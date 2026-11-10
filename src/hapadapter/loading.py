#!/usr/bin/env python3
"""
hapadapter.loading -- load any HapAdapter checkpoint
"""
import re
import torch
import numpy as np

from hapadapter.adapter.head import HapAdapterHead
from hapadapter.trunk.loading import (
    _load_state_dict,
    apply_overrides,
    build_model,
    find_weights_file,
    load_model_weights,
    load_run_config,
    load_tokenizer,
)

def _head_kind(state_dict) -> str:
    roots = {k.split(".")[0] for k in state_dict}
    if "proj" in roots or "dist_a" in roots:
        return "contrast"
    if "main_head" in roots:
        return "single_window"
    raise ValueError(
        f"checkpoint has neither proj/dist_a (contrast head) nor main_head (single-window "
        f"readout); top-level keys are {sorted(roots)[:8]}. Refusing to guess which it is!"
    )



def _head_from_weights(head_sd, pool_dropout: float) -> HapAdapterHead:
    """Rebuild a HapAdapter twin contrast head from its weights"""
    def _idx(k):
        m = re.search(r"\.(\d+)\.weight$", k)
        return int(m.group(1)) if m else -1

    # sort by sequential index, not lexically
    lin = sorted((k for k in head_sd if k.endswith(".weight") and k.startswith("proj")), key=_idx)
    if not lin:
        raise ValueError(f"contrast checkpoint has no proj.*.weight; got {sorted(head_sd)}")

    d_in = head_sd[lin[0]].shape[1] # first linear layer input dim
    proj_dim = head_sd[lin[-1]].shape[0] # last linear layer output dim
    n_layers = len(lin)

    head = HapAdapterHead(
        d_in=int(d_in),
        head_variant="twin",
        proj_dim=int(proj_dim),
        head_num_layers=n_layers,
        head_hidden_size=int(head_sd[lin[0]].shape[0]) if n_layers > 1 else -1,
        pool_dropout=pool_dropout,
    )

    head.load_state_dict(head_sd, strict=True)
    return head

def load_model_and_tokenizer(checkpoint_dir: str, device: str = "cpu", overrides: dict = None):
    """Returns (model, tokenizer, run_config) for either checkpoint shape"""
    run_config = apply_overrides(load_run_config(checkpoint_dir), overrides or {})
    weights = find_weights_file(checkpoint_dir)
    state_dict = _load_state_dict(weights)
    tokenizer = load_tokenizer(run_config)

    if _head_kind(state_dict) == "single_window":
        model = build_model(run_config, device=device)
        load_model_weights(model, weights)
        model.eval()
        return model, tokenizer, run_config

    # contrast checkpoint: encoder trunk + HapAdapter head
    encoder = build_model(dict(run_config, task="regression", num_labels=1), device=device)
    bb = {k: v for k, v in state_dict.items() if k.split(".")[0] == "backbone"}
    if not bb:
        raise ValueError(
            f"{weights} has a HapAdapter twin contrast head but no backbone.* weights. "
            f"A head-only checkpoint needs its backbone named in run_config['init_backbone_from']."
        )

    enc_missing, _ = encoder.load_state_dict(bb, strict=False)
    still = [k for k in enc_missing if k.split(".")[0] == "backbone"]
    if still:
        raise RuntimeError(
            f"{len(still)} backbone tensors did not load ({still[:4]}...). The checkpoint and "
            f"the run_config architecture disagree. Refusing to score on a partly random encoder."
        )

    head_sd = {k: v for k, v in state_dict.items()
               if k.split(".")[0] in ("proj", "dist_a", "dist_b")}
    head = _head_from_weights(head_sd, pool_dropout=float(encoder.dropout.p))

    from hapadapter.adapter.train import HapAdapterForAlleleSpecificPrediction
    model = HapAdapterForAlleleSpecificPrediction(encoder, head, neff_s=run_config.get("neff_s", 20.0)).to(device)
    model.eval()
    return model, tokenizer, run_config

@torch.no_grad()
def logits_and_embeddings(model, input_ids, attention_mask,
                          input_ids_alt=None, attention_mask_alt=None,
                          return_pools=False):
    """
    Score sequence(s) with the model's backbone -> pooling -> head

    Returns:
        return_pools=False:  (logits, pooled_contrast)
        return_pools=True:   (logits, pooled_contrast, pool1, pool2)   (pool2=None if single sequence)
    """
    is_contrast = hasattr(model, "head")
    encoder = model.encoder if is_contrast else model

    # Route through the encoder's own pooling so it matches training exactly
    def _pool(ids, mask):
        return encoder._pool_one(ids, mask)

    pool1 = _pool(input_ids, attention_mask)
    pool2 = None if input_ids_alt is None else _pool(input_ids_alt, attention_mask_alt)

    if is_contrast:
        if input_ids_alt is None:
            raise ValueError(
                "a twin contrast head scores a haplotype PAIR (hap1, hap2), "
                "but a single sequence was given!"
            )
        logits = model.head(pool1, pool2).unsqueeze(-1)   # (N,1)
        pooled = model.head.proj(pool1) - model.head.proj(pool2)  # projected contrast
        if return_pools:
            return logits, pooled, pool1, pool2
        return logits, pooled

    # single-window readout
    # shared by the regression encoder and single-window classification
    logits = model.main_head(pool1)
    pooled = pool1

    if return_pools:
        return logits, pooled, pool1, pool2
    return logits, pooled

# ---------------------------------------------------------------------------
# Batched inference over a list of sequences or sequence pairs
# ---------------------------------------------------------------------------

def run_inference(checkpoint_dir, texts, batch_size=64, device="cpu", overrides=None, dump_pools=False):
    """
    Load a checkpoint and score a list of inputs

    dump_pools=False (default):
        returns (logits, emb, run_config)
            logits : (N, 1) or (N, T) for multi-track regression
            emb    : (N, D)
        where N = number of inputs, D = embedding dimension

    dump_pools=True (pair inputs only):
        returns (logits, emb, pool_ref, pool_alt, run_config)
            pool_ref = pool(window1), pool_alt = pool(window2) -- RAW per-window pools

    Convention: window1 is the ref / hap1 window, window2 is the alt / hap2 window
    """
    model, tokenizer, run_config = load_model_and_tokenizer(
        checkpoint_dir, device=device, overrides=overrides)

    is_pair = len(texts) > 0 and isinstance(texts[0], (list, tuple))
    if dump_pools and not is_pair:
        raise ValueError("dump_pools=True requires paired inputs ([window1, window2]).")

    _is_contrast = hasattr(model, "head")
    if _is_contrast and not is_pair:
        raise ValueError("a contrast head scores a haplotype PAIR ([window1, window2]); "
                         "this checkpoint has one, but a flat list of sequences was given")
    if not _is_contrast and is_pair and not dump_pools:
        raise ValueError(
            "this checkpoint scores ONE window, so window2 would be discarded. Pass "
            "dump_pools=True to get both windows' pooled embeddings back.")

    mml = run_config.get("model_max_length", 512)

    all_logits, all_emb, all_ref, all_alt = [], [], [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        if is_pair:
            ref = [b[0] for b in batch]
            alt = [b[1] for b in batch]
            enc_r = tokenizer(ref, return_tensors="pt", padding="longest",
                              max_length=mml, truncation=True)
            enc_a = tokenizer(alt, return_tensors="pt", padding="longest",
                              max_length=mml, truncation=True)
            out = logits_and_embeddings(
                model, enc_r["input_ids"].to(device), enc_r["attention_mask"].to(device),
                input_ids_alt=enc_a["input_ids"].to(device),
                attention_mask_alt=enc_a["attention_mask"].to(device),
                return_pools=dump_pools)
            if dump_pools:
                logits, pooled, pool1, pool2 = out
                all_ref.append(pool1.detach().cpu().numpy())
                all_alt.append(pool2.detach().cpu().numpy())
            else:
                logits, pooled = out
        else:
            enc = tokenizer(batch, return_tensors="pt", padding="longest",
                            max_length=mml, truncation=True)
            logits, pooled = logits_and_embeddings(
                model, enc["input_ids"].to(device), enc["attention_mask"].to(device))
        all_logits.append(logits.detach().cpu().numpy())
        all_emb.append(pooled.detach().cpu().numpy())
    
    logits_out = np.concatenate(all_logits, axis=0)
    emb_out = np.concatenate(all_emb, axis=0)
    if dump_pools:
        return (logits_out, emb_out,
                np.concatenate(all_ref, axis=0), np.concatenate(all_alt, axis=0),
                run_config)
                
    return logits_out, emb_out, run_config
