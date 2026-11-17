#!/usr/bin/env python3
"""
hapadapter.fit_adapter -- fit a HapAdapterHead, from either backend option

    --backend cached     train on PRECOMPUTED pooled embeddings (npz caches).
                         Any backbone, including ones that are not torch models at all
                         (e.g. AlphaGenome runs in JAX, in a different environment)
                         Only requires CPUs.

    --backend sequence   train with a live but FROZEN torch encoder in the loop, from the build's
                         sequence CSVs. Requires GPU + transformers.

Same head with same loss fit either way
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from hapadapter.adapter.head import HapAdapterHead
from hapadapter.adapter.train import PooledPairDataset, fit_cached, save_head
from hapadapter.backbones.cache import load_pair_cache, resolve_coord_columns

# ---------------------------------------------------------------------------------------------
# Option 1: cached embeddings
# ---------------------------------------------------------------------------------------------

def load_cached_split(cache, pair_csv, meta_csv, label_col, depth_col, cov_col, transform, tag, min_ok_frac=0.0):
    """For one split, embeddings joined to the build's labels, depth, and coordinates"""
    pair = pd.read_csv(pair_csv)
    meta = pd.read_csv(meta_csv)
    if len(pair) != len(meta):
        raise SystemExit(
            f"{tag}: {pair_csv} has {len(pair)} rows, {meta_csv} has {len(meta)}. These are "
            f"written as a matched pair by build_experiment; a mismatch means one of them was "
            f"regenerated or subsampled separately!")
    if label_col not in pair.columns:
        raise SystemExit(f"{pair_csv}: no {label_col!r} column; present: {list(pair.columns)}")
    cc, pc = resolve_coord_columns(meta, meta_csv)

    got = load_pair_cache(cache, meta[cc], meta[pc], transform=transform, tag=tag, min_ok_frac=min_ok_frac)
    ok = got["ok"]
    n_miss = int((~ok).sum())
    if n_miss:
        # missing loci come back as zero rows with ok=False
        print(f"[{tag}] {n_miss} of {len(ok)} loci absent from the cache -- dropped.")

    out = dict(
        h1=got["h1"][ok],
        h2=got["h2"][ok],
        D=got["D"],
        transform=got["transform"],
        y=pair[label_col].to_numpy(float)[ok],
        depth=pair[depth_col].to_numpy(float)[ok] if depth_col in pair.columns else None,
        cov=meta[cov_col].to_numpy(float)[ok] if cov_col in meta.columns else None,
        chrom=meta[cc].astype(str).to_numpy()[ok],
        pos=meta[pc].astype(int).to_numpy()[ok],
    )
    _announce(out, tag, pair_csv, depth_col, extra=f"| d_in {out['D']} | transform {out['transform']}")
    return out

# ---------------------------------------------------------------------------------------------
# Option 2: "live" frozen encoder
# ---------------------------------------------------------------------------------------------

def load_sequence_split(build_dir, split, label_col, depth_col, cov_col, tag):
    """
    One split's raw haplotype sequences plus the same labels/coords the cached path uses
    ** returns the SAME dict shape as load_cached_split minus h1/h2/D **
    """
    pair_csv = os.path.join(build_dir, f"{split}.csv")
    meta_csv = os.path.join(build_dir, f"{split}.meta.csv")
    for p in (pair_csv, meta_csv):
        if not os.path.exists(p):
            raise SystemExit(f"{tag}: missing {p}. --backend sequence reads <build_dir>/"
                             f"{{train,dev,test}}.csv and the matching .meta.csv.")
    pair = pd.read_csv(pair_csv)
    meta = pd.read_csv(meta_csv)
    if len(pair) != len(meta):
        raise SystemExit(f"{tag}: {len(pair)} rows in {pair_csv} vs {len(meta)} in {meta_csv}.")
    for c in ("sequence1", "sequence2"):
        if c not in pair.columns:
            raise SystemExit(
                f"{pair_csv}: no {c!r} column. The contrast head needs BOTH haplotype windows; "
                f"a ref_single build has only 'sequence' and would train the head on one window "
                f"twice. Present: {list(pair.columns)}")
    if label_col not in pair.columns:
        raise SystemExit(f"{pair_csv}: no {label_col!r} column; present: {list(pair.columns)}")
    cc, pc = resolve_coord_columns(meta, meta_csv)

    out = dict(
        texts1=pair["sequence1"].astype(str).tolist(),
        texts2=pair["sequence2"].astype(str).tolist(),
        y=pair[label_col].to_numpy(float),
        depth=pair[depth_col].to_numpy(float) if depth_col in pair.columns else None,
        cov=meta[cov_col].to_numpy(float) if cov_col in meta.columns else None,
        chrom=meta[cc].astype(str).to_numpy(),
        pos=meta[pc].astype(int).to_numpy(),
    )
    ident = int(sum(a == b for a, b in zip(out["texts1"], out["texts2"])))
    if ident == len(out["y"]):
        raise SystemExit(
            f"{tag}: sequence1 == sequence2 at ALL {ident:,} loci. The contrast is identically "
            f"zero, so the head can only learn the bias term.")
    _announce(out, tag, pair_csv, depth_col,
              extra=f"| {ident} loci with identical haplotype windows")
    return out

def _announce(out, tag, pair_csv, depth_col, extra=""):
    if out["depth"] is None:
        print(f"[{tag}] WARNING no {depth_col!r} column in {pair_csv} -- fitting unweighted.")
    n = len(out["y"])
    n_pos = int((out["y"] > 0.5).sum())
    print(f"[{tag}] {n} loci | {n_pos} positive ({n_pos / max(n, 1):.2%}) {extra}")

def _make_paired_dataset(tokenizer, texts1, texts2, labels, depth, model_max_length):
    """Tokenize both haplotype windows into one Dataset"""
    import torch

    class PairedSeqDataset(torch.utils.data.Dataset):
        def __init__(self):
            e1 = tokenizer(texts1, return_tensors="pt", padding="longest",
                           max_length=model_max_length, truncation=True)
            e2 = tokenizer(texts2, return_tensors="pt", padding="longest",
                           max_length=model_max_length, truncation=True)

            self.ids1, self.mask1 = e1["input_ids"], e1["attention_mask"]
            self.ids2, self.mask2 = e2["input_ids"], e2["attention_mask"]
            self.labels = [float(v) for v in labels]
            self.depth = None if depth is None else [float(v) for v in depth]

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            row = dict(input_ids=self.ids1[i], attention_mask=self.mask1[i],
                       input_ids_alt=self.ids2[i], attention_mask_alt=self.mask2[i],
                       labels=self.labels[i])
            if self.depth is not None:
                row["depth"] = self.depth[i]
            return row

    return PairedSeqDataset()

def _make_paired_collator(tokenizer):
    """Pad both haplotypes and emit the keys HapAdapterForPairedPrediction.forward expects"""
    import torch

    pad = tokenizer.pad_token_id

    def collate(instances):
        def stack(key):
            return torch.nn.utils.rnn.pad_sequence(
                [x[key] for x in instances], batch_first=True, padding_value=pad)

        ids, ids_alt = stack("input_ids"), stack("input_ids_alt")
        batch = dict(input_ids=ids, attention_mask=ids.ne(pad),
                     input_ids_alt=ids_alt, attention_mask_alt=ids_alt.ne(pad),
                     labels=torch.tensor([x["labels"] for x in instances], dtype=torch.float))
        
        if "depth" in instances[0]:
            batch["depth"] = torch.tensor([x["depth"] for x in instances], dtype=torch.float)
        return batch

    return collate

def build_encoder(trunk_dir, backbone_override=None):
    """The frozen encoder, with pooling settings read from the trunk"""
    from hapadapter.trunk.loading import DEFAULT_MODEL_MAX_LENGTH, load_run_config, load_tokenizer
    from hapadapter.trunk.model import FunctionalFinetuneForSequencePrediction

    rc = load_run_config(trunk_dir)
    backbone = backbone_override or rc.get("model_name_or_path")
    if not backbone:
        raise SystemExit(f"{trunk_dir}/run_config.json records no 'model_name_or_path' and no "
                         f"--backbone was given.")
    mml = int(rc.get("model_max_length", DEFAULT_MODEL_MAX_LENGTH))

    enc = FunctionalFinetuneForSequencePrediction(
        model_name_or_path=backbone,
        cache_dir=rc.get("cache_dir"),
        pooling_mode=rc.get("pooling_mode", "center_mean"),
        center_pool_width=int(rc.get("center_pool_width", 5)),
        # the encoder's own readout is never used here -- the HapAdapterHead is the readout
        # so it is built here at its cheapest shape to be discarded
        head_num_layers=1, head_hidden_size=-1,
        head_activation=rc.get("head_activation", "gelu"),
        head_dropout=float(rc.get("head_dropout", 0.1)),
        neff_s=float(rc.get("neff_s", 20.0)),
        task="regression", num_labels=1)

    # include_head=False: we just want the trunk's BACKBONE, not its regression readout
    from hapadapter.trunk.loading import find_weights_file
    wsrc = trunk_dir if os.path.isfile(trunk_dir) else find_weights_file(trunk_dir)
    n_loaded = enc.init_backbone_from(wsrc, include_head=False)
    print(f"[encoder] backbone weights from {wsrc}")
    if os.path.basename(os.path.dirname(wsrc)).startswith("checkpoint-"):
        print(f"          NOTE this is a mid-training checkpoint, not a top-level saved model.")
    print(f"          {n_loaded} backbone tensors loaded (head deliberately not restored)")

    tok = load_tokenizer({**rc, "model_name_or_path": backbone, "model_max_length": mml})
    print(f"[encoder] backbone {backbone}")
    print(f"          pooling {rc.get('pooling_mode', 'center_mean')} "
          f"width {rc.get('center_pool_width', 5)} | model_max_length {mml} | "
          f"d_in {enc.backbone.config.hidden_size}")
    return enc, tok, mml

def _pool_split(encoder, texts1, texts2, tokenizer, mml, batch_size, device):
    """Pooled (h1, h2) for a split, encoder in eval()"""
    import torch

    encoder.eval().to(device)
    out = []
    for texts in (texts1, texts2):
        chunks = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = tokenizer(texts[i:i + batch_size], return_tensors="pt", padding="longest",
                                max_length=mml, truncation=True)
                h = encoder._pool_one(enc["input_ids"].to(device),
                                      enc["attention_mask"].to(device),
                                      apply_dropout=False)
                chunks.append(h.detach().cpu().numpy())
        out.append(np.vstack(chunks))
    return out[0], out[1]

# ---------------------------------------------------------------------------------------------
# Shared behavior
# ---------------------------------------------------------------------------------------------

def cadence_args(a):
    """extra_args switching _trainer from per-epoch to per-step eval, or {} to leave it"""
    if a.eval_strategy != "steps":
        return {}
    import inspect

    import transformers
    fields = set(inspect.signature(transformers.TrainingArguments.__init__).parameters)
    key = "eval_strategy" if "eval_strategy" in fields else "evaluation_strategy"
    return {key: "steps", "save_strategy": "steps",
            "eval_steps": a.eval_steps, "save_steps": a.save_steps}

def build_head(a, d_in):
    """Construct the head"""
    import torch
    torch.manual_seed(a.seed)

    head = HapAdapterHead(d_in=d_in,
                          head_variant=a.head_variant,
                          proj_dim=a.proj_dim,
                          head_num_layers=a.head_num_layers,
                          head_hidden_size=a.head_hidden_size,
                          head_activation=a.head_activation,
                          head_dropout=a.head_dropout,
                          pool_dropout=a.pool_dropout)

    print(f"[head] {a.head_variant} on d_in={d_in} "
          f"({sum(p.numel() for p in head.parameters()):,} parameters) "
          f"| {a.head_num_layers} layer(s), hidden {a.head_hidden_size}, proj {a.proj_dim}")
    return head

def report_test(a, head, t, out_dir):
    """Score the test split and write test_perlocus.csv"""
    import torch
    from hapadapter.adapter.train import CachedPairedPrediction
    from hapadapter.scoring.stats import balanced_auroc

    scorer = CachedPairedPrediction(head, neff_s=a.neff_s).eval()
    dev = next(scorer.parameters()).device
    with torch.no_grad():
        ell = scorer(torch.as_tensor(t["h1"], dtype=torch.float32).to(dev),
                     torch.as_tensor(t["h2"], dtype=torch.float32).to(dev))["logits"]
    ell = ell.squeeze(-1).cpu().numpy()
    y = (t["y"] > 0.5).astype(int)

    print("\n  TEST")
    if t["cov"] is None:
        print(f"  no {a.cov_col!r} column in the test meta -- reporting the raw AUROC only.")
    for name, score in (("hapadapter head", ell),
                        ("||h1-h2|| (untrained)", np.linalg.norm(t["h1"] - t["h2"], axis=1))):
        pt, _, ci, m = balanced_auroc(score, y, cover=t["cov"], n_boot=a.n_boot, seed=a.seed)
        kind = "coverage-matched" if t["cov"] is not None else "unmatched"
        print(f"  {name:24}{pt:.4f}  CI [{ci[0]:.4f}, {ci[1]:.4f}]  n/class {m}  ({kind})")

    p = os.path.join(out_dir, "test_perlocus.csv")
    pd.DataFrame(dict(chr=t["chrom"], pos=t["pos"], ell=ell, label=y)).to_csv(p, index=False)
    print(f"[write] {p}")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", required=True, choices=["cached", "sequence"])
    # ---- backend: cached
    for split in ("train", "dev", "test"):
        ap.add_argument(f"--{split}_cache")
        ap.add_argument(f"--{split}_pair")
        ap.add_argument(f"--{split}_meta")
    ap.add_argument("--min_ok_frac", type=float, default=0.0,
                    help="cached: minimum fraction of the meta table's loci that must be in "
                         "the cache. Default 0 for a full build; adjust if you are sampling.")
    # ---- backend: sequence
    ap.add_argument("--build_dir",
                    help="sequence: directory holding train/dev/test.csv + .meta.csv")
    ap.add_argument("--init_backbone_from",
                    help="sequence: the (possibly fine-tuned) trunk directory. Its weights initialise the "
                         "frozen encoder and its run_config supplies the pooling settings.")
    ap.add_argument("--backbone", default=None,
                    help="sequence: override the backbone recorded in the trunk's run_config "
                         "(use when that path has moved)")
    ap.add_argument("--balanced_sampler", action="store_true",
                    help="sequence: class-balanced WeightedRandomSampler.")
    ap.add_argument("--warmup_steps", type=int, default=0)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pool_batch_size", type=int, default=64,
                    help="sequence: batch size for the TEST-split pooling pass only")
    # ---- shared
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--depth_col", default="depth", help="the build's precision weight")
    ap.add_argument("--cov_col", default="total_reads", help="coverage column in the META table")
    ap.add_argument("--head_variant", default="twin", choices=["twin", "single"])
    ap.add_argument("--proj_dim", type=int, default=128)
    ap.add_argument("--head_num_layers", type=int, default=2)
    ap.add_argument("--head_hidden_size", type=int, default=128)
    ap.add_argument("--head_activation", default="gelu")
    ap.add_argument("--head_dropout", type=float, default=0.1)
    ap.add_argument("--pool_dropout", type=float, default=0.1,
                    help="p of the SINGLE dropout on the pooled vector")
    ap.add_argument("--neff_s", type=float, default=20.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20)
    ap.add_argument("--eval_strategy", default="epoch", choices=["epoch", "steps"])
    ap.add_argument("--eval_steps", type=int, default=250)
    ap.add_argument("--save_steps", type=int, default=250)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--backbone_label", default=None, help="recorded in run_config.json")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()

    if a.backend == "cached":
        missing = [f"--{s}_{k}" for s in ("train",) for k in ("cache", "pair", "meta")
                   if not getattr(a, f"{s}_{k}")]
        if missing:
            raise SystemExit(f"--backend cached needs {', '.join(missing)}")
        splits, d_in, xf = _load_cached(a)
        provenance = dict(train_cache=os.path.abspath(a.train_cache), cache_transform=xf)
    else:
        for flag in ("build_dir", "init_backbone_from"):
            if not getattr(a, flag):
                raise SystemExit(f"--backend sequence needs --{flag}")
        splits, d_in, encoder, tokenizer, mml = _load_sequence(a)
        provenance = dict(build_dir=os.path.abspath(a.build_dir),
                          init_backbone_from=os.path.abspath(a.init_backbone_from),
                          balanced_sampler=bool(a.balanced_sampler),
                          warmup_steps=a.warmup_steps, bf16=bool(a.bf16))

    head = build_head(a, d_in)
    os.makedirs(a.out_dir, exist_ok=True)
    trainer_out = os.path.join(a.out_dir, "_trainer")

    if a.backend == "cached":
        def ds(s):
            # the single-window head ignores h2, so passing it is harmless either way
            return PooledPairDataset(s["h1"], s["h2"], s["y"], s["depth"])

        model, dev_metrics = fit_cached(
            head, ds(splits["train"]), ds(splits["dev"]) if "dev" in splits else None,
            neff_s=a.neff_s, lr=a.lr, weight_decay=a.weight_decay, epochs=a.epochs,
            batch_size=a.batch_size, seed=a.seed, out_dir=trainer_out,
            extra_args=cadence_args(a) or None)
    else:
        model, dev_metrics = _fit_sequence(a, encoder, head, splits, tokenizer, mml, trainer_out)

    if dev_metrics:
        print(f"[dev] {json.dumps({k: round(v, 5) for k, v in dev_metrics.items() if isinstance(v, float)})}")

    save_head(head, a.out_dir, neff_s=a.neff_s,
              encoder=encoder if a.backend == "sequence" else None,
              init_backbone_from=a.init_backbone_from if a.backend == "sequence" else None,
              extra=dict(backbone_label=a.backbone_label, d_in=d_in, backend=a.backend,
                         n_train=len(splits["train"]["y"]), lr=a.lr,
                         weight_decay=a.weight_decay, epochs=a.epochs,
                         batch_size=a.batch_size, seed=a.seed, label_col=a.label_col,
                        depth_col=a.depth_col, eval_strategy=a.eval_strategy,
                         eval_steps=a.eval_steps if a.eval_strategy == "steps" else None,
                         **provenance))

    print(f"[write] {a.out_dir}")

    if "test" in splits:
        t = splits["test"]
        if a.backend == "sequence":
            t["h1"], t["h2"] = _pool_split(encoder, t["texts1"], t["texts2"], tokenizer, mml,
                                           a.pool_batch_size, a.device)
        report_test(a, head, t, a.out_dir)
    return 0

def _load_cached(a):
    train = load_cached_split(a.train_cache, a.train_pair, a.train_meta, a.label_col,
                              a.depth_col, a.cov_col, "auto", "train", a.min_ok_frac)
    xf = train["transform"]
    splits = {"train": train}
    for name in ("dev", "test"):
        cache = getattr(a, f"{name}_cache")
        if not cache:
            continue
        splits[name] = load_cached_split(cache, getattr(a, f"{name}_pair"),
                                         getattr(a, f"{name}_meta"), a.label_col, a.depth_col,
                                         a.cov_col, xf, name, a.min_ok_frac)
    widths = {k: v["D"] for k, v in splits.items()}
    if len(set(widths.values())) > 1:
        raise SystemExit(
            f"embedding widths disagree across splits: {widths}. These caches came from "
            f"different backbones (or different pooling modes of one)!")
    return splits, train["D"], xf

def _load_sequence(a):
    encoder, tokenizer, mml = build_encoder(a.init_backbone_from, a.backbone)
    splits = {}
    for name in ("train", "dev", "test"):
        p = os.path.join(a.build_dir, f"{name}.csv")
        if name != "train" and not os.path.exists(p):
            continue
        splits[name] = load_sequence_split(a.build_dir, name, a.label_col, a.depth_col,
                                           a.cov_col, name)
    return splits, int(encoder.backbone.config.hidden_size), encoder, tokenizer, mml

def _fit_sequence(a, encoder, head, splits, tokenizer, mml, trainer_out):
    from hapadapter.adapter.train import fit_end_to_end
    from hapadapter.trunk.finetune import BalancedTrainer

    class _BalancedTrainer(BalancedTrainer):
        def __init__(self, *args, **kw):
            super().__init__(*args, **kw)
            self.args.balanced_sampler = True

    def mk(name):
        s = splits[name]
        return _make_paired_dataset(tokenizer, s["texts1"], s["texts2"], s["y"], s["depth"], mml)

    extra = dict(cadence_args(a))
    if a.warmup_steps:
        extra["warmup_steps"] = a.warmup_steps
    if a.bf16:
        extra["bf16"] = True

    print(f"[fit] sequence backend | epochs {a.epochs} batch {a.batch_size} lr {a.lr} "
          f"wd {a.weight_decay} seed {a.seed} | balanced_sampler {bool(a.balanced_sampler)} "
          f"| warmup {a.warmup_steps} | bf16 {bool(a.bf16)} | eval {a.eval_strategy}"
          + (f"/{a.eval_steps}" if a.eval_strategy == "steps" else ""))

    return fit_end_to_end(
        encoder, head, mk("train"), mk("dev") if "dev" in splits else None,
        neff_s=a.neff_s, lr=a.lr, weight_decay=a.weight_decay, epochs=a.epochs,
        batch_size=a.batch_size, seed=a.seed, out_dir=trainer_out,
        data_collator=_make_paired_collator(tokenizer),
        trainer_cls=_BalancedTrainer if a.balanced_sampler else None,
        extra_args=extra or None)

if __name__ == "__main__":
    sys.exit(main())
