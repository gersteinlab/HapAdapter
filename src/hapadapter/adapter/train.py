#!/usr/bin/env python3
"""
hapadapter.adapter.train -- fit the HapAdapter head, on cached embeddings or a live encoder
** the head is identical either way, only where h1/h2 come from differs: **

    fit_cached        h1, h2 are precomputed pooled embeddings. Required when the backbone
                      cannot be expressed in a torch graph at all (AlphaGenome)
    fit_end_to_end    h1, h2 are pooled from an encoder inside the training step

Both run through the SAME transformers.Trainer with the same objective,
so heads fitted either way are comparable
"""
import hashlib
import json
import os
import torch

from hapadapter.adapter.head import HapAdapterHead, asb_loss

# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------

class _PairedHeadBase(torch.nn.Module):
    """
    Shared loss/output plumbing for both backends,
    returns a plain dict rather than a transformers output type
    """

    def __init__(self, head: HapAdapterHead, neff_s: float = 20.0):
        super().__init__()
        if neff_s <= 0:
            raise ValueError(f"neff_s must be > 0, got {neff_s}.")
        self.head = head
        self.neff_s = float(neff_s) # loss hyperparameter, not a head parameter

    def _finish(self, ell, labels, depth):
        loss = None if labels is None else asb_loss(ell, labels, depth, self.neff_s)
        return {"loss": loss, "logits": ell.unsqueeze(-1)}

class CachedPairedPrediction(_PairedHeadBase):
    """Head only; h1/h2 arrive already pooled"""

    def forward(self, h1, h2, labels=None, depth=None, **kwargs):
        return self._finish(self.head(h1, h2), labels, depth)

class HapAdapterForPairedPrediction(_PairedHeadBase):
    """Encoder + head"""
    def __init__(self, encoder, head: HapAdapterHead, neff_s: float = 20.0):
        super().__init__(head, neff_s)
        self.encoder = encoder

    def forward(self, input_ids, attention_mask, input_ids_alt=None,
                attention_mask_alt=None, labels=None, depth=None, **kwargs):
        if input_ids_alt is None:
            raise RuntimeError(
                "the contrast head requires a paired (hap1, hap2) batch, but input_ids_alt did "
                "not reach forward. Use --input_mode hap_pair and the paired collator."
            )
        h1 = self.encoder._pool_one(input_ids, attention_mask, apply_dropout=False)
        h2 = self.encoder._pool_one(input_ids_alt, attention_mask_alt, apply_dropout=False)
        return self._finish(self.head(h1, h2), labels, depth)

def freeze_encoder(encoder) -> int:
    """
    requires_grad=False on every encoder parameter; returns how many were frozen
    """
    n = 0
    for p in encoder.parameters():
        p.requires_grad = False
        n += 1
    return n

# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def backbone_fingerprint(encoder) -> str:
    """
    sha1 over the encoder's backbone tensors, in sorted key order
    """
    h = hashlib.sha1()
    sd = encoder.state_dict()
    for k in sorted(k for k in sd if k.split(".")[0] == "backbone"):
        t = sd[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()

def save_head(head: HapAdapterHead, out_dir: str, *, neff_s: float,
              encoder=None, init_backbone_from: str = None, extra: dict = None) -> str:
    """Write a HEAD-ONLY checkpoint: head weights + everything needed to rebuild it"""
    os.makedirs(out_dir, exist_ok=True)

    wpath = os.path.join(out_dir, "pytorch_model.bin")
    wtmp = wpath + ".tmp"
    torch.save(head.state_dict(), wtmp)
    os.replace(wtmp, wpath)

    cfg = dict(head.config())
    cfg.update(task="classification", neff_s=float(neff_s), checkpoint_kind="head_only")
    
    if init_backbone_from:
        cfg["init_backbone_from"] = init_backbone_from
    if encoder is not None:
        cfg["backbone_fingerprint"] = backbone_fingerprint(encoder)
    if extra:
        cfg.update(extra)
    
    path = os.path.join(out_dir, "run_config.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(cfg, fh, indent=2)
    os.replace(tmp, path)
    return out_dir

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class PooledPairDataset(torch.utils.data.Dataset):
    """(h1, h2, label, depth) rows of precomputed embeddings"""

    def __init__(self, h1, h2, labels, depth=None):
        if not (len(h1) == len(h2) == len(labels)):
            raise ValueError(f"length mismatch: h1={len(h1)} h2={len(h2)} labels={len(labels)}")
        if depth is not None and len(depth) != len(h1):
            raise ValueError(f"depth has {len(depth)} rows, embeddings have {len(h1)}")
        self.h1 = torch.as_tensor(h1, dtype=torch.float32)
        self.h2 = torch.as_tensor(h2, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)
        self.depth = None if depth is None else torch.as_tensor(depth, dtype=torch.float32)

    def __len__(self):
        return len(self.h1)

    def __getitem__(self, i):
        row = dict(h1=self.h1[i], h2=self.h2[i], labels=self.labels[i])
        if self.depth is not None:
            row["depth"] = self.depth[i]
        return row

# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def _binary_metrics(eval_pred):
    """AUROC/AUPRC on the head's single logit ell = softplus(a)*||z1-z2|| + b, for epoch-level model selection"""
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score
    logits, labels = eval_pred
    ell = np.asarray(logits, dtype=float).reshape(-1)
    y = (np.asarray(labels, dtype=float).reshape(-1) > 0.5).astype(int)
    n_pos = int(y.sum())
    if 0 < n_pos < len(y):
        return {"auroc": float(roc_auc_score(y, ell)),
                "auprc": float(average_precision_score(y, ell)), "n_pos": float(n_pos)}
    return {"auroc": float("nan"), "auprc": float("nan"), "n_pos": float(n_pos)}

def _trainer(model, train_ds, eval_ds, out_dir, *, lr, weight_decay, epochs, batch_size,
             seed, select_on="auroc", extra_args=None):
    """One place where TrainingArguments are set, so both backends agree on them"""
    import transformers
    args = dict(output_dir=out_dir, learning_rate=lr, weight_decay=weight_decay,
                num_train_epochs=epochs, per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size, seed=seed,
                report_to=[], save_strategy="no", logging_steps=50)
    if eval_ds is not None:
        import inspect
        fields = set(inspect.signature(transformers.TrainingArguments.__init__).parameters)
        key = "eval_strategy" if "eval_strategy" in fields else "evaluation_strategy"
        if key not in fields:
            raise RuntimeError(
                f"transformers {transformers.__version__} TrainingArguments accepts neither "
                f"'eval_strategy' nor 'evaluation_strategy'; pass the right name via extra_args."
            )
                args[key] = "epoch"
        if select_on:
            args["save_strategy"] = "epoch"
            args["save_total_limit"] = 1
            args["load_best_model_at_end"] = True
            args["metric_for_best_model"] = f"eval_{select_on}"
            args["greater_is_better"] = select_on != "loss" # greater is better unless selecting on loss
    if extra_args:
        args.update(extra_args)
    return transformers.Trainer(
        model=model, args=transformers.TrainingArguments(**args),
        train_dataset=train_ds, eval_dataset=eval_ds,
        compute_metrics=_binary_metrics if eval_ds is not None else None)

def fit_cached(head, train, dev=None, *, neff_s=20.0, lr=1e-3, weight_decay=0.01, epochs=25,
               batch_size=32, seed=20, out_dir="./head_run", select_on="auroc"):
    """Fit `head` on precomputed pooled embeddings. `train`/`dev` are PooledPairDataset"""
    model = CachedPairedPrediction(head, neff_s=neff_s)
    tr = _trainer(model, train, dev, out_dir, lr=lr, weight_decay=weight_decay, epochs=epochs,
                  batch_size=batch_size, seed=seed, select_on=select_on)
    tr.train()
    return model, (tr.evaluate() if dev is not None else None)

def fit_end_to_end(encoder, head, train, dev=None, *, neff_s=20.0, lr=1e-3, weight_decay=0.01,
                   epochs=25, batch_size=8, seed=20, out_dir="./head_run", select_on="auroc",
                   data_collator=None, trainer_cls=None, extra_args=None):
    """
    Fit `head` with a live encoder (frozen but left in train() mode for dropout variation)
    """
    import transformers
    freeze_encoder(encoder)
    model = HapAdapterForPairedPrediction(encoder, head, neff_s=neff_s)
    tr = _trainer(model, train, dev, out_dir, lr=lr, weight_decay=weight_decay, epochs=epochs,
                  batch_size=batch_size, seed=seed, select_on=select_on, extra_args=extra_args)
    if data_collator is not None:
        tr.data_collator = data_collator
    if trainer_cls is not None and trainer_cls is not transformers.Trainer:
        tr = trainer_cls(model=model, args=tr.args, train_dataset=train, eval_dataset=dev,
                         data_collator=data_collator)
    tr.train()
    return model, (tr.evaluate() if dev is not None else None)
