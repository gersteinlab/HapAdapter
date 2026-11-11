#!/usr/bin/env python3
"""
hapadapter.fit_head -- fit a HapAdapterHead on cached embeddings from any backbone
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from hapadapter.adapter.head import HapAdapterHead
from hapadapter.adapter.train import PooledPairDataset, fit_cached, save_head
from hapadapter.backbones.cache import cache_width, load_pair_cache, resolve_coord_columns

def load_split(cache, pair_csv, meta_csv, label_col, depth_col, cov_col, transform, tag):
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

    got = load_pair_cache(cache, meta[cc], meta[pc], transform=transform, tag=tag)
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

    if out["depth"] is None:
        print(f"[{tag}] WARNING no {depth_col!r} column in {pair_csv} -- fitting unweighted.")
    n_pos = int((out["y"] > 0.5).sum())
    print(f"[{tag}] {len(out['y'])} loci | {n_pos} positive ({n_pos/max(len(out['y']),1):.2%}) "
          f"| d_in {out['D']} | transform {out['transform']}")
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for split in ("train", "dev", "test"):
        req = split == "train"
        ap.add_argument(f"--{split}_cache", required=req)
        ap.add_argument(f"--{split}_pair", required=req)
        ap.add_argument(f"--{split}_meta", required=req)
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--depth_col", default="depth",
                    help="the build's precision weight")
    ap.add_argument("--cov_col", default="total_reads",
                    help="coverage column in the META table")
    ap.add_argument("--head_variant", default="twin", choices=["twin", "single"])
    ap.add_argument("--proj_dim", type=int, default=128)
    ap.add_argument("--head_num_layers", type=int, default=1)
    ap.add_argument("--head_hidden_size", type=int, default=-1)
    ap.add_argument("--head_activation", default="gelu")
    ap.add_argument("--head_dropout", type=float, default=0.1)
    ap.add_argument("--pool_dropout", type=float, default=0.1)
    ap.add_argument("--neff_s", type=float, default=20.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--backbone_label", default=None,
                    help="recorded in run_config.json")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()

    train = load_split(a.train_cache, a.train_pair, a.train_meta, a.label_col, a.depth_col, a.cov_col, "auto", "train")
    xf = train["transform"]

    splits = {"train": train}
    for name in ("dev", "test"):
        cache = getattr(a, f"{name}_cache")
        if not cache:
            continue
        splits[name] = load_split(cache, getattr(a, f"{name}_pair"),
                                  getattr(a, f"{name}_meta"), a.label_col, a.depth_col,
                                  a.cov_col, xf, name)

    widths = {k: v["D"] for k, v in splits.items()}
    if len(set(widths.values())) > 1:
        raise SystemExit(
            f"embedding widths disagree across splits: {widths}. These caches came from "
            f"different backbones (or different pooling modes of one)!")

    d_in = train["D"]
    head = HapAdapterHead(d_in=d_in,
                          head_variant=a.head_variant,
                          proj_dim=a.proj_dim,
                          head_num_layers=a.head_num_layers,
                          head_hidden_size=a.head_hidden_size,
                          head_activation=a.head_activation,
                          head_dropout=a.head_dropout,
                          pool_dropout=a.pool_dropout)

    print(f"[head] {a.head_variant} on d_in={d_in} "
          f"({sum(p.numel() for p in head.parameters()):,} parameters)")

    def ds(s):
        # the single-window head ignores h2, so it harmless to pass either way
        return PooledPairDataset(s["h1"], s["h2"], s["y"], s["depth"])

    model, dev_metrics = fit_cached(
        head, ds(train), ds(splits["dev"]) if "dev" in splits else None,
        neff_s=a.neff_s, lr=a.lr, weight_decay=a.weight_decay, epochs=a.epochs,
        batch_size=a.batch_size, seed=a.seed, out_dir=os.path.join(a.out_dir, "_trainer"))
    if dev_metrics:
        print(f"[dev] {json.dumps({k: round(v, 5) for k, v in dev_metrics.items() if isinstance(v, float)})}")

    save_head(head, a.out_dir, neff_s=a.neff_s,
              extra=dict(backbone_label=a.backbone_label, d_in=d_in, cache_transform=xf,
                         train_cache=os.path.abspath(a.train_cache),
                         n_train=len(train["y"]), lr=a.lr, weight_decay=a.weight_decay,
                         epochs=a.epochs, batch_size=a.batch_size, seed=a.seed,
                         label_col=a.label_col, depth_col=a.depth_col))
    print(f"[write] {a.out_dir}")

    if "test" in splits:
        import torch
        from hapadapter.scoring.stats import balanced_auroc
        t = splits["test"]
        model.eval()
        with torch.no_grad():
            ell = model(torch.as_tensor(t["h1"], dtype=torch.float32),
                        torch.as_tensor(t["h2"], dtype=torch.float32))["logits"]
        ell = ell.squeeze(-1).cpu().numpy()
        y = (t["y"] > 0.5).astype(int)

        print("\n  TEST")
        if t["cov"] is None:
            print(f"  no {a.cov_col!r} column in the test meta -- reporting the raw AUROC only.")
        for name, score in (("hapadapter head", ell),
                            ("||h1-h2|| (untrained)", np.linalg.norm(t["h1"] - t["h2"], axis=1))):
            pt, _, ci, m = balanced_auroc(score, y, cover=t["cov"], n_boot=a.n_boot,
                                          seed=a.seed)
            kind = "coverage-matched" if t["cov"] is not None else "unmatched"
            print(f"  {name:24}{pt:.4f}  CI [{ci[0]:.4f}, {ci[1]:.4f}]  "
                  f"n/class {m}  ({kind})")

        pd.DataFrame(dict(chr=t["chrom"], pos=t["pos"], ell=ell, label=y)).to_csv(
            os.path.join(a.out_dir, "test_perlocus.csv"), index=False)
        print(f"[write] {os.path.join(a.out_dir, 'test_perlocus.csv')}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
