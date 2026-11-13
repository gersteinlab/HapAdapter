#!/usr/bin/env python3
"""
hapadapter.scoring.score_dnabert2_ft -- DNABERT-2 (fine-tuned) forward pass on each hap and predictions differenced
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

def resolve_coords(meta, path, chrom_col=None, pos_col=None):
    """(chrom, pos) arrays from a build's meta CSV"""
    cc = chrom_col or ("chr" if "chr" in meta.columns else
                       next((c for c in ("chrom", "chromosome") if c in meta.columns), None))
    pc = pos_col or next((c for c in ("anchor", "pos", "ref_start", "SNV") if c in meta.columns), None)
    if cc is None or pc is None:
        raise SystemExit(
            f"{path}: cannot resolve coordinate columns (chrom={cc!r}, pos={pc!r}); "
            f"present: {list(meta.columns)}. Pass --chrom_col/--pos_col.")
    return cc, pc

def main():
    from hapadapter.loading import run_inference

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint_dir", required=True,
                    help="the fine-tuned trunk, not a HapAdapter head")
    ap.add_argument("--pair_csv", required=True,
                    help="the build's test.csv: sequence1 = hap1, sequence2 = hap2")
    ap.add_argument("--meta_csv", required=True,
                    help="the matching test.meta.csv, for the chr/anchor coordinates that "
                         "score_matched joins on")
    ap.add_argument("--chrom_col", default=None)
    ap.add_argument("--pos_col", default=None)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0,
                    help="Score the first N loci only to test the setup is working")
    a = ap.parse_args()

    pair = pd.read_csv(a.pair_csv)
    meta = pd.read_csv(a.meta_csv)
    if len(pair) != len(meta):
        raise SystemExit(f"row-count mismatch: {a.pair_csv} has {len(pair):,} rows, "
                         f"{a.meta_csv} has {len(meta):,}. These must be the same build's "
                         f"paired files!")
    for c in ("sequence1", "sequence2"):
        if c not in pair.columns:
            raise SystemExit(f"{a.pair_csv}: no {c!r} column; present: {list(pair.columns)}.")
    cc, pc = resolve_coords(meta, a.meta_csv, a.chrom_col, a.pos_col)

    if a.limit:
        pair, meta = pair.head(a.limit), meta.head(a.limit)
        print(f"[pilot] first {len(pair):,} loci only")

    s1 = pair["sequence1"].astype(str).tolist()
    s2 = pair["sequence2"].astype(str).tolist()
    n = len(s1)
    lens = {len(x) for x in s1[:200]} | {len(x) for x in s2[:200]}
    print(f"[input] {n:,} loci | window {min(lens)}-{max(lens)} bp | coords {cc!r}/{pc!r}")

    n_ident = sum(x == y for x, y in zip(s1, s2))
    if n_ident == n:
        raise SystemExit(
            f"FATAL: sequence1 == sequence2 at ALL {n:,} loci. Check to see if "
            f"you accidentally input reference (instead of personal) windows!")
    if n_ident:
        # may get some that line up due to indels (not necessarily an issue)
        print(f"[warn] {n_ident:,} of {n:,} loci ({n_ident/n:.1%}) have identical haplotype windows.")

    # one forward pass over both haplotypes
    logits, _emb, rc = run_inference(a.checkpoint_dir, s1 + s2, a.batch_size, a.device, None)
    arr = np.asarray(logits, dtype=float)
    n_out = arr.shape[1] if arr.ndim > 1 else 1

    task = str(rc.get("task", "")).lower()
    if task and task != "regression":
        raise SystemExit(
            f"FATAL: {a.checkpoint_dir} has task={task!r}. This script is currently "
            f"only built for a trunk fine-tuned with a regression head.")
    if int(rc.get("num_labels", 1)) > 1 or n_out > 1:
        raise SystemExit(
            f"FATAL: {a.checkpoint_dir} is a MULTI-TRACK trunk (run_config "
            f"num_labels={rc.get('num_labels')}, output width {n_out}). This script is "
            f"currently only built for a trunk fine-tuned with a SINGLE-TRACK regression head.")
    if arr.shape[0] != 2 * n:
        raise SystemExit(f"FATAL: expected {2*n:,} predictions for {n:,} loci x 2 haplotypes, "
                         f"got {arr.shape[0]:,}. The loader did not score one window per input.")

    mu = arr.reshape(-1) if arr.ndim == 1 else arr[:, 0]
    mu1, mu2 = mu[:n], mu[n:]
    delta = mu1 - mu2

    out = pd.DataFrame({"chr": meta[cc].astype(str).to_numpy(),
                        "pos": meta[pc].astype(int).to_numpy(),
                        "s1": mu1, "s2": mu2, "delta": delta})
    os.makedirs(os.path.dirname(a.out_csv) or ".", exist_ok=True)
    tmp = a.out_csv + ".tmp"
    out.to_csv(tmp, index=False)
    os.replace(tmp, a.out_csv)

    nz = int((np.abs(delta) > 0).sum())
    print(f"[delta] |delta| median {np.median(np.abs(delta)):.5f} "
          f"p90 {np.quantile(np.abs(delta), 0.9):.5f} max {np.abs(delta).max():.5f} "
          f"| nonzero {nz:,}/{n:,}")
    print(f"[mu]    hap1 mean {mu1.mean():+.4f} sd {mu1.std():.4f} | "
          f"hap2 mean {mu2.mean():+.4f} sd {mu2.std():.4f}")
    if nz == 0:
        print("[warn] every delta is exactly 0 -- the trunk gives both haplotypes the same "
              "score at every locus. Check that the windows actually differ.")
    print(f"[write] {a.out_csv}  ({len(out):,} rows: chr,pos,s1,s2,delta)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
