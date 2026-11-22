#!/usr/bin/env python3
"""
hapadapter.scoring.score_matched -- scoring for the backbone-comparison figure for one assay
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from hapadapter.backbones.cache import load_pair_cache, resolve_coord_columns
from hapadapter.scoring.stats import balanced_auroc

def _loci_from(path, tag):
    """(chr, pos) set from a per-locus CSV or a pair cache"""
    if path.endswith(".npz"):
        with np.load(path, allow_pickle=True) as d:
            if "keys" not in d.files:
                raise SystemExit(f"{path}: not a pair cache (no 'keys'); has {list(d.files)}")
            keys = np.asarray(d["keys"]).astype(str)
        out = set()
        for k in keys:
            c, p, _h = k.rsplit(":", 2)
            out.add((c, int(p)))
        return out
    t = pd.read_csv(path)
    cc, pc = resolve_coord_columns(t, f"{path} ({tag})")
    return set(zip(t[cc].astype(str), t[pc].astype(int)))

def _rung(rows, assay, backbone, rung, score, y, rank, note="", n_shared=None, min_frac=0.9):
    """Score one rung on a random class-balanced subsample and append a row"""
    if score is None or len(score) == 0:
        rows.append(dict(assay=assay, backbone=backbone, rung=rung, auroc_plain=np.nan,
                         ci_lo=np.nan, ci_hi=np.nan, n_per_class=0, n_loci=0,
                         frac_shared=np.nan, ranked_on=rank, note=note or "input missing"))
        return

    frac = (len(score) / n_shared) if n_shared else np.nan
    if n_shared and frac < min_frac:
        print(f"  [warn] {backbone}/{rung}: {len(score)}/{n_shared} of the shared loci ({frac:.1%}).")

    pt, _ap, ci, m = balanced_auroc(score, y, cover=None, n_boot=1000, seed=20)
    rows.append(dict(assay=assay, backbone=backbone, rung=rung, auroc_plain=pt,
                     ci_lo=ci[0], ci_hi=ci[1], n_per_class=m, n_loci=len(score),
                     frac_shared=frac, ranked_on=rank, note=note))

    print(f"  {backbone:12}{rung:18}{pt:>8.4f}  CI [{ci[0]:.4f}, {ci[1]:.4f}]  "
          f"n/class {m:>5}  ranked on {rank}")

def _norm_from_cache(path, chrom, pos, tag):
    """||h1-h2|| for the given loci, or (None, note) when the cache cannot provide them"""
    if not path or not os.path.exists(path):
        return None, f"{os.path.basename(path) if path else 'cache'} not found"
    g = load_pair_cache(path, chrom, pos, transform="none", tag=tag, min_ok_frac=0.0)
    ok = g["ok"]
    if not ok.any():
        return None, "zero loci joined"
    d = np.full(len(chrom), np.nan)
    d[ok] = np.linalg.norm(g["h1"][ok] - g["h2"][ok], axis=1) # ||h1-h2||
    return d, f"{int(ok.sum())}/{len(ok)} joined, D={g['D']}"

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assay", required=True)
    ap.add_argument("--pair_csv", required=True, help="the build's test.csv")
    ap.add_argument("--meta_csv", required=True, help="the matching test.meta.csv")
    ap.add_argument("--restrict_loci", required=True,
                    help="the shared locus set: any per-locus CSV with chr/pos.")
    ap.add_argument("--label_col", default="label")
    # DNABERT-2 rungs
    ap.add_argument("--raw_cache", default=None, help="pretrained DNABERT-2 pair cache")
    ap.add_argument("--trunk_cache", default=None, help="fine-tuned trunk pair cache")
    ap.add_argument("--head_dir", default=None,
                    help="trained HapAdapter head checkpoint")
    # AlphaGenome rungs
    ap.add_argument("--ag_cache", default=None, help="AlphaGenome test pair cache")
    ap.add_argument("--ag_perlocus", default=None,
                    help="fit_adapter's test_perlocus.csv (columns chr,pos,ell)")
    ap.add_argument("--ag_zeroshot", default=None,
                    help="score_alphagenome_api.py output (chr,pos,s1,s2,delta): AlphaGenome's "
                         "own predicted track, differenced across haplotypes.")
    ap.add_argument("--dnabert2ft_zeroshot", default=None,
                    help="score_dnabert2_ft.py output (same columns, same order): the "
                         "DNABERT-2 analogue, mu(hap1) - mu(hap2) from the fine-tuned trunk.")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_csv", required=True)
    a = ap.parse_args()

    pair = pd.read_csv(a.pair_csv)
    meta = pd.read_csv(a.meta_csv)
    if len(pair) != len(meta):
        raise SystemExit(f"{a.pair_csv} has {len(pair)} rows, {a.meta_csv} has {len(meta)} "
                         f"but these should have been written as a matched pair!")
    cc, pc = resolve_coord_columns(meta, a.meta_csv)

    want = _loci_from(a.restrict_loci, "restrict_loci")
    key = list(zip(meta[cc].astype(str), meta[pc].astype(int)))
    sel = np.array([k in want for k in key])

    if sel.sum() == 0:
        ex_w = sorted(want)[:2]
        raise SystemExit(
            f"ZERO of {len(meta)} build loci are in --restrict_loci ({len(want)} loci). "
            f"Build has e.g. {key[:2]}, restriction has e.g. {ex_w}.")

    print(f"[loci] {int(sel.sum())} of {len(meta)} build loci in the shared set "
          f"({len(want)} requested; {len(want) - int(sel.sum())} of those absent from this build)")

    pair, meta = pair[sel].reset_index(drop=True), meta[sel].reset_index(drop=True)
    y = (pair[a.label_col].to_numpy(float) > 0.5).astype(int)
    chrom, pos = meta[cc].astype(str), meta[pc].astype(int)
    n_shared = len(y) # every rung is measured against this, reported as frac_shared
    print(f"[loci] {n_shared} loci, {int(y.sum())} positive ({y.mean():.1%})\n")

    rows = []

    # DNABERT-2: the two embedding-distance rungs
    for path, rung in ((a.raw_cache, "pretrained-norm"), (a.trunk_cache, "trunk-norm")):
        d, note = _norm_from_cache(path, chrom, pos, rung) if path else (None, "not requested")
        ok = np.isfinite(d) if d is not None else None
        _rung(rows, a.assay, "dnabert2", rung,
              None if d is None else d[ok], None if d is None else y[ok], "distance", note,
              n_shared=n_shared)

    # AlphaGenome trunk norm
    d, note = (_norm_from_cache(a.ag_cache, chrom, pos, "ag-norm") if a.ag_cache
               else (None, "not requested"))
    ok = np.isfinite(d) if d is not None else None
    _rung(rows, a.assay, "alphagenome", "trunk-norm",
          None if d is None else d[ok], None if d is None else y[ok], "distance", note,
          n_shared=n_shared)

    # the per-locus score file
    for path, backbone, rung, rank, use_abs in (
            (a.ag_perlocus, "alphagenome", "hapadapter", "oriented logit", False),
            (a.ag_zeroshot, "alphagenome", "track-delta", "|log2 ratio|", True),
            (a.dnabert2ft_zeroshot, "dnabert2", "trunk-delta", "|mu1 - mu2|", True)):

        if not path or not os.path.exists(path):
            _rung(rows, a.assay, backbone, rung, None, None, rank,
                  "not requested" if not path else f"{os.path.basename(path)} not found",
                  n_shared=n_shared)
            continue

        t = pd.read_csv(path)
        col = next((c for c in ("ell", "delta", "score") if c in t.columns), None)
        if col is None:
            raise SystemExit(f"{path}: no ell/delta/score column; has {list(t.columns)}")

        tcc, tpc = resolve_coord_columns(t, path)
        m = pd.DataFrame(dict(chr=chrom.to_numpy(), pos=pos.to_numpy(), _y=y)).merge(
            t[[tcc, tpc, col]].rename(columns={tcc: "chr", tpc: "pos"}).assign(
                chr=lambda z: z["chr"].astype(str), pos=lambda z: z["pos"].astype(int)),
            on=["chr", "pos"], how="inner")
        s = m[col].to_numpy(float)

        if use_abs:
            s = np.abs(s)

        _rung(rows, a.assay, backbone, rung, s, m["_y"].to_numpy(int), rank,
              f"{len(m)}/{len(y)} joined", n_shared=n_shared)

    if a.head_dir:
        from hapadapter.loading import run_inference
        s1 = pair["sequence1"].astype(str).tolist()
        s2 = pair["sequence2"].astype(str).tolist()
        logits, _e, _rc = run_inference(a.head_dir, [[x, z] for x, z in zip(s1, s2)],
                                        a.batch_size, a.device, None)
        ell = np.asarray(logits, float).reshape(len(s1), -1)[:, 0]
        _rung(rows, a.assay, "dnabert2", "hapadapter", ell, y, "oriented logit",
              f"{len(ell)} loci", n_shared=n_shared)
    
    else:
        _rung(rows, a.assay, "dnabert2", "hapadapter", None, None, "oriented logit",
              "not requested", n_shared=n_shared)
    
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out_csv) or ".", exist_ok=True)
    out.to_csv(a.out_csv, index=False)
    print(f"\n[write] {a.out_csv} ({len(out)} rungs, "
          f"{int(out.auroc_plain.notna().sum())} with a number)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
