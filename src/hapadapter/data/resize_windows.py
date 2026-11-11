#!/usr/bin/env python3
"""
hapadapter.data.resize_windows -- re-extract an existing build's loci at a DIFFERENT window
length, longer or shorter, without running a full build

Indel content measured:

    indel_bp_hapN = (lift(pos+right) - lift(pos-left)) - (left + right)
    indel_bp_ref  = max(|indel_bp_hap1|, |indel_bp_hap2|)  -- how far either haplotype departs from the ref span
    het_indel_bp  = |indel_bp_hap1 - indel_bp_hap2|        -- HETEROZYGOUS indel content

Usage:
  python -m hapadapter.data.resize_windows \\
    --src_csv      experiments/ctcf_ENC-002/stage2_personal_data/test.csv \\
    --src_meta_csv experiments/ctcf_ENC-002/stage2_personal_data/test.meta.csv \\
    --hap1_fasta .../enc002.hap1.fa --hap1_chain .../enc002.hap1.chain \\
    --hap2_fasta .../enc002.hap2.fa --hap2_chain .../enc002.hap2.chain \\
    --left 8192 --right 8191 --backbone alphagenome \\
    --n_neg_per_pos 1 --match_depth --split test --out_dir windows/ctcf_long
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

from hapadapter.backbones.cache import resolve_coord_columns

def _check_alphagenome(L: int) -> str:
    try:
        from alphagenome.models import dna_client
    except ImportError:
        return ("NOT CHECKED -- the alphagenome client is not installed in this environment.")
    try:
        dna_client.validate_sequence_length(L)
    except ValueError as e:
        sys.exit(f"{e}\n  left + 1 + right = {L}. Supported AlphaGenome lengths only.")
    return "accepted by alphagenome"

def _check_none(L: int) -> str:
    return ("NOT CHECKED. Pass --backbone to validate the length against your model backbone.")

# add for more backbones
LENGTH_CHECKS = {
    "alphagenome": _check_alphagenome,
    "none": _check_none,
}
_OVERLAP_LOSER = "dev" # keep sequence in train split if longer sequences led to train/dev leakage

def cross_split_overlap(this_split, chrom, pos, src_dir, left, right, chrom_col, pos_col):
    """
    Catalogues which (if any) of this split's loci have a window overlapping a different split's window
    ** two anchors a and b overlap iff |position a - position b| <= left + right **

    Returns (mask, detail) where
    - mask[i] is True if locus i overlaps
    - detail maps the other split name -> number of overlapping pairs found
    """
    span = left + right
    mine = pd.DataFrame({"chrom": np.asarray(chrom).astype(str),
                         "pos": np.asarray(pos).astype(int)})
    mask = np.zeros(len(mine), bool)
    detail = {}

    for other in ("train", "dev", "test"):
        if other == this_split:
            continue
        p = os.path.join(src_dir, f"{other}.meta.csv")
        if not os.path.exists(p):
            continue
        om = pd.read_csv(p)
        try:
            occ, opc = resolve_coord_columns(om, p)
        except ValueError:
            print(f"[overlap] {other}.meta.csv has no usable coordinates -- SKIPPED, so the "
                  f"overlap count below is a LOWER BOUND")
            continue

        hits = 0
        for ch, grp in mine.groupby("chrom"):
            op = np.sort(om.loc[om[occ].astype(str) == ch, opc].astype(int).to_numpy())
            if not len(op):
                continue
            mp = grp["pos"].to_numpy()
            # nearest other-split anchor on each side; overlap iff either is within span
            j = np.searchsorted(op, mp)
            near = np.full(len(mp), np.iinfo(np.int64).max, dtype=np.int64)
            left_ok = j > 0
            near[left_ok] = np.abs(mp[left_ok] - op[j[left_ok] - 1])
            right_ok = j < len(op)
            near[right_ok] = np.minimum(near[right_ok], np.abs(op[j[right_ok]] - mp[right_ok]))
            hit = near <= span
            mask[grp.index.to_numpy()[hit]] = True
            hits += int(hit.sum())
        detail[other] = hits

    return mask, detail

def neg_ratio(v):
    """argparse type for --n_neg_per_pos: a non-negative float, or 'none'/'all' meaning None"""
    if str(v).strip().lower() in ("none", "all"):
        return None
    try:
        f = float(v)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number or 'none'/'all', got {v!r}")
    if f < 0:
        raise argparse.ArgumentTypeError(f"--n_neg_per_pos must be >= 0, got {f}")
    return f

def select_loci(df, n_neg_per_pos, match_depth, cov_col, seed):
    """
    All positives, plus negatives drawn 1:n
    With --match_depth the negatives come from within the positives' depth deciles
    """
    pos = df.index[df["label"] == 1].to_numpy()
    neg = df.index[df["label"] == 0].to_numpy()
    rng = np.random.default_rng(seed)

    if n_neg_per_pos is None:
        keep = np.concatenate([pos, neg])
        print(f"[select] {len(pos)} positives + {len(neg)} negatives = {len(keep)} loci "
              f"(ALL kept, source balance preserved)")
        return df.loc[np.sort(keep)].reset_index(drop=True)

    want = int(round(n_neg_per_pos * len(pos)))

    if not match_depth or cov_col not in df.columns:
        if match_depth and cov_col not in df.columns:
            print(f"[select] no {cov_col!r} column -- falling back to a RANDOM negative draw")
        sel_neg = rng.choice(neg, size=min(want, len(neg)), replace=False)
    else:
        c = df[cov_col].to_numpy(float)
        edges = np.quantile(c[pos], np.linspace(0, 1, 11))
        edges[0], edges[-1] = -np.inf, np.inf
        pb = np.digitize(c[pos], edges[1:-1])
        nb = np.digitize(c[neg], edges[1:-1])
        picks, short = [], 0
        for b in range(10):
            k = int(round((pb == b).sum() * n_neg_per_pos))
            pool = neg[nb == b]
            if len(pool) < k:
                short += k - len(pool)
                k = len(pool)
            if k:
                picks.append(rng.choice(pool, size=k, replace=False))
        sel_neg = np.concatenate(picks) if picks else np.array([], dtype=int)
        if short:
            print(f"[select] {short} negatives unavailable in the matching depth deciles "
                  f"(positives skew deep) -- drew {len(sel_neg)} of {want}")

    keep = np.concatenate([pos, sel_neg])
    print(f"[select] {len(pos)} positives + {len(sel_neg)} negatives = {len(keep)} loci "
          f"({'depth-decile matched' if match_depth else 'random negatives'})")
    return df.loc[np.sort(keep)].reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src_csv", required=True,
                    help="an existing short build's split CSV -- train.csv, dev.csv or test.csv")
    ap.add_argument("--src_meta_csv", required=True, help="its matching .meta.csv")
    ap.add_argument("--hap1_fasta", required=True)
    ap.add_argument("--hap1_chain", required=True)
    ap.add_argument("--hap2_fasta", required=True)
    ap.add_argument("--hap2_chain", required=True)
    ap.add_argument("--left", type=int, default=8192)
    ap.add_argument("--right", type=int, default=8191,
                    help="window is left + 1 + right bp; see --backbone for validation")
    ap.add_argument("--backbone", default="none", choices=sorted(LENGTH_CHECKS),
                    help="which foundation model will serve as the backbone for the HapAdapter head")
    ap.add_argument("--n_neg_per_pos", type=neg_ratio, default=1.0, metavar="N|none",
                    help="negatives per positive (1:1 balanced default). Pass 'none' "
                         "to TRAIN a head on these windows")
    ap.add_argument("--match_depth", action="store_true",
                    help="draw negatives from within the positives' depth deciles")
    ap.add_argument("--cov_col", default="depth")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="PILOT: first N selected loci")
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"],
                    help="names the output files. Must match what --src_csv actually is")
    ap.add_argument("--drop_cross_split_overlap", action="store_true",
                    help="drop loci whose resized window overlaps a locus across train/dev. "
                         f"In those cases, only the {_OVERLAP_LOSER!r} side is dropped.")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    if a.n_neg_per_pos is None and a.match_depth:
        raise SystemExit(
            "--match_depth with --n_neg_per_pos none asks for two incompatible things: "
            "depth matching works by SUBSAMPLING negatives to the positives' depth deciles, "
            "and 'none' keeps every negative. Drop --match_depth, or give a ratio.")

    L = a.left + 1 + a.right
    print(f"[window] left {a.left} + 1 + right {a.right} = {L} bp "
          f"-- {LENGTH_CHECKS[a.backbone](L)}")

    df = pd.read_csv(a.src_csv)
    meta = pd.read_csv(a.src_meta_csv)
    if len(df) != len(meta):
        sys.exit(f"row mismatch: {len(df)} in {os.path.basename(a.src_csv)} vs {len(meta)} in "
                 f"{os.path.basename(a.src_meta_csv)}")

    carry = [c for c in ("chr", "chrom", "anchor", "ref_start", "ref_allele", "hap1_allele",
                         "hap2_allele", "total_reads", "weight_depth") if c in meta.columns]
    keepcols = [c for c in ("label", a.cov_col) if c in df.columns]

    src = pd.concat([df[keepcols].reset_index(drop=True),
                     meta[carry].reset_index(drop=True)], axis=1)
    src = src.loc[:, ~src.columns.duplicated()]

    for c in ("label", "ref_allele", "hap1_allele", "hap2_allele"):
        if c not in src.columns:
            sys.exit(f"missing {c!r}; have {list(src.columns)}")

    chrom_col, pos_col = resolve_coord_columns(src, a.src_meta_csv)
    print(f"[load] {len(src)} source loci | chroms "
          f"{sorted(src[chrom_col].astype(str).unique())} | pos_rate {src['label'].mean():.4f}")

    # cross-split dedup (before the negative draw so the count is over the whole split)
    ov_mask, ov_detail = cross_split_overlap(
        a.split, src[chrom_col], src[pos_col], os.path.dirname(a.src_meta_csv) or ".",
        a.left, a.right, chrom_col, pos_col)
    n_ov = int(ov_mask.sum())
    if ov_detail:
        print(f"[overlap] at {L} bp, {n_ov} of {len(src)} {a.split} loci ({n_ov/len(src):.1%}) "
              f"have a window overlapping another split: {ov_detail}")
    if n_ov and a.split == _OVERLAP_LOSER and a.drop_cross_split_overlap:
        src = src.loc[~ov_mask].reset_index(drop=True)
        print(f"[overlap] dropped -> {len(src)} loci remain")
    elif n_ov and a.split == _OVERLAP_LOSER:
        print(f"[overlap] NOT dropped. Pass --drop_cross_split_overlap to remove them; a head "
              f"selected on this dev set has otherwise seen train sequence.")
    elif n_ov:
        print(f"[overlap] nothing dropped here by design -- the {_OVERLAP_LOSER!r} side is the "
              f"one that loses loci. Run --split {_OVERLAP_LOSER} "
              f"--drop_cross_split_overlap to close it.")

    src = select_loci(src, a.n_neg_per_pos, a.match_depth, a.cov_col, a.seed)
    if a.limit:
        src = src.head(a.limit).reset_index(drop=True)
        print(f"[pilot] --limit {a.limit}: {len(src)} loci")

    # personal genome handling, compute indels
    import pyfaidx
    from hapadapter.data.personal_seq import PersonalGenome, parse_chain_file, lift_point
    print("[genome] parsing chains ...", flush=True)
    c1, c2 = parse_chain_file(a.hap1_chain), parse_chain_file(a.hap2_chain)
    f1 = pyfaidx.Fasta(a.hap1_fasta, as_raw=True, sequence_always_upper=True)
    f2 = pyfaidx.Fasta(a.hap2_fasta, as_raw=True, sequence_always_upper=True)
    pg = PersonalGenome(c1, f1, c2, f2, donor=None, validate=True)

    def net_indel(chain, chrom, p0):
        """(lifted span) - (reference span); None if an edge will not lift"""
        lo = lift_point(chain, chrom, p0 - a.left)
        hi = lift_point(chain, chrom, p0 + a.right)
        if lo is None or hi is None or lo[0] != hi[0]:
            return None
        return (hi[1] - lo[1]) - (a.left + a.right)

    rows, fails = [], {}
    for i, r in enumerate(src.itertuples()):
        ch, p0 = str(getattr(r, chrom_col)), int(getattr(r, pos_col))
        res, why = pg.pair_windows(ch, p0, r.hap1_allele, r.hap2_allele,
                                   left=a.left, right=a.right)
        if res is None:
            k = str(why).split("(")[0].strip()
            fails[k] = fails.get(k, 0) + 1
        else:
            d1, d2 = net_indel(c1, ch, p0), net_indel(c2, ch, p0)
            rec = dict(sequence1=res["seq1"], sequence2=res["seq2"], label=int(r.label))
            if a.cov_col in src.columns:
                rec[a.cov_col] = float(getattr(r, a.cov_col))
            rec.update({chrom_col: ch, pos_col: p0, "ref_allele": r.ref_allele,
                        "hap1_allele": r.hap1_allele, "hap2_allele": r.hap2_allele,
                        "offset": int(res["offset"]), "swapped": bool(res["swapped"]),
                        "indel_bp_hap1": (np.nan if d1 is None else int(d1)),
                        "indel_bp_hap2": (np.nan if d2 is None else int(d2))})
            rows.append(rec)
        if (i + 1) % 200 == 0:
            print(f"[extract] {i+1}/{len(src)} | kept {len(rows)}", flush=True)

    if not rows:
        sys.exit("no loci extracted -- check the chain/FASTA pairing and the allele columns")
    out = pd.DataFrame(rows)
    pair = out[["indel_bp_hap1", "indel_bp_hap2"]].to_numpy(float)
    both_nan = np.all(np.isnan(pair), axis=1) # both NaN -> "All-NaN slice" is expected behavior
    ref_dep = np.full(len(pair), np.nan)
    if (~both_nan).any():
        ref_dep[~both_nan] = np.nanmax(np.abs(pair[~both_nan]), axis=1)
    out["indel_bp_ref"] = ref_dep
    out["het_indel_bp"] = np.abs(pair[:, 0] - pair[:, 1])

    print(f"\n[extract] kept {len(out)} of {len(src)} | allele match rate {pg.match_rate():.3f}")
    if fails:
        print(f"[extract] failures by reason: {fails}")
        lost = len(src) - len(out)
        if lost > 0.05 * len(src):
            print(f"[extract] WARNING {lost/len(src):.1%} of loci dropped. At {L} bp this is "
                  f"large enough that the survivors may not represent the source split -- "
                  f"check the reasons above before comparing this arm to a short-window one.")
    short = int((out["sequence1"].str.len() != L).sum() + (out["sequence2"].str.len() != L).sum())
    if short:
        sys.exit(f"{short} sequences are not {L} bp -- extract_window returned a short window")
    nonacgt = int(out["sequence1"].map(lambda s: not set(s) <= set("ACGTN")).sum() +
                  out["sequence2"].map(lambda s: not set(s) <= set("ACGTN")).sum())
    if nonacgt:
        print(f"[extract] WARNING: {nonacgt} sequences contain non-ACGTN characters! "
        "Some backbones (e.g. AlphaGenome will throw an error on these sequences)")

    h = out["het_indel_bp"]
    print(f"\n[indel] HETEROZYGOUS indel content in the window:")
    print(f"        resolved for {int(h.notna().sum())}/{len(out)} loci "
          f"({int(h.isna().sum())} had an unliftable window edge)")
    for lo, hi, lab in ((0, 1, "0 (none)"), (1, 128, "1-127 (sub-bin)"),
                        (128, 1024, "128-1023"), (1024, np.inf, ">=1024")):
        m = (h >= lo) & (h < hi)
        pr = out.loc[m, "label"].mean() if bool(m.any()) else float("nan")
        print(f"        {lab:16} n={int(m.sum()):>6}  pos_rate={pr:.4f}")
    print(f"[indel] departure from the reference span (indel_bp_ref): "
          f"median {out['indel_bp_ref'].median():.0f} max {out['indel_bp_ref'].max():.0f}")

    os.makedirs(a.out_dir, exist_ok=True)
    seq_cols = (["sequence1", "sequence2", "label"]
                + ([a.cov_col] if a.cov_col in out.columns else []))
    p_seq = os.path.join(a.out_dir, f"{a.split}.csv")
    p_meta = os.path.join(a.out_dir, f"{a.split}.meta.csv")
    out[seq_cols].to_csv(p_seq, index=False)
    out[[c for c in out.columns if c not in ("sequence1", "sequence2")]].to_csv(p_meta, index=False)

    print(f"\n[write] {p_seq} ({os.path.getsize(p_seq)/1e6:.0f} MB)")
    print(f"[write] {p_meta}")

if __name__ == "__main__":
    main()
