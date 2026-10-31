#!/usr/bin/env python
"""
entexbert2.analysis.score_external -- score ANY set of checkpoints on an EXTERNAL variant cohort
(ADASTRA or similar), on a single shared locus set.

This exists because the twin-vs-single comparison is only meaningful if both arms are scored on
IDENTICAL loci with an IDENTICAL leak filter and an IDENTICAL coverage vector. The script therefore
does the locus work ONCE, up front, and then loops checkpoints over that frozen set:

    external table -> label + coverage -> leak filter (shared) -> build_windows (shared)
                   -> for each checkpoint: dispatch on head_variant, score, coverage-matched AUROC

A per-checkpoint pipeline cannot give that guarantee, which is the flaw in the archived ADASTRA
numbers: the ref_single arm's leak filter was derived from its own bin-split metadata while the
paired arm used a different basis, so the two were never demonstrably on matched loci.

Writes one CSV: checkpoint, label, head_variant, auroc_cov_matched, auroc_plain, n, n_pos,
n_cov_matched, plus the shared-locus provenance columns.

File written by Amy Metrick in collaboration with Anthropic's Claude Science Opus 5 Agent
"""
import argparse
import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def head_variant_of(ckpt):
    """twin|single off the checkpoint. A MISSING DIRECTORY is a hard error (an unset shell variable
    yields a path like '/runs/.../seed20'); a directory that exists without run_config.json is the
    legacy case -> twin, matching the getattr default in model.py / model_io.py."""
    if not os.path.isdir(ckpt):
        raise FileNotFoundError(
            f"checkpoint does not exist: {ckpt!r}. If the path starts with '/' where you expected a "
            f"prefix, a shell variable was unset. Nothing has been scored.")
    p = os.path.join(ckpt, "run_config.json")
    if not os.path.exists(p):
        print(f"[head] WARNING: {ckpt} exists but has no run_config.json -- assuming "
              f"head_variant='twin' (legacy checkpoints predate the key)")
        return "twin"
    return json.load(open(p)).get("head_variant", "twin")


def derive_label(df, a):
    """Binary ASB label, either read directly or derived from per-allele FDR columns.

    Printed explicitly because the label definition is the single most consequential choice in an
    external comparison -- a different threshold is a different experiment.
    """
    if a.label_col:
        if a.label_col not in df.columns:
            raise ValueError(f"--label_col {a.label_col!r} not in {list(df.columns)[:12]}")
        y = (pd.to_numeric(df[a.label_col], errors="coerce") > 0.5).astype(int).to_numpy()
        print(f"[label] from column {a.label_col!r}: {y.sum()} positive / {len(y)-y.sum()} negative")
        return y
    missing = [c for c in a.fdr_cols if c not in df.columns]
    if missing:
        raise ValueError(f"--fdr_cols {missing} not in {list(df.columns)[:12]}; "
                         f"pass --label_col instead if the table already carries a binary label")
    fdr = np.column_stack([pd.to_numeric(df[c], errors="coerce").to_numpy() for c in a.fdr_cols])
    # ASB = significant imbalance toward EITHER allele -> min over the per-allele FDRs
    y = (np.nanmin(fdr, axis=1) < a.fdr_thresh).astype(int)
    print(f"[label] min({', '.join(a.fdr_cols)}) < {a.fdr_thresh}: "
          f"{y.sum()} positive / {len(y)-y.sum()} negative")
    return y


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--variants", required=True,
                    help="external cohort table (ADASTRA release TSV/CSV); one row per variant")
    ap.add_argument("--ref_fasta", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="one or more checkpoint dirs; each is dispatched on its own head_variant")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="display label per checkpoint (same order). Defaults to the dir basename")
    ap.add_argument("--out", required=True)

    g = ap.add_argument_group("external table columns")
    g.add_argument("--chrom_col", default="#chr")
    g.add_argument("--pos_col", default="pos")
    g.add_argument("--ref_col", default="ref")
    g.add_argument("--alt_col", default="alt")
    g.add_argument("--cov_col", default=None,
                   help="read-depth column for coverage matching. WITHOUT it the 'matched' AUROC "
                        "is class-balanced only and is NOT depth-controlled -- the comparison then "
                        "inherits exactly the confound this project exists to remove")
    g.add_argument("--pos_is_1based", action="store_true", default=True,
                   help="ADASTRA releases are 1-based (default)")
    g.add_argument("--pos_is_0based", dest="pos_is_1based", action="store_false")

    g2 = ap.add_argument_group("label definition")
    g2.add_argument("--label_col", default=None, help="use an existing binary column")
    g2.add_argument("--fdr_cols", nargs="+", default=["fdrp_bh_ref", "fdrp_bh_alt"],
                    help="per-allele FDR columns; label = min(these) < --fdr_thresh")
    g2.add_argument("--fdr_thresh", type=float, default=0.05)

    g3 = ap.add_argument_group("leakage control (SHARED across all checkpoints)")
    g3.add_argument("--train_meta", nargs="+", required=True,
                    help="the CONDITIONS' shared train/dev *.meta.csv -- every checkpoint in "
                         "--checkpoints must have been trained on this same split")
    g3.add_argument("--bin_size", type=int, default=100000)

    g4 = ap.add_argument_group("windows / scoring")
    g4.add_argument("--left_bp", type=int, default=128)
    g4.add_argument("--right_bp", type=int, default=128)
    g4.add_argument("--drop_refmismatch", action="store_true",
                    help="drop rows where the hg38 base disagrees with --ref_col")
    g4.add_argument("--batch_size", type=int, default=64)
    g4.add_argument("--device", default="cuda")
    g4.add_argument("--dry_run", action="store_true",
                    help="stop after the shared locus set is built; prints the label/coverage/leak/"
                         "window diagnostics without loading any model. RUN THIS FIRST.")
    a = ap.parse_args()

    from entexbert2.eval_utils import seen_bins_from_meta, flag_leaky, build_windows, balanced_auroc

    # ---- head variants first: a bad path must fail before any slow work ------------------------
    variants = [head_variant_of(c) for c in a.checkpoints]
    labels = a.labels or [os.path.basename(c.rstrip("/")) for c in a.checkpoints]
    if len(labels) != len(a.checkpoints):
        raise ValueError(f"--labels has {len(labels)} entries for {len(a.checkpoints)} checkpoints")
    for c, v, l in zip(a.checkpoints, variants, labels):
        print(f"[head] {l:28} {v:7} {c}")

    # ---- the shared locus set, built ONCE ------------------------------------------------------
    sep = "\t" if a.variants.endswith((".tsv", ".tsv.gz", ".txt")) else ","
    df = pd.read_csv(a.variants, sep=sep, low_memory=False)
    print(f"[load] {a.variants}: {len(df)} rows, {len(df.columns)} columns")
    need = [a.chrom_col, a.pos_col, a.ref_col, a.alt_col] + ([a.cov_col] if a.cov_col else [])
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"missing columns {miss}; table has {list(df.columns)[:14]}...")

    df = df.copy()
    df["_y"] = derive_label(df, a)

    if a.cov_col:
        df["_cov"] = pd.to_numeric(df[a.cov_col], errors="coerce")
        bad = int((~np.isfinite(df["_cov"])).sum())
        if bad:
            print(f"[cov] dropping {bad} rows with non-finite {a.cov_col}")
            df = df[np.isfinite(df["_cov"])].reset_index(drop=True)
        print(f"[cov] {a.cov_col}: median {df['_cov'].median():.1f} "
              f"IQR {df['_cov'].quantile(.25):.1f}-{df['_cov'].quantile(.75):.1f}")
        print(f"[cov] depth by class -- AS median {df.loc[df._y==1,'_cov'].median():.1f}, "
              f"non-AS median {df.loc[df._y==0,'_cov'].median():.1f} "
              f"(a large difference is exactly what coverage matching removes)")
    else:
        print("[cov] WARNING: no --cov_col; the 'matched' AUROC will be CLASS-BALANCED ONLY and "
              "NOT depth-controlled. The comparison then carries the depth confound.")

    seen = seen_bins_from_meta(a.train_meta, a.bin_size)
    if not seen:
        raise ValueError("--train_meta produced no bins; the leak filter would be a no-op. "
                         "Check the paths -- scoring an external cohort without a leak filter "
                         "against the training split is not a valid comparison.")
    leaky = flag_leaky(df, seen, a.bin_size, a.chrom_col, a.pos_col, a.pos_is_1based)
    print(f"[leakage] {int(leaky.sum())}/{len(df)} external loci fall in a training bin -> dropped")
    df = df[~leaky].reset_index(drop=True)

    seqs_ref, seqs_alt, keep = build_windows(
        df, a.ref_fasta, a.left_bp, a.right_bp,
        chrom_col=a.chrom_col, pos_col=a.pos_col, refbase_col=a.ref_col,
        a1_col=a.ref_col, a2_col=a.alt_col, pos_is_1based=a.pos_is_1based,
        drop_refmismatch=a.drop_refmismatch)

    df = df[keep].reset_index(drop=True)
    seqs_ref = [s for s, k in zip(seqs_ref, keep) if k]
    seqs_alt = [s for s, k in zip(seqs_alt, keep) if k]
    y = df["_y"].to_numpy(int)
    cov = df["_cov"].to_numpy(float) if a.cov_col else None
    assert len(seqs_ref) == len(seqs_alt) == len(y) == len(df), "shared locus set is inconsistent"

    print(f"\n[shared] {len(y)} loci survive to scoring | {int(y.sum())} AS / {int((1-y).sum())} non-AS "
          f"| pos_rate {y.mean():.4f}")
    print(f"[shared] EVERY checkpoint below is scored on exactly these {len(y)} loci, with one "
          f"leak filter and one coverage vector.")
    if y.sum() < 50:
        print(f"[shared] WARNING: only {int(y.sum())} positives -- AUROC CIs will be very wide")

    if a.dry_run:
        print("\n[dry_run] stopping before model loading. Re-run without --dry_run to score.")
        return

    # ---- score every checkpoint on the frozen set ----------------------------------------------
    from entexbert2.model_io import run_inference
    rows = []
    for ckpt, variant, lab in zip(a.checkpoints, variants, labels):
        if variant == "twin":
            texts = [[r, t] for r, t in zip(seqs_ref, seqs_alt)]   # [ref window, alt window]
        elif variant == "single":
            texts = list(seqs_ref)      # reference window only -- cannot see the allele by design
        else:
            raise ValueError(f"{ckpt}: unknown head_variant {variant!r}")

        logits, _emb, _rc = run_inference(ckpt, texts, a.batch_size, a.device)
        ell = np.asarray(logits).reshape(-1)
        assert len(ell) == len(y), f"{lab}: got {len(ell)} scores for {len(y)} loci"

        plain = float(roc_auc_score(y, ell)) if y.min() != y.max() else float("nan")
        m, _aupr, ci, n_m = balanced_auroc(ell, y, cover=cov)
        rows.append(dict(checkpoint=ckpt, label=lab, head_variant=variant,
                         auroc_cov_matched=float(m), auroc_plain=plain,
                         ci_lo=ci[0], ci_hi=ci[1], n=len(y), n_pos=int(y.sum()),
                         n_cov_matched=int(n_m), depth_controlled=bool(a.cov_col),
                         leak_bins=len(seen), bin_size=a.bin_size))
        print(f"  {lab:28} {variant:7} cov-matched={m:.4f} [{ci[0]:.3f},{ci[1]:.3f}]  "
              f"plain={plain:.4f}  n_cov_matched={n_m}")

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"\n[write] {a.out} ({len(out)} rows)")

    tw = out[out.head_variant == "twin"]["auroc_cov_matched"]
    sg = out[out.head_variant == "single"]["auroc_cov_matched"]
    if len(tw) and len(sg):
        print(f"[compare] twin best {tw.max():.4f} vs single best {sg.max():.4f} -> "
              f"{tw.max()-sg.max():+.4f} in {'TWIN' if tw.max() > sg.max() else 'SINGLE'}'s favour "
              f"(coverage-matched, {len(y)} shared loci)")


if __name__ == "__main__":
    main()
