#!/usr/bin/env python
"""
hapadapter.scoring.dump_trunk_pred
-- dump fine-tuned trunk predictions binding predictions (mu) vs actual label on the held-out test split per cell

Currently configured for SCALAR TRUNKS ONLY (run_config num_labels == 1)
"""
import argparse
import json
import os
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr

def preflight(checkpoint_dir):
    """
    Read run_config.json and refuse an unusable trunk before the forward pass
    (checkpoints with no run_config.json left to the post-inference check)
    """
    rcp = os.path.join(checkpoint_dir, "run_config.json")
    if not os.path.exists(rcp):
        print(f"[preflight] no run_config.json in {checkpoint_dir}; the track-count check "
              f"runs after the forward pass instead.")
        return None
    with open(rcp) as fh:
        rc = json.load(fh)
    if int(rc.get("num_labels", 1)) > 1:
        raise SystemExit(
            f"FATAL: {checkpoint_dir} is a MULTI-TRACK trunk (run_config "
            f"num_labels={rc.get('num_labels')}). This script currently only "
            f"supports a single-track trunk. Use a scalar (num_labels=1) trunk, or "
            f"extend this script.")
    task = str(rc.get("task", "")).lower()
    if task and task != "regression":
        raise SystemExit(
            f"FATAL: {checkpoint_dir} has task={task!r}. This script is currently "
            f"only configured for regression checkpoints.")
    print(f"[preflight] {os.path.basename(checkpoint_dir)}: task={task or 'unset'} "
          f"num_labels={rc.get('num_labels', 1)} -- usable")
    return rc

def main():
    from hapadapter.loading import run_inference # keeps --help torch-free

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True, help="trunk fine-tuned on regression task")
    ap.add_argument("--test_csv", required=True, help="the test set (just has sequence + label)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    df = pd.read_csv(a.test_csv)
    seqcol = next((c for c in ("sequence", "sequence1") if c in df.columns), None)
    missing = [n for n, c in (("a sequence column (sequence or sequence1)", seqcol),
                              ("label", "label" if "label" in df.columns else None)) if c is None]
    if missing:
        raise SystemExit(f"{a.test_csv}: needs {' and '.join(missing)}; "
                         f"present: {list(df.columns)}")

    preflight(a.checkpoint_dir)
    if os.path.dirname(a.out):
        os.makedirs(os.path.dirname(a.out), exist_ok=True)

    logits, _emb, rc = run_inference(a.checkpoint_dir, df[seqcol].astype(str).tolist(),
                                     a.batch_size, a.device, None)
    arr = np.asarray(logits)
    n_out = arr.shape[1] if arr.ndim > 1 else 1

    if int(rc.get("num_labels", 1)) > 1 or n_out > 1:
        raise SystemExit(
            f"FATAL: {a.checkpoint_dir} is a MULTI-TRACK trunk "
            f"(run_config num_labels={rc.get('num_labels')}, {n_out} output columns). This script "
            f"dumps a single scalar mu per locus and has no multi-track handling. Use a scalar "
            f"(num_labels=1) trunk.")

    mu = arr.reshape(-1)
    out = pd.DataFrame({"label": df["label"].astype(float).to_numpy(), "mu": mu})
    tmp = a.out + ".tmp"
    out.to_csv(tmp, index=False)
    os.replace(tmp, a.out)

    y, m = out["label"].to_numpy(), out["mu"].to_numpy()
    ok = np.isfinite(y) & np.isfinite(m)

    # pred vs. actual: r2 plus both correlations with their p-values (null = correlation of 0)
    n_ok = int(ok.sum())
    r2 = pear = pear_p = spear = spear_p = float("nan")
    if n_ok > 2:
        yy, mm = y[ok], m[ok]
        r2 = 1.0 - np.sum((yy - mm) ** 2) / max(np.sum((yy - yy.mean()) ** 2), 1e-12)
        if np.ptp(mm) == 0 or np.ptp(yy) == 0:
            print(f"[warn] {'mu' if np.ptp(mm) == 0 else 'label'} is constant over the "
                  f"{n_ok} finite loci -- correlations are undefined.")
        else:
            pr = pearsonr(mm, yy)
            sr = spearmanr(mm, yy)
            pear, pear_p = float(pr[0]), float(pr[1])
            spear, spear_p = float(sr.correlation), float(sr.pvalue)
    elif n_ok > 1:
        yy, mm = y[ok], m[ok]
        r2 = 1.0 - np.sum((yy - mm) ** 2) / max(np.sum((yy - yy.mean()) ** 2), 1e-12)
        print(f"[warn] only {n_ok} finite loci -- correlations need at least 3.")

    drop = len(out) - n_ok

    print(f"[trunk_pred] {len(out)} loci -> {a.out}  r2={r2:.3f} (on {n_ok} finite"
          + (f", {drop} dropped" if drop else "") + f")  mu[{mu.min():.2f},{mu.max():.2f}]")
    print(f"[corr]       pearson r={pear:+.4f} p={pear_p:.3g} | "
          f"spearman rho={spear:+.4f} p={spear_p:.3g}  (n={n_ok}; p tests rho=0 and assumes "
          f"independent loci)")

if __name__ == "__main__":
    main()
