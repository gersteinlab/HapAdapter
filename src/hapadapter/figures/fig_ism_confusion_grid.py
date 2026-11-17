#!/usr/bin/env python
"""
Per-assay GRID of head-ISM allelic saliency, split by the confusion 2x2 (TP/TN/FP/FN).

One subplot per assay; each plots the mean per-position saliency of loci in each confusion class,
centered on the variant (x = position - center). Reproduces the "Attention Scores" small-multiples
layout but from ISM saliency + the model's own calls. Runs for ONE arm (reference or personal): the
reference grid reads ism_head_ref_{AS,nonAS}.npz + the ref perVariant; the personal panel reads
ism_head_personal_{AS,nonAS}.npz + the personal perVariant (personal-window scores).

Confusion assignment (identical to fig_ism_aggregate Panel B, per assay):
  truth      = npz as_label (AS|nonAS)
  prediction = perVariant score_col joined by (chr, anchor)==(chr, ref_start); predicted-AS = ell > thr
  threshold  = 0 by default (the model's own p=0.5 boundary, signed ell; comparable across assays);
               'auto' = per-assay Youden-J (unstable on weak heads); or a shared float.
The per-assay join HIT-RATE and class counts are printed; a low hit-rate = coord off-by-one (the
+/-1 rates are printed) or the ISM loci barely overlap the eval set (FP/FN then tiny -> noisy).

  python fig_ism_confusion_grid.py --exp_root $HOME/palmer_scratch/entexbert2-models/experiments \
      --assays CTCF,EP300,POLR2A,POLR2AphosphoS5,ATAC --donor ENC-002 --arm ref \
      --title "Reference genome, ENC-002" --window 75 --out fig_ism_confusion_grid_ref.png
Per-assay perVariant defaults to <cell>/eval/score_<arm>_hetsnv_perVariant.csv.gz; override the
directory with --scores_dir (files named <assay_lower>_<arm>_perVariant.csv.gz there).
"""
import argparse, os, glob, math
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# TP blue / FP green / FN orange / TN red -- matches the reference figure; CVD-distinguishable.
CONF = {"TP": ("#4c72b0", "-"), "FP": ("#55a868", "-"), "FN": ("#dd8452", "--"), "TN": ("#c44e52", "--")}


def youden_threshold(score, truth):
    """Operating point on the AS-vs-nonAS ROC of `score` maximizing TPR - FPR."""
    order = np.argsort(-score)
    s, t = score[order], truth[order].astype(int)
    P, N = max(int(t.sum()), 1), max(int((1 - t).sum()), 1)
    tpr = np.cumsum(t) / P
    fpr = np.cumsum(1 - t) / N
    j = np.argmax(tpr - fpr)
    return s[j]


def load_arm_npz(cell, arm):
    """Concatenate the AS + nonAS head npzs for one arm -> (importance N×L, as_label, chr, anchor)."""
    sal, aslab, chrom, anchor = [], [], [], []
    for cls in ("AS", "nonAS"):
        p = os.path.join(cell, "ism", f"ism_head_{arm}_{cls}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=True)
        imp = z["importance"]
        n = imp.shape[0]
        sal.append(imp)
        aslab.append(z["as_label"].astype(str) if "as_label" in z else np.array([cls] * n))
        chrom.append(z["chr"].astype(str) if "chr" in z else np.array(["?"] * n))
        anchor.append(z["anchor"].astype(np.int64) if "anchor" in z else np.full(n, -1))
    if not sal:
        return None
    return (np.concatenate(sal), np.concatenate(aslab),
            np.concatenate(chrom), np.concatenate(anchor))


def clean_universe(pv_csv, drop_leaky, donor_kind):
    """The set of clean (chr, ref_start) coords a perVariant defines -- used to hold every arm to
    the SAME held-out, leak-free loci so ref vs personal is apples-to-apples."""
    sc = pd.read_csv(pv_csv)
    if donor_kind != "all" and "donor_kind" in sc.columns:
        sc = sc[sc["donor_kind"] == donor_kind]
    if drop_leaky and "leaky" in sc.columns:
        sc = sc[sc["leaky"] == 0]
    return {(str(c), int(p)) for c, p in zip(sc["chr"], sc["ref_start"])}


def assign_confusion(aslab, chrom, anchor, pv_csv, score_col, thr_arg, donor_kind,
                     drop_leaky=False, universe=None):
    """Join npz loci -> perVariant prediction; return (masks dict, thr, hit_rate, counts, offby, nkept).
    drop_leaky: drop rows with leaky==1 from this arm's perVariant. universe: if given, keep only loci
    whose (chr, anchor) is in this shared clean coord set (both arms -> identical loci)."""
    sc = pd.read_csv(pv_csv)
    if donor_kind != "all" and "donor_kind" in sc.columns:
        sc = sc[sc["donor_kind"] == donor_kind]
    if drop_leaky and "leaky" in sc.columns:
        sc = sc[sc["leaky"] == 0]
    agg = sc.groupby(["chr", "ref_start"])[score_col].mean()
    key = {(str(c), int(p)): v for (c, p), v in agg.items()}
    def hits_at(shift):
        return np.array([key.get((c, int(an) + shift), np.nan) for c, an in zip(chrom, anchor)])
    pred = hits_at(0)
    hit = np.isfinite(pred)
    rate = float(hit.mean()) if len(hit) else 0.0
    offby = {sh: float(np.isfinite(hits_at(sh)).mean()) for sh in (-1, 1)} if rate < 0.5 else {}
    if universe is not None:                       # restrict to the shared clean held-out coord set
        inuni = np.array([(str(c), int(an)) in universe for c, an in zip(chrom, anchor)])
        hit = hit & inuni
    is_as = (aslab == "AS")
    # SIGNED ell: score_col=delta is the classification logit (higher = more ASB), so the model's own
    # call is ell>0 (p=0.5). Thresholding on |ell| corrupts the split wherever the learned bias b is
    # negative, and per-assay Youden swings to degenerate extremes on weak heads (H3K4me1). Default 0.
    if thr_arg == "auto":
        thr = youden_threshold(pred[hit], is_as[hit].astype(int)) if hit.any() else np.inf
    else:
        thr = float(thr_arg)
    pas = pred > thr
    masks = {"TP": hit & is_as & pas, "FN": hit & is_as & ~pas,
             "FP": hit & ~is_as & pas, "TN": hit & ~is_as & ~pas}
    counts = {k: int(m.sum()) for k, m in masks.items()}
    return masks, thr, rate, counts, offby, int(hit.sum())


def load_assay_confusion(exp_root, assay, donor, arm, *, scores_dir=None, score_col="ell",
                         threshold=None, donor_kind="all", universe_mode="ref",
                         keep_leaky=False, verbose=True):
    """
    Everything one panel needs for one assay, or (None, reason) when an input is absent.

    Returns ((sal, masks, counts), None) on success and (None, "<why>") otherwise, so the
    caller decides whether a missing cell is a placeholder panel or a hard error. Split out of
    main() so the composite figure and the standalone grid share the loading, joining and
    thresholding -- the part where a second implementation would silently pick a different
    threshold or a different held-out universe and produce a plausible but different panel.
    """
    cell = os.path.join(exp_root, f"{assay.lower()}_{donor}")
    loaded = load_arm_npz(cell, arm)
    if loaded is None:
        return None, f"no {arm} npz"
    sal, aslab, chrom, anchor = loaded

    def pv_path(which):
        return (os.path.join(scores_dir, f"{assay.lower()}_{which}_perVariant.csv.gz")
                if scores_dir
                else os.path.join(cell, "eval", f"score_{which}_hetsnv_perVariant.csv.gz"))

    pv = pv_path(arm)
    if not os.path.exists(pv):
        return None, "no perVariant"
    drop_leaky = not keep_leaky
    uni = None                     # shared clean held-out universe (from the reference eval)
    if universe_mode == "ref" and os.path.exists(pv_path("ref")):
        uni = clean_universe(pv_path("ref"), drop_leaky, donor_kind)
    masks, thr, rate, counts, offby, nkept = assign_confusion(
        aslab, chrom, anchor, pv, score_col, threshold, donor_kind,
        drop_leaky=drop_leaky, universe=uni)
    if verbose:
        uni_s = f" universe={len(uni)}" if uni is not None else ""
        msg = (f"[{assay}] join hit-rate={rate:.2f}{uni_s} kept={nkept} "
               f"thr={thr:.3g} counts={counts}")
        if offby:
            msg += f"  offby={offby}"
        print(msg)
    return (sal, masks, counts), None


def draw_confusion_panel(ax, sal, masks, counts, *, window=100, smooth=0, title=None,
                         fontsize=None, legend=False, xlabel=None, ylabel=None):
    """Draw the four confusion-class saliency profiles onto an EXISTING axis."""
    fs = fontsize
    L = sal.shape[1]; c = L // 2
    x = np.arange(L) - c
    keep = np.abs(x) <= window
    for name in ("TP", "TN", "FP", "FN"):
        m = masks[name]
        if m.sum() == 0:
            continue
        y = (gaussian_filter1d(np.nanmean(sal[m], axis=0), smooth) if smooth
             else np.nanmean(sal[m], axis=0))
        col, ls = CONF[name]
        ax.plot(x[keep], y[keep], color=col, ls=ls, lw=1.3,
                label=f"{name} (n={counts[name]})")
    if title is not None:
        ax.set_title(title, loc="center", fontsize=(10 if fs is None else fs + 1))
    ax.axvline(0, color="0.85", lw=0.8, zorder=0)
    ax.tick_params(labelsize=(8 if fs is None else fs - 1))
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=(8 if fs is None else fs))
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=(8 if fs is None else fs))
    if legend:
        ax.legend(fontsize=(8 if fs is None else fs - 1), framealpha=0.9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_root", required=True)
    ap.add_argument("--assays", required=True, help="comma list, e.g. CTCF,EP300,POLR2A,POLR2AphosphoS5,ATAC")
    ap.add_argument("--donor", required=True)
    ap.add_argument("--arm", default="ref", choices=["ref", "personal"])
    ap.add_argument("--scores_dir", default=None,
                    help="dir of <assay_lower>_<arm>_perVariant.csv.gz; default <cell>/eval/score_<arm>_hetsnv_perVariant.csv.gz")
    ap.add_argument("--score_col", default="delta", help="prediction column (head: delta = ASB logit)")
    ap.add_argument("--donor_kind", default="all", help="filter perVariant to this donor_kind")
    ap.add_argument("--threshold", default="0",
                    help="signed ell cutoff: '0' = the model's own p=0.5 boundary (default, comparable "
                         "across assays); 'auto' = per-assay Youden J (unstable on weak heads); or a float")
    ap.add_argument("--universe", default="ref", choices=["ref", "self", "none"],
                    help="hold every arm to the REFERENCE perVariant's clean coords (default) so ref vs "
                         "personal use IDENTICAL loci; 'self'=each arm's own loci; 'none'=no restriction")
    ap.add_argument("--keep_leaky", action="store_true", help="keep leaky==1 loci (default: drop them)")
    ap.add_argument("--window", type=int, default=75, help="+/- bp around the variant to plot")
    ap.add_argument("--smooth", type=float, default=3.0)
    ap.add_argument("--ncols", type=int, default=4)
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    assays = [s for s in a.assays.split(",") if s]
    ncols = a.ncols
    nrows = math.ceil(len(assays) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.5 * nrows),
                             sharex=True, squeeze=False)
    axes_flat = axes.ravel()

    for ax in axes_flat[len(assays):]:
        ax.axis("off")

    for ax, assay in zip(axes_flat, assays):
        payload, why = load_assay_confusion(
            a.exp_root, assay, a.donor, a.arm, scores_dir=a.scores_dir,
            score_col=a.score_col, threshold=a.threshold, donor_kind=a.donor_kind,
            universe_mode=a.universe, keep_leaky=a.keep_leaky)
        if payload is None:
            ax.text(0.5, 0.5, f"{assay}\n({why})", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="0.5")
            ax.set_title(assay, loc="center")
            continue
        sal, masks, counts = payload
        draw_confusion_panel(ax, sal, masks, counts, window=a.window, smooth=a.smooth,
                             title=assay)

    # one shared legend (class colours), figure-level
    handles = [plt.Line2D([0], [0], color=CONF[k][0], ls=CONF[k][1], lw=1.6) for k in ("TP", "FP", "FN", "TN")]
    fig.legend(handles, ["TP", "FP", "FN", "TN"], loc="lower center", ncol=4, frameon=False, fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("position relative to variant (bp)", fontsize=8)
    for r in range(nrows):
        axes[r][0].set_ylabel("mean saliency", fontsize=8)
    if a.title:
        fig.suptitle(f"Head-ISM allelic saliency by TP/FP/FN/TN — {a.title}", fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96 if a.title else 1.0])
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print(f"[wrote] {a.out}  ({len(assays)} assays, arm={a.arm})")


if __name__ == "__main__":
    main()
