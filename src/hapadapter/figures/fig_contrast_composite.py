#!/usr/bin/env python3
"""
hapadapter.figures.fig_contrast_composite -- one ROW of four diagnostic panels per
(backbone, assay) cell, from the npz files dump_contrast_embeddings writes.

Each row reads ONE npz and renders, left to right:

  1  TRUNK contrast norm  ||h1-h2||        ECDF by class. What the frozen trunk already
                                           separates, BEFORE the learned projection.
  2  HEAD contrast norm   s = ||P(h1)-P(h2)||
                                           ECDF by class, with the decision radius -b/a
                                           marked when the npz carries the link scalars.
  3  decision axis        ell = logit P(ASB)
                                           histogram by class, with p=0.5 (ell=0).
  4  distance gained      z(head) - z(trunk)
                                           violin by class. Raw head-minus-trunk is
                                           scale-confounded (different spaces), so each is
                                           standardised across loci first.

Panels 1, 2 and 4 are taken unchanged from plot_contrast_composite_cell.py and
fig_contrast_violin_grid.py, so a row here is directly comparable to the older per-cell
figures rather than a re-derivation.

PATHS ARE EXPLICIT, which is the point of this script. The per-cell version composed
<exp_root>/<assay>_<donor>/eval/<npz_name> from four flags, which only works for cells laid
out that way -- not for a rebuild tree, a scratch directory, or two backbones side by side.
Here each row is one --row argument carrying its own path.

Usage:
    python -m hapadapter.figures.fig_contrast_composite \\
      --row "DNABERT-2:CTCF:ENC-002:/path/ctcf_eb2.npz" \\
      --row "DNABERT-2:H3K27ac:ENC-002:/path/h3k27ac_eb2.npz" \\
      --row "AlphaGenome:CTCF:ENC-002:/path/ctcf_ag.npz" \\
      --out fig_contrast_composite.png --csv_out fig_contrast_composite.csv

    # trailing descriptors are optional -- these are all valid
    --row "DNABERT-2:CTCF:/path/x.npz"      # no individual
    --row "CTCF:/path/x.npz"                # assay only

File written by Amy Metrick in collaboration with Anthropic's Claude Science Opus 5 Agent
"""
import argparse
import os
import re
import sys

import numpy as np

#: Class colours, matched to the earlier contrast figures so rows cut from this script and
#: the per-cell ones can sit in the same panel without a palette change.
#: A HUE PAIR PER BACKBONE, COOL = non-AS AND WARM = AS IN BOTH. The comparison the eye makes
#: most on this figure is AS against non-AS inside one panel, so that pair gets the stronger
#: encoding: two different hues rather than two lightnesses of one. The backbone is then carried
#: by WHICH pair -- blue/red for DNABERT-2, purple/orange for AlphaGenome.
#:
#: THE TRADE THIS MAKES, STATED. Hue-for-class means the two backbones' non-AS colours (blue,
#: purple) and their two AS colours (red, orange) are closer to each other than the old
#: lightness scheme made them, and under deuteranopia red/orange in particular converge. That
#: is acceptable here ONLY because backbone identity never rests on colour: every row carries a
#: text label at the left edge naming it, and the key at the foot spells out both pairs. Class
#: identity, which does rest on colour within a panel, is the one that stays unambiguous --
#: blue vs red and purple vs orange both survive the common CVD types and greyscale.
#:
#: Cool-for-non-AS is held across both families on purpose: a reader who learns the mapping on
#: the first row should not have to relearn it on the second.
BACKBONE_COLORS = {
    "dnabert":     {"nonas": "#3b6fb6", "as": "#c0392b", "key": "#c0392b"},
    "alphagenome": {"nonas": "#6a4c93", "as": "#e08214", "key": "#e08214"},
}
#: Neutral rather than a third pair: an unrecognised backbone should look unstyled, not like a
#: deliberate third family the reader is meant to interpret.
DEFAULT_COLORS = {"nonas": "#95a5a6", "as": "#2c3e50", "key": "#2c3e50"}


def row_colors(backbone):
    """Colour family for a row, matched on a substring of the backbone descriptor."""
    key = str(backbone or "").lower().replace("-", "").replace(" ", "")
    for frag, cols in BACKBONE_COLORS.items():
        if frag in key:
            return cols
    return DEFAULT_COLORS


def _auroc(y, score):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, score) if 0 < y.sum() < len(y) else float("nan")


def _softplus(x):
    # log(1+e^x), numerically stable; no scipy dependency
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def zscore(v):
    s = v.std()
    return (v - v.mean()) / s if s > 0 else v - v.mean()


def resolve_ab(d):
    """
    The trained logistic-link scalars a>0 and b for ell = a*s + b, so the decision boundary
    p=0.5 (ell=0) maps to a distance radius -b/a on the head-norm axis.

    Returns (a, b), or (None, None) when the npz has neither pair -- which is the normal case
    for a SINGLE-variant head, since it has no distance scalars at all.
    """
    keys = set(d.files) if hasattr(d, "files") else set(d.keys())

    def _get(k):
        return float(np.asarray(d[k]).reshape(-1)[0]) if k in keys else None

    b = _get("b") if _get("b") is not None else _get("dist_b")
    if "a" in keys:
        a = _get("a")
    elif "a_raw" in keys or "dist_a" in keys:
        raw = _get("a_raw") if "a_raw" in keys else _get("dist_a")
        a = float(_softplus(np.array(raw)))
    else:
        a = None
    if a is not None and b is not None and a > 0:
        return a, b
    return None, None


def ecdf(ax, val, y, title, xlabel, colors=None, logx="auto", vline=None, vlabel=None, fs=8):
    """
    ECDF of `val` by class; separation is the vertical gap between the two curves.

    logx="auto" uses a log x-axis only when the values span more than ~1.5 decades: the head
    norm has a zero-pileup plus a long tail and needs it, the trunk norm sits in a narrow band
    where log-x only crowds the ticks.
    """
    cols = colors or DEFAULT_COLORS
    for lab, c, name in [(0, cols["nonas"], "non-AS"), (1, cols["as"], "AS")]:
        v = np.sort(val[y == lab])
        f = np.arange(1, len(v) + 1) / len(v)
        ax.plot(v, f, color=c, lw=1.8, label=f"{name} (n={int((y == lab).sum())})")
    pos = val[val > 0]
    if logx == "auto":
        span = (np.log10(np.percentile(pos, 99.5)) - np.log10(np.percentile(pos, 1))) \
            if len(pos) else 0
        logx = span > 1.5
    if logx:
        ax.set_xscale("log")
        lo = max(np.percentile(pos, 0.5), 1e-6) if len(pos) else 1e-6
        ax.set_xlim(lo, np.percentile(val, 99.9))
    else:
        lo, hi = np.percentile(val, [0.2, 99.5])
        ax.set_xlim(lo, hi)
    au = _auroc(y, val)
    if vline is not None and vline > 0:
        ax.axvline(vline, color="k", lw=1.2, ls="--", alpha=0.8,
                   label=(vlabel or "decision radius"))
    elif vlabel is not None:
        # SAY WHY THE LINE IS ABSENT. With several rows stacked, a panel missing the dashed
        # boundary that its neighbours have reads as a rendering failure. It usually is not:
        # a single-variant head has no distance scalars, so there is no radius to draw.
        ax.plot([], [], " ", label=vlabel)
    ax.set_xlabel(xlabel, fontsize=fs)
    ax.set_ylabel("Cumulative fraction", fontsize=fs)
    # The class medians and the exact decision radius are NOT printed on the figure: they go
    # to --csv_out and belong in the caption. A title carrying three numbers is read as a
    # label rather than read for its numbers, and the AUROC is the one that earns the space.
    ax.set_title(f"{title}\nAUROC = {au:.3f}", fontsize=fs)
    ax.legend(loc="upper left", frameon=False, fontsize=fs - 1)
    ax.grid(alpha=0.25, which="both")
    ax.tick_params(labelsize=fs - 1)
    return au, float(np.median(val[y == 1])), float(np.median(val[y == 0]))


def ell_hist(ax, ell, y, row_label, colors=None, fs=8):
    lo, hi = np.percentile(ell, [0.5, 99.5])
    bins = np.linspace(lo, hi, 45)
    cols = colors or DEFAULT_COLORS
    ax.hist(ell[y == 0], bins=bins, color=cols["nonas"], alpha=0.55, density=True, label="non-AS")
    ax.hist(ell[y == 1], bins=bins, color=cols["as"], alpha=0.55, density=True, label="AS")
    # SAME boundary as delta* = -b/a on the head-norm axis, which is why panel 2's dashed line
    # and this one are the same decision, drawn in two coordinate systems.
    ax.axvline(0.0, color="k", lw=1.2, ls="--", alpha=0.8, label="decision boundary")
    au = _auroc(y, ell)
    ax.set_xlabel("Model logit  (dashed: decision boundary)", fontsize=fs)
    ax.set_ylabel("Density", fontsize=fs)
    # Short, like its siblings: the row label carries which cell this is.
    ax.set_title(f"Head logit\nAUROC = {au:.3f}", fontsize=fs)
    ax.legend(loc="best", frameon=False, fontsize=fs - 1)
    ax.tick_params(labelsize=fs - 1)
    return au


def dist_hist(ax, val, y, xlabel, title, colors=None, logx=False, vline=None, fs=8):
    """
    Density of a per-locus quantity by class, on shared bins.

    SHARED BINS, NOT TWO INDEPENDENT HISTOGRAMS. Matplotlib would otherwise choose a different
    edge set per class and the two densities would not be comparable bar-for-bar -- the visual
    difference between them would partly be a binning difference. The range is clipped to the
    pooled 0.5-99.5 percentiles for display only, so one long tail cannot compress the body of
    both distributions into a single bar.
    """
    cols = colors or DEFAULT_COLORS
    val = np.asarray(val, dtype=float)
    lo, hi = np.percentile(val, [0.5, 99.5])
    if logx and lo > 0:
        bins = np.logspace(np.log10(lo), np.log10(hi), 45)
        ax.set_xscale("log")
    else:
        bins = np.linspace(lo, hi, 45)
    ax.hist(val[y == 0], bins=bins, color=cols["nonas"], alpha=0.55, density=True, label="non-AS")
    ax.hist(val[y == 1], bins=bins, color=cols["as"], alpha=0.55, density=True, label="AS")
    if vline is not None and vline > 0:
        ax.axvline(vline, color="k", lw=1.2, ls="--", alpha=0.8, label="decision radius")
    ax.set_xlabel(xlabel, fontsize=fs)
    ax.set_ylabel("Density", fontsize=fs)
    ax.set_title(title, fontsize=fs)
    ax.legend(loc="upper right", frameon=False, fontsize=fs - 1)
    ax.tick_params(labelsize=fs - 1)
    return float(np.median(val[y == 1])), float(np.median(val[y == 0]))


def median_gain(hnorm, tnorm, y):
    """
    Distance the HEAD gains over the trunk, per locus -- COMPUTED, NOT DRAWN.

    Raw head_norm - trunk_norm is scale-confounded (the two live in different spaces), so each
    is standardised across loci first: a locus the head ranks more separated than the trunk is
    positive, less is negative, centred at zero by construction. The violin panel that used to
    show this is gone, but the statistic still goes to --csv_out -- dropping the panel should
    not drop the number, since it is the one the caption quotes.
    """
    val = zscore(np.asarray(hnorm, dtype=float)) - zscore(np.asarray(tnorm, dtype=float))
    return float(np.median(val[y == 1]) - np.median(val[y == 0]))
def parse_row(spec):
    """
    "<backbone>:<assay>:<individual>:<path>", or the same with the individual or both
    descriptors omitted. Returns (backbone, assay, individual, path); omitted fields are "".

    Split from the LEFT with a bounded count rather than on every colon, so a path containing
    one does not silently shift the fields.

    Decided on the COLON COUNT alone, never on whether the file exists. Probing the filesystem
    here made a typo'd row re-parse one field short, so the "does not exist" message named
    "<assay>:<path>" rather than the path the user actually wrote.
    """
    parts = spec.split(":", 3)
    if len(parts) == 4:
        return tuple(x.strip() for x in parts)
    if len(parts) == 3:
        return parts[0].strip(), parts[1].strip(), "", parts[2].strip()
    if len(parts) == 2:
        return "", parts[0].strip(), "", parts[1].strip()
    raise SystemExit(
        f"--row {spec!r} is not '<backbone>:<assay>:<individual>:<path>' "
        f"(or with trailing descriptors dropped).\n"
        f"  A row carries its own path so backbone, assay, individual and file cannot be\n"
        f"  mismatched by position the way parallel lists can.")


def pretty_individual(raw, overrides):
    """
    Display name for an individual: "ENC-002" -> "individual 2".

    THE CSV KEEPS THE RAW ID. Only the panel titles are rewritten, so --csv_out stays joinable
    against build directories and cache filenames, which are all keyed on the ENC id.

    The numeric suffix is taken as the individual's number. That is a guess about YOUR
    numbering, not a fact about the data -- if the donor numbering you cite elsewhere does not
    match the ENC suffix, pass --individual_label "ENC-002=individual 4" and this is bypassed.
    Anything that does not match ENC-<digits> is passed through unchanged, so a row already
    reading "individual 2" stays as written.
    """
    if raw in overrides:
        return overrides[raw]
    m = re.fullmatch(r"ENC-0*(\d+)", raw.strip(), flags=re.I)
    return f"individual {int(m.group(1))}" if m else raw


def load_row(path):
    d = np.load(path, allow_pickle=True)
    keys = set(d.files)
    need = {"labels"}
    missing = need - keys
    if missing:
        raise SystemExit(f"{path} has no {sorted(missing)}. Present: {sorted(keys)}\n"
                         f"  Is this a dump_contrast_embeddings npz?")
    y = d["labels"].astype(int)
    tnorm = d["trunk_norm"] if "trunk_norm" in keys else (
        np.linalg.norm(d["trunk_contrast"], axis=1) if "trunk_contrast" in keys else None)
    hnorm = d["head_norm"] if "head_norm" in keys else (
        np.linalg.norm(d["head_contrast"], axis=1) if "head_contrast" in keys else None)
    if tnorm is None or hnorm is None:
        raise SystemExit(f"{path} carries neither the norms nor the contrast matrices "
                         f"(present: {sorted(keys)}).")
    ell = d["ell"] if "ell" in keys else hnorm
    return d, y, np.asarray(tnorm, float), np.asarray(hnorm, float), np.asarray(ell, float)


def main():
    ap = argparse.ArgumentParser(
        description="Four-panel contrast diagnostics, one row per (backbone, assay) npz.")
    ap.add_argument("--row", action="append", required=True, metavar="BACKBONE:ASSAY:INDIV:PATH",
                    help="repeatable; '<backbone>:<assay>:<individual>:<path>', with the "
                         "individual or both descriptors droppable. Rows render top to bottom "
                         "in the order given.")
    ap.add_argument("--individual_label", action="append", default=[], metavar="RAW=DISPLAY",
                    help="repeatable override, e.g. 'ENC-002=individual 4'. Without one, "
                         "ENC-<n> displays as 'individual <n>'. The CSV always keeps the raw "
                         "id so it stays joinable against build and cache paths.")
    ap.add_argument("--panel_w", type=float, default=3.6, help="width of one panel, inches")
    ap.add_argument("--panel_h", type=float, default=3.0, help="height of one row, inches")
    ap.add_argument("--fontsize", type=float, default=8)
    ap.add_argument("--linx", action="store_true",
                    help="force linear x on both ECDFs (default: log-x when the values span "
                         ">1.5 decades, which is usually the head norm only)")
    ap.add_argument("--suptitle", default="Separation of haplotypes in embedding space",
                    help="figure-level title. Panel titles name only the QUANTITY and the row "
                         "label at the left names the cell, so everything else -- assay, "
                         "individual, what the dashed lines mean, the medians -- belongs in the "
                         "caption and in --csv_out, not on the image.")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--csv_out", default=None,
                    help="write the per-row AUROCs and median gain, so numbers quoted from "
                         "this figure are recomputable rather than read off the image")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = [parse_row(r) for r in a.row]
    overrides = {}
    for spec in a.individual_label:
        if "=" not in spec:
            raise SystemExit(f"--individual_label {spec!r} is not 'RAW=DISPLAY'")
        k, v = spec.split("=", 1)
        overrides[k.strip()] = v.strip()
    missing = [p for *_, p in rows if not os.path.exists(p)]
    if missing:
        raise SystemExit("these --row paths do not exist:\n  " + "\n  ".join(missing))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.legend_handler import HandlerTuple

    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(a.panel_w * 4, a.panel_h * n), squeeze=False)

    summary, row_meta = [], []
    for i, (backbone, assay, individual, path) in enumerate(rows):
        d, y, tnorm, hnorm, ell = load_row(path)
        # Comma-joined, omitting whatever was not supplied, so a single-donor figure does not
        # carry a dangling separator and a cross-donor one names the individual on every row.
        label = ", ".join(x for x in (backbone, assay,
                                      pretty_individual(individual, overrides)) if x)
        aa, bb = resolve_ab(d)
        radius = (-bb / aa) if (aa is not None and bb is not None and -bb / aa > 0) else None
        lx = False if a.linx else "auto"

        # X AXIS IS A DISTANCE, SO IT IS LABELLED AS ONE. ||h1-h2|| grows as the two
        # haplotypes move APART in embedding space, and the AS curve lying to the right of
        # non-AS is the result: allele-specific loci have haplotypes the model represents as
        # more different. Naming this axis a "similarity" would make the same curve read as
        # the opposite finding.
        cols = row_colors(backbone)
        row_meta.append((label, cols, backbone))

        # COLUMN ORDER PAIRS EACH SPACE WITH ITS OWN DISTRIBUTION: backbone ECDF, backbone
        # density, head ECDF, head logit density. The violin that used to sit in column 4
        # compared the two spaces to each other, which put a cross-space quantity between two
        # within-space ones; reading left to right now walks trunk -> head once.
        #
        # TITLES ARE SHORT AND THE ROW IDENTITY IS NOT IN THEM. "(DNABERT-2, CTCF)" repeated
        # across four panels is the same string four times; it is now a row label at the left
        # edge, which also makes the backbone grouping visible without reading any title.
        au_t, mt1, mt0 = ecdf(
            axes[i][0], tnorm, y, "Backbone embedding",
            "Haplotype dissimilarity  ||h1 − h2||", colors=cols, logx=lx, fs=a.fontsize)
        dist_hist(
            axes[i][1], tnorm, y, "Haplotype dissimilarity  ||h1 − h2||",
            "Backbone embedding (density)", colors=cols,
            logx=(lx is not False and lx != "auto"), fs=a.fontsize)
        au_h, mh1, mh0 = ecdf(
            axes[i][2], hnorm, y, "HapAdapter head",
            "Haplotype dissimilarity  ||P(h1) − P(h2)||", colors=cols, logx=lx,
            vline=radius,
            vlabel=("decision radius" if radius else "no decision radius: npz has no a/b"),
            fs=a.fontsize)
        au_e = ell_hist(axes[i][2 + 1], ell, y, label, colors=cols, fs=a.fontsize)
        med_gain = median_gain(hnorm, tnorm, y)

        summary.append(dict(backbone=backbone, assay=assay, individual=individual,
                            n_loci=int(len(y)),
                            n_as=int(y.sum()), n_nonas=int((y == 0).sum()),
                            trunk_norm_auroc=round(float(au_t), 4),
                            head_norm_auroc=round(float(au_h), 4),
                            ell_auroc=round(float(au_e), 4),
                            decision_radius=(round(radius, 4) if radius else None),
                            # Medians carried here rather than in the panel titles, so the
                            # caption can quote them without anyone reading them off the image.
                            trunk_median_as=round(mt1, 4), trunk_median_nonas=round(mt0, 4),
                            head_median_as=round(mh1, 4), head_median_nonas=round(mh0, 4),
                            median_gain_as_minus_nonas=round(med_gain, 4),
                            npz=path))
        print(f"  {label:28} n={len(y):>7,} AS={int(y.sum()):>6,} | "
              f"trunk {au_t:.3f} head {au_h:.3f} ell {au_e:.3f} | "
              f"radius {('%.3g' % radius) if radius else 'n/a':>7} | gain {med_gain:+.3f}")

    fig.suptitle(a.suptitle, fontsize=a.fontsize + 4, y=0.995)
    # Leave a strip on the left for the row labels and one at the bottom for the backbone key.
    fig.tight_layout(rect=(0.035, 0.030, 1.0, 0.975))

    # ROW LABELS AFTER LAYOUT, NOT BEFORE. The row's vertical centre is only known once
    # tight_layout has placed the axes; positioning these from nominal coordinates puts them
    # off-centre by a margin that changes with the number of rows and the font size.
    fig.canvas.draw()
    for i, (lab, cols, _bb) in enumerate(row_meta):
        bb0 = axes[i][0].get_position()
        bb3 = axes[i][-1].get_position()
        fig.text(0.008, (bb0.y0 + bb0.y1) / 2, lab, rotation=90, va="center", ha="left",
                 fontsize=a.fontsize + 1, fontweight="bold", color=cols["key"])

    # THE KEY SHOWS BOTH SWATCHES OF EACH PAIR. With lightness carrying the class a single
    # swatch stood for a whole family; with two hues it would stand for only half of one, so
    # each entry is the pair drawn side by side. Classes are still keyed inside every panel --
    # this only says which pair belongs to which backbone.
    seen, handles, labels = set(), [], []
    for _lab, cols, bb in row_meta:
        name = bb or "backbone"
        if name in seen:
            continue
        seen.add(name)
        handles.append((mpatches.Patch(facecolor=cols["nonas"], edgecolor="none"),
                        mpatches.Patch(facecolor=cols["as"], edgecolor="none")))
        labels.append(f"{name}  (non-AS / AS)")
    if len(handles) > 1:
        fig.legend(handles=handles, labels=labels, loc="lower center",
                   ncol=min(4, len(handles)), frameon=False, fontsize=a.fontsize,
                   bbox_to_anchor=(0.5, 0.0), handlelength=3.0,
                   handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)})

    fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight")
    print(f"[write] {a.out}")

    if a.csv_out:
        import pandas as pd
        pd.DataFrame(summary).to_csv(a.csv_out, index=False)
        print(f"[write] {a.csv_out}")


if __name__ == "__main__":
    sys.exit(main())
