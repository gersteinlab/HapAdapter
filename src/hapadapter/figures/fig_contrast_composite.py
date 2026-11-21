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

import matplotlib as mpl
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
    # NO NUMBERS ON THIS FIGURE AT ALL. Its claim is that the two classes SEPARATE, which the
    # curves show directly; an AUROC printed beside them invites reading it as a performance
    # result, and these are plain (unmatched) AUROCs that carry the read-depth confound --
    # the one number here that should not travel on an image. Every AUROC is still computed
    # and still written to --csv_out, so the caption can quote whichever is defensible.
    ax.set_title(title, fontsize=fs)
    ax.legend(loc="upper left", frameon=False, fontsize=fs - 1)
    ax.grid(alpha=0.25, which="both")
    ax.tick_params(labelsize=fs - 1)
    return au, float(np.median(val[y == 1])), float(np.median(val[y == 0]))


def ell_hist(ax, ell, y, row_label, colors=None, fs=8, clip_q=99.5, nbins=45,
             mode="distance", dist=None, radius=None, xlabel=None):
    """
    Head-logit density. Returns (auroc, kept_fraction, hi).

    THE BINS ARE COMPUTED OVER THE CLIPPED RANGE, NOT JUST THE AXIS LIMIT. The AS logit has a
    long right tail -- a handful of loci reach +17 while the bulk sits inside [-1, 1] -- so a
    range set by the full extent spends 45 bins on mostly-empty space and the separation near
    the boundary renders as two or three bars. Setting xlim alone would not fix that: the bin
    WIDTH is what hides the structure, and it is chosen here. Binning inside the clipped range
    puts all 45 bins where the classes actually differ.

    WHAT CLIPPING COSTS, AND WHY IT IS ACCEPTABLE HERE. Mass beyond `hi` is dropped from the
    drawn histogram, so the visible bars no longer integrate to 1 and the two classes are
    clipped by DIFFERENT amounts (the tail is almost entirely AS). The figure's claim is that
    the classes separate, and discarding part of the AS tail makes that claim HARDER to see,
    not easier -- the clipped loci are the most confidently-AS ones. The kept fraction is
    returned, printed and written to --csv_out so the caption can state it rather than a reader
    having to infer it from a truncated axis.
    """
    # COORDINATE, NOT ZOOM. For a twin head ell = a*||P(h1) - P(h2)|| + b with a > 0, i.e. a
    # strictly monotone LINEAR map of a non-negative, heavily right-skewed distance. A linear
    # axis therefore always piles the mass against the floor b and strings a thin tail to the
    # right, and no choice of clip percentile fixes that -- clipping moves the right edge, it
    # does not change the skew. Column 3 already handles the same skew with a log axis, so in
    # "distance" mode this panel is drawn in the SAME coordinate as column 3: log-x on
    # ||P(h1) - P(h2)||, with the decision RADIUS -b/a where the boundary was. No information
    # is lost -- the map is monotone, so the AUROC is identical either way -- and the two
    # right-hand panels become the ECDF/density pair that columns 1 and 2 already are.
    radius_offscale = False
    use_dist = (mode == "distance") and (dist is not None) and (radius is not None) \
               and np.all(np.asarray(dist) > 0)
    if use_dist:
        val = np.asarray(dist, dtype=float)
        lo = float(np.percentile(val, 100 - clip_q))
        hi = float(np.percentile(val, clip_q))
        # THE AXIS FOLLOWS THE DATA; THE RADIUS IS ANNOTATED IF IT FALLS OUTSIDE. Forcing the
        # radius onto the axis stretched rows whose threshold sits far above every locus --
        # row 4 of the real figure had its mass in 1e-3..1e-1 and the radius near 1, so most
        # of the panel was empty. That the threshold sits off the mass is a genuine finding
        # (it is why recall at the default threshold is low), so it is reported rather than
        # dropped -- just not at the cost of making the distributions unreadable.
        radius_offscale = not (lo <= radius <= hi)
        bins = np.logspace(np.log10(lo), np.log10(hi), nbins)
        ax.set_xscale("log")
        # NO MINOR TICK LABELS. When a log axis spans less than a decade matplotlib starts
        # labelling the MINOR ticks too ("2 x 10^-1", "3 x 10^-1", ...), and at this panel
        # width they overprint into an unreadable smear. Suppressing them can leave a decade
        # with no labelled tick at all, so when fewer than two majors land in range, plain
        # decimal labels are placed at the ends and the midpoint instead.
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        _dec = np.log10(hi) - np.log10(lo)
        if _dec < 1.05:
            mid = float(np.sqrt(lo * hi))
            ax.set_xticks([lo, mid, hi])
            ax.xaxis.set_major_formatter(
                mpl.ticker.FuncFormatter(lambda v, _p: f"{v:.3g}"))
        boundary, blabel = radius, "decision radius"
        # Inherit the caller's label so --x_norm shows in BOTH head panels, not just column 3.
        xlab = (xlabel or "Haplotype dissimilarity  ||P(h1) − P(h2)||") + "  (log)"
    else:
        val = np.asarray(ell, dtype=float)
        lo = float(np.percentile(val, 0.5))
        hi = float(np.percentile(val, clip_q))
        if not np.isfinite(hi) or hi <= lo:               # degenerate (e.g. a constant logit)
            lo, hi = float(np.min(val)), float(np.max(val)) or 1.0
        # The decision boundary must stay on the axis: it is the one annotation that makes this
        # panel readable, and a clip that excluded it would be a different plot.
        hi, lo = max(hi, 1e-9), min(lo, -1e-9)
        bins = np.linspace(lo, hi, nbins)
        boundary, blabel = 0.0, "decision boundary"
        xlab = "Model logit  (dashed: decision boundary)"
    kept = float(np.mean((val >= lo) & (val <= hi)))
    cols = colors or DEFAULT_COLORS
    # FRACTION PER BIN ON A LOG AXIS, NOT density=True. density divides each count by its bin
    # WIDTH, and log-spaced bins differ in width by orders of magnitude -- so the narrow bins
    # at small x are divided by a tiny number and tower over everything else. That is what put
    # row 4's y-axis at 80 while its neighbours sat near 1: an artifact of the normalisation,
    # not a property of the distribution. Weighting by 1/n instead makes every bar "fraction
    # of this class's loci", comparable left to right and across rows.
    for cls, key, lab in ((0, "nonas", "non-AS"), (1, "as", "AS")):
        v = val[y == cls]
        v = v[(v >= lo) & (v <= hi)]
        ax.hist(v, bins=bins, color=cols[key], alpha=0.55, label=lab,
                weights=np.full(len(v), 1.0 / max((y == cls).sum(), 1)))
    ax.set_xlim(lo, hi)
    # SAME boundary as delta* = -b/a on the head-norm axis, which is why panel 2's dashed line
    # and this one are the same decision, drawn in two coordinate systems.
    if use_dist and radius_offscale:
        side = 0.985 if radius > hi else 0.015
        ax.annotate(f"{blabel} {radius:.3g} {'→' if radius > hi else '←'}",
                    xy=(side, 0.97), xycoords="axes fraction",
                    ha="right" if radius > hi else "left", va="top",
                    fontsize=fs - 1.5, color="0.35")
    else:
        ax.axvline(boundary, color="k", lw=1.2, ls="--", alpha=0.8, label=blabel)
    au = _auroc(y, ell)
    ax.set_xlabel(xlab, fontsize=fs)
    ax.set_ylabel("Fraction of loci", fontsize=fs)
    # Short, like its siblings: the row label carries which cell this is, and the AUROC goes
    # to --csv_out rather than onto the panel (see the note in ecdf).
    ax.set_title("Head separation" if use_dist else "Head logit", fontsize=fs)
    # FRAMED AND OPAQUE, not frameon=False. The decision-radius rule is a vertical line that
    # loc="best" cannot avoid -- "best" minimises overlap with the DATA, and a single thin line
    # is cheap to cross, so it routinely draws straight through the entry text. A white backing
    # fixes legibility without moving the legend off the emptiest corner. This is also the one
    # collision a text-vs-text overlap check cannot see, since the line is not text.
    ax.legend(loc="best", frameon=True, framealpha=0.82, edgecolor="none",
              facecolor="white", fontsize=fs - 1)
    ax.tick_params(labelsize=fs - 1)
    return au, kept, hi


def dist_hist(ax, val, y, xlabel, title, colors=None, logx=False, vline=None, fs=8,
              clip_q=99.5):
    """
    Density of a per-locus quantity by class, on shared bins. -> (median_AS, median_nonAS, kept)

    SHARED BINS, NOT TWO INDEPENDENT HISTOGRAMS. Matplotlib would otherwise choose a different
    edge set per class and the two densities would not be comparable bar-for-bar -- the visual
    difference between them would partly be a binning difference.

    SAME CLIP CONVENTION AS THE HEAD PANEL, which is why clip_q is a parameter rather than the
    fixed 99.5 it used to be. This panel and the head-logit panel plot DIFFERENT quantities, so
    their absolute ranges were never comparable -- but when one dropped 0.5% of its tail and
    the other dropped 15%, the apparent "this panel is zoomed in further" was an artifact of
    two different conventions rather than a property of the distributions. Dropping the same
    FRACTION from both makes the two panels in a row read as the same kind of view.
    """
    cols = colors or DEFAULT_COLORS
    val = np.asarray(val, dtype=float)
    lo, hi = np.percentile(val, [100 - clip_q, clip_q])
    kept = float(np.mean((val >= lo) & (val <= hi)))
    if logx and lo > 0:
        bins = np.logspace(np.log10(lo), np.log10(hi), 45)
        ax.set_xscale("log")
    else:
        bins = np.linspace(lo, hi, 45)
    # FRACTION PER BIN, NOT density=True -- the same convention as the head panel. A density
    # divides by bin WIDTH, so its height carries the units of whatever embedding produced it:
    # AlphaGenome's dissimilarities are ~6x larger than DNABERT-2's, which alone put one row's
    # y-max at 0.26 and another's at 1.45 with no difference in how peaked they are. A reader
    # comparing two rows would have been reading a unit conversion. As a fraction, a flat
    # distribution is 1/nbins in EVERY panel, so height means the same thing in all of them.
    for _cls, _key, _lab in ((0, "nonas", "non-AS"), (1, "as", "AS")):
        _v = val[(y == _cls) & (val >= lo) & (val <= hi)]
        ax.hist(_v, bins=bins, color=cols[_key], alpha=0.55, label=_lab,
                weights=np.full(len(_v), 1.0 / max((y == _cls).sum(), 1)))
    if vline is not None and vline > 0:
        ax.axvline(vline, color="k", lw=1.2, ls="--", alpha=0.8, label="decision radius")
    ax.set_xlabel(xlabel, fontsize=fs)
    ax.set_ylabel("Fraction of loci", fontsize=fs)
    ax.set_title(title, fontsize=fs)
    ax.legend(loc="upper right", frameon=True, framealpha=0.82, edgecolor="none",
              facecolor="white", fontsize=fs - 1)
    ax.set_xlim(lo, hi)
    ax.tick_params(labelsize=fs - 1)
    return float(np.median(val[y == 1])), float(np.median(val[y == 0])), kept


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
    ap.add_argument("--logit_clip_pct", default="99.0",
                    help="percentile of the head logit at which the density panel's x-axis "
                         "ends, AND over which its bins are computed -- setting only the limit "
                         "would leave the bin width chosen over the full range, which is what "
                         "hides the separation. Mass beyond it is dropped from the drawing, "
                         "almost all of it AS, so the clip makes the separation harder to see "
                         "rather than easier. 100 disables it. The drawn fraction is printed "
                         "and written to --csv_out. Accepts ONE value for every row, or a "
                         "comma-separated value PER row in --row order -- the percentile is "
                         "evaluated within each row, so one number gives different absolute "
                         "limits per row and a row whose bulk is narrower needs its own. The "
                         "SAME percentile governs the backbone-density panel, so both density "
                         "panels in a row drop the same FRACTION of their tail -- otherwise "
                         "one looks more zoomed than the other for no reason to do with the "
                         "data. Their absolute ranges are still incomparable; they plot "
                         "different quantities.")
    ap.add_argument("--x_norm", choices=["none", "nonas_median"], default="none",
                    help="divide each row's dissimilarity axes by that row's NON-AS median, so "
                         "every row reads in units of a typical non-AS locus and the rows are "
                         "comparable left to right. AUROC, the standardised gain and the ECDF "
                         "shapes are invariant to a positive scale, so only the axis changes; "
                         "the divisors are written to --csv_out. Without it the two backbones' "
                         "axes are in different units and the caption has to say so.")
    ap.add_argument("--head_density", choices=["distance", "logit"], default="distance",
                    help="coordinate for the right-hand density panel. 'distance' draws "
                         "||P(h1)-P(h2)|| on a LOG axis with the decision radius -- the same "
                         "coordinate as the panel to its left, so the two right panels are an "
                         "ECDF/density pair exactly as columns 1 and 2 are. 'logit' draws "
                         "a*||.||+b on a linear axis. The two are monotonically related, so "
                         "the AUROC is identical; they differ only in readability, and the "
                         "linear one piles a right-skewed quantity against its floor. Falls "
                         "back to 'logit' when the npz carries no a/b (no radius to draw).")
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

    # ONE VALUE OR ONE PER ROW. Parsed here, beside `rows`, so a length mismatch is reported
    # against the number of --row flags rather than failing later with an index error.
    _cp = [x.strip() for x in str(a.logit_clip_pct).split(",") if x.strip()]
    if len(_cp) == 1:
        clip_pcts = [float(_cp[0])] * len(rows)
    elif len(_cp) == len(rows):
        clip_pcts = [float(x) for x in _cp]
    else:
        raise SystemExit(f"--logit_clip_pct has {len(_cp)} values for {len(rows)} --row flags; "
                         f"pass one value or one per row, in --row order")

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

        # SHARED X UNITS ACROSS ROWS. The two backbones' embeddings differ in magnitude by
        # roughly 6x, so raw dissimilarity axes are in different units per row and nothing is
        # comparable left to right -- the reader has to be told so in the caption. Dividing
        # each column by that row's NON-AS MEDIAN puts every row on "dissimilarity relative to
        # a typical non-AS locus", where 1.0 means the same thing everywhere. AUROC, the
        # standardised gain and the ECDF shapes are all invariant to a positive scale factor,
        # so this changes the axis and nothing else; the divisors go to --csv_out so absolute
        # values remain recoverable. Non-AS rather than the pooled median because it is the
        # class the AS distribution is being compared AGAINST.
        _rel = (a.x_norm == "nonas_median")
        _xlab_t = ("Haplotype dissimilarity  ||h1 − h2||  / non-AS median" if _rel
                   else "Haplotype dissimilarity  ||h1 − h2||")
        _xlab_h = ("Haplotype dissimilarity  ||P(h1) − P(h2)||  / non-AS median" if _rel
                   else "Haplotype dissimilarity  ||P(h1) − P(h2)||")
        t_scale = h_scale = 1.0
        if a.x_norm == "nonas_median":
            t_scale = float(np.median(np.asarray(tnorm)[y == 0])) or 1.0
            h_scale = float(np.median(np.asarray(hnorm)[y == 0])) or 1.0
            tnorm = np.asarray(tnorm, dtype=float) / t_scale
            hnorm = np.asarray(hnorm, dtype=float) / h_scale
            if radius:
                radius = radius / h_scale

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
            _xlab_t, colors=cols, logx=lx, fs=a.fontsize)
        _mt1d, _mt0d, trunk_kept = dist_hist(
            axes[i][1], tnorm, y, _xlab_t,
            "Backbone embedding (distribution)", colors=cols,
            logx=(lx is not False and lx != "auto"), fs=a.fontsize,
            clip_q=clip_pcts[i])
        au_h, mh1, mh0 = ecdf(
            axes[i][2], hnorm, y, "HapAdapter head",
            _xlab_h, colors=cols, logx=lx,
            vline=radius,
            vlabel=("decision radius" if radius else "no decision radius: npz has no a/b"),
            fs=a.fontsize)
        au_e, ell_kept, ell_hi = ell_hist(axes[i][2 + 1], ell, y, label, colors=cols,
                                          fs=a.fontsize, clip_q=clip_pcts[i],
                                          mode=a.head_density, dist=hnorm, radius=radius,
                                          xlabel=_xlab_h)
        if ell_kept < 1.0:
            n_cut = int(round((1 - ell_kept) * len(y)))
            n_cut_as = int(((ell > ell_hi) & (y == 1)).sum())
            # Name the coordinate actually drawn: in distance mode ell_hi is a DISSIMILARITY,
            # not a logit, and calling it "head-logit" made 7.1 look like a logit value next to
            # a radius of 2.06 on the same row.
            _coord = ("head-separation" if a.head_density == "distance" else "head-logit")
            print(f"  [clip] {label}: {_coord} axis ends at {ell_hi:.3g}; "
                  f"{n_cut} of {len(y)} loci beyond it ({1 - ell_kept:.1%}), "
                  f"{n_cut_as} of them AS")
        med_gain = median_gain(hnorm, tnorm, y)

        summary.append(dict(backbone=backbone, assay=assay, individual=individual,
                            n_loci=int(len(y)),
                            n_as=int(y.sum()), n_nonas=int((y == 0).sum()),
                            trunk_norm_auroc=round(float(au_t), 4),
                            head_norm_auroc=round(float(au_h), 4),
                            ell_auroc=round(float(au_e), 4),
                            # The AUROC above is computed on ALL loci; the panel only DRAWS
                            # the kept fraction. Recording both stops a reader reconciling a
                            # truncated axis against a statistic that used the whole tail.
                            trunk_x_divisor=round(float(t_scale), 6),
                            head_x_divisor=round(float(h_scale), 6),
                            trunk_drawn_fraction=round(float(trunk_kept), 4),
                            ell_axis_max=round(float(ell_hi), 4),
                            ell_drawn_fraction=round(float(ell_kept), 4),
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
