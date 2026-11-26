#!/usr/bin/env python3
"""
hapadapter.figures.fig_backbone_comparison -- the ladder figure: for each assay, what the
raw representation achieves and what HapAdapter adds, on both backbones.

WHAT EACH BAR IS. Within a backbone the bars are a LADDER over the same frozen encoder:

  DNABERT-2  sequence-only            ||h1-h2|| from the pretrained backbone, no training
  DNABERT-2  functionally supervised  ||h1-h2|| from the binding-finetuned trunk
  DNABERT-2  + HapAdapter             the trained twin head on that same trunk
  AlphaGenome functionally supervised ||h1-h2|| from AlphaGenome as shipped
  AlphaGenome + HapAdapter            the trained twin head on those embeddings

The two "functionally supervised" bars are NOT the same rung and the shared shade does not
claim they are: one is a trunk this project finetuned, the other is a model that arrives
functionally supervised. The comparison that carries no confound is WITHIN a backbone --
same encoder, same loci, same label -- which is why hue is bound to backbone and shade to
rung, and why the two families are separated by a gap on the x axis.

METRIC. --metric defaults to auroc_plain (unmatched). auroc_matched is the coverage-matched
statistic; on this test set the two differ by a median of ~0.02-0.03 because the loci were
depth-matched at SELECTION on `depth` while the matched statistic controls `total_reads`,
which was never matched. Both columns are in the input CSV; the figure plots one and the
caption should name which.

The input is the concatenation of score_matched.py's per-assay CSVs. Rows with
backbone == "control" (the read-depth rung) are excluded from the bars by design -- depth is
a confound control, not a model, and its matched value is near-tautological since it ranks
coverage after matching on coverage.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NAME = {"ctcf": "CTCF", "ep300": "EP300", "polr2a": "POLR2A",
        "polr2aphosphos5": "POLR2AphosphoS5", "h3k4me3": "H3K4me3",
        "h3k4me1": "H3K4me1", "h3k27ac": "H3K27ac", "atac": "ATAC-seq",
        "h3k36me3": "H3K36me3", "h3k27me3": "H3K27me3", "h3k9me3": "H3K9me3"}

#: Panel order is by REGULATORY REGIME -- TF, promoter, enhancer mark, accessibility,
#: heterochromatin -- not alphabetical and not by effect size. Keep it stable across
#: regenerated versions so panels stay comparable between drafts.
#: The six-assay version's relative order is PRESERVED inside this list (ctcf < polr2a <
#: h3k4me3 < h3k27ac < atac < h3k9me3), so a six-panel draft and an eleven-panel one put the
#: same assays in the same sequence and can be read side by side.
ORDER = ["ctcf", "ep300", "polr2a", "polr2aphosphos5", "h3k4me3", "h3k4me1",
         "h3k27ac", "atac", "h3k36me3", "h3k27me3", "h3k9me3"]

#: Hierarchical colour: the OUTER level (backbone) picks the hue family, the INNER level
#: (rung) samples within it by lightness. Blue/orange rather than any red-green pair, so the
#: backbone distinction survives deuteranopia; the within-family steps are lightness, which
#: survives it too.
#: Within each family the order is a LADDER OF INCREASING SUPERVISION, and the third rung in
#: each is the one a practitioner would actually reach for: run the model on both haplotypes
#: and difference ITS OWN OUTPUT. It sits immediately left of the adapter in both families
#: because that is the comparison a reader asks about ("why not just run the model twice?").
#:
#: IT IS NOT RELIABLY THE STRONGER BASELINE, and an earlier version of this comment claimed it
#: was, generalising from CTCF where it scores 0.660 against 0.581 for the embedding norm.
#: Measured across all six assays it beats the best distance rung in only 4 of 12 cells, and
#: DNABERT-2's H3K4me3 readout-difference is BELOW chance at 0.493. Include it because it is
#: the published protocol, not because it is a harder bar -- and quote the per-assay numbers
#: from hapadapter_gain_by_assay.csv rather than any summary claim about which rung is higher.
BARS = [("dnabert2",    "pretrained-norm", "#D3E0EF", "sequence-only distance"),
        ("dnabert2",    "trunk-norm",      "#9BBCDF", "fine-tuned distance"),
        ("dnabert2",    "trunk-delta",     "#5287C0", "own readout, differenced"),
        ("dnabert2",    "hapadapter",      "#17497F", "+ HapAdapter"),
        ("alphagenome", "trunk-norm",      "#F7D4AC", "embedding distance"),
        ("alphagenome", "track-delta",     "#E99A4A", "own track, differenced"),
        ("alphagenome", "hapadapter",      "#B35F10", "+ HapAdapter ")]

#: The 0.6 gap between index 3 and 4 is the backbone separation. Without it the seven bars read
#: as one ladder, which would invite a cross-backbone comparison at the distance rungs that the
#: data does not support -- and the two families' third bars are DIFFERENT quantities (a
#: regression readout vs a predicted track), comparable to their own neighbours but not to
#: each other.
XPOS = [0, 1, 2, 3, 4.6, 5.6, 6.6]

GREY = "#6E6E6E"


def style(base=8.0):
    """Minimal publication defaults. Self-contained so the script runs anywhere."""
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300,
        "font.size": base, "axes.titlesize": base, "axes.labelsize": base - 1,
        "xtick.labelsize": base - 2, "ytick.labelsize": base - 2,
        "legend.fontsize": base - 1.7,
        "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.direction": "out", "ytick.direction": "out",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "figure.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def build(d, metric="auroc_plain", title=None, size=6.8, bar_width=0.62, ncols=3,
          legend_slot="last",
          keep=None):
    bars = d[d.backbone != "control"]
    # ORDER always sets the SEQUENCE; `keep` only subsets it. Passing a reordered --assays
    # therefore cannot silently reorder the panels away from the canonical layout the other
    # figures use, which is what makes two drafts comparable panel-by-panel.
    present = set(bars.assay)
    if keep is not None:
        want = {k.strip().lower() for k in keep if k.strip()}
        missing = sorted(want - present)
        if missing:
            print(f"[warn] --assays named {missing}, absent from the input; ignoring them")
        present = present & want
    assays = [a for a in ORDER if a in present]
    if not assays:
        raise SystemExit(f"no assays left to plot; input has {sorted(set(d.assay))}, "
                         f"--assays asked for {sorted(keep) if keep else '(all)'}")

    vals = bars[metric].to_numpy(float)
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        raise SystemExit(f"column {metric!r} is entirely NaN -- nothing to plot")
    # Pad around the data rather than fixing a round floor: a wider floor leaves a dead band
    # under the chance line in every panel that has no sub-chance bar.
    lo = min(0.49, float(vals.min()) - 0.015)
    hi = max(0.51, float(vals.max()) + 0.015)
    # Ticks are placed INSIDE [lo, hi]; the limits are not rounded out to the nearest tick.
    # Rounding out is what re-opens a dead band under the chance line -- a single bar at
    # 0.433 would drag the floor to 0.40 and leave every other panel mostly empty.
    ticks = np.arange(np.ceil(lo * 20) / 20, np.floor(hi * 20) / 20 + 1e-9, 0.05)

    # A square CANVAS requires fixed margins and no bbox_inches="tight" at save time --
    # tight cropping reflows to the content and silently breaks the aspect the caller asked
    # for. The legend lives in the reserved bottom strip.
    # GRID FROM THE ASSAY COUNT. This was a hardcoded 2x3, which silently plotted only the
    # first six assays and dropped the rest without a word -- a figure that looks finished and
    # is missing panels. ncols is fixed and rows follow, so six assays still render exactly
    # 2x3 and nothing about the existing draft changes.
    rows_n = int(np.ceil(len(assays) / ncols))
    # Height tracks the grid so a panel keeps its shape as rows are added; the 1.5 reproduces
    # the original square canvas at 2 rows x 3 cols.
    fig = plt.figure(figsize=(size, size * (rows_n / ncols) * 1.5))
    # The legend strip is a fixed fraction of a TALLER canvas, so reserve it in inches rather
    # than in figure fraction: at 4 rows a 0.085 bottom is nearly twice the strip it was at 2.
    # THE LEGEND GOES IN A FREE GRID SLOT WHEN THERE IS ONE, and only falls back to a reserved
    # strip below the axes when the grid is exactly full. An incompletely filled grid already
    # has blank panels; spending extra canvas height on a strip as well wastes the space twice
    # and shrinks every panel. The reserved value stays 0.145 for the full-grid case, which was
    # swept and measured: 0.085 overlaps the bottom axes, 0.115 clears by ~10 px, 0.145 by ~40.
    n_free = rows_n * ncols - len(assays)
    # 0.125 for the full-grid strip. 0.145 cleared by ~33 px, which read as the key floating
    # away from the panels; swept down and measured at 9 assays / ncols=3, where the legend is
    # tallest relative to the canvas: 0.105 OVERLAPS the bottom row by 12.6 px, 0.115 still
    # touches at -1.2 px, 0.125 clears by 10.2 px. Taking the first value that clears rather
    # than the smallest that renders, since a longer rung label grows the legend downward.
    bot_in = (0.125 if n_free == 0 else 0.03) * size
    top_in = (1 - (0.945 if title is None else 0.915)) * size
    h_in = size * (rows_n / ncols) * 1.5
    gs = fig.add_gridspec(rows_n, ncols, left=0.095, right=0.985,
                          top=1 - top_in / h_in, bottom=bot_in / h_in,
                          hspace=0.17, wspace=0.09)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(ncols)] for r in range(rows_n)]
    flat = [x for row in axes for x in row]

    # WHICH SLOT THE LEGEND TAKES, AND THEREFORE WHICH SLOTS THE PANELS GET. The default is
    # the last slot, so panels fill the grid in order and the key lands bottom-right.
    # "first_row_end" instead reserves the end of the FIRST row, which turns 9 panels at
    # ncols=5 from 5+4 with the key below into 4+5 with the key beside the top row. The
    # panels have to skip that slot rather than flow through it, or the assay order would
    # silently shift by one against every previous draft.
    legend_idx = None
    if n_free > 0:
        legend_idx = (ncols - 1) if legend_slot == "first_row_end" else len(flat) - 1
    elif legend_slot == "first_row_end":
        print(f"[legend] --legend_slot first_row_end needs a free grid slot, but "
              f"{len(assays)} assays fill {rows_n}x{ncols} exactly. Using the reserved "
              f"strip below the panels; --ncols {ncols + 1} would free one.")
    panel_ix = [i for i in range(len(flat)) if i != legend_idx]
    if len(assays) > len(panel_ix):
        raise SystemExit(f"{len(assays)} assays do not fit {rows_n}x{ncols} with the legend "
                         f"in a slot; raise --ncols or use the default --legend_slot last.")
    used = set(panel_ix[:len(assays)])
    for i, ax in enumerate(flat):
        if i not in used and i != legend_idx:
            ax.set_visible(False)

    for i, a in zip(panel_ix, assays):
        ax = flat[i]
        sub = bars[bars.assay == a]
        for x, (bb, rg, col, lab) in zip(XPOS, BARS):
            r = sub[(sub.backbone == bb) & (sub.rung == rg)]
            if not len(r) or not np.isfinite(r[metric].iloc[0]):
                # An absent bar is reported, never silently skipped: a gap in a bar chart
                # reads as zero, and "input missing" is not a measurement of zero.
                ax.text(x, 0.503, "n.d.", ha="center", va="bottom", fontsize=5.5,
                        rotation=90, color=GREY)
                continue
            ax.bar(x, r[metric].iloc[0] - 0.5, bottom=0.5, width=bar_width, color=col,
                   edgecolor="black", linewidth=0.4, zorder=2,
                   label=lab if a == assays[0] else None)
        ax.axhline(0.5, color="black", lw=0.8, zorder=3)
        n = int(sub.n_per_class.max()) if sub.n_per_class.notna().any() else 0
        ax.set_title(f"{NAME.get(a, a)}   n = {n:,}/class", loc="center", pad=3)
        ax.set_xticks([])
        ax.set_xlim(-0.72, 7.32)
        ax.set_ylim(lo, hi)
        ax.set_yticks(ticks)
        ax.grid(axis="y", lw=0.4, alpha=0.3, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # The y-label goes on the LEFTMOST VISIBLE PANEL of each row, not on column 0 blindly:
    # with the legend in the first row's last slot every row still starts at column 0, but a
    # future arrangement that reserves a left-hand slot would otherwise label a blank axes
    # and leave the first real panel without one.
    for r, row in enumerate(axes):
        vis = [c for c, ax in enumerate(row) if r * ncols + c in used]
        if not vis:
            continue
        row[vis[0]].set_ylabel("AUROC", labelpad=3)
        for c in vis[1:]:
            row[c].tick_params(labelleft=False)

    if title:
        fig.suptitle(title, y=0.985, fontsize=9)

    # ONE LEGEND PER BACKBONE, NAME IN THE TITLE. The labels used to repeat "DNABERT-2" and
    # "AlphaGenome" on every row, so the backbone -- the thing the hue already encodes -- was
    # the widest and most repeated text in the legend while the rung, which is what a reader
    # is actually looking up, sat in the leftover space. The name now appears once per group.
    #
    # The adapter labels differ by a trailing space ("+ HapAdapter" / "+ HapAdapter ") so the
    # two groups keep separate legend entries; identical label strings would otherwise be
    # merged by get_legend_handles_labels into one, dropping a swatch.
    h, l = axes[0][0].get_legend_handles_labels()
    by_label = dict(zip(l, h))
    groups = [("DNABERT-2", [lab for _bb, _rg, _c, lab in BARS if _bb == "dnabert2"]),
              ("AlphaGenome", [lab for _bb, _rg, _c, lab in BARS if _bb == "alphagenome"])]
    # In-grid: the two groups stack in the LAST free slot, which is the bottom-right corner.
    # Stacked rather than side by side because one slot is a panel's width, not the figure's.
    host_ax = None
    if legend_idx is not None:
        host_ax = flat[legend_idx]
        host_ax.set_visible(True)
        host_ax.axis("off")
    # THE SECOND GROUP IS PLACED FROM THE FIRST'S MEASURED HEIGHT, not at a guessed fraction of
    # the slot. A fixed offset is wrong at any other panel size or rung count -- it either
    # strands the two groups at opposite ends of a tall slot or overlaps them in a short one.
    y_next = 1.0
    for gi, (name, labs) in enumerate(groups):
        labs = [x for x in labs if x in by_label]
        if not labs:
            continue
        handles_g, texts_g = [by_label[x] for x in labs], [x.strip() for x in labs]
        if host_ax is not None:
            # Legend(...) + add_artist rather than ax.legend(): ax.legend() also REBINDS
            # ax.legend_, so the last group would be registered twice (once as the axes legend,
            # once as an artist) and drawn twice.
            leg = mpl.legend.Legend(host_ax, handles_g, texts_g, title=name, frameon=False,
                                    loc="upper left", bbox_to_anchor=(0.0, y_next),
                                    bbox_transform=host_ax.transAxes, handlelength=1.2,
                                    labelspacing=0.45, alignment="left")
            host_ax.add_artist(leg)
            fig.canvas.draw()                      # realise the layout before measuring it
            bb = leg.get_window_extent(fig.canvas.get_renderer())
            h_ax = bb.transformed(host_ax.transAxes.inverted()).height
            y_next -= h_ax + 0.06                  # one blank line between the groups
        else:
            leg = fig.legend(handles_g, texts_g, title=name, frameon=False, loc="lower center",
                             bbox_to_anchor=(0.285 if gi == 0 else 0.715, 0.012),
                             handlelength=1.2, labelspacing=0.45, alignment="left")
        # From the legend's OWN entry size, not plt.rcParams: rcParams["legend.fontsize"] is
        # the string "medium" unless style() has run, so reading it breaks any caller that
        # imports build() directly.
        leg.get_title().set_fontsize(leg.get_texts()[0].get_fontsize() + 0.6)
        leg.get_title().set_fontweight("bold")
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", nargs="+", required=True,
                    help="score_matched output(s); several are concatenated")
    ap.add_argument("--metric", default="auroc_plain",
                    choices=["auroc_plain", "auroc_matched"],
                    help="auroc_plain is unmatched; auroc_matched is coverage-matched. "
                         "Name whichever you choose in the caption -- they are not "
                         "interchangeable on this test set.")
    ap.add_argument("--title", default=None)
    ap.add_argument("--size", type=float, default=6.8,
                    help="canvas WIDTH in inches. Height follows the grid, so the panel shape "
                         "is held constant as assays are added; at 3 columns and 2 rows this "
                         "reproduces the original square canvas exactly.")
    ap.add_argument("--assays", default=None,
                    help="comma-separated assays to KEEP (canonical names, case-insensitive). "
                         "Panel order always follows the canonical ORDER regardless of how "
                         "this is written. Omit for every assay in the input.")
    ap.add_argument("--ncols", type=int, default=3,
                    help="panel columns. Rows follow from the assay count, and trailing empty "
                         "slots are hidden. 6 assays at the default is the original 2x3.")
    ap.add_argument("--legend_slot", default="last", choices=["last", "first_row_end"],
                    help="which free grid slot holds the key. 'last' is the bottom-right "
                         "corner (panels fill the grid in order). 'first_row_end' reserves "
                         "the end of the top row instead, so 9 assays at --ncols 5 read as "
                         "4 panels + key, then 5 panels. Needs a free slot; with none the "
                         "key falls back to the strip below the panels.")
    ap.add_argument("--bar_width", type=float, default=0.62)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    frames = []
    for p in a.csv:
        if not os.path.exists(p):
            raise SystemExit(f"missing input: {p}")
        frames.append(pd.read_csv(p))
    d = pd.concat(frames, ignore_index=True)
    missing = {"assay", "backbone", "rung", a.metric} - set(d.columns)
    if missing:
        raise SystemExit(f"input is missing column(s) {sorted(missing)}; present: "
                         f"{list(d.columns)}")

    style()
    fig = build(d, metric=a.metric, title=a.title, size=a.size, bar_width=a.bar_width,
                ncols=a.ncols, legend_slot=a.legend_slot,
                keep=(a.assays.split(",") if a.assays else None))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    # No bbox_inches="tight": it would crop to content and break the square aspect.
    fig.savefig(a.out)
    print(f"[write] {a.out}  ({a.metric}, {len(d)} rows, "
          f"{d[d.backbone != 'control'].assay.nunique()} assays)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
