#!/usr/bin/env python3
"""
hapadapter.figures.fig_ism_head -- the HEAD-only ISM figure for one (assay, individual):

    +---------------------------+---------------------------+
    |  A  head ISM saliency     |  B  cofactor motif density |
    |     by TP / TN / FP / FN  |     AS vs non-AS, N motifs |
    +---------------------------+---------------------------+
    |  C  AS     contribution logo                           |
    +--------------------------------------------------------+
    |  D  non-AS contribution logo    (SAME y-scale as C)    |
    +--------------------------------------------------------+
    |  E  reference PWM, shared: both logos align to it      |
    +--------------------------------------------------------+

THE TWO LOGOS SHARE A Y-SCALE, AND THAT IS THE POINT OF THE PANEL. Drawn independently,
matplotlib autoscales each to its own range, so a non-AS logo carrying a tenth of the signal
would fill its axis exactly as fully as the AS one and the pair would look equivalent. The
comparison here is of MAGNITUDE -- does the head attribute more structure at AS loci than at
non-AS ones -- so both axes are forced to the same symmetric limit and the limit is printed.

WHAT TO EXPECT, AND WHY A FLAT NON-AS LOGO IS A RESULT. Binding is motif-driven but allele-
specific binding largely is not: only 4.2% of CTCF ENC-002 trunk-ISM windows carry a canonical
MA0139.2 hit covering the variant, and ~87% of AS windows have no CTCF motif within +/-10 bp.
So neither logo should be read as "the model found the motif" -- the averaged contribution is
dominated by windows with no motif to find. The informative quantity is the DIFFERENCE between
the two panels, and the motif/flank amplitude ratios printed for each.

NO TRUNK HERE. For the trunk-and-head version with peak-vs-background tracks, use
fig_ism_composite. This figure is the head's own story: where its saliency sits by prediction
outcome, what cofactor context those windows have, and whether its attribution looks different
at AS and non-AS loci.

NO PANEL MATH LIVES HERE -- every panel calls the function its standalone script uses:

    A    fig_ism_confusion_grid.load_assay_confusion -> draw_confusion_panel
    B    plot_as_motif_enrichment.compute_density    -> draw_motif_enrichment
    C,D  plot_ism_logo.align_contributions           -> draw_ism_logo
"""
import os
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

import argparse
import json
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hapadapter.figures.plot_as_motif_enrichment import (compute_density, draw_motif_enrichment,
                                                         load_motifs, motif_hit_mask,
                                                         normalise_density)
from hapadapter.figures.plot_ism_logo import (align_contributions, centre_contributions,
                                              draw_ism_logo, pfm_to_prob, prob_to_ic)
from hapadapter.figures.fig_ism_confusion_grid import (load_assay_confusion,
                                                       draw_confusion_panel)


def pretty_individual(raw):
    """"ENC-002" -> "individual 2", matching the other figures in this package."""
    if not raw:
        return ""
    m = re.fullmatch(r"ENC-0*(\d+)", str(raw).strip(), flags=re.I)
    return f"individual {int(m.group(1))}" if m else str(raw)


def placeholder(ax, name, why, fs):
    # A panel that was never allocated (axD when --logos AS) has nothing to label. Returning
    # here rather than at every call site means a future panel cannot reintroduce the
    # AttributeError by forgetting to filter.
    if ax is None:
        return
    ax.text(0.5, 0.5, f"{name}\n({why})", ha="center", va="center",
            transform=ax.transAxes, fontsize=fs, color="0.45")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("0.85")
    print(f"  [skip] {name}: {why}")


def density_seqs_from_csv(paths, seq_col, label_col, dedup=True):
    """
    AS and non-AS sequences for the DENSITY panels, pooled over one or more windows CSVs.

    SEPARATE FROM THE LOGO'S WINDOWS ON PURPOSE. compute_density is FIMO over sequences; no
    model, checkpoint or prediction enters it. The leak filter and the held-out restriction
    exist so the HEAD's attributions do not describe loci it memorised, and a motif scan has
    nothing to memorise -- so this panel can use every locus in the assay while the saliency and
    logo panels stay on the held-out set.

    DUPLICATE SEQUENCES ARE DROPPED BY DEFAULT. Pooling donors repeats a locus whenever no
    donor-specific variant falls inside the window, and a byte-identical sequence carries no
    extra motif information. Counting it would shrink the apparent shot noise without shrinking
    the real shot noise, which is the one thing this panel is being enlarged to fix.
    """
    import pandas as pd
    frames = []
    for q in paths:
        d = pd.read_csv(q, usecols=lambda c: c in (seq_col, label_col))
        for need in (seq_col, label_col):
            if need not in d.columns:
                raise SystemExit(f"{q} has no {need!r} column")
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    n_raw = len(d)
    if dedup:
        # PER LABEL GROUP, not globally. A locus can be AS in one split/donor and non-AS in
        # another; a global drop_duplicates would keep whichever row came first and assign the
        # sequence to that group by file order. Deduping within each group keeps both
        # occurrences and the overlap is reported below, where it is a fact about the labels
        # rather than an artefact of row order.
        d = d.drop_duplicates(subset=[seq_col, label_col])
    # TWO LABEL VOCABULARIES. build_asb_ism_windows writes as_label = "AS"/"nonAS"; the Stage-2
    # build CSVs written by build_inputs rename imbalance_significance to label = 1/0. Accept
    # both so the panel can be fed from either, and refuse anything else rather than silently
    # classing an unrecognised value as non-AS.
    lab = d[label_col].astype(str).str.strip().str.upper()
    AS_V, NON_V = {"AS", "1", "1.0", "TRUE"}, {"NONAS", "NON-AS", "0", "0.0", "FALSE"}
    unknown = sorted(set(lab) - AS_V - NON_V)
    if unknown:
        raise SystemExit(f"--density_label_col {label_col!r} has unrecognised value(s) "
                         f"{unknown[:6]}; expected AS/nonAS or 1/0")
    A = [str(x) for x in d.loc[lab.isin(AS_V), seq_col]]
    N = [str(x) for x in d.loc[lab.isin(NON_V), seq_col]]
    both = len(set(A) & set(N))
    src = (f"{len(paths)} CSV(s), {n_raw:,} rows -> {len(d):,} unique (sequence, label) pairs "
           f"({len(A):,} AS / {len(N):,} non-AS)")
    print(f"  [density] {src}")
    if dedup and n_raw != len(d):
        print(f"  [density] dropped {n_raw - len(d):,} byte-identical sequences "
              f"({(n_raw - len(d)) / max(n_raw, 1):.0%}); pass --density_keep_duplicate_seqs "
              f"to count them.")
    if both:
        print(f"  [density] {both:,} sequence(s) appear in BOTH groups -- the same locus called "
              f"AS in one\n            split or donor and non-AS in another. They are counted "
              f"on both sides, which\n            is the conservative choice: it cannot "
              f"manufacture a difference between the groups.")
    if not A or not N:
        raise SystemExit(f"--density_csv gave {len(A)} AS and {len(N)} non-AS sequences; "
                         f"check --density_label_col (values seen: "
                         f"{sorted(lab.unique())[:6]})")
    return A, N, src


def npz_seqs(path):
    with np.load(path, allow_pickle=True) as z:
        if "seqs" not in z:
            raise SystemExit(f"{path} has no 'seqs' array; keys are {sorted(z.files)}")
        return [str(s) for s in z["seqs"]]


def logo_agg(path, ref_profile, W, flank, strand_aware, control="none", seed=0, keep=None):
    """
    Aligned average for one class, plus counts and the per-window motif offsets.

    keep: optional boolean mask over windows, applied BEFORE alignment. Restricting to windows
    that contain a motif changes the question from "what does the head attribute across AS
    windows" to "when a site is present, does the head attribute to it" -- a sharper question,
    and one the unrestricted average cannot answer because windows with no site dominate it and
    align to whatever noise scores best.
    """
    with np.load(path, allow_pickle=True) as z:
        for need in ("onehot", "contrib", "importance"):
            if need not in z:
                raise SystemExit(f"{path} has no {need!r}; keys are {sorted(z.files)}")
        if keep is not None:
            if len(keep) != z["onehot"].shape[0]:
                raise SystemExit(f"{path}: motif filter has {len(keep)} entries for "
                                 f"{z['onehot'].shape[0]} windows")
            if not keep.any():
                return None
            z = {k: (z[k][keep] if getattr(z[k], "shape", (0,))[:1] == (len(keep),) else z[k])
                 for k in ("onehot", "contrib", "importance")}
        agg, used, n_flip, offs = align_contributions(
            z["onehot"], z["contrib"], z["importance"], ref_profile, W, flank,
            control=control, strand_aware=strand_aware, return_offsets=True,
            rng=np.random.default_rng(seed))
        total, Lw = int(z["onehot"].shape[0]), int(z["onehot"].shape[1])
    return agg, int(used), int(n_flip), total, offs, Lw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assay", required=True)
    ap.add_argument("--donor", default="")

    ap.add_argument("--as_npz", required=True, help="ism_head_<arm>_AS.npz")
    ap.add_argument("--nonas_npz", required=True, help="ism_head_<arm>_nonAS.npz")

    ap.add_argument("--exp_root", default=None, help="panel A")
    ap.add_argument("--arm", default="personal", help="panel A")
    ap.add_argument("--scores_dir", default=None, help="panel A")
    ap.add_argument("--score_col", default="delta", help="panel A ranking column")
    ap.add_argument("--threshold", default="0", help="panel A; 0 = p=0.5, 'auto' = Youden")
    ap.add_argument("--donor_kind", default="all", help="panel A")
    ap.add_argument("--universe", default="ref", choices=["ref", "none"], help="panel A")
    ap.add_argument("--keep_leaky", action="store_true", help="panel A")
    ap.add_argument("--window", type=int, default=100, help="panel A half-width (bp)")
    ap.add_argument("--conf_smooth", type=int, default=3, help="panel A smoothing")

    ap.add_argument("--jaspar", default=None, help="panel B motif JSON")
    ap.add_argument("--tfs", default="CTCF", help="panel B; comma-separated cofactors")
    ap.add_argument("--fimo_threshold", type=float, default=1e-4)
    ap.add_argument("--smooth", type=int, default=21,
                    help="density smoothing width in bp. At the cofactor level these curves are "
                         "counting noise rather than structure: a motif at density 0.01 over 588 "
                         "windows is ~6 windows per position, so the shot noise is ~41%% of the "
                         "mean. 21 bp with a Gaussian kernel cuts roughness ~2.5x while keeping "
                         "a real central peak at 98%% of its height.")
    ap.add_argument("--density_kernel", choices=["gaussian", "box"], default="gaussian",
                    help="'box' is the moving average plot_as_motif_enrichment uses; it leaks "
                         "high frequencies and leaves the sharp spikes that make a rare motif's "
                         "counting noise read as structure. 'gaussian' (sigma = --smooth/4) is "
                         "strictly better at the same nominal width -- measured on a simulated "
                         "profile at the cofactor rate, roughness 3.4 against 8.7. Use 'box' "
                         "only to reproduce an earlier figure exactly.")
    ap.add_argument("--density_csv", nargs="+", default=None, metavar="CSV",
                    help="windows CSV(s) for the DENSITY panels only, instead of the ISM npz. "
                         "The density panels run FIMO over sequences and never touch the model, "
                         "so they are not bound by the leak filter or the held-out split that "
                         "the saliency and logo panels need -- pass a build made WITHOUT "
                         "--train_coords/--restrict_coords, and several (one per donor) to "
                         "pool. This makes the density panels describe a DIFFERENT, larger "
                         "locus set than the logo; the panel titles carry their own n and the "
                         "difference is recorded in --csv_out.")
    ap.add_argument("--density_seq_col", default="sequence1",
                    help="--density_csv: sequence column. sequence1 is haplotype 1.")
    ap.add_argument("--density_label_col", default="as_label")
    ap.add_argument("--density_keep_duplicate_seqs", action="store_true",
                    help="--density_csv: keep byte-identical sequences. Off by default: pooling "
                         "donors repeats loci where no donor-specific variant falls in the "
                         "window, and an identical sequence adds no motif information while "
                         "inflating n -- which would narrow the apparent noise without "
                         "narrowing the real noise.")
    ap.add_argument("--density_flank", type=int, default=75,
                    help="bp either side of the variant to SHOW in the density facets. The "
                         "curves are computed over the whole 257 bp window; this crops the view. "
                         "75 matches plot_as_motif_enrichment's default. Showing the full +/-128 "
                         "in a 2.6 in facet packs ~99 points per inch against the standalone's "
                         "~38, which renders a rare motif's shot noise as visible spikes.")
    ap.add_argument("--density_norm", choices=["none", "fold"], default="none",
                    help="'fold' rescales each motif to its OWN flank baseline, pooled across "
                         "AS and non-AS. Use it when a dominant motif flattens the cofactors: "
                         "CTCF peaks near 0.155 here while SP1/YY1 sit around 0.01, so on a "
                         "shared absolute axis the cofactors carry no visible structure. The "
                         "pooled denominator keeps the AS-vs-non-AS difference intact; a "
                         "per-group one would divide it out.")
    ap.add_argument("--density_edge_frac", type=float, default=0.25,
                    help="--density_norm fold: fraction of the window at EACH end used as the "
                         "baseline region.")

    ap.add_argument("--pwm_json", default=None, help="panels C/D alignment reference")
    ap.add_argument("--matrix_id", default="MA0139.2", help="panels C/D")
    ap.add_argument("--logos", choices=["AS", "both"], default="AS",
                    help="which contribution logos to DRAW. 'both' adds the non-AS panel. The "
                         "non-AS aggregate is computed either way: it sets the shared y-limit "
                         "and its peak goes to --csv_out, so the AS panel's height stays "
                         "comparable and the claim that the two look alike remains checkable "
                         "from the table even when only one is shown.")
    ap.add_argument("--logo_align", choices=["variant", "motif"], default="motif",
                    help="'variant' (default) averages a FIXED window centred on the variant -- "
                         "the question a head figure asks, since the variant is the only "
                         "position whose identity differs between haplotypes. 'motif' slides "
                         "each window to its best match against --matrix_id first, which suits "
                         "a trunk logo but places the AS windows with no canonical motif at "
                         "whatever offset scores best out of noise.")
    ap.add_argument("--logo_require_motif", default=None, metavar="TF",
                    help="restrict the LOGO to windows containing a FIMO hit for this TF (a "
                         "name from --tfs) within --logo_require_within bp of the variant. The "
                         "unrestricted aligned average is dominated by windows with no site -- "
                         "for CTCF about 87%% of them -- which align to whatever noise scores "
                         "best, so the mean reports the alignment procedure. This asks the "
                         "sharper question instead: WHEN a site is present, does the head "
                         "attribute to it? Applied identically to both classes and to the "
                         "--logo_control nulls, so every comparison stays on one subset.")
    ap.add_argument("--logo_require_within", type=int, default=25,
                    help="--logo_require_motif: max distance (bp) from the variant to the hit.")
    ap.add_argument("--logo_subsample", type=int, default=0, metavar="N",
                    help="cap each class at N windows (0 = no cap), drawn without replacement "
                         "after the motif filter. THIS IS A POWER CONTROL, not a convenience: "
                         "comparing one motif's aligned logo against another's is confounded "
                         "when their hit rates differ -- CTCF+ gives ~183 AS windows and SP1+ "
                         "about 29, so a flat SP1 logo cannot be told apart from an "
                         "underpowered one. Re-running the STRONG motif at the WEAK motif's N "
                         "is what separates specificity from sample size.")
    ap.add_argument("--logo_subsample_seed", type=int, default=0)
    ap.add_argument("--logo_control", choices=["none", "shuffle_pwm", "shuffle_contrib"],
                    default="none",
                    help="--logo_align motif only. best_offset takes the MAXIMUM correlation "
                         "over ~239 candidate offsets, which finds motif-like segments in "
                         "sequence that contains no motif -- so a clean aligned logo is not "
                         "evidence by itself. 'shuffle_contrib' permutes each window's "
                         "positions before aligning and 'shuffle_pwm' shuffles the reference "
                         "columns; either run gives the logo this same selection procedure "
                         "produces from noise. Compare logo_AS_motif_over_flank between the "
                         "real run and the control: if they are close, the match is the "
                         "procedure, not the model.")
    ap.add_argument("--logo_control_seed", type=int, default=0)
    ap.add_argument("--logo_halfwidth", type=int, default=20,
                    help="--logo_align variant: bp either side of the variant to show.")
    ap.add_argument("--logo_flank", type=int, default=5,
                    help="--logo_align motif: bp either side of the motif span.")
    ap.add_argument("--strand_aware", action="store_true")
    ap.add_argument("--flip_logo", action="store_true",
                    help="draw the contribution logos pointing UP. An ASB head's contributions "
                         "are negative almost everywhere, which puts the logo below the axis "
                         "where logomaker draws characters upside down. This negates the matrix "
                         "so letters read upright and relabels the ticks so the axis still "
                         "shows the true (negative) values -- say so in the caption.")
    ap.add_argument("--show_reference", dest="show_reference", action="store_true", default=True,
                    help="draw the reference PWM strip under the two logos (default)")
    ap.add_argument("--no_reference", dest="show_reference", action="store_false",
                    help="omit the reference strip; both logos are still ALIGNED to it, so "
                         "--pwm_json stays required")

    ap.add_argument("--size", type=float, nargs=2, default=None,
                    help="default scales with the number of motifs: 2.6 in per density facet "
                         "(min 9.5) by 9.8 in. Four facets on a 9.5 in canvas leaves each about "
                         "2 in of axis, which is too narrow for a 257 bp profile.")
    ap.add_argument("--fontsize", type=float, default=8)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv_out", default=None)
    a = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    fs = a.fontsize
    who = ", ".join(x for x in (a.assay, pretty_individual(a.donor)) if x)
    tfs = [t.strip() for t in a.tfs.split(",") if t.strip()]
    summary = {}

    # Motifs are resolved BEFORE the figure exists because they set the number of density
    # facets, and the gridspec column count cannot be changed afterwards.
    motifs = {}
    if a.jaspar and tfs:
        motifs = load_motifs(a.jaspar, tfs)
        missing = [t for t in tfs if t.upper() not in motifs]
        if missing:
            print(f"  [warn] motifs not found in {a.jaspar}: {missing}")
        for tf, mats in motifs.items():
            print(f"  [motifs] {tf}: {len(mats)} JASPAR matrices {sorted(mats)} (unioned)")

    # ONE PANEL PER MOTIF, EACH WITH ITS OWN Y-AXIS. A shared axis is unreadable here because
    # absolute motif densities differ by an order of magnitude between a figure's primary TF and
    # its cofactors -- CTCF near 0.09 against KLF4 near 0.02 -- so every cofactor is pressed flat
    # against the bottom and a real centred peak in one of them is invisible. Faceting gives each
    # motif the full height of its own axis, which is what makes a cofactor's enrichment legible,
    # and it needs no denominator: the curves stay in directly interpretable density units rather
    # than a derived fold value whose baseline can be unstable when a motif is rare.
    # The reference strip is a MOTIF-ALIGNMENT artefact: it states "the motif sits here", which
    # is only true when the windows were slid to a common motif position. Decided before the
    # gridspec so variant mode drops the row outright rather than leaving it blank.
    show_ref = a.show_reference and a.logo_align != "variant"
    n_facet = max(1, len(motifs))
    # 3.2 in per facet, not 2.6: at 2.6 a 151 bp cropped profile still sits at ~58 points per
    # inch, against ~38 in the standalone figure these facets are meant to reproduce.
    size = a.size or [max(9.5, 3.2 * n_facet), 9.8]
    _motif_mc = None
    _motif_W = 1
    _logo_span = _logo_spread = None
    fig = plt.figure(figsize=tuple(size))
    both_logos = a.logos == "both"
    nrow = (4 if both_logos else 3) + (1 if show_ref else 0)
    heights = [1.05, 0.95, 0.85] + ([0.85] if both_logos else []) + \
              ([0.45] if show_ref else [])
    gs = fig.add_gridspec(nrow, n_facet, height_ratios=heights, hspace=0.62, wspace=0.42)
    axA = fig.add_subplot(gs[0, :])
    axB = [fig.add_subplot(gs[1, i]) for i in range(n_facet)]
    axC = fig.add_subplot(gs[2, :])
    axD = fig.add_subplot(gs[3, :]) if both_logos else None
    axE = fig.add_subplot(gs[nrow - 1, :]) if show_ref else None
    # C, D and E are the same positional axis; only the bottom one keeps tick labels.
    if a.logo_align == "variant":
        # Each logo carries its OWN variant-relative axis in this mode, so both keep their tick
        # labels. Suppressing them (correct when the three panels share one motif coordinate)
        # would leave the x-label naming an axis whose numbers are hidden.
        pass
    else:
        axC.tick_params(labelbottom=False)
        if axE is not None:
            (axD or axC).tick_params(labelbottom=False)

    # ---- cofactor motif density, one facet per motif ------------------------------------
    if motifs:
        from memelite import fimo
        if a.density_csv:
            A_seqs, N_seqs, _dsrc = density_seqs_from_csv(
                a.density_csv, a.density_seq_col, a.density_label_col,
                dedup=not a.density_keep_duplicate_seqs)
        else:
            A_seqs, N_seqs = npz_seqs(a.as_npz), npz_seqs(a.nonas_npz)
            _dsrc = "ism npz (held-out, leak-filtered -- same loci as the logo)"
        L0 = len(A_seqs[0])
        A_seqs = [s_ for s_ in A_seqs if len(s_) == L0]
        N_seqs = [s_ for s_ in N_seqs if len(s_) == L0]
        groups = [("AS", compute_density(A_seqs, motifs, fimo, a.fimo_threshold)),
                  ("non-AS", compute_density(N_seqs, motifs, fimo, a.fimo_threshold))]
        flank = L0 // 2
        if a.density_kernel == "gaussian":
            # Smooth HERE and hand the draw function already-smoothed curves (smooth_w=1), so
            # plot_as_motif_enrichment keeps its box-kernel behaviour byte-for-byte for the
            # standalone figures that were produced with it.
            from scipy.ndimage import gaussian_filter1d
            sig = max(a.smooth, 1) / 4.0
            groups = [(g, {k: gaussian_filter1d(v, sig) for k, v in dd.items()})
                      for g, dd in groups]
            smooth_w = 1
        else:
            smooth_w = a.smooth
        dlab = "motif density"
        if a.density_norm == "fold":
            groups, _base, _kept = normalise_density(groups, list(motifs),
                                                     edge_frac=a.density_edge_frac)
            dlab = "motif density (fold over flank)"
        order = list(motifs)
        for i, nm in enumerate(order):
            ax = axB[i]
            if a.density_norm == "fold":
                ax.axhline(1.0, color="0.6", lw=0.8, ls=":", zorder=1)
            draw_motif_enrichment(
                ax, groups, [nm], flank, smooth_w=smooth_w, title=nm,
                fontsize=fs, legend=(i == 0),
                # y-label on the leftmost facet only: the axes carry DIFFERENT scales, which is
                # the point of faceting, but repeating the same words beside each one wastes the
                # width the facets need.
                # n BELONGS TO THE ROW, NOT TO THE FIRST MOTIF. Hanging it under the leftmost
                # panel's TITLE read as "SP1: n=..." when SP1 happened to be listed first; every
                # facet is computed from the same sequences. On the shared y-label it attaches
                # to the quantity instead, and it is stated once rather than four times.
                ylabel=(f"{dlab}\n(n={len(A_seqs):,} AS / {len(N_seqs):,} non-AS)"
                        if i == 0 else ""),
                xlabel="position relative to variant (bp)" if i == len(order) - 1 else "")
            # Crop the VIEW, not the computation: the density is still calculated over the whole
            # window so edge positions are not special-cased, but the shown range matches the
            # standalone figure rather than packing 257 points into a 3 in facet.
            if a.density_flank and a.density_flank < flank:
                ax.set_xlim(-a.density_flank, a.density_flank)
            # EACH GROUP AGAINST ITS OWN FLANK. The AS and non-AS curves can sit at different
            # LEVELS for reasons that have nothing to do with position -- AS loci are GC-richer,
            # so every GC-rich motif is lifted everywhere, flanks included. A between-group
            # ratio at the variant cannot tell that apart from positional enrichment. Dividing
            # each group by its OWN flank does: a level offset cancels, and what survives is how
            # much the motif concentrates at the variant WITHIN that group.
            #
            # The flank is the outer third of the SHOWN window, so the number matches what the
            # reader sees rather than reaching into cropped-off positions.
            _fl = a.density_flank if (a.density_flank and a.density_flank < flank) else flank
            _c = slice(flank - 5, flank + 6)                    # +/-5 bp of the variant
            _lo, _hi = int(_fl * 2 / 3), _fl
            for glab, dens in groups:
                y = dens[nm]
                centre = float(np.mean(y[_c]))
                base = float(np.mean(np.r_[y[flank - _hi:flank - _lo],
                                           y[flank + _lo + 1:flank + _hi + 1]]))
                summary[f"density_{nm}_center_{glab}"] = round(float(y[flank]), 5)
                summary[f"density_{nm}_flank_{glab}"] = round(base, 5)
                summary[f"density_{nm}_center_over_flank_{glab}"] = (
                    round(centre / base, 4) if base > 0 else float("nan"))
        summary["logo_control"] = a.logo_control
        summary["density_source"] = _dsrc
        summary["density_same_loci_as_logo"] = int(not a.density_csv)
        summary["density_n_AS"] = len(A_seqs)
        summary["density_n_nonAS"] = len(N_seqs)
        print(f"  [density] {len(order)} facets, independent y-axes: {', '.join(order)}")
    else:
        placeholder(axB[0], "cofactor motif density", "no --jaspar/--tfs", fs)
        for ax in axB[1:]:
            ax.axis("off")

    # ---- the contribution logo(s), on ONE scale ------------------------------------------
    # THE PWM IS ONLY NEEDED FOR MOTIF ALIGNMENT. Under --logo_align variant the logo is a fixed
    # slice around the variant: no reference profile is consulted, nothing is slid, and the
    # reference strip is already dropped. Requiring --pwm_json there turned a perfectly valid
    # variant-centred run into a placeholder -- which is exactly the combination an assay with
    # no motif of its own (POLR2A, ATAC, the histone marks) has to use.
    variant_mode = a.logo_align == "variant"
    _need_pwm = (not variant_mode) or show_ref
    _pj = _entry = None
    W, ref_ic, tf = 0, None, None
    if a.pwm_json and os.path.exists(a.pwm_json):
        _pj = json.load(open(a.pwm_json))
        if a.matrix_id in _pj:
            _entry = _pj[a.matrix_id]
            P = pfm_to_prob(_entry.get("pfm") or _entry.get("counts"))
            if a.logo_control == "shuffle_pwm":
                # IMPLEMENTED HERE, NOT IN align_contributions, which only handles
                # shuffle_contrib -- passing "shuffle_pwm" down returned results byte-identical
                # to the real run, i.e. a control that silently did nothing.
                #
                # Permuting POSITIONS keeps the matrix's base composition and information
                # content exactly and destroys only its positional structure, which is the
                # property best_offset matches on. The resulting logo is what this selection
                # procedure extracts from a reference profile carrying no real motif shape.
                P = P[np.random.default_rng(a.logo_control_seed).permutation(P.shape[0])]
            W = P.shape[0]
            ref_ic = prob_to_ic(P)
            tf = _entry.get("name", a.matrix_id)

    if _need_pwm and _entry is None:
        why = ("no --pwm_json" if _pj is None
               else f"{a.matrix_id} not in {os.path.basename(a.pwm_json)}")
        # axD is None whenever --logos AS (the default), so the axis list must be filtered:
        # placeholder() takes an axis and calls .text on it.
        for ax in (x for x in (axC, axD) if x is not None):
            placeholder(ax, "contribution logo", why, fs)
        if axE is not None:
            axE.axis("off")
    else:
        if True:

            # BOTH aggregates are computed BEFORE either is drawn, because the shared y-limit
            # is a property of the pair. Drawing C then deciding D's scale would either
            # rescale C after the fact or leave the two axes inconsistent.
            # MOTIF-PRESENCE FILTER, computed once per class and shared by every arm and null.
            _keep, _kept = {}, {}
            if a.logo_require_motif:
                if not motifs:
                    raise SystemExit("--logo_require_motif needs --jaspar/--tfs to scan with")
                _rtf = next((k for k in motifs if k.upper() == a.logo_require_motif.upper()), None)
                if _rtf is None:
                    raise SystemExit(f"--logo_require_motif {a.logo_require_motif!r} is not in "
                                     f"--tfs; loaded {sorted(motifs)}")
                from memelite import fimo as _fimo
                for lab, path in (("AS", a.as_npz), ("non-AS", a.nonas_npz)):
                    sq = npz_seqs(path)
                    Lw_ = len(sq[0])
                    mask = motif_hit_mask(sq, {_rtf: motifs[_rtf]}, _fimo, a.fimo_threshold)[_rtf]
                    lo = max(0, Lw_ // 2 - a.logo_require_within)
                    hi = min(Lw_, Lw_ // 2 + a.logo_require_within + 1)
                    _keep[lab] = mask[:, lo:hi].any(axis=1)
                    _kept[lab] = (int(_keep[lab].sum()), len(sq))
                    print(f"  [logo] {lab}: {_kept[lab][0]}/{_kept[lab][1]} windows carry a "
                          f"{_rtf} hit within +/-{a.logo_require_within} bp "
                          f"({_kept[lab][0] / max(_kept[lab][1], 1):.0%})")
                    summary[f"logo_{lab}_motif_filter_kept"] = _kept[lab][0]
                    summary[f"logo_{lab}_motif_filter_total"] = _kept[lab][1]
                summary["logo_require_motif"] = _rtf
                summary["logo_require_within"] = a.logo_require_within

            # SUBSAMPLE AFTER THE FILTER, independently per class but on one seed, so a matched-N
            # run is reproducible and both classes are thinned the same way.
            if a.logo_subsample > 0:
                _rng = np.random.default_rng(a.logo_subsample_seed)
                for lab, path in (("AS", a.as_npz), ("non-AS", a.nonas_npz)):
                    with np.load(path, allow_pickle=True) as z:
                        _n = int(z["onehot"].shape[0])
                    m = _keep.get(lab)
                    if m is None:
                        m = np.ones(_n, dtype=bool)
                    idx = np.flatnonzero(m)
                    if len(idx) > a.logo_subsample:
                        drop = _rng.choice(idx, size=len(idx) - a.logo_subsample, replace=False)
                        m = m.copy(); m[drop] = False
                        print(f"  [logo] {lab}: subsampled {len(idx)} -> {int(m.sum())} windows "
                              f"(seed {a.logo_subsample_seed})")
                    else:
                        print(f"  [logo] {lab}: {len(idx)} windows, below the "
                              f"--logo_subsample cap of {a.logo_subsample}; not thinned")
                    _keep[lab] = m
                    summary[f"logo_{lab}_subsampled_to"] = int(m.sum())
                summary["logo_subsample"] = a.logo_subsample

            packs = {}
            for lab, path in (("AS", a.as_npz), ("non-AS", a.nonas_npz)):
                if variant_mode:
                    with np.load(path, allow_pickle=True) as z:
                        agg, used = centre_contributions(z["contrib"], a.logo_halfwidth)
                        total = int(z["contrib"].shape[0])
                    packs[lab] = (agg, used, 0, total, None, None)
                else:
                    packs[lab] = logo_agg(path, ref_ic.sum(axis=1), W, a.logo_flank,
                                          a.strand_aware, control=a.logo_control,
                                          seed=a.logo_control_seed, keep=_keep.get(lab))
            if any(v is None for v in packs.values()):
                _empty = [k for k, v in packs.items() if v is None]
                raise SystemExit(
                    f"--logo_require_motif {a.logo_require_motif} left {', '.join(_empty)} with "
                    f"no windows. Widen --logo_require_within, raise --fimo_threshold, or drop "
                    f"the filter -- a logo cannot be averaged from nothing.")
            ymax = max(float(np.abs(p[0]).sum(axis=1).max()) for p in packs.values())
            ylim = ymax * 1.08

            # WHERE THE VARIANT FALLS IN MOTIF COORDINATES. The variant sits at Lw//2 in every
            # window, so after aligning to a motif starting at `off` it lands at
            # flank + (Lw//2 - off) on the logo's axis. Pooled over AS and non-AS because the
            # band marks the same thing on both panels.
            vpos = vq = None
            if not variant_mode:
                allo, Lw = [], None
                for _lab, pk in packs.items():
                    if pk[4] is not None:
                        allo.append(pk[4]); Lw = pk[5]
                if allo:
                    o = np.concatenate(allo)
                    vp = a.logo_flank + (Lw // 2 - o)
                    vpos = float(np.median(vp))
                    vq = (float(np.percentile(vp, 25)), float(np.percentile(vp, 75)))
                    # ONLY MARK A POSITION THAT IS ACTUALLY CONCENTRATED. Windows with no real
                    # motif align to whatever scores best out of noise, so their offsets are
                    # near-uniform over the window. When they dominate, the interquartile range
                    # spans most of the panel -- and a median line plus a band covering
                    # everything claims a location the alignment did not find. Suppress both and
                    # report the number instead, which is the honest form of the same answer.
                    _iqr = vq[1] - vq[0]
                    if _iqr > 2 * W:
                        print(f"  [align] variant position NOT marked: IQR is {_iqr:.0f} bp "
                              f"against a {W} bp motif, so the\n          offsets are not "
                              f"concentrated -- most windows carry no match and align to noise.\n"
                              f"          Median {vpos:.1f} and quartiles are in --csv_out.")
                        vpos = vq = None
                    # In VARIANT coordinates, for the top saliency panel.
                    mc = (o + W / 2.0) - Lw // 2
                    summary["motif_center_vs_variant_median"] = round(float(np.median(mc)), 2)
                    summary["motif_center_vs_variant_q25"] = round(float(np.percentile(mc, 25)), 2)
                    summary["motif_center_vs_variant_q75"] = round(float(np.percentile(mc, 75)), 2)
                    _motif_mc = mc
                    _motif_W = W
                    # THE LOGO'S OWN FOOTPRINT, in variant-relative bp. The logo shows
                    # [off - flank, off + W + flank) of each window; at the median offset that
                    # is this span once the variant (at Lw//2) is taken as zero. It is a fixed
                    # WIDTH -- W + 2*flank -- whose POSITION varies per window, so the label
                    # carries the spread rather than implying the band is exact.
                    _mo = float(np.median(o))
                    _logo_span = (_mo - a.logo_flank - Lw // 2,
                                  _mo + W + a.logo_flank - Lw // 2)
                    _logo_spread = float(np.percentile(o, 75) - np.percentile(o, 25))
                    summary["logo_window_lo_bp"] = round(_logo_span[0], 1)
                    summary["logo_window_hi_bp"] = round(_logo_span[1], 1)
                    summary["logo_offset_iqr_bp"] = round(_logo_spread, 1)
                    print(f"  [align] logo covers {_logo_span[0]:+.0f} to {_logo_span[1]:+.0f} bp "
                          f"relative to the variant at the median offset "
                          f"({W + 2 * a.logo_flank} bp wide; per-window offset IQR "
                          f"{_logo_spread:.0f} bp)")
                    print(f"  [align] motif centre vs variant: median "
                          f"{np.median(mc):+.1f} bp, IQR [{np.percentile(mc, 25):+.1f}, "
                          f"{np.percentile(mc, 75):+.1f}] over {len(mc)} windows")
                    if vpos is not None:
                        print(f"  [align] variant falls at logo position {vpos:.1f} "
                              f"(IQR {vq[0]:.1f}-{vq[1]:.1f}); the motif span is "
                              f"{a.logo_flank}-{a.logo_flank + W}")
            # MEASURE BOTH CLASSES, DRAW ONLY THE REQUESTED ONES. The non-AS numbers are what
            # make "the two logos look the same" a checkable statement rather than an
            # impression, so they go to --csv_out whether or not that panel is rendered.
            for lab in ("AS", "non-AS"):
                agg, used, n_flip, total = packs[lab][:4]
                if variant_mode:
                    h = a.logo_halfwidth
                    m_amp = float(np.abs(agg[h - 2:h + 3]).sum(axis=1).mean())
                    k = max(1, (2 * h + 1) // 4)
                    f_amp = float(np.abs(np.r_[agg[:k], agg[-k:]]).sum(axis=1).mean())
                else:
                    m_amp = float(np.abs(agg[a.logo_flank:a.logo_flank + W]).sum(axis=1).mean())
                    f_amp = (float(np.abs(agg[:a.logo_flank]).sum(axis=1).mean())
                             if a.logo_flank else float("nan"))
                summary[f"logo_{lab}_windows_used"] = used
                summary[f"logo_{lab}_windows_total"] = total
                summary[f"logo_{lab}_strand_flips"] = n_flip
                summary[f"logo_{lab}_peak_contribution"] = round(
                    float(np.abs(agg).sum(axis=1).max()), 5)
                summary[f"logo_{lab}_center_over_flank" if variant_mode
                        else f"logo_{lab}_motif_over_flank"] = (
                    round(m_amp / f_amp, 4) if f_amp and f_amp > 0 else float("nan"))
                summary[f"logo_{lab}_drawn"] = int(lab == "AS" or axD is not None)

            _panels = [(axC, "AS")] + ([(axD, "non-AS")] if axD is not None else [])
            for ax, lab in _panels:
                agg, used, n_flip, total = packs[lab][:4]
                where = (f"centred on the variant, +/-{a.logo_halfwidth} bp" if variant_mode
                         else f"aligned to {tf}"
                              + ("" if not a.logo_require_motif
                                 else f", {a.logo_require_motif}+ windows only")
                              + ("" if a.logo_control == "none"
                                 else f" -- {a.logo_control.upper()} CONTROL"))
                draw_ism_logo(ax, agg, a.logo_flank, W,
                              title=f"{lab} contribution logo "
                                    f"(mean of {used}/{total} windows, {where})",
                              fontsize=fs, flip=a.flip_logo,
                              xoffset=a.logo_halfwidth if variant_mode else None,
                              shade=not variant_mode, mark_center=variant_mode,
                              # The span is already named by the panel title ("aligned to
                              # CTCF") and shown by the shaded band; a third statement of it sat
                              # on top of the letters.
                              shade_label=None,
                              mark_at=vpos,
                              mark_label=None if vpos is None else "median variant")
                if vq is not None:
                    # The variant is NOT at a fixed position under motif alignment; the band is
                    # the middle half of where it actually falls. A bare median line would imply
                    # a precision the alignment does not have.
                    ax.axvspan(vq[0], vq[1], color="red", alpha=0.07, zorder=0)
                if variant_mode:
                    ax.set_xlabel("position relative to variant (bp)",
                                  **({} if fs is None else {"fontsize": fs}))
                # Symmetric and identical on every panel, and set from BOTH aggregates even
                # when only one is drawn: the comparison is of magnitude, and autoscaling the
                # surviving panel would make its height meaningless across runs.
                ax.set_ylim(-ylim, ylim)
            summary["logo_shared_ylim"] = round(ylim, 5)
            print(f"  [logo] shared y-limit +/-{ylim:.4g}; peak |contribution| "
                  f"AS {summary['logo_AS_peak_contribution']:.4g} vs "
                  f"non-AS {summary['logo_non-AS_peak_contribution']:.4g}")

            if axE is not None and variant_mode:
                # The JASPAR strip states "the motif sits HERE". Under variant centring the motif
                # falls at a different offset in every window, so printing it beneath the logos
                # would assert an alignment that was never performed.
                axE.axis("off")
                print("  [logo] reference PWM strip omitted: with --logo_align variant there is "
                      "no common\n         motif position for it to line up with. "
                      "Pass --logo_align motif for that comparison.")
            elif axE is not None:
                # One reference strip for both logos: they are aligned to the SAME profile, so
                # drawing it twice would claim two references where there is one.
                draw_ism_logo(axE, np.zeros((a.logo_flank * 2 + W, 4)), a.logo_flank, W,
                              ax_ref=axE, ref_ic=ref_ic,
                              ref_title=f"JASPAR {a.matrix_id} ({tf}) reference, bits",
                              fontsize=fs)

    # DRAWN LAST, THOUGH IT IS THE TOP PANEL. Under --logo_align motif the logo block measures
    # where each window's motif sits relative to the variant, and this panel marks that position
    # -- so it has to run after the measurement exists. Axis creation is unaffected: axA was
    # allocated with the gridspec, only the drawing moved.
    # ---- head ISM by confusion outcome ---------------------------------------------------
    # CONSISTENCY GATE. load_assay_confusion derives its OWN saliency npz from exp_root as
    # <exp_root>/<assay>_<donor>/ism/ism_head_<arm>_{AS,nonAS}.npz -- it does not look at
    # --as_npz/--nonas_npz. Pointing those at a rebuild while exp_root still names the original
    # tree would draw the logos from one head and the confusion panel from another, with no
    # error and a figure that looks fine. Refuse instead: a silently mixed figure is worse than
    # a missing panel, and the message names the two paths so the fix is obvious.
    if a.exp_root:
        _cell = os.path.join(a.exp_root, f"{a.assay.lower()}_{a.donor}")
        _want = {cls: os.path.realpath(
                     os.path.join(_cell, "ism", f"ism_head_{a.arm}_{cls}.npz"))
                 for cls in ("AS", "nonAS")}
        _have = {"AS": os.path.realpath(a.as_npz), "nonAS": os.path.realpath(a.nonas_npz)}
        _bad = [c for c in ("AS", "nonAS")
                if os.path.exists(_want[c]) and _want[c] != _have[c]]
        if _bad:
            raise SystemExit(
                "the confusion panel and the logos would use DIFFERENT saliency files:\n"
                + "".join(f"  {c}:\n    logos      {_have[c]}\n    confusion  {_want[c]}\n"
                          for c in _bad)
                + "The confusion panel takes its npz from --exp_root, not from --as_npz. Point\n"
                  "--exp_root at the tree the npz files live in, or drop --exp_root to render\n"
                  "without that panel.")
    if a.exp_root:
        thr = a.threshold
        if thr is not None and thr != "auto":
            thr = float(thr)
        payload, why = load_assay_confusion(
            a.exp_root, a.assay, a.donor, a.arm, scores_dir=a.scores_dir,
            score_col=a.score_col, threshold=thr, donor_kind=a.donor_kind,
            universe_mode=a.universe, keep_leaky=a.keep_leaky)
        if payload is None:
            placeholder(axA, "head ISM by outcome", why, fs)
        else:
            sal, masks, counts = payload
            draw_confusion_panel(axA, sal, masks, counts, window=a.window, smooth=a.conf_smooth,
                                 title=f"head ISM saliency by outcome ({who})",
                                 fontsize=fs, legend=True,
                                 xlabel="position relative to variant (bp)",
                                 ylabel="mean saliency")
            # WHERE THE MOTIF SITS, in this panel's own (variant-relative) coordinates. Drawn as
            # a median line inside the interquartile band rather than a line alone: under motif
            # alignment the motif is at a DIFFERENT offset in every window, and a lone dashed
            # line would assert a fixed position the data does not have.
            if _logo_span is not None:
                # WHAT REGION THE LOGO PANELS BELOW ARE SHOWING. Dotted edges rather than a solid
                # band so it is not confused with the interquartile band of the motif centre,
                # which answers a different question.
                lo, hi = _logo_span
                axA.axvspan(lo, hi, color="#2c7fb8", alpha=0.08, zorder=0)
                for xx in (lo, hi):
                    axA.axvline(xx, color="#2c7fb8", ls=":", lw=1.1, zorder=3)
                # TOP-LEFT CORNER, NOT OVER THE BAND. The band sits near the variant, which is
                # exactly where every saliency curve peaks, so an annotation centred on it lands
                # on the data. The dotted edges already say where it is; this only has to name
                # it. The per-window spread moved to --csv_out and the log.
                axA.annotate(f"logo window: {lo:+.0f} to {hi:+.0f} bp",
                             xy=(0.015, 0.97), xycoords="axes fraction",
                             ha="left", va="top", color="#1a5a85",
                             fontsize=(8 if fs is None else fs - 1))
            if _motif_mc is not None and len(_motif_mc):
                md = float(np.median(_motif_mc))
                q1, q3 = (float(np.percentile(_motif_mc, 25)),
                          float(np.percentile(_motif_mc, 75)))
                if (q3 - q1) <= 2 * _motif_W:
                    axA.axvspan(q1, q3, color="0.55", alpha=0.12, zorder=0)
                    axA.axvline(md, color="0.35", ls="--", lw=1.1, zorder=4)
                    axA.annotate(f"{a.matrix_id} centre  median {md:+.0f} bp  "
                                 f"IQR {q1:+.0f}..{q3:+.0f}",
                                 xy=(md, 0.90), xycoords=("data", "axes fraction"),
                                 xytext=(4, 0), textcoords="offset points",
                                 ha="left", va="top", color="0.35",
                                 fontsize=(8 if fs is None else fs - 1.5), bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))
                else:
                    print(f"  [align] motif centre NOT marked on the saliency panel: IQR "
                          f"{q3 - q1:.0f} bp over a {_motif_W} bp motif.")
            for k in ("TP", "TN", "FP", "FN"):
                summary[f"confusion_n_{k}"] = int(counts[k])
    else:
        placeholder(axA, "head ISM by outcome", "no --exp_root", fs)


    fig.suptitle(a.title or f"HapAdapter head ISM: saliency, cofactor context and "
                            f"attribution ({who})", fontsize=fs + 3, y=0.995)
    fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight")
    print(f"[write] {a.out}")

    if a.csv_out:
        import pandas as pd
        pd.DataFrame([dict(assay=a.assay, individual=a.donor, arm=a.arm, **summary)]) \
            .to_csv(a.csv_out, index=False)
        print(f"[write] {a.csv_out}")
    for k, v in summary.items():
        print(f"    {k:34}{v}")


if __name__ == "__main__":
    sys.exit(main())
