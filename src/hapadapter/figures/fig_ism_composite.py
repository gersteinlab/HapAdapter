#!/usr/bin/env python3
"""
hapadapter.figures.fig_ism_composite -- the five-panel ISM figure for one (assay, individual):

    +---------------------------+---------------------------+
    |  A  trunk ISM saliency    |  B  head ISM saliency     |
    |     peak vs background    |     by TP / TN / FP / FN  |
    +---------------------------+---------------------------+
    |  C  motif density in      |  D  motif density in      |
    |     TRUNK windows         |     HEAD windows          |
    |     peak vs background    |     AS vs non-AS          |
    +---------------------------+---------------------------+
    |  E  trunk ISM contribution logo, over the JASPAR reference PWM   |
    +-----------------------------------------------------------------+

EACH COLUMN USES THE CONTRAST ITS OWN MODEL WAS TRAINED ON. The trunk is the Stage-1 BINDING
regressor and never saw an allele-specific label, so its panels contrast peak vs background;
the head is the ASB classifier, so its panels contrast AS vs non-AS (and, in A/B, the four
confusion outcomes). Colouring the trunk panels by AS/non-AS would ask a model to separate a
distinction it was never trained on -- measured at AUROC ~0.46-0.55, i.e. chance -- and would
put a guaranteed non-result beside four informative panels.

THE LOGO IS TRUNK-ONLY, AND THAT IS A RESULT RATHER THAN AN OMISSION. Binding is motif-driven,
so the trunk's averaged attribution recovers a motif. ASB largely is not: 87.4% of AS windows
carry no CTCF motif within +/-10bp, so an averaged head logo would be dominated by windows with
nothing to recover and would come out near-flat. The honest head equivalent is a PER-LOCUS logo
for the minority that do have a motif -- a different panel making a different claim.

NO PANEL MATH LIVES HERE. Every panel is drawn by the same function its standalone script uses:

    A   plot_saliency_tracks.load_tracks          -> draw_saliency_tracks
    B   fig_ism_confusion_grid.load_assay_confusion -> draw_confusion_panel
    C,D plot_as_motif_enrichment.compute_density  -> draw_motif_enrichment
    E   plot_ism_logo.align_contributions         -> draw_ism_logo

so this figure and the standalone ones cannot drift apart. A panel whose inputs are absent is
drawn as a labelled placeholder and named on stdout, rather than silently omitted or faked --
an incomplete composite should be obviously incomplete.

Usage
-----
    python -m hapadapter.figures.fig_ism_composite \\
      --assay CTCF --donor ENC-002 \\
      --trunk_peak_npz  $C/ism/ism_trunk_peak.npz \\
      --trunk_bg_npz    $C/ism/ism_trunk_background.npz \\
      --exp_root        $EXP --arm personal \\
      --trunk_peak_seqs $C/ism/seqs_trunk_peak.npy \\
      --trunk_bg_seqs   $C/ism/seqs_trunk_background.npy \\
      --head_as_seqs    $C/ism/seqs_head_AS.npy \\
      --head_nonas_seqs $C/ism/seqs_head_nonAS.npy \\
      --logo_ism        $C/ism/ism_trunk_peak.npz \\
      --jaspar jaspar.json --pwm_json jaspar.json --matrix_id MA0139.2 --tfs CTCF,SP1 \\
      --out fig_ism_composite_ctcf.png --csv_out fig_ism_composite_ctcf.csv
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

from hapadapter.figures.plot_saliency_tracks import (load_tracks, draw_saliency_tracks,
                                                     load_motifs)
from hapadapter.figures.plot_as_motif_enrichment import compute_density, draw_motif_enrichment
from hapadapter.figures.plot_ism_logo import (align_contributions, draw_ism_logo, pfm_to_prob,
                                              prob_to_ic)
from hapadapter.figures.fig_ism_confusion_grid import (load_assay_confusion,
                                                       draw_confusion_panel)

PANEL_LETTERS = ["A", "B", "C", "D", "E"]


def pretty_individual(raw):
    """"ENC-002" -> "individual 2", matching fig_contrast_composite and the arrow figures."""
    if not raw:
        return ""
    m = re.fullmatch(r"ENC-0*(\d+)", str(raw).strip(), flags=re.I)
    return f"individual {int(m.group(1))}" if m else str(raw)


def placeholder(ax, letter, name, why, fs):
    """A panel whose inputs are missing. Labelled, not blank: a reader must be able to tell
    an absent panel from an empty result, and a reviewer must be able to tell both from a
    panel that was quietly dropped."""
    ax.text(0.5, 0.5, f"{name}\n({why})", ha="center", va="center",
            transform=ax.transAxes, fontsize=fs, color="0.45")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("0.85")
    print(f"  [skip] panel {letter} ({name}): {why}")


def seq_list(path, want_len=None):
    """
    Window sequences from a .npy array OR straight out of an ISM .npz's `seqs` array.

    The npz route exists because ism_saliency already stores the exact sequences it scored,
    so panels C/D need no separate export -- and more importantly cannot drift from the windows
    panels A/B are computed on, which a separately-dumped .npy could.
    """
    if str(path).endswith(".npz"):
        with np.load(path, allow_pickle=True) as z:
            if "seqs" not in z:
                raise SystemExit(f"{path} has no 'seqs' array; keys are {sorted(z.files)}")
            seqs = [str(s) for s in z["seqs"]]
    else:
        seqs = [str(s) for s in np.load(path, allow_pickle=True)]
    if not seqs:
        return []
    L = want_len or len(seqs[0])
    return [s for s in seqs if len(s) == L]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assay", required=True)
    ap.add_argument("--donor", default="", help="ENC id; titles render it as 'individual N'")

    ap.add_argument("--trunk_npz", "--trunk_peak_npz", dest="trunk_npz", default=None,
                    help="panel A: the trunk ISM npz. ism_trunk.npz is already the PEAK set -- "
                         "ism_saliency defaults to --feature_keep peak and ranks by "
                         "binding_label_raw.")
    ap.add_argument("--trunk_bg_npz", default=None,
                    help="panel A: optional background counterpart. Omit it and panel A is a "
                         "SINGLE track with an IQR band, which is what the data supports "
                         "today; supply it and the panel becomes peak vs background.")

    ap.add_argument("--exp_root", default=None, help="panel B")
    ap.add_argument("--arm", default="personal", help="panel B")
    ap.add_argument("--scores_dir", default=None, help="panel B")
    ap.add_argument("--score_col", default="delta", help="panel B ranking column")
    # Mirror the standalone's default exactly. None is not accepted downstream -- 0 is the
    # model's own p=0.5 boundary, 'auto' selects Youden.
    ap.add_argument("--threshold", default="0", help="panel B; 0 = p=0.5 boundary, 'auto' = Youden")
    ap.add_argument("--donor_kind", default="all", help="panel B")
    ap.add_argument("--universe", default="ref", choices=["ref", "none"], help="panel B")
    ap.add_argument("--keep_leaky", action="store_true", help="panel B")

    # Each of these takes a .npy of sequences OR an ISM .npz (its `seqs` array is used), so
    # the head panels can point straight at ism_head_<arm>_{AS,nonAS}.npz with no export step.
    # The second member of each pair is optional: without it the panel draws one curve, the
    # same way panel A degrades, rather than being dropped.
    ap.add_argument("--trunk_peak_seqs", default=None, help="panel C; .npy or ISM .npz")
    ap.add_argument("--trunk_bg_seqs", default=None, help="panel C; optional second group")
    ap.add_argument("--head_as_seqs", default=None, help="panel D; .npy or ISM .npz")
    ap.add_argument("--head_nonas_seqs", default=None, help="panel D; optional second group")

    ap.add_argument("--logo_ism", default=None, help="panel E; the TRUNK ism npz")
    ap.add_argument("--pwm_json", default=None, help="panel E")
    ap.add_argument("--matrix_id", default="MA0139.2", help="panel E")
    ap.add_argument("--logo_flank", type=int, default=5, help="panel E")
    ap.add_argument("--strand_aware", action="store_true", help="panel E")

    ap.add_argument("--jaspar", default=None, help="motif JSON for panels A, C, D")
    ap.add_argument("--tfs", default="CTCF", help="comma-separated, for panels A, C, D")
    ap.add_argument("--fimo_threshold", type=float, default=1e-4)
    ap.add_argument("--smooth", type=int, default=11, help="panels A, C, D")
    ap.add_argument("--window", type=int, default=100, help="panel B half-width (bp)")
    ap.add_argument("--conf_smooth", type=int, default=3, help="panel B smoothing")

    ap.add_argument("--size", type=float, nargs=2, default=[9.5, 9.0])
    ap.add_argument("--fontsize", type=float, default=8)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv_out", default=None)
    a = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fs = a.fontsize
    who = ", ".join(x for x in (a.assay, pretty_individual(a.donor)) if x)
    tfs = [t.strip() for t in a.tfs.split(",") if t.strip()]

    # ONE loader for every panel. plot_as_motif_enrichment now uses the same grouped
    # {TF: {matrix_id: pfm}} representation as plot_saliency_tracks, so panels A, C and D
    # union all of a TF's JASPAR matrices into one curve and mean the same thing by
    # "CTCF density". Before that they did not: CTCF has three matrices in the current JASPAR
    # file and the flat loader silently kept one.
    motifs = {}
    if a.jaspar and tfs:
        motifs = load_motifs(a.jaspar, tfs)
        missing = [t for t in tfs if t.upper() not in motifs]
        if missing:
            print(f"  [warn] motifs not found in {a.jaspar}: {missing}")
        for tf, mats in motifs.items():
            print(f"  [motifs] {tf}: {len(mats)} JASPAR matrices {sorted(mats)} (unioned)")

    # Row heights: the two analysis rows carry the comparison and get equal space; the logo row
    # is shorter because a sequence logo is read horizontally and gains nothing from height.
    fig = plt.figure(figsize=tuple(a.size))
    # wspace is wide because panel A carries a SECOND y-axis on its right edge (motif
    # density); at the default it collided with panel B's y-label.
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.85], hspace=0.42, wspace=0.34)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])
    # Panel E is TWO stacked axes: the averaged contribution over the reference PWM. The claim
    # of that panel is a comparison ("the trunk recovers this motif"), and a comparison needs
    # both halves present -- otherwise the reader takes the recovery on faith.
    gsE = gs[2, :].subgridspec(2, 1, height_ratios=[2, 1], hspace=0.30)
    axE1 = fig.add_subplot(gsE[0])
    axE2 = fig.add_subplot(gsE[1])
    # E1 and E2 are the SAME positional axis, so E1's tick labels are redundant -- and they sat
    # exactly where E2's title goes. Hiding them removes the collision and makes the pair read
    # as one stacked comparison rather than two charts.
    axE1.tick_params(labelbottom=False)

    summary = {}

    # ---- A: trunk ISM, peak vs background ------------------------------------------------
    if a.trunk_npz and os.path.exists(a.trunk_npz):
        pairs = [("peak", a.trunk_npz)]
        if a.trunk_bg_npz and os.path.exists(a.trunk_bg_npz):
            pairs.append(("background", a.trunk_bg_npz))
        else:
            print("  [note] panel A: no background track -- drawing the peak track alone with "
                  "an IQR band. Add --trunk_bg_npz once the background ISM run exists.")
        tracks = load_tracks(pairs)
        # shared_windows stays FALSE: peak and background are DIFFERENT locus sets, so each
        # track needs its own motif density. Sharing would draw the peak windows' density
        # under both curves -- which looks right and is wrong.
        profs, _ = draw_saliency_tracks(
            axA, tracks, motifs, shared_windows=False, fimo_threshold=a.fimo_threshold,
            smooth_w=a.smooth,
            # The band is only drawn for a SINGLE track (two overlapping bands are unreadable),
            # so it appears exactly when it is the panel's only indication of spread.
            band="iqr", normalize=False,
            title=f"A  trunk ISM saliency ({who})", fontsize=fs, legend=True)
        for lab, pr in profs.items():
            summary[f"A_{lab}_peak_saliency"] = round(float(pr.max()), 6)
            summary[f"A_{lab}_argmax_bp"] = int(np.argmax(pr)) - len(pr) // 2
        for lab, imp, _oh in tracks:
            summary[f"A_n_{lab}"] = int(imp.shape[0])
    else:
        placeholder(axA, "A", "trunk ISM", "no --trunk_npz", fs)

    # ---- B: head ISM by confusion outcome ------------------------------------------------
    if a.exp_root:
        thr = a.threshold
        if thr is not None and thr != "auto":
            thr = float(thr)
        payload, why = load_assay_confusion(
            a.exp_root, a.assay, a.donor, a.arm, scores_dir=a.scores_dir,
            score_col=a.score_col, threshold=thr, donor_kind=a.donor_kind,
            universe_mode=a.universe, keep_leaky=a.keep_leaky)
        if payload is None:
            placeholder(axB, "B", "head ISM", why, fs)
        else:
            sal, masks, counts = payload
            draw_confusion_panel(axB, sal, masks, counts, window=a.window,
                                 smooth=a.conf_smooth,
                                 title=f"B  head ISM saliency by outcome ({who})",
                                 fontsize=fs, legend=True,
                                 xlabel="position relative to variant (bp)",
                                 ylabel="mean saliency")
            for k in ("TP", "TN", "FP", "FN"):
                summary[f"B_n_{k}"] = int(counts[k])
    else:
        placeholder(axB, "B", "head ISM", "no --exp_root", fs)

    # ---- C, D: motif density in each model's own windows ---------------------------------
    from memelite import fimo
    for ax, letter, name, pa, pb, la, lb in (
            (axC, "C", "motif density, trunk windows", a.trunk_peak_seqs, a.trunk_bg_seqs,
             "peak", "background"),
            (axD, "D", "motif density, head windows", a.head_as_seqs, a.head_nonas_seqs,
             "AS", "non-AS")):
        if not (motifs and pa and os.path.exists(pa)):
            placeholder(ax, letter, name,
                        "no sequences" if motifs else "no --jaspar/--tfs", fs)
            continue
        A = seq_list(pa)
        if not A:
            placeholder(ax, letter, name, "empty after length filter", fs)
            continue
        groups = [(la, compute_density(A, motifs, fimo, a.fimo_threshold))]
        summary[f"{letter}_n_{la}"] = len(A)
        if pb and os.path.exists(pb):
            B = seq_list(pb, want_len=len(A[0]))
            if B:
                groups.append((lb, compute_density(B, motifs, fimo, a.fimo_threshold)))
                summary[f"{letter}_n_{lb}"] = len(B)
        else:
            print(f"  [note] panel {letter}: no {lb!r} group -- drawing {la!r} alone.")
        order = list(motifs)
        flank = len(A[0]) // 2
        draw_motif_enrichment(ax, groups, order, flank, smooth_w=a.smooth,
                              title=f"{letter}  {name} ({who})", fontsize=fs, legend=True,
                              xlabel="position relative to variant (bp)")
        for glab, dens in groups:
            for nm in order:
                summary[f"{letter}_{nm}_center_{glab}"] = round(float(dens[nm][flank]), 5)

    # ---- E: trunk contribution logo over the reference PWM --------------------------------
    if a.logo_ism and a.pwm_json and os.path.exists(a.logo_ism) and os.path.exists(a.pwm_json):
        pj = json.load(open(a.pwm_json))
        if a.matrix_id not in pj:
            placeholder(axE1, "E", "contribution logo",
                        f"{a.matrix_id} not in {os.path.basename(a.pwm_json)}", fs)
            axE2.axis("off")
        else:
            entry = pj[a.matrix_id]
            P = pfm_to_prob(entry.get("pfm") or entry.get("counts"))
            W = P.shape[0]
            ref_ic = prob_to_ic(P)
            d = np.load(a.logo_ism, allow_pickle=True)
            agg, used, n_flip = align_contributions(
                d["onehot"], d["contrib"], d["importance"], ref_ic.sum(axis=1), W,
                a.logo_flank, control="none", strand_aware=a.strand_aware)
            tf = entry.get("name", a.matrix_id)
            draw_ism_logo(axE1, agg, a.logo_flank, W, ax_ref=axE2, ref_ic=ref_ic,
                          title=f"E  trunk ISM contribution logo "
                                f"(mean of {used}/{d['onehot'].shape[0]} {tf} windows)",
                          ref_title=f"JASPAR {a.matrix_id} ({tf}) reference, bits",
                          fontsize=fs)
            # The statistic the panel is really making a claim about, so it lands in the CSV
            # rather than only in a reader's impression of the picture.
            motif_amp = float(np.abs(agg[a.logo_flank:a.logo_flank + W]).sum(axis=1).mean())
            flank_amp = (float(np.abs(agg[:a.logo_flank]).sum(axis=1).mean())
                         if a.logo_flank else float("nan"))
            summary["E_windows_used"] = int(used)
            summary["E_windows_total"] = int(d["onehot"].shape[0])
            summary["E_motif_over_flank"] = (round(motif_amp / flank_amp, 4)
                                             if flank_amp and flank_amp > 0 else float("nan"))
    else:
        placeholder(axE1, "E", "contribution logo", "no --logo_ism/--pwm_json", fs)
        axE2.axis("off")

    fig.suptitle(a.title or f"ISM and motif context: trunk vs HapAdapter head ({who})",
                 fontsize=fs + 3, y=0.995)
    fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight")
    print(f"[write] {a.out}")

    if a.csv_out:
        import pandas as pd
        row = dict(assay=a.assay, individual=a.donor, arm=a.arm, **summary)
        pd.DataFrame([row]).to_csv(a.csv_out, index=False)
        print(f"[write] {a.csv_out}")
    for k, v in summary.items():
        print(f"    {k:34}{v}")


if __name__ == "__main__":
    sys.exit(main())
