#!/usr/bin/env python3
"""
hapadapter.figures.fig_ism_full_composite -- the four-section head ISM figure as ONE file.

    +------------------------------------------------------------------+
    | A  head ISM saliency by TP / TN / FP / FN                         |
    +------------------------------------------------------------------+
    | B  cofactor motif density, one facet per TF, AS vs non-AS         |
    +------------------------------------------------------------------+
    | C  AS contribution logo, over the JASPAR reference PWM            |
    +------------------------------------------------------------------+
    | D  one AS locus: ISM saliency and its motif-hit lanes             |
    +------------------------------------------------------------------+

A, B AND C COME FROM ONE RENDER, D FROM ANOTHER. fig_ism_head already builds A/B/C (and
the reference strip) in a single GridSpec, so splitting them apart to retile them would
throw away a layout that is already correct. fig_ism_motif_example builds D entirely
inside its own main() with no drawable entry point, so it cannot be asked to draw into a
supplied axes without a rewrite. Rendering the two and stacking them keeps one
implementation of each panel.

THE LETTERS ARE STAMPED BY THE SOURCE SCRIPTS, NOT HERE. A tiler can put one letter on a
tile; the top tile carries three sections, whose positions only the script that laid them
out knows. Each script therefore takes its own --panel_letters / --panel_letter, and this
builder passes A,B,C to one and D to the other.

Usage:
  python -m hapadapter.figures.fig_ism_full_composite \
      --assay CTCF --donor ENC-002 --exp_root $EXPROOT \
      --as_npz   $ISM/ism_head_personal_AS.npz \
      --nonas_npz $ISM/ism_head_personal_nonAS.npz \
      --locus_npz $ISM/ism_head_personal_AS.npz \
      --jaspar jaspar.json --pwm_json jaspar.json --matrix_id MA0139.2 \
      --tfs "SP1,CTCF,EGR1,KLF4" \
      --out fig_ism_full.png
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

HEAD = "hapadapter.figures.fig_ism_head"
LOCUS = "hapadapter.figures.fig_ism_motif_example"

# Flags the two panel scripts must accept for this builder to drive them.
# Checked against each script's own --help before anything runs, because a wrong flag name
# otherwise surfaces as an argparse usage dump after the first panel's data has been read.
# Both lists were taken from the scripts, not from memory: the locus panel's npz flag is
# --npz (not --ism) and its lane motifs are --motifs (not --tfs).
REQUIRED = {HEAD: ("--exact_size", "--panel_letters", "--size", "--logos",
                   "--show_reference", "--flip_logo", "--density_csv", "--fimo_threshold",
                   "--no_suptitle", "--hspace", "--density_legend"),
            LOCUS: ("--exact_size", "--panel_letter", "--size", "--npz", "--motifs",
                    "--count_motifs", "--exact_motifs", "--fixed_lanes", "--fimo_threshold",
                    "--lane_color")}

TOL_PX = 2


def preflight():
    """Fail with one actionable line if either panel script is too old."""
    for mod, flags in REQUIRED.items():
        r = subprocess.run([sys.executable, "-m", mod, "--help"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"could not run {mod} --help:\n{(r.stderr or '')[-700:]}")
        missing = [f for f in flags if f not in r.stdout]
        if missing:
            raise SystemExit(
                f"the installed {mod} does not accept {missing}.\n"
                f"  This builder drives it as a subprocess, so the two must be updated\n"
                f"  together. Copy the current {mod.rsplit('.', 1)[-1]}.py into\n"
                f"  <your package>/hapadapter/figures/ and re-run.")


def render(mod, args_list, out_png, size_in, dpi, label):
    """Run one panel script at an exact size; fail loudly if it produced nothing."""
    cmd = [sys.executable, "-m", mod] + args_list + [
        "--size", f"{size_in[0]:.3f}", f"{size_in[1]:.3f}",
        "--dpi", str(dpi), "--exact_size", "--out", out_png]
    print(f"\n[{label}] " + " ".join(cmd[2:]))
    r = subprocess.run(cmd, capture_output=True, text=True)
    for line in (r.stdout or "").splitlines():
        if line.strip():
            print(f"    {line}")
    if r.returncode != 0 or not os.path.exists(out_png):
        sys.stderr.write((r.stderr or "")[-1500:] + "\n")
        raise SystemExit(f"[{label}] {mod} failed; see above.")
    return out_png


def _trim_rows(im, pad_px, bg=250):
    """
    Crop an image's blank TOP AND BOTTOM to its content plus pad_px. Vertical only.

    Vertical only, deliberately. The dead space between two stacked tiles is each tile's
    own subplot margin -- matplotlib reserves a bottom margin for an x-label and a top
    margin for a title whether or not they need all of it, and under --exact_size nothing
    trims it. Cropping the sides as well would move each tile's axes horizontally by a
    different amount, since the tiles differ in label width, and the panels would no longer
    line up -- which is the one property the fixed left margin exists to guarantee.
    """
    import numpy as np
    a = np.asarray(im.convert("RGB"))
    nonblank = np.where((a < bg).any(axis=(1, 2)))[0]
    if len(nonblank) == 0:
        return im
    top = max(int(nonblank[0]) - pad_px, 0)
    bot = min(int(nonblank[-1]) + 1 + pad_px, a.shape[0])
    return im.crop((0, top, im.size[0], bot))


def stack(paths, out_path, dpi, gutter_mm, width_px, trim_pad_mm=None):
    """
    Stack the tiles vertically at a common width.

    No rescaling: each tile was rendered at the slot width, and resizing a raster panel
    scales its text with it, so two tiles fitted to one width from different render sizes
    would end up carrying different effective font sizes. Heights are whatever the tiles
    are after trimming, so the gap between sections is exactly `gutter_mm`.
    """
    from PIL import Image
    g = int(gutter_mm * dpi / 25.4)
    pad = None if trim_pad_mm is None else int(trim_pad_mm * dpi / 25.4)
    ims = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        h0 = im.size[1]
        if pad is not None:
            im = _trim_rows(im, pad)
            if im.size[1] != h0:
                print(f"[trim] {os.path.basename(p)}: {h0} -> {im.size[1]} px "
                      f"({(h0 - im.size[1]) / dpi * 25.4:.0f} mm of blank margin removed)")
        if im.size[0] != width_px:
            print(f"[stack] {os.path.basename(p)} is {im.size[0]} px wide, expected "
                  f"{width_px} -- resizing, which rescales its text relative to the other "
                  f"tile.")
            im = im.resize((width_px, im.size[1]))
        ims.append(im)
    H = sum(i.size[1] for i in ims) + g * (len(ims) - 1)
    canvas = Image.new("RGB", (width_px, H), "white")
    y = 0
    for im in ims:
        canvas.paste(im, (0, y))
        y += im.size[1] + g
    canvas.save(out_path)
    return out_path, (width_px, H)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--assay", required=True)
    p.add_argument("--donor", required=True)
    p.add_argument("--exp_root", required=True)
    p.add_argument("--arm", default="personal")
    p.add_argument("--as_npz", required=True)
    p.add_argument("--nonas_npz", required=True)
    p.add_argument("--locus_npz", default=None,
                   help="npz the single-locus panel D is drawn from; defaults to --as_npz")
    p.add_argument("--jaspar", required=True)
    p.add_argument("--pwm_json", default=None, help="defaults to --jaspar")
    p.add_argument("--matrix_id", required=True)
    p.add_argument("--tfs", default="CTCF,SP1,KLF4,EGR1",
                   help="panel B facets AND panel D lanes, in this order. CTCF first "
                        "because it is the assay's own motif; the cofactors follow.")
    p.add_argument("--lane_tfs", default=None,
                   help="motifs for panel D's lanes; defaults to --tfs")
    p.add_argument("--density_csv", nargs="+", default=None, metavar="CSV",
                   help="windows CSV(s) the panel-B densities are computed over, instead of "
                        "the ISM npz. The density panels run FIMO over sequences and never "
                        "touch the model, so they are not bound by the held-out split the "
                        "saliency and logo panels need -- a TRAINING build gives them far "
                        "more sequence. Panel B is then on different loci from A/C/D, which "
                        "belongs in the caption.")
    p.add_argument("--density_label_col", default=None, metavar="COL",
                   help="label column in --density_csv. Left unset, fig_ism_head tries "
                        "as_label then label, which covers both producers.")
    p.add_argument("--density_legend_mode", default="above",
                   choices=["above", "inside", "none"],
                   help="panel B's AS/non-AS key placement")
    p.add_argument("--density_seq_col", default=None, metavar="COL",
                   help="sequence column in --density_csv (default sequence1 = haplotype 1)")
    p.add_argument("--fimo_threshold", type=float, default=1e-3,
                   help="motif-hit p-value for panels B and D. 1e-4 is a genome-scan "
                        "threshold: a ~10 bp GC-rich matrix (SP1, EGR1, KLF4) rarely clears "
                        "it inside one 257 bp window while the 19 bp CTCF matrix does, so at "
                        "1e-4 an empty cofactor lane is a threshold effect as much as a "
                        "biological one.")
    p.add_argument("--no_flip_logo", action="store_true",
                   help="leave the contribution logo pointing down. An ASB head's "
                        "contributions are negative almost everywhere, so unflipped the "
                        "letters hang below the axis and logomaker draws them inverted.")
    p.add_argument("--head_extra", default="", help="extra args passed through to fig_ism_head")
    p.add_argument("--locus_extra", default="", help="extra args passed through to the locus panel")
    p.add_argument("--width_mm", type=float, default=183.0)
    p.add_argument("--head_mm", type=float, default=150.0, help="height of the A/B/C tile")
    p.add_argument("--locus_mm", type=float, default=95.0, help="height of the D tile")
    p.add_argument("--lane_color", default="track",
                   choices=["track", "pvalue", "cofactor"],
                   help="panel D lane colour. 'track' draws every lane in the saliency "
                        "curve's colour with no colorbar -- right when the roster is an "
                        "explicit four, where colour would encode nothing. 'pvalue' is the "
                        "viridis-by-significance scale with a colorbar.")
    p.add_argument("--hspace", type=float, default=1.25,
                   help="vertical gap between the A/B/C rows, in units of row height. This "
                        "is the A-B and B-C spacing; the C-D gap is --gutter_mm.")
    p.add_argument("--trim_pad_mm", type=float, default=1.5,
                   help="white margin left above and below each tile's content after "
                        "trimming. 0 crops flush; a negative value disables trimming, "
                        "leaving each tile's own subplot margins in place.")
    p.add_argument("--gutter_mm", type=float, default=4.0)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--fontsize", type=float, default=8)
    p.add_argument("--letter_pt", type=float, default=11.0)
    p.add_argument("--keep_panels", default=None, metavar="DIR")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    preflight()

    mm = a.dpi / 25.4
    W = int(a.width_mm * mm)
    h_head, h_locus = int(a.head_mm * mm), int(a.locus_mm * mm)
    pwm = a.pwm_json or a.jaspar
    outdir = a.keep_panels or tempfile.mkdtemp(prefix="ism_panels_")
    os.makedirs(outdir, exist_ok=True)

    # A/B/C + the reference strip. --logos as, because the second (non-AS) logo is a
    # separate claim and this figure's C is the AS logo over the reference.
    head_args = ["--assay", a.assay, "--donor", a.donor, "--exp_root", a.exp_root,
                 "--arm", a.arm, "--as_npz", a.as_npz, "--nonas_npz", a.nonas_npz,
                 "--jaspar", a.jaspar, "--pwm_json", pwm, "--matrix_id", a.matrix_id,
                 "--tfs", a.tfs, "--logos", "AS", "--show_reference",
                 "--fontsize", str(a.fontsize), "--letter_pt", str(a.letter_pt),
                 "--fimo_threshold", str(a.fimo_threshold),
                 "--hspace", str(a.hspace), "--no_suptitle",
                 "--density_legend", a.density_legend_mode,
                 "--panel_letters", "A,B,C"] + \
                ([] if a.no_flip_logo else ["--flip_logo"]) + \
                (["--density_csv"] + list(a.density_csv) if a.density_csv else []) + \
                (["--density_label_col", a.density_label_col] if a.density_label_col else []) + \
                (["--density_seq_col", a.density_seq_col] if a.density_seq_col else []) + \
                (a.head_extra.split() if a.head_extra else [])

    # EXACTLY THE NAMED MOTIFS, AS LANES THAT EXIST WHETHER OR NOT THEY HIT.
    # --exact_motifs: without it "CTCF" also matches CTCFL and "SP1" matches SP110, so the
    #   lane set is whatever the substring happened to pull in.
    # --fixed_lanes: without it only motifs WITH a hit get a row, so an absent cofactor is
    #   indistinguishable from one that was never scanned. It keys off --count_motifs
    #   rather than --motifs, so both carry the same list.
    lanes = a.lane_tfs or a.tfs
    locus_args = ["--npz", a.locus_npz or a.as_npz, "--jaspar", a.jaspar,
                  "--motifs", lanes, "--count_motifs", lanes,
                  "--exact_motifs", "--fixed_lanes", "--lane_color", a.lane_color,
                  "--fimo_threshold", str(a.fimo_threshold),
                  "--panel_letter", "D", "--letter_pt", str(a.letter_pt)] + \
                 (a.locus_extra.split() if a.locus_extra else [])

    top = render(HEAD, head_args, os.path.join(outdir, "tile_abc.png"),
                 (W / a.dpi, h_head / a.dpi), a.dpi, "A/B/C")
    bot = render(LOCUS, locus_args, os.path.join(outdir, "tile_d.png"),
                 (W / a.dpi, h_locus / a.dpi), a.dpi, "D")

    out_path, (w, h) = stack([top, bot], a.out, a.dpi, a.gutter_mm, W,
                             trim_pad_mm=(None if a.trim_pad_mm < 0 else a.trim_pad_mm))
    print(f"\n[write] {out_path}  ({w} x {h} px, {a.width_mm:.0f} mm wide at {a.dpi} dpi)")
    if a.keep_panels:
        print(f"[write] tiles in {outdir}")


if __name__ == "__main__":
    main()
