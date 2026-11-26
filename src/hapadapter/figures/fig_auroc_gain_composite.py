#!/usr/bin/env python3
"""
hapadapter.figures.fig_auroc_gain_composite -- the three AUROC-change panels as ONE figure.

    +--------------------------------------------------------------+
    | a  Change in AUROC, reference to personal (DNABERT-2)         |
    +------------------------------+-------------------------------+
    | b  within -> cross individual| c  within -> cross individual |
    |    (DNABERT-2)               |    (AlphaGenome)              |
    +------------------------------+-------------------------------+

WHY A COMPOSER RATHER THAN A THREE-PANEL PLOTTER. fig_auroc_gain_arrows builds its own
figure and places its legend relative to that figure, and the three panels differ in arm
pair, backbone, donor set and assay list. Rendering each through the existing, reviewed
plotter and tiling the results keeps one implementation of the arrow drawing instead of a
second copy that can drift from it.

EACH PANEL IS RENDERED AT ITS FINAL SIZE, NOT RESIZED AFTERWARDS. The tiler would happily
scale a panel into its slot, but scaling a raster panel scales its text with it, so two
panels rendered at different sizes and then fitted to the same slot end up with different
effective font sizes. Panel geometry is therefore computed first and passed to each render
as --size/--dpi, so every panel is drawn once, at the size it will occupy.

EP300 IS EXCLUDED FROM PANEL c ONLY. AlphaGenome's EP300 cell is underpowered, so its
cross-individual arrow would be read as a result rather than as an artefact of n. It
remains in a and b, where it is supported, and the exclusion is printed at run time so it
cannot become a silent difference between panels.

Usage:
  python -m hapadapter.figures.fig_auroc_gain_composite \
      --perlocus_root $EXPROOT --crossdonor_root $EXPROOT \
      --ag_crossdonor_root $ROOT/experiments \
      --out fig_auroc_gain_composite.png
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

PLOTTER = "hapadapter.figures.fig_auroc_gain_arrows"

ASSAYS = ["CTCF", "EP300", "POLR2A", "POLR2AphosphoS5", "ATAC", "H3K4me3",
          "H3K27ac", "H3K4me1", "H3K9me3", "H3K36me3", "H3K27me3"]
DONORS = ["ENC-001", "ENC-002", "ENC-003", "ENC-004"]

# The assay string is the join key into the experiment tree; only the tick text changes.
ASSAY_LABELS = ["ATAC=ATAC-seq"]

TITLES = {"a": "Change in AUROC, reference to personal (DNABERT-2)",
          "b": "Change in AUROC, within to cross individual (DNABERT-2)",
          "c": "Change in AUROC, within to cross individual (AlphaGenome)"}

# Underpowered in the AlphaGenome arm; see the module docstring.
DROP_FROM_C = {"EP300"}


def grid_geom(ol, dpi, gutter_mm):
    """Pixel geometry of the 12-column grid: column width, row heights, row origins."""
    mm = dpi / 25.4
    W = int(ol["width_mm"] * mm)
    g = int(gutter_mm * mm)
    colw = (W - g * (ol["ncol"] - 1)) // ol["ncol"]
    rowh = [int(h * mm) for h in ol["row_heights_mm"]]
    row_y = [sum(rowh[:i]) + g * i for i in range(len(rowh))]
    return W, colw, rowh, row_y, g


def panel_box(ol, letter, dpi, gutter_mm):
    """(x, y, w, h) in pixels for one panel's slot."""
    W, colw, rowh, row_y, g = grid_geom(ol, dpi, gutter_mm)
    q = next(x for x in ol["panels"] if x["letter"] == letter)
    cs, rs, r, c = q["colspan"], q.get("rowspan", 1), q["row"], q["col"]
    return (colw * c + g * c, row_y[r],
            colw * cs + g * (cs - 1), sum(rowh[r:r + rs]) + g * (rs - 1))


# Slots are computed in mm and panels in inches, so a slot and its panel can disagree by a
# pixel. Below this, pad; above it, something is actually wrong and gets reported.
TOL_PX = 2


def tile(ol, paths, out_path, dpi, gutter_mm, letter_case, letter_pt=9):
    """
    Paste each panel into its slot and stamp the panel letter.

    NO RESIZE HAPPENS HERE IF THE PANELS WERE RENDERED AT THEIR SLOT SIZE, which is the
    point: a resize would scale the panel's text along with everything else, so two panels
    drawn at different sizes and then fitted to equal slots would carry different effective
    font sizes. A mismatch is reported rather than silently corrected.
    """
    from PIL import Image, ImageDraw, ImageFont
    W, colw, rowh, row_y, g = grid_geom(ol, dpi, gutter_mm)
    H = row_y[-1] + rowh[-1]
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    # RESOLVE THE FONT THROUGH MATPLOTLIB, which ships DejaVu and is already a dependency.
    # A bare "DejaVuSans-Bold.ttf" only resolves if it happens to be on the system font
    # path; when it is not, PIL falls back to a ~10 px bitmap and the panel letters come
    # out as specks rather than failing, which is easy to miss on a large canvas.
    ft = None
    try:
        from matplotlib import font_manager
        ft = ImageFont.truetype(font_manager.findfont("DejaVu Sans:bold"),
                                int(letter_pt / 72 * dpi))
    except Exception:
        for cand in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Helvetica.ttc"):
            try:
                ft = ImageFont.truetype(cand, int(letter_pt / 72 * dpi)); break
            except Exception:
                continue
    if ft is None:
        print("[tile] no scalable bold font found; panel letters will be very small.")
        ft = ImageFont.load_default()
    for q in ol["panels"]:
        L = q["letter"]
        x, y, w, h = panel_box(ol, L, dpi, gutter_mm)
        im = Image.open(paths[L]).convert("RGBA")
        dw, dh = w - im.size[0], h - im.size[1]
        if abs(dw) <= TOL_PX and abs(dh) <= TOL_PX:
            # Sub-pixel rounding between mm-based slots and inch-based figsize. Centre the
            # panel in its slot rather than resizing: a one-pixel rescale is invisible but
            # it is still a rescale, and doing it per panel is how the text scales drift.
            x, y = x + max(dw, 0) // 2, y + max(dh, 0) // 2
        elif im.size != (w, h):
            print(f"[tile] panel {L} rendered at {im.size}, slot is {(w, h)} -- resizing, "
                  f"which rescales its text relative to the other panels. Pass --exact_size "
                  f"through to the panel renderer to avoid this.")
            im = im.resize((w, h))
        canvas.paste(im, (x, y), im)
        stamp = L.lower() if letter_case == "lower" else L.upper()
        draw.text((x + int(1.5 / 25.4 * dpi), y + int(1 / 25.4 * dpi)),
                  stamp, fill="black", font=ft)
    canvas.save(out_path)
    return out_path, (W, H)


def outline(width_mm, row_mm, layout="wide", sub_colspan=9):
    """
    12-column grid.

      wide     a full width on row 0; b and c side by side on row 1
      stacked  a, b and c each full width on their own row

    WHICH ONE DEPENDS ON b AND c, NOT ON a. Side by side gives each of them half the
    canvas for the same eleven assays, so their per-assay annotations sit about as wide
    apart as the annotation text itself and collide. Stacking trades figure height for
    twice the horizontal room per assay, which is the only thing that actually separates
    them; widening the whole figure buys each of them half of whatever is added.
    """
    if layout == "stacked":
        # A keeps the full twelve columns; B and C take `sub_colspan` of them, both starting
        # at column 0 so their left edges coincide. Their AXES coincide too only because the
        # panels are rendered with a shared --left_margin_in; without it each panel's plot
        # area would start wherever its own tick labels allowed.
        rows = [row_mm[0], row_mm[1], row_mm[1]]
        geom = {"a": (0, 0, 12), "b": (1, 0, sub_colspan), "c": (2, 0, sub_colspan)}
    else:
        rows = list(row_mm)
        geom = {"a": (0, 0, 12), "b": (1, 0, 6), "c": (1, 6, 6)}
    return {
        "claim": "Personalised sequence improves allele-specific binding prediction, and "
                 "the gain does not transfer across individuals.",
        "width_mm": width_mm, "ncol": 12, "row_heights_mm": rows,
        "panels": [
            dict(letter=L, role=("primary" if L == "a" else "evidence"),
                 row=geom[L][0], col=geom[L][1], colspan=geom[L][2],
                 chart_family="paired arrows", message=TITLES[L], data_vid=None, ask="")
            for L in "abc"
        ],
    }


# Flags this composer passes that older copies of the plotter do not have. Checked once,
# before any panel runs, because the failure otherwise arrives as an argparse usage dump
# after the first panel's data has already been scored.
REQUIRED_FLAGS = ("--exact_size", "--assay_label", "--title_fontsize")


def preflight():
    """Fail with one actionable line if the installed arrows plotter is too old."""
    r = subprocess.run([sys.executable, "-m", PLOTTER, "--help"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"could not run {PLOTTER} --help:\n{(r.stderr or '')[-600:]}")
    missing = [f for f in REQUIRED_FLAGS if f not in r.stdout]
    if missing:
        raise SystemExit(
            f"the installed {PLOTTER} does not accept {missing}.\n"
            f"  This composer drives that module as a subprocess, so the two have to be\n"
            f"  updated together. Copy the current fig_auroc_gain_arrows.py into\n"
            f"  <your package>/hapadapter/figures/ and re-run.\n"
            f"  Check with: python -m {PLOTTER} --help | grep -- --exact_size")


def render(args_list, out_png, size_in, dpi, fontsize, label, expect_cells=None):
    """
    Run the arrows plotter once, at an exact size.

    A PANEL THAT DREW ALMOST NOTHING IS REPORTED, NOT JUST ONE THAT CRASHED. The plotter
    treats a missing run as a [miss] line and carries on, so pointing it at a tree that
    does not hold the files it reads produces a successful run and a near-empty panel --
    one assay out of eleven, with a mean arrow computed over that one. That is worse than
    a failure because it still composes. The expected cell count is checked against the
    plotter's own summary line.
    """
    cmd = [sys.executable, "-m", PLOTTER] + args_list + [
        "--size", f"{size_in[0]:.3f}", f"{size_in[1]:.3f}",
        "--dpi", str(dpi), "--fontsize", str(fontsize), "--out", out_png]
    print(f"\n[panel {label}] " + " ".join(cmd[2:]))
    r = subprocess.run(cmd, capture_output=True, text=True)
    for line in (r.stdout or "").splitlines():
        if line.strip():
            print(f"    {line}")
    if r.returncode != 0 or not os.path.exists(out_png):
        sys.stderr.write((r.stderr or "")[-1200:] + "\n")
        n_miss = sum(1 for x in (r.stdout or "").splitlines() if "[miss]" in x)
        hint = (f"\n  Every input was missing ({n_miss} [miss] lines), which usually means "
                f"the root\n  for this panel points somewhere the runs are not."
                if n_miss else "")
        raise SystemExit(f"[panel {label}] the arrows plotter failed; see above.{hint}")
    if expect_cells:
        m = re.search(r"^\s*(\d+) cells", r.stdout or "", re.M)
        got = int(m.group(1)) if m else 0
        if got < max(2, expect_cells // 2):
            print(f"[panel {label}] WARNING: drew {got} of about {expect_cells} expected "
                  f"cells. The rest are the [miss] lines above -- check that this panel's "
                  f"root and --source hold the files it reads.")
    return out_png


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--perlocus_root", required=True, metavar="EXP_ROOT",
                   help="tree holding <assay>_<donor>/stage2_<arm>/runs/*/test_perlocus.csv, "
                        "used for panel a (reference -> personal)")
    p.add_argument("--crossdonor_root", default=None, metavar="EXP_ROOT",
                   help="tree for panels b and c (within -> cross individual). Defaults to "
                        "--perlocus_root for b; --ag_crossdonor_root overrides it for c.")
    p.add_argument("--ag_rows_csv", default=None, metavar="CSV",
                   help="PREFERRED input for panel c: a long table with columns "
                        "assay, individual, arm, auroc and arm in {within, cross} -- the "
                        "shape score_crossdonor's aggregate already writes "
                        "($ROOT/eval/crossdonor_ag_rows.csv). Takes precedence over "
                        "--ag_crossdonor_root.")
    p.add_argument("--ag_crossdonor_root", default=None, metavar="EXP_ROOT",
                   help="fallback for panel c: an experiment tree laid out as "
                        "<assay>_<donor>/eval/crossdonor_<arm>.csv. The AlphaGenome results "
                        "are NOT in this layout -- they are flat per-assay files -- so "
                        "--ag_rows_csv is normally the right flag.")
    p.add_argument("--ag_donor", default="ENC-002",
                   help="AlphaGenome has no per-individual encoder, so panel c is one donor")
    p.add_argument("--assays", default=",".join(ASSAYS))
    p.add_argument("--donors", default=",".join(DONORS))
    p.add_argument("--head_arm", default="personal", choices=["ref", "personal"])
    p.add_argument("--source", default="eval_results",
                   choices=["eval_results", "perlocus"],
                   help="where panel A reads its reference and personal AUROCs. "
                        "'eval_results' reads each run's eval_results.json, which is what "
                        "the trunk trainer writes and what panels B and C already use, so "
                        "all three panels are then the same measure. 'perlocus' recomputes "
                        "a balanced AUROC from test_perlocus.csv, which only exists for "
                        "cells fitted by fit_adapter.")
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--width_mm", type=float, default=183.0,
                   help="figure width. 183 mm is the usual double-column maximum; going "
                        "wider only helps if the figure is placed at its natural size "
                        "rather than scaled to the text width.")
    p.add_argument("--row_mm", type=float, nargs=2, default=[78.0, 58.0], metavar=("TOP", "BOT"))
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--fontsize", type=float, default=8)
    p.add_argument("--title_fontsize_wide", type=float, default=10.0)
    p.add_argument("--title_fontsize_narrow", type=float, default=7.0)
    p.add_argument("--letter_pt", type=float, default=11.0)
    p.add_argument("--gutter_mm", type=float, default=4.0)
    p.add_argument("--layout", default="wide", choices=["wide", "stacked"],
                   help="'wide' puts b and c side by side; 'stacked' gives each of them the "
                        "full width on its own row, which is what separates their per-assay "
                        "annotations when eleven assays crowd a half-width panel.")
    p.add_argument("--sub_colspan", type=int, default=9, choices=range(4, 13),
                   metavar="N", help="stacked layout only: how many of the 12 columns B and "
                                     "C occupy. A always spans all 12. 12 makes all three "
                                     "the same width.")
    p.add_argument("--left_margin_in", type=float, default=0.62,
                   help="shared distance from each panel's left edge to its plot area, so "
                        "the three panels' axes align despite different tick-label widths. "
                        "0 lets each panel choose its own.")
    p.add_argument("--letter_case", default="upper", choices=["lower", "upper"])
    p.add_argument("--keep_panels", default=None, metavar="DIR",
                   help="write the individual panel PNGs here instead of a temp dir")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    if not (a.ag_rows_csv or a.ag_crossdonor_root):
        raise SystemExit("panel c needs one of --ag_rows_csv (a long table with columns "
                         "assay, individual, arm, auroc) or --ag_crossdonor_root (a tree of "
                         "<assay>_<donor>/eval/crossdonor_<arm>.csv).")

    preflight()

    assays = [x.strip() for x in a.assays.split(",") if x.strip()]
    c_assays = [x for x in assays if x.upper() not in {s.upper() for s in DROP_FROM_C}]
    dropped = [x for x in assays if x not in c_assays]
    if dropped:
        print(f"[panel c] excluding {dropped} -- underpowered in the AlphaGenome arm. "
              f"Panels a and b keep all {len(assays)}.")

    ol = outline(a.width_mm, a.row_mm, a.layout, a.sub_colspan)
    cd_root = a.crossdonor_root or a.perlocus_root
    # --exact_size so each panel lands on its slot without a per-panel rescale; see tile().
    common = ["--annotate", "gain", "--exact_size", "--n_boot", str(a.n_boot)] + \
             (["--left_margin_in", str(a.left_margin_in)] if a.left_margin_in else []) + \
             sum([["--assay_label", s] for s in ASSAY_LABELS], [])

    spec = {
        "a": [("--from_perlocus" if a.source == "perlocus" else "--from_eval_results"),
              a.perlocus_root, "--assays", ",".join(assays),
              "--donors", a.donors, "--arm_from", "ref", "--arm_to", "personal",
              "--show_mean", "--title", TITLES["a"]],
        "b": ["--from_crossdonor", cd_root, "--assays", ",".join(assays),
              "--donors", a.donors, "--head_arm", a.head_arm,
              "--average", "--title", TITLES["b"]],
        # Panel c reads the aggregate long table when given one. Its `arm` column is already
        # within/cross, so no arm derivation happens here and nothing is recomputed -- the
        # AUROCs drawn are the ones in that file.
        "c": (["--csv", a.ag_rows_csv, "--arm_from", "within", "--arm_to", "cross",
               "--assays", ",".join(c_assays), "--title", TITLES["c"]]
              if a.ag_rows_csv else
              ["--from_crossdonor", a.ag_crossdonor_root, "--assays", ",".join(c_assays),
               "--donors", a.ag_donor, "--head_arm", a.head_arm, "--title", TITLES["c"]]),
    }

    outdir = a.keep_panels or tempfile.mkdtemp(prefix="auroc_panels_")
    os.makedirs(outdir, exist_ok=True)
    paths = {}
    for L in "abc":
        _x, _y, w, h = panel_box(ol, L, a.dpi, a.gutter_mm)
        # Panel a is full width and carries the larger title; b and c are half width, where
        # the same string at the same size would run off the canvas.
        _full = next(q for q in ol["panels"] if q["letter"] == L)["colspan"] == ol["ncol"]
        tfs = a.title_fontsize_wide if _full else a.title_fontsize_narrow
        n_exp = (len(assays) * len(a.donors.split(",")) if L == "a"
                 else len(assays) if L == "b" else len(c_assays))
        paths[L] = render(spec[L] + common + ["--title_fontsize", str(tfs)],
                          os.path.join(outdir, f"panel_{L}.png"),
                          (w / a.dpi, h / a.dpi), a.dpi, a.fontsize, L,
                          expect_cells=n_exp)

    out_path, (W, H) = tile(ol, paths, a.out, a.dpi, a.gutter_mm, a.letter_case,
                            letter_pt=a.letter_pt)
    print(f"\n[write] {out_path}  ({W} x {H} px, {a.width_mm:.0f} mm wide at {a.dpi} dpi)")
    if a.keep_panels:
        print(f"[write] individual panels in {outdir}")


if __name__ == "__main__":
    main()
