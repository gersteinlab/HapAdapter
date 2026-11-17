#!/usr/bin/env python3
"""
hapadapter.figures.fig_auroc_gain_arrows -- reference -> personal AUROC gain as one arrow per
(assay, individual), coloured by individual. DNABERT-2 only: AlphaGenome has no per-individual
encoder, so there is no reference-vs-personal contrast to draw for it.

WHAT THE FIGURE ENCODES. x is the assay, y is AUROC. Each arrow starts at that cell's REFERENCE
AUROC and ends at its PERSONAL AUROC, so arrow length is the gain and direction is its sign. An
arrow pointing down is a cell where personal sequence HURT, and it is drawn pointing down rather
than dropped.

ARROWS ARE VERTICAL, dodged sideways only to separate individuals within one assay. A slanted
arrow would make the x displacement mean something it does not -- x is a category, and a reader
measuring horizontal travel would be measuring the dodge.

INPUT IS A LONG-FORM CSV, not a directory layout. The metric this figure should carry is still
open: the per-assay bar chart it replaces read eval_results.json, which that script's own
docstring flags as NOT coverage-matched and generally higher than the honest score_asb number --
on CTCF the unmatched 0.71 sits on top of the 0.708 that read depth alone reaches on this data.
The reference-vs-personal DELTA survives that (both arms share the per-assay fold and salt) but
the LEVELS the arrow tips sit at do not. Taking a CSV means switching metric is a different
input file, not a different script.

Required columns: assay, individual, arm, auroc   (arm values: reference | personal)
Optional:         ci_lo, ci_hi, n_per_class

  --from_eval_results walks the old layout and writes those rows for you, so the existing
  numbers are reproducible, but it prints the same metric warning every time.

Usage:
    python -m hapadapter.figures.fig_auroc_gain_arrows \\
      --csv gain_rows.csv --out fig_auroc_gain_arrows.png

    # rebuild the rows from the experiment tree (unmatched metric -- see above)
    python -m hapadapter.figures.fig_auroc_gain_arrows \\
      --from_eval_results $EXP --donors ENC-002 \\
      --assays CTCF,EP300,POLR2A,ATAC,H3K4me3,H3K27ac \\
      --csv_out gain_rows.csv --out fig_auroc_gain_arrows.png

File written by Amy Metrick in collaboration with Anthropic's Claude Science Opus 5 Agent
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

#: Colour PER INDIVIDUAL NUMBER, not per position in --donors. Individuals 1-2 are shades of
#: blue and 3-4 shades of red/pink, encoding donor sex. Keying on the number rather than the
#: order a donor happens to appear in means a given individual keeps its colour across every
#: figure in the paper, including ones that plot a subset of donors.
#:
#: Within a family the two shades differ in LIGHTNESS, so the pairs stay separable in
#: greyscale and under red-green colour vision deficiency; the families differ in hue.
INDIV_COLORS_BY_NUM = {1: "#08519c", 2: "#6baed6", 3: "#a50f15", 4: "#e7718a"}
#: Fallback for anything that is not ENC-<n> with n in the map above.
INDIV_COLORS = ["#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9", "#f0e442"]
#: Used when --average collapses the individual dimension away.
MEAN_COLOR = "#3a3a3a"
MEAN_LABEL = "__mean__"


def color_for(individual, position):
    """Colour by individual NUMBER when the id parses, else by position in the input."""
    if individual == MEAN_LABEL:
        return MEAN_COLOR
    m = re.fullmatch(r"ENC-0*(\d+)", str(individual).strip(), flags=re.I)
    if m and int(m.group(1)) in INDIV_COLORS_BY_NUM:
        return INDIV_COLORS_BY_NUM[int(m.group(1))]
    return INDIV_COLORS[position % len(INDIV_COLORS)]

#: Spelling conveniences for the DEFAULT pair only. Any other --arm_from/--arm_to values are
#: matched literally (after strip+lower), so a new comparison does not need an entry here.
ARM_ALIASES = {"ref": "reference", "pers": "personal"}

KEY_CANDIDATES = ["eval_auroc", "eval_roc_auc", "eval_auc", "auroc", "roc_auc", "auc"]


def pretty_individual(raw, overrides):
    """"ENC-002" -> "individual 2"; the CSV keeps the raw id. See fig_contrast_composite."""
    if raw == MEAN_LABEL:
        return "mean across individuals"
    if raw in overrides:
        return overrides[raw]
    m = re.fullmatch(r"ENC-0*(\d+)", str(raw).strip(), flags=re.I)
    return f"individual {int(m.group(1))}" if m else str(raw)


def find_auroc(d, forced=None):
    if forced:
        return (d.get(forced), forced)
    for k in KEY_CANDIDATES:
        if k in d:
            return d[k], k
    cands = [k for k in d if "auroc" in k.lower()] or [k for k in d if "auc" in k.lower()]
    cands.sort(key=lambda k: ("auroc" not in k.lower(), k))
    return (d[cands[0]], cands[0]) if cands else (None, None)


def rows_from_eval_results(exp_root, assays, donors, run_subdir, metric_key,
                           perlocus_fallback=True, n_boot=1000, seed=20):
    """
    Walk <exp_root>/<assay>_<donor>/stage2_<arm>/runs/<run_subdir>/**/eval_results.json.

    Reproduces the per-assay bar chart's numbers. This metric is NOT coverage-matched; the
    warning is printed once here rather than buried, because a figure built from it can be
    read as a publishable level when it is not one.

    PER-LOCUS FALLBACK. fit_adapter writes test_perlocus.csv and NO eval_results.json, so a
    rebuilt run is invisible to the glob above and its cell silently leaves the figure. When a
    run directory has no eval_results.json but does have test_perlocus.csv, the AUROC is
    computed from that file instead.

    THE TWO ARE THE SAME QUANTITY, NOT TWO METRICS: both are plain AUROC of the head's score
    against the label on that run's own test split -- one logged by the trainer, one recomputed
    from the per-locus scores it was logged from. They are nonetheless RECORDED SEPARATELY in
    the metric_key column ("auroc_from_perlocus"), and every fallback cell is listed at the end
    of this function, because no cell in this project carries both files and that equality is
    therefore an argument rather than something measured here.
    """
    import glob as _glob
    from hapadapter.scoring.stats import balanced_auroc
    print("[metric] eval_results.json is the HF Trainer held-out metric. It is leak-free for a\n"
          "         matched reference-vs-personal comparison (both arms share the per-assay fold\n"
          "         and salt) but it is NOT coverage-matched, so the LEVELS run high. The gain\n"
          "         (arrow length) is the quantity this figure is for; quote levels from\n"
          "         score_matched instead.")
    out, fallback = [], []
    for assay in assays:
        for donor in donors:
            for arm_dir, arm in (("ref", "reference"), ("personal", "personal")):
                base = os.path.join(exp_root, f"{assay.lower()}_{donor}",
                                    f"stage2_{arm_dir}", "runs", run_subdir)
                hits = (glob.glob(os.path.join(base, "results", "*", "eval_results.json"))
                        or glob.glob(os.path.join(base, "**", "eval_results.json"),
                                     recursive=True))
                if not hits:
                    pl = ""
                    if perlocus_fallback:
                        cand = [os.path.join(d, "test_perlocus.csv")
                                for d in sorted(_glob.glob(base)) if os.path.isdir(d)]
                        pl = next((c for c in cand if os.path.exists(c)), "")
                    if not pl:
                        print(f"  [miss] {assay:16} {donor:8} {arm:9} no eval_results.json "
                              f"under {base}")
                        continue
                    d = pd.read_csv(pl)
                    missing = {"ell", "label"} - set(d.columns)
                    if missing:
                        print(f"  [miss] {assay:16} {donor:8} {arm:9} {pl} lacks {sorted(missing)}")
                        continue
                    pt, _, _ci, _n = balanced_auroc(d["ell"].to_numpy(float),
                                                    d["label"].to_numpy(int),
                                                    cover=None, n_boot=n_boot, seed=seed)
                    out.append(dict(assay=assay, individual=donor, arm=arm, auroc=float(pt),
                                    metric_key="auroc_from_perlocus", path=pl))
                    fallback.append(f"{assay} {donor} {arm}")
                    continue
                with open(sorted(hits)[0]) as fh:
                    v, k = find_auroc(json.load(fh), metric_key)
                if v is None:
                    print(f"  [miss] {assay:16} {donor:8} {arm:9} no AUROC key in {sorted(hits)[0]}")
                    continue
                out.append(dict(assay=assay, individual=donor, arm=arm, auroc=float(v),
                                metric_key=k, path=sorted(hits)[0]))
    if fallback:
        print(f"[fallback] {len(fallback)} arm(s) had no eval_results.json and were scored from\n"
              f"           test_perlocus.csv instead -- these are the runs written by fit_adapter:")
        for f in fallback:
            print(f"             {f}")
        print("           In --csv_out these read 'auroc_from_perlocus' in the metric_<arm>\n"
              "           column for that arm specifically, so which cells are recomputed is\n"
              "           recoverable per row. Same quantity as eval_auroc (plain AUROC on that\n"
              "           run's own test split), recomputed from the scores it was logged from.")
    return pd.DataFrame(out)


def rows_from_perlocus(perlocus_root, build_root, assays, donors, arms, run_subdir,
                       metric="plain", cov_col="total_reads", n_boot=1000, seed=20):
    """
    Rows scored from each run's test_perlocus.csv (chr,pos,ell,label).

    PLAIN AUROC BY DEFAULT, which is the metric these arrow figures report and the one the
    y-axis is labelled with. Plain needs nothing but the per-locus file -- it already carries
    both the score and the label -- so build_root is unused unless metric="matched".

    WHY THIS EXISTS RATHER THAN --from_eval_results. fit_adapter writes test_perlocus.csv and
    run_config.json but NO eval_results.json, so rebuilt runs are invisible to that loader.
    Scoring from the per-locus file also computes the metric here rather than trusting whatever
    the trainer happened to log, so an original cell and a rebuilt one are measured the same way.

    metric="matched" additionally reads the split's test.meta.csv for depth. build_root is
    separate from perlocus_root because the rebuild writes runs to a new tree while still
    reading the ORIGINAL builds. Rows are aligned POSITIONALLY -- both files are written in
    test-split order -- with the chr column checked for agreement; a join on (chr,pos) would
    silently fan out wherever two loci share a coordinate.
    """
    import glob as _glob
    from hapadapter.scoring.stats import balanced_auroc
    from hapadapter.backbones.cache import resolve_coord_columns

    rows = []
    for assay in assays:
        for donor in donors:
            cell = f"{assay.lower()}_{donor}"
            for arm, label in arms:
                hits = [d for d in _glob.glob(
                    os.path.join(perlocus_root, cell, f"stage2_{arm}", "runs", run_subdir))
                    if os.path.isdir(d)]
                if not hits:
                    print(f"  [miss] {cell} {arm}: no run under "
                          f"{os.path.join(perlocus_root, cell, f'stage2_{arm}', 'runs', run_subdir)}")
                    continue
                pl = os.path.join(sorted(hits)[-1], "test_perlocus.csv")
                mt = os.path.join(build_root, cell, f"stage2_{arm}_data", "test.meta.csv")
                if not os.path.exists(pl):
                    print(f"  [miss] {cell} {arm}: {pl}")
                    continue
                d = pd.read_csv(pl)
                cov = None
                if metric == "matched" and os.path.exists(mt):
                    m = pd.read_csv(mt)
                    cc, _pc = resolve_coord_columns(m, mt)
                    if len(m) != len(d):
                        print(f"  [warn] {cell} {arm}: meta has {len(m)} rows, per-locus has "
                              f"{len(d)} -- skipping coverage matching for this cell.")
                    elif (m[cc].astype(str).to_numpy() != d["chr"].astype(str).to_numpy()).any():
                        print(f"  [warn] {cell} {arm}: chr columns disagree between the build "
                              f"meta and the per-locus file -- skipping coverage matching.")
                    elif cov_col in m.columns:
                        cov = m[cov_col].to_numpy(float)
                if metric == "matched" and cov is None:
                    print(f"  [warn] {cell} {arm}: fell back to PLAIN AUROC (no usable "
                          f"{cov_col!r}); not comparable to the matched cells in this figure.")
                pt, _, ci, n = balanced_auroc(d["ell"].to_numpy(float),
                                              d["label"].to_numpy(int),
                                              cover=cov, n_boot=n_boot, seed=seed)
                rows.append(dict(assay=assay, individual=donor, arm=label,
                                 auroc=float(pt), ci_lo=float(ci[0]), ci_hi=float(ci[1]),
                                 n_per_class=int(n),
                                 metric="auroc_matched" if cov is not None else "auroc_plain"))
    return pd.DataFrame(rows)


def rows_from_crossdonor(exp_root, assays, donors, head_arm, run_subdir, metric_key):
    """
    Build within/cross rows the way fig_matched_vs_crossdonor did:

      within  <exp_root>/<assay>_<donor>/stage2_<arm>/runs/<run_subdir>/**/eval_results.json
      cross   <exp_root>/<assay>_<donor>/eval/crossdonor_<arm>.csv, rows with
              donor_kind == "cross-donor", mean of auroc over the other donors

    Both are the same quantity (roc_auc on ell over that donor's Stage-2 test.csv), so the
    within->cross change is a clean generalisation measure. It is NOT coverage-matched; see
    the module docstring.
    """
    out = []
    for assay in assays:
        for donor in donors:
            cell = os.path.join(exp_root, f"{assay.lower()}_{donor}")
            base = os.path.join(cell, f"stage2_{head_arm}", "runs", run_subdir)
            hits = (glob.glob(os.path.join(base, "results", "*", "eval_results.json"))
                    or glob.glob(os.path.join(base, "**", "eval_results.json"), recursive=True))
            if hits:
                with open(sorted(hits)[0]) as fh:
                    v, k = find_auroc(json.load(fh), metric_key)
                if v is not None:
                    out.append(dict(assay=assay, individual=donor, arm="within",
                                    auroc=float(v), metric_key=k, path=sorted(hits)[0]))
            else:
                print(f"  [miss] {assay:16} {donor:10} within   no eval_results.json under {base}")

            xp = os.path.join(cell, "eval", f"crossdonor_{head_arm}.csv")
            if not os.path.exists(xp):
                print(f"  [miss] {assay:16} {donor:10} cross    {xp}")
                continue
            cd = pd.read_csv(xp)
            need = {"donor_kind", "auroc"}
            if not need <= set(cd.columns):
                print(f"  [miss] {assay:16} {donor:10} cross    {xp} lacks {sorted(need - set(cd.columns))}")
                continue
            cd = cd[(cd["donor_kind"] == "cross-donor") & cd["auroc"].notna()]
            if cd.empty:
                print(f"  [miss] {assay:16} {donor:10} cross    no cross-donor rows in {xp}")
                continue
            out.append(dict(assay=assay, individual=donor, arm="cross",
                            auroc=float(cd["auroc"].mean()), n_cross=len(cd), path=xp))
    return pd.DataFrame(out)


def load_rows(paths):
    frames = []
    for p in paths:
        if not os.path.exists(p):
            raise SystemExit(f"--csv {p} does not exist")
        frames.append(pd.read_csv(p))
    d = pd.concat(frames, ignore_index=True)
    need = {"assay", "individual", "arm", "auroc"}
    missing = need - set(d.columns)
    if missing:
        raise SystemExit(f"missing column(s) {sorted(missing)}; present: {list(d.columns)}")
    d["arm"] = (d["arm"].astype(str).str.strip().str.lower()
                .map(lambda v: ARM_ALIASES.get(v, v)))
    return d


def attach_row_metrics(pairs, d, arm_from, arm_to, mean_label):
    """
    Record, FOR EACH ROW, which metric each end of its arrow came from.

    WHY PER ROW AND NOT ONE GLOBAL STRING. The loaders can mix sources within a single figure:
    --from_eval_results falls back to test_perlocus.csv for runs that fit_adapter wrote, so one
    cell can be recomputed while its neighbours are trainer-logged. A single joined value --
    which is what this used to write -- stamped every row with the union, so a reader could see
    THAT the figure mixed sources but not WHICH cells did, and the per-cell attribution the
    fallback message promises was not in fact in the file.

    Adds two columns named for the arrow's ends. Aggregated rows (the mean) take the sorted
    union over the individuals they average, since a mean of one recomputed cell and three
    logged ones is neither one thing nor the other and should say so.
    """
    col = ("metric_key" if "metric_key" in d.columns
           else "metric" if "metric" in d.columns else None)
    if col is None:
        return pairs
    mk = {(str(r.assay), str(r.individual), str(r.arm)): str(getattr(r, col))
          for r in d.itertuples()}
    by_assay = {}
    for (assay, _ind, arm), v in mk.items():
        by_assay.setdefault((assay, arm), set()).add(v)

    def lookup(assay, indiv, arm):
        if indiv == mean_label:
            return "+".join(sorted(by_assay.get((assay, arm), {"?"})))
        return mk.get((assay, indiv, arm), "?")

    for arm in (arm_from, arm_to):
        pairs[f"metric_{arm}"] = [lookup(str(r.assay), str(r.individual), arm)
                                  for r in pairs.itertuples()]
    return pairs


def to_pairs(d, arm_from, arm_to):
    """
    One row per (assay, individual) carrying both arms. A cell missing either has no
    transition to draw, so it is REPORTED AND DROPPED rather than plotted at an invented
    baseline.
    """
    present = sorted(d["arm"].unique())
    for want in (arm_from, arm_to):
        if want not in present:
            raise SystemExit(f"no rows with arm={want!r}; the input has {present}. "
                             f"Set --arm_from / --arm_to to two of those.")
    d = d[d["arm"].isin([arm_from, arm_to])]
    w = d.pivot_table(index=["assay", "individual"], columns="arm", values="auroc",
                      aggfunc="mean")
    for c in (arm_from, arm_to):
        if c not in w.columns:
            w[c] = np.nan
    for (assay, indiv), r in w[w[[arm_from, arm_to]].isna().any(axis=1)].iterrows():
        have = [c for c in (arm_from, arm_to) if not pd.isna(r[c])]
        print(f"  [skip] {assay:16} {indiv:10} only has {have} -- no transition to draw")
    w = w.dropna(subset=[arm_from, arm_to]).reset_index()
    w["delta"] = w[arm_to] - w[arm_from]
    return w


def main():
    ap = argparse.ArgumentParser(
        description="reference -> personal AUROC gain, one arrow per (assay, individual)")
    ap.add_argument("--csv", action="append", default=[],
                    help="long-form CSV: assay, individual, arm, auroc. Repeatable.")
    ap.add_argument("--from_perlocus", default=None, metavar="ROOT",
                    help="score every cell from <ROOT>/<assay>_<donor>/stage2_<arm>/runs/"
                         "<run_subdir>/test_perlocus.csv, coverage-matched against the build's "
                         "test.meta.csv. Use this for rebuilt runs: fit_adapter writes no "
                         "eval_results.json, and this metric is the coverage-matched one the "
                         "paper reports rather than the trainer's unmatched AUROC.")
    ap.add_argument("--perlocus_metric", choices=["plain", "matched"], default="plain",
                    help="--from_perlocus only. plain (default) is the AUROC these figures "
                         "report and needs nothing but test_perlocus.csv; matched additionally "
                         "reads the build's test.meta.csv to coverage-match.")
    ap.add_argument("--no_perlocus_fallback", action="store_true",
                    help="--from_eval_results only: do NOT fall back to test_perlocus.csv for a "
                         "run that has no eval_results.json. The fallback exists because "
                         "fit_adapter writes no eval_results.json, so a rebuilt cell would "
                         "silently leave the figure; pass this when the figure must contain "
                         "ONLY trainer-logged numbers.")
    ap.add_argument("--n_boot", type=int, default=1000,
                    help="--from_perlocus only: bootstrap draws for the CI columns in "
                         "--csv_out. The arrows use the point estimate, so 200 is plenty.")
    ap.add_argument("--seed", type=int, default=20,
                    help="--from_perlocus only: bootstrap seed.")
    ap.add_argument("--build_root", default=None, metavar="ROOT",
                    help="where the stage2_<arm>_data builds live; defaults to --from_perlocus. "
                         "Set it when the runs were written to a rebuild tree while still "
                         "reading the ORIGINAL builds.")
    ap.add_argument("--from_crossdonor", default=None, metavar="EXP_ROOT",
                    help="build within/cross rows from eval_results.json and "
                         "<cell>/eval/crossdonor_<arm>.csv. Implies --arm_from within "
                         "--arm_to cross unless you override them.")
    ap.add_argument("--head_arm", choices=["ref", "personal"], default="personal",
                    help="--from_crossdonor: which trained head (NOT the arrow's two ends)")
    ap.add_argument("--from_eval_results", default=None, metavar="EXP_ROOT",
                    help="build the rows by walking the experiment tree instead (unmatched "
                         "metric -- see the module docstring)")
    ap.add_argument("--assays", default=None,
                    help="comma-separated, also sets left-to-right panel order. Defaults to "
                         "the order encountered in the input.")
    ap.add_argument("--donors", default="ENC-002", help="comma-separated (--from_eval_results)")
    ap.add_argument("--run_subdir", default="clf_s20_seed20")
    ap.add_argument("--metric_key", default=None)
    ap.add_argument("--individual_label", action="append", default=[], metavar="RAW=DISPLAY")
    ap.add_argument("--show_mean", action="store_true",
                    help="draw the mean-across-individuals arrow ALONGSIDE the per-individual "
                         "ones, at the right of each assay's group. Only that arrow is "
                         "labelled with a number, so the group stays readable.")
    ap.add_argument("--group_frac", type=float, default=0.45,
                    help="fraction of one assay's slot that the arrows occupy. Lower packs "
                         "the arrows within an assay tighter and widens the gap between "
                         "assays; the dodge is derived from it and the number of arrows, so "
                         "the rhythm does not change when the donor count does. "
                         "--dodge overrides it.")
    ap.add_argument("--average", action="store_true",
                    help="one arrow per assay, averaged over individuals. For the cross-donor "
                         "figure this is a SECOND average: each cell's cross value is already "
                         "the mean over the other donors, and this then averages over the "
                         "training donors.")
    ap.add_argument("--arm_from", default="reference", help="arm the arrow TAIL sits at")
    ap.add_argument("--arm_to", default="personal", help="arm the arrow HEAD sits at")
    ap.add_argument("--geometry", choices=["vertical", "slanted"], default="vertical",
                    help="vertical: arrows rise in place, length = the change. Use when x is "
                         "only a category. slanted: arrows run left-to-right across a fixed "
                         "horizontal span so their SLOPE is the change and FLAT means no "
                         "change -- use when the two arms are a transition (within-individual "
                         "-> cross-donor), where the horizontal axis genuinely means something.")
    ap.add_argument("--span", type=float, default=None,
                    help="slanted only: horizontal length of one arrow, in x units. Defaults "
                         "to 0.75*dodge so arrows from neighbouring individuals cannot overlap.")
    ap.add_argument("--dodge", type=float, default=None,
                    help="explicit horizontal separation between arrows within one assay, in "
                         "x units. Overrides --group_frac.")
    ap.add_argument("--annotate", choices=["none", "gain", "endpoints"], default="gain",
                    help="gain: print the delta beside each arrow. endpoints: print both AUROCs.")
    ap.add_argument("--ymin", type=float, default=None, help="default: just below the lowest point")
    ap.add_argument("--ymax", type=float, default=None)
    ap.add_argument("--size", type=float, nargs=2, default=[7.2, 4.2], metavar=("W", "H"))
    ap.add_argument("--fontsize", type=float, default=8)
    ap.add_argument("--xtick_rotation", type=float, default=30,
                    help="assay label rotation. A narrower figure needs a steeper angle: the "
                         "longest assay name sets how close the ticks can sit before they "
                         "collide, not the arrows.")
    ap.add_argument("--title", default="")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--csv_out", default=None,
                    help="write the resolved per-cell rows (reference, personal, gain)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    if a.csv_out:
        os.makedirs(os.path.dirname(os.path.abspath(a.csv_out)) or ".", exist_ok=True)

    sources = [bool(a.csv), bool(a.from_perlocus), bool(a.from_eval_results),
               bool(a.from_crossdonor)]
    if not any(sources):
        raise SystemExit("give one of --csv, --from_perlocus, --from_eval_results, "
                         "--from_crossdonor")
    if sum(sources) > 1:
        raise SystemExit("--csv, --from_perlocus, --from_eval_results and --from_crossdonor "
                         "are alternatives; pass exactly one")

    assays_cli = [x.strip() for x in a.assays.split(",")] if a.assays else None
    if a.from_perlocus:
        # Without --assays there is nothing to walk and the loop raised an opaque
        # "'NoneType' object is not iterable" from inside the loader. Say so here instead.
        if not assays_cli:
            raise SystemExit("--from_perlocus needs --assays (comma-separated), since the "
                             "run directories are addressed as <root>/<assay>_<donor>/...")
        d = rows_from_perlocus(a.from_perlocus, a.build_root or a.from_perlocus,
                               assays_cli, [x.strip() for x in a.donors.split(",")],
                               [("ref", "reference"), ("personal", "personal")],
                               a.run_subdir, metric=a.perlocus_metric, n_boot=a.n_boot,
                               seed=a.seed)
        if d.empty:
            raise SystemExit("found no test_perlocus.csv for any (assay, donor, arm)")
        # Carry the metric through to the output table: a figure of plain AUROC and one of
        # matched AUROC are not interchangeable, and the CSV is what a methods line is written
        # from months later.
        mixed = set(d["metric"])
        if len(mixed) > 1:
            print(f"  [warn] this figure MIXES {sorted(mixed)} across cells. The matched and "
                  f"unmatched metrics are not on the same scale; fix the missing coverage "
                  f"column rather than comparing these arrows.")
    elif a.from_crossdonor:
        if not assays_cli:
            raise SystemExit("--from_crossdonor needs --assays")
        # Defaults that make this source plot the right figure without four extra flags. An
        # explicit --arm_from/--arm_to/--geometry still wins, since argparse already recorded
        # whatever the user passed and these only fire on the untouched defaults.
        if (a.arm_from, a.arm_to) == ("reference", "personal"):
            a.arm_from, a.arm_to = "within", "cross"
        d = rows_from_crossdonor(a.from_crossdonor, assays_cli,
                                 [x.strip() for x in a.donors.split(",")],
                                 a.head_arm, a.run_subdir, a.metric_key)
        if d.empty:
            raise SystemExit("found no within or cross rows for any (assay, donor)")
    elif a.from_eval_results:
        if not assays_cli:
            raise SystemExit("--from_eval_results needs --assays")
        d = rows_from_eval_results(a.from_eval_results, assays_cli,
                                   [x.strip() for x in a.donors.split(",")],
                                   a.run_subdir, a.metric_key,
                                   perlocus_fallback=not a.no_perlocus_fallback,
                                   n_boot=a.n_boot, seed=a.seed)
        if d.empty:
            raise SystemExit("no eval_results.json or test_perlocus.csv found for any "
                             "(assay, donor, arm)")
    else:
        d = load_rows(a.csv)

    arm_from = ARM_ALIASES.get(a.arm_from.strip().lower(), a.arm_from.strip().lower())
    arm_to = ARM_ALIASES.get(a.arm_to.strip().lower(), a.arm_to.strip().lower())
    if arm_from == arm_to:
        raise SystemExit("--arm_from and --arm_to are the same value")
    if a.average and a.show_mean:
        raise SystemExit("--average REPLACES the individual arrows with the mean; "
                         "--show_mean ADDS it beside them. Pass one.")
    pairs = to_pairs(d, arm_from, arm_to)
    if a.show_mean and not pairs.empty:
        n_by_assay = pairs.groupby("assay")["individual"].nunique()
        m = pairs.groupby("assay", as_index=False)[[arm_from, arm_to, "delta"]].mean()
        m["n_individuals"] = m["assay"].map(n_by_assay).astype(int)
        m["individual"] = MEAN_LABEL
        pairs = pd.concat([pairs, m], ignore_index=True)
    if a.average and not pairs.empty:
        # COLLAPSE THE INDIVIDUAL DIMENSION. Mean over the individuals that have BOTH arms, so
        # an assay where one donor is missing a side is averaged over the donors it does have
        # rather than dropped -- n_individuals records how many, because a mean over 2 and a
        # mean over 4 should not be read as equally firm.
        n_by_assay = pairs.groupby("assay")["individual"].nunique()
        pairs = (pairs.groupby("assay", as_index=False)[[arm_from, arm_to, "delta"]].mean())
        pairs["n_individuals"] = pairs["assay"].map(n_by_assay).astype(int)
        pairs["individual"] = MEAN_LABEL
        print("  [average] one arrow per assay, mean over individuals: "
              + ", ".join(f"{r.assay}={r.n_individuals}" for r in pairs.itertuples()))
    if pairs.empty:
        raise SystemExit(f"no (assay, individual) cell has both {arm_from!r} and {arm_to!r}")
    # After every aggregation, so a mean row reports the union over what it averaged.
    pairs = attach_row_metrics(pairs, d, arm_from, arm_to, MEAN_LABEL)


    order = assays_cli or list(dict.fromkeys(d["assay"]))
    order = [x for x in order if x in set(pairs["assay"])]
    indivs = [x for x in dict.fromkeys(pairs["individual"]) if x != MEAN_LABEL]
    if MEAN_LABEL in set(pairs["individual"]):
        indivs.append(MEAN_LABEL)        # always rightmost in the group
    xpos = {s: i for i, s in enumerate(order)}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch

    fig, ax = plt.subplots(figsize=tuple(a.size))

    # Centre the fan on the assay's tick: k individuals occupy (k-1)*dodge and are centred,
    # so a single-individual figure sits exactly on the tick rather than off to one side.
    k = len(indivs)
    # Derive the dodge from --group_frac so "tight within an assay, clear gap between assays"
    # is one number that holds whether there are 2 arrows or 5. --dodge still overrides.
    dodge = a.dodge if a.dodge is not None else (a.group_frac / max(k - 1, 1))
    offs = {ind: (j - (k - 1) / 2) * dodge for j, ind in enumerate(indivs)}

    lo = min(pairs[arm_from].min(), pairs[arm_to].min())
    hi = max(pairs[arm_from].max(), pairs[arm_to].max())
    pad = max(0.02, 0.10 * (hi - lo))
    y0 = a.ymin if a.ymin is not None else min(0.5, lo) - pad
    y1 = a.ymax if a.ymax is not None else hi + pad

    span = a.span if a.span is not None else dodge * 0.75
    ax.axhline(0.5, color="0.55", lw=0.8, ls=":", zorder=1)

    for _, r in pairs.iterrows():
        if r["assay"] not in xpos:
            continue
        x = xpos[r["assay"]] + offs[r["individual"]]
        c = color_for(r["individual"], indivs.index(r["individual"]))
        if a.geometry == "slanted":
            x0, x1 = x - span / 2, x + span / 2
        else:
            x0 = x1 = x
        ax.add_patch(FancyArrowPatch(
            (x0, r[arm_from]), (x1, r[arm_to]),
            arrowstyle="-|>", mutation_scale=9, lw=1.6, color=c,
            shrinkA=0, shrinkB=0, zorder=3))
        # A short HORIZONTAL tick at the tail. In vertical mode it marks the starting level,
        # which an arrowhead alone does not show. In slanted mode it does more work: it is the
        # zero-slope reference the eye compares the arrow against, which is what makes "flat"
        # readable as a quantity rather than an impression.
        tick = (span * 0.40) if a.geometry == "slanted" else (dodge * 0.26)
        ax.plot([x0 - tick, x0 + tick], [r[arm_from]] * 2, color=c, lw=1.0, zorder=3)
        # With --show_mean the per-individual arrows are deliberately unlabelled: five numbers
        # per assay is unreadable at this width, and the mean is the one being quoted.
        if a.show_mean and r["individual"] != MEAN_LABEL:
            continue
        if a.annotate == "gain":
            top = max(r[arm_from], r[arm_to])
            # THREE DECIMALS, not two: at two, a real -0.004 prints as "-0.00", which reads as
            # a formatting fault rather than as a small negative number -- and on the
            # cross-donor figure almost every delta is that size.
            #
            # The mean's label is LEFT-ALIGNED just right of its arrow rather than centred on
            # it. Centred, it overhung the neighbouring individual's arrowhead, because the
            # mean sits at the right edge of the group.
            if r["individual"] == MEAN_LABEL:
                ax.text(x + dodge * 0.30, top, f"{r['delta']:+.03f}", ha="left", va="bottom",
                        fontsize=a.fontsize - 1.5, color=c, zorder=5)
            else:
                ax.text(x, top + 0.004, f"{r['delta']:+.03f}", ha="center", va="bottom",
                        fontsize=a.fontsize - 2, color=c, zorder=4)
        elif a.annotate == "endpoints":
            ax.text(x1, r[arm_to] + 0.004, f"{r[arm_to]:.03f}", ha="center",
                    va="bottom", fontsize=a.fontsize - 2, color=c, zorder=4)
            ax.text(x0, r[arm_from] - 0.006, f"{r[arm_from]:.03f}", ha="center",
                    va="top", fontsize=a.fontsize - 2, color="0.35", zorder=4)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=a.fontsize, rotation=a.xtick_rotation,
                       ha="right" if a.xtick_rotation else "center")
    # Margin scales with how much width the arrow group actually uses. A half-slot margin is
    # right for a five-arrow group but leaves conspicuous dead space on an --average figure,
    # where each assay is a single arrow occupying no width at all.
    margin = 0.34 + (k - 1) * dodge / 2
    ax.set_xlim(-margin, len(order) - 1 + margin + (0.22 if a.show_mean else 0.0))
    ax.set_ylim(y0, y1)
    ax.set_ylabel("AUROC", fontsize=a.fontsize)
    ax.tick_params(labelsize=a.fontsize - 1)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if a.title:
        # suptitle rather than set_title: the legend now occupies the space directly above the
        # axes, and a set_title would be laid out underneath it.
        fig.suptitle(a.title, fontsize=a.fontsize + 2, y=1.0)

    overrides = {}
    for spec in a.individual_label:
        if "=" not in spec:
            raise SystemExit(f"--individual_label {spec!r} is not 'RAW=DISPLAY'")
        kk, vv = spec.split("=", 1)
        overrides[kk.strip()] = vv.strip()
    handles = [Line2D([], [], color=color_for(ind, j), lw=1.8, marker=">", markersize=4,
                      label=pretty_individual(ind, overrides))
               for j, ind in enumerate(indivs)]
    handles.append(Line2D([], [], color="0.35", lw=0, marker="_", markersize=7,
                          label=f"tail = {arm_from}, head = {arm_to}"))
    # TWO ROWS, WITH THE TAIL KEY DIRECTLY BELOW THE MEAN. matplotlib fills a legend
    # COLUMN-MAJOR and spreads n entries over ncol columns as evenly as it can, so 6 entries in
    # 4 columns becomes 2,2,1,1 -- which lands the mean and the tail key side by side in the two
    # short columns. Forcing an even two-row split (ncol = n/2) makes the final column exactly
    # [mean, tail]. With an odd number of individuals that split would place the mean one slot
    # early, so a blank entry pads the individuals to an even count first.
    legend_ncol = min(4, len(handles))
    if MEAN_LABEL in indivs and len(handles) >= 4:
        if (len(indivs) - 1) % 2:
            handles.insert(len(indivs) - 1, Line2D([], [], color="none", lw=0, label=" "))
        legend_ncol = len(handles) // 2
    # ABOVE THE AXES, NOT INSIDE. The lower-left is where the weakest assay's arrows live --
    # EP300's tails reach below the chance line -- so an in-axes legend sat on top of real
    # data in the multi-individual case. Putting it above costs vertical space and collides
    # with nothing, at any number of individuals.
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.005),
              frameon=False, fontsize=a.fontsize - 1,
              ncol=legend_ncol, columnspacing=1.4, handletextpad=0.5)

    fig.tight_layout()
    fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight")
    print(f"[write] {a.out}")

    n_up = int((pairs["delta"] > 0).sum())
    flat = int((pairs["delta"].abs() <= 0.02).sum())
    print(f"  {len(pairs)} cells | {arm_to} ahead in {n_up}/{len(pairs)} | "
          f"median {pairs['delta'].median():+.4f} | "
          f"range {pairs['delta'].min():+.4f} to {pairs['delta'].max():+.4f}")
    print(f"  within +/-0.02 of no change: {flat}/{len(pairs)} cells")

    if a.csv_out:
        pairs.to_csv(a.csv_out, index=False)
        print(f"[write] {a.csv_out}")


if __name__ == "__main__":
    sys.exit(main())
