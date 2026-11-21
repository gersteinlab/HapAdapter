#!/usr/bin/env python3
"""
fig_ism_motif_example.py -- ONE ISM window: per-base allelic saliency vs JASPAR motif hits.
Standalone (reads only the npz + a JASPAR PFM json; no model stack).

Top panel : gray = per-position motif-hit COUNT over the chosen motifs (left y, "Motif Count");
            blue = ISM saliency track (right y). Dotted line = the variant at window center.
Lower panel: motif-hit lanes -- one row per motif with a hit (name at left), bars at [start,end]
            colored by -log10 p (viridis). Capped to --max_rows most-significant motifs.

Which motifs to include is YOUR choice: --motifs is a comma-separated list of JASPAR TF-name
substrings (e.g. "CTCF,CTCFL") or matrix ids, or "ALL".

Title: "{--title}\n{chr}: {start}-{end}" -- pass e.g. --title "Ind. 1, ATACseq"; the genomic
window span is appended automatically. Overlap metrics (spearman/enrichment/mass) are printed
to stdout (not drawn on the figure).

Window selection: --select top_depth (highest total_reads), or index:<N>, or chr:<anchor>.

  python fig_ism_motif_example.py --npz ism_ref_AS.npz --jaspar jaspar2024_core_vert_pfms.json \
      --motifs CTCF --select top_depth --title "Ind. 1, CTCF" --zoom_bp 75 --out fig_ism_motif_ctcf.png
"""
import argparse, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr

B = {"A": 0, "C": 1, "G": 2, "T": 3}
plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 8, "axes.titlesize": 9,
    "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4,
})
C_SAL = "#2c6fbb"


def rank_by_ism_peak(by_motif, sal, peak_q, n_null, seed, smooth=0.0):
    """
    Order lanes by how strongly their hits sit on ISM saliency, with a shift null.

    SCORE. For each motif, the best of its hits by mean |saliency| across the hit's span. Mean
    rather than max so a long matrix is not rewarded for containing one salient base, and the
    best hit rather than the mean over hits so a motif with one strong site is not diluted by
    its own background occurrences.

    WHY THERE IS A NULL AT ALL. Selecting motifs BECAUSE they overlap the saliency peaks and
    then drawing them under the saliency track is circular: the alignment is guaranteed by the
    selection, so the panel cannot be evidence that the model attends to those motifs. The null
    is what makes the display interpretable -- saliency is circularly shifted n_null times with
    the hit positions held fixed, and p is the fraction of shifts scoring at least as high. A
    motif that beats the null is one whose placement relative to THIS saliency profile is
    unusual; one that does not is a motif that occurs often enough to land on a peak anyway.

    Returns (order, table) with table rows (name, score, p, n_hits, peak_frac).
    """
    a_sal = np.abs(gaussian_filter1d(sal, smooth) if smooth else sal)
    L = len(a_sal)
    thr = np.quantile(a_sal, peak_q)
    rng = np.random.default_rng(seed)
    shifts = rng.integers(1, L, size=n_null) if n_null else np.empty(0, dtype=int)

    def best(arr, spans):
        return max(float(arr[s:min(e, L)].mean()) for s, e in spans if min(e, L) > s)

    table = []
    for nm, hs in by_motif.items():
        spans = [(s, min(e, L)) for s, e, _ in hs if min(e, L) > s]
        if not spans:
            continue
        obs = best(a_sal, spans)
        # peak_frac is reported, not used for ranking: it answers "is this lane ON a peak"
        # separately from "how high is the saliency there", which differ for a broad shoulder.
        cov = np.zeros(L, dtype=bool)
        for s, e in spans:
            cov[s:e] = True
        pk = float((a_sal[cov] >= thr).mean()) if cov.any() else 0.0
        if n_null:
            null = np.array([best(np.roll(a_sal, int(k)), spans) for k in shifts])
            p = float((null >= obs).sum() + 1) / (n_null + 1)
        else:
            p = float("nan")
        table.append((nm, obs, p, len(spans), pk))
    table.sort(key=lambda r: -r[1])
    return [r[0] for r in table], table


def select_pwms(jaspar_json, motifs_arg, pseudo=0.0, exact=False):
    """{matrix_id: (pfm(4,W), display_name)} for motifs matching any --motifs token.

    exact=False (default) matches tokens as SUBSTRINGS of the name, which is what --motifs and
    --count_motifs document ("CTCF" also pulls CTCFL). exact=True requires the whole name to equal
    the token -- used by the motif_* window selectors so their reported percentages agree with
    fig_as_motif_tracks.load_motifs, which matches exactly. Mixing the two silently doubled the
    reported central-hit rate (24.0% vs the aggregate test's 12.6%).
    """
    J = json.load(open(jaspar_json))
    toks = [t.strip().upper() for t in motifs_arg.split(",") if t.strip()]
    allm = (len(toks) == 1 and toks[0] == "ALL")
    out = {}
    for mid, v in J.items():
        name = v.get("name", mid)
        hit = (name.upper() in toks or mid.upper() in toks) if exact else \
              any(t in name.upper() or t == mid.upper() for t in toks)
        if allm or hit:
            p = v["pfm"]; M = np.array([p["A"], p["C"], p["G"], p["T"]], dtype=np.float64)
            if pseudo:
                # A pseudocount FLATTENS a probability PFM ([.05 .85 .05 .05] -> [.107 .679 ...],
                # max log2-odds 1.77 -> 1.44), so fimo accepts weaker matches at the same p-value
                # threshold -- measured ~1.8x more total hits. fig_as_motif_tracks.load_motifs uses
                # the RAW matrix, so a nonzero pseudo here makes the two scripts disagree. Default 0.
                M = M + pseudo
                M = M / M.sum(0, keepdims=True)
            out[mid] = (M, name)
    return out


def suppress_overlaps(hits, max_overlap):
    """
    Collapse runs of overlapping hits of the SAME motif, keeping the best-scoring one.

    WHY THIS IS NOT OPTIONAL AT A LOOSE THRESHOLD. fimo reports every offset that clears the
    threshold, so one strong site yields a RUN of passing offsets one or two bp apart. Drawn
    as lanes they merge into a single bar whose length is the length of the run, not of the
    motif -- a 19 bp CTCF matrix rendering as a ~100 bp bar, which a reader reads as a footprint.
    It also inflates the count track and the union coverage behind peak_frac.

    Greedy by score: take the best hit, drop any same-motif hit overlapping it by more than
    max_overlap of the SHORTER interval, repeat. max_overlap >= 1 disables suppression.
    """
    if max_overlap >= 1.0:
        return hits, 0
    out, by = [], {}
    for h in hits:
        by.setdefault(h[0], []).append(h)
    for nm, hs in by.items():
        kept = []
        for name, s, e, val in sorted(hs, key=lambda r: -r[3]):
            ov = False
            for _n, ks, ke, _v in kept:
                inter = min(e, ke) - max(s, ks)
                if inter > 0 and inter > max_overlap * min(e - s, ke - ks):
                    ov = True
                    break
            if not ov:
                kept.append((name, s, e, val))
        out.extend(kept)
    return sorted(out, key=lambda r: (r[0], r[1])), len(hits) - len(out)


def scan(seq, pwms, threshold):
    """fimo over one sequence -> list of (name, start, end, neglogp). neglogp from the p-value
    column if present, else the score (label adapts)."""
    from memelite import fimo
    X = np.zeros((1, 4, len(seq)), dtype=np.float32)
    for i, ch in enumerate(seq.upper()):
        if ch in B:
            X[0, B[ch], i] = 1.0
    flat = {mid: pwm for mid, (pwm, _) in pwms.items()}
    name_of = {mid: nm for mid, (_, nm) in pwms.items()}
    res = fimo(flat, X, threshold=threshold)
    hits, pcol = [], None
    for df in res:
        if len(df) == 0:
            continue
        if pcol is None:
            pcol = next((c for c in df.columns if "p-value" in c.lower() or c.lower() in ("pval", "pvalue")), None)
        for _, r in df.iterrows():
            mid = str(r["motif_name"])
            s, e = int(r["start"]), int(r["end"])
            if pcol is not None and np.isfinite(r[pcol]) and r[pcol] > 0:
                val = -np.log10(float(r[pcol]))
            else:
                val = float(r["score"])
            hits.append((name_of.get(mid, mid), s, e, val))
    return hits, ("\u2212log10 p" if pcol is not None else "motif score")


def _windows_with_central_hit(z, jaspar_json, tf, central_bp, threshold, whole_window=False,
                              pfm_pseudo=0.0):
    """Which windows carry a `tf` motif hit overlapping the central +/-central_bp? -> (n,) bool.

    ONE fimo pass over all windows (not a per-window loop): the same per-window overlap definition
    the aggregate central-enrichment test uses, so an example picked here is an instance of the
    population that test measures.
    """
    from memelite import fimo
    seqs = [str(x).upper() for x in z["seqs"]]
    n = len(seqs)
    Lw = len(seqs[0])
    if any(len(x) != Lw for x in seqs):
        raise SystemExit("windows are not all the same length; cannot batch-scan")
    sel = select_pwms(jaspar_json, tf, pseudo=pfm_pseudo, exact=True)        # same definition as the aggregate test
    if not sel:
        raise SystemExit(f"--select ...:{tf} matched no JASPAR entry")
    codes = np.full(256, -1, dtype=np.int16)
    for ch, k in B.items():
        codes[ord(ch)] = k
    raw = np.frombuffer("".join(seqs).encode("ascii"), dtype=np.uint8).reshape(n, Lw)
    idx = codes[raw]
    X = np.zeros((n, 4, Lw), dtype=np.float32)
    ii, jj = np.nonzero(idx >= 0)
    X[ii, idx[ii, jj], jj] = 1.0
    c = Lw // 2
    lo, hi = max(c - central_bp, 0), min(c + central_bp + 1, Lw)
    has = np.zeros(n, dtype=bool)
    for df in fimo({mid: pwm for mid, (pwm, _) in sel.items()}, X, threshold=threshold):
        if len(df) == 0:
            continue
        for _, r in df.iterrows():
            if whole_window or (int(r["end"]) > lo and int(r["start"]) < hi):
                has[int(r["sequence_name"])] = True
    return has


def parse_lane_groups(arg, pwms, exact):
    """'label:TF1,TF2;label2:TF3' -> [(label, [display_name, ...]), ...], order preserved.

    Family / cofactor-class assignment is a biological judgement, not a lookup, so it is declared
    here rather than inferred. Groups set BOTH the lane order (members contiguous) and the colour,
    so same-class lanes sit together and the legend stays short.
    """
    names = {nm for _, nm in pwms.values()}
    out, seen = [], set()
    for blk in [b.strip() for b in arg.split(";") if b.strip()]:
        if ":" not in blk:
            raise SystemExit(f"--lane_groups block {blk!r} must look like 'label:TF1,TF2'")
        lab, csv = blk.split(":", 1)
        lab = lab.strip()
        members, missing = [], []
        for t in [x.strip() for x in csv.split(",") if x.strip()]:
            tu = t.upper()
            hit = sorted(nm for nm in names
                         if (nm.upper() == tu if exact else tu in nm.upper()))
            if not hit:
                missing.append(t)
            for nm in hit:
                if nm not in seen:                 # first group wins on a duplicate
                    members.append(nm); seen.add(nm)
        if missing:
            print(f"[warn] --lane_groups '{lab}': no JASPAR entry for {', '.join(missing)}")
        if members:
            out.append((lab, members))
        else:
            print(f"[warn] --lane_groups '{lab}' has no matching motif -> group dropped")
    return out


def cofactor_lane_order(motifs_arg, pwms, exact):
    """Display names for a FIXED lane roster, in the order the tokens were given.

    Lanes normally appear only for motifs that happen to hit this window, so the row set changes
    between loci and two panels cannot be compared. With a fixed roster every known cofactor gets
    a row whether or not it hits here -- an empty row is the finding ("this cofactor is absent").
    """
    names = {nm for _, nm in pwms.values()}
    out, unmatched = [], []
    for t in [x.strip() for x in motifs_arg.split(",") if x.strip()]:
        tu = t.upper()
        matched = sorted(nm for nm in names
                         if (nm.upper() == tu if exact else tu in nm.upper()))
        if not matched:
            unmatched.append(t)                 # a typo would otherwise drop the lane silently
        for nm in matched:
            if nm not in out:
                out.append(nm)
    if unmatched:
        print(f"[warn] {len(unmatched)} roster token(s) matched no JASPAR entry and have NO lane: "
              + ", ".join(unmatched)
              + ("  (exact matching is on; check spelling/case-insensitive exact name)" if exact
                 else "  (check spelling)"))
    for nm in sorted(names):            # anything the tokens did not cover, appended
        if nm not in out:
            out.append(nm)
    return out


def pick_window(z, select, jaspar_json=None, threshold=1e-4, central_bp=10,
                prefer_nonempty=False, count_motifs=None, pfm_pseudo=0.0):
    n = z["importance"].shape[0]
    if select == "top_depth":
        tr = z["total_reads"].astype(float) if "total_reads" in z else np.zeros(n)
        return int(np.argmax(tr))
    if select.startswith("index:"):
        return int(select.split(":", 1)[1])
    if select.startswith("chr:"):
        an = int(select.split(":", 1)[1]); anc = z["anchor"].astype(np.int64)
        w = np.where(anc == an)[0]
        if len(w) == 0:
            raise SystemExit(f"no window with anchor {an}")
        return int(w[0])
    for kind, want in (("motif_at_center:", True), ("no_motif_at_center:", False)):
        if select.startswith(kind):
            tf = select.split(":", 1)[1]
            has = _windows_with_central_hit(z, jaspar_json, tf, central_bp, threshold,
                                         pfm_pseudo=pfm_pseudo)
            ok = np.where(has == want) [0]
            if len(ok) == 0:
                raise SystemExit(
                    f"no window {'with' if want else 'without'} a central {tf} hit "
                    f"(+/-{central_bp}bp) among {len(has)}; "
                    f"{int(has.sum())}/{len(has)} windows have one")
            tr = z["total_reads"].astype(float) if "total_reads" in z else np.zeros(len(has))
            n_qualify = len(ok)          # before any restriction, so the message stays truthful
            if prefer_nonempty and count_motifs:
                # a window with NO cofactor hit anywhere makes an empty panel, so the pair would
                # differ in motif PRESENCE as well as position. Restrict to windows carrying at
                # least one count-set hit when any qualify.
                any_hit = _windows_with_central_hit(z, jaspar_json, count_motifs, central_bp,
                                                    threshold, whole_window=True,
                                                    pfm_pseudo=pfm_pseudo)
                ok2 = ok[any_hit[ok]]
                if len(ok2):
                    print(f"[select]   restricted to {len(ok2)}/{len(ok)} with >=1 cofactor hit "
                          "somewhere in the window (--prefer_nonempty)")
                    ok = ok2
                else:
                    print("[select]   --prefer_nonempty: no qualifying window has a cofactor hit; "
                          "keeping the full set")
            w = int(ok[np.argmax(tr[ok])])                      # deepest qualifying window
            print(f"[select] {kind}{tf}: {n_qualify}/{len(has)} windows qualify "
                  f"({has.mean():.1%} have a central hit)"
                  + (f", {len(ok)} after --prefer_nonempty" if len(ok) != n_qualify else "")
                  + f" -> picked idx={w} (highest total_reads among them)")
            return w
    raise SystemExit("--select must be top_depth | index:N | chr:ANCHOR | "
                     f"motif_at_center:TF | no_motif_at_center:TF, got {select!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--jaspar", required=True)
    ap.add_argument("--motifs", default="ALL", help="motifs shown as LANES: comma-sep JASPAR TF-name substrings / matrix ids, or ALL")
    ap.add_argument("--count_motifs", default=None,
                    help="motifs used for the gray COUNT track (known cofactors); comma-sep substrings/ids. "
                         "Default = same as --motifs. ALL is nearly flat (~3 hits/base genome-wide), so a "
                         "curated cofactor set makes the count track carry signal.")
    ap.add_argument("--select", default="top_depth",
                    help="top_depth | index:N | chr:ANCHOR | motif_at_center:TF | "
                         "no_motif_at_center:TF. The motif_* forms pick the DEEPEST window that "
                         "does (or does not) carry a TF hit overlapping the central +/--central_bp, "
                         "i.e. an instance of the population the aggregate enrichment test measures. "
                         "A figure so chosen is a declared example, not a random one -- say so in the caption.")
    ap.add_argument("--group", action="append", default=None, metavar="LABEL:TF1,TF2",
                    help="repeatable form of --lane_groups, one group per flag: "
                         "--group insulator:CTCF --group E-box:ZEB1,SNAI1. PREFER THIS: it needs no "
                         "semicolons, which the shell mangles (a ';' inside a quoted variable has "
                         "twice silently arrived here as an unrelated number).")
    ap.add_argument("--lane_groups", default=None,
                    help="colour lanes by COFACTOR CLASS / protein family instead of per motif: "
                         "'label:TF1,TF2;label2:TF3,TF4'. One hue per group, members drawn "
                         "contiguously, motifs in no group drawn gray as 'other', and a legend "
                         "replaces the colorbar. Implies --lane_color cofactor.")
    ap.add_argument("--fixed_lanes", action="store_true",
                    help="draw ONE LANE PER --count_motifs entry, in the order given, whether or "
                         "not it hits this window (instead of only motifs with hits, ranked by "
                         "significance). Makes the row set identical across loci so two panels are "
                         "comparable, and an empty row reads as 'this cofactor is absent here'.")
    ap.add_argument("--exact_motifs", action="store_true",
                    help="match --motifs/--count_motifs tokens as WHOLE names instead of substrings, "
                         "so 'CTCF' does not also pull CTCFL (nor 'SP1' -> SP110). Use this to make "
                         "the lanes/count set identical to fig_as_motif_tracks' --tfs.")
    ap.add_argument("--prefer_nonempty", action="store_true",
                    help="among windows the motif_* selector accepts, prefer those carrying at "
                         "least one --count_motifs hit ANYWHERE in the window, so an AS/non-AS pair "
                         "contrasts motif POSITION rather than bottoming out at an empty panel")
    ap.add_argument("--central_bp", type=int, default=10,
                    help="half-width of the central band used by the motif_* selectors "
                         "(default 10 = what the aggregate central-enrichment test uses)")
    ap.add_argument("--sal_ylim", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="fix the saliency (right) axis, so a pair of loci share one scale; "
                         "without it each panel autoscales and the two are not visually comparable")
    ap.add_argument("--sal_key", default="importance")
    ap.add_argument("--nms_overlap", type=float, default=0.0,
                    help="collapse same-motif hits overlapping by more than this fraction of "
                         "the shorter one, keeping the best-scoring. 1.0 disables it. "
                         "DEFAULT 0 = no overlap at all, which is the only value that bounds "
                         "a drawn bar to the matrix width: fimo reports every passing offset, "
                         "and at any fraction above 0 a chain of hits each overlapping its "
                         "neighbour by LESS than the fraction survives intact and draws as one "
                         "continuous bar. Measured on a chain of 19 bp hits every 8-10 bp: "
                         "1.0 -> 91 bp drawn, 0.5 -> 83, 0.25 -> 83, 0.0 -> 19.")
    ap.add_argument("--fimo_threshold", type=float, default=1e-4)
    ap.add_argument("--max_rows", type=int, default=25, help="lane rows (top motifs by significance)")
    ap.add_argument("--lane_select", choices=["significance", "ism_peak"], default="significance",
                    help="which motifs get a lane when the roster is not fixed. significance: "
                         "best FIMO score (the original behaviour). ism_peak: motifs whose hits "
                         "sit on ISM saliency, ranked by mean |saliency| over the best hit. "
                         "ism_peak SEARCHES for cofactors at the saliency peaks -- it selects on "
                         "the same quantity the top panel draws, so the visible alignment is "
                         "guaranteed by construction and is not evidence. Read the printed p "
                         "against the shift null, not the picture.")
    ap.add_argument("--peak_quantile", type=float, default=0.75,
                    help="--lane_select ism_peak: |saliency| quantile defining a peak, used for "
                         "the reported peak_frac (ranking uses mean |saliency|, not this).")
    ap.add_argument("--lane_null", type=int, default=200,
                    help="--lane_select ism_peak: circular shifts of the saliency track for the "
                         "null p. 0 disables it, which also removes the only thing that makes "
                         "this selection interpretable.")
    ap.add_argument("--lane_max_p", type=float, default=None,
                    help="--lane_select ism_peak: drop lanes whose shift-null p exceeds this. "
                         "Unset keeps every lane and lets the printed table speak.")
    ap.add_argument("--count_track", action="store_true",
                    help="draw the gray per-position motif-count track behind the saliency "
                         "curve. Off by default: with cofactor lanes directly below, the count "
                         "is the same information summed, and it competes with the saliency "
                         "curve for the panel. The overlap statistics are printed either way.")
    ap.add_argument("--smooth", type=float, default=2.0)
    ap.add_argument("--lane_lw", type=float, default=1.8, help="motif-hit line thickness in the lanes")
    ap.add_argument("--row_height", type=float, default=0.12,
                    help="lanes-panel height per row (figure-height units); lower = more condensed")
    ap.add_argument("--lane_fontsize", type=float, default=7.5, help="motif-name label size in the lanes")
    ap.add_argument("--pfm_pseudo", type=float, default=0.0,
                    help="pseudocount added to each PFM cell before renormalising. 0 (default) "
                         "matches fig_as_motif_tracks, which uses the raw matrix; the former 0.1 "
                         "flattened the PWM and inflated hit counts ~1.8x, so the two scripts "
                         "disagreed (22.6%% vs 12.6%% of AS windows with a central CTCF hit)")
    ap.add_argument("--lane_color", choices=["pvalue", "cofactor", "track"], default="pvalue",
                    help="'pvalue' = viridis by -log10 p (colorbar); 'cofactor' = one categorical "
                         "colour per motif in the COUNT set (the cofactors feeding the gray track), "
                         "other lanes gray, and the y-tick labels tinted to match so no legend "
                         "is needed; 'track' = every lane in the saliency curve's colour, with "
                         "no colorbar and no legend. Use 'track' when the lane roster is "
                         "explicit and short: colour then carries no information, and encoding "
                         "nothing is better than encoding a p-value nobody reads off a strip.")
    ap.add_argument("--zoom_bp", type=int, default=0,
                    help="crop both panels to +/- this many bp around the variant (0 = full window; "
                         "use 75 to match the target layout)")
    ap.add_argument("--sal_label", default="ISM saliency", help="right-axis label for the blue track")
    ap.add_argument("--panel_label", default="", help="optional panel letter drawn top-left (e.g. B)")
    ap.add_argument("--title", default="",
                    help="top title line (e.g. 'Ind. 1, ATACseq'); the chr:start-end line is appended")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.group:                      # repeatable flags -> the same ';'-joined string
        joined = ";".join(a.group)
        if a.lane_groups and a.lane_groups != joined:
            raise SystemExit("pass --group OR --lane_groups, not both")
        a.lane_groups = joined
    if a.lane_groups:
        a.lane_color = "cofactor"   # grouping is a colour scheme, not a p-value scale

    z = np.load(a.npz, allow_pickle=True)
    w = pick_window(z, a.select, a.jaspar, a.fimo_threshold, a.central_bp,
                    a.prefer_nonempty, a.count_motifs, a.pfm_pseudo)
    seq = str(z["seqs"][w])
    sal = z[a.sal_key][w].astype(np.float64)                 # (L,)
    L = len(seq); center = L // 2; x = np.arange(L) - center
    anchor = int(z["anchor"][w]) if "anchor" in z else -1
    chrom = str(z["chr"][w]) if "chr" in z else "?"
    gstart, gend = anchor - center, anchor - center + L      # genomic window span
    print(f"[window] idx={w} {chrom}:{anchor} L={L} span={gstart}-{gend}")

    lane_pwms = select_pwms(a.jaspar, a.motifs, pseudo=a.pfm_pseudo, exact=a.exact_motifs)
    if not lane_pwms:
        raise SystemExit(f"--motifs {a.motifs!r} matched no JASPAR entries")
    count_pwms = select_pwms(a.jaspar, a.count_motifs, pseudo=a.pfm_pseudo, exact=a.exact_motifs) if a.count_motifs else lane_pwms
    if a.count_motifs and not count_pwms:
        raise SystemExit(f"--count_motifs {a.count_motifs!r} matched no JASPAR entries")
    lane_names = {nm for _, nm in lane_pwms.values()}
    count_names = {nm for _, nm in count_pwms.values()}
    hits, plabel = scan(seq, {**count_pwms, **lane_pwms}, a.fimo_threshold)   # one fimo pass over the union
    hits, n_sup = suppress_overlaps(hits, a.nms_overlap)
    if n_sup:
        _w = {}
        for nm, s, e, _v in hits:
            _w[nm] = max(_w.get(nm, 0), e - s)
        # The DRAWN extent, not the widest single hit: a chain of short hits is what reads as
        # a long footprint, and reporting max(e-s) would have said 19bp while the figure drew 83.
        _run = {}
        for nm in _w:
            iv = sorted((s, e) for n_, s, e, _ in hits if n_ == nm)
            cur, longest = list(iv[0]), 0
            for s, e in iv[1:]:
                if s <= cur[1]:
                    cur[1] = max(cur[1], e)
                else:
                    longest = max(longest, cur[1] - cur[0]); cur = [s, e]
            _run[nm] = max(longest, cur[1] - cur[0])
        print(f"[nms] suppressed {n_sup} overlapping hit(s) at --nms_overlap {a.nms_overlap}; "
              f"longest DRAWN extent per motif (matrix width in brackets): "
              + ", ".join(f"{nm} {_run[nm]}bp [{_w[nm]}]"
                          for nm in sorted(_run, key=lambda k: -_run[k])[:6]))
    for tag, arg, pw in (("--motifs", a.motifs, lane_pwms),
                         ("--count_motifs", a.count_motifs, count_pwms)):
        if arg and arg.upper() != "ALL":
            ntok = len([t for t in arg.split(",") if t.strip()])
            if len(pw) > ntok:                 # substring matching pulled in extra matrices
                print(f"[motifs] {tag} {ntok} token(s) -> {len(pw)} matrices (substring match): "
                      + ", ".join(sorted({nm for _, nm in pw.values()})))
    print(f"[scan] lanes={len(lane_pwms)} count-motifs={len(count_pwms)} -> {len(hits)} hits "
          f"(thr p<{a.fimo_threshold})")

    # per-position motif COUNT track + covered mask -- built ONLY from the count-motif (cofactor) set
    count = np.zeros(L); covered = np.zeros(L, dtype=bool)
    for name, s, e, _ in hits:
        if name in count_names:
            e = min(e, L); count[s:e] += 1.0; covered[s:e] = True
    if not covered.any():
        _present = sorted({nm for nm, _, _, _ in hits if nm in lane_names})[:12]
        print("[warn] the count-motif set matched 0 hits in this window -> gray count track is EMPTY. "
              + (f"Motifs that DO occur here: {', '.join(_present)}" if _present
                 else "No --motifs matched anywhere in this window either."))

    # ---- OVERLAP metrics (printed only) --------------------------------------------------------
    absal = np.abs(sal)
    rho = spearmanr(sal, count).correlation if count.any() else np.nan
    enr = (absal[covered].mean() / absal[~covered].mean()
           if covered.any() and (~covered).any() and absal[~covered].mean() > 0 else np.nan)
    mass = absal[covered].sum() / absal.sum() if absal.sum() > 0 else np.nan
    print(f"[overlap] spearman(sal,count)={rho:.3f} | enrichment(in/out)={enr:.3f} | "
          f"saliency-mass-in-motifs={mass:.3f} | covered={covered.mean():.2f}")

    # ---- figure: top (count + saliency), bottom motif-hit lanes --------------------------------
    by_motif = {}
    for name, s, e, val in hits:
        if name in lane_names:
            by_motif.setdefault(name, []).append((s, e, val))
    if a.zoom_bp > 0:                                        # drop motifs whose hits all fall outside the crop
        by_motif = {m: hs for m, hs in by_motif.items()
                    if any(x[s] <= a.zoom_bp and x[min(e, L) - 1] >= -a.zoom_bp for s, e, _ in hs)}
    groups = parse_lane_groups(a.lane_groups, count_pwms, a.exact_motifs) if a.lane_groups else []
    grp_of = {nm: gi for gi, (_, mem) in enumerate(groups) for nm in mem}
    if a.fixed_lanes:
        if not a.count_motifs:
            raise SystemExit("--fixed_lanes needs --count_motifs (it is the lane roster)")
        roster = cofactor_lane_order(a.count_motifs, count_pwms, a.exact_motifs)
        if groups:            # group order drives lane order so same-class lanes sit together
            order = [nm for _, mem in groups for nm in mem if nm in roster]
            order += [nm for nm in roster if nm not in grp_of]
        else:
            order = roster
        n_empty = sum(1 for m in order if m not in by_motif)
        print(f"[lanes] fixed roster of {len(order)}: {', '.join(order)}"
              + (f"  ({n_empty} with no hit in this window)" if n_empty else ""))
    elif a.lane_select == "ism_peak":
        order, lane_table = rank_by_ism_peak(by_motif, sal, a.peak_quantile, a.lane_null,
                                             20, a.smooth)
        if a.lane_max_p is not None:
            keep = {nm for nm, _s, p, _n, _f in lane_table if not (p > a.lane_max_p)}
            dropped = [nm for nm in order if nm not in keep]
            order = [nm for nm in order if nm in keep]
            if dropped:
                print(f"[lanes] --lane_max_p {a.lane_max_p}: dropped {len(dropped)} "
                      f"({', '.join(dropped[:8])}{' ...' if len(dropped) > 8 else ''})")
        order = order[:a.max_rows]
        print(f"  {'motif':18}{'mean|sal|':>11}{'p_shift':>9}{'hits':>6}{'peak_frac':>11}")
        for nm, sc, p, nh, pf in lane_table[:a.max_rows]:
            print(f"  {nm:18}{sc:>11.4f}{p:>9.3f}{nh:>6}{pf:>11.2f}")
        if a.lane_null:
            n_sig = sum(1 for _n, _s, p, _h, _f in lane_table if p <= 0.05)
            print(f"[lanes] {n_sig}/{len(lane_table)} motifs beat the shift null at p<=0.05 "
                  f"({0.05 * len(lane_table):.1f} expected by chance at this many tests)")
    else:
        order = sorted(by_motif, key=lambda m: -max(v for _, _, v in by_motif[m]))[:a.max_rows]

    # lanes-panel height follows the row count: a fixed 1.7 leaves a 4-lane panel mostly blank
    h_lanes = float(np.clip(a.row_height * max(len(order), 1), 0.42, 1.7))
    # a class legend needs real width in the right column; a colorbar only needs a strip
    w_right = 0.17 if a.lane_groups else 0.02
    fig = plt.figure(figsize=(8.8 * (1.0 + w_right - 0.02), 2.1 + 2.0 * h_lanes))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, w_right], height_ratios=[1.0, h_lanes],
                          hspace=0.18, wspace=0.02)
    axT = fig.add_subplot(gs[0, 0])                          # col 0 -> axT and axH stay x-aligned
    axH = fig.add_subplot(gs[1, 0], sharex=axT)
    cax = fig.add_subplot(gs[1, 1])                          # colorbar beside the lanes only

    # TOP PANEL. With --count_track the old two-axis form is kept (gray count left, saliency
    # right on a twinx). Without it -- the default -- saliency owns the LEFT axis and there is
    # no twin: a lone curve on a right-hand axis reads as the secondary series in a plot whose
    # primary series is missing, and the empty left axis still consumes label width.
    sal_curve = gaussian_filter1d(sal, a.smooth) if a.smooth else sal
    if a.count_track:
        axT.fill_between(x, gaussian_filter1d(count, a.smooth) if a.smooth else count,
                         color="0.75", linewidth=0, zorder=1)
        axT.set_ylabel("Motif Count", color="0.4"); axT.tick_params(axis="y", colors="0.4")
        axT.set_ylim(bottom=0)                                # counts are non-negative
        if not count.any():
            # all-zero counts autoscale to a meaningless fractional range (0.00-0.05 ticks on
            # an integer "Motif Count" axis); pin it so the empty track reads as empty
            axT.set_ylim(0, 1)
        axT.margins(x=0.02)
        axS = axT.twinx(); axS.spines["top"].set_visible(False)
        axS.plot(x, sal_curve, color=C_SAL, zorder=3)
        axS.set_ylabel(a.sal_label, color=C_SAL); axS.tick_params(axis="y", colors=C_SAL)
    else:
        axT.margins(x=0.02)
        axT.plot(x, sal_curve, color=C_SAL, zorder=3)
        axT.set_ylabel(a.sal_label, color=C_SAL); axT.tick_params(axis="y", colors=C_SAL)
        axT.spines["right"].set_visible(False); axT.spines["top"].set_visible(False)
        axS = axT
    if a.sal_ylim:
        axS.set_ylim(*a.sal_ylim)
    for ax in {id(axT): axT, id(axS): axS}.values():
        ax.axvline(0, color="0.4", ls=":", lw=0.8, zorder=2)
    title = ((a.title.strip() + "\n") if a.title.strip() else "") + f"{chrom}: {gstart}-{gend}"
    axT.set_title(title)

    # bottom: motif-hit lanes -- one row per motif (name at left), bars colored by -log10 p (no colorbar)
    vals = [v for m in order for _, _, v in by_motif.get(m, [])]
    norm = Normalize(min(vals), max(vals)) if vals else Normalize(0, 1)
    cmap = plt.get_cmap("viridis")
    if a.lane_color == "cofactor":
        # categorical colour per COUNT-set motif (the cofactors that build the gray track);
        # lanes outside that set stay gray so incidental hits recede.
        # tab10 for <=10: tab20 pairs adjacent entries as dark/light of ONE hue, which makes
        # consecutive entries read as related and the light tints are faint as tick labels.
        gpal = plt.get_cmap("tab10")
        if groups:
            lane_col = {nm: gpal(gi % 10) for nm, gi in grp_of.items()}   # one hue per CLASS
            if len(groups) > 10:
                print(f"[warn] {len(groups)} groups but the palette has 10 colours -- hues repeat")
        else:
            cof = [m for m in order if m in count_names]
            n_pal = 10 if len(cof) <= 10 else 20
            pal = plt.get_cmap(f"tab{n_pal}")
            if len(cof) > n_pal:
                print(f"[warn] {len(cof)} cofactor lanes but the palette has {n_pal} colours -- "
                      "colours repeat; narrow --count_motifs or lower --max_rows")
            lane_col = {m: pal(i % n_pal) for i, m in enumerate(cof)}
    else:
        lane_col = {}
    for row, name in enumerate(order):
        yv = len(order) - 1 - row
        axH.hlines(yv, x[0], x[-1], color="0.93", lw=0.8, zorder=0)   # lane guide
        for s, e, val in by_motif.get(name, []):
            col = (C_SAL if a.lane_color == "track" else
               lane_col.get(name, "0.62") if a.lane_color == "cofactor" else cmap(norm(val)))
            axH.hlines(yv, x[s], x[min(e, L) - 1], color=col, lw=a.lane_lw)
    axH.set_yticks(range(len(order)))
    axH.set_yticklabels(list(reversed(order)), fontsize=a.lane_fontsize)
    if a.lane_color == "track":
        for lab in axH.get_yticklabels():
            lab.set_color(C_SAL)
    if a.lane_color == "cofactor":
        # tint each tick label to its lane colour -- the axis IS the key, so no legend/colorbar
        for t, nm in zip(axH.get_yticklabels(), list(reversed(order))):
            t.set_color(lane_col.get(nm, "0.45"))
    axH.set_ylim(-0.6, len(order) - 0.4)
    axH.axvline(0, color="0.4", ls=":", lw=0.8)
    axH.set_xlabel("position relative to variant (bp)")
    axH.margins(x=0.02)
    axH.tick_params(axis="y", length=0)
    if a.lane_color == "track":
        cax.axis("off")                       # nothing to key: every lane is the same colour
    elif a.lane_color == "cofactor":
        cax.axis("off")                       # no colorbar: colour encodes identity, not a scale
        if groups:
            from matplotlib.lines import Line2D
            handles = [Line2D([0], [0], color=gpal(gi % 10), lw=3, label=lab)
                       for gi, (lab, _) in enumerate(groups)]
            if any(nm not in grp_of for nm in order):
                handles.append(Line2D([0], [0], color="0.62", lw=3, label="other"))
            # anchor at cax's own left edge: a negative offset pushes the legend back INTO the
            # lanes panel and collides with the lane guides/labels
            cax.legend(handles=handles, loc="center left", bbox_to_anchor=(0.0, 0.5),
                       frameon=False, fontsize=7, handlelength=1.3, borderaxespad=0,
                       title="cofactor class", title_fontsize=7.5)
    else:
        sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
        cb = fig.colorbar(sm, cax=cax); cb.set_label(f"motif hit {plabel}", fontsize=7)
        cb.ax.tick_params(labelsize=6)

    if a.zoom_bp > 0:
        axT.set_xlim(-a.zoom_bp, a.zoom_bp)                  # sharex -> crops both panels
    if a.panel_label:
        fig.text(0.005, 0.99, a.panel_label, fontsize=16, fontweight="bold", va="top", ha="left")

    fig.savefig(a.out, bbox_inches="tight")
    print(f"[fig] wrote {a.out}  (lanes colored by "
          + ("the saliency track colour -- colour carries no information here"
             if a.lane_color == "track" else
             "cofactor class; only lanes are in the count set" if a.lane_color == "cofactor"
             else f"{plabel}")
          + f"; window {'full' if not a.zoom_bp else f'+/-{a.zoom_bp} bp'})")


if __name__ == "__main__":
    main()
