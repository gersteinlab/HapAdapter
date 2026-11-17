#!/usr/bin/env python
"""
Motif-enrichment figure around CTCF ASB SNVs, in the style of "Motif enrichments surrounding
CTCF AS SNVs" (CTCF, SP1 panels). For each SNV, take a +/-flank window of hg38 REFERENCE sequence
centered on the variant, FIMO-scan the requested motifs, and plot the per-position hit density
for allele-specific (AS, label=1) vs non-allele-specific (non-AS, label=0) SNVs.

This is a pure SEQUENCE analysis (no model): it asks whether AS variants sit in stronger motif
context than non-AS variants. Groups are balanced (equal N) so the density curves are comparable.

Inputs are two arrays of equal-length sequences (AS, non-AS), one motif set. Local; needs
NUMBA_CACHE_DIR set before import.
"""
import os
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)
import argparse, json, numpy as np
import matplotlib.pyplot as plt

BASES = "ACGT"
IDX = {b: i for i, b in enumerate(BASES)}


def load_motifs(path, tfs):
    """
    {CANONICAL_UPPER_NAME: {matrix_id: (4,W)}} -- EVERY JASPAR matrix for each requested TF,
    grouped under one canonical name.

    This replaces a flat {name: (4,W)} dict that kept only the LAST matrix of each name. CTCF
    has three matrices in the current JASPAR file (MA0139.2, MA1929.2, MA1930.2), so the flat
    form silently scanned with one arbitrary matrix and split case variants (JUN vs 'Jun') into
    separate curves. The format now matches plot_saliency_tracks.load_motifs so a composite
    figure's panels mean the same thing by "CTCF density".
    """
    db = json.load(open(path)); want = {t.upper() for t in tfs}; out = {}
    for k, v in db.items():
        nm = v.get("name", k)
        up = nm.upper()
        if up in want:
            out.setdefault(up, {})[k] = np.array([v["pfm"][b] for b in BASES], dtype=np.float64)
    return out


def onehot_stack(seqs):
    """List of equal-length ACGT strings -> (N,4,L); non-ACGT bases (N) become all-zero columns."""
    L = len(seqs[0]); N = len(seqs)
    oh = np.zeros((N, 4, L), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j, b in enumerate(s):
            k = IDX.get(b)
            if k is not None:
                oh[i, k, j] = 1.0
    return oh


def motif_hit_mask(seqs, motifs, fimo, threshold):
    """
    Per-WINDOW, per-position motif occupancy. -> {tf: (N, L) bool}

    This is the array compute_density was already building internally and then summing away.
    It is split out rather than recomputed because the logo's motif-presence filter and the
    density curves must agree about what "this window contains a GABPA site" means -- two
    independent FIMO calls with drifting thresholds or matrix unions would let a window be
    counted in one panel and excluded from the other with nothing to flag it.
    """
    L = len(seqs[0]); N = len(seqs)
    oh = onehot_stack(seqs)                                   # (N,4,L)
    flat, mid2tf = {}, {}
    for tf, mats in motifs.items():
        for mid, pfm in mats.items():
            flat[mid] = pfm; mid2tf[mid] = tf
    res = fimo(flat, oh, threshold=threshold)
    per_win = {tf: np.zeros((N, L), dtype=bool) for tf in motifs}
    for r in res:
        if len(r) == 0:
            continue
        mid = str(r["motif_name"].iloc[0])
        tf = mid2tf.get(mid)
        if tf is None:
            continue
        for _, row in r.iterrows():
            w = int(row["sequence_name"]) if "sequence_name" in row else int(row.get("seq_idx", 0))
            s0, e0 = int(row["start"]), int(min(row["end"], L))
            per_win[tf][w, s0:e0] = True
    return per_win


def compute_density(seqs, motifs, fimo, threshold):
    """
    Per-position fraction of WINDOWS with a motif hit, per TF. -> {tf: (L,)}

    motifs = {TF: {matrix_id: (4,W)}}; all of a TF's matrices are unioned into one curve, and a
    position counts ONCE PER WINDOW however many matrices hit it. Delegates the scan to
    motif_hit_mask so the logo's motif-presence filter cannot disagree with these curves.
    """
    per_win = motif_hit_mask(seqs, motifs, fimo, threshold)
    N = len(seqs)
    return {tf: per_win[tf].sum(0).astype(float) / N for tf in motifs}


def smooth(x, w):
    if w <= 1:
        return x
    return np.convolve(x, np.ones(w) / w, mode="same")


#: Group colours, in the order groups are passed. The first two reproduce the original
#: AS / non-AS pair exactly; the rest extend it for other contrasts (peak / background).
GROUP_COLORS = ["#1f4e9c", "#9db8e0", "#b2182b", "#ef9a9a"]
GROUP_LW = [1.8, 1.4, 1.8, 1.4]
#: When several motifs share ONE axis (the composite's single-panel form) the group stays the
#: colour and the motif becomes the line style. Colouring by motif instead would have made the
#: standalone per-motif figure change appearance, which is the thing this refactor must not do.
MOTIF_STYLES = ["-", "--", ":", "-."]


def normalise_density(dens_by_group, order, edge_frac=0.25, min_base=2e-3, verbose=True):
    """
    Rescale each motif's curves to FOLD OVER ITS OWN FLANK BASELINE. -> (dens, baselines, kept)

    WHY. Absolute motif density is dominated by how common the motif is, not by how enriched it
    is at the variant: in CTCF ENC-002 the CTCF curve peaks near 0.155 while SP1 and YY1 sit
    around 0.01, so on a shared axis the cofactors are flat lines against the bottom and any
    structure they have is invisible. Dividing by the motif's own background puts every curve in
    the same unit -- "how many times its usual rate" -- which is the quantity the panel is
    actually asking about.

    ONE BASELINE PER MOTIF, POOLED ACROSS GROUPS, computed from the outer `edge_frac` of the
    window on each side. Pooling matters: giving AS and non-AS their own baselines would divide
    out exactly the difference the panel exists to show, leaving two curves that look alike
    however far apart their absolute rates are. A shared denominator removes the cross-motif
    scale and keeps the cross-group comparison.

    A motif whose baseline is below `min_base` is LEFT UNNORMALISED and reported, not silently
    divided: a near-zero denominator turns a handful of chance hits into a huge fold value, and
    a cofactor that is simply absent would otherwise produce the most dramatic curve on the
    panel.
    """
    L = len(next(iter(dens_by_group[0][1].values())))
    k = max(1, int(round(L * edge_frac)))
    edge = np.r_[np.arange(k), np.arange(L - k, L)]
    out, base, kept = [], {}, []
    for nm in order:
        vals = [d[nm][edge] for _g, d in dens_by_group if nm in d]
        base[nm] = float(np.mean(np.concatenate(vals))) if vals else 0.0
    for glab, d in dens_by_group:
        nd = {}
        for nm, v in d.items():
            if nm in base and base[nm] >= min_base:
                nd[nm] = v / base[nm]
            else:
                nd[nm] = v
        out.append((glab, nd))
    kept = [nm for nm in order if base.get(nm, 0.0) >= min_base]
    if verbose:
        for nm in order:
            b = base.get(nm, 0.0)
            flagged = "" if b >= min_base else "   NOT NORMALISED (baseline below floor)"
            print(f"  [norm] {nm:10} flank baseline {b:.5f}{flagged}")
        if len(kept) != len(order):
            print("  [norm] mixing normalised and unnormalised curves on one axis is not "
                  "readable;\n         drop the flagged motif(s) from --tfs or use "
                  "--density_norm none.")
    return out, base, kept


def draw_motif_enrichment(ax, dens_by_group, order, flank, *, smooth_w=7, title=None,
                          fontsize=None, legend=True, ylabel="motif density",
                          xlabel="position relative to SNV (bp)"):
    """
    Draw per-position motif density onto an EXISTING axis.

    `dens_by_group` is [(group_label, {motif: (L,)}), ...] as compute_density returns, and
    `order` the motifs to draw. With one motif this reproduces the standalone figure's axis
    exactly; with several it puts them all on one axis, group = colour, motif = line style.

    Split out of main() so the composite figure and the standalone CLI render through the same
    code rather than two copies that can drift apart.
    """
    fs = fontsize
    L = len(next(iter(dens_by_group[0][1].values())))
    x = np.arange(L) - flank
    multi = len(order) > 1
    for gi, (glab, dens) in enumerate(dens_by_group):
        for mi, nm in enumerate(order):
            ax.plot(x, smooth(dens[nm], smooth_w),
                    color=GROUP_COLORS[gi % len(GROUP_COLORS)],
                    lw=GROUP_LW[gi % len(GROUP_LW)],
                    ls=MOTIF_STYLES[mi % len(MOTIF_STYLES)] if multi else "-",
                    label=f"{glab} {nm}" if multi else glab)
    ax.axvline(0, color="red", ls="--", lw=1)
    if title is not None:
        ax.set_title(title, fontsize=(11 if fs is None else fs + 1))
    if ylabel:
        ax.set_ylabel(ylabel, **({} if fs is None else {"fontsize": fs}))
    if xlabel:
        ax.set_xlabel(xlabel, **({} if fs is None else {"fontsize": fs}))
    ax.margins(x=0)
    if legend:
        ax.legend(fontsize=(8 if fs is None else fs - 1), loc="upper right", framealpha=0.9)
    if fs is not None:
        ax.tick_params(labelsize=fs - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--as_seqs", required=True, help=".npy array of AS window sequences")
    ap.add_argument("--nonas_seqs", required=True, help=".npy array of non-AS window sequences")
    ap.add_argument("--jaspar", required=True)
    ap.add_argument("--tfs", default="CTCF,SP1")
    ap.add_argument("--flank", type=int, default=75, help="half-window; SNV assumed at center")
    ap.add_argument("--fimo_threshold", type=float, default=1e-4)
    ap.add_argument("--smooth", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from memelite import fimo

    AS = [str(s) for s in np.load(a.as_seqs, allow_pickle=True)]
    NON = [str(s) for s in np.load(a.nonas_seqs, allow_pickle=True)]
    # keep only full-length windows (drop any short fetches) and equalize length
    L = 2 * a.flank + 1
    AS = [s for s in AS if len(s) == L]; NON = [s for s in NON if len(s) == L]
    tfs = [t.strip() for t in a.tfs.split(",")]
    motifs = load_motifs(a.jaspar, tfs)
    order = [nm for nm in motifs]                     # display names actually found
    print(f"[enrich] AS={len(AS)} nonAS={len(NON)} L={L}; motifs={order}")

    dAS = compute_density(AS, motifs, fimo, a.fimo_threshold)
    dNON = compute_density(NON, motifs, fimo, a.fimo_threshold)

    n = len(order)
    fig, axes = plt.subplots(n, 1, figsize=(5.2, 2.2 * n), sharex=True)
    if n == 1:
        axes = [axes]
    groups = [("AS", dAS), ("non-AS", dNON)]
    for i, (ax, nm) in enumerate(zip(axes, order)):
        draw_motif_enrichment(ax, groups, [nm], a.flank, smooth_w=a.smooth, title=nm,
                              xlabel=("position relative to SNV (bp)" if i == n - 1 else ""))
    fig.suptitle("Motif enrichments surrounding CTCF AS SNVs", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(a.out, dpi=150, bbox_inches="tight")
    print(f"[out] {a.out}")
    for nm in order:
        c = a.flank
        print(f"   {nm:6s} center density AS={smooth(dAS[nm],a.smooth)[c]:.4f} "
              f"non-AS={smooth(dNON[nm],a.smooth)[c]:.4f}")


if __name__ == "__main__":
    main()
