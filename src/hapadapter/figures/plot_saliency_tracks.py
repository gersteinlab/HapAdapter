#!/usr/bin/env python
"""
Multi-track aggregate ISM-saliency figure with JASPAR cofactor-density overlay.

Generalizes plot_ism_aggregate_density.py (single track) to overlay several .npz tracks on one
axis, so it covers every Wave-2 figure:

  Fig C (trunk)      : one track   -> ism_trunk.npz
  Fig G-head (AS/nonAS): two tracks -> ism_head_<arm>_<set>_AS_<base>.npz + ..._nonAS_...  (per-group
                        motif density: AS and non-AS are different loci, so each track's density is
                        computed from its own windows)

Each track npz carries importance (N,L) and onehot (N,L,4), as ism_saliency.py writes them.

--shared_windows plots ONE motif-density curve instead of one per track. It is correct only
when the tracks are the SAME loci scored differently; every current caller compares DIFFERENT
locus sets (AS vs non-AS, peak vs background), where each track needs its own density. Passing
it for those would draw the first track's density under both curves -- which looks right and
is wrong. No figure in the repo uses it today.

FIMO (memelite) runs LOCALLY; local numba compiles it fine (the cluster's older numba is the one
that needs the sed patch). Set NUMBA_CACHE_DIR before import.
"""
import os
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)
import argparse, json, numpy as np
import matplotlib.pyplot as plt

BASES = "ACGT"


def load_motifs(path, tfs):
    """{CANONICAL_UPPER_NAME: {matrix_id: (4,W)}} -- keep EVERY JASPAR matrix for each requested TF
    (JUN/Jun, both FOS, both HNF4A ...) grouped under one canonical name, so density unions all of
    a TF's matrices into a single curve instead of silently dropping all but one (dict collision)
    or double-listing case variants (JUN vs 'Jun')."""
    db = json.load(open(path)); want = {t.upper() for t in tfs}; out = {}
    for k, v in db.items():
        nm = v.get("name", k)
        up = nm.upper()
        if up in want:
            out.setdefault(up, {})[k] = np.array([v["pfm"][b] for b in BASES], dtype=np.float64)
    return out


def smooth(x, w):
    if w <= 1:
        return x
    return np.convolve(x, np.ones(w) / w, mode="same")


def motif_density(onehot, motifs, fimo, threshold, smooth_w):
    """Per-position fraction of windows with a motif hit, per TF. motifs = {TF: {matrix_id:(4,W)}};
    all of a TF's matrices are unioned into one curve. A position counts once per window even if
    several matrices of the same TF hit it (avoids double-counting overlapping JASPAR variants)."""
    N, L, _ = onehot.shape
    seqs = np.transpose(onehot, (0, 2, 1)).astype(np.float32)     # (N,4,L)
    flat = {}                          # matrix_id -> pfm, plus map matrix_id -> TF
    mid2tf = {}
    for tf, mats in motifs.items():
        for mid, pfm in mats.items():
            flat[mid] = pfm; mid2tf[mid] = tf
    res = fimo(flat, seqs, threshold=threshold)
    # per TF, per window: boolean covered[pos]; then sum across windows so each window counts once
    cover = {tf: np.zeros(L) for tf in motifs}
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
            s, e = int(row["start"]), int(min(row["end"], L))
            per_win[tf][w, s:e] = True
    for tf in motifs:
        cover[tf] = per_win[tf].sum(0).astype(float)
    return {tf: smooth(cover[tf] / N, smooth_w) for tf in motifs}


def draw_saliency_tracks(ax, tracks, motifs=None, *, shared_windows=False,
                         fimo_threshold=1e-4, smooth_w=11, band="iqr", normalize=False,
                         mode="ism", title=None, fontsize=None, legend=True):
    """
    Draw the saliency tracks (and optional cofactor-density overlay) onto an EXISTING axis.

    Split out of main() so the composite figure and the standalone CLI render through the same
    code. The drawing is unchanged from the single-figure version; only the axis it targets and
    the font sizes are now parameters.

    `tracks` is [(label, importance (N,L), onehot (N,L,4)), ...] as load_tracks returns.
    Returns (profs, ax2) where ax2 is the twin density axis, or None when no motifs were given.
    """
    fs = fontsize
    L = tracks[0][1].shape[1]
    x = np.arange(L) - L // 2
    tcmap = plt.get_cmap("viridis")
    single = len(tracks) == 1

    profs = {}
    for lab, imp, _oh in tracks:
        profs[lab] = smooth(imp.mean(0), smooth_w)
    norm_by = max((p.max() or 1.0) for p in profs.values()) if normalize else 1.0

    # Fixed high-contrast colors for the common AS / non-AS pair so the saliency tracks POP
    # against the (many, noisy) cofactor-density curves; fall back to viridis for other labels.
    FIXED = {"AS": "#d1495b", "nonAS": "#3d6cb3", "non-AS": "#3d6cb3",
             "peak": "#d1495b", "background": "#3d6cb3"}
    for i, (lab, imp, _oh) in enumerate(tracks):
        prof = profs[lab] / norm_by
        col = FIXED.get(lab, tcmap(0.12 + 0.72 * i / max(1, len(tracks) - 1))
                        if len(tracks) > 1 else "#4b0082")
        lw = 3.4 if i == 0 else 3.0        # thick: saliency is the message, density is context
        ax.plot(x, prof, color=col, lw=lw, label=f"{lab} saliency", zorder=10 - i,
                solid_capstyle="round")
        if single and band != "none":
            if band == "iqr":
                lo, hi = np.percentile(imp, [25, 75], axis=0)
            else:
                sd = imp.std(0); lo, hi = imp.mean(0) - sd, imp.mean(0) + sd
            ax.fill_between(x, smooth(lo, smooth_w) / norm_by, smooth(hi, smooth_w) / norm_by,
                            color="#7030a0", alpha=0.16, lw=0)

    # ---- motif density overlay (SECONDARY y-axis: density is a fraction in [0,1], a different
    #      quantity from saliency units, so it gets its own axis instead of sharing the left one)
    ax2 = None
    if motifs:
        from memelite import fimo
        cmap = plt.get_cmap("tab10")
        ax2 = ax.twinx()
        if shared_windows:
            dens = motif_density(tracks[0][2], motifs, fimo, fimo_threshold, smooth_w)
            for j, nm in enumerate(dens):
                ax2.plot(x, dens[nm], color=cmap(j), lw=1.0, ls="-", alpha=0.55, zorder=3,
                         label=f"{nm} density")
        else:
            for i, (lab, _imp, oh) in enumerate(tracks):
                dens = motif_density(oh, motifs, fimo, fimo_threshold, smooth_w)
                for j, nm in enumerate(dens):
                    ax2.plot(x, dens[nm], color=cmap(j),
                             lw=0.9, ls="--" if i else "-", alpha=0.5, zorder=3,
                             label=f"{nm} density ({lab})")
        ax2.set_ylabel("motif-hit density (fraction of windows)",
                       **({} if fs is None else {"fontsize": fs}))
        ax2.set_ylim(bottom=0)
        ax2.margins(x=0)
        if fs is not None:
            ax2.tick_params(labelsize=fs - 1)

    ax.axvline(0, color="red", ls="--", lw=1, zorder=1)
    ax.set_xlabel("position relative to center (bp)", **({} if fs is None else {"fontsize": fs}))
    ax.set_ylabel(("normalized " if normalize else "") + "ISM saliency (Δ score / substitution)",
                  **({} if fs is None else {"fontsize": fs}))
    ax.set_title(title or "aggregate saliency vs cofactor motif density",
                 fontsize=(11 if fs is None else fs + 1))
    if legend:
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = (ax2.get_legend_handles_labels() if ax2 is not None else ([], []))
        ax.legend(h1 + h2, l1 + l2, fontsize=(7.5 if fs is None else fs - 1.5),
                  ncol=1, loc="upper right", framealpha=0.9)
    ax.margins(x=0)
    if fs is not None:
        ax.tick_params(labelsize=fs - 1)
    return profs, ax2


def load_tracks(pairs):
    """[(label, path), ...] -> [(label, importance, onehot), ...]; all must share one length."""
    tracks, L = [], None
    for lab, path in pairs:
        d = np.load(path, allow_pickle=True)
        imp, onehot = d["importance"], d["onehot"]
        if L is None:
            L = imp.shape[1]
        elif imp.shape[1] != L:
            raise SystemExit(f"track {lab}: L={imp.shape[1]} != {L}")
        tracks.append((lab, imp, onehot))
    return tracks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", nargs="+", required=True,
                    help="LABEL=path.npz entries (1+). First track is drawn boldest.")
    ap.add_argument("--jaspar", default=None, help="omit to skip motif overlay entirely")
    ap.add_argument("--tfs", default="", help="comma-separated TF names (need JASPAR motifs)")
    ap.add_argument("--shared_windows", action="store_true",
                    help="tracks share the same loci -> compute ONE density curve. Correct "
                         "only when the tracks are the same loci scored differently; for "
                         "different locus sets (AS vs non-AS, peak vs background) leave it "
                         "off so each track gets its own density.")
    ap.add_argument("--fimo_threshold", type=float, default=1e-4)
    ap.add_argument("--smooth", type=int, default=11)
    ap.add_argument("--band", choices=["iqr", "std", "none"], default="iqr",
                    help="variability band; drawn only when a SINGLE track is plotted")
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    pairs = []
    for t in a.tracks:
        if "=" not in t:
            raise SystemExit(f"--tracks entry must be LABEL=path.npz, got {t!r}")
        lab, path = t.split("=", 1)
        pairs.append((lab, path))

    tracks = load_tracks(pairs)
    # L was previously a by-product of main()'s inline loading loop; load_tracks owns that now,
    # so take it from the data. The summary print below is the only other user.
    L = tracks[0][1].shape[1]

    motifs = {}
    if a.jaspar and a.tfs.strip():
        from memelite import fimo
        motifs = load_motifs(a.jaspar, [t.strip() for t in a.tfs.split(",")])

    fig, ax = plt.subplots(figsize=(13, 4.4))
    profs, _ax2 = draw_saliency_tracks(
        ax, tracks, motifs, shared_windows=a.shared_windows,
        fimo_threshold=a.fimo_threshold, smooth_w=a.smooth, band=a.band,
        normalize=a.normalize, title=a.title)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150, bbox_inches="tight")
    for lab in profs:
        p = profs[lab]
        print(f"[track] {lab}: peak {p.max():.4f} at {int(np.argmax(p))-L//2:+d} bp")
    print(f"[out] {a.out}")


if __name__ == "__main__":
    main()