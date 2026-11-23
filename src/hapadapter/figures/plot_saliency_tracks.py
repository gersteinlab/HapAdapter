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


def _axis_for(vidx, L, align):
    """x coordinates for one track: bp relative to the window centre, or to its variant."""
    return np.arange(L) - (L // 2 if align == "center" else vidx)


def draw_saliency_tracks(ax, tracks, motifs=None, *, shared_windows=False,
                         fimo_threshold=1e-4, smooth_w=11, band="iqr", normalize=False,
                         mode="ism", title=None, fontsize=None, legend=True,
                         align="center", mark_variants=False):
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
    # PER-TRACK x. With align="center" every track shares one axis and this is the old
    # behaviour exactly. With align="variant" each track is shifted so its own variant sits
    # at 0, which is the view that answers "does the peak travel with the variant": tracking
    # peaks superimpose, a peak pinned to the window centre fans out by the offsets instead.
    xs = {lab: _axis_for(vidx, L, align) for lab, _i, _o, vidx in tracks}
    x = _axis_for(L // 2, L, "center")           # shared axis, used by shared_windows only
    tcmap = plt.get_cmap("viridis")
    single = len(tracks) == 1

    profs = {}
    for lab, imp, _oh, _vi in tracks:
        profs[lab] = smooth(imp.mean(0), smooth_w)
    norm_by = max((p.max() or 1.0) for p in profs.values()) if normalize else 1.0

    # Fixed high-contrast colors for the common AS / non-AS pair so the saliency tracks POP
    # against the (many, noisy) cofactor-density curves; fall back to viridis for other labels.
    FIXED = {"AS": "#d1495b", "nonAS": "#3d6cb3", "non-AS": "#3d6cb3",
             "peak": "#d1495b", "background": "#3d6cb3"}
    for i, (lab, imp, _oh, vidx) in enumerate(tracks):
        prof = profs[lab] / norm_by
        xt = xs[lab]
        col = FIXED.get(lab, tcmap(0.12 + 0.72 * i / max(1, len(tracks) - 1))
                        if len(tracks) > 1 else "#4b0082")
        lw = 3.4 if i == 0 else 3.0        # thick: saliency is the message, density is context
        ax.plot(xt, prof, color=col, lw=lw, label=f"{lab} saliency", zorder=10 - i,
                solid_capstyle="round")
        if mark_variants and align == "center":
            # where this track's variant actually is. Without it the reader has to take the
            # label's word for which offset a curve belongs to.
            ax.axvline(vidx - L // 2, color=col, ls=":", lw=1.1, alpha=0.75, zorder=2)
        if single and band != "none":
            if band == "iqr":
                lo, hi = np.percentile(imp, [25, 75], axis=0)
            else:
                sd = imp.std(0); lo, hi = imp.mean(0) - sd, imp.mean(0) + sd
            ax.fill_between(xt, smooth(lo, smooth_w) / norm_by, smooth(hi, smooth_w) / norm_by,
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
            for i, (lab, _imp, oh, _vi) in enumerate(tracks):
                dens = motif_density(oh, motifs, fimo, fimo_threshold, smooth_w)
                for j, nm in enumerate(dens):
                    ax2.plot(xs[lab], dens[nm], color=cmap(j),
                             lw=0.9, ls="--" if i else "-", alpha=0.5, zorder=3,
                             label=f"{nm} density ({lab})")
        ax2.set_ylabel("motif-hit density (fraction of windows)",
                       **({} if fs is None else {"fontsize": fs}))
        ax2.set_ylim(bottom=0)
        ax2.margins(x=0)
        if fs is not None:
            ax2.tick_params(labelsize=fs - 1)

    ax.axvline(0, color="red", ls="--", lw=1, zorder=1)
    ax.set_xlabel("position relative to " + ("center" if align == "center" else "variant")
                  + " (bp)", **({} if fs is None else {"fontsize": fs}))
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
    """
    [(label, path), ...] -> [(label, importance, onehot, variant_index), ...].

    VARIANT_INDEX IS CARRIED, NOT ASSUMED. Every window set built before the jitter work put
    the variant at the centre, so L//2 was the variant position and nothing needed to record
    it. A jittered build puts it at left_bp instead, and an npz from one is indistinguishable
    from an npz from the other by its arrays alone -- which is exactly why ism_saliency now
    writes variant_index. Falling back to L//2 keeps every pre-jitter npz working unchanged;
    the warning is there so a jittered run that lost the column cannot pass silently.

    All tracks must share one length: the figure plots them on one axis.
    """
    tracks, L = [], None
    for lab, path in pairs:
        d = np.load(path, allow_pickle=True)
        imp, onehot = d["importance"], d["onehot"]
        if L is None:
            L = imp.shape[1]
        elif imp.shape[1] != L:
            raise SystemExit(f"track {lab}: L={imp.shape[1]} != {L}")
        if "variant_index" in d:
            vi = np.unique(np.asarray(d["variant_index"]).astype(int))
            if len(vi) != 1:
                raise SystemExit(f"track {lab}: windows disagree on variant_index {vi[:5]}; "
                                 f"a track must be one offset arm.")
            vidx = int(vi[0])
            if not 0 <= vidx < L:
                raise SystemExit(f"track {lab}: variant_index {vidx} outside the window (L={L}).")
        else:
            vidx = L // 2
            print(f"[warn] {lab}: no variant_index in {path}; assuming the centre ({vidx}). "
                  f"Correct for any window set built before the jitter work, WRONG for a "
                  f"jittered one -- rebuild it if this track is part of an offset sweep.")
        tracks.append((lab, imp, onehot, vidx))
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
    ap.add_argument("--align", choices=["center", "variant"], default="center",
                    help="x origin. 'center' (default, the pre-jitter behaviour) plots bp from "
                         "the window centre, so an offset sweep shows its peaks AT the offsets. "
                         "'variant' shifts each track by its own variant_index, so tracking "
                         "peaks superimpose at 0 and a centre-pinned peak fans out instead.")
    ap.add_argument("--mark_variants", action="store_true",
                    help="dotted vertical line at each track's variant position (--align center)")
    ap.add_argument("--csv_out", default=None,
                    help="per-track table: expected offset, observed peak, error, peak height")
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
        normalize=a.normalize, title=a.title, align=a.align,
        mark_variants=a.mark_variants)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150, bbox_inches="tight")

    # THE TWO NUMBERS, not an eyeball of the figure.
    #   error  = observed peak - expected peak, in bp. Near 0 across an offset sweep is the
    #            result; an error that equals MINUS the expected offset means the peak never
    #            moved off the window centre, which is the artifact the control rules out.
    #   height = peak saliency. Read as a TREND across offsets: a dip confined to the extreme
    #            arms points at the window edge truncating context, not at weaker signal,
    #            and that distinction is not visible in the error column.
    rows = []
    print(f"\n  {'track':<14}{'expected':>10}{'observed':>10}{'error':>8}{'peak':>10}")
    for lab, _imp, _oh, vidx in tracks:
        prof = profs[lab]
        obs = int(np.argmax(prof)) - L // 2        # always reported centre-relative
        exp = vidx - L // 2
        rows.append(dict(track=lab, window_len=L, variant_index=vidx,
                         expected_bp=exp, observed_bp=obs, error_bp=obs - exp,
                         peak=float(prof.max())))
        print(f"  {lab:<14}{exp:>+10d}{obs:>+10d}{obs-exp:>+8d}{prof.max():>10.4f}")
    if len(rows) > 1:
        err = np.array([r["error_bp"] for r in rows])
        pk = np.array([r["peak"] for r in rows])
        print(f"\n  |error| median {np.median(np.abs(err)):.0f} bp, max {np.abs(err).max():.0f} bp"
              f"  |  peak range {pk.min():.4f}-{pk.max():.4f} "
              f"({'flat' if pk.max() <= 1.5 * pk.min() else 'VARIES >1.5x across arms'})")
    if a.csv_out:
        import pandas as pd
        pd.DataFrame(rows).to_csv(a.csv_out, index=False)
        print(f"[out] {a.csv_out}")
    print(f"[out] {a.out}")


if __name__ == "__main__":
    main()