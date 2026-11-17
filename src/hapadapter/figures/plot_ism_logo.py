#!/usr/bin/env python
"""
Plot an averaged ISM contribution logo for the entexBERT-2 binding trunk and overlay the
JASPAR CTCF PWM (MA0139.2 core) aligned to the recovered motif.

Inputs:
  --ism      : .npz from ism_saliency.py  (onehot, contrib, importance, seqs)
  --pwm_json : jaspar2024_core_vert_pfms.json  (JASPAR PFMs keyed by matrix_id; entry['pfm'][base])
Because ISM windows are not motif-centered, we align by cross-correlating each window's
per-position importance against the PWM information profile, roll each to a common center,
then average the contribution logos. The top panel is the model's learned logo; the bottom
panel is the JASPAR reference logo at the same width.
"""
import argparse, json, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import logomaker

BASES = "ACGT"


def pfm_to_prob(pfm):
    """JASPAR PFM dict {A:[...],C:[...],G:[...],T:[...]} -> (W,4) probability matrix."""
    M = np.array([pfm[b] for b in BASES], dtype=np.float64).T  # (W,4)
    M = M + 1e-9
    return M / M.sum(axis=1, keepdims=True)


def prob_to_ic(P):
    """probabilities -> per-base information content (bits), for the reference logo + profile."""
    ic_pos = 2.0 + (P * np.log2(P)).sum(axis=1)          # (W,)
    return P * ic_pos[:, None]                            # (W,4)


_COMP = [3, 2, 1, 0]          # A<->T, C<->G for a (.,4) array in ACGT order


def revcomp(mat):
    """(W,4) -> reverse-complemented (W,4): reverse positions AND swap base channels."""
    return mat[::-1, _COMP]


def best_offset(imp, profile, W, strand_aware=False, imp_rc=None):
    """Slide the PWM info-profile across a window's importance; return (offset, best, flip).

    WHY THIS IS A SELECTION PROCEDURE, AND WHY THE CONTROLS BELOW EXIST. This takes the MAXIMUM
    correlation over L-W+1 candidate offsets (243 for a 257bp window and a 15bp motif). Taking a
    max over hundreds of candidates finds motif-like segments in sequence that contains no motif,
    so averaging the selected segments can manufacture a clean consensus logo out of noise -- the
    same failure mode as maximising a PWM score over placements. A clean averaged logo is
    therefore NOT evidence on its own; it only means something relative to the same pipeline run
    on a shuffled matrix or shuffled attributions.

    strand_aware also scores the reverse orientation. Without it, sites on opposite strands are
    aligned with mirrored context, so a cofactor at a fixed genomic offset lands on opposite
    sides of the motif in the two halves and cancels -- which matters far more for reading the
    FLANK than for recovering the motif itself.
    """
    L = imp.shape[0]
    prof = (profile - profile.mean()) / (profile.std() + 1e-9)
    best, bo, flip = -np.inf, 0, False
    for off in range(0, L - W + 1):
        seg = imp[off:off + W]
        seg = (seg - seg.mean()) / (seg.std() + 1e-9)
        c = float((seg * prof).sum())
        if c > best:
            best, bo, flip = c, off, False
        if strand_aware:
            seg_r = imp_rc[off:off + W]
            seg_r = (seg_r - seg_r.mean()) / (seg_r.std() + 1e-9)
            cr = float((seg_r * prof).sum())
            if cr > best:
                best, bo, flip = cr, off, True
    return bo, best, flip


def align_contributions(onehot, contrib, imp, profile, W, flank, *, control="none",
                        strand_aware=False, rng=None, return_offsets=False):
    """
    Align every window to the reference information profile and average the contributions.

    Returns (agg (W+2*flank, 4), used, n_flip), or that plus an `offsets` array of the chosen
    motif-start index per USED window when return_offsets=True. The offsets are what let a
    caller say where the variant sits relative to the aligned motif: the variant is at L//2 in
    every window, so its position in motif coordinates is flank + (L//2 - offset). Opt-in rather
    than always returned, so existing three-value unpacking keeps working.

    Split out of main() so the composite figure
    and the standalone CLI compute the aligned average through the SAME code -- the alignment
    is the part of this script that a second implementation would most easily get subtly wrong.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    N, L, _ = onehot.shape
    span = W + 2 * flank
    agg = np.zeros((span, 4), dtype=np.float64)
    used = n_flip = 0
    offsets = []
    for w in range(N):
        imp_w, con_w = imp[w], contrib[w]
        if control == "shuffle_contrib":
            q = rng.permutation(L)
            imp_w, con_w = imp_w[q], con_w[q]
        con_rc = revcomp(con_w) if strand_aware else None
        imp_rc = con_rc.sum(axis=1) if strand_aware else None
        off, _sc, flip = best_offset(imp_w, profile, W, strand_aware, imp_rc)
        s0 = off - flank; e0 = off + W + flank
        if s0 < 0 or e0 > L:
            continue
        agg += (con_rc if flip else con_w)[s0:e0, :]
        used += 1
        n_flip += int(flip)
        offsets.append(off)
    agg /= max(used, 1)
    if return_offsets:
        return agg, used, n_flip, np.asarray(offsets, dtype=np.int64)
    return agg, used, n_flip


def centre_contributions(contrib, half, *, center=None):
    """
    Average contributions in a FIXED window around the VARIANT. -> (agg (2*half+1, 4), used)

    THE OPPOSITE CHOICE FROM align_contributions, and a different claim. That function slides
    each window to wherever its best motif match sits, so the average answers "what does the
    model attribute across the motif". This one holds every window at the variant, so it answers
    "what does the model attribute around the base whose allele actually differs" -- which is the
    question a HEAD figure is asking, since the head's input is variant-centred and the variant
    is the only position whose identity differs between the two haplotypes.

    It also avoids a failure mode specific to this data. Most AS windows contain no canonical
    motif at all, so motif alignment places them at whatever offset scores best out of noise and
    averages real signal from the minority with arbitrary offsets from the majority. A fixed
    slice has no such degree of freedom.

    EVERY WINDOW IS USED. align_contributions drops windows whose aligned span falls off the
    edge; a centred slice of a variant-centred window cannot, as long as half <= L // 2, so
    `used` always equals N and the count in a title is not a filter applied silently.
    """
    contrib = np.asarray(contrib)
    N, L, _ = contrib.shape
    c = L // 2 if center is None else int(center)
    s0, e0 = c - half, c + half + 1
    if s0 < 0 or e0 > L:
        raise ValueError(f"half={half} does not fit a {L} bp window centred at {c}; "
                         f"the largest that fits is {min(c, L - 1 - c)}")
    return contrib[:, s0:e0, :].mean(axis=0), int(N)


def draw_ism_logo(ax_logo, agg, flank, W, *, ax_ref=None, ref_ic=None, title=None,
                  ref_title=None, fontsize=None, ylabel="ISM contribution", flip=False,
                  xoffset=None, shade=True, mark_center=False, shade_label=None,
                  mark_at=None, mark_label=None):
    """
    Draw the averaged contribution logo onto an EXISTING axis, optionally with the reference
    PWM logo on a second one. Drawing only -- align_contributions does the work.

    flip=True NEGATES THE MATRIX AND RELABELS THE TICKS, rather than inverting the axis. It
    exists because an ASB head's contributions are negative almost everywhere -- substituting
    away from the observed base makes the two haplotypes more different, so it RAISES the
    logit -- which puts the whole logo below the axis where logomaker draws characters UPSIDE
    DOWN (flip_below defaults to True). The result is an unreadable logo of mirrored glyphs.

    ax_logo.invert_yaxis() would not fix it: inverting the coordinate mapping mirrors the
    glyph paths too, so the letters stay upside down and merely move. Negating the values
    draws them upright and above the axis; the tick FORMATTER then prints the original signed
    number, so the axis still reads negative and the figure makes no false claim. Say so in the
    caption -- "contributions are negative throughout; the axis is inverted so letters read
    upright" -- because a reader who does not notice the minus signs will read the sign wrong.

    The reference PWM strip is never flipped: information content is non-negative and its
    letters already point up.
    """
    fs = fontsize
    mat = -np.asarray(agg, dtype=float) if flip else agg
    lm = pd.DataFrame(mat, columns=list(BASES))
    logomaker.Logo(lm, ax=ax_logo, color_scheme="classic")
    if title is not None:
        ax_logo.set_title(title, fontsize=(10 if fs is None else fs + 1))
    ax_logo.set_ylabel(ylabel, **({} if fs is None else {"fontsize": fs}))
    if shade:
        # The shaded band marks the MOTIF SPAN and is meaningful only when the windows were
        # aligned to a motif. On a variant-centred average there is no common motif position to
        # shade, so drawing it anyway would assert an alignment that was not performed.
        ax_logo.axvspan(flank - 0.5, flank + W - 0.5, color="0.9", zorder=-5)
        if shade_label:
            # Name the shaded band ON the axes. A grey rectangle with no label is read as
            # decoration; the band is the motif span and the reader cannot recover which motif
            # from the panel alone once the reference strip is not drawn beneath it.
            # INSIDE the axes, not above: above the top spine is where the panel title is laid
            # out, and the two collide at any font size.
            ax_logo.annotate(shade_label, xy=(flank + W / 2 - 0.5, 0.97),
                             xycoords=("data", "axes fraction"),
                             xytext=(0, 0), textcoords="offset points",
                             ha="center", va="top", color="0.35",
                             fontsize=(8 if fs is None else fs - 1.5),
                             # The logo fills the axes, so any in-axes label lands on letters.
                             # A translucent plate keeps it readable without moving it off the
                             # band it names.
                             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))
    if mark_at is not None:
        ax_logo.axvline(mark_at, color="red", ls="--", lw=1.1, zorder=4)
        if mark_label:
            ax_logo.annotate(mark_label, xy=(mark_at, 0.02), xycoords=("data", "axes fraction"),
                             xytext=(3, 0), textcoords="offset points",
                             ha="left", va="bottom", color="red",
                             fontsize=(8 if fs is None else fs - 1.5), bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))
    if xoffset is not None:
        # Relabel to coordinates relative to the variant. logomaker indexes positions 0..n-1;
        # the ticks should read -half..+half so the axis states what the centre is.
        n = lm.shape[0]
        step = max(1, int(round(n / 8 / 5)) * 5)
        ticks = [t for t in range(0, n) if (t - xoffset) % step == 0]
        ax_logo.set_xticks(ticks)
        ax_logo.set_xticklabels([str(t - xoffset) for t in ticks])
    if mark_center:
        ax_logo.axvline(xoffset if xoffset is not None else (flank + W / 2 - 0.5),
                        color="red", ls="--", lw=1, zorder=4)
    if flip:
        from matplotlib.ticker import FuncFormatter
        # Print the UNNEGATED value, so the drawn height is -v and the label says v.
        ax_logo.yaxis.set_major_formatter(
            FuncFormatter(lambda v, _pos: "0" if v == 0 else f"{-v:g}"))
    if fs is not None:
        ax_logo.tick_params(labelsize=fs - 1)
    if ax_ref is not None and ref_ic is not None:
        rl = pd.DataFrame(np.vstack([np.zeros((flank, 4)), ref_ic, np.zeros((flank, 4))]),
                          columns=list(BASES))
        logomaker.Logo(rl, ax=ax_ref, color_scheme="classic")
        if ref_title is not None:
            ax_ref.set_title(ref_title, fontsize=(10 if fs is None else fs + 1))
        ax_ref.set_ylabel("bits", **({} if fs is None else {"fontsize": fs}))
        ax_ref.set_xlabel("position (motif shaded)", **({} if fs is None else {"fontsize": fs}))
        if fs is not None:
            ax_ref.tick_params(labelsize=fs - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ism", required=True)
    ap.add_argument("--pwm_json", required=True)
    ap.add_argument("--matrix_id", default="MA0139.2",
                    help="JASPAR matrix id present in --pwm_json (CTCF core = MA0139.2)")
    ap.add_argument("--flank", type=int, default=4, help="bp of flank to show around the motif")
    ap.add_argument("--tf_label", default="", help="TF name for the title (defaults to JASPAR entry name)")
    ap.add_argument("--control", default="none",
                    choices=["none", "shuffle_pwm", "shuffle_contrib"],
                    help="shuffle_pwm = column-permute the matrix before aligning (same IC "
                         "distribution, wrong order); shuffle_contrib = permute positions within "
                         "each window. EITHER control recovering a clean logo means the alignment "
                         "is manufacturing it, not the model")
    ap.add_argument("--strand_aware", action="store_true",
                    help="also score the reverse orientation and flip that window before "
                         "averaging. Needed to read the FLANK; the motif survives without it")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    d = np.load(a.ism, allow_pickle=True)
    onehot, contrib, imp = d["onehot"], d["contrib"], d["importance"]
    N, L, _ = onehot.shape

    pj = json.load(open(a.pwm_json))
    entry = pj[a.matrix_id]
    pfm = entry.get("pfm") or entry.get("counts")
    P = pfm_to_prob(pfm); W = P.shape[0]
    ref_ic = prob_to_ic(P)
    profile = ref_ic.sum(axis=1)                          # (W,) info profile to align against
    if a.control == "shuffle_pwm":
        # permute COLUMNS: identical per-position information content, destroyed order. The
        # alignment still has a structured profile of the same width to maximise against, so a
        # clean result here isolates the selection step from the motif's actual identity.
        perm = rng.permutation(W)
        ref_ic, profile = ref_ic[perm], profile[perm]
        print(f"[control] shuffle_pwm: columns permuted (seed {a.seed})")
    elif a.control == "shuffle_contrib":
        print(f"[control] shuffle_contrib: positions permuted within each window (seed {a.seed})")

    span = W + 2 * a.flank
    agg, used, n_flip = align_contributions(onehot, contrib, imp, profile, W, a.flank,
                                            control=a.control, strand_aware=a.strand_aware,
                                            rng=rng)

    # QUANTITATIVE readout, so real-vs-control is a number rather than two pictures. Correlate the
    # averaged contribution over the MOTIF SPAN against the reference information profile.
    span_obs = np.abs(agg[a.flank:a.flank + W]).sum(axis=1)
    _o = (span_obs - span_obs.mean()) / (span_obs.std() + 1e-9)
    _p = (profile - profile.mean()) / (profile.std() + 1e-9)
    r_motif = float((_o * _p).mean())
    flank_amp = (float(np.abs(agg[:a.flank]).sum(axis=1).mean()) if a.flank else float("nan"))
    print(f"[logo] control={a.control} strand_aware={a.strand_aware} flipped={n_flip}/{used}")
    print(f"[logo] motif-span r with reference IC profile = {r_motif:+.3f}   "
          f"mean |contribution| motif {np.abs(agg[a.flank:a.flank+W]).sum(axis=1).mean():.4f} "
          f"vs flank {flank_amp:.4f}")
    enrich = (np.abs(agg[a.flank:a.flank + W]).sum(axis=1).mean() / flank_amp
              if a.flank and flank_amp > 0 else float("nan"))
    print(f"[logo] MOTIF/FLANK ENRICHMENT = {enrich:.2f}   <- the statistic to compare")
    print(f"[read] Run the same command with --control shuffle_pwm and compare ENRICHMENT, not r.")
    print(f"       Real signal COLLAPSES under the control; an alignment artifact does not.")
    print(f"       Measured: CTCF ENC-002 trunk 5.42 -> 0.78 (7.0x collapse, real); a planted")
    print(f"       positive 365 -> 0.62; a pure-noise negative 3.33 -> 3.20 (1.0x, no collapse).")
    print(f"       DO NOT use the r above as the discriminator: the alignment maximises the")
    print(f"       correlation it measures, so r is high in every run -- shuffle_contrib, which")
    print(f"       destroys all positional structure, still scored +0.713 on real data.")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(6, span * 0.35), 4.4),
                                   gridspec_kw={"height_ratios": [2, 1]})
    _tf = a.tf_label or entry.get("name", "")
    draw_ism_logo(ax1, agg, a.flank, W, ax_ref=ax2, ref_ic=ref_ic,
                  title=f"entexBERT-2 trunk — ISM contribution logo "
                        f"(avg of {used}/{N} {_tf} windows)",
                  ref_title=f"JASPAR {a.matrix_id} ({entry.get('name','CTCF')}) — reference (bits)")
    fig.tight_layout()
    fig.savefig(a.out, dpi=150, bbox_inches="tight")
    print(f"[plot] saved {a.out} | {used}/{N} windows aligned | motif width {W} + {a.flank}bp flank")


if __name__ == "__main__":
    main()
