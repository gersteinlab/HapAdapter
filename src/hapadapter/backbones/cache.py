#!/usr/bin/env python3
"""
hapadapter.backbones.cache -- the embedding-cache format for backbone -> adapter transfer

We define a "backbone" as anything that can turn a pair of haplotype windows
into a pair of fixed-length vectors (embeddings)

    npz with two arrays:
        keys : (2N,) str   "<chrom>:<pos>:<tag>", tag in {h1, h2}
        vals : (2N, D)     float32, one pooled vector per window
"""
import os
import numpy as np

TAGS = ("h1", "h2")

_CHROM_COLS = ("chr", "chrom")
_POS_COLS = ("anchor", "ref_start", "pos")

def pair_key(chrom, pos, tag: str) -> str:
    """
    The cache key for one window
    """
    if tag not in TAGS:
        raise ValueError(f"tag must be one of {TAGS}, got {tag!r}")
    return f"{chrom}:{int(pos)}:{tag}"

def resolve_coord_columns(df, path: str = "<dataframe>"):
    """(chrom_col, pos_col) for a meta table, or raise naming what was actually present"""
    cc = next((c for c in _CHROM_COLS if c in df.columns), None)
    pc = next((c for c in _POS_COLS if c in df.columns), None)
    if cc is None or pc is None:
        raise ValueError(
            f"{path}: need a chromosome column from {_CHROM_COLS} and a position column from "
            f"{_POS_COLS}; got {list(df.columns)[:12]}. The cache is keyed by locus, so these "
            f"must be the SAME coordinates the extractor was given."
        )
    return cc, pc

def write_pair_cache(path: str, keys, vals) -> str:
    """Write a cache atomically, returning the path"""
    keys = np.asarray(keys)
    vals = np.asarray(vals, dtype=np.float32)
    if len(keys) != len(vals):
        raise ValueError(f"{len(keys)} keys but {len(vals)} vectors -- these must match!")
    if vals.ndim != 2:
        raise ValueError(f"vals must be (n_windows, D); got shape {vals.shape}.")
    out = path if path.endswith(".npz") else path + ".npz"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tmp = out[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, keys=keys, vals=vals)
    os.replace(tmp, out)
    return out

def load_pair_cache(path: str, chrom, pos, transform: str = "auto", tag: str = "",
                    min_ok_frac: float = 0.5, verbose: bool = True):
    """
    Join a cache onto a locus list
    Returns dict(h1, h2, ok, D, transform)

    `chrom` and `pos`: the locus list to score
    ** Loci absent from the cache come back as zero rows with ok=False -- filter on `ok` before fitting **

    transform = 'auto' picks none for a matrix with negative entries and log1p otherwise
    """
    chrom = np.asarray(chrom).astype(str)
    pos = np.asarray(pos).astype(int)
    if len(chrom) != len(pos):
        raise ValueError(f"chrom has {len(chrom)} entries, pos has {len(pos)}")

    d = np.load(path, allow_pickle=True)
    missing_arrays = [k for k in ("keys", "vals") if k not in d.files]
    if missing_arrays:
        raise ValueError(
            f"{path}: not a pair cache -- missing {missing_arrays}; has {list(d.files)}. "
            f"A per-track score cache from the zero-shot scorer has a different layout."
        )

    index = {str(k): i for i, k in enumerate(d["keys"])}
    vals = np.asarray(d["vals"], np.float32)
    n, D = len(chrom), vals.shape[1]

    h1 = np.zeros((n, D), np.float32)
    h2 = np.zeros((n, D), np.float32)
    ok = np.zeros(n, bool)

    for i, (c, p) in enumerate(zip(chrom, pos)):
        ia = index.get(pair_key(c, p, "h1"))
        ib = index.get(pair_key(c, p, "h2"))
        if ia is not None and ib is not None: # only mark as "ok" if both haplotypes are passed in
            h1[i], h2[i], ok[i] = vals[ia], vals[ib], True

    if not ok.any():
        raise ValueError(
            f"{path}: ZERO of {n} loci joined. The cache holds {len(index) // 2} loci, e.g. "
            f"{sorted(index)[:2]}, against requested e.g. "
            f"{[pair_key(chrom[0], pos[0], 'h1')] if n else []}. Mismatched coordinate "
            f"convention (0- vs 1-based) or a different locus set -- not a missing-data issue."
        )
    frac = float(ok.mean())
    if frac < min_ok_frac:
        raise ValueError(
            f"{path}: only {int(ok.sum())} of {n} loci joined ({frac:.1%}, below "
            f"min_ok_frac={min_ok_frac:.0%}). Fitting on this subset would change "
            f"which loci the result describes. Pass min_ok_frac=0 to proceed deliberately."
        )

    neg = min(float(h1[ok].min()), float(h2[ok].min()))
    xf = transform
    if xf == "auto":
        xf = "none" if neg < -1e-6 else "log1p"
    elif xf not in ("none", "log1p"):
        raise ValueError(f"transform must be auto/none/log1p, got {transform!r}")
    if xf == "log1p":
        if neg < -1e-6:
            raise ValueError(
                f"{tag or path}: transform='log1p' on a matrix with min {neg:+.4g} would clip "
                f"every negative coordinate to zero. Use 'none' for trunk embeddings."
            )
        h1, h2 = np.log1p(h1), np.log1p(h2)

    if verbose:
        print(f"[cache] {os.path.basename(path)}  n={n} ok={int(ok.sum())} ({frac:.1%}) "
              f"D={D}  transform={xf} (min {neg:+.3g})")
    return dict(h1=h1, h2=h2, ok=ok, D=D, transform=xf)

def cache_width(path: str) -> int:
    """Embedding width d_in, without materialising the matrix"""
    with np.load(path, allow_pickle=True) as d:
        return int(np.asarray(d["vals"]).shape[1])
