#!/usr/bin/env python3
"""
hapadapter.scoring.score_alphagenome -- AlphaGenome API forward pass on each hap and predictions differenced
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

def _client(api_key):
    """Import the AlphaGenome client across the version 0.x -> 2.x reorganization"""
    mod = None
    for path in ("alphagenome.models.dna_client",
                 "alphagenome.models.v2.dna_client",
                 "alphagenome.models.v1.dna_client"):
        try:
            mod = __import__(path, fromlist=["dna_client"])
            break
        except ImportError:
            continue
    if mod is None:
        raise SystemExit("could not import dna_client from alphagenome.models{,.v1,.v2} -- "
                         "check the installed alphagenome version")
    return mod, mod.create(api_key)

# assay -> (OutputType attribute, metadata attribute, metadata column holding the assay name)
ASSAY_MAP = {
    "ctcf":     ("CHIP_TF",      "chip_tf",      "transcription_factor"),
    "polr2a":   ("CHIP_TF",      "chip_tf",      "transcription_factor"),
    "h3k4me3":  ("CHIP_HISTONE", "chip_histone", "histone_mark"),
    "h3k4me1":  ("CHIP_HISTONE", "chip_histone", "histone_mark"),
    "h3k27ac":  ("CHIP_HISTONE", "chip_histone", "histone_mark"),
    "h3k9me3":  ("CHIP_HISTONE", "chip_histone", "histone_mark"),
    "h3k27me3": ("CHIP_HISTONE", "chip_histone", "histone_mark"),
    "h3k36me3": ("CHIP_HISTONE", "chip_histone", "histone_mark"),
    "atac":     ("ATAC",         "atac",         None),
}

# ENCODE spelling per assay tag for matching the metadata column
ENCODE_NAME = {"ctcf": "CTCF", "polr2a": "POLR2A", "h3k4me3": "H3K4me3", "h3k4me1": "H3K4me1",
               "h3k27ac": "H3K27ac", "h3k9me3": "H3K9me3", "h3k27me3": "H3K27me3",
               "h3k36me3": "H3K36me3", "atac": "ATAC"}

def select_tracks(md, assay, track_col, label):
    """Row mask over the track metadata"""
    if track_col is None:
        print(f"[tracks] {label}: {len(md)} tracks, no assay column needed for this output type")
        return np.ones(len(md), bool)
    if track_col not in md.columns:
        raise SystemExit(
            f"[tracks] {label}: metadata has no column {track_col!r}. Columns are "
            f"{list(md.columns)}. Pass --track_col with the right one.")
    want = ENCODE_NAME.get(assay, assay)
    m = md[track_col].astype(str).str.casefold() == want.casefold()
    if not m.any():
        vals = sorted(set(md[track_col].astype(str)))[:12]
        raise SystemExit(
            f"[tracks] {label}: no track has {track_col}=={want!r}. Present values include {vals}.")
    print(f"[tracks] {label}: {int(m.sum())} of {len(md)} tracks match {track_col}=={want!r}")
    return m.to_numpy()

def center_mask(n_bins, seq_len, width_bp):
    """
    Bin indices covering +/- width_bp around the center, and the bp they actually span
    ** relevant for when the API bins the output into > 1 bp chunks **
    """
    res = max(seq_len // n_bins, 1)
    c = seq_len // 2
    lo = max((c - width_bp) // res, 0)
    hi = min((c + width_bp) // res + 1, n_bins)
    return np.arange(lo, hi), (hi - lo) * res

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assay", required=True, help=f"one of {sorted(ASSAY_MAP)}")
    ap.add_argument("--pair_csv", required=True,
                    help="Genomic window CSV (sequence1/sequence2) -- can build with resize_windows.py")
    ap.add_argument("--meta_csv", required=True, help="its .meta.csv (chr/anchor)")
    ap.add_argument("--restrict_loci", default=None,
                    help="pair cache (.npz) or per-locus CSV; score ONLY these loci")
    ap.add_argument("--out_csv", required=True, help="chr,pos,s1,s2,delta")
    ap.add_argument("--cache", default=None,
                    help="npz of already-scored loci to avoid unecessarily re-scoring.")
    ap.add_argument("--track_col", default=None,
                    help="override the metadata column holding the assay name")
    ap.add_argument("--ontology", default=None,
                    help="restrict to tracks whose ontology_curie matches (e.g. the donor's "
                         "tissue). If None, will average over all matching tracks.")
    ap.add_argument("--mask_width", type=int, default=501,
                    help="center window in bp; the effective width is reported, since output "
                         "can be binned and the real span is a multiple of the bin size")
    ap.add_argument("--limit", type=int, default=0, help="only score  N loci to test setup (0 -> all)")
    ap.add_argument("--organism", default="HOMO_SAPIENS")
    a = ap.parse_args()

    if a.assay not in ASSAY_MAP:
        raise SystemExit(f"--assay {a.assay!r} not in {sorted(ASSAY_MAP)}; add it to ASSAY_MAP "
                         f"with its output type and metadata column or choose a different assay")
    ot_name, md_attr, default_col = ASSAY_MAP[a.assay]
    track_col = a.track_col if a.track_col is not None else default_col

    key = os.environ.get("ALPHAGENOME_API_KEY")
    if not key:
        raise SystemExit("ALPHAGENOME_API_KEY is not set")

    pair = pd.read_csv(a.pair_csv)
    meta = pd.read_csv(a.meta_csv)
    if len(pair) != len(meta):
        raise SystemExit(f"{len(pair)} window rows vs {len(meta)} meta rows -- these "
                         f"should have been written as a matched pair!")
    from hapadapter.backbones.cache import resolve_coord_columns
    cc, pc = resolve_coord_columns(meta, a.meta_csv)

    if a.restrict_loci:
        if a.restrict_loci.endswith(".npz"):
            d = np.load(a.restrict_loci, allow_pickle=True)
            want = set()
            for k in np.asarray(d["keys"]).astype(str):
                c, p, _h = k.rsplit(":", 2)
                want.add((c, int(p)))
        else:
            t = pd.read_csv(a.restrict_loci)
            tcc = "chr" if "chr" in t.columns else t.columns[0]
            tpc = next(c for c in ("pos", "anchor") if c in t.columns)
            want = set(zip(t[tcc].astype(str), t[tpc].astype(int)))
        sel = np.array([(str(c), int(p)) in want for c, p in zip(meta[cc], meta[pc])])
        if not sel.any():
            raise SystemExit(
                f"ZERO of {len(meta)} window loci are in --restrict_loci ({len(want)}). Window "
                f"has e.g. {list(zip(meta[cc], meta[pc]))[:2]}, restriction e.g. "
                f"{sorted(want)[:2]}. Probably a coordinate-convention mismatch (0-based vs 1-based)")
        pair, meta = pair[sel].reset_index(drop=True), meta[sel].reset_index(drop=True)
        print(f"[restrict] {int(sel.sum())} of {len(sel)} loci kept ({len(want)} requested)")

    if a.limit:
        pair, meta = pair.head(a.limit), meta.head(a.limit)
        print(f"[limit] pilot on {len(pair)} loci")

    seq_len = len(str(pair["sequence1"].iloc[0]))
    print(f"[window] {seq_len} bp")

    dna_client, client = _client(key)
    OT = dna_client.OutputType
    if not hasattr(OT, ot_name):
        raise SystemExit(f"this client has no OutputType.{ot_name}; available: "
                         f"{[m for m in dir(OT) if m.isupper()]}")
    ot = getattr(OT, ot_name)

    _Org = getattr(dna_client, "Organism", None)
    if _Org is None:
        from alphagenome.data import genome as _g
        _Org = getattr(_g, "Organism", None)
    if _Org is None:
        raise SystemExit("cannot locate the Organism enum in this alphagenome version; "
                         "check `dir(dna_client)` for where it moved")
    try:
        organism = _Org[a.organism.upper()]
    except KeyError:
        raise SystemExit(f"--organism {a.organism!r} is not one of {[m.name for m in _Org]}")
    meta_all = client.output_metadata(organism=organism)

    md = getattr(meta_all, md_attr, None)
    if md is None:
        raise SystemExit(f"output_metadata has no {md_attr!r}; available: "
                         f"{[x for x in dir(meta_all) if not x.startswith('_')]}")

    keep = select_tracks(md, a.assay, track_col, f"{a.assay}/{ot_name}")
    if a.ontology:
        if "ontology_curie" not in md.columns:
            raise SystemExit("--ontology given but metadata has no ontology_curie column")
        keep = keep & (md["ontology_curie"].astype(str) == a.ontology).to_numpy()
        if not keep.any():
            raise SystemExit(f"no track matches both the assay and ontology {a.ontology!r}")
        print(f"[tracks] {int(keep.sum())} after ontology filter {a.ontology!r}")

    # create both output directories before any scoring in case of crash
    for _p in (a.out_csv, a.cache):
        if _p and os.path.dirname(_p):
            os.makedirs(os.path.dirname(_p), exist_ok=True)

    done = {}
    if a.cache and os.path.exists(a.cache):
        z = np.load(a.cache, allow_pickle=True)
        done = {k: (float(s1), float(s2)) for k, s1, s2
                in zip(np.asarray(z["keys"]).astype(str), z["s1"], z["s2"])}
        print(f"[cache] {len(done)} loci already scored")

    def score(seq):
        out = client.predict_sequence(sequence=seq, organism=organism, requested_outputs=[ot])
        td = getattr(out, md_attr)
        v = np.asarray(td.values, dtype=float) # (bins, tracks)
        idx, eff = center_mask(v.shape[0], seq_len, a.mask_width)
        return float(np.nanmean(v[np.ix_(idx, np.where(keep)[0])])), eff

    rows, eff_bp, n_new = [], None, 0
    import time
    t0 = time.time()
    for i, (r, m) in enumerate(zip(pair.itertuples(index=False), meta.itertuples(index=False))):
        k = (f"{getattr(m, cc)}:{int(getattr(m, pc))}:{a.assay}:{a.mask_width}"
             f":{a.ontology or 'all'}:{int(keep.sum())}")
        if k in done:
            s1, s2 = done[k]
        else:
            s1, eff_bp = score(str(r.sequence1))
            s2, eff_bp = score(str(r.sequence2))
            done[k] = (s1, s2)
            n_new += 1
            if n_new % 25 == 0:
                el = time.time() - t0
                print(f"  [score] {i+1}/{len(pair)} | {el/max(n_new,1):.2f} s/locus | "
                      f"eta {(len(pair)-i-1)*el/max(n_new,1)/60:.0f} min", flush=True)
                if a.cache:
                    _save_cache(a.cache, done)
        # log2(hap1/hap2)
        eps = 1e-6
        rows.append((str(getattr(m, cc)), int(getattr(m, pc)), s1, s2,
                     float(np.log2((max(s1, 0) + eps) / (max(s2, 0) + eps)))))

    if a.cache:
        _save_cache(a.cache, done)
    out = pd.DataFrame(rows, columns=["chr", "pos", "s1", "s2", "delta"])
    os.makedirs(os.path.dirname(a.out_csv) or ".", exist_ok=True)
    out.to_csv(a.out_csv, index=False)
    if eff_bp is None:
        print(f"\n[mask] requested {a.mask_width} bp -> effective width NOT MEASURED this run "
              f"(every locus came from --cache). Re-run one locus without the cache to report it.")
    else:
        print(f"\n[mask] requested {a.mask_width} bp -> effective {eff_bp} bp "
              f"(output is binned -- the real span is a multiple of the bin size)")
    print(f"[write] {a.out_csv} ({len(out)} loci, {n_new} newly scored)")
    print(f"  delta: median |d| {np.median(np.abs(out.delta)):.4f}, "
          f"{int((out.delta == 0).sum())} exactly zero")
    if (out.delta == 0).all():
        print("  EVERY delta is ZERO -- the two haplotypes scored identically at every locus. "
              "Check that sequence1 and sequence2 actually differ in these windows.")
    return 0

def _save_cache(path, done):
    ks = sorted(done)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, keys=np.array(ks),
                        s1=np.array([done[k][0] for k in ks], float),
                        s2=np.array([done[k][1] for k in ks], float))
    os.replace(tmp, path)

if __name__ == "__main__":
    sys.exit(main())
