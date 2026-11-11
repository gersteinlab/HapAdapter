#!/usr/bin/env python3
"""
hapadapter.backbones.alphagenome -- pooled AlphaGenome trunk embeddings for both haplotypes of each locus
"""
import argparse
import dataclasses
import os
import time

import numpy as np
import pandas as pd

from hapadapter.backbones.cache import pair_key, resolve_coord_columns, write_pair_cache

BASES = "ACGT"

def offline_settings(dna_model):
    """Avoid pulling from unecessary AlphaGenome download paths"""
    return {
        org: dataclasses.replace(
            st, fasta_path=None,
            gtf_feather_path=None,
            pas_feather_path=None,
            splice_site_starts_feather_path=None,
            splice_site_ends_feather_path=None,
            calibration_path=None)
        for org, st in dna_model.default_organism_settings().items()
    }

def pool_embeddings(emb, mode, width, seq_len):
    """1bp-resolution embeddings -> (vector, effective_bp, source_array_name)"""
    a = emb.embeddings_1bp[0] # (S, 1536), still on device
    if mode == "mean":
        return np.asarray(a.mean(axis=0)), a.shape[0], "embeddings_1bp"
    center, L = seq_len // 2, a.shape[0]
    lo, hi = max(center - width, 0), min(center + width + 1, L)
    return np.asarray(a[lo:hi].mean(axis=0)), hi - lo, "embeddings_1bp"

def load_trunk(model_version):
    """(encoder, trunk apply function, params, state, organism_index)"""
    from alphagenome.models import dna_client
    from alphagenome_research.model import dna_model

    t0 = time.time()
    model = dna_model.create_from_huggingface(
        model_version, organism_settings=offline_settings(dna_model))
    print(f"[load] {model_version} in {time.time() - t0:.0f}s", flush=True)

    need = ("_one_hot_encoder", "_trunk_apply_fn", "_params", "_state")
    missing = [n for n in need if not hasattr(model, n)]
    if missing:
        raise RuntimeError(
            f"this alphagenome_research build does not expose {missing} on the model object. "
            f"The trunk-only path depends on those private attributes; check the current version "
            f"for a rename since v0.3.0."
        )
    org = dna_model.convert_to_organism_index(dna_client.Organism.HOMO_SAPIENS)
    return model._one_hot_encoder, model._trunk_apply_fn, model._params, model._state, org

def _done_keys(paths):
    """Find the keys already present in every existing output cache to allow for one forward pass"""
    sets, existing = [], {}
    for p in paths:
        if not os.path.exists(p):
            return set(), {}
        with np.load(p, allow_pickle=True) as d:
            ks = [str(k) for k in d["keys"]]
            existing[p] = (ks, np.asarray(d["vals"], np.float32))
        sets.append(set(ks))
    return set.intersection(*sets), existing

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair_csv", required=True, help="columns sequence1, sequence2")
    ap.add_argument("--meta_csv", required=True, help="columns chr/chrom and anchor/ref_start")
    ap.add_argument("--model_version", default="fold_0",
                    help="fold_0 holds out the most chr4+chr17; all_folds holds out nothing")
    ap.add_argument("--pool", nargs="+", default=["center_bp"],
                    choices=["center_bp", "mean"],
                    help="one or more pooling modes")
    ap.add_argument("--pool_width", nargs="+", type=int, default=[2],
                    help="bp either side of the variant, one per --pool mode (a single value "
                         "is broadcast). center_bp = 2 -> 5 bp in pool.")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--flush_every", type=int, default=1000,
                    help="forwards between cache writes. Writes are atomic, so a flush cannot "
                         "corrupt a previous one; flush_every = 0 disables and writes only at the end.")
    ap.add_argument("--overwrite", action="store_true",
                    help="ignore existing caches and recompute every locus")
    ap.add_argument("--out", nargs="+", required=True, help="one npz per --pool mode")
    a = ap.parse_args()

    if len(a.pool_width) == 1:
        a.pool_width = a.pool_width * len(a.pool)
    if not (len(a.pool) == len(a.out) == len(a.pool_width)):
        raise SystemExit(f"--pool has {len(a.pool)} modes, --out has {len(a.out)} paths, "
                         f"--pool_width has {len(a.pool_width)}: give one --out per mode (and "
                         f"either one --pool_width or one per mode).")
    if len(set(a.out)) != len(a.out):
        raise SystemExit(f"--out paths must be distinct, got {a.out}")
    a.out = [p if p.endswith(".npz") else p + ".npz" for p in a.out]

    df = pd.read_csv(a.pair_csv)
    meta = pd.read_csv(a.meta_csv)
    for c in ("sequence1", "sequence2"):
        if c not in df.columns:
            raise SystemExit(f"{a.pair_csv}: need sequence1/sequence2; got {list(df.columns)}")
    if len(df) != len(meta):
        raise SystemExit(f"row mismatch: pair_csv {len(df)} vs meta_csv {len(meta)} -- these "
                        f"must be the matched pair written by resize_windows.py")
    
    cc, pc = resolve_coord_columns(meta, a.meta_csv)
    if a.limit:
        df, meta = df.iloc[:a.limit], meta.iloc[:a.limit]
    rows = np.arange(len(df))
    
    if a.n_shards > 1:
        bad = [p for p in a.out if str(a.shard) not in os.path.basename(p)]
        if bad:
            raise SystemExit(
                f"--n_shards {a.n_shards} but {bad} does not contain the shard index {a.shard} "
                f"in the filename. Use e.g. --out emb/ag_test_1bp_s{a.shard}.npz")
        rows = rows[a.shard::a.n_shards]

    # resume instead of starting over in the case of interrupt
    done, existing = (set(), {}) if a.overwrite else _done_keys(a.out)
    if done:
        keep = [i for i in rows
                if not all(pair_key(meta[cc].iloc[i], meta[pc].iloc[i], t) in done
                           for t in ("h1", "h2"))]
        print(f"[resume] {len(done) // 2} loci already cached -> {len(keep)} of {len(rows)} "
              f"left in this shard")
        rows = np.array(keep, dtype=int)
    keys = [k for k in (existing[a.out[0]][0] if existing else [])]
    vals = {m: ([existing[p][1]] if existing else [])
            for m, p in zip(a.pool, a.out)}
    if existing:
        # re-key every mode off ONE order so a flush cannot interleave them differently
        order = {k: i for i, k in enumerate(existing[a.out[0]][0])}
        for m, p in zip(a.pool, a.out):
            ks, vs = existing[p]
            vals[m] = [vs[np.array([order[k] for k in ks])]] if ks else []

    seq_len = len(str(df["sequence1"].iloc[0]))
    n_fwd = 2 * len(rows)
    print(f"[in] {len(df)} loci | window {seq_len} bp | shard {a.shard}/{a.n_shards} "
          f"-> {len(rows)} loci, {n_fwd} forwards")
    if n_fwd == 0:
        print("[done] nothing left to extract")
        return

    from alphagenome.models import dna_client
    try:
        dna_client.validate_sequence_length(seq_len)
    except ValueError as e:
        raise SystemExit(f"{seq_len} bp is not a supported AlphaGenome length: {e}. Rebuild "
                         f"windows at a supported length.")
    enc, trunk, params, state, org = load_trunk(a.model_version)

    info = {}
    t0, done_n = time.time(), 0

    def flush():
        K = np.array(keys)
        for mode, dest in zip(a.pool, a.out):
            V = np.vstack(vals[mode])
            assert len(V) == len(K), f"{mode}: {len(V)} vectors vs {len(K)} keys"
            write_pair_cache(dest, K, V)

    for i in rows:
        for tag, col in (("h1", "sequence1"), ("h2", "sequence2")):
            seq = str(df[col].iloc[i]).upper()
            bad = set(seq) - set(BASES + "N")
            if bad:
                raise SystemExit(f"row {i} {col}: non-ACGTN characters {sorted(bad)[:4]}")
            emb = trunk(params, state, enc.encode(seq)[np.newaxis],
                        np.asarray([org], dtype=np.int32))
            # EVERY mode is reduced off this ONE forward, before `emb` is discarded
            for mode, w in zip(a.pool, a.pool_width):
                v, eff_bp, src = pool_embeddings(emb, mode, w, seq_len)
                vals[mode].append(np.asarray(v, dtype=np.float32)[np.newaxis])
                info[mode] = (eff_bp, src)
            keys.append(pair_key(meta[cc].iloc[i], meta[pc].iloc[i], tag))
            done_n += 1
            if a.flush_every and done_n % a.flush_every == 0:
                flush()
                print(f"[flush] {len(keys) // 2} loci written", flush=True)
        if done_n % 200 == 0:
            el = time.time() - t0
            print(f"[fwd] {done_n}/{n_fwd} | {el / done_n:.3f} s/forward | "
                  f"eta {(n_fwd - done_n) * el / done_n / 60:.1f} min", flush=True)

    flush()
    for mode, w, dest in zip(a.pool, a.pool_width, a.out):
        eff_bp, src = info[mode]
        V = np.vstack(vals[mode])
        req = "whole window" if mode == "mean" else f"requested +/-{w} bp"
        # effective bp may be slightly off due to edge clipping
        print(f"[pool] {mode:12} from {src}: {req} -> effective {eff_bp} bp")
        print(f"[write] {dest} | {V.shape[0]} vectors x {V.shape[1]} dims "
              f"({V.nbytes / 1e6:.0f} MB)")

if __name__ == "__main__":
    main()
