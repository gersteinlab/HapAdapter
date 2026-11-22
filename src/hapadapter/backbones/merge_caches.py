#!/usr/bin/env python3
"""
hapadapter.backbones.merge_caches -- combine sharded pair caches into the single file that fit_adapter expects
** allows embeddings to be extracted in parallel shards and then merged if multiple GPUs are available **
"""
import argparse
import os
import sys
import numpy as np

from hapadapter.backbones.cache import write_pair_cache

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shards", nargs="+", help="shard .npz files (shell glob is fine too)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect_shards", type=int, default=0,
                    help="refuse unless exactly this many inputs are present.")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    if os.path.exists(a.out) and not a.overwrite:
        print(f"[skip] {a.out} exists (use --overwrite)")
        return 0
    if a.expect_shards and len(a.shards) != a.expect_shards:
        raise SystemExit(f"expected {a.expect_shards} shards, got {len(a.shards)}: "
                         f"{[os.path.basename(s) for s in a.shards]}")

    keys, vals, widths = [], [], {}
    for p in sorted(a.shards):
        with np.load(p, allow_pickle=True) as d:
            missing = [k for k in ("keys", "vals") if k not in d.files]
            if missing:
                raise SystemExit(f"{p}: not a pair cache -- missing {missing}; "
                                 f"has {list(d.files)}")
            k = np.asarray(d["keys"]).astype(str)
            v = np.asarray(d["vals"], dtype=np.float32)
        if len(k) != len(v):
            raise SystemExit(f"{p}: {len(k)} keys vs {len(v)} vectors")
        # key by full path
        widths[p] = v.shape[1]
        keys.append(k)
        vals.append(v)
        print(f"[in] {os.path.basename(p):48} {len(k):>8,} rows  D={v.shape[1]}")

    if len(set(widths.values())) > 1:
        raise SystemExit(f"embedding widths disagree across shards: {widths}. These came from "
                         f"different pooling modes or different backbones!")

    K = np.concatenate(keys)
    uniq, counts = np.unique(K, return_counts=True)
    if len(uniq) != len(K):
        dup = uniq[counts > 1]
        raise SystemExit(
            f"{len(K) - len(uniq)} duplicate keys across shards (e.g. {list(dup[:3])}). The "
            f"shards are not disjoint -- almost always different --n_shards values between "
            f"runs. Re-extract with one consistent --n_shards rather than merging these.")

    tags = {}
    for key in K:
        locus, tag = key.rsplit(":", 1)
        tags.setdefault(locus, set()).add(tag)
    only_one_hap = sorted(l for l, t in tags.items() if t != {"h1", "h2"})
    if only_one_hap:
        print(f"[warn] {len(only_one_hap)} locus/loci carry only one haplotype "
              f"(e.g. {only_one_hap[:3]}). These join as ok=False and drop out at fit time. "
              f"Re-run the extractor on the affected shard -- its resume path drops "
              f"and recomputes these loci.", file=sys.stderr)

    V = np.vstack(vals)
    write_pair_cache(a.out, K, V)
    mb = os.path.getsize(a.out) / 1e6

    print(f"[write] {a.out} | {len(K):,} rows ({len(tags) - len(only_one_hap):,} complete loci"
          + (f", {len(only_one_hap):,} incomplete" if only_one_hap else "")
          + f") x {V.shape[1]} dims | {mb:.0f} MB")

    return 0

if __name__ == "__main__":
    sys.exit(main())
