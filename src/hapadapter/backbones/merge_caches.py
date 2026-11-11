#!/usr/bin/env python3
"""
hapadapter.backbones.merge_caches -- combine sharded pair caches into the single file that
fit_head expects.

WHY THIS EXISTS. extract_* shards by STRIDE (rows[shard::n_shards]), so N concurrent GPUs each
produce a disjoint slice of the same locus list. fit_head takes one --train_cache, so the
slices have to be reunited. Nothing here is backbone-specific: it operates on the {keys, vals}
cache format and never looks at what the vectors mean.

WHAT IT REFUSES, AND WHY EACH MATTERS MORE THAN IT LOOKS

  duplicate keys   Two shards claiming the same locus means they were NOT disjoint -- most
                   likely different --n_shards values between runs. Concatenating would put
                   two rows under one key, and load_pair_cache's index would silently keep
                   whichever came last. Refused.

  width mismatch   Shards from different pooling modes or backbones. A vstack would raise, but
                   with a shape error that does not say which file is the odd one out.

  missing shard    Cannot be detected from the files alone -- a 3-of-4 merge looks exactly like
                   a complete 3-shard run. Pass --expect_shards to make it checkable; without
                   it the count is reported and trusted, which is how a quietly smaller
                   training set gets through.

Usage:
    python -m hapadapter.backbones.merge_caches \\
        --out emb/ag_ctcf_ENC-002_train_1bp.npz --expect_shards 4 \\
        emb/shards/ag_ctcf_ENC-002_train_1bp_s*.npz
"""
import argparse
import os
import sys

import numpy as np

from hapadapter.backbones.cache import write_pair_cache


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shards", nargs="+", help="shard .npz files (shell glob is fine)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect_shards", type=int, default=0,
                    help="refuse unless exactly this many inputs are present. A merge of 3 of "
                         "4 shards is indistinguishable from a complete 3-shard run, so "
                         "without this the only symptom is a training set that is quietly "
                         "25%% smaller.")
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
        d = np.load(p, allow_pickle=True)
        missing = [k for k in ("keys", "vals") if k not in d.files]
        if missing:
            raise SystemExit(f"{p}: not a pair cache -- missing {missing}; has {list(d.files)}")
        k = np.asarray(d["keys"]).astype(str)
        v = np.asarray(d["vals"], dtype=np.float32)
        if len(k) != len(v):
            raise SystemExit(f"{p}: {len(k)} keys vs {len(v)} vectors")
        widths[os.path.basename(p)] = v.shape[1]
        keys.append(k); vals.append(v)
        print(f"[in] {os.path.basename(p):48} {len(k):>8,} rows  D={v.shape[1]}")

    if len(set(widths.values())) > 1:
        raise SystemExit(f"embedding widths disagree across shards: {widths}. These came from "
                         f"different pooling modes or different backbones.")

    K = np.concatenate(keys)
    uniq, counts = np.unique(K, return_counts=True)
    if len(uniq) != len(K):
        dup = uniq[counts > 1]
        raise SystemExit(
            f"{len(K) - len(uniq)} duplicate keys across shards (e.g. {list(dup[:3])}). The "
            f"shards are not disjoint -- almost always different --n_shards values between "
            f"runs. Re-extract with one consistent --n_shards rather than merging these.")

    V = np.vstack(vals)
    write_pair_cache(a.out, K, V)
    mb = os.path.getsize(a.out) / 1e6
    print(f"[write] {a.out} | {len(K):,} rows ({len(K)//2:,} loci) x {V.shape[1]} dims | {mb:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
