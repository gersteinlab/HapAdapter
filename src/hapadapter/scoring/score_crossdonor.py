#!/usr/bin/env python
"""
hapadapter.scoring.score_crossdonor -- cross-donor test set for HapAdapter head scoring
(both ref/alt and hap1/hap2 test sets supported)

Writes one per-head CSV: assay, train_donor, eval_donor, arm, donor_kind, auroc, n, n_pos
"""
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def _head_variant(head_dir):
    """Read head_variant off the checkpoint"""
    p = os.path.join(head_dir, "run_config.json")
    if not os.path.exists(p):
        print(f"[head] WARNING: no run_config.json in {head_dir}; assuming head_variant='twin'")
        return "twin"
    import json
    return json.load(open(p)).get("head_variant", "twin")

def score_on(head_dir, test_csv, batch_size, device):
    """Score one donor's test split; returns (auroc_matched, auroc_plain, n, n_pos, n_matched)"""
    from hapadapter.loading import run_inference # to keep --help torch-free
    from hapadapter.scoring.stats import balanced_auroc

    df = pd.read_csv(test_csv)
    variant = _head_variant(head_dir)

    if "label" not in df.columns:
        raise ValueError(f"{test_csv}: no label column; got {list(df.columns)[:6]}")

    if variant != "twin":
        raise ValueError(f"{head_dir}: head_variant={variant!r}, but only 'twin' heads can be "
                         f"scored here. Re-fit this head to score.")

    if "sequence1" not in df.columns or "sequence2" not in df.columns:
        raise ValueError(f"{test_csv}: a twin head needs sequence1,sequence2; "
                         f"got {list(df.columns)[:6]}.")

    texts = [[s1, s2] for s1, s2 in zip(df["sequence1"].astype(str), df["sequence2"].astype(str))]

    logits, _emb, _rc = run_inference(head_dir, texts, batch_size, device)
    ell = np.asarray(logits).reshape(-1)
    y = df["label"].astype(int).to_numpy()
    if y.min() == y.max():
        return float("nan"), float("nan"), len(y), int(y.sum()), 0

    plain = float(roc_auc_score(y, ell))
    cov_col = next((c for c in ("depth", "total_reads") if c in df.columns), None)
    if cov_col is None:
        print(f"[cov] WARNING: {test_csv} has no depth/total_reads column -- the matched AUROC "
              f"falls back to class-balancing only and is not depth-controlled")
        cov = None
    else:
        cov = df[cov_col].to_numpy(float)
        if not np.isfinite(cov).all():
            print(f"[cov] WARNING: non-finite values in {cov_col} -- depth control disabled")
            cov = None
            
    matched, _aupr, _ci, n_matched = balanced_auroc(ell, y, cover=cov)
    return float(matched), plain, len(y), int(y.sum()), int(n_matched)

DATA_TPL = "{exp_root}/{assay_lower}_{donor}/stage2_{arm}_data"

def split_dir(exp_root, assay, donor, arm, data_tpl=None):
    """Directory holding one donor-arm's HapAdapter training CSVs, from --data_tpl"""
    try:
        return (data_tpl or DATA_TPL).format(
            exp_root=exp_root, assay=assay, assay_lower=assay.lower(), donor=donor, arm=arm)
    except KeyError as e:
        raise SystemExit(f"--data_tpl {data_tpl!r}: unknown placeholder {e}. "
                         f"Available: {{exp_root}} {{assay}} {{assay_lower}} "
                         f"{{donor}} {{arm}}")

def _meta_coords(path):
    """(chr, pos) set from a *.meta.csv"""
    tc = pd.read_csv(path)
    chrom_col = next((c for c in ("chr", "chrom") if c in tc.columns), None)
    pos_col = next((c for c in ("SNV", "pos", "anchor", "ref_start") if c in tc.columns), None)
    if chrom_col is None or pos_col is None:
        raise ValueError(f"{path}: need a chromosome column from ('chr', 'chrom') and a "
                         f"position column from ('SNV', 'pos', 'anchor', 'ref_start'); "
                         f"got {list(tc.columns)[:8]}")

def verify_folds(exp_root, assay, arm, train_donor, donors, bin_size, data_tpl=None):
    """
    Check the crucial two cross-donor comparison conditions:

      (1) LEAKAGE: the head's train+dev coordinates must not appear in any eval donor's test set (checked within bin_size)
      (2) COMPARABILITY: every eval donor's test set should hold the SAME coordinates

    Returns True if no leakage was found
    Comparability differences warn but do not fail
    """
    def meta(d, split):
        return os.path.join(split_dir(exp_root, assay, d, arm, data_tpl), f"{split}.meta.csv")

    seen = set()
    for split in ("train", "dev"):
        p = meta(train_donor, split)
        if not os.path.exists(p):
            print(f"[verify_folds] WARNING: {p} not found -- leak check INCOMPLETE")
            continue
        seen |= _meta_coords(p)

    if not seen:
        print(f"[verify_folds] FAIL: no train/dev coords found for {train_donor} under "
              f"{exp_root}; leakage cannot be verified.")
        return False

    seen_bins = {(c, p // bin_size) for c, p in seen}
    print(f"[verify_folds] {train_donor} train+dev: {len(seen)} loci, {len(seen_bins)} bins")

    test_sets, leaked = {}, False
    for ed in donors:
        p = meta(ed, "test")
        if not os.path.exists(p):
            print(f"[verify_folds] {ed}: no test.meta.csv -- skipped")
            continue
        t = _meta_coords(p)
        test_sets[ed] = t
        ex = len(t & seen)
        bn = len({(c, q // bin_size) for c, q in t} & seen_bins)
        flag = "LEAK" if (ex or bn) else "clean"
        print(f"[verify_folds] {ed} test: {len(t)} loci | exact overlap {ex} | "
              f"bin overlap {bn} -> {flag}")
        if ex or bn:
            leaked = True

    if len(test_sets) > 1:
        # check that no test bins for one donor end up in train or dev for another
        seen_by = {}
        for d in test_sets:
            s = set()
            for split in ("train", "dev"):
                p = meta(d, split)
                if os.path.exists(p):
                    s |= {(c, q // bin_size) for c, q in _meta_coords(p)}
            seen_by[d] = s
        tbins = {d: {(c, q // bin_size) for c, q in t} for d, t in test_sets.items()}
        bad = {f"{d1}.test n {d2}.train/dev": len(tbins[d1] & seen_by[d2])
               for d1 in tbins for d2 in seen_by
               if d1 != d2 and seen_by[d2] and tbins[d1] & seen_by[d2]}
        if bad:
            print(f"[verify_folds] WARNING: the fold is INCONSISTENT across donors -- bins that are "
                  f"test for one donor are train/dev for another: {bad}")
        else:
            cov = {d: len(t) for d, t in test_sets.items()}
            print(f"[verify_folds] fold is consistent: no donor's test bin is any other donor's train/dev bin!")

    return not leaked

def score_cached(head_dir, cache, pair_csv, meta_csv, cov_col, label_col="label", cache_transform=None, min_ok_frac=0.5):
    """Score one donor's test split from an embedding cache"""
    import json
    import torch
    from hapadapter.adapter.head import HapAdapterHead
    from hapadapter.fit_adapter import load_cached_split
    from hapadapter.trunk.loading import find_weights_file, _load_state_dict
    from hapadapter.scoring.stats import balanced_auroc

    cfg = json.load(open(os.path.join(head_dir, "run_config.json")))
    if cfg.get("head_variant") != "twin":
        raise SystemExit(f"{head_dir} is a {cfg.get('head_variant')!r} head; the cached "
                         f"cross-donor path scores twin heads only.")

    xf = cache_transform or cfg.get("cache_transform")
    if xf is None:
        raise SystemExit(
            f"{head_dir}/run_config.json records no 'cache_transform', so the embedding "
            f"preprocessing used at fit time cannot be reproduced. Pass --cache_transform "
            f"explicitly if you know what the fit used.")
    split = load_cached_split(cache, pair_csv, meta_csv, label_col, "depth", cov_col,
                              xf, "crossdonor", min_ok_frac)

    import inspect
    accepted = set(inspect.signature(HapAdapterHead.__init__).parameters) - {"self"}
    head = HapAdapterHead(**{k: v for k, v in cfg.items() if k in accepted})
    head.load_state_dict(_load_state_dict(find_weights_file(head_dir)), strict=True)
    head.eval()

    if cfg.get("d_in") and split["D"] != cfg["d_in"]:
        raise SystemExit(f"cache width {split['D']} != head d_in {cfg['d_in']} -- this cache "
                         f"came from a different backbone or pooling mode.")

    with torch.no_grad():
        ell = head(torch.as_tensor(split["h1"], dtype=torch.float32),
                   torch.as_tensor(split["h2"], dtype=torch.float32)).numpy().reshape(-1)
    y = split["y"].astype(int)
    if y.min() == y.max():
        return float("nan"), float("nan"), len(y), int(y.sum()), 0

    plain = float(roc_auc_score(y, ell))
    cov = split["cov"]
    if cov is None:
        print(f"[cov] WARNING: {meta_csv} has no {cov_col!r} column -- the matched AUROC falls "
              f"back to class-balancing only and is not depth-controlled")
    matched, _ap, _ci, n_match = balanced_auroc(ell, y, cover=cov)
    return float(matched), plain, len(y), int(y.sum()), int(n_match)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True, help="the trained head (train_donor's arm)")
    ap.add_argument("--assay", required=True)
    ap.add_argument("--train_donor", required=True)
    ap.add_argument("--arm", choices=["ref", "personal", "ref_single", "personal_single"], default="ref")
    ap.add_argument("--data_tpl", default=DATA_TPL,
                    help=f"directory holding one donor-arm's HapAdapter training split CSVs. Placeholders: "
                         f"{{exp_root}} {{assay}} {{assay_lower}} {{donor}} {{arm}}. "
                         f"Default: {DATA_TPL}")
    ap.add_argument("--exp_root", required=True)
    ap.add_argument("--donors", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="per-head output CSV (overwritten, not appended)")
    ap.add_argument("--backend", choices=["sequence", "cached"], default="sequence",
                    help="sequence: forward the haplotype pairs through trunk+head or "
                         "cached: read pooled vectors from an embedding cache and apply only "
                         "the head.")
    ap.add_argument("--ag_root", default=None,
                    help="cached backend: root holding emb/. Used by --cache_tpl.")
    ap.add_argument("--cache_tpl", default="{ag_root}/emb/ag_{assay}_{donor}_test_1bp.npz",
                    help="cached backend: per-donor cache path template, {assay} and {donor} substituted.")
    ap.add_argument("--cache_transform", default=None, choices=[None, "auto", "none", "log1p"],
                    help="cached backend: override the transform recorded in the head's "
                         "run_config.json. Only needed for a head fitted before that field "
                         "existed.")
    ap.add_argument("--cov_col", default="total_reads",
                    help="cached backend: coverage column in the META table, for the matched "
                         "AUROC. Default matches fit_adapter")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--verify_folds", action="store_true",
                    help="before scoring, check that for cross-donor leakage and comparability")
    ap.add_argument("--min_ok_frac", type=float, default=0.5,
                    help="cached backend: refuse if fewer than this fraction of the donor's "
                         "test loci are present in the cache. 0 disables the check.")
    ap.add_argument("--bin_size", type=int, default=100000,
                    help="leak-bin size for --verify_folds; must match the scorers' bin_size")
    a = ap.parse_args()

    if a.verify_folds:
        if not verify_folds(a.exp_root, a.assay, a.arm, a.train_donor, a.donors, a.bin_size, data_tpl=a.data_tpl):
            raise SystemExit(
                "FATAL: the head's train/dev coordinates overlap an eval donor's test set -- "
                "the cross-donor AUROCs would be leaky! Rebuild the train/dev/test folds, or drop the "
                "offending donor from --donors.")

    if a.backend == "cached" and not a.ag_root and "{ag_root}" in a.cache_tpl:
        raise SystemExit("--backend cached needs --ag_root (or a --cache_tpl without "
                         "{ag_root} in it)")
    rows = []

    for ed in a.donors:
        sd = split_dir(a.exp_root, a.assay, ed, a.arm, a.data_tpl)
        tcsv = os.path.join(sd, "test.csv")
        if not os.path.exists(tcsv):
            print(f"[skip] {a.assay}/{ed} {a.arm}: no test.csv at {tcsv}")
            continue

        if a.backend == "cached":
            cache = a.cache_tpl.format(ag_root=a.ag_root or "", assay=a.assay.lower(), donor=ed)
            if not os.path.exists(cache):
                print(f"[skip] {a.assay}/{ed}: no cache at {cache}")
                continue
            mcsv = os.path.join(sd, "test.meta.csv")
            if not os.path.exists(mcsv):
                print(f"[skip] {a.assay}/{ed}: no meta beside {tcsv}")
                continue
            au, au_plain, n, npos, n_match = score_cached(
                a.checkpoint_dir, cache, tcsv, mcsv, a.cov_col,
                cache_transform=a.cache_transform, min_ok_frac=a.min_ok_frac)
        else:
            au, au_plain, n, npos, n_match = score_on(
                a.checkpoint_dir, tcsv, a.batch_size, a.device)

        kind = "matched" if ed == a.train_donor else "cross-donor"
        rows.append(dict(assay=a.assay, train_donor=a.train_donor, eval_donor=ed, arm=a.arm,
                         head_variant=_head_variant(a.checkpoint_dir), donor_kind=kind,
                         auroc_cov_matched=au, auroc_plain=au_plain,
                         n=n, n_pos=npos, n_cov_matched=n_match))
        print(f"  [{a.assay}] {a.train_donor} -> {ed} (donor: {kind}) {a.arm}: "
              f"AUROC cov-matched={au:.4f} plain={au_plain:.4f} "
              f"n={n} n_pos={npos} n_cov_matched={n_match}")

    if not rows:
        print("[warn] no rows produced (no test.csv found)"); return

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"[write] {a.out} ({len(rows)} rows)")

if __name__ == "__main__":
    main()
