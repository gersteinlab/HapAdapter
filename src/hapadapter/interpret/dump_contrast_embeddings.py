#!/usr/bin/env python
"""
hapadapter.interpret.dump_contrast_embeddings -- dump per-locus contrast representations

  * TRUNK contrast: h1 - h2      (pool_ref - pool_alt, dimension = d_model)
        -- what the frozen trunk already produces, BEFORE the stage 2 learned projection

  * HEAD contrast: P(h1) - P(h2) (dimension = proj_dim)

Also save: the model's twin classification logit ell (= a*s + b = logit P(AS)) and the binary AS label
(if the input CSV carries total_reads / chr / anchor, they are also passed through to enable leak filtering)

Input:
a HapAdapter build CSV (columns: sequence1, sequence2, label[, depth])
OR an ASB-windows CSV (sequence1, sequence2, as_label, total_reads, chr, anchor)
"""
import argparse, csv, os, sys
import numpy as np

def _write_npz(out_path, *, labels, ell, trunk_contrast, head_contrast, total_reads,
               chrom, anchor, checkpoint_dir, data_csv, extra):
    n = len(labels)
    wrong = [f"{nm} ({len(v)})" for nm, v in
             (("ell", ell), ("trunk_contrast", trunk_contrast), ("head_contrast", head_contrast),
              ("total_reads", total_reads), ("chrom", chrom), ("anchor", anchor))
             if len(v) != n]
    if wrong:
        raise SystemExit(f"row mismatch against {n} labels: {', '.join(wrong)}. A selection "
                         f"step (restrict / subsample) filtered some arrays and not others.")
    # tmp + replace in cash of mid-run crash
    tmp = out_path + ".tmp.npz"
    np.savez_compressed(
        tmp,
        labels=np.asarray(labels).astype(np.int8),
        ell=np.asarray(ell, dtype=np.float32),
        trunk_contrast=np.asarray(trunk_contrast, dtype=np.float32),
        head_contrast=np.asarray(head_contrast, dtype=np.float32),
        trunk_norm=np.linalg.norm(trunk_contrast, axis=1).astype(np.float32),
        head_norm=np.linalg.norm(head_contrast, axis=1).astype(np.float32),
        total_reads=np.asarray(total_reads, dtype=np.float32),
        chrom=np.asarray(chrom),
        anchor=np.asarray(anchor),
        checkpoint_dir=np.array(checkpoint_dir),
        data_csv=np.array(data_csv),
        **extra,
    )
    os.replace(tmp, out_path)
    print(f"[write] {out_path}")

def _cached(args):
    """For the cached backend"""
    import json
    import os
    import torch
    from hapadapter.adapter.head import HapAdapterHead
    from hapadapter.fit_adapter import load_cached_split
    from hapadapter.trunk.loading import find_weights_file, _load_state_dict

    cfg = json.load(open(os.path.join(args.checkpoint_dir, "run_config.json")))
    if cfg.get("head_variant") != "twin":
        raise SystemExit(
            f"{args.checkpoint_dir} is a {cfg.get('head_variant')!r} head. "
            f"Only pass twin (contrast) heads here.")

    xf = args.cache_transform or cfg.get("cache_transform")
    if xf is None:
        raise SystemExit(
            f"{args.checkpoint_dir}/run_config.json records no 'cache_transform', so the "
            f"embedding preprocessing used at fit time cannot be reproduced. Pass "
            f"--cache_transform explicitly if you know what the fit used.")
    if args.cache_transform and cfg.get("cache_transform") not in (None, args.cache_transform):
        print(f"[warn] --cache_transform {args.cache_transform!r} overrides the "
              f"{cfg['cache_transform']!r} recorded in the head's run_config. The recorded "
              f"value is what the head was trained under.")

    split = load_cached_split(args.cache, args.pair_csv, args.meta_csv, args.label_col,
                              "depth", "depth", xf, "dump", 0.0)

    if cfg.get("d_in") and split["D"] != cfg["d_in"]:
        raise SystemExit(
            f"cache width {split['D']} != the head's d_in {cfg['d_in']}. This cache came from "
            f"a different backbone, or a different pooling mode of the same one.")

    import inspect
    accepted = set(inspect.signature(HapAdapterHead.__init__).parameters) - {"self"}
    head = HapAdapterHead(**{k: v for k, v in cfg.items() if k in accepted})
    head.load_state_dict(_load_state_dict(find_weights_file(args.checkpoint_dir)), strict=True)
    head.eval()

    h1 = torch.as_tensor(split["h1"], dtype=torch.float32)
    h2 = torch.as_tensor(split["h2"], dtype=torch.float32)

    with torch.no_grad():
        ell = head(h1, h2).numpy()
        head_contrast = (head.proj(h1) - head.proj(h2)).numpy()
    trunk_contrast = (h1 - h2).numpy()

    extra = {}
    sd = head.state_dict()
    if "dist_a" in sd and "dist_b" in sd:
        a_raw = float(sd["dist_a"].reshape(-1)[0]); b = float(sd["dist_b"].reshape(-1)[0])
        a = float(torch.nn.functional.softplus(torch.tensor(a_raw)))
        extra = {"a_raw": np.float32(a_raw), "a": np.float32(a), "b": np.float32(b)}
        print(f"[dump] head scalars: a_raw={a_raw:.4g} -> a={a:.4g} | b={b:.4g} "
              f"| decision radius -b/a={-b / a:.4g}")

    y = (split["y"] > 0.5).astype(int)
    print(f"[dump] cached backend | {len(y)} loci | pos={int(y.sum())} "
          f"neg={int((y == 0).sum())} | d_in={split['D']} transform={xf}")
    return dict(labels=y, ell=ell, trunk_contrast=trunk_contrast,
                head_contrast=head_contrast,
                total_reads=(split["cov"] if split["cov"] is not None
                             else np.zeros(len(y), dtype=np.float32)),
                chrom=split["chrom"], anchor=split["pos"],
                checkpoint_dir=args.checkpoint_dir, data_csv=args.pair_csv, extra=extra)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["sequence", "cached"], default="sequence",
                    help="sequence: forward the haplotype pairs through the trunk+head (needs "
                         "a torch encoder). cached: read the pooled vectors "
                         "from an embedding cache and apply only the head.")
    ap.add_argument("--checkpoint_dir", required=True, help="trained classification run directory")
    ap.add_argument("--data_csv", help="sequence backend: build CSV (sequence1,sequence2,label)")
    ap.add_argument("--cache", help="cached backend: the embedding cache npz")
    ap.add_argument("--pair_csv", help="cached backend: the split's pair CSV (labels)")
    ap.add_argument("--meta_csv", help="cached backend: the split's meta CSV (coordinates)")
    ap.add_argument("--cache_transform", default=None,
                    help="cached backend: override the transform recorded in the head's "
                         "run_config. Only pass this if you know the fit used something else; "
                         "the recorded value is the one the head was trained under.")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_rows", type=int, default=0,
                    help="0 = all rows; else cap number of rows by randomly subsetting negatives")
    ap.add_argument("--neg_per_pos", type=float, default=None,
                    help="negatives to keep PER POSITIVE (1.0 = balanced); overrides --max_rows if given")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label_col", default="label",
                    help="label column; numeric (e.g. 0.5 for '> 0.5 => positive') OR categorical AS/nonAS (as_label)")
    ap.add_argument("--restrict_loci", default=None,
                    help="sequence backend: a pair cache (.npz) or per-locus CSV; keep ONLY "
                         "these (chr, pos) loci.")
    ap.add_argument("--coord_csv", default=None,
                    help="sequence backend: row-aligned meta CSV supplying chr/anchor (and "
                         "total_reads) when --data_csv lacks them. "
                         "Required by --restrict_loci unless the "
                         "build CSV already carries coordinates.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    if args.backend == "cached":
        missing = [f"--{f}" for f in ("cache", "pair_csv", "meta_csv")
                   if not getattr(args, f)]
        if missing:
            raise SystemExit(f"--backend cached needs {', '.join(missing)}")
        _write_npz(args.out, **_cached(args))
        return
    if not args.data_csv:
        raise SystemExit("--backend sequence needs --data_csv")

    from hapadapter.loading import run_inference

    with open(args.data_csv) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"empty CSV: {args.data_csv}")
    cols = rows[0].keys()
    for need in ("sequence1", "sequence2", args.label_col):
        if need not in cols:
            sys.exit(f"{args.data_csv} missing column '{need}'; has {list(cols)}")

    # allow for label options
    def _lab(v):
        s = str(v).strip()
        if s.lower() in ("as", "1", "true"):  return 1
        if s.lower() in ("nonas", "non-as", "non_as", "non as", "0", "false"):  return 0
        try:
            return int(float(s) > 0.5) # fallback to predicting AS if logit > 0.5, non-AS if < 0.5
        except ValueError:
            raise SystemExit(f"--label_col {args.label_col!r}: unrecognised value {v!r}; expected "
                             f"AS/nonAS/1/0/true/false or a number")

    labels = np.array([_lab(r[args.label_col]) for r in rows], dtype=int)

    coord = None
    if args.coord_csv:
        with open(args.coord_csv, newline="") as fh:
            coord = list(csv.DictReader(fh))
        if len(coord) != len(rows):
            sys.exit(f"--coord_csv has {len(coord)} rows, --data_csv has {len(rows)}. These "
                     f"are written as a matched pair; a mismatch means one was regenerated "
                     f"or subsampled separately.")

    def _col(name, default, cast):
        src = coord if (coord is not None and name in coord[0]) else (rows if name in cols else None)
        if src is None:
            return np.array([default] * len(rows))
        return np.array([cast(r.get(name) or default) for r in src])

    if args.restrict_loci:
        from hapadapter.scoring.score_matched import _loci_from
        want = _loci_from(args.restrict_loci, "restrict_loci")
        _c = _col("chr", "?", str)
        _p = _col("anchor", 0, lambda v: int(float(v))).astype(np.int64)
        if set(_c) == {"?"}:
            sys.exit("--restrict_loci needs coordinates, but neither --data_csv nor "
                     "--coord_csv carries a 'chr' column. Pass --coord_csv <build>.meta.csv.")
        sel = np.array([(str(c), int(p)) in want for c, p in zip(_c, _p)])
        if not sel.any():
            sys.exit(f"ZERO of {len(rows)} rows are in --restrict_loci ({len(want)} loci). "
                     f"Data has e.g. {list(zip(_c, _p))[:2]}, restriction has e.g. "
                     f"{sorted(want)[:2]}.")
        rows = [r for r, k in zip(rows, sel) if k]
        if coord is not None:
            coord = [r for r, k in zip(coord, sel) if k]
        labels = labels[sel]
        print(f"[restrict] {int(sel.sum())} of {len(sel)} rows kept ({len(want)} requested) "
              f"| {int((labels == 1).sum())} pos {int((labels == 0).sum())} neg")

    idx = np.arange(len(rows)) # row indices the subsample selects from
    
    # optional subsample to bound GPU cost
    if args.neg_per_pos is not None or (args.max_rows and args.max_rows < len(rows)):
        rng = np.random.default_rng(args.seed)
        pos, neg = idx[labels == 1], idx[labels == 0]
        if args.neg_per_pos is not None:
            n_neg_keep = int(round(args.neg_per_pos * len(pos)))
            mode = f"--neg_per_pos {args.neg_per_pos}"
        else:
            n_neg_keep = max(0, args.max_rows - len(pos))
            mode = f"--max_rows {args.max_rows}"

        neg_keep = rng.choice(neg, size=min(n_neg_keep, len(neg)), replace=False)
        idx = np.sort(np.concatenate([pos, neg_keep]))
        rows = [rows[i] for i in idx]
        labels = labels[idx]

        if coord is not None:
            coord = [coord[i] for i in idx]

        print(f"[subsample] {mode} -> kept {len(rows)} rows "
              f"({int((labels==1).sum())} pos, {int((labels==0).sum())} neg)")

        if (labels == 1).sum() == 0 or (labels == 0).sum() == 0:
            raise SystemExit(
                f"{mode} left a SINGLE-CLASS npz ({int((labels==1).sum())} pos, "
                f"{int((labels==0).sum())} neg) -- every consumer computes an AUROC and would "
                f"return nan. Use --neg_per_pos 1.0, or raise --max_rows above the positive count.")

    texts = [[r["sequence1"], r["sequence2"]] for r in rows]
    print(f"[in] {len(texts)} loci | pos={int((labels==1).sum())} neg={int((labels==0).sum())}")

    # carry coverage and coordinates through to the npz if present
    total_reads = _col("total_reads", 0.0, float).astype(np.float32)
    chrom = _col("chr", "?", str)
    anchor = _col("anchor", 0, lambda v: int(float(v))).astype(np.int64)

    logits, emb, pool_ref, pool_alt, rc = run_inference(
        args.checkpoint_dir, texts, batch_size=args.batch_size,
        device=args.device, dump_pools=True)

    logits = np.asarray(logits, dtype=np.float32).reshape(-1)     # ell = logit P(AS)
    head_contrast = np.asarray(emb, dtype=np.float32)             # (N, proj_dim)
    pool_ref = np.asarray(pool_ref, dtype=np.float32)             # (N, H)
    pool_alt = np.asarray(pool_alt, dtype=np.float32)             # (N, H)
    trunk_contrast = pool_ref - pool_alt                          # (N, H)

    assert head_contrast.shape[0] == trunk_contrast.shape[0] == len(labels), "row mismatch"
    print(f"[dump] trunk_contrast {trunk_contrast.shape} | head_contrast {head_contrast.shape} "
          f"| task={rc.get('task')} proj_dim={rc.get('proj_dim')}")

    # Stash the trained logistic-link scalars a > 0 and b for plotting
    extra = {}
    try:
        import torch
        from hapadapter.trunk.loading import find_weights_file, _load_state_dict
        sd = _load_state_dict(find_weights_file(args.checkpoint_dir))
        if "dist_a" in sd and "dist_b" in sd:
            a_raw = float(sd["dist_a"].reshape(-1)[0])
            b = float(sd["dist_b"].reshape(-1)[0])
            a = float(torch.nn.functional.softplus(torch.tensor(a_raw)))
            extra = {"a_raw": np.float32(a_raw), "a": np.float32(a), "b": np.float32(b)}
            print(f"[dump] head scalars: a_raw={a_raw:.4g} -> a={a:.4g} | b={b:.4g} "
                  f"| decision radius -b/a={-b/a:.4g}")
        else:
            print(f"[dump] no dist_a/dist_b in the checkpoint: this is a single-variant head, "
                  f"which has no distance scalars to read.")
    except Exception as e:
        print(f"[dump] could not read a/b from checkpoint ({e}); plot will need --a_raw/--b")

    _write_npz(args.out, labels=labels, ell=logits, trunk_contrast=trunk_contrast,
               head_contrast=head_contrast, total_reads=total_reads, chrom=chrom,
               anchor=anchor, checkpoint_dir=args.checkpoint_dir, data_csv=args.data_csv,
               extra=extra)

if __name__ == "__main__":
    main()
