#!/usr/bin/env python
"""
hapadapter.backbones.dnabert2 -- pooled haplotype embeddings from the pre-trained DNABERT-2 backbone
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default=None,
                    help="HuggingFace DNABERT-2 checkpoint dir (no run_config.json needed)")
    ap.add_argument("--trunk_config", required=True,
                    help="Functional fine-tuned trunk run_config.json")
    ap.add_argument("--pair_csv", required=True, help="build split csv (sequence1/sequence2)")
    ap.add_argument("--pair_meta", required=True, help="its .meta.csv (chr/anchor)")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoint_dir", default=None,
                    help="a fine-tuned checkpoint to pool from")
    ap.add_argument("--restrict_loci", default=None,
                    help="a pair cache (.npz) or per-locus CSV; extract only these loci "
                         "to pool over and score")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-extract even if --out exists")
    ap.add_argument("--no_verify_pretrained", action="store_true",
                    help="skip the random-initialisation check (NOT recommended)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if bool(a.raw_dir) == bool(a.checkpoint_dir):
        raise SystemExit("give exactly one of --raw_dir (pre-trained only) "
                         "or --checkpoint_dir (fine-tuned on a functional task).")

    _out = a.out if a.out.endswith(".npz") else a.out + ".npz"
    if os.path.exists(_out) and not a.overwrite:
        print(f"[skip] {_out} exists (use --overwrite)")
        return

    from hapadapter.backbones.cache import pair_key, write_pair_cache
    from hapadapter.trunk.loading import build_model, load_tokenizer

    with open(a.trunk_config) as f:
        trunk_rc = json.load(f)
    
    POOL_KEYS = ("pooling_mode", "center_pool_width", "model_max_length")
    pooling = {k: trunk_rc[k] for k in POOL_KEYS if k in trunk_rc}
    missing = [k for k in ("pooling_mode",) if k not in pooling]

    if missing:
        raise SystemExit(f"{a.trunk_config} has no {missing} -- cannot guarantee the raw bar is "
                         f"pooled like the fine-tuned bar. Please pass a trunk config that has it.")
    print(f"[pool] copied from trunk config: {pooling}")

    rc = dict(model_name_or_path=a.raw_dir, task="regression", num_labels=1,
              head_num_layers=1, head_hidden_size=-1, head_variant="single", **pooling)
    rc.setdefault("center_pool_width", 5)

    if a.checkpoint_dir: # fine-tuned
        from hapadapter.loading import load_model_and_tokenizer
        model, tok, rc = load_model_and_tokenizer(a.checkpoint_dir, device=a.device)
        src_desc = f"fine-tuned checkpoint {a.checkpoint_dir}"
    else: # raw pre-trained
        model = build_model(rc, device=a.device)
        tok = load_tokenizer(rc)
        src_desc = f"pretrained {a.raw_dir}"
    model.eval()
    print(f"[model] backbone from {src_desc}")

    if a.raw_dir and not a.no_verify_pretrained:
        import transformers
        ref = transformers.AutoModel.from_pretrained(a.raw_dir, trust_remote_code=True)
        rsd = ref.state_dict()
        msd = {k: v for k, v in model.state_dict().items()}
        hits, checked = 0, 0
        for rk, rv in list(rsd.items()):
            if rv.ndim < 2 or rv.numel() < 1000:
                continue
            cand = [mk for mk in msd if mk.endswith(rk) and msd[mk].shape == rv.shape]
            if not cand:
                continue
            checked += 1
            if torch.allclose(msd[cand[0]].detach().cpu().float(), rv.detach().cpu().float(), atol=1e-5):
                hits += 1
            if checked >= 12:
                break
        del ref, rsd
        frac = hits / max(checked, 1)
        print(f"[verify] {hits}/{checked} large backbone tensors match "
              f"AutoModel.from_pretrained({os.path.basename(a.raw_dir)})")
        if checked == 0 or frac < 0.9:
            raise SystemExit(
                f"BACKBONE IS NOT THE PRETRAINED ONE ({hits}/{checked} matched). "
                f"May be randomly initialized! Check that "
                f"--raw_dir holds real DNABERT-2 weights.")

    df = pd.read_csv(a.pair_csv)
    meta = pd.read_csv(a.pair_meta)

    if len(df) != len(meta):
        raise SystemExit(f"pair_csv has {len(df)} rows, pair_meta has {len(meta)} but these must "
                         f"come from the same split!")

    for c in ("sequence1", "sequence2"):
        if c not in df.columns:
            raise SystemExit(f"{a.pair_csv} has no '{c}'; present: {list(df.columns)[:12]}. "
                             f"A hap_pair build is required.")

    cc = "chr" if "chr" in meta.columns else "chrom"
    pc = next((c for c in ("anchor", "pos", "ref_start") if c in meta.columns), None)
    if pc is None:
        raise SystemExit(f"{a.pair_meta} has no anchor/pos/ref_start; present: {list(meta.columns)}")

    # pool embeddings
    @torch.no_grad()
    def pool_all(seqs):
        out = []
        for i in range(0, len(seqs), a.batch_size):
            b = [str(x) for x in seqs[i:i + a.batch_size]]
            enc = tok(b, return_tensors="pt", padding=True, truncation=True)
            ids = enc["input_ids"].to(a.device)
            msk = enc["attention_mask"].to(a.device)
            out.append(model._pool_one(ids, msk, apply_dropout=False)
                       .detach().cpu().float().numpy())
            if i and (i // a.batch_size) % 200 == 0:
                print(f"  ... {i:,}/{len(seqs):,}", flush=True)
        return np.concatenate(out, 0)

    if a.restrict_loci:
        if a.restrict_loci.endswith(".npz"):
            _d = np.load(a.restrict_loci, allow_pickle=True)
            want = set()
            for k in np.asarray(_d["keys"]).astype(str):
                _c, _p, _h = k.rsplit(":", 2)
                want.add((_c, int(_p)))
        else:
            _t = pd.read_csv(a.restrict_loci)
            _cc = "chr" if "chr" in _t.columns else _t.columns[0]
            _pc = next(c for c in ("pos", "anchor") if c in _t.columns)
            want = set(zip(_t[_cc].astype(str), _t[_pc].astype(int)))
        sel = np.array([(str(c), int(p)) in want for c, p in zip(meta[cc], meta[pc])])
        if not sel.any():
            raise SystemExit(
                f"ZERO of {len(meta)} build loci are in --restrict_loci ({len(want)} loci). "
                f"Build has e.g. {list(zip(meta[cc], meta[pc]))[:2]}, restriction has e.g. "
                f"{sorted(want)[:2]}.")
        df, meta = df[sel].reset_index(drop=True), meta[sel].reset_index(drop=True)
        print(f"[restrict] {int(sel.sum())} of {len(sel)} build loci kept ({len(want)} requested)")

    print(f"[infer] h1: {len(df):,} windows")
    p1 = pool_all(df["sequence1"].tolist())
    print(f"[infer] h2: {len(df):,} windows")
    p2 = pool_all(df["sequence2"].tolist())

    # output the cached embeddings
    keys, vals = [], []
    for i, r in enumerate(meta.itertuples(index=False)):
        c, p = getattr(r, cc), int(getattr(r, pc))
        keys.append(pair_key(c, p, "h1")); vals.append(p1[i])
        keys.append(pair_key(c, p, "h2")); vals.append(p2[i])

    out = a.out if a.out.endswith(".npz") else a.out + ".npz"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    write_pair_cache(out, keys, np.asarray(vals, np.float32))
    print(f"[write] {out} ({len(keys)} sequences x {p1.shape[1]} dims)")

if __name__ == "__main__":
    main()
