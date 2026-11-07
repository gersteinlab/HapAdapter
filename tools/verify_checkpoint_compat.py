#!/usr/bin/env python
"""
verify_checkpoint_compat.py -- prove the refactor did not touch a single trained model.

Loads EVERY checkpoint under --checkpoints twice: once with the pre-migration package, once
with the post-migration package, and compares them. Exits nonzero on any difference, so it
can gate the merge.

THIS IS THE POINT. The HapAdapter refactor shares the head's construction and forward math
between trunk/model.py and adapter/head.py, but deliberately does NOT move any attribute:
`backbone`, `proj`, `dist_a`, `dist_b`, `main_head` keep their names, because those names ARE
the state_dict keys. The claim "nothing you trained is affected" is therefore checkable rather
than argued -- and this script is the check.

THREE LEVELS, cheapest first. Level 1 alone catches any key rename, which is the only way a
checkpoint can actually be invalidated.

  1. KEYS      state_dict key sets identical, and every tensor equal elementwise.
  2. HEAD      head-only forward on fixed random pooled vectors, bit-identical.
  3. FULL      whole-model forward incl. backbone (--full; needs the backbone weights).

eval() is forced everywhere: dropout active would make even a correct refactor differ.

Usage:
  python verify_checkpoint_compat.py \
      --old_src $HOME/entexBERT-2-premigration/src --new_src $HOME/HapAdapter/src \
      --checkpoints '$HOME/palmer_scratch/entexbert2-models/experiments/*/stage*/runs/*' \
      --full
"""
import argparse
import glob
import importlib
import os
import sys
import json

import torch


def _import_pkg(src_dir, pkg, alias):
    """Import `pkg` from `src_dir` under a private alias so both versions coexist."""
    for m in [k for k in list(sys.modules) if k == pkg or k.startswith(pkg + ".")]:
        del sys.modules[m]
    sys.path.insert(0, src_dir)
    try:
        mod = importlib.import_module(pkg)
        sys.modules[alias] = mod
        return mod
    finally:
        sys.path.pop(0)


def _state_dict(path):
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and "backbone.embeddings" not in str(sd.keys()):
        sd = sd["state_dict"]
    return sd


def _weights_file(d):
    for n in ("pytorch_model.bin", "model.safetensors"):
        p = os.path.join(d, n)
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old_src", required=True, help="src/ of the PRE-migration package")
    ap.add_argument("--new_src", required=True, help="src/ of the POST-migration package")
    ap.add_argument("--old_pkg", default="entexbert2")
    ap.add_argument("--new_pkg", default="hapadapter")
    ap.add_argument("--checkpoints", required=True, help="glob of checkpoint DIRECTORIES")
    ap.add_argument("--full", action="store_true",
                    help="also compare a whole-model forward (loads the backbone; slower)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strict", action="store_true",
                    help="FAIL if any directory holds weights but no run_config.json. Across a "
                         "44-cell sweep a silent skip lets an unverified checkpoint through, "
                         "which is the failure mode this script exists to prevent.")
    ap.add_argument("--report", default=None)
    a = ap.parse_args()

    allm = sorted(d for d in glob.glob(os.path.expandvars(a.checkpoints)) if os.path.isdir(d))
    with_w = [d for d in allm if _weights_file(d)]
    dirs = [d for d in with_w if os.path.exists(os.path.join(d, "run_config.json"))]
    orphan = [d for d in with_w if d not in dirs]
    if orphan:
        print(f"[orphan] {len(orphan)} director{'y holds' if len(orphan)==1 else 'ies hold'} "
              f"weights but NO run_config.json:")
        for d in orphan[:12]:
            print(f"    {d}")
        if len(orphan) > 12:
            print(f"    ... and {len(orphan)-12} more")
        if a.strict:
            raise SystemExit(
                "--strict: refusing to report a pass while real checkpoints go unchecked. "
                "build_model needs the config to rebuild the architecture, so these cannot be "
                "verified until each has one.")
        print("  -> these are SKIPPED. Re-run with --strict to make that a failure.\n")
    if not dirs:
        raise SystemExit(f"no checkpoint dirs with weights + run_config.json matched "
                         f"{a.checkpoints!r} -- nothing to verify, which is NOT a pass")
    print(f"[scan] {len(dirs)} checkpoints to verify"
          f"{f', {len(orphan)} skipped' if orphan else ''}\n")

    old = _import_pkg(a.old_src, a.old_pkg, "_OLD")
    new = _import_pkg(a.new_src, a.new_pkg, "_NEW")
    old_io = importlib.import_module(f"{a.old_pkg}.model_io")
    sys.modules.pop(f"{a.new_pkg}.model_io", None)
    sys.path.insert(0, a.new_src)
    new_io = importlib.import_module(f"{a.new_pkg}.model_io")
    sys.path.pop(0)

    rows, bad = [], 0
    for d in dirs:
        name = os.path.relpath(d)
        rc = json.load(open(os.path.join(d, "run_config.json")))
        task, var = rc.get("task", "regression"), rc.get("head_variant", "twin")
        rec = dict(checkpoint=name, task=task, head_variant=var,
                   keys="-", tensors="-", head="-", full="-")

        mo = old_io.build_model(rc, device="cpu")
        mn = new_io.build_model(rc, device="cpu")

        ko, kn = sorted(mo.state_dict()), sorted(mn.state_dict())
        rec["keys"] = "OK" if ko == kn else "MISMATCH"
        if ko != kn:
            rec["detail"] = (f"only-old={sorted(set(ko)-set(kn))[:6]} "
                             f"only-new={sorted(set(kn)-set(ko))[:6]}")
            bad += 1
            rows.append(rec); print(f"  {name}: KEY MISMATCH -- {rec['detail']}"); continue

        # ---- level 1: the saved tensors land in the same places, bit for bit --------------
        sd = _state_dict(_weights_file(d))
        old_io.load_model_weights(mo, _weights_file(d))
        new_io.load_model_weights(mn, _weights_file(d))
        so, sn = mo.state_dict(), mn.state_dict()
        diff = [k for k in so if not torch.equal(so[k], sn[k])]
        rec["tensors"] = "OK" if not diff else f"DIFFER x{len(diff)}"
        if diff:
            rec["detail"] = f"first differing: {diff[:5]}"
            bad += 1

        # ---- level 2: head-only forward on fixed pooled vectors ---------------------------
        mo.eval(); mn.eval()
        torch.manual_seed(a.seed)
        D = getattr(getattr(mo, "backbone", None), "config", None)
        D = getattr(D, "hidden_size", None) or rc.get("hidden_size") or 768
        p1, p2 = torch.randn(4, D), torch.randn(4, D)
        with torch.no_grad():
            def head_fwd(m):
                if task == "classification" and var == "twin":
                    z1, z2 = m.proj(m.dropout(p1)), m.proj(m.dropout(p2))
                    s = torch.linalg.vector_norm(z1 - z2, dim=-1)
                    return torch.nn.functional.softplus(m.dist_a) * s + m.dist_b
                return m.main_head(m.dropout(p1))
            ho, hn = head_fwd(mo), head_fwd(mn)
        rec["head"] = "OK" if torch.equal(ho, hn) else "DIFFER"
        if not torch.equal(ho, hn):
            rec["detail"] = f"max|dh| = {(ho-hn).abs().max().item():.3e}"
            bad += 1

        # ---- level 3: whole model, backbone included --------------------------------------
        if a.full:
            torch.manual_seed(a.seed)
            L = int(rc.get("model_max_length", 64))
            ids = torch.randint(5, 100, (2, min(L, 64)))
            msk = torch.ones_like(ids)
            with torch.no_grad():
                try:
                    lo = old_io.logits_and_embeddings(mo, ids, msk, ids, msk)[0]
                    ln = new_io.logits_and_embeddings(mn, ids, msk, ids, msk)[0]
                    rec["full"] = "OK" if torch.equal(lo, ln) else "DIFFER"
                    if not torch.equal(lo, ln):
                        rec["detail"] = f"max|dlogit| = {(lo-ln).abs().max().item():.3e}"
                        bad += 1
                except Exception as e:
                    rec["full"] = f"ERROR {type(e).__name__}"
                    bad += 1

        rows.append(rec)
        print(f"  {name:58} keys={rec['keys']:9} tensors={rec['tensors']:11} "
              f"head={rec['head']:7} full={rec['full']}")

    print(f"\n=== {len(rows)} checkpoints verified, {bad} problem(s)"
          f"{f', {len(orphan)} SKIPPED for want of a run_config.json' if orphan else ''}")
    if a.report:
        import pandas as pd
        pd.DataFrame(rows).to_csv(a.report, index=False)
        print(f"[write] {a.report}")
    if bad:
        print("  DO NOT MERGE. Every row above must read OK; a single key rename silently")
        print("  invalidates the checkpoint it names.")
        return 1
    if orphan:
        # Not an unqualified pass: a clean result over a subset says nothing about the rest.
        print(f"  The {len(rows)} verified checkpoints load identically under both packages.")
        print(f"  {len(orphan)} were NOT checked (no run_config.json). Safe to merge only if")
        print(f"  you accept those as out of scope -- --strict makes that a failure instead.")
    else:
        print("  All checkpoints load identically under both packages. Safe to merge.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
