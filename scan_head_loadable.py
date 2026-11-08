#!/usr/bin/env python
"""
scan_head_loadable.py -- can every trained Stage-2 head be loaded into a bare HapAdapterHead?

THE QUESTION THIS SETTLES. Moving Stage-2 training out of trunk/model.py and into
adapter/train.py changes how heads are PRODUCED. It must not change whether the heads already
trained are still usable, because re-training the 88-checkpoint suite is not on the table.

WHY IT SHOULD WORK, AND WHY THAT IS NOT ENOUGH. HapAdapterHead binds its parameters to the
same attribute names the flat checkpoints use -- proj / dist_a / dist_b for twin, main_head
for single -- so the head's keys are a SUBSET of a flat checkpoint's keys and a strict load of
the prefix-filtered subset should succeed with no remap. That is an argument. This script is
the measurement, on the real files, and it is the gate before trusting the suite.

STRICT ON PURPOSE. strict=True is the whole point: strict=False would quietly accept a
checkpoint whose head does not match the architecture and hand back a partly-random head,
which is the failure this exists to catch.

It also records the head's SHAPES, so a cell whose proj width differs from its siblings shows
up here rather than as a confusing error later.

Usage:
  python scan_head_loadable.py --root $HOME/palmer_scratch/entexbert2-models/experiments
  python scan_head_loadable.py --root ... --report head_loadable.csv

Reads every checkpoint, so submit it rather than running on a login node.
"""
import argparse
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.environ.get("HAPADAPTER_SRC", ""))
from hapadapter.adapter.head import HapAdapterHead  # noqa: E402

WEIGHTS = ("pytorch_model.bin", "model.safetensors")
TWIN_PREFIXES = ("proj", "dist_a", "dist_b")
SINGLE_PREFIXES = ("main_head",)


def weights_file(d):
    for w in WEIGHTS:
        p = os.path.join(d, w)
        if os.path.exists(p):
            return p
    return None


def load_sd(path):
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    sd = torch.load(path, map_location="cpu")
    return sd.get("state_dict", sd) if isinstance(sd, dict) and "state_dict" in sd else sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--report", default=None)
    a = ap.parse_args()

    cfgs = sorted(glob.glob(os.path.join(a.root, "**", "run_config.json"), recursive=True))
    rows, n_ok, n_bad = [], 0, 0
    for c in cfgs:
        d = os.path.dirname(c)
        rc = json.load(open(c))
        if rc.get("task") != "classification":
            continue                      # Stage-1 trunks keep living in trunk/, untouched
        w = weights_file(d)
        name = os.path.relpath(d, a.root)
        if w is None:
            print(f"  SKIP      {name}: no weights file")
            continue

        sd = load_sd(w)
        # Infer the variant from the WEIGHTS, not from run_config: only 2 of 134 checkpoints
        # record head_variant, so trusting the config would mislabel almost every cell.
        if any(k.startswith("proj.") for k in sd):
            variant, pre = "twin", TWIN_PREFIXES
        elif any(k.startswith("main_head.") for k in sd):
            variant, pre = "single", SINGLE_PREFIXES
        else:
            print(f"  NO HEAD   {name}: neither proj.* nor main_head.* present")
            rows.append(dict(cell=name, variant="?", ok=0, detail="no head keys"))
            n_bad += 1
            continue

        sub = {k: v for k, v in sd.items() if k.split(".")[0] in pre}
        # Head geometry comes from the weights too, so a non-default cell is handled rather
        # than assumed away.
        if variant == "twin":
            pw = [sub[k] for k in sub if k.endswith("proj.weight") or k == "proj.weight"]
            lin = sorted((k for k in sub if k.endswith(".weight") or k == "proj.weight"))
            d_in = sub[lin[0]].shape[1]
            proj_dim = sub[lin[-1]].shape[0]
            n_layers = len(lin)
            hidden = sub[lin[0]].shape[0] if n_layers > 1 else -1
            kw = dict(d_in=int(d_in), head_variant="twin", proj_dim=int(proj_dim),
                      head_num_layers=n_layers, head_hidden_size=int(hidden))
            shape = f"d_in={d_in} proj_dim={proj_dim} layers={n_layers}"
        else:
            lin = sorted(k for k in sub if k.endswith(".weight"))
            d_in = sub[lin[0]].shape[1]
            n_layers = len(lin)
            hidden = sub[lin[0]].shape[0] if n_layers > 1 else -1
            kw = dict(d_in=int(d_in), head_variant="single", head_num_layers=n_layers,
                      head_hidden_size=int(hidden))
            shape = f"d_in={d_in} layers={n_layers}"

        try:
            head = HapAdapterHead(**kw)
            head.load_state_dict(sub, strict=True)
            print(f"  OK        {name}  [{variant}]  {shape}")
            rows.append(dict(cell=name, variant=variant, ok=1, detail=shape))
            n_ok += 1
        except Exception as e:
            msg = str(e).splitlines()[0][:110]
            print(f"  FAIL      {name}  [{variant}]  {msg}")
            rows.append(dict(cell=name, variant=variant, ok=0, detail=msg))
            n_bad += 1

    print(f"\n=== {n_ok} loadable, {n_bad} failed, of {n_ok + n_bad} Stage-2 checkpoints")
    if n_ok and not n_bad:
        print("  Every trained head loads into the refactored HapAdapterHead with strict=True")
        print("  and no key remap, so the suite does NOT need re-training. New checkpoints")
        print("  will be head-only; these stay flat, and loading.py handles both shapes.")
    elif n_bad:
        print("  A failure here means the refactored head's architecture disagrees with what")
        print("  was trained. Read the shapes above before changing any constructor default:")
        print("  almost no checkpoint records head_variant or the MLP geometry, so a default")
        print("  decides what gets built, and the wrong one silently fails a strict load.")
    else:
        print("  NOTHING WAS SCANNED -- this is not a pass. Check --root.")
        return 1
    if a.report and rows:
        import csv
        with open(a.report, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"\n[write] {a.report}")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
