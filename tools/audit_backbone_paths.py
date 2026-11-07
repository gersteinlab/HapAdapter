#!/usr/bin/env python
"""
audit_backbone_paths.py -- find and repair the absolute backbone path baked into every
checkpoint's run_config.json.

THE PROBLEM THIS EXISTS FOR. finetune records `model_name_or_path` verbatim, and it was given
as an absolute path:

    "model_name_or_path": "/home/asm242/entexBERT-2/DNABERT-2-117M-attention"

model_io.build_model hands that straight to transformers.AutoModel.from_pretrained, so the
moment that directory moves -- a repo rename, a move to another filesystem, a clone by anyone
else -- EVERY checkpoint fails to load with an HFValidationError that names a hub repo id
rather than the real problem.

This is NOT caused by the HapAdapter rename. The rename only exposed it. These checkpoints were
never loadable on any machine but the one that trained them, which also means a checkpoint
published to HuggingFace in this state is unusable by anyone who downloads it. It has to be
fixed before publication regardless of what happens to the repo.

AUDIT FIRST, THEN REPAIR. Default is read-only: it reports the distinct values, how many
checkpoints carry each, and whether each still resolves. --apply rewrites, and only then.

Usage:
  python audit_backbone_paths.py --root $HOME/palmer_scratch/entexbert2-models/experiments
  python audit_backbone_paths.py --root ... --new_path $HOME/backbones/DNABERT-2-117M-attention --apply
"""
import argparse
import glob
import json
import os
import shutil
import sys
from collections import Counter

KEY = "model_name_or_path"
# A backbone directory must have these for AutoModel.from_pretrained to work at all.
NEEDED = ("config.json",)


def resolves(path):
    """Is this a usable local backbone directory? (A hub id is not a path and is fine as-is.)"""
    if not path or not path.startswith("/"):
        return None          # hub id or relative -- not our problem
    if not os.path.isdir(path):
        return False
    return all(os.path.exists(os.path.join(path, n)) for n in NEEDED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="directory to search recursively for run_config.json")
    ap.add_argument("--new_path", default=None,
                    help="value to write into model_name_or_path. A directory that does NOT "
                         "live inside the repo is the point: put the backbone somewhere a "
                         "future rename cannot reach, e.g. $HOME/backbones/<name>.")
    ap.add_argument("--only_broken", action="store_true",
                    help="rewrite only entries whose current path does not resolve, leaving "
                         "already-correct ones untouched")
    ap.add_argument("--apply", action="store_true",
                    help="actually write. Without it this is a read-only audit.")
    ap.add_argument("--backup_suffix", default=".bak",
                    help="each rewritten file is copied to <name>.json<suffix> first; set "
                         "empty to disable (not advised)")
    a = ap.parse_args()

    cfgs = sorted(glob.glob(os.path.join(a.root, "**", "run_config.json"), recursive=True))
    if not cfgs:
        raise SystemExit(f"no run_config.json under {a.root} -- nothing to audit, which is NOT "
                         f"the same as nothing to fix. Check the path.")
    print(f"[scan] {len(cfgs)} run_config.json under {a.root}\n")

    vals, bad = Counter(), []
    for c in cfgs:
        try:
            rc = json.load(open(c))
        except Exception as e:
            print(f"  UNREADABLE {c}: {type(e).__name__}")
            continue
        v = rc.get(KEY)
        vals[v] += 1
        if resolves(v) is False:
            bad.append(c)

    print(f"  {'count':>7}  {'resolves':>9}  {KEY}")
    for v, n in vals.most_common():
        r = resolves(v)
        tag = {True: "yes", False: "NO", None: "n/a (hub id)"}[r]
        print(f"  {n:>7}  {tag:>9}  {v}")

    print(f"\n  {len(bad)} of {len(cfgs)} checkpoints point at a path that does not resolve.")
    if not bad and not a.new_path:
        print("  Nothing to repair. Note that 'resolves' is true only ON THIS MACHINE -- an "
              "absolute\n  path under $HOME is still unusable to anyone else, which matters "
              "for publication.")
        return 0

    if not a.new_path:
        print("\n  Pass --new_path to repair. Recommended target is OUTSIDE the repo, so the "
              "next\n  rename or move cannot break it again:")
        print("      --new_path $HOME/backbones/DNABERT-2-117M-attention")
        return 1

    if resolves(a.new_path) is False:
        raise SystemExit(f"--new_path does not resolve as a backbone directory: {a.new_path}\n"
                         f"  expected to find: {', '.join(NEEDED)}\n"
                         f"  Move the backbone there FIRST, then re-run. Writing a path that is "
                         f"also broken would convert a loud failure into a confusing one.")

    targets = bad if a.only_broken else [c for c in cfgs
                                         if json.load(open(c)).get(KEY) != a.new_path]
    print(f"\n  would rewrite {len(targets)} file(s) -> {a.new_path}")
    if not a.apply:
        print("  DRY RUN. Re-run with --apply to write.")
        for c in targets[:5]:
            print(f"    {c}")
        if len(targets) > 5:
            print(f"    ... and {len(targets)-5} more")
        return 0

    n = 0
    for c in targets:
        rc = json.load(open(c))
        if a.backup_suffix:
            shutil.copy2(c, c + a.backup_suffix)
        rc[KEY] = a.new_path
        # rewrite via a temp file + replace: a crash mid-write must not leave a truncated
        # run_config.json, which would make the checkpoint unloadable rather than mislocated.
        tmp = c + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(rc, fh, indent=2)
        os.replace(tmp, c)
        n += 1
    print(f"  rewrote {n} file(s)"
          f"{f'; originals kept as *.json{a.backup_suffix}' if a.backup_suffix else ''}")
    print("  Re-run without --apply to confirm every row now reads 'resolves: yes'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
