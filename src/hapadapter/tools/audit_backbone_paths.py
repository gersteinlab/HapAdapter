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
import re
import os
import shutil
import sys
from collections import Counter

KEY = "model_name_or_path"
# AutoModel.from_pretrained needs the config to BUILD the architecture and a weights file to
# populate it. Checking only the config would let a weightless directory pass here and fail
# later inside transformers -- the confusing failure this tool exists to prevent.
NEEDED = ("config.json",)
WEIGHTS = ("pytorch_model.bin", "model.safetensors")


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

    # A HUB ID IS NOT A PATH AND MUST SKIP THE FILESYSTEM CHECKS. For publication the recorded
    # backbone has to be something a DOWNLOADER can resolve, which an absolute local path never
    # is -- that is the defect that makes a shared checkpoint unloadable on any other machine.
    # The validation that replaces the disk check is structural: owner/name, no scheme, no
    # leading slash, no spaces. It cannot confirm the repo exists (that needs the network), so
    # it says so rather than implying it verified anything.
    is_hub = bool(re.fullmatch(r"[A-Za-z0-9][\w.-]*/[\w.-]+", a.new_path or ""))
    if is_hub:
        print(f"\n  --new_path looks like a HUB ID ({a.new_path}), so the on-disk checks are "
              f"skipped.")
        print(f"  NOT VERIFIED: that this repo exists or is public -- that needs a network "
              f"call.\n  Confirm with:  huggingface-cli download {a.new_path} config.json "
              f"--quiet\n  before publishing anything that records it.")
        print(f"  Use this ONLY on copies destined for release. A local training run needs a "
              f"local\n  path, and rewriting a working tree to a hub id makes every checkpoint "
              f"re-download.")
    elif resolves(a.new_path) is not True:
        # `is not True`, not `is False`. resolves() returns None for anything that is not an
        # absolute path, which is the right answer when CLASSIFYING a recorded value (a hub id
        # there is legitimately not a local path to check) but the wrong gate on --new_path:
        # under `is False`, a typo like "not a/valid id" or a pasted URL is neither a hub id nor
        # an absolute path, so it fell through and got written to every checkpoint.
        why = ("is neither a hub id (owner/name) nor an absolute path"
               if not a.new_path.startswith("/") else
               f"is an absolute path but is not a usable backbone directory "
               f"(expected {', '.join(NEEDED)})")
        raise SystemExit(f"--new_path {why}: {a.new_path!r}\n"
                         f"  A hub id must match owner/name with no scheme and no leading "
                         f"slash -- 'https://huggingface.co/owner/name' is a URL, not an id.\n"
                         f"  For a local path, move the backbone there FIRST, then re-run: "
                         f"writing a path that is also broken converts a loud failure into a "
                         f"confusing one.")

    if not is_hub and a.new_path.startswith("/") and not any(
            os.path.exists(os.path.join(a.new_path, w)) for w in WEIGHTS):
        raise SystemExit(
            f"--new_path has {NEEDED[0]} but NO weights file ({' or '.join(WEIGHTS)}):\n"
            f"  {a.new_path}\n"
            f"  transformers would build the architecture and then fail to populate it, or "
            f"silently\n  return a RANDOMLY INITIALISED backbone depending on the call path. "
            f"Copy the weights\n  there first. If the weights are gitignored they were never "
            f"tracked, so check the\n  ORIGINAL directory still has them before moving "
            f"anything.")

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
