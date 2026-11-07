#!/bin/bash
#SBATCH --job-name=verify_rename
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/verify_rename_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=amy.metrick@yale.edu
#
# Step 2d -- prove the rename did not touch a single trained model.
#
# WHY A BATCH JOB. 44+ checkpoints at ~468 MB, read from BOTH sides = tens of GB of I/O.
# Trivial in compute, but not a login-node command. No GPU: the comparison is state_dict
# tensors, and --full runs two short sequences on CPU.
#
# TWO PASSES, ON PURPOSE:
#   pass 1  every checkpoint, KEY + TENSOR comparison, no forward pass. Fast, and it is the
#           pass that can actually fail on a rename -- a renamed attribute changes the key set.
#   pass 2  ONE Stage-2 twin checkpoint with --full. The forward pass is the only thing that
#           can catch a change in the softplus(dist_a)*||z1-z2|| + dist_b arithmetic, which
#           would move results without moving any key. It needs the real backbone, so it is
#           slow per checkpoint and pointless to run 44 times for a pure rename.
#
# No `set -e`: a failure in pass 1 must not discard pass 2's result, and vice versa.

set -uo pipefail

REPO=${REPO:-$HOME/HapAdapter}
OLD=${OLD:-$HOME/HapAdapter-premigration}
MODELS=${MODELS:-$HOME/palmer_scratch/entexbert2-models}
PY=${PY:-$HOME/.conda/envs/eb2/bin/python}
GLOB=${GLOB:-"$MODELS/experiments/*/stage*/runs/*"}
FULL_ONE=${FULL_ONE:-"$MODELS/experiments/ctcf_ENC-002/stage2_ref_data"}

cd "$REPO" || { echo "FATAL: no $REPO" >&2; exit 1; }
mkdir -p logs

echo "=== $(date) | repo=$REPO old=$OLD"
[ -x "$PY" ] || { echo "FATAL: interpreter not executable: $PY" >&2; exit 1; }

# ---- preflight ---------------------------------------------------------------------------
# The premigration worktree is this job's control. If the mv in 2c broke its link and
# `git worktree repair` was not run, the directory may exist but be empty or stale -- in which
# case the comparison is vacuous rather than failed, which is the worse outcome.
for F in "$OLD/src/entexbert2/model.py" "$OLD/src/entexbert2/model_io.py" \
         "$REPO/src/hapadapter/model.py" "$REPO/tools/verify_checkpoint_compat.py"; do
  [ -f "$F" ] || { echo "FATAL: missing $F" >&2; exit 1; }
done
echo "[pre] both trees present"

if grep -q "pair_dropout" "$OLD/src/entexbert2/model.py"; then
  echo "[pre] NOTE premigration tree INCLUDES the pair_dropout patch."
  echo "      Still a valid control (that patch is key-neutral and bit-identical on the"
  echo "      default path) but it is not the exact code that trained the checkpoints."
else
  echo "[pre] premigration tree predates the pair_dropout patch -- the cleanest control."
fi

N=$(compgen -G "$GLOB" | wc -l | tr -d ' ')
echo "[pre] $N directories match the checkpoint glob"
[ "$N" -gt 0 ] || { echo "FATAL: glob matched nothing: $GLOB" >&2; exit 1; }

RC=0

# ---- pass 1: every checkpoint, keys + tensors ---------------------------------------------
# NOT --strict on the first run. Across a sweep this size there are usually directories that
# hold weights but no run_config.json (interrupted runs, hand-copied dirs). --strict turns
# those into a failure; without it they are named individually and skipped, and the closing
# line says "N verified / M NOT checked" rather than an unqualified pass. Read that list,
# satisfy yourself the orphans are not checkpoints you care about, THEN re-run with --strict.
echo
echo "=== pass 1: all checkpoints, key + tensor comparison"
"$PY" tools/verify_checkpoint_compat.py \
  --old_src "$OLD/src" --new_src "$REPO/src" \
  --checkpoints "$GLOB" \
  --report compat_after_rename.csv || RC=1

# ---- pass 2: one Stage-2 twin, with a forward pass ----------------------------------------
echo
echo "=== pass 2: --full on one Stage-2 twin checkpoint"
if compgen -G "$FULL_ONE/runs/*" > /dev/null; then
  "$PY" tools/verify_checkpoint_compat.py \
    --old_src "$OLD/src" --new_src "$REPO/src" \
    --checkpoints "$FULL_ONE/runs/*" --full \
    --report compat_full_one.csv || RC=1
else
  echo "  SKIP: no runs under $FULL_ONE -- set FULL_ONE to a Stage-2 twin experiment"
  echo "  This pass is the ONLY check on the twin forward arithmetic. Do not call 2d done"
  echo "  without it."
fi

echo
echo "=== exit $RC"
if [ "$RC" -eq 0 ]; then
  echo "  A rename cannot change a state_dict key, so every row should read OK. It does."
  echo "  Re-run pass 1 with --strict once you have read the orphan list, to convert"
  echo "  'skipped' into 'must not exist'."
else
  echo "  DO NOT COMMIT. The rename commit is the only change in the tree, so it is the"
  echo "  cause. Check first for an attribute that was renamed along with the package:"
  echo "  backbone / proj / dist_a / dist_b / main_head must keep their names."
fi
exit $RC
