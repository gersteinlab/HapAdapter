#!/bin/bash
# reorg_step1.sh -- create the HapAdapter tree and git mv every file into it.
#
# WHAT THIS DOES AND DOES NOT DO.
# Moves files. Edits NOTHING inside them. After this runs the package is BROKEN -- every
# intra-package import still names the old module path -- and that is intended: the commit is
# a pure move, so `git log --follow` tracks each file and the next commit's diff is the real
# content change rather than a rename tangled up with edits.
#
# Run from the repo root. Dry by default; pass --apply to execute.
#
#   bash reorg_step1.sh            # print what it would do
#   bash reorg_step1.sh --apply    # do it

set -uo pipefail
APPLY=0; [ "${1:-}" = "--apply" ] && APPLY=1
S=src/hapadapter

[ -d "$S" ] || { echo "FATAL: run from the HapAdapter repo root (no $S)" >&2; exit 1; }
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo "FATAL: not a git repo" >&2; exit 1; }
# TRACKED changes only. Untracked files are fine -- this script itself is usually one of
# them, and `git status --porcelain` would otherwise make the script refuse to run because
# it exists.
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "FATAL: tracked files are modified. Commit or stash first -- a pure-move commit must"
  echo "       contain nothing but moves, or the next verifier failure has two suspects." >&2
  git status --short --untracked-files=no | sed 's/^/       /' >&2
  exit 1
fi

run() { if [ "$APPLY" = 1 ]; then "$@"; else echo "  would: $*"; fi; }

echo "=== 1. create the tree"
for D in adapter trunk backbones data scoring interpret figures; do
  run mkdir -p "$S/$D"
  if [ ! -f "$S/$D/__init__.py" ]; then
    if [ "$APPLY" = 1 ]; then : > "$S/$D/__init__.py"; git add "$S/$D/__init__.py"
    else echo "  would: create $S/$D/__init__.py"; fi
  fi
done

echo
echo "=== 2. move files  (old -> new)"
# trunk/ -- the end-to-end torch path. `loading.py` not `io.py`: `io` shadows a stdlib module
# and an installed top-level package. `finetune.py` not `finetune_entexbert2.py`: the old name
# carries the retired project name, and nothing imports it (CLI entry point only).
mv_one() {  # $1 = path under $S, $2 = destination path under $S
  [ -f "$S/$1" ] || { echo "  SKIP (absent): $1"; return; }
  run git mv "$S/$1" "$S/$2"
}

mv_one model.py                              trunk/model.py
mv_one model_io.py                           trunk/loading.py
mv_one finetune_entexbert2.py                trunk/finetune.py

# data/ -- everything that builds inputs. No model import among them.
mv_one build_inputs.py                       data/build_inputs.py
mv_one build_experiment.py                   data/build_experiment.py
mv_one build_hap_counts.py                   data/build_hap_counts.py
mv_one personal_seq.py                       data/personal_seq.py
mv_one select_heldout_chroms.py              data/select_heldout_chroms.py

# scoring/ -- was `eval/`, renamed because `eval` is a Python BUILTIN.
# eval_utils.py -> stats.py: it is the home of balanced_auroc, which the cached backend
# currently vendors as asb_stat.py. One copy, under a name that says what it is.
mv_one eval_utils.py                         scoring/stats.py
mv_one analysis/score_crossdonor.py          scoring/score_crossdonor.py
mv_one analysis/score_external.py            scoring/score_external.py
mv_one analysis/score_individual.py          scoring/score_individual.py
mv_one analysis/dump_trunk_pred.py           scoring/dump_trunk_pred.py

# interpret/ -- ISM and attention. dump_contrast_embeddings is here rather than in scoring/
# because its output feeds the interpretability figures, not a metric.
mv_one analysis/ism_saliency.py              interpret/ism_saliency.py
mv_one analysis/build_asb_ism_windows.py     interpret/build_asb_ism_windows.py
mv_one analysis/dump_contrast_embeddings.py  interpret/dump_contrast_embeddings.py

# figures/ -- anything whose product is a rendered file.
mv_one analysis/plot_ism_logo.py             figures/plot_ism_logo.py

echo
echo "=== 3. retire the emptied packages"
for D in analysis alphagenome; do
  if [ -d "$S/$D" ]; then
    N=$(find "$S/$D" -name "*.py" ! -name "__init__.py" | wc -l | tr -d ' ')
    if [ "$N" = 0 ]; then
      run git rm -q "$S/$D/__init__.py"
      run rmdir "$S/$D"
    else
      echo "  KEEP $S/$D -- still holds $N module(s); nothing in the plan claims them"
    fi
  fi
done

echo
echo "=== 4. what this leaves BROKEN, deliberately"
cat <<'EOF'
  Every intra-package import still names the old path. Expect `import hapadapter` to fail
  until the next commit. The count to fix:
EOF
if [ "$APPLY" = 1 ]; then
  grep -rn "from hapadapter\.\(model\|model_io\|eval_utils\|build_\|personal_seq\|select_\|analysis\)" \
    --include='*.py' "$S" | wc -l | sed 's/^/    import sites: /'
  grep -rn "^from \.\|^from \.\." --include='*.py' "$S" | wc -l | sed 's/^/    relative imports: /'
  echo
  echo "  __init__.py's lazy _EXPORTS map points at old module names and must be rewritten."
  echo "  jobs/*.sbatch and README invoke python -m hapadapter.finetune_entexbert2 -- now"
  echo "  hapadapter.trunk.finetune."
else
  echo "    (run with --apply to count)"
fi

echo
if [ "$APPLY" = 1 ]; then
  echo "=== moved. Commit this ALONE:"
  echo "    git commit -m 'Reorganise package into adapter/trunk/data/scoring/interpret/figures (pure move)'"
else
  echo "=== dry run. Re-run with --apply."
fi
