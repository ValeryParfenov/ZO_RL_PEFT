#!/bin/bash
# Entry point for one rebuttal run on the mlsub cloud gateway (one GPU per job).
# Usage (as mlsub --entry with --args): scripts/mlsub_run.sh <model> <mode> <method> [steps_override]
#
# Platform rules this follows: a failed job shows no logs, so the script always exits 0 and prints
# the tail of its own log; only /home/jovyan survives the job, so the run log and results.json are
# copied there (checkpoints stay in /tmp, they are only needed during the run).
set -u
MODEL=$1; MODE=$2; METHOD=$3; STEPS=${4:-}
NAME="sst2-${MODEL}-${MODE}-${METHOD}"
[[ -n $STEPS ]] && NAME="$NAME-smoke"
OUT=/home/jovyan/zo_ldsd/$NAME
mkdir -p "$OUT" /home/jovyan/tmp

export PYTHONUSERBASE=/home/jovyan/.local-zoldsd
export PATH=$PYTHONUSERBASE/bin:$PATH
export HF_HOME=/home/jovyan/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTHON=python

echo "START $NAME $(date '+%F %T') $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
if [[ ! -f $PYTHONUSERBASE/.zoldsd_deps ]]; then
    pip install --user --quiet transformers==4.28.1 tokenizers==0.13.3 accelerate==0.25.0 \
        datasets==2.16.0 huggingface-hub==0.20.1 scikit-learn==1.3.2 scipy numpy==1.26.4 \
        wandb==0.16.1 sentencepiece > "$OUT/pip.log" 2>&1 && touch "$PYTHONUSERBASE/.zoldsd_deps"
    echo "pip exit=$? (log $OUT/pip.log)"
fi

# Mirror progress to NFS every 5 minutes so a running job can be read from outside.
LOG=runs/$NAME/stdout.log
( while sleep 300; do [[ -f $LOG ]] && cp "$LOG" "$OUT/stdout.log"; done ) &
MIRROR=$!

if [[ -n $STEPS ]]; then
    bash scripts/rebuttal_runs.sh "$MODEL" "$MODE" "$METHOD" "$STEPS" -smoke > "$OUT/run.log" 2>&1
else
    bash scripts/rebuttal_runs.sh "$MODEL" "$MODE" "$METHOD" > "$OUT/run.log" 2>&1
fi
CODE=$?
kill $MIRROR 2>/dev/null

[[ -f $LOG ]] && cp "$LOG" "$OUT/stdout.log"
[[ -f result/$NAME/results.json ]] && cp "result/$NAME/results.json" "$OUT/results.json"
cp -r runs/lab-runs "$OUT/" 2>/dev/null
echo "EXIT=$CODE $(date '+%F %T')"
grep -oE "Evaluating at step [0-9]+|'test_acc': [0-9.]+, 'val_acc': [0-9.]+" "$OUT/stdout.log" 2>/dev/null | paste -d' ' - - | tail -60
cat "$OUT/results.json" 2>/dev/null
tail -40 "$OUT/run.log"
exit 0
