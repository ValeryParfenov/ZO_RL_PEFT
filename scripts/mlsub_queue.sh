#!/bin/bash
# Entry point for a sequence of rebuttal jobs on one mlsub GPU.
# Usage (mlsub --entry with --args): scripts/mlsub_queue.sh <job spec> [<job spec> ...]
#   job spec as in scripts/run_job.sh, e.g. roberta:lora:zoar:-lr1e-3:LR=1e-3,MAX_STEPS=3000
# Jobs run one after another through run_job.sh, so the collapse watchdog applies and LR=auto
# picks the best sweep run finished earlier in this same mlsub job.
# The job log is the only thing readable from outside while it runs, so every 10 minutes one
# PROGRESS line is printed (tqdm position and last evaluation of the current run), and every
# finished run prints its RESULT line. Logs and results.json are copied to /home/jovyan, which
# survives the job. The script always exits 0 (a failed job shows no logs on the platform).
set -u
ROOT=/home/jovyan/zo_ldsd
mkdir -p "$ROOT" /home/jovyan/tmp

export PYTHONUSERBASE=/home/jovyan/.local-zoldsd
export PATH=$PYTHONUSERBASE/bin:$PATH
export HF_HOME=/home/jovyan/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTHON=python

echo "START queue $(date '+%F %T') $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
if [[ ! -f $PYTHONUSERBASE/.zoldsd_deps ]]; then
    pip install --user --quiet transformers==4.28.1 tokenizers==0.13.3 accelerate==0.25.0 \
        datasets==2.16.0 huggingface-hub==0.20.1 scikit-learn==1.3.2 scipy numpy==1.26.4 \
        wandb==0.16.1 sentencepiece > "$ROOT/pip-queue.log" 2>&1 && touch "$PYTHONUSERBASE/.zoldsd_deps"
    echo "pip exit=$? (log $ROOT/pip-queue.log)"
fi

progress() {  # one line about run $1: tqdm position and last evaluation
    local log=runs/$1/stdout.log
    [[ -f $log ]] || return 0
    local bar ev acc
    bar=$(grep -oE "[0-9]+/[0-9]+ \[[0-9:]+<" "$log" | tail -1)
    ev=$(grep -oE "Evaluating at step [0-9]+" "$log" | tail -1)
    acc=$(grep -oE "'test_acc': [0-9.]+, 'val_acc': [0-9.]+" "$log" | tail -1)
    echo "PROGRESS $1 $(date '+%F %T') bar=${bar:-none} ${ev:+eval_step=${ev##* }} $acc"
}

for SPEC in "$@"; do
    IFS=: read -r MODEL MODE METHOD SUFFIX _ <<< "$SPEC"
    NAME="sst2-$MODEL-$MODE-$METHOD${SUFFIX:-}"
    OUT=$ROOT/$NAME
    mkdir -p "$OUT"
    echo "JOB_START $NAME $(date '+%F %T') spec=$SPEC"
    ( while sleep 600; do progress "$NAME"; [[ -f runs/$NAME/stdout.log ]] \
        && cp "runs/$NAME/stdout.log" "$OUT/stdout.log"; done ) &
    TICKER=$!
    bash scripts/run_job.sh "$SPEC" > "$OUT/run.log" 2>&1
    CODE=$?
    kill $TICKER 2>/dev/null
    progress "$NAME"
    [[ -f runs/$NAME/stdout.log ]] && cp "runs/$NAME/stdout.log" "$OUT/stdout.log"
    [[ -f result/$NAME/results.json ]] && cp "result/$NAME/results.json" "$OUT/results.json"
    [[ -f runs/collapsed/$NAME ]] && cp "runs/collapsed/$NAME" "$OUT/collapsed"
    grep -oE "Evaluating at step [0-9]+|'test_acc': [0-9.]+, 'val_acc': [0-9.]+" "$OUT/stdout.log" 2>/dev/null \
        | paste -d' ' - - | sed "s/^/EVAL $NAME /"
    echo "RESULT $NAME code=$CODE collapsed=$([[ -f $OUT/collapsed ]] && echo 1 || echo 0)" \
        "$(tr -d '\n ' < "$OUT/results.json" 2>/dev/null)"
    tail -5 "$OUT/run.log"
done
cp -r runs/lab-runs "$ROOT/" 2>/dev/null
echo "QUEUE_DONE $(date '+%F %T')"
# The platform drops the last lines a job prints before it exits.
sleep 60
exit 0
