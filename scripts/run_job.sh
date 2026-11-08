#!/bin/bash
# Run one queue job with a collapse watchdog.
# Usage: scripts/run_job.sh <model>:<mode>:<method>[:<suffix>[:KEY=VAL,KEY=VAL]]
#   e.g. roberta:lora:ldsd:-diag:LDSD_DIAG_EVERY=50
#        roberta:lora:zoar:-lr1e-3:LR=1e-3,MAX_STEPS=3000,EVAL_STEPS=500
# KEY=VAL pairs are exported for rebuttal_runs.sh (LR, MAX_STEPS, EVAL_STEPS, LDSD_DIAG_EVERY).
# Every 5 minutes scripts/collapse_check.py looks at the run log; on collapse the run is
# stopped and runs/collapsed/<name> is written, so it is not mistaken for a result.
cd "$(dirname "$0")/.."
IFS=: read -r MODEL MODE METHOD SUFFIX ENVS <<< "$1"
if [[ -n ${ENVS:-} ]]; then
    IFS=, read -ra PAIRS <<< "$ENVS"
    for kv in "${PAIRS[@]}"; do export "${kv?}"; done
fi
NAME="sst2-$MODEL-$MODE-$METHOD${SUFFIX:-}"
# LR=auto: take the learning rate of the best finished, non-collapsed sweep run
# result/sst2-<model>-<mode>-<method>-lr<value> (highest val accuracy of the best-dev-loss checkpoint).
if [[ ${LR:-} == auto ]]; then
    LR=$(python3 - "sst2-$MODEL-$MODE-$METHOD-lr" <<'EOF'
import glob, json, os, sys
prefix = sys.argv[1]
best = None
for path in glob.glob(f"result/{prefix}*/results.json"):
    name = path.split("/")[1]
    if os.path.exists(f"runs/collapsed/{name}"):
        continue
    val = json.load(open(path)).get("val_accuracy", 0.0)
    if best is None or val > best[0]:
        best = (val, name[len(prefix):])
print(best[1] if best else "")
EOF
)
    if [[ -z $LR ]]; then echo "$(date '+%F %T') no usable sweep result for $NAME" >&2; exit 4; fi
    echo "$(date '+%F %T') $NAME uses swept LR=$LR" >&2
    export LR
fi
LOG=runs/$NAME/stdout.log

setsid bash scripts/rebuttal_runs.sh "$MODEL" "$MODE" "$METHOD" "" "${SUFFIX:-}" &
CHILD=$!
stop_child() { pkill -TERM -g "$CHILD" 2>/dev/null; sleep 20; pkill -KILL -g "$CHILD" 2>/dev/null; }
trap 'stop_child; exit 143' TERM INT

while kill -0 "$CHILD" 2>/dev/null; do
    sleep 300 & wait $!
    if python3 scripts/collapse_check.py "$LOG"; then
        mkdir -p runs/collapsed; date '+%F %T' > "runs/collapsed/$NAME"
        echo "$(date '+%F %T') COLLAPSE $NAME, stopping" >&2
        stop_child; exit 3
    fi
done
wait "$CHILD"
