#!/bin/bash
# Extra baselines for ZO-LDSD on SST-2 (Table 1 of the paper).
# Usage: scripts/rebuttal_runs.sh <opt13|roberta> <lora|ft> <ldsd|greedy|loren|zoar> [steps_override] [name_suffix]
# RoBERTa goes through AutoModelForCausalLM, as in the authors' dev-branch scripts (test 0.885-0.900).
# The caller sets CUDA_DEVICE_ORDER / CUDA_VISIBLE_DEVICES.
#
# Budget matching (forward passes per run, as in Table 1):
#   ldsd, greedy : zo_rl_sgd, K=5 candidates + 2-sided step, 20k steps
#   loren        : 6 one-sided queries per step,             20k steps
#   zoar         : 2 one-sided queries per step,             60k steps (3x)
set -euo pipefail

MODEL=$1
MODE=$2
METHOD=$3
STEPS_OVERRIDE=${4:-}
SUFFIX=${5:-}

export TMPDIR=$HOME/tmp; mkdir -p "$TMPDIR"
export WANDB_DISABLED=true
export LAB_RUN_OFFLINE=1
export LAB_RUN_DIR=runs/lab-runs
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

# Base learning rates from Table 4 of the paper (ZO-SGD and ZO-AdaMM rows).
# ZoAR on OPT-1.3B LoRA uses its own published value (5e-5, Qiu et al. 2025, App. D.4).
case $MODEL-$MODE in
    roberta-lora) HF=roberta-large;    LORA_FLAG="--lora"; SGD_LR=5e-4; ZOAR_LR=3e-3 ;;
    roberta-ft)   HF=roberta-large;    LORA_FLAG="";       SGD_LR=4e-8; ZOAR_LR=3e-5 ;;
    opt13-lora)   HF=facebook/opt-1.3b; LORA_FLAG="--lora"; SGD_LR=1e-6; ZOAR_LR=5e-5 ;;
    opt13-ft)     HF=facebook/opt-1.3b; LORA_FLAG="";       SGD_LR=4e-8; ZOAR_LR=1e-5 ;;
    *) echo "unknown model/mode $MODEL-$MODE"; exit 1 ;;
esac

case $METHOD in
    ldsd)
        STEPS=20000; EVAL=500
        OPT="--trainer=zo_rl_sgd --learning_rate=$SGD_LR --momentum=0.9 --k_value=5 --variance=1 --lr_mu=1e-3 --zo_eps=1e-3"
        WHY="Anchor: ZO-SGD + LDSD (Algorithm 2) re-run under the same config, to check it reproduces Table 1." ;;
    greedy)
        STEPS=20000; EVAL=500
        OPT="--trainer=zo_rl_sgd --learning_rate=$SGD_LR --momentum=0.9 --k_value=5 --variance=1 --lr_mu=0 --zo_eps=1e-3"
        WHY="Ablation: ZO-SGD with Gaussian sampling, K=5 and greedy argmin, mu frozen at 0. If it matches LDSD, the gain comes from selection, not from learning mu." ;;
    loren)
        STEPS=20000; EVAL=500
        OPT="--trainer=loren --learning_rate=$SGD_LR --momentum=0.9 --k_value=6 --zo_eps=1e-3 --loren_damping=0.1 --loren_lr_cov=1e-3"
        WHY="Baseline LOREN (Seung et al. 2026), 6 queries per step, same forward budget as LDSD." ;;
    zoar)
        STEPS=60000; EVAL=1500
        OPT="--trainer=zoar --learning_rate=$ZOAR_LR --k_value=2 --zoar_history=15 --zo_eps=1e-2 --beta1=0.9 --beta2=0.999"
        WHY="Baseline ZoAR (Qiu et al. 2025), 2 queries per step and 3x steps, same forward budget as LDSD." ;;
    *) echo "unknown method $METHOD"; exit 1 ;;
esac

if [[ -n $STEPS_OVERRIDE ]]; then STEPS=$STEPS_OVERRIDE; EVAL=$STEPS; fi
SAVE=$((2 * EVAL))
[[ -n $STEPS_OVERRIDE ]] && SAVE=$EVAL
NAME="sst2-${MODEL}-${MODE}-${METHOD}${SUFFIX}"

"${PYTHON:-$HOME/venvs/zoldsd/bin/python}" scripts/launch_run.py --name "$NAME" --objective "$WHY" -- \
    --model_name=$HF --task_name=SST2 $LORA_FLAG \
    --num_train=1000 --num_dev=500 --num_eval=1000 --train_as_classification --train_set_seed=0 \
    --per_device_train_batch_size=16 --max_steps=$STEPS --num_training_steps=$STEPS \
    --scheduler=cosine --warmup_steps=0 --min_lr_ratio=0.1 --scheduler_cycle_length=1 \
    --evaluation_strategy=steps --eval_steps=$EVAL \
    --save_strategy=steps --save_steps=$SAVE --save_total_limit=1 --load_best_model_at_end \
    --logging_steps=10 --report_to=none --no_use_wandb --overwrite_output_dir \
    --perturbation_mode=two_side --weight_decay=0.0 $OPT
