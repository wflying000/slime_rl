#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || true)
if [[ "$GPU_NAME" =~ (B300|GB300) ]]; then
    IS_B300=1
else
    IS_B300=0
fi
echo "GPU_NAME: ${GPU_NAME:-unknown}; IS_B300: $IS_B300"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

source "${PROJECT_DIR}/scripts/models/qwen3.5-35B-A3B.sh"

MODEL_NAME="${MODEL_NAME:-qwen3.5-35b-a3b-sft-605-mtp}"
HF_CHECKPOINT="${HF_CHECKPOINT:-/opt/users/models/${MODEL_NAME}}"
REF_CHECKPOINT="${REF_CHECKPOINT:-/opt/users/models/${MODEL_NAME}_miles}"
TRAIN_CHECKPOINT="${TRAIN_CHECKPOINT:-/mnt/code/wangfei61/models/${MODEL_NAME}-RL-Nemotron-Structured-Outputs-MILES}"

NEMOTRON_DATA_DIR="${NEMOTRON_DATA_DIR:-/Users/a58/instruction_following/dataset/nemotron_structured_outputs_v2}"
COMBINED_DATA="${COMBINED_DATA:-/tmp/nemotron_structured_outputs_v2_all_slmfmt.jsonl}"
TRAIN_DATA="${TRAIN_DATA:-${COMBINED_DATA}}"

if [ ! -f "${TRAIN_DATA}" ] || [ "${REBUILD_COMBINED_DATA:-0}" = "1" ]; then
   cat \
      "${NEMOTRON_DATA_DIR}/direct_generation_slmfmt.jsonl" \
      "${NEMOTRON_DATA_DIR}/diversified_tasks_slmfmt.jsonl" \
      "${NEMOTRON_DATA_DIR}/tool_calling_extraction_slmfmt.jsonl" \
      > "${COMBINED_DATA}"
   TRAIN_DATA="${COMBINED_DATA}"
fi

PROJECT_NAME="${PROJECT_NAME:-Miles-RL-Nemotron-Structured-Outputs}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${PROJECT_NAME}-${MODEL_NAME}}"
SWANLOG_DIR="${SWANLOG_DIR:-/opt/users/swanlog}"
WANDB_KEY="${WANDB_KEY:-}"

CKPT_ARGS=(
   --hf-checkpoint ${HF_CHECKPOINT}
   --ref-load ${REF_CHECKPOINT}
   --load ${TRAIN_CHECKPOINT}
   --save ${TRAIN_CHECKPOINT}
   --save-interval 50
   --no-save-optim
)

ROLLOUT_ARGS=(
   --prompt-data ${TRAIN_DATA}
   --input-key messages
   --label-key label
   --metadata-key metadata
   --tool-key tools
   --apply-chat-template
   --apply-chat-template-kwargs '{"enable_thinking":false}'
   --rollout-shuffle
   --custom-rm-path recipe.nemotron_structured_outputs.structured_outputs_reward.structured_outputs_reward
   --num-rollout 3000
   --rollout-batch-size 96
   --n-samples-per-prompt 8
   --rollout-max-prompt-len 8192
   --rollout-max-response-len 8192
   --rollout-temperature 1.0

   --num-steps-per-rollout 1
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 50
   --eval-prompt-data nemotron_structured_outputs ${TRAIN_DATA}@[0:512]
   --eval-input-key messages
   --eval-label-key label
   --eval-tool-key tools
   --n-samples-per-eval-prompt 1
   --eval-max-prompt-len 8192
   --eval-max-response-len 8192
   --eval-top-p 0.95
   --eval-temperature 0.6
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 2
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 12288
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project ${PROJECT_NAME}
   # --wandb-group ${EXPERIMENT_NAME}
   # --wandb-key ${WANDB_KEY}
)

SWANLAB_ARGS=(
   # --use-swanlab
   # --swanlab-logdir ${SWANLOG_DIR}
   # --swanlab-project ${PROJECT_NAME}
   # --swanlab-name ${EXPERIMENT_NAME}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.75
   --sglang-ep-size 1

   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)

   --sglang-speculative-algorithm EAGLE
   --sglang-speculative-num-steps 2
   --sglang-speculative-eagle-topk 1
   --sglang-speculative-num-draft-tokens 3
   --sglang-mamba-scheduler-strategy extra_buffer

   --sglang-server-concurrency 256
   --sglang-max-running-requests 256
)

TRAIN_ATTENTION_BACKEND=flash
if [ "$IS_B300" -eq 1 ]; then
   SGLANG_ARGS+=(
      --sglang-moe-runner-backend flashinfer_cutlass
      --sglang-attention-backend trtllm_mha
   )
   TRAIN_ATTENTION_BACKEND=auto
fi

MTP_ARGS=(
   # --enable-mtp-training
   # --mtp-num-layers 1
   # --mtp-loss-scaling-factor 0.2
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend ${TRAIN_ATTENTION_BACKEND}

   --moe-token-dispatcher-type flex
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"

ray start --head \
   --node-ip-address ${MASTER_ADDR} \
   --num-gpus 8 \
   --disable-usage-stats \
   --dashboard-host=0.0.0.0 \
   --dashboard-port=8265 \
   --dashboard-agent-grpc-port=20001 \
   --dashboard-agent-listen-port=20002 \
   --runtime-env-agent-port=20003 \
   --metrics-export-port=20004 \
   --object-manager-port=20005 \
   --node-manager-port=20006 \
   --min-worker-port=21000 \
   --max-worker-port=29999

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"SGLANG_ENABLE_SPEC_V2\": \"1\",
    \"DEPRECATED_MEGATRON_COMPATIBLE\": \"1\",
    \"no_proxy\": \"${no_proxy}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${SWANLAB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MTP_ARGS[@]} \
   ${MISC_ARGS[@]}
