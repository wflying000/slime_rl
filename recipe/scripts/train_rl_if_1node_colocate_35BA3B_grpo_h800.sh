#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
# Eight training ranks share the host; prevent Ray from serializing CPU Adam.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/scripts/models/qwen3.5-35B-A3B.sh"

NUM_GPUS=${NUM_GPUS:-8}
TP_SIZE=${TP_SIZE:-2}
EP_SIZE=${EP_SIZE:-8}
ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-2}
# IF performance and memory measurements: /data/zwd/logs/qwen35_35ba3b_if_tune_20261008
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-16384}
RECOMPUTE_MODE=${RECOMPUTE_MODE:-full}
OPTIMIZER_OFFLOAD_FRACTION=${OPTIMIZER_OFFLOAD_FRACTION:-1.0}
MOE_DISPATCHER=${MOE_DISPATCHER:-flex}
ENABLE_SPECULATIVE=${ENABLE_SPECULATIVE:-1}
ENABLE_EVAL=${ENABLE_EVAL:-1}
ENABLE_SAVE=${ENABLE_SAVE:-1}
ENABLE_WANDB=${ENABLE_WANDB:-1}
QWEN_GDN_BACKEND=${QWEN_GDN_BACKEND:-flashqla}
SGLANG_CONCURRENCY=${SGLANG_CONCURRENCY:-64}

MODEL_NAME="Qwen3.5-35B-A3B"
MODEL_DIR="/model/dir"
HF_CHECKPOINT="${HF_CHECKPOINT:-${MODEL_DIR}/${MODEL_NAME}}"
REF_CHECKPOINT="${REF_CHECKPOINT:-${MODEL_DIR}/${MODEL_NAME}_slime}"
TRAIN_CHECKPOINT="${TRAIN_CHECKPOINT:-${MODEL_DIR}/${MODEL_NAME}-RL-IF-GRPO}"
DATA_DIR="/data/dir"
TRAIN_DATAPATH="${TRAIN_DATAPATH:-${DATA_DIR}/instruction-following-rl-66k.jsonl}"
PROJECT_NAME="Slime-RL-IF"
EXPERIMENT_NAME="${PROJECT_NAME}-${MODEL_NAME}"

# Resume saved model weights when available; otherwise use the converted model.
if [[ -z "${LOAD_CHECKPOINT:-}" ]]; then
   LOAD_CHECKPOINT="${REF_CHECKPOINT}"
   if [[ -f "${TRAIN_CHECKPOINT}/latest_checkpointed_iteration.txt" ]]; then
      LOAD_CHECKPOINT="${TRAIN_CHECKPOINT}"
   fi
fi

export JOINT_REWARD_WORKERS="${JOINT_REWARD_WORKERS:-8}"
export JOINT_REWARD_SAMPLE_TIMEOUT_S="${JOINT_REWARD_SAMPLE_TIMEOUT_S:-600}"
NOW_TIME=$(date "+%Y%m%d_%H%M%S")
REWARD_LOG_DIR="${REWARD_LOG_DIR:-/data/zwd/logs/${EXPERIMENT_NAME}/${NOW_TIME}}"
export IF_REWARD_LOG_PATH="${IF_REWARD_LOG_PATH-${REWARD_LOG_DIR}/if_reward.jsonl}"
export MATH_REWARD_LOG_PATH="${MATH_REWARD_LOG_PATH-${REWARD_LOG_DIR}/klear_math_reward.jsonl}"
export CODE_REWARD_LOG_PATH="${CODE_REWARD_LOG_PATH-${REWARD_LOG_DIR}/klear_code_reward.jsonl}"
export GURU_RL_REWARD_LOG_PATH="${GURU_RL_REWARD_LOG_PATH-${REWARD_LOG_DIR}/guru_rl_reward.jsonl}"
export REASONING_GYM_REWARD_LOG_PATH="${REASONING_GYM_REWARD_LOG_PATH-${REWARD_LOG_DIR}/reasoning_gym_reward.jsonl}"
export MATH_LOGIC_PUZZLE_ZEBRALOGIC_REWARD_LOG_PATH="${MATH_LOGIC_PUZZLE_ZEBRALOGIC_REWARD_LOG_PATH-${REWARD_LOG_DIR}/math_logic_puzzle_reward.jsonl}"


CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${REF_CHECKPOINT}"
   --load "${LOAD_CHECKPOINT}"
   --no-save-optim
   --no-load-optim
)

if [[ "${ENABLE_SAVE}" == 1 ]]; then
   CKPT_ARGS+=(--save "${TRAIN_CHECKPOINT}" --save-interval "${SAVE_INTERVAL:-20}")
fi

ROLLOUT_ARGS=(
   --prompt-data "${TRAIN_DATAPATH}"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --apply-chat-template
   --apply-chat-template-kwargs '{"enable_thinking":false}'
   --rollout-shuffle
   --custom-rm-path recipe.rl_reward.joint_reward.joint_reward
   --custom-reward-post-process-path recipe.rl_reward.reward_post_process.reward_post_process
   --num-rollout "${NUM_ROLLOUT:-5000}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-48}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-16}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-4096}"
   --rollout-temperature 1.0
   --global-batch-size "${GLOBAL_BATCH_SIZE:-768}"
   --num-steps-per-rollout 1
   --balance-data
   --start-rollout-id 0
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data \
      ifbench "${DATA_DIR}/ifbench_test.jsonl" \
      aime25 "${DATA_DIR}/aime25_klear.jsonl" \
      livecodebench-v6 "${DATA_DIR}/livecodebench_test6_175_slmfmt.jsonl" \
      zebralogic "${DATA_DIR}/zebralogicbench_private_grid_mode_slmfmt_1k.jsonl"
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 16384
   --eval-top-p 0.95
   --eval-temperature 0.6
)

PERF_ARGS=(
   --qwen-gdn-backend "${QWEN_GDN_BACKEND}"
   --tensor-model-parallel-size "${TP_SIZE}"
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size "${EP_SIZE}"
   --expert-tensor-parallel-size 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
)

# Recompute modes are benchmarked with identical IF samples and longest sequences.
case "${RECOMPUTE_MODE}" in
   full) PERF_ARGS+=(--recompute-granularity full --recompute-method uniform --recompute-num-layers 1) ;;
   selective) echo "Selective MoE recomputation failed finite-gradient validation on this stack; use full for this IF workload." >&2; exit 2 ;;
   none) ;;
   *) echo "Invalid RECOMPUTE_MODE: ${RECOMPUTE_MODE}" >&2; exit 2 ;;
esac
if [[ "${ENABLE_EVAL}" != 1 ]]; then EVAL_ARGS=(); fi

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-rollout-logprobs
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --use-distributed-optimizer
   --use-precision-aware-optimizer
)

# FP32 Adam state for 35B parameters otherwise consumes most of 8x80GB.
# Partial offload is tunable after measuring the long-sequence memory peak.
if [[ "${OPTIMIZER_OFFLOAD_FRACTION}" != 0 && "${OPTIMIZER_OFFLOAD_FRACTION}" != 0.0 ]]; then
   OPTIMIZER_ARGS+=(--optimizer-cpu-offload --optimizer-offload-fraction "${OPTIMIZER_OFFLOAD_FRACTION}" --overlap-cpu-optimizer-d2h-h2d)
fi

WANDB_ARGS=(
   --use-wandb
   --wandb-project "${PROJECT_NAME}"
   --wandb-group "${EXPERIMENT_NAME}"
)

if [[ "${ENABLE_WANDB}" != 1 ]]; then WANDB_ARGS=(); fi

# Measured inference layout; retain headroom for graph and weight-sync buffers.
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${ROLLOUT_TP_SIZE}"
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION:-0.65}"
   --sglang-ep-size 1
   --sglang-mamba-full-memory-ratio "${SGLANG_MAMBA_FULL_MEMORY_RATIO:-3}"
   --sglang-chunked-prefill-size "${SGLANG_CHUNKED_PREFILL_SIZE:-1024}"

   --sglang-max-prefill-tokens "${SGLANG_MAX_PREFILL_TOKENS:-1024}"

   --sglang-cuda-graph-bs-decode 1 2 4 8 16 24 32 48 64

   --sglang-mamba-radix-cache-strategy extra_buffer

   --sglang-server-concurrency "${SGLANG_CONCURRENCY}"
   --sglang-max-running-requests "${SGLANG_CONCURRENCY}"
)

if [[ "${ENABLE_SPECULATIVE}" == 1 ]]; then
   SGLANG_ARGS+=(--sglang-speculative-algorithm NEXTN --sglang-speculative-num-steps "${SPECULATIVE_STEPS:-3}" --sglang-speculative-eagle-topk 1 --sglang-speculative-num-draft-tokens "${SPECULATIVE_DRAFT_TOKENS:-4}")
fi

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --moe-token-dispatcher-type "${MOE_DISPATCHER}"

   --update-weights-interval 1
   --update-weight-buffer-size "${UPDATE_WEIGHT_BUFFER_SIZE:-2147483648}"
)

if [[ "${MOE_DISPATCHER}" == flex ]]; then MISC_ARGS+=(--moe-enable-deepep); fi

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export no_proxy="${no_proxy:+${no_proxy},}localhost,127.0.0.1,${MASTER_ADDR}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}localhost,127.0.0.1,${MASTER_ADDR}"
export RAY_ADDRESS="${MASTER_ADDR}:16379"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
HAS_NVLINK=0
if [ "${NVLINK_COUNT}" -gt 0 ]; then
   HAS_NVLINK=1
fi

# Dedicated ports avoid the other container's Ray cluster on this host.
if ! timeout 10s ray status --address="${RAY_ADDRESS}" >/dev/null 2>&1; then
   ray start --head \
      --node-ip-address "${MASTER_ADDR}" \
      --port 16379 \
      --num-gpus "${NUM_GPUS}" \
      --disable-usage-stats \
      --temp-dir /tmp/ray-slime-rl-if-h800-35ba3b \
      --dashboard-host=0.0.0.0 --dashboard-port 18265 \
      --dashboard-agent-grpc-port 20001 --dashboard-agent-listen-port 20002 \
      --runtime-env-agent-port 20003 --metrics-export-port 20004 \
      --object-manager-port 20005 --node-manager-port 20006 \
      --min-worker-port 21000 --max-worker-port 29999
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${PROJECT_DIR}:/root/Megatron-LM/\",
    \"no_proxy\": \"${no_proxy}\",
    \"NO_PROXY\": \"${NO_PROXY}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"PYTORCH_ALLOC_CONF\": \"${PYTORCH_ALLOC_CONF}\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"OMP_NUM_THREADS\": \"${OMP_NUM_THREADS}\",
    \"MKL_NUM_THREADS\": \"${MKL_NUM_THREADS}\",
    \"NCCL_DEBUG\": \"WARN\",
    \"NVSHMEM_DISABLE_NCCL\": \"1\",
    \"DEPRECATED_MEGATRON_COMPATIBLE\": \"1\",
    \"JOINT_REWARD_WORKERS\": \"${JOINT_REWARD_WORKERS}\",
    \"JOINT_REWARD_SAMPLE_TIMEOUT_S\": \"${JOINT_REWARD_SAMPLE_TIMEOUT_S}\",
    \"IF_REWARD_LOG_PATH\": \"${IF_REWARD_LOG_PATH}\",
    \"MATH_REWARD_LOG_PATH\": \"${MATH_REWARD_LOG_PATH}\",
    \"CODE_REWARD_LOG_PATH\": \"${CODE_REWARD_LOG_PATH}\",
    \"GURU_RL_REWARD_LOG_PATH\": \"${GURU_RL_REWARD_LOG_PATH}\",
    \"REASONING_GYM_REWARD_LOG_PATH\": \"${REASONING_GYM_REWARD_LOG_PATH}\",
    \"MATH_LOGIC_PUZZLE_ZEBRALOGIC_REWARD_LOG_PATH\": \"${MATH_LOGIC_PUZZLE_ZEBRALOGIC_REWARD_LOG_PATH}\"
  }
}"

ray job submit --address="http://127.0.0.1:18265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 "${PROJECT_DIR}/train.py" \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${NUM_GPUS}" \
   --num-gpus-per-node "${NUM_GPUS}" \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "$@"
