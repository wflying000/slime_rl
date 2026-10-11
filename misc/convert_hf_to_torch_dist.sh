#!/bin/bash
set -ex

export CUDA_VISIBLE_DEVICES=0,1,2,3

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
SLIME_DIR=$(dirname "$SCRIPT_DIR")
echo "slime dir: ${SLIME_DIR}"

source "${SLIME_DIR}/scripts/models/qwen3.5-9B.sh"

MODEL_NAME="Qwen3.5-9B"
HF_CHECKPOINT=/data/models/${MODEL_NAME}
SAVE_PATH=/data/models/${MODEL_NAME}_slime

PYTHONPATH=/root/Megatron-LM python ${SLIME_DIR}/tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint ${HF_CHECKPOINT} \
    --save ${SAVE_PATH}


# 多卡转换（4卡），单卡转换将 torchrun 行替换为 python
# PYTHONPATH=/root/Megatron-LM torchrun --nproc_per_node=4 ${SLIME_DIR}/tools/convert_hf_to_torch_dist.py \
#     ${MODEL_ARGS[@]} \
#     --hf-checkpoint ${HF_CHECKPOINT} \
#     --save ${SAVE_PATH}

# MASTER_PORT="${MASTER_PORT:-29501}"

# PYTHONPATH=/root/Megatron-LM torchrun \
#     --master_addr=127.0.0.1 \
#     --master_port="${MASTER_PORT}" \
#     --nproc_per_node=4 \
#     "${SLIME_DIR}/tools/convert_hf_to_torch_dist.py" \
#     "${MODEL_ARGS[@]}" \
#     --hf-checkpoint "${HF_CHECKPOINT}" \
#     --save "${SAVE_PATH}"


