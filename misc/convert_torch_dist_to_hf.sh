

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PROJECT_DIR=$(dirname "$SCRIPT_DIR")
echo "project dir: ${PROJECT_DIR}"


INPUT_DIR=/path/to/saved_checkpoint
OUTPUT_DIR=/path/to/output
ORIGIN_HF_DIR=/path/to/origin_hf_model


python ${PROJECT_DIR}/tools/convert_torch_dist_to_hf.py \
    --input-dir ${INPUT_DIR} \
    --output-dir  ${OUTPUT_DIR} \
    --origin-hf-dir  ${ORIGIN_HF_DIR} \
    --add-missing-from-origin-hf

