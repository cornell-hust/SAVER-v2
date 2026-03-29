#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/common_experiment.sh"

DATA_ROOT="${DATA_ROOT:-/mnt/shared-storage-user/mineru2-shared/zengweijun}"
EXP_ROOT="${EXP_ROOT:-${DATA_ROOT}/Wmh/ideas/idea2_v2}"
DATA_UTILS_DIR="${DATA_UTILS_DIR:-${CODE_DIR}/data_utils}"
configure_experiment_layout "${CODE_DIR}" "${EXP_ROOT}" "${DATA_UTILS_DIR}"
ANNOTATION_DIR="${ANNOTATION_DIR:-${DEFAULT_ANNOTATION_DIR}}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${DEFAULT_ARTIFACT_DIR}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${DEFAULT_CHECKPOINT_DIR}}"
MODEL_ROOT="${MODEL_ROOT:-${DATA_ROOT}/Wmh/MLLMs}"

PREPARED_TRAIN_JSONL="${PREPARED_TRAIN_JSONL:-${ARTIFACT_DIR}/msad_saver_agent_train.prepared_sft.jsonl}"
EVAL_DATA="${EVAL_DATA:-${ANNOTATION_DIR}/msad_saver_oracle_sft_test.jsonl}"
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/qwen3-vl-8b-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_DIR}/saver_sft_qwen3vl_8b_eval_ddp}"

INCLUDE_SPLITS="${INCLUDE_SPLITS:-train}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2.0}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_3}"

# 设为 0 表示评完整个 eval 数据，不传 --eval-max-records。
EVAL_MAX_RECORDS="${EVAL_MAX_RECORDS:-0}"
EVAL_ROLLOUT_MAX_TURNS="${EVAL_ROLLOUT_MAX_TURNS:-12}"
EVAL_VERIFIER_BACKEND="${EVAL_VERIFIER_BACKEND:-heuristic}"
EVAL_PROGRESS_EVERY="${EVAL_PROGRESS_EVERY:-1}"

MAX_IMAGE_SIDE="${MAX_IMAGE_SIDE:-0}"
MAX_IMAGE_PIXELS="${MAX_IMAGE_PIXELS:-0}"
KEEP_RECENT_TOOL_IMAGE_MESSAGES="${KEEP_RECENT_TOOL_IMAGE_MESSAGES:-0}"
MAX_TOTAL_IMAGES="${MAX_TOTAL_IMAGES:-0}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"
KEEP_RECENT_TEXT_MESSAGES="${KEEP_RECENT_TEXT_MESSAGES:-12}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-2}"
DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-1}"

mkdir -p "${CHECKPOINT_DIR}"
cd "${CODE_DIR}"

if [[ ! -f "${PREPARED_TRAIN_JSONL}" ]]; then
  echo "Missing prepared SFT file: ${PREPARED_TRAIN_JSONL}" >&2
  exit 1
fi
if [[ ! -f "${EVAL_DATA}" ]]; then
  echo "Missing eval data file: ${EVAL_DATA}" >&2
  exit 1
fi

launcher=(python)
if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  launcher=(torchrun --nproc_per_node="${NPROC_PER_NODE}")
fi

cmd=(
  "${launcher[@]}"
  train_saver_sft.py
  --prepared-data "${PREPARED_TRAIN_JSONL}"
  --include-splits "${INCLUDE_SPLITS}"
  --model-path "${MODEL_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --eval-data "${EVAL_DATA}"
  --eval-data-root "${DATA_ROOT}"
  --eval-rollout-max-turns "${EVAL_ROLLOUT_MAX_TURNS}"
  --eval-verifier-backend "${EVAL_VERIFIER_BACKEND}"
  --eval-progress-every "${EVAL_PROGRESS_EVERY}"
  --lora
  --bf16
  --gradient-checkpointing
  --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
  --learning-rate "${LEARNING_RATE}"
  --num-train-epochs "${NUM_TRAIN_EPOCHS}"
  --dataloader-num-workers "${DATALOADER_NUM_WORKERS}"
  --dataloader-prefetch-factor "${DATALOADER_PREFETCH_FACTOR}"
  --logging-steps "${LOGGING_STEPS}"
)
if [[ "${DATALOADER_PERSISTENT_WORKERS}" == "1" ]]; then
  cmd+=(--dataloader-persistent-workers)
fi
if [[ -n "${ATTN_IMPLEMENTATION}" ]]; then
  cmd+=(--attn-implementation "${ATTN_IMPLEMENTATION}")
fi
if [[ "${MAX_IMAGE_SIDE}" != "0" ]]; then
  cmd+=(--max-image-side "${MAX_IMAGE_SIDE}")
fi
if [[ "${MAX_IMAGE_PIXELS}" != "0" ]]; then
  cmd+=(--max-image-pixels "${MAX_IMAGE_PIXELS}")
fi
if [[ "${KEEP_RECENT_TOOL_IMAGE_MESSAGES}" != "0" ]]; then
  cmd+=(--keep-recent-tool-image-messages "${KEEP_RECENT_TOOL_IMAGE_MESSAGES}")
fi
if [[ "${MAX_TOTAL_IMAGES}" != "0" ]]; then
  cmd+=(--max-total-images "${MAX_TOTAL_IMAGES}")
fi
if [[ "${MAX_SEQ_LENGTH}" != "0" ]]; then
  cmd+=(--max-seq-length "${MAX_SEQ_LENGTH}")
fi
if [[ "${KEEP_RECENT_TEXT_MESSAGES}" != "0" ]]; then
  cmd+=(--keep-recent-text-messages "${KEEP_RECENT_TEXT_MESSAGES}")
fi
if [[ "${EVAL_MAX_RECORDS}" != "0" ]]; then
  cmd+=(--eval-max-records "${EVAL_MAX_RECORDS}")
fi

"${cmd[@]}"
