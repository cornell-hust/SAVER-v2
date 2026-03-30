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
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${DEFAULT_CHECKPOINT_DIR}}"
MODEL_ROOT="${MODEL_ROOT:-${DATA_ROOT}/Wmh/MLLMs}"

DATA_PATH="${DATA_PATH:-${ANNOTATION_DIR}/msad_saver_agent_train.jsonl}"
INCLUDE_SPLITS="${INCLUDE_SPLITS:-train}"
MODEL_PATH="${MODEL_PATH:-${CHECKPOINT_DIR}/saver_sft_qwen3vl_8b_eval_ddp}"
REFERENCE_MODEL_PATH="${REFERENCE_MODEL_PATH:-${MODEL_PATH}}"
OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_DIR}/saver_cea_grpo_v1}"
PROPOSAL_MODEL_PATH="${PROPOSAL_MODEL_PATH:-}"
PROPOSAL_TORCH_DTYPE="${PROPOSAL_TORCH_DTYPE:-auto}"
PROPOSAL_DEVICE="${PROPOSAL_DEVICE:-}"

EVAL_DATA="${EVAL_DATA:-${ANNOTATION_DIR}/msad_saver_oracle_sft_test.jsonl}"
EVAL_INCLUDE_SPLITS="${EVAL_INCLUDE_SPLITS:-}"
EVAL_PROPOSAL_MODEL_PATH="${EVAL_PROPOSAL_MODEL_PATH:-${PROPOSAL_MODEL_PATH}}"
EVAL_PROPOSAL_TORCH_DTYPE="${EVAL_PROPOSAL_TORCH_DTYPE:-${PROPOSAL_TORCH_DTYPE}}"
EVAL_PROPOSAL_DEVICE="${EVAL_PROPOSAL_DEVICE:-${PROPOSAL_DEVICE}}"
# 设为 0 表示评完整个 eval 数据，不传 --eval-max-records。
EVAL_MAX_RECORDS="${EVAL_MAX_RECORDS:-0}"
EVAL_ROLLOUT_MAX_TURNS="${EVAL_ROLLOUT_MAX_TURNS:-12}"
EVAL_VERIFIER_BACKEND="${EVAL_VERIFIER_BACKEND:-heuristic}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3}"
ROLLOUT_COUNT="${ROLLOUT_COUNT:-16}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
ROLLOUT_MAX_TURNS="${ROLLOUT_MAX_TURNS:-12}"
POLICY_DO_SAMPLE="${POLICY_DO_SAMPLE:-1}"
POLICY_TEMPERATURE="${POLICY_TEMPERATURE:-0.7}"
POLICY_TOP_P="${POLICY_TOP_P:-0.9}"

MIN_WEIGHT="${MIN_WEIGHT:-0.1}"
ADVANTAGE_CLIP="${ADVANTAGE_CLIP:-3.0}"
KL_BETA="${KL_BETA:-0.02}"
GRPO_VARIANT="${GRPO_VARIANT:-cea_grpo}"
CEA_ENABLE_SEARCH_GROUP="${CEA_ENABLE_SEARCH_GROUP:-1}"
CEA_ENABLE_ALERT_GROUP="${CEA_ENABLE_ALERT_GROUP:-1}"
CEA_ENABLE_EVIDENCE_GROUP="${CEA_ENABLE_EVIDENCE_GROUP:-1}"
CEA_SEARCH_LOCAL_ALPHA="${CEA_SEARCH_LOCAL_ALPHA:-0.5}"
CEA_ALERT_LOCAL_ALPHA="${CEA_ALERT_LOCAL_ALPHA:-0.5}"
CEA_EVIDENCE_LOCAL_ALPHA="${CEA_EVIDENCE_LOCAL_ALPHA:-0.5}"
CEA_LOCAL_VERIFIER_BACKEND="${CEA_LOCAL_VERIFIER_BACKEND:-heuristic}"
CEA_LOCAL_USE_REFERENCE_SUPERVISION="${CEA_LOCAL_USE_REFERENCE_SUPERVISION:-0}"
CEA_MAX_SEARCH_ANCHORS_PER_ROLLOUT="${CEA_MAX_SEARCH_ANCHORS_PER_ROLLOUT:-2}"
CEA_MAX_ALERT_ANCHORS_PER_ROLLOUT="${CEA_MAX_ALERT_ANCHORS_PER_ROLLOUT:-2}"
CEA_MAX_EVIDENCE_ANCHORS_PER_ROLLOUT="${CEA_MAX_EVIDENCE_ANCHORS_PER_ROLLOUT:-2}"
VERIFIER_BACKEND="${VERIFIER_BACKEND:-hybrid}"
VERIFIER_MODEL_PATH="${VERIFIER_MODEL_PATH:-${MODEL_ROOT}/qwen3-vl-8b-Instruct}"
VERIFIER_TORCH_DTYPE="${VERIFIER_TORCH_DTYPE:-auto}"
VERIFIER_DEVICE_MAP="${VERIFIER_DEVICE_MAP:-auto}"
VERIFIER_ATTN_IMPLEMENTATION="${VERIFIER_ATTN_IMPLEMENTATION:-}"
VERIFIER_MAX_NEW_TOKENS="${VERIFIER_MAX_NEW_TOKENS:-512}"
VERIFIER_HYBRID_ALPHA="${VERIFIER_HYBRID_ALPHA:-0.7}"

LEARNING_RATE="${LEARNING_RATE:-5e-6}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1.0}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_3}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
MAX_IMAGE_SIDE="${MAX_IMAGE_SIDE:-640}"
MAX_IMAGE_PIXELS="${MAX_IMAGE_PIXELS:-0}"
KEEP_RECENT_TOOL_IMAGE_MESSAGES="${KEEP_RECENT_TOOL_IMAGE_MESSAGES:-0}"
MAX_TOTAL_IMAGES="${MAX_TOTAL_IMAGES:-44}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"
KEEP_RECENT_TEXT_MESSAGES="${KEEP_RECENT_TEXT_MESSAGES:-12}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-2}"
DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-1}"

mkdir -p "${OUTPUT_DIR}"
cd "${CODE_DIR}"

launcher=(python)
if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  launcher=(torchrun --nproc_per_node="${NPROC_PER_NODE}")
fi

cmd=(
  "${launcher[@]}"
  train_saver_rl.py
  --data "${DATA_PATH}"
  --data-root "${DATA_ROOT}"
  --include-splits "${INCLUDE_SPLITS}"
  --model-path "${MODEL_PATH}"
  --reference-model-path "${REFERENCE_MODEL_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --eval-data "${EVAL_DATA}"
  --eval-data-root "${DATA_ROOT}"
  --eval-rollout-max-turns "${EVAL_ROLLOUT_MAX_TURNS}"
  --eval-verifier-backend "${EVAL_VERIFIER_BACKEND}"
  --num-iterations "${NUM_ITERATIONS}"
  --rollout-count "${ROLLOUT_COUNT}"
  --num-generations "${NUM_GENERATIONS}"
  --rollout-max-turns "${ROLLOUT_MAX_TURNS}"
  --min-weight "${MIN_WEIGHT}"
  --advantage-clip "${ADVANTAGE_CLIP}"
  --grpo-variant "${GRPO_VARIANT}"
  --cea-search-local-alpha "${CEA_SEARCH_LOCAL_ALPHA}"
  --cea-alert-local-alpha "${CEA_ALERT_LOCAL_ALPHA}"
  --cea-evidence-local-alpha "${CEA_EVIDENCE_LOCAL_ALPHA}"
  --cea-local-verifier-backend "${CEA_LOCAL_VERIFIER_BACKEND}"
  --cea-max-search-anchors-per-rollout "${CEA_MAX_SEARCH_ANCHORS_PER_ROLLOUT}"
  --cea-max-alert-anchors-per-rollout "${CEA_MAX_ALERT_ANCHORS_PER_ROLLOUT}"
  --cea-max-evidence-anchors-per-rollout "${CEA_MAX_EVIDENCE_ANCHORS_PER_ROLLOUT}"
  --kl-beta "${KL_BETA}"
  --verifier-backend "${VERIFIER_BACKEND}"
  --verifier-model-path "${VERIFIER_MODEL_PATH}"
  --verifier-torch-dtype "${VERIFIER_TORCH_DTYPE}"
  --verifier-device-map "${VERIFIER_DEVICE_MAP}"
  --verifier-max-new-tokens "${VERIFIER_MAX_NEW_TOKENS}"
  --verifier-hybrid-alpha "${VERIFIER_HYBRID_ALPHA}"
  --lora
  --bf16
  --gradient-checkpointing
  --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
  --learning-rate "${LEARNING_RATE}"
  --num-train-epochs "${NUM_TRAIN_EPOCHS}"
  --max-image-side "${MAX_IMAGE_SIDE}"
  --max-image-pixels "${MAX_IMAGE_PIXELS}"
  --keep-recent-tool-image-messages "${KEEP_RECENT_TOOL_IMAGE_MESSAGES}"
  --max-total-images "${MAX_TOTAL_IMAGES}"
  --max-seq-length "${MAX_SEQ_LENGTH}"
  --keep-recent-text-messages "${KEEP_RECENT_TEXT_MESSAGES}"
  --dataloader-num-workers "${DATALOADER_NUM_WORKERS}"
  --dataloader-prefetch-factor "${DATALOADER_PREFETCH_FACTOR}"
  --logging-steps "${LOGGING_STEPS}"
)
if [[ "${DATALOADER_PERSISTENT_WORKERS}" == "1" ]]; then
  cmd+=(--dataloader-persistent-workers)
fi
if [[ -n "${EVAL_INCLUDE_SPLITS}" ]]; then
  cmd+=(--eval-include-splits "${EVAL_INCLUDE_SPLITS}")
fi
if [[ -n "${ATTN_IMPLEMENTATION}" ]]; then
  cmd+=(--attn-implementation "${ATTN_IMPLEMENTATION}")
fi
if [[ -n "${PROPOSAL_MODEL_PATH}" ]]; then
  cmd+=(
    --proposal-model-path "${PROPOSAL_MODEL_PATH}"
    --proposal-torch-dtype "${PROPOSAL_TORCH_DTYPE}"
  )
  if [[ -n "${PROPOSAL_DEVICE}" ]]; then
    cmd+=(--proposal-device "${PROPOSAL_DEVICE}")
  fi
fi
if [[ -n "${EVAL_PROPOSAL_MODEL_PATH}" ]]; then
  cmd+=(
    --eval-proposal-model-path "${EVAL_PROPOSAL_MODEL_PATH}"
    --eval-proposal-torch-dtype "${EVAL_PROPOSAL_TORCH_DTYPE}"
  )
  if [[ -n "${EVAL_PROPOSAL_DEVICE}" ]]; then
    cmd+=(--eval-proposal-device "${EVAL_PROPOSAL_DEVICE}")
  fi
fi
if [[ "${CEA_ENABLE_SEARCH_GROUP}" == "1" ]]; then
  cmd+=(--cea-enable-search-group)
fi
if [[ "${CEA_ENABLE_ALERT_GROUP}" == "1" ]]; then
  cmd+=(--cea-enable-alert-group)
fi
if [[ "${CEA_ENABLE_EVIDENCE_GROUP}" == "1" ]]; then
  cmd+=(--cea-enable-evidence-group)
fi
if [[ "${CEA_LOCAL_USE_REFERENCE_SUPERVISION}" == "1" ]]; then
  cmd+=(--cea-local-use-reference-supervision)
fi
if [[ "${POLICY_DO_SAMPLE}" == "1" ]]; then
  cmd+=(--policy-do-sample --policy-temperature "${POLICY_TEMPERATURE}" --policy-top-p "${POLICY_TOP_P}")
fi
if [[ -n "${VERIFIER_ATTN_IMPLEMENTATION}" ]]; then
  cmd+=(--verifier-attn-implementation "${VERIFIER_ATTN_IMPLEMENTATION}")
fi
if [[ "${EVAL_MAX_RECORDS}" != "0" ]]; then
  cmd+=(--eval-max-records "${EVAL_MAX_RECORDS}")
fi

"${cmd[@]}"
