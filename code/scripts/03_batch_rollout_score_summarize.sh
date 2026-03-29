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
ROLLOUT_ROOT="${ROLLOUT_ROOT:-${DEFAULT_ROLLOUT_ROOT}}"
MODEL_ROOT="${MODEL_ROOT:-${DATA_ROOT}/Wmh/MLLMs}"

DATA_PATH="${DATA_PATH:-${ANNOTATION_DIR}/msad_saver_oracle_sft_test.jsonl}"
INCLUDE_SPLITS="${INCLUDE_SPLITS:-}"
MODEL_PATH="${MODEL_PATH:-${CHECKPOINT_DIR}/saver_sft_qwen3vl_8b_eval_ddp}"
RUN_NAME="${RUN_NAME:-sft_rollout_eval}"
RUN_DIR="${RUN_DIR:-${ROLLOUT_ROOT}/${RUN_NAME}}"

START_INDEX="${START_INDEX:-0}"
# 设为 0 表示从 START_INDEX 开始一路跑到该 split 结束，不截断。
COUNT="${COUNT:-0}"
MAX_TURNS="${MAX_TURNS:-12}"
PROGRESS_EVERY="${PROGRESS_EVERY:-5}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_3}"

DO_SAMPLE="${DO_SAMPLE:-0}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
TOP_K="${TOP_K:-}"
REPETITION_PENALTY="${REPETITION_PENALTY:-}"

VERIFIER_BACKEND="${VERIFIER_BACKEND:-heuristic}"
VERIFIER_MODEL_PATH="${VERIFIER_MODEL_PATH:-${MODEL_ROOT}/qwen3-vl-8b-Instruct}"
VERIFIER_TORCH_DTYPE="${VERIFIER_TORCH_DTYPE:-auto}"
VERIFIER_DEVICE_MAP="${VERIFIER_DEVICE_MAP:-auto}"
VERIFIER_ATTN_IMPLEMENTATION="${VERIFIER_ATTN_IMPLEMENTATION:-}"
VERIFIER_MAX_NEW_TOKENS="${VERIFIER_MAX_NEW_TOKENS:-512}"
VERIFIER_HYBRID_ALPHA="${VERIFIER_HYBRID_ALPHA:-0.7}"

RAW_OUTPUT="${RAW_OUTPUT:-${RUN_DIR}/rollouts.raw.jsonl}"
SCORED_OUTPUT="${SCORED_OUTPUT:-${RUN_DIR}/rollouts.scored.jsonl}"
SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-${RUN_DIR}/summary.json}"

mkdir -p "${RUN_DIR}"
cd "${CODE_DIR}"

batch_cmd=(
  python batch_run_saver_rollout.py
  --data "${DATA_PATH}"
  --data-root "${DATA_ROOT}"
  --start-index "${START_INDEX}"
  --count "${COUNT}"
  --max-turns "${MAX_TURNS}"
  --policy-backend qwen
  --model-path "${MODEL_PATH}"
  --torch-dtype "${TORCH_DTYPE}"
  --device-map "${DEVICE_MAP}"
  --output "${RAW_OUTPUT}"
  --progress-every "${PROGRESS_EVERY}"
  --verifier-backend "${VERIFIER_BACKEND}"
  --verifier-model-path "${VERIFIER_MODEL_PATH}"
  --verifier-torch-dtype "${VERIFIER_TORCH_DTYPE}"
  --verifier-device-map "${VERIFIER_DEVICE_MAP}"
  --verifier-max-new-tokens "${VERIFIER_MAX_NEW_TOKENS}"
  --verifier-hybrid-alpha "${VERIFIER_HYBRID_ALPHA}"
)
if [[ -n "${INCLUDE_SPLITS}" ]]; then
  batch_cmd+=(--include-splits "${INCLUDE_SPLITS}")
fi
if [[ -n "${ATTN_IMPLEMENTATION}" ]]; then
  batch_cmd+=(--attn-implementation "${ATTN_IMPLEMENTATION}")
fi
if [[ "${DO_SAMPLE}" == "1" ]]; then
  batch_cmd+=(--do-sample --temperature "${TEMPERATURE}" --top-p "${TOP_P}")
fi
if [[ -n "${TOP_K}" ]]; then
  batch_cmd+=(--top-k "${TOP_K}")
fi
if [[ -n "${REPETITION_PENALTY}" ]]; then
  batch_cmd+=(--repetition-penalty "${REPETITION_PENALTY}")
fi
if [[ -n "${VERIFIER_ATTN_IMPLEMENTATION}" ]]; then
  batch_cmd+=(--verifier-attn-implementation "${VERIFIER_ATTN_IMPLEMENTATION}")
fi

echo "[1/3] Running batch rollout to ${RAW_OUTPUT}"
"${batch_cmd[@]}"

score_cmd=(
  python score_saver_rollout.py
  --input "${RAW_OUTPUT}"
  --output "${SCORED_OUTPUT}"
  --data "${DATA_PATH}"
  --data-root "${DATA_ROOT}"
  --verifier-backend "${VERIFIER_BACKEND}"
  --verifier-model-path "${VERIFIER_MODEL_PATH}"
  --verifier-torch-dtype "${VERIFIER_TORCH_DTYPE}"
  --verifier-device-map "${VERIFIER_DEVICE_MAP}"
  --verifier-max-new-tokens "${VERIFIER_MAX_NEW_TOKENS}"
  --verifier-hybrid-alpha "${VERIFIER_HYBRID_ALPHA}"
  --progress-every "${PROGRESS_EVERY}"
  --force-reverify
)
if [[ -n "${VERIFIER_ATTN_IMPLEMENTATION}" ]]; then
  score_cmd+=(--verifier-attn-implementation "${VERIFIER_ATTN_IMPLEMENTATION}")
fi

echo "[2/3] Scoring rollouts to ${SCORED_OUTPUT}"
"${score_cmd[@]}"

echo "[3/3] Summarizing scored rollouts to ${SUMMARY_OUTPUT}"
python summarize_saver_scores.py \
  --input "${SCORED_OUTPUT}" \
  --output "${SUMMARY_OUTPUT}" \
  --data "${DATA_PATH}" \
  --data-root "${DATA_ROOT}"

echo "Done."
echo "RAW_OUTPUT=${RAW_OUTPUT}"
echo "SCORED_OUTPUT=${SCORED_OUTPUT}"
echo "SUMMARY_OUTPUT=${SUMMARY_OUTPUT}"
