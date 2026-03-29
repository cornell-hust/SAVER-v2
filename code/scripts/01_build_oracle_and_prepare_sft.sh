#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-/mnt/shared-storage-user/mineru2-shared/zengweijun}"
EXP_ROOT="${EXP_ROOT:-${DATA_ROOT}/Wmh/ideas/idea2_v2}"
DATA_UTILS_DIR="${DATA_UTILS_DIR:-${CODE_DIR}/data_utils}"
ANNOTATION_DIR="${ANNOTATION_DIR:-${DATA_UTILS_DIR}}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${DATA_UTILS_DIR}}"

CANONICAL_JSONL="${CANONICAL_JSONL:-${ANNOTATION_DIR}/msad_saver_with_qwen.jsonl}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
ADAPTER="${ADAPTER:-msad_saver_qwen}"

AGENT_TRAIN_JSONL="${AGENT_TRAIN_JSONL:-${ANNOTATION_DIR}/msad_saver_agent_train.jsonl}"
ORACLE_TRAIN_JSONL="${ORACLE_TRAIN_JSONL:-${ANNOTATION_DIR}/msad_saver_oracle_sft_train.jsonl}"
ORACLE_EVAL_JSONL="${ORACLE_EVAL_JSONL:-${ANNOTATION_DIR}/msad_saver_oracle_sft_${EVAL_SPLIT}.jsonl}"
PREPARED_TRAIN_JSONL="${PREPARED_TRAIN_JSONL:-${ARTIFACT_DIR}/msad_saver_agent_train.prepared_sft.jsonl}"

PROGRESS_EVERY="${PROGRESS_EVERY:-25}"
VALIDATE_MATERIALIZATION="${VALIDATE_MATERIALIZATION:-0}"
VALIDATION_MAX_EXAMPLES="${VALIDATION_MAX_EXAMPLES:-0}"

mkdir -p "${ANNOTATION_DIR}" "${ARTIFACT_DIR}"
cd "${CODE_DIR}"

echo "[1/4] Building agent_train view: ${AGENT_TRAIN_JSONL}"
python convert_to_saver_agent.py \
  --input "${CANONICAL_JSONL}" \
  --output "${AGENT_TRAIN_JSONL}" \
  --adapter "${ADAPTER}" \
  --mode agent_train \
  --include-splits "${TRAIN_SPLIT}"

echo "[2/4] Building oracle_sft train view: ${ORACLE_TRAIN_JSONL}"
python convert_to_saver_agent.py \
  --input "${CANONICAL_JSONL}" \
  --output "${ORACLE_TRAIN_JSONL}" \
  --adapter "${ADAPTER}" \
  --mode oracle_sft \
  --include-splits "${TRAIN_SPLIT}"

echo "[3/4] Building oracle_sft eval view (${EVAL_SPLIT}): ${ORACLE_EVAL_JSONL}"
python convert_to_saver_agent.py \
  --input "${CANONICAL_JSONL}" \
  --output "${ORACLE_EVAL_JSONL}" \
  --adapter "${ADAPTER}" \
  --mode oracle_sft \
  --include-splits "${EVAL_SPLIT}"

echo "[4/4] Preparing lightweight SFT JSONL: ${PREPARED_TRAIN_JSONL}"
prepare_cmd=(
  python train_saver_sft.py
  --data "${AGENT_TRAIN_JSONL}"
  --data-root "${DATA_ROOT}"
  --include-splits "${TRAIN_SPLIT}"
  --prepare-output "${PREPARED_TRAIN_JSONL}"
  --prepare-only
  --validate-prepared-data
  --progress-every "${PROGRESS_EVERY}"
)
if [[ "${VALIDATE_MATERIALIZATION}" == "1" ]]; then
  prepare_cmd+=(--validate-materialization)
fi
if [[ "${VALIDATION_MAX_EXAMPLES}" != "0" ]]; then
  prepare_cmd+=(--validation-max-examples "${VALIDATION_MAX_EXAMPLES}")
fi
"${prepare_cmd[@]}"

echo "Done."
echo "AGENT_TRAIN_JSONL=${AGENT_TRAIN_JSONL}"
echo "ORACLE_TRAIN_JSONL=${ORACLE_TRAIN_JSONL}"
echo "ORACLE_EVAL_JSONL=${ORACLE_EVAL_JSONL}"
echo "PREPARED_TRAIN_JSONL=${PREPARED_TRAIN_JSONL}"
