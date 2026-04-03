#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/common_experiment.sh"

DATA_ROOT="${DATA_ROOT:-/mnt/shared-storage-user/mineru2-shared/zengweijun}"
EXP_ROOT="${EXP_ROOT:-${DATA_ROOT}/Wmh/ideas/idea2_v2}"
DATA_UTILS_DIR="${DATA_UTILS_DIR:-${CODE_DIR}/data_utils}"
configure_experiment_layout "${CODE_DIR}" "${EXP_ROOT}" "${DATA_UTILS_DIR}"
LOG_DIR="${LOG_DIR:-${DEFAULT_SFT_LOG_DIR}}"
configure_script_logging "02_train_sft_with_rollout_eval"
ANNOTATION_DIR="${ANNOTATION_DIR:-${DEFAULT_ANNOTATION_DIR}}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${DEFAULT_ARTIFACT_DIR}}"
MODEL_ROOT="${MODEL_ROOT:-${DATA_ROOT}/Wmh/MLLMs}"

if [[ -n "${SFT_CHECKPOINT_DIR:-}" ]]; then
  CHECKPOINT_DIR="${SFT_CHECKPOINT_DIR}"
elif [[ -n "${CHECKPOINT_DIR:-}" ]]; then
  CHECKPOINT_DIR="${CHECKPOINT_DIR}"
else
  CHECKPOINT_DIR="${DEFAULT_SFT_CHECKPOINT_DIR}"
fi
ROLLOUT_EVAL_OUTPUT_DIR="${ROLLOUT_EVAL_OUTPUT_DIR:-${DEFAULT_SFT_EVAL_DIR}}"

PREPARED_TRAIN_JSONL="${PREPARED_TRAIN_JSONL:-${DATA_UTILS_DIR}/msad_saver_sft_train.jsonl}"
TEACHER_PREPARED_TRAIN_JSONL="${TEACHER_PREPARED_TRAIN_JSONL:-${DATA_UTILS_DIR}/msad_saver_sft_train.teacher.jsonl}"
EVAL_DATA="${EVAL_DATA:-${DATA_UTILS_DIR}/msad_saver_runtime_test.jsonl}"
EVAL_INCLUDE_SPLITS="${EVAL_INCLUDE_SPLITS:-test}"
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/qwen3-vl-8b-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_DIR}/saver_sft_qwen3vl_8b_eval_ddp}"
PROPOSAL_MODEL_PATH="${PROPOSAL_MODEL_PATH:-}"
PROPOSAL_TORCH_DTYPE="${PROPOSAL_TORCH_DTYPE:-auto}"
PROPOSAL_DEVICE="${PROPOSAL_DEVICE:-}"
EVAL_PROPOSAL_MODEL_PATH="${EVAL_PROPOSAL_MODEL_PATH:-${PROPOSAL_MODEL_PATH}}"
EVAL_PROPOSAL_TORCH_DTYPE="${EVAL_PROPOSAL_TORCH_DTYPE:-${PROPOSAL_TORCH_DTYPE}}"
EVAL_PROPOSAL_DEVICE="${EVAL_PROPOSAL_DEVICE:-${PROPOSAL_DEVICE}}"
TEACHER_JUDGE_MODEL_PATH="${TEACHER_JUDGE_MODEL_PATH:-}"
TEACHER_JUDGE_INPUT_MODE="${TEACHER_JUDGE_INPUT_MODE:-auto}"
TEACHER_JUDGE_TORCH_DTYPE="${TEACHER_JUDGE_TORCH_DTYPE:-auto}"
TEACHER_JUDGE_DEVICE_MAP="${TEACHER_JUDGE_DEVICE_MAP:-auto}"
TEACHER_JUDGE_ATTN_IMPLEMENTATION="${TEACHER_JUDGE_ATTN_IMPLEMENTATION:-}"
TEACHER_JUDGE_MAX_NEW_TOKENS="${TEACHER_JUDGE_MAX_NEW_TOKENS:-384}"
TEACHER_JUDGE_MAX_IMAGES="${TEACHER_JUDGE_MAX_IMAGES:-8}"
TEACHER_JUDGE_TOPK_FRAMES_PER_VIEW="${TEACHER_JUDGE_TOPK_FRAMES_PER_VIEW:-4}"
TEACHER_JUDGE_PROGRESS_EVERY="${TEACHER_JUDGE_PROGRESS_EVERY:-25}"
TEACHER_JUDGE_OVERWRITE_EXISTING="${TEACHER_JUDGE_OVERWRITE_EXISTING:-0}"

INCLUDE_SPLITS="${INCLUDE_SPLITS:-train}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
INLINE_ROLLOUT_EVAL="${INLINE_ROLLOUT_EVAL:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2.0}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_3}"

# 设为 0 表示评完整个 eval 数据，不传 --eval-max-records。
EVAL_MAX_RECORDS="${EVAL_MAX_RECORDS:-0}"
EVAL_ROLLOUT_MAX_TURNS="${EVAL_ROLLOUT_MAX_TURNS:-14}"
EVAL_MAX_NEW_TOKENS_PER_TURN="${EVAL_MAX_NEW_TOKENS_PER_TURN:-256}"
EVAL_MAX_TOTAL_IMAGES="${EVAL_MAX_TOTAL_IMAGES:-24}"
EVAL_VERIFIER_BACKEND="${EVAL_VERIFIER_BACKEND:-heuristic}"
EVAL_ATTACH_REFERENCE_DIAGNOSTICS="${EVAL_ATTACH_REFERENCE_DIAGNOSTICS:-0}"
EVAL_VERIFIER_MODEL_PATH="${EVAL_VERIFIER_MODEL_PATH:-${MODEL_ROOT}/qwen3-vl-8b-Instruct}"
EVAL_VERIFIER_TORCH_DTYPE="${EVAL_VERIFIER_TORCH_DTYPE:-auto}"
EVAL_VERIFIER_DEVICE_MAP="${EVAL_VERIFIER_DEVICE_MAP:-auto}"
EVAL_VERIFIER_ATTN_IMPLEMENTATION="${EVAL_VERIFIER_ATTN_IMPLEMENTATION:-}"
EVAL_VERIFIER_MAX_NEW_TOKENS="${EVAL_VERIFIER_MAX_NEW_TOKENS:-512}"
EVAL_VERIFIER_HYBRID_ALPHA="${EVAL_VERIFIER_HYBRID_ALPHA:-0.7}"
EVAL_PROGRESS_EVERY="${EVAL_PROGRESS_EVERY:-1}"

MAX_IMAGE_SIDE="${MAX_IMAGE_SIDE:-0}"
MAX_IMAGE_PIXELS="${MAX_IMAGE_PIXELS:-0}"
KEEP_RECENT_TOOL_IMAGE_MESSAGES="${KEEP_RECENT_TOOL_IMAGE_MESSAGES:-0}"
MAX_TOTAL_IMAGES="${MAX_TOTAL_IMAGES:-24}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"
KEEP_RECENT_TEXT_MESSAGES="${KEEP_RECENT_TEXT_MESSAGES:-12}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-2}"
DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-1}"

is_complete_model_checkpoint_dir() {
  local candidate="$1"
  [[ -d "${candidate}" ]] || return 1
  [[ -f "${candidate}/adapter_config.json" || -f "${candidate}/config.json" ]] || return 1
  [[ -f "${candidate}/adapter_model.safetensors" || -f "${candidate}/pytorch_model.bin" || -f "${candidate}/model.safetensors" ]] || return 1
  [[ -f "${candidate}/preprocessor_config.json" || -f "${candidate}/processor_config.json" || -f "${candidate}/tokenizer_config.json" ]] || return 1
  return 0
}

rollout_eval_metrics_path_for_epoch() {
  local rollout_eval_output_dir="$1"
  local epoch_index="$2"
  printf '%s\n' "${rollout_eval_output_dir}/epoch_$(printf '%03d' "${epoch_index}")/metrics.json"
}

legacy_rollout_eval_metrics_path_for_epoch() {
  local rollout_eval_output_dir="$1"
  local epoch_index="$2"
  printf '%s\n' "${rollout_eval_output_dir}/rollout_eval/epoch_$(printf '%03d' "${epoch_index}")/metrics.json"
}

rollout_eval_metrics_exist_for_epoch() {
  local rollout_eval_output_dir="$1"
  local epoch_index="$2"
  local metrics_path=""
  local legacy_metrics_path=""

  metrics_path="$(rollout_eval_metrics_path_for_epoch "${rollout_eval_output_dir}" "${epoch_index}")"
  legacy_metrics_path="$(legacy_rollout_eval_metrics_path_for_epoch "${rollout_eval_output_dir}" "${epoch_index}")"
  [[ -f "${metrics_path}" || -f "${legacy_metrics_path}" ]]
}

run_pending_external_rollout_evals() {
  local output_dir="$1"
  local rollout_eval_output_dir="$2"
  local checkpoint_dir=""
  local base_name=""
  local epoch_index=""
  local metrics_path=""

  for checkpoint_dir in "${output_dir}"/epoch_resume/epoch_*; do
    [[ -d "${checkpoint_dir}" ]] || continue
    base_name="$(basename "${checkpoint_dir}")"
    if [[ ! "${base_name}" =~ ^epoch_([0-9]+)$ ]]; then
      continue
    fi
    if ! is_complete_model_checkpoint_dir "${checkpoint_dir}"; then
      continue
    fi
    epoch_index="${BASH_REMATCH[1]}"
    if rollout_eval_metrics_exist_for_epoch "${rollout_eval_output_dir}" "${epoch_index}"; then
      continue
    fi
    eval_cmd=(
      "${launcher[@]}"
      train_saver_sft.py
      --prepared-data "${EFFECTIVE_PREPARED_TRAIN_JSONL}"
      --include-splits "${INCLUDE_SPLITS}"
      --model-path "${MODEL_PATH}"
      --output-dir "${OUTPUT_DIR}"
      --log-dir "${LOG_DIR}"
      --rollout-eval-output-dir "${ROLLOUT_EVAL_OUTPUT_DIR}"
      --eval-data "${EVAL_DATA}"
      --eval-data-root "${DATA_ROOT}"
      --eval-include-splits "${EVAL_INCLUDE_SPLITS}"
      --eval-rollout-max-turns "${EVAL_ROLLOUT_MAX_TURNS}"
      --eval-max-new-tokens-per-turn "${EVAL_MAX_NEW_TOKENS_PER_TURN}"
      --eval-progress-every "${EVAL_PROGRESS_EVERY}"
      --resume-from-checkpoint "${checkpoint_dir}"
      --resume-rollout-eval-only
    )
    if [[ -n "${ATTN_IMPLEMENTATION}" ]]; then
      eval_cmd+=(--attn-implementation "${ATTN_IMPLEMENTATION}")
    fi
    if [[ "${EVAL_MAX_RECORDS}" != "0" ]]; then
      eval_cmd+=(--eval-max-records "${EVAL_MAX_RECORDS}")
    fi
    if [[ "${EVAL_MAX_TOTAL_IMAGES}" != "0" ]]; then
      eval_cmd+=(--eval-max-total-images "${EVAL_MAX_TOTAL_IMAGES}")
    fi
    if [[ "${EVAL_ATTACH_REFERENCE_DIAGNOSTICS}" == "1" ]]; then
      eval_cmd+=(
        --eval-attach-reference-diagnostics
        --eval-verifier-backend "${EVAL_VERIFIER_BACKEND}"
        --eval-verifier-model-path "${EVAL_VERIFIER_MODEL_PATH}"
        --eval-verifier-torch-dtype "${EVAL_VERIFIER_TORCH_DTYPE}"
        --eval-verifier-device-map "${EVAL_VERIFIER_DEVICE_MAP}"
        --eval-verifier-max-new-tokens "${EVAL_VERIFIER_MAX_NEW_TOKENS}"
        --eval-verifier-hybrid-alpha "${EVAL_VERIFIER_HYBRID_ALPHA}"
      )
      if [[ -n "${EVAL_VERIFIER_ATTN_IMPLEMENTATION}" ]]; then
        eval_cmd+=(--eval-verifier-attn-implementation "${EVAL_VERIFIER_ATTN_IMPLEMENTATION}")
      fi
    fi
    if [[ -n "${EVAL_PROPOSAL_MODEL_PATH}" ]]; then
      eval_cmd+=(
        --eval-proposal-model-path "${EVAL_PROPOSAL_MODEL_PATH}"
        --eval-proposal-torch-dtype "${EVAL_PROPOSAL_TORCH_DTYPE}"
      )
      if [[ -n "${EVAL_PROPOSAL_DEVICE}" ]]; then
        eval_cmd+=(--eval-proposal-device "${EVAL_PROPOSAL_DEVICE}")
      fi
    fi
    "${eval_cmd[@]}"
  done
}

cd "${CODE_DIR}"

if [[ ! -f "${PREPARED_TRAIN_JSONL}" ]]; then
  echo "Missing prepared SFT file: ${PREPARED_TRAIN_JSONL}" >&2
  exit 1
fi
if [[ ! -f "${EVAL_DATA}" ]]; then
  echo "Missing eval data file: ${EVAL_DATA}" >&2
  exit 1
fi

EFFECTIVE_PREPARED_TRAIN_JSONL="${PREPARED_TRAIN_JSONL}"
if [[ -n "${TEACHER_JUDGE_MODEL_PATH}" ]]; then
  if [[ ! -f "${TEACHER_PREPARED_TRAIN_JSONL}" ]]; then
    teacher_cmd=(
      python annotate_teacher_judge_sft.py
      --input "${PREPARED_TRAIN_JSONL}"
      --output "${TEACHER_PREPARED_TRAIN_JSONL}"
      --include-splits "${INCLUDE_SPLITS}"
      --model-path "${TEACHER_JUDGE_MODEL_PATH}"
      --input-mode "${TEACHER_JUDGE_INPUT_MODE}"
      --torch-dtype "${TEACHER_JUDGE_TORCH_DTYPE}"
      --device-map "${TEACHER_JUDGE_DEVICE_MAP}"
      --max-new-tokens "${TEACHER_JUDGE_MAX_NEW_TOKENS}"
      --max-images "${TEACHER_JUDGE_MAX_IMAGES}"
      --topk-frames-per-view "${TEACHER_JUDGE_TOPK_FRAMES_PER_VIEW}"
      --progress-every "${TEACHER_JUDGE_PROGRESS_EVERY}"
    )
    if [[ -n "${TEACHER_JUDGE_ATTN_IMPLEMENTATION}" ]]; then
      teacher_cmd+=(--attn-implementation "${TEACHER_JUDGE_ATTN_IMPLEMENTATION}")
    fi
    if [[ "${TEACHER_JUDGE_OVERWRITE_EXISTING}" == "1" ]]; then
      teacher_cmd+=(--overwrite-existing)
    fi
    "${teacher_cmd[@]}"
  fi
  EFFECTIVE_PREPARED_TRAIN_JSONL="${TEACHER_PREPARED_TRAIN_JSONL}"
fi

launcher=(python)
if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  launcher=(torchrun --nproc_per_node="${NPROC_PER_NODE}")
fi

cmd=(
  "${launcher[@]}"
  train_saver_sft.py
  --prepared-data "${EFFECTIVE_PREPARED_TRAIN_JSONL}"
  --include-splits "${INCLUDE_SPLITS}"
  --model-path "${MODEL_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --log-dir "${LOG_DIR}"
  --rollout-eval-output-dir "${ROLLOUT_EVAL_OUTPUT_DIR}"
  --eval-data "${EVAL_DATA}"
  --eval-data-root "${DATA_ROOT}"
  --eval-include-splits "${EVAL_INCLUDE_SPLITS}"
  --eval-rollout-max-turns "${EVAL_ROLLOUT_MAX_TURNS}"
  --eval-max-new-tokens-per-turn "${EVAL_MAX_NEW_TOKENS_PER_TURN}"
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
if [[ "${INLINE_ROLLOUT_EVAL}" == "1" ]]; then
  cmd+=(--inline-rollout-eval)
fi
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
if [[ "${EVAL_MAX_TOTAL_IMAGES}" != "0" ]]; then
  cmd+=(--eval-max-total-images "${EVAL_MAX_TOTAL_IMAGES}")
fi
if [[ "${EVAL_ATTACH_REFERENCE_DIAGNOSTICS}" == "1" ]]; then
  cmd+=(
    --eval-attach-reference-diagnostics
    --eval-verifier-backend "${EVAL_VERIFIER_BACKEND}"
    --eval-verifier-model-path "${EVAL_VERIFIER_MODEL_PATH}"
    --eval-verifier-torch-dtype "${EVAL_VERIFIER_TORCH_DTYPE}"
    --eval-verifier-device-map "${EVAL_VERIFIER_DEVICE_MAP}"
    --eval-verifier-max-new-tokens "${EVAL_VERIFIER_MAX_NEW_TOKENS}"
    --eval-verifier-hybrid-alpha "${EVAL_VERIFIER_HYBRID_ALPHA}"
  )
  if [[ -n "${EVAL_VERIFIER_ATTN_IMPLEMENTATION}" ]]; then
    cmd+=(--eval-verifier-attn-implementation "${EVAL_VERIFIER_ATTN_IMPLEMENTATION}")
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

"${cmd[@]}"
run_pending_external_rollout_evals "${OUTPUT_DIR}" "${ROLLOUT_EVAL_OUTPUT_DIR}"
