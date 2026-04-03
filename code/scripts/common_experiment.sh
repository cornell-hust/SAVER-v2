#!/usr/bin/env bash

trim_experiment_value() {
  local value="${1-}"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

validate_experiment_name() {
  local value
  value="$(trim_experiment_value "${1-}")"
  if [[ -z "${value}" ]]; then
    return 0
  fi
  if [[ "${value}" == "." || "${value}" == ".." ]]; then
    echo "EXP_NAME cannot be '.' or '..'." >&2
    return 1
  fi
  if [[ "${value}" == */* ]]; then
    echo "EXP_NAME must be a single directory name without '/'." >&2
    return 1
  fi
  return 0
}

configure_experiment_layout() {
  local code_dir="$1"
  local default_exp_root="$2"
  local default_annotation_dir="$3"
  local experiment_base_dir="${EXPERIMENT_BASE_DIR:-${code_dir}/ckpt}"
  local exp_name
  exp_name="$(trim_experiment_value "${EXP_NAME:-}")"

  if [[ -z "${exp_name}" && "${PROMPT_EXP_NAME:-1}" == "1" ]]; then
    local has_manual_output_override=0
    local override_var
    for override_var in \
      ARTIFACT_DIR \
      CHECKPOINT_DIR \
      ROLLOUT_ROOT \
      OUTPUT_DIR \
      RUN_DIR \
      SFT_OUTPUT_DIR \
      RL_OUTPUT_DIR \
      PREPARED_TRAIN_JSONL \
      RAW_OUTPUT \
      SCORED_OUTPUT \
      SUMMARY_OUTPUT \
      FRAME_CACHE_SUMMARY_OUTPUT \
      FEATURE_CACHE_SUMMARY_OUTPUT; do
      if [[ -n "${!override_var:-}" ]]; then
        has_manual_output_override=1
        break
      fi
    done
    if [[ "${has_manual_output_override}" == "0" && -t 0 && -t 1 ]]; then
      local input_name=""
      read -r -p "Experiment name for outputs under ${experiment_base_dir} (blank = default layout): " input_name || true
      exp_name="$(trim_experiment_value "${input_name}")"
    fi
  fi

  validate_experiment_name "${exp_name}"

  DEFAULT_ANNOTATION_DIR="${default_annotation_dir}"
  if [[ -n "${exp_name}" ]]; then
    RUN_BASE_DIR="${experiment_base_dir}/${exp_name}"
    DEFAULT_ARTIFACT_DIR="${RUN_BASE_DIR}/artifacts"
    DEFAULT_CHECKPOINT_DIR="${RUN_BASE_DIR}/checkpoints"
    DEFAULT_SFT_CHECKPOINT_DIR="${DEFAULT_CHECKPOINT_DIR}/sft"
    DEFAULT_RL_CHECKPOINT_DIR="${DEFAULT_CHECKPOINT_DIR}/rl"
    DEFAULT_EVAL_ROOT="${RUN_BASE_DIR}/eval"
    DEFAULT_SFT_EVAL_DIR="${DEFAULT_EVAL_ROOT}/sft_epoch_end"
    DEFAULT_RL_EVAL_DIR="${DEFAULT_EVAL_ROOT}/rl"
    DEFAULT_ROLLOUT_ROOT="${DEFAULT_EVAL_ROOT}/batch_rollout"
    DEFAULT_LOG_ROOT="${RUN_BASE_DIR}/logs"
    DEFAULT_PIPELINE_LOG_DIR="${DEFAULT_LOG_ROOT}/pipeline"
    DEFAULT_SFT_LOG_DIR="${DEFAULT_LOG_ROOT}/sft"
    DEFAULT_RL_LOG_DIR="${DEFAULT_LOG_ROOT}/rl"
    DEFAULT_ROLLOUT_LOG_DIR="${DEFAULT_LOG_ROOT}/rollout"
    DEFAULT_LOG_DIR="${DEFAULT_PIPELINE_LOG_DIR}"
    echo "[main] experiment output base: ${RUN_BASE_DIR}"
  else
    RUN_BASE_DIR=""
    DEFAULT_ARTIFACT_DIR="${default_annotation_dir}"
    DEFAULT_CHECKPOINT_DIR="${default_exp_root}/checkpoints"
    DEFAULT_SFT_CHECKPOINT_DIR="${DEFAULT_CHECKPOINT_DIR}/sft"
    DEFAULT_RL_CHECKPOINT_DIR="${DEFAULT_CHECKPOINT_DIR}/rl"
    DEFAULT_EVAL_ROOT="${default_exp_root}/eval"
    DEFAULT_SFT_EVAL_DIR="${DEFAULT_EVAL_ROOT}/sft_epoch_end"
    DEFAULT_RL_EVAL_DIR="${DEFAULT_EVAL_ROOT}/rl"
    DEFAULT_ROLLOUT_ROOT="${DEFAULT_EVAL_ROOT}/batch_rollout"
    DEFAULT_LOG_ROOT="${experiment_base_dir}/logs"
    DEFAULT_PIPELINE_LOG_DIR="${DEFAULT_LOG_ROOT}/pipeline"
    DEFAULT_SFT_LOG_DIR="${DEFAULT_LOG_ROOT}/sft"
    DEFAULT_RL_LOG_DIR="${DEFAULT_LOG_ROOT}/rl"
    DEFAULT_ROLLOUT_LOG_DIR="${DEFAULT_LOG_ROOT}/rollout"
    DEFAULT_LOG_DIR="${DEFAULT_PIPELINE_LOG_DIR}"
  fi
}

configure_script_logging() {
  local script_name="${1:-script}"
  LOG_DIR="${LOG_DIR:-${DEFAULT_LOG_DIR}}"
  mkdir -p "${LOG_DIR}"
  if [[ "${SCRIPT_LOGGING_ENABLED:-1}" != "1" ]]; then
    return 0
  fi
  local timestamp
  timestamp="$(date +%Y%m%d_%H%M%S)"
  SCRIPT_LOG_FILE="${SCRIPT_LOG_FILE:-${LOG_DIR}/${script_name}.${timestamp}.log}"
  mkdir -p "$(dirname "${SCRIPT_LOG_FILE}")"
  exec > >(
    python3 -c '
import sys
import re
from datetime import datetime
from pathlib import Path

log_path = Path(sys.argv[1])
log_path.parent.mkdir(parents=True, exist_ok=True)
timestamp_prefix = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]")

with log_path.open("a", encoding="utf-8") as handle:
    for raw_line in sys.stdin:
        line = raw_line.rstrip("\n")
        if timestamp_prefix.match(line):
            rendered = f"{line}\n"
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            rendered = f"[{timestamp}] {line}\n"
        sys.stdout.write(rendered)
        sys.stdout.flush()
        handle.write(rendered)
        handle.flush()
' "${SCRIPT_LOG_FILE}"
  ) 2>&1
  echo "[main] script log: ${SCRIPT_LOG_FILE}"
}
