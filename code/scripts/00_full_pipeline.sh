#!/usr/bin/env bash
set -euo pipefail

# Full SAVER pipeline:
#   1. 如果缺失，则构建 runtime_train / runtime_test / sft_train(.teacher)
#   2. 预建 .frame_cache / .feature_cache
#   3. 多卡训练 SFT，并在每个 epoch 后做 rollout 评估
#   4. 用 proposal runtime 跑批量 rollout、离线打分、汇总
#   5. 多卡训练 RL
#
# 使用方式:
#   bash code/scripts/00_full_pipeline.sh
#
# 推荐覆盖方式:
#   DATA_ROOT=/your/root MODEL_PATH=/your/model bash code/scripts/00_full_pipeline.sh
#
# 说明:
#   - 这个脚本顶部所有变量都支持通过同名环境变量覆盖。
#   - Stage 1 会检查目标文件是否已存在；存在则直接跳过该子阶段。
#   - 如果你想强制重建某个阶段，删除对应输出文件后重新运行即可。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/common_experiment.sh"

# -----------------------------
# 基础路径
# 这些通常是你最先需要改的参数。
# -----------------------------
# 数据根目录。大多数路径都从这里派生。
DATA_ROOT="${DATA_ROOT:-/mnt/shared-storage-user/mineru2-shared/zengweijun}"
# 实验根目录。
EXP_ROOT="${EXP_ROOT:-${DATA_ROOT}/Wmh/ideas/idea2_v2}"
DATA_UTILS_DIR="${DATA_UTILS_DIR:-${CODE_DIR}/data_utils}"
configure_experiment_layout "${CODE_DIR}" "${EXP_ROOT}" "${DATA_UTILS_DIR}"
LOG_DIR="${LOG_DIR:-${DEFAULT_PIPELINE_LOG_DIR}}"
configure_script_logging "00_full_pipeline"
# 预处理类 json/jsonl/json summary 固定放在 code/data_utils；
# checkpoint / rollout / train artifacts 继续走 ckpt/<EXP_NAME>。
ANNOTATION_DIR="${ANNOTATION_DIR:-${DEFAULT_ANNOTATION_DIR}}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${DEFAULT_ARTIFACT_DIR}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${DEFAULT_CHECKPOINT_DIR}}"
ROLLOUT_ROOT="${ROLLOUT_ROOT:-${DEFAULT_ROLLOUT_ROOT}}"
SFT_EVAL_OUTPUT_DIR="${SFT_EVAL_OUTPUT_DIR:-${DEFAULT_SFT_EVAL_DIR}}"
SFT_LOG_DIR="${SFT_LOG_DIR:-${DEFAULT_SFT_LOG_DIR}}"
RL_LOG_DIR="${RL_LOG_DIR:-${DEFAULT_RL_LOG_DIR}}"
ROLLOUT_LOG_DIR="${ROLLOUT_LOG_DIR:-${DEFAULT_ROLLOUT_LOG_DIR}}"
# 模型目录，默认假设 Qwen3-VL 权重放在这里。
MODEL_ROOT="${MODEL_ROOT:-${DATA_ROOT}/Wmh/MLLMs}"

# -----------------------------
# 数据文件
# 如果你已经手工准备过这些文件，可以直接指向已有文件。
# -----------------------------
CANONICAL_JSONL="${CANONICAL_JSONL:-${DATA_UTILS_DIR}/msad_saver_with_qwen.jsonl}"
LEGACY_AGENT_TRAIN_JSONL="${AGENT_TRAIN_JSONL:-}"
LEGACY_ORACLE_JSONL="${ORACLE_JSONL:-}"
LEGACY_PREPARED_TRAIN_JSONL="${PREPARED_TRAIN_JSONL:-}"
LEGACY_TEACHER_PREPARED_TRAIN_JSONL="${TEACHER_PREPARED_TRAIN_JSONL:-}"
RUNTIME_TRAIN_JSONL="${RUNTIME_TRAIN_JSONL:-${LEGACY_AGENT_TRAIN_JSONL:-${DATA_UTILS_DIR}/msad_saver_runtime_train.jsonl}}"
RUNTIME_TEST_JSONL="${RUNTIME_TEST_JSONL:-${DATA_UTILS_DIR}/msad_saver_runtime_test.jsonl}"
SFT_TRAIN_JSONL="${SFT_TRAIN_JSONL:-${LEGACY_PREPARED_TRAIN_JSONL:-${DATA_UTILS_DIR}/msad_saver_sft_train.jsonl}}"
TEACHER_SFT_TRAIN_JSONL="${TEACHER_SFT_TRAIN_JSONL:-${LEGACY_TEACHER_PREPARED_TRAIN_JSONL:-${DATA_UTILS_DIR}/msad_saver_sft_train.teacher.jsonl}}"
SFT_TRAIN_METADATA_JSON="${SFT_TRAIN_METADATA_JSON:-${SFT_TRAIN_JSONL}.meta.json}"
TEACHER_SFT_TRAIN_METADATA_JSON="${TEACHER_SFT_TRAIN_METADATA_JSON:-${TEACHER_SFT_TRAIN_JSONL}.meta.json}"

# -----------------------------
# 模型与输出
# MODEL_PATH 是 SFT 初始模型；SFT_OUTPUT_DIR 会作为后续 rollout 和 RL 的默认输入。
# -----------------------------
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/qwen3-vl-8b-Instruct}"
SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-${DEFAULT_SFT_CHECKPOINT_DIR}/saver_sft_qwen3vl_8b_eval_ddp}"
RL_OUTPUT_DIR="${RL_OUTPUT_DIR:-${DEFAULT_RL_CHECKPOINT_DIR}/saver_cea_grpo_v1}"

# -----------------------------
# 通用数据过滤与容错
# EVAL_SPLIT 默认用 test。runtime builder 会把 canonical JSONL 拆成 runtime_train/runtime_test。
# -----------------------------
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
ADAPTER="${ADAPTER:-msad_saver_qwen}"
# 设为 1 时，遇到坏 JSONL 行会跳过；更推荐修复上游文件。
SKIP_INVALID_JSONL_LINES="${SKIP_INVALID_JSONL_LINES:-0}"

# -----------------------------
# Stage 1: prepared SFT 数据准备
# VALIDATE_MATERIALIZATION=1 会很慢，但能更彻底检查 image_ref 是否能回读。
# -----------------------------
PREPARE_PROGRESS_EVERY="${PREPARE_PROGRESS_EVERY:-25}"
VALIDATE_MATERIALIZATION="${VALIDATE_MATERIALIZATION:-0}"
VALIDATION_MAX_EXAMPLES="${VALIDATION_MAX_EXAMPLES:-0}"

# -----------------------------
# Stage 2: frame / feature cache
# BUILD_FEATURE_CACHE=auto 时，只在提供 PROPOSAL_MODEL_PATH 后自动开启。
# Stage 4 的真实 rollout 也会在同样条件下自动接 proposal runtime。
# -----------------------------
BUILD_FRAME_CACHE="${BUILD_FRAME_CACHE:-1}"
BUILD_FEATURE_CACHE="${BUILD_FEATURE_CACHE:-auto}"
FRAME_CACHE_DATA="${FRAME_CACHE_DATA:-${LEGACY_ORACLE_JSONL:-}}"
FRAME_CACHE_INCLUDE_SPLITS="${FRAME_CACHE_INCLUDE_SPLITS:-${TRAIN_SPLIT},${EVAL_SPLIT}}"
FRAME_CACHE_VIDEO_FPS="${FRAME_CACHE_VIDEO_FPS:-2.0}"
FRAME_CACHE_MAX_FRAMES="${FRAME_CACHE_MAX_FRAMES:-256}"
FRAME_CACHE_PROGRESS_EVERY="${FRAME_CACHE_PROGRESS_EVERY:-50}"
FRAME_CACHE_OVERWRITE="${FRAME_CACHE_OVERWRITE:-0}"
FRAME_CACHE_SUMMARY_OUTPUT="${FRAME_CACHE_SUMMARY_OUTPUT:-${DATA_UTILS_DIR}/frame_cache_summary.json}"
PROPOSAL_MODEL_PATH="${PROPOSAL_MODEL_PATH:-/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/MLLMs/siglip}"
PROPOSAL_TORCH_DTYPE="${PROPOSAL_TORCH_DTYPE:-auto}"
PROPOSAL_DEVICE="${PROPOSAL_DEVICE:-}"
ROLLOUT_USE_PROPOSAL_RUNTIME="${ROLLOUT_USE_PROPOSAL_RUNTIME:-auto}"
FEATURE_CACHE_DATA="${FEATURE_CACHE_DATA:-${FRAME_CACHE_DATA}}"
FEATURE_CACHE_INCLUDE_SPLITS="${FEATURE_CACHE_INCLUDE_SPLITS:-${FRAME_CACHE_INCLUDE_SPLITS}}"
FEATURE_CACHE_DEVICE="${FEATURE_CACHE_DEVICE:-${PROPOSAL_DEVICE:-cpu}}"
FEATURE_CACHE_PROGRESS_EVERY="${FEATURE_CACHE_PROGRESS_EVERY:-25}"
FEATURE_CACHE_OVERWRITE="${FEATURE_CACHE_OVERWRITE:-0}"
FEATURE_CACHE_SUMMARY_OUTPUT="${FEATURE_CACHE_SUMMARY_OUTPUT:-${DATA_UTILS_DIR}/feature_cache_summary.json}"
EVAL_PROPOSAL_MODEL_PATH="${EVAL_PROPOSAL_MODEL_PATH:-${PROPOSAL_MODEL_PATH}}"
EVAL_PROPOSAL_TORCH_DTYPE="${EVAL_PROPOSAL_TORCH_DTYPE:-${PROPOSAL_TORCH_DTYPE}}"
EVAL_PROPOSAL_DEVICE="${EVAL_PROPOSAL_DEVICE:-${PROPOSAL_DEVICE}}"
TEACHER_JUDGE_ENABLE="${TEACHER_JUDGE_ENABLE:-auto}"
TEACHER_JUDGE_MODEL_PATH="${TEACHER_JUDGE_MODEL_PATH:-}"
TEACHER_JUDGE_DEFAULT_MODEL_PATH="${TEACHER_JUDGE_DEFAULT_MODEL_PATH:-${MODEL_ROOT}/Qwen3-VL-32B-Instruct}"
TEACHER_JUDGE_INPUT_MODE="${TEACHER_JUDGE_INPUT_MODE:-auto}"
TEACHER_JUDGE_TORCH_DTYPE="${TEACHER_JUDGE_TORCH_DTYPE:-auto}"
TEACHER_JUDGE_DEVICE_MAP="${TEACHER_JUDGE_DEVICE_MAP:-auto}"
TEACHER_JUDGE_ATTN_IMPLEMENTATION="${TEACHER_JUDGE_ATTN_IMPLEMENTATION:-}"
TEACHER_JUDGE_MAX_NEW_TOKENS="${TEACHER_JUDGE_MAX_NEW_TOKENS:-384}"
TEACHER_JUDGE_MAX_IMAGES="${TEACHER_JUDGE_MAX_IMAGES:-8}"
TEACHER_JUDGE_TOPK_FRAMES_PER_VIEW="${TEACHER_JUDGE_TOPK_FRAMES_PER_VIEW:-4}"
TEACHER_JUDGE_PROGRESS_EVERY="${TEACHER_JUDGE_PROGRESS_EVERY:-25}"
TEACHER_JUDGE_OVERWRITE_EXISTING="${TEACHER_JUDGE_OVERWRITE_EXISTING:-0}"
MAX_TEACHER_DISAGREEMENT_CASES="${MAX_TEACHER_DISAGREEMENT_CASES:-50}"

# -----------------------------
# 推荐统一预算
# 训练、epoch-end eval、外部 rollout 默认共用这组预算；
# 需要单独覆盖时，再改各 stage 的专属变量。
# -----------------------------
PIPELINE_MAX_IMAGE_SIDE="${PIPELINE_MAX_IMAGE_SIDE:-640}"
PIPELINE_MAX_IMAGE_PIXELS="${PIPELINE_MAX_IMAGE_PIXELS:-0}"
PIPELINE_MAX_TOTAL_IMAGES="${PIPELINE_MAX_TOTAL_IMAGES:-28}"
PIPELINE_MAX_SEQ_LENGTH="${PIPELINE_MAX_SEQ_LENGTH:-6144}"
PIPELINE_KEEP_RECENT_TEXT_MESSAGES="${PIPELINE_KEEP_RECENT_TEXT_MESSAGES:-20}"
PIPELINE_NUM_PREVIEW_FRAMES="${PIPELINE_NUM_PREVIEW_FRAMES:-8}"
PIPELINE_PREVIEW_SAMPLING_FPS="${PIPELINE_PREVIEW_SAMPLING_FPS:-}"

# -----------------------------
# Stage 3: SFT
# 最常改:
#   - SFT_NPROC_PER_NODE
#   - SFT_INLINE_ROLLOUT_EVAL
#   - SFT_NUM_TRAIN_EPOCHS
#   - SFT_GRADIENT_ACCUMULATION_STEPS
#   - SFT_MAX_TOTAL_IMAGES / SFT_MAX_IMAGE_SIDE
# -----------------------------
SFT_NPROC_PER_NODE="${SFT_NPROC_PER_NODE:-4}"
SFT_INLINE_ROLLOUT_EVAL="${SFT_INLINE_ROLLOUT_EVAL:-1}"
SFT_LEARNING_RATE="${SFT_LEARNING_RATE:-1e-5}"
SFT_NUM_TRAIN_EPOCHS="${SFT_NUM_TRAIN_EPOCHS:-2.0}"
SFT_PER_DEVICE_TRAIN_BATCH_SIZE="${SFT_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
SFT_GRADIENT_ACCUMULATION_STEPS="${SFT_GRADIENT_ACCUMULATION_STEPS:-8}"
SFT_LOGGING_STEPS="${SFT_LOGGING_STEPS:-5}"
SFT_ATTN_IMPLEMENTATION="${SFT_ATTN_IMPLEMENTATION:-flash_attention_3}"
# 设为 0 表示评完整个 EVAL_SPLIT，不传 --eval-max-records。
SFT_EVAL_MAX_RECORDS="${SFT_EVAL_MAX_RECORDS:-0}"
SFT_EVAL_ROLLOUT_MAX_TURNS="${SFT_EVAL_ROLLOUT_MAX_TURNS:-14}"
SFT_EVAL_MAX_NEW_TOKENS_PER_TURN="${SFT_EVAL_MAX_NEW_TOKENS_PER_TURN:-256}"
SFT_EVAL_MAX_TOTAL_IMAGES="${SFT_EVAL_MAX_TOTAL_IMAGES:-${PIPELINE_MAX_TOTAL_IMAGES}}"
SFT_EVAL_ATTACH_REFERENCE_DIAGNOSTICS="${SFT_EVAL_ATTACH_REFERENCE_DIAGNOSTICS:-0}"
SFT_EVAL_DIAGNOSTIC_VERIFIER_BACKEND="${SFT_EVAL_DIAGNOSTIC_VERIFIER_BACKEND:-${SFT_EVAL_VERIFIER_BACKEND:-heuristic}}"
SFT_EVAL_PROGRESS_EVERY="${SFT_EVAL_PROGRESS_EVERY:-1}"
# epoch-end rollout eval 会自动继承 SFT_MAX_IMAGE_SIDE / SFT_MAX_IMAGE_PIXELS。
#   - SFT_MAX_IMAGE_SIDE: 等比缩放图片最长边
#   - SFT_MAX_IMAGE_PIXELS: 等比缩放到目标像素数上限
#   - SFT_KEEP_RECENT_TOOL_IMAGE_MESSAGES: 仅保留最近 N 条 tool 图片
#   - SFT_MAX_TOTAL_IMAGES: 每个样本保留的总图片数上限
#   - SFT_MAX_SEQ_LENGTH / SFT_KEEP_RECENT_TEXT_MESSAGES: 文本预算控制
SFT_MAX_IMAGE_SIDE="${SFT_MAX_IMAGE_SIDE:-${PIPELINE_MAX_IMAGE_SIDE}}"
SFT_MAX_IMAGE_PIXELS="${SFT_MAX_IMAGE_PIXELS:-${PIPELINE_MAX_IMAGE_PIXELS}}"
SFT_KEEP_RECENT_TOOL_IMAGE_MESSAGES="${SFT_KEEP_RECENT_TOOL_IMAGE_MESSAGES:-0}"
SFT_MAX_TOTAL_IMAGES="${SFT_MAX_TOTAL_IMAGES:-${PIPELINE_MAX_TOTAL_IMAGES}}"
SFT_MAX_SEQ_LENGTH="${SFT_MAX_SEQ_LENGTH:-${PIPELINE_MAX_SEQ_LENGTH}}"
SFT_KEEP_RECENT_TEXT_MESSAGES="${SFT_KEEP_RECENT_TEXT_MESSAGES:-${PIPELINE_KEEP_RECENT_TEXT_MESSAGES}}"
SFT_DATALOADER_NUM_WORKERS="${SFT_DATALOADER_NUM_WORKERS:-4}"
SFT_DATALOADER_PREFETCH_FACTOR="${SFT_DATALOADER_PREFETCH_FACTOR:-2}"
SFT_DATALOADER_PERSISTENT_WORKERS="${SFT_DATALOADER_PERSISTENT_WORKERS:-1}"

# -----------------------------
# Stage 4: 批量 rollout / score / summarize
# 默认从 SFT checkpoint 在 test split 上跑。
# 如果只想快速 smoke test，可以把 ROLLOUT_COUNT 调小。
# -----------------------------
ROLLOUT_RUN_NAME="${ROLLOUT_RUN_NAME:-sft_rollout_eval}"
ROLLOUT_RUN_DIR="${ROLLOUT_RUN_DIR:-${ROLLOUT_ROOT}/${ROLLOUT_RUN_NAME}}"
ROLLOUT_NPROC_PER_NODE="${ROLLOUT_NPROC_PER_NODE:-${SFT_NPROC_PER_NODE:-1}}"
ROLLOUT_START_INDEX="${ROLLOUT_START_INDEX:-0}"
# 设为 0 表示从 ROLLOUT_START_INDEX 开始一路跑到该 split 结束，不截断。
ROLLOUT_COUNT="${ROLLOUT_COUNT:-0}"
ROLLOUT_MAX_TURNS="${ROLLOUT_MAX_TURNS:-14}"
ROLLOUT_PROGRESS_EVERY="${ROLLOUT_PROGRESS_EVERY:-5}"
ROLLOUT_TORCH_DTYPE="${ROLLOUT_TORCH_DTYPE:-auto}"
ROLLOUT_DEVICE_MAP="${ROLLOUT_DEVICE_MAP:-auto}"
ROLLOUT_ATTN_IMPLEMENTATION="${ROLLOUT_ATTN_IMPLEMENTATION:-flash_attention_3}"
ROLLOUT_MAX_NEW_TOKENS="${ROLLOUT_MAX_NEW_TOKENS:-256}"
ROLLOUT_MAX_IMAGE_SIDE="${ROLLOUT_MAX_IMAGE_SIDE:-${PIPELINE_MAX_IMAGE_SIDE}}"
ROLLOUT_MAX_IMAGE_PIXELS="${ROLLOUT_MAX_IMAGE_PIXELS:-${PIPELINE_MAX_IMAGE_PIXELS}}"
ROLLOUT_MAX_TOTAL_IMAGES="${ROLLOUT_MAX_TOTAL_IMAGES:-${PIPELINE_MAX_TOTAL_IMAGES}}"
# 采样开关。评估更稳时通常设 0；想看更多策略多样性时设 1。
ROLLOUT_DO_SAMPLE="${ROLLOUT_DO_SAMPLE:-0}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.7}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.9}"
ROLLOUT_TOP_K="${ROLLOUT_TOP_K:-}"
ROLLOUT_REPETITION_PENALTY="${ROLLOUT_REPETITION_PENALTY:-}"
ROLLOUT_DIAGNOSTIC_VERIFIER_BACKEND="${ROLLOUT_DIAGNOSTIC_VERIFIER_BACKEND:-${ROLLOUT_VERIFIER_BACKEND:-heuristic}}"
ROLLOUT_DIAGNOSTIC_VERIFIER_MODEL_PATH="${ROLLOUT_DIAGNOSTIC_VERIFIER_MODEL_PATH:-${ROLLOUT_VERIFIER_MODEL_PATH:-${MODEL_ROOT}/qwen3-vl-8b-Instruct}}"
ROLLOUT_DIAGNOSTIC_VERIFIER_TORCH_DTYPE="${ROLLOUT_DIAGNOSTIC_VERIFIER_TORCH_DTYPE:-${ROLLOUT_VERIFIER_TORCH_DTYPE:-auto}}"
ROLLOUT_DIAGNOSTIC_VERIFIER_DEVICE_MAP="${ROLLOUT_DIAGNOSTIC_VERIFIER_DEVICE_MAP:-${ROLLOUT_VERIFIER_DEVICE_MAP:-auto}}"
ROLLOUT_DIAGNOSTIC_VERIFIER_ATTN_IMPLEMENTATION="${ROLLOUT_DIAGNOSTIC_VERIFIER_ATTN_IMPLEMENTATION:-${ROLLOUT_VERIFIER_ATTN_IMPLEMENTATION:-}}"
ROLLOUT_DIAGNOSTIC_VERIFIER_MAX_NEW_TOKENS="${ROLLOUT_DIAGNOSTIC_VERIFIER_MAX_NEW_TOKENS:-${ROLLOUT_VERIFIER_MAX_NEW_TOKENS:-512}}"
ROLLOUT_DIAGNOSTIC_VERIFIER_HYBRID_ALPHA="${ROLLOUT_DIAGNOSTIC_VERIFIER_HYBRID_ALPHA:-${ROLLOUT_VERIFIER_HYBRID_ALPHA:-0.7}}"
ROLLOUT_DIAGNOSTIC_ONLINE_VERIFIER_FALLBACK="${ROLLOUT_DIAGNOSTIC_ONLINE_VERIFIER_FALLBACK:-0}"
ROLLOUT_DIAGNOSTIC_ATTACH_REFERENCE_OFFLINE_VERIFIER="${ROLLOUT_DIAGNOSTIC_ATTACH_REFERENCE_OFFLINE_VERIFIER:-0}"
ROLLOUT_DIAGNOSTIC_FORCE_REVERIFY="${ROLLOUT_DIAGNOSTIC_FORCE_REVERIFY:-0}"
RAW_ROLLOUT_OUTPUT="${RAW_ROLLOUT_OUTPUT:-${ROLLOUT_RUN_DIR}/rollouts.raw.jsonl}"
SCORED_ROLLOUT_OUTPUT="${SCORED_ROLLOUT_OUTPUT:-${ROLLOUT_RUN_DIR}/rollouts.scored.jsonl}"
ROLLOUT_SUMMARY_OUTPUT="${ROLLOUT_SUMMARY_OUTPUT:-${ROLLOUT_RUN_DIR}/summary.json}"

# -----------------------------
# Stage 5: RL
# 默认用 SFT 输出作为当前 policy 和 reference policy 的起点。
# 如果你想锁定一个不同的 reference checkpoint，单独改 RL_REFERENCE_MODEL_PATH。
# RL update 阶段和 SFT 一样，建议显式打开视觉 + 文本双预算，
# 否则长轨迹里的 tool JSON / 历史消息仍可能拖慢训练甚至触发 OOM。
# -----------------------------
RL_NPROC_PER_NODE="${RL_NPROC_PER_NODE:-4}"
RL_REFERENCE_MODEL_PATH="${RL_REFERENCE_MODEL_PATH:-${SFT_OUTPUT_DIR}}"
# 设为 0 表示评完整个 EVAL_SPLIT，不传 --eval-max-records。
RL_EVAL_MAX_RECORDS="${RL_EVAL_MAX_RECORDS:-0}"
RL_EVAL_ROLLOUT_MAX_TURNS="${RL_EVAL_ROLLOUT_MAX_TURNS:-14}"
RL_EVAL_MAX_NEW_TOKENS_PER_TURN="${RL_EVAL_MAX_NEW_TOKENS_PER_TURN:-256}"
RL_EVAL_MAX_TOTAL_IMAGES="${RL_EVAL_MAX_TOTAL_IMAGES:-${PIPELINE_MAX_TOTAL_IMAGES}}"
RL_EVAL_ATTACH_REFERENCE_DIAGNOSTICS="${RL_EVAL_ATTACH_REFERENCE_DIAGNOSTICS:-0}"
RL_EVAL_DIAGNOSTIC_VERIFIER_BACKEND="${RL_EVAL_DIAGNOSTIC_VERIFIER_BACKEND:-${RL_EVAL_VERIFIER_BACKEND:-heuristic}}"
RL_NUM_ITERATIONS="${RL_NUM_ITERATIONS:-3}"
RL_ROLLOUT_COUNT="${RL_ROLLOUT_COUNT:-16}"
RL_NUM_GENERATIONS="${RL_NUM_GENERATIONS:-4}"
RL_ROLLOUT_MAX_TURNS="${RL_ROLLOUT_MAX_TURNS:-14}"
RL_POLICY_MAX_NEW_TOKENS="${RL_POLICY_MAX_NEW_TOKENS:-256}"
RL_POLICY_DO_SAMPLE="${RL_POLICY_DO_SAMPLE:-1}"
RL_POLICY_TEMPERATURE="${RL_POLICY_TEMPERATURE:-0.7}"
RL_POLICY_TOP_P="${RL_POLICY_TOP_P:-0.9}"
RL_MIN_WEIGHT="${RL_MIN_WEIGHT:-0.1}"
RL_ADVANTAGE_CLIP="${RL_ADVANTAGE_CLIP:-3.0}"
RL_GRPO_VARIANT="${RL_GRPO_VARIANT:-cea_grpo}"
RL_CEA_ENABLE_SEARCH_GROUP="${RL_CEA_ENABLE_SEARCH_GROUP:-1}"
RL_CEA_ENABLE_ALERT_GROUP="${RL_CEA_ENABLE_ALERT_GROUP:-1}"
RL_CEA_ENABLE_EVIDENCE_GROUP="${RL_CEA_ENABLE_EVIDENCE_GROUP:-1}"
RL_CEA_SEARCH_LOCAL_ALPHA="${RL_CEA_SEARCH_LOCAL_ALPHA:-0.5}"
RL_CEA_ALERT_LOCAL_ALPHA="${RL_CEA_ALERT_LOCAL_ALPHA:-0.5}"
RL_CEA_EVIDENCE_LOCAL_ALPHA="${RL_CEA_EVIDENCE_LOCAL_ALPHA:-0.5}"
RL_TEACHER_JUDGE_LOCAL_ALPHA="${RL_TEACHER_JUDGE_LOCAL_ALPHA:-0.5}"
RL_CEA_LOCAL_VERIFIER_BACKEND="${RL_CEA_LOCAL_VERIFIER_BACKEND:-self_teacher}"
RL_CEA_LOCAL_USE_REFERENCE_SUPERVISION="${RL_CEA_LOCAL_USE_REFERENCE_SUPERVISION:-0}"
RL_CEA_MAX_SEARCH_ANCHORS_PER_ROLLOUT="${RL_CEA_MAX_SEARCH_ANCHORS_PER_ROLLOUT:-2}"
RL_CEA_MAX_ALERT_ANCHORS_PER_ROLLOUT="${RL_CEA_MAX_ALERT_ANCHORS_PER_ROLLOUT:-2}"
RL_CEA_MAX_EVIDENCE_ANCHORS_PER_ROLLOUT="${RL_CEA_MAX_EVIDENCE_ANCHORS_PER_ROLLOUT:-2}"
RL_KL_BETA="${RL_KL_BETA:-0.02}"
RL_DIAGNOSTIC_VERIFIER_BACKEND="${RL_DIAGNOSTIC_VERIFIER_BACKEND:-${RL_VERIFIER_BACKEND:-heuristic}}"
RL_DIAGNOSTIC_VERIFIER_MODEL_PATH="${RL_DIAGNOSTIC_VERIFIER_MODEL_PATH:-${RL_VERIFIER_MODEL_PATH:-${MODEL_ROOT}/qwen3-vl-8b-Instruct}}"
RL_DIAGNOSTIC_VERIFIER_TORCH_DTYPE="${RL_DIAGNOSTIC_VERIFIER_TORCH_DTYPE:-${RL_VERIFIER_TORCH_DTYPE:-auto}}"
RL_DIAGNOSTIC_VERIFIER_DEVICE_MAP="${RL_DIAGNOSTIC_VERIFIER_DEVICE_MAP:-${RL_VERIFIER_DEVICE_MAP:-auto}}"
RL_DIAGNOSTIC_VERIFIER_ATTN_IMPLEMENTATION="${RL_DIAGNOSTIC_VERIFIER_ATTN_IMPLEMENTATION:-${RL_VERIFIER_ATTN_IMPLEMENTATION:-}}"
RL_DIAGNOSTIC_VERIFIER_MAX_NEW_TOKENS="${RL_DIAGNOSTIC_VERIFIER_MAX_NEW_TOKENS:-${RL_VERIFIER_MAX_NEW_TOKENS:-512}}"
RL_DIAGNOSTIC_VERIFIER_HYBRID_ALPHA="${RL_DIAGNOSTIC_VERIFIER_HYBRID_ALPHA:-${RL_VERIFIER_HYBRID_ALPHA:-0.7}}"
RL_DIAGNOSTIC_ONLINE_VERIFIER_FALLBACK="${RL_DIAGNOSTIC_ONLINE_VERIFIER_FALLBACK:-0}"
RL_DIAGNOSTIC_ATTACH_REFERENCE_OFFLINE_VERIFIER="${RL_DIAGNOSTIC_ATTACH_REFERENCE_OFFLINE_VERIFIER:-0}"
RL_DIAGNOSTIC_FORCE_REVERIFY="${RL_DIAGNOSTIC_FORCE_REVERIFY:-0}"
RL_LEARNING_RATE="${RL_LEARNING_RATE:-5e-6}"
RL_NUM_TRAIN_EPOCHS="${RL_NUM_TRAIN_EPOCHS:-1.0}"
RL_PER_DEVICE_TRAIN_BATCH_SIZE="${RL_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
RL_GRADIENT_ACCUMULATION_STEPS="${RL_GRADIENT_ACCUMULATION_STEPS:-16}"
RL_ATTN_IMPLEMENTATION="${RL_ATTN_IMPLEMENTATION:-flash_attention_3}"
RL_LOGGING_STEPS="${RL_LOGGING_STEPS:-10}"
RL_MAX_IMAGE_SIDE="${RL_MAX_IMAGE_SIDE:-${PIPELINE_MAX_IMAGE_SIDE}}"
RL_MAX_IMAGE_PIXELS="${RL_MAX_IMAGE_PIXELS:-${PIPELINE_MAX_IMAGE_PIXELS}}"
RL_KEEP_RECENT_TOOL_IMAGE_MESSAGES="${RL_KEEP_RECENT_TOOL_IMAGE_MESSAGES:-0}"
RL_MAX_TOTAL_IMAGES="${RL_MAX_TOTAL_IMAGES:-${PIPELINE_MAX_TOTAL_IMAGES}}"
RL_MAX_SEQ_LENGTH="${RL_MAX_SEQ_LENGTH:-${PIPELINE_MAX_SEQ_LENGTH}}"
RL_KEEP_RECENT_TEXT_MESSAGES="${RL_KEEP_RECENT_TEXT_MESSAGES:-${PIPELINE_KEEP_RECENT_TEXT_MESSAGES}}"
RL_DATALOADER_NUM_WORKERS="${RL_DATALOADER_NUM_WORKERS:-8}"
RL_DATALOADER_PREFETCH_FACTOR="${RL_DATALOADER_PREFETCH_FACTOR:-4}"
RL_DATALOADER_PERSISTENT_WORKERS="${RL_DATALOADER_PERSISTENT_WORKERS:-1}"
RL_TENSOR_CACHE_DIR="${RL_TENSOR_CACHE_DIR:-}"
RL_TENSOR_CACHE_PROGRESS_EVERY="${RL_TENSOR_CACHE_PROGRESS_EVERY:-25}"
RL_TENSOR_CACHE_OVERWRITE_EXISTING="${RL_TENSOR_CACHE_OVERWRITE_EXISTING:-0}"

is_complete_model_checkpoint_dir() {
  local checkpoint_dir="$1"
  local has_weights=1
  local has_config=1
  local has_processor=1
  local found=1

  [[ -d "${checkpoint_dir}" ]] || return 1

  found=1
  for filename in \
    adapter_model.safetensors \
    adapter_model.bin \
    pytorch_model.bin \
    pytorch_model.bin.index.json \
    model.safetensors \
    model.safetensors.index.json; do
    if [[ -f "${checkpoint_dir}/${filename}" ]]; then
      found=0
      break
    fi
  done
  has_weights=$found

  found=1
  for filename in adapter_config.json config.json; do
    if [[ -f "${checkpoint_dir}/${filename}" ]]; then
      found=0
      break
    fi
  done
  has_config=$found

  found=1
  for filename in \
    preprocessor_config.json \
    processor_config.json \
    tokenizer_config.json \
    tokenizer.json; do
    if [[ -f "${checkpoint_dir}/${filename}" ]]; then
      found=0
      break
    fi
  done
  has_processor=$found

  [[ "${has_weights}" == "0" && "${has_config}" == "0" && "${has_processor}" == "0" ]]
}

resolve_complete_model_checkpoint_dir() {
  local output_dir="$1"
  local candidate=""
  local best_dir=""
  local best_step=-1
  local base_name=""
  local step=""

  if is_complete_model_checkpoint_dir "${output_dir}"; then
    printf '%s\n' "${output_dir}"
    return 0
  fi

  for candidate in "${output_dir}"/checkpoint-*; do
    [[ -d "${candidate}" ]] || continue
    base_name="$(basename "${candidate}")"
    if [[ ! "${base_name}" =~ ^checkpoint-([0-9]+)$ ]]; then
      continue
    fi
    step="${BASH_REMATCH[1]}"
    if ! is_complete_model_checkpoint_dir "${candidate}"; then
      continue
    fi
    if (( step > best_step )); then
      best_step=$step
      best_dir="${candidate}"
    fi
  done

  if [[ -n "${best_dir}" ]]; then
    printf '%s\n' "${best_dir}"
    return 0
  fi
  return 1
}

resolve_latest_numeric_checkpoint_dir() {
  local output_dir="$1"
  local candidate=""
  local best_dir=""
  local best_step=-1
  local base_name=""
  local step=""

  for candidate in "${output_dir}"/checkpoint-*; do
    [[ -d "${candidate}" ]] || continue
    base_name="$(basename "${candidate}")"
    if [[ ! "${base_name}" =~ ^checkpoint-([0-9]+)$ ]]; then
      continue
    fi
    step="${BASH_REMATCH[1]}"
    if ! is_complete_model_checkpoint_dir "${candidate}"; then
      continue
    fi
    if (( step > best_step )); then
      best_step=$step
      best_dir="${candidate}"
    fi
  done

  if [[ -n "${best_dir}" ]]; then
    printf '%s\n' "${best_dir}"
    return 0
  fi
  return 1
}

resolve_latest_sft_epoch_resume_dir() {
  local output_dir="$1"
  local candidate=""
  local best_dir=""
  local best_epoch=-1
  local base_name=""
  local epoch=""

  for candidate in "${output_dir}"/epoch_resume/epoch_*; do
    [[ -d "${candidate}" ]] || continue
    base_name="$(basename "${candidate}")"
    if [[ ! "${base_name}" =~ ^epoch_([0-9]+)$ ]]; then
      continue
    fi
    epoch="${BASH_REMATCH[1]}"
    if ! is_complete_model_checkpoint_dir "${candidate}"; then
      continue
    fi
    if (( epoch > best_epoch )); then
      best_epoch=$epoch
      best_dir="${candidate}"
    fi
  done

  if [[ -n "${best_dir}" ]]; then
    printf '%s\n' "${best_dir}"
    return 0
  fi
  return 1
}

epoch_resume_epoch_index() {
  local checkpoint_dir="$1"
  local base_name=""

  base_name="$(basename "${checkpoint_dir}")"
  if [[ "${base_name}" =~ ^epoch_([0-9]+)$ ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
    return 0
  fi
  return 1
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

list_pending_sft_rollout_eval_checkpoints() {
  local output_dir="$1"
  local rollout_eval_output_dir="$2"
  local candidate=""
  local epoch_index=""
  local base_name=""

  for candidate in "${output_dir}"/epoch_resume/epoch_*; do
    [[ -d "${candidate}" ]] || continue
    base_name="$(basename "${candidate}")"
    if [[ ! "${base_name}" =~ ^epoch_([0-9]+)$ ]]; then
      continue
    fi
    if ! is_complete_model_checkpoint_dir "${candidate}"; then
      continue
    fi
    epoch_index="${BASH_REMATCH[1]}"
    if ! rollout_eval_metrics_exist_for_epoch "${rollout_eval_output_dir}" "${epoch_index}"; then
      printf '%s\n' "${candidate}"
    fi
  done
}

run_pending_sft_rollout_evals() {
  local output_dir="$1"
  local rollout_eval_output_dir="$2"
  local pending_checkpoints=()
  local checkpoint_dir=""

  while IFS= read -r checkpoint_dir; do
    [[ -n "${checkpoint_dir}" ]] || continue
    pending_checkpoints+=("${checkpoint_dir}")
  done < <(list_pending_sft_rollout_eval_checkpoints "${output_dir}" "${rollout_eval_output_dir}")

  if [[ "${#pending_checkpoints[@]}" == "0" ]]; then
    echo "  - No pending external SFT rollout eval checkpoints."
    return 0
  fi

  for checkpoint_dir in "${pending_checkpoints[@]}"; do
    echo "  - Run external SFT rollout eval from ${checkpoint_dir}"
    build_stage3_sft_cmd "${checkpoint_dir}" 1
    "${SFT_STAGE3_CMD[@]}"
  done
}

checkpoint_has_remaining_training() {
  local checkpoint_dir="$1"
  python3 - "${checkpoint_dir}" <<'PY'
import json
import sys
from pathlib import Path

checkpoint_dir = Path(sys.argv[1])
trainer_state_path = checkpoint_dir / "trainer_state.json"
if not trainer_state_path.exists():
    print(0)
    raise SystemExit(0)

try:
    state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
except Exception:
    print(0)
    raise SystemExit(0)

try:
    epoch = float(state.get("epoch") or 0.0)
    num_train_epochs = float(state.get("num_train_epochs") or 0.0)
except Exception:
    print(0)
    raise SystemExit(0)

print(1 if epoch + 1e-6 < num_train_epochs else 0)
PY
}

build_stage3_sft_cmd() {
  local resume_path="${1:-}"
  local resume_rollout_eval_only="${2:-0}"

  local launcher=(python)
  if [[ "${SFT_NPROC_PER_NODE}" -gt 1 ]]; then
    launcher=(torchrun --nproc_per_node="${SFT_NPROC_PER_NODE}")
  fi

  SFT_STAGE3_CMD=(
    "${launcher[@]}"
    train_saver_sft.py
    --prepared-data "${EFFECTIVE_PREPARED_TRAIN_JSONL}"
    --include-splits "${TRAIN_SPLIT}"
    --model-path "${MODEL_PATH}"
    --output-dir "${SFT_OUTPUT_DIR}"
    --log-dir "${SFT_LOG_DIR}"
    --rollout-eval-output-dir "${SFT_EVAL_OUTPUT_DIR}"
    --eval-data "${RUNTIME_TEST_JSONL}"
    --eval-data-root "${DATA_ROOT}"
    --eval-include-splits "${EVAL_SPLIT}"
    --eval-rollout-max-turns "${SFT_EVAL_ROLLOUT_MAX_TURNS}"
    --eval-max-new-tokens-per-turn "${SFT_EVAL_MAX_NEW_TOKENS_PER_TURN}"
    --eval-progress-every "${SFT_EVAL_PROGRESS_EVERY}"
    --lora
    --bf16
    --gradient-checkpointing
    --per-device-train-batch-size "${SFT_PER_DEVICE_TRAIN_BATCH_SIZE}"
    --gradient-accumulation-steps "${SFT_GRADIENT_ACCUMULATION_STEPS}"
    --learning-rate "${SFT_LEARNING_RATE}"
    --num-train-epochs "${SFT_NUM_TRAIN_EPOCHS}"
    --dataloader-num-workers "${SFT_DATALOADER_NUM_WORKERS}"
    --dataloader-prefetch-factor "${SFT_DATALOADER_PREFETCH_FACTOR}"
    --logging-steps "${SFT_LOGGING_STEPS}"
    --num-preview-frames "${PIPELINE_NUM_PREVIEW_FRAMES}"
  )
  if [[ -n "${PIPELINE_PREVIEW_SAMPLING_FPS}" ]]; then
    SFT_STAGE3_CMD+=(--preview-sampling-fps "${PIPELINE_PREVIEW_SAMPLING_FPS}")
  fi
  if [[ -n "${resume_path}" ]]; then
    SFT_STAGE3_CMD+=(--resume-from-checkpoint "${resume_path}")
  fi
  if [[ "${resume_rollout_eval_only}" == "1" ]]; then
    SFT_STAGE3_CMD+=(--resume-rollout-eval-only)
  elif [[ "${SFT_INLINE_ROLLOUT_EVAL}" == "1" ]]; then
    SFT_STAGE3_CMD+=(--inline-rollout-eval)
  else
    SFT_STAGE3_CMD+=(--defer-rollout-eval)
  fi
  if [[ "${SFT_DATALOADER_PERSISTENT_WORKERS}" == "1" ]]; then
    SFT_STAGE3_CMD+=(--dataloader-persistent-workers)
  fi
  if [[ -n "${SFT_ATTN_IMPLEMENTATION}" ]]; then
    SFT_STAGE3_CMD+=(--attn-implementation "${SFT_ATTN_IMPLEMENTATION}")
  fi
  if [[ "${SFT_MAX_IMAGE_SIDE}" != "0" ]]; then
    SFT_STAGE3_CMD+=(--max-image-side "${SFT_MAX_IMAGE_SIDE}")
  fi
  if [[ "${SFT_MAX_IMAGE_PIXELS}" != "0" ]]; then
    SFT_STAGE3_CMD+=(--max-image-pixels "${SFT_MAX_IMAGE_PIXELS}")
  fi
  if [[ "${SFT_KEEP_RECENT_TOOL_IMAGE_MESSAGES}" != "0" ]]; then
    SFT_STAGE3_CMD+=(--keep-recent-tool-image-messages "${SFT_KEEP_RECENT_TOOL_IMAGE_MESSAGES}")
  fi
  if [[ "${SFT_MAX_TOTAL_IMAGES}" != "0" ]]; then
    SFT_STAGE3_CMD+=(--max-total-images "${SFT_MAX_TOTAL_IMAGES}")
  fi
  if [[ "${SFT_MAX_SEQ_LENGTH}" != "0" ]]; then
    SFT_STAGE3_CMD+=(--max-seq-length "${SFT_MAX_SEQ_LENGTH}")
  fi
  if [[ "${SFT_KEEP_RECENT_TEXT_MESSAGES}" != "0" ]]; then
    SFT_STAGE3_CMD+=(--keep-recent-text-messages "${SFT_KEEP_RECENT_TEXT_MESSAGES}")
  fi
  if [[ "${SFT_EVAL_MAX_RECORDS}" != "0" ]]; then
    SFT_STAGE3_CMD+=(--eval-max-records "${SFT_EVAL_MAX_RECORDS}")
  fi
  if [[ "${SFT_EVAL_MAX_TOTAL_IMAGES}" != "0" ]]; then
    SFT_STAGE3_CMD+=(--eval-max-total-images "${SFT_EVAL_MAX_TOTAL_IMAGES}")
  fi
  if [[ "${SFT_EVAL_ATTACH_REFERENCE_DIAGNOSTICS}" == "1" ]]; then
    SFT_STAGE3_CMD+=(
      --eval-attach-reference-diagnostics
      --eval-verifier-backend "${SFT_EVAL_DIAGNOSTIC_VERIFIER_BACKEND}"
      --eval-verifier-model-path "${ROLLOUT_DIAGNOSTIC_VERIFIER_MODEL_PATH}"
      --eval-verifier-torch-dtype "${ROLLOUT_DIAGNOSTIC_VERIFIER_TORCH_DTYPE}"
      --eval-verifier-device-map "${ROLLOUT_DIAGNOSTIC_VERIFIER_DEVICE_MAP}"
      --eval-verifier-max-new-tokens "${ROLLOUT_DIAGNOSTIC_VERIFIER_MAX_NEW_TOKENS}"
      --eval-verifier-hybrid-alpha "${ROLLOUT_DIAGNOSTIC_VERIFIER_HYBRID_ALPHA}"
    )
    if [[ -n "${ROLLOUT_DIAGNOSTIC_VERIFIER_ATTN_IMPLEMENTATION}" ]]; then
      SFT_STAGE3_CMD+=(--eval-verifier-attn-implementation "${ROLLOUT_DIAGNOSTIC_VERIFIER_ATTN_IMPLEMENTATION}")
    fi
  fi
  if [[ -n "${EVAL_PROPOSAL_MODEL_PATH}" ]]; then
    SFT_STAGE3_CMD+=(
      --eval-proposal-model-path "${EVAL_PROPOSAL_MODEL_PATH}"
      --eval-proposal-torch-dtype "${EVAL_PROPOSAL_TORCH_DTYPE}"
    )
    if [[ -n "${EVAL_PROPOSAL_DEVICE}" ]]; then
      SFT_STAGE3_CMD+=(--eval-proposal-device "${EVAL_PROPOSAL_DEVICE}")
    fi
  fi
}

resolve_toggle() {
  local value="${1:-0}"
  local auto_nonempty="${2:-}"

  case "${value}" in
    1|true|TRUE|yes|YES|on|ON)
      return 0
      ;;
    0|false|FALSE|no|NO|off|OFF)
      return 1
      ;;
    auto|AUTO)
      [[ -n "${auto_nonempty}" ]]
      return
      ;;
    *)
      echo "Unsupported toggle value: ${value}" >&2
      exit 1
      ;;
  esac
}

summary_output_with_suffix() {
  local base_path="$1"
  local suffix="$2"
  if [[ "${base_path}" == *.json ]]; then
    printf '%s.%s.json\n' "${base_path%.json}" "${suffix}"
    return 0
  fi
  printf '%s.%s\n' "${base_path}" "${suffix}"
}

resolve_effective_teacher_judge_model_path() {
  local enable_value="${1:-auto}"
  local explicit_path="${2:-}"
  local default_path="${3:-}"

  case "${enable_value}" in
    1|true|TRUE|yes|YES|on|ON)
      if [[ -n "${explicit_path}" ]]; then
        printf '%s\n' "${explicit_path}"
        return 0
      fi
      if [[ -e "${default_path}" ]]; then
        printf '%s\n' "${default_path}"
        return 0
      fi
      echo "TEACHER_JUDGE_ENABLE is on, but no teacher model path is available." >&2
      exit 1
      ;;
    0|false|FALSE|no|NO|off|OFF)
      printf '%s\n' ""
      return 0
      ;;
    auto|AUTO)
      if [[ -n "${explicit_path}" ]]; then
        printf '%s\n' "${explicit_path}"
      elif [[ -e "${default_path}" ]]; then
        printf '%s\n' "${default_path}"
      else
        printf '%s\n' ""
      fi
      return 0
      ;;
    *)
      echo "Unsupported teacher toggle value: ${enable_value}" >&2
      exit 1
      ;;
  esac
}

cd "${CODE_DIR}"

EFFECTIVE_TEACHER_JUDGE_MODEL_PATH="$(
  resolve_effective_teacher_judge_model_path \
    "${TEACHER_JUDGE_ENABLE}" \
    "${TEACHER_JUDGE_MODEL_PATH}" \
    "${TEACHER_JUDGE_DEFAULT_MODEL_PATH}"
)"

if [[ -n "${EFFECTIVE_TEACHER_JUDGE_MODEL_PATH}" ]]; then
  echo "Teacher judge enabled: ${EFFECTIVE_TEACHER_JUDGE_MODEL_PATH}"
  echo "  - SFT verify supervision: on"
  echo "  - RL auxiliary teacher signal: on (teacher_judge_local_alpha=${RL_TEACHER_JUDGE_LOCAL_ALPHA})"
else
  echo "Teacher judge disabled."
fi

echo "[Stage 1/5] Build data artifacts if missing"
stage1_targets=(
  "${RUNTIME_TRAIN_JSONL}"
  "${RUNTIME_TEST_JSONL}"
  "${SFT_TRAIN_JSONL}"
  "${SFT_TRAIN_METADATA_JSON}"
)
if [[ -n "${EFFECTIVE_TEACHER_JUDGE_MODEL_PATH}" ]]; then
  stage1_targets+=("${TEACHER_SFT_TRAIN_JSONL}" "${TEACHER_SFT_TRAIN_METADATA_JSON}")
fi

stage1_missing_targets=()
for target_path in "${stage1_targets[@]}"; do
  if [[ ! -f "${target_path}" ]]; then
    stage1_missing_targets+=("${target_path}")
  fi
done

if [[ "${#stage1_missing_targets[@]}" == "0" ]]; then
  echo "  - Skip simplified data build; all runtime/SFT artifacts already exist."
else
  echo "  - Building simplified runtime/SFT artifacts"
  build_data_cmd=(
    python build_saver_data.py
    --input "${CANONICAL_JSONL}"
    --runtime-train-output "${RUNTIME_TRAIN_JSONL}"
    --runtime-test-output "${RUNTIME_TEST_JSONL}"
    --sft-train-output "${SFT_TRAIN_JSONL}"
    --data-root "${DATA_ROOT}"
    --adapter "${ADAPTER}"
    --train-splits "${TRAIN_SPLIT}"
    --test-splits "${EVAL_SPLIT}"
    --validate-sft-data
    --progress-every "${PREPARE_PROGRESS_EVERY}"
    --num-preview-frames "${PIPELINE_NUM_PREVIEW_FRAMES}"
  )
  if [[ -n "${PIPELINE_PREVIEW_SAMPLING_FPS}" ]]; then
    build_data_cmd+=(--preview-sampling-fps "${PIPELINE_PREVIEW_SAMPLING_FPS}")
  fi
  if [[ "${SKIP_INVALID_JSONL_LINES}" == "1" ]]; then
    build_data_cmd+=(--skip-invalid-jsonl-lines)
  fi
  if [[ "${VALIDATE_MATERIALIZATION}" == "1" ]]; then
    build_data_cmd+=(--validate-materialization)
  fi
  if [[ "${VALIDATION_MAX_EXAMPLES}" != "0" ]]; then
    build_data_cmd+=(--validation-max-examples "${VALIDATION_MAX_EXAMPLES}")
  fi
  if [[ -n "${PROPOSAL_MODEL_PATH}" ]]; then
    build_data_cmd+=(
      --proposal-model-path "${PROPOSAL_MODEL_PATH}"
      --proposal-torch-dtype "${PROPOSAL_TORCH_DTYPE}"
    )
    if [[ -n "${PROPOSAL_DEVICE}" ]]; then
      build_data_cmd+=(--proposal-device "${PROPOSAL_DEVICE}")
    fi
  fi
  if [[ -n "${EFFECTIVE_TEACHER_JUDGE_MODEL_PATH}" ]]; then
    build_data_cmd+=(
      --teacher-output "${TEACHER_SFT_TRAIN_JSONL}"
      --teacher-judge-model-path "${EFFECTIVE_TEACHER_JUDGE_MODEL_PATH}"
      --teacher-judge-input-mode "${TEACHER_JUDGE_INPUT_MODE}"
      --teacher-judge-torch-dtype "${TEACHER_JUDGE_TORCH_DTYPE}"
      --teacher-judge-device-map "${TEACHER_JUDGE_DEVICE_MAP}"
      --teacher-judge-max-new-tokens "${TEACHER_JUDGE_MAX_NEW_TOKENS}"
      --teacher-judge-max-images "${TEACHER_JUDGE_MAX_IMAGES}"
      --teacher-judge-topk-frames-per-view "${TEACHER_JUDGE_TOPK_FRAMES_PER_VIEW}"
    )
    if [[ -n "${TEACHER_JUDGE_ATTN_IMPLEMENTATION}" ]]; then
      build_data_cmd+=(--teacher-judge-attn-implementation "${TEACHER_JUDGE_ATTN_IMPLEMENTATION}")
    fi
    if [[ "${TEACHER_JUDGE_OVERWRITE_EXISTING}" == "1" ]]; then
      build_data_cmd+=(--teacher-judge-overwrite-existing)
    fi
  fi
  "${build_data_cmd[@]}"
fi

EFFECTIVE_PREPARED_TRAIN_JSONL="${SFT_TRAIN_JSONL}"
if [[ -n "${EFFECTIVE_TEACHER_JUDGE_MODEL_PATH}" ]]; then
  EFFECTIVE_PREPARED_TRAIN_JSONL="${TEACHER_SFT_TRAIN_JSONL}"
fi

echo "[Stage 2/5] Build frame cache / feature cache"
if resolve_toggle "${BUILD_FRAME_CACHE}" "1"; then
  frame_cache_data_paths=()
  frame_cache_splits=()
  if [[ -n "${FRAME_CACHE_DATA}" ]]; then
    frame_cache_data_paths+=("${FRAME_CACHE_DATA}")
    frame_cache_splits+=("${FRAME_CACHE_INCLUDE_SPLITS}")
  else
    frame_cache_data_paths+=("${RUNTIME_TRAIN_JSONL}" "${RUNTIME_TEST_JSONL}")
    frame_cache_splits+=("${TRAIN_SPLIT}" "${EVAL_SPLIT}")
  fi
  for cache_index in "${!frame_cache_data_paths[@]}"; do
    frame_summary_output="${FRAME_CACHE_SUMMARY_OUTPUT}"
    if [[ "${#frame_cache_data_paths[@]}" -gt 1 ]]; then
      frame_summary_label="${frame_cache_splits[$cache_index]}"
      frame_summary_label="${frame_summary_label//,/__}"
      frame_summary_label="${frame_summary_label// /}"
      [[ -n "${frame_summary_label}" ]] || frame_summary_label="part${cache_index}"
      frame_summary_output="$(summary_output_with_suffix "${FRAME_CACHE_SUMMARY_OUTPUT}" "${frame_summary_label}")"
    fi
    frame_cmd=(
      python build_frame_cache.py
      --data "${frame_cache_data_paths[$cache_index]}"
      --data-root "${DATA_ROOT}"
      --cache-video-fps "${FRAME_CACHE_VIDEO_FPS}"
      --max-cache-frames "${FRAME_CACHE_MAX_FRAMES}"
      --progress-every "${FRAME_CACHE_PROGRESS_EVERY}"
      --summary-output "${frame_summary_output}"
    )
    if [[ -n "${frame_cache_splits[$cache_index]}" ]]; then
      frame_cmd+=(--include-splits "${frame_cache_splits[$cache_index]}")
    fi
    if [[ "${FRAME_CACHE_OVERWRITE}" == "1" ]]; then
      frame_cmd+=(--overwrite)
    fi
    "${frame_cmd[@]}"
  done
else
  echo "  - Skip frame cache stage (BUILD_FRAME_CACHE=${BUILD_FRAME_CACHE})"
fi

if resolve_toggle "${BUILD_FEATURE_CACHE}" "${PROPOSAL_MODEL_PATH}"; then
  if [[ -z "${PROPOSAL_MODEL_PATH}" ]]; then
    echo "BUILD_FEATURE_CACHE is enabled, but PROPOSAL_MODEL_PATH is empty." >&2
    exit 1
  fi
  feature_cache_data_paths=()
  feature_cache_splits=()
  if [[ -n "${FEATURE_CACHE_DATA}" ]]; then
    feature_cache_data_paths+=("${FEATURE_CACHE_DATA}")
    feature_cache_splits+=("${FEATURE_CACHE_INCLUDE_SPLITS}")
  else
    feature_cache_data_paths+=("${RUNTIME_TRAIN_JSONL}" "${RUNTIME_TEST_JSONL}")
    feature_cache_splits+=("${TRAIN_SPLIT}" "${EVAL_SPLIT}")
  fi
  for cache_index in "${!feature_cache_data_paths[@]}"; do
    feature_summary_output="${FEATURE_CACHE_SUMMARY_OUTPUT}"
    if [[ "${#feature_cache_data_paths[@]}" -gt 1 ]]; then
      feature_summary_label="${feature_cache_splits[$cache_index]}"
      feature_summary_label="${feature_summary_label//,/__}"
      feature_summary_label="${feature_summary_label// /}"
      [[ -n "${feature_summary_label}" ]] || feature_summary_label="part${cache_index}"
      feature_summary_output="$(summary_output_with_suffix "${FEATURE_CACHE_SUMMARY_OUTPUT}" "${feature_summary_label}")"
    fi
    feature_cmd=(
      python build_feature_cache.py
      --data "${feature_cache_data_paths[$cache_index]}"
      --data-root "${DATA_ROOT}"
      --model-path "${PROPOSAL_MODEL_PATH}"
      --torch-dtype "${PROPOSAL_TORCH_DTYPE}"
      --device "${FEATURE_CACHE_DEVICE}"
      --progress-every "${FEATURE_CACHE_PROGRESS_EVERY}"
      --summary-output "${feature_summary_output}"
    )
    if [[ -n "${feature_cache_splits[$cache_index]}" ]]; then
      feature_cmd+=(--include-splits "${feature_cache_splits[$cache_index]}")
    fi
    if [[ "${FEATURE_CACHE_OVERWRITE}" == "1" ]]; then
      feature_cmd+=(--overwrite)
    fi
    "${feature_cmd[@]}"
  done
else
  if [[ -n "${PROPOSAL_MODEL_PATH}" ]]; then
    echo "  - Skip feature cache stage (BUILD_FEATURE_CACHE=${BUILD_FEATURE_CACHE})"
  else
    echo "  - Skip feature cache stage because PROPOSAL_MODEL_PATH is empty."
  fi
fi

echo "[Stage 3/5] Multi-GPU SFT with immediate epoch-end rollout evaluation by default and deferred recovery fallback"
RESOLVED_SFT_MODEL_PATH=""
if is_complete_model_checkpoint_dir "${SFT_OUTPUT_DIR}"; then
  RESOLVED_SFT_MODEL_PATH="${SFT_OUTPUT_DIR}"
  echo "  - Skip SFT, found complete root checkpoint: ${RESOLVED_SFT_MODEL_PATH}"
else
  LATEST_SFT_EPOCH_RESUME_DIR="$(resolve_latest_sft_epoch_resume_dir "${SFT_OUTPUT_DIR}" || true)"
  LATEST_SFT_NUMERIC_CHECKPOINT="$(resolve_latest_numeric_checkpoint_dir "${SFT_OUTPUT_DIR}" || true)"
  LATEST_SFT_EPOCH_RESUME_PENDING=0
  if [[ -n "${LATEST_SFT_EPOCH_RESUME_DIR}" ]]; then
    LATEST_SFT_EPOCH_INDEX="$(epoch_resume_epoch_index "${LATEST_SFT_EPOCH_RESUME_DIR}" || true)"
    if ! rollout_eval_metrics_exist_for_epoch "${SFT_EVAL_OUTPUT_DIR}" "${LATEST_SFT_EPOCH_INDEX}"; then
      LATEST_SFT_EPOCH_RESUME_PENDING=1
      echo "  - Recover missing epoch-end rollout eval from ${LATEST_SFT_EPOCH_RESUME_DIR}"
      build_stage3_sft_cmd "${LATEST_SFT_EPOCH_RESUME_DIR}" 1
      "${SFT_STAGE3_CMD[@]}"
    fi
  fi
  if [[ "${LATEST_SFT_EPOCH_RESUME_PENDING}" == "1" && "$(checkpoint_has_remaining_training "${LATEST_SFT_EPOCH_RESUME_DIR}")" == "1" ]]; then
    echo "  - Resume remaining SFT epochs from ${LATEST_SFT_EPOCH_RESUME_DIR}"
    build_stage3_sft_cmd "${LATEST_SFT_EPOCH_RESUME_DIR}" 0
    "${SFT_STAGE3_CMD[@]}"
  elif [[ -n "${LATEST_SFT_NUMERIC_CHECKPOINT}" && "$(checkpoint_has_remaining_training "${LATEST_SFT_NUMERIC_CHECKPOINT}")" == "1" ]]; then
    echo "  - Resume SFT training from checkpoint: ${LATEST_SFT_NUMERIC_CHECKPOINT}"
    build_stage3_sft_cmd "${LATEST_SFT_NUMERIC_CHECKPOINT}" 0
    "${SFT_STAGE3_CMD[@]}"
  elif [[ -n "${LATEST_SFT_EPOCH_RESUME_DIR}" && "$(checkpoint_has_remaining_training "${LATEST_SFT_EPOCH_RESUME_DIR}")" == "1" ]]; then
    echo "  - Resume remaining SFT epochs from ${LATEST_SFT_EPOCH_RESUME_DIR}"
    build_stage3_sft_cmd "${LATEST_SFT_EPOCH_RESUME_DIR}" 0
    "${SFT_STAGE3_CMD[@]}"
  elif [[ -n "${LATEST_SFT_NUMERIC_CHECKPOINT}" ]]; then
    RESOLVED_SFT_MODEL_PATH="${LATEST_SFT_NUMERIC_CHECKPOINT}"
    echo "  - Skip SFT, found reusable checkpoint: ${RESOLVED_SFT_MODEL_PATH}"
  else
    echo "  - Start fresh SFT training"
    build_stage3_sft_cmd "" 0
    "${SFT_STAGE3_CMD[@]}"
  fi

  if [[ -z "${RESOLVED_SFT_MODEL_PATH}" ]]; then
    if is_complete_model_checkpoint_dir "${SFT_OUTPUT_DIR}"; then
      RESOLVED_SFT_MODEL_PATH="${SFT_OUTPUT_DIR}"
    fi
  fi
  if [[ -z "${RESOLVED_SFT_MODEL_PATH}" ]]; then
    LATEST_SFT_EPOCH_RESUME_DIR="$(resolve_latest_sft_epoch_resume_dir "${SFT_OUTPUT_DIR}" || true)"
    if [[ -n "${LATEST_SFT_EPOCH_RESUME_DIR}" ]]; then
      LATEST_SFT_EPOCH_INDEX="$(epoch_resume_epoch_index "${LATEST_SFT_EPOCH_RESUME_DIR}" || true)"
      if rollout_eval_metrics_exist_for_epoch "${SFT_EVAL_OUTPUT_DIR}" "${LATEST_SFT_EPOCH_INDEX}" && [[ "$(checkpoint_has_remaining_training "${LATEST_SFT_EPOCH_RESUME_DIR}")" != "1" ]]; then
        RESOLVED_SFT_MODEL_PATH="${LATEST_SFT_EPOCH_RESUME_DIR}"
      fi
    fi
  fi
  if [[ -z "${RESOLVED_SFT_MODEL_PATH}" ]]; then
    RESOLVED_SFT_MODEL_PATH="$(resolve_latest_numeric_checkpoint_dir "${SFT_OUTPUT_DIR}" || true)"
  fi
  if [[ -z "${RESOLVED_SFT_MODEL_PATH}" ]]; then
    echo "SFT recovery finished, but no complete checkpoint was found under ${SFT_OUTPUT_DIR}" >&2
    exit 1
  fi
  echo "  - Resolved Stage 3 checkpoint: ${RESOLVED_SFT_MODEL_PATH}"
fi

run_pending_sft_rollout_evals "${SFT_OUTPUT_DIR}" "${SFT_EVAL_OUTPUT_DIR}"

EFFECTIVE_RL_REFERENCE_MODEL_PATH="${RL_REFERENCE_MODEL_PATH}"
if [[ "${EFFECTIVE_RL_REFERENCE_MODEL_PATH}" == "${SFT_OUTPUT_DIR}" ]]; then
  EFFECTIVE_RL_REFERENCE_MODEL_PATH="${RESOLVED_SFT_MODEL_PATH}"
fi

echo "[Stage 4/5] Batch rollout, scoring, and summary"
need_batch_rollout=0
need_score_rollout=0
need_summarize_rollout=0

if [[ ! -f "${RAW_ROLLOUT_OUTPUT}" ]]; then
  need_batch_rollout=1
  need_score_rollout=1
  need_summarize_rollout=1
elif [[ ! -f "${SCORED_ROLLOUT_OUTPUT}" ]]; then
  need_score_rollout=1
  need_summarize_rollout=1
elif [[ ! -f "${ROLLOUT_SUMMARY_OUTPUT}" ]]; then
  need_summarize_rollout=1
fi

if [[ "${need_batch_rollout}" == "0" && "${need_score_rollout}" == "0" && "${need_summarize_rollout}" == "0" ]]; then
  echo "  - Skip rollout, score, and summary; all Stage 3 artifacts already exist."
else
  if [[ "${need_batch_rollout}" == "1" ]]; then
    batch_cmd_prefix=(python)
    if [[ "${ROLLOUT_NPROC_PER_NODE}" != "1" ]]; then
      batch_cmd_prefix=(torchrun --standalone --nproc_per_node="${ROLLOUT_NPROC_PER_NODE}")
    fi
    batch_cmd=(
      "${batch_cmd_prefix[@]}"
      batch_run_saver_rollout.py
      --data "${RUNTIME_TEST_JSONL}"
      --data-root "${DATA_ROOT}"
      --include-splits "${EVAL_SPLIT}"
      --start-index "${ROLLOUT_START_INDEX}"
      --count "${ROLLOUT_COUNT}"
      --max-turns "${ROLLOUT_MAX_TURNS}"
      --policy-backend qwen
      --model-path "${RESOLVED_SFT_MODEL_PATH}"
      --torch-dtype "${ROLLOUT_TORCH_DTYPE}"
      --device-map "${ROLLOUT_DEVICE_MAP}"
      --max-new-tokens "${ROLLOUT_MAX_NEW_TOKENS}"
      --max-image-side "${ROLLOUT_MAX_IMAGE_SIDE}"
      --max-image-pixels "${ROLLOUT_MAX_IMAGE_PIXELS}"
      --max-total-images "${ROLLOUT_MAX_TOTAL_IMAGES}"
      --num-preview-frames "${PIPELINE_NUM_PREVIEW_FRAMES}"
      --output "${RAW_ROLLOUT_OUTPUT}"
      --progress-every "${ROLLOUT_PROGRESS_EVERY}"
    )
    if [[ -n "${PIPELINE_PREVIEW_SAMPLING_FPS}" ]]; then
      batch_cmd+=(--preview-sampling-fps "${PIPELINE_PREVIEW_SAMPLING_FPS}")
    fi
    if [[ -n "${ROLLOUT_ATTN_IMPLEMENTATION}" ]]; then
      batch_cmd+=(--attn-implementation "${ROLLOUT_ATTN_IMPLEMENTATION}")
    fi
    if resolve_toggle "${ROLLOUT_USE_PROPOSAL_RUNTIME}" "${PROPOSAL_MODEL_PATH}"; then
      if [[ -z "${PROPOSAL_MODEL_PATH}" ]]; then
        echo "ROLLOUT_USE_PROPOSAL_RUNTIME is enabled, but PROPOSAL_MODEL_PATH is empty." >&2
        exit 1
      fi
      batch_cmd+=(
        --proposal-model-path "${PROPOSAL_MODEL_PATH}"
        --proposal-torch-dtype "${PROPOSAL_TORCH_DTYPE}"
      )
      if [[ -n "${PROPOSAL_DEVICE}" ]]; then
        batch_cmd+=(--proposal-device "${PROPOSAL_DEVICE}")
      fi
    fi
    if [[ "${ROLLOUT_DO_SAMPLE}" == "1" ]]; then
      batch_cmd+=(--do-sample --temperature "${ROLLOUT_TEMPERATURE}" --top-p "${ROLLOUT_TOP_P}")
    fi
    if [[ -n "${ROLLOUT_TOP_K}" ]]; then
      batch_cmd+=(--top-k "${ROLLOUT_TOP_K}")
    fi
    if [[ -n "${ROLLOUT_REPETITION_PENALTY}" ]]; then
      batch_cmd+=(--repetition-penalty "${ROLLOUT_REPETITION_PENALTY}")
    fi
    if [[ "${ROLLOUT_DIAGNOSTIC_ONLINE_VERIFIER_FALLBACK}" == "1" ]]; then
      batch_cmd+=(
        --diagnostic-online-verifier-fallback
        --verifier-backend "${ROLLOUT_DIAGNOSTIC_VERIFIER_BACKEND}"
        --verifier-model-path "${ROLLOUT_DIAGNOSTIC_VERIFIER_MODEL_PATH}"
        --verifier-torch-dtype "${ROLLOUT_DIAGNOSTIC_VERIFIER_TORCH_DTYPE}"
        --verifier-device-map "${ROLLOUT_DIAGNOSTIC_VERIFIER_DEVICE_MAP}"
        --verifier-max-new-tokens "${ROLLOUT_DIAGNOSTIC_VERIFIER_MAX_NEW_TOKENS}"
        --verifier-hybrid-alpha "${ROLLOUT_DIAGNOSTIC_VERIFIER_HYBRID_ALPHA}"
      )
      if [[ -n "${ROLLOUT_DIAGNOSTIC_VERIFIER_ATTN_IMPLEMENTATION}" ]]; then
        batch_cmd+=(--verifier-attn-implementation "${ROLLOUT_DIAGNOSTIC_VERIFIER_ATTN_IMPLEMENTATION}")
      fi
    fi
    "${batch_cmd[@]}"
  else
    echo "  - Skip batch rollout, file exists: ${RAW_ROLLOUT_OUTPUT}"
  fi

  if [[ "${need_score_rollout}" == "1" ]]; then
    score_cmd=(
      python score_saver_rollout.py
      --input "${RAW_ROLLOUT_OUTPUT}"
      --output "${SCORED_ROLLOUT_OUTPUT}"
      --data "${RUNTIME_TEST_JSONL}"
      --data-root "${DATA_ROOT}"
      --progress-every "${ROLLOUT_PROGRESS_EVERY}"
    )
    if [[ "${ROLLOUT_DIAGNOSTIC_ATTACH_REFERENCE_OFFLINE_VERIFIER}" == "1" ]]; then
      score_cmd+=(
        --attach-reference-offline-verifier
        --verifier-backend "${ROLLOUT_DIAGNOSTIC_VERIFIER_BACKEND}"
        --verifier-model-path "${ROLLOUT_DIAGNOSTIC_VERIFIER_MODEL_PATH}"
        --verifier-torch-dtype "${ROLLOUT_DIAGNOSTIC_VERIFIER_TORCH_DTYPE}"
        --verifier-device-map "${ROLLOUT_DIAGNOSTIC_VERIFIER_DEVICE_MAP}"
        --verifier-max-new-tokens "${ROLLOUT_DIAGNOSTIC_VERIFIER_MAX_NEW_TOKENS}"
        --verifier-hybrid-alpha "${ROLLOUT_DIAGNOSTIC_VERIFIER_HYBRID_ALPHA}"
      )
      if [[ "${ROLLOUT_DIAGNOSTIC_FORCE_REVERIFY}" == "1" ]]; then
        score_cmd+=(--force-reverify)
      fi
      if [[ -n "${ROLLOUT_DIAGNOSTIC_VERIFIER_ATTN_IMPLEMENTATION}" ]]; then
        score_cmd+=(--verifier-attn-implementation "${ROLLOUT_DIAGNOSTIC_VERIFIER_ATTN_IMPLEMENTATION}")
      fi
    fi
    if [[ -n "${EFFECTIVE_TEACHER_JUDGE_MODEL_PATH}" ]]; then
      score_cmd+=(
        --teacher-judge-model-path "${EFFECTIVE_TEACHER_JUDGE_MODEL_PATH}"
        --teacher-judge-input-mode "${TEACHER_JUDGE_INPUT_MODE}"
        --teacher-judge-torch-dtype "${TEACHER_JUDGE_TORCH_DTYPE}"
        --teacher-judge-device-map "${TEACHER_JUDGE_DEVICE_MAP}"
        --teacher-judge-max-new-tokens "${TEACHER_JUDGE_MAX_NEW_TOKENS}"
        --teacher-judge-max-images "${TEACHER_JUDGE_MAX_IMAGES}"
        --teacher-judge-topk-frames-per-view "${TEACHER_JUDGE_TOPK_FRAMES_PER_VIEW}"
      )
      if [[ -n "${TEACHER_JUDGE_ATTN_IMPLEMENTATION}" ]]; then
        score_cmd+=(--teacher-judge-attn-implementation "${TEACHER_JUDGE_ATTN_IMPLEMENTATION}")
      fi
    fi
    "${score_cmd[@]}"
  else
    echo "  - Skip score, file exists: ${SCORED_ROLLOUT_OUTPUT}"
  fi

  if [[ "${need_summarize_rollout}" == "1" ]]; then
    python summarize_saver_scores.py \
      --input "${SCORED_ROLLOUT_OUTPUT}" \
      --output "${ROLLOUT_SUMMARY_OUTPUT}" \
      --data "${RUNTIME_TEST_JSONL}" \
      --data-root "${DATA_ROOT}" \
      --max-teacher-disagreement-cases "${MAX_TEACHER_DISAGREEMENT_CASES}"
  else
    echo "  - Skip summarize, file exists: ${ROLLOUT_SUMMARY_OUTPUT}"
  fi
fi

if [[ ! -f "${RAW_ROLLOUT_OUTPUT}" || ! -f "${SCORED_ROLLOUT_OUTPUT}" || ! -f "${ROLLOUT_SUMMARY_OUTPUT}" ]]; then
  echo "Stage 3 did not produce a complete rollout/score/summary bundle under ${ROLLOUT_RUN_DIR}" >&2
  exit 1
fi

echo "[Stage 5/5] RL training"
# RL_NPROC_PER_NODE=1 时退回单进程 python；
# 大于 1 时自动用 torchrun 多卡训练。
rl_launcher=(python)
if [[ "${RL_NPROC_PER_NODE}" -gt 1 ]]; then
  rl_launcher=(torchrun --nproc_per_node="${RL_NPROC_PER_NODE}")
fi
rl_cmd=(
  "${rl_launcher[@]}"
  train_saver_rl.py
  --data "${RUNTIME_TRAIN_JSONL}"
  --data-root "${DATA_ROOT}"
  --include-splits "${TRAIN_SPLIT}"
  --model-path "${RESOLVED_SFT_MODEL_PATH}"
  --reference-model-path "${EFFECTIVE_RL_REFERENCE_MODEL_PATH}"
  --output-dir "${RL_OUTPUT_DIR}"
  --log-dir "${RL_LOG_DIR}"
  --eval-data "${RUNTIME_TEST_JSONL}"
  --eval-data-root "${DATA_ROOT}"
  --eval-include-splits "${EVAL_SPLIT}"
  --eval-rollout-max-turns "${RL_EVAL_ROLLOUT_MAX_TURNS}"
  --num-iterations "${RL_NUM_ITERATIONS}"
  --rollout-count "${RL_ROLLOUT_COUNT}"
  --num-generations "${RL_NUM_GENERATIONS}"
  --rollout-max-turns "${RL_ROLLOUT_MAX_TURNS}"
  --policy-max-new-tokens "${RL_POLICY_MAX_NEW_TOKENS}"
  --min-weight "${RL_MIN_WEIGHT}"
  --advantage-clip "${RL_ADVANTAGE_CLIP}"
  --grpo-variant "${RL_GRPO_VARIANT}"
  --cea-search-local-alpha "${RL_CEA_SEARCH_LOCAL_ALPHA}"
  --cea-alert-local-alpha "${RL_CEA_ALERT_LOCAL_ALPHA}"
  --cea-evidence-local-alpha "${RL_CEA_EVIDENCE_LOCAL_ALPHA}"
  --teacher-judge-local-alpha "${RL_TEACHER_JUDGE_LOCAL_ALPHA}"
  --cea-local-verifier-backend "${RL_CEA_LOCAL_VERIFIER_BACKEND}"
  --cea-max-search-anchors-per-rollout "${RL_CEA_MAX_SEARCH_ANCHORS_PER_ROLLOUT}"
  --cea-max-alert-anchors-per-rollout "${RL_CEA_MAX_ALERT_ANCHORS_PER_ROLLOUT}"
  --cea-max-evidence-anchors-per-rollout "${RL_CEA_MAX_EVIDENCE_ANCHORS_PER_ROLLOUT}"
  --kl-beta "${RL_KL_BETA}"
  --lora
  --bf16
  --gradient-checkpointing
  --per-device-train-batch-size "${RL_PER_DEVICE_TRAIN_BATCH_SIZE}"
  --gradient-accumulation-steps "${RL_GRADIENT_ACCUMULATION_STEPS}"
  --learning-rate "${RL_LEARNING_RATE}"
  --num-train-epochs "${RL_NUM_TRAIN_EPOCHS}"
  --max-image-side "${RL_MAX_IMAGE_SIDE}"
  --max-image-pixels "${RL_MAX_IMAGE_PIXELS}"
  --keep-recent-tool-image-messages "${RL_KEEP_RECENT_TOOL_IMAGE_MESSAGES}"
  --max-total-images "${RL_MAX_TOTAL_IMAGES}"
  --max-seq-length "${RL_MAX_SEQ_LENGTH}"
  --keep-recent-text-messages "${RL_KEEP_RECENT_TEXT_MESSAGES}"
  --dataloader-num-workers "${RL_DATALOADER_NUM_WORKERS}"
  --dataloader-prefetch-factor "${RL_DATALOADER_PREFETCH_FACTOR}"
  --logging-steps "${RL_LOGGING_STEPS}"
  --num-preview-frames "${PIPELINE_NUM_PREVIEW_FRAMES}"
)
if [[ -n "${PIPELINE_PREVIEW_SAMPLING_FPS}" ]]; then
  rl_cmd+=(--preview-sampling-fps "${PIPELINE_PREVIEW_SAMPLING_FPS}")
fi
if [[ -n "${PROPOSAL_MODEL_PATH}" ]]; then
  rl_cmd+=(
    --proposal-model-path "${PROPOSAL_MODEL_PATH}"
    --proposal-torch-dtype "${PROPOSAL_TORCH_DTYPE}"
  )
  if [[ -n "${PROPOSAL_DEVICE}" ]]; then
    rl_cmd+=(--proposal-device "${PROPOSAL_DEVICE}")
  fi
fi
if [[ -n "${EVAL_PROPOSAL_MODEL_PATH}" ]]; then
  rl_cmd+=(
    --eval-proposal-model-path "${EVAL_PROPOSAL_MODEL_PATH}"
    --eval-proposal-torch-dtype "${EVAL_PROPOSAL_TORCH_DTYPE}"
  )
  if [[ -n "${EVAL_PROPOSAL_DEVICE}" ]]; then
    rl_cmd+=(--eval-proposal-device "${EVAL_PROPOSAL_DEVICE}")
  fi
fi
if [[ -n "${EFFECTIVE_TEACHER_JUDGE_MODEL_PATH}" ]]; then
  rl_cmd+=(
    --teacher-judge-model-path "${EFFECTIVE_TEACHER_JUDGE_MODEL_PATH}"
    --teacher-judge-input-mode "${TEACHER_JUDGE_INPUT_MODE}"
    --teacher-judge-torch-dtype "${TEACHER_JUDGE_TORCH_DTYPE}"
    --teacher-judge-device-map "${TEACHER_JUDGE_DEVICE_MAP}"
    --teacher-judge-max-new-tokens "${TEACHER_JUDGE_MAX_NEW_TOKENS}"
    --teacher-judge-max-images "${TEACHER_JUDGE_MAX_IMAGES}"
    --teacher-judge-topk-frames-per-view "${TEACHER_JUDGE_TOPK_FRAMES_PER_VIEW}"
    --teacher-judge-progress-every "${TEACHER_JUDGE_PROGRESS_EVERY}"
  )
  if [[ -n "${TEACHER_JUDGE_ATTN_IMPLEMENTATION}" ]]; then
    rl_cmd+=(--teacher-judge-attn-implementation "${TEACHER_JUDGE_ATTN_IMPLEMENTATION}")
  fi
fi
if [[ "${RL_DATALOADER_PERSISTENT_WORKERS}" == "1" ]]; then
  rl_cmd+=(--dataloader-persistent-workers)
fi
if [[ -n "${RL_TENSOR_CACHE_DIR}" ]]; then
  rl_cmd+=(
    --tensor-cache-dir "${RL_TENSOR_CACHE_DIR}"
    --tensor-cache-progress-every "${RL_TENSOR_CACHE_PROGRESS_EVERY}"
  )
  if [[ "${RL_TENSOR_CACHE_OVERWRITE_EXISTING}" == "1" ]]; then
    rl_cmd+=(--tensor-cache-overwrite-existing)
  fi
fi
if [[ -n "${RL_ATTN_IMPLEMENTATION}" ]]; then
  rl_cmd+=(--attn-implementation "${RL_ATTN_IMPLEMENTATION}")
fi
if [[ "${RL_CEA_ENABLE_SEARCH_GROUP}" == "1" ]]; then
  rl_cmd+=(--cea-enable-search-group)
fi
if [[ "${RL_CEA_ENABLE_ALERT_GROUP}" == "1" ]]; then
  rl_cmd+=(--cea-enable-alert-group)
fi
if [[ "${RL_CEA_ENABLE_EVIDENCE_GROUP}" == "1" ]]; then
  rl_cmd+=(--cea-enable-evidence-group)
fi
if [[ "${RL_CEA_LOCAL_USE_REFERENCE_SUPERVISION}" == "1" ]]; then
  rl_cmd+=(--cea-local-use-reference-supervision)
fi
if [[ "${RL_POLICY_DO_SAMPLE}" == "1" ]]; then
  rl_cmd+=(--policy-do-sample --policy-temperature "${RL_POLICY_TEMPERATURE}" --policy-top-p "${RL_POLICY_TOP_P}")
fi
if [[ "${RL_DIAGNOSTIC_ONLINE_VERIFIER_FALLBACK}" == "1" ]]; then
  rl_cmd+=(
    --diagnostic-online-verifier-fallback
    --verifier-backend "${RL_DIAGNOSTIC_VERIFIER_BACKEND}"
    --verifier-model-path "${RL_DIAGNOSTIC_VERIFIER_MODEL_PATH}"
    --verifier-torch-dtype "${RL_DIAGNOSTIC_VERIFIER_TORCH_DTYPE}"
    --verifier-device-map "${RL_DIAGNOSTIC_VERIFIER_DEVICE_MAP}"
    --verifier-max-new-tokens "${RL_DIAGNOSTIC_VERIFIER_MAX_NEW_TOKENS}"
    --verifier-hybrid-alpha "${RL_DIAGNOSTIC_VERIFIER_HYBRID_ALPHA}"
  )
  if [[ -n "${RL_DIAGNOSTIC_VERIFIER_ATTN_IMPLEMENTATION}" ]]; then
    rl_cmd+=(--verifier-attn-implementation "${RL_DIAGNOSTIC_VERIFIER_ATTN_IMPLEMENTATION}")
  fi
fi
if [[ "${RL_DIAGNOSTIC_ATTACH_REFERENCE_OFFLINE_VERIFIER}" == "1" ]]; then
  rl_cmd+=(
    --diagnostic-attach-reference-offline-verifier
    --verifier-backend "${RL_DIAGNOSTIC_VERIFIER_BACKEND}"
    --verifier-model-path "${RL_DIAGNOSTIC_VERIFIER_MODEL_PATH}"
    --verifier-torch-dtype "${RL_DIAGNOSTIC_VERIFIER_TORCH_DTYPE}"
    --verifier-device-map "${RL_DIAGNOSTIC_VERIFIER_DEVICE_MAP}"
    --verifier-max-new-tokens "${RL_DIAGNOSTIC_VERIFIER_MAX_NEW_TOKENS}"
    --verifier-hybrid-alpha "${RL_DIAGNOSTIC_VERIFIER_HYBRID_ALPHA}"
  )
  if [[ -n "${RL_DIAGNOSTIC_VERIFIER_ATTN_IMPLEMENTATION}" ]]; then
    rl_cmd+=(--verifier-attn-implementation "${RL_DIAGNOSTIC_VERIFIER_ATTN_IMPLEMENTATION}")
  fi
fi
if [[ "${RL_DIAGNOSTIC_FORCE_REVERIFY}" == "1" ]]; then
  rl_cmd+=(--diagnostic-force-reverify)
fi
if [[ "${RL_EVAL_MAX_RECORDS}" != "0" ]]; then
  rl_cmd+=(--eval-max-records "${RL_EVAL_MAX_RECORDS}")
fi
if [[ "${RL_EVAL_MAX_NEW_TOKENS_PER_TURN}" != "0" ]]; then
  rl_cmd+=(--eval-max-new-tokens-per-turn "${RL_EVAL_MAX_NEW_TOKENS_PER_TURN}")
fi
if [[ "${RL_EVAL_MAX_TOTAL_IMAGES}" != "0" ]]; then
  rl_cmd+=(--eval-max-total-images "${RL_EVAL_MAX_TOTAL_IMAGES}")
fi
if [[ "${RL_EVAL_ATTACH_REFERENCE_DIAGNOSTICS}" == "1" ]]; then
  rl_cmd+=(
    --eval-attach-reference-diagnostics
    --eval-verifier-backend "${RL_EVAL_DIAGNOSTIC_VERIFIER_BACKEND}"
    --eval-verifier-model-path "${RL_DIAGNOSTIC_VERIFIER_MODEL_PATH}"
    --eval-verifier-torch-dtype "${RL_DIAGNOSTIC_VERIFIER_TORCH_DTYPE}"
    --eval-verifier-device-map "${RL_DIAGNOSTIC_VERIFIER_DEVICE_MAP}"
    --eval-verifier-max-new-tokens "${RL_DIAGNOSTIC_VERIFIER_MAX_NEW_TOKENS}"
    --eval-verifier-hybrid-alpha "${RL_DIAGNOSTIC_VERIFIER_HYBRID_ALPHA}"
  )
  if [[ -n "${RL_DIAGNOSTIC_VERIFIER_ATTN_IMPLEMENTATION}" ]]; then
    rl_cmd+=(--eval-verifier-attn-implementation "${RL_DIAGNOSTIC_VERIFIER_ATTN_IMPLEMENTATION}")
  fi
fi
"${rl_cmd[@]}"

echo "Full pipeline finished."
echo "FRAME_CACHE_SUMMARY_OUTPUT=${FRAME_CACHE_SUMMARY_OUTPUT}"
echo "FEATURE_CACHE_SUMMARY_OUTPUT=${FEATURE_CACHE_SUMMARY_OUTPUT}"
echo "PROPOSAL_MODEL_PATH=${PROPOSAL_MODEL_PATH}"
echo "SFT_OUTPUT_DIR=${SFT_OUTPUT_DIR}"
echo "RESOLVED_SFT_MODEL_PATH=${RESOLVED_SFT_MODEL_PATH}"
echo "RAW_ROLLOUT_OUTPUT=${RAW_ROLLOUT_OUTPUT}"
echo "SCORED_ROLLOUT_OUTPUT=${SCORED_ROLLOUT_OUTPUT}"
echo "ROLLOUT_SUMMARY_OUTPUT=${ROLLOUT_SUMMARY_OUTPUT}"
echo "RL_OUTPUT_DIR=${RL_OUTPUT_DIR}"
