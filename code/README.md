# SAVER 命令行流程

这个 README 只保留直接可运行的命令。

- 不使用 `scripts/*.sh`
- 预处理产生的 `json/jsonl` 统一放在 [`data_utils`](/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/data_utils)
- 训练和 rollout 输出放在 `code/ckpt/<EXP_NAME>/...`

以下命令默认在 [`/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code`](/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code) 目录执行。

## 1. 通用变量

```bash
conda activate qwen3-vl
cd /mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code

export DATA_ROOT=/mnt/shared-storage-user/mineru2-shared/zengweijun
export PROJECT_ROOT=${DATA_ROOT}/Wmh/ideas/idea2_v2
export MODEL_ROOT=${DATA_ROOT}/Wmh/MLLMs

export EXP_NAME=exp1
export EXP_DIR=$(pwd)/ckpt/${EXP_NAME}
export CKPT_DIR=${EXP_DIR}/checkpoints
export ROLLOUT_DIR=${EXP_DIR}/rollouts
export DATA_UTILS_DIR=$(pwd)/data_utils

mkdir -p "${CKPT_DIR}" "${ROLLOUT_DIR}" "${DATA_UTILS_DIR}"

export CANONICAL_JSONL=${PROJECT_ROOT}/benchmark_annotations/msad_saver_with_qwen.jsonl

export AGENT_TRAIN_JSONL=${DATA_UTILS_DIR}/msad_saver_agent_train.jsonl
export ORACLE_TRAIN_JSONL=${DATA_UTILS_DIR}/msad_saver_oracle_sft_train.jsonl
export ORACLE_TEST_JSONL=${DATA_UTILS_DIR}/msad_saver_oracle_sft_test.jsonl
export PREPARED_TRAIN_JSONL=${DATA_UTILS_DIR}/msad_saver_agent_train.prepared_sft.jsonl

export MODEL_PATH=${MODEL_ROOT}/qwen3-vl-8b-Instruct
export VERIFIER_MODEL_PATH=${MODEL_ROOT}/qwen3-vl-8b-Instruct
export PROPOSAL_MODEL_PATH=${MODEL_ROOT}/siglip

export TRAIN_SPLIT=train
export EVAL_SPLIT=test
```

常用变量：

- `EXP_NAME`: 实验名，决定 `ckpt/<EXP_NAME>` 输出目录
- `DATA_UTILS_DIR`: 预处理 `json/jsonl` 输出目录
- `MODEL_PATH`: policy 初始模型
- `VERIFIER_MODEL_PATH`: verifier 模型
- `PROPOSAL_MODEL_PATH`: proposal encoder，通常是 SigLIP

## 2. 数据预处理

### 2.1 canonical -> agent_train

```bash
python convert_to_saver_agent.py \
  --input ../benchmark_annotations/msad_saver_with_qwen.jsonl \
  --output data_utils/msad_saver_agent_all.jsonl \
  --mode agent_train \
  --adapter msad_saver_qwen \
  --include-splits "train, test"
```

### 2.2 canonical -> oracle_sft_train train 和 test 都要

```bash
python convert_to_saver_agent.py \
  --input ../benchmark_annotations/msad_saver_with_qwen.jsonl \
  --output data_utils/msad_saver_oracle_sft_train.jsonl \
  --mode oracle_sft \
  --adapter msad_saver_qwen \
  --include-splits "train"
```

```bash
python convert_to_saver_agent.py \
  --input ../benchmark_annotations/msad_saver_with_qwen.jsonl \
  --output data_utils/msad_saver_oracle_sft_test.jsonl \
  --mode oracle_sft \
  --adapter msad_saver_qwen \
  --include-splits "test"
```

### 2.3 agent_train -> prepared_sft

```bash
python train_saver_sft.py \
  --data data_utils/msad_saver_agent_all.jsonl \
  --data-root /mnt/shared-storage-user/mineru2-shared/zengweijun \
  --include-splits "train" \
  --prepare-output data_utils/msad_saver_agent_train.prepared_sft.jsonl \
  --prepare-only \
  --validate-prepared-data \
  --progress-every 25
```

这一阶段产物都在 [`data_utils`](/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/data_utils)：

- `msad_saver_agent_train.jsonl`
- `msad_saver_oracle_sft_train.jsonl`
- `msad_saver_oracle_sft_test.jsonl`
- `msad_saver_agent_train.prepared_sft.jsonl`

相关变量：

- `CANONICAL_JSONL`
- `AGENT_TRAIN_JSONL`
- `ORACLE_TRAIN_JSONL`
- `ORACLE_TEST_JSONL`
- `PREPARED_TRAIN_JSONL`
- `TRAIN_SPLIT`
- `EVAL_SPLIT`

## 3. 构建 `.frame_cache`

### 3.1 train

```bash
python build_frame_cache.py \
  --data data_utils/msad_saver_oracle_sft_train.jsonl \
  --data-root /mnt/shared-storage-user/mineru2-shared/zengweijun \
  --include-splits "train" \
  --cache-video-fps 2.0 \
  --max-cache-frames 256 \
  --progress-every 50 \
  --summary-output data_utils/frame_cache_train_summary.json
```

### 3.2 test

```bash
python build_frame_cache.py \
  --data data_utils/msad_saver_oracle_sft_test.jsonl \
  --data-root /mnt/shared-storage-user/mineru2-shared/zengweijun \
  --include-splits "test" \
  --cache-video-fps 2.0 \
  --max-cache-frames 256 \
  --progress-every 50 \
  --summary-output data_utils/frame_cache_test_summary.json
```

相关变量：

- `--cache-video-fps`
- `--max-cache-frames`
- `--overwrite`

## 4. 构建 `.feature_cache`

### 4.1 train

```bash
python build_feature_cache.py \
  --data data_utils/msad_saver_oracle_sft_train.jsonl \
  --data-root /mnt/shared-storage-user/mineru2-shared/zengweijun \
  --include-splits "train" \
  --model-path /mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/MLLMs/siglip \
  --torch-dtype auto \
  --device cuda:0 \
  --progress-every 25 \
  --summary-output data_utils/feature_cache_train_summary.json
```

### 4.2 test

```bash
python build_feature_cache.py \
  --data data_utils/msad_saver_oracle_sft_test.jsonl \
  --data-root /mnt/shared-storage-user/mineru2-shared/zengweijun \
  --include-splits "test" \
  --model-path /mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/MLLMs/siglip \
  --torch-dtype auto \
  --device cuda:0 \
  --progress-every 25 \
  --summary-output data_utils/feature_cache_test_summary.json
```

相关变量：

- `PROPOSAL_MODEL_PATH`
- `--device`
- `--torch-dtype`
- `--overwrite`

## 5. SFT

4 卡训练直接用 `torchrun`。如果只是单卡调试，把 `torchrun --nproc_per_node=4` 改成 `python` 即可。

```bash
export SFT_OUTPUT_DIR=${CKPT_DIR}/saver_sft_qwen3vl_8b_eval_ddp

export SFT_NPROC_PER_NODE=4
export SFT_LEARNING_RATE=1e-5
export SFT_NUM_TRAIN_EPOCHS=2.0
export SFT_PER_DEVICE_TRAIN_BATCH_SIZE=1
export SFT_GRADIENT_ACCUMULATION_STEPS=16
export SFT_EVAL_ROLLOUT_MAX_TURNS=12

export SFT_MAX_IMAGE_SIDE=0
export SFT_MAX_IMAGE_PIXELS=0
export SFT_KEEP_RECENT_TOOL_IMAGE_MESSAGES=0
export SFT_MAX_TOTAL_IMAGES=0
export SFT_MAX_SEQ_LENGTH=4096
export SFT_KEEP_RECENT_TEXT_MESSAGES=12

export SFT_DATALOADER_NUM_WORKERS=4
export SFT_DATALOADER_PREFETCH_FACTOR=2

torchrun --nproc_per_node=${SFT_NPROC_PER_NODE} train_saver_sft.py \
  --prepared-data "${PREPARED_TRAIN_JSONL}" \
  --include-splits "${TRAIN_SPLIT}" \
  --model-path "${MODEL_PATH}" \
  --output-dir "${SFT_OUTPUT_DIR}" \
  --eval-data "${ORACLE_TEST_JSONL}" \
  --eval-data-root "${DATA_ROOT}" \
  --eval-rollout-max-turns "${SFT_EVAL_ROLLOUT_MAX_TURNS}" \
  --eval-verifier-backend heuristic \
  --eval-progress-every 1 \
  --lora \
  --bf16 \
  --gradient-checkpointing \
  --per-device-train-batch-size "${SFT_PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient-accumulation-steps "${SFT_GRADIENT_ACCUMULATION_STEPS}" \
  --learning-rate "${SFT_LEARNING_RATE}" \
  --num-train-epochs "${SFT_NUM_TRAIN_EPOCHS}" \
  --max-image-side "${SFT_MAX_IMAGE_SIDE}" \
  --max-image-pixels "${SFT_MAX_IMAGE_PIXELS}" \
  --keep-recent-tool-image-messages "${SFT_KEEP_RECENT_TOOL_IMAGE_MESSAGES}" \
  --max-total-images "${SFT_MAX_TOTAL_IMAGES}" \
  --max-seq-length "${SFT_MAX_SEQ_LENGTH}" \
  --keep-recent-text-messages "${SFT_KEEP_RECENT_TEXT_MESSAGES}" \
  --dataloader-num-workers "${SFT_DATALOADER_NUM_WORKERS}" \
  --dataloader-prefetch-factor "${SFT_DATALOADER_PREFETCH_FACTOR}" \
  --dataloader-persistent-workers \
  --attn-implementation flash_attention_3 \
  --logging-steps 5
```

相关变量：

- `SFT_OUTPUT_DIR`
- `SFT_NPROC_PER_NODE`
- `SFT_LEARNING_RATE`
- `SFT_NUM_TRAIN_EPOCHS`
- `SFT_PER_DEVICE_TRAIN_BATCH_SIZE`
- `SFT_GRADIENT_ACCUMULATION_STEPS`
- `SFT_EVAL_ROLLOUT_MAX_TURNS`
- `SFT_MAX_IMAGE_SIDE`
- `SFT_MAX_IMAGE_PIXELS`
- `SFT_KEEP_RECENT_TOOL_IMAGE_MESSAGES`
- `SFT_MAX_TOTAL_IMAGES`
- `SFT_MAX_SEQ_LENGTH`
- `SFT_KEEP_RECENT_TEXT_MESSAGES`
- `SFT_DATALOADER_NUM_WORKERS`
- `SFT_DATALOADER_PREFETCH_FACTOR`

## 6. Batch Rollout

```bash
export ROLLOUT_NAME=sft_rollout_eval
export ROLLOUT_RUN_DIR=${ROLLOUT_DIR}/${ROLLOUT_NAME}
export RAW_ROLLOUT_JSONL=${ROLLOUT_RUN_DIR}/rollouts.raw.jsonl

mkdir -p "${ROLLOUT_RUN_DIR}"

python batch_run_saver_rollout.py \
  --data "${ORACLE_TEST_JSONL}" \
  --data-root "${DATA_ROOT}" \
  --include-splits "${EVAL_SPLIT}" \
  --start-index 0 \
  --count 0 \
  --max-turns 12 \
  --policy-backend qwen \
  --model-path "${SFT_OUTPUT_DIR}" \
  --proposal-model-path "${PROPOSAL_MODEL_PATH}" \
  --proposal-torch-dtype auto \
  --proposal-device cuda:0 \
  --torch-dtype auto \
  --device-map auto \
  --attn-implementation flash_attention_3 \
  --max-new-tokens 512 \
  --output "${RAW_ROLLOUT_JSONL}" \
  --progress-every 5 \
  --verifier-backend heuristic \
  --verifier-model-path "${VERIFIER_MODEL_PATH}" \
  --verifier-device-map auto \
  --verifier-torch-dtype auto \
  --verifier-hybrid-alpha 0.7
```

相关变量：

- `ROLLOUT_NAME`
- `RAW_ROLLOUT_JSONL`
- `--count`: `0` 表示从 `start-index` 跑到该 split 结束
- `--max-turns`
- `--proposal-model-path`
- `--proposal-device`
- `--do-sample`
- `--temperature`
- `--top-p`
- `--top-k`
- `--repetition-penalty`

## 7. Offline Score

```bash
export SCORED_ROLLOUT_JSONL=${ROLLOUT_RUN_DIR}/rollouts.scored.jsonl

python score_saver_rollout.py \
  --input "${RAW_ROLLOUT_JSONL}" \
  --output "${SCORED_ROLLOUT_JSONL}" \
  --data "${ORACLE_TEST_JSONL}" \
  --data-root "${DATA_ROOT}" \
  --verifier-backend hybrid \
  --force-reverify \
  --verifier-model-path "${VERIFIER_MODEL_PATH}" \
  --verifier-device-map auto \
  --verifier-torch-dtype auto \
  --verifier-attn-implementation flash_attention_3 \
  --verifier-hybrid-alpha 0.7 \
  --progress-every 5
```

相关变量：

- `SCORED_ROLLOUT_JSONL`
- `--verifier-backend`
- `--force-reverify`
- `--verifier-model-path`
- `--verifier-device-map`
- `--verifier-torch-dtype`
- `--verifier-attn-implementation`
- `--verifier-hybrid-alpha`

## 8. Summarize

```bash
export SUMMARY_JSON=${ROLLOUT_RUN_DIR}/summary.json

python summarize_saver_scores.py \
  --input "${SCORED_ROLLOUT_JSONL}" \
  --output "${SUMMARY_JSON}" \
  --data "${ORACLE_TEST_JSONL}" \
  --data-root "${DATA_ROOT}"
```

输出文件：

- `RAW_ROLLOUT_JSONL`
- `SCORED_ROLLOUT_JSONL`
- `SUMMARY_JSON`

## 9. RL

4 卡训练直接用 `torchrun`。如果只是单卡调试，把 `torchrun --nproc_per_node=4` 改成 `python` 即可。

```bash
export RL_OUTPUT_DIR=${CKPT_DIR}/saver_cea_grpo_v1

export RL_NPROC_PER_NODE=4
export RL_NUM_ITERATIONS=3
export RL_ROLLOUT_COUNT=16
export RL_NUM_GENERATIONS=4
export RL_ROLLOUT_MAX_TURNS=12

export RL_LEARNING_RATE=5e-6
export RL_NUM_TRAIN_EPOCHS=1.0
export RL_PER_DEVICE_TRAIN_BATCH_SIZE=1
export RL_GRADIENT_ACCUMULATION_STEPS=16

export RL_KL_BETA=0.02
export RL_GRPO_VARIANT=cea_grpo

export RL_CEA_SEARCH_LOCAL_ALPHA=0.5
export RL_CEA_ALERT_LOCAL_ALPHA=0.5
export RL_CEA_EVIDENCE_LOCAL_ALPHA=0.5

export RL_MAX_IMAGE_SIDE=640
export RL_MAX_IMAGE_PIXELS=0
export RL_KEEP_RECENT_TOOL_IMAGE_MESSAGES=0
export RL_MAX_TOTAL_IMAGES=44
export RL_MAX_SEQ_LENGTH=4096
export RL_KEEP_RECENT_TEXT_MESSAGES=12

torchrun --nproc_per_node=${RL_NPROC_PER_NODE} train_saver_rl.py \
  --data "${AGENT_TRAIN_JSONL}" \
  --data-root "${DATA_ROOT}" \
  --include-splits "${TRAIN_SPLIT}" \
  --model-path "${SFT_OUTPUT_DIR}" \
  --reference-model-path "${SFT_OUTPUT_DIR}" \
  --output-dir "${RL_OUTPUT_DIR}" \
  --eval-data "${ORACLE_TEST_JSONL}" \
  --eval-data-root "${DATA_ROOT}" \
  --eval-rollout-max-turns 12 \
  --eval-verifier-backend heuristic \
  --num-iterations "${RL_NUM_ITERATIONS}" \
  --rollout-count "${RL_ROLLOUT_COUNT}" \
  --num-generations "${RL_NUM_GENERATIONS}" \
  --rollout-max-turns "${RL_ROLLOUT_MAX_TURNS}" \
  --grpo-variant "${RL_GRPO_VARIANT}" \
  --cea-enable-search-group \
  --cea-enable-alert-group \
  --cea-enable-evidence-group \
  --cea-search-local-alpha "${RL_CEA_SEARCH_LOCAL_ALPHA}" \
  --cea-alert-local-alpha "${RL_CEA_ALERT_LOCAL_ALPHA}" \
  --cea-evidence-local-alpha "${RL_CEA_EVIDENCE_LOCAL_ALPHA}" \
  --cea-local-verifier-backend heuristic \
  --kl-beta "${RL_KL_BETA}" \
  --verifier-backend hybrid \
  --verifier-model-path "${VERIFIER_MODEL_PATH}" \
  --verifier-torch-dtype auto \
  --verifier-device-map auto \
  --verifier-attn-implementation flash_attention_3 \
  --verifier-hybrid-alpha 0.7 \
  --policy-do-sample \
  --policy-temperature 0.7 \
  --policy-top-p 0.9 \
  --lora \
  --bf16 \
  --gradient-checkpointing \
  --per-device-train-batch-size "${RL_PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient-accumulation-steps "${RL_GRADIENT_ACCUMULATION_STEPS}" \
  --learning-rate "${RL_LEARNING_RATE}" \
  --num-train-epochs "${RL_NUM_TRAIN_EPOCHS}" \
  --max-image-side "${RL_MAX_IMAGE_SIDE}" \
  --max-image-pixels "${RL_MAX_IMAGE_PIXELS}" \
  --keep-recent-tool-image-messages "${RL_KEEP_RECENT_TOOL_IMAGE_MESSAGES}" \
  --max-total-images "${RL_MAX_TOTAL_IMAGES}" \
  --max-seq-length "${RL_MAX_SEQ_LENGTH}" \
  --keep-recent-text-messages "${RL_KEEP_RECENT_TEXT_MESSAGES}" \
  --dataloader-num-workers 4 \
  --dataloader-prefetch-factor 2 \
  --dataloader-persistent-workers \
  --attn-implementation flash_attention_3 \
  --logging-steps 10
```

相关变量：

- `RL_OUTPUT_DIR`
- `RL_NPROC_PER_NODE`
- `RL_NUM_ITERATIONS`
- `RL_ROLLOUT_COUNT`
- `RL_NUM_GENERATIONS`
- `RL_ROLLOUT_MAX_TURNS`
- `RL_LEARNING_RATE`
- `RL_NUM_TRAIN_EPOCHS`
- `RL_PER_DEVICE_TRAIN_BATCH_SIZE`
- `RL_GRADIENT_ACCUMULATION_STEPS`
- `RL_KL_BETA`
- `RL_GRPO_VARIANT`
- `RL_CEA_SEARCH_LOCAL_ALPHA`
- `RL_CEA_ALERT_LOCAL_ALPHA`
- `RL_CEA_EVIDENCE_LOCAL_ALPHA`
- `RL_MAX_IMAGE_SIDE`
- `RL_MAX_IMAGE_PIXELS`
- `RL_KEEP_RECENT_TOOL_IMAGE_MESSAGES`
- `RL_MAX_TOTAL_IMAGES`
- `RL_MAX_SEQ_LENGTH`
- `RL_KEEP_RECENT_TEXT_MESSAGES`

## 10. 最短顺序

按顺序执行下面这些入口即可：

```text
python convert_to_saver_agent.py   # agent_train
python convert_to_saver_agent.py   # oracle_sft_train
python convert_to_saver_agent.py   # oracle_sft_test
python train_saver_sft.py --prepare-only
python build_frame_cache.py        # train
python build_frame_cache.py        # test
python build_feature_cache.py      # train
python build_feature_cache.py      # test
torchrun train_saver_sft.py
python batch_run_saver_rollout.py
python score_saver_rollout.py
python summarize_saver_scores.py
torchrun train_saver_rl.py
```
