# SAVER x TimeSearch-R 代码库

这个目录是当前 SAVER 最终版代码基线。它借鉴了 TimeSearch-R 的多轮 rollout / tool-use 风格，但核心目标不是纯 proposal 搜索，而是围绕 SAVER 的完整闭环来训练和评估：

`preview -> search -> alert -> verify -> finalize`

当前版本已经包含：

- schema-driven tool-use prompt
- oracle verifier branch supervision
- Qwen policy / verifier 接口
- batch rollout / offline scoring / summarize
- token-level clipped policy optimization + reference KL
- Counterfactual Evidence-and-Alert GRPO, 即 CEA-GRPO

## 1. 一眼看懂整体流程

推荐把整套流程理解成下面 5 个阶段：

1. `canonical 标注 -> agent/oracle 数据`
2. `为真实视频预建 .frame_cache`
3. `prepared SFT -> SFT warm start`
4. `batch rollout -> score -> summarize`
5. `CEA-GRPO RL`

仓库内对应的主要入口如下：

- `build_frame_cache.py`
  - 为训练/评测视频生成 `video.mp4.frame_cache`
- `convert_to_saver_agent.py`
  - 把 canonical 标注转换成 `agent_train` / `oracle_sft`
- `train_saver_sft.py`
  - 负责 `prepare-only` 和正式 SFT
- `batch_run_saver_rollout.py`
  - 批量生成 rollout
- `score_saver_rollout.py`
  - 用 reward + verifier 做离线打分
- `summarize_saver_scores.py`
  - 汇总 rollout 打分结果
- `train_saver_rl.py`
  - 默认启用 CEA-GRPO 的 RL 管线

如果你想少记命令，直接记住这些脚本：

- `scripts/00_full_pipeline.sh`
  - 从数据准备一路跑到 RL
- `scripts/01_build_oracle_and_prepare_sft.sh`
  - 重建 `agent_train / oracle / prepared_sft`
- `scripts/02_train_sft_with_rollout_eval.sh`
  - 只跑 SFT
- `scripts/03_batch_rollout_score_summarize.sh`
  - 只跑 rollout / score / summarize
- `scripts/04_train_rl.sh`
  - 只跑 RL

## 2. 核心概念

### 2.1 四个模型角色

- `model`
  - 主 SAVER policy
  - 决定何时 search、何时 alert、何时 verify、何时 finalize
- `old policy`
  - PPO/GRPO 比率里的分母
  - 每个 RL iteration 开始时冻结
- `reference policy`
  - 只用于 KL regularization
  - 通常固定为初始 SFT checkpoint
- `verifier-model`
  - `verify_hypothesis` 内部使用的模型
  - 判断当前证据子集是否足以且有必要支撑当前 claim / alert

### 2.2 verifier 现在是什么，不是什么

SAVER 里的 verifier 不再被定义成“答案对不对”，而是：

`当前选中的证据子集，是否足以支撑当前 anomaly claim 和当前 alert 决策。`

因此它服务的是反事实验证，而不是普通分类打分。

### 2.3 oracle verifier feedback 与 SFT 分支监督

当前 oracle SFT 不再是“verify 一下然后立刻 finalize”的固定脚本。

`verify_hypothesis` 之后现在显式监督三类后续分支：

- `revise_claim`
- `continue_search`
- `finalize`

这些反馈写在 `oracle_sft.trajectory[*].oracle_verifier_feedback` 中，`train_saver_sft.py --prepare-only` 会把它重新注入 verify 的 tool observation，所以 prepared SFT 里学到的是“verifier 作为局部控制器”，不是“verifier 作为盖章器”。

### 2.4 显式双预算

对 SAVER 来说，单独的 TimeSearch-R 式 `video token budget` 不够。

因为长轨迹的 token 不只来自帧，还来自：

- tool observation
- JSON 参数
- verifier feedback
- 多轮 assistant/tool 历史

所以现在训练默认使用 `multimodal dual budget`：

- 视觉预算
  - `--max-image-side`
  - `--max-image-pixels`
  - `--keep-recent-tool-image-messages`
  - `--max-total-images`
- 文本预算
  - `--max-seq-length`
  - `--keep-recent-text-messages`

## 3. 数据、cache 与产物

### 3.1 常见输入文件

- canonical 标注
  - `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/benchmark_annotations/msad_saver_with_qwen.jsonl`
- agent-ready 数据
  - `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/benchmark_annotations/msad_saver_agent_train.jsonl`

### 3.2 典型数据链路

```text
msad_saver_with_qwen.jsonl
  -> convert_to_saver_agent.py --mode agent_train
  -> msad_saver_agent_train.jsonl
  -> train_saver_sft.py --prepare-only
  -> msad_saver_agent_train.prepared_sft.jsonl
  -> train_saver_sft.py
  -> SFT checkpoint
  -> batch rollout / score / summarize
  -> train_saver_rl.py
```

### 3.3 `.frame_cache` 是什么

训练时，prepared SFT 中的图片不会直接把原始 tensor 塞进 JSONL，而是以 `image_ref` 形式回指到视频。

如果旁边存在：

```text
video.mp4.frame_cache
```

训练会优先从 cache 读帧；否则会退回到真实视频解码。

这就是为什么在共享存储上直接训练时，经常会看到：

```text
Epoch 1/2: 0%| | 0/40
```

长时间不动。很多时候不是训练死了，而是首个 batch 还在做真实视频读取和解码。

### 3.4 `train_artifacts` 与 `checkpoints`

- `train_artifacts/`
  - 更像上游准备产物
  - 例如 `prepared_sft.jsonl`、summary、details
- `checkpoints/`
  - 真正的 SFT / RL 模型产物
- `rollouts/`
  - rollout、score 和 summarize 输出

## 4. 先做这一步：预建 frame cache

`build_frame_cache.py` 现在是强烈推荐的 Stage 0。

它会：

- 读取 agent/oracle JSONL
- 按 split 过滤
- 解析真实视频路径
- 用与 `SaverAgentDataset` 一致的策略采样帧
- 在原视频旁边写 `video.mp4.frame_cache`

### 4.1 给训练集建 cache

```bash
conda activate qwen3-vl
cd /mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code

python build_frame_cache.py \
  --data /mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/benchmark_annotations/msad_saver_agent_train.jsonl \
  --data-root /mnt/shared-storage-user/mineru2-shared/zengweijun \
  --include-splits train \
  --cache-video-fps 2.0 \
  --max-cache-frames 256 \
  --progress-every 50 \
  --summary-output /mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/train_artifacts/frame_cache_train_summary.json
```

### 4.2 给测试/评测集建 cache

```bash
python build_frame_cache.py \
  --data /mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/benchmark_annotations/msad_saver_oracle_sft.jsonl \
  --data-root /mnt/shared-storage-user/mineru2-shared/zengweijun \
  --include-splits test \
  --cache-video-fps 2.0 \
  --max-cache-frames 256 \
  --progress-every 50 \
  --summary-output /mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/train_artifacts/frame_cache_test_summary.json
```

### 4.3 什么时候加 `--overwrite`

如果你怀疑旧 cache 是按错误路径、错误 fps 或旧数据生成的，重建时加：

```bash
--overwrite
```

## 5. 推荐运行思路

这一节是最重要的。以后跑实验，优先按这里选路径。

### 5.1 从零开始完整跑

这是最推荐的主路径：

1. 先建 train/test 的 `.frame_cache`
2. 再跑 `scripts/00_full_pipeline.sh`

对你当前的 4xH200，我推荐这套正式版配置：

- 主 policy: `Qwen3-VL-8B-Instruct`
- verifier: `Qwen3-VL-32B-Instruct`
- SFT / RL 都用 `max_seq_length=6144`
- SFT / RL 都保留更多 recent text history
- rollout / RL 都使用 hybrid verifier

直接运行：

```bash
conda activate qwen3-vl
cd /mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code

MODEL_ROOT=/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/MLLMs \
MODEL_PATH=/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/MLLMs/qwen3-vl-8b-Instruct \
ROLLOUT_VERIFIER_MODEL_PATH=/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/MLLMs/Qwen3-VL-32B-Instruct \
RL_VERIFIER_MODEL_PATH=/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/MLLMs/Qwen3-VL-32B-Instruct \
SFT_OUTPUT_DIR=/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/checkpoints/saver_sft_qwen3vl_8b_h200_ctx6k \
RL_OUTPUT_DIR=/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/checkpoints/saver_cea_grpo_h200_ctx6k \
ROLLOUT_RUN_NAME=sft_rollout_eval_h200_ctx6k \
VALIDATE_MATERIALIZATION=1 \
VALIDATION_MAX_EXAMPLES=256 \
SFT_NPROC_PER_NODE=4 \
SFT_NUM_TRAIN_EPOCHS=2.0 \
SFT_GRADIENT_ACCUMULATION_STEPS=4 \
SFT_MAX_IMAGE_SIDE=640 \
SFT_KEEP_RECENT_TOOL_IMAGE_MESSAGES=2 \
SFT_MAX_TOTAL_IMAGES=44 \
SFT_MAX_SEQ_LENGTH=6144 \
SFT_KEEP_RECENT_TEXT_MESSAGES=16 \
SFT_EVAL_MAX_RECORDS=160 \
ROLLOUT_COUNT=0 \
ROLLOUT_DO_SAMPLE=0 \
ROLLOUT_VERIFIER_BACKEND=hybrid \
ROLLOUT_VERIFIER_HYBRID_ALPHA=0.8 \
RL_NPROC_PER_NODE=4 \
RL_NUM_ITERATIONS=4 \
RL_ROLLOUT_COUNT=32 \
RL_NUM_GENERATIONS=6 \
RL_POLICY_DO_SAMPLE=1 \
RL_POLICY_TEMPERATURE=0.7 \
RL_POLICY_TOP_P=0.9 \
RL_KL_BETA=0.02 \
RL_MIN_WEIGHT=0.1 \
RL_ADVANTAGE_CLIP=3.0 \
RL_EVAL_MAX_RECORDS=160 \
RL_VERIFIER_BACKEND=hybrid \
RL_VERIFIER_HYBRID_ALPHA=0.8 \
RL_NUM_TRAIN_EPOCHS=1.0 \
RL_GRADIENT_ACCUMULATION_STEPS=8 \
RL_MAX_IMAGE_SIDE=768 \
RL_KEEP_RECENT_TOOL_IMAGE_MESSAGES=4 \
RL_MAX_TOTAL_IMAGES=56 \
RL_MAX_SEQ_LENGTH=6144 \
RL_KEEP_RECENT_TEXT_MESSAGES=16 \
SFT_DATALOADER_NUM_WORKERS=4 \
SFT_DATALOADER_PREFETCH_FACTOR=2 \
SFT_DATALOADER_PERSISTENT_WORKERS=1 \
RL_DATALOADER_NUM_WORKERS=4 \
RL_DATALOADER_PREFETCH_FACTOR=2 \
RL_DATALOADER_PERSISTENT_WORKERS=1 \
bash scripts/00_full_pipeline.sh
```

### 5.2 只重准备 SFT 数据

如果你已经有 `agent_train.jsonl`，只是想吃到最新的：

- schema-driven prompt
- oracle verifier branch supervision
- prepared SFT replay 逻辑

推荐直接：

```bash
conda activate qwen3-vl
cd /mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code

rm -f /mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/train_artifacts/msad_saver_agent_train.prepared_sft.jsonl

python train_saver_sft.py \
  --data /mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/benchmark_annotations/msad_saver_agent_train.jsonl \
  --data-root /mnt/shared-storage-user/mineru2-shared/zengweijun \
  --include-splits train \
  --prepare-output /mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/train_artifacts/msad_saver_agent_train.prepared_sft.jsonl \
  --prepare-only \
  --validate-prepared-data \
  --validate-materialization \
  --progress-every 25
```

### 5.3 只重建 `agent_train / oracle / prepared_sft`

如果你改了 canonical 标注，或者想从最上游整套刷新：

```bash
conda activate qwen3-vl
cd /mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code

bash scripts/01_build_oracle_and_prepare_sft.sh
```

### 5.4 只跑 SFT

如果你已经有 `prepared_sft.jsonl`，直接跑：

```bash
conda activate qwen3-vl
cd /mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code

NPROC_PER_NODE=4 \
MODEL_PATH=/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/MLLMs/qwen3-vl-8b-Instruct \
OUTPUT_DIR=/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/checkpoints/saver_sft_qwen3vl_8b_eval_ddp \
MAX_SEQ_LENGTH=6144 \
KEEP_RECENT_TEXT_MESSAGES=16 \
MAX_IMAGE_SIDE=640 \
MAX_TOTAL_IMAGES=44 \
DATALOADER_NUM_WORKERS=4 \
DATALOADER_PREFETCH_FACTOR=2 \
DATALOADER_PERSISTENT_WORKERS=1 \
GRADIENT_ACCUMULATION_STEPS=4 \
bash scripts/02_train_sft_with_rollout_eval.sh
```

### 5.5 只跑 rollout / score / summarize

如果 SFT checkpoint 已经有了，想先看看真实 rollout 状态分布：

```bash
conda activate qwen3-vl
cd /mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code

MODEL_PATH=/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/checkpoints/saver_sft_qwen3vl_8b_eval_ddp \
VERIFIER_MODEL_PATH=/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/MLLMs/Qwen3-VL-32B-Instruct \
VERIFIER_BACKEND=hybrid \
RUN_NAME=sft_rollout_eval \
bash scripts/03_batch_rollout_score_summarize.sh
```

### 5.6 只跑 RL / CEA-GRPO

如果 SFT 和 Stage 3 都已经有了，直接：

```bash
conda activate qwen3-vl
cd /mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code

NPROC_PER_NODE=4 \
MODEL_PATH=/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/checkpoints/saver_sft_qwen3vl_8b_eval_ddp \
REFERENCE_MODEL_PATH=/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/checkpoints/saver_sft_qwen3vl_8b_eval_ddp \
VERIFIER_MODEL_PATH=/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/MLLMs/Qwen3-VL-32B-Instruct \
VERIFIER_BACKEND=hybrid \
MAX_SEQ_LENGTH=6144 \
KEEP_RECENT_TEXT_MESSAGES=16 \
MAX_IMAGE_SIDE=768 \
MAX_TOTAL_IMAGES=56 \
DATALOADER_NUM_WORKERS=4 \
DATALOADER_PREFETCH_FACTOR=2 \
DATALOADER_PERSISTENT_WORKERS=1 \
NUM_ITERATIONS=4 \
ROLLOUT_COUNT=32 \
NUM_GENERATIONS=6 \
bash scripts/04_train_rl.sh
```

## 6. 评估口径：main eval vs diagnostic eval

这个边界一定要清楚。

### 6.1 主结果看什么

论文主结果和默认 epoch-end eval，应该优先看 rollout 本身和参考标注直接计算出的指标，例如：

- `existence_ap`
- `category_macro_f1`
- `temporal_miou`
- `precursor_miou`
- `alert_utility`
- `premature_alert_rate`
- `false_alert_rate`
- `evidence_f1_at_3`
- `counterfactual_type_accuracy`
- `protocol_compliance_rate`

### 6.2 诊断结果看什么

这些更适合作为训练诊断，而不是论文主表：

- `reward_summary`
- `mean_total_reward`
- `primary_status_counts`
- `alert_status_counts`
- `offline verifier` 的 reference-conditioned 结果

也就是说：

- `run_rollout_evaluation(...)` / 默认 epoch-end eval
  - 是 `main eval`
- `score_saver_rollout.py` / `summarize_saver_scores.py`
  - 是 `diagnostic eval`

## 7. 重要的跳过逻辑

`scripts/00_full_pipeline.sh` 会自动跳过已有产物：

- 如果 `AGENT_TRAIN_JSONL` 已存在，就跳过数据转换
- 如果 `PREPARED_TRAIN_JSONL` 已存在，就跳过 prepared SFT
- 如果 `SFT_OUTPUT_DIR` 下发现完整 checkpoint，就跳过 SFT
- 如果 rollout / scored / summary 都存在，就跳过 Stage 3

这很方便，但也意味着：

- 如果你更新了 `oracle_verifier_feedback` 相关逻辑
  - 需要删掉旧的 `prepared_sft.jsonl`
- 如果你希望真正重训 SFT
  - 需要删掉或改名旧 checkpoint 目录

## 8. 为什么训练看起来像“卡住”

### 8.1 `Epoch 1/2: 0/40` 很久不动

最常见不是死锁，而是：

- 进度条按 `optimizer step` 更新，不是按 micro-batch 更新
- 如果 `gradient_accumulation_steps` 大，首个 step 本来就很慢
- 如果没有 `.frame_cache`，首个 batch 还在真实视频读取 / 解码

所以看到：

```text
Epoch 1/2: 0/40
```

并不自动等于“挂了”。

### 8.2 怎么判断是真卡还是只是慢

建议同时开两个窗口：

```bash
watch -n 1 "nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader"
```

```bash
pidstat -dru -p $(pgrep -d, -f train_saver_sft.py) 1
```

如果 GPU、CPU、磁盘读都在波动，它只是还在准备和跑首个 step。

### 8.3 真想让首步快一点怎么做

优先顺序是：

1. 先预建 `.frame_cache`
2. 降低 `gradient_accumulation_steps`
3. 缩小 `max_image_side`
4. 减少 `max_total_images`

## 9. 目前最推荐的模型组合

- 主 policy warm start
  - `Qwen3-VL-8B-Instruct`
- rollout teacher / verifier
  - 优先 `Qwen3-VL-32B-Instruct`
- RL reference policy
  - 初始 SFT checkpoint
- local CEA verifier backend
  - 先用 `heuristic`
- rollout / RL verifier backend
  - 更推荐 `hybrid`

## 10. 这份 README 建议怎么使用

如果你只记 3 件事：

1. 训练前先跑 `build_frame_cache.py`
2. 想省事就直接跑 `scripts/00_full_pipeline.sh`
3. 论文主结果看 `main eval`，不要把 diagnostic verifier summary 直接当主结果

如果你只记 1 条命令：

```bash
bash scripts/00_full_pipeline.sh
```

但前提是：

- 你知道它会自动跳过已有阶段
- 你已经先把 `.frame_cache` 建好了
- 你确认当前 `prepared_sft.jsonl` 和 checkpoint 不是旧版本残留
