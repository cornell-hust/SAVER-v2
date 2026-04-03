# SAVER Simplified Data Pipeline Design

## Goal

将当前多阶段、多中间 JSONL 的 SAVER 预处理链路，收敛成一套更接近 TimeSearch-R 风格的稳定数据入口：

- `data_utils/msad_saver_runtime_train.jsonl`
- `data_utils/msad_saver_runtime_test.jsonl`
- `data_utils/msad_saver_sft_train.jsonl`
- `data_utils/msad_saver_sft_train.teacher.jsonl`

其中：

- `runtime_*` 是 episode-level 运行时数据，供 rollout eval 和 RL 直接使用
- `sft_train` 是 step-level 最终 SFT 样本
- `sft_train.teacher` 是 teacher 标注后的最终 SFT 样本

## Context

当前代码已经有两块稳定能力，不需要重写：

1. [convert_to_saver_agent.py](/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/convert_to_saver_agent.py)
   的 `oracle_sft` 模式已经能输出接近最终 runtime episode 的结构。
2. [train_saver_sft.py](/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/train_saver_sft.py)
   的 `build_prepared_sft_examples_from_jsonl(...)` 已经能把 episode JSONL 转成最终 step-level SFT 样本。

因此，最稳的方案不是重写训练数据语义，而是新增一个统一 builder，把这两段现有稳定逻辑接起来。

## Chosen Approach

新增 [build_saver_data.py](/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/build_saver_data.py) 作为默认主入口。

它负责：

1. canonical JSONL -> `runtime_train.jsonl`
2. canonical JSONL -> `runtime_test.jsonl`
3. `runtime_train.jsonl` -> `sft_train.jsonl`
4. 可选：`sft_train.jsonl` -> `sft_train.teacher.jsonl`

## Data Contracts

### Runtime JSONL

每条记录对应一个视频 episode，保留 rollout / eval / RL 所需的全部字段，包括：

- `video_id`, `video_path`, `split`, `video_meta`
- `scene`, `key_objects`
- `label`, `temporal`, `evidence`, `counterfactual`
- `proposal_supervision`
- `agent_task`, `structured_target`, `tool_io`
- `oracle_sft`

这里直接复用 `convert_to_saver_agent.py --mode oracle_sft` 的记录形状。

### SFT JSONL

每条记录对应一个训练 step，保留：

- `messages`
- `target_response`
- `target_action`
- `split`
- `video_id`
- 训练时需要的 image refs / multimodal payload

这里直接复用 `build_prepared_sft_examples_from_jsonl(...)` 的输出形状。

## Script Changes

默认脚本改为使用新文件名：

- [scripts/00_full_pipeline.sh](/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/scripts/00_full_pipeline.sh)
- [scripts/02_train_sft_with_rollout_eval.sh](/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/scripts/02_train_sft_with_rollout_eval.sh)
- [scripts/03_batch_rollout_score_summarize.sh](/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/scripts/03_batch_rollout_score_summarize.sh)
- [scripts/04_train_rl.sh](/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/scripts/04_train_rl.sh)

新的默认关系：

- SFT 训练读 `msad_saver_sft_train(.teacher).jsonl`
- rollout eval 读 `msad_saver_runtime_test.jsonl`
- RL 训练读 `msad_saver_runtime_train.jsonl`

## Cache Strategy

frame cache / feature cache 仍然沿用现有视频旁路缓存，不改缓存格式。

但是因为 episode JSONL 从一个文件拆成了 train/test 两个 runtime 文件，full pipeline 的 cache 构建阶段需要显式对 train 和 test 两份 runtime 数据分别调用构建脚本，避免重新引入 `runtime_all.jsonl` 这类新的中间产物。

## Backward Tolerance

训练核心代码不强制移除旧路径兼容性，但新默认入口、脚本和测试全部切到新命名。

这意味着：

- 旧文件如果还在，手动指定时仍可工作
- 新主路径不再依赖 `agent_train.jsonl` / `oracle_sft.jsonl` / `prepared_sft.jsonl`

## Validation

本次改动需要覆盖三类测试：

1. 新增 `build_saver_data.py` 的单元测试
2. 更新 full pipeline 脚本测试，断言新文件名与新阶段顺序
3. 更新 experiment-name 脚本测试，断言 `EXP_NAME` 下仍然只把 JSONL 放在 `data_utils`
