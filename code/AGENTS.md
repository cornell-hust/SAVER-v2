# SAVER 项目协作指南

## 目的

本文件定义了本仓库内人类与编码 agent 的协作规则。

作用范围：

- 本指南对应的仓库根目录：`/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code`
- 本指南是项目管理与日常执行层的约束文档
- 当论文、README、旧笔记与当前代码冲突时，以最新代码为准

最重要的两个配套文档是：

- [README.md](/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/README.md)：可直接运行的命令
- [pipeline.md](/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/pipeline.md)：与当前代码对齐的训练与推理流程说明

---

## 项目定位

本仓库实现的是一套 SAVER 风格的主动视频异常理解 agent。

当前主路径定义：

- 单个 `Qwen3-VL` policy
- 多轮工具调用 rollout
- `verify_hypothesis` 使用 `policy self-verification`
- `teacher judge` 仅用于训练
- legacy external verifier 仅保留为 diagnostic 路径
- 只有挂载 `proposal_runtime` 时，proposal 检索才是真正的 query-conditioned

文档与论文中推荐使用的表述：

- `turn-budgeted active anomaly understanding`

避免继续使用以下过时表述：

- 默认外接 verifier 的主路径
- 文本 judge + 视觉 judge 的双 teacher 主线
- 严格 frame-budgeted 的 observation process

---

## 仓库结构

关键目录：

- `saver_agent/`：核心 runtime、rollout、tools、reward、training utilities
- `data_utils/`：所有预处理得到的 `json/jsonl/json summary` 产物
- `ckpt/`：实验输出、checkpoints、logs、rollouts
- `scripts/`：便捷 shell 脚本
- `tests/`：单元测试

重要的顶层脚本：

- `convert_to_saver_agent.py`
- `build_frame_cache.py`
- `build_feature_cache.py`
- `annotate_teacher_judge_sft.py`
- `prepare_sft_tensor_cache.py`
- `train_saver_sft.py`
- `train_saver_rl.py`
- `batch_run_saver_rollout.py`
- `score_saver_rollout.py`
- `summarize_saver_scores.py`

---

## 产物落盘规则

以下规则为强约束。

### 预处理产物

所有预处理输出必须放在：

- `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/data_utils`

包括但不限于：

- canonical passthrough 导出结果
- `agent_train` JSONL
- `oracle_sft` JSONL
- `prepared_sft` JSONL
- `prepared_sft.teacher` JSONL
- frame-cache summary JSON
- feature-cache summary JSON
- 由 prepared JSONL 派生的 tensor cache 目录

不要把这些产物放进实验 checkpoint 目录。

### 实验产物

所有训练、rollout、打分和日志输出必须放在：

- `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/ckpt/<EXP_NAME>/`

典型结构：

- `ckpt/<EXP_NAME>/checkpoints/...`
- `ckpt/<EXP_NAME>/rollouts/...`
- `ckpt/<EXP_NAME>/logs/...`

### 视频旁路缓存

重型、可复用的逐视频缓存保留在视频源旁边：

- `*.frame_cache`
- `*.feature_cache`

---

## 主路径技术规则

除非有明确的重设计，否则以下规则必须保持一致。

### Verification

主路径要求：

- `verify_hypothesis` 必须使用紧凑版 `policy self-verification` payload
- policy 负责自己产出结构化验证结果
- runtime 负责把该 payload 写入 state，并反馈给后续轮次

仅 diagnostic 路径允许：

- heuristic verifier
- qwen self verifier
- hybrid verifier

绝不能让 diagnostic fallback 悄悄重新进入主 rollout 或主 eval。

### Teacher judge

Teacher judge 仅用于训练。

主要职责：

- 给 SFT 的 verify 样本补标注
- 校准 verify-step supervision
- 在 RL 中提供辅助 teacher-alignment signal

Teacher judge 不是线上最终推理的决策者。

### Proposal retrieval

只有同时具备以下两项时，`seek_evidence` 才是真正的 query-conditioned：

- `feature_cache`
- `proposal_runtime`

如果缺少 `proposal_runtime`，即使 query 文本存在，也会退化成 uniform fallback。

### Evidence 语义

`scan_timeline` 不是 evidence。

规则如下：

- broad scan 用于全局覆盖与定位
- `seek_evidence` 才用于收集证据
- evidence ledger 的语义不能被全局扫描污染

### Category 语义

所有 policy 可见的 anomaly category 都必须使用 canonical label enum。

不要让自由文本类别重新进入以下位置：

- tool schema
- oracle targets
- verify payloads
- finalize payloads

---

## 标准流程

预期的数据与训练流程为：

1. `convert_to_saver_agent.py`
2. `train_saver_sft.py --prepare-only`
3. `annotate_teacher_judge_sft.py`
4. `build_frame_cache.py`
5. `build_feature_cache.py`
6. `prepare_sft_tensor_cache.py`
7. `train_saver_sft.py`
8. 基于保存的 epoch checkpoint 做外部 rollout eval
9. `train_saver_rl.py`

解释：

- 预处理产物应尽量一次构建、反复复用
- SFT 的 epoch-end eval 已外部化
- RL 依赖 rollout traces 和 reward-weighted updates

---

## 重建规则

在重跑昂贵任务前，先参考下表。

| 改动内容 | 必须重建 |
| --- | --- |
| 仅训练超参数变化 | 默认无需重建 `data_utils/` 中产物 |
| oracle trajectory logic 变化 | `oracle_sft -> prepared_sft -> prepared_sft.teacher -> tensor_cache` |
| `verify_hypothesis` schema 或 payload 形状变化 | `oracle_sft -> prepared_sft -> prepared_sft.teacher -> tensor_cache` |
| teacher judge 标注逻辑变化 | `prepared_sft.teacher -> tensor_cache` |
| tokenizer / processor / 图像预算 / 文本预算变化 | `tensor_cache` |
| frame extraction 设置变化 | `frame_cache`，必要时继续重建下游 `feature_cache` |
| proposal encoder 或特征提取设置变化 | `feature_cache` |
| 仅 rollout-time proposal 挂载方式变化 | 无需重建 JSONL，但必须已有 `feature_cache` |

经验法则：

- 如果变的是语义目标，就重建 JSONL。
- 如果变的是样本物化方式，就重建 tensor cache。

---

## 实验命名与恢复规则

推荐实验布局：

- 输入 `EXP_NAME=exp1`
- 输出根目录为：
  `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/ckpt/exp1`

命名要求：

- 一个实验名只对应一套有意义的设置组合
- 不要用旧实验名复用不兼容的新实验
- logs、summaries、rollout outputs 应保持在同一个实验根目录下

SFT 恢复规则：

- epoch-end training 应保存 `epoch_resume/epoch_xxx`
- 外部 rollout eval 应消费这些 checkpoints
- 如果训练在某个 epoch 结束后、eval 完成前中断，重启时应从保存的 epoch checkpoint 继续 eval，而不是重复已经完成的训练

---

## 日志规则

每一次重要运行都应在实验目录下留下可读日志。

必须保留的日志包括：

- training config snapshot
- final result snapshot
- rollout eval metrics
- RL iteration metrics
- 如果使用 shell 脚本，还应保留 pipeline shell logs

推荐位置：

- `ckpt/<EXP_NAME>/logs/`
- 若 checkpoint 输出目录已经支持独立 `logs/`，则保留其模型子目录日志

不要把重要诊断信息只留在终端滚动输出中。

---

## 文档同步规则

只要代码行为变化会影响 pipeline 语义，就必须在同一工作流中同步更新文档。

以下变化需要更新 `pipeline.md`：

- 主路径推理流程
- self-verification 行为
- teacher judge 角色
- proposal runtime 语义
- eval / rollout 语义
- budget 语义

以下变化需要更新 `README.md`：

- 可运行命令
- 必要环境变量
- 输出路径
- 预处理顺序

以下变化需要更新本 `AGENTS.md`：

- 项目级约定
- 目录规则
- 实验管理策略
- source-of-truth 决策

如果代码和文档冲突，必须立刻修正文档。

---

## 性能规则

默认性能优化哲学：

- 先保证 anomaly evidence flow 的正确性
- 再通过预算化压缩降低显存与运行时间

推荐优化顺序：

1. 降低每轮生成长度
2. 限制传给 policy 的总图片数
3. 增加预算化的二阶段帧选择
4. 在全局视觉预算下增加动态 resize
5. 最后再考虑更激进的裁剪

避免以下取巧方式：

- 为了掩盖 policy verify 失败而把 external verifier 恢复到主路径
- 把 `max_turns` 缩到协议根本跑不完
- 在不考虑证据覆盖率的前提下盲目减少图片数量

---

## 当前高置信项目假设

除非有明确改动，这些假设应被继续保持：

- `verify_hypothesis` 的 compact payload 是正确的 policy-facing protocol
- normal 与 anomaly 样本共用同一套 self-verification 接口
- teacher model 默认是 `Qwen3-VL-32B-Instruct`
- teacher input mode 默认是 `auto`
- proposal encoder 默认是 SigLIP
- SFT 与 RL 都应能从 tensor cache 中获益
- rollout eval 主路径应继续保持 reference-free，且禁用 external verifier fallback

---

## 效果差时优先检查什么

在提出新想法前，优先按下面顺序检查：

1. `verify_hypothesis` 的 JSON 是否合法、是否与 schema 对齐？
2. rollout 是否真的挂上了 `proposal_runtime`？
3. 主 eval 是否意外重新打开了 diagnostic external verifier fallback？
4. `prepared_sft.teacher` 和 tensor cache 是否与当前配置一致？
5. `max_turns`、`max_new_tokens_per_turn`、图像预算是否过紧，导致协议跑不完？
6. oracle supervision 是否仍与当前 compact self-verification 语义保持一致？

只有排除以上问题后，才应该继续深入到 reward 或更大架构改动。

---

## 面向未来 agent 的编辑规则

在修改本仓库时：

- 不要重新引入过时的 verifier-main-path 逻辑
- 不要把预处理 JSONL 产物移出 `data_utils/`
- 不要把实验输出写到 `ckpt/<EXP_NAME>/` 之外
- 不要在不更新文档的情况下修改 pipeline 语义
- 不要假设旧论文表述仍然正确

拿不准时：

- 先相信当前代码
- 用测试验证
- 再同步更新文档

## 如果代码发生修改，尤其是超参数的修改，一定要检查脚本00_full_pipeline.sh 和 README.md 中的指令是否也得到了同步更新
