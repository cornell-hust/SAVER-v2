# SAVER 项目交接摘要

> 目的：这份文档用于在开启一段新对话时，快速让新的 assistant 或协作者进入当前工作上下文。
> 原则：只强调“核心 idea”和“已经落地的改动”，同时明确“当前仍未解决的问题”。
> 更新时间：`2026-04-01`

---

## 1. 一句话说明当前项目

当前 SAVER 的代码主线已经从“外接 verifier 的工具型系统”收敛为：

**一个在 turn budget 下运行的、单 `Qwen3-VL` policy 主导的主动异常理解 agent。**

它的主路径是：

`search -> alert -> self-verify -> finalize -> answer`

不是：

`search -> external verifier -> answer`

---

## 2. 当前最重要的核心 idea

### 2.1 主路径定义

当前代码中，SAVER 的核心设定是：

- 一个单一 `Qwen3-VL` policy 负责多轮决策
- policy 通过工具主动搜索证据
- policy 自己输出结构化 `verify_hypothesis` 结果
- `teacher judge` 只在训练阶段使用，不进入默认推理主路径
- legacy external verifier 只保留作 diagnostic 工具

### 2.2 推荐使用的表述

当前最贴切的论文/文档表述是：

- `turn-budgeted active anomaly understanding`

当前代码里并没有实现“按已看帧数/时长连续扣费”的严格统一 observation budget。

当前真正实现的是：

- 用 `max_turns` 控制 episode 轮数上限
- 再配合 `max_new_tokens_per_turn`、`max_total_images` 等做局部预算控制

所以更准确的是：

- `B ≈ max_turns`

而不是：

- 严格 frame-budgeted 或 duration-budgeted process

---

## 3. 已经落地的关键改动

下面这些是已经进入代码主线、并且应该视为当前基线的改动。

### 3.1 `verify_hypothesis` 改成了紧凑版 self-verification protocol

已经落地：

- policy 现在要直接输出紧凑版结构化 payload
- payload 重点字段包括：
  - `verification_mode`
  - `verification_decision`
  - `recommended_action`
  - `selected_window_ids`
  - `selected_evidence_moment_ids`
  - `claim`
  - `alert`
  - `query`
  - `sufficiency_score`
  - `necessity_score`
  - `alertability_score`
  - `counterfactual_faithfulness`
  - `rationale`

已经删除或弱化了旧版冗余字段逻辑，不再把长而杂的 legacy verifier payload 当主目标。

意义：

- 减少 verify target 长度
- 降低 parse-error 风险
- 更符合“policy 自验证”而不是“外部 verifier 补锅”

### 3.2 Oracle / SFT supervision 已经对齐到 compact self-verification

已经落地：

- `convert_to_saver_agent.py` 中 oracle verify 步使用 compact payload
- `training_data.py` 中 verify 训练目标与 tool observation 也对齐到 compact payload

意义：

- 训练目标与 runtime schema 不再断层
- 减少“模型学的是一种格式，线上要求的是另一种格式”的问题

### 3.3 Rollout eval 主路径已禁用 external verifier fallback

已经落地：

- rollout eval 的样本会附带 `disable_external_verifier_fallback=True`
- 如果 `verify_hypothesis` 没有给出合法 self-verification payload，不允许自动退回 external verifier

意义：

- 避免主评估路径偷偷变成“policy + 外接 verifier”
- 避免 eval 时额外懒加载 Qwen verifier 造成显存膨胀
- 保证评估的是“真正的 self-verifying policy”

### 3.4 `scan_timeline` 已从 evidence 语义中剥离

已经落地：

- `scan_timeline` 的 broad scan 不再当作 evidence 记账
- `seek_evidence` 才是真正的 evidence 收集动作

意义：

- evidence ledger 不再被全局扫描污染
- verifier / evidence metrics 语义更干净

### 3.5 `seek_evidence` 已支持 query-conditioned proposal runtime

已经落地：

- runtime 中可以挂 `proposal_runtime`
- `seek_evidence` 会把 query 交给 proposal encoder
- proposal encoder 再和 `feature_cache` 做相似度检索

但要注意：

- 只有同时有 `feature_cache + proposal_runtime`，query 才是真正生效的
- 如果 `proposal_runtime is None`，会退化成 uniform fallback

### 3.6 Teacher judge 路线已经落地到 SFT 和 RL

已经落地：

- `annotate_teacher_judge_sft.py` 给 prepared SFT 的 verify 样本补 teacher 标签
- teacher judge 支持：
  - `text_only`
  - `multimodal_visual`
  - `auto`
- teacher package 使用四个视角：
  - `full`
  - `keep`
  - `drop`
  - `alert_prefix`
- RL 中 teacher signal 已接到 reward / diagnostics

当前 teacher 路线的准确定义：

- 单 teacher model：`Qwen3-VL-32B-Instruct`
- 两种输入方式：文本 judge / 多模态 visual judge
- 默认 `auto`

### 3.7 SFT epoch-end rollout eval 已外部化

已经落地：

- SFT epoch 结束时不再直接在训练进程内跑 rollout eval
- 训练只保存 `epoch_resume/epoch_xxx` checkpoint
- rollout eval 由外部进程单独加载 checkpoint 执行

意义：

- 避免训练态模型和 eval 附加运行时叠显存
- 降低 OOM 风险
- 支持中断后从 epoch resume checkpoint 接着做 eval

### 3.8 预处理产物与实验产物的目录规则已经统一

已经落地：

- 所有 `json/jsonl/json summary` 统一放在：
  - `code/data_utils/`
- 所有实验输出统一放在：
  - `code/ckpt/<EXP_NAME>/`
- `scripts/00_full_pipeline.sh` 和相关脚本已按这条规则对齐

意义：

- 预处理数据和实验结果语义分离
- 便于复用数据、管理实验

### 3.9 `prepare_sft_tensor_cache.py` 已支持分片并行

已经落地：

- 支持 `--num-shards / --shard-index`
- 可以并行构建 tensor cache
- RL iteration 的 reward examples 也能走 tensor cache 加速

意义：

- 加快 SFT / RL update 阶段的多模态预处理

### 3.10 teacher annotation 脚本已支持更实用的批处理与多卡用法

已经落地：

- `annotate_teacher_judge_sft.py` 支持 `--batch-size`
- 支持 `torchrun --nproc_per_node=...`
- 支持分 shard 输出再合并
- 增加了进度可视化

### 3.11 precursor 规则已经收紧

已经落地：

- 只有 `precursor_end <= anomaly_start` 时，才保留 `precursor_interval_sec`
- 否则 `precursor_interval_sec = null`
- 非严格 precursor moment 不再继续作为真正 `precursor` 强监督，而会降级成 `trigger` 或普通 evidence
- `precursor_miou` 只在 GT 确实有 precursor 时统计
- verifier 在 GT 无 precursor 时屏蔽 precursor_support 并重归一化

意义：

- 避免模型学到“precursor 可以发生在 anomaly 之后”的错误时序

### 3.12 README、pipeline、AGENTS 已经对齐到最新代码

已经落地：

- `README.md`：保留可直接运行的命令链
- `pipeline.md`：详细讲清当前代码口径
- `AGENTS.md`：项目级协作、目录与实验规则

---

## 4. 当前默认运行口径

当前默认配置中，几个关键预算已经稳定为：

- `max_turns = 12`
- SFT eval / RL eval 常用 `max_new_tokens_per_turn = 256`
- SFT eval / RL eval 常用 `max_total_images = 24`

这比最早的 `6 turns` 评估口径更合理，因为 oracle 轨迹实际通常需要更长协议链。

---

## 5. 当前重要目录与文件

### 5.1 核心目录

- 代码根目录：
  - `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code`
- 预处理产物目录：
  - `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/data_utils`
- 实验输出目录：
  - `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/ckpt`

### 5.2 当前最重要的说明文档

- [AGENTS.md](/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/AGENTS.md)
- [README.md](/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/README.md)
- [pipeline.md](/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/pipeline.md)

### 5.3 当前最关键的实现文件

- `saver_agent/self_verification.py`
- `saver_agent/tools.py`
- `saver_agent/evaluation.py`
- `saver_agent/training_data.py`
- `saver_agent/teacher_judge.py`
- `saver_agent/reward.py`
- `convert_to_saver_agent.py`
- `train_saver_sft.py`
- `train_saver_rl.py`

---

## 6. 当前仍未解决的核心问题

尽管主线语义已经基本理顺，但**模型效果仍然不好**，这是当前最大的现实问题。

已观察到的主要现象：

- `temporal_miou` 仍然偏低，甚至接近 0
- `precursor_miou` 偏低
- `category_macro_f1` 很差
- `protocol_compliance_rate` 仍然不高
- 训练和 eval 仍然容易受到多轮上下文长度、视觉输入规模的影响

也就是说：

- 当前已经解决了不少“协议/实现错位”的硬伤
- 但还没有解决“性能真正变强”的问题

应把当前阶段理解为：

- **主路径语义基本对齐**
- **性能优化仍未完成**

---

## 7. 当前最值得继续推进的方向

这些是下一阶段最有价值的方向，但**还没有完整落地**。

### 7.1 统一视觉预算

当前我们已经有：

- `max_turns`
- `max_new_tokens_per_turn`
- `max_total_images`

但还没有：

- 整个 episode 级别的统一视觉预算

最值得借鉴 TimeSearch-R 的点不是“更少 turns”，而是：

- `episode-level visual budget`

### 7.2 动态二阶段帧压缩

当前我们已经有：

- frame cache
- feature cache
- proposal runtime

但还缺：

- 在 rollout / eval 时，对已经取出的候选帧再做一次预算化压缩

更准确地说，未来最值得做的是：

- 二次抽帧
- 动态 resize

而不是只靠继续压 `max_total_images`

### 7.3 infer-time 文本预算

当前训练侧已经有：

- `max_seq_length`
- `keep_recent_tool_image_messages`
- `keep_recent_text_messages`

但在线 rollout 侧还缺更明确的：

- `max_prompt_length` 风格控制

### 7.4 继续提升 verify 成功率与协议完成率

虽然 compact self-verification 已落地，但还需要继续关注：

- verify JSON 是否稳定合法
- verify turn 是否过于保守或过于激进
- finalize 触发是否合理

---

## 8. 新对话最应该先知道的事实

如果开一段新对话，希望新的 assistant 快速上手，最重要的是先告诉它下面这些：

1. 当前代码主路径是 `single self-verifying policy`，不是默认外接 verifier
2. teacher judge 只在训练期使用
3. rollout eval 主路径已经禁用 external verifier fallback
4. `seek_evidence` 只有挂了 `proposal_runtime` 才是 query-conditioned
5. 所有 `json/jsonl` 预处理产物统一放在 `code/data_utils/`
6. 所有实验输出统一放在 `code/ckpt/<EXP_NAME>/`
7. SFT epoch-end eval 已外部化，使用 `epoch_resume` checkpoint
8. 当前最大的未解问题是“效果仍差”，不是“主线路径还没理顺”

---

## 9. 建议新对话开头直接贴的摘要

下面这段文字可以直接复制到新对话中：

```text
请以 /mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code 为工作根目录，并先阅读：
1. code/docs/project_handoff.md
2. code/AGENTS.md
3. code/pipeline.md
4. code/README.md

当前 SAVER 主路径已经对齐为：
- single Qwen3-VL policy
- search -> alert -> self-verify -> finalize -> answer
- verify_hypothesis 使用 compact policy self-verification payload
- teacher judge 仅用于训练
- rollout eval 主路径禁用 external verifier fallback
- proposal retrieval 只有在 feature_cache + proposal_runtime 同时存在时才真正 query-conditioned

当前目录规则：
- 所有 json/jsonl 预处理产物在 code/data_utils/
- 所有实验输出在 code/ckpt/<EXP_NAME>/

当前最重要的现实问题不是主线路径定义，而是性能仍差：
- temporal_miou / precursor_miou / category_macro_f1 / protocol_compliance_rate 仍然偏低

请先以“最新代码为准”理解项目，不要回退到旧论文或旧 verifier 主路径口径。
```

---

## 10. 最后一句判断

如果要用一句话概括当前阶段：

**我们已经把 SAVER 的主线语义从“混合 verifier 系统”整理成了“单 policy 自验证系统”，但性能优化还远未结束。**

