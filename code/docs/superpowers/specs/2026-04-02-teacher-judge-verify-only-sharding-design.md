# Teacher Judge Verify-Only Sharding Design

## Goal

修正 `annotate_teacher_judge_sft.py` 在 `torchrun` 多卡下的 teacher-judge 负载不均问题，但只做最小行为改动：

- `verify_hypothesis` 样本单独按候选顺序均分到各个 shard
- 非 `verify_hypothesis` 样本继续保持原本基于原始行号的 shard 归属
- 最终 merged 输出仍然恢复原始 JSONL 顺序

## Context

当前实现直接对整份输入做 `index % num_shards` 静态分片。对这条脚本来说，真正重的工作只发生在 `verify_hypothesis` 样本上，因此即使每个 shard 的总样本数接近，也会因为 verify 候选数不均导致尾部空转。

## Chosen Approach

在 `annotate_teacher_judge_sft.py` 内新增 teacher-judge 专用分片映射：

1. 扫描整份输入
2. 对 `verify_hypothesis` 候选单独编号并 round-robin 到各 shard
3. 对非候选样本继续使用原始 `index % num_shards`
4. 主进程 merge shard 输出时使用同一份映射恢复原始顺序

## Non-Goals

- 不引入动态调度 / work stealing
- 不修改 `saver_agent/runtime.py` 里的通用 `shard_sequence(...)` 语义
- 不改变 teacher judge 的 batch 推理逻辑

## Validation

需要覆盖：

1. verify-only 分片 helper 的行为测试
2. 自定义 shard 映射下的 merge 顺序恢复测试
3. 原有 `annotate_teacher_judge_sft.py` 相关回归测试不被破坏
