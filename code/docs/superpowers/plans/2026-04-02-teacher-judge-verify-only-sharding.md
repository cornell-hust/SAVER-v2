# Teacher Judge Verify-Only Sharding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Balance multi-GPU teacher-judge work by redistributing only `verify_hypothesis` samples across shards while leaving non-candidates on their original strided shards.

**Architecture:** Add a teacher-judge-specific shard index planner inside `annotate_teacher_judge_sft.py`, use it for local row selection and shard merge validation, and keep `saver_agent/runtime.py` unchanged so other scripts retain their current partition semantics.

**Tech Stack:** Python, unittest, existing SAVER teacher-judge helpers

---

### Task 1: Lock the desired shard behavior in tests

**Files:**
- Modify: `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/tests/test_annotate_teacher_judge_sft.py`

- [ ] Step 1: Add a failing test for verify-only shard redistribution.
- [ ] Step 2: Add a failing test for merge order restoration under a custom shard mapping.
- [ ] Step 3: Run the targeted annotate-teacher-judge tests and verify the new assertions fail first.

### Task 2: Implement verify-only shard planning

**Files:**
- Modify: `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/annotate_teacher_judge_sft.py`

- [ ] Step 1: Add helper(s) that compute per-shard source indices with verify-only redistribution.
- [ ] Step 2: Switch local row selection to those computed indices.
- [ ] Step 3: Update shard wait/merge validation to use the same mapping.
- [ ] Step 4: Keep final merged output order identical to the input JSONL order.

### Task 3: Regression verification

**Files:**
- Test: `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/tests/test_annotate_teacher_judge_sft.py`
- Test: `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/tests/test_saver_agent_runtime.py`

- [ ] Step 1: Re-run the targeted annotate-teacher-judge tests.
- [ ] Step 2: Re-run runtime shard tests to confirm generic strided partitioning stays unchanged.
- [ ] Step 3: Report exact commands and observed pass/fail results.
