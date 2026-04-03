# SAVER Simplified Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current multi-artifact preprocessing path with a single builder that emits runtime train/test JSONLs and final SFT train JSONLs, then switch the default scripts to those artifacts.

**Architecture:** Reuse the existing canonical-to-runtime conversion in `convert_to_saver_agent.py` and the existing runtime-to-final-SFT materialization in `train_saver_sft.py`. Add one orchestration script, then repoint shell scripts and tests to the new filenames while keeping training/runtime internals stable.

**Tech Stack:** Python, bash, unittest, existing SAVER data/build utilities

---

### Task 1: Lock the new artifact contract in tests

**Files:**
- Create: `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/tests/test_build_saver_data.py`
- Modify: `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/tests/test_full_pipeline_script.py`
- Modify: `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/tests/test_experiment_name_scripts.py`

- [ ] Step 1: Add a failing unit test for canonical -> runtime_train/test -> sft_train.
- [ ] Step 2: Run the new test and verify it fails because `build_saver_data.py` does not exist yet.
- [ ] Step 3: Update existing script tests to expect `runtime_train/test` and `sft_train(.teacher)` filenames.
- [ ] Step 4: Run the targeted script tests and verify they fail on the old defaults.

### Task 2: Implement the unified data builder

**Files:**
- Create: `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/build_saver_data.py`

- [ ] Step 1: Implement JSONL loading and canonical split filtering using `convert_to_saver_agent.iter_jsonl(...)`.
- [ ] Step 2: Materialize runtime train/test outputs by calling `convert_record(..., mode="oracle_sft")`.
- [ ] Step 3: Materialize `sft_train.jsonl` by calling `build_prepared_sft_examples_from_jsonl(...)` on `runtime_train.jsonl`.
- [ ] Step 4: Add optional inline teacher-judge annotation that reuses existing teacher utilities when requested.
- [ ] Step 5: Run the new builder unit tests and make them pass.

### Task 3: Switch the default shell scripts to the simplified path

**Files:**
- Modify: `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/scripts/00_full_pipeline.sh`
- Modify: `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/scripts/02_train_sft_with_rollout_eval.sh`
- Modify: `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/scripts/03_batch_rollout_score_summarize.sh`
- Modify: `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/scripts/04_train_rl.sh`

- [ ] Step 1: Replace old default filenames with the new runtime/sft filenames.
- [ ] Step 2: Make Stage 1 of `00_full_pipeline.sh` call `build_saver_data.py` instead of the old multi-step JSONL generation path.
- [ ] Step 3: Update Stage 2 cache building so train/test runtime files are both covered without introducing a new combined JSONL.
- [ ] Step 4: Run the shell-script tests and verify the new defaults are wired through.

### Task 4: Regression verification

**Files:**
- Test: `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/tests/test_build_saver_data.py`
- Test: `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/tests/test_full_pipeline_script.py`
- Test: `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/tests/test_experiment_name_scripts.py`

- [ ] Step 1: Run the targeted unittest set for the new builder and scripts.
- [ ] Step 2: Fix any failures without reintroducing old default file names.
- [ ] Step 3: Report the exact commands run and the final pass/fail status.
