# SFT Resume Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow repeated experiment names to resume SFT safely, including the case where training for an epoch already finished but epoch-end rollout eval crashed; the next pipeline run should skip the finished training work and continue from the missing eval or resume the next unfinished training step.

**Architecture:** Add explicit SFT recovery metadata and helpers so the pipeline can distinguish between a complete SFT run, a resumable trainer checkpoint, and a missing epoch-end rollout eval. Teach `train_saver_sft.py` a small eval-only recovery mode and teach `00_full_pipeline.sh` to choose between eval-only recovery, trainer resume, or normal fresh training.

**Tech Stack:** Bash, Python, Hugging Face Trainer, existing SAVER rollout-eval callback/tests

---

### Task 1: Define recovery state and failure modes

**Files:**
- Modify: `Wmh/ideas/idea2_v2/code/train_saver_sft.py`
- Modify: `Wmh/ideas/idea2_v2/code/saver_agent/training.py`
- Test: `Wmh/ideas/idea2_v2/code/tests/test_train_saver_sft.py`

- [ ] Step 1: Write failing tests for resolving latest checkpoint epoch and missing eval epoch.
- [ ] Step 2: Run targeted unittest cases and confirm failure.
- [ ] Step 3: Add minimal helpers to resolve latest checkpoint path, infer epoch index, and detect whether rollout eval metrics exist for that epoch.
- [ ] Step 4: Re-run targeted tests and confirm pass.

### Task 2: Add eval-only recovery path to SFT entrypoint

**Files:**
- Modify: `Wmh/ideas/idea2_v2/code/train_saver_sft.py`
- Modify: `Wmh/ideas/idea2_v2/code/saver_agent/training.py`
- Test: `Wmh/ideas/idea2_v2/code/tests/test_train_saver_sft.py`

- [ ] Step 1: Write failing tests for `--resume-from-checkpoint` and `--resume-rollout-eval-only` behavior.
- [ ] Step 2: Run targeted unittest cases and confirm failure.
- [ ] Step 3: Implement eval-only recovery that loads the checkpoint model, rebuilds rollout-eval config, and writes the missing epoch metrics without re-entering trainer training.
- [ ] Step 4: Implement trainer resume path using `resume_from_checkpoint` when eval is already complete but later training is unfinished.
- [ ] Step 5: Re-run targeted tests and confirm pass.

### Task 3: Teach full_pipeline to choose recovery branch

**Files:**
- Modify: `Wmh/ideas/idea2_v2/code/scripts/00_full_pipeline.sh`
- Test: `Wmh/ideas/idea2_v2/code/tests/test_full_pipeline_script.py`

- [ ] Step 1: Write failing tests for the three pipeline branches: fresh train, eval-only recovery, trainer resume.
- [ ] Step 2: Run targeted unittest cases and confirm failure.
- [ ] Step 3: Implement bash helpers that inspect `checkpoint-*` and `rollout_eval/epoch_XXX/metrics.json`.
- [ ] Step 4: Wire Stage 3 to call `train_saver_sft.py --resume-rollout-eval-only` when training finished but eval is missing for the latest checkpoint epoch.
- [ ] Step 5: Wire Stage 3 to call `train_saver_sft.py --resume-from-checkpoint <latest>` when a resumable checkpoint exists and eval is already complete.
- [ ] Step 6: Re-run targeted tests and confirm pass.

### Task 4: Verify end-to-end compatibility

**Files:**
- Modify: `Wmh/ideas/idea2_v2/code/tests/test_full_pipeline_script.py`
- Modify: `Wmh/ideas/idea2_v2/code/tests/test_train_saver_sft.py`

- [ ] Step 1: Add regression coverage for existing skip-to-RL path.
- [ ] Step 2: Run targeted unittest suite.
- [ ] Step 3: Run `py_compile` on touched Python files.
