# SAVER Multi-GPU Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make SAVER training, rollout, scoring, and summarization practical on 4 GPUs by adding clear distributed/sharding support and visible runtime progress logs.

**Architecture:** Keep `train_saver_sft.py` and `train_saver_rl.py` compatible with `torchrun`/DDP through Hugging Face `Trainer`, while inference-style scripts use explicit dataset sharding so one process can be bound to one GPU. Centralize shard/distributed/progress helpers in a small runtime utility module and reuse them across entrypoints.

**Tech Stack:** Python, Transformers Trainer, PyTorch distributed env vars, existing SAVER rollout/scoring stack.

---

### Task 1: Add runtime helper tests

**Files:**
- Create: `tests/test_saver_agent_runtime.py`
- Modify: `tests/test_score_saver_rollout.py`
- Modify: `tests/test_summarize_saver_scores.py`

- [ ] **Step 1: Write failing tests for sharding helpers**
- [ ] **Step 2: Run targeted tests to confirm they fail**
- [ ] **Step 3: Add minimal helper expectations for env-based rank/world-size detection and shard slicing**
- [ ] **Step 4: Re-run targeted tests**

### Task 2: Implement shared runtime helpers

**Files:**
- Create: `saver_agent/runtime.py`

- [ ] **Step 1: Add helper API for distributed env detection**
- [ ] **Step 2: Add helper API for dataset/list sharding**
- [ ] **Step 3: Add helper API for progress logging and main-process-only printing**
- [ ] **Step 4: Run targeted tests**

### Task 3: Wire sharding and logging into rollout/scoring/summarization

**Files:**
- Modify: `batch_run_saver_rollout.py`
- Modify: `score_saver_rollout.py`
- Modify: `summarize_saver_scores.py`
- Modify: `saver_agent/offline_scoring.py`

- [ ] **Step 1: Add CLI args for shard index / num shards / progress interval**
- [ ] **Step 2: Apply sharding before per-record processing**
- [ ] **Step 3: Emit progress logs during long loops and on startup**
- [ ] **Step 4: Re-run rollout/scoring/summarize tests**

### Task 4: Improve training entrypoint logging and distributed guidance

**Files:**
- Modify: `train_saver_sft.py`
- Modify: `train_saver_rl.py`
- Modify: `README.md`

- [ ] **Step 1: Add startup/progress logging with main-process guards**
- [ ] **Step 2: Add explicit `torchrun` usage guidance and 4-GPU examples**
- [ ] **Step 3: Explain why terminal output used to look silent**
- [ ] **Step 4: Run training-related tests**

### Task 5: Verification

**Files:**
- Modify only if needed during fixes

- [ ] **Step 1: Run focused unit tests**
- [ ] **Step 2: Run full `python -m unittest discover -s tests`**
- [ ] **Step 3: Summarize exact multi-GPU support matrix and recommended commands**
