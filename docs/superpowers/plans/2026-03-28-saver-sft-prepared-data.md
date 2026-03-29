# SAVER SFT Prepared Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an offline-preparable SFT example format that avoids storing inline image tensors and lets training lazily materialize frames.

**Architecture:** Keep oracle rollout expansion logic, but add a serialized message mode that replaces inline images with frame references. The training dataset resolves those references on demand from frame caches or source videos, and `train_saver_sft.py` gains CLI support for writing and reusing prepared JSONL files.

**Tech Stack:** Python, PyTorch, decord, unittest

---

### Task 1: Cover the New Behavior with Tests

**Files:**
- Modify: `code/tests/test_saver_agent_training_data.py`
- Modify: `code/tests/test_saver_agent_training.py`
- Modify: `code/tests/test_train_saver_sft.py`

- [x] **Step 1: Write failing tests**
- [x] **Step 2: Run focused tests and verify they fail for missing serialization/materialization support**

### Task 2: Serialize Oracle SFT Messages Without Inline Tensors

**Files:**
- Modify: `code/saver_agent/dataset.py`
- Modify: `code/saver_agent/tools.py`
- Modify: `code/saver_agent/training_data.py`

- [ ] **Step 1: Attach frame index metadata to preview/tool image content**
- [ ] **Step 2: Add serialized-message mode for oracle SFT example building**
- [ ] **Step 3: Keep existing in-memory example behavior intact for old call sites**

### Task 3: Lazily Materialize Frame References During Training

**Files:**
- Modify: `code/saver_agent/training.py`

- [ ] **Step 1: Teach the training dataset to resolve `image_ref` entries on access**
- [ ] **Step 2: Support frame-cache-backed and video-backed resolution paths**
- [ ] **Step 3: Keep collator behavior unchanged once messages are materialized**

### Task 4: Add CLI Support for Preparing and Reusing Prepared Data

**Files:**
- Modify: `code/train_saver_sft.py`

- [ ] **Step 1: Add `--prepared-data` and `--prepare-output` arguments**
- [ ] **Step 2: Build lightweight prepared examples from raw data when needed**
- [ ] **Step 3: Save prepared JSONL for reuse and train directly from prepared data**

### Task 5: Verify End-to-End

**Files:**
- Modify: `code/tests/test_saver_agent_training_data.py`
- Modify: `code/tests/test_saver_agent_training.py`
- Modify: `code/tests/test_train_saver_sft.py`

- [ ] **Step 1: Re-run focused unittests**
- [ ] **Step 2: Run a small `train_saver_sft.py` dry-run with `--prepare-output`**
- [ ] **Step 3: Confirm prepared JSONL is JSON-safe and training can load it**
