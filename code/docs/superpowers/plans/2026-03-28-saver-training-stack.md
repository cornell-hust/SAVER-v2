# SAVER Training Stack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a usable SAVER training stack with oracle-SFT warm start, reward-guided RL v1, and stronger policy guidance for timely finalize/answer decisions.

**Architecture:** Reuse the existing SAVER environment, adapter, rollout, verifier, and reward stack. Build a shared training-data layer that converts `oracle_sft` trajectories and scored rollouts into stepwise chat-supervision examples, then expose two CLI entrypoints: `train_saver_sft.py` for warm start and `train_saver_rl.py` for rollout-score-update loops.

**Tech Stack:** Python, PyTorch, Transformers/Qwen3-VL, existing SAVER dataset/environment/verifier code.

---

### Task 1: Shared Training Data Layer

**Files:**
- Create: `saver_agent/training_data.py`
- Modify: `tests/test_convert_to_saver_agent.py`
- Modify: `tests/test_saver_agent_rollout.py`

- [ ] Write failing tests for oracle-SFT example generation and final-answer synthesis after `finalize_case`.
- [ ] Run focused tests and confirm they fail for the intended missing behavior.
- [ ] Implement stepwise oracle transcript replay into chat-supervision examples.
- [ ] Implement rollout-to-weighted-example conversion for later RL updates.
- [ ] Re-run focused tests and confirm they pass.

### Task 2: Policy Finalize/Answer Guidance

**Files:**
- Modify: `saver_agent/adapter.py`
- Modify: `tests/test_saver_agent_rollout.py`

- [ ] Write failing tests for verifier/finalize tool follow-up prompts.
- [ ] Run focused tests and confirm they fail.
- [ ] Add tool-specific prompt guidance for `verify_hypothesis`, `finalize_case`, and `emit_alert`.
- [ ] Re-run focused tests and confirm they pass.

### Task 3: SFT Training Entry

**Files:**
- Create: `train_saver_sft.py`
- Create: `tests/test_train_saver_sft.py`
- Modify: `saver_agent/__init__.py` if exports are needed

- [ ] Write failing CLI/unit tests for SFT dataset preparation and config parsing.
- [ ] Run focused tests and confirm they fail.
- [ ] Implement SFT script with oracle trajectory conversion, processor/model loading, optional LoRA, and Trainer-based updates.
- [ ] Re-run focused tests and confirm they pass.

### Task 4: RL v1 Training Entry

**Files:**
- Create: `train_saver_rl.py`
- Create: `tests/test_train_saver_rl.py`
- Modify: `saver_agent/offline_scoring.py` only if helper reuse is needed

- [ ] Write failing tests for RL config parsing and reward-weighted dataset building.
- [ ] Run focused tests and confirm they fail.
- [ ] Implement rollout collection, scoring, reward filtering/weighting, and iterative update orchestration.
- [ ] Re-run focused tests and confirm they pass.

### Task 5: Verification

**Files:**
- Modify: none unless test fixes are required

- [ ] Run all focused test files.
- [ ] Run the full CPU unit-test suite.
- [ ] Summarize runnable GPU commands for SFT and RL.
