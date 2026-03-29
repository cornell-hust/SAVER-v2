# CEA-GRPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Counterfactual Evidence-and-Alert GRPO to SAVER with alert/evidence counterfactual groups, local group advantages, component-aware token credit routing, and RL CLI plumbing.

**Architecture:** Keep the current token-level clipped policy optimization and reference KL objective intact. Extend the rollout trace with counterfactual-friendly anchor metadata, add verifier-side alert/evidence branch scorers, convert scored rollouts into CEA-GRPO examples with global and local advantages, and route those advantage components into token-level training weights.

**Tech Stack:** Python, PyTorch, transformers, unittest, existing SAVER rollout/verifier/training stack.

---

### Task 1: Rollout Trace Support

**Files:**
- Modify: `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/saver_agent/rollout.py`
- Test: `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/tests/test_saver_agent_rollout.py`

- [ ] Step 1: Write failing tests for new trace fields and anchor tags.
- [ ] Step 2: Run targeted rollout tests and confirm they fail for missing fields.
- [ ] Step 3: Implement observed-horizon, latest-claim/alert, selected-window/evidence, and anchor-tag trace fields.
- [ ] Step 4: Run targeted rollout tests and confirm they pass.

### Task 2: Verifier Counterfactual Branch Scoring

**Files:**
- Modify: `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/saver_agent/verifier.py`
- Test: `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/tests/test_saver_agent_verifier.py`

- [ ] Step 1: Write failing tests for alert/evidence branch scoring and local group normalization.
- [ ] Step 2: Run targeted verifier tests and confirm they fail.
- [ ] Step 3: Implement alert/evidence branch scorers and group scorers on top of `run_counterfactual_verifier`.
- [ ] Step 4: Run targeted verifier tests and confirm they pass.

### Task 3: CEA-GRPO Example Construction

**Files:**
- Modify: `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/saver_agent/training_data.py`
- Test: `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/tests/test_saver_agent_training_data.py`

- [ ] Step 1: Write failing tests for anchor extraction, counterfactual groups, and example `advantage_components`.
- [ ] Step 2: Run targeted training-data tests and confirm they fail.
- [ ] Step 3: Implement alert/evidence anchor extraction, local-advantage attachment, and CEA-GRPO example building.
- [ ] Step 4: Run targeted training-data tests and confirm they pass.

### Task 4: Component-Aware Token Credit

**Files:**
- Modify: `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/saver_agent/training.py`
- Test: `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/tests/test_saver_agent_training.py`

- [ ] Step 1: Write failing tests for composing token advantages from global/alert/evidence components.
- [ ] Step 2: Run targeted training tests and confirm they fail.
- [ ] Step 3: Implement component-aware token advantage composition in the collator/training helpers.
- [ ] Step 4: Run targeted training tests and confirm they pass.

### Task 5: RL CLI Integration

**Files:**
- Modify: `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/train_saver_rl.py`
- Test: `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/tests/test_train_saver_rl.py`

- [ ] Step 1: Write failing tests for CEA-GRPO CLI/config plumbing and example builder selection.
- [ ] Step 2: Run targeted RL CLI tests and confirm they fail.
- [ ] Step 3: Implement `grpo_variant`, CEA arguments, artifact paths, and counterfactual example builder wiring.
- [ ] Step 4: Run targeted RL CLI tests and confirm they pass.

### Task 6: Verification

**Files:**
- Test: `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/tests/test_saver_agent_rollout.py`
- Test: `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/tests/test_saver_agent_verifier.py`
- Test: `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/tests/test_saver_agent_training_data.py`
- Test: `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/tests/test_saver_agent_training.py`
- Test: `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/tests/test_train_saver_rl.py`

- [ ] Step 1: Run the five targeted test files.
- [ ] Step 2: Run the full test suite with `python -m unittest discover -s tests`.
- [ ] Step 3: Inspect failures, fix regressions, and rerun until green.
