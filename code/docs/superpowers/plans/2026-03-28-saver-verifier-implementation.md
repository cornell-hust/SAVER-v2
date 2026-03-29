# SAVER Counterfactual Verifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a rollout-native SAVER counterfactual verifier, reward utilities, and tool/trace integration matching the approved spec.

**Architecture:** Add a deterministic verifier core that constructs `full/keep/drop/alert_prefix` views from rollout state and derives structured verdicts. Wire the verifier into `verify_hypothesis`, rollout traces, and a reward helper so the existing agent stack can consume verification outputs without free-form parsing.

**Tech Stack:** Python, unittest, existing `saver_agent` package, JSON-like structured tool I/O

---

### Task 1: Add Verifier Tests

**Files:**
- Create: `tests/test_saver_agent_verifier.py`
- Test: `tests/test_saver_agent_verifier.py`

- [ ] **Step 1: Write the failing test**

Add tests for:
- complete verdict when selected evidence includes decisive support
- incomplete verdict when selected evidence is missing trigger support
- redundant verdict when drop view remains strong
- alert status detection for justified vs premature alerts

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest /mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/tests/test_saver_agent_verifier.py -v`
Expected: FAIL because `saver_agent.verifier` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Create `saver_agent/verifier.py` with:
- state view extraction helpers
- heuristic per-view support scoring
- reducer for `primary_status` and `alert_status`
- public `run_counterfactual_verifier(...)`

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest /mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/tests/test_saver_agent_verifier.py -v`
Expected: PASS

### Task 2: Add Reward Tests

**Files:**
- Create: `tests/test_saver_agent_reward.py`
- Test: `tests/test_saver_agent_reward.py`

- [ ] **Step 1: Write the failing test**

Add tests for:
- reward decomposition from verifier-rich rollout traces
- complete verdict scoring higher than incomplete or misaligned
- premature alert penalty

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest /mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/tests/test_saver_agent_reward.py -v`
Expected: FAIL because `saver_agent.reward` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Create `saver_agent/reward.py` with:
- reward weight defaults
- verdict-to-score mapping
- `score_rollout_trace(...)`

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest /mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/tests/test_saver_agent_reward.py -v`
Expected: PASS

### Task 3: Integrate Tool, State, and Trace

**Files:**
- Modify: `saver_agent/schema.py`
- Modify: `saver_agent/tool_registry.py`
- Modify: `saver_agent/tools.py`
- Modify: `saver_agent/rollout.py`
- Test: `tests/test_saver_agent_tools.py`
- Test: `tests/test_saver_agent_rollout.py`

- [ ] **Step 1: Write the failing tests**

Extend existing tests to require:
- stable visited `window_id`
- upgraded `verify_hypothesis` verdict payload
- rollout trace carrying verifier status fields

- [ ] **Step 2: Run tests to verify they fail**

Run:
- `python -m unittest /mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/tests/test_saver_agent_tools.py -v`
- `python -m unittest /mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/tests/test_saver_agent_rollout.py -v`

Expected: FAIL on missing verifier fields and identifiers.

- [ ] **Step 3: Write minimal implementation**

Implement:
- state fields needed by verifier
- richer `verify_hypothesis` tool schema
- verifier invocation from tool implementation
- rollout extraction of verifier-specific fields

- [ ] **Step 4: Run tests to verify they pass**

Run:
- `python -m unittest /mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/tests/test_saver_agent_tools.py -v`
- `python -m unittest /mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/tests/test_saver_agent_rollout.py -v`

Expected: PASS

### Task 4: Full Regression Check

**Files:**
- Verify only

- [ ] **Step 1: Run focused verifier/reward/tool/rollout tests**

Run:
`python -m unittest /mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/tests/test_saver_agent_verifier.py /mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/tests/test_saver_agent_reward.py /mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/tests/test_saver_agent_tools.py /mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/tests/test_saver_agent_rollout.py -v`

- [ ] **Step 2: Run full suite**

Run:
`python -m unittest discover -s /mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code/tests -v`

- [ ] **Step 3: Summarize verification evidence**

Record:
- test command outputs
- any residual gaps
- GPU-dependent pieces intentionally not run
