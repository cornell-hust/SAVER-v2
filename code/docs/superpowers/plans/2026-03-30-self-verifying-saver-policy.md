# Self-Verifying SAVER Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train SAVER as a single self-verifying policy that learns counterfactual evidence and alert verification during SFT and RL, so inference no longer depends on an external verifier model.

**Architecture:** Replace the current external `verify_hypothesis -> verifier backend -> recommended_action` control loop with an internalized self-verification loop. The policy still emits a `verify_hypothesis` tool call, but the tool becomes a lightweight recorder/executor of the policy's own structured verification output instead of calling an external heuristic/Qwen verifier. Training uses oracle verifier supervision in SFT and a single teacher judge based on `Qwen3-VL-32B-Instruct`, which supports both text-structured judging and multimodal frame-aware judging for reward calibration and distillation in RL/SFT.

**Tech Stack:** Python, Transformers/Qwen3-VL, current SAVER rollout stack, existing CEA-GRPO pipeline, single multimodal teacher judge via `Qwen3-VL-32B-Instruct` and VLLM/OpenAI-compatible endpoint.

---

## File Structure

**Create**
- `saver_agent/self_verification.py`
  Purpose: shared schema, parsers, label builders, reward helpers for policy-internal verification.
- `tests/test_saver_agent_self_verification.py`
  Purpose: unit tests for self-verification parsing, schema normalization, and reward extraction.

**Modify**
- `saver_agent/tool_registry.py`
  Purpose: extend `verify_hypothesis` schema from backend selector to structured self-verification payload.
- `saver_agent/tools.py`
  Purpose: make `verify_hypothesis` consume policy-produced verification JSON instead of external verifier verdicts.
- `saver_agent/adapter.py`
  Purpose: change follow-up prompting so tool observations reflect self-verification state rather than external verdict language.
- `convert_to_saver_agent.py`
  Purpose: export oracle self-verification labels and multi-branch verifier supervision for SFT.
- `saver_agent/training_data.py`
  Purpose: inject self-verification supervision into prepared SFT and convert RL traces into self-verification training targets / rewards.
- `saver_agent/rollout.py`
  Purpose: record policy self-verification fields in trace and expose them to RL credit assignment.
- `saver_agent/reward.py`
  Purpose: split reward into task reward and self-verification reward; add unified teacher-judge calibration reward from one Qwen3-VL backend.
- `train_saver_sft.py`
  Purpose: add switches for self-verification SFT and optional Qwen3-VL teacher-judge distillation targets.
- `train_saver_rl.py`
  Purpose: add same-policy self-verification mode, `Qwen3-VL-32B-Instruct` teacher-judge endpoint hooks, teacher input-mode controls, and GRPO-style self-verification reward mixing.
- `README.md`
  Purpose: update training/eval instructions to "single policy inference, optional teacher judge during training".

**Retain but downgrade**
- `saver_agent/verifier.py`
  Purpose: keep as diagnostic / fallback / ablation backend, not as default online verifier.
- `saver_agent/qwen_verifier.py`
  Purpose: optional frozen Qwen-VL teacher verifier bootstrap with both text-only and multimodal judging modes, not inference-time dependency.

---

### Task 1: Define Policy-Internal Self-Verification Contract

**Files:**
- Create: `saver_agent/self_verification.py`
- Modify: `saver_agent/tool_registry.py`
- Test: `tests/test_saver_agent_self_verification.py`

- [ ] **Step 1: Write failing parser/schema tests**

```python
def test_parse_self_verification_payload_accepts_complete_case():
    payload = {
        "claim": {"existence": "anomaly", "category": "assault"},
        "selected_window_ids": ["w0001", "w0002"],
        "sufficiency_score": 0.82,
        "necessity_score": 0.63,
        "alertability_score": 0.74,
        "verification_decision": "sufficient",
        "recommended_action": "finalize",
    }
    parsed = parse_self_verification_payload(payload)
    assert parsed["recommended_action"] == "finalize"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_saver_agent_self_verification -v`
Expected: FAIL with missing module/function errors.

- [ ] **Step 3: Add minimal self-verification schema helpers**

Implement:
- `SELF_VERIFICATION_DECISIONS = {"insufficient", "sufficient", "misaligned", "redundant"}`
- `SELF_VERIFICATION_ACTIONS = {"continue_search", "revise_claim", "refine_evidence", "finalize"}`
- `parse_self_verification_payload(payload)`
- `build_self_verification_tool_schema()`

Required verification payload fields:
- `claim`
- `selected_window_ids`
- `selected_evidence_ids`
- `selected_evidence_moment_ids`
- `sufficiency_score`
- `necessity_score`
- `alertability_score`
- `counterfactual_faithfulness`
- `verification_decision`
- `recommended_action`
- `rationale`

- [ ] **Step 4: Update `verify_hypothesis` tool schema**

Replace backend-oriented arguments with:
- optional `verification_mode`
- required structured self-verification payload fields above
- keep legacy selectors for compatibility during migration

- [ ] **Step 5: Run tests**

Run: `python -m unittest tests.test_saver_agent_self_verification -v`
Expected: PASS


### Task 2: Convert `verify_hypothesis` From External Judge Call to State Recorder

**Files:**
- Modify: `saver_agent/tools.py`
- Modify: `saver_agent/adapter.py`
- Modify: `saver_agent/rollout.py`
- Test: `tests/test_saver_agent_tools.py`
- Test: `tests/test_saver_agent_rollout.py`

- [ ] **Step 1: Write failing tests for self-verification tool execution**

Add tests asserting:
- `verify_hypothesis` no longer calls external verifier by default
- tool output records the policy-provided verification JSON
- `state.active_evidence_window_ids` comes from payload selectors
- rollout summary fields still populate `verifier_primary_status`-like trace slots, now from self-verification payload

- [ ] **Step 2: Implement tool-side migration**

In `saver_agent/tools.py`:
- add `online_verifier_mode = "self_report"` path as default
- normalize the policy payload into a stored `verification_record`
- map `verification_decision -> primary_status` for backward compatibility:
  - `sufficient -> complete`
  - `insufficient -> incomplete`
  - `misaligned -> misaligned`
  - `redundant -> redundant`
- never call `run_counterfactual_verifier()` unless explicit fallback flag is enabled

- [ ] **Step 3: Update observation prompting**

In `saver_agent/adapter.py`:
- consume self-reported `recommended_action`
- avoid language that implies an external authority judged the case
- use wording like:
  - "Your verification judged the current evidence sufficient. Call finalize_case next."
  - "Your verification judged the evidence insufficient. Search more or revise the claim."

- [ ] **Step 4: Preserve rollout trace compatibility**

In `saver_agent/rollout.py`:
- keep current summary keys for downstream training code
- populate them from self-verification payload
- add new trace fields:
  - `self_verification_decision`
  - `self_verification_scores`
  - `self_verification_selected_window_ids`
  - `self_verification_confidence`

- [ ] **Step 5: Run focused tests**

Run:
- `python -m unittest tests.test_saver_agent_tools -v`
- `python -m unittest tests.test_saver_agent_rollout -v`

Expected: PASS


### Task 3: Teach Self-Verification Explicitly in SFT

**Files:**
- Modify: `convert_to_saver_agent.py`
- Modify: `saver_agent/training_data.py`
- Test: `tests/test_convert_to_saver_agent.py`
- Test: `tests/test_saver_agent_training_data.py`

- [ ] **Step 1: Write failing tests for oracle self-verification labels**

Add tests asserting oracle SFT examples contain:
- structured self-verification payloads
- multi-branch outputs for `continue_search`, `revise_claim`, `refine_evidence`, `finalize`
- no dependence on runtime external verifier output

- [ ] **Step 2: Expand oracle labels**

In `convert_to_saver_agent.py`:
- replace minimal `oracle_verifier_feedback` with richer labels:
  - `selected_window_ids`
  - `sufficiency_score`
  - `necessity_score`
  - `alertability_score`
  - `counterfactual_faithfulness`
  - `verification_decision`
  - `recommended_action`
- explicitly create three verifier outcomes where applicable:
  - insufficient -> continue_search
  - misaligned -> revise_claim
  - sufficient -> finalize
- for evidence-pruning cases, emit `redundant -> refine_evidence`

- [ ] **Step 3: Serialize richer tool supervision**

In `saver_agent/training_data.py`:
- make `_apply_oracle_verifier_feedback()` emit the richer payload
- add verifier-turn weight bonus for correct structured self-verification, not only final action
- ensure prepared SFT examples supervise:
  - when to verify
  - what evidence subset to claim
  - what self-verification scores/decision to produce
  - what next action to take

- [ ] **Step 4: Add optional distillation labels**

Allow prepared SFT records to optionally store:
- `teacher_judge_scores`
- `teacher_judge_decision`
- `teacher_judge_rationale`

These remain optional and are ignored when absent.

- [ ] **Step 5: Run focused tests**

Run:
- `python -m unittest tests.test_convert_to_saver_agent -v`
- `python -m unittest tests.test_saver_agent_training_data -v`

Expected: PASS


### Task 4: Convert RL From External Verifier Reward to Self-Verification Reward

**Files:**
- Modify: `saver_agent/training_data.py`
- Modify: `saver_agent/reward.py`
- Modify: `train_saver_rl.py`
- Test: `tests/test_saver_agent_reward.py`
- Test: `tests/test_train_saver_rl.py`

- [ ] **Step 1: Write failing tests for self-verification reward extraction**

Add tests asserting RL examples receive reward from:
- self-verification sufficiency/necessity correctness
- self-verification action calibration
- agreement with Qwen3-VL teacher judge in both `text_only` and `multimodal_visual` modes

- [ ] **Step 2: Redefine CEA-GRPO local scoring inputs**

In `saver_agent/training_data.py`:
- replace mandatory dependency on `local_verifier_backend` for actual branch scoring
- compute actual-branch local advantage from rollout trace self-verification first
- keep heuristic/Qwen verifier only as fallback / teacher comparison

Add new per-turn fields:
- `self_verification_target`
- `self_verification_agreement_reward`
- `self_verification_consistency_reward`
- `teacher_judge_agreement`

- [ ] **Step 3: Redefine reward composition**

In `saver_agent/reward.py`:
- split rewards into:
  - `task_outcome_reward`
  - `search_efficiency_reward`
  - `self_verification_quality_reward`
  - `teacher_judge_alignment_reward` (optional)
- allow turning teacher reward on/off independently

Recommended default mixing:
- task outcome: 0.45
- alert/evidence/search counterfactual outcome: 0.25
- self-verification quality: 0.20
- teacher judge alignment: 0.10

- [ ] **Step 4: Add RL switches**

In `train_saver_rl.py`, add:
- `--online-verifier-mode {self_report,fallback_external}`
- `--teacher-judge-backend {none,qwen3_vl_teacher,openai_compatible_vl}`
- `--teacher-judge-model`
- `--teacher-judge-url`
- `--teacher-judge-input-mode {text_only,multimodal_visual,auto}`
- `--teacher-judge-anchor-policy {verify_only,verify_and_finalize,hard_examples}`
- `--teacher-judge-topk-frames-per-view`
- `--teacher-judge-weight`
- `--self-verification-reward-weight`
- `--disable-external-online-verifier`

Default target behavior:
- online inference / rollout collection uses `self_report`
- external verifier only used for fallback or diagnostics
- teacher judge defaults to `text_only` on most anchors and upgrades to `multimodal_visual` for hard cases when `auto` is enabled

- [ ] **Step 5: Run focused tests**

Run:
- `python -m unittest tests.test_saver_agent_reward -v`
- `python -m unittest tests.test_train_saver_rl -v`

Expected: PASS


### Task 5: Add Training-Time Qwen3-VL Teacher Judge Support

**Files:**
- Create: `saver_agent/self_verification.py`
- Modify: `train_saver_sft.py`
- Modify: `train_saver_rl.py`
- Modify: `README.md`
- Test: `tests/test_saver_agent_self_verification.py`

- [ ] **Step 1: Add teacher-judge client abstraction**

Implement minimal helpers:
- `score_self_verification_with_teacher(...)`
- `build_teacher_judge_prompt(...)`
- `build_teacher_judge_package(...)`
- `select_teacher_judge_input_mode(...)`
- parser for structured teacher outputs:
  - `sufficiency_score`
  - `necessity_score`
  - `alertability_score`
  - `verification_decision`
  - `recommended_action`

- [ ] **Step 2: Wire teacher usage into SFT and RL**

SFT:
- optional offline label enrichment
- optional KL / regression-to-teacher auxiliary loss later

RL:
- optional teacher score at sampled verification turns
- optional sparse teacher calls only at anchors to save compute
- optional mixed teacher usage:
  - `text_only`: judge structured claim + summaries + timestamps + self-verification output
  - `multimodal_visual`: judge the same package plus sampled raw frames for `full / keep / drop / alert_prefix`
  - `auto`: default to `text_only`, escalate to `multimodal_visual` on hard/ambiguous anchors

- [ ] **Step 3: Document recommended teacher setup**

Document `Qwen3-VL-32B-Instruct` as the default single training-time teacher judge.
Document two supported input modes from the same model:
- `text_only`
- `multimodal_visual`

Document that the base teacher input package includes:
- policy-produced claim
- `full / keep / drop / alert_prefix` view summaries
- selected timestamps
- policy self-verification output

Document that `multimodal_visual` additionally includes:
- sampled raw frames from each view
- optional preview frames for broader context

Document sparse-calling defaults to control cost:
- only verification anchor turns
- only top-K frames per view
- optional hard-example-only relabeling
- `auto` mode escalation triggers:
  - low policy self-verification confidence
  - keep/drop scores too close
  - alert decision near threshold
  - disagreement with oracle/teacher history on similar anchors

- [ ] **Step 4: Run parser/client tests**

Run: `python -m unittest tests.test_saver_agent_self_verification -v`
Expected: PASS


### Task 6: Update Eval Semantics and Backward Compatibility

**Files:**
- Modify: `saver_agent/evaluation.py`
- Modify: `saver_agent/metrics.py`
- Modify: `README.md`
- Test: `tests/test_saver_agent_evaluation.py`
- Test: `tests/test_saver_agent_metrics.py`

- [ ] **Step 1: Preserve current metric keys while changing provenance**

Metrics should still expose:
- protocol compliance
- evidence F1
- temporal mIoU
- alert utility

But self-verification metrics should become explicit:
- `self_verification_decision_accuracy`
- `self_verification_action_accuracy`
- `self_verification_teacher_agreement`
- `finalize_after_sufficient_rate`
- `continue_search_after_insufficient_rate`

- [ ] **Step 2: Keep fallback evaluator path**

Maintain `heuristic` / `hybrid` verifier in eval only for:
- ablation
- diagnosis
- debugging regression cases

Do not make them default online inference dependencies.

- [ ] **Step 3: Run evaluation/metrics tests**

Run:
- `python -m unittest tests.test_saver_agent_evaluation -v`
- `python -m unittest tests.test_saver_agent_metrics -v`

Expected: PASS


### Task 7: Docs and Migration Commands

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add migration note**

Explain:
- old mode: external verifier tool judges policy
- new mode: policy emits self-verification, training may use teacher judge
- teacher judge is a single `Qwen3-VL-32B-Instruct` backend with two input modes, not two separate models

- [ ] **Step 2: Add recommended staged rollout**

Document three phases:
- Phase A: self-verification SFT warm start
- Phase B: CEA-GRPO with self-verification reward + `Qwen3-VL-32B-Instruct` teacher judge
- Phase C: inference with single policy only

- [ ] **Step 3: Add command templates**

Include:
- SFT with self-verification labels
- RL with `--online-verifier-mode self_report`
- optional teacher judge endpoint using `Qwen3-VL-32B-Instruct`
- examples for:
  - `--teacher-judge-input-mode text_only`
  - `--teacher-judge-input-mode multimodal_visual`
  - `--teacher-judge-input-mode auto`

- [ ] **Step 4: Final regression sweep**

Run:
- `python -m unittest discover -s tests`
- `python -m py_compile train_saver_sft.py train_saver_rl.py saver_agent/*.py`

Expected: all tests pass, no syntax errors.
