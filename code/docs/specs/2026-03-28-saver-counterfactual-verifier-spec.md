# SAVER Counterfactual Verifier Spec

## Goal

Define a rollout-native verifier for SAVER that matches the core counterfactual idea while staying stylistically close to TimeSearch-R:

- the verifier is not an answer checker
- the verifier judges whether the currently searched and selected evidence subset is enough to support the current anomaly claim and alert decision
- the verifier must answer two counterfactual questions:
  - if we drop the selected evidence, does support collapse
  - if we keep only the selected evidence, does support remain

The verifier should be directly callable inside the existing `saver_agent` rollout loop and should also emit structured signals that can be consumed by reward code later.

## TimeSearch-R Alignment

This design follows the overall spirit of TimeSearch-R:

- use a unified rollout environment instead of a detached post-hoc verifier pipeline
- make verification a structured component inside tool-using interaction
- allow the same backbone family to serve both policy and verifier roles
- let reward consume verifier outputs instead of hiding verification inside loose free-form text

This design does **not** assume a multi-stage verifier training schedule. SFT warm-start and rollout-time RL can be added later, but the verifier interface is designed to work in a single unified system from the start.

## Problem Definition

Given:

- a searched trajectory with visited windows `W`
- a current candidate evidence subset `E`, where `E` is drawn from `W`
- a current structured claim `C`
- an optional alert decision `A`

the verifier should determine:

1. sufficiency: whether `E` alone is enough to support `C`
2. necessity: whether removing `E` from `W` materially weakens support for `C`
3. alert validity: whether the evidence available at the alert time was already actionable
4. evidence quality type: whether the selected evidence is complete, incomplete, redundant, or misaligned

This is the operational definition of SAVER counterfactual verification.

## Chosen Design

We use a **claim-conditioned multi-view verifier** with four views:

- `full`: all currently visited windows `W`
- `keep`: selected evidence subset `E`
- `drop`: complement subset `W - E`
- `alert_prefix`: the subset of visited windows whose timestamps are not later than the proposed alert time

The first three views implement the counterfactual core. The fourth is a SAVER-specific check for early warning validity.

## Claim Schema

The verifier does not validate generic answers. It validates a structured claim.

The canonical claim schema is:

```json
{
  "existence": "normal|anomaly",
  "category": "string",
  "severity": 0,
  "anomaly_interval_sec": [0.0, 1.0],
  "precursor_interval_sec": [0.0, 1.0],
  "earliest_alert_sec": 0.0,
  "evidence_ids": ["e1", "e2"],
  "counterfactual_type": "string",
  "summary": "string",
  "rationale": "string"
}
```

Only `existence` is mandatory for early rollout checks. The rest may be partially filled during intermediate verification.

The verifier should accept partial claims and degrade gracefully.

## Verifier Input Contract

The tool-level verifier input is:

```json
{
  "verification_mode": "soft_alert_check|hard_alert_check|final_check|reward_only",
  "claim": {
    "existence": "anomaly",
    "category": "assault",
    "severity": 4,
    "anomaly_interval_sec": [1.8, 14.2],
    "precursor_interval_sec": [0.2, 1.8],
    "earliest_alert_sec": 1.8,
    "counterfactual_type": "remove_actor_interaction"
  },
  "candidate_evidence_ids": ["ev2", "ev3"],
  "candidate_window_ids": ["evidence_2", "evidence_3"],
  "alert": {
    "decision": "soft_alert",
    "alert_sec": 1.8
  },
  "query": "optional hypothesis text"
}
```

### Resolution Rules

- `candidate_evidence_ids` refers to GT-like semantic evidence ids when available
- `candidate_window_ids` refers to searched windows stored in environment state
- if only one of them is provided, the verifier resolves the other when possible
- if neither is provided, verifier falls back to the currently accumulated evidence ledger

## Verifier Output Contract

The verifier returns a structured verdict:

```json
{
  "primary_status": "complete|incomplete|redundant|misaligned",
  "alert_status": "justified|premature|late|not_applicable",
  "recommended_action": "continue_search|refine_evidence|finalize|revise_claim",
  "view_scores": {
    "full": {
      "exist_support": 0.92,
      "category_support": 0.81,
      "temporal_support": 0.77,
      "precursor_support": 0.68,
      "alert_support": 0.88,
      "counterfactual_support": 0.74,
      "overall_support": 0.83
    },
    "keep": {
      "exist_support": 0.84,
      "category_support": 0.78,
      "temporal_support": 0.72,
      "precursor_support": 0.51,
      "alert_support": 0.80,
      "counterfactual_support": 0.69,
      "overall_support": 0.76
    },
    "drop": {
      "exist_support": 0.31,
      "category_support": 0.18,
      "temporal_support": 0.25,
      "precursor_support": 0.14,
      "alert_support": 0.22,
      "counterfactual_support": 0.20,
      "overall_support": 0.22
    },
    "alert_prefix": {
      "exist_support": 0.79,
      "category_support": 0.71,
      "temporal_support": 0.55,
      "precursor_support": 0.83,
      "alert_support": 0.76,
      "counterfactual_support": 0.62,
      "overall_support": 0.71
    }
  },
  "derived_scores": {
    "sufficiency": 0.76,
    "necessity": 0.61,
    "consistency": 0.93,
    "alertability": 0.76,
    "counterfactual_faithfulness": 0.74
  },
  "verified_window_ids": ["evidence_2", "evidence_3"],
  "best_effort_window_ids": ["evidence_2", "evidence_3", "evidence_4"],
  "failure_reasons": [],
  "explanation": "Selected evidence is sufficient and mostly necessary. Alert timing is supported.",
  "verifier_backend": "heuristic|qwen_self_verifier|hybrid"
}
```

## Status Semantics

### `primary_status`

- `complete`
  - `keep` remains strong
  - `drop` is clearly weaker than `full`
  - `keep` agrees with the main claim
- `incomplete`
  - `full` is reasonably strong
  - `keep` is materially weaker than `full`
  - missing decisive support is likely
- `redundant`
  - `keep` is strong
  - `drop` is also strong
  - selected evidence is not necessary enough
- `misaligned`
  - selected evidence supports a different category, wrong interval, wrong polarity, or no actionable anomaly

### `alert_status`

- `justified`
  - `alert_prefix` already supports the alert decision at the proposed alert time
- `premature`
  - current alert happened before evidence was actionable
- `late`
  - the system waited after evidence was already sufficient
- `not_applicable`
  - no alert decision is under verification

## Core Derived Scores

Derived scores are computed from view-level supports.

### Base Per-View Score

For each view, define:

```text
overall_support =
  w_exist * exist_support +
  w_cat   * category_support +
  w_temp  * temporal_support +
  w_pre   * precursor_support +
  w_alert * alert_support +
  w_cf    * counterfactual_support
```

Suggested default weights:

- `w_exist = 0.20`
- `w_cat = 0.20`
- `w_temp = 0.20`
- `w_pre = 0.10`
- `w_alert = 0.20`
- `w_cf = 0.10`

These should live in config, not code constants.

### Derived Verifier Metrics

```text
sufficiency = overall_support(keep)
necessity = clamp(overall_support(full) - overall_support(drop), 0, 1)
consistency = 1 - abs(overall_support(full) - overall_support(keep))
alertability = overall_support(alert_prefix)
counterfactual_faithfulness = 0.5 * sufficiency + 0.5 * necessity
```

## Reducer Rules

The reducer converts raw scores into statuses. The first implementation should be deterministic.

Suggested defaults:

- if `keep.exist_support < 0.4` for an anomaly claim, mark `misaligned`
- else if `keep.category_support < 0.35` and category is non-empty, mark `misaligned`
- else if `sufficiency < 0.55` and `full.overall_support >= 0.60`, mark `incomplete`
- else if `sufficiency >= 0.60` and `necessity < 0.20`, mark `redundant`
- else if `sufficiency >= 0.60` and `necessity >= 0.20` and `consistency >= 0.75`, mark `complete`
- else mark `incomplete`

Alert reducer:

- if no alert provided, `not_applicable`
- else if `alertability >= 0.65`, `justified`
- else if `claim.earliest_alert_sec` exists and `alert.alert_sec > claim.earliest_alert_sec + tolerance`, `late`
- else `premature`

Suggested `tolerance = 0.5s` initially.

## Backend Strategy

The verifier should support three backends behind the same interface.

### 1. `heuristic`

No model call. Score overlap and structural agreement using:

- IoU with GT anomaly interval
- overlap with GT evidence windows
- whether selected windows include trigger-like evidence
- whether selected windows occur before or after alert
- whether selected evidence covers precursor plus decisive trigger

Use this backend for:

- early reward scaffolding
- unit tests
- rollout fallback

### 2. `qwen_self_verifier`

Use Qwen3-VL as a structured judge over the four views.

Prompt should ask the model to assess:

- whether this view alone supports the claim
- whether the claim category and interval are supported
- whether an alert is justified from this view
- whether removing selected evidence destroys support

The model must emit strict JSON matching the per-view schema.

### 3. `hybrid`

Run both:

- heuristic scorer
- qwen self-verifier

Then fuse:

```text
final_score = alpha * heuristic + (1 - alpha) * qwen
```

Suggested default:

- `alpha = 0.7` for rollout-time control
- `alpha = 0.5` for offline evaluation

## Why Four Views Instead of Only Three

The `full/keep/drop` tri-view block captures counterfactual evidence faithfulness, but SAVER also cares about whether the **alert was justified at that time**.

That is not fully equivalent to tri-view evidence testing, because:

- the selected evidence may be correct in hindsight
- but the alert may still have been premature given the prefix actually observed at alert time

Therefore `alert_prefix` is necessary to make the verifier serve SAVER rather than generic evidence scoring.

## Environment State Extensions

Extend `SaverEnvironmentState` with:

- `last_claim: Optional[Dict[str, Any]]`
- `active_evidence_window_ids: List[str]`
- `verifier_cache: List[Dict[str, Any]]`

Each `visited_window` and `evidence_ledger` entry should have stable ids.

### Required Stable Identifiers

Add:

- `window_id` for visited windows
- `evidence_id` should remain stable and unique

Current state already tracks `evidence_id`; visited windows still need stable ids.

## Tool Contract Changes

### `emit_alert`

No major semantic change, but output should include:

- `alert_id`
- `alert_sec`
- `decision`

### `verify_hypothesis`

Upgrade from lightweight status response to full verifier call.

New schema:

```json
{
  "verification_mode": "soft_alert_check|hard_alert_check|final_check|reward_only",
  "claim": {"type": "object"},
  "candidate_window_ids": {"type": "array", "items": {"type": "string"}},
  "candidate_evidence_ids": {"type": "array", "items": {"type": "string"}},
  "alert": {"type": "object"},
  "query": {"type": "string"},
  "verifier_backend": {
    "type": "string",
    "enum": ["heuristic", "qwen_self_verifier", "hybrid"]
  }
}
```

Required:

- `verification_mode`

Behavior:

- resolve candidate subset
- construct the four views
- run selected backend
- reduce scores to statuses
- append full verdict to `state.verification_records`

### `finalize_case`

Should optionally consume:

- `verified_window_ids`
- `best_effort_window_ids`
- `verifier_status`
- `alert_status`

These fields should not be mandatory in the first iteration, but the schema should allow them.

## Rollout Trace Additions

The rollout trace should record verifier-specific fields so reward code does not need to re-parse raw tool text.

For each verifier turn, add:

- `verifier_mode`
- `verifier_backend`
- `verifier_primary_status`
- `verifier_alert_status`
- `verifier_recommended_action`
- `verifier_derived_scores`
- `verifier_verified_window_ids`
- `verifier_best_effort_window_ids`
- `verifier_failure_reasons`

Episode-level aggregates:

- `latest_verifier_status`
- `latest_alert_status`
- `verification_turn_count`
- `final_verified_window_ids`

## Reward Integration

Reward code should consume structured verifier verdicts rather than raw natural language.

Suggested terminal reward decomposition:

- `decision_reward`
  - anomaly existence
  - category
  - severity
- `temporal_reward`
  - anomaly interval
  - precursor interval
- `alert_reward`
  - justified alert bonus
  - premature alert penalty
  - late alert penalty
- `verification_reward`
  - `complete` bonus
  - `incomplete` penalty
  - `redundant` mild penalty
  - `misaligned` strong penalty
- `efficiency_reward`
  - fewer turns
  - fewer repeated windows
- `counterfactual_reward`
  - based on `counterfactual_faithfulness`

Initial shaped reward suggestion:

```text
R =
  1.2 * decision_reward +
  0.8 * temporal_reward +
  0.8 * alert_reward +
  1.0 * verification_reward +
  0.4 * efficiency_reward +
  0.6 * counterfactual_reward
```

The exact weights should be configurable.

## Supervision Construction From Current Data

Current `msad_saver_agent_train.jsonl` is sufficient to bootstrap verifier supervision.

Construct synthetic verifier targets as follows:

### Positive `complete`

- choose subsets that include trigger evidence
- optionally include precursor or confirmation evidence
- ensure subset overlaps the annotated anomaly interval

### Positive `incomplete`

- remove trigger evidence
- keep only precursor or aftermath windows
- or keep a very narrow subset that misses decisive support

### Positive `redundant`

- include correct trigger evidence
- add extra windows far outside decisive evidence
- or include nearly all visited windows when a much smaller subset is sufficient

### Positive `misaligned`

- pick windows outside anomaly interval
- pick only non-actionable precursor while claiming final anomaly
- or pick windows that support the wrong temporal segment

### Alert labels

- `justified`: prefix includes actionable trigger evidence
- `premature`: prefix ends before actionable evidence
- `late`: alert after annotated earliest alert plus tolerance

This enables verifier training and evaluation without introducing a separate bespoke annotation pass at the start.

## Implementation Plan

### New Files

- `saver_agent/verifier.py`
- `saver_agent/reward.py`
- `tests/test_saver_agent_verifier.py`
- `tests/test_saver_agent_reward.py`

### Files To Modify

- `saver_agent/schema.py`
- `saver_agent/tools.py`
- `saver_agent/tool_registry.py`
- `saver_agent/rollout.py`
- `run_saver_rollout.py`

### Optional Later Files

- `saver_agent/qwen_verifier.py`
- `score_saver_rollout.py`

## Module Responsibilities

### `saver_agent/verifier.py`

Owns:

- view construction
- heuristic scoring
- backend dispatch
- reducer
- verdict schema helpers

Primary API:

```python
def run_counterfactual_verifier(
    *,
    state,
    multimodal_cache,
    verification_mode,
    claim,
    candidate_window_ids=None,
    candidate_evidence_ids=None,
    alert=None,
    query="",
    backend="heuristic",
):
    ...
```

### `saver_agent/reward.py`

Owns:

- reward decomposition from rollout traces
- score aggregation
- future trainer-facing reward function wrappers

### `saver_agent/tools.py`

Owns:

- turning verifier verdicts into tool messages
- state updates

## Qwen Self-Verifier Prompt Shape

The verifier prompt should be strict and structured, not free-form CoT.

Prompt inputs:

- current task
- claim JSON
- view name
- compact timeline of windows in the view
- sampled images from that view
- explicit instruction to emit JSON only

Prompt questions:

- does this view support anomaly existence
- does it support the claimed category
- does it support the claimed temporal grounding
- is the proposed alert justified from this view
- if this is a keep-only view, is the evidence sufficient
- if this is a drop view, does support collapse

The output parser should be resilient but strict enough to reject malformed JSON.

## First Deliverable Scope

The first concrete implementation should deliver:

- deterministic heuristic backend
- upgraded `verify_hypothesis` tool
- richer verification trace fields
- offline reward computation using structured verifier outputs
- tests with synthetic states

The first deliverable should **not** depend on GPU inference.

## Second Deliverable Scope

After the first deliverable is stable:

- add `qwen_self_verifier`
- add `hybrid` fusion
- expose verifier backend selection through CLI and config

## Non-Goals For V1

- no dedicated separate verifier model training script yet
- no end-to-end RL trainer integration yet
- no object-level causal intervention
- no claim calibration over all language outputs yet

## Success Criteria

The spec is successful if:

1. `verify_hypothesis` returns structured counterfactual verdicts rather than weak status strings
2. the verdict can distinguish `complete`, `incomplete`, `redundant`, and `misaligned`
3. alert timing is evaluated separately through `alert_prefix`
4. reward code can consume verifier outputs without reparsing free-form text
5. the entire V1 implementation can run in CPU-only unit tests with the heuristic backend
