from __future__ import annotations

from typing import Any, Dict, Optional

from saver_agent.teacher_judge import compute_teacher_judge_signal


DEFAULT_COMPONENT_WEIGHTS = {
    "decision_reward": 1.2,
    "protocol_reward": 1.0,
    "temporal_grounding_reward": 0.8,
    "alert_calibration_reward": 0.8,
    "self_verification_quality_reward": 1.0,
    "efficiency_reward": 0.4,
    "invalid_attempt_penalty": 0.6,
    "counterfactual_reward": 0.6,
    "teacher_agreement_reward": 0.6,
    "temporal_reward": 0.8,
    "alert_reward": 0.8,
    "verification_reward": 1.0,
    "teacher_judge_reward": 0.6,
}

LEGACY_COMPONENT_ALIASES = {
    "temporal_reward": "temporal_grounding_reward",
    "alert_reward": "alert_calibration_reward",
    "verification_reward": "self_verification_quality_reward",
    "teacher_judge_reward": "teacher_agreement_reward",
}

PRIMARY_STATUS_REWARD = {
    "complete": 1.0,
    "redundant": 0.35,
    "incomplete": -0.35,
    "misaligned": -1.0,
}

ALERT_STATUS_REWARD = {
    "justified": 1.0,
    "not_applicable": 0.0,
    "late": -0.35,
    "premature": -0.8,
}


def _normalize_component_weights(weights: Optional[Dict[str, float]]) -> Dict[str, float]:
    merged = dict(DEFAULT_COMPONENT_WEIGHTS)
    for key, value in (weights or {}).items():
        canonical_key = LEGACY_COMPONENT_ALIASES.get(str(key), str(key))
        merged[canonical_key] = float(value)
    return merged


def _latest_verifier_turn(rollout_trace: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for turn in reversed(rollout_trace.get("turns") or []):
        if turn.get("tool_name") == "verify_hypothesis":
            copied = dict(turn)
            copied["_verifier_source"] = "online_turn"
            copied["_uses_reference_conditioned_verifier"] = False
            return copied
    offline_verifier = rollout_trace.get("offline_verifier")
    if isinstance(offline_verifier, dict):
        uses_reference_conditioned_verifier = bool(offline_verifier.get("reference_conditioned", True))
        return {
            "tool_name": "verify_hypothesis",
            "verifier_primary_status": offline_verifier.get("primary_status"),
            "verifier_alert_status": offline_verifier.get("alert_status"),
            "verifier_recommended_action": offline_verifier.get("recommended_action"),
            "verifier_derived_scores": offline_verifier.get("derived_scores") or {},
            "_verifier_source": (
                "offline_reference_conditioned" if uses_reference_conditioned_verifier else "offline_unconditioned"
            ),
            "_uses_reference_conditioned_verifier": uses_reference_conditioned_verifier,
        }
    return None


def _first_matching_turn_index(
    turns: list[Dict[str, Any]],
    *,
    tool_name: Optional[str] = None,
    action: Optional[str] = None,
) -> Optional[int]:
    for turn in turns:
        if tool_name is not None and turn.get("tool_name") == tool_name:
            return int(turn.get("step_index") or 0)
        if action is not None and turn.get("action") == action:
            return int(turn.get("step_index") or 0)
    return None


def _count_invalid_attempts(rollout_trace: Dict[str, Any]) -> int:
    explicit = rollout_trace.get("num_invalid_attempts")
    if explicit is not None:
        try:
            return max(int(explicit), 0)
        except Exception:
            pass
    return len(list(rollout_trace.get("invalid_attempts") or []))


def _decision_reward(rollout_trace: Dict[str, Any]) -> float:
    final_answer = rollout_trace.get("final_answer")
    finalized_case = (rollout_trace.get("state") or {}).get("finalized_case")
    has_structured_answer = isinstance(final_answer, dict)
    has_finalized_case = isinstance(finalized_case, dict)
    return 1.0 if has_structured_answer and has_finalized_case else 0.0


def _protocol_reward(rollout_trace: Dict[str, Any], verifier_turn: Optional[Dict[str, Any]]) -> float:
    turns = list(rollout_trace.get("turns") or [])
    state = rollout_trace.get("state") or {}
    final_answer = rollout_trace.get("final_answer")

    verify_turn_index = _first_matching_turn_index(turns, tool_name="verify_hypothesis")
    finalize_turn_index = _first_matching_turn_index(turns, tool_name="finalize_case")
    answer_turn_index = _first_matching_turn_index(turns, action="answer")

    has_verify = verify_turn_index is not None or verifier_turn is not None
    has_finalize_artifact = finalize_turn_index is not None or isinstance(state.get("finalized_case"), dict)
    has_answer_artifact = answer_turn_index is not None and isinstance(final_answer, dict)

    if not has_answer_artifact:
        return -1.0 if not has_finalize_artifact else -0.5
    if not has_finalize_artifact:
        return -1.0
    if not has_verify:
        return -0.75
    if finalize_turn_index is not None and answer_turn_index is not None:
        if verify_turn_index is not None and verify_turn_index < finalize_turn_index < answer_turn_index:
            return 1.0
        return -1.0
    if answer_turn_index is not None and verify_turn_index is not None and verify_turn_index < answer_turn_index:
        return 0.5
    return -0.5


def _temporal_grounding_reward(verifier_turn: Optional[Dict[str, Any]]) -> float:
    if not verifier_turn:
        return 0.0
    derived = verifier_turn.get("verifier_derived_scores") or verifier_turn.get("self_verification_scores") or {}
    temporal_like = 0.5 * float(derived.get("consistency", 0.0)) + 0.5 * float(derived.get("sufficiency", 0.0))
    return max(-1.0, min(1.0, temporal_like))


def _alert_calibration_reward(verifier_turn: Optional[Dict[str, Any]]) -> float:
    if not verifier_turn:
        return 0.0
    status = verifier_turn.get("verifier_alert_status")
    return float(ALERT_STATUS_REWARD.get(status, 0.0))


def _self_verification_quality_reward(verifier_turn: Optional[Dict[str, Any]]) -> float:
    if not verifier_turn:
        return 0.0
    primary_status = verifier_turn.get("verifier_primary_status")
    derived = verifier_turn.get("verifier_derived_scores") or verifier_turn.get("self_verification_scores") or {}
    base = float(PRIMARY_STATUS_REWARD.get(primary_status, 0.0))
    shaping = 0.2 * float(derived.get("sufficiency", 0.0)) + 0.2 * float(derived.get("necessity", 0.0))
    return max(-1.0, min(1.0, base + shaping))


def _efficiency_reward(rollout_trace: Dict[str, Any]) -> float:
    final_answer = rollout_trace.get("final_answer")
    finalized_case = (rollout_trace.get("state") or {}).get("finalized_case")
    if not (isinstance(final_answer, dict) and isinstance(finalized_case, dict)):
        return 0.0
    num_turns = int(rollout_trace.get("num_turns") or 0)
    num_turns += _count_invalid_attempts(rollout_trace)
    if num_turns <= 0:
        return 0.0
    return max(-1.0, min(1.0, 1.0 - 0.15 * max(num_turns - 1, 0)))


def _invalid_attempt_penalty(rollout_trace: Dict[str, Any]) -> float:
    invalid_attempts = _count_invalid_attempts(rollout_trace)
    if invalid_attempts <= 0:
        return 0.0
    return -min(1.0, 0.4 * float(invalid_attempts))


def _counterfactual_reward(verifier_turn: Optional[Dict[str, Any]]) -> float:
    if not verifier_turn:
        return 0.0
    derived = verifier_turn.get("verifier_derived_scores") or verifier_turn.get("self_verification_scores") or {}
    return max(-1.0, min(1.0, float(derived.get("counterfactual_faithfulness", 0.0))))


def _teacher_agreement_reward(verifier_turn: Optional[Dict[str, Any]]) -> float:
    if not verifier_turn:
        return 0.0
    return float(compute_teacher_judge_signal(verifier_turn).get("teacher_judge_reward") or 0.0)


def score_rollout_trace(
    rollout_trace: Dict[str, Any],
    *,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    normalized_weights = _normalize_component_weights(weights)
    verifier_turn = _latest_verifier_turn(rollout_trace)

    core_components = {
        "decision_reward": _decision_reward(rollout_trace),
        "protocol_reward": _protocol_reward(rollout_trace, verifier_turn),
        "temporal_grounding_reward": _temporal_grounding_reward(verifier_turn),
        "alert_calibration_reward": _alert_calibration_reward(verifier_turn),
        "self_verification_quality_reward": _self_verification_quality_reward(verifier_turn),
        "efficiency_reward": _efficiency_reward(rollout_trace),
        "invalid_attempt_penalty": _invalid_attempt_penalty(rollout_trace),
        "counterfactual_reward": _counterfactual_reward(verifier_turn),
        "teacher_agreement_reward": _teacher_agreement_reward(verifier_turn),
    }
    total_reward = 0.0
    for key, value in core_components.items():
        total_reward += float(normalized_weights.get(key, 0.0)) * float(value)

    components = dict(core_components)
    for legacy_key, canonical_key in LEGACY_COMPONENT_ALIASES.items():
        components[legacy_key] = components[canonical_key]

    teacher_signal = compute_teacher_judge_signal(verifier_turn or {})
    return {
        "total_reward": round(total_reward, 6),
        "components": {key: round(value, 6) for key, value in components.items()},
        "weights": dict(normalized_weights),
        "latest_verifier_turn_present": verifier_turn is not None,
        "verifier_source": (
            str(verifier_turn.get("_verifier_source")) if verifier_turn is not None else "none"
        ),
        "uses_reference_conditioned_verifier": bool(
            verifier_turn.get("_uses_reference_conditioned_verifier") if verifier_turn is not None else False
        ),
        "teacher_judge_present": bool(teacher_signal.get("teacher_judge_present")),
        "teacher_judge_alignment": teacher_signal.get("teacher_judge_alignment"),
    }
