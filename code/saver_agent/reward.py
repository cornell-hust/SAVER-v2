from __future__ import annotations

from typing import Any, Dict, Optional


DEFAULT_COMPONENT_WEIGHTS = {
    "decision_reward": 1.2,
    "temporal_reward": 0.8,
    "alert_reward": 0.8,
    "verification_reward": 1.0,
    "efficiency_reward": 0.4,
    "counterfactual_reward": 0.6,
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


def _decision_reward(rollout_trace: Dict[str, Any]) -> float:
    final_answer = rollout_trace.get("final_answer")
    finalized_case = (rollout_trace.get("state") or {}).get("finalized_case")
    if final_answer or finalized_case:
        return 1.0
    if rollout_trace.get("terminated_reason") == "answered":
        return 0.5
    return 0.0


def _temporal_reward(verifier_turn: Optional[Dict[str, Any]]) -> float:
    if not verifier_turn:
        return 0.0
    derived = verifier_turn.get("verifier_derived_scores") or {}
    temporal_like = 0.5 * float(derived.get("consistency", 0.0)) + 0.5 * float(
        derived.get("sufficiency", 0.0)
    )
    return max(-1.0, min(1.0, temporal_like))


def _alert_reward(verifier_turn: Optional[Dict[str, Any]]) -> float:
    if not verifier_turn:
        return 0.0
    status = verifier_turn.get("verifier_alert_status")
    return float(ALERT_STATUS_REWARD.get(status, 0.0))


def _verification_reward(verifier_turn: Optional[Dict[str, Any]]) -> float:
    if not verifier_turn:
        return 0.0
    primary_status = verifier_turn.get("verifier_primary_status")
    derived = verifier_turn.get("verifier_derived_scores") or {}
    base = float(PRIMARY_STATUS_REWARD.get(primary_status, 0.0))
    shaping = 0.2 * float(derived.get("sufficiency", 0.0)) + 0.2 * float(derived.get("necessity", 0.0))
    return max(-1.0, min(1.0, base + shaping))


def _efficiency_reward(rollout_trace: Dict[str, Any]) -> float:
    num_turns = int(rollout_trace.get("num_turns") or 0)
    if num_turns <= 0:
        return 0.0
    return max(-1.0, min(1.0, 1.0 - 0.15 * max(num_turns - 1, 0)))


def _counterfactual_reward(verifier_turn: Optional[Dict[str, Any]]) -> float:
    if not verifier_turn:
        return 0.0
    derived = verifier_turn.get("verifier_derived_scores") or {}
    return max(-1.0, min(1.0, float(derived.get("counterfactual_faithfulness", 0.0))))


def score_rollout_trace(
    rollout_trace: Dict[str, Any],
    *,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    weights = {**DEFAULT_COMPONENT_WEIGHTS, **(weights or {})}
    verifier_turn = _latest_verifier_turn(rollout_trace)

    components = {
        "decision_reward": _decision_reward(rollout_trace),
        "temporal_reward": _temporal_reward(verifier_turn),
        "alert_reward": _alert_reward(verifier_turn),
        "verification_reward": _verification_reward(verifier_turn),
        "efficiency_reward": _efficiency_reward(rollout_trace),
        "counterfactual_reward": _counterfactual_reward(verifier_turn),
    }
    total_reward = 0.0
    for key, value in components.items():
        total_reward += float(weights.get(key, 0.0)) * float(value)

    return {
        "total_reward": round(total_reward, 6),
        "components": {key: round(value, 6) for key, value in components.items()},
        "weights": dict(weights),
        "latest_verifier_turn_present": verifier_turn is not None,
        "verifier_source": (
            str(verifier_turn.get("_verifier_source")) if verifier_turn is not None else "none"
        ),
        "uses_reference_conditioned_verifier": bool(
            verifier_turn.get("_uses_reference_conditioned_verifier") if verifier_turn is not None else False
        ),
    }
