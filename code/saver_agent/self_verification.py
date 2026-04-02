from __future__ import annotations

from typing import Any, Dict, List, Optional

from saver_agent.categories import CANONICAL_POLICY_CATEGORIES, validate_canonical_category_payload

SELF_VERIFICATION_DECISIONS = {"insufficient", "sufficient", "misaligned", "redundant"}
SELF_VERIFICATION_ACTIONS = {"continue_search", "revise_claim", "refine_evidence", "finalize"}
SELF_VERIFICATION_MODES = {
    "soft_alert_check",
    "hard_alert_check",
    "final_check",
    "full_keep_drop",
    "reward_only",
    "search_step_check",
}
LEGACY_VERIFICATION_MODE_ALIASES = {
    "normal_check": "final_check",
    "declare_normal": "final_check",
}
PRIMARY_STATUS_TO_DECISION = {
    "complete": "sufficient",
    "incomplete": "insufficient",
    "misaligned": "misaligned",
    "redundant": "redundant",
}
DECISION_TO_PRIMARY_STATUS = {
    "sufficient": "complete",
    "insufficient": "incomplete",
    "misaligned": "misaligned",
    "redundant": "redundant",
}
POLICY_SELF_VERIFICATION_CLAIM_KEYS = ("existence", "category", "earliest_alert_sec")
POLICY_SELF_VERIFICATION_ALERT_KEYS = ("decision", "existence", "category", "alert_sec", "earliest_alert_sec")
POLICY_SELF_VERIFICATION_REQUIRED_FIELDS = (
    "verification_mode",
    "selected_window_ids",
    "verification_decision",
    "recommended_action",
    "sufficiency_score",
    "necessity_score",
    "alertability_score",
    "counterfactual_faithfulness",
)


def _compact_payload_object(payload: Dict[str, Any], *, keys: tuple[str, ...]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for key in keys:
        if key not in payload:
            continue
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        compact[key] = value
    return compact


def _coerce_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        result: List[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                result.append(text)
        return result
    text = str(value).strip()
    return [text] if text else []


def _coerce_score(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return float(default)


def _normalize_decision(payload: Dict[str, Any]) -> str:
    decision = str(payload.get("verification_decision") or "").strip().lower()
    if decision in SELF_VERIFICATION_DECISIONS:
        return decision
    primary_status = str(payload.get("primary_status") or "").strip().lower()
    return PRIMARY_STATUS_TO_DECISION.get(primary_status, "insufficient")


def _normalize_action(payload: Dict[str, Any], decision: str) -> str:
    action = str(payload.get("recommended_action") or "").strip().lower()
    if action in SELF_VERIFICATION_ACTIONS:
        return action
    if decision == "sufficient":
        return "finalize"
    if decision == "misaligned":
        return "revise_claim"
    if decision == "redundant":
        return "refine_evidence"
    return "continue_search"


def normalize_self_verification_mode(value: Any, *, default: str = "reward_only") -> str:
    text = str(value or "").strip().lower()
    if text in LEGACY_VERIFICATION_MODE_ALIASES:
        return LEGACY_VERIFICATION_MODE_ALIASES[text]
    if text in SELF_VERIFICATION_MODES:
        return text
    return str(default or "reward_only")


def _derive_alert_status(
    *,
    alert: Optional[Dict[str, Any]],
    claim: Dict[str, Any],
    alertability_score: float,
) -> str:
    if not isinstance(alert, dict) or not alert:
        return "not_applicable"
    if alertability_score >= 0.65:
        return "justified"
    try:
        alert_sec = float(alert.get("alert_sec", alert.get("earliest_alert_sec")))
    except Exception:
        alert_sec = None
    try:
        expected_sec = float(claim.get("earliest_alert_sec"))
    except Exception:
        expected_sec = None
    if alert_sec is not None and expected_sec is not None and alert_sec > expected_sec + 0.5:
        return "late"
    return "premature"


def _derive_failure_reasons(primary_status: str, alert_status: str) -> List[str]:
    reasons: List[str] = []
    if primary_status == "incomplete":
        reasons.append("selected_evidence_not_sufficient")
    elif primary_status == "misaligned":
        reasons.append("selected_evidence_not_aligned_with_claim")
    elif primary_status == "redundant":
        reasons.append("selected_evidence_not_necessary_enough")
    if alert_status == "premature":
        reasons.append("alert_prefix_not_actionable")
    elif alert_status == "late":
        reasons.append("alert_after_expected_actionable_time")
    return reasons


def parse_self_verification_payload(
    payload: Dict[str, Any],
    *,
    fallback_claim: Optional[Dict[str, Any]] = None,
    fallback_alert: Optional[Dict[str, Any]] = None,
    verification_mode: str = "reward_only",
) -> Dict[str, Any]:
    payload = dict(payload or {})
    claim = validate_canonical_category_payload(
        dict(payload.get("claim") or fallback_claim or {}),
        payload_name="claim",
        require_category_for_anomaly=True,
    )
    alert_value = payload.get("alert")
    alert = validate_canonical_category_payload(
        dict(alert_value),
        payload_name="alert",
    ) if isinstance(alert_value, dict) else (
        validate_canonical_category_payload(
            dict(fallback_alert),
            payload_name="alert",
        ) if isinstance(fallback_alert, dict) else None
    )
    verification_mode_normalized = normalize_self_verification_mode(
        payload.get("verification_mode") or verification_mode,
        default=verification_mode,
    )

    decision = _normalize_decision(payload)
    primary_status = DECISION_TO_PRIMARY_STATUS.get(decision, "incomplete")
    recommended_action = _normalize_action(payload, decision)

    derived_scores_in = dict(payload.get("derived_scores") or {})
    sufficiency = _coerce_score(payload.get("sufficiency_score", derived_scores_in.get("sufficiency", 0.0)))
    necessity = _coerce_score(payload.get("necessity_score", derived_scores_in.get("necessity", 0.0)))
    alertability = _coerce_score(payload.get("alertability_score", derived_scores_in.get("alertability", 0.0)))
    counterfactual_faithfulness = _coerce_score(
        payload.get(
            "counterfactual_faithfulness",
            derived_scores_in.get("counterfactual_faithfulness", 0.0),
        )
    )
    consistency = _coerce_score(
        derived_scores_in.get("consistency", 1.0 - abs(sufficiency - necessity) if (sufficiency or necessity) else 0.0)
    )
    derived_scores = {
        "sufficiency": round(sufficiency, 6),
        "necessity": round(necessity, 6),
        "consistency": round(consistency, 6),
        "alertability": round(alertability, 6),
        "counterfactual_faithfulness": round(counterfactual_faithfulness, 6),
    }

    selected_window_ids = _coerce_string_list(
        payload.get("selected_window_ids")
        or payload.get("verified_window_ids")
        or payload.get("best_effort_window_ids")
    )
    candidate_window_ids = _coerce_string_list(payload.get("candidate_window_ids"))
    selected_evidence_ids = _coerce_string_list(
        payload.get("selected_evidence_ids")
        or payload.get("candidate_evidence_ids")
        or payload.get("evidence_ids")
    )
    selected_evidence_moment_ids = _coerce_string_list(
        payload.get("selected_evidence_moment_ids")
        or payload.get("candidate_evidence_moment_ids")
        or payload.get("evidence_moment_ids")
    )
    best_effort_window_ids = list(selected_window_ids or candidate_window_ids)
    alert_status = str(payload.get("alert_status") or "").strip().lower()
    if alert_status not in {"justified", "premature", "late", "not_applicable"}:
        alert_status = _derive_alert_status(alert=alert, claim=claim, alertability_score=alertability)
    failure_reasons = _coerce_string_list(payload.get("failure_reasons")) or _derive_failure_reasons(
        primary_status,
        alert_status,
    )

    parsed = {
        "verification_mode": verification_mode_normalized,
        "claim": claim,
        "alert": alert,
        "query": str(payload.get("query") or ""),
        "candidate_window_ids": candidate_window_ids,
        "candidate_evidence_ids": _coerce_string_list(payload.get("candidate_evidence_ids")),
        "candidate_evidence_moment_ids": _coerce_string_list(
            payload.get("candidate_evidence_moment_ids") or payload.get("evidence_moment_ids")
        ),
        "selected_window_ids": selected_window_ids,
        "selected_evidence_ids": selected_evidence_ids,
        "selected_evidence_moment_ids": selected_evidence_moment_ids,
        "verification_decision": decision,
        "recommended_action": recommended_action,
        "rationale": str(payload.get("rationale") or payload.get("explanation") or ""),
        "primary_status": primary_status,
        "alert_status": alert_status,
        "derived_scores": derived_scores,
        "verified_window_ids": list(selected_window_ids),
        "best_effort_window_ids": best_effort_window_ids,
        "failure_reasons": failure_reasons,
        "verifier_backend": "self_report",
        "self_verification_scores": dict(derived_scores),
        "self_verification_selected_window_ids": list(selected_window_ids),
        "self_verification_confidence": round(max(sufficiency, necessity), 6),
    }
    return parsed


def validate_policy_self_verification_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(payload or {})
    missing = [
        field_name
        for field_name in POLICY_SELF_VERIFICATION_REQUIRED_FIELDS
        if field_name not in normalized
    ]
    if missing:
        raise ValueError(
            "verify_hypothesis self-verification payload is missing required field(s): "
            + ", ".join(missing)
            + "."
        )
    if not _coerce_string_list(normalized.get("selected_window_ids")):
        raise ValueError(
            "verify_hypothesis self-verification payload must include at least one selected_window_id."
        )
    return normalized


def build_policy_self_verification_payload(
    payload: Dict[str, Any],
    *,
    include_query: bool = True,
    include_rationale: bool = True,
) -> Dict[str, Any]:
    source = dict(payload or {})
    decision = _normalize_decision(source)
    recommended_action = _normalize_action(source, decision)
    compact: Dict[str, Any] = {
        "verification_mode": normalize_self_verification_mode(
            source.get("verification_mode"),
            default="reward_only",
        ),
        "verification_decision": decision,
        "recommended_action": recommended_action,
        "sufficiency_score": round(_coerce_score(source.get("sufficiency_score")), 6),
        "necessity_score": round(_coerce_score(source.get("necessity_score")), 6),
        "alertability_score": round(_coerce_score(source.get("alertability_score")), 6),
        "counterfactual_faithfulness": round(_coerce_score(source.get("counterfactual_faithfulness")), 6),
    }

    claim = source.get("claim")
    if isinstance(claim, dict) and claim:
        normalized_claim = validate_canonical_category_payload(
            dict(claim),
            payload_name="claim",
            require_category_for_anomaly=True,
        )
        compact_claim = _compact_payload_object(
            normalized_claim,
            keys=POLICY_SELF_VERIFICATION_CLAIM_KEYS,
        )
        if compact_claim:
            compact["claim"] = compact_claim

    alert = source.get("alert")
    if isinstance(alert, dict) and alert:
        normalized_alert = validate_canonical_category_payload(
            dict(alert),
            payload_name="alert",
        )
        compact_alert = _compact_payload_object(
            normalized_alert,
            keys=POLICY_SELF_VERIFICATION_ALERT_KEYS,
        )
        if compact_alert:
            compact["alert"] = compact_alert

    query = str(source.get("query") or "").strip()
    if include_query and query:
        compact["query"] = query

    selected_window_ids = _coerce_string_list(
        source.get("selected_window_ids")
        or source.get("verified_window_ids")
        or source.get("best_effort_window_ids")
    )
    if selected_window_ids:
        compact["selected_window_ids"] = selected_window_ids

    selected_evidence_moment_ids = _coerce_string_list(
        source.get("selected_evidence_moment_ids")
        or source.get("evidence_moment_ids")
        or source.get("candidate_evidence_moment_ids")
    )
    if selected_evidence_moment_ids:
        compact["selected_evidence_moment_ids"] = selected_evidence_moment_ids

    rationale = str(source.get("rationale") or source.get("explanation") or "").strip()
    if include_rationale and rationale:
        compact["rationale"] = rationale

    return compact


def build_self_verification_tool_schema() -> Dict[str, Any]:
    claim_schema = {
        "type": "object",
        "properties": {
            "existence": {"type": "string", "enum": ["normal", "anomaly"]},
            "category": {"type": "string", "enum": list(CANONICAL_POLICY_CATEGORIES)},
            "earliest_alert_sec": {"oneOf": [{"type": "null"}, {"type": "number"}]},
        },
    }
    alert_schema = {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": ["soft_alert", "hard_alert", "declare_normal"]},
            "existence": {"type": "string", "enum": ["normal", "anomaly"]},
            "category": {"type": "string", "enum": list(CANONICAL_POLICY_CATEGORIES)},
            "alert_sec": {"oneOf": [{"type": "null"}, {"type": "number"}]},
            "earliest_alert_sec": {"oneOf": [{"type": "null"}, {"type": "number"}]},
        },
    }
    return {
        "type": "object",
        "properties": {
            "verification_mode": {
                "type": "string",
                "enum": sorted(SELF_VERIFICATION_MODES),
            },
            "selected_window_ids": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            },
            "selected_evidence_moment_ids": {"type": "array", "items": {"type": "string"}},
            "claim": claim_schema,
            "alert": alert_schema,
            "query": {"type": "string"},
            "verification_decision": {"type": "string", "enum": sorted(SELF_VERIFICATION_DECISIONS)},
            "recommended_action": {"type": "string", "enum": sorted(SELF_VERIFICATION_ACTIONS)},
            "sufficiency_score": {"type": "number"},
            "necessity_score": {"type": "number"},
            "alertability_score": {"type": "number"},
            "counterfactual_faithfulness": {"type": "number"},
            "rationale": {"type": "string"},
        },
        "required": list(POLICY_SELF_VERIFICATION_REQUIRED_FIELDS),
    }
