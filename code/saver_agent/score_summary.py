from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PRIMARY_STATUS_KEYS = ["complete", "incomplete", "redundant", "misaligned", "unknown"]
ALERT_STATUS_KEYS = ["justified", "premature", "late", "not_applicable", "unknown"]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _latest_turn_verdict(record: Dict[str, Any]) -> Tuple[str, str]:
    turns = record.get("turns") or []
    for turn in reversed(turns):
        primary_status = turn.get("verifier_primary_status")
        alert_status = turn.get("verifier_alert_status")
        if primary_status or alert_status:
            return str(primary_status or "unknown"), str(alert_status or "unknown")
    return "unknown", "unknown"


def extract_verifier_statuses(record: Dict[str, Any]) -> Tuple[str, str]:
    offline_verifier = record.get("offline_verifier")
    if isinstance(offline_verifier, dict):
        primary_status = str(offline_verifier.get("primary_status") or "unknown")
        alert_status = str(offline_verifier.get("alert_status") or "unknown")
        return primary_status, alert_status
    return _latest_turn_verdict(record)


def _initialize_counter(keys: Sequence[str]) -> Dict[str, int]:
    return {key: 0 for key in keys}


def summarize_scored_rollouts(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    num_records = len(records)
    primary_counter = Counter()
    alert_counter = Counter()
    total_rewards: List[float] = []
    num_turns: List[float] = []
    component_values: Dict[str, List[float]] = {}

    for record in records:
        primary_status, alert_status = extract_verifier_statuses(record)
        if primary_status not in PRIMARY_STATUS_KEYS:
            primary_status = "unknown"
        if alert_status not in ALERT_STATUS_KEYS:
            alert_status = "unknown"
        primary_counter[primary_status] += 1
        alert_counter[alert_status] += 1

        total_rewards.append(_safe_float((record.get("reward_summary") or {}).get("total_reward"), 0.0))
        num_turns.append(_safe_float(record.get("num_turns"), 0.0))

        reward_components = (record.get("reward_summary") or {}).get("components") or {}
        for key, value in reward_components.items():
            component_values.setdefault(str(key), []).append(_safe_float(value, 0.0))

    primary_counts = _initialize_counter(PRIMARY_STATUS_KEYS)
    primary_counts.update(primary_counter)
    alert_counts = _initialize_counter(ALERT_STATUS_KEYS)
    alert_counts.update(alert_counter)

    primary_ratios = {
        key: (_safe_float(primary_counts[key]) / num_records if num_records else 0.0)
        for key in PRIMARY_STATUS_KEYS
    }
    alert_ratios = {
        key: (_safe_float(alert_counts[key]) / num_records if num_records else 0.0)
        for key in ALERT_STATUS_KEYS
    }
    mean_component_values = {key: _mean(values) for key, values in sorted(component_values.items())}

    return {
        "num_records": num_records,
        "primary_status_counts": primary_counts,
        "primary_status_ratios": primary_ratios,
        "alert_status_counts": alert_counts,
        "alert_status_ratios": alert_ratios,
        "mean_total_reward": _mean(total_rewards),
        "min_total_reward": min(total_rewards) if total_rewards else 0.0,
        "max_total_reward": max(total_rewards) if total_rewards else 0.0,
        "mean_num_turns": _mean(num_turns),
        "min_num_turns": min(num_turns) if num_turns else 0.0,
        "max_num_turns": max(num_turns) if num_turns else 0.0,
        "mean_reward_components": mean_component_values,
    }
