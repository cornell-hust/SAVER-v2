from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from saver_agent.score_summary import extract_verifier_statuses, summarize_scored_rollouts


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _normalize_interval(interval: Sequence[float] | None) -> Optional[Tuple[float, float]]:
    if not interval or len(interval) != 2:
        return None
    start_sec = _safe_float(interval[0], 0.0)
    end_sec = _safe_float(interval[1], 0.0)
    if end_sec < start_sec:
        start_sec, end_sec = end_sec, start_sec
    return start_sec, end_sec


def _interval_length(interval: Sequence[float] | None) -> float:
    normalized = _normalize_interval(interval)
    if normalized is None:
        return 0.0
    return max(0.0, normalized[1] - normalized[0])


def _interval_overlap(interval_a: Sequence[float] | None, interval_b: Sequence[float] | None) -> float:
    a = _normalize_interval(interval_a)
    b = _normalize_interval(interval_b)
    if a is None or b is None:
        return 0.0
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def _interval_iou(interval_a: Sequence[float] | None, interval_b: Sequence[float] | None) -> float:
    overlap = _interval_overlap(interval_a, interval_b)
    if overlap <= 0:
        return 0.0
    len_a = _interval_length(interval_a)
    len_b = _interval_length(interval_b)
    union = max(len_a + len_b - overlap, 1e-6)
    return max(0.0, min(1.0, overlap / union))


def _merge_intervals(intervals: Iterable[Sequence[float]]) -> List[Tuple[float, float]]:
    normalized = sorted(
        (value for value in (_normalize_interval(interval) for interval in intervals) if value is not None),
        key=lambda item: item[0],
    )
    if not normalized:
        return []
    merged: List[Tuple[float, float]] = [normalized[0]]
    for start_sec, end_sec in normalized[1:]:
        prev_start, prev_end = merged[-1]
        if start_sec <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end_sec))
        else:
            merged.append((start_sec, end_sec))
    return merged


def _union_length(intervals: Iterable[Sequence[float]]) -> float:
    return sum(end_sec - start_sec for start_sec, end_sec in _merge_intervals(intervals))


def _binary_average_precision(targets: Sequence[int], scores: Sequence[float]) -> float:
    if not targets or not scores or len(targets) != len(scores):
        return 0.0
    total_positives = sum(1 for value in targets if int(value) > 0)
    if total_positives <= 0:
        return 0.0
    ranked_indices = sorted(range(len(targets)), key=lambda idx: float(scores[idx]), reverse=True)
    precision_sum = 0.0
    true_positives = 0
    for rank, idx in enumerate(ranked_indices, start=1):
        if int(targets[idx]) > 0:
            true_positives += 1
            precision_sum += true_positives / rank
    return precision_sum / total_positives


def _macro_f1(gt_labels: Sequence[str], pred_labels: Sequence[str]) -> float:
    classes = sorted({label for label in gt_labels if label and label != "normal"})
    if not classes or len(gt_labels) != len(pred_labels):
        return 0.0
    per_class_f1: List[float] = []
    for label in classes:
        tp = fp = fn = 0
        for gt_label, pred_label in zip(gt_labels, pred_labels):
            if gt_label == label and pred_label == label:
                tp += 1
            elif gt_label != label and pred_label == label:
                fp += 1
            elif gt_label == label and pred_label != label:
                fn += 1
        denom = 2 * tp + fp + fn
        per_class_f1.append((2 * tp / denom) if denom > 0 else 0.0)
    return _mean(per_class_f1)


def _extract_reference_target(record: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(record.get("structured_target"), dict):
        return dict(record["structured_target"])
    label = dict(record.get("label") or {})
    temporal = dict(record.get("temporal") or {})
    return {
        "existence": "anomaly" if label.get("is_anomaly") else "normal",
        "category": label.get("category"),
        "severity": label.get("severity"),
        "hard_normal": label.get("hard_normal"),
        "anomaly_interval_sec": temporal.get("anomaly_interval_sec"),
        "precursor_interval_sec": temporal.get("precursor_interval_sec"),
        "earliest_alert_sec": temporal.get("earliest_alert_sec"),
        "counterfactual_type": (record.get("counterfactual") or {}).get("type", "none"),
    }


def _extract_reference_evidence_windows(record: Dict[str, Any]) -> List[Tuple[float, float]]:
    tool_io = dict(record.get("tool_io") or {})
    raw_windows = tool_io.get("oracle_windows_sec") or []
    intervals: List[Tuple[float, float]] = []
    for entry in raw_windows:
        interval = entry.get("window") or entry.get("window_sec")
        normalized = _normalize_interval(interval)
        if normalized is not None:
            intervals.append(normalized)
    if intervals:
        return intervals
    target = _extract_reference_target(record)
    for entry in target.get("evidence_windows_sec") or []:
        interval = entry.get("window") or entry.get("window_sec")
        normalized = _normalize_interval(interval)
        if normalized is not None:
            intervals.append(normalized)
    return intervals


def _predicted_existence_score(record: Dict[str, Any]) -> float:
    claim = _infer_claim_from_rollout(record)
    if claim:
        return 1.0 if str(claim.get("existence") or "").lower() == "anomaly" else 0.0
    first_alert = _infer_first_alert(record)
    if first_alert and str(first_alert.get("decision") or "") in {"soft_alert", "hard_alert"}:
        return 0.75
    return 0.0


def _normalize_existence(value: Any) -> str:
    text = str(value or "").strip().lower()
    return "anomaly" if text == "anomaly" else "normal"


def _normalize_category(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text or "unknown"


def _normalize_counterfactual_type(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text or "none"


def _infer_claim_from_rollout(record: Dict[str, Any]) -> Dict[str, Any]:
    state = record.get("state") or {}
    finalized_case = state.get("finalized_case")
    if isinstance(finalized_case, dict):
        return dict(finalized_case)
    final_answer = record.get("final_answer")
    if isinstance(final_answer, dict):
        return dict(final_answer)
    last_claim = state.get("last_claim")
    if isinstance(last_claim, dict):
        return dict(last_claim)
    return {}


def _infer_first_alert(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    state = record.get("state") or {}
    alerts = list(state.get("alerts") or [])
    if alerts:
        for alert in alerts:
            if str(alert.get("decision") or "") in {"soft_alert", "hard_alert"}:
                return dict(alert)
        return dict(alerts[0])
    for turn in record.get("turns") or []:
        parsed_tool_call = turn.get("parsed_tool_call") or {}
        if str(parsed_tool_call.get("name") or "") != "emit_alert":
            continue
        arguments = parsed_tool_call.get("arguments") or {}
        if str(arguments.get("decision") or "") in {"soft_alert", "hard_alert"}:
            return dict(arguments)
    return None


def _infer_candidate_window_ids(record: Dict[str, Any]) -> List[str]:
    turns = record.get("turns") or []
    for turn in reversed(turns):
        if turn.get("verifier_verified_window_ids"):
            return list(turn.get("verifier_verified_window_ids") or [])
    state = record.get("state") or {}
    if state.get("active_evidence_window_ids"):
        return list(state.get("active_evidence_window_ids") or [])
    return [str(entry.get("window_id")) for entry in (state.get("evidence_ledger") or []) if entry.get("window_id")]


def _resolve_state_windows(record: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    state = record.get("state") or {}
    entries = list(state.get("visited_windows") or []) + list(state.get("evidence_ledger") or [])
    resolved: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        window_id = entry.get("window_id")
        if window_id and window_id not in resolved:
            resolved[str(window_id)] = dict(entry)
    return resolved


def _first_matching_turn_index(
    turns: Sequence[Dict[str, Any]],
    *,
    tool_name: Optional[str] = None,
    action: Optional[str] = None,
) -> Optional[int]:
    for turn in turns:
        if tool_name is not None and turn.get("tool_name") == tool_name:
            return _safe_int(turn.get("step_index"), 0)
        if action is not None and turn.get("action") == action:
            return _safe_int(turn.get("step_index"), 0)
    return None


def _protocol_compliance_flag(record: Dict[str, Any]) -> float:
    turns = list(record.get("turns") or [])
    state = record.get("state") or {}

    finalize_turn_index = _first_matching_turn_index(turns, tool_name="finalize_case")
    answer_turn_index = _first_matching_turn_index(turns, action="answer")

    has_finalize_artifact = finalize_turn_index is not None or bool(state.get("finalized_case"))
    has_answer_artifact = answer_turn_index is not None or bool(record.get("final_answer"))

    if not has_finalize_artifact or not has_answer_artifact:
        return 0.0
    if finalize_turn_index is not None and answer_turn_index is not None:
        return 1.0 if finalize_turn_index < answer_turn_index else 0.0
    return 1.0


def _select_predicted_evidence_windows(record: Dict[str, Any], *, top_k: int) -> List[Tuple[float, float]]:
    if top_k <= 0:
        return []
    candidate_ids = list(_infer_candidate_window_ids(record) or [])
    by_window_id = _resolve_state_windows(record)
    selected: List[Tuple[float, float]] = []
    seen = set()
    for window_id in candidate_ids:
        window = by_window_id.get(str(window_id))
        if window is None:
            continue
        normalized = _normalize_interval((window.get("start_sec"), window.get("end_sec")))
        if normalized is None or normalized in seen:
            continue
        seen.add(normalized)
        selected.append(normalized)
        if len(selected) >= top_k:
            return selected
    if selected:
        return selected[:top_k]
    for window in by_window_id.values():
        normalized = _normalize_interval((window.get("start_sec"), window.get("end_sec")))
        if normalized is None or normalized in seen:
            continue
        seen.add(normalized)
        selected.append(normalized)
        if len(selected) >= top_k:
            break
    return selected


def _evidence_match_counts(
    predicted_windows: Sequence[Tuple[float, float]],
    reference_windows: Sequence[Tuple[float, float]],
    *,
    iou_threshold: float,
) -> Tuple[int, int, int]:
    matched_reference = set()
    matched_predicted = 0
    for pred_index, predicted in enumerate(predicted_windows):
        best_idx = None
        best_iou = 0.0
        for ref_index, reference in enumerate(reference_windows):
            if ref_index in matched_reference:
                continue
            iou = _interval_iou(predicted, reference)
            if iou >= float(iou_threshold) and iou > best_iou:
                best_iou = iou
                best_idx = ref_index
        if best_idx is not None:
            matched_reference.add(best_idx)
            matched_predicted += 1
    return matched_predicted, len(predicted_windows), len(reference_windows)


def summarize_saver_metrics(
    scored_records: Sequence[Dict[str, Any]],
    *,
    reference_data: Any,
    evidence_top_k: int = 3,
    evidence_iou_threshold: float = 0.3,
    include_diagnostic_summary: bool = False,
) -> Dict[str, Any]:
    existence_targets: List[int] = []
    existence_scores: List[float] = []
    existence_predictions: List[int] = []
    anomaly_gt_categories: List[str] = []
    anomaly_pred_categories: List[str] = []
    temporal_ious: List[float] = []
    precursor_ious: List[float] = []
    alert_utilities: List[float] = []
    premature_flags: List[float] = []
    late_flags: List[float] = []
    false_alert_flags: List[float] = []
    hard_normal_false_alert_flags: List[float] = []
    evidence_precisions: List[float] = []
    evidence_recalls: List[float] = []
    evidence_f1s: List[float] = []
    counterfactual_hits: List[float] = []
    severity_errors: List[float] = []
    inspected_clip_ratios: List[float] = []
    search_steps: List[float] = []
    total_turns: List[float] = []
    tool_validity_values: List[float] = []
    protocol_flags: List[float] = []
    primary_status_counter = Counter()
    alert_status_counter = Counter()

    for record in scored_records:
        video_id = record.get("video_id")
        if not video_id or video_id not in reference_data.by_video_id:
            continue
        reference_record = reference_data.by_video_id[str(video_id)]
        target = _extract_reference_target(reference_record)
        claim = _infer_claim_from_rollout(record)
        gt_existence = _normalize_existence(target.get("existence"))
        pred_existence = _normalize_existence(claim.get("existence"))

        existence_targets.append(1 if gt_existence == "anomaly" else 0)
        existence_predictions.append(1 if pred_existence == "anomaly" else 0)
        existence_scores.append(_predicted_existence_score(record))

        if gt_existence == "anomaly":
            anomaly_gt_categories.append(_normalize_category(target.get("category")))
            anomaly_pred_categories.append(
                _normalize_category(claim.get("category")) if pred_existence == "anomaly" else "normal"
            )
            temporal_ious.append(_interval_iou(claim.get("anomaly_interval_sec"), target.get("anomaly_interval_sec")))
            precursor_ious.append(
                _interval_iou(claim.get("precursor_interval_sec"), target.get("precursor_interval_sec"))
            )
            gt_counterfactual_type = _normalize_counterfactual_type(target.get("counterfactual_type"))
            if gt_counterfactual_type != "none":
                counterfactual_hits.append(
                    1.0 if _normalize_counterfactual_type(claim.get("counterfactual_type")) == gt_counterfactual_type else 0.0
                )
            gt_severity = target.get("severity")
            pred_severity = claim.get("severity")
            if gt_severity is not None and pred_severity is not None:
                severity_errors.append(abs(_safe_float(pred_severity) - _safe_float(gt_severity)))

            gt_evidence_windows = _extract_reference_evidence_windows(reference_record)
            if gt_evidence_windows:
                pred_evidence_windows = _select_predicted_evidence_windows(record, top_k=evidence_top_k)
                matched, pred_count, gt_count = _evidence_match_counts(
                    pred_evidence_windows,
                    gt_evidence_windows,
                    iou_threshold=evidence_iou_threshold,
                )
                precision = matched / pred_count if pred_count > 0 else 0.0
                recall = matched / gt_count if gt_count > 0 else 0.0
                denom = precision + recall
                f1 = (2 * precision * recall / denom) if denom > 0 else 0.0
                evidence_precisions.append(precision)
                evidence_recalls.append(recall)
                evidence_f1s.append(f1)

        duration = _safe_float((reference_record.get("video_meta") or {}).get("duration_sec"), 0.0)
        state = record.get("state") or {}
        visited_intervals = [
            (entry.get("start_sec"), entry.get("end_sec"))
            for entry in list(state.get("visited_windows") or [])
            if _normalize_interval((entry.get("start_sec"), entry.get("end_sec"))) is not None
        ]
        inspected_length = _union_length(visited_intervals)
        inspected_clip_ratios.append(inspected_length / duration if duration > 0 else 0.0)
        total_turns.append(_safe_float(record.get("num_turns"), 0.0))
        search_steps.append(
            float(
                sum(
                    1
                    for turn in (record.get("turns") or [])
                    if turn.get("tool_name") in {"scan_timeline", "seek_evidence"}
                )
            )
        )

        turns = list(record.get("turns") or [])
        if turns:
            tool_validity_values.append(
                sum(1.0 for turn in turns if bool(turn.get("valid_action", turn.get("action") in {"tool_call", "answer"})))
                / len(turns)
            )
        else:
            tool_validity_values.append(0.0)

        protocol_flags.append(_protocol_compliance_flag(record))

        primary_status, alert_status = extract_verifier_statuses(record)
        primary_status_counter[str(primary_status or "unknown")] += 1
        alert_status_counter[str(alert_status or "unknown")] += 1

        first_alert = _infer_first_alert(record)
        has_anomaly_alert = bool(first_alert) and str(first_alert.get("decision") or "") in {"soft_alert", "hard_alert"}
        if gt_existence == "normal":
            false_alert_flags.append(1.0 if has_anomaly_alert else 0.0)
            if bool(target.get("hard_normal")):
                hard_normal_false_alert_flags.append(1.0 if has_anomaly_alert else 0.0)
            alert_utilities.append(-1.0 if has_anomaly_alert else 1.0)
        else:
            gt_alert_sec = target.get("earliest_alert_sec")
            anomaly_interval = target.get("anomaly_interval_sec")
            gt_alert_sec = _safe_float(
                gt_alert_sec if gt_alert_sec is not None else ((_normalize_interval(anomaly_interval) or (0.0, 0.0))[0]),
                0.0,
            )
            precursor_interval = target.get("precursor_interval_sec")
            precursor_duration = max(_interval_length(precursor_interval), 1.0)
            anomaly_duration = max(_interval_length(anomaly_interval), 1.0)
            tolerance_sec = max(0.5, 0.02 * max(duration, 0.0))
            if not has_anomaly_alert:
                premature_flags.append(0.0)
                late_flags.append(0.0)
                alert_utilities.append(0.0)
            else:
                pred_alert_sec = _safe_float(
                    first_alert.get("alert_sec", first_alert.get("earliest_alert_sec")),
                    gt_alert_sec,
                )
                is_premature = pred_alert_sec < gt_alert_sec - tolerance_sec
                is_late = pred_alert_sec > gt_alert_sec + tolerance_sec
                premature_flags.append(1.0 if is_premature else 0.0)
                late_flags.append(1.0 if is_late else 0.0)
                if is_premature:
                    alert_utilities.append(-min(1.0, (gt_alert_sec - pred_alert_sec) / precursor_duration))
                else:
                    alert_utilities.append(1.0 - min(1.0, max(0.0, pred_alert_sec - gt_alert_sec) / anomaly_duration))

    num_records = len(existence_targets)
    primary_ratios = {
        key: (primary_status_counter.get(key, 0) / num_records if num_records else 0.0)
        for key in ["complete", "incomplete", "redundant", "misaligned", "unknown"]
    }
    alert_ratios = {
        key: (alert_status_counter.get(key, 0) / num_records if num_records else 0.0)
        for key in ["justified", "premature", "late", "not_applicable", "unknown"]
    }
    summary = {
        "num_records": num_records,
        "existence_ap": _binary_average_precision(existence_targets, existence_scores),
        "existence_accuracy": (
            sum(1.0 for gt, pred in zip(existence_targets, existence_predictions) if gt == pred) / num_records
            if num_records
            else 0.0
        ),
        "category_macro_f1": _macro_f1(anomaly_gt_categories, anomaly_pred_categories),
        "temporal_miou": _mean(temporal_ious),
        "precursor_miou": _mean(precursor_ious),
        "alert_utility": _mean(alert_utilities),
        "premature_alert_rate": _mean(premature_flags),
        "late_alert_rate": _mean(late_flags),
        "false_alert_rate": _mean(false_alert_flags),
        "hard_normal_false_alert_rate": _mean(hard_normal_false_alert_flags),
        "evidence_precision_at_3": _mean(evidence_precisions),
        "evidence_recall_at_3": _mean(evidence_recalls),
        "evidence_f1_at_3": _mean(evidence_f1s),
        "counterfactual_type_accuracy": _mean(counterfactual_hits),
        "severity_mae": _mean(severity_errors),
        "mean_inspected_clip_ratio": _mean(inspected_clip_ratios),
        "mean_search_steps": _mean(search_steps),
        "mean_num_turns": _mean(total_turns),
        "tool_call_validity_rate": _mean(tool_validity_values),
        "protocol_compliance_rate": _mean(protocol_flags),
        "main_metrics_reference_free": True,
    }
    if include_diagnostic_summary:
        summary["diagnostic_summary"] = {
            "primary_status_ratios": primary_ratios,
            "alert_status_ratios": alert_ratios,
            "score_summary": summarize_scored_rollouts(scored_records),
        }
    return summary
