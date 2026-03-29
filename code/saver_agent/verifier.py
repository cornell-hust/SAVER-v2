from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from saver_agent.schema import SaverEnvironmentState


DEFAULT_SUPPORT_WEIGHTS = {
    "exist_support": 0.20,
    "category_support": 0.20,
    "temporal_support": 0.20,
    "precursor_support": 0.10,
    "alert_support": 0.20,
    "counterfactual_support": 0.10,
}

PRIMARY_STATUS_VALUES = {"complete", "incomplete", "redundant", "misaligned"}
ALERT_STATUS_VALUES = {"justified", "premature", "late", "not_applicable"}
_VERIFIER_RUNTIME_CACHE: Dict[Tuple[str, str, str, str, int], Any] = {}


def _normalize_interval(interval: Sequence[float] | None) -> Optional[Tuple[float, float]]:
    if not interval or len(interval) != 2:
        return None
    start_sec = float(interval[0])
    end_sec = float(interval[1])
    if end_sec < start_sec:
        start_sec, end_sec = end_sec, start_sec
    return start_sec, end_sec


def _interval_overlap(interval_a: Sequence[float] | None, interval_b: Sequence[float] | None) -> float:
    a = _normalize_interval(interval_a)
    b = _normalize_interval(interval_b)
    if a is None or b is None:
        return 0.0
    start_sec = max(a[0], b[0])
    end_sec = min(a[1], b[1])
    return max(0.0, end_sec - start_sec)


def _interval_duration(interval: Sequence[float] | None) -> float:
    normalized = _normalize_interval(interval)
    if normalized is None:
        return 0.0
    return max(0.0, normalized[1] - normalized[0])


def _overlap_ratio(interval_a: Sequence[float] | None, interval_b: Sequence[float] | None) -> float:
    duration = _interval_duration(interval_b)
    if duration <= 0:
        return 0.0
    return max(0.0, min(1.0, _interval_overlap(interval_a, interval_b) / duration))


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


def _coverage_ratio(windows: Sequence[Dict[str, Any]], target_interval: Sequence[float] | None) -> float:
    target = _normalize_interval(target_interval)
    if target is None:
        return 0.0
    clipped = []
    for window in windows:
        interval = _normalize_interval((window.get("start_sec"), window.get("end_sec")))
        if interval is None:
            continue
        overlap = _interval_overlap(interval, target)
        if overlap <= 0:
            continue
        clipped.append((max(interval[0], target[0]), min(interval[1], target[1])))
    if not clipped:
        return 0.0
    total = sum(end_sec - start_sec for start_sec, end_sec in _merge_intervals(clipped))
    target_duration = max(target[1] - target[0], 1e-6)
    return max(0.0, min(1.0, total / target_duration))


def _extract_target(
    multimodal_cache: Dict[str, Any],
    *,
    use_reference_supervision: bool = True,
) -> Dict[str, Any]:
    if not use_reference_supervision:
        return {}
    return dict(multimodal_cache.get("structured_target") or {})


def _extract_oracle_windows(
    multimodal_cache: Dict[str, Any],
    *,
    use_reference_supervision: bool = True,
) -> List[Dict[str, Any]]:
    if not use_reference_supervision:
        return []
    tool_io = multimodal_cache.get("tool_io") or {}
    raw_windows = tool_io.get("oracle_windows_sec") or []
    if raw_windows:
        windows = []
        for entry in raw_windows:
            interval = entry.get("window") or entry.get("window_sec")
            normalized = _normalize_interval(interval)
            if normalized is None:
                continue
            windows.append(
                {
                    "moment_id": entry.get("moment_id"),
                    "role": str(entry.get("role") or "").lower(),
                    "start_sec": normalized[0],
                    "end_sec": normalized[1],
                    "description": entry.get("description"),
                }
            )
        if windows:
            return windows

    target = _extract_target(multimodal_cache, use_reference_supervision=use_reference_supervision)
    evidence_windows = target.get("evidence_windows_sec") or []
    windows = []
    for entry in evidence_windows:
        interval = entry.get("window_sec") or entry.get("window")
        normalized = _normalize_interval(interval)
        if normalized is None:
            continue
        windows.append(
            {
                "moment_id": entry.get("moment_id"),
                "role": str(entry.get("role") or "").lower(),
                "start_sec": normalized[0],
                "end_sec": normalized[1],
                "description": entry.get("description"),
            }
        )
    return windows


def _resolve_windows(
    state: SaverEnvironmentState,
    *,
    candidate_window_ids: Optional[Sequence[str]] = None,
    candidate_evidence_ids: Optional[Sequence[str]] = None,
    candidate_evidence_moment_ids: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    by_window_id = {entry.get("window_id"): entry for entry in state.visited_windows}
    by_evidence_id = {entry.get("evidence_id"): entry for entry in state.evidence_ledger}
    by_moment_id: Dict[str, List[Dict[str, Any]]] = {}
    for entry in state.evidence_ledger:
        moment_id = entry.get("moment_id")
        if moment_id is None:
            continue
        by_moment_id.setdefault(str(moment_id), []).append(entry)
    resolved: List[Dict[str, Any]] = []
    seen = set()
    selectors_provided = bool(candidate_window_ids) or bool(candidate_evidence_ids) or bool(candidate_evidence_moment_ids)

    for window_id in candidate_window_ids or []:
        if window_id in by_window_id and window_id not in seen:
            seen.add(window_id)
            resolved.append(by_window_id[window_id])
    for evidence_id in candidate_evidence_ids or []:
        entry = by_evidence_id.get(evidence_id)
        if entry is None:
            continue
        window_id = entry.get("window_id")
        if window_id in seen:
            continue
        seen.add(window_id)
        resolved.append(entry)
    for moment_id in candidate_evidence_moment_ids or []:
        for entry in by_moment_id.get(str(moment_id), []):
            window_id = entry.get("window_id")
            if window_id in seen:
                continue
            seen.add(window_id)
            resolved.append(entry)

    if resolved:
        return resolved
    if selectors_provided:
        return []
    return list(state.evidence_ledger)


def _window_ids(windows: Sequence[Dict[str, Any]]) -> List[str]:
    return [str(entry.get("window_id")) for entry in windows if entry.get("window_id")]


def _role_scores(view_windows: Sequence[Dict[str, Any]], oracle_windows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for oracle in oracle_windows:
        role = str(oracle.get("role") or "").lower()
        oracle_interval = (oracle.get("start_sec"), oracle.get("end_sec"))
        best = 0.0
        for window in view_windows:
            candidate_interval = (window.get("start_sec"), window.get("end_sec"))
            best = max(best, _overlap_ratio(candidate_interval, oracle_interval))
        scores[role] = max(scores.get(role, 0.0), best)
    return scores


def _fallback_view_scores(view_windows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if not view_windows:
        return {
            "exist_support": 0.0,
            "category_support": 0.0,
            "temporal_support": 0.0,
            "precursor_support": 0.0,
            "alert_support": 0.0,
            "counterfactual_support": 0.0,
            "overall_support": 0.0,
        }
    coverage = min(1.0, 0.25 + 0.15 * len(view_windows))
    scores = {
        "exist_support": coverage,
        "category_support": coverage * 0.9,
        "temporal_support": coverage * 0.85,
        "precursor_support": coverage * 0.4,
        "alert_support": coverage * 0.75,
        "counterfactual_support": coverage * 0.8,
    }
    scores["overall_support"] = _weighted_support(scores)
    return scores


def _video_coverage_ratio(view_windows: Sequence[Dict[str, Any]], multimodal_cache: Dict[str, Any]) -> float:
    intervals = _merge_intervals(
        (window.get("start_sec"), window.get("end_sec"))
        for window in view_windows
    )
    if not intervals:
        return 0.0
    covered_duration = sum(max(0.0, end_sec - start_sec) for start_sec, end_sec in intervals)
    duration = 0.0
    try:
        duration = float(multimodal_cache.get("duration") or 0.0)
    except Exception:
        duration = 0.0
    if duration <= 0.0:
        duration = max(float(intervals[-1][1]), 1e-6)
    return max(0.0, min(1.0, covered_duration / max(duration, 1e-6)))


def _fallback_normal_view_scores(
    view_windows: Sequence[Dict[str, Any]],
    multimodal_cache: Dict[str, Any],
) -> Dict[str, float]:
    if not view_windows:
        return {
            "exist_support": 0.0,
            "category_support": 0.0,
            "temporal_support": 0.0,
            "precursor_support": 0.0,
            "alert_support": 0.0,
            "counterfactual_support": 0.0,
            "overall_support": 0.0,
        }

    coverage = _video_coverage_ratio(view_windows, multimodal_cache)
    search_windows = float(len(view_windows))
    exist_support = min(1.0, 0.25 + 0.60 * coverage + 0.05 * search_windows)
    category_support = min(1.0, 0.20 + 0.55 * coverage + 0.05 * search_windows)
    temporal_support = min(1.0, 0.15 + 0.60 * coverage + 0.05 * max(0.0, search_windows - 1.0))
    alert_support = min(1.0, 0.15 + 0.50 * coverage + 0.05 * search_windows)
    counterfactual_support = min(1.0, 0.10 + 0.50 * coverage + 0.05 * search_windows)

    scores = {
        "exist_support": round(exist_support, 6),
        "category_support": round(category_support, 6),
        "temporal_support": round(temporal_support, 6),
        "precursor_support": 0.0,
        "alert_support": round(alert_support, 6),
        "counterfactual_support": round(counterfactual_support, 6),
    }
    scores["overall_support"] = round(_weighted_support(scores), 6)
    return scores


def _weighted_support(scores: Dict[str, float]) -> float:
    total = 0.0
    for key, weight in DEFAULT_SUPPORT_WEIGHTS.items():
        total += float(scores.get(key, 0.0)) * float(weight)
    return max(0.0, min(1.0, total))


def _score_view(
    view_windows: Sequence[Dict[str, Any]],
    *,
    claim: Dict[str, Any],
    multimodal_cache: Dict[str, Any],
    use_reference_supervision: bool = True,
) -> Dict[str, float]:
    oracle_windows = _extract_oracle_windows(
        multimodal_cache,
        use_reference_supervision=use_reference_supervision,
    )
    target = _extract_target(
        multimodal_cache,
        use_reference_supervision=use_reference_supervision,
    )
    claim_existence = str(claim.get("existence") or target.get("existence") or "").strip().lower()
    if claim_existence == "normal" and not oracle_windows:
        return _fallback_normal_view_scores(view_windows, multimodal_cache)
    if not oracle_windows and not target:
        return _fallback_view_scores(view_windows)

    scores_by_role = _role_scores(view_windows, oracle_windows)
    precursor_score = scores_by_role.get("precursor", 0.0)
    trigger_score = scores_by_role.get("trigger", 0.0)
    peak_score = max(scores_by_role.get("peak_action", 0.0), scores_by_role.get("peak", 0.0))
    confirm_score = max(scores_by_role.get("confirmation", 0.0), scores_by_role.get("aftermath", 0.0))

    anomaly_interval = claim.get("anomaly_interval_sec") or target.get("anomaly_interval_sec")
    precursor_interval = claim.get("precursor_interval_sec") or target.get("precursor_interval_sec")
    anomaly_coverage = _coverage_ratio(view_windows, anomaly_interval)
    precursor_coverage = _coverage_ratio(view_windows, precursor_interval)

    if trigger_score >= 0.25 and (peak_score >= 0.25 or confirm_score >= 0.25):
        category_support = 0.92
    elif trigger_score >= 0.25:
        category_support = 0.78
    elif peak_score >= 0.25 and confirm_score >= 0.25:
        category_support = 0.74
    elif peak_score >= 0.25:
        category_support = 0.48
    elif confirm_score >= 0.25:
        category_support = 0.36
    elif precursor_score >= 0.25:
        category_support = 0.20
    elif anomaly_coverage > 0:
        category_support = 0.10
    else:
        category_support = 0.0

    temporal_support = min(1.0, 0.5 * anomaly_coverage + 0.5 * max(trigger_score, peak_score, confirm_score))
    exist_support = max(category_support, anomaly_coverage * 0.9)
    precursor_support = max(precursor_coverage, precursor_score * 0.8)
    if trigger_score >= 0.25:
        alert_support = 0.85
    elif peak_score >= 0.25 and confirm_score >= 0.25:
        alert_support = 0.72
    elif peak_score >= 0.25:
        alert_support = 0.45
    elif confirm_score >= 0.25:
        alert_support = 0.30
    elif precursor_score >= 0.25:
        alert_support = 0.15
    else:
        alert_support = 0.0

    counterfactual_support = min(1.0, 0.5 * category_support + 0.5 * temporal_support)

    scores = {
        "exist_support": round(exist_support, 6),
        "category_support": round(category_support, 6),
        "temporal_support": round(temporal_support, 6),
        "precursor_support": round(precursor_support, 6),
        "alert_support": round(alert_support, 6),
        "counterfactual_support": round(counterfactual_support, 6),
    }
    scores["overall_support"] = round(_weighted_support(scores), 6)
    return scores


def _derive_scores(view_scores: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    full_score = float(view_scores["full"].get("overall_support", 0.0))
    keep_score = float(view_scores["keep"].get("overall_support", 0.0))
    drop_score = float(view_scores["drop"].get("overall_support", 0.0))
    alert_score = float(view_scores["alert_prefix"].get("overall_support", 0.0))

    sufficiency = keep_score
    necessity = max(0.0, min(1.0, full_score - drop_score))
    consistency = max(0.0, min(1.0, 1.0 - abs(full_score - keep_score)))
    counterfactual_faithfulness = max(0.0, min(1.0, 0.5 * sufficiency + 0.5 * necessity))
    return {
        "sufficiency": round(sufficiency, 6),
        "necessity": round(necessity, 6),
        "consistency": round(consistency, 6),
        "alertability": round(alert_score, 6),
        "counterfactual_faithfulness": round(counterfactual_faithfulness, 6),
    }


def _view_payload(
    *,
    view_name: str,
    windows: Sequence[Dict[str, Any]],
    multimodal_cache: Dict[str, Any],
) -> Dict[str, Any]:
    frames = multimodal_cache.get("video")
    fps = float(multimodal_cache.get("fps") or 1.0)
    images = []
    timestamps: List[float] = []
    for window in windows:
        for timestamp in window.get("selected_timestamps") or []:
            timestamps.append(float(timestamp))
            if frames is None:
                continue
            frame_index = int(round(float(timestamp) * fps))
            frame_index = max(0, min(frame_index, len(frames) - 1))
            images.append(frames[frame_index])
    window_summary = "; ".join(
        f"{entry.get('window_id')}[{float(entry.get('start_sec', 0.0)):.3f},{float(entry.get('end_sec', 0.0)):.3f}]"
        for entry in windows
    ) or "no windows"
    return {
        "name": view_name,
        "window_ids": _window_ids(windows),
        "timestamps": timestamps[:8],
        "images": images[:8],
        "summary_text": f"windows={window_summary}",
    }


def _resolve_qwen_verifier_runtime(multimodal_cache: Dict[str, Any]) -> Optional[Any]:
    if multimodal_cache.get("verifier_runtime") is not None:
        return multimodal_cache["verifier_runtime"]

    model_path = str(multimodal_cache.get("verifier_model_path") or "")
    if not model_path:
        try:
            from saver_agent.qwen_verifier import DEFAULT_VERIFIER_MODEL_PATH

            model_path = DEFAULT_VERIFIER_MODEL_PATH
        except Exception:
            model_path = ""
    torch_dtype = str(multimodal_cache.get("verifier_torch_dtype") or "auto")
    device_map = str(multimodal_cache.get("verifier_device_map") or "auto")
    attn_implementation = str(multimodal_cache.get("verifier_attn_implementation") or "")
    max_new_tokens = int(multimodal_cache.get("verifier_max_new_tokens") or 512)
    cache_key = (model_path, torch_dtype, device_map, attn_implementation, max_new_tokens)
    if cache_key in _VERIFIER_RUNTIME_CACHE:
        runtime = _VERIFIER_RUNTIME_CACHE[cache_key]
        multimodal_cache["verifier_runtime"] = runtime
        return runtime

    try:
        from saver_agent.qwen_verifier import QwenSelfVerifier

        runtime = QwenSelfVerifier.from_pretrained(
            model_path=model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation=attn_implementation or None,
            max_new_tokens=max_new_tokens,
        )
    except Exception:
        return None

    _VERIFIER_RUNTIME_CACHE[cache_key] = runtime
    multimodal_cache["verifier_runtime"] = runtime
    return runtime


def _blend_view_scores(
    heuristic_scores: Dict[str, Dict[str, float]],
    qwen_scores: Dict[str, Dict[str, float]],
    *,
    alpha: float,
) -> Dict[str, Dict[str, float]]:
    alpha = max(0.0, min(1.0, float(alpha)))
    blended: Dict[str, Dict[str, float]] = {}
    for view_name in ("full", "keep", "drop", "alert_prefix"):
        blended[view_name] = {}
        for key in (
            "exist_support",
            "category_support",
            "temporal_support",
            "precursor_support",
            "alert_support",
            "counterfactual_support",
            "overall_support",
        ):
            heuristic_value = float((heuristic_scores.get(view_name) or {}).get(key, 0.0))
            qwen_value = float((qwen_scores.get(view_name) or {}).get(key, 0.0))
            blended[view_name][key] = round(alpha * heuristic_value + (1.0 - alpha) * qwen_value, 6)
    return blended


def _reduce_primary_status(
    *,
    view_scores: Dict[str, Dict[str, float]],
    derived_scores: Dict[str, float],
) -> str:
    keep_scores = view_scores["keep"]
    full_scores = view_scores["full"]
    drop_scores = view_scores["drop"]

    if keep_scores["exist_support"] < 0.2 and keep_scores["temporal_support"] < 0.2:
        return "misaligned"
    if keep_scores["category_support"] < 0.15 and keep_scores["overall_support"] < 0.35:
        return "misaligned"
    if keep_scores["overall_support"] < 0.55 and full_scores["overall_support"] >= 0.60:
        return "incomplete"
    if keep_scores["overall_support"] >= 0.60 and drop_scores["overall_support"] >= 0.65:
        return "redundant"
    if (
        keep_scores["overall_support"] >= 0.60
        and derived_scores["necessity"] >= 0.20
        and derived_scores["consistency"] >= 0.75
    ):
        return "complete"
    return "incomplete"


def _reduce_alert_status(
    *,
    alert: Optional[Dict[str, Any]],
    claim: Dict[str, Any],
    derived_scores: Dict[str, float],
    tolerance_sec: float = 0.5,
) -> str:
    if not alert:
        return "not_applicable"

    alert_sec = alert.get("alert_sec", alert.get("earliest_alert_sec"))
    try:
        alert_sec = float(alert_sec)
    except Exception:
        alert_sec = None

    expected_alert_sec = claim.get("earliest_alert_sec")
    try:
        expected_alert_sec = float(expected_alert_sec)
    except Exception:
        expected_alert_sec = None

    if derived_scores["alertability"] >= 0.65:
        return "justified"
    if alert_sec is not None and expected_alert_sec is not None and alert_sec > expected_alert_sec + tolerance_sec:
        return "late"
    return "premature"


def _recommended_action(primary_status: str, alert_status: str, verification_mode: str) -> str:
    if primary_status == "misaligned":
        return "revise_claim"
    if primary_status == "incomplete":
        return "continue_search"
    if primary_status == "redundant":
        return "refine_evidence"
    if primary_status == "complete":
        if verification_mode in {"final_check", "hard_alert_check", "full_keep_drop", "reward_only"}:
            return "finalize"
        if alert_status in {"justified", "not_applicable"}:
            return "finalize"
        return "continue_search"
    return "continue_search"


def _failure_reasons(primary_status: str, alert_status: str) -> List[str]:
    reasons: List[str] = []
    if primary_status == "incomplete":
        reasons.append("selected_evidence_not_sufficient")
    elif primary_status == "redundant":
        reasons.append("selected_evidence_not_necessary_enough")
    elif primary_status == "misaligned":
        reasons.append("selected_evidence_not_aligned_with_claim")

    if alert_status == "premature":
        reasons.append("alert_prefix_not_actionable")
    elif alert_status == "late":
        reasons.append("alert_after_expected_actionable_time")
    return reasons


def run_counterfactual_verifier(
    *,
    state: SaverEnvironmentState,
    multimodal_cache: Dict[str, Any],
    verification_mode: str,
    claim: Optional[Dict[str, Any]] = None,
    candidate_window_ids: Optional[Sequence[str]] = None,
    candidate_evidence_ids: Optional[Sequence[str]] = None,
    candidate_evidence_moment_ids: Optional[Sequence[str]] = None,
    alert: Optional[Dict[str, Any]] = None,
    query: str = "",
    backend: str = "heuristic",
    use_reference_supervision: bool = True,
) -> Dict[str, Any]:
    requested_backend = str(backend or "heuristic")
    actual_backend = "heuristic"
    claim = dict(claim or {})
    target = _extract_target(
        multimodal_cache,
        use_reference_supervision=use_reference_supervision,
    )
    merged_claim = {**target, **claim} if use_reference_supervision else dict(claim)
    selected_windows = _resolve_windows(
        state,
        candidate_window_ids=candidate_window_ids,
        candidate_evidence_ids=candidate_evidence_ids,
        candidate_evidence_moment_ids=candidate_evidence_moment_ids,
    )

    selected_window_ids = set(_window_ids(selected_windows))
    full_windows = list(state.evidence_ledger)
    if not full_windows:
        full_windows = [
            entry for entry in state.visited_windows
            if str(entry.get("kind") or "") != "scan"
        ]
    drop_windows = [entry for entry in full_windows if entry.get("window_id") not in selected_window_ids]

    alert_sec = None
    if alert:
        alert_value = alert.get("alert_sec", alert.get("earliest_alert_sec"))
        try:
            alert_sec = float(alert_value)
        except Exception:
            alert_sec = None
    if alert_sec is None:
        try:
            alert_sec = float(merged_claim.get("earliest_alert_sec"))
        except Exception:
            alert_sec = None
    if alert_sec is None:
        alert_prefix_windows = full_windows
    else:
        alert_prefix_windows = [entry for entry in full_windows if float(entry.get("start_sec", 0.0)) <= alert_sec]

    heuristic_view_scores = {
        "full": _score_view(
            full_windows,
            claim=merged_claim,
            multimodal_cache=multimodal_cache,
            use_reference_supervision=use_reference_supervision,
        ),
        "keep": _score_view(
            selected_windows,
            claim=merged_claim,
            multimodal_cache=multimodal_cache,
            use_reference_supervision=use_reference_supervision,
        ),
        "drop": _score_view(
            drop_windows,
            claim=merged_claim,
            multimodal_cache=multimodal_cache,
            use_reference_supervision=use_reference_supervision,
        ),
        "alert_prefix": _score_view(
            alert_prefix_windows,
            claim=merged_claim,
            multimodal_cache=multimodal_cache,
            use_reference_supervision=use_reference_supervision,
        ),
    }
    view_scores = heuristic_view_scores
    if requested_backend in {"qwen_self_verifier", "hybrid"}:
        runtime = _resolve_qwen_verifier_runtime(multimodal_cache)
        if runtime is not None:
            view_payloads = {
                "full": _view_payload(view_name="full", windows=full_windows, multimodal_cache=multimodal_cache),
                "keep": _view_payload(view_name="keep", windows=selected_windows, multimodal_cache=multimodal_cache),
                "drop": _view_payload(view_name="drop", windows=drop_windows, multimodal_cache=multimodal_cache),
                "alert_prefix": _view_payload(
                    view_name="alert_prefix", windows=alert_prefix_windows, multimodal_cache=multimodal_cache
                ),
            }
            qwen_view_scores = runtime.score_views(
                views=view_payloads,
                claim=merged_claim,
                verification_mode=verification_mode,
                question=str(multimodal_cache.get("question") or query),
            )
            if requested_backend == "qwen_self_verifier":
                view_scores = qwen_view_scores
                actual_backend = "qwen_self_verifier"
            else:
                alpha = float(multimodal_cache.get("verifier_hybrid_alpha") or 0.7)
                view_scores = _blend_view_scores(heuristic_view_scores, qwen_view_scores, alpha=alpha)
                actual_backend = "hybrid"
    derived_scores = _derive_scores(view_scores)
    primary_status = _reduce_primary_status(view_scores=view_scores, derived_scores=derived_scores)
    alert_status = _reduce_alert_status(
        alert=alert,
        claim=merged_claim,
        derived_scores=derived_scores,
    )
    recommended_action = _recommended_action(primary_status, alert_status, verification_mode)

    verified_window_ids = _window_ids(selected_windows)
    selectors_provided = bool(candidate_window_ids) or bool(candidate_evidence_ids) or bool(candidate_evidence_moment_ids)
    best_effort_window_ids = list(verified_window_ids)
    if not best_effort_window_ids and full_windows and not selectors_provided:
        best_effort_window_ids = _window_ids(full_windows[:1])

    failure_reasons = _failure_reasons(primary_status, alert_status)
    if requested_backend != actual_backend:
        failure_reasons.append("requested_backend_fell_back_to_heuristic")
    explanation = (
        f"Verification mode {verification_mode}: selected evidence is {primary_status} "
        f"and alert is {alert_status}."
    )

    verdict = {
        "verification_mode": verification_mode,
        "primary_status": primary_status,
        "alert_status": alert_status,
        "recommended_action": recommended_action,
        "view_scores": view_scores,
        "derived_scores": derived_scores,
        "verified_window_ids": verified_window_ids,
        "best_effort_window_ids": best_effort_window_ids,
        "failure_reasons": failure_reasons,
        "explanation": explanation,
        "requested_verifier_backend": requested_backend,
        "verifier_backend": actual_backend,
        "candidate_window_ids": list(candidate_window_ids or []),
        "candidate_evidence_ids": list(candidate_evidence_ids or []),
        "candidate_evidence_moment_ids": list(candidate_evidence_moment_ids or []),
        "query": query,
        "claim": merged_claim,
        "alert": dict(alert or {}),
        "use_reference_supervision": bool(use_reference_supervision),
    }
    if primary_status not in PRIMARY_STATUS_VALUES:
        raise ValueError(f"Unexpected verifier primary status: {primary_status}")
    if alert_status not in ALERT_STATUS_VALUES:
        raise ValueError(f"Unexpected verifier alert status: {alert_status}")
    return verdict


def _coerce_state_like(state_like: SaverEnvironmentState | Dict[str, Any]) -> SaverEnvironmentState:
    if isinstance(state_like, SaverEnvironmentState):
        return state_like
    payload = dict(state_like or {})
    return SaverEnvironmentState(
        visited_windows=list(payload.get("visited_windows") or []),
        evidence_ledger=list(payload.get("evidence_ledger") or []),
        alerts=list(payload.get("alerts") or []),
        verification_records=list(payload.get("verification_records") or []),
        finalized_case=dict(payload["finalized_case"]) if isinstance(payload.get("finalized_case"), dict) else None,
        last_claim=dict(payload["last_claim"]) if isinstance(payload.get("last_claim"), dict) else None,
        active_evidence_window_ids=list(payload.get("active_evidence_window_ids") or []),
        verifier_cache=list(payload.get("verifier_cache") or []),
        next_evidence_id=int(payload.get("next_evidence_id") or 1),
        next_window_id=int(payload.get("next_window_id") or 1),
        next_alert_id=int(payload.get("next_alert_id") or 1),
    )


def _group_relative_advantages(records: List[Dict[str, Any]], *, eps: float = 1e-6) -> List[Dict[str, Any]]:
    if not records:
        return []
    rewards = [float(record.get("branch_reward") or 0.0) for record in records]
    mean_reward = sum(rewards) / float(len(rewards))
    variance = sum((reward - mean_reward) ** 2 for reward in rewards) / float(len(rewards))
    std_reward = variance ** 0.5
    updated: List[Dict[str, Any]] = []
    for record, reward in zip(records, rewards):
        local_advantage = 0.0 if std_reward <= eps else (reward - mean_reward) / (std_reward + eps)
        enriched = dict(record)
        enriched["local_advantage"] = round(float(local_advantage), 6)
        enriched["group_reward_mean"] = round(float(mean_reward), 6)
        enriched["group_reward_std"] = round(float(std_reward), 6)
        updated.append(enriched)
    return updated


def _primary_status_bonus(primary_status: str) -> float:
    return {
        "complete": 0.3,
        "incomplete": -0.2,
        "redundant": -0.3,
        "misaligned": -0.6,
    }.get(str(primary_status), 0.0)


def _alert_status_bonus(alert_status: str) -> float:
    return {
        "justified": 0.4,
        "premature": -0.5,
        "late": -0.15,
        "not_applicable": 0.0,
    }.get(str(alert_status), 0.0)


def _evidence_ids_for_window_ids(state: SaverEnvironmentState, window_ids: Sequence[str]) -> List[str]:
    selected = set(str(value) for value in window_ids or [])
    if not selected:
        return []
    evidence_ids: List[str] = []
    for entry in state.evidence_ledger:
        window_id = str(entry.get("window_id") or "")
        evidence_id = entry.get("evidence_id")
        if window_id in selected and evidence_id:
            evidence_ids.append(str(evidence_id))
    return evidence_ids


def _choose_minimal_subset_window_ids(
    state: SaverEnvironmentState,
    selected_window_ids: Sequence[str],
    *,
    subset_size: int = 2,
) -> List[str]:
    entries = [entry for entry in state.evidence_ledger if str(entry.get("window_id") or "") in set(selected_window_ids or [])]
    if not entries:
        return []
    role_priority = {
        "trigger": 0,
        "peak_action": 1,
        "peak": 1,
        "precursor": 2,
        "confirmation": 3,
        "aftermath": 3,
        "": 4,
        "none": 4,
        None: 4,
    }
    entries = sorted(
        entries,
        key=lambda entry: (
            role_priority.get(str(entry.get("role") or "").lower(), 4),
            -float(entry.get("end_sec") or 0.0),
            str(entry.get("window_id") or ""),
        ),
    )
    keep = max(1, min(int(subset_size), len(entries)))
    return [str(entry.get("window_id")) for entry in entries[:keep] if entry.get("window_id")]


def _entry_identifier(entry: Dict[str, Any]) -> Tuple[str, str, float, float]:
    return (
        str(entry.get("window_id") or ""),
        str(entry.get("evidence_id") or ""),
        float(entry.get("start_sec") or 0.0),
        float(entry.get("end_sec") or 0.0),
    )


def _build_delta_only_state(
    state_before: SaverEnvironmentState,
    state_after: SaverEnvironmentState,
) -> SaverEnvironmentState:
    before_visited = {_entry_identifier(entry) for entry in state_before.visited_windows}
    before_evidence = {_entry_identifier(entry) for entry in state_before.evidence_ledger}
    delta_visited = [dict(entry) for entry in state_after.visited_windows if _entry_identifier(entry) not in before_visited]
    delta_evidence = [dict(entry) for entry in state_after.evidence_ledger if _entry_identifier(entry) not in before_evidence]
    delta_window_ids = [str(entry.get("window_id")) for entry in delta_evidence if entry.get("window_id")]
    return SaverEnvironmentState(
        visited_windows=delta_visited,
        evidence_ledger=delta_evidence,
        alerts=list(state_after.alerts),
        verification_records=list(state_after.verification_records),
        finalized_case=dict(state_after.finalized_case) if isinstance(state_after.finalized_case, dict) else None,
        last_claim=dict(state_after.last_claim) if isinstance(state_after.last_claim, dict) else None,
        active_evidence_window_ids=delta_window_ids,
        verifier_cache=list(state_after.verifier_cache),
        next_evidence_id=int(state_after.next_evidence_id),
        next_window_id=int(state_after.next_window_id),
        next_alert_id=int(state_after.next_alert_id),
    )


def _state_clip_ratio(state: SaverEnvironmentState, duration_sec: float) -> float:
    if duration_sec <= 1e-8:
        return 0.0
    intervals: List[Tuple[float, float]] = []
    source_entries = list(state.visited_windows or []) or list(state.evidence_ledger or [])
    for entry in source_entries:
        try:
            start_sec = float(entry.get("start_sec") or 0.0)
            end_sec = float(entry.get("end_sec") or start_sec)
        except Exception:
            continue
        if end_sec < start_sec:
            start_sec, end_sec = end_sec, start_sec
        if end_sec <= start_sec:
            continue
        intervals.append((start_sec, end_sec))
    if not intervals:
        return 0.0
    intervals.sort()
    merged: List[Tuple[float, float]] = []
    current_start, current_end = intervals[0]
    for start_sec, end_sec in intervals[1:]:
        if start_sec <= current_end:
            current_end = max(current_end, end_sec)
            continue
        merged.append((current_start, current_end))
        current_start, current_end = start_sec, end_sec
    merged.append((current_start, current_end))
    covered = sum(max(0.0, end_sec - start_sec) for start_sec, end_sec in merged)
    return min(1.0, max(0.0, covered / float(duration_sec)))


def score_alert_counterfactual_branch(
    *,
    state: SaverEnvironmentState | Dict[str, Any],
    multimodal_cache: Dict[str, Any],
    claim: Dict[str, Any],
    alert: Dict[str, Any],
    branch_type: str,
    anchor_turn_index: int,
    source_turn_index: Optional[int] = None,
    verifier_backend: str = "heuristic",
    use_reference_supervision: bool = False,
    delay_penalty: float = 0.10,
) -> Dict[str, Any]:
    runtime_state = _coerce_state_like(state)
    verdict = run_counterfactual_verifier(
        state=runtime_state,
        multimodal_cache=multimodal_cache,
        verification_mode="soft_alert_check",
        claim=claim,
        alert=alert,
        backend=verifier_backend,
        use_reference_supervision=use_reference_supervision,
    )
    derived = verdict.get("derived_scores") or {}
    reward_components = {
        "alertability_reward": 0.40 * float(derived.get("alertability", 0.0) or 0.0),
        "claim_support_reward": 0.30 * float(derived.get("sufficiency", 0.0) or 0.0),
        "alert_status_bonus": 0.20 * _alert_status_bonus(str(verdict.get("alert_status") or "")),
        "primary_status_bonus": 0.10 * {
            "complete": 0.2,
            "incomplete": -0.1,
            "redundant": -0.05,
            "misaligned": -0.4,
        }.get(str(verdict.get("primary_status") or ""), 0.0),
        "delay_penalty": 0.0,
    }
    earliest_alert = claim.get("earliest_alert_sec")
    alert_sec = (alert or {}).get("alert_sec", (alert or {}).get("earliest_alert_sec"))
    try:
        reward_components["delay_penalty"] = -abs(float(delay_penalty)) * max(0.0, float(alert_sec) - float(earliest_alert))
    except Exception:
        reward_components["delay_penalty"] = 0.0
    branch_reward = round(sum(float(value) for value in reward_components.values()), 6)
    selected_window_ids = list(verdict.get("verified_window_ids") or verdict.get("best_effort_window_ids") or [])
    return {
        "group_kind": "alert",
        "branch_type": str(branch_type),
        "anchor_turn_index": int(anchor_turn_index),
        "source_turn_index": int(source_turn_index if source_turn_index is not None else anchor_turn_index),
        "selected_window_ids": selected_window_ids,
        "selected_evidence_ids": _evidence_ids_for_window_ids(runtime_state, selected_window_ids),
        "synthetic_alert": dict(alert or {}),
        "branch_reward_components": reward_components,
        "branch_reward": branch_reward,
        "verifier_verdict": verdict,
        "primary_status": verdict.get("primary_status"),
        "alert_status": verdict.get("alert_status"),
    }


def score_alert_counterfactual_group(
    *,
    branches: Sequence[Dict[str, Any]],
    multimodal_cache: Dict[str, Any],
    verifier_backend: str = "heuristic",
    use_reference_supervision: bool = False,
    delay_penalty: float = 0.10,
) -> List[Dict[str, Any]]:
    records = [
        score_alert_counterfactual_branch(
            state=branch.get("state"),
            multimodal_cache=multimodal_cache,
            claim=dict(branch.get("claim") or {}),
            alert=dict(branch.get("alert") or {}),
            branch_type=str(branch.get("branch_type") or ""),
            anchor_turn_index=int(branch.get("anchor_turn_index") or 0),
            source_turn_index=branch.get("source_turn_index"),
            verifier_backend=verifier_backend,
            use_reference_supervision=use_reference_supervision,
            delay_penalty=delay_penalty,
        )
        for branch in branches
    ]
    return _group_relative_advantages(records)


def score_evidence_counterfactual_group(
    *,
    state: SaverEnvironmentState | Dict[str, Any],
    multimodal_cache: Dict[str, Any],
    claim: Dict[str, Any],
    selected_window_ids: Sequence[str],
    anchor_turn_index: int,
    alert: Optional[Dict[str, Any]] = None,
    verifier_backend: str = "heuristic",
    use_reference_supervision: bool = False,
    minimal_subset_window_ids: Optional[Sequence[str]] = None,
    subset_size_penalty: float = 0.10,
) -> List[Dict[str, Any]]:
    runtime_state = _coerce_state_like(state)
    full_window_ids = [str(entry.get("window_id")) for entry in runtime_state.evidence_ledger if entry.get("window_id")]
    keep_window_ids = [str(value) for value in selected_window_ids or [] if value]
    drop_window_ids = [window_id for window_id in full_window_ids if window_id not in set(keep_window_ids)]
    minimal_window_ids = (
        [str(value) for value in minimal_subset_window_ids or [] if value]
        if minimal_subset_window_ids
        else _choose_minimal_subset_window_ids(runtime_state, keep_window_ids)
    )
    branch_specs = [
        ("full_ledger", full_window_ids),
        ("keep_selected", keep_window_ids),
        ("drop_selected", drop_window_ids),
        ("minimal_subset", minimal_window_ids),
    ]
    records: List[Dict[str, Any]] = []
    full_ledger_size = max(1, len(full_window_ids))
    for branch_type, branch_window_ids in branch_specs:
        verdict = run_counterfactual_verifier(
            state=runtime_state,
            multimodal_cache=multimodal_cache,
            verification_mode="full_keep_drop",
            claim=claim,
            candidate_window_ids=branch_window_ids,
            alert=alert,
            backend=verifier_backend,
            use_reference_supervision=use_reference_supervision,
        )
        derived = verdict.get("derived_scores") or {}
        subset_ratio = float(len(branch_window_ids)) / float(full_ledger_size)
        reward_components = {
            "sufficiency_reward": 0.35 * float(derived.get("sufficiency", 0.0) or 0.0),
            "necessity_reward": 0.35 * float(derived.get("necessity", 0.0) or 0.0),
            "counterfactual_reward": 0.20 * float(derived.get("counterfactual_faithfulness", 0.0) or 0.0),
            "primary_status_bonus": 0.10 * _primary_status_bonus(str(verdict.get("primary_status") or "")),
            "subset_size_penalty": -abs(float(subset_size_penalty)) * subset_ratio,
        }
        branch_reward = round(sum(float(value) for value in reward_components.values()), 6)
        records.append(
            {
                "group_kind": "evidence",
                "branch_type": branch_type,
                "anchor_turn_index": int(anchor_turn_index),
                "source_turn_index": int(anchor_turn_index),
                "selected_window_ids": list(branch_window_ids),
                "selected_evidence_ids": _evidence_ids_for_window_ids(runtime_state, branch_window_ids),
                "branch_reward_components": reward_components,
                "branch_reward": branch_reward,
                "verifier_verdict": verdict,
                "primary_status": verdict.get("primary_status"),
                "alert_status": verdict.get("alert_status"),
            }
        )
    return _group_relative_advantages(records)


def score_search_counterfactual_group(
    *,
    state_before: SaverEnvironmentState | Dict[str, Any],
    state_after: SaverEnvironmentState | Dict[str, Any],
    multimodal_cache: Dict[str, Any],
    claim: Dict[str, Any],
    anchor_turn_index: int,
    alert: Optional[Dict[str, Any]] = None,
    verifier_backend: str = "heuristic",
    use_reference_supervision: bool = False,
    search_cost_penalty: float = 0.10,
) -> List[Dict[str, Any]]:
    runtime_before = _coerce_state_like(state_before)
    runtime_after = _coerce_state_like(state_after)
    delta_only_state = _build_delta_only_state(runtime_before, runtime_after)
    duration_sec = float(multimodal_cache.get("duration") or 0.0)
    branch_specs = [
        ("skip_search", runtime_before),
        ("use_search", runtime_after),
        ("delta_only", delta_only_state),
    ]
    records: List[Dict[str, Any]] = []
    for branch_type, branch_state in branch_specs:
        verdict = run_counterfactual_verifier(
            state=branch_state,
            multimodal_cache=multimodal_cache,
            verification_mode="search_step_check",
            claim=claim,
            alert=alert,
            backend=verifier_backend,
            use_reference_supervision=use_reference_supervision,
        )
        derived = verdict.get("derived_scores") or {}
        clip_ratio = _state_clip_ratio(branch_state, duration_sec)
        reward_components = {
            "sufficiency_reward": 0.35 * float(derived.get("sufficiency", 0.0) or 0.0),
            "necessity_reward": 0.20 * float(derived.get("necessity", 0.0) or 0.0),
            "counterfactual_reward": 0.20 * float(derived.get("counterfactual_faithfulness", 0.0) or 0.0),
            "alertability_reward": 0.10 * float(derived.get("alertability", 0.0) or 0.0),
            "primary_status_bonus": 0.15 * _primary_status_bonus(str(verdict.get("primary_status") or "")),
            "search_cost_penalty": -abs(float(search_cost_penalty)) * clip_ratio,
        }
        branch_reward = round(sum(float(value) for value in reward_components.values()), 6)
        selected_window_ids = (
            list(verdict.get("verified_window_ids") or verdict.get("best_effort_window_ids") or [])
            or [str(value) for value in branch_state.active_evidence_window_ids if value]
        )
        records.append(
            {
                "group_kind": "search",
                "branch_type": branch_type,
                "anchor_turn_index": int(anchor_turn_index),
                "source_turn_index": int(anchor_turn_index),
                "selected_window_ids": selected_window_ids,
                "selected_evidence_ids": _evidence_ids_for_window_ids(branch_state, selected_window_ids),
                "branch_reward_components": reward_components,
                "branch_reward": branch_reward,
                "verifier_verdict": verdict,
                "primary_status": verdict.get("primary_status"),
                "alert_status": verdict.get("alert_status"),
            }
        )
    return _group_relative_advantages(records)
