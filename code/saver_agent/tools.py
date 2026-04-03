from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from saver_agent.categories import canonicalize_category_payload, validate_canonical_category_payload
from saver_agent.proposal import (
    coerce_feature_cache_payload,
    feature_guided_frame_proposal,
    normalize_query_package,
    normalize_query_text,
    summarize_query_package,
)
from saver_agent.schema import SaverEnvironmentState, validate_required_fields
from saver_agent.self_verification import (
    normalize_self_verification_mode,
    parse_self_verification_payload,
    validate_policy_self_verification_payload,
)
from saver_agent.verifier import run_counterfactual_verifier


MAX_NUM_KEY_FRAMES = 8
SELF_VERIFICATION_VERDICT_KEYS = (
    "verification_decision",
    "primary_status",
    "alert_status",
    "recommended_action",
    "sufficiency_score",
    "necessity_score",
    "alertability_score",
    "counterfactual_faithfulness",
    "derived_scores",
    "failure_reasons",
    "rationale",
    "explanation",
)


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _frame_bounds(multimodal_cache: Dict) -> Tuple[int, float, int]:
    frames = multimodal_cache.get("video")
    fps = float(multimodal_cache.get("fps") or 1.0)
    if frames is not None:
        total_frames = int(frames.shape[0])
    else:
        duration = float(multimodal_cache.get("duration") or 0.0)
        total_frames = max(int(math.ceil(duration * fps)), 1)
    return total_frames, fps, max(total_frames - 1, 0)


def _select_uniform_indices(start_idx: int, end_idx: int, num_frames: int) -> List[int]:
    if end_idx < start_idx:
        return []
    total = end_idx - start_idx + 1
    num_frames = max(1, min(int(num_frames), total, MAX_NUM_KEY_FRAMES))
    if total <= num_frames:
        return list(range(start_idx, end_idx + 1))
    return np.round(np.linspace(start_idx, end_idx, num_frames)).astype(int).tolist()


def _select_stride_indices(start_idx: int, end_idx: int, stride_frames: int) -> List[int]:
    if end_idx < start_idx:
        return []
    stride_frames = max(1, int(stride_frames))
    indices = list(range(start_idx, end_idx + 1, stride_frames))
    if not indices or indices[-1] != end_idx:
        indices.append(end_idx)
    indices = list(dict.fromkeys(int(index) for index in indices))
    if len(indices) > MAX_NUM_KEY_FRAMES:
        return _select_uniform_indices(indices[0], indices[-1], MAX_NUM_KEY_FRAMES)
    return indices


def _resolve_window(args: Dict[str, Any], multimodal_cache: Dict) -> Tuple[float, float, List[int], float]:
    total_frames, fps, last_frame_idx = _frame_bounds(multimodal_cache)
    duration = float(multimodal_cache.get("duration") or (total_frames / fps))
    start_sec = max(0.0, _coerce_float(args.get("start_sec", args.get("start_time", 0.0)), 0.0))
    end_sec = _coerce_float(args.get("end_sec", args.get("end_time", duration)), duration)
    end_sec = min(max(start_sec, end_sec), duration)
    start_idx = max(0, min(int(math.floor(start_sec * fps)), last_frame_idx))
    end_idx = max(start_idx, min(int(math.floor(end_sec * fps)), last_frame_idx))
    if args.get("stride_sec") is not None:
        stride_sec = _coerce_float(args.get("stride_sec"), 0.0)
        stride_frames = max(1, int(round(stride_sec * fps))) if stride_sec > 0 else 1
        selected_indices = _select_stride_indices(start_idx, end_idx, stride_frames)
    else:
        num_frames = _coerce_int(args.get("num_frames", MAX_NUM_KEY_FRAMES), MAX_NUM_KEY_FRAMES)
        selected_indices = _select_uniform_indices(start_idx, end_idx, num_frames)
    return start_sec, end_sec, selected_indices, fps


def _indices_to_timestamps(indices: List[int], fps: float) -> List[float]:
    return [round(float(idx) / fps, 6) for idx in indices]


def _build_minimal_claim_from_alert_payload(alert_payload: Dict[str, Any]) -> Dict[str, Any]:
    claim: Dict[str, Any] = {}
    existence = str(alert_payload.get("existence") or "").strip()
    category = str(alert_payload.get("category") or "").strip()
    earliest_alert_sec = alert_payload.get("earliest_alert_sec", alert_payload.get("alert_sec"))
    if existence:
        claim["existence"] = existence
    if category:
        claim["category"] = category
    if earliest_alert_sec is not None:
        claim["earliest_alert_sec"] = float(earliest_alert_sec)
    return claim


def _build_visual_content(indices: List[int], multimodal_cache: Dict, footer: str) -> List[Dict[str, Any]]:
    fps = float(multimodal_cache.get("fps") or 1.0)
    frames = multimodal_cache.get("video")
    frame_indices = multimodal_cache.get("frame_indices") or []
    timestamps = _indices_to_timestamps(indices, fps)
    content: List[Dict[str, Any]] = []
    for i, timestamp in zip(indices, timestamps):
        content.append({"type": "text", "text": f"{timestamp:.3f}s"})
        if frames is not None and 0 <= i < len(frames):
            image_item = {
                "type": "image",
                "image": frames[i],
                "sampled_frame_index": int(i),
                "timestamp_sec": float(timestamp),
            }
            if 0 <= int(i) < len(frame_indices):
                image_item["raw_frame_index"] = int(frame_indices[int(i)])
            content.append(image_item)
    content.append({"type": "text", "text": footer})
    return content


def _dedupe_string_list(values: List[Any] | Tuple[Any, ...] | None) -> List[str]:
    deduped: List[str] = []
    seen = set()
    for value in values or []:
        text = str(value).strip()
        if not text or text in seen:
            continue
        deduped.append(text)
        seen.add(text)
    return deduped


def _resolve_legacy_selected_window_alias(
    raw_window_id: str,
    *,
    evidence_window_ids: List[str],
    compact_alias_mode: bool,
) -> str | None:
    if not evidence_window_ids:
        return None

    evidence_match = re.fullmatch(r"evidence_(\d+)", raw_window_id)
    if evidence_match:
        index = int(evidence_match.group(1))
        if 0 <= index < len(evidence_window_ids):
            return evidence_window_ids[index]
        return None

    zero_based_window_match = re.fullmatch(r"w_(\d+)", raw_window_id)
    if zero_based_window_match:
        index = int(zero_based_window_match.group(1))
        if 0 <= index < len(evidence_window_ids):
            return evidence_window_ids[index]
        return None

    compact_window_match = re.fullmatch(r"w(\d+)", raw_window_id)
    if compact_alias_mode and compact_window_match:
        index = int(compact_window_match.group(1)) - 1
        if 0 <= index < len(evidence_window_ids):
            return evidence_window_ids[index]

    return None


def _append_window(
    state: SaverEnvironmentState,
    *,
    kind: str,
    query: str | None,
    query_package: Dict[str, Any] | None,
    query_normalized: str | None,
    query_source: str | None,
    moment_id: str | None,
    role: str | None,
    start_sec: float,
    end_sec: float,
    selected_indices: List[int],
    fps: float,
    record_as_evidence: bool = True,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    window_id = f"w{state.next_window_id:04d}"
    state.next_window_id += 1
    evidence_id = f"e{state.next_evidence_id:04d}"
    state.next_evidence_id += 1
    entry = {
        "window_id": window_id,
        "evidence_id": evidence_id,
        "kind": kind,
        "query": query,
        "query_package": dict(query_package or {}),
        "query_normalized": query_normalized,
        "query_source": query_source,
        "moment_id": moment_id,
        "role": role,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "selected_frame_indices": [int(index) for index in selected_indices],
        "selected_timestamps": _indices_to_timestamps(selected_indices, fps),
        "selected_frame_count": len(selected_indices),
    }
    if metadata:
        entry.update(dict(metadata))
    state.visited_windows.append(entry)
    if record_as_evidence:
        state.evidence_ledger.append(entry)
    return entry


def _has_self_verification_payload(arguments: Dict[str, Any]) -> bool:
    return any(key in arguments for key in SELF_VERIFICATION_VERDICT_KEYS)


def _looks_like_legacy_verification_request(arguments: Dict[str, Any]) -> bool:
    for key in (
        "claim",
        "alert",
        "query",
        "candidate_window_ids",
        "candidate_evidence_ids",
        "candidate_evidence_moment_ids",
        "evidence_ids",
        "evidence_moment_ids",
    ):
        value = arguments.get(key)
        if isinstance(value, dict) and value:
            return True
        if isinstance(value, (list, tuple, set)) and value:
            return True
        if isinstance(value, str) and value.strip():
            return True
    return False


def _normalize_verification_arguments(arguments: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(arguments or {})
    normalized["verification_mode"] = normalize_self_verification_mode(
        normalized.get("verification_mode"),
        default="reward_only",
    )
    if isinstance(normalized.get("claim"), dict):
        normalized["claim"] = validate_canonical_category_payload(
            normalized["claim"],
            payload_name="claim",
            require_category_for_anomaly=True,
        )
    if isinstance(normalized.get("alert"), dict):
        normalized["alert"] = validate_canonical_category_payload(
            normalized["alert"],
            payload_name="alert",
        )
    return normalized


def _resolve_selected_window_ids(
    state: SaverEnvironmentState,
    *,
    selected_window_ids: List[str],
    selected_evidence_ids: List[str],
    selected_evidence_moment_ids: List[str],
    candidate_window_ids: List[str],
) -> Dict[str, Any]:
    requested_selected_window_ids = _dedupe_string_list(selected_window_ids)
    selected_evidence_ids = _dedupe_string_list(selected_evidence_ids)
    selected_evidence_moment_ids = _dedupe_string_list(selected_evidence_moment_ids)
    candidate_window_ids = _dedupe_string_list(candidate_window_ids)
    raw_window_selector_ids = _dedupe_string_list(requested_selected_window_ids + candidate_window_ids)

    resolved: List[str] = []
    seen = set()
    invalid_selected_window_ids: List[str] = []
    resolution_sources: List[str] = []
    evidence_window_ids = [
        str(entry.get("window_id")).strip()
        for entry in state.evidence_ledger
        if str(entry.get("window_id") or "").strip()
    ]
    compact_alias_mode = bool(evidence_window_ids and evidence_window_ids[0] != "w0001")
    valid_window_ids = {
        str(entry.get("window_id")).strip()
        for entry in state.evidence_ledger
        if str(entry.get("window_id") or "").strip()
    }
    by_evidence_id = {
        str(entry.get("evidence_id")): entry for entry in state.evidence_ledger if entry.get("evidence_id")
    }
    by_moment_id: Dict[str, List[Dict[str, Any]]] = {}
    for entry in state.evidence_ledger:
        moment_id = entry.get("moment_id")
        if moment_id is None:
            continue
        by_moment_id.setdefault(str(moment_id), []).append(entry)

    selected_window_added = False
    for raw_window_id in requested_selected_window_ids:
        resolved_window_id = _resolve_legacy_selected_window_alias(
            raw_window_id,
            evidence_window_ids=evidence_window_ids,
            compact_alias_mode=compact_alias_mode,
        )
        if resolved_window_id is None and raw_window_id in valid_window_ids:
            resolved_window_id = raw_window_id
        if resolved_window_id is None:
            if raw_window_id not in invalid_selected_window_ids:
                invalid_selected_window_ids.append(raw_window_id)
            continue
        if resolved_window_id not in valid_window_ids:
            if raw_window_id not in invalid_selected_window_ids:
                invalid_selected_window_ids.append(raw_window_id)
            continue
        if resolved_window_id not in seen:
            seen.add(resolved_window_id)
            resolved.append(resolved_window_id)
            selected_window_added = True
    if selected_window_added:
        resolution_sources.append("selected_window_ids")

    evidence_added = False
    for evidence_id in selected_evidence_ids:
        entry = by_evidence_id.get(str(evidence_id))
        if entry is None:
            continue
        window_id = str(entry.get("window_id") or "").strip()
        if window_id and window_id not in seen:
            seen.add(window_id)
            resolved.append(window_id)
            evidence_added = True
    if evidence_added:
        resolution_sources.append("selected_evidence_ids")

    moment_added = False
    for moment_id in selected_evidence_moment_ids:
        for entry in by_moment_id.get(str(moment_id), []):
            window_id = str(entry.get("window_id") or "").strip()
            if window_id and window_id not in seen:
                seen.add(window_id)
                resolved.append(window_id)
                moment_added = True
    if moment_added:
        resolution_sources.append("selected_evidence_moment_ids")

    valid_candidate_window_ids: List[str] = []
    candidate_added = False
    candidate_fallback_allowed = not resolved
    for window_id in candidate_window_ids:
        if window_id not in valid_window_ids:
            continue
        if window_id not in valid_candidate_window_ids:
            valid_candidate_window_ids.append(window_id)
        if not candidate_fallback_allowed:
            continue
        if window_id not in seen:
            seen.add(window_id)
            resolved.append(window_id)
            candidate_added = True
    if candidate_added:
        resolution_sources.append("candidate_window_ids")

    selection_requested = bool(
        requested_selected_window_ids or selected_evidence_ids or selected_evidence_moment_ids or candidate_window_ids
    )
    if resolution_sources:
        selection_resolution_source = "+".join(resolution_sources)
    elif selection_requested:
        selection_resolution_source = "unresolved"
    else:
        selection_resolution_source = "none"

    return {
        "resolved_window_ids": resolved,
        "requested_selected_window_ids": requested_selected_window_ids,
        "invalid_selected_window_ids": invalid_selected_window_ids,
        "selection_resolution_source": selection_resolution_source,
        "selection_requested": selection_requested,
        "selection_unresolved": bool(selection_requested and not resolved),
        "selected_evidence_ids": selected_evidence_ids,
        "selected_evidence_moment_ids": selected_evidence_moment_ids,
        "valid_candidate_window_ids": valid_candidate_window_ids,
        "window_selector_ids": raw_window_selector_ids,
    }


def _finalize_verification_payload(
    verification: Dict[str, Any],
    *,
    selection_info: Dict[str, Any],
    verification_parse_mode: str,
    legacy_compatibility_used: bool,
) -> Dict[str, Any]:
    finalized = dict(verification or {})
    finalized["verified_window_ids"] = _dedupe_string_list(finalized.get("verified_window_ids") or [])
    finalized["best_effort_window_ids"] = _dedupe_string_list(finalized.get("best_effort_window_ids") or [])
    if finalized.get("self_verification_selected_window_ids") is not None:
        finalized["self_verification_selected_window_ids"] = _dedupe_string_list(
            finalized.get("self_verification_selected_window_ids") or []
        )

    failure_reasons = _dedupe_string_list(finalized.get("failure_reasons") or [])
    if bool(selection_info.get("selection_unresolved")):
        failure_reasons = _dedupe_string_list(
            failure_reasons + ["selected_evidence_not_resolved_to_known_windows"]
        )
        finalized["verified_window_ids"] = []
        finalized["best_effort_window_ids"] = []
        if finalized.get("self_verification_selected_window_ids") is not None:
            finalized["self_verification_selected_window_ids"] = []
    finalized["failure_reasons"] = failure_reasons
    finalized["verification_parse_mode"] = str(verification_parse_mode)
    finalized["requested_selected_window_ids"] = list(selection_info.get("requested_selected_window_ids") or [])
    finalized["invalid_selected_window_ids"] = list(selection_info.get("invalid_selected_window_ids") or [])
    finalized["selection_resolution_source"] = str(selection_info.get("selection_resolution_source") or "none")
    finalized["legacy_compatibility_used"] = bool(legacy_compatibility_used)
    if bool(legacy_compatibility_used):
        finalized["legacy_request_compatibility_used"] = True
    return finalized


def scan_timeline(arguments: Dict[str, Any], multimodal_cache: Dict, state: SaverEnvironmentState):
    start_sec, end_sec, selected_indices, fps = _resolve_window(arguments, multimodal_cache)
    entry = _append_window(
        state,
        kind="scan",
        query=arguments.get("purpose"),
        query_package=None,
        query_normalized=normalize_query_text(str(arguments.get("purpose") or "")),
        query_source="scan_purpose",
        moment_id=None,
        role=None,
        start_sec=start_sec,
        end_sec=end_sec,
        selected_indices=selected_indices,
        fps=fps,
        record_as_evidence=False,
        metadata={
            "proposal_backend": "uniform",
            "feature_cache_used": False,
            "proposal_candidate_count": 0,
            "proposal_candidate_frame_indices": [int(index) for index in selected_indices],
            "proposal_candidate_frame_scores": [],
            "proposal_candidate_windows": [],
            "selected_frame_scores": [],
            "proposal_fallback_reason": "scan_timeline_uniform",
        },
    )
    footer = (
        f"Scanned timeline window [{start_sec:.3f}, {end_sec:.3f}] and selected "
        f"{len(selected_indices)} frames."
    )
    return _build_visual_content(selected_indices, multimodal_cache, footer), state, entry


def seek_evidence(arguments: Dict[str, Any], multimodal_cache: Dict, state: SaverEnvironmentState):
    start_sec, end_sec, selected_indices, fps = _resolve_window(arguments, multimodal_cache)
    query_package = normalize_query_package(
        arguments.get("query_package"),
        fallback_query=str(arguments.get("query") or ""),
        rewrite_reason=str(arguments.get("query_source") or "model"),
    )
    query = summarize_query_package(query_package)
    feature_cache = coerce_feature_cache_payload(
        multimodal_cache.get("embedding"),
        fps=fps,
        frame_indices=multimodal_cache.get("frame_indices") or [],
    )
    proposal_metadata = feature_guided_frame_proposal(
        feature_cache=feature_cache,
        proposal_runtime=multimodal_cache.get("proposal_runtime"),
        query=query,
        query_package=query_package,
        start_sec=start_sec,
        end_sec=end_sec,
        fps=fps,
        num_frames=int(arguments.get("num_frames") or 0),
        top_k_candidates=int(arguments.get("top_k_candidates") or 8),
        candidate_merge_gap_sec=float(arguments.get("candidate_merge_gap_sec") or 1.0),
        query_source=str(arguments.get("query_source") or "model"),
    )
    if proposal_metadata.get("selected_frame_indices"):
        selected_indices = [int(index) for index in proposal_metadata["selected_frame_indices"]]
    else:
        proposal_metadata["selected_frame_indices"] = [int(index) for index in selected_indices]
    entry = _append_window(
        state,
        kind="evidence",
        query=query,
        query_package=query_package,
        query_normalized=str(proposal_metadata.get("query_normalized") or normalize_query_text(query)),
        query_source=str(proposal_metadata.get("query_source") or arguments.get("query_source") or "model"),
        moment_id=str(arguments.get("moment_id")) if arguments.get("moment_id") is not None else None,
        role=str(arguments.get("role")) if arguments.get("role") is not None else None,
        start_sec=start_sec,
        end_sec=end_sec,
        selected_indices=selected_indices,
        fps=fps,
        record_as_evidence=True,
        metadata=proposal_metadata,
    )
    evidence_window_id = str(entry.get("window_id") or "").strip()
    evidence_id = str(entry.get("evidence_id") or "").strip()
    moment_id = str(entry.get("moment_id") or "").strip()
    role = str(entry.get("role") or "").strip()
    registration_parts = []
    if evidence_window_id:
        registration_parts.append(f"window_id={evidence_window_id}")
    if evidence_id:
        registration_parts.append(f"evidence_id={evidence_id}")
    if role:
        registration_parts.append(f"role={role}")
    if moment_id:
        registration_parts.append(f"moment_id={moment_id}")
    verification_hint = ""
    if evidence_window_id:
        verification_hint = f' If you later call verify_hypothesis on this evidence, use selected_window_ids=["{evidence_window_id}"].'
        if moment_id:
            verification_hint += f' You may also include selected_evidence_moment_ids=["{moment_id}"].'
    footer = (
        f"Registered evidence {' '.join(registration_parts)}. "
        f"Searched evidence for query '{query}' in window [{start_sec:.3f}, {end_sec:.3f}] "
        f"and selected {len(selected_indices)} frames."
        f"{verification_hint}"
    )
    return _build_visual_content(selected_indices, multimodal_cache, footer), state, entry


def emit_alert(arguments: Dict[str, Any], multimodal_cache: Dict, state: SaverEnvironmentState):
    normalized_arguments = validate_canonical_category_payload(
        arguments,
        payload_name="alert",
        require_category_for_anomaly=True,
    )
    alert_id = f"a{state.next_alert_id:04d}"
    state.next_alert_id += 1
    alert = {
        "alert_id": alert_id,
        "decision": normalized_arguments.get("decision"),
        "existence": normalized_arguments.get("existence"),
        "category": normalized_arguments.get("category"),
        "earliest_alert_sec": normalized_arguments.get("earliest_alert_sec"),
        "alert_sec": normalized_arguments.get("alert_sec", normalized_arguments.get("earliest_alert_sec")),
        "reason": normalized_arguments.get("reason"),
    }
    state.alerts.append(alert)
    state.last_claim = _build_minimal_claim_from_alert_payload(alert)
    content = [
        {
            "type": "text",
            "text": json.dumps(
                {
                    "status": "alert_recorded",
                    "alert_index": len(state.alerts) - 1,
                    "alert": alert,
                },
                ensure_ascii=False,
            ),
        }
    ]
    return content, state, alert


def verify_hypothesis(arguments: Dict[str, Any], multimodal_cache: Dict, state: SaverEnvironmentState):
    arguments = _normalize_verification_arguments(arguments)
    selected_window_ids = [str(value) for value in (arguments.get("selected_window_ids") or []) if str(value).strip()]
    selected_evidence_ids = [str(value) for value in (arguments.get("selected_evidence_ids") or []) if str(value).strip()]
    selected_evidence_moment_ids = [
        str(value)
        for value in (
            arguments.get("selected_evidence_moment_ids")
            or arguments.get("evidence_moment_ids")
            or []
        )
        if str(value).strip()
    ]
    candidate_window_ids = [str(value) for value in (arguments.get("candidate_window_ids") or []) if str(value).strip()]
    candidate_evidence_ids = [
        str(value)
        for value in (arguments.get("candidate_evidence_ids") or arguments.get("evidence_ids") or [])
        if str(value).strip()
    ]
    candidate_evidence_moment_ids = [
        str(value)
        for value in (
            arguments.get("candidate_evidence_moment_ids")
            or arguments.get("evidence_moment_ids")
            or arguments.get("selected_evidence_moment_ids")
            or []
        )
        if str(value).strip()
    ]
    selection_info = _resolve_selected_window_ids(
        state,
        selected_window_ids=selected_window_ids,
        selected_evidence_ids=selected_evidence_ids,
        selected_evidence_moment_ids=selected_evidence_moment_ids,
        candidate_window_ids=candidate_window_ids,
    )

    has_self_verification_payload = _has_self_verification_payload(arguments)
    looks_like_legacy_request = _looks_like_legacy_verification_request(arguments)
    allow_legacy_verify_compatibility = bool(multimodal_cache.get("allow_legacy_verify_compatibility"))
    allow_external_verifier_fallback = bool(
        arguments.get("allow_external_verifier_fallback")
        or multimodal_cache.get("allow_external_verifier_fallback")
    )
    disable_external_verifier_fallback = bool(multimodal_cache.get("disable_external_verifier_fallback"))

    if has_self_verification_payload:
        payload = dict(arguments)
        payload["selected_window_ids"] = list(selection_info["resolved_window_ids"])
        payload["selected_evidence_ids"] = list(selection_info["selected_evidence_ids"])
        payload["selected_evidence_moment_ids"] = list(selection_info["selected_evidence_moment_ids"])
        payload["candidate_window_ids"] = list(selection_info["valid_candidate_window_ids"])
        payload = validate_policy_self_verification_payload(payload)
        verification = parse_self_verification_payload(
            payload,
            fallback_claim=arguments.get("claim") or state.last_claim or {},
            fallback_alert=arguments.get("alert") or (state.alerts[-1] if state.alerts else None),
            verification_mode=str(arguments.get("verification_mode") or "reward_only"),
        )
        verification = _finalize_verification_payload(
            verification,
            selection_info=selection_info,
            verification_parse_mode="self_report",
            legacy_compatibility_used=False,
        )
    elif looks_like_legacy_request:
        if allow_legacy_verify_compatibility:
            verification_parse_mode = "legacy_compatibility"
            legacy_compatibility_used = True
        elif disable_external_verifier_fallback:
            raise ValueError(
                "External verifier fallback is disabled for this rollout path. "
                "verify_hypothesis must use the policy self-verification payload."
            )
        elif allow_external_verifier_fallback:
            verification_parse_mode = "external_verifier_fallback"
            legacy_compatibility_used = False
        else:
            raise ValueError(
                "verify_hypothesis now requires a policy-produced self-verification payload. "
                "External verifier fallback is diagnostic-only and must be enabled explicitly."
            )
        legacy_candidate_window_ids = (
            list(selection_info["resolved_window_ids"])
            if selection_info["resolved_window_ids"]
            else list(selection_info["window_selector_ids"])
        )
        verification = run_counterfactual_verifier(
            state=state,
            multimodal_cache=multimodal_cache,
            verification_mode=str(arguments.get("verification_mode") or "reward_only"),
            claim=arguments.get("claim") or state.last_claim or {},
            candidate_window_ids=legacy_candidate_window_ids,
            candidate_evidence_ids=list(selection_info["selected_evidence_ids"] or candidate_evidence_ids),
            candidate_evidence_moment_ids=list(
                selection_info["selected_evidence_moment_ids"] or candidate_evidence_moment_ids
            ),
            alert=arguments.get("alert") or (state.alerts[-1] if state.alerts else None),
            query=str(arguments.get("query") or ""),
            backend=str(arguments.get("verifier_backend") or multimodal_cache.get("verifier_backend") or "heuristic"),
            use_reference_supervision=False,
        )
        verification = _finalize_verification_payload(
            verification,
            selection_info=selection_info,
            verification_parse_mode=verification_parse_mode,
            legacy_compatibility_used=legacy_compatibility_used,
        )
    else:
        raise ValueError(
            "verify_hypothesis must provide either a verdict-bearing self-verification payload "
            "or a legacy verifier request with explicit compatibility enabled."
        )
    state.last_claim = dict(verification.get("claim") or {})
    state.active_evidence_window_ids = list(
        verification.get("verified_window_ids") or verification.get("best_effort_window_ids") or []
    )
    state.verification_records.append(verification)
    state.verifier_cache.append(verification)
    content = [{"type": "text", "text": json.dumps(verification, ensure_ascii=False)}]
    return content, state, verification


def finalize_case(arguments: Dict[str, Any], multimodal_cache: Dict, state: SaverEnvironmentState):
    arguments = validate_canonical_category_payload(
        arguments,
        payload_name="finalize_case",
        require_category_for_anomaly=True,
    )
    schema = multimodal_cache.get("tool_io", {}).get("finalize_case_schema")
    validate_required_fields(arguments, schema)
    state.finalized_case = dict(arguments)
    state.last_claim = dict(arguments)
    content = [
        {
            "type": "text",
            "text": json.dumps(
                {"status": "finalized", "finalized_case": state.finalized_case},
                ensure_ascii=False,
            ),
        }
    ]
    return content, state, state.finalized_case
