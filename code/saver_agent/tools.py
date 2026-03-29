from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from saver_agent.proposal import coerce_feature_cache_payload, feature_guided_frame_proposal, normalize_query_text
from saver_agent.schema import SaverEnvironmentState, validate_required_fields
from saver_agent.verifier import run_counterfactual_verifier


MAX_NUM_KEY_FRAMES = 8


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


def _append_window(
    state: SaverEnvironmentState,
    *,
    kind: str,
    query: str | None,
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


def scan_timeline(arguments: Dict[str, Any], multimodal_cache: Dict, state: SaverEnvironmentState):
    start_sec, end_sec, selected_indices, fps = _resolve_window(arguments, multimodal_cache)
    entry = _append_window(
        state,
        kind="scan",
        query=arguments.get("purpose"),
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
    query = str(arguments.get("query") or "")
    feature_cache = coerce_feature_cache_payload(
        multimodal_cache.get("embedding"),
        fps=fps,
        frame_indices=multimodal_cache.get("frame_indices") or [],
    )
    proposal_metadata = feature_guided_frame_proposal(
        feature_cache=feature_cache,
        proposal_runtime=multimodal_cache.get("proposal_runtime"),
        query=query,
        start_sec=start_sec,
        end_sec=end_sec,
        fps=fps,
        num_frames=max(1, len(selected_indices)),
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
    footer = (
        f"Searched evidence for query '{query}' in window [{start_sec:.3f}, {end_sec:.3f}] "
        f"and selected {len(selected_indices)} frames."
    )
    return _build_visual_content(selected_indices, multimodal_cache, footer), state, entry


def emit_alert(arguments: Dict[str, Any], multimodal_cache: Dict, state: SaverEnvironmentState):
    alert_id = f"a{state.next_alert_id:04d}"
    state.next_alert_id += 1
    alert = {
        "alert_id": alert_id,
        "decision": arguments.get("decision"),
        "existence": arguments.get("existence"),
        "category": arguments.get("category"),
        "earliest_alert_sec": arguments.get("earliest_alert_sec"),
        "alert_sec": arguments.get("alert_sec", arguments.get("earliest_alert_sec")),
        "reason": arguments.get("reason"),
    }
    state.alerts.append(alert)
    if isinstance(arguments.get("claim"), dict):
        state.last_claim = dict(arguments["claim"])
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
    verification = run_counterfactual_verifier(
        state=state,
        multimodal_cache=multimodal_cache,
        verification_mode=str(arguments.get("verification_mode") or "reward_only"),
        claim=arguments.get("claim") or state.last_claim or {},
        candidate_window_ids=arguments.get("candidate_window_ids"),
        candidate_evidence_ids=arguments.get("candidate_evidence_ids") or arguments.get("evidence_ids"),
        candidate_evidence_moment_ids=arguments.get("evidence_moment_ids"),
        alert=arguments.get("alert") or (state.alerts[-1] if state.alerts else None),
        query=str(arguments.get("query") or ""),
        backend=str(arguments.get("verifier_backend") or multimodal_cache.get("verifier_backend") or "heuristic"),
        use_reference_supervision=False,
    )
    state.last_claim = dict(verification.get("claim") or {})
    state.active_evidence_window_ids = list(verification.get("verified_window_ids") or [])
    state.verification_records.append(verification)
    state.verifier_cache.append(verification)
    content = [{"type": "text", "text": json.dumps(verification, ensure_ascii=False)}]
    return content, state, verification


def finalize_case(arguments: Dict[str, Any], multimodal_cache: Dict, state: SaverEnvironmentState):
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
