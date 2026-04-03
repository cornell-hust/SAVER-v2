#!/usr/bin/env python3
"""Convert canonical SAVER-style annotations into agent-ready training views.

This script keeps the benchmark-facing canonical JSONL untouched and derives
stable training/evaluation views for a TimeSearch-R-style SAVER agent.

Current adapters:
- ``msad_saver_qwen``: canonical records from
  ``msad_saver_with_qwen.jsonl``

Output modes:
- ``canonical_passthrough``: enrich canonical records with derived second-based
  fields while preserving the original structure.
- ``agent_train``: add stable task/schema/tool supervision for the future agent.
- ``oracle_sft``: add a heuristic oracle trajectory for warm-start SFT.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from split_utils import parse_include_splits
from saver_agent.categories import CANONICAL_POLICY_CATEGORIES, canonicalize_saver_category
from saver_agent.proposal import (
    build_proposal_supervision,
    normalize_query_package,
    normalize_query_text,
    select_query_for_moment,
)
from saver_agent.self_verification import build_policy_self_verification_payload


SCHEMA_VERSION = "saver_agent.v2"
ALLOWED_TOOLS = [
    "scan_timeline",
    "seek_evidence",
    "emit_alert",
    "verify_hypothesis",
    "finalize_case",
]

FINALIZE_CASE_SCHEMA = {
    "type": "object",
    "properties": {
        "existence": {"type": "string", "enum": ["normal", "anomaly"]},
        "category": {"type": "string", "enum": list(CANONICAL_POLICY_CATEGORIES)},
        "severity": {"type": "integer"},
        "anomaly_interval_sec": {
            "oneOf": [
                {"type": "null"},
                {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                },
            ]
        },
        "precursor_interval_sec": {
            "oneOf": [
                {"type": "null"},
                {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                },
            ]
        },
        "earliest_alert_sec": {"oneOf": [{"type": "null"}, {"type": "number"}]},
        "evidence_moment_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "counterfactual_type": {"type": "string"},
        "summary": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "required": [
        "existence",
        "category",
        "severity",
        "anomaly_interval_sec",
        "precursor_interval_sec",
        "earliest_alert_sec",
        "evidence_moment_ids",
        "counterfactual_type",
        "summary",
        "rationale",
    ],
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def round6(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), 6)


def frame_to_second(
    frame: Optional[int],
    *,
    fps: float,
    frame_index_base: int,
    duration_sec: Optional[float] = None,
) -> Optional[float]:
    """Convert a frame index to the start time of that frame."""
    if frame is None:
        return None
    second = (float(frame) - float(frame_index_base)) / float(fps)
    if duration_sec is not None:
        second = clamp(second, 0.0, float(duration_sec))
    return round6(second)


def frame_interval_to_seconds(
    interval_frames: Optional[List[int]],
    *,
    fps: float,
    frame_index_base: int,
    duration_sec: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """Convert an inclusive frame interval into a second interval.

    The end timestamp follows an inclusive-end policy:
    ``end_sec = (end_frame - base + 1) / fps``.
    This gives the right duration for inclusive frame annotations and behaves
    well with retrieval tools that operate on seconds.
    """
    if not interval_frames:
        return None, None
    start_frame, end_frame = int(interval_frames[0]), int(interval_frames[1])
    start_sec = frame_to_second(
        start_frame, fps=fps, frame_index_base=frame_index_base, duration_sec=duration_sec
    )
    end_sec = (float(end_frame) - float(frame_index_base) + 1.0) / float(fps)
    if duration_sec is not None:
        end_sec = clamp(end_sec, 0.0, float(duration_sec))
    return start_sec, round6(end_sec)


def ensure_frame_interval(interval_frames: Optional[List[int]]) -> Optional[List[int]]:
    if not interval_frames:
        return None
    start_frame, end_frame = int(interval_frames[0]), int(interval_frames[1])
    if start_frame > end_frame:
        start_frame, end_frame = end_frame, start_frame
    return [start_frame, end_frame]


def union_frame_intervals(intervals: Iterable[List[int]]) -> Optional[List[int]]:
    cleaned = [ensure_frame_interval(interval) for interval in intervals if interval]
    cleaned = [interval for interval in cleaned if interval]
    if not cleaned:
        return None
    start_frame = min(interval[0] for interval in cleaned)
    end_frame = max(interval[1] for interval in cleaned)
    return [start_frame, end_frame]


def _is_strict_precursor_interval(
    interval_frames: Optional[List[int]],
    *,
    anomaly_start_frame: Optional[int],
) -> bool:
    interval = ensure_frame_interval(interval_frames)
    if interval is None or anomaly_start_frame is None:
        return interval is not None
    # The second-domain interval uses an exclusive end timestamp:
    # end_sec = (end_frame - base + 1) / fps. Therefore a precursor that ends
    # on the same frame as anomaly_start already overlaps the anomaly in time.
    return int(interval[1]) < int(anomaly_start_frame)


def _sanitize_qa_pairs(
    qa_pairs: Any,
    *,
    precursor_interval_sec: Optional[List[float]],
    precursor_resolution_source: Optional[str],
) -> List[Dict[str, Any]]:
    cleaned_pairs: List[Dict[str, Any]] = []
    keep_precursor_temporal = (
        precursor_interval_sec is not None and str(precursor_resolution_source or "") == "annotation"
    )
    for qa in list(qa_pairs or []):
        if not isinstance(qa, dict):
            continue
        if str(qa.get("type") or "") == "precursor_temporal" and not keep_precursor_temporal:
            continue
        cleaned_pairs.append(dict(qa))
    return cleaned_pairs


def _normalize_evidence_role(
    role: Any,
    *,
    start_frame: int,
    end_frame: int,
    anomaly_start_frame: Optional[int],
) -> str:
    normalized_role = str(role or "unspecified")
    if normalized_role != "precursor" or anomaly_start_frame is None:
        return normalized_role
    if _is_strict_precursor_interval(
        [start_frame, end_frame],
        anomaly_start_frame=anomaly_start_frame,
    ):
        return normalized_role
    if int(start_frame) >= int(anomaly_start_frame):
        return "trigger"
    return "evidence"


def normalize_evidence_moment(
    moment: Dict[str, Any],
    *,
    fps: float,
    frame_index_base: int,
    duration_sec: float,
    anomaly_start_frame: Optional[int] = None,
) -> Dict[str, Any]:
    start_frame = int(moment["start_frame"])
    end_frame = int(moment["end_frame"])
    start_sec, end_sec = frame_interval_to_seconds(
        [start_frame, end_frame],
        fps=fps,
        frame_index_base=frame_index_base,
        duration_sec=duration_sec,
    )
    normalized_role = _normalize_evidence_role(
        moment.get("role"),
        start_frame=start_frame,
        end_frame=end_frame,
        anomaly_start_frame=anomaly_start_frame,
    )
    return {
        "moment_id": moment.get("moment_id", f"{normalized_role or 'ev'}_{start_frame}_{end_frame}"),
        "role": normalized_role,
        "description": moment.get("description"),
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_sec": start_sec,
        "end_sec": end_sec,
    }


def complete_precursor_interval(
    record: Dict[str, Any],
    *,
    heuristic_seconds: float = 2.0,
    heuristic_fraction: float = 0.2,
) -> Dict[str, Any]:
    """Resolve or synthesize a precursor interval for anomaly videos."""
    temporal = record["temporal"]
    label = record["label"]
    fps = float(record["video_meta"]["fps"])
    duration_sec = float(record["video_meta"]["duration_sec"])
    frame_index_base = int(record.get("frame_index_base", 0))

    if not label.get("is_anomaly", False):
        return {
            "frames": None,
            "seconds": None,
            "source": "not_applicable_normal",
            "auto_completed": False,
        }

    annotated = ensure_frame_interval(temporal.get("precursor_interval_frames"))
    anomaly_interval = ensure_frame_interval(temporal.get("anomaly_interval_frames"))
    anomaly_start = anomaly_interval[0] if anomaly_interval else None
    saw_explicit_non_strict_precursor = False

    if annotated:
        if _is_strict_precursor_interval(annotated, anomaly_start_frame=anomaly_start):
            seconds = frame_interval_to_seconds(
                annotated,
                fps=fps,
                frame_index_base=frame_index_base,
                duration_sec=duration_sec,
            )
            return {
                "frames": annotated,
                "seconds": list(seconds),
                "source": "annotation",
                "auto_completed": False,
            }
        saw_explicit_non_strict_precursor = True

    evidence_moments = record.get("evidence", {}).get("evidence_moments", [])

    precursor_intervals = []
    preceding_intervals = []
    for moment in evidence_moments:
        interval = ensure_frame_interval([moment["start_frame"], moment["end_frame"]])
        if moment.get("role") == "precursor":
            if _is_strict_precursor_interval(interval, anomaly_start_frame=anomaly_start):
                precursor_intervals.append(interval)
            else:
                saw_explicit_non_strict_precursor = True
        if anomaly_start is not None and interval[1] < anomaly_start:
            preceding_intervals.append(interval)

    if precursor_intervals:
        resolved = union_frame_intervals(precursor_intervals)
        seconds = frame_interval_to_seconds(
            resolved,
            fps=fps,
            frame_index_base=frame_index_base,
            duration_sec=duration_sec,
        )
        return {
            "frames": resolved,
            "seconds": list(seconds),
            "source": "evidence_precursor_role",
            "auto_completed": False,
        }

    if saw_explicit_non_strict_precursor:
        return {
            "frames": None,
            "seconds": None,
            "source": "non_strict_precursor_dropped",
            "auto_completed": False,
        }

    if preceding_intervals:
        resolved = union_frame_intervals(preceding_intervals)
        seconds = frame_interval_to_seconds(
            resolved,
            fps=fps,
            frame_index_base=frame_index_base,
            duration_sec=duration_sec,
        )
        return {
            "frames": resolved,
            "seconds": list(seconds),
            "source": "evidence_preceding_event",
            "auto_completed": True,
        }

    if anomaly_start is None:
        return {
            "frames": None,
            "seconds": None,
            "source": "missing_without_anomaly_interval",
            "auto_completed": True,
        }

    anomaly_length = anomaly_interval[1] - anomaly_interval[0] + 1
    heuristic_frames = max(
        int(round(float(heuristic_seconds) * fps)),
        int(round(float(heuristic_fraction) * anomaly_length)),
    )
    end_frame = anomaly_start - 1
    start_frame = max(frame_index_base, end_frame - heuristic_frames + 1)
    if end_frame < start_frame:
        return {
            "frames": None,
            "seconds": None,
            "source": "heuristic_unavailable",
            "auto_completed": True,
        }

    resolved = [start_frame, end_frame]
    seconds = frame_interval_to_seconds(
        resolved,
        fps=fps,
        frame_index_base=frame_index_base,
        duration_sec=duration_sec,
    )
    return {
        "frames": resolved,
        "seconds": list(seconds),
        "source": "heuristic_preceding_window",
        "auto_completed": True,
    }


@dataclass
class ConverterConfig:
    heuristic_seconds: float = 2.0
    heuristic_fraction: float = 0.2


class CanonicalSaverAdapter:
    """Adapter for canonical SAVER benchmark records."""

    name = "msad_saver_qwen"

    def __init__(self, config: ConverterConfig):
        self.config = config

    def convert(self, record: Dict[str, Any], mode: str) -> Dict[str, Any]:
        base = self._build_base_view(record)
        if mode == "canonical_passthrough":
            return base
        if mode == "agent_train":
            return self._build_agent_train_view(base)
        if mode == "oracle_sft":
            agent_view = self._build_agent_train_view(base)
            agent_view["oracle_sft"] = self._build_oracle_sft(agent_view)
            return agent_view
        raise ValueError(f"Unsupported conversion mode: {mode}")

    def _build_base_view(self, record: Dict[str, Any]) -> Dict[str, Any]:
        fps = float(record["video_meta"]["fps"])
        duration_sec = float(record["video_meta"]["duration_sec"])
        frame_index_base = int(record.get("frame_index_base", 0))

        anomaly_interval_frames = ensure_frame_interval(record["temporal"].get("anomaly_interval_frames"))
        anomaly_start_frame = anomaly_interval_frames[0] if anomaly_interval_frames else None
        anomaly_interval_sec = (
            list(
                frame_interval_to_seconds(
                    anomaly_interval_frames,
                    fps=fps,
                    frame_index_base=frame_index_base,
                    duration_sec=duration_sec,
                )
            )
            if anomaly_interval_frames
            else None
        )

        precursor_info = complete_precursor_interval(
            record,
            heuristic_seconds=self.config.heuristic_seconds,
            heuristic_fraction=self.config.heuristic_fraction,
        )
        earliest_alert_frame = record["temporal"].get("earliest_alert_frame")
        earliest_alert_sec = frame_to_second(
            earliest_alert_frame,
            fps=fps,
            frame_index_base=frame_index_base,
            duration_sec=duration_sec,
        )

        evidence_moments = [
            normalize_evidence_moment(
                moment,
                fps=fps,
                frame_index_base=frame_index_base,
                duration_sec=duration_sec,
                anomaly_start_frame=anomaly_start_frame,
            )
            for moment in record.get("evidence", {}).get("evidence_moments", [])
        ]

        base = {
            "schema_version": SCHEMA_VERSION,
            "record_origin": self.name,
            "video_id": record["video_id"],
            "file_name": record["file_name"],
            "video_path": record["video_path"],
            "source_dataset": record["source_dataset"],
            "source_split": record["source_split"],
            "split": record["split"],
            "frame_index_base": frame_index_base,
            "video_meta": {
                **record["video_meta"],
                "time_basis": {
                    "frame_index_base": frame_index_base,
                    "interval_end_policy": "inclusive_frame_end_converted_to_exclusive_second_end",
                },
            },
            "scene": record.get("scene", {}),
            "key_objects": record.get("key_objects", []),
            "label": {
                **record["label"],
                "category": canonicalize_saver_category(
                    record.get("label", {}).get("category"),
                    existence="anomaly" if record.get("label", {}).get("is_anomaly") else "normal",
                ),
            },
            "temporal": {
                **record["temporal"],
                "anomaly_interval_sec": anomaly_interval_sec,
                "precursor_interval_frames": precursor_info["frames"],
                "precursor_interval_sec": precursor_info["seconds"],
                "earliest_alert_sec": earliest_alert_sec,
                "precursor_resolution": {
                    "source": precursor_info["source"],
                    "auto_completed": precursor_info["auto_completed"],
                },
            },
            "evidence": {
                **record.get("evidence", {}),
                "evidence_moments": evidence_moments,
            },
            "counterfactual": record.get("counterfactual", {}),
            "language": record.get("language", {}),
            "qa_pairs": _sanitize_qa_pairs(
                record.get("qa_pairs", []),
                precursor_interval_sec=precursor_info["seconds"],
                precursor_resolution_source=precursor_info["source"],
            ),
            "provenance": record.get("provenance", {}),
            "qwen_preannotation": record.get("qwen_preannotation", {}),
            "auto_completed": {
                "precursor_interval": bool(precursor_info["auto_completed"]),
            },
        }
        base["proposal_supervision"] = build_proposal_supervision(
            key_objects=base.get("key_objects") or [],
            evidence_moments=evidence_moments,
        )
        return base

    def _build_agent_train_view(self, base: Dict[str, Any]) -> Dict[str, Any]:
        label = base["label"]
        temporal = base["temporal"]
        evidence_moments = base["evidence"]["evidence_moments"]
        language = base.get("language") or {}
        existence = "anomaly" if label["is_anomaly"] else "normal"
        anomaly_interval_sec = temporal.get("anomaly_interval_sec")
        precursor_interval_sec = temporal.get("precursor_interval_sec")
        earliest_alert_sec = temporal.get("earliest_alert_sec")

        task_prompt = self._build_task_prompt(base)
        structured_target = {
            "existence": existence,
            "category": canonicalize_saver_category(label["category"], existence=existence),
            "severity": label["severity"],
            "hard_normal": label["hard_normal"],
            "anomaly_interval_sec": anomaly_interval_sec,
            "precursor_interval_sec": precursor_interval_sec,
            "earliest_alert_sec": earliest_alert_sec,
            "evidence_moment_ids": [moment["moment_id"] for moment in evidence_moments],
            "evidence_windows_sec": [
                {
                    "moment_id": moment["moment_id"],
                    "role": moment["role"],
                    "window_sec": [moment["start_sec"], moment["end_sec"]],
                }
                for moment in evidence_moments
            ],
            "counterfactual_type": base["counterfactual"].get("type", "none"),
            "counterfactual_text": base["counterfactual"].get("text"),
            "summary": language.get("summary"),
            "rationale": language.get("rationale"),
        }
        tool_io = {
            "allowed_tools": ALLOWED_TOOLS,
            "initial_scan_window_frames": [
                base["frame_index_base"],
                int(base["video_meta"]["total_frames"]) + base["frame_index_base"] - 1,
            ],
            "initial_scan_window_sec": [0.0, round6(base["video_meta"]["duration_sec"])],
            "oracle_windows_frames": [
                {
                    "moment_id": moment["moment_id"],
                    "role": moment["role"],
                    "window": [moment["start_frame"], moment["end_frame"]],
                    "description": moment["description"],
                }
                for moment in evidence_moments
            ],
            "oracle_windows_sec": [
                {
                    "moment_id": moment["moment_id"],
                    "role": moment["role"],
                    "window": [moment["start_sec"], moment["end_sec"]],
                    "description": moment["description"],
                }
                for moment in evidence_moments
            ],
            "finalize_case_schema": FINALIZE_CASE_SCHEMA,
        }
        agent_task = {
            "task_type": "video_anomaly_search_alert_verify",
            "query_mode": "internal_hypothesis_generation",
            "task_prompt": task_prompt,
            "success_criteria": [
                "Discover whether an actionable anomaly exists under limited search.",
                "Alert early but avoid false alerts on normal videos.",
                "Ground the decision in visited evidence only.",
                "Verify the current evidence subset before finalizing.",
            ],
        }
        base["agent_task"] = agent_task
        base["structured_target"] = structured_target
        base["tool_io"] = tool_io
        return base

    def _build_task_prompt(self, base: Dict[str, Any]) -> str:
        scene = base.get("scene", {}).get("scenario")
        duration = round6(base["video_meta"]["duration_sec"])
        target_text = (
            "Determine whether the video contains an actionable anomaly or is normal, search for precursor and trigger "
            "evidence when relevant, raise a soft or hard alert only when justified, verify that the searched evidence "
            "is sufficient, and then return a structured final decision."
        )
        return (
            f"Video duration: {duration} seconds. "
            f"Scene: {scene}. "
            f"{target_text}"
        )

    @staticmethod
    def _select_proposal_query_group(
        moment: Dict[str, Any],
        *,
        proposal_supervision: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(proposal_supervision, dict):
            return None
        moment_id = str(moment.get("moment_id") or "")
        role = str(moment.get("role") or "")
        moment_tokens = {
            token
            for token in normalize_query_text(f"{moment.get('description') or ''} {role}").split()
            if token
        }
        best_group: Optional[Dict[str, Any]] = None
        best_score = -1.0
        for query_group in proposal_supervision.get("queries") or []:
            linked_moment_ids = {str(value) for value in query_group.get("linked_moment_ids") or [] if value}
            linked_roles = {str(value) for value in query_group.get("linked_roles") or [] if value}
            if moment_id and linked_moment_ids and moment_id not in linked_moment_ids and role not in linked_roles:
                continue
            score = 0.0
            if moment_id and moment_id in linked_moment_ids:
                score += 4.0
            if role and role in linked_roles:
                score += 2.0
            best_entry_score = 0.0
            for entry in query_group.get("normalized_queries") or []:
                text = str(entry.get("text") or "").strip()
                if not text:
                    continue
                query_tokens = {
                    token
                    for token in normalize_query_text(text).split()
                    if token
                }
                overlap = len(query_tokens & moment_tokens) / float(max(len(query_tokens), 1)) if query_tokens else 0.0
                best_entry_score = max(best_entry_score, float(entry.get("weight") or 0.0) + overlap)
            score += best_entry_score
            if score > best_score:
                best_score = score
                best_group = dict(query_group)
        return best_group

    @staticmethod
    def _generic_hypothesis_for_role(role: str) -> str:
        if role == "precursor":
            return "suspected pre-anomaly cue"
        if role in {"trigger", "peak_action"}:
            return "suspected actionable event"
        if role == "confirmation":
            return "suspected anomaly confirmation"
        if role == "normal_check":
            return "check whether this interval remains normal"
        if role == "context":
            return "suspected anomaly context"
        return "suspected anomaly evidence"

    def _build_generic_query_package(
        self,
        base: Dict[str, Any],
        *,
        role: str,
        query_text: str,
        query_source: str,
        key_objects: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        scene_context = str((base.get("scene") or {}).get("scenario") or "").strip()
        return normalize_query_package(
            {
                "event_cue": str(query_text or "").strip(),
                "key_objects": list(key_objects or []),
                "scene_context": scene_context,
                "hypothesis": self._generic_hypothesis_for_role(role),
                "negative_constraints": [],
                "rewrite_reason": str(query_source or "role_fallback"),
            },
            fallback_query=str(query_text or "").strip(),
            fallback_scene_context=scene_context,
            rewrite_reason=str(query_source or "role_fallback"),
        )

    def _build_query_package_for_moment(
        self,
        base: Dict[str, Any],
        *,
        moment: Dict[str, Any],
        query_text: str,
        query_source: str,
        proposal_supervision: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        role = str(moment.get("role") or "")
        query_group = self._select_proposal_query_group(moment, proposal_supervision=proposal_supervision)
        normalized_entries = list((query_group or {}).get("normalized_queries") or [])
        event_candidates = [
            str(entry.get("text") or "").strip()
            for entry in normalized_entries
            if str(entry.get("kind") or "").strip() == "event_relation" and str(entry.get("text") or "").strip()
        ]
        key_object_candidates = [
            str(entry.get("text") or "").strip()
            for entry in normalized_entries
            if str(entry.get("kind") or "").strip() in {"object", "attribute_object"}
            and str(entry.get("text") or "").strip()
        ]
        event_cue = event_candidates[0] if event_candidates else str(query_text or "").strip()
        if not key_object_candidates and query_text and query_text != event_cue:
            key_object_candidates = [str(query_text).strip()]
        return self._build_generic_query_package(
            base,
            role=role,
            query_text=event_cue or str(query_text or "").strip(),
            query_source=query_source,
            key_objects=key_object_candidates[:4],
        )

    def _build_oracle_sft(self, base: Dict[str, Any]) -> Dict[str, Any]:
        trajectory: List[Dict[str, Any]] = []
        language = base.get("language") or {}
        structured_target = base.get("structured_target") or {}
        duration = round6(base["video_meta"]["duration_sec"])
        next_window_id = 1
        next_evidence_id = 1
        searched_real_moments: List[Dict[str, Any]] = []
        searched_real_refs: List[Dict[str, Any]] = []
        searched_supplemental_refs: List[Dict[str, Any]] = []
        normal_search_refs: List[Dict[str, Any]] = []

        def append_step(
            step: Dict[str, Any],
            *,
            ref_bucket: Optional[List[Dict[str, Any]]] = None,
        ) -> Optional[Dict[str, Any]]:
            nonlocal next_window_id, next_evidence_id
            trajectory.append(step)
            tool_name = str(step.get("tool") or "")
            if tool_name not in {"scan_timeline", "seek_evidence"}:
                return None
            runtime_ref = {
                "window_id": f"w{next_window_id:04d}",
                "evidence_id": f"e{next_evidence_id:04d}",
                "tool": tool_name,
                "moment_id": (step.get("arguments") or {}).get("moment_id"),
                "role": (step.get("arguments") or {}).get("role"),
            }
            next_window_id += 1
            next_evidence_id += 1
            if ref_bucket is not None:
                ref_bucket.append(runtime_ref)
            return runtime_ref

        append_step(
            {
                "tool": "scan_timeline",
                "arguments": {
                    "start_sec": 0.0,
                    "end_sec": duration,
                    "stride_sec": max(round6(duration / 8.0) or 0.5, 0.5),
                    "purpose": "global_overview",
                },
            }
        )

        evidence_moments = base["evidence"]["evidence_moments"]
        label = base["label"]
        temporal = base["temporal"]

        if label["is_anomaly"]:
            category = canonicalize_saver_category(label.get("category") or "anomaly", existence="anomaly")
            sorted_moments = self._sorted_evidence_moments(evidence_moments)
            used_window_keys = {
                self._window_key(moment.get("start_sec"), moment.get("end_sec"))
                for moment in sorted_moments
            }

            proposal_supervision = base.get("proposal_supervision")

            def append_real_seek(moment: Dict[str, Any]) -> None:
                runtime_ref = append_step(
                    self._seek_evidence_step(
                        base,
                        moment,
                        category=category,
                        proposal_supervision=proposal_supervision,
                    ),
                    ref_bucket=searched_real_refs,
                )
                searched_real_moments.append(moment)
                if runtime_ref is not None:
                    runtime_ref["moment_id"] = moment.get("moment_id")
                    runtime_ref["role"] = moment.get("role")

            def append_supplemental_seeks(count: int) -> List[Dict[str, Any]]:
                added_refs: List[Dict[str, Any]] = []
                steps = self._supplemental_seek_steps(
                    base,
                    category=category,
                    used_window_keys=used_window_keys,
                    count=count,
                )
                for step in steps:
                    runtime_ref = append_step(step, ref_bucket=searched_supplemental_refs)
                    if runtime_ref is not None:
                        added_refs.append(runtime_ref)
                return added_refs

            remaining_real_moments = list(sorted_moments)
            primary_moment = remaining_real_moments.pop(0) if remaining_real_moments else None
            if primary_moment is not None:
                append_real_seek(primary_moment)

            trajectory.append(
                {
                    "tool": "emit_alert",
                    "arguments": {
                        "decision": "soft_alert",
                        "existence": "anomaly",
                        "category": category,
                        "earliest_alert_sec": temporal["earliest_alert_sec"],
                    },
                }
            )

            if remaining_real_moments:
                trajectory.append(
                    {
                        "tool": "verify_hypothesis",
                        "arguments": {
                            "verification_mode": "soft_alert_check",
                            "evidence_moment_ids": [primary_moment["moment_id"]] if primary_moment is not None else [],
                            "claim": {
                                "existence": "anomaly",
                                "category": category,
                            },
                        },
                        "oracle_verifier_feedback": self._oracle_verifier_feedback(
                            verification_mode="soft_alert_check",
                            verification_decision="misaligned",
                            primary_status="misaligned",
                            alert_status="premature",
                            recommended_action="revise_claim",
                            selected_refs=searched_real_refs[:1],
                            selected_evidence_moment_ids=[primary_moment["moment_id"]] if primary_moment is not None else [],
                            sufficiency_score=0.18,
                            necessity_score=0.24,
                            alertability_score=0.12,
                            counterfactual_faithfulness=0.21,
                            failure_reasons=[
                                "current_claim_is_more_specific_than_the_observed_precursor_evidence",
                                "alert_prefix_not_actionable",
                            ],
                            rationale=(
                                "The current evidence only supports a tentative suspicious precursor, so the anomaly "
                                "claim should be revised before continuing."
                            ),
                        ),
                    }
                )

            first_support_batch = remaining_real_moments[:2]
            remaining_real_moments = remaining_real_moments[2:]
            for moment in first_support_batch:
                append_real_seek(moment)

            first_support_needed = max(0, 2 - len(first_support_batch))
            append_supplemental_seeks(first_support_needed)

            selected_subset_ids = [
                str(moment.get("moment_id"))
                for moment in searched_real_moments[:2]
                if moment.get("moment_id") is not None
            ]
            if selected_subset_ids:
                trajectory.append(
                    {
                        "tool": "verify_hypothesis",
                        "arguments": {
                            "verification_mode": "full_keep_drop",
                            "evidence_moment_ids": selected_subset_ids,
                            "claim": {
                                "existence": "anomaly",
                                "category": category,
                                "earliest_alert_sec": temporal.get("earliest_alert_sec"),
                            },
                        },
                        "oracle_verifier_feedback": self._oracle_verifier_feedback(
                            verification_mode="full_keep_drop",
                            verification_decision="insufficient",
                            primary_status="incomplete",
                            alert_status="premature",
                            recommended_action="continue_search",
                            selected_refs=searched_real_refs[:2],
                            selected_evidence_moment_ids=selected_subset_ids,
                            sufficiency_score=0.47,
                            necessity_score=0.31,
                            alertability_score=0.34,
                            counterfactual_faithfulness=0.39,
                            failure_reasons=[
                                "selected_evidence_not_sufficient",
                                "alert_prefix_not_actionable",
                            ],
                            rationale=(
                                "The compact evidence subset is promising but still incomplete, so more decisive or "
                                "confirmatory evidence should be searched before finalizing."
                            ),
                        ),
                    }
                )

            if remaining_real_moments:
                append_real_seek(remaining_real_moments[0])
            else:
                append_supplemental_seeks(1)

            if len(searched_real_refs) >= 3:
                redundant_context_refs = append_supplemental_seeks(1)
                if redundant_context_refs:
                    trajectory.append(
                        {
                            "tool": "verify_hypothesis",
                            "arguments": {
                                "verification_mode": "full_keep_drop",
                                "evidence_moment_ids": [
                                    str(moment.get("moment_id"))
                                    for moment in searched_real_moments
                                    if moment.get("moment_id") is not None
                                ],
                                "claim": {
                                    "existence": "anomaly",
                                    "category": category,
                                    "earliest_alert_sec": temporal.get("earliest_alert_sec"),
                                },
                            },
                            "oracle_verifier_feedback": self._oracle_verifier_feedback(
                                verification_mode="full_keep_drop",
                                verification_decision="redundant",
                                primary_status="redundant",
                                alert_status="premature",
                                recommended_action="refine_evidence",
                                selected_refs=searched_real_refs + redundant_context_refs,
                                selected_evidence_moment_ids=[
                                    str(moment.get("moment_id"))
                                    for moment in searched_real_moments
                                    if moment.get("moment_id") is not None
                                ],
                                sufficiency_score=0.86,
                                necessity_score=0.18,
                                alertability_score=0.62,
                                counterfactual_faithfulness=0.52,
                                failure_reasons=[
                                    "selected_evidence_not_necessary_enough",
                                    "alert_prefix_not_actionable",
                                ],
                                rationale=(
                                    "The searched evidence is now strong enough, but the current subset still includes "
                                    "redundant broad context that should be pruned before finalizing."
                                ),
                            ),
                        }
                    )

            trajectory.append(
                {
                    "tool": "emit_alert",
                    "arguments": {
                        "decision": "hard_alert",
                        "existence": "anomaly",
                        "category": category,
                        "earliest_alert_sec": temporal["earliest_alert_sec"],
                    },
                }
            )
            trajectory.append(
                {
                    "tool": "verify_hypothesis",
                    "arguments": {
                        "verification_mode": "hard_alert_check",
                        "claim": {
                            "existence": "anomaly",
                            "category": category,
                            "earliest_alert_sec": temporal.get("earliest_alert_sec"),
                        },
                    },
                    "oracle_verifier_feedback": self._oracle_verifier_feedback(
                        verification_mode="hard_alert_check",
                        verification_decision="sufficient",
                        primary_status="complete",
                        alert_status="justified",
                        recommended_action="finalize",
                        selected_refs=searched_real_refs,
                        selected_evidence_moment_ids=[
                            str(moment.get("moment_id"))
                            for moment in searched_real_moments
                            if moment.get("moment_id") is not None
                        ],
                        sufficiency_score=0.92,
                        necessity_score=0.71,
                        alertability_score=0.88,
                        counterfactual_faithfulness=0.82,
                        failure_reasons=[],
                        rationale="The selected evidence subset is sufficient, necessary enough, and actionably supports the current anomaly claim.",
                    ),
                }
            )
        else:
            for step in self._normal_followup_scan_steps(base, count=3):
                append_step(step, ref_bucket=normal_search_refs)
            trajectory.append(
                {
                    "tool": "verify_hypothesis",
                    "arguments": {
                        "verification_mode": "final_check",
                        "claim": {
                            "existence": "normal",
                            "category": canonicalize_saver_category(label.get("category"), existence="normal"),
                        },
                    },
                    "oracle_verifier_feedback": self._oracle_verifier_feedback(
                        verification_mode="final_check",
                        verification_decision="sufficient",
                        primary_status="complete",
                        alert_status="not_applicable",
                        recommended_action="finalize",
                        selected_refs=normal_search_refs,
                        selected_evidence_moment_ids=[],
                        sufficiency_score=0.9,
                        necessity_score=0.58,
                        alertability_score=0.0,
                        counterfactual_faithfulness=0.74,
                        failure_reasons=[],
                        rationale="The searched windows are enough to justify a normal decision, so the case can be finalized.",
                    ),
                }
            )

        trajectory.append(
            {
                "tool": "finalize_case",
                "arguments": base["structured_target"],
            }
        )
        for step in trajectory:
            if str(step.get("tool") or "") != "verify_hypothesis":
                continue
            feedback = step.get("oracle_verifier_feedback") or {}
            if not isinstance(feedback, dict):
                continue
            merged_arguments = dict(step.get("arguments") or {})
            merged_arguments.update(feedback)
            step["arguments"] = build_policy_self_verification_payload(merged_arguments)
        return {
            "trajectory": trajectory,
            "final_decision": base["structured_target"],
        }

    @staticmethod
    def _window_key(start_sec: Any, end_sec: Any) -> Tuple[Optional[float], Optional[float]]:
        return round6(start_sec), round6(end_sec)

    @staticmethod
    def _sorted_evidence_moments(evidence_moments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        role_priority = {
            "precursor": 0,
            "trigger": 1,
            "peak_action": 2,
            "peak": 2,
            "confirmation": 3,
            "aftermath": 4,
        }
        return sorted(
            evidence_moments,
            key=lambda moment: (
                role_priority.get(str(moment.get("role") or "").lower(), 5),
                float(moment.get("start_sec") or 0.0),
                float(moment.get("end_sec") or 0.0),
                str(moment.get("moment_id") or ""),
            ),
        )

    def _seek_evidence_step(
        self,
        base: Dict[str, Any],
        moment: Dict[str, Any],
        *,
        category: str,
        proposal_supervision: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        role = str(moment.get("role") or "")
        fallback_query = self._query_for_role(role, category)
        query_text, query_source = select_query_for_moment(
            moment=moment,
            proposal_supervision=proposal_supervision,
            fallback_query=fallback_query,
        )
        query_package = self._build_query_package_for_moment(
            base,
            moment=moment,
            query_text=query_text,
            query_source=query_source,
            proposal_supervision=proposal_supervision,
        )
        return {
            "tool": "seek_evidence",
            "arguments": {
                "query": query_text,
                "query_package": query_package,
                "start_sec": moment["start_sec"],
                "end_sec": moment["end_sec"],
                "moment_id": moment["moment_id"],
                "role": role,
                "query_source": query_source,
            },
        }

    def _supplemental_seek_steps(
        self,
        base: Dict[str, Any],
        *,
        category: str,
        used_window_keys: set[Tuple[Optional[float], Optional[float]]],
        count: int,
    ) -> List[Dict[str, Any]]:
        if count <= 0:
            return []
        duration = round6(base["video_meta"]["duration_sec"]) or 0.0
        temporal = base.get("temporal") or {}
        anomaly_interval = temporal.get("anomaly_interval_sec") or [0.0, duration]
        precursor_interval = temporal.get("precursor_interval_sec")
        anomaly_start = float((anomaly_interval or [0.0, duration])[0] or 0.0)
        anomaly_end = float((anomaly_interval or [0.0, duration])[1] or duration)
        if anomaly_end <= anomaly_start:
            anomaly_end = max(anomaly_start + 0.5, duration)
        anomaly_span = max(anomaly_end - anomaly_start, 0.5)

        candidate_specs: List[Dict[str, Any]] = []
        if precursor_interval:
            candidate_specs.append(
                {
                    "query": self._query_for_role("precursor", category),
                    "start_sec": precursor_interval[0],
                    "end_sec": precursor_interval[1],
                    "role": "precursor",
                    "query_source": "oracle_role_fallback",
                }
            )
        candidate_specs.extend(
            [
                {
                    "query": self._query_for_role("peak_action", category),
                    "start_sec": anomaly_start + 0.15 * anomaly_span,
                    "end_sec": anomaly_start + 0.55 * anomaly_span,
                    "role": "peak_action",
                    "query_source": "oracle_role_fallback",
                },
                {
                    "query": self._query_for_role("confirmation", category),
                    "start_sec": max(anomaly_start, anomaly_end - max(0.35 * anomaly_span, 0.5)),
                    "end_sec": anomaly_end,
                    "role": "confirmation",
                    "query_source": "oracle_role_fallback",
                },
                {
                    "query": "look for broader temporal context around the suspected anomaly",
                    "start_sec": max(0.0, anomaly_start - max(0.15 * anomaly_span, 0.5)),
                    "end_sec": min(duration, anomaly_end + max(0.15 * anomaly_span, 0.5)),
                    "role": "context",
                    "query_source": "oracle_context_broad",
                },
                {
                    "query": "look for the full temporal context of the suspected anomaly",
                    "start_sec": anomaly_start,
                    "end_sec": anomaly_end,
                    "role": "context",
                    "query_source": "oracle_context_full",
                },
            ]
        )

        supplemental_steps: List[Dict[str, Any]] = []
        for spec in candidate_specs:
            start_sec = round6(spec.get("start_sec"))
            end_sec = round6(spec.get("end_sec"))
            if start_sec is None or end_sec is None:
                continue
            if end_sec <= start_sec:
                end_sec = round6(min(duration, float(start_sec) + 0.5))
            window_key = self._window_key(start_sec, end_sec)
            if window_key in used_window_keys:
                continue
            used_window_keys.add(window_key)
            supplemental_steps.append(
                {
                    "tool": "seek_evidence",
                    "arguments": {
                        "query": spec["query"],
                        "query_package": self._build_generic_query_package(
                            base,
                            role=str(spec.get("role") or ""),
                            query_text=str(spec["query"]),
                            query_source=str(spec.get("query_source") or "oracle_role_fallback"),
                        ),
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "role": spec.get("role"),
                        "query_source": spec.get("query_source"),
                    },
                }
            )
            if len(supplemental_steps) >= count:
                break
        return supplemental_steps

    def _normal_followup_scan_steps(self, base: Dict[str, Any], *, count: int = 3) -> List[Dict[str, Any]]:
        duration = round6(base["video_meta"]["duration_sec"]) or 0.0
        if duration <= 0.0 or count <= 0:
            return []
        boundaries = [round6(duration * float(idx) / float(count)) for idx in range(count + 1)]
        steps: List[Dict[str, Any]] = []
        for idx in range(count):
            start_sec = boundaries[idx]
            end_sec = boundaries[idx + 1]
            if end_sec is None or start_sec is None:
                continue
            if idx == count - 1:
                end_sec = duration
            if end_sec <= start_sec:
                end_sec = round6(min(duration, float(start_sec) + max(duration / max(count, 1), 0.5)))
            if end_sec <= start_sec:
                continue
            steps.append(
                {
                    "tool": "seek_evidence",
                    "arguments": {
                        "query": f"check whether segment {idx + 1} contains any actionable anomaly evidence or supports a normal conclusion",
                        "query_package": self._build_generic_query_package(
                            base,
                            role="normal_check",
                            query_text=f"check whether segment {idx + 1} contains any actionable anomaly evidence or supports a normal conclusion",
                            query_source="oracle_normal_search",
                        ),
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "role": "normal_check",
                        "query_source": "oracle_normal_search",
                    },
                }
            )
        return steps

    @staticmethod
    def _oracle_verifier_feedback(
        *,
        verification_mode: str,
        verification_decision: str,
        primary_status: str,
        alert_status: str,
        recommended_action: str,
        selected_refs: Optional[List[Dict[str, Any]]] = None,
        selected_evidence_moment_ids: Optional[List[str]] = None,
        sufficiency_score: float,
        necessity_score: float,
        alertability_score: float,
        counterfactual_faithfulness: float,
        failure_reasons: List[str],
        rationale: str,
        teacher_judge_scores: Optional[Dict[str, Any]] = None,
        teacher_judge_decision: Optional[str] = None,
        teacher_judge_rationale: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = build_policy_self_verification_payload(
            {
            "verification_mode": verification_mode,
            "verification_decision": verification_decision,
            "primary_status": primary_status,
            "alert_status": alert_status,
            "recommended_action": recommended_action,
            "selected_window_ids": [
                str(ref.get("window_id"))
                for ref in list(selected_refs or [])
                if ref.get("window_id") is not None
            ],
            "selected_evidence_ids": [
                str(ref.get("evidence_id"))
                for ref in list(selected_refs or [])
                if ref.get("evidence_id") is not None
            ],
            "selected_evidence_moment_ids": [
                str(moment_id)
                for moment_id in list(selected_evidence_moment_ids or [])
                if str(moment_id).strip()
            ],
            "sufficiency_score": round6(sufficiency_score),
            "necessity_score": round6(necessity_score),
            "alertability_score": round6(alertability_score),
            "counterfactual_faithfulness": round6(counterfactual_faithfulness),
            "failure_reasons": list(failure_reasons or []),
            "rationale": rationale,
            }
        )
        if teacher_judge_scores is not None:
            payload["teacher_judge_scores"] = dict(teacher_judge_scores)
        if teacher_judge_decision is not None:
            payload["teacher_judge_decision"] = str(teacher_judge_decision)
        if teacher_judge_rationale is not None:
            payload["teacher_judge_rationale"] = str(teacher_judge_rationale)
        return payload

    @staticmethod
    def _query_for_role(role: str, category: str) -> str:
        if role == "precursor":
            return "look for early cues, suspicious lead-up, or precursor evidence before a possible anomaly"
        if role == "trigger":
            return "look for the decisive trigger event or first clearly actionable moment"
        if role == "peak_action":
            return "look for the strongest visible anomalous action or interaction"
        if role == "confirmation":
            return "look for confirmation that the suspected anomaly has actually occurred"
        return "look for evidence relevant to the suspected anomaly"


ADAPTERS = {
    "msad_saver_qwen": CanonicalSaverAdapter,
    "canonical_saver_v1": CanonicalSaverAdapter,
}


def convert_record(
    record: Dict[str, Any],
    *,
    mode: str,
    adapter_name: str = "msad_saver_qwen",
    heuristic_seconds: float = 2.0,
    heuristic_fraction: float = 0.2,
) -> Dict[str, Any]:
    adapter_cls = ADAPTERS[adapter_name]
    adapter = adapter_cls(
        ConverterConfig(
            heuristic_seconds=heuristic_seconds,
            heuristic_fraction=heuristic_fraction,
        )
    )
    return adapter.convert(record, mode)


def iter_jsonl(
    path: Path,
    *,
    include_splits: Optional[str | List[str]] = None,
    skip_invalid_lines: bool = False,
) -> Iterable[Dict[str, Any]]:
    allowed_splits = set(parse_include_splits(include_splits) or [])
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                preview = line.replace("\t", " ")
                if len(preview) > 240:
                    preview = preview[:240] + "..."
                message = f"Invalid JSONL at {path}:{line_number}: {exc}. Line preview: {preview}"
                if not skip_invalid_lines:
                    raise ValueError(message) from exc
                print(json.dumps({"warning": "skipped_invalid_jsonl_line", "message": message}, ensure_ascii=False))
                continue
            if allowed_splits and str(record.get("split") or "").strip() not in allowed_splits:
                continue
            yield record


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input canonical JSONL path.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument(
        "--mode",
        default="agent_train",
        choices=["canonical_passthrough", "agent_train", "oracle_sft"],
        help="Derived view to generate.",
    )
    parser.add_argument(
        "--adapter",
        default="msad_saver_qwen",
        choices=sorted(ADAPTERS.keys()),
        help="Input adapter.",
    )
    parser.add_argument(
        "--heuristic-seconds",
        type=float,
        default=2.0,
        help="Fallback precursor window length in seconds when precursor is missing.",
    )
    parser.add_argument(
        "--heuristic-fraction",
        type=float,
        default=0.2,
        help="Fallback precursor window length as a fraction of anomaly duration.",
    )
    parser.add_argument(
        "--include-splits",
        default="",
        help="Optional comma-separated split whitelist, e.g. train or train,val.",
    )
    parser.add_argument(
        "--skip-invalid-jsonl-lines",
        action="store_true",
        help="Skip malformed JSONL lines instead of failing immediately. Prefer regenerating the source file when possible.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)
    input_records = iter_jsonl(
        input_path,
        include_splits=args.include_splits,
        skip_invalid_lines=args.skip_invalid_jsonl_lines,
    )
    converted_rows = (
        convert_record(
            record,
            mode=args.mode,
            adapter_name=args.adapter,
            heuristic_seconds=args.heuristic_seconds,
            heuristic_fraction=args.heuristic_fraction,
        )
        for record in input_records
    )
    write_jsonl(output_path, converted_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
