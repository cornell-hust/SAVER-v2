from __future__ import annotations

from typing import Any, Dict, List, Tuple

from saver_agent.schema import SaverEnvironmentState
from saver_agent import tools as saver_tools


TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "scan_timeline",
            "description": "Uniformly inspect a time window to build a global or local overview.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_sec": {"type": "number"},
                    "end_sec": {"type": "number"},
                    "num_frames": {"type": "integer"},
                    "stride_sec": {"type": "number"},
                    "purpose": {"type": "string"},
                },
                "required": ["start_sec", "end_sec"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "seek_evidence",
            "description": "Search a time window for frames relevant to a textual anomaly hypothesis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "start_sec": {"type": "number"},
                    "end_sec": {"type": "number"},
                    "num_frames": {"type": "integer"},
                    "moment_id": {"type": "string"},
                    "role": {"type": "string"},
                    "query_source": {"type": "string"},
                    "top_k_candidates": {"type": "integer"},
                    "candidate_merge_gap_sec": {"type": "number"},
                },
                "required": ["query", "start_sec", "end_sec"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "emit_alert",
            "description": "Record a soft alert, hard alert, or normal declaration proposal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "decision": {
                        "type": "string",
                        "enum": ["soft_alert", "hard_alert", "declare_normal"],
                    },
                    "existence": {"type": "string", "enum": ["normal", "anomaly"]},
                    "category": {"type": "string"},
                    "earliest_alert_sec": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["decision", "existence"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_hypothesis",
            "description": "Verify whether the currently visited evidence supports the active anomaly hypothesis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "verification_mode": {
                        "type": "string",
                        "enum": ["soft_alert_check", "hard_alert_check", "final_check", "full_keep_drop", "reward_only"],
                    },
                    "candidate_window_ids": {"type": "array", "items": {"type": "string"}},
                    "candidate_evidence_ids": {"type": "array", "items": {"type": "string"}},
                    "evidence_moment_ids": {"type": "array", "items": {"type": "string"}},
                    "claim": {"type": "object"},
                    "alert": {"type": "object"},
                    "query": {"type": "string"},
                    "verifier_backend": {
                        "type": "string",
                        "enum": ["heuristic", "qwen_self_verifier", "hybrid"],
                    },
                },
                "required": ["verification_mode"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize_case",
            "description": "Finalize the structured anomaly decision once enough evidence has been gathered.",
            "parameters": {"type": "object"},
        },
    },
]

TOOL_IMPLS = {
    "scan_timeline": saver_tools.scan_timeline,
    "seek_evidence": saver_tools.seek_evidence,
    "emit_alert": saver_tools.emit_alert,
    "verify_hypothesis": saver_tools.verify_hypothesis,
    "finalize_case": saver_tools.finalize_case,
}


def get_tool_schemas() -> List[Dict[str, Any]]:
    return [tool.copy() for tool in TOOL_SCHEMAS]


def execute_tool_call(
    params: Dict[str, Any],
    multimodal_cache: Dict[str, Any],
    state: SaverEnvironmentState,
) -> Tuple[Dict[str, Any], SaverEnvironmentState]:
    func = params.get("function", {})
    name = func.get("name")
    arguments = func.get("arguments", {})
    if name not in TOOL_IMPLS:
        raise ValueError(f"Unknown tool name: {name}")
    content, state, _ = TOOL_IMPLS[name](arguments, multimodal_cache, state)
    message = {
        "role": "tool",
        "name": name,
        "arguments": arguments,
        "content": content,
    }
    return message, state
