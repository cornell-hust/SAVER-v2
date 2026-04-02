from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple

from saver_agent.categories import CANONICAL_POLICY_CATEGORIES
from saver_agent.schema import SaverEnvironmentState
from saver_agent.self_verification import build_self_verification_tool_schema
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
            "description": "Record a soft alert, hard alert, or normal declaration proposal using canonical SAVER category labels.",
            "parameters": {
                "type": "object",
                "properties": {
                    "decision": {
                        "type": "string",
                        "enum": ["soft_alert", "hard_alert", "declare_normal"],
                    },
                    "existence": {"type": "string", "enum": ["normal", "anomaly"]},
                    "category": {"type": "string", "enum": list(CANONICAL_POLICY_CATEGORIES)},
                    "earliest_alert_sec": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["decision", "existence", "category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_hypothesis",
            "description": (
                "Verify whether the currently selected evidence subset is sufficient, necessary enough, "
                "and actionable for the active anomaly hypothesis using a compact policy-produced "
                "self-verification verdict."
            ),
            "parameters": build_self_verification_tool_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize_case",
            "description": "Finalize the structured anomaly decision once enough evidence has been gathered.",
            "parameters": {
                "type": "object",
                "properties": {
                    "existence": {"type": "string", "enum": ["normal", "anomaly"]},
                    "category": {"type": "string", "enum": list(CANONICAL_POLICY_CATEGORIES)},
                },
            },
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


def get_tool_schemas(*, finalize_case_schema: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    tool_schemas = copy.deepcopy(TOOL_SCHEMAS)
    if not finalize_case_schema:
        return tool_schemas
    for tool in tool_schemas:
        function = tool.get("function") or {}
        if function.get("name") != "finalize_case":
            continue
        function["parameters"] = copy.deepcopy(finalize_case_schema)
        break
    return tool_schemas


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
