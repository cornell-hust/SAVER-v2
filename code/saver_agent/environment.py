from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

from saver_agent.schema import SaverEnvironmentState
from saver_agent.tool_registry import execute_tool_call


INVALID_TOOL_CALL_PROMPT = (
    "The previous response is invalid. Retry immediately with exactly one of these formats: "
    '<tool_call>{"name":"scan_timeline","arguments":{"start_sec":0.0,"end_sec":4.0}}</tool_call> '
    'or <answer>{"existence":"normal"}</answer>. '
    "Do not describe the intended tool call in plain English. Do not output bare tool names."
)


def cleanup_llm_response(response_str: str) -> str:
    tagged_blocks = list(re.finditer(r"<(tool_call|answer)>(.*?)</\1>", response_str, re.DOTALL))
    if tagged_blocks:
        return tagged_blocks[-1].group(0)
    if "<think>" in response_str:
        response_str = "<think>" + response_str.split("<think>")[-1]
    return response_str


def _finalize_retry_prompt(multimodal_cache: Dict[str, Any] | None = None) -> str:
    schema = ((multimodal_cache or {}).get("tool_io") or {}).get("finalize_case_schema") or {}
    required = [str(field_name) for field_name in list(schema.get("required") or []) if str(field_name).strip()]
    required_text = ", ".join(required) if required else "the required finalize_case fields"
    return (
        "The previous finalize_case call was invalid. Retry immediately with exactly one "
        '<tool_call>{"name":"finalize_case","arguments":{...}}</tool_call>. '
        f"finalize_case requires: {required_text}. "
        "Do not output <answer> yet. Do not describe the intended tool call in plain English."
    )


def invalid_tool_call_message(
    *,
    tool_name: str | None = None,
    multimodal_cache: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    prompt_text = _finalize_retry_prompt(multimodal_cache) if tool_name == "finalize_case" else INVALID_TOOL_CALL_PROMPT
    return {
        "role": "tool",
        "name": "parse_error",
        "content": [{"type": "text", "text": prompt_text}],
    }


def _parse_sec_value(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if text.endswith("s"):
        text = text[:-1].strip()
    if ":" in text:
        parts = text.split(":")
        try:
            total = 0.0
            for part in parts:
                total = total * 60.0 + float(part)
            return total
        except Exception:
            return value
    try:
        return float(text)
    except Exception:
        return value


def _normalize_tool_arguments(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(arguments)
    if "start_time" in normalized and "start_sec" not in normalized:
        normalized["start_sec"] = _parse_sec_value(normalized.pop("start_time"))
    if "end_time" in normalized and "end_sec" not in normalized:
        normalized["end_sec"] = _parse_sec_value(normalized.pop("end_time"))
    if "start_sec" in normalized:
        normalized["start_sec"] = _parse_sec_value(normalized["start_sec"])
    if "end_sec" in normalized:
        normalized["end_sec"] = _parse_sec_value(normalized["end_sec"])
    if "earliest_alert_sec" in normalized:
        normalized["earliest_alert_sec"] = _parse_sec_value(normalized["earliest_alert_sec"])
    if "alert_sec" in normalized:
        normalized["alert_sec"] = _parse_sec_value(normalized["alert_sec"])

    if name == "emit_alert":
        alert_type = str(normalized.get("alert_type") or "").strip().lower()
        if alert_type and "decision" not in normalized:
            normalized["decision"] = {
                "soft": "soft_alert",
                "hard": "hard_alert",
                "normal": "declare_normal",
            }.get(alert_type, alert_type)
        if "reason" not in normalized and "description" in normalized:
            normalized["reason"] = normalized["description"]
        if "existence" not in normalized:
            decision = str(normalized.get("decision") or "").strip().lower()
            if decision in {"soft_alert", "hard_alert"}:
                normalized["existence"] = "anomaly"
            elif decision == "declare_normal":
                normalized["existence"] = "normal"
    return normalized


def _parse_tool_payload(content: str) -> Dict[str, Any] | None:
    parsed: Dict[str, Any] | None = None
    try:
        candidate = json.loads(content)
    except Exception:
        candidate = None

    if isinstance(candidate, dict):
        if "name" in candidate:
            parsed = {"name": candidate.get("name"), "arguments": candidate.get("arguments") or {}}
        elif len(candidate) == 1:
            tool_name, arguments = next(iter(candidate.items()))
            if isinstance(arguments, dict):
                parsed = {"name": tool_name, "arguments": arguments}

    if parsed is None:
        match = re.fullmatch(r"\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(\{.*\})\s*\}", content, re.DOTALL)
        if match:
            tool_name = match.group(1)
            try:
                arguments = json.loads(match.group(2))
            except Exception:
                arguments = None
            if isinstance(arguments, dict):
                parsed = {"name": tool_name, "arguments": arguments}

    if parsed is None:
        return None

    name = str(parsed.get("name") or "").strip()
    arguments = parsed.get("arguments") or {}
    if not name or not isinstance(arguments, dict):
        return None
    return {"name": name, "arguments": _normalize_tool_arguments(name, arguments)}


def parse_actions_and_contents(predictions: List[Any]) -> Tuple[List[str | None], List[Any]]:
    actions: List[str | None] = []
    contents: List[Any] = []
    for prediction in predictions:
        stripped_prediction = re.sub(r"<think>.*?</think>", "", prediction, flags=re.DOTALL)
        matches = re.findall(r"<(tool_call|answer)>(.*?)</\1>", stripped_prediction, re.DOTALL)
        if not matches:
            matches = re.findall(r"<(tool_call|answer)>(.*?)</\1>", prediction, re.DOTALL)
        match = matches[-1] if matches else None
        if not match:
            actions.append(None)
            contents.append("")
            continue
        action = match[0]
        content = match[1].strip()
        if action == "tool_call":
            func = _parse_tool_payload(content)
            if func is None:
                actions.append(None)
                contents.append("")
            else:
                actions.append("tool_call")
                contents.append({"type": "function", "function": func})
        else:
            actions.append("answer")
            contents.append(content)
    return actions, contents


class SaverVideoInteraction:
    """Step-level environment wrapper for SAVER-style tool use."""

    def execute_predictions(
        self,
        predictions: List[str],
        multimodal_cache_batch: List[Dict[str, Any]],
        states: List[SaverEnvironmentState],
        active_mask: List[bool],
    ) -> Tuple[List[Any], List[int], List[int], List[int], List[SaverEnvironmentState]]:
        cleaned = [cleanup_llm_response(prediction) for prediction in predictions]
        actions, contents = parse_actions_and_contents(cleaned)

        next_obs: List[Any] = []
        dones: List[int] = []
        valid_actions: List[int] = []
        is_search: List[int] = []
        next_states: List[SaverEnvironmentState] = []

        for action, content, multimodal_cache, state, active in zip(
            actions, contents, multimodal_cache_batch, states, active_mask
        ):
            if not active:
                next_obs.append(None)
                dones.append(1)
                valid_actions.append(0)
                is_search.append(0)
                next_states.append(state)
                continue

            if action == "answer":
                next_obs.append(None)
                dones.append(1)
                valid_actions.append(1)
                is_search.append(0)
                next_states.append(state)
                continue

            if action == "tool_call":
                try:
                    message, next_state = execute_tool_call(content, multimodal_cache, state)
                    next_obs.append(message)
                    dones.append(0)
                    valid_actions.append(1)
                    is_search.append(1)
                    next_states.append(next_state)
                except Exception:
                    next_obs.append(
                        invalid_tool_call_message(
                            tool_name=content["function"]["name"],
                            multimodal_cache=multimodal_cache,
                        )
                    )
                    dones.append(0)
                    valid_actions.append(0)
                    is_search.append(0)
                    next_states.append(state)
                continue

            next_obs.append(invalid_tool_call_message(multimodal_cache=multimodal_cache))
            dones.append(0)
            valid_actions.append(0)
            is_search.append(0)
            next_states.append(state)

        return next_obs, dones, valid_actions, is_search, next_states
