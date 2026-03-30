from __future__ import annotations

import copy
import json
import re
from typing import Any, Dict, List, Optional

from saver_agent.config import SaverAgentConfig
from saver_agent.prompts import build_tool_response_prompt


TIMESTAMP_PATTERN = re.compile(r"^\d+(?:\.\d+)?s$")
THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


class TimeSearchRolloutAdapter:
    """Adapt SAVER observations into a TimeSearch-R style rollout transcript."""

    def __init__(self, *, config: SaverAgentConfig | None = None):
        self.config = copy.deepcopy(config) if config is not None else SaverAgentConfig()

    def build_initial_messages(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        return copy.deepcopy(item.get("messages", []))

    def build_assistant_message(self, response_text: str) -> Dict[str, Any]:
        return {
            "role": "assistant",
            "content": [{"type": "text", "text": response_text}],
        }

    def adapt_tool_observation(
        self,
        tool_message: Dict[str, Any],
        multimodal_cache: Dict[str, Any],
    ) -> Dict[str, Any]:
        adapted = copy.deepcopy(tool_message)
        content = list(adapted.get("content", []))
        if adapted.get("name") == "parse_error":
            content.append(
                {
                    "type": "text",
                    "text": (
                        "Retry now with exactly one valid <tool_call>{...}</tool_call> or "
                        "<answer>{...}</answer>. Do not explain the action in plain English."
                    ),
                }
            )
            adapted["content"] = content
            return adapted
        if adapted.get("name") == "verify_hypothesis":
            verification = self._extract_json_payload(content)
            recommended_action = str((verification or {}).get("recommended_action") or "")
            finalize_required = self._finalize_required_fields(multimodal_cache)
            finalize_suffix = ""
            if finalize_required:
                finalize_suffix = " Required finalize_case fields: " + ", ".join(finalize_required) + "."
            if recommended_action == "finalize":
                content.append(
                    {
                        "type": "text",
                        "text": (
                            "Verifier says the current evidence is sufficient. "
                            "Call finalize_case next using your current structured claim supported by searched evidence only. "
                            "After finalize_case returns, output the final answer inside <answer></answer>."
                            f"{finalize_suffix}"
                        ),
                    }
                )
            elif recommended_action == "continue_search":
                content.append(
                    {
                        "type": "text",
                        "text": (
                            "Verifier says the current evidence is not yet sufficient. "
                            "Search more evidence or refine the claim before finalizing."
                        ),
                    }
                )
            elif recommended_action == "revise_claim":
                content.append(
                    {
                        "type": "text",
                        "text": "Verifier says the current evidence does not align with the claim. Revise the claim before continuing.",
                    }
                )
            adapted["content"] = content
            return adapted
        if adapted.get("name") == "finalize_case":
            finalized_payload = self._extract_json_payload(content)
            finalized_case = (finalized_payload or {}).get("finalized_case") or {}
            content.append(
                {
                    "type": "text",
                    "text": (
                        "Output the final answer now inside <answer></answer>. "
                        f"Use the finalized case JSON {json.dumps(finalized_case, ensure_ascii=False)}. "
                        "Do not call more tools."
                    ),
                }
            )
            adapted["content"] = content
            return adapted
        if adapted.get("name") == "emit_alert":
            alert_payload = self._extract_json_payload(content)
            alert = (alert_payload or {}).get("alert") or {}
            decision = alert.get("decision")
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"The alert decision {decision!r} has been recorded. "
                        "If the evidence is now sufficient, call verify_hypothesis or finalize_case instead of repeating search."
                    ),
                }
            )
            adapted["content"] = content
            return adapted
        prompt_text = build_tool_response_prompt(
            self._extract_timestamps(content),
            question=str(multimodal_cache.get("question") or ""),
            duration=multimodal_cache.get("duration"),
            prompt_config=self.config.prompt,
        )
        content.append({"type": "text", "text": prompt_text})
        adapted["content"] = content
        return adapted

    @staticmethod
    def parse_answer_text(response_text: str) -> Optional[str]:
        stripped = THINK_BLOCK_PATTERN.sub("", response_text)
        matches = re.findall(r"<answer>(.*?)</answer>", stripped, re.DOTALL)
        if matches:
            return matches[-1].strip()
        matches = re.findall(r"<answer>(.*?)</answer>", response_text, re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None

    @staticmethod
    def _extract_timestamps(content: List[Dict[str, Any]]) -> List[str]:
        timestamps: List[str] = []
        for item in content:
            if item.get("type") != "text":
                continue
            text = str(item.get("text", "")).strip()
            if TIMESTAMP_PATTERN.match(text):
                timestamps.append(text)
        return timestamps

    @staticmethod
    def _extract_json_payload(content: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        for item in content:
            if item.get("type") != "text":
                continue
            text = str(item.get("text", "")).strip()
            try:
                payload = json.loads(text)
            except Exception:
                continue
            if isinstance(payload, dict):
                return payload
        return None

    @staticmethod
    def _finalize_required_fields(multimodal_cache: Dict[str, Any]) -> List[str]:
        schema = (multimodal_cache.get("tool_io") or {}).get("finalize_case_schema") or {}
        required = list(schema.get("required") or [])
        return [str(field_name) for field_name in required if str(field_name).strip()]
