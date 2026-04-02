from __future__ import annotations

import copy
import json
from dataclasses import asdict
from typing import Any, Callable, Dict, Iterable, List, Optional

from saver_agent.adapter import TimeSearchRolloutAdapter
from saver_agent.config import RolloutTraceConfig, SaverAgentConfig
from saver_agent.environment import SaverVideoInteraction, parse_actions_and_contents
from saver_agent.schema import SaverEnvironmentState
from saver_agent.self_verification import parse_self_verification_payload


PolicyFn = Callable[[List[Dict[str, Any]], Dict[str, Any], SaverEnvironmentState, int], str]


class ReplayPolicy:
    """Deterministic response replay for rollout smoke tests and CLI usage."""

    def __init__(self, responses: Iterable[str]):
        self.responses = list(responses)
        self.cursor = 0

    def __call__(
        self,
        messages: List[Dict[str, Any]],
        multimodal_cache: Dict[str, Any],
        state: SaverEnvironmentState,
        step_index: int,
    ) -> str:
        if self.cursor >= len(self.responses):
            raise IndexError(f"ReplayPolicy exhausted at step {step_index}")
        response = self.responses[self.cursor]
        self.cursor += 1
        return response


class SaverRolloutRunner:
    """Minimal step runner that mirrors TimeSearch-R style tool-using rollouts."""

    def __init__(
        self,
        *,
        environment: Optional[SaverVideoInteraction] = None,
        adapter: Optional[TimeSearchRolloutAdapter] = None,
        max_turns: int = 12,
        config: Optional[SaverAgentConfig] = None,
    ):
        self.config = copy.deepcopy(config) if config is not None else SaverAgentConfig()
        self.environment = environment or SaverVideoInteraction()
        self.adapter = adapter or TimeSearchRolloutAdapter(config=self.config)
        self.max_turns = int(max_turns)

    def run_episode(
        self,
        item: Dict[str, Any],
        policy: PolicyFn,
        *,
        initial_state: Optional[SaverEnvironmentState] = None,
    ) -> Dict[str, Any]:
        messages = self.adapter.build_initial_messages(item)
        multimodal_cache = item["multimodal_cache"]
        state = copy.deepcopy(initial_state or SaverEnvironmentState())
        turns: List[Dict[str, Any]] = []
        invalid_attempts: List[Dict[str, Any]] = []
        final_answer = None
        final_answer_text = None
        terminated_reason = "max_turns"
        terminated_at_step = self.max_turns
        needs_state_delta = bool(self.config.rollout_trace.record_state_deltas)
        needs_counterfactual_trace = bool(self.config.rollout_trace.record_counterfactual_trace)
        needs_state_snapshots = bool(needs_state_delta or needs_counterfactual_trace)
        formal_step_index = 0
        total_attempts = 0
        max_total_attempts = max(self.max_turns * 4, self.max_turns)

        while formal_step_index < self.max_turns:
            total_attempts += 1
            if total_attempts > max_total_attempts:
                terminated_reason = "max_invalid_retries"
                terminated_at_step = formal_step_index
                break
            step_index = formal_step_index + 1
            state_before = asdict(copy.deepcopy(state)) if needs_state_snapshots else None
            response_text = policy(messages, multimodal_cache, state, step_index)
            actions, contents = parse_actions_and_contents([response_text])
            action = actions[0]
            parsed_content = contents[0]

            next_obs, dones, valid_actions, is_search, next_states = self.environment.execute_predictions(
                [response_text],
                [multimodal_cache],
                [state],
                [True],
            )
            next_state = next_states[0]
            is_valid_action = bool(valid_actions[0])
            if is_valid_action:
                messages.append(self.adapter.build_assistant_message(response_text))
                formal_step_index += 1
            state = next_state
            state_after = asdict(copy.deepcopy(state)) if needs_state_snapshots else None

            turn_info = {
                "step_index": step_index,
                "response": response_text,
                "assistant_response_raw": response_text,
                "action": action,
                "assistant_action": action,
                "done": bool(dones[0]),
                "valid_action": bool(valid_actions[0]),
                "is_search": bool(is_search[0]),
                "tool_name": None,
                "parsed_tool_call": None,
                "tool_observation_summary": None,
                "tool_timestamps": [],
                "tool_image_count": 0,
                "new_evidence_ids": [],
                "new_alerts": [],
                "new_verifications": [],
                "new_finalized_case": None,
                "verifier_mode": None,
                "verifier_backend": None,
                "verifier_primary_status": None,
                "verifier_alert_status": None,
                "verifier_recommended_action": None,
                "verifier_derived_scores": None,
                "verifier_verified_window_ids": None,
                "verifier_best_effort_window_ids": None,
                "verifier_failure_reasons": None,
                "verification_parse_mode": None,
                "legacy_compatibility_used": None,
                "invalid_selected_window_ids": [],
                "selection_resolution_source": None,
                "self_verification_decision": None,
                "self_verification_scores": None,
                "self_verification_selected_window_ids": None,
                "self_verification_confidence": None,
                "proposal_backend": None,
                "feature_cache_used": None,
                "proposal_query_raw": None,
                "proposal_query_normalized": None,
                "proposal_query_source": None,
                "proposal_candidate_count": None,
                "proposal_candidate_frame_indices": [],
                "proposal_candidate_frame_scores": [],
                "proposal_candidate_windows": [],
                "proposal_selected_frame_indices": [],
                "proposal_selected_frame_scores": [],
                "proposal_fallback_reason": None,
            }
            tool_message = next_obs[0] if isinstance(next_obs[0], dict) and next_obs[0].get("role") == "tool" else None

            if action == "tool_call":
                turn_info["tool_name"] = parsed_content["function"]["name"]
                turn_info["parsed_tool_call"] = parsed_content["function"]
            elif action == "answer":
                final_answer_text = self.adapter.parse_answer_text(response_text)
                final_answer = self._coerce_answer_payload(final_answer_text)
                terminated_reason = "answered"
                terminated_at_step = step_index
                turn_info["parsed_answer"] = final_answer
                if not is_valid_action:
                    final_answer = None
                    final_answer_text = None
                    terminated_reason = "max_turns"
                    terminated_at_step = self.max_turns
                else:
                    if needs_state_delta and state_before is not None and state_after is not None:
                        state_delta = self._compute_state_delta(state_before, state_after)
                        turn_info["state_before"] = state_before
                        turn_info["state_after"] = state_after
                        turn_info["state_delta"] = state_delta
                        turn_info["new_evidence_ids"] = [entry["evidence_id"] for entry in state_delta["new_evidence_windows"]]
                        turn_info["new_alerts"] = state_delta["new_alerts"]
                        turn_info["new_verifications"] = state_delta["new_verifications"]
                        turn_info["new_finalized_case"] = state_delta["new_finalized_case"]
                        self._attach_search_turn_trace(turn_info, state_delta=state_delta)
                    if needs_counterfactual_trace and state_before is not None and state_after is not None:
                        self._attach_counterfactual_turn_trace(
                            turn_info,
                            state_before=state_before,
                            state_after=state_after,
                        )
                    turns.append(turn_info)
                    break
            if tool_message is not None:
                adapted_tool_message = self.adapter.adapt_tool_observation(tool_message, multimodal_cache)
                messages.append(adapted_tool_message)
                if turn_info["tool_name"] is None:
                    turn_info["tool_name"] = tool_message.get("name")
                turn_info.update(self._summarize_tool_message(tool_message))

            if needs_state_delta and state_before is not None and state_after is not None:
                state_delta = self._compute_state_delta(state_before, state_after)
                turn_info["state_before"] = state_before
                turn_info["state_after"] = state_after
                turn_info["state_delta"] = state_delta
                turn_info["new_evidence_ids"] = [entry["evidence_id"] for entry in state_delta["new_evidence_windows"]]
                turn_info["new_alerts"] = state_delta["new_alerts"]
                turn_info["new_verifications"] = state_delta["new_verifications"]
                turn_info["new_finalized_case"] = state_delta["new_finalized_case"]
                self._attach_search_turn_trace(turn_info, state_delta=state_delta)
            if action != "answer":
                turn_info["parsed_answer"] = None
            if needs_counterfactual_trace and state_before is not None and state_after is not None:
                self._attach_counterfactual_turn_trace(
                    turn_info,
                    state_before=state_before,
                    state_after=state_after,
                )
            if is_valid_action:
                turns.append(turn_info)
            else:
                invalid_attempts.append(turn_info)

        result = {
            "video_id": item.get("video_id"),
            "terminated_reason": terminated_reason,
            "num_turns": len(turns),
            "num_invalid_attempts": len(invalid_attempts),
            "final_answer": final_answer,
            "final_answer_text": final_answer_text,
            "turns": turns,
            "invalid_attempts": invalid_attempts,
            "state": asdict(state),
            "config_snapshot": self.config.to_dict(),
            "preview_trace": self._build_preview_trace(multimodal_cache),
            "termination_trace": {
                "reason": terminated_reason,
                "terminated_at_step": terminated_at_step if turns else 0,
                "final_answer_present": final_answer is not None,
                "latest_verifier_status": self._latest_verifier_status(turns),
                "latest_alert_status": self._latest_alert_status(turns),
                "verification_turn_count": self._verification_turn_count(turns),
                "final_verified_window_ids": self._final_verified_window_ids(turns),
            },
            "counterfactual_anchor_summary": self._build_counterfactual_anchor_summary(turns),
            "decision_turn_indices": self._decision_turn_indices(turns),
            "latest_claim_trace": self._build_latest_claim_trace(turns),
            "latest_alert_trace": self._build_latest_alert_trace(turns),
            "search_trace": self._build_search_trace(turns),
        }
        if self.config.rollout_trace.record_message_history:
            result["messages"] = messages
        else:
            result["messages"] = None
        return result

    @staticmethod
    def _coerce_answer_payload(answer_text: Optional[str]) -> Any:
        if answer_text is None:
            return None
        try:
            payload = json.loads(answer_text)
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _compute_state_delta(state_before: Dict[str, Any], state_after: Dict[str, Any]) -> Dict[str, Any]:
        new_visited_windows = state_after["visited_windows"][len(state_before["visited_windows"]) :]
        new_evidence_windows = state_after["evidence_ledger"][len(state_before["evidence_ledger"]) :]
        new_alerts = state_after["alerts"][len(state_before["alerts"]) :]
        new_verifications = state_after["verification_records"][len(state_before["verification_records"]) :]
        new_finalized_case = None
        if state_before.get("finalized_case") != state_after.get("finalized_case"):
            new_finalized_case = state_after.get("finalized_case")
        return {
            "new_visited_windows": new_visited_windows,
            "new_evidence_windows": new_evidence_windows,
            "new_alerts": new_alerts,
            "new_verifications": new_verifications,
            "new_finalized_case": new_finalized_case,
            "next_evidence_id_delta": state_after["next_evidence_id"] - state_before["next_evidence_id"],
        }

    def _attach_counterfactual_turn_trace(
        self,
        turn_info: Dict[str, Any],
        *,
        state_before: Dict[str, Any],
        state_after: Dict[str, Any],
    ) -> None:
        turn_info["observed_horizon_sec_before"] = self._observed_horizon_sec(state_before)
        turn_info["observed_horizon_sec_after"] = self._observed_horizon_sec(state_after)
        turn_info["latest_claim_before"] = self._latest_claim(state_before)
        turn_info["latest_claim_after"] = self._latest_claim(state_after)
        turn_info["latest_alert_before"] = self._latest_alert(state_before)
        turn_info["latest_alert_after"] = self._latest_alert(state_after)
        turn_info["selected_window_ids_before"] = self._selected_window_ids(state_before)
        turn_info["selected_window_ids_after"] = self._selected_window_ids(state_after)
        turn_info["selected_evidence_ids_before"] = self._selected_evidence_ids(state_before)
        turn_info["selected_evidence_ids_after"] = self._selected_evidence_ids(state_after)
        turn_info["counterfactual_anchor_tags"] = self._counterfactual_anchor_tags(turn_info)
        turn_info["counterfactual_actual_search_branch"] = self._actual_search_branch(turn_info)
        turn_info["counterfactual_actual_alert_branch"] = self._actual_alert_branch(turn_info)
        turn_info["counterfactual_actual_evidence_branch"] = self._actual_evidence_branch(turn_info)

    @staticmethod
    def _attach_search_turn_trace(turn_info: Dict[str, Any], *, state_delta: Dict[str, Any]) -> None:
        if str(turn_info.get("tool_name") or "") != "seek_evidence":
            return
        new_windows = list(state_delta.get("new_evidence_windows") or [])
        if not new_windows:
            return
        latest = dict(new_windows[-1])
        turn_info["proposal_backend"] = latest.get("proposal_backend")
        turn_info["feature_cache_used"] = latest.get("feature_cache_used")
        turn_info["proposal_query_raw"] = latest.get("query")
        turn_info["proposal_query_normalized"] = latest.get("query_normalized")
        turn_info["proposal_query_source"] = latest.get("query_source")
        turn_info["proposal_candidate_count"] = latest.get("proposal_candidate_count")
        turn_info["proposal_candidate_frame_indices"] = list(latest.get("proposal_candidate_frame_indices") or [])
        turn_info["proposal_candidate_frame_scores"] = list(latest.get("proposal_candidate_frame_scores") or [])
        turn_info["proposal_candidate_windows"] = copy.deepcopy(latest.get("proposal_candidate_windows") or [])
        turn_info["proposal_selected_frame_indices"] = list(latest.get("selected_frame_indices") or [])
        turn_info["proposal_selected_frame_scores"] = list(latest.get("selected_frame_scores") or [])
        turn_info["proposal_fallback_reason"] = latest.get("proposal_fallback_reason")

    def _summarize_tool_message(self, tool_message: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not tool_message:
            return {
                "tool_observation_summary": None,
                "tool_timestamps": [],
                "tool_image_count": 0,
                "tool_observation_content": None,
                "verifier_mode": None,
                "verifier_backend": None,
                "verifier_primary_status": None,
                "verifier_alert_status": None,
                "verifier_recommended_action": None,
                "verifier_derived_scores": None,
                "verifier_verified_window_ids": None,
                "verifier_best_effort_window_ids": None,
                "verifier_failure_reasons": None,
                "verification_parse_mode": None,
                "legacy_compatibility_used": None,
                "invalid_selected_window_ids": [],
                "selection_resolution_source": None,
                "self_verification_decision": None,
                "self_verification_scores": None,
                "self_verification_selected_window_ids": None,
                "self_verification_confidence": None,
            }

        content = tool_message.get("content", [])
        tool_timestamps = []
        tool_image_count = 0
        tool_observation_summary = None
        parsed_json_payload = None
        for item in content:
            if item.get("type") == "image":
                tool_image_count += 1
            elif item.get("type") == "text":
                text = str(item.get("text", ""))
                if text.endswith("s"):
                    tool_timestamps.append(text)
                tool_observation_summary = text
                if parsed_json_payload is None:
                    try:
                        parsed_json_payload = json.loads(text)
                    except Exception:
                        parsed_json_payload = None
        summary = {
            "tool_observation_summary": tool_observation_summary,
            "tool_timestamps": tool_timestamps,
            "tool_image_count": tool_image_count,
            "verifier_mode": None,
            "verifier_backend": None,
            "verifier_primary_status": None,
            "verifier_alert_status": None,
            "verifier_recommended_action": None,
            "verifier_derived_scores": None,
            "verifier_verified_window_ids": None,
            "verifier_best_effort_window_ids": None,
            "verifier_failure_reasons": None,
            "self_verification_decision": None,
            "self_verification_scores": None,
            "self_verification_selected_window_ids": None,
            "self_verification_confidence": None,
        }
        if tool_message.get("name") == "verify_hypothesis" and isinstance(parsed_json_payload, dict):
            normalized_payload = parsed_json_payload
            if (
                normalized_payload.get("primary_status") is None
                and normalized_payload.get("verification_decision") is not None
            ):
                normalized_payload = parse_self_verification_payload(normalized_payload)
            summary.update(
                {
                    "verifier_mode": normalized_payload.get("verification_mode"),
                    "verifier_backend": normalized_payload.get("verifier_backend"),
                    "verifier_primary_status": normalized_payload.get("primary_status"),
                    "verifier_alert_status": normalized_payload.get("alert_status"),
                    "verifier_recommended_action": normalized_payload.get("recommended_action"),
                    "verifier_derived_scores": normalized_payload.get("derived_scores"),
                    "verifier_verified_window_ids": normalized_payload.get("verified_window_ids"),
                    "verifier_best_effort_window_ids": normalized_payload.get("best_effort_window_ids"),
                    "verifier_failure_reasons": normalized_payload.get("failure_reasons"),
                    "verification_parse_mode": normalized_payload.get("verification_parse_mode"),
                    "legacy_compatibility_used": normalized_payload.get("legacy_compatibility_used"),
                    "invalid_selected_window_ids": list(normalized_payload.get("invalid_selected_window_ids") or []),
                    "selection_resolution_source": normalized_payload.get("selection_resolution_source"),
                    "self_verification_decision": normalized_payload.get("verification_decision"),
                    "self_verification_scores": normalized_payload.get("self_verification_scores")
                    or normalized_payload.get("derived_scores"),
                    "self_verification_selected_window_ids": normalized_payload.get("selected_window_ids")
                    or normalized_payload.get("verified_window_ids"),
                    "self_verification_confidence": normalized_payload.get("self_verification_confidence"),
                }
            )
        if self.config.rollout_trace.record_observation_content:
            summary["tool_observation_content"] = content
        else:
            summary["tool_observation_content"] = None
        return summary

    @staticmethod
    def _build_preview_trace(multimodal_cache: Dict[str, Any]) -> Dict[str, Any]:
        preview_frames = multimodal_cache.get("preview_frames")
        preview_timestamps = multimodal_cache.get("preview_timestamps") or []
        preview_frame_count = 0 if preview_frames is None else int(len(preview_frames))
        return {
            "preview_frame_count": preview_frame_count,
            "preview_timestamps": preview_timestamps,
        }

    @staticmethod
    def _latest_verifier_status(turns: List[Dict[str, Any]]) -> Optional[str]:
        for turn in reversed(turns):
            if turn.get("verifier_primary_status"):
                return turn["verifier_primary_status"]
        return None

    @staticmethod
    def _latest_alert_status(turns: List[Dict[str, Any]]) -> Optional[str]:
        for turn in reversed(turns):
            if turn.get("verifier_alert_status"):
                return turn["verifier_alert_status"]
        return None

    @staticmethod
    def _verification_turn_count(turns: List[Dict[str, Any]]) -> int:
        return sum(1 for turn in turns if turn.get("tool_name") == "verify_hypothesis")

    @staticmethod
    def _final_verified_window_ids(turns: List[Dict[str, Any]]) -> Optional[List[str]]:
        for turn in reversed(turns):
            if turn.get("verifier_verified_window_ids"):
                return turn["verifier_verified_window_ids"]
        return None

    @staticmethod
    def _observed_horizon_sec(state_dict: Dict[str, Any]) -> float:
        max_end = 0.0
        for entry in state_dict.get("visited_windows") or []:
            try:
                max_end = max(max_end, float(entry.get("end_sec") or 0.0))
            except Exception:
                continue
        return round(float(max_end), 6)

    @staticmethod
    def _latest_claim(state_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        claim = state_dict.get("last_claim")
        if isinstance(claim, dict):
            return copy.deepcopy(claim)
        finalized_case = state_dict.get("finalized_case")
        if isinstance(finalized_case, dict):
            return copy.deepcopy(finalized_case)
        return None

    @staticmethod
    def _latest_alert(state_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        alerts = state_dict.get("alerts") or []
        if alerts:
            return copy.deepcopy(alerts[-1])
        return None

    @staticmethod
    def _selected_window_ids(state_dict: Dict[str, Any]) -> List[str]:
        active = state_dict.get("active_evidence_window_ids") or []
        if active:
            return [str(value) for value in active]
        verification_records = state_dict.get("verification_records") or []
        if verification_records:
            latest = verification_records[-1]
            values = latest.get("verified_window_ids") or latest.get("best_effort_window_ids") or []
            return [str(value) for value in values]
        return []

    @staticmethod
    def _selected_evidence_ids(state_dict: Dict[str, Any]) -> List[str]:
        window_ids = set(SaverRolloutRunner._selected_window_ids(state_dict))
        if not window_ids:
            return []
        evidence_ids: List[str] = []
        for entry in state_dict.get("evidence_ledger") or []:
            window_id = str(entry.get("window_id") or "")
            evidence_id = entry.get("evidence_id")
            if window_id in window_ids and evidence_id:
                evidence_ids.append(str(evidence_id))
        return evidence_ids

    @staticmethod
    def _counterfactual_anchor_tags(turn_info: Dict[str, Any]) -> List[str]:
        tool_name = str(turn_info.get("tool_name") or "")
        tags: List[str] = []
        if tool_name in {"scan_timeline", "seek_evidence"}:
            tags.append("search_anchor")
        if tool_name in {"emit_alert", "verify_hypothesis"}:
            tags.append("alert_anchor")
        if tool_name in {"verify_hypothesis", "finalize_case"}:
            tags.append("evidence_anchor")
        return tags

    @staticmethod
    def _actual_search_branch(turn_info: Dict[str, Any]) -> Optional[str]:
        tags = turn_info.get("counterfactual_anchor_tags") or []
        if "search_anchor" not in tags:
            return None
        return "use_search"

    @staticmethod
    def _actual_alert_branch(turn_info: Dict[str, Any]) -> Optional[str]:
        tags = turn_info.get("counterfactual_anchor_tags") or []
        if "alert_anchor" not in tags:
            return None
        tool_name = str(turn_info.get("tool_name") or "")
        if tool_name == "emit_alert":
            return "alert_now"
        if tool_name == "finalize_case":
            return "defer_to_final"
        return "defer_to_next_decision"

    @staticmethod
    def _actual_evidence_branch(turn_info: Dict[str, Any]) -> Optional[str]:
        tags = turn_info.get("counterfactual_anchor_tags") or []
        if "evidence_anchor" not in tags:
            return None
        selected = turn_info.get("selected_window_ids_after") or []
        return "keep_selected" if selected else "full_ledger"

    @staticmethod
    def _build_counterfactual_anchor_summary(turns: List[Dict[str, Any]]) -> Dict[str, Any]:
        search_anchor_turns = [
            int(turn.get("step_index") or 0)
            for turn in turns
            if "search_anchor" in (turn.get("counterfactual_anchor_tags") or [])
        ]
        alert_anchor_turns = [
            int(turn.get("step_index") or 0)
            for turn in turns
            if "alert_anchor" in (turn.get("counterfactual_anchor_tags") or [])
        ]
        evidence_anchor_turns = [
            int(turn.get("step_index") or 0)
            for turn in turns
            if "evidence_anchor" in (turn.get("counterfactual_anchor_tags") or [])
        ]
        return {
            "num_search_anchors": len(search_anchor_turns),
            "num_alert_anchors": len(alert_anchor_turns),
            "num_evidence_anchors": len(evidence_anchor_turns),
            "search_anchor_turn_indices": search_anchor_turns,
            "alert_anchor_turn_indices": alert_anchor_turns,
            "evidence_anchor_turn_indices": evidence_anchor_turns,
        }

    @staticmethod
    def _decision_turn_indices(turns: List[Dict[str, Any]]) -> List[int]:
        return [
            int(turn.get("step_index") or 0)
            for turn in turns
            if str(turn.get("tool_name") or "") in {"emit_alert", "verify_hypothesis", "finalize_case"}
            or str(turn.get("action") or "") == "answer"
        ]

    @staticmethod
    def _build_latest_claim_trace(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        trace: List[Dict[str, Any]] = []
        for turn in turns:
            claim = turn.get("latest_claim_after")
            if claim is None:
                continue
            trace.append(
                {
                    "step_index": int(turn.get("step_index") or 0),
                    "claim": copy.deepcopy(claim),
                }
            )
        return trace

    @staticmethod
    def _build_latest_alert_trace(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        trace: List[Dict[str, Any]] = []
        for turn in turns:
            alert = turn.get("latest_alert_after")
            if alert is None:
                continue
            trace.append(
                {
                    "step_index": int(turn.get("step_index") or 0),
                    "alert": copy.deepcopy(alert),
                }
            )
        return trace

    @staticmethod
    def _build_search_trace(turns: List[Dict[str, Any]]) -> Dict[str, Any]:
        seek_turns = [turn for turn in turns if str(turn.get("tool_name") or "") == "seek_evidence"]
        backend_counts: Dict[str, int] = {}
        query_sequence: List[str] = []
        num_feature_guided = 0
        num_uniform_fallback = 0
        for turn in seek_turns:
            backend = str(turn.get("proposal_backend") or "unknown")
            backend_counts[backend] = backend_counts.get(backend, 0) + 1
            if backend == "feature_topk":
                num_feature_guided += 1
            if backend == "uniform":
                num_uniform_fallback += 1
            query_text = str(turn.get("proposal_query_normalized") or turn.get("proposal_query_raw") or "").strip()
            if query_text:
                query_sequence.append(query_text)
        num_query_revisions = 0
        for previous, current in zip(query_sequence, query_sequence[1:]):
            if current != previous:
                num_query_revisions += 1
        return {
            "num_seek_turns": len(seek_turns),
            "num_feature_guided_searches": num_feature_guided,
            "num_uniform_fallback_searches": num_uniform_fallback,
            "num_query_revisions": num_query_revisions,
            "query_sequence": query_sequence,
            "proposal_backend_counts": backend_counts,
        }
