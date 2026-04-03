from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Optional, Sequence

from convert_to_saver_agent import CanonicalSaverAdapter, ConverterConfig

from saver_agent.adapter import TimeSearchRolloutAdapter
from saver_agent.config import SaverAgentConfig
from saver_agent.environment import SaverEnvironmentState, SaverVideoInteraction
from saver_agent.proposal import normalize_query_text, render_query_package_texts, summarize_query_package
from saver_agent.reward import ALERT_STATUS_REWARD, DEFAULT_COMPONENT_WEIGHTS, PRIMARY_STATUS_REWARD
from saver_agent.self_verification import (
    DECISION_TO_PRIMARY_STATUS,
    build_policy_self_verification_payload,
)
from saver_agent.teacher_judge import compute_teacher_judge_signal
from saver_agent.verifier import (
    score_alert_counterfactual_group,
    score_evidence_counterfactual_group,
    score_search_counterfactual_group,
)


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _clip_text(text: str, *, max_len: int = 160) -> str:
    normalized = " ".join(str(text or "").strip().split())
    if len(normalized) <= max_len:
        return normalized
    return normalized[: max_len - 3].rstrip() + "..."


def _format_time_span(start_sec: Any, end_sec: Any) -> str:
    try:
        start_value = float(start_sec)
        end_value = float(end_sec)
        return f"{start_value:.1f}-{end_value:.1f}s"
    except Exception:
        return "this interval"


def _find_evidence_moment(record: Dict[str, Any], arguments: Dict[str, Any]) -> Dict[str, Any]:
    moment_id = arguments.get("moment_id")
    role = str(arguments.get("role") or "")
    start_sec = arguments.get("start_sec")
    end_sec = arguments.get("end_sec")
    for moment in (record.get("evidence") or {}).get("evidence_moments", []):
        if moment_id is not None and str(moment.get("moment_id")) == str(moment_id):
            return copy.deepcopy(moment)
    for moment in (record.get("evidence") or {}).get("evidence_moments", []):
        if role and str(moment.get("role") or "") == role:
            if (
                start_sec is None
                or end_sec is None
                or (
                    float(moment.get("start_sec") or -1.0) == float(start_sec)
                    and float(moment.get("end_sec") or -1.0) == float(end_sec)
                )
            ):
                return copy.deepcopy(moment)
    return {}


def _tool_reasoning_text(
    tool_name: str,
    arguments: Dict[str, Any],
    *,
    record: Dict[str, Any],
    state: SaverEnvironmentState,
) -> str:
    label = record.get("label") or {}
    structured_target = record.get("structured_target") or {}
    category = str(label.get("category") or structured_target.get("category") or "event")
    task_prompt = str((record.get("agent_task") or {}).get("task_prompt") or "").strip()
    summary = str((record.get("language") or {}).get("summary") or structured_target.get("summary") or "").strip()
    rationale = str((record.get("language") or {}).get("rationale") or structured_target.get("rationale") or "").strip()
    span_text = _format_time_span(arguments.get("start_sec"), arguments.get("end_sec"))

    if tool_name == "scan_timeline":
        if str(arguments.get("purpose") or "") == "global_overview":
            if task_prompt:
                return _clip_text(
                    f"I only have a limited preview, so I should scan {span_text} for a global overview before deciding how to investigate {task_prompt.lower()}."
                )
            return _clip_text(
                f"I only have a limited preview, so I should scan {span_text} for a global overview before making a decision about the clip."
            )
        return _clip_text(f"I should inspect {span_text} to refine my overview before taking the next step.")

    if tool_name == "seek_evidence":
        moment = _find_evidence_moment(record, arguments)
        role = str(arguments.get("role") or moment.get("role") or "evidence")
        description = str(moment.get("description") or "").strip()
        query = str(
            arguments.get("query")
            or summarize_query_package(arguments.get("query_package"))
            or ""
        ).strip()
        if description:
            return _clip_text(
                f"The next useful clue is the {role} around {span_text}; I should inspect it to check whether {description.lower()}."
            )
        if query:
            return _clip_text(
                f"I should inspect {span_text} for the {role} evidence and test whether the video shows {query.lower()}."
            )
        return _clip_text(f"I should inspect {span_text} for more targeted evidence about the suspected {category}.")

    if tool_name == "emit_alert":
        decision = str(arguments.get("decision") or "")
        evidence_count = len(state.evidence_ledger)
        if decision == "soft_alert":
            return _clip_text(
                f"I have enough visible support from {max(evidence_count, 1)} searched evidence window(s) to raise a soft alert for a possible {category}, but I still want verification before finalizing."
            )
        if decision == "hard_alert":
            return _clip_text(
                f"The searched evidence now looks strong and temporally consistent, so I should escalate this to a hard alert for {category}."
            )
        if decision == "declare_normal":
            if rationale:
                return _clip_text(
                    f"I do not see enough support for an anomaly in the inspected frames, so I should record a normal decision: {rationale}"
                )
            return _clip_text("I do not see enough support for an anomaly in the inspected frames, so I should record a normal decision.")
        return _clip_text("I should record the current alert decision so the next step can reason over it explicitly.")

    if tool_name == "verify_hypothesis":
        claim = arguments.get("claim") or state.last_claim or {}
        claim_existence = str(claim.get("existence") or structured_target.get("existence") or "current")
        claim_category = str(claim.get("category") or category)
        selected_window_ids = list(arguments.get("selected_window_ids") or [])
        selected_evidence_moment_ids = list(
            arguments.get("selected_evidence_moment_ids")
            or arguments.get("candidate_evidence_moment_ids")
            or arguments.get("evidence_moment_ids")
            or []
        )
        if selected_window_ids:
            candidate_count = len(selected_window_ids)
        else:
            candidate_count = len(selected_evidence_moment_ids)
        candidate_phrase = f"{candidate_count} selected evidence item(s)" if candidate_count > 0 else "the currently gathered evidence"
        return _clip_text(f"I should verify whether {candidate_phrase} are enough for the {claim_existence} claim about {claim_category}.")

    if tool_name == "finalize_case":
        existence = str(arguments.get("existence") or structured_target.get("existence") or "unknown")
        final_category = str(arguments.get("category") or category)
        if existence == "normal":
            return _clip_text("The inspected evidence is enough to conclude that the clip is normal, so I can finalize the case now.")
        if summary:
            return _clip_text(
                f"The searched evidence is sufficient and consistent, so I can finalize this as {final_category}: {summary}"
            )
        return _clip_text(f"The searched evidence is sufficient and consistent, so I can finalize this as {existence} / {final_category}.")

    if rationale:
        return _clip_text(f"I should act on the current evidence and rationale: {rationale}")
    return "I should take the next tool step based on the evidence collected so far."


def _answer_reasoning_text(answer_payload: Dict[str, Any], *, record: Dict[str, Any]) -> str:
    existence = str(answer_payload.get("existence") or (record.get("structured_target") or {}).get("existence") or "unknown")
    category = str(answer_payload.get("category") or (record.get("structured_target") or {}).get("category") or "case")
    summary = str(answer_payload.get("summary") or (record.get("structured_target") or {}).get("summary") or "").strip()
    if summary:
        return _clip_text(f"The evidence gathering is complete, so I can return the final structured answer: {summary}")
    if existence == "normal":
        return "The evidence gathering is complete, so I can return the final structured answer that this clip is normal."
    return _clip_text(f"The evidence gathering is complete, so I can return the final structured answer for the {category} case.")


def _assistant_tool_response(
    tool_name: str,
    arguments: Dict[str, Any],
    *,
    record: Dict[str, Any],
    state: SaverEnvironmentState,
) -> str:
    think_text = _tool_reasoning_text(tool_name, arguments, record=record, state=state)
    tool_payload = {"name": tool_name, "arguments": arguments}
    return f"<think>{think_text}</think><tool_call>{_json_dumps(tool_payload)}</tool_call>"


def _assistant_answer_response(answer_payload: Dict[str, Any], *, record: Dict[str, Any]) -> str:
    think_text = _answer_reasoning_text(answer_payload, record=record)
    return f"<think>{think_text}</think><answer>{_json_dumps(answer_payload)}</answer>"


def _extract_teacher_judge_labels(payload: Dict[str, Any]) -> Dict[str, Any]:
    labels: Dict[str, Any] = {}
    for key in ("teacher_judge_scores", "teacher_judge_decision", "teacher_judge_rationale"):
        if key in payload:
            labels[key] = copy.deepcopy(payload[key])
    return labels


def _merge_verify_arguments_with_oracle_feedback(
    arguments: Dict[str, Any],
    oracle_feedback: Any,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    if not isinstance(oracle_feedback, dict):
        return build_policy_self_verification_payload(arguments), {}
    merged_source = copy.deepcopy(arguments)
    merged_source.update(copy.deepcopy(oracle_feedback))
    return build_policy_self_verification_payload(merged_source), _extract_teacher_judge_labels(oracle_feedback)


def _apply_oracle_verifier_feedback(
    tool_message: Dict[str, Any],
    *,
    step: Dict[str, Any],
) -> Dict[str, Any]:
    if str(step.get("tool") or "") != "verify_hypothesis":
        return tool_message
    oracle_feedback = step.get("oracle_verifier_feedback")
    if not isinstance(oracle_feedback, dict):
        return tool_message

    arguments = dict(step.get("arguments") or {})
    merged_source = copy.deepcopy(arguments)
    merged_source.update(copy.deepcopy(oracle_feedback))
    payload = build_policy_self_verification_payload(merged_source)
    payload.update(_extract_teacher_judge_labels(oracle_feedback))
    return {
        "role": "tool",
        "name": "verify_hypothesis",
        "content": [{"type": "text", "text": _json_dumps(payload)}],
    }


def _serialize_message_content(
    content: List[Dict[str, Any]],
    *,
    multimodal_cache: Dict[str, Any],
) -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []
    video_path = str(multimodal_cache.get("video_path") or "")
    for item in content:
        if item.get("type") != "image" or "image" not in item:
            serialized.append(copy.deepcopy(item))
            continue

        image_ref: Dict[str, Any] = {"video_path": video_path}
        if item.get("sampled_frame_index") is not None:
            image_ref["sampled_frame_index"] = int(item["sampled_frame_index"])
        if item.get("raw_frame_index") is not None:
            image_ref["raw_frame_index"] = int(item["raw_frame_index"])
        if item.get("timestamp_sec") is not None:
            image_ref["timestamp_sec"] = float(item["timestamp_sec"])

        serialized_item = copy.deepcopy(item)
        serialized_item.pop("image", None)
        serialized_item.pop("sampled_frame_index", None)
        serialized_item.pop("raw_frame_index", None)
        serialized_item.pop("timestamp_sec", None)
        serialized_item["image_ref"] = image_ref
        serialized.append(serialized_item)
    return serialized


def _serialize_messages(
    messages: List[Dict[str, Any]],
    *,
    multimodal_cache: Dict[str, Any],
) -> List[Dict[str, Any]]:
    serialized_messages: List[Dict[str, Any]] = []
    for message in messages:
        serialized_message = copy.deepcopy(message)
        serialized_message["content"] = _serialize_message_content(
            list(message.get("content") or []),
            multimodal_cache=multimodal_cache,
        )
        serialized_messages.append(serialized_message)
    return serialized_messages


def _serialize_message(
    message: Dict[str, Any],
    *,
    multimodal_cache: Dict[str, Any],
) -> Dict[str, Any]:
    serialized_message = copy.deepcopy(message)
    serialized_message["content"] = _serialize_message_content(
        list(message.get("content") or []),
        multimodal_cache=multimodal_cache,
    )
    return serialized_message


def _ensure_agent_train_view(record: Dict[str, Any]) -> Dict[str, Any]:
    required_keys = {"agent_task", "structured_target", "tool_io"}
    if required_keys.issubset(record.keys()):
        return record
    adapter = CanonicalSaverAdapter(ConverterConfig())
    return adapter.convert(record, mode="agent_train")


def _ensure_oracle_sft(record: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(record.get("oracle_sft"), dict):
        return record["oracle_sft"]
    adapter = CanonicalSaverAdapter(ConverterConfig())
    agent_train_view = _ensure_agent_train_view(record)
    return adapter._build_oracle_sft(agent_train_view)  # Reuse the existing oracle heuristic for warm-start supervision.


def _clamp_score(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def _weighted_verifier_turn_credit(turn: Dict[str, Any]) -> float:
    derived = turn.get("verifier_derived_scores") or {}
    consistency = float(derived.get("consistency", 0.0) or 0.0)
    sufficiency = float(derived.get("sufficiency", 0.0) or 0.0)
    necessity = float(derived.get("necessity", 0.0) or 0.0)
    counterfactual = float(derived.get("counterfactual_faithfulness", 0.0) or 0.0)
    primary_status = str(turn.get("verifier_primary_status") or "")
    alert_status = str(turn.get("verifier_alert_status") or "")

    verification_reward = _clamp_score(
        float(PRIMARY_STATUS_REWARD.get(primary_status, 0.0)) + 0.2 * sufficiency + 0.2 * necessity
    )
    temporal_reward = _clamp_score(0.5 * consistency + 0.5 * sufficiency)
    alert_reward = float(ALERT_STATUS_REWARD.get(alert_status, 0.0))
    counterfactual_reward = _clamp_score(counterfactual)
    return (
        DEFAULT_COMPONENT_WEIGHTS["verification_reward"] * verification_reward
        + DEFAULT_COMPONENT_WEIGHTS["temporal_reward"] * temporal_reward
        + DEFAULT_COMPONENT_WEIGHTS["alert_reward"] * alert_reward
        + DEFAULT_COMPONENT_WEIGHTS["counterfactual_reward"] * counterfactual_reward
    )


def _compute_turn_credit(
    turn: Dict[str, Any],
    *,
    search_bonus: float,
    evidence_bonus: float,
    finalize_bonus: float,
    invalid_penalty: float,
    invalid_attempt_count: int = 0,
) -> float:
    valid_action = bool(turn.get("valid_action", turn.get("action") in {"tool_call", "answer"}))
    tool_name = str(turn.get("tool_name") or "")
    step_index = int(turn.get("step_index") or 0)
    turn_credit = 0.0

    if int(invalid_attempt_count) > 0:
        turn_credit -= abs(float(invalid_penalty)) * float(int(invalid_attempt_count))

    if not valid_action:
        turn_credit -= abs(float(invalid_penalty))

    if valid_action and tool_name in {"scan_timeline", "seek_evidence"}:
        turn_credit += float(search_bonus)

    turn_credit += float(evidence_bonus) * float(len(turn.get("new_evidence_ids") or []))

    if tool_name == "verify_hypothesis":
        turn_credit += _weighted_verifier_turn_credit(turn)

    if tool_name == "finalize_case" and turn.get("new_finalized_case") is not None:
        turn_credit += float(finalize_bonus)

    if str(turn.get("action") or "") == "answer":
        turn_credit += float(DEFAULT_COMPONENT_WEIGHTS["decision_reward"])

    if step_index > 1:
        turn_credit -= float(DEFAULT_COMPONENT_WEIGHTS["efficiency_reward"]) * 0.15 * float(step_index - 1)

    return float(turn_credit)


def _compute_turn_level_advantages(
    rollout: Dict[str, Any],
    *,
    gamma: float,
    alpha: float,
    search_bonus: float,
    evidence_bonus: float,
    finalize_bonus: float,
    invalid_penalty: float,
) -> List[Dict[str, float]]:
    turns = list(rollout.get("turns") or [])
    if not turns:
        return []

    invalid_attempts_by_step: Dict[int, int] = {}
    for invalid_attempt in list(rollout.get("invalid_attempts") or []):
        try:
            step_index = int(invalid_attempt.get("step_index") or 0)
        except Exception:
            step_index = 0
        if step_index <= 0:
            continue
        invalid_attempts_by_step[step_index] = invalid_attempts_by_step.get(step_index, 0) + 1

    rollout_advantage = float(
        rollout.get("group_advantage", (rollout.get("reward_summary") or {}).get("total_reward", 0.0)) or 0.0
    )
    turn_credits = [
        _compute_turn_credit(
            turn,
            search_bonus=search_bonus,
            evidence_bonus=evidence_bonus,
            finalize_bonus=finalize_bonus,
            invalid_penalty=invalid_penalty,
            invalid_attempt_count=invalid_attempts_by_step.get(int(turn.get("step_index") or 0), 0),
        )
        for turn in turns
    ]

    discounted_returns = [0.0 for _ in turn_credits]
    running_return = 0.0
    for idx in range(len(turn_credits) - 1, -1, -1):
        running_return = float(turn_credits[idx]) + float(gamma) * running_return
        discounted_returns[idx] = running_return

    if abs(rollout_advantage) < 1e-8:
        return [
            {
                "rollout_advantage": rollout_advantage,
                "turn_credit": float(turn_credits[idx]),
                "discounted_return": float(discounted_returns[idx]),
                "advantage": 0.0,
            }
            for idx in range(len(turn_credits))
        ]

    mean_turn_credit = sum(turn_credits) / float(len(turn_credits))
    centered_turn_credits = [value - mean_turn_credit for value in turn_credits]
    mean_abs_centered = sum(abs(value) for value in centered_turn_credits) / float(len(centered_turn_credits))

    if mean_abs_centered <= 1e-8:
        return [
            {
                "rollout_advantage": rollout_advantage,
                "turn_credit": float(turn_credits[idx]),
                "discounted_return": float(discounted_returns[idx]),
                "advantage": float(rollout_advantage),
            }
            for idx in range(len(turn_credits))
        ]

    advantages: List[Dict[str, float]] = []
    credit_scale = max(abs(float(rollout_advantage)), 0.25)
    for idx, centered_value in enumerate(centered_turn_credits):
        normalized_centered = float(centered_value) / float(mean_abs_centered)
        turn_advantage = float(rollout_advantage) + float(alpha) * credit_scale * normalized_centered
        advantages.append(
            {
                "rollout_advantage": rollout_advantage,
                "turn_credit": float(turn_credits[idx]),
                "discounted_return": float(discounted_returns[idx]),
                "advantage": float(turn_advantage),
            }
        )
    return advantages


def build_oracle_sft_examples(
    item: Dict[str, Any],
    record: Dict[str, Any],
    *,
    config: Optional[SaverAgentConfig] = None,
    serialize_messages: bool = False,
) -> List[Dict[str, Any]]:
    config = copy.deepcopy(config) if config is not None else SaverAgentConfig()
    adapter = TimeSearchRolloutAdapter(config=config)
    environment = SaverVideoInteraction()
    messages = adapter.build_initial_messages(item)
    multimodal_cache = item["multimodal_cache"]
    serialized_messages = (
        _serialize_messages(messages, multimodal_cache=multimodal_cache)
        if serialize_messages
        else None
    )
    state = SaverEnvironmentState()
    oracle_sft = _ensure_oracle_sft(record)
    trajectory = list(oracle_sft.get("trajectory") or [])
    examples: List[Dict[str, Any]] = []
    final_decision = copy.deepcopy(
        oracle_sft.get("final_decision")
        or record.get("structured_target")
        or state.finalized_case
        or {}
    )
    total_supervision_steps = len(trajectory) + (1 if final_decision else 0)
    normalized_sample_weight = 1.0 / float(max(total_supervision_steps, 1))

    for step_index, step in enumerate(trajectory, start=1):
        tool_name = str(step.get("tool") or "")
        arguments = copy.deepcopy(step.get("arguments") or {})
        response_arguments = copy.deepcopy(arguments)
        teacher_judge_labels: Dict[str, Any] = {}
        if tool_name == "verify_hypothesis":
            response_arguments, teacher_judge_labels = _merge_verify_arguments_with_oracle_feedback(
                arguments,
                step.get("oracle_verifier_feedback"),
            )
        response_text = _assistant_tool_response(
            tool_name,
            response_arguments,
            record=record,
            state=state,
        )
        example = {
            "video_id": item.get("video_id"),
            "split": item.get("split"),
            "step_index": step_index,
            "source": "oracle_sft",
            "target_action": "tool_call",
            "target_response": response_text,
            "messages": (
                copy.deepcopy(serialized_messages)
                if serialize_messages
                else copy.deepcopy(messages)
            ),
            "sample_weight": normalized_sample_weight,
            "tool_name": tool_name,
            "proposal_supervision": (
                _proposal_supervision_for_query(
                    record,
                    arguments.get("query", ""),
                    query_package=arguments.get("query_package"),
                )
                if tool_name == "seek_evidence"
                else {}
            ),
        }
        if teacher_judge_labels:
            example.update(teacher_judge_labels)
        examples.append(example)

        next_obs, _, _, _, next_states = environment.execute_predictions(
            [response_text],
            [multimodal_cache],
            [state],
            [True],
        )
        state = next_states[0]
        assistant_message = adapter.build_assistant_message(response_text)
        messages.append(assistant_message)
        if serialized_messages is not None:
            serialized_messages.append(copy.deepcopy(assistant_message))
        tool_message = next_obs[0]
        if isinstance(tool_message, dict) and tool_message.get("role") == "tool":
            tool_message = _apply_oracle_verifier_feedback(tool_message, step=step)
            adapted_tool_message = adapter.adapt_tool_observation(tool_message, multimodal_cache)
            messages.append(adapted_tool_message)
            if serialized_messages is not None:
                serialized_messages.append(
                    _serialize_message(
                        adapted_tool_message,
                        multimodal_cache=multimodal_cache,
                    )
                )

    if final_decision:
        examples.append(
            {
                "video_id": item.get("video_id"),
                "split": item.get("split"),
                "step_index": len(examples) + 1,
                "source": "oracle_sft",
                "target_action": "answer",
                "target_response": _assistant_answer_response(final_decision, record=record),
                "messages": (
                    copy.deepcopy(serialized_messages)
                    if serialize_messages
                    else copy.deepcopy(messages)
                ),
                "sample_weight": normalized_sample_weight,
                "tool_name": None,
            }
        )
    return examples


def _coerce_state_from_turn(turn: Dict[str, Any], key: str) -> SaverEnvironmentState:
    payload = _sanitize_state_selection_payload(dict(turn.get(key) or {}))
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


def _known_evidence_window_ids_from_state_payload(state_payload: Dict[str, Any] | SaverEnvironmentState) -> set[str]:
    if isinstance(state_payload, SaverEnvironmentState):
        entries = list(state_payload.evidence_ledger or [])
    else:
        entries = list((state_payload or {}).get("evidence_ledger") or [])
    return {
        str(entry.get("window_id")).strip()
        for entry in entries
        if str(entry.get("window_id") or "").strip()
    }


def _sanitize_window_ids_against_state(
    window_ids: Sequence[str],
    *,
    state_payload: Dict[str, Any] | SaverEnvironmentState,
) -> List[str]:
    known_window_ids = _known_evidence_window_ids_from_state_payload(state_payload)
    sanitized: List[str] = []
    seen = set()
    for value in window_ids or []:
        window_id = str(value).strip()
        if not window_id or window_id not in known_window_ids or window_id in seen:
            continue
        sanitized.append(window_id)
        seen.add(window_id)
    return sanitized


def _sanitize_state_selection_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = dict(payload or {})
    sanitized["active_evidence_window_ids"] = _sanitize_window_ids_against_state(
        sanitized.get("active_evidence_window_ids") or [],
        state_payload=sanitized,
    )
    verification_records = []
    for record in sanitized.get("verification_records") or []:
        updated_record = dict(record or {})
        for key in ("selected_window_ids", "verified_window_ids", "best_effort_window_ids"):
            if key in updated_record:
                updated_record[key] = _sanitize_window_ids_against_state(
                    updated_record.get(key) or [],
                    state_payload=sanitized,
                )
        verification_records.append(updated_record)
    sanitized["verification_records"] = verification_records
    return sanitized


def _latest_claim_from_turn(turn: Dict[str, Any]) -> Dict[str, Any]:
    claim = turn.get("latest_claim_after") or (turn.get("state_after") or {}).get("last_claim") or {}
    return copy.deepcopy(claim) if isinstance(claim, dict) else {}


def _latest_alert_from_turn(turn: Dict[str, Any]) -> Dict[str, Any]:
    alert = turn.get("latest_alert_after")
    if isinstance(alert, dict):
        return copy.deepcopy(alert)
    state_after = turn.get("state_after") or {}
    alerts = state_after.get("alerts") or []
    if alerts:
        return copy.deepcopy(alerts[-1])
    return {}


def _proposal_supervision_for_query(
    record: Dict[str, Any],
    query: str,
    *,
    query_package: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    proposal_supervision = record.get("proposal_supervision") or {}
    candidate_texts = set()
    normalized_query = normalize_query_text(str(query or ""))
    if normalized_query:
        candidate_texts.add(normalized_query)
    if isinstance(query_package, dict):
        summary_query = normalize_query_text(summarize_query_package(query_package))
        if summary_query:
            candidate_texts.add(summary_query)
        rendered = render_query_package_texts(query_package)
        for entry in rendered.get("positive_texts") or []:
            text = normalize_query_text(str(entry.get("text") or ""))
            if text:
                candidate_texts.add(text)

    best_match: Dict[str, Any] = {}
    best_score = 0.0
    for query_group in proposal_supervision.get("queries") or []:
        normalized_entries = list(query_group.get("normalized_queries") or [])
        normalized_texts = {
            normalize_query_text(str(entry.get("text") or ""))
            for entry in normalized_entries
            if str(entry.get("text") or "").strip()
        }
        overlap_count = len(candidate_texts & normalized_texts)
        if overlap_count <= 0:
            continue
        weight_bonus = 0.0
        for entry in normalized_entries:
            text = normalize_query_text(str(entry.get("text") or ""))
            if text in candidate_texts:
                weight_bonus = max(weight_bonus, float(entry.get("weight") or 0.0))
        score = float(overlap_count) + weight_bonus
        if score <= best_score:
            continue
        best_score = score
        best_match = {
            "query_id": query_group.get("query_id"),
            "raw_text": query_group.get("raw_text"),
            "normalized_queries": copy.deepcopy(normalized_entries),
            "linked_moment_ids": list(query_group.get("linked_moment_ids") or []),
            "linked_roles": list(query_group.get("linked_roles") or []),
            "linked_windows_sec": copy.deepcopy(query_group.get("linked_windows_sec") or []),
            "alignment_source": query_group.get("alignment_source"),
        }
    return best_match


def _proposal_metadata_from_turn(turn: Dict[str, Any]) -> Dict[str, Any]:
    if str(turn.get("tool_name") or "") != "seek_evidence":
        return {}
    return {
        "backend": turn.get("proposal_backend"),
        "feature_cache_used": turn.get("feature_cache_used"),
        "query_raw": turn.get("proposal_query_raw"),
        "query_normalized": turn.get("proposal_query_normalized"),
        "query_source": turn.get("proposal_query_source"),
        "candidate_count": turn.get("proposal_candidate_count"),
        "candidate_frame_indices": list(turn.get("proposal_candidate_frame_indices") or []),
        "candidate_frame_scores": list(turn.get("proposal_candidate_frame_scores") or []),
        "candidate_windows": copy.deepcopy(turn.get("proposal_candidate_windows") or []),
        "selected_frame_indices": list(turn.get("proposal_selected_frame_indices") or []),
        "selected_frame_scores": list(turn.get("proposal_selected_frame_scores") or []),
        "fallback_reason": turn.get("proposal_fallback_reason"),
    }


def _selected_window_ids_from_turn(turn: Dict[str, Any]) -> List[str]:
    state_after = _sanitize_state_selection_payload(dict(turn.get("state_after") or {}))
    values = (
        turn.get("selected_window_ids_after")
        or turn.get("verifier_verified_window_ids")
        or turn.get("verifier_best_effort_window_ids")
        or state_after.get("active_evidence_window_ids")
        or []
    )
    return _sanitize_window_ids_against_state(values, state_payload=state_after)


def _selected_evidence_ids_from_turn(turn: Dict[str, Any]) -> List[str]:
    values = turn.get("selected_evidence_ids_after") or []
    if values:
        return [str(value) for value in values if value]
    state_after = _sanitize_state_selection_payload(dict(turn.get("state_after") or {}))
    selected_window_ids = set(_selected_window_ids_from_turn(turn))
    evidence_ids: List[str] = []
    for entry in state_after.get("evidence_ledger") or []:
        window_id = str(entry.get("window_id") or "")
        evidence_id = entry.get("evidence_id")
        if window_id in selected_window_ids and evidence_id:
            evidence_ids.append(str(evidence_id))
    return evidence_ids


def _clamp_unit_score(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _clamp_signed_score(value: Any) -> float:
    try:
        return max(-1.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _group_relative_advantages_local(records: List[Dict[str, Any]], *, eps: float = 1e-6) -> List[Dict[str, Any]]:
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


def _selected_window_ids_from_state_payload(state_payload: Dict[str, Any]) -> List[str]:
    if isinstance(state_payload, SaverEnvironmentState):
        state_payload = {
            "evidence_ledger": list(state_payload.evidence_ledger or []),
            "active_evidence_window_ids": list(state_payload.active_evidence_window_ids or []),
            "verification_records": list(state_payload.verification_records or []),
        }
    state_payload = _sanitize_state_selection_payload(dict(state_payload or {}))
    active = state_payload.get("active_evidence_window_ids") or []
    if active:
        return [str(value) for value in active if value]
    verification_records = state_payload.get("verification_records") or []
    if verification_records:
        latest = verification_records[-1]
        values = latest.get("verified_window_ids") or latest.get("best_effort_window_ids") or []
        return [str(value) for value in values if value]
    return []


def _selected_evidence_ids_for_state_payload(
    state_payload: Dict[str, Any],
    window_ids: Sequence[str],
) -> List[str]:
    if isinstance(state_payload, SaverEnvironmentState):
        state_payload = {
            "evidence_ledger": list(state_payload.evidence_ledger or []),
        }
    selected = {str(value) for value in window_ids or [] if str(value)}
    if not selected:
        return []
    evidence_ids: List[str] = []
    for entry in state_payload.get("evidence_ledger") or []:
        window_id = str(entry.get("window_id") or "")
        evidence_id = entry.get("evidence_id")
        if window_id in selected and evidence_id:
            evidence_ids.append(str(evidence_id))
    return evidence_ids


def _state_clip_ratio_local(state_payload: Dict[str, Any], *, duration_sec: float) -> float:
    if isinstance(state_payload, SaverEnvironmentState):
        state_payload = {
            "visited_windows": list(state_payload.visited_windows or []),
            "evidence_ledger": list(state_payload.evidence_ledger or []),
        }
    if duration_sec <= 1e-8:
        return 0.0
    intervals: List[tuple[float, float]] = []
    entries = list(state_payload.get("visited_windows") or []) or list(state_payload.get("evidence_ledger") or [])
    for entry in entries:
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
    merged: List[tuple[float, float]] = []
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


def _policy_verification_signal_from_turn(turn: Dict[str, Any]) -> Dict[str, Any]:
    state_after = turn.get("state_after") or {}
    latest_record = {}
    verification_records = state_after.get("verification_records") or []
    if verification_records:
        latest_record = dict(verification_records[-1] or {})

    score_payload = (
        turn.get("self_verification_scores")
        or latest_record.get("self_verification_scores")
        or latest_record.get("derived_scores")
        or turn.get("verifier_derived_scores")
        or {}
    )
    sufficiency = _clamp_unit_score(score_payload.get("sufficiency"))
    necessity = _clamp_unit_score(score_payload.get("necessity"))
    alertability = _clamp_unit_score(score_payload.get("alertability"))
    counterfactual_faithfulness = _clamp_unit_score(score_payload.get("counterfactual_faithfulness"))
    decision = str(
        turn.get("self_verification_decision")
        or latest_record.get("verification_decision")
        or turn.get("verification_decision")
        or ""
    ).strip().lower()
    primary_status = str(
        turn.get("verifier_primary_status")
        or latest_record.get("primary_status")
        or DECISION_TO_PRIMARY_STATUS.get(decision, "")
    ).strip().lower()
    alert_status = str(
        turn.get("verifier_alert_status")
        or latest_record.get("alert_status")
        or "not_applicable"
    ).strip().lower()
    teacher_source = dict(turn)
    teacher_source["self_verification_decision"] = decision
    teacher_source["self_verification_scores"] = {
        "sufficiency": sufficiency,
        "necessity": necessity,
        "alertability": alertability,
        "counterfactual_faithfulness": counterfactual_faithfulness,
    }
    if latest_record.get("teacher_judge_scores") is not None and teacher_source.get("teacher_judge_scores") is None:
        teacher_source["teacher_judge_scores"] = copy.deepcopy(latest_record.get("teacher_judge_scores") or {})
    if latest_record.get("teacher_judge_decision") is not None and teacher_source.get("teacher_judge_decision") is None:
        teacher_source["teacher_judge_decision"] = latest_record.get("teacher_judge_decision")
    teacher_signal = compute_teacher_judge_signal(teacher_source)
    verification_quality = _clamp_signed_score(
        0.35 * float(PRIMARY_STATUS_REWARD.get(primary_status, 0.0))
        + 0.20 * (2.0 * sufficiency - 1.0)
        + 0.15 * (2.0 * necessity - 1.0)
        + 0.15 * (2.0 * counterfactual_faithfulness - 1.0)
        + 0.15 * (2.0 * alertability - 1.0)
    )
    return {
        "has_signal": bool(decision or primary_status or score_payload),
        "decision": decision,
        "primary_status": primary_status,
        "alert_status": alert_status,
        "scores": {
            "sufficiency": sufficiency,
            "necessity": necessity,
            "alertability": alertability,
            "counterfactual_faithfulness": counterfactual_faithfulness,
        },
        "verification_quality": verification_quality,
        "teacher_signal": teacher_signal,
        "teacher_reward": _clamp_signed_score(teacher_signal.get("teacher_judge_reward") or 0.0),
    }


def _find_future_verification_turn(turns: List[Dict[str, Any]], *, start_index: int) -> Optional[Dict[str, Any]]:
    for turn in turns:
        step_index = int(turn.get("step_index") or 0)
        if step_index <= int(start_index):
            continue
        signal = _policy_verification_signal_from_turn(turn)
        if signal.get("has_signal"):
            return turn
    return None


def _alert_timeliness_reward(claim: Dict[str, Any], alert: Dict[str, Any], *, delay_penalty: float) -> float:
    try:
        alert_sec = float(alert.get("alert_sec", alert.get("earliest_alert_sec")))
        earliest_alert_sec = float(claim.get("earliest_alert_sec"))
    except Exception:
        return 0.0
    if alert_sec <= earliest_alert_sec + 0.5:
        return 1.0
    return _clamp_signed_score(1.0 - abs(float(delay_penalty)) * max(0.0, alert_sec - earliest_alert_sec))


def _build_self_teacher_search_counterfactual_group(
    rollout: Dict[str, Any],
    anchor_turn: Dict[str, Any],
    *,
    multimodal_cache: Dict[str, Any],
    search_cost_penalty: float = 0.10,
) -> List[Dict[str, Any]]:
    state_before = dict(anchor_turn.get("state_before") or {})
    state_after = dict(anchor_turn.get("state_after") or {})
    current_signal = _policy_verification_signal_from_turn(anchor_turn)
    duration_sec = float(multimodal_cache.get("duration") or 0.0)
    state_delta = anchor_turn.get("state_delta") or {}
    new_evidence_windows = list(state_delta.get("new_evidence_windows") or [])
    new_visited_windows = list(state_delta.get("new_visited_windows") or [])
    new_window_ids = {
        str(entry.get("window_id") or "")
        for entry in (new_evidence_windows or new_visited_windows)
        if entry.get("window_id")
    }
    current_selected_window_ids = set(_selected_window_ids_from_turn(anchor_turn)) or set(new_window_ids)
    before_selected_window_ids = set(anchor_turn.get("selected_window_ids_before") or [])
    selected_overlap_den = len(current_selected_window_ids) or len(new_window_ids) or 1
    selected_overlap = float(len(new_window_ids & current_selected_window_ids)) / float(selected_overlap_den)
    new_evidence_gain = min(1.0, float(len(anchor_turn.get("new_evidence_ids") or [])))
    current_support = 1.0 if current_selected_window_ids else 0.0
    preexisting_support = 1.0 if before_selected_window_ids else 0.0
    proposal_backend = str(anchor_turn.get("proposal_backend") or "")
    if proposal_backend == "siglip_dpp":
        proposal_quality_bonus = 0.12
    elif proposal_backend == "feature_topk":
        proposal_quality_bonus = 0.10
    else:
        proposal_quality_bonus = 0.0
    coverage_gain = max(
        0.0,
        _state_clip_ratio_local(state_after, duration_sec=duration_sec)
        - _state_clip_ratio_local(state_before, duration_sec=duration_sec),
    )
    use_search_reward = _clamp_signed_score(
        0.35 * float(current_support)
        + 0.25 * float(new_evidence_gain)
        + 0.15 * float(selected_overlap)
        + 0.10 * float(current_signal.get("verification_quality") or 0.0)
        + 0.05 * float(current_signal.get("teacher_reward") or 0.0)
        + proposal_quality_bonus
        - abs(float(search_cost_penalty)) * float(coverage_gain)
    )
    skip_search_reward = _clamp_signed_score(
        0.55 * float(preexisting_support)
        - 0.20 * float(current_support)
        - 0.15 * float(new_evidence_gain)
        - 0.15 * float(selected_overlap)
        - 0.05 * proposal_quality_bonus
    )
    delta_only_reward = _clamp_signed_score(
        0.30 * float(current_support)
        + 0.35 * float(new_evidence_gain)
        + 0.20 * float(selected_overlap)
        + 0.05 * float(current_signal.get("teacher_reward") or 0.0)
        - 0.05 * abs(float(search_cost_penalty)) * float(coverage_gain)
    )
    records = [
        {
            "group_kind": "search",
            "branch_type": "skip_search",
            "anchor_turn_index": int(anchor_turn.get("step_index") or 0),
            "source_turn_index": int(anchor_turn.get("step_index") or 0),
            "selected_window_ids": list(before_selected_window_ids),
            "selected_evidence_ids": _selected_evidence_ids_for_state_payload(state_before, list(before_selected_window_ids)),
            "branch_reward": skip_search_reward,
            "branch_reward_components": {
                "preexisting_support": round(float(preexisting_support), 6),
                "current_support_penalty": round(-float(current_support), 6),
                "new_evidence_penalty": round(-float(new_evidence_gain), 6),
                "selected_overlap_penalty": round(-float(selected_overlap), 6),
            },
        },
        {
            "group_kind": "search",
            "branch_type": "use_search",
            "anchor_turn_index": int(anchor_turn.get("step_index") or 0),
            "source_turn_index": int(anchor_turn.get("step_index") or 0),
            "selected_window_ids": list(current_selected_window_ids or new_window_ids),
            "selected_evidence_ids": _selected_evidence_ids_from_turn(anchor_turn) or list(anchor_turn.get("new_evidence_ids") or []),
            "branch_reward": use_search_reward,
            "branch_reward_components": {
                "current_support": round(float(current_support), 6),
                "selected_overlap": round(float(selected_overlap), 6),
                "new_evidence_gain": round(float(new_evidence_gain), 6),
                "coverage_gain": round(float(coverage_gain), 6),
                "verification_quality": round(float(current_signal.get("verification_quality") or 0.0), 6),
                "teacher_reward": round(float(current_signal.get("teacher_reward") or 0.0), 6),
                "proposal_quality_bonus": round(float(proposal_quality_bonus), 6),
            },
        },
        {
            "group_kind": "search",
            "branch_type": "delta_only",
            "anchor_turn_index": int(anchor_turn.get("step_index") or 0),
            "source_turn_index": int(anchor_turn.get("step_index") or 0),
            "selected_window_ids": list(new_window_ids),
            "selected_evidence_ids": list(anchor_turn.get("new_evidence_ids") or []),
            "branch_reward": delta_only_reward,
            "branch_reward_components": {
                "current_support": round(float(current_support), 6),
                "marginal_overlap": round(max(float(selected_overlap), float(new_evidence_gain)), 6),
                "coverage_gain": round(float(coverage_gain), 6),
                "teacher_reward": round(float(current_signal.get("teacher_reward") or 0.0), 6),
            },
        },
    ]
    return _group_relative_advantages_local(records)


def _alert_claim_consistency_reward_local(claim: Dict[str, Any], alert: Dict[str, Any]) -> float:
    claim_existence = str(claim.get("existence") or "").strip().lower()
    decision = str(alert.get("decision") or "").strip().lower()
    if claim_existence == "anomaly":
        if decision in {"soft_alert", "hard_alert"}:
            return 1.0
        if decision == "declare_normal":
            return -1.0
        return -0.25
    if claim_existence == "normal":
        if decision == "declare_normal":
            return 1.0
        if decision in {"soft_alert", "hard_alert"}:
            return -1.0
        return 0.25
    return 0.0


def _build_self_teacher_alert_counterfactual_group(
    anchor_turn: Dict[str, Any],
    *,
    delay_penalty: float = 0.10,
) -> List[Dict[str, Any]]:
    if str(anchor_turn.get("tool_name") or "") != "emit_alert":
        return []
    signal = _policy_verification_signal_from_turn(anchor_turn)
    claim = _latest_claim_from_turn(anchor_turn)
    alert = _latest_alert_from_turn(anchor_turn)
    if not claim or not alert:
        return []

    state_before = dict(anchor_turn.get("state_before") or {})
    state_after = dict(anchor_turn.get("state_after") or {})
    before_selected_window_ids = _selected_window_ids_from_state_payload(state_before)
    selected_window_ids = _selected_window_ids_from_turn(anchor_turn) or _selected_window_ids_from_state_payload(state_after)
    selected_evidence_ids = _selected_evidence_ids_from_turn(anchor_turn) or _selected_evidence_ids_for_state_payload(
        state_after,
        selected_window_ids,
    )
    alertability = float((signal.get("scores") or {}).get("alertability") or 0.0)
    timeliness_reward = _alert_timeliness_reward(claim, alert, delay_penalty=delay_penalty)
    consistency_reward = _alert_claim_consistency_reward_local(claim, alert)
    claim_existence = str(claim.get("existence") or "").strip().lower()
    hold_reward = 0.75 if claim_existence == "normal" else (-0.65 if claim_existence == "anomaly" else 0.0)

    alert_now_reward = _clamp_signed_score(
        0.40 * float(consistency_reward)
        + 0.25 * float(timeliness_reward)
        + 0.15 * float(signal.get("verification_quality") or 0.0)
        + 0.10 * (2.0 * alertability - 1.0)
        + 0.10 * float(signal.get("teacher_reward") or 0.0)
    )
    hold_alert_reward = _clamp_signed_score(
        0.55 * float(hold_reward)
        - 0.20 * float(consistency_reward)
        - 0.15 * float(timeliness_reward if claim_existence == "anomaly" else 0.0)
        - 0.10 * float(signal.get("verification_quality") or 0.0)
    )
    records = [
        {
            "group_kind": "alert",
            "branch_type": "alert_now",
            "anchor_turn_index": int(anchor_turn.get("step_index") or 0),
            "source_turn_index": int(anchor_turn.get("step_index") or 0),
            "selected_window_ids": list(selected_window_ids),
            "selected_evidence_ids": list(selected_evidence_ids),
            "synthetic_alert": dict(alert),
            "branch_reward": alert_now_reward,
            "branch_reward_components": {
                "claim_alert_consistency": round(float(consistency_reward), 6),
                "timeliness_reward": round(float(timeliness_reward), 6),
                "verification_quality": round(float(signal.get("verification_quality") or 0.0), 6),
                "alertability": round(float(alertability), 6),
                "teacher_reward": round(float(signal.get("teacher_reward") or 0.0), 6),
            },
            "primary_status": signal.get("primary_status"),
            "alert_status": signal.get("alert_status"),
        },
        {
            "group_kind": "alert",
            "branch_type": "hold_alert",
            "anchor_turn_index": int(anchor_turn.get("step_index") or 0),
            "source_turn_index": int(anchor_turn.get("step_index") or 0),
            "selected_window_ids": list(before_selected_window_ids),
            "selected_evidence_ids": _selected_evidence_ids_for_state_payload(state_before, before_selected_window_ids),
            "synthetic_alert": {},
            "branch_reward": hold_alert_reward,
            "branch_reward_components": {
                "hold_reward": round(float(hold_reward), 6),
                "consistency_penalty": round(-float(consistency_reward), 6),
                "timeliness_penalty": round(-float(timeliness_reward if claim_existence == "anomaly" else 0.0), 6),
                "verification_quality_penalty": round(-float(signal.get("verification_quality") or 0.0), 6),
            },
            "primary_status": signal.get("primary_status"),
            "alert_status": signal.get("alert_status"),
        },
    ]
    return _group_relative_advantages_local(records)


def _choose_minimal_subset_window_ids_local(
    state_payload: Dict[str, Any],
    selected_window_ids: Sequence[str],
    *,
    subset_size: int = 2,
) -> List[str]:
    entries = [
        entry
        for entry in state_payload.get("evidence_ledger") or []
        if str(entry.get("window_id") or "") in {str(value) for value in selected_window_ids or []}
    ]
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
    ordered = sorted(
        entries,
        key=lambda entry: (
            role_priority.get(str(entry.get("role") or "").lower(), 4),
            -float(entry.get("end_sec") or 0.0),
            str(entry.get("window_id") or ""),
        ),
    )
    keep = max(1, min(int(subset_size), len(ordered)))
    return [str(entry.get("window_id")) for entry in ordered[:keep] if entry.get("window_id")]


def _build_self_teacher_evidence_counterfactual_group(
    rollout: Dict[str, Any],
    anchor_turn: Dict[str, Any],
    *,
    subset_size_penalty: float = 0.10,
) -> List[Dict[str, Any]]:
    signal_turn = anchor_turn
    signal = _policy_verification_signal_from_turn(signal_turn)
    if not signal.get("has_signal"):
        return []
    state_payload = dict(anchor_turn.get("state_after") or {})
    full_window_ids = [str(entry.get("window_id") or "") for entry in state_payload.get("evidence_ledger") or [] if entry.get("window_id")]
    keep_window_ids = [str(value) for value in _selected_window_ids_from_turn(anchor_turn) if value]
    drop_window_ids = [window_id for window_id in full_window_ids if window_id not in set(keep_window_ids)]
    minimal_window_ids = _choose_minimal_subset_window_ids_local(state_payload, keep_window_ids)
    full_ledger_size = max(1, len(full_window_ids))
    keep_ratio = float(len(keep_window_ids)) / float(full_ledger_size)
    minimal_ratio = float(len(minimal_window_ids)) / float(full_ledger_size) if minimal_window_ids else 0.0
    scores = signal.get("scores") or {}
    sufficiency = float(scores.get("sufficiency") or 0.0)
    necessity = float(scores.get("necessity") or 0.0)
    counterfactual_faithfulness = float(scores.get("counterfactual_faithfulness") or 0.0)
    signed_sufficiency = 2.0 * sufficiency - 1.0
    signed_necessity = 2.0 * necessity - 1.0
    signed_counterfactual = 2.0 * counterfactual_faithfulness - 1.0
    decision = str(signal.get("decision") or "")
    keep_reward = _clamp_signed_score(
        0.45 * float(signal.get("verification_quality") or 0.0)
        + 0.20 * signed_sufficiency
        + 0.20 * signed_necessity
        + 0.10 * signed_counterfactual
        + 0.05 * float(signal.get("teacher_reward") or 0.0)
        - 0.25 * abs(float(subset_size_penalty)) * keep_ratio
    )
    full_reward = _clamp_signed_score(
        0.35 * float(signal.get("verification_quality") or 0.0)
        + 0.20 * signed_sufficiency
        - 0.20 * signed_necessity
        + 0.10 * signed_counterfactual
        - abs(float(subset_size_penalty))
    )
    drop_reward = _clamp_signed_score(
        -0.45 * signed_sufficiency
        - 0.20 * float(signal.get("verification_quality") or 0.0)
        + 0.20 * (1.0 if decision in {"misaligned", "redundant"} else -0.5)
        - 0.05 * float(signal.get("teacher_reward") or 0.0)
    )
    minimal_reward = _clamp_signed_score(
        0.35 * float(signal.get("verification_quality") or 0.0)
        + 0.20 * signed_sufficiency
        + 0.30 * signed_necessity
        + 0.10 * signed_counterfactual
        + 0.05 * float(signal.get("teacher_reward") or 0.0)
        - abs(float(subset_size_penalty)) * minimal_ratio
    )
    if not keep_window_ids:
        full_reward = _clamp_signed_score(full_reward + 0.35)
        keep_reward = _clamp_signed_score(keep_reward - 0.35)
        minimal_reward = _clamp_signed_score(minimal_reward - 0.25)
    elif decision == "redundant":
        minimal_reward = _clamp_signed_score(minimal_reward + 0.15)
        full_reward = _clamp_signed_score(full_reward - 0.15)
    records = [
        {
            "group_kind": "evidence",
            "branch_type": "full_ledger",
            "anchor_turn_index": int(anchor_turn.get("step_index") or 0),
            "source_turn_index": int(signal_turn.get("step_index") or anchor_turn.get("step_index") or 0),
            "selected_window_ids": list(full_window_ids),
            "selected_evidence_ids": _selected_evidence_ids_for_state_payload(state_payload, full_window_ids),
            "branch_reward": full_reward,
            "branch_reward_components": {
                "verification_quality": round(float(signal.get("verification_quality") or 0.0), 6),
                "sufficiency": round(sufficiency, 6),
                "necessity_penalty": round(-signed_necessity, 6),
            },
            "primary_status": signal.get("primary_status"),
            "alert_status": signal.get("alert_status"),
        },
        {
            "group_kind": "evidence",
            "branch_type": "keep_selected",
            "anchor_turn_index": int(anchor_turn.get("step_index") or 0),
            "source_turn_index": int(signal_turn.get("step_index") or anchor_turn.get("step_index") or 0),
            "selected_window_ids": list(keep_window_ids),
            "selected_evidence_ids": _selected_evidence_ids_for_state_payload(state_payload, keep_window_ids),
            "branch_reward": keep_reward,
            "branch_reward_components": {
                "verification_quality": round(float(signal.get("verification_quality") or 0.0), 6),
                "sufficiency": round(sufficiency, 6),
                "necessity": round(necessity, 6),
                "teacher_reward": round(float(signal.get("teacher_reward") or 0.0), 6),
            },
            "primary_status": signal.get("primary_status"),
            "alert_status": signal.get("alert_status"),
        },
        {
            "group_kind": "evidence",
            "branch_type": "drop_selected",
            "anchor_turn_index": int(anchor_turn.get("step_index") or 0),
            "source_turn_index": int(signal_turn.get("step_index") or anchor_turn.get("step_index") or 0),
            "selected_window_ids": list(drop_window_ids),
            "selected_evidence_ids": _selected_evidence_ids_for_state_payload(state_payload, drop_window_ids),
            "branch_reward": drop_reward,
            "branch_reward_components": {
                "drop_bonus": round(1.0 if decision in {"misaligned", "redundant"} else -0.5, 6),
                "sufficiency_penalty": round(-signed_sufficiency, 6),
                "verification_quality": round(float(signal.get("verification_quality") or 0.0), 6),
            },
            "primary_status": signal.get("primary_status"),
            "alert_status": signal.get("alert_status"),
        },
        {
            "group_kind": "evidence",
            "branch_type": "minimal_subset",
            "anchor_turn_index": int(anchor_turn.get("step_index") or 0),
            "source_turn_index": int(signal_turn.get("step_index") or anchor_turn.get("step_index") or 0),
            "selected_window_ids": list(minimal_window_ids),
            "selected_evidence_ids": _selected_evidence_ids_for_state_payload(state_payload, minimal_window_ids),
            "branch_reward": minimal_reward,
            "branch_reward_components": {
                "verification_quality": round(float(signal.get("verification_quality") or 0.0), 6),
                "sufficiency": round(sufficiency, 6),
                "necessity": round(necessity, 6),
                "compactness_bonus": round(signed_necessity, 6),
            },
            "primary_status": signal.get("primary_status"),
            "alert_status": signal.get("alert_status"),
        },
    ]
    return _group_relative_advantages_local(records)


def _extract_alert_anchors(turns: List[Dict[str, Any]], *, max_anchors: int) -> List[Dict[str, Any]]:
    anchors: List[Dict[str, Any]] = []
    for turn in turns:
        tool_name = str(turn.get("tool_name") or "")
        tags = set(turn.get("counterfactual_anchor_tags") or [])
        if "alert_anchor" in tags or tool_name in {"emit_alert", "verify_hypothesis", "finalize_case"}:
            anchors.append(turn)
        if len(anchors) >= max(0, int(max_anchors)):
            break
    return anchors


def _extract_evidence_anchors(turns: List[Dict[str, Any]], *, max_anchors: int) -> List[Dict[str, Any]]:
    anchors: List[Dict[str, Any]] = []
    for turn in turns:
        tool_name = str(turn.get("tool_name") or "")
        tags = set(turn.get("counterfactual_anchor_tags") or [])
        if (
            "evidence_anchor" in tags
            or tool_name in {"verify_hypothesis", "finalize_case"}
        ) and (
            _selected_window_ids_from_turn(turn)
            or (turn.get("state_after") or {}).get("evidence_ledger")
        ):
            anchors.append(turn)
        if len(anchors) >= max(0, int(max_anchors)):
            break
    return anchors


def _extract_search_anchors(turns: List[Dict[str, Any]], *, max_anchors: int) -> List[Dict[str, Any]]:
    anchors: List[Dict[str, Any]] = []
    for turn in turns:
        tool_name = str(turn.get("tool_name") or "")
        state_delta = turn.get("state_delta") or {}
        new_evidence_windows = list(state_delta.get("new_evidence_windows") or [])
        new_visited_windows = list(state_delta.get("new_visited_windows") or [])
        if tool_name == "seek_evidence" and (
            turn.get("new_evidence_ids")
            or new_evidence_windows
            or new_visited_windows
        ):
            anchors.append(turn)
        if len(anchors) >= max(0, int(max_anchors)):
            break
    return anchors


def _find_next_decision_turn(turns: List[Dict[str, Any]], start_index: int) -> Optional[Dict[str, Any]]:
    for turn in turns:
        step_index = int(turn.get("step_index") or 0)
        if step_index <= int(start_index):
            continue
        if str(turn.get("tool_name") or "") in {"emit_alert", "verify_hypothesis", "finalize_case"}:
            return turn
    return None


def _find_future_claim_turn(turns: List[Dict[str, Any]], start_index: int) -> Optional[Dict[str, Any]]:
    next_decision_turn = _find_next_decision_turn(turns, start_index)
    if next_decision_turn is not None and _latest_claim_from_turn(next_decision_turn):
        return next_decision_turn
    for turn in turns:
        step_index = int(turn.get("step_index") or 0)
        if step_index <= int(start_index):
            continue
        if _latest_claim_from_turn(turn):
            return turn
    return turns[-1] if turns else None


def _build_search_counterfactual_group(
    rollout: Dict[str, Any],
    anchor_turn: Dict[str, Any],
    *,
    multimodal_cache: Dict[str, Any],
    local_verifier_backend: str,
    local_use_reference_supervision: bool,
) -> List[Dict[str, Any]]:
    if str(local_verifier_backend or "").strip().lower() == "self_teacher":
        records = _build_self_teacher_search_counterfactual_group(
            rollout,
            anchor_turn,
            multimodal_cache=multimodal_cache,
        )
        group_id = f"{rollout.get('video_id', 'unknown')}:search:{int(anchor_turn.get('step_index') or 0)}"
        for record in records:
            record["group_id"] = group_id
        return records
    turns = list(rollout.get("turns") or [])
    target_turn = _find_future_claim_turn(turns, int(anchor_turn.get("step_index") or 0)) or anchor_turn
    claim = _latest_claim_from_turn(target_turn) or _latest_claim_from_turn(anchor_turn)
    alert = _latest_alert_from_turn(target_turn) or _latest_alert_from_turn(anchor_turn) or None
    records = score_search_counterfactual_group(
        state_before=_coerce_state_from_turn(anchor_turn, "state_before"),
        state_after=_coerce_state_from_turn(anchor_turn, "state_after"),
        multimodal_cache=multimodal_cache,
        claim=claim,
        anchor_turn_index=int(anchor_turn.get("step_index") or 0),
        alert=alert,
        verifier_backend=local_verifier_backend,
        use_reference_supervision=local_use_reference_supervision,
    )
    group_id = f"{rollout.get('video_id', 'unknown')}:search:{int(anchor_turn.get('step_index') or 0)}"
    for record in records:
        record["group_id"] = group_id
    return records


def _build_alert_counterfactual_group(
    rollout: Dict[str, Any],
    anchor_turn: Dict[str, Any],
    *,
    multimodal_cache: Dict[str, Any],
    local_verifier_backend: str,
    local_use_reference_supervision: bool,
) -> List[Dict[str, Any]]:
    if str(local_verifier_backend or "").strip().lower() == "self_teacher":
        records = _build_self_teacher_alert_counterfactual_group(anchor_turn)
        group_id = f"{rollout.get('video_id', 'unknown')}:alert:{int(anchor_turn.get('step_index') or 0)}"
        for record in records:
            record["group_id"] = group_id
        return records
    turns = list(rollout.get("turns") or [])
    next_decision_turn = _find_next_decision_turn(turns, int(anchor_turn.get("step_index") or 0))
    final_turn = turns[-1] if turns else anchor_turn
    claim_now = _latest_claim_from_turn(anchor_turn)
    latest_alert_now = _latest_alert_from_turn(anchor_turn)
    if not latest_alert_now:
        latest_alert_now = {
            "decision": "soft_alert",
            "existence": claim_now.get("existence"),
            "category": claim_now.get("category"),
            "alert_sec": float(anchor_turn.get("observed_horizon_sec_after") or 0.0),
        }
    branches = [
        {
            "branch_type": "alert_now",
            "anchor_turn_index": int(anchor_turn.get("step_index") or 0),
            "source_turn_index": int(anchor_turn.get("step_index") or 0),
            "source_turn": anchor_turn,
            "state": _coerce_state_from_turn(anchor_turn, "state_after"),
            "claim": claim_now,
            "alert": latest_alert_now,
        }
    ]
    if next_decision_turn is not None:
        branches.append(
            {
                "branch_type": "defer_to_next_decision",
                "anchor_turn_index": int(anchor_turn.get("step_index") or 0),
                "source_turn_index": int(next_decision_turn.get("step_index") or 0),
                "source_turn": next_decision_turn,
                "state": _coerce_state_from_turn(next_decision_turn, "state_after"),
                "claim": _latest_claim_from_turn(next_decision_turn) or claim_now,
                "alert": _latest_alert_from_turn(next_decision_turn)
                or {
                    "decision": "hard_alert",
                    "existence": claim_now.get("existence"),
                    "category": claim_now.get("category"),
                    "alert_sec": float(next_decision_turn.get("observed_horizon_sec_after") or 0.0),
                },
            }
        )
    else:
        branches.append(
            {
                "branch_type": "defer_to_next_decision",
                "anchor_turn_index": int(anchor_turn.get("step_index") or 0),
                "source_turn_index": int(anchor_turn.get("step_index") or 0),
                "source_turn": anchor_turn,
                "state": _coerce_state_from_turn(anchor_turn, "state_after"),
                "claim": claim_now,
                "alert": latest_alert_now,
            }
        )
    branches.append(
        {
            "branch_type": "defer_to_final",
            "anchor_turn_index": int(anchor_turn.get("step_index") or 0),
            "source_turn_index": int(final_turn.get("step_index") or 0),
            "source_turn": final_turn,
            "state": _coerce_state_from_turn(final_turn, "state_after"),
            "claim": _latest_claim_from_turn(final_turn) or claim_now,
            "alert": _latest_alert_from_turn(final_turn)
            or {
                "decision": "hard_alert",
                "existence": claim_now.get("existence"),
                "category": claim_now.get("category"),
                "alert_sec": float(final_turn.get("observed_horizon_sec_after") or 0.0),
            },
        }
    )
    records = score_alert_counterfactual_group(
        branches=branches,
        multimodal_cache=multimodal_cache,
        verifier_backend=local_verifier_backend,
        use_reference_supervision=local_use_reference_supervision,
    )
    group_id = f"{rollout.get('video_id', 'unknown')}:alert:{int(anchor_turn.get('step_index') or 0)}"
    for record in records:
        record["group_id"] = group_id
    return records


def _build_evidence_counterfactual_group(
    rollout: Dict[str, Any],
    anchor_turn: Dict[str, Any],
    *,
    multimodal_cache: Dict[str, Any],
    local_verifier_backend: str,
    local_use_reference_supervision: bool,
) -> List[Dict[str, Any]]:
    if str(local_verifier_backend or "").strip().lower() == "self_teacher":
        records = _build_self_teacher_evidence_counterfactual_group(
            rollout,
            anchor_turn,
        )
        group_id = f"{rollout.get('video_id', 'unknown')}:evidence:{int(anchor_turn.get('step_index') or 0)}"
        for record in records:
            record["group_id"] = group_id
        return records
    selected_window_ids = _selected_window_ids_from_turn(anchor_turn)
    records = score_evidence_counterfactual_group(
        state=_coerce_state_from_turn(anchor_turn, "state_after"),
        multimodal_cache=multimodal_cache,
        claim=_latest_claim_from_turn(anchor_turn),
        selected_window_ids=selected_window_ids,
        anchor_turn_index=int(anchor_turn.get("step_index") or 0),
        alert=_latest_alert_from_turn(anchor_turn) or None,
        verifier_backend=local_verifier_backend,
        use_reference_supervision=local_use_reference_supervision,
    )
    group_id = f"{rollout.get('video_id', 'unknown')}:evidence:{int(anchor_turn.get('step_index') or 0)}"
    for record in records:
        record["group_id"] = group_id
    return records


def _turn_component_weights(
    turn: Dict[str, Any],
    *,
    search_local_alpha: float,
    alert_local_alpha: float,
    evidence_local_alpha: float,
    evidence_hit: bool = False,
    teacher_present: bool = False,
) -> Dict[str, float]:
    del search_local_alpha
    del alert_local_alpha
    del evidence_local_alpha
    tool_name = str(turn.get("tool_name") or "")
    action = str(turn.get("action") or "")
    weights = {"global": 1.0, "search_local": 0.0, "alert_local": 0.0, "evidence_local": 0.0, "teacher_local": 0.0}
    if tool_name == "seek_evidence":
        weights["search_local"] = 1.0
    if tool_name == "emit_alert":
        weights["alert_local"] = 1.0
        weights["evidence_local"] = 0.25
    elif tool_name == "verify_hypothesis":
        weights["alert_local"] = 0.5
        weights["evidence_local"] = 1.0
        if teacher_present:
            weights["teacher_local"] = 1.0
    elif tool_name == "finalize_case":
        weights["alert_local"] = 0.25
        weights["evidence_local"] = 1.0
    elif tool_name == "seek_evidence" and evidence_hit:
        weights["evidence_local"] = 0.5
    elif action == "answer":
        weights["alert_local"] = 0.5
        weights["evidence_local"] = 0.5
    return weights


def build_counterfactual_grpo_examples(
    item: Dict[str, Any],
    rollout: Dict[str, Any],
    *,
    config: Optional[SaverAgentConfig] = None,
    serialize_messages: bool = False,
    include_invalid: bool = False,
    turn_advantage_gamma: float = 0.9,
    turn_advantage_alpha: float = 0.5,
    turn_search_bonus: float = 0.05,
    turn_evidence_bonus: float = 0.1,
    turn_finalize_bonus: float = 0.2,
    turn_invalid_penalty: float = 0.75,
    local_verifier_backend: str = "self_teacher",
    local_use_reference_supervision: bool = False,
    search_local_alpha: float = 0.5,
    alert_local_alpha: float = 0.5,
    evidence_local_alpha: float = 0.5,
    teacher_local_alpha: float = 0.5,
    max_search_anchors: int = 2,
    max_alert_anchors: int = 2,
    max_evidence_anchors: int = 2,
    enable_search_group: bool = True,
    enable_alert_group: bool = True,
    enable_evidence_group: bool = True,
) -> List[Dict[str, Any]]:
    config = copy.deepcopy(config) if config is not None else SaverAgentConfig()
    adapter = TimeSearchRolloutAdapter(config=config)
    environment = SaverVideoInteraction()
    messages = adapter.build_initial_messages(item)
    multimodal_cache = item["multimodal_cache"]
    state = SaverEnvironmentState()
    turns = list(rollout.get("turns") or [])
    sample_weight = float((rollout.get("reward_summary") or {}).get("total_reward") or 0.0)
    turn_advantages = _compute_turn_level_advantages(
        rollout,
        gamma=turn_advantage_gamma,
        alpha=turn_advantage_alpha,
        search_bonus=turn_search_bonus,
        evidence_bonus=turn_evidence_bonus,
        finalize_bonus=turn_finalize_bonus,
        invalid_penalty=turn_invalid_penalty,
    )
    global_advantage_by_turn = {
        int(turn.get("step_index") or idx + 1): float(turn_advantages[idx]["advantage"]) if idx < len(turn_advantages) else sample_weight
        for idx, turn in enumerate(turns)
    }
    search_local_by_turn = {int(turn.get("step_index") or 0): 0.0 for turn in turns}
    alert_local_by_turn = {int(turn.get("step_index") or 0): 0.0 for turn in turns}
    evidence_local_by_turn = {int(turn.get("step_index") or 0): 0.0 for turn in turns}
    teacher_local_by_turn = {int(turn.get("step_index") or 0): 0.0 for turn in turns}
    group_ids_by_turn = {int(turn.get("step_index") or 0): [] for turn in turns}
    teacher_signal_by_turn: Dict[int, Dict[str, Any]] = {}

    for turn in turns:
        turn_index = int(turn.get("step_index") or 0)
        teacher_signal = compute_teacher_judge_signal(turn)
        teacher_signal_by_turn[turn_index] = teacher_signal
        if teacher_signal.get("teacher_judge_present"):
            teacher_local_by_turn[turn_index] = float(teacher_local_alpha) * float(
                teacher_signal.get("teacher_judge_reward") or 0.0
            )

    if enable_search_group:
        for anchor_turn in _extract_search_anchors(turns, max_anchors=max_search_anchors):
            records = _build_search_counterfactual_group(
                rollout,
                anchor_turn,
                multimodal_cache=multimodal_cache,
                local_verifier_backend=local_verifier_backend,
                local_use_reference_supervision=local_use_reference_supervision,
            )
            actual_record = next(
                (record for record in records if record.get("branch_type") == "use_search"),
                records[0] if records else None,
            )
            if actual_record is None:
                continue
            turn_index = int(anchor_turn.get("step_index") or 0)
            scaled_local_advantage = float(search_local_alpha) * float(actual_record.get("local_advantage") or 0.0)
            search_local_by_turn[turn_index] = search_local_by_turn.get(turn_index, 0.0) + scaled_local_advantage
            group_ids_by_turn.setdefault(turn_index, []).append(actual_record.get("group_id"))

    if enable_alert_group:
        for anchor_turn in _extract_alert_anchors(turns, max_anchors=max_alert_anchors):
            records = _build_alert_counterfactual_group(
                rollout,
                anchor_turn,
                multimodal_cache=multimodal_cache,
                local_verifier_backend=local_verifier_backend,
                local_use_reference_supervision=local_use_reference_supervision,
            )
            actual_branch = str(anchor_turn.get("counterfactual_actual_alert_branch") or "")
            if not actual_branch:
                actual_branch = (
                    "alert_now" if str(anchor_turn.get("tool_name") or "") == "emit_alert" else "defer_to_next_decision"
                )
            actual_record = next(
                (record for record in records if record.get("branch_type") == actual_branch),
                records[0] if records else None,
            )
            if actual_record is None:
                continue
            turn_index = int(anchor_turn.get("step_index") or 0)
            scaled_local_advantage = float(alert_local_alpha) * float(actual_record.get("local_advantage") or 0.0)
            alert_local_by_turn[turn_index] = alert_local_by_turn.get(turn_index, 0.0) + scaled_local_advantage
            group_ids_by_turn.setdefault(turn_index, []).append(actual_record.get("group_id"))

    if enable_evidence_group:
        for anchor_turn in _extract_evidence_anchors(turns, max_anchors=max_evidence_anchors):
            records = _build_evidence_counterfactual_group(
                rollout,
                anchor_turn,
                multimodal_cache=multimodal_cache,
                local_verifier_backend=local_verifier_backend,
                local_use_reference_supervision=local_use_reference_supervision,
            )
            actual_branch = str(anchor_turn.get("counterfactual_actual_evidence_branch") or "keep_selected")
            actual_record = next(
                (record for record in records if record.get("branch_type") == actual_branch),
                records[0] if records else None,
            )
            if actual_record is None:
                continue
            turn_index = int(anchor_turn.get("step_index") or 0)
            scaled_local_advantage = float(evidence_local_alpha) * float(actual_record.get("local_advantage") or 0.0)
            evidence_local_by_turn[turn_index] = evidence_local_by_turn.get(turn_index, 0.0) + scaled_local_advantage
            group_ids_by_turn.setdefault(turn_index, []).append(actual_record.get("group_id"))
            selected_evidence_ids = set(actual_record.get("selected_evidence_ids") or [])
            if selected_evidence_ids:
                for turn in turns:
                    if str(turn.get("tool_name") or "") != "seek_evidence":
                        continue
                    step_index = int(turn.get("step_index") or 0)
                    new_ids = set(turn.get("new_evidence_ids") or [])
                    if new_ids & selected_evidence_ids:
                        evidence_local_by_turn[step_index] = (
                            evidence_local_by_turn.get(step_index, 0.0) + 0.5 * scaled_local_advantage
                        )
                        group_ids_by_turn.setdefault(step_index, []).append(actual_record.get("group_id"))

    examples: List[Dict[str, Any]] = []
    for step_index, turn in enumerate(turns, start=1):
        response_text = str(turn.get("assistant_response_raw") or turn.get("response") or "")
        if not response_text:
            continue
        valid_action = bool(turn.get("valid_action", turn.get("action") in {"tool_call", "answer"}))
        if not include_invalid and not valid_action:
            continue
        global_advantage = float(global_advantage_by_turn.get(step_index, sample_weight))
        search_local = float(search_local_by_turn.get(step_index, 0.0))
        alert_local = float(alert_local_by_turn.get(step_index, 0.0))
        evidence_local = float(evidence_local_by_turn.get(step_index, 0.0))
        teacher_local = float(teacher_local_by_turn.get(step_index, 0.0))
        teacher_signal = teacher_signal_by_turn.get(step_index, {})
        evidence_hit = bool(turn.get("new_evidence_ids")) and evidence_local != 0.0
        turn_component_weights = _turn_component_weights(
            turn,
            search_local_alpha=search_local_alpha,
            alert_local_alpha=alert_local_alpha,
            evidence_local_alpha=evidence_local_alpha,
            evidence_hit=evidence_hit,
            teacher_present=bool(teacher_signal.get("teacher_judge_present")),
        )
        total_advantage = (
            float(turn_component_weights["global"]) * global_advantage
            + float(turn_component_weights["search_local"]) * search_local
            + float(turn_component_weights["alert_local"]) * alert_local
            + float(turn_component_weights["evidence_local"]) * evidence_local
            + float(turn_component_weights["teacher_local"]) * teacher_local
        )
        example = {
            "video_id": item.get("video_id"),
            "split": item.get("split"),
            "step_index": step_index,
            "source": "counterfactual_grpo_rollout",
            "target_action": turn.get("action"),
            "target_response": response_text,
            "messages": (
                _serialize_messages(messages, multimodal_cache=multimodal_cache)
                if serialize_messages
                else copy.deepcopy(messages)
            ),
            "sample_weight": max(float(total_advantage), 0.0),
            "advantage": float(total_advantage),
            "rollout_advantage": float(global_advantage),
            "advantage_components": {
                "global": float(global_advantage),
                "search_local": float(search_local),
                "alert_local": float(alert_local),
                "evidence_local": float(evidence_local),
                "teacher_local": float(teacher_local),
            },
            "turn_component_weights": turn_component_weights,
            "counterfactual_group_ids": [group_id for group_id in group_ids_by_turn.get(step_index, []) if group_id],
            "counterfactual_anchor_kind": list(turn.get("counterfactual_anchor_tags") or []),
            "counterfactual_anchor_turn_index": int(step_index),
            "counterfactual_metadata": {
                "actual_search_branch": turn.get("counterfactual_actual_search_branch"),
                "actual_alert_branch": turn.get("counterfactual_actual_alert_branch"),
                "actual_evidence_branch": turn.get("counterfactual_actual_evidence_branch"),
                "selected_window_ids": _selected_window_ids_from_turn(turn),
                "selected_evidence_ids": _selected_evidence_ids_from_turn(turn),
            },
            "tool_name": turn.get("tool_name"),
            "proposal_metadata": _proposal_metadata_from_turn(turn),
            "teacher_judge_present": bool(teacher_signal.get("teacher_judge_present")),
            "teacher_judge_alignment": teacher_signal.get("teacher_judge_alignment"),
            "teacher_judge_score_agreement": float(teacher_signal.get("teacher_judge_score_agreement") or 0.0),
            "teacher_judge_confidence": float(teacher_signal.get("teacher_judge_confidence") or 0.0),
            "teacher_judge_reward": float(teacher_signal.get("teacher_judge_reward") or 0.0),
        }
        if "teacher_judge_scores" in turn:
            example["teacher_judge_scores"] = copy.deepcopy(turn.get("teacher_judge_scores") or {})
        if turn.get("teacher_judge_decision") is not None:
            example["teacher_judge_decision"] = turn.get("teacher_judge_decision")
        if turn.get("teacher_judge_rationale") is not None:
            example["teacher_judge_rationale"] = turn.get("teacher_judge_rationale")
        examples.append(example)

        next_obs, _, _, _, next_states = environment.execute_predictions(
            [response_text],
            [multimodal_cache],
            [state],
            [True],
        )
        state = next_states[0]
        messages.append(adapter.build_assistant_message(response_text))
        tool_message = next_obs[0]
        if isinstance(tool_message, dict) and tool_message.get("role") == "tool":
            messages.append(adapter.adapt_tool_observation(tool_message, multimodal_cache))
    return examples


def build_reward_weighted_examples(
    item: Dict[str, Any],
    rollout: Dict[str, Any],
    *,
    config: Optional[SaverAgentConfig] = None,
    include_invalid: bool = False,
    serialize_messages: bool = False,
    turn_advantage_gamma: float = 0.9,
    turn_advantage_alpha: float = 0.5,
    turn_search_bonus: float = 0.05,
    turn_evidence_bonus: float = 0.1,
    turn_finalize_bonus: float = 0.2,
    turn_invalid_penalty: float = 0.75,
) -> List[Dict[str, Any]]:
    config = copy.deepcopy(config) if config is not None else SaverAgentConfig()
    adapter = TimeSearchRolloutAdapter(config=config)
    environment = SaverVideoInteraction()
    messages = adapter.build_initial_messages(item)
    multimodal_cache = item["multimodal_cache"]
    state = SaverEnvironmentState()
    sample_weight = float((rollout.get("reward_summary") or {}).get("total_reward") or 0.0)
    turn_advantages = _compute_turn_level_advantages(
        rollout,
        gamma=turn_advantage_gamma,
        alpha=turn_advantage_alpha,
        search_bonus=turn_search_bonus,
        evidence_bonus=turn_evidence_bonus,
        finalize_bonus=turn_finalize_bonus,
        invalid_penalty=turn_invalid_penalty,
    )
    examples: List[Dict[str, Any]] = []

    for step_index, turn in enumerate(rollout.get("turns") or [], start=1):
        response_text = str(turn.get("assistant_response_raw") or turn.get("response") or "")
        if not response_text:
            continue
        valid_action = bool(turn.get("valid_action", turn.get("action") in {"tool_call", "answer"}))
        teacher_signal = compute_teacher_judge_signal(turn)
        turn_advantage_info = (
            turn_advantages[step_index - 1]
            if step_index - 1 < len(turn_advantages)
            else {
                "rollout_advantage": sample_weight,
                "turn_credit": 0.0,
                "discounted_return": 0.0,
                "advantage": sample_weight,
            }
        )
        if include_invalid or valid_action:
            example = {
                "video_id": item.get("video_id"),
                "split": item.get("split"),
                "step_index": step_index,
                "source": "reward_weighted_rollout",
                "target_action": turn.get("action"),
                "target_response": response_text,
                "messages": (
                    _serialize_messages(messages, multimodal_cache=multimodal_cache)
                    if serialize_messages
                    else copy.deepcopy(messages)
                ),
                "sample_weight": float(turn_advantage_info["advantage"]),
                "advantage": float(turn_advantage_info["advantage"]),
                "rollout_advantage": float(turn_advantage_info["rollout_advantage"]),
                "advantage_metadata": {
                    "turn_credit": float(turn_advantage_info["turn_credit"]),
                    "discounted_return": float(turn_advantage_info["discounted_return"]),
                    "gamma": float(turn_advantage_gamma),
                    "alpha": float(turn_advantage_alpha),
                    "teacher_judge_reward": float(teacher_signal.get("teacher_judge_reward") or 0.0),
                    "teacher_judge_alignment": teacher_signal.get("teacher_judge_alignment"),
                },
                "tool_name": turn.get("tool_name"),
                "proposal_metadata": _proposal_metadata_from_turn(turn),
                "teacher_judge_present": bool(teacher_signal.get("teacher_judge_present")),
                "teacher_judge_alignment": teacher_signal.get("teacher_judge_alignment"),
                "teacher_judge_score_agreement": float(teacher_signal.get("teacher_judge_score_agreement") or 0.0),
                "teacher_judge_confidence": float(teacher_signal.get("teacher_judge_confidence") or 0.0),
                "teacher_judge_reward": float(teacher_signal.get("teacher_judge_reward") or 0.0),
            }
            if "teacher_judge_scores" in turn:
                example["teacher_judge_scores"] = copy.deepcopy(turn.get("teacher_judge_scores") or {})
            if turn.get("teacher_judge_decision") is not None:
                example["teacher_judge_decision"] = turn.get("teacher_judge_decision")
            if turn.get("teacher_judge_rationale") is not None:
                example["teacher_judge_rationale"] = turn.get("teacher_judge_rationale")
            examples.append(example)

        next_obs, _, _, _, next_states = environment.execute_predictions(
            [response_text],
            [multimodal_cache],
            [state],
            [True],
        )
        state = next_states[0]
        messages.append(adapter.build_assistant_message(response_text))
        tool_message = next_obs[0]
        if isinstance(tool_message, dict) and tool_message.get("role") == "tool":
            messages.append(adapter.adapt_tool_observation(tool_message, multimodal_cache))
    return examples
