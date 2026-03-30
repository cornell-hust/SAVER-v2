from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


DEFAULT_INITIAL_USER_TEMPLATE = (
    "Case ID: {public_case_id}\n"
    "Scene: {scene}\n"
    "Duration (sec): {duration}\n"
    "Task: {task_prompt}\n"
    "Success Criteria:\n"
    "{criteria_text}"
)

DEFAULT_PREVIEW_INSTRUCTION = (
    "You are given temporally ordered preview frames from the video. "
    "Use these preview frames to decide the next best action. "
    "If they are insufficient, call a tool to inspect more evidence."
)

DEFAULT_TOOL_RESPONSE_TEMPLATE = (
    "Here are selected frames. They are located at {timestamps}.\n"
    "Use them to decide the next tool step.\n"
    "If the evidence is already sufficient, call finalize_case next instead of answering directly. "
    "Only output <answer></answer> after a finalize_case observation explicitly asks for the terminal answer. "
    "Otherwise invoke the next tool with exactly one "
    '<tool_call>{{"name":"...","arguments":{{...}}}}</tool_call>.\n'
    "Do not describe the intended tool call in plain English. Do not output bare tool names. "
    "Do not output <answer> yet unless finalize_case has already returned.\n"
)


@dataclass
class PreviewConfig:
    num_preview_frames: int = 8
    preview_sampling_fps: Optional[float] = None
    max_preview_frames: int = 8


@dataclass
class PromptConfig:
    initial_user_template: str = DEFAULT_INITIAL_USER_TEMPLATE
    preview_instruction: str = DEFAULT_PREVIEW_INSTRUCTION
    tool_response_template: str = DEFAULT_TOOL_RESPONSE_TEMPLATE


@dataclass
class RolloutTraceConfig:
    record_observation_content: bool = False
    record_state_deltas: bool = True
    record_message_history: bool = True


@dataclass
class SaverAgentConfig:
    preview: PreviewConfig = field(default_factory=PreviewConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    rollout_trace: RolloutTraceConfig = field(default_factory=RolloutTraceConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
