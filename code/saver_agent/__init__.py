"""SAVER agent package."""

from .adapter import TimeSearchRolloutAdapter
from .config import PromptConfig, PreviewConfig, RolloutTraceConfig, SaverAgentConfig
from .dataset import SaverAgentDataset
from .environment import SaverEnvironmentState, SaverVideoInteraction
from .qwen_policy import QwenGenerationPolicy
from .teacher_judge import (
    QwenTeacherJudge,
    annotate_teacher_judge_examples,
    apply_teacher_judge_reweighting,
    attach_teacher_judge_labels,
    build_teacher_judge_messages,
    compute_teacher_judge_alignment,
    compute_teacher_judge_signal,
    compute_teacher_judge_weight_multiplier,
    has_teacher_judge_labels,
    is_teacher_judge_candidate,
    normalize_teacher_judge_result,
    parse_teacher_judge_response,
    reweight_teacher_judge_examples,
)
from .qwen_verifier import QwenSelfVerifier
from .reward import score_rollout_trace
from .rollout import ReplayPolicy, SaverRolloutRunner
from .self_verification import parse_self_verification_payload
from .verifier import run_counterfactual_verifier


def build_oracle_sft_examples(*args, **kwargs):
    from .training_data import build_oracle_sft_examples as _build_oracle_sft_examples

    return _build_oracle_sft_examples(*args, **kwargs)


def build_reward_weighted_examples(*args, **kwargs):
    from .training_data import build_reward_weighted_examples as _build_reward_weighted_examples

    return _build_reward_weighted_examples(*args, **kwargs)

__all__ = [
    "PromptConfig",
    "PreviewConfig",
    "QwenGenerationPolicy",
    "QwenTeacherJudge",
    "QwenSelfVerifier",
    "RolloutTraceConfig",
    "SaverAgentConfig",
    "TimeSearchRolloutAdapter",
    "SaverAgentDataset",
    "SaverEnvironmentState",
    "SaverVideoInteraction",
    "ReplayPolicy",
    "SaverRolloutRunner",
    "build_oracle_sft_examples",
    "build_reward_weighted_examples",
    "annotate_teacher_judge_examples",
    "apply_teacher_judge_reweighting",
    "build_teacher_judge_messages",
    "compute_teacher_judge_alignment",
    "compute_teacher_judge_signal",
    "compute_teacher_judge_weight_multiplier",
    "is_teacher_judge_candidate",
    "has_teacher_judge_labels",
    "parse_teacher_judge_response",
    "normalize_teacher_judge_result",
    "reweight_teacher_judge_examples",
    "attach_teacher_judge_labels",
    "parse_self_verification_payload",
    "run_counterfactual_verifier",
    "score_rollout_trace",
]
