"""SAVER agent package."""

from .adapter import TimeSearchRolloutAdapter
from .config import PromptConfig, PreviewConfig, RolloutTraceConfig, SaverAgentConfig
from .dataset import SaverAgentDataset
from .environment import SaverEnvironmentState, SaverVideoInteraction
from .qwen_policy import QwenGenerationPolicy
from .qwen_verifier import QwenSelfVerifier
from .reward import score_rollout_trace
from .rollout import ReplayPolicy, SaverRolloutRunner
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
    "run_counterfactual_verifier",
    "score_rollout_trace",
]
