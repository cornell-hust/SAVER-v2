#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from split_utils import parse_include_splits

from run_saver_rollout import _serialize_result
from saver_agent.adapter import TimeSearchRolloutAdapter
from saver_agent.config import PromptConfig, PreviewConfig, RolloutTraceConfig, SaverAgentConfig
from saver_agent.dataset import SaverAgentDataset
from saver_agent.evaluation import RolloutEvaluationConfig
from saver_agent.offline_scoring import (
    ReferenceDataProvider,
    load_rollout_records,
    save_rollout_records,
    score_rollout_records,
)
from saver_agent.qwen_policy import DEFAULT_MODEL_PATH, QwenGenerationPolicy
from saver_agent.qwen_verifier import DEFAULT_VERIFIER_MODEL_PATH, QwenSelfVerifier
from saver_agent.runtime import (
    distributed_barrier,
    distributed_runtime_from_env,
    init_torch_distributed,
    resolve_inference_device_map,
    resolve_shard_spec,
    runtime_log,
    shard_sequence,
    should_log_progress,
)
from saver_agent.rollout import SaverRolloutRunner
from saver_agent.training import run_weighted_sft
from saver_agent.training_data import build_counterfactual_grpo_examples, build_reward_weighted_examples


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAVER RL: grouped rollouts + rollout/turn/token credit assignment + PPO/GRPO-style clipped policy updates."
    )
    parser.add_argument("--data", required=True, help="Path to SAVER agent/oracle JSONL data.")
    parser.add_argument("--data-root", default="", help="Root path used to resolve relative video paths.")
    parser.add_argument("--include-splits", default="", help="Optional comma-separated split whitelist for --data, e.g. train or train,val.")
    parser.add_argument("--output-dir", required=True, help="Directory to store iterative RL outputs.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Initial Qwen policy checkpoint.")
    parser.add_argument(
        "--reference-model-path",
        default="",
        help="Optional fixed reference policy checkpoint. Defaults to the initial --model-path.",
    )
    parser.add_argument("--num-iterations", type=int, default=1, help="Number of rollout-update iterations.")
    parser.add_argument("--rollout-count", type=int, default=16, help="Number of videos per iteration.")
    parser.add_argument("--num-generations", type=int, default=4, help="Number of sampled rollouts per video per iteration.")
    parser.add_argument("--rollout-start-index", type=int, default=0, help="Start index for the first iteration.")
    parser.add_argument("--rollout-max-turns", type=int, default=12, help="Maximum rollout turns.")
    parser.add_argument("--dry-run", action="store_true", help="Collect/score examples but skip gradient updates.")
    parser.add_argument(
        "--min-weight",
        type=float,
        default=0.1,
        help="Minimum absolute rollout advantage kept for updates. Applies to both positive and negative examples.",
    )
    parser.add_argument("--advantage-clip", type=float, default=3.0, help="Absolute clip value for group-relative advantages.")
    parser.add_argument(
        "--training-objective",
        choices=["grpo", "weighted_sft"],
        default="grpo",
        help="Policy optimization objective used for each RL update. `grpo` is the recommended default.",
    )
    parser.add_argument(
        "--grpo-variant",
        choices=["standard", "cea_grpo"],
        default="cea_grpo",
        help="GRPO data construction variant. Defaults to `cea_grpo`, which enables Counterfactual Evidence-and-Alert GRPO examples.",
    )
    parser.add_argument(
        "--ppo-clip-epsilon",
        type=float,
        default=0.2,
        help="Clipping epsilon used by the PPO/GRPO-style surrogate objective.",
    )
    parser.add_argument("--kl-beta", type=float, default=0.0, help="KL regularization weight against the fixed reference policy.")
    parser.add_argument("--max-train-examples", type=int, default=0, help="Optional limit on RL training examples per iteration.")
    parser.add_argument("--torch-dtype", default="auto", help="Torch dtype passed to from_pretrained.")
    parser.add_argument("--attn-implementation", default="", help="Optional attention backend for policy model.")
    parser.add_argument("--policy-device-map", default="auto", help="device_map used for rollout policy inference.")
    parser.add_argument("--policy-do-sample", action="store_true", help="Enable sampling for rollout generation.")
    parser.add_argument("--policy-temperature", type=float, default=None, help="Sampling temperature.")
    parser.add_argument("--policy-top-p", type=float, default=None, help="Sampling top-p.")
    parser.add_argument("--policy-top-k", type=int, default=None, help="Sampling top-k.")
    parser.add_argument("--policy-repetition-penalty", type=float, default=None, help="Sampling repetition penalty.")
    parser.add_argument("--policy-max-new-tokens", type=int, default=512, help="Policy generation length.")
    parser.add_argument("--verifier-backend", choices=["heuristic", "qwen_self_verifier", "hybrid"], default="hybrid")
    parser.add_argument("--verifier-model-path", default=DEFAULT_VERIFIER_MODEL_PATH, help="Verifier model path.")
    parser.add_argument("--verifier-torch-dtype", default="auto", help="Torch dtype for verifier.")
    parser.add_argument("--verifier-device-map", default="auto", help="device_map used for verifier inference.")
    parser.add_argument("--verifier-attn-implementation", default="", help="Optional attention backend for verifier.")
    parser.add_argument("--verifier-max-new-tokens", type=int, default=512, help="Verifier generation length.")
    parser.add_argument("--verifier-hybrid-alpha", type=float, default=0.7, help="Hybrid verifier alpha.")
    parser.add_argument("--turn-advantage-gamma", type=float, default=0.9, help="Discount factor used when logging per-turn credit returns.")
    parser.add_argument("--turn-advantage-alpha", type=float, default=0.5, help="Strength of turn-level credit redistribution around the rollout-level advantage.")
    parser.add_argument("--turn-search-bonus", type=float, default=0.05, help="Small positive shaping for valid search turns in turn-level advantage assignment.")
    parser.add_argument("--turn-evidence-bonus", type=float, default=0.1, help="Bonus per newly added evidence item in turn-level advantage assignment.")
    parser.add_argument("--turn-finalize-bonus", type=float, default=0.2, help="Bonus for successful finalize_case turns in turn-level advantage assignment.")
    parser.add_argument("--turn-invalid-penalty", type=float, default=0.75, help="Penalty for invalid turns in turn-level advantage assignment.")
    parser.add_argument(
        "--cea-enable-alert-group",
        action="store_true",
        help="Enable alert counterfactual groups for CEA-GRPO. If no CEA group flags are set, both alert and evidence groups are enabled by default.",
    )
    parser.add_argument(
        "--cea-enable-evidence-group",
        action="store_true",
        help="Enable evidence counterfactual groups for CEA-GRPO. If no CEA group flags are set, both alert and evidence groups are enabled by default.",
    )
    parser.add_argument(
        "--cea-enable-search-group",
        action="store_true",
        help="Enable search counterfactual groups for CEA-GRPO. If no CEA group flags are set, alert/evidence/search groups are all enabled by default.",
    )
    parser.add_argument(
        "--cea-alert-local-alpha",
        type=float,
        default=0.5,
        help="Scale applied to alert-local CEA advantages before they are mixed into the turn/token objective.",
    )
    parser.add_argument(
        "--cea-evidence-local-alpha",
        type=float,
        default=0.5,
        help="Scale applied to evidence-local CEA advantages before they are mixed into the turn/token objective.",
    )
    parser.add_argument(
        "--cea-search-local-alpha",
        type=float,
        default=0.5,
        help="Scale applied to search-local CEA advantages before they are mixed into the turn/token objective.",
    )
    parser.add_argument(
        "--cea-local-use-reference-supervision",
        action="store_true",
        help="Allow reference-conditioned supervision inside local counterfactual branch scoring. Use for ablations/diagnostics only.",
    )
    parser.add_argument(
        "--cea-local-verifier-backend",
        choices=["heuristic", "qwen_self_verifier", "hybrid"],
        default="heuristic",
        help="Verifier backend used for local counterfactual branch scoring in CEA-GRPO.",
    )
    parser.add_argument(
        "--cea-max-alert-anchors-per-rollout",
        type=int,
        default=2,
        help="Maximum number of alert anchors extracted from each rollout when building CEA-GRPO examples.",
    )
    parser.add_argument(
        "--cea-max-evidence-anchors-per-rollout",
        type=int,
        default=2,
        help="Maximum number of evidence anchors extracted from each rollout when building CEA-GRPO examples.",
    )
    parser.add_argument(
        "--cea-max-search-anchors-per-rollout",
        type=int,
        default=2,
        help="Maximum number of search anchors extracted from each rollout when building CEA-GRPO examples.",
    )
    parser.add_argument("--progress-every", type=int, default=1, help="Log rollout/score progress every N local items.")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable model gradient checkpointing.")
    parser.add_argument("--learning-rate", type=float, default=5e-6, help="Update learning rate.")
    parser.add_argument("--num-train-epochs", type=float, default=1.0, help="Update epochs per iteration.")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1, help="Per-device update batch size.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16, help="Gradient accumulation steps.")
    parser.add_argument("--dataloader-num-workers", type=int, default=0, help="DataLoader worker count for each RL update stage.")
    parser.add_argument(
        "--dataloader-prefetch-factor",
        type=int,
        default=0,
        help="Optional DataLoader prefetch_factor for RL updates. Only used when dataloader_num_workers > 0.",
    )
    parser.add_argument(
        "--dataloader-persistent-workers",
        action="store_true",
        help="Keep RL update DataLoader workers alive across epochs. Only used when dataloader_num_workers > 0.",
    )
    parser.add_argument("--logging-steps", type=int, default=10, help="Trainer logging steps.")
    parser.add_argument("--save-steps", type=int, default=100, help="Trainer save steps.")
    parser.add_argument("--save-total-limit", type=int, default=2, help="Trainer save_total_limit.")
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="Trainer warmup ratio.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Trainer weight decay.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Trainer max grad norm.")
    parser.add_argument("--bf16", action="store_true", help="Enable bf16 training.")
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 training.")
    parser.add_argument("--lora", action="store_true", help="Use PEFT LoRA adapters.")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument("--lora-target-modules", default="", help="Comma-separated LoRA target module names.")
    parser.add_argument("--max-image-side", type=int, default=640, help="Optional training-time max image side length in pixels for the RL update stage. 0 disables resizing.")
    parser.add_argument("--max-image-pixels", type=int, default=0, help="Optional training-time max image area in pixels for the RL update stage. 0 disables resizing.")
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=4096,
        help="Explicit tokenizer/processor max_length for RL update examples. Uses left truncation to keep recent turns and the supervised response. 0 disables truncation.",
    )
    parser.add_argument(
        "--keep-recent-tool-image-messages",
        type=int,
        default=0,
        help="If >0, keep images only for the N most recent tool messages during RL updates; older tool images are dropped.",
    )
    parser.add_argument(
        "--keep-recent-text-messages",
        type=int,
        default=12,
        help="If >0, keep full text only for the N most recent non-initial history messages during RL updates; older assistant/tool history is dropped before tokenization.",
    )
    parser.add_argument(
        "--max-total-images",
        type=int,
        default=44,
        help="Optional hard cap on total images kept in each RL update example after pruning. 0 keeps all images.",
    )
    parser.add_argument("--num-preview-frames", type=int, default=8, help="Preview frames for initial prompt.")
    parser.add_argument("--preview-sampling-fps", type=float, default=None, help="Preview sampling fps.")
    parser.add_argument("--eval-data", default="", help="Optional raw saver_agent/oracle JSONL used for rollout metrics after each update epoch.")
    parser.add_argument("--eval-data-root", default="", help="Root path used to resolve relative video paths for epoch-end rollout eval.")
    parser.add_argument("--eval-include-splits", default="", help="Optional comma-separated split whitelist for --eval-data.")
    parser.add_argument("--eval-max-records", type=int, default=0, help="Optional cap on eval records per epoch-end rollout eval.")
    parser.add_argument("--eval-rollout-max-turns", type=int, default=12, help="Maximum rollout turns for epoch-end rollout eval.")
    parser.add_argument(
        "--eval-verifier-backend",
        choices=["heuristic", "qwen_self_verifier", "hybrid"],
        default="heuristic",
        help="Offline verifier backend used for epoch-end rollout metrics.",
    )
    parser.add_argument("--eval-verifier-model-path", default="", help="Optional verifier model path for epoch-end rollout eval.")
    parser.add_argument("--eval-verifier-torch-dtype", default="auto", help="Torch dtype for epoch-end eval verifier.")
    parser.add_argument("--eval-verifier-device-map", default="auto", help="device_map for epoch-end eval verifier.")
    parser.add_argument("--eval-verifier-attn-implementation", default="", help="Attention backend for epoch-end eval verifier.")
    parser.add_argument("--eval-verifier-max-new-tokens", type=int, default=512, help="Generation length for epoch-end eval verifier.")
    parser.add_argument("--eval-verifier-hybrid-alpha", type=float, default=0.7, help="Hybrid alpha for epoch-end eval verifier.")
    parser.add_argument(
        "--eval-attach-reference-diagnostics",
        action="store_true",
        help="Attach reference-conditioned offline verifier diagnostics during epoch-end rollout eval.",
    )
    parser.add_argument("--eval-progress-every", type=int, default=1, help="Log epoch-end rollout eval progress every N local items.")
    return parser.parse_args(argv)


def select_iteration_indices(dataset_size: int, rollout_count: int, start_index: int, iteration: int) -> List[int]:
    if dataset_size <= 0:
        return []
    offset = (int(start_index) + int(iteration) * int(rollout_count)) % int(dataset_size)
    return [(offset + i) % int(dataset_size) for i in range(int(rollout_count))]


def filter_reward_weighted_examples(examples: List[Dict[str, Any]], *, min_weight: float) -> List[Dict[str, Any]]:
    threshold = abs(float(min_weight))
    if threshold <= 0.0:
        return list(examples)
    filtered: List[Dict[str, Any]] = []
    for example in examples:
        value = float(example.get("advantage", example.get("sample_weight", 0.0)) or 0.0)
        if abs(value) >= threshold:
            filtered.append(example)
    return filtered


def resolve_reference_model_path(model_path: str | Path, reference_model_path: str | Path | None) -> str:
    reference_text = str(reference_model_path or "").strip()
    return reference_text or str(model_path)


def expand_grouped_rollout_specs(indices: List[int], num_generations: int) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for dataset_index in indices:
        group_id = f"idx{int(dataset_index):06d}"
        for generation_id in range(int(num_generations)):
            specs.append(
                {
                    "dataset_index": int(dataset_index),
                    "group_id": group_id,
                    "generation_id": int(generation_id),
                }
            )
    return specs


def compute_group_relative_advantages(
    scored_records: List[Dict[str, Any]],
    *,
    clip_value: Optional[float] = None,
    eps: float = 1e-6,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in scored_records:
        group_id = str(record.get("group_id") or record.get("video_id") or "ungrouped")
        grouped.setdefault(group_id, []).append(record)

    advantaged_records: List[Dict[str, Any]] = []
    for group_id, group_records in grouped.items():
        rewards = [float((record.get("reward_summary") or {}).get("total_reward") or 0.0) for record in group_records]
        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
        variance = sum((reward - mean_reward) ** 2 for reward in rewards) / len(rewards) if rewards else 0.0
        std_reward = math.sqrt(max(variance, 0.0))
        for record, reward in zip(group_records, rewards):
            advantage = 0.0 if std_reward < eps else (reward - mean_reward) / (std_reward + eps)
            if clip_value is not None:
                advantage = max(-float(clip_value), min(float(clip_value), advantage))
            updated = copy.deepcopy(record)
            updated["group_id"] = group_id
            updated["group_reward"] = reward
            updated["group_reward_mean"] = mean_reward
            updated["group_reward_std"] = std_reward
            updated["group_advantage"] = round(float(advantage), 6)
            advantaged_records.append(updated)
    return advantaged_records


def _load_jsonl(
    path: str | Path,
    *,
    include_splits: Optional[str | List[str]] = None,
) -> List[Dict[str, Any]]:
    allowed_splits = set(parse_include_splits(include_splits) or [])
    with Path(path).open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not allowed_splits:
        return rows
    return [row for row in rows if str(row.get("split") or "").strip() in allowed_splits]


def _build_config(args: argparse.Namespace) -> SaverAgentConfig:
    return SaverAgentConfig(
        preview=PreviewConfig(
            num_preview_frames=args.num_preview_frames,
            preview_sampling_fps=args.preview_sampling_fps,
            max_preview_frames=args.num_preview_frames,
        ),
        prompt=PromptConfig(),
        rollout_trace=RolloutTraceConfig(
            record_observation_content=True,
            record_state_deltas=True,
            record_message_history=True,
        ),
    )


def _build_policy(model_path: str | Path, args: argparse.Namespace, *, runtime: Any) -> QwenGenerationPolicy:
    return QwenGenerationPolicy.from_pretrained(
        model_path,
        torch_dtype=args.torch_dtype,
        device_map=resolve_inference_device_map(args.policy_device_map, runtime=runtime),
        attn_implementation=args.attn_implementation or None,
        max_new_tokens=args.policy_max_new_tokens,
        do_sample=args.policy_do_sample,
        temperature=args.policy_temperature,
        top_p=args.policy_top_p,
        top_k=args.policy_top_k,
        repetition_penalty=args.policy_repetition_penalty,
    )


def _build_verifier_runtime(args: argparse.Namespace, *, runtime: Any) -> Optional[QwenSelfVerifier]:
    if args.verifier_backend not in {"qwen_self_verifier", "hybrid"}:
        return None
    return QwenSelfVerifier.from_pretrained(
        args.verifier_model_path,
        torch_dtype=args.verifier_torch_dtype,
        device_map=resolve_inference_device_map(args.verifier_device_map, runtime=runtime),
        attn_implementation=args.verifier_attn_implementation or None,
        max_new_tokens=args.verifier_max_new_tokens,
    )


def _attach_verifier_context(
    item: Dict[str, Any],
    args: argparse.Namespace,
    verifier_runtime: Any,
    *,
    verifier_device_map: Any,
) -> None:
    cache = item["multimodal_cache"]
    cache["verifier_backend"] = args.verifier_backend
    cache["verifier_model_path"] = args.verifier_model_path
    cache["verifier_torch_dtype"] = args.verifier_torch_dtype
    cache["verifier_device_map"] = verifier_device_map
    cache["verifier_attn_implementation"] = args.verifier_attn_implementation
    cache["verifier_max_new_tokens"] = args.verifier_max_new_tokens
    cache["verifier_hybrid_alpha"] = args.verifier_hybrid_alpha
    if verifier_runtime is not None:
        cache["verifier_runtime"] = verifier_runtime


def build_training_kwargs(
    *,
    current_model_path: str | Path,
    checkpoint_dir: str | Path,
    args: argparse.Namespace,
    reference_model_path: str | Path,
    config: SaverAgentConfig,
) -> Dict[str, Any]:
    lora_target_modules = [module.strip() for module in args.lora_target_modules.split(",") if module.strip()]
    rollout_eval_config = None
    if args.eval_data:
        rollout_eval_config = RolloutEvaluationConfig(
            data_path=args.eval_data,
            data_root=args.eval_data_root or args.data_root,
            include_splits=parse_include_splits(args.eval_include_splits),
            max_records=args.eval_max_records,
            rollout_max_turns=args.eval_rollout_max_turns,
            verifier_backend=args.eval_verifier_backend,
            verifier_model_path=args.eval_verifier_model_path or args.verifier_model_path or current_model_path,
            verifier_torch_dtype=args.eval_verifier_torch_dtype,
            verifier_device_map=args.eval_verifier_device_map,
            verifier_attn_implementation=args.eval_verifier_attn_implementation,
            verifier_max_new_tokens=args.eval_verifier_max_new_tokens,
            verifier_hybrid_alpha=args.eval_verifier_hybrid_alpha,
            attach_reference_diagnostics=args.eval_attach_reference_diagnostics,
            progress_every=args.eval_progress_every,
            saver_config=config,
        )
    return {
        "model_path": str(current_model_path),
        "output_dir": checkpoint_dir,
        "training_objective": args.training_objective,
        "old_policy_model_path": str(current_model_path) if args.training_objective == "grpo" else "",
        "ppo_clip_epsilon": float(args.ppo_clip_epsilon),
        "reference_model_path": str(reference_model_path),
        "kl_beta": float(args.kl_beta),
        "torch_dtype": args.torch_dtype,
        "attn_implementation": args.attn_implementation or None,
        "gradient_checkpointing": args.gradient_checkpointing,
        "use_lora": args.lora,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": lora_target_modules or None,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "dataloader_num_workers": args.dataloader_num_workers,
        "dataloader_prefetch_factor": args.dataloader_prefetch_factor,
        "dataloader_persistent_workers": args.dataloader_persistent_workers,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "max_image_side": args.max_image_side,
        "max_image_pixels": args.max_image_pixels,
        "keep_recent_tool_image_messages": args.keep_recent_tool_image_messages,
        "max_total_images": args.max_total_images,
        "max_seq_length": args.max_seq_length,
        "keep_recent_text_messages": args.keep_recent_text_messages,
        "rollout_eval_config": rollout_eval_config,
    }


def collect_rollouts(
    *,
    data_path: str | Path,
    data_root: str | Path,
    rollout_specs: List[Dict[str, Any]],
    model_path: str | Path,
    args: argparse.Namespace,
    runtime: Any,
    verifier_runtime: Any = None,
) -> List[Dict[str, Any]]:
    if not rollout_specs:
        return []
    config = _build_config(args)
    include_splits = parse_include_splits(args.include_splits)
    dataset = SaverAgentDataset(data_path, data_root=data_root, config=config, include_splits=include_splits)
    if hasattr(dataset, "format_frame_cache_status"):
        runtime_log(
            dataset.format_frame_cache_status(prefix="RL rollout frame cache"),
            runtime=runtime,
            main_process_only=True,
        )
    runner = SaverRolloutRunner(
        adapter=TimeSearchRolloutAdapter(config=config),
        max_turns=args.rollout_max_turns,
        config=config,
    )
    policy = _build_policy(model_path, args, runtime=runtime)
    effective_verifier_device_map = resolve_inference_device_map(args.verifier_device_map, runtime=runtime)
    rollouts: List[Dict[str, Any]] = []
    total_rollouts = len(rollout_specs)
    for completed, spec in enumerate(rollout_specs, start=1):
        dataset_index = int(spec["dataset_index"])
        item = dataset[dataset_index]
        _attach_verifier_context(
            item,
            args,
            verifier_runtime,
            verifier_device_map=effective_verifier_device_map,
        )
        result = runner.run_episode(item, policy)
        serialized = _serialize_result(result)
        serialized["dataset_index"] = dataset_index
        serialized["group_id"] = spec.get("group_id")
        serialized["generation_id"] = spec.get("generation_id")
        rollouts.append(serialized)
        if should_log_progress(completed, total_rollouts, args.progress_every):
            runtime_log(
                f"rollout progress: {completed}/{total_rollouts} dataset_index={dataset_index} "
                f"video_id={serialized.get('video_id', '')}",
                runtime=runtime,
            )
    return rollouts


def build_reward_examples_from_scored_rollouts(
    *,
    data_path: str | Path,
    data_root: str | Path,
    scored_records: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    config = _build_config(args)
    include_splits = parse_include_splits(args.include_splits)
    dataset = SaverAgentDataset(data_path, data_root=data_root, config=config, include_splits=include_splits)
    raw_records = list(dataset.records)
    index_by_video_id = {record.get("video_id"): idx for idx, record in enumerate(raw_records) if record.get("video_id")}
    examples: List[Dict[str, Any]] = []
    for record in scored_records:
        video_id = record.get("video_id")
        if video_id not in index_by_video_id:
            continue
        item = dataset[index_by_video_id[video_id]]
        rollout_examples = build_reward_weighted_examples(
            item,
            record,
            config=config,
            turn_advantage_gamma=args.turn_advantage_gamma,
            turn_advantage_alpha=args.turn_advantage_alpha,
            turn_search_bonus=args.turn_search_bonus,
            turn_evidence_bonus=args.turn_evidence_bonus,
            turn_finalize_bonus=args.turn_finalize_bonus,
            turn_invalid_penalty=args.turn_invalid_penalty,
        )
        for example in rollout_examples:
            turn_advantage = float(example.get("advantage", record.get("group_advantage", 0.0)) or 0.0)
            example["sample_weight"] = max(turn_advantage, 0.0)
            example["advantage"] = turn_advantage
            example["group_id"] = record.get("group_id")
            example["generation_id"] = record.get("generation_id")
        examples.extend(rollout_examples)
    examples = filter_reward_weighted_examples(examples, min_weight=args.min_weight)
    if args.max_train_examples > 0:
        examples = examples[: args.max_train_examples]
    return examples


def _resolve_cea_group_settings(args: argparse.Namespace) -> tuple[bool, bool]:
    if str(args.grpo_variant) != "cea_grpo":
        return False, False
    if (
        bool(args.cea_enable_alert_group)
        or bool(args.cea_enable_evidence_group)
        or bool(getattr(args, "cea_enable_search_group", False))
    ):
        return bool(args.cea_enable_alert_group), bool(args.cea_enable_evidence_group)
    return True, True


def _resolve_cea_search_group_enabled(args: argparse.Namespace) -> bool:
    if str(args.grpo_variant) != "cea_grpo":
        return False
    if (
        bool(args.cea_enable_alert_group)
        or bool(args.cea_enable_evidence_group)
        or bool(getattr(args, "cea_enable_search_group", False))
    ):
        return bool(getattr(args, "cea_enable_search_group", False))
    return True


def build_cea_grpo_examples_from_scored_rollouts(
    *,
    data_path: str | Path,
    data_root: str | Path,
    scored_records: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    config = _build_config(args)
    include_splits = parse_include_splits(args.include_splits)
    dataset = SaverAgentDataset(data_path, data_root=data_root, config=config, include_splits=include_splits)
    raw_records = list(dataset.records)
    index_by_video_id = {record.get("video_id"): idx for idx, record in enumerate(raw_records) if record.get("video_id")}
    enable_alert_group, enable_evidence_group = _resolve_cea_group_settings(args)
    enable_search_group = _resolve_cea_search_group_enabled(args)
    examples: List[Dict[str, Any]] = []
    for record in scored_records:
        video_id = record.get("video_id")
        if video_id not in index_by_video_id:
            continue
        item = dataset[index_by_video_id[video_id]]
        rollout_examples = build_counterfactual_grpo_examples(
            item,
            record,
            config=config,
            turn_advantage_gamma=args.turn_advantage_gamma,
            turn_advantage_alpha=args.turn_advantage_alpha,
            turn_search_bonus=args.turn_search_bonus,
            turn_evidence_bonus=args.turn_evidence_bonus,
            turn_finalize_bonus=args.turn_finalize_bonus,
            turn_invalid_penalty=args.turn_invalid_penalty,
            local_verifier_backend=args.cea_local_verifier_backend,
            local_use_reference_supervision=args.cea_local_use_reference_supervision,
            search_local_alpha=args.cea_search_local_alpha,
            alert_local_alpha=args.cea_alert_local_alpha,
            evidence_local_alpha=args.cea_evidence_local_alpha,
            max_search_anchors=args.cea_max_search_anchors_per_rollout,
            max_alert_anchors=args.cea_max_alert_anchors_per_rollout,
            max_evidence_anchors=args.cea_max_evidence_anchors_per_rollout,
            enable_search_group=enable_search_group,
            enable_alert_group=enable_alert_group,
            enable_evidence_group=enable_evidence_group,
        )
        for example in rollout_examples:
            example["sample_weight"] = max(float(example.get("advantage", 0.0) or 0.0), 0.0)
            example["group_id"] = record.get("group_id")
            example["generation_id"] = record.get("generation_id")
        examples.extend(rollout_examples)
    examples = filter_reward_weighted_examples(examples, min_weight=args.min_weight)
    if args.max_train_examples > 0:
        examples = examples[: args.max_train_examples]
    return examples


def build_training_examples_from_scored_rollouts(
    *,
    data_path: str | Path,
    data_root: str | Path,
    scored_records: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    if str(args.grpo_variant) == "cea_grpo":
        return build_cea_grpo_examples_from_scored_rollouts(
            data_path=data_path,
            data_root=data_root,
            scored_records=scored_records,
            args=args,
        )
    return build_reward_examples_from_scored_rollouts(
        data_path=data_path,
        data_root=data_root,
        scored_records=scored_records,
        args=args,
    )


def main() -> None:
    args = parse_args()
    runtime = distributed_runtime_from_env()
    init_torch_distributed(runtime)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_data = ReferenceDataProvider(data_path=args.data, data_root=args.data_root)
    raw_records = _load_jsonl(args.data, include_splits=args.include_splits)
    current_model_path = str(args.model_path)
    latest_checkpoint = current_model_path
    reference_model_path = resolve_reference_model_path(args.model_path, args.reference_model_path)
    shard_spec = resolve_shard_spec(runtime=runtime)

    runtime_log(
        "RL startup: "
        f"num_iterations={args.num_iterations} rollout_count={args.rollout_count} "
        f"num_generations={args.num_generations} world_size={runtime.world_size} "
        f"model_path={current_model_path} include_splits={parse_include_splits(args.include_splits) or 'all'}",
        runtime=runtime,
        main_process_only=True,
    )

    for iteration in range(int(args.num_iterations)):
        iter_dir = output_dir / f"iter_{iteration:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        rollout_shard_dir = iter_dir / "rollout_shards"
        scored_shard_dir = iter_dir / "scored_shards"
        rollout_shard_dir.mkdir(parents=True, exist_ok=True)
        scored_shard_dir.mkdir(parents=True, exist_ok=True)
        indices = select_iteration_indices(len(raw_records), args.rollout_count, args.rollout_start_index, iteration)
        rollout_specs = expand_grouped_rollout_specs(indices, args.num_generations)
        local_rollout_specs = shard_sequence(
            rollout_specs,
            num_shards=shard_spec.num_shards,
            shard_index=shard_spec.shard_index,
        )
        runtime_log(
            f"iteration {iteration}: total_specs={len(rollout_specs)} local_specs={len(local_rollout_specs)}",
            runtime=runtime,
            main_process_only=True,
        )

        verifier_runtime = None
        if args.verifier_backend in {"qwen_self_verifier", "hybrid"} and local_rollout_specs:
            runtime_log(
                f"loading verifier model from {args.verifier_model_path} for rollout/score",
                runtime=runtime,
            )
            verifier_runtime = _build_verifier_runtime(args, runtime=runtime)

        local_rollouts = collect_rollouts(
            data_path=args.data,
            data_root=args.data_root,
            rollout_specs=local_rollout_specs,
            model_path=current_model_path,
            args=args,
            runtime=runtime,
            verifier_runtime=verifier_runtime,
        )
        local_rollout_path = rollout_shard_dir / f"part.rank{runtime.rank:02d}-of-{runtime.world_size:02d}.jsonl"
        save_rollout_records(local_rollouts, local_rollout_path, metadata={"input_kind": "jsonl"})
        distributed_barrier(runtime)

        raw_rollout_path = iter_dir / "rollouts.jsonl"
        if runtime.is_main_process:
            merged_rollouts, _ = load_rollout_records(rollout_shard_dir)
            save_rollout_records(merged_rollouts, raw_rollout_path, metadata={"input_kind": "jsonl"})
            runtime_log(
                f"iteration {iteration}: merged {len(merged_rollouts)} rollouts to {raw_rollout_path}",
                runtime=runtime,
                main_process_only=True,
            )
        distributed_barrier(runtime)

        verifier_kwargs = {
            "verifier_backend": args.verifier_backend,
            "verifier_model_path": args.verifier_model_path,
            "verifier_torch_dtype": args.verifier_torch_dtype,
            "verifier_device_map": resolve_inference_device_map(args.verifier_device_map, runtime=runtime),
            "verifier_attn_implementation": args.verifier_attn_implementation,
            "verifier_max_new_tokens": args.verifier_max_new_tokens,
            "verifier_hybrid_alpha": args.verifier_hybrid_alpha,
        }
        if verifier_runtime is not None:
            verifier_kwargs["verifier_runtime"] = verifier_runtime
        local_scored_records = score_rollout_records(
            local_rollouts,
            reference_data=reference_data,
            verifier_backend=args.verifier_backend,
            force_reverify=True,
            verifier_kwargs=verifier_kwargs,
            progress_every=args.progress_every,
            progress_label=f"iteration {iteration} score progress",
            runtime=runtime,
        )
        local_scored_path = scored_shard_dir / f"part.rank{runtime.rank:02d}-of-{runtime.world_size:02d}.jsonl"
        save_rollout_records(local_scored_records, local_scored_path, metadata={"input_kind": "jsonl"})
        distributed_barrier(runtime)

        scored_path = iter_dir / "rollouts.scored.jsonl"
        scored_records: List[Dict[str, Any]] = []
        if runtime.is_main_process:
            merged_scored_records, _ = load_rollout_records(scored_shard_dir)
            scored_records = compute_group_relative_advantages(merged_scored_records, clip_value=args.advantage_clip)
            save_rollout_records(scored_records, scored_path, metadata={"input_kind": "jsonl"})
            runtime_log(
                f"iteration {iteration}: merged and rescored {len(scored_records)} rollouts to {scored_path}",
                runtime=runtime,
                main_process_only=True,
            )
        distributed_barrier(runtime)

        examples_path = iter_dir / "training_examples.json"
        reward_examples: List[Dict[str, Any]] = []
        summary = {
            "iteration": iteration,
            "current_model_path": current_model_path,
            "reference_model_path": reference_model_path,
            "kl_beta": float(args.kl_beta),
            "grpo_variant": args.grpo_variant,
            "num_rollouts": len(rollout_specs),
            "num_groups": len(indices),
            "num_generations": int(args.num_generations),
            "num_training_examples": 0,
            "num_reward_examples": 0,
            "indices": indices,
            "world_size": runtime.world_size,
        }
        if runtime.is_main_process:
            scored_records, _ = load_rollout_records(scored_path)
            reward_examples = build_training_examples_from_scored_rollouts(
                data_path=args.data,
                data_root=args.data_root,
                scored_records=scored_records,
                args=args,
            )
            examples_path.write_text(json.dumps(reward_examples, ensure_ascii=False, indent=2), encoding="utf-8")
            summary["num_training_examples"] = len(reward_examples)
            summary["num_reward_examples"] = len(reward_examples)
        distributed_barrier(runtime)

        if examples_path.exists():
            reward_examples = json.loads(examples_path.read_text(encoding="utf-8"))
        summary["num_training_examples"] = len(reward_examples)
        summary["num_reward_examples"] = len(reward_examples)

        if args.dry_run or not reward_examples:
            if runtime.is_main_process:
                (iter_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            distributed_barrier(runtime)
            continue

        checkpoint_dir = iter_dir / "checkpoint"
        runtime_log(
            f"iteration {iteration}: starting distributed update with {len(reward_examples)} reward examples",
            runtime=runtime,
            main_process_only=True,
        )
        train_result = run_weighted_sft(
            reward_examples,
            **build_training_kwargs(
                current_model_path=current_model_path,
                checkpoint_dir=checkpoint_dir,
                args=args,
                reference_model_path=reference_model_path,
                config=_build_config(args),
            ),
        )
        latest_checkpoint = str(checkpoint_dir)
        current_model_path = latest_checkpoint
        if runtime.is_main_process:
            (iter_dir / "summary.json").write_text(
                json.dumps({**summary, **train_result}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        distributed_barrier(runtime)

    if runtime.is_main_process:
        (output_dir / "latest_checkpoint.txt").write_text(str(latest_checkpoint), encoding="utf-8")
        print(json.dumps({"latest_checkpoint": str(latest_checkpoint)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
