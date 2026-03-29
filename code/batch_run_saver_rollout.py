#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from split_utils import parse_include_splits

from run_saver_rollout import _serialize_result
from saver_agent.adapter import TimeSearchRolloutAdapter
from saver_agent.config import PromptConfig, PreviewConfig, RolloutTraceConfig, SaverAgentConfig
from saver_agent.dataset import SaverAgentDataset
from saver_agent.proposal import SiglipFeatureEncoder
from saver_agent.qwen_policy import DEFAULT_MODEL_PATH, QwenGenerationPolicy
from saver_agent.qwen_verifier import DEFAULT_VERIFIER_MODEL_PATH, QwenSelfVerifier
from saver_agent.runtime import (
    distributed_runtime_from_env,
    resolve_inference_device_map,
    resolve_shard_spec,
    runtime_log,
    shard_sequence,
    sharded_output_path,
    should_log_progress,
)
from saver_agent.rollout import ReplayPolicy, SaverRolloutRunner


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-run SAVER rollouts over a dataset slice.")
    parser.add_argument("--data", required=True, help="Path to saver_agent JSONL data.")
    parser.add_argument("--data-root", default="", help="Root path used to resolve relative video paths.")
    parser.add_argument("--include-splits", default="", help="Optional comma-separated split whitelist for --data.")
    parser.add_argument(
        "--indices",
        default="",
        help="Optional explicit dataset indices, e.g. '0,1,5-7'. Overrides --start-index/--count.",
    )
    parser.add_argument("--start-index", type=int, default=0, help="Start dataset index for batch rollout.")
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="Number of samples to roll out from start-index. Use 0 to run until the end of the filtered dataset.",
    )
    parser.add_argument("--max-turns", type=int, default=12, help="Maximum rollout turns per sample.")
    parser.add_argument(
        "--policy-backend",
        choices=["replay", "qwen"],
        default="replay",
        help="Use replayed responses or real Qwen generation.",
    )
    parser.add_argument(
        "--response",
        action="append",
        default=[],
        help="Replayed model response for one turn. Reused for every sample when policy-backend=replay.",
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Local Qwen model path.")
    parser.add_argument("--proposal-model-path", default="", help="Optional local SigLIP/CLIP path for feature-guided proposal.")
    parser.add_argument("--proposal-torch-dtype", default="auto", help="Torch dtype for the proposal encoder.")
    parser.add_argument("--proposal-device", default="", help="Optional device for the proposal encoder. Defaults to cpu or current local cuda device.")
    parser.add_argument("--torch-dtype", default="auto", help="Torch dtype passed to from_pretrained.")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map argument.")
    parser.add_argument("--attn-implementation", default="", help="Optional attention backend.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Generation length for Qwen policy.")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling for Qwen policy.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=None, help="Sampling top-p.")
    parser.add_argument("--top-k", type=int, default=None, help="Sampling top-k.")
    parser.add_argument("--repetition-penalty", type=float, default=None, help="Optional repetition penalty.")
    parser.add_argument(
        "--num-preview-frames",
        type=int,
        default=8,
        help="Maximum preview frames injected into the first user turn.",
    )
    parser.add_argument(
        "--preview-sampling-fps",
        type=float,
        default=None,
        help="Target preview sampling fps before capping by preview frame count.",
    )
    parser.add_argument("--initial-user-template", default="", help="Optional custom template for the first user prompt.")
    parser.add_argument("--preview-instruction", default="", help="Optional custom preview instruction.")
    parser.add_argument("--tool-response-template", default="", help="Optional custom tool follow-up prompt template.")
    parser.add_argument("--record-observation-content", action="store_true", help="Store full tool observation content.")
    parser.add_argument(
        "--no-record-message-history",
        action="store_true",
        help="Disable storing the full message history in the rollout output.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Batch rollout output path (.jsonl, .json, or directory).",
    )
    parser.add_argument("--num-shards", type=int, default=0, help="Optional number of shard workers.")
    parser.add_argument("--shard-index", type=int, default=-1, help="Optional shard index for this process.")
    parser.add_argument("--progress-every", type=int, default=1, help="Log rollout progress every N local samples.")
    parser.add_argument(
        "--verifier-backend",
        choices=["heuristic", "qwen_self_verifier", "hybrid"],
        default="heuristic",
        help="Verifier backend used by verify_hypothesis calls that do not explicitly specify one.",
    )
    parser.add_argument(
        "--verifier-model-path",
        default=DEFAULT_VERIFIER_MODEL_PATH,
        help="Local Qwen verifier model path.",
    )
    parser.add_argument("--verifier-torch-dtype", default="auto", help="Torch dtype for Qwen verifier.")
    parser.add_argument("--verifier-device-map", default="auto", help="device_map for Qwen verifier.")
    parser.add_argument(
        "--verifier-attn-implementation",
        default="",
        help="Optional attention backend for Qwen verifier.",
    )
    parser.add_argument(
        "--verifier-max-new-tokens",
        type=int,
        default=512,
        help="Generation length for Qwen self-verifier.",
    )
    parser.add_argument(
        "--verifier-hybrid-alpha",
        type=float,
        default=0.7,
        help="Hybrid verifier weight on heuristic scores.",
    )
    return parser.parse_args(argv)


def _build_config(args: argparse.Namespace) -> SaverAgentConfig:
    return SaverAgentConfig(
        preview=PreviewConfig(
            num_preview_frames=args.num_preview_frames,
            preview_sampling_fps=args.preview_sampling_fps,
            max_preview_frames=args.num_preview_frames,
        ),
        prompt=PromptConfig(
            initial_user_template=args.initial_user_template or PromptConfig().initial_user_template,
            preview_instruction=args.preview_instruction or PromptConfig().preview_instruction,
            tool_response_template=args.tool_response_template or PromptConfig().tool_response_template,
        ),
        rollout_trace=RolloutTraceConfig(
            record_observation_content=args.record_observation_content,
            record_state_deltas=True,
            record_message_history=not args.no_record_message_history,
        ),
    )


def _parse_indices(indices_text: str) -> List[int]:
    values: List[int] = []
    if not indices_text.strip():
        return values
    for chunk in indices_text.split(","):
        token = chunk.strip()
        if not token:
            continue
        range_match = re.fullmatch(r"(-?\d+)\s*-\s*(-?\d+)", token)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            step = 1 if end >= start else -1
            values.extend(list(range(start, end + step, step)))
            continue
        values.append(int(token))
    return values


def _resolve_dataset_indices(args: argparse.Namespace, dataset_size: int) -> List[int]:
    if args.indices:
        indices = _parse_indices(args.indices)
    else:
        if args.start_index < 0 or args.start_index >= dataset_size:
            raise SystemExit(
                f"Dataset index out of range: {args.start_index}. Valid range is [0, {max(dataset_size - 1, 0)}]."
            )
        if args.count < 0:
            raise SystemExit("Provide either --indices or a non-negative --count for batch rollout.")
        if args.count == 0:
            indices = list(range(args.start_index, dataset_size))
        else:
            indices = list(range(args.start_index, args.start_index + args.count))

    if not indices:
        raise SystemExit("No dataset indices were resolved for batch rollout.")
    invalid = [index for index in indices if index < 0 or index >= dataset_size]
    if invalid:
        raise SystemExit(
            f"Dataset index out of range: {invalid[0]}. Valid range is [0, {max(dataset_size - 1, 0)}]."
        )
    return indices


def _attach_verifier_runtime(
    item: Dict[str, Any],
    args: argparse.Namespace,
    verifier_runtime: Any,
    *,
    verifier_device_map: Any,
) -> None:
    item["multimodal_cache"]["verifier_backend"] = args.verifier_backend
    item["multimodal_cache"]["verifier_model_path"] = args.verifier_model_path
    item["multimodal_cache"]["verifier_torch_dtype"] = args.verifier_torch_dtype
    item["multimodal_cache"]["verifier_device_map"] = verifier_device_map
    item["multimodal_cache"]["verifier_attn_implementation"] = args.verifier_attn_implementation
    item["multimodal_cache"]["verifier_max_new_tokens"] = args.verifier_max_new_tokens
    item["multimodal_cache"]["verifier_hybrid_alpha"] = args.verifier_hybrid_alpha
    if verifier_runtime is not None:
        item["multimodal_cache"]["verifier_runtime"] = verifier_runtime


def _attach_proposal_runtime(
    item: Dict[str, Any],
    proposal_runtime: Any,
) -> None:
    if proposal_runtime is not None:
        item["multimodal_cache"]["proposal_runtime"] = proposal_runtime


def _build_qwen_policy(args: argparse.Namespace, *, runtime: Any) -> QwenGenerationPolicy:
    return QwenGenerationPolicy.from_pretrained(
        args.model_path,
        torch_dtype=args.torch_dtype,
        device_map=resolve_inference_device_map(args.device_map, runtime=runtime),
        attn_implementation=args.attn_implementation or None,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )


def _build_qwen_verifier(args: argparse.Namespace, *, runtime: Any) -> QwenSelfVerifier | None:
    if args.verifier_backend not in {"qwen_self_verifier", "hybrid"}:
        return None
    return QwenSelfVerifier.from_pretrained(
        args.verifier_model_path,
        torch_dtype=args.verifier_torch_dtype,
        device_map=resolve_inference_device_map(args.verifier_device_map, runtime=runtime),
        attn_implementation=args.verifier_attn_implementation or None,
        max_new_tokens=args.verifier_max_new_tokens,
    )


def _build_proposal_runtime(args: argparse.Namespace, *, runtime: Any) -> SiglipFeatureEncoder | None:
    if not args.proposal_model_path:
        return None
    if args.proposal_device:
        device = args.proposal_device
    else:
        try:
            import torch
        except Exception:
            device = "cpu"
        else:
            device = f"cuda:{int(runtime.local_rank)}" if torch.cuda.is_available() else "cpu"
    return SiglipFeatureEncoder.from_pretrained(
        args.proposal_model_path,
        torch_dtype=args.proposal_torch_dtype,
        device=device,
    )


def _save_batch_results(records: List[Dict[str, Any]], output_path: Path) -> None:
    if output_path.suffix in {".jsonl", ".json"}:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix == ".jsonl":
            with output_path.open("w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            return
        output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    output_path.mkdir(parents=True, exist_ok=True)
    for record in records:
        safe_video_id = str(record.get("video_id") or "sample").replace("/", "_")
        dataset_index = int(record.get("dataset_index", 0))
        file_path = output_path / f"{dataset_index:06d}_{safe_video_id}.json"
        file_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.policy_backend == "replay" and not args.response:
        raise SystemExit("At least one --response is required for replay rollout.")

    runtime = distributed_runtime_from_env()
    shard_spec = resolve_shard_spec(num_shards=args.num_shards, shard_index=args.shard_index, runtime=runtime)
    config = _build_config(args)
    dataset = SaverAgentDataset(
        args.data,
        data_root=args.data_root,
        config=config,
        include_splits=parse_include_splits(args.include_splits),
    )
    if hasattr(dataset, "format_frame_cache_status"):
        runtime_log(
            dataset.format_frame_cache_status(prefix="rollout frame cache"),
            runtime=runtime,
            main_process_only=True,
        )
    indices = _resolve_dataset_indices(args, len(dataset))
    local_indices = shard_sequence(indices, num_shards=shard_spec.num_shards, shard_index=shard_spec.shard_index)
    output_path = sharded_output_path(args.output, num_shards=shard_spec.num_shards, shard_index=shard_spec.shard_index)
    effective_policy_device_map = resolve_inference_device_map(args.device_map, runtime=runtime)
    effective_verifier_device_map = resolve_inference_device_map(args.verifier_device_map, runtime=runtime)

    runtime_log(
        "batch rollout startup: "
        f"dataset_size={len(dataset)} total_indices={len(indices)} local_indices={len(local_indices)} "
        f"policy_backend={args.policy_backend} include_splits={parse_include_splits(args.include_splits) or 'all'} "
        f"output={output_path}",
        runtime=runtime,
    )

    verifier_runtime = None
    if args.verifier_backend in {"qwen_self_verifier", "hybrid"} and local_indices:
        runtime_log(
            f"loading verifier model from {args.verifier_model_path} with device_map={effective_verifier_device_map}",
            runtime=runtime,
        )
        verifier_runtime = _build_qwen_verifier(args, runtime=runtime)
    proposal_runtime = None
    if args.proposal_model_path and local_indices:
        runtime_log(
            f"loading proposal model from {args.proposal_model_path}",
            runtime=runtime,
        )
        proposal_runtime = _build_proposal_runtime(args, runtime=runtime)
    qwen_policy = None
    if args.policy_backend == "qwen" and local_indices:
        runtime_log(
            f"loading policy model from {args.model_path} with device_map={effective_policy_device_map}",
            runtime=runtime,
        )
        qwen_policy = _build_qwen_policy(args, runtime=runtime)
    runner = SaverRolloutRunner(
        adapter=TimeSearchRolloutAdapter(config=config),
        max_turns=args.max_turns,
        config=config,
    )

    results: List[Dict[str, Any]] = []
    total_local = len(local_indices)
    for completed, dataset_index in enumerate(local_indices, start=1):
        item = dataset[dataset_index]
        _attach_verifier_runtime(
            item,
            args,
            verifier_runtime,
            verifier_device_map=effective_verifier_device_map,
        )
        _attach_proposal_runtime(item, proposal_runtime)
        policy = qwen_policy if qwen_policy is not None else ReplayPolicy(args.response)
        result = runner.run_episode(item, policy)
        serialized = _serialize_result(result)
        serialized["dataset_index"] = dataset_index
        results.append(serialized)
        if should_log_progress(completed, total_local, args.progress_every):
            runtime_log(
                f"rollout progress: {completed}/{total_local} dataset_index={dataset_index} "
                f"video_id={serialized.get('video_id', '')}",
                runtime=runtime,
            )

    _save_batch_results(results, output_path)
    runtime_log(f"saved {len(results)} rollout records to {output_path}", runtime=runtime)


if __name__ == "__main__":
    main()
