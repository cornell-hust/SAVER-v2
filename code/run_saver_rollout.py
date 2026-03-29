#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from split_utils import parse_include_splits

from saver_agent.config import PromptConfig, PreviewConfig, RolloutTraceConfig, SaverAgentConfig
from saver_agent.adapter import TimeSearchRolloutAdapter
from saver_agent.dataset import SaverAgentDataset
from saver_agent.proposal import SiglipFeatureEncoder
from saver_agent.qwen_policy import DEFAULT_MODEL_PATH, QwenGenerationPolicy
from saver_agent.qwen_verifier import DEFAULT_VERIFIER_MODEL_PATH, QwenSelfVerifier
from saver_agent.rollout import ReplayPolicy, SaverRolloutRunner


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal SAVER rollout with replayed responses.")
    parser.add_argument("--data", required=True, help="Path to saver_agent JSONL data.")
    parser.add_argument("--data-root", default="", help="Root path used to resolve relative video paths.")
    parser.add_argument("--include-splits", default="", help="Optional comma-separated split whitelist for --data.")
    parser.add_argument("--index", type=int, default=0, help="Dataset sample index.")
    parser.add_argument("--max-turns", type=int, default=12, help="Maximum rollout turns.")
    parser.add_argument(
        "--policy-backend",
        choices=["replay", "qwen"],
        default="replay",
        help="Use replayed responses or real Qwen generation.",
    )
    parser.add_argument("--response", action="append", default=[], help="Replayed model response for one turn.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Local Qwen model path.")
    parser.add_argument("--proposal-model-path", default="", help="Optional local SigLIP/CLIP path for feature-guided proposal.")
    parser.add_argument("--proposal-torch-dtype", default="auto", help="Torch dtype for the proposal encoder.")
    parser.add_argument("--proposal-device", default="cpu", help="Device for the proposal encoder, e.g. cpu or cuda:0.")
    parser.add_argument("--torch-dtype", default="auto", help="Torch dtype passed to from_pretrained.")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map argument.")
    parser.add_argument("--attn-implementation", default="", help="Optional attention backend, e.g. flash_attention_2.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Generation length for Qwen policy.")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling for Qwen policy.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=None, help="Sampling top-p.")
    parser.add_argument("--top-k", type=int, default=None, help="Sampling top-k.")
    parser.add_argument("--repetition-penalty", type=float, default=None, help="Optional repetition penalty.")
    parser.add_argument("--num-preview-frames", type=int, default=8, help="Maximum preview frames injected into the first user turn.")
    parser.add_argument("--preview-sampling-fps", type=float, default=None, help="Target preview sampling fps before capping by preview frame count.")
    parser.add_argument("--initial-user-template", default="", help="Optional custom template for the first user prompt.")
    parser.add_argument("--preview-instruction", default="", help="Optional custom preview instruction.")
    parser.add_argument("--tool-response-template", default="", help="Optional custom tool follow-up prompt template.")
    parser.add_argument("--record-observation-content", action="store_true", help="Store full tool observation content in rollout traces.")
    parser.add_argument(
        "--no-record-message-history",
        action="store_true",
        help="Disable storing the full message history in the rollout output.",
    )
    parser.add_argument("--output", default="", help="Optional JSON output path.")
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


def _serialize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    return _to_jsonable(result)


def _serialize_message(message: Dict[str, Any]) -> Dict[str, Any]:
    content: List[Dict[str, Any]] = []
    for item in message.get("content", []):
        if item.get("type") == "image":
            image = item.get("image")
            shape = list(image.shape) if hasattr(image, "shape") else None
            content.append({"type": "image", "shape": shape})
        else:
            content.append({"type": item.get("type"), "text": item.get("text")})
    return {
        "role": message.get("role"),
        "name": message.get("name"),
        "content": content,
    }


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        if {"role", "content"}.issubset(value.keys()):
            return _serialize_message(value)
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return {"type": "tensor", "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.ndarray):
        return {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
    return value


def main() -> None:
    args = parse_args()
    if args.policy_backend == "replay" and not args.response:
        raise SystemExit("At least one --response is required for replay rollout.")

    config = SaverAgentConfig(
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

    dataset = SaverAgentDataset(
        args.data,
        data_root=args.data_root,
        config=config,
        include_splits=parse_include_splits(args.include_splits),
    )
    item = dataset[args.index]
    item["multimodal_cache"]["verifier_backend"] = args.verifier_backend
    item["multimodal_cache"]["verifier_model_path"] = args.verifier_model_path
    item["multimodal_cache"]["verifier_torch_dtype"] = args.verifier_torch_dtype
    item["multimodal_cache"]["verifier_device_map"] = args.verifier_device_map
    item["multimodal_cache"]["verifier_attn_implementation"] = args.verifier_attn_implementation
    item["multimodal_cache"]["verifier_max_new_tokens"] = args.verifier_max_new_tokens
    item["multimodal_cache"]["verifier_hybrid_alpha"] = args.verifier_hybrid_alpha
    if args.verifier_backend in {"qwen_self_verifier", "hybrid"}:
        verifier_runtime = QwenSelfVerifier.from_pretrained(
            args.verifier_model_path,
            torch_dtype=args.verifier_torch_dtype,
            device_map=args.verifier_device_map,
            attn_implementation=args.verifier_attn_implementation or None,
            max_new_tokens=args.verifier_max_new_tokens,
        )
        item["multimodal_cache"]["verifier_runtime"] = verifier_runtime
    if args.proposal_model_path:
        item["multimodal_cache"]["proposal_runtime"] = SiglipFeatureEncoder.from_pretrained(
            args.proposal_model_path,
            torch_dtype=args.proposal_torch_dtype,
            device=args.proposal_device,
        )
    runner = SaverRolloutRunner(
        adapter=TimeSearchRolloutAdapter(config=config),
        max_turns=args.max_turns,
        config=config,
    )
    if args.policy_backend == "qwen":
        policy = QwenGenerationPolicy.from_pretrained(
            args.model_path,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
            attn_implementation=args.attn_implementation or None,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )
    else:
        policy = ReplayPolicy(args.response)
    result = runner.run_episode(item, policy)
    serialized = _serialize_result(result)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(serialized, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(serialized, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
