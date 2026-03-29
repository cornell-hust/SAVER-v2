#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from saver_agent.offline_scoring import (
    ReferenceDataProvider,
    load_rollout_records,
    save_rollout_records,
    score_rollout_records,
)
from saver_agent.qwen_verifier import DEFAULT_VERIFIER_MODEL_PATH, QwenSelfVerifier
from saver_agent.runtime import (
    distributed_runtime_from_env,
    resolve_inference_device_map,
    resolve_shard_spec,
    runtime_log,
    shard_sequence,
    sharded_output_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch score SAVER rollout files with reward and offline verifier.")
    parser.add_argument("--input", required=True, help="Input rollout path (.json, .jsonl, or directory of .json files).")
    parser.add_argument("--output", required=True, help="Output scored path (.json or .jsonl).")
    parser.add_argument("--num-shards", type=int, default=0, help="Optional number of shard workers.")
    parser.add_argument("--shard-index", type=int, default=-1, help="Optional shard index for this process.")
    parser.add_argument("--progress-every", type=int, default=1, help="Log score progress every N local records.")
    parser.add_argument("--data", default="", help="Optional saver_agent JSONL data used for offline re-verification.")
    parser.add_argument("--data-root", default="", help="Root path used to resolve relative video paths.")
    parser.add_argument(
        "--verifier-backend",
        choices=["heuristic", "qwen_self_verifier", "hybrid"],
        default="heuristic",
        help="Backend used for offline verifier attachment.",
    )
    parser.add_argument(
        "--force-reverify",
        action="store_true",
        help="Re-run offline verifier even if rollout already contains verifier turns.",
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
        help="Generation length for the Qwen verifier.",
    )
    parser.add_argument(
        "--verifier-hybrid-alpha",
        type=float,
        default=0.7,
        help="Hybrid verifier weight on heuristic scores.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = distributed_runtime_from_env()
    shard_spec = resolve_shard_spec(num_shards=args.num_shards, shard_index=args.shard_index, runtime=runtime)
    try:
        records, metadata = load_rollout_records(args.input)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))
    local_records = shard_sequence(records, num_shards=shard_spec.num_shards, shard_index=shard_spec.shard_index)
    output_path = sharded_output_path(args.output, num_shards=shard_spec.num_shards, shard_index=shard_spec.shard_index)
    effective_verifier_device_map = resolve_inference_device_map(args.verifier_device_map, runtime=runtime)

    runtime_log(
        f"score startup: total_records={len(records)} local_records={len(local_records)} "
        f"verifier_backend={args.verifier_backend} output={output_path}",
        runtime=runtime,
    )

    reference_data = None
    verifier_kwargs = {
        "verifier_backend": args.verifier_backend,
        "verifier_model_path": args.verifier_model_path,
        "verifier_torch_dtype": args.verifier_torch_dtype,
        "verifier_device_map": effective_verifier_device_map,
        "verifier_attn_implementation": args.verifier_attn_implementation,
        "verifier_max_new_tokens": args.verifier_max_new_tokens,
        "verifier_hybrid_alpha": args.verifier_hybrid_alpha,
    }
    if args.data:
        reference_data = ReferenceDataProvider(data_path=args.data, data_root=args.data_root)
        if args.verifier_backend in {"qwen_self_verifier", "hybrid"} and local_records:
            runtime_log(
                f"loading verifier model from {args.verifier_model_path} with device_map={effective_verifier_device_map}",
                runtime=runtime,
            )
            verifier_kwargs["verifier_runtime"] = QwenSelfVerifier.from_pretrained(
                args.verifier_model_path,
                torch_dtype=args.verifier_torch_dtype,
                device_map=effective_verifier_device_map,
                attn_implementation=args.verifier_attn_implementation or None,
                max_new_tokens=args.verifier_max_new_tokens,
            )

    scored_records = score_rollout_records(
        local_records,
        reference_data=reference_data,
        verifier_backend=args.verifier_backend,
        force_reverify=args.force_reverify,
        verifier_kwargs=verifier_kwargs,
        progress_every=args.progress_every,
        progress_label="score progress",
        runtime=runtime,
    )
    save_rollout_records(scored_records, output_path, metadata=metadata)
    runtime_log(f"saved {len(scored_records)} scored records to {output_path}", runtime=runtime)


if __name__ == "__main__":
    main()
