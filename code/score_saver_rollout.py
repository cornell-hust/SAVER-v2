#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from saver_agent.offline_scoring import (
    ReferenceDataProvider,
    attach_teacher_judge_to_records,
    load_rollout_records,
    save_rollout_records,
    score_rollout_records,
)
from saver_agent.teacher_judge import QwenTeacherJudge
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
        "--attach-reference-offline-verifier",
        action="store_true",
        help="Attach reference-conditioned offline verifier results. Diagnostic only; disabled by default.",
    )
    parser.add_argument(
        "--verifier-backend",
        choices=["heuristic", "qwen_self_verifier", "hybrid"],
        default="heuristic",
        help="Backend used for diagnostic offline verifier attachment.",
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
    parser.add_argument(
        "--teacher-judge-model-path",
        default="",
        help="Optional offline Qwen teacher judge model path used for diagnostic verify_hypothesis annotation.",
    )
    parser.add_argument(
        "--teacher-judge-input-mode",
        choices=["text_only", "multimodal_visual", "auto"],
        default="auto",
        help="Teacher judge input mode for diagnostic scoring.",
    )
    parser.add_argument("--teacher-judge-torch-dtype", default="auto", help="Torch dtype for the teacher judge.")
    parser.add_argument("--teacher-judge-device-map", default="auto", help="device_map for the teacher judge.")
    parser.add_argument("--teacher-judge-attn-implementation", default="", help="Optional attention backend for the teacher judge.")
    parser.add_argument("--teacher-judge-max-new-tokens", type=int, default=384, help="Generation length for the teacher judge.")
    parser.add_argument("--teacher-judge-max-images", type=int, default=8, help="Maximum images passed to the teacher judge per verify turn.")
    parser.add_argument(
        "--teacher-judge-topk-frames-per-view",
        type=int,
        default=4,
        help="Maximum raw frames sampled for each teacher-judge view.",
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

    diagnostic_reference_eval = bool(args.attach_reference_offline_verifier)
    teacher_diagnostic = bool(args.teacher_judge_model_path)
    verifier_status = f"on(backend={args.verifier_backend})" if diagnostic_reference_eval else "off"
    teacher_status = "on" if teacher_diagnostic else "off"
    runtime_log(
        (
            f"score startup: total_records={len(records)} local_records={len(local_records)} "
            f"reference_diagnostic={verifier_status} teacher_diagnostic={teacher_status} "
            f"output={output_path}"
        ),
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
    if args.attach_reference_offline_verifier and not args.data:
        raise SystemExit("--attach-reference-offline-verifier requires --data.")

    if args.data:
        reference_data = ReferenceDataProvider(data_path=args.data, data_root=args.data_root)
        if (
            args.attach_reference_offline_verifier
            and args.verifier_backend in {"qwen_self_verifier", "hybrid"}
            and local_records
        ):
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
    elif args.teacher_judge_model_path:
        raise SystemExit("--teacher-judge-model-path requires --data so diagnostic teacher annotation can rebuild verify-turn messages.")

    if args.teacher_judge_model_path and local_records:
        effective_teacher_device_map = resolve_inference_device_map(args.teacher_judge_device_map, runtime=runtime)
        runtime_log(
            f"loading teacher judge model from {args.teacher_judge_model_path} with device_map={effective_teacher_device_map}",
            runtime=runtime,
        )
        teacher_judge = QwenTeacherJudge.from_pretrained(
            args.teacher_judge_model_path,
            torch_dtype=args.teacher_judge_torch_dtype,
            device_map=effective_teacher_device_map,
            attn_implementation=args.teacher_judge_attn_implementation or None,
            input_mode=args.teacher_judge_input_mode,
            max_new_tokens=args.teacher_judge_max_new_tokens,
            max_images=args.teacher_judge_max_images,
            topk_frames_per_view=args.teacher_judge_topk_frames_per_view,
        )
        local_records, teacher_summary = attach_teacher_judge_to_records(
            local_records,
            reference_data=reference_data,
            judge=teacher_judge,
            input_mode=args.teacher_judge_input_mode,
            progress_every=args.progress_every,
            progress_label="teacher judge progress",
            runtime=runtime,
        )
        runtime_log(
            (
                f"teacher judge annotated "
                f"{teacher_summary['num_teacher_judge_annotated_turns']}/"
                f"{teacher_summary['num_teacher_judge_candidates']} verify turns"
            ),
            runtime=runtime,
        )

    scored_records = score_rollout_records(
        local_records,
        reference_data=reference_data,
        verifier_backend=args.verifier_backend,
        force_reverify=args.force_reverify,
        attach_reference_offline_verifier=args.attach_reference_offline_verifier,
        verifier_kwargs=verifier_kwargs,
        progress_every=args.progress_every,
        progress_label="score progress",
        runtime=runtime,
    )
    save_rollout_records(scored_records, output_path, metadata=metadata)
    runtime_log(f"saved {len(scored_records)} scored records to {output_path}", runtime=runtime)


if __name__ == "__main__":
    main()
