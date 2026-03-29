#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from split_utils import parse_include_splits

from saver_agent.config import PromptConfig, PreviewConfig, RolloutTraceConfig, SaverAgentConfig
from saver_agent.dataset import SaverAgentDataset
from saver_agent.evaluation import RolloutEvaluationConfig
from saver_agent.qwen_policy import DEFAULT_MODEL_PATH
from saver_agent.runtime import distributed_runtime_from_env, runtime_log, should_log_progress
from saver_agent.training import run_weighted_sft, validate_prepared_examples
from saver_agent.training_data import build_oracle_sft_examples


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warm-start SAVER policy with oracle stepwise SFT supervision.")
    parser.add_argument("--data", default="", help="Path to SAVER agent/oracle JSONL data.")
    parser.add_argument("--prepared-data", default="", help="Optional path to pre-generated lightweight SFT JSONL.")
    parser.add_argument("--prepare-output", default="", help="Optional path to write lightweight prepared SFT JSONL.")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare lightweight SFT JSONL and exit before training.")
    parser.add_argument("--data-root", default="", help="Root path used to resolve relative video paths.")
    parser.add_argument("--include-splits", default="", help="Optional comma-separated split whitelist for --data/--prepared-data, e.g. train or train,val.")
    parser.add_argument("--output-dir", default="", help="Training output directory.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Local Qwen model path.")
    parser.add_argument("--max-records", type=int, default=0, help="Optional limit on the number of records.")
    parser.add_argument("--max-train-examples", type=int, default=0, help="Optional limit on built SFT examples.")
    parser.add_argument("--dry-run", action="store_true", help="Build examples and print summary without training.")
    parser.add_argument("--dry-run-json", default="", help="Optional path to dump built SFT examples as JSON.")
    parser.add_argument("--validate-prepared-data", action="store_true", help="Validate prepared SFT examples before exiting or training.")
    parser.add_argument("--validate-materialization", action="store_true", help="During validation, resolve image_ref payloads back to frames.")
    parser.add_argument("--validation-max-examples", type=int, default=0, help="Optional cap on how many prepared examples to materialize during validation.")
    parser.add_argument("--progress-every", type=int, default=25, help="Log data preparation or validation progress every N items. First and last items are always logged.")
    parser.add_argument("--skip-invalid-jsonl-lines", action="store_true", help="Skip malformed JSONL lines instead of failing immediately. Prefer regenerating the source file when possible.")
    parser.add_argument("--torch-dtype", default="auto", help="Torch dtype passed to from_pretrained.")
    parser.add_argument("--attn-implementation", default="", help="Optional attention backend for Qwen.")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable model gradient checkpointing.")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Training learning rate.")
    parser.add_argument("--num-train-epochs", type=float, default=1.0, help="Number of training epochs.")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1, help="Per-device batch size.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16, help="Gradient accumulation steps.")
    parser.add_argument("--dataloader-num-workers", type=int, default=0, help="DataLoader worker count for training.")
    parser.add_argument(
        "--dataloader-prefetch-factor",
        type=int,
        default=0,
        help="Optional DataLoader prefetch_factor. Only used when dataloader_num_workers > 0.",
    )
    parser.add_argument(
        "--dataloader-persistent-workers",
        action="store_true",
        help="Keep DataLoader workers alive across epochs. Only used when dataloader_num_workers > 0.",
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
    parser.add_argument("--max-image-side", type=int, default=0, help="Optional training-time max image side length in pixels. 0 disables resizing.")
    parser.add_argument("--max-image-pixels", type=int, default=0, help="Optional training-time max image area in pixels. 0 disables resizing.")
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=4096,
        help="Explicit tokenizer/processor max_length for SFT examples. Uses left truncation to keep recent turns and the response. 0 disables truncation.",
    )
    parser.add_argument(
        "--keep-recent-tool-image-messages",
        type=int,
        default=0,
        help="If >0, keep images only for the N most recent tool messages during SFT; older tool images are dropped.",
    )
    parser.add_argument(
        "--keep-recent-text-messages",
        type=int,
        default=12,
        help="If >0, keep full text only for the N most recent non-initial history messages during SFT; older assistant/tool history is dropped before tokenization.",
    )
    parser.add_argument(
        "--max-total-images",
        type=int,
        default=0,
        help="Optional hard cap on total images kept in each SFT example after pruning. 0 keeps all images.",
    )
    parser.add_argument("--num-preview-frames", type=int, default=8, help="Preview frames for initial prompt.")
    parser.add_argument("--preview-sampling-fps", type=float, default=None, help="Preview sampling fps.")
    parser.add_argument("--initial-user-template", default="", help="Optional custom initial user template.")
    parser.add_argument("--preview-instruction", default="", help="Optional custom preview instruction.")
    parser.add_argument("--tool-response-template", default="", help="Optional custom tool response template.")
    parser.add_argument("--eval-data", default="", help="Optional raw saver_agent/oracle JSONL used for rollout metrics after each epoch.")
    parser.add_argument("--eval-data-root", default="", help="Root path used to resolve relative video paths for epoch-end rollout eval.")
    parser.add_argument("--eval-include-splits", default="", help="Optional comma-separated split whitelist for --eval-data.")
    parser.add_argument("--eval-max-records", type=int, default=0, help="Optional cap on how many eval records to use per epoch.")
    parser.add_argument("--eval-rollout-max-turns", type=int, default=6, help="Maximum rollout turns for epoch-end eval.")
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


def _jsonl_decode_error_message(path: str | Path, line_number: int, line: str, exc: Exception) -> str:
    preview = line.strip().replace("\t", " ")
    if len(preview) > 240:
        preview = preview[:240] + "..."
    return f"Invalid JSONL at {path}:{line_number}: {exc}. Line preview: {preview}"


def _load_jsonl(
    path: str | Path,
    *,
    skip_invalid_lines: bool = False,
    include_splits: Optional[str | List[str]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    invalid_messages: List[str] = []
    allowed_splits = set(parse_include_splits(include_splits) or [])
    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                message = _jsonl_decode_error_message(path, line_number, line, exc)
                if not skip_invalid_lines:
                    raise ValueError(message) from exc
                invalid_messages.append(message)
                continue
            if allowed_splits and str(row.get("split") or "").strip() not in allowed_splits:
                continue
            rows.append(row)
    if invalid_messages:
        print(
            json.dumps(
                {
                    "warning": "skipped_invalid_jsonl_lines",
                    "path": str(path),
                    "num_skipped": len(invalid_messages),
                    "first_error": invalid_messages[0],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return rows


def _write_jsonl(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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
            record_observation_content=True,
            record_state_deltas=True,
            record_message_history=True,
        ),
    )


def _build_rollout_eval_config(
    args: argparse.Namespace,
    *,
    config: SaverAgentConfig,
) -> Optional[RolloutEvaluationConfig]:
    if not args.eval_data:
        return None
    return RolloutEvaluationConfig(
        data_path=args.eval_data,
        data_root=args.eval_data_root or args.data_root,
        include_splits=parse_include_splits(args.eval_include_splits),
        max_records=args.eval_max_records,
        rollout_max_turns=args.eval_rollout_max_turns,
        verifier_backend=args.eval_verifier_backend,
        verifier_model_path=args.eval_verifier_model_path or args.model_path,
        verifier_torch_dtype=args.eval_verifier_torch_dtype,
        verifier_device_map=args.eval_verifier_device_map,
        verifier_attn_implementation=args.eval_verifier_attn_implementation,
        verifier_max_new_tokens=args.eval_verifier_max_new_tokens,
        verifier_hybrid_alpha=args.eval_verifier_hybrid_alpha,
        attach_reference_diagnostics=args.eval_attach_reference_diagnostics,
        progress_every=args.eval_progress_every,
        saver_config=config,
    )


def build_sft_examples_from_jsonl(
    *,
    data_path: str | Path,
    data_root: str | Path = "",
    max_records: int = 0,
    skip_invalid_jsonl_lines: bool = False,
    include_splits: Optional[str | List[str]] = None,
    progress_every: int = 0,
    runtime=None,
    config: Optional[SaverAgentConfig] = None,
) -> List[Dict[str, Any]]:
    config = config or SaverAgentConfig()
    runtime = runtime or distributed_runtime_from_env()
    dataset = SaverAgentDataset(
        data_path,
        data_root=data_root,
        skip_invalid_jsonl_lines=skip_invalid_jsonl_lines,
        config=config,
        include_splits=include_splits,
    )
    runtime_log(
        dataset.format_frame_cache_status(prefix="SFT source frame cache"),
        runtime=runtime,
        main_process_only=True,
    )
    raw_records = list(dataset.records)
    if max_records > 0:
        raw_records = raw_records[:max_records]
    total_records = len(raw_records)
    examples: List[Dict[str, Any]] = []
    for idx, record in enumerate(raw_records):
        item = dataset[idx]
        examples.extend(build_oracle_sft_examples(item, record, config=config))
        completed = idx + 1
        if should_log_progress(completed, total_records, int(progress_every)):
            runtime_log(
                f"SFT example build progress: records={completed}/{total_records} examples={len(examples)}",
                runtime=runtime,
                main_process_only=True,
            )
    return examples


def build_prepared_sft_examples_from_jsonl(
    *,
    data_path: str | Path,
    data_root: str | Path = "",
    max_records: int = 0,
    skip_invalid_jsonl_lines: bool = False,
    include_splits: Optional[str | List[str]] = None,
    progress_every: int = 0,
    runtime=None,
    config: Optional[SaverAgentConfig] = None,
) -> List[Dict[str, Any]]:
    config = config or SaverAgentConfig()
    runtime = runtime or distributed_runtime_from_env()
    dataset = SaverAgentDataset(
        data_path,
        data_root=data_root,
        skip_invalid_jsonl_lines=skip_invalid_jsonl_lines,
        config=config,
        include_splits=include_splits,
    )
    runtime_log(
        dataset.format_frame_cache_status(prefix="prepared SFT source frame cache"),
        runtime=runtime,
        main_process_only=True,
    )
    raw_records = list(dataset.records)
    if max_records > 0:
        raw_records = raw_records[:max_records]
    total_records = len(raw_records)
    examples: List[Dict[str, Any]] = []
    for idx, record in enumerate(raw_records):
        item = dataset[idx]
        examples.extend(
            build_oracle_sft_examples(
                item,
                record,
                config=config,
                serialize_messages=True,
            )
        )
        completed = idx + 1
        if should_log_progress(completed, total_records, int(progress_every)):
            runtime_log(
                f"Prepared SFT build progress: records={completed}/{total_records} examples={len(examples)}",
                runtime=runtime,
                main_process_only=True,
            )
    return examples


def _summarize_examples(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not examples:
        return {"num_examples": 0, "num_records": 0}
    record_ids = {example.get("video_id") for example in examples}
    return {
        "num_examples": len(examples),
        "num_records": len(record_ids),
        "num_answer_examples": sum(1 for example in examples if example.get("target_action") == "answer"),
        "num_tool_examples": sum(1 for example in examples if example.get("target_action") == "tool_call"),
    }


def _run_validation(args: argparse.Namespace, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not args.validate_prepared_data:
        return {}
    validation = validate_prepared_examples(
        examples,
        materialize_images=args.validate_materialization,
        max_materialized_examples=args.validation_max_examples,
        progress_every=args.progress_every,
    )
    if validation.get("num_errors", 0) > 0:
        error_preview = dict(validation)
        error_preview["errors"] = list(validation.get("errors") or [])[:10]
        raise ValueError(
            "Prepared SFT data validation failed. "
            f"Preview: {json.dumps(error_preview, ensure_ascii=False, indent=2)}"
        )
    return validation


def main() -> None:
    args = parse_args()
    if not args.data and not args.prepared_data:
        raise ValueError("Either --data or --prepared-data must be provided.")
    runtime = distributed_runtime_from_env()
    runtime_log(
        (
            f"SFT startup: data={args.data or '(none)'} prepared_data={args.prepared_data or '(none)'} "
            f"model_path={args.model_path} output_dir={args.output_dir or '(dry-run)'} "
            f"include_splits={parse_include_splits(args.include_splits) or 'all'}"
        ),
        runtime=runtime,
        main_process_only=True,
    )
    config = _build_config(args)
    rollout_eval_config = _build_rollout_eval_config(args, config=config)
    if args.prepared_data:
        runtime_log(f"loading prepared SFT examples from {args.prepared_data}", runtime=runtime, main_process_only=True)
        examples = _load_jsonl(
            args.prepared_data,
            skip_invalid_lines=args.skip_invalid_jsonl_lines,
            include_splits=args.include_splits,
        )
    else:
        if runtime.is_distributed:
            runtime_log(
                "building prepared SFT examples directly from --data under torchrun will repeat data work on every rank; "
                "prefer `--prepare-only --prepare-output ...` first, then train with `--prepared-data`.",
                runtime=runtime,
                main_process_only=True,
            )
        runtime_log("building lightweight oracle SFT examples from dataset", runtime=runtime, main_process_only=True)
        examples = build_prepared_sft_examples_from_jsonl(
            data_path=args.data,
            data_root=args.data_root,
            max_records=args.max_records,
            skip_invalid_jsonl_lines=args.skip_invalid_jsonl_lines,
            include_splits=args.include_splits,
            progress_every=args.progress_every,
            runtime=runtime,
            config=config,
        )
    if args.max_train_examples > 0:
        examples = examples[: args.max_train_examples]

    summary = _summarize_examples(examples)
    validation = _run_validation(args, examples)
    if args.prepare_output and runtime.is_main_process:
        _write_jsonl(args.prepare_output, examples)
    if args.dry_run_json:
        if runtime.is_main_process:
            output_path = Path(args.dry_run_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.prepare_only or args.dry_run or not args.output_dir:
        if runtime.is_main_process:
            print(json.dumps({**summary, **validation}, ensure_ascii=False, indent=2))
        return

    lora_target_modules = [module.strip() for module in args.lora_target_modules.split(",") if module.strip()]
    result = run_weighted_sft(
        examples,
        model_path=args.model_path,
        output_dir=args.output_dir,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation or None,
        gradient_checkpointing=args.gradient_checkpointing,
        use_lora=args.lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=lora_target_modules or None,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_prefetch_factor=args.dataloader_prefetch_factor,
        dataloader_persistent_workers=args.dataloader_persistent_workers,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        bf16=args.bf16,
        fp16=args.fp16,
        max_image_side=args.max_image_side,
        max_image_pixels=args.max_image_pixels,
        max_seq_length=args.max_seq_length,
        keep_recent_tool_image_messages=args.keep_recent_tool_image_messages,
        keep_recent_text_messages=args.keep_recent_text_messages,
        max_total_images=args.max_total_images,
        rollout_eval_config=rollout_eval_config,
    )
    if runtime.is_main_process:
        print(json.dumps({**summary, **validation, **result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
