#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from split_utils import parse_include_splits

from saver_agent.config import (
    DEFAULT_POLICY_MAX_NEW_TOKENS,
    DEFAULT_TOTAL_VISUAL_BUDGET,
    PromptConfig,
    PreviewConfig,
    RolloutTraceConfig,
    SaverAgentConfig,
)
from saver_agent.dataset import SaverAgentDataset
from saver_agent.evaluation import RolloutEvaluationConfig
from saver_agent.experiment_logging import resolve_experiment_log_dir, utc_timestamp, write_json
from saver_agent.prepared_metadata import ensure_prepared_sft_metadata, write_prepared_sft_metadata
from saver_agent.proposal import SiglipFeatureEncoder
from saver_agent.qwen_policy import DEFAULT_MODEL_PATH
from saver_agent.runtime import distributed_runtime_from_env, runtime_log, should_log_progress
from saver_agent.teacher_judge import (
    QwenTeacherJudge,
    annotate_teacher_judge_examples,
    reweight_teacher_judge_examples,
)
from saver_agent.training import (
    _FrameReferenceResolver,
    default_sft_tensor_cache_dir,
    run_rollout_eval_from_checkpoint,
    run_weighted_sft,
    validate_prepared_examples,
)
from saver_agent.training_data import build_oracle_sft_examples


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warm-start SAVER policy with oracle stepwise SFT supervision.")
    parser.add_argument("--data", default="", help="Path to SAVER agent/oracle JSONL data.")
    parser.add_argument("--prepared-data", default="", help="Optional path to pre-generated lightweight SFT JSONL.")
    parser.add_argument(
        "--tensor-cache-dir",
        default="",
        help="Optional offline tensor cache directory. If omitted and --prepared-data is set, defaults to <prepared-data>.tensor_cache.",
    )
    parser.add_argument("--prepare-output", default="", help="Optional path to write lightweight prepared SFT JSONL.")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare lightweight SFT JSONL and exit before training.")
    parser.add_argument("--teacher-judge-output", default="", help="Optional path to write teacher-annotated prepared SFT JSONL.")
    parser.add_argument("--teacher-judge-only", action="store_true", help="Annotate prepared SFT examples with the teacher judge and exit before training.")
    parser.add_argument("--data-root", default="", help="Root path used to resolve relative video paths.")
    parser.add_argument("--include-splits", default="", help="Optional comma-separated split whitelist for --data/--prepared-data, e.g. train or train,val.")
    parser.add_argument("--output-dir", default="", help="Training output directory.")
    parser.add_argument("--log-dir", default="", help="Optional directory for SFT logs. Defaults to <output-dir>/logs.")
    parser.add_argument(
        "--rollout-eval-output-dir",
        default="",
        help="Optional directory for epoch-end rollout eval outputs. Defaults to <output-dir>.",
    )
    parser.add_argument("--resume-from-checkpoint", default="", help="Optional Trainer/model checkpoint used to resume SFT or replay a missing epoch-end rollout eval.")
    parser.add_argument("--resume-rollout-eval-only", action="store_true", help="Only replay the missing epoch-end rollout eval for --resume-from-checkpoint, then exit.")
    parser.set_defaults(inline_rollout_eval=True)
    parser.add_argument(
        "--inline-rollout-eval",
        dest="inline_rollout_eval",
        action="store_true",
        help="Run epoch-end rollout eval inline before the next epoch. This is the default behavior.",
    )
    parser.add_argument(
        "--defer-rollout-eval",
        dest="inline_rollout_eval",
        action="store_false",
        help="Defer epoch-end rollout eval to the external recovery path instead of running it immediately after each epoch.",
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Local Qwen model path.")
    parser.add_argument("--proposal-model-path", default="", help="Optional local SigLIP/CLIP path for query-conditioned proposal during SFT example preparation.")
    parser.add_argument("--proposal-torch-dtype", default="auto", help="Torch dtype for the proposal encoder used during SFT example preparation.")
    parser.add_argument("--proposal-device", default="", help="Optional device for the proposal encoder during SFT example preparation.")
    parser.add_argument("--max-records", type=int, default=0, help="Optional limit on the number of records.")
    parser.add_argument("--max-train-examples", type=int, default=0, help="Optional limit on built SFT examples.")
    parser.add_argument("--dry-run", action="store_true", help="Build examples and print summary without training.")
    parser.add_argument("--dry-run-json", default="", help="Optional path to dump built SFT examples as JSON.")
    parser.add_argument("--validate-prepared-data", action="store_true", help="Validate prepared SFT examples before exiting or training.")
    parser.add_argument("--validate-materialization", action="store_true", help="During validation, resolve image_ref payloads back to frames.")
    parser.add_argument("--validation-max-examples", type=int, default=0, help="Optional cap on how many prepared examples to materialize during validation.")
    parser.add_argument("--progress-every", type=int, default=25, help="Log data preparation or validation progress every N items. First and last items are always logged.")
    parser.add_argument("--skip-invalid-jsonl-lines", action="store_true", help="Skip malformed JSONL lines instead of failing immediately. Prefer regenerating the source file when possible.")
    parser.add_argument("--teacher-judge-model-path", default="", help="Optional Qwen teacher judge model path used to annotate verify_hypothesis SFT examples.")
    parser.add_argument(
        "--teacher-judge-input-mode",
        choices=["text_only", "multimodal_visual", "auto"],
        default="auto",
        help="Teacher judge input mode used during prepared-data annotation.",
    )
    parser.add_argument("--teacher-judge-torch-dtype", default="auto", help="Torch dtype for the teacher judge.")
    parser.add_argument("--teacher-judge-device-map", default="auto", help="device_map for the teacher judge.")
    parser.add_argument("--teacher-judge-attn-implementation", default="", help="Optional attention backend for the teacher judge.")
    parser.add_argument("--teacher-judge-max-new-tokens", type=int, default=384, help="Generation length for the teacher judge.")
    parser.add_argument("--teacher-judge-max-images", type=int, default=8, help="Maximum images passed to the teacher judge per example.")
    parser.add_argument(
        "--teacher-judge-topk-frames-per-view",
        type=int,
        default=4,
        help="Maximum number of frames sampled into each teacher-judge view package.",
    )
    parser.add_argument("--teacher-judge-overwrite-existing", action="store_true", help="Overwrite existing teacher judge labels when annotating prepared SFT examples.")
    parser.add_argument("--teacher-judge-frame-cache-max-cached-videos", type=int, default=64, help="How many frame_cache tensors/video readers to keep open while resolving image_ref payloads for teacher judging.")
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
        default=DEFAULT_TOTAL_VISUAL_BUDGET,
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
    parser.add_argument("--eval-rollout-max-turns", type=int, default=14, help="Maximum rollout turns for epoch-end eval.")
    parser.add_argument(
        "--eval-max-new-tokens-per-turn",
        type=int,
        default=DEFAULT_POLICY_MAX_NEW_TOKENS,
        help="Generation length budget for each epoch-end rollout eval turn.",
    )
    parser.add_argument(
        "--eval-total-visual-budget",
        type=int,
        default=0,
        help="Alias for a coarse epoch-end rollout visual budget. Currently resolved as a total-image cap when --eval-max-total-images is unset.",
    )
    parser.add_argument(
        "--eval-max-total-images",
        type=int,
        default=DEFAULT_TOTAL_VISUAL_BUDGET,
        help="Optional hard cap on total images preserved in each epoch-end rollout eval prompt. 0 keeps all images.",
    )
    parser.add_argument(
        "--eval-verifier-backend",
        choices=["heuristic", "qwen_self_verifier", "hybrid"],
        default="heuristic",
        help="Diagnostic offline verifier backend used only when --eval-attach-reference-diagnostics is enabled.",
    )
    parser.add_argument("--eval-verifier-model-path", default="", help="Optional verifier model path for epoch-end rollout eval.")
    parser.add_argument("--eval-proposal-model-path", default="", help="Optional proposal encoder path for epoch-end rollout eval. Defaults to --proposal-model-path.")
    parser.add_argument("--eval-proposal-torch-dtype", default="auto", help="Torch dtype for epoch-end eval proposal encoder.")
    parser.add_argument("--eval-proposal-device", default="", help="Optional device for epoch-end eval proposal encoder.")
    parser.add_argument("--eval-verifier-torch-dtype", default="auto", help="Torch dtype for epoch-end eval verifier.")
    parser.add_argument("--eval-verifier-device-map", default="auto", help="device_map for epoch-end eval verifier.")
    parser.add_argument("--eval-verifier-attn-implementation", default="", help="Attention backend for epoch-end eval verifier.")
    parser.add_argument("--eval-verifier-max-new-tokens", type=int, default=512, help="Generation length for epoch-end eval verifier.")
    parser.add_argument("--eval-verifier-hybrid-alpha", type=float, default=0.7, help="Hybrid alpha for epoch-end eval verifier.")
    parser.add_argument(
        "--eval-attach-reference-diagnostics",
        action="store_true",
        help="Attach reference-conditioned offline verifier diagnostics during epoch-end rollout eval. Main metrics remain reference-free.",
    )
    parser.add_argument("--eval-progress-every", type=int, default=1, help="Log epoch-end rollout eval progress every N local items.")
    return parser.parse_args(argv)


def _resolve_eval_max_total_images(args: argparse.Namespace) -> int:
    explicit_max_total_images = int(getattr(args, "eval_max_total_images", 0) or 0)
    if explicit_max_total_images > 0:
        return explicit_max_total_images
    return max(0, int(getattr(args, "eval_total_visual_budget", 0) or 0))


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
        inline_rollout_eval=bool(args.inline_rollout_eval),
        rollout_max_turns=args.eval_rollout_max_turns,
        policy_max_new_tokens=args.eval_max_new_tokens_per_turn,
        max_total_images=_resolve_eval_max_total_images(args),
        max_image_side=args.max_image_side,
        max_image_pixels=args.max_image_pixels,
        verifier_backend=args.eval_verifier_backend,
        verifier_model_path=args.eval_verifier_model_path or args.model_path,
        proposal_model_path=args.eval_proposal_model_path or args.proposal_model_path,
        proposal_torch_dtype=args.eval_proposal_torch_dtype,
        proposal_device=args.eval_proposal_device,
        verifier_torch_dtype=args.eval_verifier_torch_dtype,
        verifier_device_map=args.eval_verifier_device_map,
        verifier_attn_implementation=args.eval_verifier_attn_implementation,
        verifier_max_new_tokens=args.eval_verifier_max_new_tokens,
        verifier_hybrid_alpha=args.eval_verifier_hybrid_alpha,
        attach_reference_diagnostics=args.eval_attach_reference_diagnostics,
        progress_every=args.eval_progress_every,
        saver_config=config,
    )


def _resolve_proposal_device(explicit_device: str, *, runtime: Any) -> str:
    if str(explicit_device or "").strip():
        return str(explicit_device)
    try:
        import torch
    except Exception:
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    try:
        visible_cuda_devices = int(torch.cuda.device_count())
    except Exception:
        visible_cuda_devices = 0
    if visible_cuda_devices <= 0:
        return "cpu"
    local_rank = int(getattr(runtime, "local_rank", 0) or 0)
    if 0 <= local_rank < visible_cuda_devices:
        return f"cuda:{local_rank}"
    runtime_log(
        (
            "SFT proposal device fallback: "
            f"local_rank={local_rank} is outside visible_cuda_devices={visible_cuda_devices}; using cuda:0"
        ),
        runtime=runtime,
        main_process_only=True,
    )
    return "cuda:0"


def _read_json_file(path: str | Path) -> Dict[str, Any]:
    candidate = Path(path)
    if not candidate.exists():
        return {}
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_resume_epoch_index(checkpoint_path: str | Path) -> int:
    checkpoint_dir = Path(checkpoint_path)
    match = re.fullmatch(r"epoch_(\d+)", checkpoint_dir.name)
    if match:
        return max(1, int(match.group(1)))

    resume_metadata = _read_json_file(checkpoint_dir / "resume_metadata.json")
    if int(resume_metadata.get("epoch_index", 0) or 0) > 0:
        return int(resume_metadata["epoch_index"])

    trainer_state = _read_json_file(checkpoint_dir / "trainer_state.json")
    epoch_value = trainer_state.get("epoch")
    try:
        epoch_float = float(epoch_value)
    except Exception as exc:
        raise ValueError(f"Unable to infer resume epoch index from {checkpoint_dir}") from exc
    epoch_index = int(round(epoch_float))
    if epoch_index <= 0:
        raise ValueError(f"Resolved non-positive epoch index from {checkpoint_dir}: epoch={epoch_float}")
    return epoch_index


def _build_proposal_runtime(args: argparse.Namespace, *, runtime: Any) -> SiglipFeatureEncoder | None:
    if not args.proposal_model_path:
        return None
    return SiglipFeatureEncoder.from_pretrained(
        args.proposal_model_path,
        torch_dtype=args.proposal_torch_dtype,
        device=_resolve_proposal_device(args.proposal_device, runtime=runtime),
    )


def _attach_proposal_runtime(item: Dict[str, Any], proposal_runtime: Any) -> None:
    if proposal_runtime is not None:
        item["multimodal_cache"]["proposal_runtime"] = proposal_runtime


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
    proposal_runtime: Any = None,
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
        _attach_proposal_runtime(item, proposal_runtime)
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
    proposal_runtime: Any = None,
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
        _attach_proposal_runtime(item, proposal_runtime)
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


def _maybe_annotate_examples_with_teacher_judge(
    args: argparse.Namespace,
    examples: List[Dict[str, Any]],
    *,
    runtime: Any,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not args.teacher_judge_model_path:
        return examples, {}
    if getattr(runtime, "is_distributed", False):
        raise ValueError(
            "Inline teacher-judge annotation inside distributed SFT is not supported. "
            "Run annotate_teacher_judge_sft.py first to produce an annotated prepared JSONL, then train with --prepared-data."
        )

    image_resolver = None
    if str(args.teacher_judge_input_mode or "").strip().lower() != "text_only":
        resolver = _FrameReferenceResolver(max_cached_videos=args.teacher_judge_frame_cache_max_cached_videos)
        image_resolver = resolver._resolve_image_ref

    runtime_log(
        f"loading teacher judge from {args.teacher_judge_model_path} with input_mode={args.teacher_judge_input_mode}",
        runtime=runtime,
        main_process_only=True,
    )
    teacher_judge = QwenTeacherJudge.from_pretrained(
        args.teacher_judge_model_path,
        torch_dtype=args.teacher_judge_torch_dtype,
        device_map=args.teacher_judge_device_map,
        attn_implementation=args.teacher_judge_attn_implementation or None,
        input_mode=args.teacher_judge_input_mode,
        max_new_tokens=args.teacher_judge_max_new_tokens,
        max_images=args.teacher_judge_max_images,
        topk_frames_per_view=args.teacher_judge_topk_frames_per_view,
        image_resolver=image_resolver,
    )
    annotated_examples, summary = annotate_teacher_judge_examples(
        examples,
        judge=teacher_judge,
        input_mode=args.teacher_judge_input_mode,
        overwrite_existing=args.teacher_judge_overwrite_existing,
        progress_every=args.progress_every,
        log_fn=lambda message: runtime_log(message, runtime=runtime, main_process_only=True),
    )
    return annotated_examples, summary


def _apply_teacher_judge_reweighting(
    examples: List[Dict[str, Any]],
    *,
    runtime: Any,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    reweighted_examples, summary = reweight_teacher_judge_examples(examples)
    if int(summary.get("num_teacher_judge_reweighted", 0)) > 0:
        runtime_log(
            f"teacher judge reweighting: reweighted={summary['num_teacher_judge_reweighted']}",
            runtime=runtime,
            main_process_only=True,
        )
    return reweighted_examples, summary


def main() -> None:
    args = parse_args()
    if not args.resume_rollout_eval_only and not args.data and not args.prepared_data:
        raise ValueError("Either --data or --prepared-data must be provided.")
    if args.resume_rollout_eval_only and not args.resume_from_checkpoint:
        raise ValueError("--resume-rollout-eval-only requires --resume-from-checkpoint.")
    runtime = distributed_runtime_from_env()
    log_dir = resolve_experiment_log_dir(
        args.log_dir,
        output_dir=args.output_dir,
        fallback_paths=[args.prepare_output, args.teacher_judge_output, args.dry_run_json],
    )
    rollout_eval_output_dir = str(args.rollout_eval_output_dir or "").strip() or str(args.output_dir or "").strip()
    if runtime.is_main_process and log_dir is not None:
        write_json(
            log_dir / "train_saver_sft_run_config.json",
            {
                "timestamp_utc": utc_timestamp(),
                "data": args.data,
                "prepared_data": args.prepared_data,
                "data_root": args.data_root,
                "include_splits": parse_include_splits(args.include_splits) or [],
                "output_dir": args.output_dir,
                "log_dir": str(log_dir),
                "rollout_eval_output_dir": str(rollout_eval_output_dir),
                "resume_from_checkpoint": args.resume_from_checkpoint,
                "resume_rollout_eval_only": bool(args.resume_rollout_eval_only),
                "inline_rollout_eval": bool(args.inline_rollout_eval),
                "model_path": args.model_path,
                "tensor_cache_dir": args.tensor_cache_dir,
                "prepare_output": args.prepare_output,
                "teacher_judge_output": args.teacher_judge_output,
                "prepare_only": bool(args.prepare_only),
                "teacher_judge_only": bool(args.teacher_judge_only),
                "dry_run": bool(args.dry_run),
                "eval_data": args.eval_data,
                "eval_rollout_max_turns": int(args.eval_rollout_max_turns),
                "eval_attach_reference_diagnostics": bool(args.eval_attach_reference_diagnostics),
                "teacher_judge_model_path": args.teacher_judge_model_path,
                "teacher_judge_input_mode": args.teacher_judge_input_mode,
            },
        )
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
    if args.resume_rollout_eval_only:
        if rollout_eval_config is None:
            raise ValueError("--resume-rollout-eval-only requires --eval-data so the missing rollout eval can be replayed.")
        epoch_index = _resolve_resume_epoch_index(args.resume_from_checkpoint)
        recovery_kwargs = dict(
            checkpoint_path=args.resume_from_checkpoint,
            output_dir=args.output_dir,
            rollout_eval_config=rollout_eval_config,
            epoch_index=epoch_index,
            model_path=args.model_path,
            torch_dtype=args.torch_dtype,
            attn_implementation=args.attn_implementation or None,
            runtime=runtime,
        )
        if str(args.rollout_eval_output_dir or "").strip():
            recovery_kwargs["rollout_eval_output_dir"] = rollout_eval_output_dir
        result = run_rollout_eval_from_checkpoint(
            **recovery_kwargs,
        )
        final_summary = {
            "resume_from_checkpoint": args.resume_from_checkpoint,
            "resume_rollout_eval_only": True,
            "resume_epoch_index": int(epoch_index),
            **(result or {}),
        }
        if runtime.is_main_process and log_dir is not None:
            write_json(log_dir / "train_saver_sft_summary.json", final_summary)
        if runtime.is_main_process:
            print(json.dumps(final_summary, ensure_ascii=False, indent=2))
        return
    proposal_runtime = None
    if not args.prepared_data and args.proposal_model_path:
        runtime_log(
            f"loading proposal model from {args.proposal_model_path} for SFT example preparation",
            runtime=runtime,
            main_process_only=True,
        )
        proposal_runtime = _build_proposal_runtime(args, runtime=runtime)
    if args.prepared_data:
        ensure_prepared_sft_metadata(
            args.prepared_data,
            config=config,
            require_config_match=True,
        )
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
            proposal_runtime=proposal_runtime,
        )
    if args.max_train_examples > 0:
        examples = examples[: args.max_train_examples]
    teacher_judge_summary: Dict[str, Any] = {}
    examples, teacher_judge_summary = _maybe_annotate_examples_with_teacher_judge(
        args,
        examples,
        runtime=runtime,
    )
    examples, teacher_judge_reweight_summary = _apply_teacher_judge_reweighting(examples, runtime=runtime)

    summary = _summarize_examples(examples)
    validation = _run_validation(args, examples)
    if args.prepare_output and runtime.is_main_process:
        _write_jsonl(args.prepare_output, examples)
        write_prepared_sft_metadata(args.prepare_output, config=config)
    if args.teacher_judge_output and runtime.is_main_process:
        _write_jsonl(args.teacher_judge_output, examples)
        write_prepared_sft_metadata(
            args.teacher_judge_output,
            config=config,
            extra_fields={"teacher_annotated": True},
        )
    if args.dry_run_json:
        if runtime.is_main_process:
            output_path = Path(args.dry_run_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.teacher_judge_only or args.prepare_only or args.dry_run or not args.output_dir:
        final_summary = {**summary, **teacher_judge_summary, **teacher_judge_reweight_summary, **validation}
        if runtime.is_main_process and log_dir is not None:
            write_json(log_dir / "train_saver_sft_summary.json", final_summary)
        if runtime.is_main_process:
            print(
                json.dumps(
                    final_summary,
                    ensure_ascii=False,
                    indent=2,
                )
            )
        return

    tensor_cache_dir = args.tensor_cache_dir
    if not tensor_cache_dir and args.prepared_data:
        tensor_cache_dir = str(default_sft_tensor_cache_dir(args.prepared_data))

    lora_target_modules = [module.strip() for module in args.lora_target_modules.split(",") if module.strip()]
    result = run_weighted_sft(
        examples,
        model_path=args.model_path,
        output_dir=args.output_dir,
        log_dir=str(log_dir) if log_dir is not None else "",
        rollout_eval_output_dir=rollout_eval_output_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
        tensor_cache_dir=tensor_cache_dir,
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
    final_summary = {**summary, **teacher_judge_summary, **teacher_judge_reweight_summary, **validation, **result}
    if runtime.is_main_process and log_dir is not None:
        write_json(log_dir / "train_saver_sft_summary.json", final_summary)
    if runtime.is_main_process:
        print(
            json.dumps(
                final_summary,
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
