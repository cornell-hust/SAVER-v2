#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from split_utils import parse_include_splits

from saver_agent.qwen_policy import DEFAULT_MODEL_PATH
from saver_agent.runtime import distributed_runtime_from_env, runtime_log, should_log_progress
from saver_agent.training import (
    _FrameReferenceResolver,
    build_processor_signature,
    build_processor_signature_summary,
    build_sft_tensor_cache_key,
    build_sft_tensor_cache_metadata,
    build_sft_tensor_cache_payload,
    default_sft_tensor_cache_dir,
    materialize_example_for_training,
    resolve_sft_tensor_cache_config_from_metadata,
    sft_tensor_cache_entry_path,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline-prepare tokenized multimodal SFT tensors so training can skip image materialization and processor work."
    )
    parser.add_argument("--prepared-data", required=True, help="Path to lightweight prepared SFT JSONL.")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output tensor cache directory. Defaults to <prepared-data>.tensor_cache.",
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Local Qwen model path used to load the processor.")
    parser.add_argument("--include-splits", default="", help="Optional comma-separated split whitelist.")
    parser.add_argument("--max-examples", type=int, default=0, help="Optional cap on the number of prepared examples.")
    parser.add_argument("--skip-invalid-jsonl-lines", action="store_true", help="Skip malformed JSONL lines instead of failing.")
    parser.add_argument("--progress-every", type=int, default=25, help="Log progress every N examples.")
    parser.add_argument("--overwrite-existing", action="store_true", help="Overwrite existing per-example tensor cache entries.")
    parser.add_argument("--max-image-side", type=int, default=0, help="Optional max image side used during training preprocessing.")
    parser.add_argument("--max-image-pixels", type=int, default=0, help="Optional max image area used during training preprocessing.")
    parser.add_argument(
        "--keep-recent-tool-image-messages",
        type=int,
        default=0,
        help="If >0, keep images only for the N most recent tool messages during offline preprocessing.",
    )
    parser.add_argument(
        "--keep-recent-text-messages",
        type=int,
        default=12,
        help="If >0, keep full text only for the N most recent non-initial history messages during offline preprocessing.",
    )
    parser.add_argument(
        "--max-total-images",
        type=int,
        default=0,
        help="Optional hard cap on total images kept in each example. 0 keeps all images.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=4096,
        help="Tokenizer/processor max_length used during offline preprocessing. 0 disables truncation.",
    )
    parser.add_argument(
        "--frame-cache-max-cached-videos",
        type=int,
        default=128,
        help="How many frame_cache tensors/video readers to keep open while materializing image_ref payloads.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split filtered examples into N deterministic shards. Use with --shard-index for parallel cache builds.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard index to build when --num-shards > 1.",
    )
    return parser.parse_args(argv)


def _jsonl_decode_error_message(path: str | Path, line_number: int, line: str, exc: Exception) -> str:
    preview = line.strip().replace("\t", " ")
    if len(preview) > 240:
        preview = preview[:240] + "..."
    return f"Invalid JSONL at {path}:{line_number}: {exc}. Line preview: {preview}"


def _load_prepared_examples(
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


def _load_processor(model_path: str | Path):
    try:
        from transformers import AutoProcessor
    except Exception as exc:
        raise ImportError("Preparing SFT tensor cache requires the `transformers` package.") from exc
    return AutoProcessor.from_pretrained(str(model_path))


def _load_existing_metadata(metadata_path: Path) -> Dict[str, Any]:
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_shard_args(*, num_shards: int, shard_index: int) -> tuple[int, int]:
    normalized_num_shards = int(num_shards)
    normalized_shard_index = int(shard_index)
    if normalized_num_shards <= 0:
        raise ValueError(f"--num-shards must be >= 1, got {normalized_num_shards}")
    if normalized_shard_index < 0 or normalized_shard_index >= normalized_num_shards:
        raise ValueError(
            f"--shard-index must be in [0, {normalized_num_shards - 1}], got {normalized_shard_index}"
        )
    return normalized_num_shards, normalized_shard_index


def _select_examples_for_shard(
    examples: List[Dict[str, Any]],
    *,
    num_shards: int,
    shard_index: int,
) -> List[Dict[str, Any]]:
    if int(num_shards) <= 1:
        return list(examples)
    total_examples = len(examples)
    start = (total_examples * int(shard_index)) // int(num_shards)
    end = (total_examples * (int(shard_index) + 1)) // int(num_shards)
    return list(examples[start:end])


def _shard_file_suffix(*, num_shards: int, shard_index: int) -> str:
    if int(num_shards) <= 1:
        return ""
    return f".shard-{int(shard_index)}-of-{int(num_shards)}"


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    num_shards, shard_index = _normalize_shard_args(num_shards=args.num_shards, shard_index=args.shard_index)
    runtime = distributed_runtime_from_env()
    prepared_data_path = Path(args.prepared_data)
    output_dir = Path(args.output_dir) if args.output_dir else default_sft_tensor_cache_dir(prepared_data_path)
    runtime_log(
        (
            f"SFT tensor cache startup: prepared_data={prepared_data_path} "
            f"output_dir={output_dir} model_path={args.model_path} "
            f"include_splits={parse_include_splits(args.include_splits) or 'all'} "
            f"shard={shard_index + 1}/{num_shards}"
        ),
        runtime=runtime,
        main_process_only=True,
    )

    examples = _load_prepared_examples(
        prepared_data_path,
        skip_invalid_lines=args.skip_invalid_jsonl_lines,
        include_splits=args.include_splits,
    )
    if args.max_examples > 0:
        examples = examples[: int(args.max_examples)]
    if not examples:
        raise ValueError("No prepared SFT examples were loaded.")
    total_examples = len(examples)
    examples = _select_examples_for_shard(examples, num_shards=num_shards, shard_index=shard_index)

    processor = _load_processor(args.model_path)
    processor_signature = build_processor_signature(processor)
    metadata = build_sft_tensor_cache_metadata(
        model_path=args.model_path,
        processor_signature=processor_signature,
        processor_signature_summary=build_processor_signature_summary(processor),
        max_image_side=args.max_image_side,
        max_image_pixels=args.max_image_pixels,
        keep_recent_tool_image_messages=args.keep_recent_tool_image_messages,
        max_total_images=args.max_total_images,
        max_seq_length=args.max_seq_length,
        keep_recent_text_messages=args.keep_recent_text_messages,
        prepared_data_path=prepared_data_path,
        num_examples=total_examples,
    )
    metadata_path = output_dir / "metadata.json"
    existing_metadata = _load_existing_metadata(metadata_path)
    if existing_metadata:
        existing_config = {
            "schema_version": str(existing_metadata.get("schema_version") or ""),
            "cache_config": resolve_sft_tensor_cache_config_from_metadata(existing_metadata),
        }
        expected_config = {
            "schema_version": str(metadata.get("schema_version") or ""),
            "cache_config": dict(metadata.get("cache_config") or {}),
        }
        if existing_config != expected_config:
            raise ValueError(
                "Existing tensor cache metadata is incompatible with the requested preprocessing config. "
                f"metadata_path={metadata_path} expected={json.dumps(expected_config, ensure_ascii=False, sort_keys=True)} "
                f"actual={json.dumps(existing_config, ensure_ascii=False, sort_keys=True)}"
            )
    resolver = _FrameReferenceResolver(max_cached_videos=args.frame_cache_max_cached_videos)

    manifest_rows: List[Dict[str, Any]] = []
    built_count = 0
    skipped_existing = 0
    total_bytes = 0
    shard_examples = len(examples)
    for idx, example in enumerate(examples, start=1):
        cache_key = build_sft_tensor_cache_key(example)
        entry_path = sft_tensor_cache_entry_path(output_dir, cache_key)
        entry_path.parent.mkdir(parents=True, exist_ok=True)
        if entry_path.exists() and not args.overwrite_existing:
            skipped_existing += 1
        else:
            materialized_example = materialize_example_for_training(example, resolver=resolver)
            payload = build_sft_tensor_cache_payload(
                processor,
                materialized_example,
                max_image_side=args.max_image_side,
                max_image_pixels=args.max_image_pixels,
                keep_recent_tool_image_messages=args.keep_recent_tool_image_messages,
                max_total_images=args.max_total_images,
                max_seq_length=args.max_seq_length,
                keep_recent_text_messages=args.keep_recent_text_messages,
            )
            torch.save(payload, entry_path)
            built_count += 1
        try:
            total_bytes += int(entry_path.stat().st_size)
        except Exception:
            pass

        manifest_rows.append(
            {
                "cache_key": cache_key,
                "cache_path": str(entry_path),
                "video_id": example.get("video_id"),
                "split": example.get("split"),
                "step_index": example.get("step_index"),
                "target_action": example.get("target_action"),
                "tool_name": example.get("tool_name"),
            }
        )
        if should_log_progress(idx, shard_examples, int(args.progress_every)):
            runtime_log(
                (
                    "SFT tensor cache progress: "
                    f"examples={idx}/{shard_examples} built={built_count} skipped_existing={skipped_existing} "
                    f"shard={shard_index + 1}/{num_shards}"
                ),
                runtime=runtime,
                main_process_only=True,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(metadata_path, metadata)
    shard_suffix = _shard_file_suffix(num_shards=num_shards, shard_index=shard_index)
    manifest_path = output_dir / f"manifest{shard_suffix}.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "output_dir": str(output_dir),
        "prepared_data_path": str(prepared_data_path),
        "num_examples": shard_examples,
        "num_examples_total": total_examples,
        "num_built": built_count,
        "num_skipped_existing": skipped_existing,
        "total_bytes": int(total_bytes),
        "num_shards": int(num_shards),
        "shard_index": int(shard_index),
        "metadata_path": str(metadata_path),
        "manifest_path": str(manifest_path),
    }
    _write_json(output_dir / f"summary{shard_suffix}.json", summary)
    if runtime.is_main_process:
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
