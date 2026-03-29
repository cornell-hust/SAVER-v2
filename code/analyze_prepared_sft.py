#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import json
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from split_utils import parse_include_splits


@dataclass(frozen=True)
class _Runtime:
    rank: int = 0
    world_size: int = 1

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def distributed_runtime_from_env() -> _Runtime:
    world_size = max(1, _safe_int(os.environ.get("WORLD_SIZE", 1), 1))
    rank = _safe_int(os.environ.get("RANK", 0), 0)
    if not 0 <= rank < world_size:
        rank = 0
    return _Runtime(rank=rank, world_size=world_size)


def runtime_log(
    message: str,
    *,
    runtime: Optional[_Runtime] = None,
    main_process_only: bool = False,
) -> None:
    runtime = runtime or distributed_runtime_from_env()
    if main_process_only and not runtime.is_main_process:
        return
    prefix = f"[rank {runtime.rank}/{runtime.world_size}]" if runtime.is_distributed else "[main]"
    print(f"{prefix} {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze prepared SAVER SFT JSONL files and report image-count-heavy examples."
    )
    parser.add_argument("--input", required=True, help="Input prepared SFT JSONL path.")
    parser.add_argument(
        "--include-splits",
        default="",
        help="Optional comma-separated split whitelist, e.g. train or train,val.",
    )
    parser.add_argument("--top-k", type=int, default=20, help="How many heaviest examples to keep in the summary.")
    parser.add_argument(
        "--output",
        default="",
        help="Optional summary JSON output path. Prints to stdout if omitted.",
    )
    parser.add_argument(
        "--details-output",
        default="",
        help="Optional JSONL path to write one per-example detail record.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Log scan progress every N examples. First and last examples are always logged.",
    )
    return parser.parse_args()


def _quantile(sorted_values: Sequence[int], q: float) -> float:
    if not sorted_values:
        return 0.0
    if q <= 0:
        return float(sorted_values[0])
    if q >= 1:
        return float(sorted_values[-1])
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = q * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return float(sorted_values[lower]) * (1.0 - weight) + float(sorted_values[upper]) * weight


def _empty_role_counts() -> Dict[str, int]:
    return {
        "system": 0,
        "user": 0,
        "assistant": 0,
        "tool": 0,
        "other": 0,
    }


def _count_example_images(example: Dict[str, Any]) -> Dict[str, Any]:
    messages = example.get("messages") or []
    if not isinstance(messages, list):
        messages = []

    role_image_counts = _empty_role_counts()
    role_message_counts = _empty_role_counts()
    num_images = 0
    num_image_refs = 0
    num_inline_images = 0
    num_text_items = 0
    message_breakdown: List[Dict[str, Any]] = []

    for message_index, message in enumerate(messages):
        role = str(message.get("role") or "other")
        if role not in role_image_counts:
            role = "other"
        role_message_counts[role] += 1
        content = message.get("content") or []
        if not isinstance(content, list):
            content = []

        image_count = 0
        text_count = 0
        for item in content:
            item_type = item.get("type")
            if item_type == "image":
                image_count += 1
                num_images += 1
                if "image_ref" in item:
                    num_image_refs += 1
                if "image" in item:
                    num_inline_images += 1
            elif item_type == "text":
                text_count += 1
                num_text_items += 1

        role_image_counts[role] += image_count
        message_breakdown.append(
            {
                "message_index": message_index,
                "role": role,
                "num_images": image_count,
                "num_text_items": text_count,
            }
        )

    nonzero_message_breakdown = [item for item in message_breakdown if item["num_images"] > 0]
    return {
        "num_messages": len(messages),
        "num_images": num_images,
        "num_image_refs": num_image_refs,
        "num_inline_images": num_inline_images,
        "num_text_items": num_text_items,
        "role_image_counts": role_image_counts,
        "role_message_counts": role_message_counts,
        "num_tool_messages_with_images": sum(
            1 for item in nonzero_message_breakdown if item["role"] == "tool"
        ),
        "max_images_in_single_message": max((item["num_images"] for item in message_breakdown), default=0),
        "message_image_breakdown": nonzero_message_breakdown,
    }


def _iter_jsonl_rows(
    input_path: Path,
    *,
    include_splits: Optional[Sequence[str]],
) -> Iterable[Tuple[int, Dict[str, Any]]]:
    allowed = set(include_splits or [])
    with input_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if allowed and str(row.get("split") or "").strip() not in allowed:
                continue
            yield line_number, row


def _summarize_numeric(values: Sequence[int]) -> Dict[str, float]:
    if not values:
        return {
            "count": 0,
            "sum": 0.0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
    sorted_values = sorted(int(value) for value in values)
    return {
        "count": len(sorted_values),
        "sum": float(sum(sorted_values)),
        "min": float(sorted_values[0]),
        "max": float(sorted_values[-1]),
        "mean": float(statistics.fmean(sorted_values)),
        "median": float(statistics.median(sorted_values)),
        "p90": _quantile(sorted_values, 0.90),
        "p95": _quantile(sorted_values, 0.95),
        "p99": _quantile(sorted_values, 0.99),
    }


def _update_counter(counter: Dict[str, int], key: Any, increment: int = 1) -> None:
    normalized_key = str(key or "").strip() or "(none)"
    counter[normalized_key] = int(counter.get(normalized_key, 0)) + int(increment)


def _ranked_counts(counter: Dict[str, int], *, top_k: int = 20) -> List[Dict[str, Any]]:
    items = sorted(counter.items(), key=lambda item: (-int(item[1]), item[0]))
    return [
        {"key": key, "count": int(value)}
        for key, value in items[: max(1, int(top_k))]
    ]


def _ranked_video_stats(video_stats: Dict[str, Dict[str, Any]], *, top_k: int = 20) -> List[Dict[str, Any]]:
    ranked = sorted(
        video_stats.values(),
        key=lambda item: (
            -int(item["max_num_images"]),
            -float(item["mean_num_images"]),
            -int(item["num_examples"]),
            str(item["video_id"]),
        ),
    )
    return ranked[: max(1, int(top_k))]


def analyze_prepared_sft(
    *,
    input_path: Path,
    include_splits: Optional[Sequence[str]],
    top_k: int,
    details_output_path: Optional[Path] = None,
    progress_every: int = 0,
) -> Dict[str, Any]:
    runtime = distributed_runtime_from_env()
    details_handle = None
    if details_output_path is not None:
        details_output_path.parent.mkdir(parents=True, exist_ok=True)
        details_handle = details_output_path.open("w", encoding="utf-8")

    num_examples = 0
    num_examples_with_images = 0
    image_counts: List[int] = []
    message_counts: List[int] = []
    split_counts: Dict[str, int] = {}
    target_action_counts: Dict[str, int] = {}
    tool_name_counts: Dict[str, int] = {}
    video_stats: Dict[str, Dict[str, Any]] = {}
    top_examples_heap: List[Tuple[int, int, int, Dict[str, Any]]] = []

    try:
        for line_number, example in _iter_jsonl_rows(input_path, include_splits=include_splits):
            stats = _count_example_images(example)
            detail = {
                "line_number": line_number,
                "video_id": example.get("video_id"),
                "split": example.get("split"),
                "step_index": example.get("step_index"),
                "target_action": example.get("target_action"),
                "tool_name": example.get("tool_name"),
                **stats,
            }

            num_examples += 1
            image_counts.append(int(stats["num_images"]))
            message_counts.append(int(stats["num_messages"]))
            if int(stats["num_images"]) > 0:
                num_examples_with_images += 1

            _update_counter(split_counts, example.get("split"))
            _update_counter(target_action_counts, example.get("target_action"))
            _update_counter(tool_name_counts, example.get("tool_name"))

            video_id = str(example.get("video_id") or "")
            video_entry = video_stats.setdefault(
                video_id,
                {
                    "video_id": video_id,
                    "split": str(example.get("split") or ""),
                    "num_examples": 0,
                    "num_tool_examples": 0,
                    "num_answer_examples": 0,
                    "total_num_images": 0,
                    "max_num_images": 0,
                },
            )
            video_entry["num_examples"] += 1
            video_entry["num_tool_examples"] += int(example.get("target_action") == "tool_call")
            video_entry["num_answer_examples"] += int(example.get("target_action") == "answer")
            video_entry["total_num_images"] += int(stats["num_images"])
            video_entry["max_num_images"] = max(int(video_entry["max_num_images"]), int(stats["num_images"]))

            heap_item = (
                int(stats["num_images"]),
                int(stats["num_messages"]),
                int(line_number),
                detail,
            )
            if len(top_examples_heap) < max(1, int(top_k)):
                heapq.heappush(top_examples_heap, heap_item)
            else:
                heapq.heappushpop(top_examples_heap, heap_item)

            if details_handle is not None:
                details_handle.write(json.dumps(detail, ensure_ascii=False) + "\n")

            if num_examples == 1 or (int(progress_every) > 0 and num_examples % int(progress_every) == 0):
                runtime_log(
                    f"prepared SFT scan progress: examples={num_examples} current_max_images={max(image_counts or [0])}",
                    runtime=runtime,
                    main_process_only=True,
                )
    finally:
        if details_handle is not None:
            details_handle.close()

    for video_entry in video_stats.values():
        num_video_examples = max(1, int(video_entry["num_examples"]))
        video_entry["mean_num_images"] = float(video_entry["total_num_images"]) / float(num_video_examples)

    heaviest_examples = [
        item[3]
        for item in sorted(
            top_examples_heap,
            key=lambda item: (-int(item[0]), -int(item[1]), int(item[2])),
        )
    ]

    return {
        "input_path": str(input_path),
        "include_splits": list(include_splits or []),
        "num_examples": num_examples,
        "num_examples_with_images": num_examples_with_images,
        "image_count_stats": _summarize_numeric(image_counts),
        "message_count_stats": _summarize_numeric(message_counts),
        "split_counts": _ranked_counts(split_counts, top_k=max(10, int(top_k))),
        "target_action_counts": _ranked_counts(target_action_counts, top_k=max(10, int(top_k))),
        "tool_name_counts": _ranked_counts(tool_name_counts, top_k=max(10, int(top_k))),
        "top_videos_by_image_load": _ranked_video_stats(video_stats, top_k=top_k),
        "heaviest_examples": heaviest_examples,
    }


def main() -> None:
    args = parse_args()
    runtime = distributed_runtime_from_env()
    if runtime.is_distributed and not runtime.is_main_process:
        runtime_log("prepared SFT analysis runs on the main process only; skipping duplicate worker.", runtime=runtime)
        return

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input path does not exist: {input_path}")

    include_splits = parse_include_splits(args.include_splits)
    runtime_log(
        f"analyzing prepared SFT file {input_path} include_splits={include_splits or 'all'}",
        runtime=runtime,
    )
    summary = analyze_prepared_sft(
        input_path=input_path,
        include_splits=include_splits,
        top_k=args.top_k,
        details_output_path=Path(args.details_output) if args.details_output else None,
        progress_every=args.progress_every,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        runtime_log(f"wrote prepared SFT summary to {output_path}", runtime=runtime)
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
