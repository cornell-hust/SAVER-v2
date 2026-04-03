#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from split_utils import parse_include_splits

from saver_agent.prepared_metadata import (
    ensure_prepared_sft_metadata,
    load_prepared_sft_metadata,
    prepared_sft_metadata_path,
)
from saver_agent.runtime import (
    distributed_barrier,
    distributed_runtime_from_env,
    init_torch_distributed,
    resolve_inference_device_map,
    resolve_shard_spec,
    runtime_log,
    sharded_output_path,
)
from saver_agent.teacher_judge import (
    QwenTeacherJudge,
    annotate_teacher_judge_examples,
    is_teacher_judge_candidate,
    reweight_teacher_judge_examples,
)
from saver_agent.training import _FrameReferenceResolver


def _load_tqdm():
    try:
        from tqdm.auto import tqdm
    except Exception:
        return None
    return tqdm


class _ProgressVisualizer:
    def __init__(self, *, runtime, enabled: bool = True):
        self.runtime = runtime
        self.enabled = bool(enabled)
        self._bars: Dict[str, Any] = {}
        self._tqdm = _load_tqdm() if self.enabled else None

    def _position_for_phase(self, phase: str) -> int:
        base = int(self.runtime.rank) * 2
        return base if phase == "scan" else base + 1

    def _desc_for_phase(self, phase: str) -> str:
        return f"rank{self.runtime.rank} scan" if phase == "scan" else f"rank{self.runtime.rank} judge"

    def _get_bar(self, phase: str, total: int):
        bar = self._bars.get(phase)
        if bar is None:
            if self._tqdm is None:
                return None
            bar = self._tqdm(
                total=max(0, int(total)),
                desc=self._desc_for_phase(phase),
                position=self._position_for_phase(phase),
                leave=True,
                dynamic_ncols=True,
            )
            self._bars[phase] = bar
            return bar
        if total and getattr(bar, "total", None) != int(total):
            try:
                bar.total = int(total)
            except Exception:
                pass
        return bar

    def __call__(self, payload: Dict[str, Any]) -> None:
        if self._tqdm is None:
            return
        phase = str(payload.get("phase") or "").strip().lower()
        if phase not in {"scan", "annotate"}:
            return
        total = max(0, int(payload.get("total") or 0))
        completed = max(0, int(payload.get("completed") or 0))
        bar = self._get_bar(phase, total)
        if bar is None:
            return
        current = max(0, int(getattr(bar, "n", 0)))
        delta = completed - current
        if delta > 0:
            bar.update(delta)
        postfix = {
            "cand": int(payload.get("candidate_examples") or 0),
            "annotated": int(payload.get("annotated_count") or 0),
            "skipped": int(payload.get("skipped_existing") or 0),
        }
        try:
            bar.set_postfix(postfix, refresh=False)
        except Exception:
            pass

    def close(self) -> None:
        for bar in self._bars.values():
            try:
                bar.close()
            except Exception:
                pass
        self._bars.clear()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-annotate prepared SFT verify_hypothesis examples with a Qwen teacher judge."
    )
    parser.add_argument("--input", required=True, help="Input prepared SFT JSONL path.")
    parser.add_argument("--output", required=True, help="Output JSONL path for annotated examples.")
    parser.add_argument("--include-splits", default="", help="Optional comma-separated split whitelist.")
    parser.add_argument("--skip-invalid-jsonl-lines", action="store_true", help="Skip malformed JSONL lines instead of failing.")
    parser.add_argument("--model-path", required=True, help="Local Qwen teacher judge model path.")
    parser.add_argument(
        "--input-mode",
        choices=["text_only", "multimodal_visual", "auto"],
        default="auto",
        help="Teacher judge input mode.",
    )
    parser.add_argument("--torch-dtype", default="auto", help="Torch dtype for the teacher judge.")
    parser.add_argument("--device-map", default="auto", help="device_map for the teacher judge.")
    parser.add_argument("--attn-implementation", default="", help="Optional attention backend for the teacher judge.")
    parser.add_argument("--max-new-tokens", type=int, default=384, help="Generation length for the teacher judge.")
    parser.add_argument("--max-images", type=int, default=8, help="Maximum images passed to the teacher judge per example.")
    parser.add_argument(
        "--topk-frames-per-view",
        type=int,
        default=4,
        help="Maximum number of frames sampled into each teacher-judge view package.",
    )
    parser.add_argument("--frame-cache-max-cached-videos", type=int, default=64, help="How many frame_cache tensors/video readers to keep open while resolving image_ref payloads.")
    parser.add_argument("--overwrite-existing", action="store_true", help="Overwrite existing teacher judge labels.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Teacher-judge micro-batch size inside each shard process.",
    )
    parser.add_argument("--progress-every", type=int, default=25, help="Log annotation progress every N examples.")
    parser.add_argument("--no-progress-bar", action="store_true", help="Disable interactive tqdm progress bars.")
    parser.add_argument("--num-shards", type=int, default=0, help="Optional number of shard workers.")
    parser.add_argument("--shard-index", type=int, default=-1, help="Optional shard index for this process.")
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


def _resolve_teacher_judge_shard_indices(
    rows: List[Dict[str, Any]],
    *,
    num_shards: int,
) -> List[List[int]]:
    if int(num_shards) < 1:
        raise ValueError("num_shards must be at least 1.")
    shard_indices_by_shard: List[List[int]] = [[] for _ in range(int(num_shards))]
    verify_candidate_index = 0
    for row_index, row in enumerate(rows):
        if is_teacher_judge_candidate(row):
            assigned_shard = verify_candidate_index % int(num_shards)
            verify_candidate_index += 1
        else:
            assigned_shard = row_index % int(num_shards)
        shard_indices_by_shard[int(assigned_shard)].append(int(row_index))
    return shard_indices_by_shard


def _expected_shard_indices(
    *,
    total_rows: int,
    num_shards: int,
    shard_index: int,
    shard_indices_by_shard: Optional[List[List[int]]] = None,
) -> List[int]:
    if shard_indices_by_shard is not None:
        if not 0 <= int(shard_index) < len(shard_indices_by_shard):
            raise ValueError(
                f"shard_index={shard_index} is outside the provided shard mapping range "
                f"[0, {len(shard_indices_by_shard) - 1}]."
            )
        return list(shard_indices_by_shard[int(shard_index)])
    return list(range(int(shard_index), int(total_rows), int(num_shards)))


def _merge_sharded_outputs(
    output_path: str | Path,
    *,
    total_rows: int,
    num_shards: int,
    shard_indices_by_shard: Optional[List[List[int]]] = None,
) -> List[Dict[str, Any]]:
    base_output_path = Path(output_path)
    if int(num_shards) <= 1:
        return _load_jsonl(base_output_path)
    merged_rows: List[Optional[Dict[str, Any]]] = [None] * int(total_rows)
    for shard_index in range(int(num_shards)):
        shard_output_path = sharded_output_path(base_output_path, num_shards=int(num_shards), shard_index=shard_index)
        shard_rows = _load_jsonl(shard_output_path)
        expected_indices = _expected_shard_indices(
            total_rows=int(total_rows),
            num_shards=int(num_shards),
            shard_index=shard_index,
            shard_indices_by_shard=shard_indices_by_shard,
        )
        if len(shard_rows) != len(expected_indices):
            raise ValueError(
                "Shard output row count does not match expected shard partition size. "
                f"shard={shard_index + 1}/{num_shards} expected={len(expected_indices)} actual={len(shard_rows)} "
                f"path={shard_output_path}"
            )
        for source_index, row in zip(expected_indices, shard_rows):
            merged_rows[source_index] = row
    if any(row is None for row in merged_rows):
        raise ValueError("Merged teacher-judge output is incomplete after combining shard files.")
    finalized_rows = [row for row in merged_rows if row is not None]
    _write_jsonl(base_output_path, finalized_rows)
    return finalized_rows


def _wait_for_sharded_outputs(
    output_path: str | Path,
    *,
    total_rows: int,
    num_shards: int,
    shard_indices_by_shard: Optional[List[List[int]]] = None,
    timeout_sec: float = 1800.0,
    poll_interval_sec: float = 1.0,
) -> None:
    base_output_path = Path(output_path)
    deadline = time.time() + max(1.0, float(timeout_sec))
    shard_status: Dict[str, str] = {}
    while time.time() < deadline:
        all_ready = True
        shard_status.clear()
        for shard_index in range(int(num_shards)):
            shard_output_path = sharded_output_path(base_output_path, num_shards=int(num_shards), shard_index=shard_index)
            expected_indices = _expected_shard_indices(
                total_rows=int(total_rows),
                num_shards=int(num_shards),
                shard_index=shard_index,
                shard_indices_by_shard=shard_indices_by_shard,
            )
            expected_count = len(expected_indices)
            if not shard_output_path.exists():
                all_ready = False
                shard_status[str(shard_output_path)] = "missing"
                continue
            try:
                shard_rows = _load_jsonl(shard_output_path)
            except Exception as exc:
                all_ready = False
                shard_status[str(shard_output_path)] = f"unreadable: {exc}"
                continue
            if len(shard_rows) != expected_count:
                all_ready = False
                shard_status[str(shard_output_path)] = f"rows={len(shard_rows)}/{expected_count}"
                continue
            shard_status[str(shard_output_path)] = "ready"
        if all_ready:
            return
        time.sleep(max(0.05, float(poll_interval_sec)))
    raise TimeoutError(
        "Timed out while waiting for sharded teacher-judge outputs to become ready: "
        + json.dumps(shard_status, ensure_ascii=False)
    )


def _build_image_resolver(args: argparse.Namespace):
    if str(args.input_mode or "").strip().lower() == "text_only":
        return None
    resolver = _FrameReferenceResolver(max_cached_videos=args.frame_cache_max_cached_videos)
    return resolver._resolve_image_ref


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    runtime = distributed_runtime_from_env()
    dist_initialized = init_torch_distributed(runtime)
    shard_spec = resolve_shard_spec(num_shards=args.num_shards, shard_index=args.shard_index, runtime=runtime)
    output_path = sharded_output_path(args.output, num_shards=shard_spec.num_shards, shard_index=shard_spec.shard_index)
    input_metadata = ensure_prepared_sft_metadata(args.input)
    rows = _load_jsonl(
        args.input,
        skip_invalid_lines=args.skip_invalid_jsonl_lines,
        include_splits=args.include_splits,
    )
    shard_indices_by_shard = _resolve_teacher_judge_shard_indices(rows, num_shards=shard_spec.num_shards)
    local_row_indices = _expected_shard_indices(
        total_rows=len(rows),
        num_shards=shard_spec.num_shards,
        shard_index=shard_spec.shard_index,
        shard_indices_by_shard=shard_indices_by_shard,
    )
    local_rows = [rows[row_index] for row_index in local_row_indices]
    effective_device_map = resolve_inference_device_map(args.device_map, runtime=runtime)
    runtime_log(
        (
            f"teacher judge startup: total_examples={len(rows)} local_examples={len(local_rows)} "
            f"input_mode={args.input_mode} batch_size={args.batch_size} "
            f"model_path={args.model_path} output={output_path}"
        ),
        runtime=runtime,
    )
    progress_visualizer = _ProgressVisualizer(
        runtime=runtime,
        enabled=(not args.no_progress_bar) and sys.stderr.isatty(),
    )
    try:
        judge = QwenTeacherJudge.from_pretrained(
            args.model_path,
            torch_dtype=args.torch_dtype,
            device_map=effective_device_map,
            attn_implementation=args.attn_implementation or None,
            input_mode=args.input_mode,
            max_new_tokens=args.max_new_tokens,
            max_images=args.max_images,
            topk_frames_per_view=args.topk_frames_per_view,
            image_resolver=_build_image_resolver(args),
        )
        annotated_rows, summary = annotate_teacher_judge_examples(
            local_rows,
            judge=judge,
            input_mode=args.input_mode,
            batch_size=args.batch_size,
            overwrite_existing=args.overwrite_existing,
            progress_every=args.progress_every,
            log_fn=lambda message: runtime_log(message, runtime=runtime),
            progress_callback=progress_visualizer,
        )
        annotated_rows, reweight_summary = reweight_teacher_judge_examples(annotated_rows)
        _write_jsonl(output_path, annotated_rows)
        runtime_log(f"saved {len(annotated_rows)} teacher-annotated examples to {output_path}", runtime=runtime)
        if not shard_spec.is_sharded:
            metadata_path = prepared_sft_metadata_path(output_path)
            metadata = dict(input_metadata)
            metadata["teacher_annotated"] = True
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        merged_output_path: Optional[Path] = None
        if shard_spec.is_sharded and runtime.is_distributed and dist_initialized:
            if runtime.is_main_process:
                _wait_for_sharded_outputs(
                    args.output,
                    total_rows=len(rows),
                    num_shards=shard_spec.num_shards,
                    shard_indices_by_shard=shard_indices_by_shard,
                )
                merged_rows = _merge_sharded_outputs(
                    args.output,
                    total_rows=len(rows),
                    num_shards=shard_spec.num_shards,
                    shard_indices_by_shard=shard_indices_by_shard,
                )
                merged_output_path = Path(args.output)
                runtime_log(
                    f"merged {len(merged_rows)} sharded teacher-judge outputs into {merged_output_path}",
                    runtime=runtime,
                )
                merged_metadata = dict(load_prepared_sft_metadata(args.input))
                merged_metadata["teacher_annotated"] = True
                prepared_sft_metadata_path(merged_output_path).write_text(
                    json.dumps(merged_metadata, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
    finally:
        progress_visualizer.close()
    if runtime.is_main_process:
        print(
            json.dumps(
                {
                    "input": str(args.input),
                    "output": str(merged_output_path or output_path),
                    "local_output": str(output_path),
                    "num_examples": len(local_rows),
                    **summary,
                    **reweight_summary,
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
