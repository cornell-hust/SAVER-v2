from __future__ import annotations

import copy
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from run_saver_rollout import _serialize_result
from saver_agent.adapter import TimeSearchRolloutAdapter
from saver_agent.config import SaverAgentConfig
from saver_agent.dataset import SaverAgentDataset
from saver_agent.metrics import summarize_saver_metrics
from saver_agent.offline_scoring import (
    ReferenceDataProvider,
    load_rollout_records,
    save_rollout_records,
    score_rollout_records,
)
from saver_agent.proposal import SiglipFeatureEncoder
from saver_agent.qwen_verifier import DEFAULT_VERIFIER_MODEL_PATH, QwenSelfVerifier
from saver_agent.rollout import SaverRolloutRunner
from saver_agent.runtime import (
    DistributedRuntime,
    distributed_barrier,
    distributed_runtime_from_env,
    init_torch_distributed,
    resolve_inference_device_map,
    resolve_shard_spec,
    runtime_log,
    shard_sequence,
    should_log_progress,
)


@dataclass
class RolloutEvaluationConfig:
    data_path: str | Path
    data_root: str | Path = ""
    include_splits: Optional[Sequence[str] | str] = None
    max_records: int = 0
    inline_rollout_eval: bool = False
    rollout_max_turns: int = 14
    policy_max_new_tokens: int = 256
    max_total_images: int = 0
    max_image_side: int = 0
    max_image_pixels: int = 0
    proposal_model_path: str | Path = ""
    proposal_torch_dtype: str = "auto"
    proposal_device: str = ""
    verifier_backend: str = "heuristic"
    verifier_model_path: str | Path = DEFAULT_VERIFIER_MODEL_PATH
    verifier_torch_dtype: str = "auto"
    verifier_device_map: Any = "auto"
    verifier_attn_implementation: str = ""
    verifier_max_new_tokens: int = 512
    verifier_hybrid_alpha: float = 0.7
    attach_reference_diagnostics: bool = False
    allow_legacy_verify_compatibility: bool = False
    progress_every: int = 1
    saver_config: Optional[SaverAgentConfig] = None


def _attach_verifier_context(
    item: Dict[str, Any],
    *,
    eval_config: RolloutEvaluationConfig,
    verifier_runtime: Any,
    verifier_device_map: Any,
) -> None:
    cache = item["multimodal_cache"]
    cache["verifier_backend"] = eval_config.verifier_backend
    cache["verifier_model_path"] = str(eval_config.verifier_model_path)
    cache["verifier_torch_dtype"] = eval_config.verifier_torch_dtype
    cache["verifier_device_map"] = verifier_device_map
    cache["verifier_attn_implementation"] = eval_config.verifier_attn_implementation
    cache["verifier_max_new_tokens"] = int(eval_config.verifier_max_new_tokens)
    cache["verifier_hybrid_alpha"] = float(eval_config.verifier_hybrid_alpha)
    if verifier_runtime is not None:
        cache["verifier_runtime"] = verifier_runtime


def _attach_proposal_context(
    item: Dict[str, Any],
    *,
    proposal_runtime: Any,
) -> None:
    if proposal_runtime is not None:
        item["multimodal_cache"]["proposal_runtime"] = proposal_runtime


def _attach_reference_free_eval_guard(
    item: Dict[str, Any],
    *,
    allow_legacy_verify_compatibility: bool,
) -> None:
    cache = item.setdefault("multimodal_cache", {})
    cache["disable_external_verifier_fallback"] = True
    cache["allow_legacy_verify_compatibility"] = bool(allow_legacy_verify_compatibility)
    cache.pop("allow_external_verifier_fallback", None)


def _resolve_proposal_device(
    explicit_device: str | None,
    *,
    runtime: DistributedRuntime,
) -> str:
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
            "eval proposal device fallback: "
            f"local_rank={local_rank} is outside visible_cuda_devices={visible_cuda_devices}; using cuda:0"
        ),
        runtime=runtime,
    )
    return "cuda:0"


def _load_verifier_runtime(
    *,
    eval_config: RolloutEvaluationConfig,
    runtime: DistributedRuntime,
) -> Any:
    if not bool(eval_config.attach_reference_diagnostics):
        return None
    if eval_config.verifier_backend not in {"qwen_self_verifier", "hybrid"}:
        return None
    resolved_device_map = resolve_inference_device_map(eval_config.verifier_device_map, runtime=runtime)
    runtime_log(
        f"loading eval verifier from {eval_config.verifier_model_path} with device_map={resolved_device_map}",
        runtime=runtime,
    )
    return QwenSelfVerifier.from_pretrained(
        eval_config.verifier_model_path,
        torch_dtype=eval_config.verifier_torch_dtype,
        device_map=resolved_device_map,
        attn_implementation=eval_config.verifier_attn_implementation or None,
        max_new_tokens=eval_config.verifier_max_new_tokens,
    )


def _load_proposal_runtime(
    *,
    eval_config: RolloutEvaluationConfig,
    runtime: DistributedRuntime,
) -> Any:
    if not str(eval_config.proposal_model_path or "").strip():
        return None
    resolved_device = _resolve_proposal_device(eval_config.proposal_device, runtime=runtime)
    runtime_log(
        f"loading eval proposal model from {eval_config.proposal_model_path} on device={resolved_device}",
        runtime=runtime,
    )
    return SiglipFeatureEncoder.from_pretrained(
        str(eval_config.proposal_model_path),
        torch_dtype=eval_config.proposal_torch_dtype,
        device=resolved_device,
    )


def _cleanup_cuda_cache(*, runtime: DistributedRuntime, reason: str) -> None:
    gc.collect()
    try:
        import torch
    except Exception:
        return
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
        runtime_log(
            f"rollout eval memory cleanup: {reason}",
            runtime=runtime,
        )
    except Exception:
        return


def _clear_stale_scored_shards(scored_shard_dir: Path) -> int:
    removed = 0
    for pattern in ("*.json", "*.jsonl"):
        for shard_path in scored_shard_dir.glob(pattern):
            if not shard_path.is_file():
                continue
            shard_path.unlink()
            removed += 1
    return removed


def _clear_rollout_eval_sync_files(*, eval_root: Path) -> int:
    removed = 0
    for path in (eval_root / "metrics.json", eval_root / "failure.json"):
        if path.exists():
            path.unlink()
            removed += 1
    return removed


def _expected_scored_shard_paths(
    *,
    scored_shard_dir: Path,
    runtime: DistributedRuntime,
) -> list[Path]:
    return [
        scored_shard_dir / f"part.rank{rank:02d}-of-{runtime.world_size:02d}.jsonl"
        for rank in range(int(runtime.world_size))
    ]


def _write_rollout_eval_failure_marker(*, failure_path: Path, exc: Exception) -> None:
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    failure_payload = {
        "error_type": exc.__class__.__name__,
        "error": str(exc),
    }
    failure_path.write_text(json.dumps(failure_payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _wait_for_current_scored_records(
    *,
    scored_shard_dir: Path,
    runtime: DistributedRuntime,
    timeout_sec: float = 1800.0,
    poll_interval_sec: float = 1.0,
) -> list[Dict[str, Any]]:
    deadline = time.time() + max(1.0, float(timeout_sec))
    last_log_time = 0.0
    shard_status: dict[str, str] = {}
    expected_paths = _expected_scored_shard_paths(scored_shard_dir=scored_shard_dir, runtime=runtime)
    while time.time() < deadline:
        shard_status.clear()
        merged_records: list[Dict[str, Any]] = []
        all_ready = True
        for shard_path in expected_paths:
            if not shard_path.exists():
                all_ready = False
                shard_status[str(shard_path)] = "missing"
                continue
            try:
                shard_records, _ = load_rollout_records(shard_path)
            except Exception as exc:
                all_ready = False
                shard_status[str(shard_path)] = f"unreadable: {exc}"
                continue
            merged_records.extend(shard_records)
            shard_status[str(shard_path)] = f"ready:{len(shard_records)}"
        if all_ready:
            return merged_records
        now = time.time()
        if now - last_log_time >= 30.0:
            runtime_log(
                "waiting for rollout-eval scored shards: " + json.dumps(shard_status, ensure_ascii=False),
                runtime=runtime,
                main_process_only=True,
            )
            last_log_time = now
        time.sleep(max(0.05, float(poll_interval_sec)))
    raise TimeoutError(
        "Timed out while waiting for rollout-eval scored shard outputs: "
        + json.dumps(shard_status, ensure_ascii=False)
    )


def _load_current_scored_records(
    *,
    scored_shard_dir: Path,
    runtime: DistributedRuntime,
) -> list[Dict[str, Any]]:
    expected_paths = _expected_scored_shard_paths(scored_shard_dir=scored_shard_dir, runtime=runtime)
    missing_paths = [str(path) for path in expected_paths if not path.exists()]
    if missing_paths:
        raise RuntimeError(
            "rollout eval is missing scored shard outputs for the current distributed run: "
            + ", ".join(missing_paths)
        )
    merged_records: list[Dict[str, Any]] = []
    for shard_path in expected_paths:
        shard_records, _ = load_rollout_records(shard_path)
        merged_records.extend(shard_records)
    return merged_records


def _wait_for_rollout_eval_completion(
    *,
    metrics_path: Path,
    failure_path: Path,
    runtime: DistributedRuntime,
    timeout_sec: float = 1800.0,
    poll_interval_sec: float = 1.0,
) -> None:
    deadline = time.time() + max(1.0, float(timeout_sec))
    last_log_time = 0.0
    while time.time() < deadline:
        if metrics_path.exists():
            return
        if failure_path.exists():
            try:
                payload = json.loads(failure_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {"error": failure_path.read_text(encoding="utf-8")}
            raise RuntimeError(
                "rollout eval failed on the main process: " + json.dumps(payload, ensure_ascii=False)
            )
        now = time.time()
        if now - last_log_time >= 30.0:
            runtime_log(
                f"waiting for rollout-eval completion marker at {metrics_path}",
                runtime=runtime,
            )
            last_log_time = now
        time.sleep(max(0.05, float(poll_interval_sec)))
    raise TimeoutError(f"Timed out while waiting for rollout-eval completion marker at {metrics_path}")


def run_rollout_evaluation(
    policy: Any,
    *,
    eval_config: RolloutEvaluationConfig,
    output_dir: str | Path,
    epoch_index: int,
    runtime: Optional[DistributedRuntime] = None,
) -> Optional[Dict[str, Any]]:
    runtime = runtime or distributed_runtime_from_env()
    init_torch_distributed(runtime)
    shard_spec = resolve_shard_spec(runtime=runtime)
    saver_config = copy.deepcopy(eval_config.saver_config) if eval_config.saver_config is not None else SaverAgentConfig()
    saver_config.rollout_trace.record_message_history = False
    saver_config.rollout_trace.record_observation_content = False
    saver_config.rollout_trace.record_state_deltas = False
    saver_config.rollout_trace.record_counterfactual_trace = False
    runtime_log(
        (
            "rollout eval policy budget: "
            f"max_new_tokens_per_turn={int(eval_config.policy_max_new_tokens)} "
            f"max_total_images={int(eval_config.max_total_images) if int(eval_config.max_total_images) > 0 else 'all'} "
            f"max_image_side={int(eval_config.max_image_side) or 'off'} "
            f"max_image_pixels={int(eval_config.max_image_pixels) or 'off'}"
        ),
        runtime=runtime,
        main_process_only=True,
    )
    dataset = SaverAgentDataset(
        eval_config.data_path,
        data_root=eval_config.data_root,
        config=saver_config,
        include_splits=eval_config.include_splits,
    )
    if hasattr(dataset, "format_frame_cache_status"):
        runtime_log(
            dataset.format_frame_cache_status(prefix="rollout eval frame cache"),
            runtime=runtime,
            main_process_only=True,
        )
    all_indices = list(range(len(dataset)))
    if int(eval_config.max_records) > 0:
        all_indices = all_indices[: int(eval_config.max_records)]
    local_indices = shard_sequence(all_indices, num_shards=shard_spec.num_shards, shard_index=shard_spec.shard_index)

    eval_root = Path(output_dir) / "rollout_eval" / f"epoch_{int(epoch_index):03d}"
    metrics_path = eval_root / "metrics.json"
    failure_path = eval_root / "failure.json"
    scored_shard_dir = eval_root / "scored_shards"
    scored_shard_dir.mkdir(parents=True, exist_ok=True)
    if runtime.is_main_process:
        removed_shards = _clear_stale_scored_shards(scored_shard_dir)
        removed_sync_files = _clear_rollout_eval_sync_files(eval_root=eval_root)
        if removed_shards > 0:
            runtime_log(
                f"cleared {removed_shards} stale rollout-eval scored shard files from {scored_shard_dir}",
                runtime=runtime,
                main_process_only=True,
            )
        if removed_sync_files > 0:
            runtime_log(
                f"cleared {removed_sync_files} stale rollout-eval sync files from {eval_root}",
                runtime=runtime,
                main_process_only=True,
            )
    distributed_barrier(runtime)

    proposal_runtime = None
    verifier_runtime = None
    reference_data = None
    summary: Optional[Dict[str, Any]] = None
    try:
        _cleanup_cuda_cache(runtime=runtime, reason="before loading rollout-eval auxiliary runtimes")
        proposal_runtime = _load_proposal_runtime(eval_config=eval_config, runtime=runtime) if local_indices else None
        verifier_runtime = _load_verifier_runtime(eval_config=eval_config, runtime=runtime) if local_indices else None
        verifier_kwargs = {
            "verifier_backend": eval_config.verifier_backend,
            "verifier_model_path": str(eval_config.verifier_model_path),
            "verifier_torch_dtype": eval_config.verifier_torch_dtype,
            "verifier_device_map": resolve_inference_device_map(eval_config.verifier_device_map, runtime=runtime),
            "verifier_attn_implementation": eval_config.verifier_attn_implementation,
            "verifier_max_new_tokens": int(eval_config.verifier_max_new_tokens),
            "verifier_hybrid_alpha": float(eval_config.verifier_hybrid_alpha),
        }
        if verifier_runtime is not None:
            verifier_kwargs["verifier_runtime"] = verifier_runtime

        runner = SaverRolloutRunner(
            adapter=TimeSearchRolloutAdapter(config=saver_config),
            max_turns=int(eval_config.rollout_max_turns),
            config=saver_config,
        )
        local_rollouts = []
        total_local = len(local_indices)
        for completed, dataset_index in enumerate(local_indices, start=1):
            item = dataset[int(dataset_index)]
            _attach_reference_free_eval_guard(
                item,
                allow_legacy_verify_compatibility=bool(
                    eval_config.attach_reference_diagnostics and eval_config.allow_legacy_verify_compatibility
                ),
            )
            _attach_proposal_context(item, proposal_runtime=proposal_runtime)
            result = runner.run_episode(item, policy)
            serialized = _serialize_result(result)
            serialized["dataset_index"] = int(dataset_index)
            local_rollouts.append(serialized)
            if should_log_progress(completed, total_local, int(eval_config.progress_every)):
                runtime_log(
                    f"rollout eval progress: {completed}/{total_local} video_id={serialized.get('video_id', '')}",
                    runtime=runtime,
                )

        reference_data = ReferenceDataProvider(data_path=eval_config.data_path, data_root=eval_config.data_root)
        local_scored_records = score_rollout_records(
            local_rollouts,
            reference_data=reference_data,
            verifier_backend=eval_config.verifier_backend,
            force_reverify=bool(eval_config.attach_reference_diagnostics),
            attach_reference_offline_verifier=bool(eval_config.attach_reference_diagnostics),
            verifier_kwargs=verifier_kwargs,
            progress_every=eval_config.progress_every,
            progress_label=f"epoch {int(epoch_index)} eval score progress",
            runtime=runtime,
        )
        local_scored_path = scored_shard_dir / f"part.rank{runtime.rank:02d}-of-{runtime.world_size:02d}.jsonl"
        save_rollout_records(local_scored_records, local_scored_path, metadata={"input_kind": "jsonl"})

        if runtime.is_main_process:
            try:
                merged_scored_records = _wait_for_current_scored_records(
                    scored_shard_dir=scored_shard_dir,
                    runtime=runtime,
                )
                summary = summarize_saver_metrics(
                    merged_scored_records,
                    reference_data=reference_data,
                    include_diagnostic_summary=bool(eval_config.attach_reference_diagnostics),
                )
                summary["epoch_index"] = int(epoch_index)
                summary["num_scored_records"] = len(merged_scored_records)
                metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
                runtime_log(
                    f"epoch {int(epoch_index)} rollout eval metrics saved to {metrics_path}",
                    runtime=runtime,
                    main_process_only=True,
                )
            except Exception as exc:
                _write_rollout_eval_failure_marker(failure_path=failure_path, exc=exc)
                raise
        elif runtime.is_distributed:
            _wait_for_rollout_eval_completion(
                metrics_path=metrics_path,
                failure_path=failure_path,
                runtime=runtime,
            )
        return summary
    finally:
        proposal_runtime = None
        verifier_runtime = None
        reference_data = None
        _cleanup_cuda_cache(runtime=runtime, reason="after releasing rollout-eval auxiliary runtimes")
