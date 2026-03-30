from __future__ import annotations

import copy
import json
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
    rollout_max_turns: int = 6
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
    if torch.cuda.is_available():
        return f"cuda:{int(runtime.local_rank)}"
    return "cpu"


def _load_verifier_runtime(
    *,
    eval_config: RolloutEvaluationConfig,
    runtime: DistributedRuntime,
) -> Any:
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


def run_rollout_evaluation(
    policy: Any,
    *,
    eval_config: RolloutEvaluationConfig,
    output_dir: str | Path,
    epoch_index: int,
    runtime: Optional[DistributedRuntime] = None,
) -> Optional[Dict[str, Any]]:
    runtime = runtime or distributed_runtime_from_env()
    shard_spec = resolve_shard_spec(runtime=runtime)
    saver_config = copy.deepcopy(eval_config.saver_config) if eval_config.saver_config is not None else SaverAgentConfig()
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
    scored_shard_dir = eval_root / "scored_shards"
    scored_shard_dir.mkdir(parents=True, exist_ok=True)

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
        _attach_proposal_context(item, proposal_runtime=proposal_runtime)
        _attach_verifier_context(
            item,
            eval_config=eval_config,
            verifier_runtime=verifier_runtime,
            verifier_device_map=verifier_kwargs["verifier_device_map"],
        )
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
    distributed_barrier(runtime)

    summary: Optional[Dict[str, Any]] = None
    if runtime.is_main_process:
        merged_scored_records, _ = load_rollout_records(scored_shard_dir)
        summary = summarize_saver_metrics(
            merged_scored_records,
            reference_data=reference_data,
            include_diagnostic_summary=bool(eval_config.attach_reference_diagnostics),
        )
        summary["epoch_index"] = int(epoch_index)
        summary["num_scored_records"] = len(merged_scored_records)
        metrics_path = eval_root / "metrics.json"
        metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        runtime_log(
            f"epoch {int(epoch_index)} rollout eval metrics saved to {metrics_path}",
            runtime=runtime,
            main_process_only=True,
        )
    distributed_barrier(runtime)
    return summary
