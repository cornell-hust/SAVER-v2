from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class DistributedRuntime:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


@dataclass(frozen=True)
class ShardSpec:
    num_shards: int = 1
    shard_index: int = 0

    @property
    def is_sharded(self) -> bool:
        return self.num_shards > 1


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def distributed_runtime_from_env(env: Optional[Mapping[str, str]] = None) -> DistributedRuntime:
    env = env or os.environ
    world_size = max(1, _safe_int(env.get("WORLD_SIZE", 1), 1))
    rank = _safe_int(env.get("RANK", 0), 0)
    local_rank = _safe_int(env.get("LOCAL_RANK", rank), rank)
    if rank < 0:
        rank = 0
    if rank >= world_size:
        rank = 0
    if local_rank < 0:
        local_rank = 0
    return DistributedRuntime(rank=rank, world_size=world_size, local_rank=local_rank)


def resolve_shard_spec(
    *,
    num_shards: int = 0,
    shard_index: int = -1,
    runtime: Optional[DistributedRuntime] = None,
) -> ShardSpec:
    runtime = runtime or distributed_runtime_from_env()
    resolved_num_shards = int(num_shards)
    resolved_shard_index = int(shard_index)

    if resolved_num_shards < 0:
        raise ValueError("num_shards must be non-negative.")
    if resolved_num_shards == 0 and resolved_shard_index >= 0:
        raise ValueError("shard_index requires num_shards to be provided.")

    if resolved_num_shards == 0:
        if runtime.is_distributed:
            resolved_num_shards = int(runtime.world_size)
            resolved_shard_index = int(runtime.rank)
        else:
            resolved_num_shards = 1
            resolved_shard_index = 0
    elif resolved_shard_index < 0:
        resolved_shard_index = int(runtime.rank) if runtime.is_distributed else 0

    if resolved_num_shards < 1:
        raise ValueError("num_shards must be at least 1 after resolution.")
    if not 0 <= resolved_shard_index < resolved_num_shards:
        raise ValueError(
            f"Resolved shard_index={resolved_shard_index} is outside [0, {resolved_num_shards - 1}]."
        )
    return ShardSpec(num_shards=resolved_num_shards, shard_index=resolved_shard_index)


def shard_sequence(values: Sequence[T], *, num_shards: int, shard_index: int) -> list[T]:
    if num_shards < 1:
        raise ValueError("num_shards must be at least 1.")
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"shard_index={shard_index} must satisfy 0 <= shard_index < {num_shards}.")
    return [value for index, value in enumerate(values) if index % num_shards == shard_index]


def sharded_output_path(output_path: str | Path, *, num_shards: int, shard_index: int) -> Path:
    path = Path(output_path)
    if num_shards <= 1:
        return path
    shard_tag = f"shard{int(shard_index):02d}-of-{int(num_shards):02d}"
    if path.suffix:
        return path.with_name(f"{path.stem}.{shard_tag}{path.suffix}")
    return path / shard_tag


def resolve_inference_device_map(device_map: Any, *, runtime: Optional[DistributedRuntime] = None) -> Any:
    runtime = runtime or distributed_runtime_from_env()
    if runtime.is_distributed and device_map == "auto":
        try:
            import torch
        except Exception:
            return {"": int(runtime.local_rank)}
        if not torch.cuda.is_available():
            return {"": int(runtime.local_rank)}
        try:
            visible_cuda_devices = int(torch.cuda.device_count())
        except Exception:
            return {"": int(runtime.local_rank)}
        if visible_cuda_devices <= 0:
            return {"": int(runtime.local_rank)}
        local_rank = int(runtime.local_rank)
        if 0 <= local_rank < visible_cuda_devices:
            return {"": local_rank}
        return {"": 0}
    return device_map


def runtime_prefix(runtime: Optional[DistributedRuntime] = None) -> str:
    runtime = runtime or distributed_runtime_from_env()
    if runtime.is_distributed:
        return f"[rank {runtime.rank}/{runtime.world_size}]"
    return "[main]"


def runtime_log(
    message: str,
    *,
    runtime: Optional[DistributedRuntime] = None,
    main_process_only: bool = False,
) -> None:
    runtime = runtime or distributed_runtime_from_env()
    if main_process_only and not runtime.is_main_process:
        return
    print(f"{runtime_prefix(runtime)} {message}", flush=True)


def should_log_progress(completed: int, total: int, every: int) -> bool:
    if total <= 0:
        return False
    if completed <= 0:
        return False
    if completed == 1 or completed == total:
        return True
    if every <= 0:
        return False
    return completed % every == 0


def init_torch_distributed(runtime: Optional[DistributedRuntime] = None) -> bool:
    runtime = runtime or distributed_runtime_from_env()
    if not runtime.is_distributed:
        return False
    try:
        import torch
    except Exception:
        return False
    if not torch.distributed.is_available():
        return False
    if torch.distributed.is_initialized():
        return True
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(int(runtime.local_rank))
    torch.distributed.init_process_group(backend=backend, init_method="env://")
    return True


def distributed_barrier(runtime: Optional[DistributedRuntime] = None) -> None:
    runtime = runtime or distributed_runtime_from_env()
    if not runtime.is_distributed:
        return
    try:
        import torch
    except Exception:
        return
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
