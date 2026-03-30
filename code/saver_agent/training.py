from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from saver_agent.evaluation import RolloutEvaluationConfig, run_rollout_evaluation
from saver_agent.qwen_policy import _to_pil_image
from saver_agent.runtime import distributed_barrier, distributed_runtime_from_env, runtime_log, should_log_progress


DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

SFT_TENSOR_CACHE_SCHEMA_VERSION = "saver_agent.sft_tensor_cache.v1"


def _frame_cache_path(video_path: Path) -> Path:
    return Path(str(video_path) + ".frame_cache")


def _print_cache_warning(message: str) -> None:
    print(f"[cache-warning] {message}", flush=True)


def _append_jsonl(path: str | Path, payload: Dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _iter_image_ref_video_paths(messages: Sequence[Dict[str, Any]]) -> Iterable[Path]:
    for message in messages:
        for item in message.get("content", []):
            if item.get("type") != "image":
                continue
            image_ref = item.get("image_ref") or {}
            video_path = str(image_ref.get("video_path") or "")
            if video_path:
                yield Path(video_path)


def summarize_example_frame_cache_status(
    examples: Sequence[Dict[str, Any]],
    *,
    max_examples: int = 5,
) -> Dict[str, Any]:
    referenced_video_paths: List[Path] = []
    seen_video_paths: set[str] = set()
    for example in examples:
        for video_path in _iter_image_ref_video_paths(example.get("messages", [])):
            key = str(video_path)
            if key in seen_video_paths:
                continue
            seen_video_paths.add(key)
            referenced_video_paths.append(video_path)

    num_cached_videos = 0
    num_missing_frame_cache = 0
    num_missing_video_files = 0
    missing_examples: List[Dict[str, str]] = []
    for video_path in referenced_video_paths:
        cache_path = _frame_cache_path(video_path)
        if cache_path.exists():
            num_cached_videos += 1
        else:
            num_missing_frame_cache += 1
            if len(missing_examples) < max(0, int(max_examples)):
                missing_examples.append(
                    {
                        "video_path": str(video_path),
                        "cache_path": str(cache_path),
                    }
                )
        if not video_path.exists():
            num_missing_video_files += 1

    return {
        "num_examples": len(examples),
        "num_referenced_videos": len(referenced_video_paths),
        "num_cached_videos": num_cached_videos,
        "num_missing_frame_cache": num_missing_frame_cache,
        "num_missing_video_files": num_missing_video_files,
        "missing_examples": missing_examples,
    }


def format_example_frame_cache_status(
    summary: Dict[str, Any],
    *,
    prefix: str = "prepared frame cache",
) -> str:
    message = (
        f"{prefix}: cached={int(summary.get('num_cached_videos', 0))}/"
        f"{int(summary.get('num_referenced_videos', 0))} "
        f"missing_frame_cache={int(summary.get('num_missing_frame_cache', 0))} "
        f"missing_video_files={int(summary.get('num_missing_video_files', 0))}"
    )
    missing_examples = list(summary.get("missing_examples") or [])
    if missing_examples:
        preview = "; ".join(
            f"cache_path={item.get('cache_path') or ''} video_path={item.get('video_path') or ''}"
            for item in missing_examples
        )
        message += f" missing_examples=[{preview}]"
    return message


def default_sft_tensor_cache_dir(prepared_data_path: str | Path) -> Path:
    return Path(str(prepared_data_path) + ".tensor_cache")


def _hash_json_payload(payload: Any) -> str:
    encoded = json.dumps(
        _strip_private_fields_for_cache_key(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_to_dict(obj: Any) -> Dict[str, Any]:
    if obj is None or not hasattr(obj, "to_dict"):
        return {}
    try:
        payload = obj.to_dict()
    except Exception:
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def build_processor_signature_payload(processor: Any) -> Dict[str, Any]:
    tokenizer = getattr(processor, "tokenizer", None)
    image_processor = getattr(processor, "image_processor", None)

    tokenizer_payload: Dict[str, Any] = {
        "class_name": type(tokenizer).__name__ if tokenizer is not None else "",
        "init_kwargs": _safe_to_dict(tokenizer) or _strip_private_fields_for_cache_key(
            getattr(tokenizer, "init_kwargs", {}) or {}
        ),
        "special_tokens_map": _strip_private_fields_for_cache_key(
            getattr(tokenizer, "special_tokens_map", {}) or {}
        ),
        "chat_template": str(
            getattr(tokenizer, "chat_template", None)
            or getattr(processor, "chat_template", None)
            or ""
        ),
    }
    if tokenizer is not None:
        try:
            added_vocab = tokenizer.get_added_vocab()
        except Exception:
            added_vocab = {}
        tokenizer_payload["added_vocab"] = _strip_private_fields_for_cache_key(added_vocab)
        try:
            vocab = tokenizer.get_vocab()
        except Exception:
            vocab = {}
        tokenizer_payload["vocab_size"] = int(len(vocab) or getattr(tokenizer, "vocab_size", 0) or 0)
        tokenizer_payload["vocab_digest"] = _hash_json_payload(vocab) if vocab else ""

    image_processor_payload = {
        "class_name": type(image_processor).__name__ if image_processor is not None else "",
        "config": _safe_to_dict(image_processor),
    }
    processor_payload = {
        "class_name": type(processor).__name__,
        "config": _safe_to_dict(processor),
        "tokenizer": tokenizer_payload,
        "image_processor": image_processor_payload,
    }
    return _strip_private_fields_for_cache_key(processor_payload)


def build_processor_signature(processor: Any) -> str:
    return _hash_json_payload(build_processor_signature_payload(processor))


def load_processor_signature_from_model_path(model_path: str | Path) -> str:
    try:
        from transformers import AutoProcessor
    except Exception as exc:
        raise ImportError("Computing processor signatures requires the `transformers` package.") from exc
    processor = AutoProcessor.from_pretrained(str(model_path))
    return build_processor_signature(processor)


def build_processor_signature_summary(processor: Any) -> Dict[str, Any]:
    payload = build_processor_signature_payload(processor)
    tokenizer_payload = dict(payload.get("tokenizer") or {})
    image_processor_payload = dict(payload.get("image_processor") or {})
    return {
        "processor_class": str(payload.get("class_name") or ""),
        "tokenizer_class": str(tokenizer_payload.get("class_name") or ""),
        "image_processor_class": str(image_processor_payload.get("class_name") or ""),
        "vocab_size": int(tokenizer_payload.get("vocab_size") or 0),
        "signature": build_processor_signature(processor),
    }


def _strip_private_fields_for_cache_key(payload: Any) -> Any:
    if payload is None or isinstance(payload, (str, int, float, bool)):
        return payload
    if isinstance(payload, dict):
        return {
            str(key): _strip_private_fields_for_cache_key(value)
            for key, value in payload.items()
            if not str(key).startswith("_")
        }
    if isinstance(payload, list):
        return [_strip_private_fields_for_cache_key(value) for value in payload]
    if isinstance(payload, tuple):
        return [_strip_private_fields_for_cache_key(value) for value in payload]
    if isinstance(payload, (set, frozenset)):
        normalized_items = [_strip_private_fields_for_cache_key(value) for value in payload]
        return sorted(
            normalized_items,
            key=lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        )
    if isinstance(payload, Path):
        return str(payload)
    if isinstance(payload, bytes):
        try:
            return payload.decode("utf-8")
        except Exception:
            return {"__type__": "bytes", "hex": payload.hex()}
    if isinstance(payload, torch.Tensor):
        return {
            "__type__": "torch.Tensor",
            "shape": list(payload.shape),
            "dtype": str(payload.dtype),
        }
    if hasattr(payload, "item") and callable(getattr(payload, "item")):
        try:
            return _strip_private_fields_for_cache_key(payload.item())
        except Exception:
            pass
    if hasattr(payload, "to_dict") and callable(getattr(payload, "to_dict")):
        try:
            return _strip_private_fields_for_cache_key(payload.to_dict())
        except Exception:
            pass
    if hasattr(payload, "__getstate__") and callable(getattr(payload, "__getstate__")):
        try:
            state = payload.__getstate__()
            if state is not None:
                return {
                    "__type__": type(payload).__name__,
                    "state": _strip_private_fields_for_cache_key(state),
                }
        except Exception:
            pass
    if hasattr(payload, "__dict__"):
        try:
            return {
                "__type__": type(payload).__name__,
                "attrs": _strip_private_fields_for_cache_key(vars(payload)),
            }
        except Exception:
            pass
    if hasattr(payload, "size") and hasattr(payload, "mode"):
        try:
            width, height = payload.size
            return {
                "__type__": type(payload).__name__,
                "size": [int(width), int(height)],
                "mode": str(payload.mode),
            }
        except Exception:
            return {"__type__": type(payload).__name__}
    return {"__type__": type(payload).__name__, "repr": repr(payload)}


def build_sft_tensor_cache_key(example: Dict[str, Any]) -> str:
    normalized_payload = _strip_private_fields_for_cache_key(example)
    encoded = json.dumps(
        normalized_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_sft_tensor_cache_config(
    *,
    processor_signature: str,
    max_image_side: int = 0,
    max_image_pixels: int = 0,
    keep_recent_tool_image_messages: int = 0,
    max_total_images: int = 0,
    max_seq_length: int = 0,
    keep_recent_text_messages: int = 0,
) -> Dict[str, Any]:
    return {
        "processor_signature": str(processor_signature or ""),
        "max_image_side": int(max_image_side),
        "max_image_pixels": int(max_image_pixels),
        "keep_recent_tool_image_messages": int(keep_recent_tool_image_messages),
        "max_total_images": int(max_total_images),
        "max_seq_length": int(max_seq_length),
        "keep_recent_text_messages": int(keep_recent_text_messages),
    }


def build_sft_tensor_cache_metadata(
    *,
    model_path: str | Path,
    processor_signature: str,
    processor_signature_summary: Optional[Dict[str, Any]] = None,
    max_image_side: int = 0,
    max_image_pixels: int = 0,
    keep_recent_tool_image_messages: int = 0,
    max_total_images: int = 0,
    max_seq_length: int = 0,
    keep_recent_text_messages: int = 0,
    prepared_data_path: str | Path = "",
    num_examples: int = 0,
) -> Dict[str, Any]:
    return {
        "schema_version": SFT_TENSOR_CACHE_SCHEMA_VERSION,
        "cache_config": normalize_sft_tensor_cache_config(
            processor_signature=processor_signature,
            max_image_side=max_image_side,
            max_image_pixels=max_image_pixels,
            keep_recent_tool_image_messages=keep_recent_tool_image_messages,
            max_total_images=max_total_images,
            max_seq_length=max_seq_length,
            keep_recent_text_messages=keep_recent_text_messages,
        ),
        "model_path": str(model_path),
        "processor_signature_summary": dict(processor_signature_summary or {}),
        "prepared_data_path": str(prepared_data_path) if prepared_data_path else "",
        "num_examples": int(num_examples),
    }


def resolve_sft_tensor_cache_config_from_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    actual_config = dict(metadata.get("cache_config") or {})
    if str(actual_config.get("processor_signature") or "").strip():
        return actual_config

    legacy_model_path = str(actual_config.get("model_path") or metadata.get("model_path") or "").strip()
    if not legacy_model_path:
        return actual_config

    migrated_config = dict(actual_config)
    migrated_config.pop("model_path", None)
    migrated_config["processor_signature"] = load_processor_signature_from_model_path(legacy_model_path)
    return migrated_config


def sft_tensor_cache_entry_path(cache_dir: str | Path, cache_key: str) -> Path:
    normalized_key = str(cache_key)
    prefix = normalized_key[:2] if len(normalized_key) >= 2 else "xx"
    return Path(cache_dir) / "entries" / prefix / f"{normalized_key}.pt"


class PreparedSFTTensorCache:
    def __init__(
        self,
        cache_dir: str | Path,
        *,
        expected_config: Dict[str, Any],
        runtime=None,
    ):
        self.cache_dir = Path(cache_dir)
        self.expected_config = dict(expected_config)
        self.runtime = runtime or distributed_runtime_from_env()
        self.metadata_path = self.cache_dir / "metadata.json"
        self.enabled = False
        self.status = "disabled"
        self.message = ""
        self.metadata: Dict[str, Any] = {}
        self._logged_missing_keys: set[str] = set()

        if not self.cache_dir.exists():
            self.status = "missing_dir"
            self.message = (
                f"SFT tensor cache not found at {self.cache_dir}; "
                "falling back to on-the-fly multimodal preprocessing."
            )
            return
        if not self.metadata_path.exists():
            self.status = "missing_metadata"
            self.message = (
                f"SFT tensor cache metadata is missing at {self.metadata_path}; "
                "falling back to on-the-fly multimodal preprocessing."
            )
            return
        try:
            self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.status = "invalid_metadata"
            self.message = (
                f"Failed to read SFT tensor cache metadata from {self.metadata_path}: {exc}; "
                "falling back to on-the-fly multimodal preprocessing."
            )
            return
        if str(self.metadata.get("schema_version") or "") != SFT_TENSOR_CACHE_SCHEMA_VERSION:
            self.status = "schema_mismatch"
            self.message = (
                f"SFT tensor cache schema mismatch at {self.metadata_path}; "
                "falling back to on-the-fly multimodal preprocessing."
            )
            return

        try:
            actual_config = resolve_sft_tensor_cache_config_from_metadata(self.metadata)
        except Exception as exc:
            self.status = "legacy_upgrade_failed"
            self.message = (
                f"Failed to resolve SFT tensor cache config from {self.metadata_path}: {exc}; "
                "falling back to on-the-fly multimodal preprocessing."
            )
            return
        if actual_config != self.expected_config:
            self.status = "config_mismatch"
            self.message = (
                f"SFT tensor cache config mismatch at {self.metadata_path}; "
                f"expected={json.dumps(self.expected_config, ensure_ascii=False, sort_keys=True)} "
                f"actual={json.dumps(actual_config, ensure_ascii=False, sort_keys=True)}. "
                "Falling back to on-the-fly multimodal preprocessing."
            )
            return

        self.enabled = True
        self.status = "enabled"
        self.message = (
            f"SFT tensor cache enabled: dir={self.cache_dir} "
            f"num_examples={int(self.metadata.get('num_examples', 0) or 0)}"
        )

    def log_status(self) -> None:
        runtime_log(self.message, runtime=self.runtime, main_process_only=True)

    def entry_path(self, cache_key: str) -> Path:
        return sft_tensor_cache_entry_path(self.cache_dir, cache_key)

    def load(self, cache_key: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        path = self.entry_path(cache_key)
        if not path.exists():
            if cache_key not in self._logged_missing_keys and len(self._logged_missing_keys) < 10:
                self._logged_missing_keys.add(cache_key)
                runtime_log(
                    (
                        f"SFT tensor cache miss: key={cache_key} path={path}; "
                        "falling back to on-the-fly multimodal preprocessing for this example."
                    ),
                    runtime=self.runtime,
                    main_process_only=True,
                )
            return None
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid SFT tensor cache payload at {path}: expected dict.")
        return payload


def _unwrap_model(model: Any) -> Any:
    return getattr(model, "module", model)


def _resize_image_for_training(
    image: Any,
    *,
    max_image_side: int = 0,
    max_image_pixels: int = 0,
) -> Any:
    pil_image = _to_pil_image(image)
    if not hasattr(pil_image, "size"):
        return pil_image

    width, height = pil_image.size
    if width <= 0 or height <= 0:
        return pil_image

    scale = 1.0
    if int(max_image_side) > 0:
        current_max_side = max(width, height)
        if current_max_side > int(max_image_side):
            scale = min(scale, float(max_image_side) / float(current_max_side))
    if int(max_image_pixels) > 0:
        current_pixels = width * height
        if current_pixels > int(max_image_pixels):
            scale = min(scale, math.sqrt(float(max_image_pixels) / float(current_pixels)))

    if scale >= 0.999:
        return pil_image

    resized_width = max(28, int(round(width * scale)))
    resized_height = max(28, int(round(height * scale)))
    return pil_image.resize((resized_width, resized_height))


def _prune_stale_tool_images(
    messages: List[Dict[str, Any]],
    *,
    keep_recent_tool_image_messages: int = 0,
) -> List[Dict[str, Any]]:
    if int(keep_recent_tool_image_messages) <= 0:
        return copy.deepcopy(messages)

    prepared = copy.deepcopy(messages)
    image_tool_message_indices = [
        message_index
        for message_index, message in enumerate(prepared)
        if message.get("role") == "tool"
        and any(item.get("type") == "image" and "image" in item for item in message.get("content", []))
    ]
    keep_indices = set(image_tool_message_indices[-int(keep_recent_tool_image_messages) :])

    for message_index in image_tool_message_indices:
        if message_index in keep_indices:
            continue
        content = prepared[message_index].get("content", [])
        prepared[message_index]["content"] = [item for item in content if item.get("type") != "image"]
    return prepared


def _prune_stale_text_history(
    messages: List[Dict[str, Any]],
    *,
    keep_recent_text_messages: int = 0,
) -> List[Dict[str, Any]]:
    if int(keep_recent_text_messages) <= 0:
        return copy.deepcopy(messages)

    prepared = copy.deepcopy(messages)
    prefix_end = 0
    while prefix_end < len(prepared) and prepared[prefix_end].get("role") in {"system", "user"}:
        prefix_end += 1

    preserved_prefix = prepared[:prefix_end]
    history = prepared[prefix_end:]
    if len(history) <= int(keep_recent_text_messages):
        return preserved_prefix + history
    return preserved_prefix + history[-int(keep_recent_text_messages) :]


def _cap_total_images(
    messages: List[Dict[str, Any]],
    *,
    max_total_images: int = 0,
) -> List[Dict[str, Any]]:
    if int(max_total_images) <= 0:
        return messages

    image_positions: List[Tuple[int, int]] = []
    for message_index, message in enumerate(messages):
        for content_index, item in enumerate(message.get("content", [])):
            if item.get("type") == "image" and ("image" in item or "image_ref" in item):
                image_positions.append((message_index, content_index))

    overflow = len(image_positions) - int(max_total_images)
    if overflow <= 0:
        return messages

    image_positions.sort()
    removals_by_message: Dict[int, set[int]] = {}
    for message_index, content_index in image_positions[:overflow]:
        removals_by_message.setdefault(message_index, set()).add(content_index)

    for message_index, removals in removals_by_message.items():
        content = list(messages[message_index].get("content", []))
        messages[message_index]["content"] = [
            item for idx, item in enumerate(content) if idx not in removals
        ]
    return messages


def _prepare_messages(
    messages: List[Dict[str, Any]],
    *,
    max_image_side: int = 0,
    max_image_pixels: int = 0,
    keep_recent_tool_image_messages: int = 0,
    max_total_images: int = 0,
    keep_recent_text_messages: int = 0,
) -> List[Dict[str, Any]]:
    prepared = _prune_stale_text_history(
        messages,
        keep_recent_text_messages=keep_recent_text_messages,
    )
    prepared = _prune_stale_tool_images(
        prepared,
        keep_recent_tool_image_messages=keep_recent_tool_image_messages,
    )
    prepared = _cap_total_images(
        prepared,
        max_total_images=max_total_images,
    )
    for message in prepared:
        for item in message.get("content", []):
            if item.get("type") == "image" and "image" in item:
                item["image"] = _resize_image_for_training(
                    item["image"],
                    max_image_side=max_image_side,
                    max_image_pixels=max_image_pixels,
                )
            elif item.get("type") == "video" and "video" in item:
                video = item["video"]
                if isinstance(video, (list, tuple)):
                    item["video"] = [
                        _resize_image_for_training(
                            frame,
                            max_image_side=max_image_side,
                            max_image_pixels=max_image_pixels,
                        )
                        for frame in video
                    ]
                elif isinstance(video, torch.Tensor) and video.ndim == 4:
                    item["video"] = [
                        _resize_image_for_training(
                            frame,
                            max_image_side=max_image_side,
                            max_image_pixels=max_image_pixels,
                        )
                        for frame in video
                    ]
    return prepared


def _extract_vision_inputs(messages: List[Dict[str, Any]]) -> Tuple[List[Any], List[Any]]:
    image_inputs: List[Any] = []
    video_inputs: List[Any] = []
    for message in messages:
        for item in message.get("content", []):
            if item.get("type") == "image" and "image" in item:
                image_inputs.append(item["image"])
            elif item.get("type") == "video" and "video" in item:
                video_inputs.append(item["video"])
    return image_inputs, video_inputs


def _build_chat_text(processor: Any, messages: List[Dict[str, Any]], *, add_generation_prompt: bool) -> str:
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    except TypeError:
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


def _tokenize_chat(
    processor: Any,
    text: str,
    messages: List[Dict[str, Any]],
    *,
    max_length: int = 0,
    truncation_side: str = "left",
) -> Dict[str, torch.Tensor]:
    image_inputs, video_inputs = _extract_vision_inputs(messages)
    has_multimodal_inputs = bool(image_inputs or video_inputs)
    processor_kwargs: Dict[str, Any] = {
        "text": text,
        "padding": False,
        "return_tensors": "pt",
    }
    if int(max_length) > 0 and not has_multimodal_inputs:
        processor_kwargs["truncation"] = True
        processor_kwargs["max_length"] = int(max_length)
    if image_inputs:
        processor_kwargs["images"] = image_inputs
    if video_inputs:
        processor_kwargs["videos"] = video_inputs
    tokenizer = getattr(processor, "tokenizer", None)
    if (
        tokenizer is None
        or int(max_length) <= 0
        or not hasattr(tokenizer, "truncation_side")
        or has_multimodal_inputs
    ):
        return processor(**processor_kwargs)
    original_side = tokenizer.truncation_side
    tokenizer.truncation_side = str(truncation_side or "left")
    try:
        return processor(**processor_kwargs)
    finally:
        tokenizer.truncation_side = original_side


def _count_text_tokens(processor: Any, text: str) -> int:
    tokenizer = getattr(processor, "tokenizer", processor)
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
    )
    input_ids = encoded["input_ids"]
    if isinstance(input_ids, torch.Tensor):
        return int(input_ids.shape[-1])
    if input_ids and isinstance(input_ids[0], list):
        return int(len(input_ids[0]))
    return int(len(input_ids))


def _count_model_input_tokens(encoded_inputs: Dict[str, torch.Tensor]) -> int:
    input_ids = encoded_inputs["input_ids"]
    if isinstance(input_ids, torch.Tensor):
        return int(input_ids.shape[-1])
    if input_ids and isinstance(input_ids[0], list):
        return int(len(input_ids[0]))
    return int(len(input_ids))


def _has_multimodal_content(messages: List[Dict[str, Any]]) -> bool:
    for message in messages:
        for item in message.get("content", []):
            item_type = item.get("type")
            if item_type == "image" and ("image" in item or "image_ref" in item):
                return True
            if item_type == "video" and "video" in item:
                return True
    return False


def _drop_oldest_multimodal_item(messages: List[Dict[str, Any]]) -> bool:
    for message_index, message in enumerate(messages):
        content = list(message.get("content", []))
        for content_index, item in enumerate(content):
            item_type = item.get("type")
            if item_type == "image" and ("image" in item or "image_ref" in item):
                del content[content_index]
                if content:
                    messages[message_index]["content"] = content
                elif message.get("role") not in {"system", "user"}:
                    del messages[message_index]
                else:
                    messages[message_index]["content"] = []
                return True
            if item_type == "video" and "video" in item:
                del content[content_index]
                if content:
                    messages[message_index]["content"] = content
                elif message.get("role") not in {"system", "user"}:
                    del messages[message_index]
                else:
                    messages[message_index]["content"] = []
                return True
    return False


def _drop_oldest_history_message(messages: List[Dict[str, Any]]) -> bool:
    prefix_end = 0
    while prefix_end < len(messages) and messages[prefix_end].get("role") in {"system", "user"}:
        prefix_end += 1
    if prefix_end >= len(messages):
        return False
    del messages[prefix_end]
    return True


def _fit_messages_to_budget(
    processor: Any,
    prompt_messages: List[Dict[str, Any]],
    *,
    target_response: str,
    max_seq_length: int = 0,
    truncation_side: str = "left",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str, str, Dict[str, torch.Tensor]]:
    fitted_prompt_messages = copy.deepcopy(prompt_messages)
    target_message = {"role": "assistant", "content": [{"type": "text", "text": target_response}]}
    max_length = int(max_seq_length)

    for _ in range(512):
        full_messages = fitted_prompt_messages + [target_message]
        prompt_text = _build_chat_text(processor, fitted_prompt_messages, add_generation_prompt=True)
        full_text = _build_chat_text(processor, full_messages, add_generation_prompt=False)
        has_multimodal = _has_multimodal_content(full_messages)
        full_inputs = _tokenize_chat(
            processor,
            full_text,
            full_messages,
            max_length=max_length if max_length > 0 and not has_multimodal else 0,
            truncation_side=truncation_side,
        )
        if max_length <= 0 or _count_model_input_tokens(full_inputs) <= max_length:
            return fitted_prompt_messages, full_messages, prompt_text, full_text, full_inputs

        if _drop_oldest_multimodal_item(fitted_prompt_messages):
            continue
        if _drop_oldest_history_message(fitted_prompt_messages):
            continue

        if has_multimodal:
            raise ValueError(
                f"Unable to fit multimodal example within max_seq_length={max_length}. "
                "Increase the sequence budget or reduce retained multimodal context."
            )
        return fitted_prompt_messages, full_messages, prompt_text, full_text, full_inputs

    raise RuntimeError("Exceeded pruning attempts while fitting a multimodal example to the sequence budget.")


def _tag_messages_for_cache(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tagged_messages: List[Dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        tagged_message = dict(message)
        tagged_message["_cache_message_index"] = int(message_index)
        tagged_content: List[Dict[str, Any]] = []
        for content_index, item in enumerate(message.get("content", [])):
            tagged_item = dict(item)
            tagged_item["_cache_content_index"] = int(content_index)
            tagged_content.append(tagged_item)
        tagged_message["content"] = tagged_content
        tagged_messages.append(tagged_message)
    return tagged_messages


def _capture_message_plan(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    plan: List[Dict[str, Any]] = []
    for message in messages:
        message_index = message.get("_cache_message_index")
        if message_index is None:
            continue
        plan.append(
            {
                "message_index": int(message_index),
                "content_indices": [
                    int(item["_cache_content_index"])
                    for item in message.get("content", [])
                    if item.get("_cache_content_index") is not None
                ],
            }
        )
    return plan


def _apply_cached_message_plan(
    original_messages: List[Dict[str, Any]],
    plan: List[Dict[str, Any]],
    *,
    max_image_side: int = 0,
    max_image_pixels: int = 0,
) -> List[Dict[str, Any]]:
    rebuilt_messages: List[Dict[str, Any]] = []
    for entry in plan:
        message_index = int(entry.get("message_index", -1))
        if not 0 <= message_index < len(original_messages):
            continue
        source_message = original_messages[message_index]
        rebuilt_message = {
            key: copy.deepcopy(value)
            for key, value in source_message.items()
            if key != "content"
        }
        rebuilt_content: List[Dict[str, Any]] = []
        source_content = list(source_message.get("content", []))
        for content_index in entry.get("content_indices", []):
            if not 0 <= int(content_index) < len(source_content):
                continue
            item = copy.deepcopy(source_content[int(content_index)])
            if item.get("type") == "image" and "image" in item:
                item["image"] = _resize_image_for_training(
                    item["image"],
                    max_image_side=max_image_side,
                    max_image_pixels=max_image_pixels,
                )
            elif item.get("type") == "video" and "video" in item:
                video = item["video"]
                if isinstance(video, (list, tuple)):
                    item["video"] = [
                        _resize_image_for_training(
                            frame,
                            max_image_side=max_image_side,
                            max_image_pixels=max_image_pixels,
                        )
                        for frame in video
                    ]
                elif isinstance(video, torch.Tensor) and video.ndim == 4:
                    item["video"] = [
                        _resize_image_for_training(
                            frame,
                            max_image_side=max_image_side,
                            max_image_pixels=max_image_pixels,
                        )
                        for frame in video
                    ]
            rebuilt_content.append(item)
        rebuilt_message["content"] = rebuilt_content
        rebuilt_messages.append(rebuilt_message)
    return rebuilt_messages


def _build_response_labels(
    input_ids: torch.Tensor,
    *,
    response_token_count: int,
) -> torch.Tensor:
    labels = input_ids.clone()
    labels.fill_(-100)
    retained_response_tokens = min(max(int(response_token_count), 0), int(labels.shape[-1]))
    if retained_response_tokens <= 0:
        return labels
    labels[:, -retained_response_tokens:] = input_ids[:, -retained_response_tokens:]
    return labels


def materialize_example_for_training(
    example: Dict[str, Any],
    *,
    resolver: Optional["_FrameReferenceResolver"] = None,
) -> Dict[str, Any]:
    prepared_example = copy.deepcopy(example)
    prepared_example.setdefault("_feature_cache_key", build_sft_tensor_cache_key(example))
    active_resolver = resolver or _FrameReferenceResolver()
    prepared_example["messages"] = active_resolver.materialize_messages(prepared_example["messages"])
    return prepared_example


def materialize_example_messages(
    example: Dict[str, Any],
    *,
    resolver: Optional["_FrameReferenceResolver"] = None,
) -> Dict[str, Any]:
    return materialize_example_for_training(example, resolver=resolver)


def _build_batch_from_feature(
    processor: Any,
    feature: Dict[str, Any],
    *,
    max_image_side: int = 0,
    max_image_pixels: int = 0,
    keep_recent_tool_image_messages: int = 0,
    max_total_images: int = 0,
    max_seq_length: int = 0,
    keep_recent_text_messages: int = 0,
    cached_plan: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    if cached_plan is not None:
        prompt_messages = _apply_cached_message_plan(
            feature["messages"],
            list(cached_plan.get("message_plan") or []),
            max_image_side=max_image_side,
            max_image_pixels=max_image_pixels,
        )
        target_message = {"role": "assistant", "content": [{"type": "text", "text": feature["target_response"]}]}
        full_messages = prompt_messages + [target_message]
        full_text = str(cached_plan.get("full_text") or "")
        full_inputs = _tokenize_chat(
            processor,
            full_text,
            full_messages,
            max_length=max_seq_length,
            truncation_side="left",
        )
        response_token_count = int(cached_plan.get("response_token_count") or 0)
        next_cached_plan = None
    else:
        tagged_messages = _tag_messages_for_cache(feature["messages"])
        prompt_messages = _prepare_messages(
            tagged_messages,
            max_image_side=max_image_side,
            max_image_pixels=max_image_pixels,
            keep_recent_tool_image_messages=keep_recent_tool_image_messages,
            max_total_images=max_total_images,
            keep_recent_text_messages=keep_recent_text_messages,
        )
        prompt_messages, full_messages, prompt_text, full_text, full_inputs = _fit_messages_to_budget(
            processor,
            prompt_messages,
            target_response=feature["target_response"],
            max_seq_length=max_seq_length,
            truncation_side="left",
        )
        response_token_count = max(
            0,
            _count_text_tokens(processor, full_text) - _count_text_tokens(processor, prompt_text),
        )
        next_cached_plan = {
            "message_plan": _capture_message_plan(prompt_messages),
            "full_text": full_text,
            "response_token_count": int(response_token_count),
        }

    labels = _build_response_labels(
        full_inputs["input_ids"],
        response_token_count=response_token_count,
    )

    batch = dict(full_inputs)
    batch["labels"] = labels
    batch["sample_weight"] = torch.tensor([float(feature.get("sample_weight", 1.0))], dtype=torch.float32)
    if "advantage" in feature:
        batch["advantage"] = torch.tensor([float(feature.get("advantage", 0.0))], dtype=torch.float32)
    response_positions = labels.ne(-100)
    response_token_count = int(response_positions.sum().item())
    if response_token_count > 0:
        token_advantages = _build_token_advantages_for_feature(
            processor=processor,
            feature=feature,
            response_token_count=response_token_count,
        )
        full_token_advantages = labels.new_zeros(labels.shape, dtype=torch.float32)
        full_token_advantages[response_positions] = torch.tensor(token_advantages, dtype=torch.float32)
        batch["token_advantages"] = full_token_advantages
    return batch, next_cached_plan


def build_sft_tensor_cache_payload(
    processor: Any,
    feature: Dict[str, Any],
    *,
    max_image_side: int = 0,
    max_image_pixels: int = 0,
    keep_recent_tool_image_messages: int = 0,
    max_total_images: int = 0,
    max_seq_length: int = 0,
    keep_recent_text_messages: int = 0,
) -> Dict[str, Any]:
    batch, _ = _build_batch_from_feature(
        processor,
        feature,
        max_image_side=max_image_side,
        max_image_pixels=max_image_pixels,
        keep_recent_tool_image_messages=keep_recent_tool_image_messages,
        max_total_images=max_total_images,
        max_seq_length=max_seq_length,
        keep_recent_text_messages=keep_recent_text_messages,
        cached_plan=None,
    )
    payload: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            payload[key] = value.detach().cpu()
        else:
            payload[key] = copy.deepcopy(value)
    return payload


class _FrameReferenceResolver:
    def __init__(self, *, max_cached_videos: int = 2):
        self.max_cached_videos = max(0, int(max_cached_videos))
        self._frame_cache_tensors: "OrderedDict[str, Optional[torch.Tensor]]" = OrderedDict()
        self._frame_cache_status: "OrderedDict[str, str]" = OrderedDict()
        self._decord_readers: "OrderedDict[str, Any]" = OrderedDict()
        self._logged_raw_frame_fallbacks: set[tuple[str, str, str]] = set()

    def materialize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        materialized = copy.deepcopy(messages)
        for message in materialized:
            for item in message.get("content", []):
                image_ref = item.pop("image_ref", None)
                if item.get("type") != "image" or image_ref is None:
                    continue
                item["image"] = self._resolve_image_ref(image_ref)
        return materialized

    def _resolve_image_ref(self, image_ref: Dict[str, Any]) -> torch.Tensor:
        video_path = Path(str(image_ref.get("video_path") or ""))
        if not str(video_path):
            raise ValueError("image_ref is missing video_path")

        sampled_frame_index = image_ref.get("sampled_frame_index")
        cache_tensor, cache_status = self._load_frame_cache_tensor(video_path)
        if cache_tensor is not None and sampled_frame_index is not None:
            index = int(sampled_frame_index)
            if 0 <= index < int(cache_tensor.shape[0]):
                return cache_tensor[index]

        fallback_reason = "missing_frame_cache"
        if cache_tensor is not None and sampled_frame_index is None:
            fallback_reason = "missing_sampled_frame_index"
        elif cache_tensor is not None and sampled_frame_index is not None:
            fallback_reason = f"sampled_frame_index_out_of_range:{int(sampled_frame_index)}"
        elif cache_status != "missing":
            fallback_reason = f"cache_status:{cache_status}"
        self._warn_raw_frame_fallback(video_path=video_path, cache_status=cache_status, fallback_reason=fallback_reason)

        raw_frame_index = image_ref.get("raw_frame_index")
        if raw_frame_index is None:
            raw_frame_index = sampled_frame_index
        if raw_frame_index is None and image_ref.get("timestamp_sec") is not None:
            raw_frame_index = self._resolve_frame_index_from_timestamp(video_path, float(image_ref["timestamp_sec"]))
        if raw_frame_index is None:
            raise ValueError(f"Cannot resolve image_ref for {video_path}")
        return self._load_raw_video_frame(video_path, int(raw_frame_index))

    def _load_frame_cache_tensor(self, video_path: Path) -> tuple[Optional[torch.Tensor], str]:
        key = str(video_path)
        if key in self._frame_cache_tensors:
            self._touch_cache_key(key)
            return self._frame_cache_tensors[key], self._frame_cache_status.get(key, "missing")

        cache_path = _frame_cache_path(video_path)
        if not cache_path.exists():
            self._store_frame_cache_entry(key, None, "missing")
            return None, "missing"

        try:
            cache = torch.load(cache_path, map_location="cpu")
        except Exception:
            self._store_frame_cache_entry(key, None, "load_error")
            return None, "load_error"

        frame_tensor = cache.get("frame_tensor")
        resolved_tensor = frame_tensor if isinstance(frame_tensor, torch.Tensor) else None
        resolved_status = "loaded" if resolved_tensor is not None else "invalid"
        self._store_frame_cache_entry(key, resolved_tensor, resolved_status)
        return resolved_tensor, resolved_status

    def _warn_raw_frame_fallback(self, *, video_path: Path, cache_status: str, fallback_reason: str) -> None:
        warning_key = (str(video_path), str(cache_status), str(fallback_reason))
        if warning_key in self._logged_raw_frame_fallbacks:
            return
        self._logged_raw_frame_fallbacks.add(warning_key)
        _print_cache_warning(
            f"video_path={video_path} cache_status={cache_status} fallback_reason={fallback_reason} "
            f"cache_path={_frame_cache_path(video_path)} falling back to raw frame materialization."
        )

    def _resolve_frame_index_from_timestamp(self, video_path: Path, timestamp_sec: float) -> int:
        try:
            reader = self._get_decord_reader(video_path)
            fps = float(reader.get_avg_fps() or 0.0)
            if fps > 0:
                return max(0, min(int(round(timestamp_sec * fps)), len(reader) - 1))
        except Exception:
            pass

        try:
            import cv2
        except Exception as exc:
            raise RuntimeError("Resolving image_ref by timestamp requires decord or cv2.") from exc

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video for timestamp resolution: {video_path}")
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            capture.release()
        if fps <= 0:
            raise RuntimeError(f"Failed to read fps for video: {video_path}")
        if frame_count <= 0:
            return max(0, int(round(timestamp_sec * fps)))
        return max(0, min(int(round(timestamp_sec * fps)), frame_count - 1))

    def _load_raw_video_frame(self, video_path: Path, frame_index: int) -> torch.Tensor:
        try:
            reader = self._get_decord_reader(video_path)
            frame = reader[int(frame_index)].asnumpy()
            return torch.from_numpy(frame).permute(2, 0, 1).contiguous()
        except Exception:
            return self._load_raw_video_frame_with_cv2(video_path, frame_index)

    def _get_decord_reader(self, video_path: Path):
        key = str(video_path)
        if key in self._decord_readers:
            self._touch_cache_key(key)
            return self._decord_readers[key]
        from decord import VideoReader, cpu

        reader = VideoReader(str(video_path), ctx=cpu(0))
        self._decord_readers[key] = reader
        self._touch_cache_key(key)
        self._prune_cached_videos()
        return reader

    def _store_frame_cache_entry(self, key: str, tensor: Optional[torch.Tensor], status: str) -> None:
        self._frame_cache_tensors[key] = tensor
        self._frame_cache_status[key] = str(status)
        self._touch_cache_key(key)
        self._prune_cached_videos()

    def _touch_cache_key(self, key: str) -> None:
        if key in self._frame_cache_tensors:
            self._frame_cache_tensors.move_to_end(key)
        if key in self._frame_cache_status:
            self._frame_cache_status.move_to_end(key)
        if key in self._decord_readers:
            self._decord_readers.move_to_end(key)

    def _prune_cached_videos(self) -> None:
        if self.max_cached_videos <= 0:
            keys_to_drop = list(self._frame_cache_tensors.keys())
            keys_to_drop.extend([key for key in self._decord_readers.keys() if key not in self._frame_cache_tensors])
            for key in list(dict.fromkeys(keys_to_drop)):
                self._evict_cache_key(key)
            return
        while len(self._frame_cache_tensors) > self.max_cached_videos:
            oldest_key = next(iter(self._frame_cache_tensors))
            self._evict_cache_key(oldest_key)
        while len(self._decord_readers) > self.max_cached_videos:
            oldest_key = next(iter(self._decord_readers))
            self._evict_cache_key(oldest_key)

    def _evict_cache_key(self, key: str) -> None:
        self._frame_cache_tensors.pop(key, None)
        self._frame_cache_status.pop(key, None)
        self._decord_readers.pop(key, None)

    @staticmethod
    def _load_raw_video_frame_with_cv2(video_path: Path, frame_index: int) -> torch.Tensor:
        try:
            import cv2
        except Exception as exc:
            raise RuntimeError("Materializing image_ref requires decord or cv2.") from exc

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video for frame materialization: {video_path}")
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame {frame_index} from {video_path}")
        frame = frame[:, :, ::-1].copy()
        return torch.from_numpy(frame).permute(2, 0, 1).contiguous()


class SingleExampleMultimodalCollator:
    """Keep batching conservative for multimodal Qwen training.

    This collator expects `per_device_train_batch_size=1`. Users can recover
    throughput with gradient accumulation and multi-GPU launch.
    """

    def __init__(
        self,
        processor: Any,
        *,
        max_image_side: int = 0,
        max_image_pixels: int = 0,
        keep_recent_tool_image_messages: int = 0,
        max_total_images: int = 0,
        max_seq_length: int = 0,
        keep_recent_text_messages: int = 0,
        feature_plan_cache_size: int = 4096,
    ):
        self.processor = processor
        self.max_image_side = int(max_image_side)
        self.max_image_pixels = int(max_image_pixels)
        self.keep_recent_tool_image_messages = int(keep_recent_tool_image_messages)
        self.max_total_images = int(max_total_images)
        self.max_seq_length = int(max_seq_length)
        self.keep_recent_text_messages = int(keep_recent_text_messages)
        self.feature_plan_cache_size = max(0, int(feature_plan_cache_size))
        self._feature_plan_cache: "OrderedDict[Tuple[str, str], Dict[str, Any]]" = OrderedDict()

    def _feature_cache_key(self, feature: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        raw_key = feature.get("_feature_cache_key")
        if raw_key is None or str(raw_key) == "":
            return None
        return str(raw_key), str(feature.get("target_response") or "")

    def _get_cached_plan(self, feature: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        cache_key = self._feature_cache_key(feature)
        if cache_key is None or self.feature_plan_cache_size <= 0:
            return None
        cached = self._feature_plan_cache.get(cache_key)
        if cached is None:
            return None
        self._feature_plan_cache.move_to_end(cache_key)
        return cached

    def _store_cached_plan(self, feature: Dict[str, Any], payload: Dict[str, Any]) -> None:
        cache_key = self._feature_cache_key(feature)
        if cache_key is None or self.feature_plan_cache_size <= 0:
            return
        self._feature_plan_cache[cache_key] = dict(payload)
        self._feature_plan_cache.move_to_end(cache_key)
        while len(self._feature_plan_cache) > self.feature_plan_cache_size:
            self._feature_plan_cache.popitem(last=False)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(features) != 1:
            raise ValueError(
                "SingleExampleMultimodalCollator currently requires per_device_train_batch_size=1 "
                "for stable multimodal padding."
            )
        feature = features[0]
        if bool(feature.get("_pretokenized", False)):
            return {
                key: value
                for key, value in feature.items()
                if not str(key).startswith("_")
            }

        feature = copy.deepcopy(feature)
        cached_plan = self._get_cached_plan(feature)
        batch, plan_payload = _build_batch_from_feature(
            self.processor,
            feature,
            max_image_side=self.max_image_side,
            max_image_pixels=self.max_image_pixels,
            keep_recent_tool_image_messages=self.keep_recent_tool_image_messages,
            max_total_images=self.max_total_images,
            max_seq_length=self.max_seq_length,
            keep_recent_text_messages=self.keep_recent_text_messages,
            cached_plan=cached_plan,
        )
        if plan_payload is not None:
            self._store_cached_plan(feature, plan_payload)
        return batch


class WeightedExampleDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        examples: Sequence[Dict[str, Any]],
        *,
        tensor_cache_dir: str | Path = "",
        tensor_cache_expected_config: Optional[Dict[str, Any]] = None,
    ):
        self.examples = list(examples)
        self._frame_reference_resolver = _FrameReferenceResolver()
        self._example_cache_keys = [build_sft_tensor_cache_key(example) for example in self.examples]
        self._tensor_cache: Optional[PreparedSFTTensorCache] = None
        if tensor_cache_dir:
            self._tensor_cache = PreparedSFTTensorCache(
                tensor_cache_dir,
                expected_config=dict(tensor_cache_expected_config or {}),
            )
            self._tensor_cache.log_status()

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        cache_key = self._example_cache_keys[idx]
        if self._tensor_cache is not None:
            cached_payload = self._tensor_cache.load(cache_key)
            if cached_payload is not None:
                batch = {
                    key: value
                    for key, value in cached_payload.items()
                }
                batch["_pretokenized"] = True
                batch["_feature_cache_key"] = cache_key
                return batch
        example = materialize_example_for_training(
            self.examples[idx],
            resolver=self._frame_reference_resolver,
        )
        example["_feature_cache_key"] = cache_key
        return example


def validate_prepared_examples(
    examples: Sequence[Dict[str, Any]],
    *,
    materialize_images: bool = False,
    max_materialized_examples: int = 0,
    progress_every: int = 0,
) -> Dict[str, Any]:
    runtime = distributed_runtime_from_env()
    summary: Dict[str, Any] = {
        "num_examples": len(examples),
        "num_examples_with_image_refs": 0,
        "num_image_refs": 0,
        "num_inline_images": 0,
        "materialized_examples": 0,
        "num_errors": 0,
        "errors": [],
    }
    error_examples = set()
    total_examples = len(examples)

    for idx, example in enumerate(examples):
        prefix = f"example[{idx}]"
        messages = example.get("messages")
        if not isinstance(messages, list) or not messages:
            summary["errors"].append(f"{prefix}: missing non-empty messages list")
            error_examples.add(idx)
            continue

        target_response = example.get("target_response")
        if not isinstance(target_response, str) or not target_response.strip():
            summary["errors"].append(f"{prefix}: missing non-empty target_response")
            error_examples.add(idx)

        example_image_ref_count = 0
        for message_idx, message in enumerate(messages):
            content = message.get("content")
            if not isinstance(content, list):
                summary["errors"].append(f"{prefix}: message[{message_idx}] content is not a list")
                error_examples.add(idx)
                continue
            for content_idx, item in enumerate(content):
                if item.get("type") != "image":
                    continue
                item_prefix = f"{prefix}: message[{message_idx}].content[{content_idx}]"
                if "image_ref" in item:
                    example_image_ref_count += 1
                    summary["num_image_refs"] += 1
                    image_ref = item.get("image_ref") or {}
                    if not str(image_ref.get("video_path") or "").strip():
                        summary["errors"].append(f"{item_prefix}: image_ref.video_path is missing")
                        error_examples.add(idx)
                    if all(
                        image_ref.get(key) is None
                        for key in ("sampled_frame_index", "raw_frame_index", "timestamp_sec")
                    ):
                        summary["errors"].append(
                            f"{item_prefix}: image_ref needs sampled_frame_index, raw_frame_index, or timestamp_sec"
                        )
                        error_examples.add(idx)
                elif "image" in item:
                    summary["num_inline_images"] += 1
                else:
                    summary["errors"].append(f"{item_prefix}: image item is missing both image and image_ref")
                    error_examples.add(idx)
        if example_image_ref_count > 0:
            summary["num_examples_with_image_refs"] += 1
        completed = idx + 1
        if should_log_progress(completed, total_examples, int(progress_every)):
            runtime_log(
                (
                    "Prepared data validation progress: "
                    f"examples={completed}/{total_examples} "
                    f"image_refs={summary['num_image_refs']} errors={len(summary['errors'])}"
                ),
                runtime=runtime,
                main_process_only=True,
            )

    if materialize_images:
        inspect_count = len(examples) if max_materialized_examples <= 0 else min(len(examples), int(max_materialized_examples))
        dataset = WeightedExampleDataset(list(examples[:inspect_count]))
        for idx in range(inspect_count):
            try:
                dataset[idx]
            except Exception as exc:
                summary["errors"].append(f"example[{idx}]: failed to materialize image_ref payloads: {exc}")
                error_examples.add(idx)
            completed = idx + 1
            if should_log_progress(completed, inspect_count, int(progress_every)):
                runtime_log(
                    (
                        "Prepared data materialization progress: "
                        f"examples={completed}/{inspect_count} errors={len(summary['errors'])}"
                    ),
                    runtime=runtime,
                    main_process_only=True,
                )
        summary["materialized_examples"] = inspect_count

    summary["num_errors"] = len(summary["errors"])
    summary["num_invalid_examples"] = len(error_examples)
    return summary


def _shift_logits_and_labels(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    response_mask = shift_labels.ne(-100)
    return shift_logits, shift_labels, response_mask


def compute_masked_response_token_log_probs(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    shift_logits, shift_labels, response_mask = _shift_logits_and_labels(logits, labels)
    if not torch.any(response_mask):
        return shift_logits.new_zeros(shift_labels.shape), response_mask

    log_probs = F.log_softmax(shift_logits, dim=-1)
    safe_labels = shift_labels.masked_fill(~response_mask, 0)
    token_log_probs = torch.gather(log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs.masked_fill(~response_mask, 0.0)
    return token_log_probs, response_mask


def compute_masked_response_nll(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    shift_logits, shift_labels, response_mask = _shift_logits_and_labels(logits, labels)
    if not torch.any(response_mask):
        return logits.new_zeros(())

    log_probs = F.log_softmax(shift_logits, dim=-1)
    safe_labels = shift_labels.masked_fill(~response_mask, 0)
    token_nll = -torch.gather(log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    masked_nll = token_nll.masked_select(response_mask)
    return masked_nll.mean() if masked_nll.numel() else logits.new_zeros(())


def compute_masked_response_log_probs(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    batch_size = int(logits.shape[0]) if logits.ndim > 0 else 1
    token_log_probs, response_mask = compute_masked_response_token_log_probs(logits=logits, labels=labels)
    if not torch.any(response_mask):
        return logits.new_zeros((batch_size,))
    token_counts = response_mask.sum(dim=-1).clamp(min=1)
    return token_log_probs.sum(dim=-1) / token_counts


def compute_masked_forward_kl(
    *,
    policy_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    policy_shift_logits, shift_labels, response_mask = _shift_logits_and_labels(policy_logits, labels)
    reference_shift_logits = reference_logits[..., :-1, :].contiguous()
    if not torch.any(response_mask):
        return policy_logits.new_zeros(())

    policy_log_probs = F.log_softmax(policy_shift_logits, dim=-1)
    reference_log_probs = F.log_softmax(reference_shift_logits, dim=-1)
    reference_probs = reference_log_probs.exp()
    token_kl = torch.sum(reference_probs * (reference_log_probs - policy_log_probs), dim=-1)
    masked_kl = token_kl.masked_select(response_mask)
    return masked_kl.mean() if masked_kl.numel() else policy_logits.new_zeros(())


def compute_grpo_surrogate_loss(
    *,
    policy_log_probs: torch.Tensor,
    old_policy_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float,
) -> torch.Tensor:
    policy_log_probs = policy_log_probs.view(-1)
    old_policy_log_probs = old_policy_log_probs.to(policy_log_probs.device).view(-1)
    advantages = advantages.to(policy_log_probs.device).view(-1)
    if (
        policy_log_probs.numel() != old_policy_log_probs.numel()
        or policy_log_probs.numel() != advantages.numel()
    ):
        raise ValueError("policy_log_probs, old_policy_log_probs, and advantages must have matching lengths.")

    ratios = torch.exp(policy_log_probs - old_policy_log_probs)
    clipped_ratios = torch.clamp(ratios, 1.0 - float(clip_epsilon), 1.0 + float(clip_epsilon))
    surrogate_unclipped = ratios * advantages
    surrogate_clipped = clipped_ratios * advantages
    return -torch.minimum(surrogate_unclipped, surrogate_clipped).mean()


def _assign_weight_range(char_weights: List[float], start: int, end: int, weight: float) -> None:
    start = max(0, int(start))
    end = min(len(char_weights), int(end))
    for idx in range(start, end):
        char_weights[idx] = float(weight)


def _find_tag_bounds(text: str, tag_name: str) -> Optional[Tuple[int, int, int, int]]:
    open_tag = f"<{tag_name}>"
    close_tag = f"</{tag_name}>"
    open_idx = text.find(open_tag)
    close_idx = text.find(close_tag)
    if open_idx < 0 or close_idx < 0 or close_idx < open_idx:
        return None
    content_start = open_idx + len(open_tag)
    return open_idx, content_start, close_idx, close_idx + len(close_tag)


def _annotate_json_like_span_weights(
    text: str,
    char_weights: List[float],
    *,
    base_offset: int = 0,
) -> None:
    critical_keys = {
        "existence",
        "category",
        "severity",
        "anomaly_interval_sec",
        "precursor_interval_sec",
        "earliest_alert_sec",
        "decision",
        "alert_sec",
        "start_sec",
        "end_sec",
        "window_id",
        "evidence_id",
        "evidence_moment_ids",
        "counterfactual_type",
        "name",
        "arguments",
    }
    punctuation_weight = 0.2
    key_weight = 0.8
    value_weight = 1.15
    critical_value_weight = 1.35
    string_weight = 1.0

    idx = 0
    expecting_key = False
    current_key = ""
    stack: List[str] = []
    while idx < len(text):
        ch = text[idx]
        global_idx = base_offset + idx
        if ch in "{[":
            _assign_weight_range(char_weights, global_idx, global_idx + 1, punctuation_weight)
            stack.append(ch)
            expecting_key = ch == "{"
            idx += 1
            continue
        if ch in "}]":
            _assign_weight_range(char_weights, global_idx, global_idx + 1, punctuation_weight)
            if stack:
                stack.pop()
            expecting_key = bool(stack and stack[-1] == "{")
            current_key = ""
            idx += 1
            continue
        if ch in ",:":
            _assign_weight_range(char_weights, global_idx, global_idx + 1, punctuation_weight)
            if ch == ",":
                expecting_key = bool(stack and stack[-1] == "{")
                current_key = ""
            else:
                expecting_key = False
            idx += 1
            continue
        if ch.isspace():
            _assign_weight_range(char_weights, global_idx, global_idx + 1, punctuation_weight)
            idx += 1
            continue
        if ch == '"':
            end_idx = idx + 1
            escaped = False
            while end_idx < len(text):
                current = text[end_idx]
                if current == '"' and not escaped:
                    end_idx += 1
                    break
                escaped = current == "\\" and not escaped
                if current != "\\":
                    escaped = False
                end_idx += 1
            literal = text[idx + 1 : max(idx + 1, end_idx - 1)]
            if expecting_key:
                _assign_weight_range(char_weights, global_idx, base_offset + end_idx, key_weight)
                current_key = literal
            else:
                is_critical = current_key in critical_keys
                _assign_weight_range(
                    char_weights,
                    global_idx,
                    base_offset + end_idx,
                    critical_value_weight if is_critical else value_weight,
                )
            idx = end_idx
            continue

        end_idx = idx + 1
        while end_idx < len(text) and text[end_idx] not in ',:{}[] \t\r\n':
            end_idx += 1
        token = text[idx:end_idx]
        is_value = not expecting_key
        weight = string_weight
        if is_value:
            weight = critical_value_weight if current_key in critical_keys else value_weight
        _assign_weight_range(char_weights, global_idx, base_offset + end_idx, weight)
        idx = end_idx


def _build_response_char_weights(
    *,
    response_text: str,
    target_action: Optional[str] = None,
    tool_name: Optional[str] = None,
) -> List[float]:
    if not response_text:
        return []
    char_weights = [1.0 for _ in response_text]

    for tag_name in ("think", "tool_call", "answer"):
        bounds = _find_tag_bounds(response_text, tag_name)
        if bounds is None:
            continue
        open_start, content_start, content_end, close_end = bounds
        _assign_weight_range(char_weights, open_start, content_start, 0.15)
        _assign_weight_range(char_weights, content_end, close_end, 0.15)
        if tag_name == "think":
            _assign_weight_range(char_weights, content_start, content_end, 0.35)
        else:
            _assign_weight_range(char_weights, content_start, content_end, 1.0)
            _annotate_json_like_span_weights(
                response_text[content_start:content_end],
                char_weights,
                base_offset=content_start,
            )

    if str(target_action or "") == "answer" and "<answer>" not in response_text:
        _annotate_json_like_span_weights(response_text, char_weights, base_offset=0)
    elif str(target_action or "") == "tool_call" and str(tool_name or "") and "<tool_call>" not in response_text:
        _annotate_json_like_span_weights(response_text, char_weights, base_offset=0)
    return char_weights


def _apply_focus_term_weights(
    response_text: str,
    char_weights: List[float],
    *,
    focus_terms: Sequence[str],
    boost_weight: float,
) -> None:
    lowered_text = response_text.lower()
    for term in focus_terms:
        lowered_term = str(term or "").strip().lower()
        if not lowered_term:
            continue
        start = 0
        while True:
            index = lowered_text.find(lowered_term, start)
            if index < 0:
                break
            _assign_weight_range(char_weights, index, index + len(lowered_term), boost_weight)
            start = index + len(lowered_term)


def _build_component_response_char_weights(
    *,
    response_text: str,
    component_name: str,
    target_action: Optional[str] = None,
    tool_name: Optional[str] = None,
) -> List[float]:
    char_weights = _build_response_char_weights(
        response_text=response_text,
        target_action=target_action,
        tool_name=tool_name,
    )
    if not char_weights or component_name == "global":
        return char_weights

    focus_terms_by_component = {
        "search_local": [
            "scan_timeline",
            "seek_evidence",
            "start_sec",
            "end_sec",
            "query",
            "purpose",
            "num_frames",
            "top_k",
            "window",
        ],
        "alert_local": [
            "alert_sec",
            "earliest_alert_sec",
            "decision",
            "soft_alert",
            "hard_alert",
            "alert",
        ],
        "evidence_local": [
            "candidate_window_ids",
            "candidate_evidence_ids",
            "candidate_evidence_moment_ids",
            "selected_window_ids",
            "selected_evidence_ids",
            "evidence_moment_ids",
            "window_id",
            "evidence_id",
        ],
    }
    _apply_focus_term_weights(
        response_text,
        char_weights,
        focus_terms=focus_terms_by_component.get(component_name, []),
        boost_weight=2.5,
    )
    return char_weights


def _token_weights_from_char_weights(
    char_weights: Sequence[float],
    offsets: Sequence[Tuple[int, int]],
) -> List[float]:
    token_weights: List[float] = []
    for start, end in offsets:
        start = max(0, int(start))
        end = min(len(char_weights), int(end))
        if end <= start:
            token_weights.append(1.0)
            continue
        token_weights.append(sum(float(value) for value in char_weights[start:end]) / float(end - start))
    return token_weights


def build_token_advantages_from_offsets(
    *,
    response_text: str,
    offsets: Sequence[Tuple[int, int]],
    base_advantage: float,
    target_action: Optional[str] = None,
    tool_name: Optional[str] = None,
    advantage_components: Optional[Dict[str, float]] = None,
    turn_component_weights: Optional[Dict[str, float]] = None,
) -> List[float]:
    if not offsets:
        return []
    component_pairs: List[Tuple[str, float]] = []
    if advantage_components:
        for component_name in ("global", "search_local", "alert_local", "evidence_local"):
            component_advantage = float(advantage_components.get(component_name, 0.0) or 0.0)
            component_weight = 1.0
            if turn_component_weights is not None:
                component_weight = float(turn_component_weights.get(component_name, 0.0) or 0.0)
            scaled_advantage = component_advantage * component_weight
            if abs(scaled_advantage) > 1e-8:
                component_pairs.append((component_name, scaled_advantage))

    if component_pairs:
        combined_advantages = [0.0 for _ in offsets]
        for component_name, component_advantage in component_pairs:
            char_weights = _build_component_response_char_weights(
                response_text=response_text,
                component_name=component_name,
                target_action=target_action,
                tool_name=tool_name,
            )
            if not char_weights:
                char_weights = [1.0 for _ in response_text]
            token_weights = _token_weights_from_char_weights(char_weights, offsets)
            mean_weight = sum(token_weights) / float(len(token_weights)) if token_weights else 1.0
            if mean_weight <= 1e-8:
                mean_weight = 1.0
            for index, weight in enumerate(token_weights):
                combined_advantages[index] += float(component_advantage) * (weight / mean_weight)
        return combined_advantages

    char_weights = _build_response_char_weights(
        response_text=response_text,
        target_action=target_action,
        tool_name=tool_name,
    )
    if not char_weights:
        return [float(base_advantage) for _ in offsets]

    token_weights = _token_weights_from_char_weights(char_weights, offsets)
    mean_weight = sum(token_weights) / float(len(token_weights)) if token_weights else 1.0
    if mean_weight <= 1e-8:
        mean_weight = 1.0
    return [float(base_advantage) * (weight / mean_weight) for weight in token_weights]


def _extract_response_offsets(processor: Any, response_text: str) -> List[Tuple[int, int]]:
    tokenizer = getattr(processor, "tokenizer", None) or processor
    try:
        encoded = tokenizer(
            response_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = encoded.get("offset_mapping") if isinstance(encoded, dict) else None
        if offsets is None:
            return []
        if hasattr(offsets, "tolist"):
            offsets = offsets.tolist()
        return [tuple(map(int, offset)) for offset in offsets]
    except Exception:
        return []


def _align_token_advantages(
    token_advantages: Sequence[float],
    *,
    response_token_count: int,
    base_advantage: float,
) -> List[float]:
    if response_token_count <= 0:
        return []
    values = [float(value) for value in token_advantages]
    if not values:
        return [float(base_advantage) for _ in range(response_token_count)]
    if len(values) == response_token_count:
        return values
    if len(values) < response_token_count:
        pad_value = values[-1] if values else float(base_advantage)
        return values + [pad_value for _ in range(response_token_count - len(values))]
    return values[:response_token_count]


def _build_token_advantages_for_feature(
    *,
    processor: Any,
    feature: Dict[str, Any],
    response_token_count: int,
) -> List[float]:
    base_advantage = float(feature.get("advantage", feature.get("sample_weight", 1.0)) or 0.0)
    response_text = str(feature.get("target_response") or "")
    if not response_text:
        return [base_advantage for _ in range(response_token_count)]
    offsets = _extract_response_offsets(processor, response_text)
    token_advantages = build_token_advantages_from_offsets(
        response_text=response_text,
        offsets=offsets,
        base_advantage=base_advantage,
        target_action=feature.get("target_action"),
        tool_name=feature.get("tool_name"),
        advantage_components=feature.get("advantage_components"),
        turn_component_weights=feature.get("turn_component_weights"),
    )
    return _align_token_advantages(
        token_advantages,
        response_token_count=response_token_count,
        base_advantage=base_advantage,
    )


def load_qwen_model_and_processor(
    model_path: str | Path,
    *,
    torch_dtype: str = "auto",
    attn_implementation: Optional[str] = None,
    gradient_checkpointing: bool = False,
    use_lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[Sequence[str]] = None,
) -> Tuple[Any, Any]:
    try:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except Exception as exc:
        raise ImportError(
            "Training requires a recent transformers build with Qwen3-VL support. "
            "Install it with `pip install git+https://github.com/huggingface/transformers accelerate`."
        ) from exc

    model_init_kwargs: Dict[str, Any] = {}
    if torch_dtype != "auto":
        model_init_kwargs["torch_dtype"] = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype
    if attn_implementation:
        model_init_kwargs["attn_implementation"] = attn_implementation

    model = Qwen3VLForConditionalGeneration.from_pretrained(str(model_path), **model_init_kwargs)
    processor = AutoProcessor.from_pretrained(str(model_path))
    if hasattr(model.config, "use_cache"):
        # KV cache is useful for autoregressive decoding, not for teacher-forced SFT.
        model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if use_lora:
        try:
            from peft import LoraConfig, get_peft_model
        except Exception as exc:
            raise ImportError("LoRA training requires `peft` to be installed.") from exc
        peft_config = LoraConfig(
            r=int(lora_r),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout),
            target_modules=list(lora_target_modules or DEFAULT_LORA_TARGET_MODULES),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
    return model, processor


def create_trainer(
    *,
    model: Any,
    processor: Any,
    train_dataset: torch.utils.data.Dataset,
    output_dir: str | Path,
    learning_rate: float,
    num_train_epochs: float,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    logging_steps: int,
    save_steps: int,
    save_total_limit: int,
    warmup_ratio: float,
    weight_decay: float,
    max_grad_norm: float,
    bf16: bool,
    fp16: bool,
    max_image_side: int = 0,
    max_image_pixels: int = 0,
    keep_recent_tool_image_messages: int = 0,
    max_total_images: int = 0,
    max_seq_length: int = 0,
    keep_recent_text_messages: int = 0,
    dataloader_num_workers: int = 0,
    dataloader_prefetch_factor: int = 0,
    dataloader_persistent_workers: bool = False,
    training_objective: str = "weighted_sft",
    old_policy_model: Optional[Any] = None,
    ppo_clip_epsilon: float = 0.2,
    reference_model: Optional[Any] = None,
    kl_beta: float = 0.0,
    callbacks: Optional[Sequence[Any]] = None,
) -> Any:
    try:
        from transformers import Trainer, TrainingArguments
    except Exception as exc:
        raise ImportError("Training requires the `transformers` package.") from exc

    class WeightedLossTrainer(Trainer):
        def __init__(
            self,
            *trainer_args,
            training_objective: str = "weighted_sft",
            old_policy_model: Optional[Any] = None,
            ppo_clip_epsilon: float = 0.2,
            reference_model: Optional[Any] = None,
            kl_beta: float = 0.0,
            **trainer_kwargs,
        ):
            super().__init__(*trainer_args, **trainer_kwargs)
            self.training_objective = str(training_objective)
            self.old_policy_model = old_policy_model
            self.ppo_clip_epsilon = float(ppo_clip_epsilon)
            self.reference_model = reference_model
            self.kl_beta = float(kl_beta)
            self._old_policy_model_device = None
            self._reference_model_device = None
            if self.old_policy_model is not None:
                self.old_policy_model.eval()
                for parameter in self.old_policy_model.parameters():
                    parameter.requires_grad_(False)
            if self.reference_model is not None:
                self.reference_model.eval()
                for parameter in self.reference_model.parameters():
                    parameter.requires_grad_(False)

        def _ensure_aux_model_device(self, aux_model: Any, current_device: Any, device_attr_name: str) -> None:
            if aux_model is None:
                return
            if getattr(self, device_attr_name) != current_device:
                aux_model.to(current_device)
                aux_model.eval()
                setattr(self, device_attr_name, current_device)

        def _ensure_old_policy_model_device(self, model: Any) -> None:
            if self.old_policy_model is None:
                return
            try:
                target_device = next(model.parameters()).device
            except StopIteration:
                return
            self._ensure_aux_model_device(self.old_policy_model, target_device, "_old_policy_model_device")

        def _ensure_reference_model_device(self, model: Any) -> None:
            if self.reference_model is None:
                return
            try:
                target_device = next(model.parameters()).device
            except StopIteration:
                return
            self._ensure_aux_model_device(self.reference_model, target_device, "_reference_model_device")

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            sample_weight = inputs.pop("sample_weight", None)
            advantage = inputs.pop("advantage", None)
            token_advantages = inputs.pop("token_advantages", None)
            labels = inputs.get("labels")
            outputs = model(**inputs)
            if labels is None:
                loss = outputs.loss
            else:
                if self.training_objective == "grpo":
                    effective_advantage = advantage if advantage is not None else sample_weight
                    if effective_advantage is None:
                        raise ValueError("GRPO training requires `advantage` or `sample_weight` in each batch.")
                    if self.old_policy_model is None:
                        raise ValueError("GRPO training requires a frozen old_policy_model.")
                    if token_advantages is not None:
                        policy_token_log_probs, response_mask = compute_masked_response_token_log_probs(
                            logits=outputs.logits,
                            labels=labels,
                        )
                    else:
                        response_mask = None
                        policy_log_probs = compute_masked_response_log_probs(logits=outputs.logits, labels=labels)
                    self._ensure_old_policy_model_device(model)
                    old_policy_inputs = {key: value for key, value in inputs.items() if key != "labels"}
                    with torch.no_grad():
                        old_policy_outputs = self.old_policy_model(**old_policy_inputs)
                    if token_advantages is not None and response_mask is not None:
                        old_policy_token_log_probs, _ = compute_masked_response_token_log_probs(
                            logits=old_policy_outputs.logits,
                            labels=labels,
                        )
                        shifted_token_advantages = token_advantages[..., 1:].to(policy_token_log_probs.device)
                        masked_policy_log_probs = policy_token_log_probs.masked_select(response_mask)
                        masked_old_policy_log_probs = old_policy_token_log_probs.masked_select(response_mask)
                        masked_token_advantages = shifted_token_advantages.masked_select(response_mask)
                        loss = compute_grpo_surrogate_loss(
                            policy_log_probs=masked_policy_log_probs,
                            old_policy_log_probs=masked_old_policy_log_probs.detach(),
                            advantages=masked_token_advantages,
                            clip_epsilon=self.ppo_clip_epsilon,
                        )
                    else:
                        old_policy_log_probs = compute_masked_response_log_probs(
                            logits=old_policy_outputs.logits,
                            labels=labels,
                        )
                        loss = compute_grpo_surrogate_loss(
                            policy_log_probs=policy_log_probs,
                            old_policy_log_probs=old_policy_log_probs.detach(),
                            advantages=effective_advantage,
                            clip_epsilon=self.ppo_clip_epsilon,
                        )
                else:
                    # The collator already masks prompt tokens with -100, so the model's
                    # native loss is the same objective without an extra full-vocab log_softmax.
                    nll_loss = getattr(outputs, "loss", None)
                    if nll_loss is None:
                        nll_loss = compute_masked_response_nll(logits=outputs.logits, labels=labels)
                    weight_value = (
                        sample_weight.to(nll_loss.device).mean()
                        if sample_weight is not None
                        else nll_loss.new_tensor(1.0)
                    )
                    loss = nll_loss * weight_value

                if self.reference_model is not None and self.kl_beta > 0.0:
                    self._ensure_reference_model_device(model)
                    reference_inputs = {key: value for key, value in inputs.items() if key != "labels"}
                    with torch.no_grad():
                        reference_outputs = self.reference_model(**reference_inputs)
                    kl_loss = compute_masked_forward_kl(
                        policy_logits=outputs.logits,
                        reference_logits=reference_outputs.logits,
                        labels=labels,
                    )
                    loss = loss + loss.new_tensor(self.kl_beta) * kl_loss

            if sample_weight is not None:
                outputs.sample_weight = sample_weight
            if advantage is not None:
                outputs.advantage = advantage
            if token_advantages is not None:
                outputs.token_advantages = token_advantages
            return (loss, outputs) if return_outputs else loss

    training_args_kwargs = {
        "output_dir": str(output_dir),
        "learning_rate": float(learning_rate),
        "num_train_epochs": float(num_train_epochs),
        "per_device_train_batch_size": int(per_device_train_batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "logging_steps": int(logging_steps),
        "save_steps": int(save_steps),
        "save_total_limit": int(save_total_limit),
        "warmup_ratio": float(warmup_ratio),
        "weight_decay": float(weight_decay),
        "max_grad_norm": float(max_grad_norm),
        "bf16": bool(bf16),
        "fp16": bool(fp16),
        "remove_unused_columns": False,
        "report_to": [],
        "disable_tqdm": True,
        "dataloader_num_workers": max(0, int(dataloader_num_workers)),
        "dataloader_persistent_workers": bool(dataloader_persistent_workers) and int(dataloader_num_workers) > 0,
    }
    if int(dataloader_num_workers) > 0 and int(dataloader_prefetch_factor) > 0:
        training_args_kwargs["dataloader_prefetch_factor"] = int(dataloader_prefetch_factor)
    args = TrainingArguments(**training_args_kwargs)
    trainer = WeightedLossTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=SingleExampleMultimodalCollator(
            processor,
            max_image_side=max_image_side,
            max_image_pixels=max_image_pixels,
            keep_recent_tool_image_messages=keep_recent_tool_image_messages,
            max_total_images=max_total_images,
            max_seq_length=max_seq_length,
            keep_recent_text_messages=keep_recent_text_messages,
        ),
        training_objective=training_objective,
        old_policy_model=old_policy_model,
        ppo_clip_epsilon=ppo_clip_epsilon,
        reference_model=reference_model,
        kl_beta=kl_beta,
        callbacks=list(callbacks or []),
    )
    trainer.add_callback(_build_epoch_progress_callback(trainer=trainer))
    return trainer


def _build_epoch_progress_callback(*, trainer: Any):
    try:
        from transformers import TrainerCallback
    except Exception as exc:
        raise ImportError("Epoch progress callbacks require the `transformers` package.") from exc
    try:
        from tqdm.auto import tqdm
    except Exception as exc:
        raise ImportError("Epoch progress callbacks require `tqdm` to be installed.") from exc

    class EpochProgressCallback(TrainerCallback):
        def __init__(self):
            self.trainer = trainer
            self.runtime = distributed_runtime_from_env()
            self.progress_bar = None
            self.current_epoch_index = 0
            self.epoch_start_global_step = 0
            self.last_epoch_step = 0
            self.steps_per_epoch = 0
            self.display_total_epochs = 0

        def on_train_begin(self, args, state, control, **kwargs):
            if not self.runtime.is_main_process:
                return control
            train_dataloader = self.trainer.get_train_dataloader()
            dataloader_steps = len(train_dataloader) if hasattr(train_dataloader, "__len__") else 0
            accumulation = max(1, int(args.gradient_accumulation_steps))
            self.steps_per_epoch = max(1, int(math.ceil(float(dataloader_steps) / float(accumulation))))
            self.display_total_epochs = max(1, int(math.ceil(float(args.num_train_epochs))))
            return control

        def on_epoch_begin(self, args, state, control, **kwargs):
            if not self.runtime.is_main_process:
                return control
            self.current_epoch_index += 1
            self.epoch_start_global_step = int(state.global_step or 0)
            self.last_epoch_step = 0
            remaining_steps = max(0, int(state.max_steps or 0) - self.epoch_start_global_step)
            epoch_total = max(1, min(int(self.steps_per_epoch or 1), remaining_steps or int(self.steps_per_epoch or 1)))
            if self.progress_bar is not None:
                self.progress_bar.close()
            self.progress_bar = tqdm(
                total=epoch_total,
                desc=f"Epoch {self.current_epoch_index}/{self.display_total_epochs}",
                leave=True,
                dynamic_ncols=True,
            )
            return control

        def on_step_end(self, args, state, control, **kwargs):
            if not self.runtime.is_main_process or self.progress_bar is None:
                return control
            current_epoch_step = max(0, int(state.global_step or 0) - self.epoch_start_global_step)
            delta = current_epoch_step - self.last_epoch_step
            if delta > 0:
                remaining = max(0, int(self.progress_bar.total or 0) - int(self.progress_bar.n or 0))
                self.progress_bar.update(min(delta, remaining))
                self.last_epoch_step = current_epoch_step
            return control

        def on_epoch_end(self, args, state, control, **kwargs):
            if not self.runtime.is_main_process or self.progress_bar is None:
                return control
            remaining = max(0, int(self.progress_bar.total or 0) - int(self.progress_bar.n or 0))
            if remaining > 0:
                self.progress_bar.update(remaining)
            self.progress_bar.close()
            self.progress_bar = None
            return control

        def on_train_end(self, args, state, control, **kwargs):
            if self.progress_bar is not None:
                self.progress_bar.close()
                self.progress_bar = None
            return control

    return EpochProgressCallback()


def _build_rollout_eval_callback(
    *,
    processor: Any,
    rollout_eval_config: RolloutEvaluationConfig,
):
    try:
        from transformers import TrainerCallback
    except Exception as exc:
        raise ImportError("Rollout evaluation callbacks require the `transformers` package.") from exc

    from saver_agent.qwen_policy import QwenGenerationPolicy

    class RolloutEvalCallback(TrainerCallback):
        def __init__(self):
            self.processor = processor
            self.rollout_eval_config = rollout_eval_config
            self.runtime = distributed_runtime_from_env()

        def on_epoch_end(self, args, state, control, model=None, **kwargs):
            if model is None:
                return control
            eval_model = _unwrap_model(model)
            was_training = bool(getattr(eval_model, "training", False))
            if hasattr(eval_model, "eval"):
                eval_model.eval()
            try:
                policy = QwenGenerationPolicy.from_components(
                    model=eval_model,
                    processor=self.processor,
                    max_new_tokens=512,
                    do_sample=False,
                )
                metrics = run_rollout_evaluation(
                    policy,
                    eval_config=self.rollout_eval_config,
                    output_dir=args.output_dir,
                    epoch_index=max(1, int(round(float(state.epoch or 0.0)))),
                    runtime=self.runtime,
                )
                if self.runtime.is_main_process and metrics is not None:
                    record = {"epoch": float(state.epoch or 0.0), **metrics}
                    _append_jsonl(Path(args.output_dir) / "rollout_eval_metrics.jsonl", record)
                    runtime_log(
                        (
                            f"epoch {float(state.epoch or 0.0):.2f} rollout eval: "
                            f"{json.dumps(metrics, ensure_ascii=False)}"
                        ),
                        runtime=self.runtime,
                        main_process_only=True,
                    )
            finally:
                if was_training and hasattr(eval_model, "train"):
                    eval_model.train()
            return control

    return RolloutEvalCallback()


def run_weighted_sft(
    examples: Sequence[Dict[str, Any]],
    *,
    model_path: str | Path,
    output_dir: str | Path,
    tensor_cache_dir: str | Path = "",
    training_objective: str = "weighted_sft",
    old_policy_model_path: str | Path | None = None,
    ppo_clip_epsilon: float = 0.2,
    reference_model_path: str | Path | None = None,
    kl_beta: float = 0.0,
    torch_dtype: str = "auto",
    attn_implementation: Optional[str] = None,
    gradient_checkpointing: bool = False,
    use_lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[Sequence[str]] = None,
    learning_rate: float = 1e-5,
    num_train_epochs: float = 1.0,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    logging_steps: int = 10,
    save_steps: int = 100,
    save_total_limit: int = 2,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.0,
    max_grad_norm: float = 1.0,
    bf16: bool = True,
    fp16: bool = False,
    max_image_side: int = 0,
    max_image_pixels: int = 0,
    keep_recent_tool_image_messages: int = 0,
    max_total_images: int = 0,
    max_seq_length: int = 0,
    keep_recent_text_messages: int = 0,
    dataloader_num_workers: int = 0,
    dataloader_prefetch_factor: int = 0,
    dataloader_persistent_workers: bool = False,
    rollout_eval_config: RolloutEvaluationConfig | None = None,
) -> Dict[str, Any]:
    if not examples:
        raise ValueError("No training examples were provided.")

    runtime = distributed_runtime_from_env()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_log(
        (
            f"SFT setup: num_examples={len(examples)} output_dir={output_dir} model_path={model_path} "
            f"training_objective={training_objective} "
            f"tensor_cache_dir={str(tensor_cache_dir) if tensor_cache_dir else 'off'}"
        ),
        runtime=runtime,
        main_process_only=True,
    )
    runtime_log(
        (
            "SFT memory controls: "
            f"max_image_side={int(max_image_side) or 'off'} "
            f"max_image_pixels={int(max_image_pixels) or 'off'} "
            f"keep_recent_tool_image_messages={int(keep_recent_tool_image_messages) or 'all'} "
            f"max_total_images={int(max_total_images) or 'all'} "
            f"max_seq_length={int(max_seq_length) or 'off'} "
            f"keep_recent_text_messages={int(keep_recent_text_messages) or 'all'}"
        ),
        runtime=runtime,
        main_process_only=True,
    )
    runtime_log(
        (
            "SFT dataloader controls: "
            f"num_workers={max(0, int(dataloader_num_workers))} "
            f"prefetch_factor={int(dataloader_prefetch_factor) if int(dataloader_num_workers) > 0 and int(dataloader_prefetch_factor) > 0 else 'off'} "
            f"persistent_workers={bool(dataloader_persistent_workers) and int(dataloader_num_workers) > 0}"
        ),
        runtime=runtime,
        main_process_only=True,
    )
    runtime_log(
        format_example_frame_cache_status(summarize_example_frame_cache_status(examples), prefix="training frame cache"),
        runtime=runtime,
        main_process_only=True,
    )
    runtime_log(f"loading policy model from {model_path}", runtime=runtime, main_process_only=True)
    model, processor = load_qwen_model_and_processor(
        model_path,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        gradient_checkpointing=gradient_checkpointing,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
    )
    processor_signature = build_processor_signature(processor)
    old_policy_model = None
    resolved_old_policy_model_path = str(old_policy_model_path) if old_policy_model_path else ""
    if str(training_objective) == "grpo":
        if not resolved_old_policy_model_path:
            resolved_old_policy_model_path = str(model_path)
        runtime_log(
            f"loading old policy model from {resolved_old_policy_model_path}",
            runtime=runtime,
            main_process_only=True,
        )
        old_policy_model, _ = load_qwen_model_and_processor(
            resolved_old_policy_model_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            gradient_checkpointing=False,
            use_lora=False,
        )
    reference_model = None
    resolved_reference_model_path = str(reference_model_path) if reference_model_path else ""
    if float(kl_beta) > 0.0 and resolved_reference_model_path:
        runtime_log(
            f"loading reference model from {resolved_reference_model_path}",
            runtime=runtime,
            main_process_only=True,
        )
        reference_model, _ = load_qwen_model_and_processor(
            resolved_reference_model_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            gradient_checkpointing=False,
            use_lora=False,
        )
    callbacks = []
    if rollout_eval_config is not None:
        runtime_log(
            (
                "epoch-end rollout evaluation enabled: "
                f"data={rollout_eval_config.data_path} max_records={rollout_eval_config.max_records or 'all'} "
                f"backend={rollout_eval_config.verifier_backend}"
            ),
            runtime=runtime,
            main_process_only=True,
        )
        callbacks.append(
            _build_rollout_eval_callback(
                processor=processor,
                rollout_eval_config=rollout_eval_config,
            )
        )
    trainer = create_trainer(
        model=model,
        processor=processor,
        train_dataset=WeightedExampleDataset(
            examples,
            tensor_cache_dir=tensor_cache_dir,
            tensor_cache_expected_config=normalize_sft_tensor_cache_config(
                processor_signature=processor_signature,
                max_image_side=max_image_side,
                max_image_pixels=max_image_pixels,
                keep_recent_tool_image_messages=keep_recent_tool_image_messages,
                max_total_images=max_total_images,
                max_seq_length=max_seq_length,
                keep_recent_text_messages=keep_recent_text_messages,
            ),
        ),
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        bf16=bf16,
        fp16=fp16,
        max_image_side=max_image_side,
        max_image_pixels=max_image_pixels,
        keep_recent_tool_image_messages=keep_recent_tool_image_messages,
        max_total_images=max_total_images,
        max_seq_length=max_seq_length,
        keep_recent_text_messages=keep_recent_text_messages,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_prefetch_factor=dataloader_prefetch_factor,
        dataloader_persistent_workers=dataloader_persistent_workers,
        training_objective=training_objective,
        old_policy_model=old_policy_model,
        ppo_clip_epsilon=ppo_clip_epsilon,
        reference_model=reference_model,
        kl_beta=kl_beta,
        callbacks=callbacks,
    )
    runtime_log("starting Trainer.train()", runtime=runtime, main_process_only=True)
    train_result = trainer.train()
    if trainer.is_world_process_zero():
        trainer.save_model(str(output_dir))
        processor.save_pretrained(str(output_dir))
    distributed_barrier(runtime)
    return {
        "num_examples": len(examples),
        "output_dir": str(output_dir),
        "tensor_cache_dir": str(tensor_cache_dir) if tensor_cache_dir else "",
        "train_loss": float(getattr(train_result, "training_loss", 0.0)),
        "training_objective": str(training_objective),
        "old_policy_model_path": resolved_old_policy_model_path,
        "ppo_clip_epsilon": float(ppo_clip_epsilon),
        "kl_beta": float(kl_beta),
        "reference_model_path": resolved_reference_model_path,
    }
