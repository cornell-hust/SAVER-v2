from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from saver_agent.self_verification import build_policy_self_verification_payload


DEFAULT_MODEL_PATH = os.environ.get(
    "SAVER_QWEN_MODEL_PATH",
    "/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/MLLMs/qwen3-vl-8b-Instruct",
)
_TIMESTAMP_ONLY_RE = re.compile(r"^\s*\d+(?:\.\d+)?s\s*$")
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
_ANSWER_BLOCK_RE = re.compile(r"<answer>.*?</answer>", re.DOTALL)
_VERIFY_COMPACT_KEYS = {
    "verification_decision",
    "recommended_action",
    "sufficiency_score",
    "necessity_score",
    "alertability_score",
    "counterfactual_faithfulness",
    "selected_window_ids",
    "selected_evidence_moment_ids",
}
_STRUCTURED_STOP_STRINGS = ("</tool_call>", "</answer>")


def _configure_qwen_processor(processor: Any) -> Any:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        try:
            tokenizer.padding_side = "left"
        except Exception:
            pass
    try:
        processor.padding_side = "left"
    except Exception:
        pass
    return processor


def _build_generation_kwargs(
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    repetition_penalty: Optional[float],
    stopping_criteria: Any = None,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
    }
    if do_sample:
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if top_k is not None:
            kwargs["top_k"] = top_k
    if repetition_penalty is not None:
        kwargs["repetition_penalty"] = repetition_penalty
    if stopping_criteria is not None:
        kwargs["stopping_criteria"] = stopping_criteria
    return kwargs


def _build_structured_stopping_criteria(processor: Any) -> Any:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None or not hasattr(tokenizer, "encode"):
        return None
    stop_token_sequences: List[List[int]] = []
    for stop_string in _STRUCTURED_STOP_STRINGS:
        try:
            token_ids = tokenizer.encode(stop_string, add_special_tokens=False)
        except Exception:
            token_ids = []
        token_ids = [int(token_id) for token_id in list(token_ids or [])]
        if token_ids:
            stop_token_sequences.append(token_ids)
    if not stop_token_sequences:
        return None

    def _should_stop(input_ids) -> bool:
        if input_ids is None or getattr(input_ids, "ndim", 0) != 2:
            return False
        for row in input_ids:
            row_ids = [int(token_id) for token_id in row.tolist()]
            matched = False
            for stop_ids in stop_token_sequences:
                if len(row_ids) >= len(stop_ids) and row_ids[-len(stop_ids) :] == stop_ids:
                    matched = True
                    break
            if not matched:
                return False
        return True

    try:
        from transformers import StoppingCriteria, StoppingCriteriaList

        class _StructuredStopCriteria(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return _should_stop(input_ids)

        return StoppingCriteriaList([_StructuredStopCriteria()])
    except Exception:
        class _FallbackStructuredStopCriteria:
            def __call__(self, input_ids, scores=None, **kwargs):
                return _should_stop(input_ids)

        return [_FallbackStructuredStopCriteria()]


def _trim_to_first_structured_block(output_text: str) -> str:
    text = str(output_text or "").strip()
    if not text:
        return text
    think_match = _THINK_BLOCK_RE.search(text)
    block_matches = [match for match in (_TOOL_CALL_BLOCK_RE.search(text), _ANSWER_BLOCK_RE.search(text)) if match]
    if not block_matches:
        return text
    chosen = min(block_matches, key=lambda match: match.start())
    prefix = ""
    if think_match is not None and think_match.start() <= chosen.start():
        prefix = think_match.group(0).strip()
    block_text = chosen.group(0).strip()
    if prefix:
        return f"{prefix}{block_text}"
    return block_text


def _compact_verify_tool_call(output_text: str) -> str:
    trimmed = _trim_to_first_structured_block(output_text)
    tool_match = _TOOL_CALL_BLOCK_RE.search(trimmed)
    if tool_match is None:
        return trimmed
    try:
        function_payload = json.loads(
            tool_match.group(0)[len("<tool_call>") : -len("</tool_call>")].strip()
        )
    except Exception:
        return trimmed
    if not isinstance(function_payload, dict):
        return trimmed
    if str(function_payload.get("name") or "") != "verify_hypothesis":
        return trimmed
    arguments = function_payload.get("arguments")
    if not isinstance(arguments, dict):
        return trimmed
    if not any(key in arguments for key in _VERIFY_COMPACT_KEYS):
        return trimmed
    try:
        compact_arguments = build_policy_self_verification_payload(
            arguments,
            include_query=False,
            include_rationale=False,
        )
    except Exception:
        return trimmed
    compact_function_payload = {
        "name": "verify_hypothesis",
        "arguments": compact_arguments,
    }
    compact_block = (
        "<tool_call>"
        + json.dumps(compact_function_payload, ensure_ascii=False, separators=(",", ":"))
        + "</tool_call>"
    )
    think_match = _THINK_BLOCK_RE.search(trimmed)
    if think_match is not None and think_match.start() == 0:
        return f"{think_match.group(0).strip()}{compact_block}"
    return compact_block


def _to_pil_image(image: Any) -> Any:
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, torch.Tensor):
        tensor = image.detach().cpu()
        if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
            tensor = tensor.clamp(0, 255).to(torch.uint8)
            if tensor.shape[0] == 1:
                tensor = tensor.repeat(3, 1, 1)
            array = tensor.permute(1, 2, 0).numpy()
            return Image.fromarray(array, mode="RGB")
    if isinstance(image, np.ndarray):
        array = image
        if array.ndim == 3 and array.shape[-1] in (1, 3):
            array = np.clip(array, 0, 255).astype(np.uint8)
            if array.shape[-1] == 1:
                array = np.repeat(array, 3, axis=-1)
            return Image.fromarray(array, mode="RGB")
    return image


def _is_image_item(item: Any) -> bool:
    return isinstance(item, dict) and item.get("type") == "image" and ("image" in item or "image_ref" in item)


def _is_timestamp_text_item(item: Any) -> bool:
    if not isinstance(item, dict) or item.get("type") != "text":
        return False
    return bool(_TIMESTAMP_ONLY_RE.match(str(item.get("text") or "").strip()))


def _prune_messages_to_max_total_images(
    messages: List[Dict[str, Any]],
    *,
    max_total_images: int = 0,
) -> List[Dict[str, Any]]:
    if int(max_total_images) <= 0:
        return copy.deepcopy(messages)

    prepared = copy.deepcopy(messages)
    image_positions: List[Tuple[int, int]] = []
    for message_index, message in enumerate(prepared):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for content_index, item in enumerate(content):
            if _is_image_item(item):
                image_positions.append((message_index, content_index))

    overflow = len(image_positions) - int(max_total_images)
    if overflow <= 0:
        return prepared

    for message_index, content_index in image_positions[:overflow]:
        content = prepared[message_index].get("content")
        if not isinstance(content, list):
            continue
        if 0 <= content_index < len(content):
            content[content_index] = None
        timestamp_index = content_index - 1
        if 0 <= timestamp_index < len(content) and _is_timestamp_text_item(content[timestamp_index]):
            content[timestamp_index] = None

    for message in prepared:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        message["content"] = [item for item in content if item is not None]
    return prepared


class QwenGenerationPolicy:
    """Single-turn Qwen generation policy for SAVER rollouts."""

    def __init__(
        self,
        *,
        model: Any,
        processor: Any,
        max_new_tokens: int = 512,
        max_total_images: int = 0,
        do_sample: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
    ):
        self.model = model
        self.processor = processor
        self.max_new_tokens = int(max_new_tokens)
        self.max_total_images = int(max_total_images)
        self.do_sample = bool(do_sample)
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.structured_stopping_criteria = _build_structured_stopping_criteria(processor)
        self._prepared_messages_cache_source_id: Optional[int] = None
        self._prepared_messages_cache_len: int = 0
        self._prepared_messages_cache: List[Dict[str, Any]] = []

    @classmethod
    def from_components(
        cls,
        *,
        model: Any,
        processor: Any,
        max_new_tokens: int = 512,
        max_total_images: int = 0,
        do_sample: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
    ) -> "QwenGenerationPolicy":
        return cls(
            model=model,
            processor=processor,
            max_new_tokens=max_new_tokens,
            max_total_images=max_total_images,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        *,
        torch_dtype: str = "auto",
        device_map: str = "auto",
        attn_implementation: Optional[str] = None,
        max_new_tokens: int = 512,
        max_total_images: int = 0,
        do_sample: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
    ) -> "QwenGenerationPolicy":
        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except Exception as exc:
            raise ImportError(
                "QwenGenerationPolicy requires a recent transformers build with Qwen3-VL support. "
                "Install it with `pip install git+https://github.com/huggingface/transformers accelerate`."
            ) from exc

        model_init_kwargs: Dict[str, Any] = {"device_map": device_map}
        if torch_dtype != "auto":
            model_init_kwargs["torch_dtype"] = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype
        else:
            model_init_kwargs["dtype"] = "auto"
        if attn_implementation:
            model_init_kwargs["attn_implementation"] = attn_implementation

        resolved_model_path = Path(model_path)
        adapter_config_path = resolved_model_path / "adapter_config.json"
        processor_path = str(resolved_model_path)
        if adapter_config_path.exists():
            try:
                from peft import PeftConfig, PeftModel
            except Exception as exc:
                raise ImportError("Loading LoRA adapter checkpoints requires `peft` to be installed.") from exc
            peft_config = PeftConfig.from_pretrained(str(resolved_model_path))
            base_model_path = str(peft_config.base_model_name_or_path)
            model = Qwen3VLForConditionalGeneration.from_pretrained(base_model_path, **model_init_kwargs)
            model = PeftModel.from_pretrained(model, str(resolved_model_path))
            if not any((resolved_model_path / filename).exists() for filename in ("preprocessor_config.json", "processor_config.json", "tokenizer_config.json", "tokenizer.json")):
                processor_path = base_model_path
        else:
            model = Qwen3VLForConditionalGeneration.from_pretrained(str(resolved_model_path), **model_init_kwargs)
        model.eval()
        processor = _configure_qwen_processor(AutoProcessor.from_pretrained(processor_path))
        return cls(
            model=model,
            processor=processor,
            max_new_tokens=max_new_tokens,
            max_total_images=max_total_images,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )

    def prepare_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.max_total_images > 0:
            pruned_messages = _prune_messages_to_max_total_images(
                messages,
                max_total_images=self.max_total_images,
            )
            self._prepared_messages_cache_source_id = None
            self._prepared_messages_cache_len = 0
            self._prepared_messages_cache = []
            return self._prepare_message_slice(pruned_messages)

        source_id = id(messages)
        if (
            self._prepared_messages_cache_source_id == source_id
            and len(messages) >= self._prepared_messages_cache_len
        ):
            new_messages = messages[self._prepared_messages_cache_len :]
            if new_messages:
                self._prepared_messages_cache.extend(self._prepare_message_slice(new_messages))
                self._prepared_messages_cache_len = len(messages)
            return self._prepared_messages_cache

        prepared = self._prepare_message_slice(messages)
        self._prepared_messages_cache_source_id = source_id
        self._prepared_messages_cache_len = len(messages)
        self._prepared_messages_cache = prepared
        return prepared

    def _prepare_message_slice(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        prepared = copy.deepcopy(messages)
        for message in prepared:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                item_type = item.get("type")
                if item_type == "image" and "image" in item:
                    item["image"] = _to_pil_image(item["image"])
                elif item_type == "video" and "video" in item:
                    item["video"] = self._prepare_video_payload(item["video"])
        return prepared

    def __call__(
        self,
        messages: List[Dict[str, Any]],
        multimodal_cache: Dict[str, Any],
        state: Any,
        step_index: int,
    ) -> str:
        prepared_messages = self.prepare_messages(messages)
        inputs = self._build_inputs(prepared_messages)
        inputs = self._move_to_model_device(inputs)
        generation_kwargs = self._generation_kwargs()
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generation_kwargs)

            input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, output_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        del inputs
        del output_ids
        del input_ids
        del generated_ids_trimmed
        return _compact_verify_tool_call(output_text[0])

    def _build_inputs(self, prepared_messages: List[Dict[str, Any]]) -> Any:
        try:
            return self.processor.apply_chat_template(
                prepared_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        except TypeError:
            prompt_text = self.processor.apply_chat_template(
                prepared_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = self._extract_vision_inputs(prepared_messages)
            processor_kwargs: Dict[str, Any] = {
                "text": prompt_text,
                "padding": True,
                "return_tensors": "pt",
            }
            if image_inputs:
                processor_kwargs["images"] = image_inputs
            if video_inputs:
                processor_kwargs["videos"] = video_inputs
            return self.processor(**processor_kwargs)

    def _move_to_model_device(self, inputs: Any) -> Any:
        if not hasattr(inputs, "to"):
            return inputs
        device = getattr(self.model, "device", None)
        if device is None:
            return inputs
        return inputs.to(device)

    def _generation_kwargs(self) -> Dict[str, Any]:
        return _build_generation_kwargs(
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            stopping_criteria=self.structured_stopping_criteria,
        )

    def _extract_vision_inputs(
        self,
        prepared_messages: List[Dict[str, Any]],
    ) -> Tuple[List[Any], List[Any]]:
        image_inputs: List[Any] = []
        video_inputs: List[Any] = []
        for message in prepared_messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                item_type = item.get("type")
                if item_type == "image" and "image" in item:
                    image_inputs.append(item["image"])
                elif item_type == "video" and "video" in item:
                    video_inputs.append(item["video"])
        return image_inputs, video_inputs

    def _prepare_video_payload(self, video: Any) -> Any:
        if isinstance(video, list):
            return [_to_pil_image(frame) for frame in video]
        if isinstance(video, tuple):
            return [_to_pil_image(frame) for frame in video]
        if isinstance(video, torch.Tensor) and video.ndim == 4:
            return [_to_pil_image(frame) for frame in video]
        return video
