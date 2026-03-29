from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


DEFAULT_MODEL_PATH = os.environ.get(
    "SAVER_QWEN_MODEL_PATH",
    "/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/MLLMs/qwen3-vl-8b-Instruct",
)


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


class QwenGenerationPolicy:
    """Single-turn Qwen generation policy for SAVER rollouts."""

    def __init__(
        self,
        *,
        model: Any,
        processor: Any,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
    ):
        self.model = model
        self.processor = processor
        self.max_new_tokens = int(max_new_tokens)
        self.do_sample = bool(do_sample)
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty

    @classmethod
    def from_components(
        cls,
        *,
        model: Any,
        processor: Any,
        max_new_tokens: int = 512,
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

        model = Qwen3VLForConditionalGeneration.from_pretrained(str(model_path), **model_init_kwargs)
        model.eval()
        processor = AutoProcessor.from_pretrained(str(model_path))
        return cls(
            model=model,
            processor=processor,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )

    def prepare_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
        return output_text[0]

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
        kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.top_k is not None:
            kwargs["top_k"] = self.top_k
        if self.repetition_penalty is not None:
            kwargs["repetition_penalty"] = self.repetition_penalty
        return kwargs

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
