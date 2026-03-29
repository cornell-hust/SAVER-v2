import sys
import unittest
from pathlib import Path

import torch
from PIL import Image


ROOT = Path("/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saver_agent.qwen_policy import QwenGenerationPolicy


class _FakeBatch(dict):
    def __init__(self):
        super().__init__()
        self["input_ids"] = torch.tensor([[11, 12, 13]])
        self.to_device = None

    @property
    def input_ids(self):
        return self["input_ids"]

    def to(self, device):
        self.to_device = device
        return self


class _FakeProcessor:
    def __init__(self):
        self.messages = None
        self.decode_inputs = None

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        return_dict=True,
        return_tensors="pt",
    ):
        self.messages = messages
        return _FakeBatch()

    def batch_decode(self, generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        self.decode_inputs = generated_ids_trimmed
        return ['<think>ok</think><answer>{"existence":"normal"}</answer>']


class _FallbackProcessor:
    def __init__(self):
        self.messages = None
        self.processor_kwargs = None
        self.decode_inputs = None

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.messages = messages
        if tokenize:
            raise TypeError("Old processor path")
        return "PROMPT"

    def __call__(self, **kwargs):
        self.processor_kwargs = kwargs
        return _FakeBatch()

    def batch_decode(self, generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        self.decode_inputs = generated_ids_trimmed
        return ['<answer>{"existence":"anomaly"}</answer>']


class _FakeModel:
    def __init__(self):
        self.device = "cpu"
        self.kwargs = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return torch.tensor([[11, 12, 13, 21, 22]])


class SaverAgentQwenPolicyTests(unittest.TestCase):
    def test_prepare_messages_converts_tensor_images_to_pil(self):
        policy = QwenGenerationPolicy(model=_FakeModel(), processor=_FakeProcessor())
        messages = [
            {
                "role": "tool",
                "content": [
                    {"type": "text", "text": "0.000s"},
                    {"type": "image", "image": torch.zeros(3, 8, 8)},
                ],
            }
        ]

        prepared = policy.prepare_messages(messages)

        self.assertIsInstance(prepared[0]["content"][1]["image"], Image.Image)

    def test_policy_uses_tokenized_chat_template_when_supported(self):
        processor = _FakeProcessor()
        model = _FakeModel()
        policy = QwenGenerationPolicy(model=model, processor=processor, max_new_tokens=32)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "system"}]},
            {"role": "user", "content": [{"type": "text", "text": "user"}]},
        ]

        response = policy(messages, multimodal_cache={}, state=None, step_index=1)

        self.assertEqual(response, '<think>ok</think><answer>{"existence":"normal"}</answer>')
        self.assertEqual(processor.decode_inputs[0].tolist(), [21, 22])
        self.assertEqual(model.kwargs["max_new_tokens"], 32)

    def test_policy_falls_back_to_processor_text_and_images_path(self):
        processor = _FallbackProcessor()
        model = _FakeModel()
        policy = QwenGenerationPolicy(model=model, processor=processor, max_new_tokens=16)
        messages = [
            {
                "role": "tool",
                "content": [
                    {"type": "text", "text": "0.000s"},
                    {"type": "image", "image": torch.zeros(3, 8, 8)},
                    {"type": "text", "text": "Here are selected frames."},
                ],
            }
        ]

        response = policy(messages, multimodal_cache={}, state=None, step_index=1)

        self.assertEqual(response, '<answer>{"existence":"anomaly"}</answer>')
        self.assertEqual(processor.processor_kwargs["text"], "PROMPT")
        self.assertEqual(len(processor.processor_kwargs["images"]), 1)
        self.assertIsInstance(processor.processor_kwargs["images"][0], Image.Image)


if __name__ == "__main__":
    unittest.main()
