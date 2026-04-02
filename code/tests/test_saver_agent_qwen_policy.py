import sys
import unittest
from pathlib import Path

import torch
from PIL import Image


ROOT = Path("/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saver_agent.qwen_policy import QwenGenerationPolicy, _configure_qwen_processor


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
        class _FakeTokenizer:
            def __init__(self):
                self.padding_side = "left"

            def encode(self, text, add_special_tokens=False):
                return [ord(ch) for ch in str(text)]

        self.tokenizer = _FakeTokenizer()
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
        class _FakeTokenizer:
            def __init__(self):
                self.padding_side = "left"

            def encode(self, text, add_special_tokens=False):
                return [ord(ch) for ch in str(text)]

        self.tokenizer = _FakeTokenizer()
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
    def test_configure_qwen_processor_sets_left_padding(self):
        class FakeTokenizer:
            def __init__(self):
                self.padding_side = "right"

        class FakeProcessor:
            def __init__(self):
                self.tokenizer = FakeTokenizer()
                self.padding_side = "right"

        processor = FakeProcessor()

        configured = _configure_qwen_processor(processor)

        self.assertIs(configured, processor)
        self.assertEqual(processor.tokenizer.padding_side, "left")
        self.assertEqual(processor.padding_side, "left")

    def test_generation_kwargs_omit_sampling_parameters_when_sampling_disabled(self):
        policy = QwenGenerationPolicy(
            model=_FakeModel(),
            processor=_FakeProcessor(),
            do_sample=False,
            temperature=0.8,
            top_p=0.9,
            top_k=20,
            repetition_penalty=1.05,
        )

        kwargs = policy._generation_kwargs()

        self.assertFalse(kwargs["do_sample"])
        self.assertNotIn("temperature", kwargs)
        self.assertNotIn("top_p", kwargs)
        self.assertNotIn("top_k", kwargs)
        self.assertEqual(kwargs["repetition_penalty"], 1.05)

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

    def test_prepare_messages_respects_max_total_images_budget(self):
        policy = QwenGenerationPolicy(
            model=_FakeModel(),
            processor=_FakeProcessor(),
            max_total_images=2,
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "0.000s"},
                    {"type": "image", "image": torch.zeros(3, 8, 8)},
                    {"type": "text", "text": "1.000s"},
                    {"type": "image", "image": torch.ones(3, 8, 8)},
                    {"type": "text", "text": "2.000s"},
                    {"type": "image", "image": torch.full((3, 8, 8), 2)},
                    {"type": "text", "text": "decide next tool"},
                ],
            }
        ]

        prepared = policy.prepare_messages(messages)

        image_items = [
            item
            for item in prepared[0]["content"]
            if isinstance(item, dict) and item.get("type") == "image"
        ]
        text_items = [
            item.get("text")
            for item in prepared[0]["content"]
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        self.assertEqual(len(image_items), 2)
        self.assertNotIn("0.000s", text_items)
        self.assertIn("1.000s", text_items)
        self.assertIn("2.000s", text_items)
        self.assertIn("decide next tool", text_items)

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

    def test_policy_uses_structured_stop_and_compacts_verify_tool_call(self):
        class _VerifyProcessor(_FakeProcessor):
            def batch_decode(self, generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False):
                self.decode_inputs = generated_ids_trimmed
                return [
                    '<think>verify</think><tool_call>{"name":"verify_hypothesis","arguments":{"verification_mode":"final_check","claim":{"existence":"anomaly","category":"assault","earliest_alert_sec":0.0},"selected_window_ids":["w0001"],"selected_evidence_ids":["e0001"],"selected_evidence_moment_ids":["m1"],"candidate_window_ids":["w0001","w0002"],"sufficiency_score":0.84,"necessity_score":0.61,"alertability_score":0.72,"counterfactual_faithfulness":0.73,"verification_decision":"sufficient","recommended_action":"finalize","allow_external_verifier_fallback":true,"rationale":"The chosen evidence is sufficient and necessary."}}</tool_call>\nextra trailing text'
                ]

        processor = _VerifyProcessor()
        model = _FakeModel()
        policy = QwenGenerationPolicy(model=model, processor=processor, max_new_tokens=256)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "system"}]},
            {"role": "user", "content": [{"type": "text", "text": "user"}]},
        ]

        response = policy(messages, multimodal_cache={}, state=None, step_index=1)

        self.assertTrue(response.endswith("</tool_call>"))
        self.assertIn('"name":"verify_hypothesis"', response)
        self.assertIn('"selected_window_ids":["w0001"]', response)
        self.assertNotIn("allow_external_verifier_fallback", response)
        self.assertNotIn("candidate_window_ids", response)
        self.assertNotIn("selected_evidence_ids", response)
        self.assertNotIn("extra trailing text", response)
        self.assertIn("stopping_criteria", model.kwargs)
        self.assertTrue(model.kwargs["stopping_criteria"])

    def test_generation_kwargs_include_sampling_parameters_when_sampling_enabled(self):
        policy = QwenGenerationPolicy(
            model=_FakeModel(),
            processor=_FakeProcessor(),
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            top_k=20,
        )

        kwargs = policy._generation_kwargs()

        self.assertTrue(kwargs["do_sample"])
        self.assertEqual(kwargs["temperature"], 0.8)
        self.assertEqual(kwargs["top_p"], 0.9)
        self.assertEqual(kwargs["top_k"], 20)


if __name__ == "__main__":
    unittest.main()
