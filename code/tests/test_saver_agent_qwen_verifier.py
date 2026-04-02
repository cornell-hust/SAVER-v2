import sys
import unittest
from pathlib import Path

import torch


ROOT = Path("/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saver_agent.qwen_verifier import QwenSelfVerifier


class _FakeBatch(dict):
    def __init__(self):
        super().__init__()
        self["input_ids"] = torch.tensor([[11, 12, 13]])

    @property
    def input_ids(self):
        return self["input_ids"]

    def to(self, device):
        return self


class _FakeProcessor:
    def __init__(self, response_text):
        self.response_text = response_text
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
        return [self.response_text]


class _FakeModel:
    def __init__(self):
        self.device = "cpu"
        self.kwargs = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return torch.tensor([[11, 12, 13, 21, 22]])


class SaverAgentQwenVerifierTests(unittest.TestCase):
    def test_generation_kwargs_omit_sampling_parameters_when_sampling_disabled(self):
        verifier = QwenSelfVerifier(
            model=_FakeModel(),
            processor=_FakeProcessor("{}"),
            do_sample=False,
            temperature=0.7,
            top_p=0.8,
            top_k=16,
            repetition_penalty=1.1,
        )

        kwargs = verifier._generation_kwargs()

        self.assertFalse(kwargs["do_sample"])
        self.assertNotIn("temperature", kwargs)
        self.assertNotIn("top_p", kwargs)
        self.assertNotIn("top_k", kwargs)
        self.assertEqual(kwargs["repetition_penalty"], 1.1)

    def test_qwen_verifier_prompt_requires_plain_json_and_exact_keys(self):
        processor = _FakeProcessor(
            '{"full": {}, "keep": {}, "drop": {}, "alert_prefix": {}}'
        )
        verifier = QwenSelfVerifier(model=_FakeModel(), processor=processor)

        verifier.score_views(
            views={
                "full": {"summary_text": "full summary", "images": [], "timestamps": []},
                "keep": {"summary_text": "keep summary", "images": [], "timestamps": []},
                "drop": {"summary_text": "drop summary", "images": [], "timestamps": []},
                "alert_prefix": {"summary_text": "prefix summary", "images": [], "timestamps": []},
            },
            claim={"existence": "anomaly", "category": "assault"},
            verification_mode="final_check",
            question="Verify the claim.",
        )

        system_text = processor.messages[0]["content"][0]["text"]
        user_text = processor.messages[1]["content"][0]["text"]

        self.assertIn("Begin with { and end with }", system_text)
        self.assertIn('"full"', user_text)
        self.assertIn('"alert_prefix"', user_text)
        self.assertIn("Do not wrap the JSON in markdown fences", system_text)

    def test_qwen_verifier_parses_fenced_json_and_normalizes_missing_scores(self):
        processor = _FakeProcessor(
            """```json
            {
              "full": {"exist_support": 1.0, "overall_support": 0.9},
              "keep": {"exist_support": 0.8, "overall_support": 0.7},
              "drop": {"exist_support": 0.2, "overall_support": 0.1},
              "alert_prefix": {"exist_support": 0.9, "overall_support": 0.8}
            }
            ```"""
        )
        verifier = QwenSelfVerifier(model=_FakeModel(), processor=processor)

        scores = verifier.score_views(
            views={
                "full": {"summary_text": "full summary", "images": [], "timestamps": []},
                "keep": {"summary_text": "keep summary", "images": [], "timestamps": []},
                "drop": {"summary_text": "drop summary", "images": [], "timestamps": []},
                "alert_prefix": {"summary_text": "prefix summary", "images": [], "timestamps": []},
            },
            claim={"existence": "anomaly", "category": "assault"},
            verification_mode="final_check",
            question="Verify the claim.",
        )

        self.assertEqual(scores["full"]["exist_support"], 1.0)
        self.assertIn("category_support", scores["full"])
        self.assertIn("overall_support", scores["alert_prefix"])
        self.assertEqual(processor.decode_inputs[0].tolist(), [21, 22])


if __name__ == "__main__":
    unittest.main()
