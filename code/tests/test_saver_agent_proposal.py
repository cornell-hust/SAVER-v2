import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path("/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from saver_agent.proposal import (
    SiglipFeatureEncoder,
    feature_guided_frame_proposal,
    normalize_query_package,
    render_query_package_texts,
    summarize_query_package,
)


class _FakeProcessor:
    def __call__(self, *, images=None, text=None, return_tensors="pt", padding=False):
        if images is not None:
            batch = len(images)
            return {"pixel_values": torch.ones((batch, 3, 4, 4), dtype=torch.float32)}
        if text is not None:
            batch = len(text)
            return {
                "input_ids": torch.ones((batch, 4), dtype=torch.long),
                "attention_mask": torch.ones((batch, 4), dtype=torch.long),
            }
        raise ValueError("Expected either images or text input.")


class _FakeModelReturningObjectFromGetImageFeatures:
    def get_image_features(self, **inputs):
        batch = int(inputs["pixel_values"].shape[0])
        features = torch.arange(batch * 4, dtype=torch.float32).reshape(batch, 4) + 1.0
        return SimpleNamespace(pooler_output=features)

    def get_text_features(self, **inputs):
        batch = int(inputs["input_ids"].shape[0])
        features = torch.arange(batch * 4, dtype=torch.float32).reshape(batch, 4) + 1.0
        return SimpleNamespace(pooler_output=features)


class SiglipFeatureEncoderTests(unittest.TestCase):
    def test_encode_images_accepts_model_output_object_from_get_image_features(self):
        encoder = SiglipFeatureEncoder(
            model=_FakeModelReturningObjectFromGetImageFeatures(),
            processor=_FakeProcessor(),
            device="cpu",
            model_name="fake_siglip",
        )
        images = torch.randint(0, 255, (2, 3, 8, 8), dtype=torch.uint8)

        features = encoder.encode_images(images)

        self.assertIsInstance(features, torch.Tensor)
        self.assertEqual(tuple(features.shape), (2, 4))

    def test_encode_texts_reuses_cached_embeddings_for_repeated_queries(self):
        class _CountingTextModel:
            def __init__(self):
                self.text_calls = 0

            def get_text_features(self, **inputs):
                self.text_calls += 1
                batch = int(inputs["input_ids"].shape[0])
                features = torch.arange(batch * 4, dtype=torch.float32).reshape(batch, 4) + 1.0
                return SimpleNamespace(pooler_output=features)

        model = _CountingTextModel()
        encoder = SiglipFeatureEncoder(
            model=model,
            processor=_FakeProcessor(),
            device="cpu",
            model_name="fake_siglip",
        )

        first = encoder.encode_texts(["person in red shirt", "physical struggle"])
        second = encoder.encode_texts(["person in red shirt", "physical struggle"])

        self.assertEqual(model.text_calls, 1)
        self.assertTrue(torch.equal(first, second))


class QueryPackageProposalTests(unittest.TestCase):
    def test_render_query_package_texts_preserves_structured_fields(self):
        query_package = normalize_query_package(
            {
                "event_cue": "decisive aggressive contact",
                "key_objects": ["person in red shirt", "person in black shirt"],
                "scene_context": "street sidewalk",
                "hypothesis": "suspected interpersonal violence",
                "negative_constraints": ["no routine walking only"],
                "rewrite_reason": "focus_trigger",
            }
        )

        rendered = render_query_package_texts(query_package)
        summary = summarize_query_package(query_package)

        self.assertGreaterEqual(len(rendered["positive_texts"]), 4)
        self.assertIn("decisive aggressive contact", [entry["text"] for entry in rendered["positive_texts"]])
        self.assertIn("person in red shirt", [entry["text"] for entry in rendered["positive_texts"]])
        self.assertIn("street sidewalk", [entry["text"] for entry in rendered["positive_texts"]])
        self.assertIn("no routine walking only", [entry["text"] for entry in rendered["negative_texts"]])
        self.assertIn("decisive aggressive contact", summary)

    def test_feature_guided_frame_proposal_uses_query_package_and_returns_dpp_metadata(self):
        class _FakeRuntime:
            def __init__(self):
                self.mapping = {
                    "decisive aggressive contact": torch.tensor([1.0, 0.0], dtype=torch.float32),
                    "person in red shirt": torch.tensor([1.0, 0.0], dtype=torch.float32),
                    "person in black shirt": torch.tensor([0.0, 1.0], dtype=torch.float32),
                    "street sidewalk": torch.tensor([0.2, 0.8], dtype=torch.float32),
                    "suspected interpersonal violence": torch.tensor([0.8, 0.2], dtype=torch.float32),
                    "no routine walking only": torch.tensor([0.0, 1.0], dtype=torch.float32),
                }

            def encode_texts(self, texts):
                return torch.stack([self.mapping[text] for text in texts], dim=0)

        feature_cache = {
            "fps": 1.0,
            "frame_indices": list(range(6)),
            "timestamps_sec": [float(i) for i in range(6)],
            "embeddings": torch.tensor(
                [
                    [1.0, 0.0],
                    [0.98, 0.02],
                    [0.70, 0.30],
                    [0.15, 0.85],
                    [0.02, 0.98],
                    [0.00, 1.00],
                ],
                dtype=torch.float32,
            ),
            "normalized": True,
        }
        query_package = {
            "event_cue": "decisive aggressive contact",
            "key_objects": ["person in red shirt", "person in black shirt"],
            "scene_context": "street sidewalk",
            "hypothesis": "suspected interpersonal violence",
            "negative_constraints": ["no routine walking only"],
            "rewrite_reason": "focus_trigger",
        }

        metadata = feature_guided_frame_proposal(
            feature_cache=feature_cache,
            proposal_runtime=_FakeRuntime(),
            query="decisive aggressive contact",
            query_package=query_package,
            start_sec=0.0,
            end_sec=5.0,
            fps=1.0,
            num_frames=0,
            top_k_candidates=6,
            candidate_merge_gap_sec=1.0,
            query_source="model",
        )

        self.assertEqual(metadata["proposal_backend"], "siglip_dpp")
        self.assertEqual(metadata["query_package"]["rewrite_reason"], "focus_trigger")
        self.assertGreaterEqual(metadata["adaptive_num_frames"], 2)
        self.assertEqual(len(metadata["selected_frame_indices"]), metadata["adaptive_num_frames"])
        self.assertIn("positive_texts", metadata["query_rendering"])
        self.assertIn("proposal_dpp_kernel_meta", metadata)

    def test_encode_texts_accepts_model_output_object_from_get_text_features(self):
        encoder = SiglipFeatureEncoder(
            model=_FakeModelReturningObjectFromGetImageFeatures(),
            processor=_FakeProcessor(),
            device="cpu",
            model_name="fake_siglip",
        )

        features = encoder.encode_texts(["person in red shirt", "physical struggle"])

        self.assertIsInstance(features, torch.Tensor)
        self.assertEqual(tuple(features.shape), (2, 4))


if __name__ == "__main__":
    unittest.main()
