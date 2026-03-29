import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path("/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from saver_agent.proposal import SiglipFeatureEncoder


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
