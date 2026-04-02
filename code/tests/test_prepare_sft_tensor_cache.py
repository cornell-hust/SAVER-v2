import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


ROOT = Path("/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


try:
    import prepare_sft_tensor_cache as prepare_sft_tensor_cache
except ModuleNotFoundError:
    prepare_sft_tensor_cache = None


class PrepareSFTTensorCacheTests(unittest.TestCase):
    def _write_prepared_data(self, path: Path, *, num_examples: int = 5) -> None:
        rows = []
        for index in range(num_examples):
            rows.append(
                {
                    "schema_version": "saver_agent.sft.v1",
                    "video_id": f"video_{index}",
                    "split": "train",
                    "step_index": index,
                    "target_action": "seek_evidence",
                    "tool_name": "seek_evidence",
                    "target_response": f"response {index}",
                    "sample_weight": 1.0,
                    "messages": [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": "system prompt"}],
                        },
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": f"user turn {index}"}],
                        },
                    ],
                }
            )
        path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _fake_payload(*args, **kwargs):
        return {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            "labels": torch.tensor([[-100, 2, 3]], dtype=torch.long),
        }

    def test_default_mode_preserves_legacy_manifest_and_summary_names(self):
        self.assertIsNotNone(prepare_sft_tensor_cache, "prepare_sft_tensor_cache.py is missing")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prepared_data = root / "prepared.jsonl"
            output_dir = root / "tensor_cache"
            self._write_prepared_data(prepared_data, num_examples=3)

            with patch.object(prepare_sft_tensor_cache, "_load_processor", return_value=object()), patch.object(
                prepare_sft_tensor_cache, "build_processor_signature", return_value="fake_signature"
            ), patch.object(
                prepare_sft_tensor_cache,
                "build_processor_signature_summary",
                return_value={"processor_class": "FakeProcessor"},
            ), patch.object(
                prepare_sft_tensor_cache,
                "materialize_example_for_training",
                side_effect=lambda example, resolver=None: dict(example),
            ), patch.object(
                prepare_sft_tensor_cache,
                "build_sft_tensor_cache_payload",
                side_effect=self._fake_payload,
            ):
                prepare_sft_tensor_cache.main(
                    [
                        "--prepared-data",
                        str(prepared_data),
                        "--output-dir",
                        str(output_dir),
                        "--progress-every",
                        "0",
                    ]
                )

            manifest_path = output_dir / "manifest.jsonl"
            summary_path = output_dir / "summary.json"
            self.assertTrue(manifest_path.exists())
            self.assertTrue(summary_path.exists())
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["num_examples"], 3)
            self.assertEqual(summary["num_built"], 3)
            self.assertEqual(summary["num_skipped_existing"], 0)

    def test_shard_mode_builds_only_selected_examples_and_uses_shard_files(self):
        self.assertIsNotNone(prepare_sft_tensor_cache, "prepare_sft_tensor_cache.py is missing")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prepared_data = root / "prepared.jsonl"
            output_dir = root / "tensor_cache"
            self._write_prepared_data(prepared_data, num_examples=5)

            with patch.object(prepare_sft_tensor_cache, "_load_processor", return_value=object()), patch.object(
                prepare_sft_tensor_cache, "build_processor_signature", return_value="fake_signature"
            ), patch.object(
                prepare_sft_tensor_cache,
                "build_processor_signature_summary",
                return_value={"processor_class": "FakeProcessor"},
            ), patch.object(
                prepare_sft_tensor_cache,
                "materialize_example_for_training",
                side_effect=lambda example, resolver=None: dict(example),
            ), patch.object(
                prepare_sft_tensor_cache,
                "build_sft_tensor_cache_payload",
                side_effect=self._fake_payload,
            ):
                prepare_sft_tensor_cache.main(
                    [
                        "--prepared-data",
                        str(prepared_data),
                        "--output-dir",
                        str(output_dir),
                        "--progress-every",
                        "0",
                        "--num-shards",
                        "2",
                        "--shard-index",
                        "1",
                    ]
                )

            summary_path = output_dir / "summary.shard-1-of-2.json"
            manifest_path = output_dir / "manifest.shard-1-of-2.jsonl"
            legacy_summary_path = output_dir / "summary.json"
            legacy_manifest_path = output_dir / "manifest.jsonl"
            self.assertTrue(summary_path.exists())
            self.assertTrue(manifest_path.exists())
            self.assertFalse(legacy_summary_path.exists())
            self.assertFalse(legacy_manifest_path.exists())

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["num_examples"], 3)
            self.assertEqual(summary["num_examples_total"], 5)
            self.assertEqual(summary["num_built"], 3)
            self.assertEqual(summary["num_skipped_existing"], 0)
            self.assertEqual(summary["num_shards"], 2)
            self.assertEqual(summary["shard_index"], 1)

            manifest_rows = [
                json.loads(line)
                for line in manifest_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(manifest_rows), 3)
            self.assertEqual([row["video_id"] for row in manifest_rows], ["video_2", "video_3", "video_4"])

    def test_shard_mode_respects_skip_existing(self):
        self.assertIsNotNone(prepare_sft_tensor_cache, "prepare_sft_tensor_cache.py is missing")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prepared_data = root / "prepared.jsonl"
            output_dir = root / "tensor_cache"
            self._write_prepared_data(prepared_data, num_examples=4)

            patches = [
                patch.object(prepare_sft_tensor_cache, "_load_processor", return_value=object()),
                patch.object(prepare_sft_tensor_cache, "build_processor_signature", return_value="fake_signature"),
                patch.object(
                    prepare_sft_tensor_cache,
                    "build_processor_signature_summary",
                    return_value={"processor_class": "FakeProcessor"},
                ),
                patch.object(
                    prepare_sft_tensor_cache,
                    "materialize_example_for_training",
                    side_effect=lambda example, resolver=None: dict(example),
                ),
                patch.object(
                    prepare_sft_tensor_cache,
                    "build_sft_tensor_cache_payload",
                    side_effect=self._fake_payload,
                ),
            ]

            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                prepare_sft_tensor_cache.main(
                    [
                        "--prepared-data",
                        str(prepared_data),
                        "--output-dir",
                        str(output_dir),
                        "--progress-every",
                        "0",
                        "--num-shards",
                        "2",
                        "--shard-index",
                        "0",
                    ]
                )
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                prepare_sft_tensor_cache.main(
                    [
                        "--prepared-data",
                        str(prepared_data),
                        "--output-dir",
                        str(output_dir),
                        "--progress-every",
                        "0",
                        "--num-shards",
                        "2",
                        "--shard-index",
                        "0",
                    ]
                )

            summary_path = output_dir / "summary.shard-0-of-2.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["num_examples"], 2)
            self.assertEqual(summary["num_examples_total"], 4)
            self.assertEqual(summary["num_built"], 0)
            self.assertEqual(summary["num_skipped_existing"], 2)

    def test_invalid_shard_index_raises(self):
        self.assertIsNotNone(prepare_sft_tensor_cache, "prepare_sft_tensor_cache.py is missing")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prepared_data = root / "prepared.jsonl"
            output_dir = root / "tensor_cache"
            self._write_prepared_data(prepared_data, num_examples=2)

            with self.assertRaises(ValueError):
                prepare_sft_tensor_cache.main(
                    [
                        "--prepared-data",
                        str(prepared_data),
                        "--output-dir",
                        str(output_dir),
                        "--num-shards",
                        "2",
                        "--shard-index",
                        "2",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
