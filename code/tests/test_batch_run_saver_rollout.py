import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path("/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import batch_run_saver_rollout as batch_run_saver_rollout


class BatchRunSaverRolloutTests(unittest.TestCase):
    def test_parse_args_defaults_max_turns_to_twelve(self):
        args = batch_run_saver_rollout.parse_args(
            [
                "--data",
                "/tmp/data.jsonl",
                "--output",
                "/tmp/out.jsonl",
                "--model-path",
                "/tmp/model",
            ]
        )

        self.assertEqual(args.max_turns, 14)
        self.assertEqual(args.max_new_tokens, 256)
        self.assertEqual(args.max_total_images, 24)
        self.assertEqual(args.max_image_side, 0)
        self.assertEqual(args.max_image_pixels, 0)

    def test_build_qwen_policy_passes_resize_budgets(self):
        args = batch_run_saver_rollout.parse_args(
            [
                "--data",
                "/tmp/data.jsonl",
                "--output",
                "/tmp/out.jsonl",
                "--model-path",
                "/tmp/model",
                "--max-image-side",
                "640",
                "--max-image-pixels",
                "307200",
            ]
        )
        runtime = batch_run_saver_rollout.distributed_runtime_from_env()

        with patch("batch_run_saver_rollout.QwenGenerationPolicy.from_pretrained") as from_pretrained:
            batch_run_saver_rollout._build_qwen_policy(args, runtime=runtime)

        _, kwargs = from_pretrained.call_args
        self.assertEqual(kwargs["max_image_side"], 640)
        self.assertEqual(kwargs["max_image_pixels"], 307200)

    def test_resolve_dataset_indices_uses_remaining_tail_when_count_is_zero(self):
        args = argparse.Namespace(indices="", start_index=2, count=0)

        indices = batch_run_saver_rollout._resolve_dataset_indices(args, dataset_size=5)

        self.assertEqual(indices, [2, 3, 4])

    def test_merge_sharded_outputs_writes_base_jsonl_in_dataset_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_output = Path(tmpdir) / "rollouts.raw.jsonl"
            shard0 = batch_run_saver_rollout.sharded_output_path(base_output, num_shards=2, shard_index=0)
            shard1 = batch_run_saver_rollout.sharded_output_path(base_output, num_shards=2, shard_index=1)
            shard0.write_text(
                '{"dataset_index": 0, "video_id": "a"}\n{"dataset_index": 2, "video_id": "c"}\n',
                encoding="utf-8",
            )
            shard1.write_text(
                '{"dataset_index": 1, "video_id": "b"}\n{"dataset_index": 3, "video_id": "d"}\n',
                encoding="utf-8",
            )

            merged = batch_run_saver_rollout._merge_sharded_outputs(base_output, num_shards=2)

            self.assertEqual([record["dataset_index"] for record in merged], [0, 1, 2, 3])
            self.assertTrue(base_output.exists())
            merged_lines = [line for line in base_output.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(merged_lines), 4)


if __name__ == "__main__":
    unittest.main()
