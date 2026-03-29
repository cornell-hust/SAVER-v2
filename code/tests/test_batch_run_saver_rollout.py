import argparse
import sys
import unittest
from pathlib import Path


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

        self.assertEqual(args.max_turns, 12)

    def test_resolve_dataset_indices_uses_remaining_tail_when_count_is_zero(self):
        args = argparse.Namespace(indices="", start_index=2, count=0)

        indices = batch_run_saver_rollout._resolve_dataset_indices(args, dataset_size=5)

        self.assertEqual(indices, [2, 3, 4])


if __name__ == "__main__":
    unittest.main()
