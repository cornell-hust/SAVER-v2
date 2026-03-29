import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path("/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code")


class SummarizeSaverScoresCliTests(unittest.TestCase):
    def test_cli_summarizes_primary_and_alert_status_counts_with_means(self):
        records = [
            {
                "video_id": "a",
                "num_turns": 3,
                "offline_verifier": {
                    "primary_status": "complete",
                    "alert_status": "justified",
                },
                "reward_summary": {
                    "total_reward": 2.0,
                    "components": {"verification_reward": 1.0, "alert_reward": 1.0},
                },
            },
            {
                "video_id": "b",
                "num_turns": 5,
                "offline_verifier": {
                    "primary_status": "incomplete",
                    "alert_status": "premature",
                },
                "reward_summary": {
                    "total_reward": -0.5,
                    "components": {"verification_reward": -0.2, "alert_reward": -0.8},
                },
            },
            {
                "video_id": "c",
                "num_turns": 4,
                "turns": [
                    {
                        "tool_name": "verify_hypothesis",
                        "verifier_primary_status": "redundant",
                        "verifier_alert_status": "late",
                    }
                ],
                "reward_summary": {
                    "total_reward": 0.5,
                    "components": {"verification_reward": 0.3, "alert_reward": -0.3},
                },
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_path = tmpdir / "scored.jsonl"
            output_path = tmpdir / "summary.json"
            with input_path.open("w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "summarize_saver_scores.py"),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                ],
                check=True,
            )

            summary = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(summary["num_records"], 3)
        self.assertEqual(summary["primary_status_counts"]["complete"], 1)
        self.assertEqual(summary["primary_status_counts"]["incomplete"], 1)
        self.assertEqual(summary["primary_status_counts"]["redundant"], 1)
        self.assertEqual(summary["alert_status_counts"]["justified"], 1)
        self.assertEqual(summary["alert_status_counts"]["premature"], 1)
        self.assertEqual(summary["alert_status_counts"]["late"], 1)
        self.assertAlmostEqual(summary["mean_total_reward"], (2.0 - 0.5 + 0.5) / 3.0, places=6)
        self.assertAlmostEqual(summary["mean_num_turns"], 4.0, places=6)

    def test_cli_summarizes_directory_of_jsonl_shards(self):
        shard_a = [
            {
                "video_id": "a",
                "num_turns": 2,
                "offline_verifier": {"primary_status": "complete", "alert_status": "justified"},
                "reward_summary": {"total_reward": 1.5, "components": {"verification_reward": 1.0}},
            }
        ]
        shard_b = [
            {
                "video_id": "b",
                "num_turns": 6,
                "offline_verifier": {"primary_status": "misaligned", "alert_status": "not_applicable"},
                "reward_summary": {"total_reward": -1.0, "components": {"verification_reward": -1.0}},
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_dir = tmpdir / "scored_shards"
            input_dir.mkdir(parents=True, exist_ok=True)
            output_path = tmpdir / "summary.json"
            for name, records in {"part0.jsonl": shard_a, "part1.jsonl": shard_b}.items():
                with (input_dir / name).open("w", encoding="utf-8") as f:
                    for record in records:
                        f.write(json.dumps(record) + "\n")

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "summarize_saver_scores.py"),
                    "--input",
                    str(input_dir),
                    "--output",
                    str(output_path),
                ],
                check=True,
            )

            summary = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(summary["num_records"], 2)
        self.assertEqual(summary["primary_status_counts"]["complete"], 1)
        self.assertEqual(summary["primary_status_counts"]["misaligned"], 1)
        self.assertAlmostEqual(summary["mean_total_reward"], 0.25, places=6)


if __name__ == "__main__":
    unittest.main()
