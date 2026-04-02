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
                    "components": {"verification_reward": 1.0, "alert_reward": 1.0, "teacher_judge_reward": 0.6},
                },
                "turns": [
                    {
                        "tool_name": "verify_hypothesis",
                        "teacher_judge_decision": "sufficient",
                        "teacher_judge_alignment": 1.0,
                        "teacher_judge_reward": 0.6,
                    }
                ],
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
                    "components": {"verification_reward": -0.2, "alert_reward": -0.8, "teacher_judge_reward": -0.4},
                },
                "turns": [
                    {
                        "tool_name": "verify_hypothesis",
                        "teacher_judge_decision": "misaligned",
                        "teacher_judge_alignment": 0.0,
                        "teacher_judge_reward": -0.4,
                    }
                ],
            },
            {
                "video_id": "c",
                "num_turns": 4,
                "turns": [
                    {
                        "tool_name": "verify_hypothesis",
                        "verifier_primary_status": "redundant",
                        "verifier_alert_status": "late",
                        "teacher_judge_decision": "redundant",
                        "teacher_judge_alignment": 1.0,
                        "teacher_judge_reward": 0.2,
                    }
                ],
                "reward_summary": {
                    "total_reward": 0.5,
                    "components": {"verification_reward": 0.3, "alert_reward": -0.3, "teacher_judge_reward": 0.2},
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
        self.assertEqual(summary["teacher_judge_decision_counts"]["sufficient"], 1)
        self.assertEqual(summary["teacher_judge_decision_counts"]["misaligned"], 1)
        self.assertEqual(summary["teacher_judge_decision_counts"]["redundant"], 1)
        self.assertEqual(summary["teacher_judge_alignment_counts"]["aligned"], 2)
        self.assertEqual(summary["teacher_judge_alignment_counts"]["misaligned"], 1)
        self.assertAlmostEqual(summary["mean_teacher_judge_reward"], (0.6 - 0.4 + 0.2) / 3.0, places=6)
        self.assertAlmostEqual(summary["mean_total_reward"], (2.0 - 0.5 + 0.5) / 3.0, places=6)
        self.assertAlmostEqual(summary["mean_num_turns"], 4.0, places=6)

    def test_cli_reports_verify_coverage_disagreement_cases_and_label_grouped_teacher_stats(self):
        records = [
            {
                "video_id": "anom_a",
                "split": "test",
                "num_turns": 4,
                "turns": [
                    {
                        "step_index": 2,
                        "tool_name": "verify_hypothesis",
                        "self_verification_decision": "sufficient",
                        "teacher_judge_decision": "sufficient",
                        "teacher_judge_alignment": 1.0,
                        "teacher_judge_reward": 0.7,
                    }
                ],
                "reward_summary": {
                    "total_reward": 1.4,
                    "components": {"teacher_judge_reward": 0.7},
                },
            },
            {
                "video_id": "anom_b",
                "split": "test",
                "num_turns": 5,
                "turns": [
                    {
                        "step_index": 3,
                        "tool_name": "verify_hypothesis",
                        "self_verification_decision": "sufficient",
                        "teacher_judge_decision": "misaligned",
                        "teacher_judge_alignment": 0.0,
                        "teacher_judge_reward": -0.5,
                    }
                ],
                "reward_summary": {
                    "total_reward": 0.2,
                    "components": {"teacher_judge_reward": -0.5},
                },
            },
            {
                "video_id": "normal_a",
                "split": "test",
                "num_turns": 2,
                "turns": [],
                "reward_summary": {
                    "total_reward": 0.1,
                    "components": {},
                },
            },
        ]
        data_records = [
            {
                "video_id": "anom_a",
                "split": "test",
                "label": {"is_anomaly": True, "category": "assault"},
                "structured_target": {"existence": "anomaly", "category": "assault"},
            },
            {
                "video_id": "anom_b",
                "split": "test",
                "label": {"is_anomaly": True, "category": "fire"},
                "structured_target": {"existence": "anomaly", "category": "fire"},
            },
            {
                "video_id": "normal_a",
                "split": "test",
                "label": {"is_anomaly": False, "category": "normal"},
                "structured_target": {"existence": "normal", "category": "normal"},
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_path = tmpdir / "scored.jsonl"
            data_path = tmpdir / "data.jsonl"
            output_path = tmpdir / "summary.json"
            with input_path.open("w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")
            with data_path.open("w", encoding="utf-8") as f:
                for record in data_records:
                    f.write(json.dumps(record) + "\n")

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "summarize_saver_scores.py"),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--data",
                    str(data_path),
                    "--max-teacher-disagreement-cases",
                    "10",
                ],
                check=True,
            )

            summary = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(summary["verify_turn_coverage"]["records_with_verify_turn"], 2)
        self.assertAlmostEqual(summary["verify_turn_coverage"]["record_ratio"], 2.0 / 3.0, places=6)
        self.assertEqual(len(summary["teacher_disagreement_cases"]), 1)
        self.assertEqual(summary["teacher_disagreement_cases"][0]["video_id"], "anom_b")
        grouped = summary["teacher_reward_by_label_group"]
        self.assertEqual(grouped["anomaly"]["num_records"], 2)
        self.assertEqual(grouped["normal"]["num_records"], 1)
        self.assertAlmostEqual(grouped["anomaly"]["mean_teacher_judge_reward"], 0.1, places=6)
        self.assertAlmostEqual(grouped["normal"]["mean_teacher_judge_reward"], 0.0, places=6)

    def test_cli_clusters_teacher_disagreements_by_category_and_stage(self):
        records = [
            {
                "video_id": "fire_alert",
                "split": "test",
                "num_turns": 4,
                "turns": [
                    {
                        "step_index": 2,
                        "tool_name": "verify_hypothesis",
                        "verification_mode": "soft_alert_check",
                        "self_verification_decision": "sufficient",
                        "teacher_judge_decision": "misaligned",
                        "teacher_judge_alignment": 0.0,
                        "teacher_judge_reward": -0.6,
                    }
                ],
                "reward_summary": {"total_reward": 0.0, "components": {"teacher_judge_reward": -0.6}},
            },
            {
                "video_id": "fire_evidence",
                "split": "test",
                "num_turns": 5,
                "turns": [
                    {
                        "step_index": 3,
                        "tool_name": "verify_hypothesis",
                        "verification_mode": "full_keep_drop",
                        "self_verification_decision": "sufficient",
                        "teacher_judge_decision": "insufficient",
                        "teacher_judge_alignment": 0.0,
                        "teacher_judge_reward": -0.3,
                    }
                ],
                "reward_summary": {"total_reward": 0.0, "components": {"teacher_judge_reward": -0.3}},
            },
            {
                "video_id": "assault_evidence",
                "split": "test",
                "num_turns": 5,
                "turns": [
                    {
                        "step_index": 4,
                        "tool_name": "verify_hypothesis",
                        "verification_mode": "final_check",
                        "self_verification_decision": "sufficient",
                        "teacher_judge_decision": "redundant",
                        "teacher_judge_alignment": 0.0,
                        "teacher_judge_reward": -0.2,
                    }
                ],
                "reward_summary": {"total_reward": 0.0, "components": {"teacher_judge_reward": -0.2}},
            },
        ]
        data_records = [
            {
                "video_id": "fire_alert",
                "split": "test",
                "label": {"is_anomaly": True, "category": "fire"},
                "structured_target": {"existence": "anomaly", "category": "fire"},
            },
            {
                "video_id": "fire_evidence",
                "split": "test",
                "label": {"is_anomaly": True, "category": "fire"},
                "structured_target": {"existence": "anomaly", "category": "fire"},
            },
            {
                "video_id": "assault_evidence",
                "split": "test",
                "label": {"is_anomaly": True, "category": "assault"},
                "structured_target": {"existence": "anomaly", "category": "assault"},
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_path = tmpdir / "scored.jsonl"
            data_path = tmpdir / "data.jsonl"
            output_path = tmpdir / "summary.json"
            with input_path.open("w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")
            with data_path.open("w", encoding="utf-8") as f:
                for record in data_records:
                    f.write(json.dumps(record) + "\n")

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "summarize_saver_scores.py"),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--data",
                    str(data_path),
                    "--max-teacher-disagreement-cases",
                    "10",
                ],
                check=True,
            )

            summary = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(summary["teacher_disagreement_case_total"], 3)
        fire_case = next(case for case in summary["teacher_disagreement_cases"] if case["video_id"] == "fire_alert")
        self.assertEqual(fire_case["category"], "fire")
        self.assertEqual(fire_case["stage"], "alert_verification")

        by_category = {row["category"]: row for row in summary["teacher_disagreement_by_category"]}
        self.assertEqual(by_category["fire"]["num_cases"], 2)
        self.assertEqual(by_category["assault"]["num_cases"], 1)

        by_stage = {row["stage"]: row for row in summary["teacher_disagreement_by_stage"]}
        self.assertEqual(by_stage["alert_verification"]["num_cases"], 1)
        self.assertEqual(by_stage["evidence_verification"]["num_cases"], 2)

        by_pair = {
            (row["category"], row["stage"]): row for row in summary["teacher_disagreement_by_category_stage"]
        }
        self.assertEqual(by_pair[("fire", "alert_verification")]["num_cases"], 1)
        self.assertEqual(by_pair[("fire", "evidence_verification")]["num_cases"], 1)
        self.assertEqual(by_pair[("assault", "evidence_verification")]["num_cases"], 1)

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

    def test_cli_reports_verify_health_stats_for_legacy_compatibility_and_invalid_selection(self):
        records = [
            {
                "video_id": "legacy_case",
                "num_turns": 3,
                "turns": [
                    {
                        "step_index": 2,
                        "tool_name": "verify_hypothesis",
                        "valid_action": False,
                        "verification_parse_mode": "legacy_compatibility",
                        "legacy_compatibility_used": True,
                        "invalid_selected_window_ids": ["window_0"],
                        "selection_resolution_source": "unresolved",
                        "verifier_failure_reasons": ["selected_evidence_not_resolved_to_known_windows"],
                    }
                ],
                "reward_summary": {"total_reward": -0.2, "components": {}},
            }
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

        self.assertEqual(summary["verify_parse_mode_counts"]["legacy_compatibility"], 1)
        self.assertEqual(summary["legacy_compatibility_used_count"], 1)
        self.assertAlmostEqual(summary["invalid_selected_window_rate"], 1.0, places=6)
        self.assertAlmostEqual(summary["unresolved_selection_rate"], 1.0, places=6)
        self.assertAlmostEqual(summary["verify_invalid_turn_rate"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
