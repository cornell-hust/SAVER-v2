import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path("/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saver_agent.offline_scoring import ReferenceDataProvider, attach_teacher_judge_to_records, score_rollout_records


class ScoreSaverRolloutCliTests(unittest.TestCase):
    def test_attach_teacher_judge_to_records_annotates_verify_turns(self):
        data_record = {
            "schema_version": "saver_agent.v1",
            "video_id": "sample_rollout_teacher",
            "video_path": "videos/sample_rollout_teacher.mp4",
            "split": "test",
            "video_meta": {"fps": 1.0, "duration_sec": 8.0, "total_frames": 8},
            "agent_task": {"task_prompt": "Determine whether an anomaly exists."},
            "structured_target": {
                "existence": "anomaly",
                "category": "assault",
                "severity": 4,
                "anomaly_interval_sec": [1.0, 6.0],
                "precursor_interval_sec": [0.0, 1.0],
                "earliest_alert_sec": 1.0,
                "counterfactual_type": "remove_actor_interaction",
            },
            "tool_io": {
                "oracle_windows_sec": [
                    {"moment_id": "ev1", "role": "trigger", "window": [1.0, 4.0], "description": "trigger"},
                ],
            },
        }
        rollout = {
            "video_id": "sample_rollout_teacher",
            "dataset_index": 0,
            "terminated_reason": "answered",
            "num_turns": 1,
            "final_answer": {"existence": "anomaly", "category": "assault"},
            "state": {
                "visited_windows": [],
                "evidence_ledger": [],
                "alerts": [],
                "verification_records": [],
                "finalized_case": {"existence": "anomaly", "category": "assault", "earliest_alert_sec": 1.0},
                "last_claim": {"existence": "anomaly", "category": "assault", "earliest_alert_sec": 1.0},
                "active_evidence_window_ids": [],
                "verifier_cache": [],
                "next_evidence_id": 1,
                "next_window_id": 1,
                "next_alert_id": 1,
            },
            "turns": [
                {
                    "step_index": 1,
                    "assistant_response_raw": '<think>verify</think><tool_call>{"name":"verify_hypothesis","arguments":{"verification_mode":"full_keep_drop","claim":{"existence":"anomaly","category":"assault"},"selected_window_ids":["w0001"],"selected_evidence_ids":["e0001"],"selected_evidence_moment_ids":["ev1"],"verification_decision":"sufficient","recommended_action":"finalize","sufficiency_score":0.8,"necessity_score":0.4}}</tool_call>',
                    "action": "tool_call",
                    "valid_action": True,
                    "tool_name": "verify_hypothesis",
                    "self_verification_decision": "sufficient",
                    "self_verification_scores": {
                        "sufficiency": 0.8,
                        "necessity": 0.4,
                        "alertability": 0.7,
                        "counterfactual_faithfulness": 0.6,
                    },
                }
            ],
        }

        class DummyJudge:
            def annotate_example(self, example, *, input_mode=None):
                updated = dict(example)
                updated["teacher_judge_scores"] = {
                    "sufficiency": 0.82,
                    "necessity": 0.42,
                    "alertability": 0.72,
                    "counterfactual_faithfulness": 0.61,
                }
                updated["teacher_judge_decision"] = "sufficient"
                updated["teacher_judge_rationale"] = "Teacher agrees."
                return updated

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data.jsonl"
            data_path.write_text(json.dumps(data_record) + "\n", encoding="utf-8")
            reference_data = ReferenceDataProvider(data_path=data_path, data_root=tmpdir)

            annotated, summary = attach_teacher_judge_to_records(
                [rollout],
                reference_data=reference_data,
                judge=DummyJudge(),
                input_mode="text_only",
            )

        self.assertEqual(summary["num_teacher_judge_candidates"], 1)
        self.assertEqual(summary["num_teacher_judge_annotated_turns"], 1)
        verify_turn = annotated[0]["turns"][0]
        self.assertEqual(verify_turn["teacher_judge_decision"], "sufficient")
        self.assertGreater(verify_turn["teacher_judge_reward"], 0.0)

    def test_cli_scores_jsonl_rollouts_and_attaches_offline_verifier(self):
        rollout = {
            "video_id": "sample_rollout",
            "terminated_reason": "answered",
            "num_turns": 2,
            "final_answer": {"existence": "anomaly", "category": "assault"},
            "state": {
                "visited_windows": [
                    {
                        "window_id": "w0001",
                        "evidence_id": "e0001",
                        "kind": "evidence",
                        "query": "possible assault",
                        "start_sec": 1.0,
                        "end_sec": 4.0,
                        "selected_timestamps": [1.0, 2.0, 3.0],
                        "selected_frame_count": 3,
                    }
                ],
                "evidence_ledger": [
                    {
                        "window_id": "w0001",
                        "evidence_id": "e0001",
                        "kind": "evidence",
                        "query": "possible assault",
                        "start_sec": 1.0,
                        "end_sec": 4.0,
                        "selected_timestamps": [1.0, 2.0, 3.0],
                        "selected_frame_count": 3,
                    }
                ],
                "alerts": [{"alert_id": "a0001", "decision": "hard_alert", "alert_sec": 1.2}],
                "verification_records": [],
                "finalized_case": {"existence": "anomaly", "category": "assault", "earliest_alert_sec": 1.0},
                "last_claim": {"existence": "anomaly", "category": "assault", "earliest_alert_sec": 1.0},
                "active_evidence_window_ids": ["w0001"],
                "verifier_cache": [],
                "next_evidence_id": 2,
                "next_window_id": 2,
                "next_alert_id": 2,
            },
            "turns": [],
        }
        data_record = {
            "schema_version": "saver_agent.v1",
            "video_id": "sample_rollout",
            "video_path": "videos/sample_rollout.mp4",
            "video_meta": {"fps": 1.0, "duration_sec": 8.0, "total_frames": 8},
            "agent_task": {"task_prompt": "Determine whether an anomaly exists."},
            "structured_target": {
                "existence": "anomaly",
                "category": "assault",
                "severity": 4,
                "anomaly_interval_sec": [1.0, 6.0],
                "precursor_interval_sec": [0.0, 1.0],
                "earliest_alert_sec": 1.0,
                "counterfactual_type": "remove_actor_interaction",
            },
            "tool_io": {
                "oracle_windows_sec": [
                    {"moment_id": "ev1", "role": "trigger", "window": [1.0, 4.0], "description": "trigger"},
                    {"moment_id": "ev2", "role": "peak_action", "window": [4.0, 6.0], "description": "peak"},
                ],
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_path = tmpdir / "rollouts.jsonl"
            data_path = tmpdir / "data.jsonl"
            output_path = tmpdir / "scored.jsonl"
            input_path.write_text(json.dumps(rollout) + "\n", encoding="utf-8")
            data_path.write_text(json.dumps(data_record) + "\n", encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "score_saver_rollout.py"),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--data",
                    str(data_path),
                    "--attach-reference-offline-verifier",
                    "--verifier-backend",
                    "heuristic",
                    "--force-reverify",
                ],
                check=True,
            )

            lines = output_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            scored = json.loads(lines[0])

        self.assertIn("reward_summary", scored)
        self.assertIn("offline_verifier", scored)
        self.assertEqual(scored["offline_verifier"]["primary_status"], "complete")
        self.assertGreater(scored["reward_summary"]["total_reward"], 0.0)

    def test_score_rollout_records_does_not_attach_reference_verifier_by_default(self):
        rollout = {
            "video_id": "sample_rollout_no_diag",
            "terminated_reason": "answered",
            "num_turns": 1,
            "final_answer": {"existence": "anomaly", "category": "assault"},
            "state": {
                "visited_windows": [],
                "evidence_ledger": [],
                "alerts": [],
                "verification_records": [],
                "finalized_case": {"existence": "anomaly", "category": "assault"},
                "last_claim": {"existence": "anomaly", "category": "assault"},
                "active_evidence_window_ids": [],
                "verifier_cache": [],
                "next_evidence_id": 1,
                "next_window_id": 1,
                "next_alert_id": 1,
            },
            "turns": [],
        }

        scored = score_rollout_records([rollout])

        self.assertNotIn("offline_verifier", scored[0])
        self.assertFalse(scored[0]["scoring_metadata"]["attach_offline_verifier"])

    def test_cli_accepts_directory_of_jsonl_shards(self):
        rollout_a = {
            "video_id": "sample_rollout_a",
            "terminated_reason": "answered",
            "num_turns": 2,
            "final_answer": {"existence": "anomaly", "category": "assault"},
            "state": {
                "visited_windows": [
                    {
                        "window_id": "w0001",
                        "evidence_id": "e0001",
                        "kind": "evidence",
                        "query": "possible assault",
                        "start_sec": 1.0,
                        "end_sec": 4.0,
                        "selected_timestamps": [1.0, 2.0],
                        "selected_frame_count": 2,
                    }
                ],
                "evidence_ledger": [
                    {
                        "window_id": "w0001",
                        "evidence_id": "e0001",
                        "kind": "evidence",
                        "query": "possible assault",
                        "start_sec": 1.0,
                        "end_sec": 4.0,
                        "selected_timestamps": [1.0, 2.0],
                        "selected_frame_count": 2,
                    }
                ],
                "alerts": [{"alert_id": "a0001", "decision": "hard_alert", "alert_sec": 1.2}],
                "verification_records": [],
                "finalized_case": {"existence": "anomaly", "category": "assault", "earliest_alert_sec": 1.0},
                "last_claim": {"existence": "anomaly", "category": "assault", "earliest_alert_sec": 1.0},
                "active_evidence_window_ids": ["w0001"],
                "verifier_cache": [],
                "next_evidence_id": 2,
                "next_window_id": 2,
                "next_alert_id": 2,
            },
            "turns": [],
        }
        rollout_b = {
            "video_id": "sample_rollout_b",
            "terminated_reason": "answered",
            "num_turns": 1,
            "final_answer": {"existence": "normal", "category": "normal"},
            "state": {
                "visited_windows": [],
                "evidence_ledger": [],
                "alerts": [],
                "verification_records": [],
                "finalized_case": {"existence": "normal", "category": "normal"},
                "last_claim": {"existence": "normal", "category": "normal"},
                "active_evidence_window_ids": [],
                "verifier_cache": [],
                "next_evidence_id": 1,
                "next_window_id": 1,
                "next_alert_id": 1,
            },
            "turns": [],
        }
        data_records = [
            {
                "schema_version": "saver_agent.v1",
                "video_id": "sample_rollout_a",
                "video_path": "videos/sample_rollout_a.mp4",
                "video_meta": {"fps": 1.0, "duration_sec": 8.0, "total_frames": 8},
                "agent_task": {"task_prompt": "Determine whether an anomaly exists."},
                "structured_target": {
                    "existence": "anomaly",
                    "category": "assault",
                    "severity": 4,
                    "anomaly_interval_sec": [1.0, 6.0],
                    "precursor_interval_sec": [0.0, 1.0],
                    "earliest_alert_sec": 1.0,
                    "counterfactual_type": "remove_actor_interaction",
                },
                "tool_io": {
                    "oracle_windows_sec": [
                        {"moment_id": "ev1", "role": "trigger", "window": [1.0, 4.0], "description": "trigger"},
                    ],
                },
            },
            {
                "schema_version": "saver_agent.v1",
                "video_id": "sample_rollout_b",
                "video_path": "videos/sample_rollout_b.mp4",
                "video_meta": {"fps": 1.0, "duration_sec": 8.0, "total_frames": 8},
                "agent_task": {"task_prompt": "Determine whether an anomaly exists."},
                "structured_target": {
                    "existence": "normal",
                    "category": "normal",
                    "severity": 0,
                    "anomaly_interval_sec": None,
                    "precursor_interval_sec": None,
                    "earliest_alert_sec": None,
                    "counterfactual_type": "none",
                },
                "tool_io": {"oracle_windows_sec": []},
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_dir = tmpdir / "rollout_shards"
            input_dir.mkdir(parents=True, exist_ok=True)
            data_path = tmpdir / "data.jsonl"
            output_path = tmpdir / "scored.jsonl"
            (input_dir / "part0.jsonl").write_text(json.dumps(rollout_a) + "\n", encoding="utf-8")
            (input_dir / "part1.jsonl").write_text(json.dumps(rollout_b) + "\n", encoding="utf-8")
            with data_path.open("w", encoding="utf-8") as f:
                for record in data_records:
                    f.write(json.dumps(record) + "\n")

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "score_saver_rollout.py"),
                    "--input",
                    str(input_dir),
                    "--output",
                    str(output_path),
                    "--data",
                    str(data_path),
                    "--verifier-backend",
                    "heuristic",
                    "--force-reverify",
                ],
                check=True,
            )

            lines = output_path.read_text(encoding="utf-8").strip().splitlines()

        self.assertEqual(len(lines), 2)


if __name__ == "__main__":
    unittest.main()
