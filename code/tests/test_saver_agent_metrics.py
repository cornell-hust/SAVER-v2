import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path("/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from saver_agent.metrics import summarize_saver_metrics
from saver_agent.offline_scoring import ReferenceDataProvider


class SaverAgentMetricsTests(unittest.TestCase):
    def test_summarize_saver_metrics_computes_core_saver_metrics(self):
        data_records = [
            {
                "video_id": "anomaly_case",
                "video_meta": {"fps": 1.0, "duration_sec": 8.0, "total_frames": 8},
                "structured_target": {
                    "existence": "anomaly",
                    "category": "assault",
                    "severity": 4,
                    "hard_normal": False,
                    "anomaly_interval_sec": [1.0, 6.0],
                    "precursor_interval_sec": [0.0, 1.0],
                    "earliest_alert_sec": 1.0,
                    "counterfactual_type": "remove_actor_interaction",
                },
                "tool_io": {
                    "oracle_windows_sec": [
                        {"moment_id": "ev1", "role": "trigger", "window": [1.0, 4.0]},
                        {"moment_id": "ev2", "role": "confirmation", "window": [4.0, 6.0]},
                    ]
                },
            },
            {
                "video_id": "normal_case",
                "video_meta": {"fps": 1.0, "duration_sec": 8.0, "total_frames": 8},
                "structured_target": {
                    "existence": "normal",
                    "category": "normal",
                    "severity": 0,
                    "hard_normal": True,
                    "anomaly_interval_sec": None,
                    "precursor_interval_sec": None,
                    "earliest_alert_sec": None,
                    "counterfactual_type": "none",
                },
                "tool_io": {"oracle_windows_sec": []},
            },
        ]
        scored_records = [
            {
                "video_id": "anomaly_case",
                "num_turns": 4,
                "turns": [
                    {"step_index": 1, "tool_name": "scan_timeline", "valid_action": True},
                    {"step_index": 2, "tool_name": "seek_evidence", "valid_action": True},
                    {"step_index": 3, "tool_name": "finalize_case", "valid_action": True},
                    {"step_index": 4, "action": "answer", "valid_action": True},
                ],
                "final_answer": {
                    "existence": "anomaly",
                    "category": "assault",
                    "severity": 4,
                    "anomaly_interval_sec": [1.0, 6.0],
                    "precursor_interval_sec": [0.0, 1.0],
                    "counterfactual_type": "remove_actor_interaction",
                },
                "state": {
                    "visited_windows": [
                        {"window_id": "w0001", "start_sec": 0.0, "end_sec": 6.0},
                        {"window_id": "w0002", "start_sec": 1.0, "end_sec": 4.0},
                    ],
                    "evidence_ledger": [
                        {"window_id": "w0002", "start_sec": 1.0, "end_sec": 4.0},
                    ],
                    "alerts": [{"alert_id": "a0001", "decision": "hard_alert", "alert_sec": 1.0}],
                    "finalized_case": {
                        "existence": "anomaly",
                        "category": "assault",
                        "severity": 4,
                        "anomaly_interval_sec": [1.0, 6.0],
                        "precursor_interval_sec": [0.0, 1.0],
                        "counterfactual_type": "remove_actor_interaction",
                    },
                    "last_claim": {"existence": "anomaly", "category": "assault"},
                    "active_evidence_window_ids": ["w0002"],
                },
                "offline_verifier": {
                    "primary_status": "complete",
                    "alert_status": "justified",
                    "verified_window_ids": ["w0002"],
                    "view_scores": {
                        "full": {"exist_support": 0.95},
                        "keep": {"exist_support": 0.90},
                    },
                },
                "reward_summary": {"total_reward": 1.5, "components": {"verification_reward": 1.0}},
            },
            {
                "video_id": "normal_case",
                "num_turns": 2,
                "turns": [
                    {"step_index": 1, "tool_name": "scan_timeline", "valid_action": True},
                    {"step_index": 2, "action": "answer", "valid_action": True},
                ],
                "final_answer": {"existence": "normal", "category": "normal", "severity": 0},
                "state": {
                    "visited_windows": [],
                    "evidence_ledger": [],
                    "alerts": [],
                    "finalized_case": {"existence": "normal", "category": "normal", "severity": 0},
                    "last_claim": {"existence": "normal", "category": "normal"},
                    "active_evidence_window_ids": [],
                },
                "offline_verifier": {
                    "primary_status": "complete",
                    "alert_status": "not_applicable",
                    "verified_window_ids": [],
                    "view_scores": {
                        "full": {"exist_support": 0.05},
                        "keep": {"exist_support": 0.05},
                    },
                },
                "reward_summary": {"total_reward": 0.8, "components": {"verification_reward": 0.4}},
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data.jsonl"
            with data_path.open("w", encoding="utf-8") as f:
                for record in data_records:
                    f.write(json.dumps(record) + "\n")
            reference_data = ReferenceDataProvider(data_path=data_path)
            summary = summarize_saver_metrics(scored_records, reference_data=reference_data)

        self.assertEqual(summary["num_records"], 2)
        self.assertAlmostEqual(summary["existence_accuracy"], 1.0, places=6)
        self.assertAlmostEqual(summary["category_macro_f1"], 1.0, places=6)
        self.assertAlmostEqual(summary["temporal_miou"], 1.0, places=6)
        self.assertAlmostEqual(summary["precursor_miou"], 1.0, places=6)
        self.assertAlmostEqual(summary["false_alert_rate"], 0.0, places=6)
        self.assertAlmostEqual(summary["hard_normal_false_alert_rate"], 0.0, places=6)
        self.assertAlmostEqual(summary["evidence_f1_at_3"], 2.0 / 3.0, places=6)
        self.assertAlmostEqual(summary["counterfactual_type_accuracy"], 1.0, places=6)
        self.assertAlmostEqual(summary["protocol_compliance_rate"], 1.0, places=6)

    def test_protocol_compliance_accepts_answer_with_state_finalized_case_even_without_finalize_turn(self):
        data_records = [
            {
                "video_id": "case",
                "video_meta": {"fps": 1.0, "duration_sec": 4.0, "total_frames": 4},
                "structured_target": {
                    "existence": "normal",
                    "category": "normal",
                    "severity": 0,
                    "hard_normal": False,
                    "anomaly_interval_sec": None,
                    "precursor_interval_sec": None,
                    "earliest_alert_sec": None,
                    "counterfactual_type": "none",
                },
                "tool_io": {"oracle_windows_sec": []},
            }
        ]
        scored_records = [
            {
                "video_id": "case",
                "num_turns": 2,
                "turns": [
                    {"step_index": 1, "tool_name": "scan_timeline", "valid_action": True},
                    {"step_index": 2, "action": "answer", "valid_action": True},
                ],
                "final_answer": {"existence": "normal", "category": "normal", "severity": 0},
                "state": {"finalized_case": {"existence": "normal", "category": "normal", "severity": 0}},
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data.jsonl"
            with data_path.open("w", encoding="utf-8") as f:
                for record in data_records:
                    f.write(json.dumps(record) + "\n")
            reference_data = ReferenceDataProvider(data_path=data_path)
            summary = summarize_saver_metrics(scored_records, reference_data=reference_data)

        self.assertAlmostEqual(summary["protocol_compliance_rate"], 1.0, places=6)

    def test_protocol_compliance_rejects_records_without_answer_or_finalize_artifacts(self):
        data_records = [
            {
                "video_id": "case",
                "video_meta": {"fps": 1.0, "duration_sec": 4.0, "total_frames": 4},
                "structured_target": {
                    "existence": "normal",
                    "category": "normal",
                    "severity": 0,
                    "hard_normal": False,
                    "anomaly_interval_sec": None,
                    "precursor_interval_sec": None,
                    "earliest_alert_sec": None,
                    "counterfactual_type": "none",
                },
                "tool_io": {"oracle_windows_sec": []},
            }
        ]
        scored_records = [
            {
                "video_id": "case",
                "num_turns": 1,
                "turns": [
                    {"step_index": 1, "tool_name": "scan_timeline", "valid_action": True},
                ],
                "state": {"visited_windows": [], "evidence_ledger": [], "alerts": []},
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data.jsonl"
            with data_path.open("w", encoding="utf-8") as f:
                for record in data_records:
                    f.write(json.dumps(record) + "\n")
            reference_data = ReferenceDataProvider(data_path=data_path)
            summary = summarize_saver_metrics(scored_records, reference_data=reference_data)

        self.assertAlmostEqual(summary["protocol_compliance_rate"], 0.0, places=6)

    def test_protocol_compliance_accepts_finalize_only_terminal_records(self):
        data_records = [
            {
                "video_id": "case",
                "video_meta": {"fps": 1.0, "duration_sec": 4.0, "total_frames": 4},
                "structured_target": {
                    "existence": "anomaly",
                    "category": "assault",
                    "severity": 3,
                    "hard_normal": False,
                    "anomaly_interval_sec": [0.5, 2.5],
                    "precursor_interval_sec": [0.0, 0.5],
                    "earliest_alert_sec": 0.5,
                    "counterfactual_type": "remove_actor_interaction",
                },
                "tool_io": {"oracle_windows_sec": [{"window": [0.5, 2.5]}]},
            }
        ]
        scored_records = [
            {
                "video_id": "case",
                "num_turns": 2,
                "turns": [
                    {"step_index": 1, "tool_name": "verify_hypothesis", "valid_action": True},
                    {"step_index": 2, "tool_name": "finalize_case", "valid_action": True},
                ],
                "final_answer": None,
                "state": {
                    "finalized_case": {
                        "existence": "anomaly",
                        "category": "assault",
                        "severity": 3,
                        "anomaly_interval_sec": [0.5, 2.5],
                        "precursor_interval_sec": [0.0, 0.5],
                        "counterfactual_type": "remove_actor_interaction",
                    },
                    "last_claim": {"existence": "anomaly", "category": "assault"},
                    "visited_windows": [],
                    "evidence_ledger": [{"window_id": "w0001", "start_sec": 0.5, "end_sec": 2.5}],
                    "alerts": [],
                    "active_evidence_window_ids": ["w0001"],
                },
                "offline_verifier": {
                    "primary_status": "complete",
                    "alert_status": "justified",
                    "verified_window_ids": ["w0001"],
                    "view_scores": {"full": {"exist_support": 0.9}, "keep": {"exist_support": 0.85}},
                },
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data.jsonl"
            with data_path.open("w", encoding="utf-8") as f:
                for record in data_records:
                    f.write(json.dumps(record) + "\n")
            reference_data = ReferenceDataProvider(data_path=data_path)
            summary = summarize_saver_metrics(scored_records, reference_data=reference_data)

        self.assertAlmostEqual(summary["protocol_compliance_rate"], 1.0, places=6)

    def test_tool_call_validity_rate_counts_invalid_attempts_not_just_formal_turns(self):
        data_records = [
            {
                "video_id": "case",
                "video_meta": {"fps": 1.0, "duration_sec": 4.0, "total_frames": 4},
                "structured_target": {
                    "existence": "normal",
                    "category": "normal",
                    "severity": 0,
                    "hard_normal": False,
                    "anomaly_interval_sec": None,
                    "precursor_interval_sec": None,
                    "earliest_alert_sec": None,
                    "counterfactual_type": "none",
                },
                "tool_io": {"oracle_windows_sec": []},
            }
        ]
        scored_records = [
            {
                "video_id": "case",
                "num_turns": 2,
                "num_invalid_attempts": 2,
                "turns": [
                    {"step_index": 1, "tool_name": "scan_timeline", "valid_action": True},
                    {"step_index": 2, "action": "answer", "valid_action": True},
                ],
                "invalid_attempts": [
                    {"step_index": 1, "action": None, "valid_action": False},
                    {"step_index": 2, "action": None, "valid_action": False},
                ],
                "final_answer": {"existence": "normal", "category": "normal", "severity": 0},
                "state": {"finalized_case": {"existence": "normal", "category": "normal", "severity": 0}},
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data.jsonl"
            with data_path.open("w", encoding="utf-8") as f:
                for record in data_records:
                    f.write(json.dumps(record) + "\n")
            reference_data = ReferenceDataProvider(data_path=data_path)
            summary = summarize_saver_metrics(scored_records, reference_data=reference_data)

        self.assertAlmostEqual(summary["tool_call_validity_rate"], 0.5, places=6)
        self.assertAlmostEqual(summary["formal_turn_validity_rate"], 1.0, places=6)

    def test_precursor_miou_skips_anomalies_without_reference_precursor(self):
        data_records = [
            {
                "video_id": "with_precursor",
                "video_meta": {"fps": 1.0, "duration_sec": 8.0, "total_frames": 8},
                "structured_target": {
                    "existence": "anomaly",
                    "category": "assault",
                    "severity": 4,
                    "hard_normal": False,
                    "anomaly_interval_sec": [1.0, 6.0],
                    "precursor_interval_sec": [0.0, 1.0],
                    "earliest_alert_sec": 1.0,
                    "counterfactual_type": "remove_actor_interaction",
                },
                "tool_io": {"oracle_windows_sec": []},
            },
            {
                "video_id": "without_precursor",
                "video_meta": {"fps": 1.0, "duration_sec": 8.0, "total_frames": 8},
                "structured_target": {
                    "existence": "anomaly",
                    "category": "fire",
                    "severity": 4,
                    "hard_normal": False,
                    "anomaly_interval_sec": [2.0, 7.0],
                    "precursor_interval_sec": None,
                    "earliest_alert_sec": 2.0,
                    "counterfactual_type": "none",
                },
                "tool_io": {"oracle_windows_sec": []},
            },
        ]
        scored_records = [
            {
                "video_id": "with_precursor",
                "num_turns": 2,
                "turns": [
                    {"step_index": 1, "tool_name": "finalize_case", "valid_action": True},
                    {"step_index": 2, "action": "answer", "valid_action": True},
                ],
                "final_answer": {
                    "existence": "anomaly",
                    "category": "assault",
                    "severity": 4,
                    "anomaly_interval_sec": [1.0, 6.0],
                    "precursor_interval_sec": [0.0, 1.0],
                },
                "state": {
                    "visited_windows": [],
                    "evidence_ledger": [],
                    "alerts": [],
                    "finalized_case": {
                        "existence": "anomaly",
                        "category": "assault",
                        "severity": 4,
                        "anomaly_interval_sec": [1.0, 6.0],
                        "precursor_interval_sec": [0.0, 1.0],
                    },
                    "last_claim": {"existence": "anomaly", "category": "assault"},
                    "active_evidence_window_ids": [],
                },
            },
            {
                "video_id": "without_precursor",
                "num_turns": 2,
                "turns": [
                    {"step_index": 1, "tool_name": "finalize_case", "valid_action": True},
                    {"step_index": 2, "action": "answer", "valid_action": True},
                ],
                "final_answer": {
                    "existence": "anomaly",
                    "category": "fire",
                    "severity": 4,
                    "anomaly_interval_sec": [2.0, 7.0],
                    "precursor_interval_sec": [0.0, 1.0],
                },
                "state": {
                    "visited_windows": [],
                    "evidence_ledger": [],
                    "alerts": [],
                    "finalized_case": {
                        "existence": "anomaly",
                        "category": "fire",
                        "severity": 4,
                        "anomaly_interval_sec": [2.0, 7.0],
                        "precursor_interval_sec": [0.0, 1.0],
                    },
                    "last_claim": {"existence": "anomaly", "category": "fire"},
                    "active_evidence_window_ids": [],
                },
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data.jsonl"
            with data_path.open("w", encoding="utf-8") as f:
                for record in data_records:
                    f.write(json.dumps(record) + "\n")
            reference_data = ReferenceDataProvider(data_path=data_path)
            summary = summarize_saver_metrics(scored_records, reference_data=reference_data)

        self.assertAlmostEqual(summary["precursor_miou"], 1.0, places=6)

    def test_protocol_compliance_rejects_answer_without_finalize_artifact(self):
        data_records = [
            {
                "video_id": "case",
                "video_meta": {"fps": 1.0, "duration_sec": 4.0, "total_frames": 4},
                "structured_target": {
                    "existence": "normal",
                    "category": "normal",
                    "severity": 0,
                    "hard_normal": False,
                    "anomaly_interval_sec": None,
                    "precursor_interval_sec": None,
                    "earliest_alert_sec": None,
                    "counterfactual_type": "none",
                },
                "tool_io": {"oracle_windows_sec": []},
            }
        ]
        scored_records = [
            {
                "video_id": "case",
                "num_turns": 1,
                "turns": [
                    {"step_index": 1, "action": "answer", "valid_action": True},
                ],
                "final_answer": {"existence": "normal", "category": "normal", "severity": 0},
                "state": {"visited_windows": [], "evidence_ledger": [], "alerts": []},
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data.jsonl"
            with data_path.open("w", encoding="utf-8") as f:
                for record in data_records:
                    f.write(json.dumps(record) + "\n")
            reference_data = ReferenceDataProvider(data_path=data_path)
            summary = summarize_saver_metrics(scored_records, reference_data=reference_data)

        self.assertAlmostEqual(summary["protocol_compliance_rate"], 0.0, places=6)

    def test_protocol_compliance_rejects_malformed_answer_artifact(self):
        data_records = [
            {
                "video_id": "case",
                "video_meta": {"fps": 1.0, "duration_sec": 4.0, "total_frames": 4},
                "structured_target": {
                    "existence": "normal",
                    "category": "normal",
                    "severity": 0,
                    "hard_normal": False,
                    "anomaly_interval_sec": None,
                    "precursor_interval_sec": None,
                    "earliest_alert_sec": None,
                    "counterfactual_type": "none",
                },
                "tool_io": {"oracle_windows_sec": []},
            }
        ]
        scored_records = [
            {
                "video_id": "case",
                "num_turns": 2,
                "turns": [
                    {"step_index": 1, "tool_name": "finalize_case", "valid_action": True},
                    {"step_index": 2, "action": "answer", "valid_action": False},
                ],
                "final_answer": "not-json",
                "state": {"finalized_case": {"existence": "normal", "category": "normal", "severity": 0}},
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data.jsonl"
            with data_path.open("w", encoding="utf-8") as f:
                for record in data_records:
                    f.write(json.dumps(record) + "\n")
            reference_data = ReferenceDataProvider(data_path=data_path)
            summary = summarize_saver_metrics(scored_records, reference_data=reference_data)

        self.assertAlmostEqual(summary["protocol_compliance_rate"], 0.0, places=6)

    def test_summarize_saver_metrics_existence_ap_is_independent_from_offline_verifier_scores(self):
        data_records = [
            {
                "video_id": "anom",
                "video_meta": {"fps": 1.0, "duration_sec": 4.0, "total_frames": 4},
                "structured_target": {
                    "existence": "anomaly",
                    "category": "assault",
                    "severity": 4,
                    "hard_normal": False,
                    "anomaly_interval_sec": [1.0, 3.0],
                    "precursor_interval_sec": None,
                    "earliest_alert_sec": 1.0,
                    "counterfactual_type": "remove_actor_interaction",
                },
                "tool_io": {"oracle_windows_sec": []},
            },
            {
                "video_id": "norm",
                "video_meta": {"fps": 1.0, "duration_sec": 4.0, "total_frames": 4},
                "structured_target": {
                    "existence": "normal",
                    "category": "normal",
                    "severity": 0,
                    "hard_normal": False,
                    "anomaly_interval_sec": None,
                    "precursor_interval_sec": None,
                    "earliest_alert_sec": None,
                    "counterfactual_type": "none",
                },
                "tool_io": {"oracle_windows_sec": []},
            },
        ]
        scored_records = [
            {
                "video_id": "anom",
                "num_turns": 2,
                "turns": [{"step_index": 1, "action": "answer", "valid_action": True}],
                "final_answer": {"existence": "anomaly", "category": "assault", "severity": 4},
                "state": {"finalized_case": {"existence": "anomaly", "category": "assault", "severity": 4}},
                "offline_verifier": {
                    "primary_status": "misaligned",
                    "alert_status": "premature",
                    "view_scores": {"full": {"exist_support": 0.0}},
                },
            },
            {
                "video_id": "norm",
                "num_turns": 2,
                "turns": [{"step_index": 1, "action": "answer", "valid_action": True}],
                "final_answer": {"existence": "normal", "category": "normal", "severity": 0},
                "state": {"finalized_case": {"existence": "normal", "category": "normal", "severity": 0}},
                "offline_verifier": {
                    "primary_status": "complete",
                    "alert_status": "not_applicable",
                    "view_scores": {"full": {"exist_support": 1.0}},
                },
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data.jsonl"
            with data_path.open("w", encoding="utf-8") as f:
                for record in data_records:
                    f.write(json.dumps(record) + "\n")
            reference_data = ReferenceDataProvider(data_path=data_path)
            summary = summarize_saver_metrics(scored_records, reference_data=reference_data)

        self.assertAlmostEqual(summary["existence_accuracy"], 1.0, places=6)
        self.assertAlmostEqual(summary["existence_ap"], 1.0, places=6)
        self.assertNotIn("score_summary", summary)

    def test_summarize_saver_metrics_evidence_f1_uses_policy_selected_windows_not_offline_verifier_windows(self):
        data_records = [
            {
                "video_id": "anom",
                "video_meta": {"fps": 1.0, "duration_sec": 6.0, "total_frames": 6},
                "structured_target": {
                    "existence": "anomaly",
                    "category": "assault",
                    "severity": 4,
                    "hard_normal": False,
                    "anomaly_interval_sec": [1.0, 4.0],
                    "precursor_interval_sec": None,
                    "earliest_alert_sec": 1.0,
                    "counterfactual_type": "remove_actor_interaction",
                },
                "tool_io": {
                    "oracle_windows_sec": [
                        {"moment_id": "ev1", "role": "trigger", "window": [1.0, 4.0]},
                    ]
                },
            }
        ]
        scored_records = [
            {
                "video_id": "anom",
                "num_turns": 3,
                "turns": [{"step_index": 2, "action": "answer", "valid_action": True}],
                "final_answer": {"existence": "anomaly", "category": "assault", "severity": 4},
                "state": {
                    "finalized_case": {"existence": "anomaly", "category": "assault", "severity": 4},
                    "active_evidence_window_ids": ["w_good"],
                    "visited_windows": [
                        {"window_id": "w_bad", "start_sec": 4.0, "end_sec": 5.0},
                        {"window_id": "w_good", "start_sec": 1.0, "end_sec": 4.0},
                    ],
                    "evidence_ledger": [
                        {"window_id": "w_bad", "start_sec": 4.0, "end_sec": 5.0},
                        {"window_id": "w_good", "start_sec": 1.0, "end_sec": 4.0},
                    ],
                },
                "offline_verifier": {
                    "primary_status": "complete",
                    "alert_status": "justified",
                    "verified_window_ids": ["w_bad"],
                },
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data.jsonl"
            with data_path.open("w", encoding="utf-8") as f:
                for record in data_records:
                    f.write(json.dumps(record) + "\n")
            reference_data = ReferenceDataProvider(data_path=data_path)
            summary = summarize_saver_metrics(scored_records, reference_data=reference_data, evidence_top_k=1)

        self.assertAlmostEqual(summary["evidence_f1_at_3"], 1.0, places=6)

    def test_summarize_saver_metrics_ignores_bogus_active_windows_and_reports_verify_health_stats(self):
        data_records = [
            {
                "video_id": "anom",
                "video_meta": {"fps": 1.0, "duration_sec": 6.0, "total_frames": 6},
                "structured_target": {
                    "existence": "anomaly",
                    "category": "assault",
                    "severity": 4,
                    "hard_normal": False,
                    "anomaly_interval_sec": [1.0, 4.0],
                    "precursor_interval_sec": None,
                    "earliest_alert_sec": 1.0,
                    "counterfactual_type": "remove_actor_interaction",
                },
                "tool_io": {
                    "oracle_windows_sec": [
                        {"moment_id": "ev1", "role": "trigger", "window": [1.0, 4.0]},
                    ]
                },
            }
        ]
        scored_records = [
            {
                "video_id": "anom",
                "num_turns": 4,
                "turns": [
                    {
                        "step_index": 2,
                        "tool_name": "verify_hypothesis",
                        "valid_action": False,
                        "verification_parse_mode": "legacy_compatibility",
                        "legacy_compatibility_used": True,
                        "invalid_selected_window_ids": ["window_0"],
                        "verifier_failure_reasons": ["selected_evidence_not_resolved_to_known_windows"],
                    },
                    {"step_index": 4, "action": "answer", "valid_action": True},
                ],
                "final_answer": {"existence": "anomaly", "category": "assault", "severity": 4},
                "state": {
                    "finalized_case": {"existence": "anomaly", "category": "assault", "severity": 4},
                    "active_evidence_window_ids": ["window_0"],
                    "visited_windows": [
                        {"window_id": "w_good", "start_sec": 1.0, "end_sec": 4.0},
                    ],
                    "evidence_ledger": [
                        {"window_id": "w_good", "start_sec": 1.0, "end_sec": 4.0},
                    ],
                },
                "offline_verifier": {
                    "primary_status": "complete",
                    "alert_status": "justified",
                    "verified_window_ids": ["w_good"],
                },
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data.jsonl"
            with data_path.open("w", encoding="utf-8") as f:
                for record in data_records:
                    f.write(json.dumps(record) + "\n")
            reference_data = ReferenceDataProvider(data_path=data_path)
            summary = summarize_saver_metrics(scored_records, reference_data=reference_data, evidence_top_k=1)

        self.assertAlmostEqual(summary["evidence_f1_at_3"], 0.0, places=6)
        self.assertEqual(summary["verify_parse_mode_counts"]["legacy_compatibility"], 1)
        self.assertEqual(summary["legacy_compatibility_used_count"], 1)
        self.assertAlmostEqual(summary["invalid_selected_window_rate"], 1.0, places=6)
        self.assertAlmostEqual(summary["unresolved_selection_rate"], 1.0, places=6)
        self.assertAlmostEqual(summary["verify_invalid_turn_rate"], 1.0, places=6)

    def test_summarize_saver_metrics_normalizes_safe_category_aliases_to_canonical_labels(self):
        data_records = [
            {
                "video_id": "anom",
                "video_meta": {"fps": 1.0, "duration_sec": 4.0, "total_frames": 4},
                "structured_target": {
                    "existence": "anomaly",
                    "category": "traffic_accident",
                    "severity": 3,
                    "hard_normal": False,
                    "anomaly_interval_sec": [1.0, 3.0],
                    "precursor_interval_sec": None,
                    "earliest_alert_sec": 1.0,
                    "counterfactual_type": "none",
                },
                "tool_io": {"oracle_windows_sec": []},
            }
        ]
        scored_records = [
            {
                "video_id": "anom",
                "num_turns": 2,
                "turns": [
                    {"step_index": 1, "tool_name": "finalize_case", "valid_action": True},
                    {"step_index": 2, "action": "answer", "valid_action": True},
                ],
                "final_answer": {
                    "existence": "anomaly",
                    "category": "vehicle_collision",
                    "severity": 3,
                    "anomaly_interval_sec": [1.0, 3.0],
                },
                "state": {
                    "finalized_case": {
                        "existence": "anomaly",
                        "category": "vehicle_collision",
                        "severity": 3,
                        "anomaly_interval_sec": [1.0, 3.0],
                    }
                },
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data.jsonl"
            with data_path.open("w", encoding="utf-8") as f:
                for record in data_records:
                    f.write(json.dumps(record) + "\n")
            reference_data = ReferenceDataProvider(data_path=data_path)
            summary = summarize_saver_metrics(scored_records, reference_data=reference_data)

        self.assertAlmostEqual(summary["existence_accuracy"], 1.0, places=6)
        self.assertAlmostEqual(summary["category_macro_f1"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
