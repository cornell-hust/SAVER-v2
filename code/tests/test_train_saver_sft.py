import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path("/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


try:
    import train_saver_sft as train_saver_sft
except ModuleNotFoundError:
    train_saver_sft = None


class TrainSaverSftTests(unittest.TestCase):
    def test_build_sft_examples_from_jsonl_returns_oracle_stepwise_examples(self):
        self.assertIsNotNone(train_saver_sft, "train_saver_sft.py is missing")

        sample = {
            "schema_version": "saver_agent.v1",
            "video_id": "sft_case",
            "video_path": "videos/sft_case.mp4",
            "split": "train",
            "source_dataset": "MSAD",
            "frame_index_base": 1,
            "video_meta": {"fps": 4.0, "duration_sec": 4.0, "total_frames": 16},
            "scene": {"scenario": "shop"},
            "label": {"is_anomaly": False, "category": "normal", "severity": 0, "hard_normal": False},
            "agent_task": {
                "task_type": "video_anomaly_search_alert_verify",
                "query_mode": "internal_hypothesis_generation",
                "task_prompt": "Check whether anything abnormal happens.",
                "success_criteria": ["criterion_a"],
            },
            "structured_target": {
                "existence": "normal",
                "category": "normal",
                "severity": 0,
                "anomaly_interval_sec": None,
                "precursor_interval_sec": None,
                "earliest_alert_sec": None,
                "evidence_moment_ids": [],
                "counterfactual_type": "none",
                "summary": "Normal activity.",
                "rationale": "No anomaly is visible.",
            },
            "temporal": {
                "anomaly_interval_sec": None,
                "precursor_interval_sec": None,
                "earliest_alert_sec": None,
            },
            "evidence": {"evidence_moments": []},
            "tool_io": {
                "allowed_tools": ["scan_timeline", "emit_alert", "finalize_case"],
                "initial_scan_window_frames": [1, 16],
                "initial_scan_window_sec": [0.0, 4.0],
                "oracle_windows_frames": [],
                "oracle_windows_sec": [],
                "finalize_case_schema": {"type": "object", "required": ["existence"]},
            },
            "oracle_sft": {
                "trajectory": [
                    {"tool": "scan_timeline", "arguments": {"start_sec": 0.0, "end_sec": 4.0, "purpose": "global_overview"}},
                    {"tool": "emit_alert", "arguments": {"decision": "declare_normal", "existence": "normal", "reason": "Routine activity only."}},
                    {"tool": "finalize_case", "arguments": {"existence": "normal", "category": "normal"}},
                ],
                "final_decision": {"existence": "normal", "category": "normal"},
            },
            "auto_completed": {"precursor_interval": False},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_path = root / "oracle_sft.jsonl"
            data_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

            examples = train_saver_sft.build_sft_examples_from_jsonl(
                data_path=data_path,
                data_root=root,
                max_records=1,
            )

        self.assertGreaterEqual(len(examples), 4)
        self.assertIn('"name":"scan_timeline"', examples[0]["target_response"])
        self.assertIn("<answer>", examples[-1]["target_response"])

    def test_build_prepared_sft_examples_from_jsonl_returns_json_safe_examples(self):
        self.assertIsNotNone(train_saver_sft, "train_saver_sft.py is missing")

        sample = {
            "schema_version": "saver_agent.v1",
            "video_id": "prepared_case",
            "video_path": "videos/prepared_case.mp4",
            "split": "train",
            "source_dataset": "MSAD",
            "frame_index_base": 1,
            "video_meta": {"fps": 4.0, "duration_sec": 4.0, "total_frames": 16},
            "scene": {"scenario": "shop"},
            "label": {"is_anomaly": False, "category": "normal", "severity": 0, "hard_normal": False},
            "agent_task": {
                "task_type": "video_anomaly_search_alert_verify",
                "query_mode": "internal_hypothesis_generation",
                "task_prompt": "Check whether anything abnormal happens.",
                "success_criteria": ["criterion_a"],
            },
            "structured_target": {
                "existence": "normal",
                "category": "normal",
                "severity": 0,
                "anomaly_interval_sec": None,
                "precursor_interval_sec": None,
                "earliest_alert_sec": None,
                "evidence_moment_ids": [],
                "counterfactual_type": "none",
                "summary": "Normal activity.",
                "rationale": "No anomaly is visible.",
            },
            "temporal": {
                "anomaly_interval_sec": None,
                "precursor_interval_sec": None,
                "earliest_alert_sec": None,
            },
            "evidence": {"evidence_moments": []},
            "tool_io": {
                "allowed_tools": ["scan_timeline", "emit_alert", "finalize_case"],
                "initial_scan_window_frames": [1, 16],
                "initial_scan_window_sec": [0.0, 4.0],
                "oracle_windows_frames": [],
                "oracle_windows_sec": [],
                "finalize_case_schema": {"type": "object", "required": ["existence"]},
            },
            "oracle_sft": {
                "trajectory": [
                    {"tool": "scan_timeline", "arguments": {"start_sec": 0.0, "end_sec": 4.0, "purpose": "global_overview"}},
                    {"tool": "emit_alert", "arguments": {"decision": "declare_normal", "existence": "normal", "reason": "Routine activity only."}},
                    {"tool": "finalize_case", "arguments": {"existence": "normal", "category": "normal"}},
                ],
                "final_decision": {"existence": "normal", "category": "normal"},
            },
            "auto_completed": {"precursor_interval": False},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_path = root / "oracle_sft.jsonl"
            data_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

            examples = train_saver_sft.build_prepared_sft_examples_from_jsonl(
                data_path=data_path,
                data_root=root,
                max_records=1,
            )

        self.assertGreaterEqual(len(examples), 4)
        json.dumps(examples)

    def test_build_prepared_sft_examples_from_agent_view_without_oracle_sft(self):
        self.assertIsNotNone(train_saver_sft, "train_saver_sft.py is missing")

        sample = {
            "schema_version": "saver_agent.v1",
            "video_id": "fallback_oracle_case",
            "video_path": "videos/fallback_oracle_case.mp4",
            "split": "train",
            "source_dataset": "MSAD",
            "frame_index_base": 1,
            "video_meta": {"fps": 4.0, "duration_sec": 4.0, "total_frames": 16},
            "scene": {"scenario": "shop"},
            "label": {"is_anomaly": False, "category": "normal", "severity": 0, "hard_normal": False},
            "agent_task": {
                "task_type": "video_anomaly_search_alert_verify",
                "query_mode": "internal_hypothesis_generation",
                "task_prompt": "Check whether anything abnormal happens.",
                "success_criteria": ["criterion_a"],
            },
            "structured_target": {
                "existence": "normal",
                "category": "normal",
                "severity": 0,
                "anomaly_interval_sec": None,
                "precursor_interval_sec": None,
                "earliest_alert_sec": None,
                "evidence_moment_ids": [],
                "counterfactual_type": "none",
                "summary": "Normal activity.",
                "rationale": "No anomaly is visible.",
            },
            "temporal": {
                "anomaly_interval_sec": None,
                "precursor_interval_sec": None,
                "earliest_alert_sec": None,
            },
            "evidence": {"evidence_moments": []},
            "tool_io": {
                "allowed_tools": ["scan_timeline", "emit_alert", "finalize_case"],
                "initial_scan_window_frames": [1, 16],
                "initial_scan_window_sec": [0.0, 4.0],
                "oracle_windows_frames": [],
                "oracle_windows_sec": [],
                "finalize_case_schema": {"type": "object", "required": ["existence"]},
            },
            "auto_completed": {"precursor_interval": False},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_path = root / "oracle_sft.jsonl"
            data_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

            examples = train_saver_sft.build_prepared_sft_examples_from_jsonl(
                data_path=data_path,
                data_root=root,
                max_records=1,
            )

        self.assertGreaterEqual(len(examples), 3)
        self.assertIn('"name":"scan_timeline"', examples[0]["target_response"])
        self.assertIn("<answer>", examples[-1]["target_response"])

    def test_build_rollout_eval_config_can_enable_reference_diagnostics(self):
        self.assertIsNotNone(train_saver_sft, "train_saver_sft.py is missing")

        args = train_saver_sft.parse_args(
            [
                "--data",
                "/tmp/data.jsonl",
                "--output-dir",
                "/tmp/sft_out",
                "--eval-data",
                "/tmp/eval.jsonl",
                "--eval-attach-reference-diagnostics",
            ]
        )

        config = train_saver_sft._build_rollout_eval_config(
            args,
            config=train_saver_sft._build_config(args),
        )

        self.assertTrue(config.attach_reference_diagnostics)

    def test_parse_args_defaults_eval_rollout_max_turns_to_twelve(self):
        self.assertIsNotNone(train_saver_sft, "train_saver_sft.py is missing")

        args = train_saver_sft.parse_args(
            [
                "--data",
                "/tmp/data.jsonl",
                "--output-dir",
                "/tmp/sft_out",
            ]
        )

        self.assertEqual(args.eval_rollout_max_turns, 12)

    def test_main_passes_dataloader_knobs_to_run_weighted_sft(self):
        self.assertIsNotNone(train_saver_sft, "train_saver_sft.py is missing")

        prepared_example = {
            "video_id": "prepared_case",
            "split": "train",
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "system"}]},
                {"role": "user", "content": [{"type": "text", "text": "user"}]},
            ],
            "target_response": "<answer>{}</answer>",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prepared_path = root / "prepared.jsonl"
            prepared_path.write_text(json.dumps(prepared_example) + "\n", encoding="utf-8")
            output_dir = root / "sft_out"
            argv = [
                "train_saver_sft.py",
                "--prepared-data",
                str(prepared_path),
                "--output-dir",
                str(output_dir),
                "--dataloader-num-workers",
                "3",
                "--dataloader-prefetch-factor",
                "4",
                "--dataloader-persistent-workers",
            ]

            with patch.object(sys, "argv", argv), patch.object(
                train_saver_sft, "run_weighted_sft", return_value={"ok": True}
            ) as mocked_run_weighted_sft:
                train_saver_sft.main()

        kwargs = mocked_run_weighted_sft.call_args.kwargs
        self.assertEqual(kwargs["dataloader_num_workers"], 3)
        self.assertEqual(kwargs["dataloader_prefetch_factor"], 4)
        self.assertTrue(kwargs["dataloader_persistent_workers"])

    def test_load_jsonl_reports_invalid_line_number(self):
        self.assertIsNotNone(train_saver_sft, "train_saver_sft.py is missing")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_path = root / "broken.jsonl"
            data_path.write_text('{"ok": 1}\n{"bad": "unterminated}\n', encoding="utf-8")

            with self.assertRaises(ValueError) as ctx:
                train_saver_sft._load_jsonl(data_path)

        self.assertIn("broken.jsonl:2", str(ctx.exception))

    def test_build_prepared_sft_examples_filters_split_without_index_mismatch(self):
        self.assertIsNotNone(train_saver_sft, "train_saver_sft.py is missing")

        def make_sample(video_id: str, split: str) -> dict:
            return {
                "schema_version": "saver_agent.v1",
                "video_id": video_id,
                "video_path": f"videos/{video_id}.mp4",
                "split": split,
                "source_dataset": "MSAD",
                "frame_index_base": 1,
                "video_meta": {"fps": 4.0, "duration_sec": 4.0, "total_frames": 16},
                "scene": {"scenario": "shop"},
                "label": {"is_anomaly": False, "category": "normal", "severity": 0, "hard_normal": False},
                "agent_task": {
                    "task_type": "video_anomaly_search_alert_verify",
                    "query_mode": "internal_hypothesis_generation",
                    "task_prompt": "Check whether anything abnormal happens.",
                    "success_criteria": ["criterion_a"],
                },
                "structured_target": {
                    "existence": "normal",
                    "category": "normal",
                    "severity": 0,
                    "anomaly_interval_sec": None,
                    "precursor_interval_sec": None,
                    "earliest_alert_sec": None,
                    "evidence_moment_ids": [],
                    "counterfactual_type": "none",
                    "summary": "Normal activity.",
                    "rationale": "No anomaly is visible.",
                },
                "temporal": {
                    "anomaly_interval_sec": None,
                    "precursor_interval_sec": None,
                    "earliest_alert_sec": None,
                },
                "evidence": {"evidence_moments": []},
                "tool_io": {
                    "allowed_tools": ["scan_timeline", "emit_alert", "finalize_case"],
                    "initial_scan_window_frames": [1, 16],
                    "initial_scan_window_sec": [0.0, 4.0],
                    "oracle_windows_frames": [],
                    "oracle_windows_sec": [],
                    "finalize_case_schema": {"type": "object", "required": ["existence"]},
                },
                "oracle_sft": {
                    "trajectory": [
                        {"tool": "scan_timeline", "arguments": {"start_sec": 0.0, "end_sec": 4.0, "purpose": "global_overview"}},
                        {"tool": "emit_alert", "arguments": {"decision": "declare_normal", "existence": "normal", "reason": "Routine activity only."}},
                        {"tool": "finalize_case", "arguments": {"existence": "normal", "category": "normal"}},
                    ],
                    "final_decision": {"existence": "normal", "category": "normal"},
                },
                "auto_completed": {"precursor_interval": False},
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_path = root / "oracle_sft.jsonl"
            data_path.write_text(
                json.dumps(make_sample("test_case", "test")) + "\n" + json.dumps(make_sample("train_case", "train")) + "\n",
                encoding="utf-8",
            )

            examples = train_saver_sft.build_prepared_sft_examples_from_jsonl(
                data_path=data_path,
                data_root=root,
                include_splits="train",
            )

        self.assertGreaterEqual(len(examples), 4)
        self.assertEqual({example["video_id"] for example in examples}, {"train_case"})
        self.assertEqual({example["split"] for example in examples}, {"train"})


if __name__ == "__main__":
    unittest.main()
