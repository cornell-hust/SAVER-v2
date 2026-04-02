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

    def test_parse_args_defaults_training_visual_budget_to_eval_budget(self):
        self.assertIsNotNone(train_saver_sft, "train_saver_sft.py is missing")

        args = train_saver_sft.parse_args(
            [
                "--data",
                "/tmp/data.jsonl",
                "--output-dir",
                "/tmp/sft_out",
            ]
        )

        self.assertEqual(args.max_total_images, 24)
        self.assertEqual(args.eval_max_new_tokens_per_turn, 256)
        self.assertEqual(args.eval_max_total_images, 24)
        self.assertEqual(args.eval_rollout_max_turns, 12)

    def test_build_rollout_eval_config_passes_proposal_runtime_fields(self):
        self.assertIsNotNone(train_saver_sft, "train_saver_sft.py is missing")

        args = train_saver_sft.parse_args(
            [
                "--data",
                "/tmp/data.jsonl",
                "--output-dir",
                "/tmp/sft_out",
                "--proposal-model-path",
                "/models/siglip_base",
                "--eval-data",
                "/tmp/eval.jsonl",
                "--eval-proposal-torch-dtype",
                "bfloat16",
                "--eval-proposal-device",
                "cuda:1",
            ]
        )

        config = train_saver_sft._build_rollout_eval_config(
            args,
            config=train_saver_sft._build_config(args),
        )

        self.assertEqual(str(config.proposal_model_path), "/models/siglip_base")
        self.assertEqual(config.proposal_torch_dtype, "bfloat16")
        self.assertEqual(config.proposal_device, "cuda:1")

    def test_build_prepared_sft_examples_attaches_proposal_runtime_to_multimodal_cache(self):
        self.assertIsNotNone(train_saver_sft, "train_saver_sft.py is missing")

        captured = {}

        class DummyDataset:
            def __init__(self, *args, **kwargs):
                self.records = [{"video_id": "vid_1"}]

            def format_frame_cache_status(self, *, prefix="frame cache", max_examples=5):
                return f"{prefix}: ok"

            def __getitem__(self, index):
                return {"video_id": "vid_1", "multimodal_cache": {}}

        def fake_build_oracle_sft_examples(item, record, **kwargs):
            captured["proposal_runtime"] = item["multimodal_cache"].get("proposal_runtime")
            return [{"video_id": record["video_id"], "target_response": "<answer>{}</answer>"}]

        with patch("train_saver_sft.SaverAgentDataset", DummyDataset), patch(
            "train_saver_sft.build_oracle_sft_examples",
            side_effect=fake_build_oracle_sft_examples,
        ):
            examples = train_saver_sft.build_prepared_sft_examples_from_jsonl(
                data_path="/tmp/data.jsonl",
                proposal_runtime="siglip_runtime",
            )

        self.assertEqual(captured["proposal_runtime"], "siglip_runtime")
        self.assertEqual(len(examples), 1)

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
        self.assertEqual(args.eval_max_new_tokens_per_turn, 256)

    def test_parse_args_accepts_resume_recovery_flags(self):
        self.assertIsNotNone(train_saver_sft, "train_saver_sft.py is missing")

        args = train_saver_sft.parse_args(
            [
                "--prepared-data",
                "/tmp/prepared.jsonl",
                "--output-dir",
                "/tmp/sft_out",
                "--resume-from-checkpoint",
                "/tmp/sft_out/epoch_resume/epoch_001",
                "--resume-rollout-eval-only",
            ]
        )

        self.assertEqual(args.resume_from_checkpoint, "/tmp/sft_out/epoch_resume/epoch_001")
        self.assertTrue(args.resume_rollout_eval_only)

    def test_parse_args_accepts_inline_rollout_eval_flag(self):
        self.assertIsNotNone(train_saver_sft, "train_saver_sft.py is missing")

        args = train_saver_sft.parse_args(
            [
                "--prepared-data",
                "/tmp/prepared.jsonl",
                "--output-dir",
                "/tmp/sft_out",
                "--eval-data",
                "/tmp/eval.jsonl",
                "--inline-rollout-eval",
            ]
        )

        self.assertTrue(args.inline_rollout_eval)

    def test_resolve_resume_epoch_index_from_epoch_resume_dir_name(self):
        self.assertIsNotNone(train_saver_sft, "train_saver_sft.py is missing")

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "epoch_resume" / "epoch_003"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            epoch_index = train_saver_sft._resolve_resume_epoch_index(checkpoint_dir)

        self.assertEqual(epoch_index, 3)

    def test_main_passes_resume_checkpoint_into_run_weighted_sft(self):
        self.assertIsNotNone(train_saver_sft, "train_saver_sft.py is missing")

        captured = {}

        def fake_run_weighted_sft(examples, **kwargs):
            captured["examples"] = list(examples)
            captured["resume_from_checkpoint"] = kwargs.get("resume_from_checkpoint")
            return {"num_examples": len(examples), "output_dir": kwargs.get("output_dir", "")}

        with tempfile.TemporaryDirectory() as tmpdir:
            prepared_path = Path(tmpdir) / "prepared.jsonl"
            prepared_path.write_text("{}\n", encoding="utf-8")
            output_dir = Path(tmpdir) / "sft_out"
            resume_dir = output_dir / "checkpoint-120"
            resume_dir.mkdir(parents=True, exist_ok=True)
            argv = [
                "train_saver_sft.py",
                "--prepared-data",
                str(prepared_path),
                "--output-dir",
                str(output_dir),
                "--resume-from-checkpoint",
                str(resume_dir),
            ]
            with patch.object(sys, "argv", argv), patch(
                "train_saver_sft._load_jsonl",
                return_value=[{"video_id": "v1", "target_response": "<answer>{}</answer>"}],
            ), patch(
                "train_saver_sft._maybe_annotate_examples_with_teacher_judge",
                side_effect=lambda args, examples, runtime: (examples, {}),
            ), patch(
                "train_saver_sft._apply_teacher_judge_reweighting",
                side_effect=lambda examples, runtime: (examples, {}),
            ), patch(
                "train_saver_sft._run_validation",
                return_value={},
            ), patch(
                "train_saver_sft.run_weighted_sft",
                side_effect=fake_run_weighted_sft,
            ):
                train_saver_sft.main()

        self.assertEqual(captured["resume_from_checkpoint"], str(resume_dir))
        self.assertEqual(len(captured["examples"]), 1)

    def test_main_eval_only_recovery_runs_rollout_eval_without_trainer(self):
        self.assertIsNotNone(train_saver_sft, "train_saver_sft.py is missing")

        captured = {}

        def fake_eval_only(*, checkpoint_path, output_dir, rollout_eval_config, epoch_index, model_path, torch_dtype, attn_implementation, runtime):
            captured["checkpoint_path"] = str(checkpoint_path)
            captured["output_dir"] = str(output_dir)
            captured["epoch_index"] = int(epoch_index)
            captured["model_path"] = str(model_path)
            captured["policy_max_new_tokens"] = int(rollout_eval_config.policy_max_new_tokens)
            return {"epoch_index": int(epoch_index), "temporal_miou": 0.25}

        with tempfile.TemporaryDirectory() as tmpdir:
            prepared_path = Path(tmpdir) / "prepared.jsonl"
            prepared_path.write_text("{}\n", encoding="utf-8")
            output_dir = Path(tmpdir) / "sft_out"
            resume_dir = output_dir / "epoch_resume" / "epoch_001"
            resume_dir.mkdir(parents=True, exist_ok=True)
            argv = [
                "train_saver_sft.py",
                "--prepared-data",
                str(prepared_path),
                "--output-dir",
                str(output_dir),
                "--eval-data",
                str(Path(tmpdir) / "eval.jsonl"),
                "--resume-from-checkpoint",
                str(resume_dir),
                "--resume-rollout-eval-only",
            ]
            with patch.object(sys, "argv", argv), patch(
                "train_saver_sft._load_jsonl",
                return_value=[{"video_id": "v1", "target_response": "<answer>{}</answer>"}],
            ), patch(
                "train_saver_sft._maybe_annotate_examples_with_teacher_judge",
                side_effect=lambda args, examples, runtime: (examples, {}),
            ), patch(
                "train_saver_sft._apply_teacher_judge_reweighting",
                side_effect=lambda examples, runtime: (examples, {}),
            ), patch(
                "train_saver_sft._run_validation",
                return_value={},
            ), patch(
                "train_saver_sft.run_weighted_sft",
                side_effect=AssertionError("run_weighted_sft should not run in eval-only recovery mode"),
            ), patch(
                "train_saver_sft.run_rollout_eval_from_checkpoint",
                side_effect=fake_eval_only,
            ):
                train_saver_sft.main()

        self.assertEqual(captured["checkpoint_path"], str(resume_dir))
        self.assertEqual(captured["output_dir"], str(output_dir))
        self.assertEqual(captured["epoch_index"], 1)
        self.assertEqual(captured["policy_max_new_tokens"], 256)

    def test_build_rollout_eval_config_resolves_eval_visual_budget_and_output_budget(self):
        self.assertIsNotNone(train_saver_sft, "train_saver_sft.py is missing")

        args = train_saver_sft.parse_args(
            [
                "--data",
                "/tmp/data.jsonl",
                "--output-dir",
                "/tmp/sft_out",
                "--eval-data",
                "/tmp/eval.jsonl",
                "--eval-total-visual-budget",
                "24",
                "--eval-max-new-tokens-per-turn",
                "256",
            ]
        )

        config = train_saver_sft._build_rollout_eval_config(
            args,
            config=train_saver_sft._build_config(args),
        )

        self.assertEqual(config.policy_max_new_tokens, 256)
        self.assertEqual(config.max_total_images, 24)

    def test_build_rollout_eval_config_carries_inline_rollout_eval_flag(self):
        self.assertIsNotNone(train_saver_sft, "train_saver_sft.py is missing")

        args = train_saver_sft.parse_args(
            [
                "--data",
                "/tmp/data.jsonl",
                "--output-dir",
                "/tmp/sft_out",
                "--eval-data",
                "/tmp/eval.jsonl",
                "--inline-rollout-eval",
            ]
        )

        config = train_saver_sft._build_rollout_eval_config(
            args,
            config=train_saver_sft._build_config(args),
        )

        self.assertTrue(config.inline_rollout_eval)

    def test_resolve_proposal_device_falls_back_to_cuda_zero_when_local_rank_exceeds_visible_devices(self):
        self.assertIsNotNone(train_saver_sft, "train_saver_sft.py is missing")

        runtime = type(
            "Runtime",
            (),
            {
                "local_rank": 3,
                "is_main_process": True,
                "is_distributed": False,
                "rank": 0,
                "world_size": 1,
            },
        )()
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.device_count", return_value=1):
            resolved = train_saver_sft._resolve_proposal_device("", runtime=runtime)

        self.assertEqual(resolved, "cuda:0")

    def test_build_rollout_eval_config_prefers_eval_max_total_images_over_visual_budget_alias(self):
        self.assertIsNotNone(train_saver_sft, "train_saver_sft.py is missing")

        args = train_saver_sft.parse_args(
            [
                "--data",
                "/tmp/data.jsonl",
                "--output-dir",
                "/tmp/sft_out",
                "--eval-data",
                "/tmp/eval.jsonl",
                "--eval-total-visual-budget",
                "24",
                "--eval-max-total-images",
                "18",
            ]
        )

        config = train_saver_sft._build_rollout_eval_config(
            args,
            config=train_saver_sft._build_config(args),
        )

        self.assertEqual(config.max_total_images, 18)

    def test_parse_args_defaults_teacher_judge_input_mode_to_auto(self):
        self.assertIsNotNone(train_saver_sft, "train_saver_sft.py is missing")

        args = train_saver_sft.parse_args(
            [
                "--data",
                "/tmp/data.jsonl",
                "--output-dir",
                "/tmp/sft_out",
            ]
        )

        self.assertEqual(args.teacher_judge_input_mode, "auto")

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

    def test_main_writes_summary_and_run_config_logs_under_output_dir(self):
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
            ]

            with patch.object(sys, "argv", argv), patch.object(
                train_saver_sft,
                "run_weighted_sft",
                return_value={"ok": True, "train_loss": 0.123},
            ):
                train_saver_sft.main()

            log_dir = output_dir / "logs"
            run_config_path = log_dir / "train_saver_sft_run_config.json"
            summary_path = log_dir / "train_saver_sft_summary.json"

            self.assertTrue(run_config_path.exists())
            self.assertTrue(summary_path.exists())

            run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(run_config["prepared_data"], str(prepared_path))
        self.assertEqual(run_config["output_dir"], str(output_dir))
        self.assertAlmostEqual(summary["train_loss"], 0.123, places=6)
        self.assertEqual(summary["num_examples"], 1)

    def test_main_can_annotate_verify_examples_with_teacher_judge_before_training(self):
        self.assertIsNotNone(train_saver_sft, "train_saver_sft.py is missing")

        verify_example = {
            "video_id": "prepared_case",
            "split": "train",
            "target_action": "tool_call",
            "tool_name": "verify_hypothesis",
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "system"}]},
                {"role": "user", "content": [{"type": "text", "text": "user"}]},
            ],
            "target_response": (
                '<think>verify</think><tool_call>{"name":"verify_hypothesis","arguments":'
                '{"verification_mode":"full_keep_drop","claim":{"existence":"anomaly","category":"assault"},'
                '"selected_window_ids":["w0001"],"verification_decision":"sufficient","recommended_action":"finalize"}}'
                "</tool_call>"
            ),
        }
        answer_example = {
            "video_id": "prepared_case",
            "split": "train",
            "target_action": "answer",
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "system"}]},
                {"role": "user", "content": [{"type": "text", "text": "user"}]},
            ],
            "target_response": '<answer>{"existence":"anomaly","category":"assault"}</answer>',
        }

        class DummyTeacherJudge:
            def annotate_example(self, example, *, input_mode=None):
                updated = dict(example)
                updated["teacher_judge_scores"] = {"sufficiency": 0.93, "necessity": 0.61}
                updated["teacher_judge_decision"] = "sufficient"
                updated["teacher_judge_rationale"] = "Teacher judge agrees."
                return updated

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prepared_path = root / "prepared.jsonl"
            prepared_path.write_text(
                json.dumps(verify_example) + "\n" + json.dumps(answer_example) + "\n",
                encoding="utf-8",
            )
            output_dir = root / "sft_out"
            teacher_output = root / "prepared.teacher.jsonl"
            argv = [
                "train_saver_sft.py",
                "--prepared-data",
                str(prepared_path),
                "--output-dir",
                str(output_dir),
                "--teacher-judge-model-path",
                "/models/Qwen3-VL-32B-Instruct",
                "--teacher-judge-output",
                str(teacher_output),
            ]

            with patch.object(
                sys,
                "argv",
                argv,
            ), patch.object(
                train_saver_sft,
                "run_weighted_sft",
                return_value={"ok": True},
            ) as mocked_run_weighted_sft, patch.object(
                train_saver_sft.QwenTeacherJudge,
                "from_pretrained",
                return_value=DummyTeacherJudge(),
            ) as mocked_teacher_loader:
                train_saver_sft.main()
                persisted = [
                    json.loads(line)
                    for line in teacher_output.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]

        mocked_teacher_loader.assert_called_once()
        examples = mocked_run_weighted_sft.call_args.args[0]
        self.assertEqual(examples[0]["teacher_judge_decision"], "sufficient")
        self.assertIn("teacher_judge_scores", examples[0])
        self.assertGreater(float(examples[0]["sample_weight"]), 1.0)
        self.assertNotIn("teacher_judge_decision", examples[1])
        self.assertEqual(persisted[0]["teacher_judge_decision"], "sufficient")
        self.assertGreater(float(persisted[0]["sample_weight"]), 1.0)
        self.assertNotIn("teacher_judge_decision", persisted[1])

    def test_main_passes_teacher_judge_topk_frames_per_view(self):
        self.assertIsNotNone(train_saver_sft, "train_saver_sft.py is missing")

        verify_example = {
            "video_id": "prepared_case",
            "split": "train",
            "target_action": "tool_call",
            "tool_name": "verify_hypothesis",
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "system"}]},
                {"role": "user", "content": [{"type": "text", "text": "user"}]},
            ],
            "target_response": (
                '<think>verify</think><tool_call>{"name":"verify_hypothesis","arguments":'
                '{"verification_mode":"full_keep_drop","claim":{"existence":"anomaly","category":"assault"},'
                '"selected_window_ids":["w0001"],"verification_decision":"sufficient","recommended_action":"finalize"}}'
                "</tool_call>"
            ),
        }

        class DummyTeacherJudge:
            def annotate_example(self, example, *, input_mode=None):
                updated = dict(example)
                updated["teacher_judge_scores"] = {"sufficiency": 0.9, "necessity": 0.6}
                updated["teacher_judge_decision"] = "sufficient"
                return updated

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prepared_path = root / "prepared.jsonl"
            prepared_path.write_text(json.dumps(verify_example) + "\n", encoding="utf-8")
            output_dir = root / "sft_out"
            argv = [
                "train_saver_sft.py",
                "--prepared-data",
                str(prepared_path),
                "--output-dir",
                str(output_dir),
                "--teacher-judge-model-path",
                "/models/Qwen3-VL-32B-Instruct",
                "--teacher-judge-topk-frames-per-view",
                "3",
            ]

            with patch.object(sys, "argv", argv), patch.object(
                train_saver_sft,
                "run_weighted_sft",
                return_value={"ok": True},
            ), patch.object(
                train_saver_sft.QwenTeacherJudge,
                "from_pretrained",
                return_value=DummyTeacherJudge(),
            ) as mocked_teacher_loader:
                train_saver_sft.main()

        self.assertEqual(mocked_teacher_loader.call_args.kwargs["topk_frames_per_view"], 3)

    def test_main_does_not_reweight_prepared_teacher_examples_twice(self):
        self.assertIsNotNone(train_saver_sft, "train_saver_sft.py is missing")

        prepared_example = {
            "video_id": "prepared_case",
            "split": "train",
            "target_action": "tool_call",
            "tool_name": "verify_hypothesis",
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "system"}]},
                {"role": "user", "content": [{"type": "text", "text": "user"}]},
            ],
            "target_response": (
                '<think>verify</think><tool_call>{"name":"verify_hypothesis","arguments":'
                '{"verification_mode":"full_keep_drop","claim":{"existence":"anomaly","category":"assault"},'
                '"selected_window_ids":["w0002"],"verification_decision":"insufficient","recommended_action":"continue_search"}}'
                "</tool_call>"
            ),
            "teacher_judge_scores": {
                "sufficiency": 0.92,
                "necessity": 0.84,
                "alertability": 0.61,
                "counterfactual_faithfulness": 0.88,
            },
            "teacher_judge_decision": "insufficient",
            "teacher_judge_rationale": "Need more evidence.",
            "teacher_judge_weight_multiplier": 1.5,
            "teacher_judge_effective_sample_weight": 1.5,
            "sample_weight": 1.5,
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
            ]

            with patch.object(
                sys,
                "argv",
                argv,
            ), patch.object(
                train_saver_sft,
                "run_weighted_sft",
                return_value={"ok": True},
            ) as mocked_run_weighted_sft:
                train_saver_sft.main()

        examples = mocked_run_weighted_sft.call_args.args[0]
        self.assertEqual(float(examples[0]["sample_weight"]), 1.5)

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
