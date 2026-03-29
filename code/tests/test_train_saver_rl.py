import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path("/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


try:
    import train_saver_rl as train_saver_rl
except ModuleNotFoundError:
    train_saver_rl = None


class TrainSaverRlTests(unittest.TestCase):
    def test_resolve_reference_model_path_defaults_to_initial_model(self):
        self.assertIsNotNone(train_saver_rl, "train_saver_rl.py is missing")

        resolved = train_saver_rl.resolve_reference_model_path(
            model_path="/models/policy_sft",
            reference_model_path="",
        )

        self.assertEqual(resolved, "/models/policy_sft")

    def test_resolve_reference_model_path_prefers_explicit_path(self):
        self.assertIsNotNone(train_saver_rl, "train_saver_rl.py is missing")

        resolved = train_saver_rl.resolve_reference_model_path(
            model_path="/models/policy_sft",
            reference_model_path="/models/policy_reference",
        )

        self.assertEqual(resolved, "/models/policy_reference")

    def test_build_training_kwargs_passes_reference_policy_and_kl_settings(self):
        self.assertIsNotNone(train_saver_rl, "train_saver_rl.py is missing")

        args = train_saver_rl.parse_args(
            [
                "--data",
                "/tmp/data.jsonl",
                "--output-dir",
                "/tmp/rl_out",
                "--reference-model-path",
                "/models/policy_reference",
                "--kl-beta",
                "0.025",
                "--lora-target-modules",
                "q_proj,k_proj",
            ]
        )

        kwargs = train_saver_rl.build_training_kwargs(
            current_model_path="/models/policy_iter_1",
            checkpoint_dir="/tmp/rl_out/iter_000/checkpoint",
            args=args,
            reference_model_path="/models/policy_reference",
            config=train_saver_rl._build_config(args),
        )

        self.assertEqual(kwargs["model_path"], "/models/policy_iter_1")
        self.assertEqual(kwargs["output_dir"], "/tmp/rl_out/iter_000/checkpoint")
        self.assertEqual(kwargs["reference_model_path"], "/models/policy_reference")
        self.assertAlmostEqual(kwargs["kl_beta"], 0.025, places=6)
        self.assertEqual(kwargs["lora_target_modules"], ["q_proj", "k_proj"])

    def test_build_training_kwargs_can_enable_reference_diagnostics_for_epoch_end_eval(self):
        self.assertIsNotNone(train_saver_rl, "train_saver_rl.py is missing")

        args = train_saver_rl.parse_args(
            [
                "--data",
                "/tmp/data.jsonl",
                "--output-dir",
                "/tmp/rl_out",
                "--eval-data",
                "/tmp/eval.jsonl",
                "--eval-attach-reference-diagnostics",
            ]
        )

        kwargs = train_saver_rl.build_training_kwargs(
            current_model_path="/models/policy_iter_1",
            checkpoint_dir="/tmp/rl_out/iter_000/checkpoint",
            args=args,
            reference_model_path="/models/policy_reference",
            config=train_saver_rl._build_config(args),
        )

        self.assertTrue(kwargs["rollout_eval_config"].attach_reference_diagnostics)

    def test_build_training_kwargs_defaults_to_grpo_objective_and_clip_settings(self):
        self.assertIsNotNone(train_saver_rl, "train_saver_rl.py is missing")

        args = train_saver_rl.parse_args(
            [
                "--data",
                "/tmp/data.jsonl",
                "--output-dir",
                "/tmp/rl_out",
                "--ppo-clip-epsilon",
                "0.18",
            ]
        )

        kwargs = train_saver_rl.build_training_kwargs(
            current_model_path="/models/policy_iter_1",
            checkpoint_dir="/tmp/rl_out/iter_000/checkpoint",
            args=args,
            reference_model_path="/models/policy_reference",
            config=train_saver_rl._build_config(args),
        )

        self.assertEqual(kwargs["training_objective"], "grpo")
        self.assertEqual(kwargs["old_policy_model_path"], "/models/policy_iter_1")
        self.assertAlmostEqual(kwargs["ppo_clip_epsilon"], 0.18, places=6)

    def test_build_training_kwargs_passes_explicit_text_and_vision_budgets(self):
        self.assertIsNotNone(train_saver_rl, "train_saver_rl.py is missing")

        args = train_saver_rl.parse_args(
            [
                "--data",
                "/tmp/data.jsonl",
                "--output-dir",
                "/tmp/rl_out",
                "--max-image-side",
                "640",
                "--max-image-pixels",
                "307200",
                "--keep-recent-tool-image-messages",
                "3",
                "--max-total-images",
                "44",
                "--max-seq-length",
                "4096",
                "--keep-recent-text-messages",
                "12",
            ]
        )

        kwargs = train_saver_rl.build_training_kwargs(
            current_model_path="/models/policy_iter_1",
            checkpoint_dir="/tmp/rl_out/iter_000/checkpoint",
            args=args,
            reference_model_path="/models/policy_reference",
            config=train_saver_rl._build_config(args),
        )

        self.assertEqual(kwargs["max_image_side"], 640)
        self.assertEqual(kwargs["max_image_pixels"], 307200)
        self.assertEqual(kwargs["keep_recent_tool_image_messages"], 3)
        self.assertEqual(kwargs["max_total_images"], 44)
        self.assertEqual(kwargs["max_seq_length"], 4096)
        self.assertEqual(kwargs["keep_recent_text_messages"], 12)

    def test_build_training_kwargs_passes_dataloader_knobs(self):
        self.assertIsNotNone(train_saver_rl, "train_saver_rl.py is missing")

        args = train_saver_rl.parse_args(
            [
                "--data",
                "/tmp/data.jsonl",
                "--output-dir",
                "/tmp/rl_out",
                "--dataloader-num-workers",
                "2",
                "--dataloader-prefetch-factor",
                "3",
                "--dataloader-persistent-workers",
            ]
        )

        kwargs = train_saver_rl.build_training_kwargs(
            current_model_path="/models/policy_iter_1",
            checkpoint_dir="/tmp/rl_out/iter_000/checkpoint",
            args=args,
            reference_model_path="/models/policy_reference",
            config=train_saver_rl._build_config(args),
        )

        self.assertEqual(kwargs["dataloader_num_workers"], 2)
        self.assertEqual(kwargs["dataloader_prefetch_factor"], 3)
        self.assertTrue(kwargs["dataloader_persistent_workers"])

    def test_parse_args_supports_cea_grpo_variant_and_counterfactual_settings(self):
        self.assertIsNotNone(train_saver_rl, "train_saver_rl.py is missing")

        args = train_saver_rl.parse_args(
            [
                "--data",
                "/tmp/data.jsonl",
                "--output-dir",
                "/tmp/rl_out",
                "--grpo-variant",
                "cea_grpo",
                "--cea-enable-alert-group",
                "--cea-enable-evidence-group",
                "--cea-enable-search-group",
                "--cea-alert-local-alpha",
                "0.6",
                "--cea-evidence-local-alpha",
                "0.7",
                "--cea-search-local-alpha",
                "0.4",
                "--cea-local-use-reference-supervision",
            ]
        )

        self.assertEqual(args.grpo_variant, "cea_grpo")
        self.assertTrue(args.cea_enable_alert_group)
        self.assertTrue(args.cea_enable_evidence_group)
        self.assertTrue(args.cea_enable_search_group)
        self.assertAlmostEqual(args.cea_alert_local_alpha, 0.6, places=6)
        self.assertAlmostEqual(args.cea_evidence_local_alpha, 0.7, places=6)
        self.assertAlmostEqual(args.cea_search_local_alpha, 0.4, places=6)
        self.assertTrue(args.cea_local_use_reference_supervision)

    def test_parse_args_defaults_to_cea_grpo_with_both_local_groups_enabled(self):
        self.assertIsNotNone(train_saver_rl, "train_saver_rl.py is missing")

        args = train_saver_rl.parse_args(
            [
                "--data",
                "/tmp/data.jsonl",
                "--output-dir",
                "/tmp/rl_out",
            ]
        )

        self.assertEqual(args.grpo_variant, "cea_grpo")
        enable_alert_group, enable_evidence_group = train_saver_rl._resolve_cea_group_settings(args)
        enable_search_group = train_saver_rl._resolve_cea_search_group_enabled(args)
        self.assertTrue(enable_alert_group)
        self.assertTrue(enable_evidence_group)
        self.assertTrue(enable_search_group)

    def test_expand_grouped_rollout_specs_repeats_each_index_for_num_generations(self):
        self.assertIsNotNone(train_saver_rl, "train_saver_rl.py is missing")

        specs = train_saver_rl.expand_grouped_rollout_specs(indices=[3, 5], num_generations=3)

        self.assertEqual(len(specs), 6)
        self.assertEqual(specs[0]["dataset_index"], 3)
        self.assertEqual(specs[0]["group_id"], "idx000003")
        self.assertEqual(specs[0]["generation_id"], 0)
        self.assertEqual(specs[-1]["dataset_index"], 5)
        self.assertEqual(specs[-1]["group_id"], "idx000005")
        self.assertEqual(specs[-1]["generation_id"], 2)

    def test_compute_group_relative_advantages_normalizes_rewards_within_each_group(self):
        self.assertIsNotNone(train_saver_rl, "train_saver_rl.py is missing")

        scored_records = [
            {"video_id": "a1", "group_id": "g1", "reward_summary": {"total_reward": 2.0}},
            {"video_id": "a2", "group_id": "g1", "reward_summary": {"total_reward": 1.0}},
            {"video_id": "a3", "group_id": "g1", "reward_summary": {"total_reward": -1.0}},
            {"video_id": "b1", "group_id": "g2", "reward_summary": {"total_reward": 0.5}},
            {"video_id": "b2", "group_id": "g2", "reward_summary": {"total_reward": 0.5}},
        ]

        advantaged = train_saver_rl.compute_group_relative_advantages(scored_records)

        g1 = [record for record in advantaged if record["group_id"] == "g1"]
        g2 = [record for record in advantaged if record["group_id"] == "g2"]
        self.assertGreater(g1[0]["group_advantage"], g1[1]["group_advantage"])
        self.assertGreater(g1[1]["group_advantage"], g1[2]["group_advantage"])
        self.assertAlmostEqual(sum(record["group_advantage"] for record in g2), 0.0, places=6)

    def test_filter_examples_by_advantage_keeps_signed_examples_above_abs_threshold(self):
        self.assertIsNotNone(train_saver_rl, "train_saver_rl.py is missing")

        examples = [
            {"video_id": "a", "sample_weight": 1.2},
            {"video_id": "b", "sample_weight": -0.3},
            {"video_id": "c", "sample_weight": 0.0},
            {"video_id": "d", "sample_weight": 0.8},
        ]

        filtered = train_saver_rl.filter_reward_weighted_examples(examples, min_weight=0.1)

        self.assertEqual([example["video_id"] for example in filtered], ["a", "b", "d"])

    def test_select_iteration_indices_returns_contiguous_slice_with_wraparound(self):
        self.assertIsNotNone(train_saver_rl, "train_saver_rl.py is missing")

        indices = train_saver_rl.select_iteration_indices(
            dataset_size=5,
            rollout_count=4,
            start_index=3,
            iteration=1,
        )

        self.assertEqual(indices, [2, 3, 4, 0])

    def test_build_reward_examples_from_scored_rollouts_uses_turn_level_advantages(self):
        self.assertIsNotNone(train_saver_rl, "train_saver_rl.py is missing")

        sample = {
            "schema_version": "saver_agent.v1",
            "video_id": "rl_turn_level_case",
            "video_path": "videos/rl_turn_level_case.mp4",
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
                "allowed_tools": ["scan_timeline", "finalize_case"],
                "initial_scan_window_frames": [1, 16],
                "initial_scan_window_sec": [0.0, 4.0],
                "oracle_windows_frames": [],
                "oracle_windows_sec": [],
                "finalize_case_schema": {"type": "object", "required": ["existence"]},
            },
            "auto_completed": {"precursor_interval": False},
        }
        scored_rollout = {
            "video_id": "rl_turn_level_case",
            "group_id": "idx000000",
            "generation_id": 0,
            "group_advantage": 1.0,
            "reward_summary": {"total_reward": 1.2},
            "turns": [
                {
                    "step_index": 1,
                    "assistant_response_raw": '<think>scan</think><tool_call>{"name":"scan_timeline","arguments":{"start_sec":0.0,"end_sec":3.0,"num_frames":2}}</tool_call>',
                    "action": "tool_call",
                    "valid_action": True,
                    "tool_name": "scan_timeline",
                    "new_evidence_ids": [],
                    "new_finalized_case": None,
                    "verifier_primary_status": None,
                    "verifier_alert_status": None,
                    "verifier_derived_scores": None,
                },
                {
                    "step_index": 2,
                    "assistant_response_raw": '<think>done</think><answer>{"existence":"normal"}</answer>',
                    "action": "answer",
                    "valid_action": True,
                    "tool_name": None,
                    "new_evidence_ids": [],
                    "new_finalized_case": None,
                    "verifier_primary_status": None,
                    "verifier_alert_status": None,
                    "verifier_derived_scores": None,
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data.jsonl"
            data_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

            args = train_saver_rl.parse_args(
                [
                    "--data",
                    str(data_path),
                    "--data-root",
                    tmpdir,
                    "--output-dir",
                    "/tmp/rl_out",
                ]
            )

            examples = train_saver_rl.build_reward_examples_from_scored_rollouts(
                data_path=data_path,
                data_root=tmpdir,
                scored_records=[scored_rollout],
                args=args,
            )

        self.assertEqual(len(examples), 2)
        self.assertLess(examples[0]["advantage"], examples[1]["advantage"])
        self.assertEqual(examples[0]["sample_weight"], max(examples[0]["advantage"], 0.0))
        self.assertEqual(examples[1]["sample_weight"], max(examples[1]["advantage"], 0.0))

    def test_load_jsonl_filters_by_split(self):
        self.assertIsNotNone(train_saver_rl, "train_saver_rl.py is missing")

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data.jsonl"
            data_path.write_text(
                json.dumps({"video_id": "test_case", "split": "test"}) + "\n"
                + json.dumps({"video_id": "train_case", "split": "train"}) + "\n",
                encoding="utf-8",
            )

            records = train_saver_rl._load_jsonl(data_path, include_splits="train")

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["video_id"], "train_case")


if __name__ == "__main__":
    unittest.main()
