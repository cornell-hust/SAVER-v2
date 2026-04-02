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

    def test_parse_args_accepts_rl_tensor_cache_flags(self):
        self.assertIsNotNone(train_saver_rl, "train_saver_rl.py is missing")

        args = train_saver_rl.parse_args(
            [
                "--data",
                "/tmp/data.jsonl",
                "--output-dir",
                "/tmp/rl_out",
                "--tensor-cache-dir",
                "/tmp/rl_tensor_cache",
                "--tensor-cache-progress-every",
                "9",
                "--tensor-cache-overwrite-existing",
            ]
        )

        self.assertEqual(args.tensor_cache_dir, "/tmp/rl_tensor_cache")
        self.assertEqual(args.tensor_cache_progress_every, 9)
        self.assertTrue(args.tensor_cache_overwrite_existing)

    def test_resolve_iteration_tensor_cache_dir_appends_iteration_suffix(self):
        self.assertIsNotNone(train_saver_rl, "train_saver_rl.py is missing")

        resolved = train_saver_rl.resolve_iteration_tensor_cache_dir(
            "/tmp/rl_tensor_cache",
            iteration=3,
        )

        self.assertEqual(str(resolved), "/tmp/rl_tensor_cache/iter_003")

    def test_build_training_kwargs_passes_tensor_cache_dir(self):
        self.assertIsNotNone(train_saver_rl, "train_saver_rl.py is missing")

        args = train_saver_rl.parse_args(
            [
                "--data",
                "/tmp/data.jsonl",
                "--output-dir",
                "/tmp/rl_out",
                "--tensor-cache-dir",
                "/tmp/rl_tensor_cache",
            ]
        )

        kwargs = train_saver_rl.build_training_kwargs(
            current_model_path="/models/policy_iter_1",
            checkpoint_dir="/tmp/rl_out/iter_000/checkpoint",
            args=args,
            reference_model_path="/models/policy_reference",
            config=train_saver_rl._build_config(args),
            tensor_cache_dir="/tmp/rl_tensor_cache/iter_000",
        )

        self.assertEqual(kwargs["tensor_cache_dir"], "/tmp/rl_tensor_cache/iter_000")

    def test_parse_args_defaults_rollout_turn_limits_to_twelve(self):
        self.assertIsNotNone(train_saver_rl, "train_saver_rl.py is missing")

        args = train_saver_rl.parse_args(
            [
                "--data",
                "/tmp/data.jsonl",
                "--output-dir",
                "/tmp/rl_out",
            ]
        )

        self.assertEqual(args.rollout_max_turns, 12)
        self.assertEqual(args.eval_rollout_max_turns, 12)
        self.assertEqual(args.teacher_judge_input_mode, "auto")
        self.assertEqual(args.cea_local_verifier_backend, "self_teacher")
        self.assertEqual(args.policy_max_new_tokens, 256)
        self.assertEqual(args.max_total_images, 24)
        self.assertEqual(args.eval_max_total_images, 24)

    def test_build_policy_passes_rollout_visual_budget_to_generation_policy(self):
        self.assertIsNotNone(train_saver_rl, "train_saver_rl.py is missing")

        args = train_saver_rl.parse_args(
            [
                "--data",
                "/tmp/data.jsonl",
                "--output-dir",
                "/tmp/rl_out",
                "--max-total-images",
                "18",
            ]
        )
        runtime = train_saver_rl.distributed_runtime_from_env()

        with patch("train_saver_rl.QwenGenerationPolicy.from_pretrained") as from_pretrained:
            train_saver_rl._build_policy("/models/policy_iter_1", args, runtime=runtime)

        _, kwargs = from_pretrained.call_args
        self.assertEqual(kwargs["max_total_images"], 18)

    def test_parse_args_supports_teacher_anchor_policy_and_frame_budget(self):
        self.assertIsNotNone(train_saver_rl, "train_saver_rl.py is missing")

        args = train_saver_rl.parse_args(
            [
                "--data",
                "/tmp/data.jsonl",
                "--output-dir",
                "/tmp/rl_out",
                "--teacher-judge-anchor-policy",
                "hard_examples",
                "--teacher-judge-topk-frames-per-view",
                "3",
            ]
        )

        self.assertEqual(args.teacher_judge_anchor_policy, "hard_examples")
        self.assertEqual(args.teacher_judge_topk_frames_per_view, 3)

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

    def test_build_training_kwargs_passes_eval_proposal_runtime_fields(self):
        self.assertIsNotNone(train_saver_rl, "train_saver_rl.py is missing")

        args = train_saver_rl.parse_args(
            [
                "--data",
                "/tmp/data.jsonl",
                "--output-dir",
                "/tmp/rl_out",
                "--eval-data",
                "/tmp/eval.jsonl",
                "--proposal-model-path",
                "/models/siglip_base",
                "--eval-proposal-torch-dtype",
                "bfloat16",
                "--eval-proposal-device",
                "cuda:2",
            ]
        )

        kwargs = train_saver_rl.build_training_kwargs(
            current_model_path="/models/policy_iter_1",
            checkpoint_dir="/tmp/rl_out/iter_000/checkpoint",
            args=args,
            reference_model_path="/models/policy_reference",
            config=train_saver_rl._build_config(args),
        )

        eval_config = kwargs["rollout_eval_config"]
        self.assertEqual(str(eval_config.proposal_model_path), "/models/siglip_base")
        self.assertEqual(eval_config.proposal_torch_dtype, "bfloat16")
        self.assertEqual(eval_config.proposal_device, "cuda:2")

    def test_build_training_kwargs_passes_eval_visual_budget_and_output_budget_fields(self):
        self.assertIsNotNone(train_saver_rl, "train_saver_rl.py is missing")

        args = train_saver_rl.parse_args(
            [
                "--data",
                "/tmp/data.jsonl",
                "--output-dir",
                "/tmp/rl_out",
                "--eval-data",
                "/tmp/eval.jsonl",
                "--eval-total-visual-budget",
                "24",
                "--eval-max-new-tokens-per-turn",
                "256",
            ]
        )

        kwargs = train_saver_rl.build_training_kwargs(
            current_model_path="/models/policy_iter_1",
            checkpoint_dir="/tmp/rl_out/iter_000/checkpoint",
            args=args,
            reference_model_path="/models/policy_reference",
            config=train_saver_rl._build_config(args),
        )

        eval_config = kwargs["rollout_eval_config"]
        self.assertEqual(eval_config.policy_max_new_tokens, 256)
        self.assertEqual(eval_config.max_total_images, 24)

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

    def test_collect_rollouts_attaches_proposal_runtime_to_multimodal_cache(self):
        self.assertIsNotNone(train_saver_rl, "train_saver_rl.py is missing")

        captured = {}

        class DummyDataset:
            def __init__(self, *args, **kwargs):
                self.records = [{"video_id": "vid_1"}]

            def format_frame_cache_status(self, *, prefix="frame cache", max_examples=5):
                return f"{prefix}: ok"

            def __getitem__(self, index):
                return {"video_id": "vid_1", "multimodal_cache": {}}

        class DummyRunner:
            def __init__(self, *args, **kwargs):
                pass

            def run_episode(self, item, policy):
                captured["proposal_runtime"] = item["multimodal_cache"].get("proposal_runtime")
                return {"video_id": item["video_id"], "turns": [], "state": {}, "num_turns": 1}

        args = train_saver_rl.parse_args(
            [
                "--data",
                "/tmp/data.jsonl",
                "--output-dir",
                "/tmp/rl_out",
            ]
        )

        with patch("train_saver_rl.SaverAgentDataset", DummyDataset), patch(
            "train_saver_rl.SaverRolloutRunner",
            DummyRunner,
        ), patch("train_saver_rl._build_policy", return_value=object()), patch(
            "train_saver_rl._serialize_result",
            side_effect=lambda result: dict(result),
        ):
            rollouts = train_saver_rl.collect_rollouts(
                data_path="/tmp/data.jsonl",
                data_root="/tmp",
                rollout_specs=[{"dataset_index": 0, "group_id": "g1", "generation_id": 0}],
                model_path="/models/policy",
                args=args,
                runtime=train_saver_rl.distributed_runtime_from_env(),
                verifier_runtime=None,
                proposal_runtime="siglip_runtime",
            )

        self.assertEqual(captured["proposal_runtime"], "siglip_runtime")
        self.assertEqual(len(rollouts), 1)

    def test_prepare_tensor_cache_for_examples_writes_metadata_and_entries(self):
        self.assertIsNotNone(train_saver_rl, "train_saver_rl.py is missing")

        example = {
            "video_id": "cache_case",
            "split": "train",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
            "target_response": "<answer>{\"existence\":\"normal\"}</answer>",
            "sample_weight": 1.0,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "rl_tensor_cache" / "iter_000"
            with patch("train_saver_rl._load_tensor_cache_processor", return_value=object()), patch(
                "train_saver_rl.build_processor_signature",
                return_value="fake_signature",
            ), patch(
                "train_saver_rl.build_processor_signature_summary",
                return_value={"processor": "fake"},
            ), patch(
                "train_saver_rl.materialize_example_for_training",
                side_effect=lambda feature, resolver=None: dict(feature),
            ), patch(
                "train_saver_rl.build_sft_tensor_cache_payload",
                return_value={"input_ids": [1, 2], "labels": [1, 2]},
            ), patch(
                "train_saver_rl.distributed_barrier",
                return_value=None,
            ):
                summary = train_saver_rl.prepare_tensor_cache_for_examples(
                    [example],
                    cache_dir=cache_dir,
                    model_path="/models/policy_iter_1",
                    runtime=train_saver_rl.distributed_runtime_from_env(),
                )

            metadata_path = cache_dir / "metadata.json"
            self.assertTrue(metadata_path.exists())
            self.assertEqual(summary["num_examples_total"], 1)
            self.assertEqual(summary["num_built"], 1)
            entry_files = list((cache_dir / "entries").rglob("*.pt"))
            self.assertEqual(len(entry_files), 1)

    def test_main_writes_central_rl_logs_under_output_dir(self):
        self.assertIsNotNone(train_saver_rl, "train_saver_rl.py is missing")

        sample_record = {"video_id": "rl_case", "split": "train"}
        fake_rollout = {
            "video_id": "rl_case",
            "dataset_index": 0,
            "group_id": "g0",
            "generation_id": 0,
            "turns": [],
            "reward_summary": {"total_reward": 0.5},
            "num_turns": 1,
        }
        fake_scored_rollout = {
            **fake_rollout,
            "group_advantage": 0.25,
            "advantage": 0.25,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_path = root / "data.jsonl"
            data_path.write_text(json.dumps(sample_record) + "\n", encoding="utf-8")
            output_dir = root / "rl_out"
            argv = [
                "train_saver_rl.py",
                "--data",
                str(data_path),
                "--output-dir",
                str(output_dir),
                "--num-iterations",
                "1",
                "--rollout-count",
                "1",
                "--num-generations",
                "1",
                "--dry-run",
            ]

            with patch.object(sys, "argv", argv), patch.object(
                train_saver_rl,
                "ReferenceDataProvider",
                return_value=object(),
            ), patch.object(
                train_saver_rl,
                "collect_rollouts",
                return_value=[fake_rollout],
            ), patch.object(
                train_saver_rl,
                "score_rollout_records",
                return_value=[fake_scored_rollout],
            ), patch.object(
                train_saver_rl,
                "compute_group_relative_advantages",
                side_effect=lambda records, clip_value: records,
            ), patch.object(
                train_saver_rl,
                "build_training_examples_from_scored_rollouts",
                return_value=[
                    {
                        "video_id": "rl_case",
                        "messages": [{"role": "user", "content": [{"type": "text", "text": "user"}]}],
                        "target_response": "<answer>{}</answer>",
                        "sample_weight": 1.0,
                    }
                ],
            ), patch.object(
                train_saver_rl,
                "distributed_barrier",
                return_value=None,
            ):
                train_saver_rl.main()

            log_dir = output_dir / "logs"
            run_config_path = log_dir / "train_saver_rl_run_config.json"
            iteration_metrics_path = log_dir / "rl_iteration_metrics.jsonl"
            final_summary_path = log_dir / "train_saver_rl_final_summary.json"

            self.assertTrue(run_config_path.exists())
            self.assertTrue(iteration_metrics_path.exists())
            self.assertTrue(final_summary_path.exists())

            run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
            iteration_records = [
                json.loads(line)
                for line in iteration_metrics_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            final_summary = json.loads(final_summary_path.read_text(encoding="utf-8"))

        self.assertEqual(run_config["output_dir"], str(output_dir))
        self.assertEqual(run_config["num_iterations"], 1)
        self.assertEqual(len(iteration_records), 1)
        self.assertEqual(iteration_records[0]["iteration"], 0)
        self.assertEqual(iteration_records[0]["num_reward_examples"], 1)
        self.assertIn("latest_checkpoint", final_summary)
        self.assertEqual(final_summary["num_iterations"], 1)

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

    def test_parse_args_accepts_teacher_judge_rl_flags(self):
        self.assertIsNotNone(train_saver_rl, "train_saver_rl.py is missing")

        args = train_saver_rl.parse_args(
            [
                "--data",
                "/tmp/data.jsonl",
                "--output-dir",
                "/tmp/rl_out",
                "--teacher-judge-model-path",
                "/models/qwen3-vl-32b",
                "--teacher-judge-input-mode",
                "multimodal_visual",
                "--teacher-judge-local-alpha",
                "0.6",
            ]
        )

        self.assertEqual(args.teacher_judge_model_path, "/models/qwen3-vl-32b")
        self.assertEqual(args.teacher_judge_input_mode, "multimodal_visual")
        self.assertAlmostEqual(args.teacher_judge_local_alpha, 0.6, places=6)

    def test_attach_teacher_judge_to_rollouts_updates_verify_turns(self):
        self.assertIsNotNone(train_saver_rl, "train_saver_rl.py is missing")

        sample = {
            "schema_version": "saver_agent.v1",
            "video_id": "teacher_rollout_case",
            "video_path": "videos/teacher_rollout_case.mp4",
            "split": "train",
            "source_dataset": "MSAD",
            "frame_index_base": 1,
            "video_meta": {"fps": 4.0, "duration_sec": 4.0, "total_frames": 16},
            "scene": {"scenario": "shop"},
            "label": {"is_anomaly": True, "category": "assault", "severity": 4, "hard_normal": False},
            "agent_task": {
                "task_type": "video_anomaly_search_alert_verify",
                "query_mode": "internal_hypothesis_generation",
                "task_prompt": "Check whether anything abnormal happens.",
                "success_criteria": ["criterion_a"],
            },
            "structured_target": {
                "existence": "anomaly",
                "category": "assault",
                "severity": 4,
                "anomaly_interval_sec": [1.0, 3.0],
                "precursor_interval_sec": [0.5, 1.0],
                "earliest_alert_sec": 1.0,
                "evidence_moment_ids": ["ev1"],
                "counterfactual_type": "remove_actor_interaction",
                "summary": "Assault occurs.",
                "rationale": "A person attacks another person.",
            },
            "temporal": {
                "anomaly_interval_sec": [1.0, 3.0],
                "precursor_interval_sec": [0.5, 1.0],
                "earliest_alert_sec": 1.0,
            },
            "evidence": {
                "evidence_moments": [
                    {
                        "moment_id": "ev1",
                        "role": "trigger",
                        "start_sec": 1.0,
                        "end_sec": 3.0,
                        "description": "attack",
                    }
                ]
            },
            "tool_io": {
                "allowed_tools": ["scan_timeline", "verify_hypothesis", "finalize_case"],
                "initial_scan_window_frames": [1, 16],
                "initial_scan_window_sec": [0.0, 4.0],
                "oracle_windows_frames": [],
                "oracle_windows_sec": [{"moment_id": "ev1", "role": "trigger", "window": [1.0, 3.0]}],
                "finalize_case_schema": {"type": "object", "required": ["existence"]},
            },
            "auto_completed": {"precursor_interval": False},
        }
        rollout = {
            "video_id": "teacher_rollout_case",
            "dataset_index": 0,
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
            "reward_summary": {"total_reward": 0.5},
            "group_advantage": 0.3,
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
            data_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

            args = train_saver_rl.parse_args(
                [
                    "--data",
                    str(data_path),
                    "--data-root",
                    tmpdir,
                    "--output-dir",
                    "/tmp/rl_out",
                    "--teacher-judge-model-path",
                    "/models/qwen3-vl-32b",
                ]
            )

            annotated_rollouts, summary = train_saver_rl.attach_teacher_judge_to_rollouts(
                data_path=data_path,
                data_root=tmpdir,
                rollouts=[rollout],
                args=args,
                judge=DummyJudge(),
            )

        self.assertEqual(summary["num_teacher_judge_annotated_turns"], 1)
        verify_turn = annotated_rollouts[0]["turns"][0]
        self.assertEqual(verify_turn["teacher_judge_decision"], "sufficient")
        self.assertGreater(verify_turn["teacher_judge_reward"], 0.0)


if __name__ == "__main__":
    unittest.main()
