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
    import annotate_teacher_judge_sft as annotate_teacher_judge_sft
except ModuleNotFoundError:
    annotate_teacher_judge_sft = None


def _write_prepared_input(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    Path(str(path) + ".meta.json").write_text(
        json.dumps({"schema_version": 1, "preview": {}, "prompt": {}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


class AnnotateTeacherJudgeSftCliTests(unittest.TestCase):
    def test_resolve_teacher_judge_shard_indices_redistributes_only_verify_candidates(self):
        self.assertIsNotNone(annotate_teacher_judge_sft, "annotate_teacher_judge_sft.py is missing")

        rows = [
            {"video_id": "nonverify_even_0", "target_action": "answer"},
            {"video_id": "verify_0", "target_action": "tool_call", "tool_name": "verify_hypothesis"},
            {"video_id": "verify_1", "target_action": "tool_call", "tool_name": "verify_hypothesis"},
            {"video_id": "nonverify_odd_3", "target_action": "answer"},
            {"video_id": "nonverify_even_4", "target_action": "answer"},
            {"video_id": "verify_2", "target_action": "tool_call", "tool_name": "verify_hypothesis"},
            {"video_id": "nonverify_even_6", "target_action": "answer"},
            {"video_id": "verify_3", "target_action": "tool_call", "tool_name": "verify_hypothesis"},
        ]

        shard_indices = annotate_teacher_judge_sft._resolve_teacher_judge_shard_indices(rows, num_shards=2)

        self.assertEqual(shard_indices[0], [0, 1, 4, 5, 6])
        self.assertEqual(shard_indices[1], [2, 3, 7])

    def test_progress_visualizer_updates_rank_aware_scan_and_annotation_bars(self):
        self.assertIsNotNone(annotate_teacher_judge_sft, "annotate_teacher_judge_sft.py is missing")

        created_bars = []

        class FakeTqdm:
            def __init__(self, *, total, desc, position, leave, dynamic_ncols):
                self.total = total
                self.desc = desc
                self.position = position
                self.leave = leave
                self.dynamic_ncols = dynamic_ncols
                self.n = 0
                self.postfix = None
                self.closed = False
                created_bars.append(self)

            def update(self, delta):
                self.n += delta

            def set_postfix(self, payload, refresh=False):
                self.postfix = dict(payload)

            def close(self):
                self.closed = True

        runtime = annotate_teacher_judge_sft.distributed_runtime_from_env(
            {"RANK": "1", "WORLD_SIZE": "4", "LOCAL_RANK": "1"}
        )

        with patch.object(annotate_teacher_judge_sft, "_load_tqdm", return_value=FakeTqdm):
            reporter = annotate_teacher_judge_sft._ProgressVisualizer(runtime=runtime, enabled=True)
            reporter(
                {
                    "phase": "scan",
                    "completed": 3,
                    "total": 10,
                    "candidate_examples": 2,
                    "annotated_count": 0,
                    "skipped_existing": 1,
                }
            )
            reporter(
                {
                    "phase": "annotate",
                    "completed": 2,
                    "total": 5,
                    "candidate_examples": 5,
                    "annotated_count": 2,
                    "skipped_existing": 1,
                }
            )
            reporter.close()

        self.assertEqual(len(created_bars), 2)
        self.assertEqual(created_bars[0].position, 2)
        self.assertEqual(created_bars[1].position, 3)
        self.assertEqual(created_bars[0].n, 3)
        self.assertEqual(created_bars[1].n, 2)
        self.assertEqual(created_bars[1].postfix["annotated"], 2)
        self.assertTrue(all(bar.closed for bar in created_bars))

    def test_parse_args_defaults_input_mode_to_auto(self):
        self.assertIsNotNone(annotate_teacher_judge_sft, "annotate_teacher_judge_sft.py is missing")

        args = annotate_teacher_judge_sft.parse_args(
            [
                "--input",
                "/tmp/prepared.jsonl",
                "--output",
                "/tmp/prepared.teacher.jsonl",
                "--model-path",
                "/models/Qwen3-VL-32B-Instruct",
            ]
        )

        self.assertEqual(args.input_mode, "auto")
        self.assertEqual(args.batch_size, 1)

    def test_main_annotates_only_verify_hypothesis_examples(self):
        self.assertIsNotNone(annotate_teacher_judge_sft, "annotate_teacher_judge_sft.py is missing")

        verify_example = {
            "video_id": "verify_case",
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
            "video_id": "answer_case",
            "split": "train",
            "target_action": "answer",
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "system"}]},
                {"role": "user", "content": [{"type": "text", "text": "user"}]},
            ],
            "target_response": '<answer>{"existence":"normal","category":"normal"}</answer>',
        }

        class DummyTeacherJudge:
            def annotate_example(self, example, *, input_mode=None):
                updated = dict(example)
                updated["teacher_judge_scores"] = {"sufficiency": 0.87, "necessity": 0.49}
                updated["teacher_judge_decision"] = "sufficient"
                updated["teacher_judge_rationale"] = "Teacher judge supports the selected evidence."
                return updated

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "prepared.jsonl"
            output_path = root / "prepared.teacher.jsonl"
            _write_prepared_input(input_path, [verify_example, answer_example])

            with patch.object(
                annotate_teacher_judge_sft.QwenTeacherJudge,
                "from_pretrained",
                return_value=DummyTeacherJudge(),
            ) as mocked_loader:
                annotate_teacher_judge_sft.main(
                    [
                        "--input",
                        str(input_path),
                        "--output",
                        str(output_path),
                        "--model-path",
                        "/models/Qwen3-VL-32B-Instruct",
                    ]
                )
                rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]

        mocked_loader.assert_called_once()
        self.assertEqual(rows[0]["teacher_judge_decision"], "sufficient")
        self.assertIn("teacher_judge_scores", rows[0])
        self.assertGreater(float(rows[0]["sample_weight"]), 1.0)
        self.assertNotIn("teacher_judge_decision", rows[1])

    def test_main_passes_topk_frames_per_view_to_teacher_loader(self):
        self.assertIsNotNone(annotate_teacher_judge_sft, "annotate_teacher_judge_sft.py is missing")

        verify_example = {
            "video_id": "verify_case",
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
                updated["teacher_judge_scores"] = {"sufficiency": 0.87, "necessity": 0.49}
                updated["teacher_judge_decision"] = "sufficient"
                return updated

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "prepared.jsonl"
            output_path = root / "prepared.teacher.jsonl"
            _write_prepared_input(input_path, [verify_example])

            with patch.object(
                annotate_teacher_judge_sft.QwenTeacherJudge,
                "from_pretrained",
                return_value=DummyTeacherJudge(),
            ) as mocked_loader:
                annotate_teacher_judge_sft.main(
                    [
                        "--input",
                        str(input_path),
                        "--output",
                        str(output_path),
                        "--model-path",
                        "/models/Qwen3-VL-32B-Instruct",
                        "--topk-frames-per-view",
                        "3",
                    ]
                )

        self.assertEqual(mocked_loader.call_args.kwargs["topk_frames_per_view"], 3)

    def test_main_uses_batch_annotation_path_when_batch_size_gt_one(self):
        self.assertIsNotNone(annotate_teacher_judge_sft, "annotate_teacher_judge_sft.py is missing")

        verify_example_a = {
            "video_id": "verify_case_a",
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
        verify_example_b = {
            **verify_example_a,
            "video_id": "verify_case_b",
        }

        class BatchOnlyTeacherJudge:
            def annotate_example(self, example, *, input_mode=None):
                raise AssertionError("single-example path should not be used when batch_size > 1")

            def annotate_examples(self, examples, *, input_mode=None):
                updated_examples = []
                for index, example in enumerate(examples):
                    updated = dict(example)
                    updated["teacher_judge_scores"] = {"sufficiency": 0.9 - 0.1 * index, "necessity": 0.4}
                    updated["teacher_judge_decision"] = "sufficient" if index == 0 else "insufficient"
                    updated["teacher_judge_rationale"] = f"batch-{index}"
                    updated_examples.append(updated)
                return updated_examples

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "prepared.jsonl"
            output_path = root / "prepared.teacher.jsonl"
            _write_prepared_input(input_path, [verify_example_a, verify_example_b])

            with patch.object(
                annotate_teacher_judge_sft.QwenTeacherJudge,
                "from_pretrained",
                return_value=BatchOnlyTeacherJudge(),
            ):
                annotate_teacher_judge_sft.main(
                    [
                        "--input",
                        str(input_path),
                        "--output",
                        str(output_path),
                        "--model-path",
                        "/models/Qwen3-VL-32B-Instruct",
                        "--batch-size",
                        "2",
                    ]
                )

            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]

        self.assertEqual(rows[0]["teacher_judge_decision"], "sufficient")
        self.assertEqual(rows[0]["teacher_judge_rationale"], "batch-0")
        self.assertEqual(rows[1]["teacher_judge_decision"], "insufficient")
        self.assertEqual(rows[1]["teacher_judge_rationale"], "batch-1")

    def test_merge_sharded_outputs_restores_original_order(self):
        self.assertIsNotNone(annotate_teacher_judge_sft, "annotate_teacher_judge_sft.py is missing")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_path = root / "prepared.teacher.jsonl"
            shard0_path = root / "prepared.teacher.shard00-of-02.jsonl"
            shard1_path = root / "prepared.teacher.shard01-of-02.jsonl"
            shard0_path.write_text(
                json.dumps({"video_id": "v0"}) + "\n" + json.dumps({"video_id": "v2"}) + "\n",
                encoding="utf-8",
            )
            shard1_path.write_text(
                json.dumps({"video_id": "v1"}) + "\n" + json.dumps({"video_id": "v3"}) + "\n",
                encoding="utf-8",
            )

            merged_rows = annotate_teacher_judge_sft._merge_sharded_outputs(
                output_path,
                total_rows=4,
                num_shards=2,
            )

            persisted_rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual([row["video_id"] for row in merged_rows], ["v0", "v1", "v2", "v3"])
        self.assertEqual([row["video_id"] for row in persisted_rows], ["v0", "v1", "v2", "v3"])

    def test_merge_sharded_outputs_restores_original_order_with_custom_verify_mapping(self):
        self.assertIsNotNone(annotate_teacher_judge_sft, "annotate_teacher_judge_sft.py is missing")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_path = root / "prepared.teacher.jsonl"
            shard0_path = root / "prepared.teacher.shard00-of-02.jsonl"
            shard1_path = root / "prepared.teacher.shard01-of-02.jsonl"
            shard0_path.write_text(
                json.dumps({"video_id": "v0"}) + "\n"
                + json.dumps({"video_id": "v1"}) + "\n"
                + json.dumps({"video_id": "v4"}) + "\n",
                encoding="utf-8",
            )
            shard1_path.write_text(
                json.dumps({"video_id": "v2"}) + "\n" + json.dumps({"video_id": "v3"}) + "\n",
                encoding="utf-8",
            )

            merged_rows = annotate_teacher_judge_sft._merge_sharded_outputs(
                output_path,
                total_rows=5,
                num_shards=2,
                shard_indices_by_shard=[[0, 1, 4], [2, 3]],
            )

            persisted_rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual([row["video_id"] for row in merged_rows], ["v0", "v1", "v2", "v3", "v4"])
        self.assertEqual([row["video_id"] for row in persisted_rows], ["v0", "v1", "v2", "v3", "v4"])

    def test_main_merges_sharded_outputs_automatically_in_distributed_mode(self):
        self.assertIsNotNone(annotate_teacher_judge_sft, "annotate_teacher_judge_sft.py is missing")

        verify_example_a = {
            "video_id": "verify_case_a",
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
        verify_example_b = {
            **verify_example_a,
            "video_id": "verify_case_b",
        }

        class DummyTeacherJudge:
            def annotate_example(self, example, *, input_mode=None):
                updated = dict(example)
                updated["teacher_judge_scores"] = {"sufficiency": 0.87, "necessity": 0.49}
                updated["teacher_judge_decision"] = "sufficient"
                updated["teacher_judge_rationale"] = f"annotated-{example['video_id']}"
                return updated

            def annotate_examples(self, examples, *, input_mode=None):
                return [self.annotate_example(example, input_mode=input_mode) for example in examples]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "prepared.jsonl"
            output_path = root / "prepared.teacher.jsonl"
            _write_prepared_input(input_path, [verify_example_a, verify_example_b])
            shard1_path = root / "prepared.teacher.shard01-of-02.jsonl"
            shard1_path.write_text(
                json.dumps(
                    {
                        **verify_example_b,
                        "teacher_judge_scores": {"sufficiency": 0.92, "necessity": 0.55},
                        "teacher_judge_decision": "sufficient",
                        "teacher_judge_rationale": "precreated-rank1",
                        "sample_weight": 1.0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            runtime = annotate_teacher_judge_sft.distributed_runtime_from_env(
                {"RANK": "0", "WORLD_SIZE": "2", "LOCAL_RANK": "0"}
            )

            with patch.object(annotate_teacher_judge_sft, "distributed_runtime_from_env", return_value=runtime), \
                patch.object(annotate_teacher_judge_sft, "init_torch_distributed", return_value=True), \
                patch.object(annotate_teacher_judge_sft, "distributed_barrier", return_value=None), \
                patch.object(
                    annotate_teacher_judge_sft.QwenTeacherJudge,
                    "from_pretrained",
                    return_value=DummyTeacherJudge(),
                ):
                annotate_teacher_judge_sft.main(
                    [
                        "--input",
                        str(input_path),
                        "--output",
                        str(output_path),
                        "--model-path",
                        "/models/Qwen3-VL-32B-Instruct",
                        "--batch-size",
                        "2",
                    ]
                )

            merged_rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual([row["video_id"] for row in merged_rows], ["verify_case_a", "verify_case_b"])
        self.assertEqual(merged_rows[0]["teacher_judge_rationale"], "annotated-verify_case_a")
        self.assertEqual(merged_rows[1]["teacher_judge_rationale"], "precreated-rank1")

    def test_main_merges_sharded_outputs_without_collective_barrier(self):
        self.assertIsNotNone(annotate_teacher_judge_sft, "annotate_teacher_judge_sft.py is missing")

        verify_example_a = {
            "video_id": "verify_case_a",
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
        verify_example_b = {
            **verify_example_a,
            "video_id": "verify_case_b",
        }

        class DummyTeacherJudge:
            def annotate_example(self, example, *, input_mode=None):
                updated = dict(example)
                updated["teacher_judge_scores"] = {"sufficiency": 0.87, "necessity": 0.49}
                updated["teacher_judge_decision"] = "sufficient"
                updated["teacher_judge_rationale"] = f"annotated-{example['video_id']}"
                return updated

            def annotate_examples(self, examples, *, input_mode=None):
                return [self.annotate_example(example, input_mode=input_mode) for example in examples]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "prepared.jsonl"
            output_path = root / "prepared.teacher.jsonl"
            _write_prepared_input(input_path, [verify_example_a, verify_example_b])
            shard1_path = root / "prepared.teacher.shard01-of-02.jsonl"
            shard1_path.write_text(
                json.dumps(
                    {
                        **verify_example_b,
                        "teacher_judge_scores": {"sufficiency": 0.92, "necessity": 0.55},
                        "teacher_judge_decision": "sufficient",
                        "teacher_judge_rationale": "precreated-rank1",
                        "sample_weight": 1.0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            runtime = annotate_teacher_judge_sft.distributed_runtime_from_env(
                {"RANK": "0", "WORLD_SIZE": "2", "LOCAL_RANK": "0"}
            )

            with patch.object(annotate_teacher_judge_sft, "distributed_runtime_from_env", return_value=runtime), \
                patch.object(annotate_teacher_judge_sft, "init_torch_distributed", return_value=True), \
                patch.object(
                    annotate_teacher_judge_sft,
                    "distributed_barrier",
                    side_effect=AssertionError("collective barrier should not be used here"),
                ), \
                patch.object(
                    annotate_teacher_judge_sft.QwenTeacherJudge,
                    "from_pretrained",
                    return_value=DummyTeacherJudge(),
                ):
                annotate_teacher_judge_sft.main(
                    [
                        "--input",
                        str(input_path),
                        "--output",
                        str(output_path),
                        "--model-path",
                        "/models/Qwen3-VL-32B-Instruct",
                        "--batch-size",
                        "2",
                    ]
                )

            merged_rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual([row["video_id"] for row in merged_rows], ["verify_case_a", "verify_case_b"])


if __name__ == "__main__":
    unittest.main()
