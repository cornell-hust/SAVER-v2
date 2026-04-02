import json
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path("/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saver_agent.teacher_judge import (
    QwenTeacherJudge,
    annotate_teacher_judge_examples,
    apply_teacher_judge_reweighting,
    attach_teacher_judge_labels,
    build_teacher_judge_package,
    build_teacher_judge_messages,
    compute_teacher_judge_signal,
    compute_teacher_judge_weight_multiplier,
    normalize_teacher_judge_result,
    parse_teacher_judge_response,
    select_teacher_judge_input_mode,
)


class SaverAgentTeacherJudgeTests(unittest.TestCase):
    def setUp(self):
        self.example = {
            "video_id": "teacher_case",
            "tool_name": "verify_hypothesis",
            "target_response": (
                '<think>verify</think>'
                '<tool_call>{"name":"verify_hypothesis","arguments":'
                '{"verification_mode":"full_keep_drop","claim":{"existence":"anomaly","category":"assault"},'
                '"selected_window_ids":["w0001"],"selected_evidence_ids":["e0001"],'
                '"selected_evidence_moment_ids":["ev1"],"sufficiency_score":0.82,"necessity_score":0.44,'
                '"alertability_score":0.66,"counterfactual_faithfulness":0.63,'
                '"verification_decision":"sufficient","recommended_action":"finalize",'
                '"rationale":"The selected trigger evidence is sufficient."}}</tool_call>'
            ),
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "system"}]},
                {
                    "role": "tool",
                    "name": "seek_evidence",
                    "arguments": {"moment_id": "ev1", "role": "trigger", "start_sec": 1.0, "end_sec": 2.0},
                    "content": [
                        {"type": "text", "text": "1.000s"},
                        {
                            "type": "image",
                            "image": torch.zeros(3, 2, 2),
                            "sampled_frame_index": 1,
                            "timestamp_sec": 1.0,
                        },
                        {"type": "text", "text": "selected 1 frame"},
                    ],
                },
            ],
        }

    def test_build_teacher_judge_messages_text_only_omits_images(self):
        messages = build_teacher_judge_messages(self.example, input_mode="text_only")

        self.assertEqual(messages[0]["role"], "system")
        user_content = messages[1]["content"]
        self.assertTrue(all(item.get("type") != "image" for item in user_content))
        serialized_text = "\n".join(str(item.get("text") or "") for item in user_content if item.get("type") == "text")
        self.assertIn('"selected_window_ids": [', serialized_text)
        self.assertIn('"w0001"', serialized_text)
        self.assertIn("teacher judge", serialized_text.lower())

    def test_build_teacher_judge_messages_multimodal_visual_includes_images(self):
        messages = build_teacher_judge_messages(self.example, input_mode="multimodal_visual")

        user_content = messages[1]["content"]
        self.assertTrue(any(item.get("type") == "image" for item in user_content))

    def test_build_teacher_judge_package_exposes_counterfactual_views(self):
        package = build_teacher_judge_package(self.example, topk_frames_per_view=1)

        self.assertEqual(package["claim"]["category"], "assault")
        self.assertIn("full", package["views"])
        self.assertIn("keep", package["views"])
        self.assertIn("drop", package["views"])
        self.assertIn("alert_prefix", package["views"])
        self.assertEqual(len(package["views"]["full"]["images"]), 1)

    def test_build_teacher_judge_package_filters_bogus_selected_window_ids_by_known_records(self):
        example = {
            **self.example,
            "target_response": (
                '<think>verify</think>'
                '<tool_call>{"name":"verify_hypothesis","arguments":'
                '{"verification_mode":"full_keep_drop","claim":{"existence":"anomaly","category":"assault"},'
                '"selected_window_ids":["window_0"],"selected_evidence_ids":["e0001"],'
                '"selected_evidence_moment_ids":["ev1"],"sufficiency_score":0.82,"necessity_score":0.44,'
                '"alertability_score":0.66,"counterfactual_faithfulness":0.63,'
                '"verification_decision":"sufficient","recommended_action":"finalize",'
                '"rationale":"The selected trigger evidence is sufficient."}}</tool_call>'
            ),
        }

        package = build_teacher_judge_package(example, topk_frames_per_view=1)

        self.assertEqual(package["selected_window_ids"], ["w0001"])

    def test_select_teacher_judge_input_mode_auto_upgrades_hard_cases(self):
        easy_package = build_teacher_judge_package(self.example, topk_frames_per_view=1)
        self.assertEqual(select_teacher_judge_input_mode(easy_package, requested_mode="auto"), "text_only")

        hard_example = {
            **self.example,
            "target_response": self.example["target_response"].replace('"verification_decision":"sufficient"', '"verification_decision":"misaligned"'),
        }
        hard_package = build_teacher_judge_package(hard_example, topk_frames_per_view=1)
        self.assertEqual(select_teacher_judge_input_mode(hard_package, requested_mode="auto"), "multimodal_visual")

    def test_parse_teacher_judge_response_normalizes_scores_and_decision(self):
        response = json.dumps(
            {
                "teacher_judge_scores": {
                    "sufficiency": 0.91,
                    "necessity": 0.58,
                    "alertability": 0.7,
                    "counterfactual_faithfulness": 0.74,
                },
                "teacher_judge_decision": "sufficient",
                "teacher_judge_rationale": "The evidence is enough.",
            },
            ensure_ascii=False,
        )

        parsed = parse_teacher_judge_response(response)

        self.assertEqual(parsed["teacher_judge_decision"], "sufficient")
        self.assertAlmostEqual(parsed["teacher_judge_scores"]["sufficiency"], 0.91, places=6)
        self.assertEqual(parsed["teacher_judge_rationale"], "The evidence is enough.")

    def test_attach_teacher_judge_labels_updates_example_metadata(self):
        result = normalize_teacher_judge_result(
            {
                "teacher_judge_scores": {"sufficiency": 0.88, "necessity": 0.51},
                "teacher_judge_decision": "sufficient",
                "teacher_judge_rationale": "Enough support.",
            }
        )

        updated = attach_teacher_judge_labels(dict(self.example), result)

        self.assertEqual(updated["teacher_judge_decision"], "sufficient")
        self.assertIn("teacher_judge_scores", updated)
        self.assertEqual(updated["teacher_judge_rationale"], "Enough support.")

    def test_compute_teacher_judge_weight_multiplier_upweights_agreeing_verify_example(self):
        example = {
            **self.example,
            "sample_weight": 1.0,
            "teacher_judge_scores": {
                "sufficiency": 0.91,
                "necessity": 0.58,
                "alertability": 0.7,
                "counterfactual_faithfulness": 0.74,
            },
            "teacher_judge_decision": "sufficient",
        }

        multiplier = compute_teacher_judge_weight_multiplier(example)

        self.assertGreater(multiplier, 1.0)

    def test_apply_teacher_judge_reweighting_updates_sample_weight(self):
        example = {
            **self.example,
            "sample_weight": 1.0,
            "teacher_judge_scores": {
                "sufficiency": 0.91,
                "necessity": 0.58,
                "alertability": 0.7,
                "counterfactual_faithfulness": 0.74,
            },
            "teacher_judge_decision": "sufficient",
        }

        updated = apply_teacher_judge_reweighting(example)

        self.assertIn("teacher_judge_weight_multiplier", updated)
        self.assertIn("teacher_judge_alignment", updated)
        self.assertGreater(updated["sample_weight"], 1.0)

    def test_compute_teacher_judge_signal_scores_policy_teacher_agreement(self):
        source = {
            **self.example,
            "teacher_judge_scores": {
                "sufficiency": 0.8,
                "necessity": 0.4,
                "alertability": 0.7,
                "counterfactual_faithfulness": 0.6,
            },
            "teacher_judge_decision": "sufficient",
        }

        signal = compute_teacher_judge_signal(source)

        self.assertTrue(signal["teacher_judge_present"])
        self.assertEqual(signal["teacher_judge_policy_decision"], "sufficient")
        self.assertEqual(signal["teacher_judge_decision"], "sufficient")
        self.assertAlmostEqual(signal["teacher_judge_alignment"], 1.0, places=6)
        self.assertGreater(signal["teacher_judge_score_agreement"], 0.8)
        self.assertGreater(signal["teacher_judge_reward"], 0.5)

    def test_compute_teacher_judge_signal_penalizes_misaligned_policy_decision(self):
        turn = {
            "tool_name": "verify_hypothesis",
            "self_verification_decision": "sufficient",
            "self_verification_scores": {
                "sufficiency": 0.8,
                "necessity": 0.4,
                "alertability": 0.7,
                "counterfactual_faithfulness": 0.6,
            },
            "teacher_judge_scores": {
                "sufficiency": 0.2,
                "necessity": 0.1,
                "alertability": 0.2,
                "counterfactual_faithfulness": 0.2,
            },
            "teacher_judge_decision": "misaligned",
        }

        signal = compute_teacher_judge_signal(turn)

        self.assertTrue(signal["teacher_judge_present"])
        self.assertEqual(signal["teacher_judge_policy_decision"], "sufficient")
        self.assertEqual(signal["teacher_judge_decision"], "misaligned")
        self.assertAlmostEqual(signal["teacher_judge_alignment"], 0.0, places=6)
        self.assertLess(signal["teacher_judge_score_agreement"], 0.6)
        self.assertLess(signal["teacher_judge_reward"], 0.0)

    def test_annotate_teacher_judge_examples_emits_progress_callback_for_scan_and_annotate(self):
        events = []

        class DummyJudge:
            def annotate_examples(self, examples, *, input_mode=None):
                updated_examples = []
                for example in examples:
                    updated = dict(example)
                    updated["teacher_judge_scores"] = {"sufficiency": 0.9, "necessity": 0.4}
                    updated["teacher_judge_decision"] = "sufficient"
                    updated_examples.append(updated)
                return updated_examples

        non_candidate = {
            "video_id": "answer_case",
            "tool_name": "finalize_case",
            "target_response": '<answer>{"existence":"normal"}</answer>',
            "messages": [],
        }

        annotated_rows, summary = annotate_teacher_judge_examples(
            [self.example, non_candidate],
            judge=DummyJudge(),
            input_mode="text_only",
            batch_size=2,
            progress_callback=lambda payload: events.append(dict(payload)),
        )

        self.assertEqual(len(annotated_rows), 2)
        self.assertEqual(summary["num_teacher_judge_annotated"], 1)
        self.assertTrue(any(event.get("phase") == "scan" for event in events))
        annotate_events = [event for event in events if event.get("phase") == "annotate"]
        self.assertTrue(annotate_events)
        self.assertEqual(annotate_events[-1]["completed"], 1)
        self.assertEqual(annotate_events[-1]["total"], 1)

    def test_qwen_teacher_judge_generation_kwargs_omit_sampling_parameters_when_sampling_disabled(self):
        class FakeProcessor:
            def batch_decode(self, generated_ids_trimmed, **kwargs):
                return ["{}"]

        class FakeModel:
            device = None

            def generate(self, **kwargs):
                return torch.tensor([[1, 2]])

        judge = QwenTeacherJudge(
            model=FakeModel(),
            processor=FakeProcessor(),
            do_sample=False,
            temperature=0.7,
            top_p=0.85,
            top_k=16,
            repetition_penalty=1.1,
        )

        kwargs = judge._generation_kwargs()

        self.assertFalse(kwargs["do_sample"])
        self.assertNotIn("temperature", kwargs)
        self.assertNotIn("top_p", kwargs)
        self.assertNotIn("top_k", kwargs)
        self.assertEqual(kwargs["repetition_penalty"], 1.1)

    def test_qwen_teacher_judge_judge_examples_batches_generate(self):
        class FakeProcessor:
            def __init__(self):
                self.chat_template_calls = []

            def apply_chat_template(self, conversation, **kwargs):
                self.chat_template_calls.append({"conversation": conversation, **kwargs})
                if not kwargs.get("tokenize", False):
                    if isinstance(conversation, list) and conversation and isinstance(conversation[0], list):
                        return [f"prompt-{idx}" for idx in range(len(conversation))]
                    return "prompt-0"
                batch_size = len(conversation) if isinstance(conversation, list) and conversation and isinstance(conversation[0], list) else 1
                input_ids = torch.tensor([[100 + idx, 200 + idx] for idx in range(batch_size)], dtype=torch.long)
                attention_mask = torch.ones_like(input_ids)
                return {"input_ids": input_ids, "attention_mask": attention_mask}

            def batch_decode(self, generated_ids_trimmed, **kwargs):
                outputs = []
                for token_ids in generated_ids_trimmed:
                    values = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
                    tail = int(values[0]) if values else -1
                    if tail == 10:
                        outputs.append(
                            json.dumps(
                                {
                                    "teacher_judge_scores": {"sufficiency": 0.9, "necessity": 0.5},
                                    "teacher_judge_decision": "sufficient",
                                    "teacher_judge_rationale": "batch-0",
                                }
                            )
                        )
                    elif tail == 11:
                        outputs.append(
                            json.dumps(
                                {
                                    "teacher_judge_scores": {"sufficiency": 0.2, "necessity": 0.1},
                                    "teacher_judge_decision": "misaligned",
                                    "teacher_judge_rationale": "batch-1",
                                }
                            )
                        )
                    else:
                        outputs.append("{}")
                return outputs

        class FakeModel:
            def __init__(self):
                self.generate_calls = 0
                self.device = None

            def generate(self, **kwargs):
                self.generate_calls += 1
                input_ids = kwargs["input_ids"]
                appended = torch.tensor([[10], [11]], dtype=input_ids.dtype)
                return torch.cat([input_ids, appended], dim=1)

        judge = QwenTeacherJudge(
            model=FakeModel(),
            processor=FakeProcessor(),
            input_mode="text_only",
            max_new_tokens=32,
        )

        results = judge.judge_examples([self.example, {**self.example, "video_id": "teacher_case_2"}], input_mode="text_only")

        self.assertEqual(judge.model.generate_calls, 1)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["teacher_judge_decision"], "sufficient")
        self.assertEqual(results[0]["teacher_judge_rationale"], "batch-0")
        self.assertEqual(results[1]["teacher_judge_decision"], "misaligned")
        self.assertEqual(results[1]["teacher_judge_rationale"], "batch-1")

    def test_qwen_teacher_judge_judge_examples_falls_back_when_apply_chat_template_batch_tensorization_fails(self):
        class FakeProcessor:
            def __init__(self):
                self.manual_processor_calls = 0

            def apply_chat_template(self, conversation, **kwargs):
                if kwargs.get("tokenize", False):
                    raise ValueError("simulated ragged input_ids error")
                if isinstance(conversation, list) and conversation and isinstance(conversation[0], list):
                    return [f"prompt-{idx}" for idx in range(len(conversation))]
                return "prompt-0"

            def __call__(self, **kwargs):
                self.manual_processor_calls += 1
                input_ids = torch.tensor([[100, 200], [101, 201]], dtype=torch.long)
                attention_mask = torch.ones_like(input_ids)
                return {"input_ids": input_ids, "attention_mask": attention_mask}

            def batch_decode(self, generated_ids_trimmed, **kwargs):
                outputs = []
                for token_ids in generated_ids_trimmed:
                    values = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
                    tail = int(values[0]) if values else -1
                    if tail == 10:
                        outputs.append(
                            json.dumps(
                                {
                                    "teacher_judge_scores": {"sufficiency": 0.9, "necessity": 0.5},
                                    "teacher_judge_decision": "sufficient",
                                    "teacher_judge_rationale": "fallback-0",
                                }
                            )
                        )
                    elif tail == 11:
                        outputs.append(
                            json.dumps(
                                {
                                    "teacher_judge_scores": {"sufficiency": 0.3, "necessity": 0.2},
                                    "teacher_judge_decision": "insufficient",
                                    "teacher_judge_rationale": "fallback-1",
                                }
                            )
                        )
                    else:
                        outputs.append("{}")
                return outputs

        class FakeModel:
            def __init__(self):
                self.device = None

            def generate(self, **kwargs):
                input_ids = kwargs["input_ids"]
                appended = torch.tensor([[10], [11]], dtype=input_ids.dtype)
                return torch.cat([input_ids, appended], dim=1)

        judge = QwenTeacherJudge(
            model=FakeModel(),
            processor=FakeProcessor(),
            input_mode="text_only",
            max_new_tokens=32,
        )

        results = judge.judge_examples([self.example, {**self.example, "video_id": "teacher_case_2"}], input_mode="text_only")

        self.assertEqual(judge.processor.manual_processor_calls, 1)
        self.assertEqual(results[0]["teacher_judge_rationale"], "fallback-0")
        self.assertEqual(results[1]["teacher_judge_rationale"], "fallback-1")


if __name__ == "__main__":
    unittest.main()
