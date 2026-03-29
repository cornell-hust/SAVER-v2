import copy
import json
import re
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path("/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saver_agent.config import SaverAgentConfig
from saver_agent.adapter import TimeSearchRolloutAdapter
from saver_agent.rollout import ReplayPolicy, SaverRolloutRunner
from saver_agent.environment import SaverEnvironmentState, SaverVideoInteraction


try:
    from saver_agent.training_data import (
        build_oracle_sft_examples,
        build_counterfactual_grpo_examples,
        build_reward_weighted_examples,
    )
except ModuleNotFoundError:
    build_oracle_sft_examples = None
    build_counterfactual_grpo_examples = None
    build_reward_weighted_examples = None


class SaverAgentTrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.item = {
            "video_id": "sample_training_data",
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "system prompt"}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "0.000s"},
                        {
                            "type": "image",
                            "image": torch.zeros(3, 2, 2),
                            "sampled_frame_index": 0,
                            "raw_frame_index": 0,
                            "timestamp_sec": 0.0,
                        },
                        {"type": "text", "text": "user prompt"},
                    ],
                },
            ],
            "multimodal_cache": {
                "video": torch.zeros(8, 3, 2, 2),
                "embedding": None,
                "fps": 1.0,
                "duration": 8.0,
                "video_path": "/tmp/sample_training_data.mp4",
                "frame_indices": list(range(8)),
                "question": "Determine whether an anomaly exists.",
                "structured_target": {
                    "existence": "anomaly",
                    "category": "assault",
                    "severity": 4,
                    "anomaly_interval_sec": [1.0, 4.0],
                    "precursor_interval_sec": [0.0, 1.0],
                    "earliest_alert_sec": 1.0,
                    "counterfactual_type": "remove_actor_interaction",
                    "summary": "An assault occurs.",
                    "rationale": "A person attacks another person.",
                },
                "tool_io": {
                    "oracle_windows_sec": [
                        {"moment_id": "ev1", "role": "trigger", "window": [1.0, 4.0], "description": "trigger"},
                    ],
                    "finalize_case_schema": {"type": "object", "required": ["existence"]},
                },
            },
        }
        self.record = {
            "video_id": "sample_training_data",
            "proposal_supervision": {
                "queries": [
                    {
                        "query_id": "pq1",
                        "raw_text": "person attacking another person",
                        "normalized_queries": [
                            {"text": "physical struggle", "kind": "event_relation", "weight": 0.9},
                            {"text": "person on ground", "kind": "event_relation", "weight": 0.8},
                        ],
                        "linked_moment_ids": ["ev1"],
                        "linked_roles": ["trigger"],
                        "linked_windows_sec": [[1.0, 4.0]],
                        "alignment_source": "weak_alignment",
                    }
                ],
                "has_oracle_supervision": True,
            },
            "label": {"is_anomaly": True, "category": "assault", "severity": 4, "hard_normal": False},
            "temporal": {
                "anomaly_interval_sec": [1.0, 4.0],
                "precursor_interval_sec": [0.0, 1.0],
                "earliest_alert_sec": 1.0,
            },
            "evidence": {
                "evidence_moments": [
                    {
                        "moment_id": "ev1",
                        "role": "trigger",
                        "start_sec": 1.0,
                        "end_sec": 4.0,
                        "description": "trigger",
                    }
                ]
            },
            "structured_target": self.item["multimodal_cache"]["structured_target"],
            "oracle_sft": {
                "trajectory": [
                    {
                        "tool": "scan_timeline",
                        "arguments": {"start_sec": 0.0, "end_sec": 8.0, "purpose": "global_overview"},
                    },
                    {
                        "tool": "finalize_case",
                        "arguments": self.item["multimodal_cache"]["structured_target"],
                    },
                ],
                "final_decision": self.item["multimodal_cache"]["structured_target"],
            },
        }

    def test_build_oracle_sft_examples_replays_tools_and_appends_final_answer(self):
        self.assertIsNotNone(build_oracle_sft_examples, "saver_agent.training_data module is missing")

        examples = build_oracle_sft_examples(self.item, self.record, config=SaverAgentConfig())

        self.assertEqual(len(examples), 3)
        self.assertIn('"name":"scan_timeline"', examples[0]["target_response"])
        self.assertIn('"name":"finalize_case"', examples[1]["target_response"])
        self.assertIn("<answer>", examples[2]["target_response"])
        self.assertEqual(examples[2]["target_action"], "answer")

    def test_build_oracle_sft_examples_uses_non_placeholder_reasoning_text(self):
        self.assertIsNotNone(build_oracle_sft_examples, "saver_agent.training_data module is missing")

        examples = build_oracle_sft_examples(self.item, self.record, config=SaverAgentConfig())

        tool_think_match = re.search(r"<think>(.*?)</think>", examples[0]["target_response"], re.DOTALL)
        answer_think_match = re.search(r"<think>(.*?)</think>", examples[-1]["target_response"], re.DOTALL)

        self.assertIsNotNone(tool_think_match)
        self.assertIsNotNone(answer_think_match)
        tool_think = tool_think_match.group(1).strip()
        answer_think = answer_think_match.group(1).strip()

        self.assertNotEqual(tool_think, "inspect_global_context")
        self.assertNotEqual(answer_think, "return_final_answer")
        self.assertIn("overview", tool_think.lower())
        self.assertTrue(
            any(keyword in answer_think.lower() for keyword in ["final", "evidence", "support"]),
            msg=answer_think,
        )

    def test_build_oracle_sft_examples_counts_selected_evidence_moment_ids_in_verify_reasoning(self):
        self.assertIsNotNone(build_oracle_sft_examples, "saver_agent.training_data module is missing")

        record = {
            **self.record,
            "oracle_sft": {
                "trajectory": [
                    {
                        "tool": "verify_hypothesis",
                        "arguments": {
                            "verification_mode": "full_keep_drop",
                            "evidence_moment_ids": ["ev1"],
                            "claim": {"existence": "anomaly", "category": "assault"},
                        },
                    }
                ],
                "final_decision": self.item["multimodal_cache"]["structured_target"],
            },
        }

        examples = build_oracle_sft_examples(self.item, record, config=SaverAgentConfig())

        think_match = re.search(r"<think>(.*?)</think>", examples[0]["target_response"], re.DOTALL)
        self.assertIsNotNone(think_match)
        self.assertIn("1 selected evidence item", think_match.group(1))

    def test_build_oracle_sft_examples_normalizes_total_sample_weight_per_record(self):
        self.assertIsNotNone(build_oracle_sft_examples, "saver_agent.training_data module is missing")

        examples = build_oracle_sft_examples(self.item, self.record, config=SaverAgentConfig())

        self.assertGreater(len(examples), 0)
        self.assertAlmostEqual(sum(float(example["sample_weight"]) for example in examples), 1.0, places=6)

    def test_build_oracle_sft_examples_attaches_matching_proposal_supervision(self):
        self.assertIsNotNone(build_oracle_sft_examples, "saver_agent.training_data module is missing")

        record = copy.deepcopy(self.record)
        record["oracle_sft"] = {
            "trajectory": [
                {
                    "tool": "seek_evidence",
                    "arguments": {
                        "query": "physical struggle",
                        "start_sec": 1.0,
                        "end_sec": 4.0,
                        "moment_id": "ev1",
                        "role": "trigger",
                    },
                }
            ],
            "final_decision": self.item["multimodal_cache"]["structured_target"],
        }

        examples = build_oracle_sft_examples(self.item, record, config=SaverAgentConfig())

        self.assertEqual(examples[0]["proposal_supervision"]["query_id"], "pq1")
        self.assertIn("ev1", examples[0]["proposal_supervision"]["linked_moment_ids"])

    def test_build_oracle_sft_examples_prefers_oracle_verifier_feedback_over_runtime_fallback(self):
        self.assertIsNotNone(build_oracle_sft_examples, "saver_agent.training_data module is missing")

        record = {
            **self.record,
            "oracle_sft": {
                "trajectory": [
                    {
                        "tool": "scan_timeline",
                        "arguments": {"start_sec": 0.0, "end_sec": 4.0, "purpose": "global_overview"},
                    },
                    {
                        "tool": "verify_hypothesis",
                        "arguments": {
                            "verification_mode": "full_keep_drop",
                            "claim": {"existence": "anomaly", "category": "assault"},
                        },
                        "oracle_verifier_feedback": {
                            "verification_mode": "full_keep_drop",
                            "primary_status": "misaligned",
                            "alert_status": "not_applicable",
                            "recommended_action": "revise_claim",
                            "failure_reasons": ["selected_evidence_not_aligned_with_claim"],
                            "explanation": "Oracle verifier says the current claim should be revised before continuing.",
                        },
                    },
                    {
                        "tool": "finalize_case",
                        "arguments": self.item["multimodal_cache"]["structured_target"],
                    },
                ],
                "final_decision": self.item["multimodal_cache"]["structured_target"],
            },
        }

        examples = build_oracle_sft_examples(self.item, record, config=SaverAgentConfig())

        self.assertGreaterEqual(len(examples), 3)
        last_message = examples[2]["messages"][-1]
        self.assertEqual(last_message["role"], "tool")
        self.assertEqual(last_message.get("name"), "verify_hypothesis")
        message_texts = [str(item.get("text") or "") for item in last_message.get("content") or [] if item.get("type") == "text"]
        self.assertTrue(any("Revise the claim" in text for text in message_texts), msg=message_texts)

    def test_build_reward_weighted_examples_assigns_turn_level_advantages(self):
        self.assertIsNotNone(build_reward_weighted_examples, "saver_agent.training_data module is missing")

        runner = SaverRolloutRunner(
            environment=SaverVideoInteraction(),
            adapter=TimeSearchRolloutAdapter(),
            max_turns=3,
        )
        policy = ReplayPolicy(
            [
                '<think>scan</think><tool_call>{"name":"scan_timeline","arguments":{"start_sec":0.0,"end_sec":3.0,"num_frames":2}}</tool_call>',
                '<think>done</think><answer>{"existence":"anomaly","category":"assault"}</answer>',
            ]
        )
        rollout = runner.run_episode(self.item, policy, initial_state=SaverEnvironmentState())
        rollout["reward_summary"] = {"total_reward": 1.75}
        rollout["group_advantage"] = 1.2

        examples = build_reward_weighted_examples(self.item, rollout)

        self.assertEqual(len(examples), 2)
        self.assertLess(examples[0]["advantage"], examples[1]["advantage"])
        self.assertAlmostEqual(examples[0]["rollout_advantage"], 1.2, places=6)
        self.assertAlmostEqual(examples[1]["rollout_advantage"], 1.2, places=6)
        self.assertIn("turn_credit", examples[0]["advantage_metadata"])
        self.assertIn("<answer>", examples[-1]["target_response"])

    def test_build_reward_weighted_examples_preserve_seek_evidence_proposal_metadata(self):
        self.assertIsNotNone(build_reward_weighted_examples, "saver_agent.training_data module is missing")

        rollout = {
            "video_id": "sample_training_data",
            "reward_summary": {"total_reward": 0.8},
            "group_advantage": 0.6,
            "turns": [
                {
                    "step_index": 1,
                    "action": "tool_call",
                    "assistant_response_raw": '<think>seek</think><tool_call>{"name":"seek_evidence","arguments":{"query":"physical struggle","start_sec":1.0,"end_sec":4.0}}</tool_call>',
                    "tool_name": "seek_evidence",
                    "valid_action": True,
                    "new_evidence_ids": ["e0001"],
                    "proposal_backend": "feature_topk",
                    "feature_cache_used": True,
                    "proposal_query_raw": "physical struggle",
                    "proposal_query_normalized": "physical struggle",
                    "proposal_query_source": "model",
                    "proposal_candidate_count": 1,
                    "proposal_candidate_frame_indices": [1, 2],
                    "proposal_candidate_frame_scores": [0.9, 0.8],
                    "proposal_candidate_windows": [{"frame_indices": [1, 2], "score_max": 0.9}],
                    "proposal_selected_frame_indices": [1, 2],
                    "proposal_selected_frame_scores": [0.9, 0.8],
                    "proposal_fallback_reason": None,
                }
            ],
        }

        examples = build_reward_weighted_examples(self.item, rollout)

        self.assertEqual(examples[0]["proposal_metadata"]["backend"], "feature_topk")
        self.assertEqual(examples[0]["proposal_metadata"]["selected_frame_indices"], [1, 2])

    def test_build_reward_weighted_examples_penalizes_invalid_turns_when_requested(self):
        self.assertIsNotNone(build_reward_weighted_examples, "saver_agent.training_data module is missing")

        rollout = {
            "video_id": "sample_training_data",
            "reward_summary": {"total_reward": -0.5},
            "group_advantage": -0.8,
            "turns": [
                {
                    "step_index": 1,
                    "assistant_response_raw": "not a valid action",
                    "action": "invalid",
                    "valid_action": False,
                    "tool_name": None,
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

        examples = build_reward_weighted_examples(self.item, rollout, include_invalid=True)

        self.assertEqual(len(examples), 2)
        self.assertLess(examples[0]["advantage"], examples[1]["advantage"])
        self.assertLess(examples[0]["advantage_metadata"]["turn_credit"], 0.0)

    def test_build_oracle_sft_examples_can_serialize_messages_without_inline_images(self):
        self.assertIsNotNone(build_oracle_sft_examples, "saver_agent.training_data module is missing")

        examples = build_oracle_sft_examples(
            self.item,
            self.record,
            config=SaverAgentConfig(),
            serialize_messages=True,
        )

        image_items = [
            content
            for example in examples
            for message in example["messages"]
            for content in message.get("content", [])
            if content.get("type") == "image"
        ]
        self.assertGreater(len(image_items), 0)
        self.assertTrue(all("image" not in item for item in image_items))
        self.assertTrue(all("image_ref" in item for item in image_items))
        json.dumps(examples)

    def test_build_counterfactual_grpo_examples_attaches_local_advantages_and_component_weights(self):
        self.assertIsNotNone(build_counterfactual_grpo_examples, "saver_agent.training_data module is missing")

        runner = SaverRolloutRunner(
            environment=SaverVideoInteraction(),
            adapter=TimeSearchRolloutAdapter(),
            max_turns=5,
        )
        policy = ReplayPolicy(
            [
                '<think>scan</think><tool_call>{"name":"scan_timeline","arguments":{"start_sec":0.0,"end_sec":4.0,"num_frames":2}}</tool_call>',
                '<think>alert</think><tool_call>{"name":"emit_alert","arguments":{"decision":"soft_alert","existence":"anomaly","category":"assault","alert_sec":0.5,"claim":{"existence":"anomaly","category":"assault","earliest_alert_sec":1.0}}}</tool_call>',
                '<think>verify</think><tool_call>{"name":"verify_hypothesis","arguments":{"verification_mode":"soft_alert_check","claim":{"existence":"anomaly","category":"assault","earliest_alert_sec":1.0},"candidate_window_ids":["w0001"],"alert":{"decision":"soft_alert","alert_sec":0.5}}}</tool_call>',
                '<think>finalize</think><tool_call>{"name":"finalize_case","arguments":{"existence":"anomaly","category":"assault"}}</tool_call>',
                '<think>done</think><answer>{"existence":"anomaly","category":"assault"}</answer>',
            ]
        )
        rollout = runner.run_episode(self.item, policy, initial_state=SaverEnvironmentState())
        rollout["reward_summary"] = {"total_reward": 1.2}
        rollout["group_advantage"] = 0.8

        examples = build_counterfactual_grpo_examples(
            self.item,
            rollout,
            config=SaverAgentConfig(),
            local_verifier_backend="heuristic",
            local_use_reference_supervision=True,
        )

        self.assertGreaterEqual(len(examples), 4)
        emit_alert_example = next(example for example in examples if example.get("tool_name") == "emit_alert")
        verify_example = next(example for example in examples if example.get("tool_name") == "verify_hypothesis")
        finalize_example = next(example for example in examples if example.get("tool_name") == "finalize_case")
        self.assertIn("advantage_components", emit_alert_example)
        self.assertIn("global", emit_alert_example["advantage_components"])
        self.assertIn("alert_local", emit_alert_example["advantage_components"])
        self.assertIn("evidence_local", emit_alert_example["advantage_components"])
        self.assertIn("turn_component_weights", verify_example)
        self.assertGreaterEqual(verify_example["turn_component_weights"]["alert_local"], 0.5)
        self.assertGreaterEqual(verify_example["turn_component_weights"]["evidence_local"], 1.0)
        self.assertGreaterEqual(finalize_example["turn_component_weights"]["evidence_local"], 1.0)
        self.assertIn("counterfactual_group_ids", verify_example)

    def test_build_counterfactual_grpo_examples_attaches_search_local_advantage_to_search_turns(self):
        self.assertIsNotNone(build_counterfactual_grpo_examples, "saver_agent.training_data module is missing")

        runner = SaverRolloutRunner(
            environment=SaverVideoInteraction(),
            adapter=TimeSearchRolloutAdapter(),
            max_turns=4,
        )
        policy = ReplayPolicy(
            [
                '<think>scan</think><tool_call>{"name":"scan_timeline","arguments":{"start_sec":0.0,"end_sec":4.0,"num_frames":2}}</tool_call>',
                '<think>verify</think><tool_call>{"name":"verify_hypothesis","arguments":{"verification_mode":"soft_alert_check","claim":{"existence":"anomaly","category":"assault","earliest_alert_sec":1.0},"candidate_window_ids":["w0001"],"alert":{"decision":"soft_alert","alert_sec":0.5}}}</tool_call>',
                '<think>finalize</think><tool_call>{"name":"finalize_case","arguments":{"existence":"anomaly","category":"assault"}}</tool_call>',
                '<think>done</think><answer>{"existence":"anomaly","category":"assault"}</answer>',
            ]
        )
        rollout = runner.run_episode(self.item, policy, initial_state=SaverEnvironmentState())
        rollout["reward_summary"] = {"total_reward": 1.0}
        rollout["group_advantage"] = 0.7

        examples = build_counterfactual_grpo_examples(
            self.item,
            rollout,
            config=SaverAgentConfig(),
            local_verifier_backend="heuristic",
            search_local_alpha=0.6,
        )

        scan_example = next(example for example in examples if example.get("tool_name") == "scan_timeline")
        self.assertIn("search_local", scan_example["advantage_components"])
        self.assertIn("search_local", scan_example["turn_component_weights"])
        self.assertNotEqual(scan_example["advantage_components"]["search_local"], 0.0)
        self.assertGreaterEqual(scan_example["turn_component_weights"]["search_local"], 1.0)
        self.assertIn("actual_search_branch", scan_example["counterfactual_metadata"])


if __name__ == "__main__":
    unittest.main()
