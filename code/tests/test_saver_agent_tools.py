import sys
import unittest
from pathlib import Path

import torch


ROOT = Path("/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saver_agent.environment import SaverEnvironmentState
from saver_agent.tool_registry import execute_tool_call, get_tool_schemas


class _FakeQwenVerifierRuntime:
    def __init__(self, view_scores):
        self.view_scores = view_scores
        self.calls = 0

    def score_views(self, *, views, claim, verification_mode, question):
        self.calls += 1
        return self.view_scores


class SaverAgentToolsTests(unittest.TestCase):
    def setUp(self):
        self.multimodal_cache = {
            "video": torch.arange(10 * 3 * 2 * 2, dtype=torch.float32).reshape(10, 3, 2, 2),
            "embedding": None,
            "fps": 1.0,
            "duration": 10.0,
            "question": "Determine whether an anomaly exists.",
            "structured_target": {
                "existence": "anomaly",
                "category": "robbery",
                "severity": 4,
                "anomaly_interval_sec": [1.0, 8.0],
                "precursor_interval_sec": [0.0, 1.0],
                "earliest_alert_sec": 1.0,
                "counterfactual_type": "remove_actor_interaction",
            },
            "tool_io": {
                "oracle_windows_sec": [
                    {"moment_id": "ev1", "role": "trigger", "window": [1.0, 4.0], "description": "trigger"},
                    {"moment_id": "ev2", "role": "peak_action", "window": [4.0, 8.0], "description": "peak"},
                ],
                "finalize_case_schema": {"type": "object", "required": ["existence"]},
            },
        }
        self.state = SaverEnvironmentState()

    def test_tool_schemas_include_expected_names(self):
        names = [tool["function"]["name"] for tool in get_tool_schemas()]
        self.assertEqual(
            names,
            ["scan_timeline", "seek_evidence", "emit_alert", "verify_hypothesis", "finalize_case"],
        )

    def test_scan_timeline_updates_visited_windows(self):
        message, state = execute_tool_call(
            {
                "function": {
                    "name": "scan_timeline",
                    "arguments": {"start_sec": 0.0, "end_sec": 9.0, "num_frames": 4},
                }
            },
            self.multimodal_cache,
            self.state,
        )

        self.assertEqual(message["role"], "tool")
        self.assertEqual(message["name"], "scan_timeline")
        self.assertEqual(len(state.visited_windows), 1)
        self.assertEqual(state.visited_windows[0]["window_id"], "w0001")
        self.assertEqual(state.visited_windows[0]["kind"], "scan")
        self.assertEqual(state.evidence_ledger[0]["selected_frame_count"], 4)

    def test_emit_alert_and_finalize_case_update_environment_state(self):
        _, state = execute_tool_call(
            {
                "function": {
                    "name": "emit_alert",
                    "arguments": {"decision": "soft_alert", "existence": "anomaly", "category": "robbery"},
                }
            },
            self.multimodal_cache,
            self.state,
        )
        self.assertEqual(state.alerts[-1]["decision"], "soft_alert")
        self.assertEqual(state.alerts[-1]["alert_id"], "a0001")

        message, state = execute_tool_call(
            {
                "function": {
                    "name": "finalize_case",
                    "arguments": {"existence": "anomaly"},
                }
            },
            self.multimodal_cache,
            state,
        )
        self.assertEqual(message["name"], "finalize_case")
        self.assertEqual(state.finalized_case["existence"], "anomaly")

    def test_verify_hypothesis_returns_structured_counterfactual_verdict(self):
        _, state = execute_tool_call(
            {
                "function": {
                    "name": "seek_evidence",
                    "arguments": {"query": "robbery trigger", "start_sec": 1.0, "end_sec": 4.0, "num_frames": 2},
                }
            },
            self.multimodal_cache,
            self.state,
        )
        _, state = execute_tool_call(
            {
                "function": {
                    "name": "seek_evidence",
                    "arguments": {"query": "robbery peak", "start_sec": 4.0, "end_sec": 8.0, "num_frames": 2},
                }
            },
            self.multimodal_cache,
            state,
        )

        message, state = execute_tool_call(
            {
                "function": {
                    "name": "verify_hypothesis",
                    "arguments": {
                        "verification_mode": "final_check",
                        "claim": {"existence": "anomaly", "category": "robbery", "earliest_alert_sec": 1.0},
                        "candidate_window_ids": ["w0001", "w0002"],
                        "alert": {"decision": "hard_alert", "alert_sec": 1.2},
                    },
                }
            },
            self.multimodal_cache,
            state,
        )

        self.assertEqual(message["name"], "verify_hypothesis")
        self.assertIn(
            state.verification_records[-1]["primary_status"],
            {"complete", "incomplete", "redundant", "misaligned"},
        )
        self.assertIn(
            state.verification_records[-1]["alert_status"],
            {"justified", "premature", "late", "not_applicable"},
        )
        self.assertEqual(state.verification_records[-1]["verified_window_ids"], ["w0001", "w0002"])

    def test_verify_hypothesis_does_not_merge_reference_target_into_online_claim(self):
        _, state = execute_tool_call(
            {
                "function": {
                    "name": "seek_evidence",
                    "arguments": {
                        "query": "robbery trigger",
                        "start_sec": 1.0,
                        "end_sec": 4.0,
                        "num_frames": 2,
                        "moment_id": "ev1",
                    },
                }
            },
            self.multimodal_cache,
            self.state,
        )

        _, state = execute_tool_call(
            {
                "function": {
                    "name": "verify_hypothesis",
                    "arguments": {
                        "verification_mode": "final_check",
                        "claim": {"existence": "anomaly", "category": "robbery"},
                        "candidate_window_ids": ["w0001"],
                    },
                }
            },
            self.multimodal_cache,
            state,
        )

        returned_claim = state.verification_records[-1]["claim"]
        self.assertEqual(returned_claim, {"existence": "anomaly", "category": "robbery"})
        self.assertNotIn("severity", returned_claim)
        self.assertNotIn("anomaly_interval_sec", returned_claim)
        self.assertNotIn("counterfactual_type", returned_claim)

    def test_verify_hypothesis_resolves_evidence_moment_ids_to_matching_runtime_windows(self):
        _, state = execute_tool_call(
            {
                "function": {
                    "name": "seek_evidence",
                    "arguments": {
                        "query": "robbery trigger",
                        "start_sec": 1.0,
                        "end_sec": 4.0,
                        "num_frames": 2,
                        "moment_id": "ev1",
                    },
                }
            },
            self.multimodal_cache,
            self.state,
        )
        _, state = execute_tool_call(
            {
                "function": {
                    "name": "seek_evidence",
                    "arguments": {
                        "query": "robbery peak",
                        "start_sec": 4.0,
                        "end_sec": 8.0,
                        "num_frames": 2,
                        "moment_id": "ev2",
                    },
                }
            },
            self.multimodal_cache,
            state,
        )

        _, state = execute_tool_call(
            {
                "function": {
                    "name": "verify_hypothesis",
                    "arguments": {
                        "verification_mode": "final_check",
                        "claim": {"existence": "anomaly", "category": "robbery"},
                        "evidence_moment_ids": ["ev2"],
                    },
                }
            },
            self.multimodal_cache,
            state,
        )

        self.assertEqual(state.verification_records[-1]["verified_window_ids"], ["w0002"])
        self.assertEqual(state.active_evidence_window_ids, ["w0002"])

    def test_verify_hypothesis_uses_multimodal_cache_default_qwen_backend(self):
        runtime = _FakeQwenVerifierRuntime(
            {
                "full": {"exist_support": 0.9, "category_support": 0.9, "temporal_support": 0.9, "precursor_support": 0.7, "alert_support": 0.9, "counterfactual_support": 0.9, "overall_support": 0.89},
                "keep": {"exist_support": 0.9, "category_support": 0.9, "temporal_support": 0.9, "precursor_support": 0.7, "alert_support": 0.9, "counterfactual_support": 0.9, "overall_support": 0.89},
                "drop": {"exist_support": 0.1, "category_support": 0.1, "temporal_support": 0.1, "precursor_support": 0.1, "alert_support": 0.1, "counterfactual_support": 0.1, "overall_support": 0.1},
                "alert_prefix": {"exist_support": 0.9, "category_support": 0.9, "temporal_support": 0.9, "precursor_support": 0.7, "alert_support": 0.9, "counterfactual_support": 0.9, "overall_support": 0.89},
            }
        )
        self.multimodal_cache["verifier_backend"] = "qwen_self_verifier"
        self.multimodal_cache["verifier_runtime"] = runtime

        _, state = execute_tool_call(
            {
                "function": {
                    "name": "seek_evidence",
                    "arguments": {"query": "robbery trigger", "start_sec": 1.0, "end_sec": 8.0, "num_frames": 2},
                }
            },
            self.multimodal_cache,
            self.state,
        )
        message, state = execute_tool_call(
            {
                "function": {
                    "name": "verify_hypothesis",
                    "arguments": {
                        "verification_mode": "final_check",
                        "claim": {"existence": "anomaly", "category": "robbery", "earliest_alert_sec": 1.0},
                        "candidate_window_ids": ["w0001"],
                        "alert": {"decision": "hard_alert", "alert_sec": 1.2},
                    },
                }
            },
            self.multimodal_cache,
            state,
        )

        self.assertEqual(state.verification_records[-1]["verifier_backend"], "qwen_self_verifier")
        self.assertEqual(message["content"][0]["type"], "text")
        self.assertEqual(runtime.calls, 1)


if __name__ == "__main__":
    unittest.main()
