import sys
import unittest
from pathlib import Path
from unittest.mock import patch

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


class _FakeProposalRuntime:
    def __init__(self, mapping):
        self.mapping = {str(key): torch.tensor(value, dtype=torch.float32) for key, value in mapping.items()}
        self.calls = []

    def encode_texts(self, texts):
        self.calls.append(list(texts))
        return torch.stack([self.mapping[str(text)] for text in texts], dim=0)


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

        scan_schema = next(tool for tool in get_tool_schemas() if tool["function"]["name"] == "scan_timeline")
        self.assertIn("stride_sec", scan_schema["function"]["parameters"]["properties"])

    def test_emit_alert_schema_requires_canonical_category(self):
        schema = next(tool for tool in get_tool_schemas() if tool["function"]["name"] == "emit_alert")
        parameters = schema["function"]["parameters"]

        self.assertIn("category", parameters["required"])
        self.assertIn("normal", parameters["properties"]["category"]["enum"])
        self.assertIn("assault", parameters["properties"]["category"]["enum"])

    def test_scan_timeline_updates_visited_windows(self):
        message, state = execute_tool_call(
            {
                "function": {
                    "name": "scan_timeline",
                    "arguments": {"start_sec": 0.0, "end_sec": 9.0, "num_frames": 4, "stride_sec": 2.0},
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
        self.assertEqual(state.visited_windows[0]["selected_frame_count"], 6)
        self.assertEqual(state.visited_windows[0]["selected_timestamps"], [0.0, 2.0, 4.0, 6.0, 8.0, 9.0])
        self.assertEqual(state.evidence_ledger, [])

    def test_emit_alert_and_finalize_case_update_environment_state(self):
        _, state = execute_tool_call(
            {
                "function": {
                    "name": "emit_alert",
                    "arguments": {
                        "decision": "soft_alert",
                        "existence": "anomaly",
                        "category": "robbery",
                        "earliest_alert_sec": 1.5,
                    },
                }
            },
            self.multimodal_cache,
            self.state,
        )
        self.assertEqual(state.alerts[-1]["decision"], "soft_alert")
        self.assertEqual(state.alerts[-1]["alert_id"], "a0001")
        self.assertEqual(
            state.last_claim,
            {"existence": "anomaly", "category": "robbery", "earliest_alert_sec": 1.5},
        )

        message, state = execute_tool_call(
            {
                "function": {
                    "name": "finalize_case",
                    "arguments": {"existence": "anomaly", "category": "robbery"},
                }
            },
            self.multimodal_cache,
            state,
        )
        self.assertEqual(message["name"], "finalize_case")
        self.assertEqual(state.finalized_case["existence"], "anomaly")
        self.assertEqual(state.finalized_case["category"], "robbery")

    def test_emit_alert_rejects_unknown_anomaly_category_after_canonicalization(self):
        with self.assertRaises(ValueError):
            execute_tool_call(
                {
                    "function": {
                        "name": "emit_alert",
                        "arguments": {
                            "decision": "hard_alert",
                            "existence": "anomaly",
                            "category": "mysterious_event",
                        },
                    }
                },
                self.multimodal_cache,
                self.state,
            )

    def test_finalize_case_rejects_missing_required_anomaly_category(self):
        multimodal_cache = dict(self.multimodal_cache)
        multimodal_cache["tool_io"] = {
            **dict(self.multimodal_cache.get("tool_io") or {}),
            "finalize_case_schema": {
                "type": "object",
                "properties": {
                    "existence": {"type": "string", "enum": ["normal", "anomaly"]},
                    "category": {"type": "string", "enum": ["normal", "robbery", "assault"]},
                },
                "required": ["existence", "category"],
            },
        }
        with self.assertRaises(ValueError):
            execute_tool_call(
                {
                    "function": {
                        "name": "finalize_case",
                        "arguments": {
                            "existence": "anomaly",
                        },
                    }
                },
                multimodal_cache,
                self.state,
            )

    def test_verify_hypothesis_rejects_unknown_anomaly_category_in_claim(self):
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

        with self.assertRaises(ValueError):
            execute_tool_call(
                {
                    "function": {
                        "name": "verify_hypothesis",
                        "arguments": {
                            "verification_mode": "final_check",
                            "claim": {"existence": "anomaly", "category": "mysterious_event"},
                            "candidate_window_ids": ["w0001"],
                            "selected_window_ids": ["w0001"],
                            "selected_evidence_ids": ["e0001"],
                            "selected_evidence_moment_ids": [],
                            "sufficiency_score": 0.84,
                            "necessity_score": 0.61,
                            "alertability_score": 0.72,
                            "counterfactual_faithfulness": 0.73,
                            "verification_decision": "sufficient",
                            "recommended_action": "finalize",
                        },
                    }
                },
                self.multimodal_cache,
                state,
            )

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
                        "allow_external_verifier_fallback": True,
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

    def test_verify_hypothesis_rejects_payloads_without_self_report_or_legacy_request_fields(self):
        with self.assertRaises(ValueError):
            execute_tool_call(
                {
                    "function": {
                        "name": "verify_hypothesis",
                        "arguments": {
                            "verification_mode": "final_check",
                        },
                    }
                },
                self.multimodal_cache,
                self.state,
            )

    def test_verify_hypothesis_legacy_request_payload_requires_explicit_diagnostic_fallback(self):
        _, state = execute_tool_call(
            {
                "function": {
                    "name": "seek_evidence",
                    "arguments": {
                        "query": "robbery trigger",
                        "start_sec": 1.0,
                        "end_sec": 4.0,
                        "num_frames": 2,
                    },
                }
            },
            self.multimodal_cache,
            self.state,
        )

        with self.assertRaises(ValueError):
            execute_tool_call(
                {
                    "function": {
                        "name": "verify_hypothesis",
                        "arguments": {
                            "verification_mode": "normal_check",
                            "query": "Is there a vehicle collision?",
                            "claim": {"existence": "anomaly", "category": "vehicle_collision"},
                            "candidate_window_ids": ["w0001"],
                        },
                    }
                },
                self.multimodal_cache,
                state,
            )

    def test_verify_hypothesis_legacy_request_payload_uses_reference_free_compatibility_fallback_when_enabled(self):
        _, state = execute_tool_call(
            {
                "function": {
                    "name": "seek_evidence",
                    "arguments": {
                        "query": "robbery trigger",
                        "start_sec": 1.0,
                        "end_sec": 4.0,
                        "num_frames": 2,
                    },
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
                        "allow_external_verifier_fallback": True,
                        "verification_mode": "normal_check",
                        "query": "Is there a vehicle collision?",
                        "claim": {"existence": "anomaly", "category": "vehicle_collision"},
                        "candidate_window_ids": ["w0001"],
                    },
                }
            },
            self.multimodal_cache,
            state,
        )

        self.assertEqual(message["name"], "verify_hypothesis")
        verdict = state.verification_records[-1]
        self.assertEqual(verdict["verification_mode"], "final_check")
        self.assertEqual(verdict["claim"]["category"], "traffic_accident")
        self.assertEqual(verdict["verified_window_ids"], ["w0001"])
        self.assertEqual(verdict["verifier_backend"], "heuristic")
        self.assertFalse(verdict["use_reference_supervision"])

    def test_verify_hypothesis_legacy_request_with_selected_ids_uses_verifier_fallback_not_self_report(self):
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

        with patch(
            "saver_agent.tools.run_counterfactual_verifier",
            return_value={
                "verification_mode": "final_check",
                "primary_status": "complete",
                "alert_status": "justified",
                "recommended_action": "finalize",
                "derived_scores": {
                    "sufficiency": 0.9,
                    "necessity": 0.6,
                    "consistency": 0.9,
                    "alertability": 0.8,
                    "counterfactual_faithfulness": 0.7,
                },
                "verified_window_ids": ["w0001"],
                "best_effort_window_ids": ["w0001"],
                "failure_reasons": [],
                "verifier_backend": "heuristic",
                "claim": {"existence": "anomaly", "category": "robbery"},
                "alert": {"decision": "hard_alert", "alert_sec": 1.1},
                "use_reference_supervision": False,
            },
        ) as verifier_mock:
            _, state = execute_tool_call(
                {
                    "function": {
                        "name": "verify_hypothesis",
                        "arguments": {
                            "allow_external_verifier_fallback": True,
                            "verification_mode": "final_check",
                            "claim": {"existence": "anomaly", "category": "robbery"},
                            "query": "Is the robbery evidence sufficient?",
                            "candidate_window_ids": ["w0001"],
                            "selected_window_ids": ["window_0"],
                            "selected_evidence_ids": ["e0001"],
                            "selected_evidence_moment_ids": ["ev1"],
                            "alert": {"decision": "hard_alert", "alert_sec": 1.1},
                        },
                    }
                },
                self.multimodal_cache,
                state,
            )

        verifier_mock.assert_called_once()
        verdict = state.verification_records[-1]
        self.assertEqual(verdict["verifier_backend"], "heuristic")
        self.assertEqual(verdict["primary_status"], "complete")
        self.assertEqual(verdict["verification_parse_mode"], "external_verifier_fallback")
        self.assertFalse(verdict["legacy_compatibility_used"])
        self.assertEqual(state.active_evidence_window_ids, ["w0001"])
        self.assertEqual(verdict["requested_selected_window_ids"], ["window_0"])
        self.assertEqual(verdict["invalid_selected_window_ids"], ["window_0"])

    def test_verify_hypothesis_disallows_external_fallback_when_main_path_disables_it(self):
        multimodal_cache = dict(self.multimodal_cache)
        multimodal_cache["disable_external_verifier_fallback"] = True

        _, state = execute_tool_call(
            {
                "function": {
                    "name": "seek_evidence",
                    "arguments": {
                        "query": "robbery trigger",
                        "start_sec": 1.0,
                        "end_sec": 4.0,
                        "num_frames": 2,
                    },
                }
            },
            multimodal_cache,
            self.state,
        )

        with self.assertRaises(ValueError):
            execute_tool_call(
                {
                    "function": {
                        "name": "verify_hypothesis",
                        "arguments": {
                            "allow_external_verifier_fallback": True,
                            "verification_mode": "normal_check",
                            "query": "Is there a vehicle collision?",
                            "claim": {"existence": "anomaly", "category": "vehicle_collision"},
                            "candidate_window_ids": ["w0001"],
                        },
                    }
                },
                multimodal_cache,
                state,
            )

    def test_verify_hypothesis_records_policy_self_report_without_external_runtime(self):
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
                    "arguments": {"query": "robbery trigger", "start_sec": 1.0, "end_sec": 4.0, "num_frames": 2},
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
                        "selected_window_ids": ["w0001"],
                        "selected_evidence_ids": ["e0001"],
                        "selected_evidence_moment_ids": [],
                        "sufficiency_score": 0.84,
                        "necessity_score": 0.61,
                        "alertability_score": 0.72,
                        "counterfactual_faithfulness": 0.73,
                        "verification_decision": "sufficient",
                        "recommended_action": "finalize",
                        "rationale": "The chosen evidence is sufficient and necessary.",
                        "alert": {"decision": "hard_alert", "alert_sec": 1.2},
                    },
                }
            },
            self.multimodal_cache,
            state,
        )

        self.assertEqual(message["name"], "verify_hypothesis")
        self.assertEqual(state.verification_records[-1]["verifier_backend"], "self_report")
        self.assertEqual(state.verification_records[-1]["primary_status"], "complete")
        self.assertEqual(state.verification_records[-1]["recommended_action"], "finalize")
        self.assertEqual(state.verification_records[-1]["verified_window_ids"], ["w0001"])
        self.assertEqual(state.active_evidence_window_ids, ["w0001"])
        self.assertEqual(runtime.calls, 0)

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
                        "selected_window_ids": ["w0001"],
                        "selected_evidence_ids": ["e0001"],
                        "selected_evidence_moment_ids": ["ev1"],
                        "sufficiency_score": 0.8,
                        "necessity_score": 0.5,
                        "alertability_score": 0.7,
                        "counterfactual_faithfulness": 0.6,
                        "verification_decision": "sufficient",
                        "recommended_action": "finalize",
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

    def test_verify_hypothesis_recovers_real_window_ids_from_selected_evidence_ids(self):
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
                        "selected_window_ids": ["window_0"],
                        "selected_evidence_ids": ["e0001"],
                        "selected_evidence_moment_ids": ["ev1"],
                        "sufficiency_score": 0.8,
                        "necessity_score": 0.5,
                        "alertability_score": 0.7,
                        "counterfactual_faithfulness": 0.6,
                        "verification_decision": "sufficient",
                        "recommended_action": "finalize",
                    },
                }
            },
            self.multimodal_cache,
            state,
        )

        verdict = state.verification_records[-1]
        self.assertEqual(verdict["verification_parse_mode"], "self_report")
        self.assertEqual(verdict["requested_selected_window_ids"], ["window_0"])
        self.assertEqual(verdict["invalid_selected_window_ids"], ["window_0"])
        self.assertEqual(verdict["selection_resolution_source"], "selected_evidence_ids")
        self.assertEqual(verdict["verified_window_ids"], ["w0001"])
        self.assertEqual(verdict["best_effort_window_ids"], ["w0001"])
        self.assertEqual(state.active_evidence_window_ids, ["w0001"])

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
                        "selected_evidence_moment_ids": ["ev2"],
                        "sufficiency_score": 0.5,
                        "necessity_score": 0.5,
                        "alertability_score": 0.2,
                        "counterfactual_faithfulness": 0.4,
                        "verification_decision": "insufficient",
                        "recommended_action": "continue_search",
                    },
                }
            },
            self.multimodal_cache,
            state,
        )

        self.assertEqual(state.verification_records[-1]["verified_window_ids"], ["w0002"])
        self.assertEqual(state.active_evidence_window_ids, ["w0002"])

    def test_verify_hypothesis_rejects_unresolved_selection_when_invalid_ids_cannot_be_resolved(self):
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

        with self.assertRaisesRegex(
            ValueError,
            "must include at least one selected_window_id",
        ):
            execute_tool_call(
                {
                    "function": {
                        "name": "verify_hypothesis",
                        "arguments": {
                            "verification_mode": "final_check",
                            "claim": {"existence": "anomaly", "category": "robbery"},
                            "selected_window_ids": ["window_0"],
                            "selected_evidence_ids": ["e9999"],
                            "selected_evidence_moment_ids": ["ev_missing"],
                            "candidate_window_ids": ["window_1"],
                            "sufficiency_score": 0.2,
                            "necessity_score": 0.1,
                            "alertability_score": 0.1,
                            "counterfactual_faithfulness": 0.2,
                            "verification_decision": "insufficient",
                            "recommended_action": "continue_search",
                        },
                    }
                },
                self.multimodal_cache,
                state,
            )

        self.assertEqual(state.verification_records, [])
        self.assertEqual(state.active_evidence_window_ids, [])

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
                        "allow_external_verifier_fallback": True,
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

    def test_seek_evidence_uses_feature_guided_proposal_when_runtime_and_cache_exist(self):
        self.multimodal_cache["embedding"] = {
            "version": "saver_feature_cache_v1",
            "model_name": "dummy-siglip",
            "fps": 1.0,
            "frame_indices": list(range(10)),
            "timestamps_sec": [float(i) for i in range(10)],
            "embeddings": torch.tensor(
                [
                    [0.1, 0.9],
                    [0.1, 0.9],
                    [0.2, 0.8],
                    [0.2, 0.8],
                    [0.3, 0.7],
                    [0.98, 0.02],
                    [0.97, 0.03],
                    [0.1, 0.9],
                    [0.1, 0.9],
                    [0.1, 0.9],
                ],
                dtype=torch.float32,
            ),
            "embedding_dim": 2,
            "normalized": True,
            "frame_cache_signature": "dummy",
        }
        self.multimodal_cache["proposal_runtime"] = _FakeProposalRuntime({"person in red shirt": [1.0, 0.0]})

        message, state = execute_tool_call(
            {
                "function": {
                    "name": "seek_evidence",
                    "arguments": {
                        "query": "person in red shirt",
                        "start_sec": 0.0,
                        "end_sec": 9.0,
                        "num_frames": 2,
                    },
                }
            },
            self.multimodal_cache,
            self.state,
        )

        self.assertEqual(message["name"], "seek_evidence")
        entry = state.evidence_ledger[-1]
        self.assertEqual(entry["proposal_backend"], "feature_topk")
        self.assertTrue(entry["feature_cache_used"])
        self.assertEqual(entry["query_normalized"], "person in red shirt")
        self.assertEqual(entry["selected_frame_indices"], [5, 6])
        self.assertGreaterEqual(entry["proposal_candidate_count"], 1)
        self.assertIsNone(entry.get("proposal_fallback_reason"))

    def test_emit_alert_and_finalize_case_normalize_category_aliases_to_canonical_labels(self):
        _, state = execute_tool_call(
            {
                "function": {
                    "name": "emit_alert",
                    "arguments": {
                        "decision": "hard_alert",
                        "existence": "anomaly",
                        "category": "vehicle_collision",
                    },
                }
            },
            self.multimodal_cache,
            self.state,
        )
        self.assertEqual(state.alerts[-1]["category"], "traffic_accident")

        _, state = execute_tool_call(
            {
                "function": {
                    "name": "finalize_case",
                    "arguments": {"existence": "anomaly", "category": "physical_altercation"},
                }
            },
            self.multimodal_cache,
            state,
        )
        self.assertEqual(state.finalized_case["category"], "fighting")


if __name__ == "__main__":
    unittest.main()
