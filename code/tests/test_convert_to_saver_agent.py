import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path("/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import convert_to_saver_agent as ctsa
from saver_agent.environment import SaverVideoInteraction
from saver_agent.schema import SaverEnvironmentState


def _collect_verifier_recommended_actions(agent_view):
    environment = SaverVideoInteraction()
    state = SaverEnvironmentState()
    multimodal_cache = {
        "fps": float(agent_view["video_meta"]["fps"]),
        "duration": float(agent_view["video_meta"]["duration_sec"]),
        "question": str((agent_view.get("agent_task") or {}).get("task_prompt") or ""),
        "structured_target": dict(agent_view.get("structured_target") or {}),
        "tool_io": dict(agent_view.get("tool_io") or {}),
    }

    recommended_actions = []
    for step in list((agent_view.get("oracle_sft") or {}).get("trajectory") or []):
        payload = json.dumps(
            {
                "name": step.get("tool"),
                "arguments": step.get("arguments") or {},
            },
            ensure_ascii=False,
        )
        prediction = f"<think>test</think><tool_call>{payload}</tool_call>"
        next_obs, _, _, _, next_states = environment.execute_predictions(
            [prediction],
            [multimodal_cache],
            [state],
            [True],
        )
        state = next_states[0]
        if step.get("tool") != "verify_hypothesis":
            continue
        tool_message = next_obs[0]
        verification = json.loads(tool_message["content"][0]["text"])
        recommended_actions.append(str(verification.get("recommended_action") or ""))
    return recommended_actions


class ConvertToSaverAgentUnitTests(unittest.TestCase):
    def test_frame_interval_to_seconds_uses_inclusive_end_policy(self):
        start_sec, end_sec = ctsa.frame_interval_to_seconds(
            [1, 30], fps=10.0, frame_index_base=1, duration_sec=10.0
        )
        self.assertAlmostEqual(start_sec, 0.0)
        self.assertAlmostEqual(end_sec, 3.0)

    def test_complete_precursor_from_evidence_role(self):
        record = {
            "frame_index_base": 1,
            "video_meta": {"fps": 10.0, "duration_sec": 12.0},
            "label": {"is_anomaly": True},
            "temporal": {
                "anomaly_interval_frames": [31, 60],
                "precursor_interval_frames": None,
                "earliest_alert_frame": 31,
            },
            "evidence": {
                "evidence_moments": [
                    {
                        "moment_id": "ev1",
                        "start_frame": 11,
                        "end_frame": 20,
                        "role": "precursor",
                        "description": "lead-up",
                    },
                    {
                        "moment_id": "ev2",
                        "start_frame": 31,
                        "end_frame": 40,
                        "role": "trigger",
                        "description": "event",
                    },
                ]
            },
        }

        completed = ctsa.complete_precursor_interval(record)
        self.assertEqual(completed["frames"], [11, 20])
        self.assertEqual(completed["source"], "evidence_precursor_role")
        self.assertFalse(completed["auto_completed"])

    def test_complete_precursor_with_heuristic_when_missing(self):
        record = {
            "frame_index_base": 1,
            "video_meta": {"fps": 10.0, "duration_sec": 20.0},
            "label": {"is_anomaly": True},
            "temporal": {
                "anomaly_interval_frames": [51, 100],
                "precursor_interval_frames": None,
                "earliest_alert_frame": 51,
            },
            "evidence": {"evidence_moments": []},
        }

        completed = ctsa.complete_precursor_interval(record, heuristic_seconds=2.0, heuristic_fraction=0.2)
        self.assertEqual(completed["frames"], [31, 50])
        self.assertEqual(completed["source"], "heuristic_preceding_window")
        self.assertTrue(completed["auto_completed"])

    def test_complete_precursor_drops_non_strict_explicit_precursor(self):
        record = {
            "frame_index_base": 1,
            "video_meta": {"fps": 10.0, "duration_sec": 12.0},
            "label": {"is_anomaly": True},
            "temporal": {
                "anomaly_interval_frames": [31, 60],
                "precursor_interval_frames": [31, 40],
                "earliest_alert_frame": 31,
            },
            "evidence": {
                "evidence_moments": [
                    {
                        "moment_id": "ev1",
                        "start_frame": 31,
                        "end_frame": 40,
                        "role": "precursor",
                        "description": "starts with the anomaly",
                    }
                ]
            },
        }

        completed = ctsa.complete_precursor_interval(record)
        self.assertIsNone(completed["frames"])
        self.assertIsNone(completed["seconds"])
        self.assertEqual(completed["source"], "non_strict_precursor_dropped")
        self.assertFalse(completed["auto_completed"])

    def test_complete_precursor_treats_boundary_touching_interval_as_non_strict(self):
        record = {
            "frame_index_base": 1,
            "video_meta": {"fps": 30.0, "duration_sec": 12.0},
            "label": {"is_anomaly": True},
            "temporal": {
                "anomaly_interval_frames": [335, 600],
                "precursor_interval_frames": [152, 335],
                "earliest_alert_frame": 335,
            },
            "evidence": {
                "evidence_moments": [
                    {
                        "moment_id": "ev1",
                        "start_frame": 152,
                        "end_frame": 335,
                        "role": "precursor",
                        "description": "touches the anomaly start frame",
                    }
                ]
            },
        }

        completed = ctsa.complete_precursor_interval(record)
        self.assertIsNone(completed["frames"])
        self.assertIsNone(completed["seconds"])
        self.assertEqual(completed["source"], "non_strict_precursor_dropped")

    def test_agent_train_conversion_includes_structured_target_and_tool_io(self):
        record = {
            "video_id": "sample_anom",
            "file_name": "sample_anom.mp4",
            "video_path": "data/sample_anom.mp4",
            "source_dataset": "MSAD",
            "source_split": "anomaly_training",
            "split": "train",
            "frame_index_base": 1,
            "video_meta": {"fps": 10.0, "width": 1280, "height": 720, "total_frames": 120, "duration_sec": 12.0},
            "scene": {"scenario": "shop"},
            "key_objects": ["person", "bag"],
            "label": {"is_anomaly": True, "category": "robbery", "severity": 4, "hard_normal": False},
            "temporal": {
                "anomaly_interval_frames": [41, 90],
                "precursor_interval_frames": [21, 40],
                "earliest_alert_frame": 41,
            },
            "evidence": {
                "evidence_moments": [
                    {
                        "moment_id": "ev1",
                        "start_frame": 21,
                        "end_frame": 40,
                        "role": "precursor",
                        "description": "suspicious approach",
                    },
                    {
                        "moment_id": "ev2",
                        "start_frame": 41,
                        "end_frame": 50,
                        "role": "trigger",
                        "description": "bag snatch",
                    },
                ]
            },
            "counterfactual": {"type": "remove_actor_interaction", "text": "No interaction, no robbery."},
            "language": {"summary": "A robbery happens.", "rationale": "The suspect approaches and snatches a bag."},
            "qa_pairs": [],
            "provenance": {"annotation_status": "qwen_preannotated"},
            "qwen_preannotation": {"model_name": "Qwen3-VL-32B-Instruct"},
        }

        converted = ctsa.convert_record(record, mode="agent_train")
        self.assertEqual(converted["schema_version"], "saver_agent.v1")
        self.assertEqual(converted["structured_target"]["existence"], "anomaly")
        self.assertEqual(converted["structured_target"]["category"], "robbery")
        self.assertEqual(converted["tool_io"]["allowed_tools"][0], "scan_timeline")
        self.assertEqual(converted["tool_io"]["oracle_windows_frames"][0]["role"], "precursor")
        self.assertIn("task_prompt", converted["agent_task"])

    def test_agent_train_conversion_builds_proposal_supervision_from_key_objects(self):
        record = {
            "video_id": "sample_proposal",
            "file_name": "sample_proposal.mp4",
            "video_path": "data/sample_proposal.mp4",
            "source_dataset": "MSAD",
            "source_split": "anomaly_training",
            "split": "train",
            "frame_index_base": 1,
            "video_meta": {"fps": 10.0, "width": 1280, "height": 720, "total_frames": 120, "duration_sec": 12.0},
            "scene": {"scenario": "shop"},
            "key_objects": ["person in a red shirt being attacked by the person in a black shirt"],
            "label": {"is_anomaly": True, "category": "assault", "severity": 4, "hard_normal": False},
            "temporal": {
                "anomaly_interval_frames": [41, 90],
                "precursor_interval_frames": [21, 40],
                "earliest_alert_frame": 41,
            },
            "evidence": {
                "evidence_moments": [
                    {
                        "moment_id": "ev1",
                        "start_frame": 21,
                        "end_frame": 40,
                        "role": "precursor",
                        "description": "A person in red approaches a person in black before the assault.",
                    },
                    {
                        "moment_id": "ev2",
                        "start_frame": 41,
                        "end_frame": 50,
                        "role": "trigger",
                        "description": "The person in red attacks the person in black and the victim falls.",
                    },
                ]
            },
            "counterfactual": {"type": "remove_actor_interaction", "text": "No interaction, no assault."},
            "language": {"summary": "An assault happens.", "rationale": "The attacker in red hits the victim in black."},
            "qa_pairs": [],
            "provenance": {"annotation_status": "qwen_preannotated"},
            "qwen_preannotation": {"model_name": "Qwen3-VL-32B-Instruct"},
        }

        converted = ctsa.convert_record(record, mode="agent_train")

        self.assertIn("proposal_supervision", converted)
        queries = converted["proposal_supervision"]["queries"]
        self.assertGreaterEqual(len(queries), 1)
        normalized_texts = [entry["text"] for entry in queries[0]["normalized_queries"]]
        self.assertIn("person in red shirt", normalized_texts)
        self.assertIn("person in black shirt", normalized_texts)
        self.assertIn("physical struggle", normalized_texts)
        self.assertEqual(queries[0]["alignment_source"], "weak_alignment")
        self.assertIn("ev1", queries[0]["linked_moment_ids"])
        self.assertIn("ev2", queries[0]["linked_moment_ids"])

    def test_agent_train_conversion_downgrades_non_strict_precursor_roles(self):
        record = {
            "video_id": "sample_non_strict_precursor",
            "file_name": "sample_non_strict_precursor.mp4",
            "video_path": "data/sample_non_strict_precursor.mp4",
            "source_dataset": "MSAD",
            "source_split": "anomaly_training",
            "split": "train",
            "frame_index_base": 1,
            "video_meta": {"fps": 10.0, "width": 1280, "height": 720, "total_frames": 120, "duration_sec": 12.0},
            "scene": {"scenario": "street"},
            "key_objects": ["person"],
            "label": {"is_anomaly": True, "category": "assault", "severity": 4, "hard_normal": False},
            "temporal": {
                "anomaly_interval_frames": [31, 90],
                "precursor_interval_frames": [31, 40],
                "earliest_alert_frame": 31,
            },
            "evidence": {
                "evidence_moments": [
                    {
                        "moment_id": "ev1",
                        "start_frame": 31,
                        "end_frame": 40,
                        "role": "precursor",
                        "description": "already part of the anomaly onset",
                    },
                    {
                        "moment_id": "ev2",
                        "start_frame": 41,
                        "end_frame": 50,
                        "role": "trigger",
                        "description": "decisive physical contact",
                    },
                ]
            },
            "counterfactual": {"type": "remove_actor_interaction", "text": "No contact, no assault."},
            "language": {"summary": "An assault happens.", "rationale": "Physical contact is visible."},
            "qa_pairs": [],
            "provenance": {"annotation_status": "qwen_preannotated"},
            "qwen_preannotation": {"model_name": "Qwen3-VL-32B-Instruct"},
        }

        converted = ctsa.convert_record(record, mode="oracle_sft")

        self.assertIsNone(converted["structured_target"]["precursor_interval_sec"])
        oracle_roles = [entry["role"] for entry in converted["tool_io"]["oracle_windows_frames"]]
        self.assertNotIn("precursor", oracle_roles)
        self.assertIn("trigger", oracle_roles)
        seek_roles = [
            str((step.get("arguments") or {}).get("role") or "")
            for step in (converted.get("oracle_sft") or {}).get("trajectory") or []
            if step.get("tool") == "seek_evidence"
        ]
        self.assertNotIn("precursor", seek_roles)

    def test_agent_train_conversion_removes_precursor_qas_when_precursor_is_recovered_from_non_annotation_source(self):
        record = {
            "video_id": "sample_non_strict_precursor_with_qa",
            "file_name": "sample_non_strict_precursor_with_qa.mp4",
            "video_path": "data/sample_non_strict_precursor_with_qa.mp4",
            "source_dataset": "MSAD",
            "source_split": "anomaly_training",
            "split": "train",
            "frame_index_base": 1,
            "video_meta": {"fps": 30.0, "width": 1280, "height": 720, "total_frames": 1800, "duration_sec": 60.0},
            "scene": {"scenario": "patio"},
            "key_objects": ["person", "red object"],
            "label": {"is_anomaly": True, "category": "fire", "severity": 4, "hard_normal": False},
            "temporal": {
                "anomaly_interval_frames": [335, 1800],
                "precursor_interval_frames": [152, 335],
                "earliest_alert_frame": 335,
            },
            "evidence": {
                "evidence_moments": [
                    {
                        "moment_id": "ev1",
                        "start_frame": 1,
                        "end_frame": 152,
                        "role": "precursor",
                        "description": "pre-anomaly handling",
                    },
                    {
                        "moment_id": "ev2",
                        "start_frame": 152,
                        "end_frame": 335,
                        "role": "trigger",
                        "description": "ignition onset",
                    },
                ]
            },
            "counterfactual": {"type": "remove_dangerous_object", "text": "Remove ignition source."},
            "language": {"summary": "A fire happens.", "rationale": "The fire starts abruptly."},
            "qa_pairs": [
                {"type": "existence", "question": "Is there an anomaly?", "answer": "Yes."},
                {
                    "type": "precursor_temporal",
                    "question": "When do visible precursor cues appear before the anomaly?",
                    "answer": "Precursor cues appear before the fire.",
                },
            ],
            "provenance": {"annotation_status": "qwen_preannotated"},
            "qwen_preannotation": {"model_name": "Qwen3-VL-32B-Instruct"},
        }

        converted = ctsa.convert_record(record, mode="oracle_sft")

        self.assertEqual(converted["temporal"]["precursor_resolution"]["source"], "evidence_precursor_role")
        self.assertEqual(converted["structured_target"]["precursor_interval_sec"], [0.0, 5.066667])
        self.assertFalse(
            any(qa.get("type") == "precursor_temporal" for qa in converted.get("qa_pairs") or [])
        )
        finalize_args = [
            step.get("arguments") or {}
            for step in (converted.get("oracle_sft") or {}).get("trajectory") or []
            if step.get("tool") == "finalize_case"
        ][-1]
        self.assertEqual(finalize_args.get("precursor_interval_sec"), [0.0, 5.066667])
        final_decision = (converted.get("oracle_sft") or {}).get("final_decision") or {}
        self.assertEqual(final_decision.get("precursor_interval_sec"), [0.0, 5.066667])

    def test_agent_train_finalize_schema_uses_canonical_category_enum(self):
        record = {
            "video_id": "sample_alias",
            "file_name": "sample_alias.mp4",
            "video_path": "data/sample_alias.mp4",
            "source_dataset": "MSAD",
            "source_split": "anomaly_training",
            "split": "train",
            "frame_index_base": 1,
            "video_meta": {"fps": 10.0, "width": 1280, "height": 720, "total_frames": 120, "duration_sec": 12.0},
            "scene": {"scenario": "shop"},
            "key_objects": ["car"],
            "label": {"is_anomaly": True, "category": "vehicle_collision", "severity": 4, "hard_normal": False},
            "temporal": {
                "anomaly_interval_frames": [41, 90],
                "precursor_interval_frames": [21, 40],
                "earliest_alert_frame": 41,
            },
            "evidence": {
                "evidence_moments": [
                    {
                        "moment_id": "ev1",
                        "start_frame": 41,
                        "end_frame": 50,
                        "role": "trigger",
                        "description": "A car collides with another vehicle.",
                    }
                ]
            },
            "counterfactual": {"type": "none", "text": None},
            "language": {"summary": "A traffic accident happens.", "rationale": "Two vehicles collide."},
            "qa_pairs": [],
            "provenance": {"annotation_status": "qwen_preannotated"},
            "qwen_preannotation": {"model_name": "Qwen3-VL-32B-Instruct"},
        }

        converted = ctsa.convert_record(record, mode="agent_train")
        schema = converted["tool_io"]["finalize_case_schema"]["properties"]["category"]

        self.assertEqual(converted["structured_target"]["category"], "traffic_accident")
        self.assertIn("traffic_accident", schema["enum"])
        self.assertIn("normal", schema["enum"])
        self.assertNotIn("vehicle_collision", schema["enum"])

    def test_oracle_sft_conversion_emits_alert_and_finalize_steps(self):
        record = {
            "video_id": "sample_normal",
            "file_name": "sample_normal.mp4",
            "video_path": "data/sample_normal.mp4",
            "source_dataset": "MSAD",
            "source_split": "normal_training",
            "split": "train",
            "frame_index_base": 1,
            "video_meta": {"fps": 10.0, "width": 1280, "height": 720, "total_frames": 100, "duration_sec": 10.0},
            "scene": {"scenario": "frontdoor"},
            "key_objects": ["worker"],
            "label": {"is_anomaly": False, "category": "normal", "severity": 0, "hard_normal": True},
            "temporal": {
                "anomaly_interval_frames": None,
                "precursor_interval_frames": None,
                "earliest_alert_frame": None,
            },
            "evidence": {"evidence_moments": []},
            "counterfactual": {"type": "none", "text": None},
            "language": {"summary": "Normal loading activity.", "rationale": "Routine work only."},
            "qa_pairs": [],
            "provenance": {"annotation_status": "qwen_preannotated"},
            "qwen_preannotation": {"model_name": "Qwen3-VL-32B-Instruct"},
        }

        converted = ctsa.convert_record(record, mode="oracle_sft")
        actions = [step["tool"] for step in converted["oracle_sft"]["trajectory"]]
        self.assertEqual(actions[0], "scan_timeline")
        self.assertIn("verify_hypothesis", actions)
        self.assertEqual(actions.count("scan_timeline"), 1)
        self.assertGreaterEqual(actions.count("seek_evidence"), 3)
        self.assertEqual(actions[-2], "verify_hypothesis")
        self.assertEqual(actions[-1], "finalize_case")
        self.assertNotIn("emit_alert", actions)

        recommended_actions = _collect_verifier_recommended_actions(converted)
        self.assertEqual(recommended_actions, ["finalize"])

    def test_oracle_sft_conversion_emits_structured_self_verification_feedback(self):
        record = {
            "video_id": "sample_structured_verify",
            "file_name": "sample_structured_verify.mp4",
            "video_path": "data/sample_structured_verify.mp4",
            "source_dataset": "MSAD",
            "source_split": "anomaly_training",
            "split": "train",
            "frame_index_base": 1,
            "video_meta": {"fps": 10.0, "width": 1280, "height": 720, "total_frames": 120, "duration_sec": 12.0},
            "scene": {"scenario": "street"},
            "key_objects": ["person in red", "person in black"],
            "label": {"is_anomaly": True, "category": "assault", "severity": 4, "hard_normal": False},
            "temporal": {
                "anomaly_interval_frames": [41, 90],
                "precursor_interval_frames": [21, 40],
                "earliest_alert_frame": 41,
            },
            "evidence": {
                "evidence_moments": [
                    {
                        "moment_id": "ev1",
                        "start_frame": 21,
                        "end_frame": 40,
                        "role": "precursor",
                        "description": "suspicious approach",
                    },
                    {
                        "moment_id": "ev2",
                        "start_frame": 41,
                        "end_frame": 50,
                        "role": "trigger",
                        "description": "first violent contact",
                    },
                    {
                        "moment_id": "ev3",
                        "start_frame": 51,
                        "end_frame": 70,
                        "role": "peak_action",
                        "description": "continued attack",
                    },
                ]
            },
            "counterfactual": {"type": "remove_actor_interaction", "text": "No attack without interaction."},
            "language": {"summary": "An assault happens.", "rationale": "A person attacks another person."},
            "qa_pairs": [],
            "provenance": {"annotation_status": "qwen_preannotated"},
            "qwen_preannotation": {"model_name": "Qwen3-VL-32B-Instruct"},
        }

        converted = ctsa.convert_record(record, mode="oracle_sft")
        verify_steps = [step for step in converted["oracle_sft"]["trajectory"] if step.get("tool") == "verify_hypothesis"]

        self.assertGreaterEqual(len(verify_steps), 3)
        for step in verify_steps:
            feedback = step.get("oracle_verifier_feedback") or {}
            self.assertIn("verification_decision", feedback)
            self.assertIn("selected_window_ids", feedback)
            self.assertIn("selected_evidence_moment_ids", feedback)
            self.assertIn("sufficiency_score", feedback)
            self.assertIn("necessity_score", feedback)
            self.assertIn("alertability_score", feedback)
            self.assertIn("counterfactual_faithfulness", feedback)
            self.assertIn("recommended_action", feedback)
            self.assertNotIn("primary_status", feedback)
            self.assertNotIn("alert_status", feedback)
            self.assertNotIn("failure_reasons", feedback)
            self.assertNotIn("selected_evidence_ids", feedback)

        recommended_actions = {
            str((step.get("oracle_verifier_feedback") or {}).get("recommended_action") or "")
            for step in verify_steps
        }
        self.assertIn("revise_claim", recommended_actions)
        self.assertIn("continue_search", recommended_actions)
        self.assertIn("finalize", recommended_actions)

    def test_oracle_sft_seek_queries_do_not_leak_category_name(self):
        record = {
            "video_id": "Assault_1",
            "file_name": "Assault_1.mp4",
            "video_path": "data/Assault_1.mp4",
            "source_dataset": "MSAD",
            "source_split": "anomaly_training",
            "split": "train",
            "frame_index_base": 1,
            "video_meta": {"fps": 10.0, "width": 1280, "height": 720, "total_frames": 120, "duration_sec": 12.0},
            "scene": {"scenario": "street"},
            "key_objects": ["person in red", "person in black"],
            "label": {"is_anomaly": True, "category": "assault", "severity": 4, "hard_normal": False},
            "temporal": {
                "anomaly_interval_frames": [41, 90],
                "precursor_interval_frames": [21, 40],
                "earliest_alert_frame": 41,
            },
            "evidence": {
                "evidence_moments": [
                    {
                        "moment_id": "ev1",
                        "start_frame": 21,
                        "end_frame": 40,
                        "role": "precursor",
                        "description": "A confrontation builds up.",
                    },
                    {
                        "moment_id": "ev2",
                        "start_frame": 41,
                        "end_frame": 50,
                        "role": "trigger",
                        "description": "Physical aggressive contact starts.",
                    },
                ]
            },
            "counterfactual": {"type": "remove_actor_interaction", "text": "No interaction, no assault."},
            "language": {"summary": "An assault happens.", "rationale": "One person attacks another."},
            "qa_pairs": [],
            "provenance": {"annotation_status": "qwen_preannotated"},
            "qwen_preannotation": {"model_name": "Qwen3-VL-32B-Instruct"},
        }

        converted = ctsa.convert_record(record, mode="oracle_sft")
        queries = [
            str((step.get("arguments") or {}).get("query") or "")
            for step in converted["oracle_sft"]["trajectory"]
            if str(step.get("tool") or "") == "seek_evidence"
        ]

        self.assertGreaterEqual(len(queries), 1)
        self.assertTrue(all("assault" not in query.lower() for query in queries))

    def test_oracle_sft_scan_timeline_arguments_match_registered_schema(self):
        record = {
            "video_id": "sample_stride",
            "file_name": "sample_stride.mp4",
            "video_path": "data/sample_stride.mp4",
            "source_dataset": "MSAD",
            "source_split": "normal_training",
            "split": "train",
            "frame_index_base": 1,
            "video_meta": {"fps": 10.0, "width": 1280, "height": 720, "total_frames": 100, "duration_sec": 10.0},
            "scene": {"scenario": "frontdoor"},
            "key_objects": ["worker"],
            "label": {"is_anomaly": False, "category": "normal", "severity": 0, "hard_normal": True},
            "temporal": {
                "anomaly_interval_frames": None,
                "precursor_interval_frames": None,
                "earliest_alert_frame": None,
            },
            "evidence": {"evidence_moments": []},
            "counterfactual": {"type": "none", "text": None},
            "language": {"summary": "Normal loading activity.", "rationale": "Routine work only."},
            "qa_pairs": [],
            "provenance": {"annotation_status": "qwen_preannotated"},
            "qwen_preannotation": {"model_name": "Qwen3-VL-32B-Instruct"},
        }

        converted = ctsa.convert_record(record, mode="oracle_sft")
        first_step = converted["oracle_sft"]["trajectory"][0]

        self.assertEqual(first_step["tool"], "scan_timeline")
        self.assertIn("stride_sec", first_step["arguments"])

    def test_oracle_sft_conversion_carries_moment_id_in_seek_evidence_arguments(self):
        record = {
            "video_id": "sample_oracle_anom",
            "file_name": "sample_oracle_anom.mp4",
            "video_path": "data/sample_oracle_anom.mp4",
            "source_dataset": "MSAD",
            "source_split": "anomaly_training",
            "split": "train",
            "frame_index_base": 1,
            "video_meta": {"fps": 10.0, "width": 1280, "height": 720, "total_frames": 120, "duration_sec": 12.0},
            "scene": {"scenario": "shop"},
            "key_objects": ["person", "bag"],
            "label": {"is_anomaly": True, "category": "robbery", "severity": 4, "hard_normal": False},
            "temporal": {
                "anomaly_interval_frames": [41, 90],
                "precursor_interval_frames": [21, 40],
                "earliest_alert_frame": 41,
            },
            "evidence": {
                "evidence_moments": [
                    {
                        "moment_id": "ev1",
                        "start_frame": 21,
                        "end_frame": 40,
                        "role": "precursor",
                        "description": "suspicious approach",
                    },
                    {
                        "moment_id": "ev2",
                        "start_frame": 41,
                        "end_frame": 50,
                        "role": "trigger",
                        "description": "bag snatch",
                    },
                ]
            },
            "counterfactual": {"type": "remove_actor_interaction", "text": "No interaction, no robbery."},
            "language": {"summary": "A robbery happens.", "rationale": "The suspect approaches and snatches a bag."},
            "qa_pairs": [],
            "provenance": {"annotation_status": "qwen_preannotated"},
            "qwen_preannotation": {"model_name": "Qwen3-VL-32B-Instruct"},
        }

        converted = ctsa.convert_record(record, mode="oracle_sft")
        seek_steps = [
            step for step in converted["oracle_sft"]["trajectory"]
            if step.get("tool") == "seek_evidence"
        ]

        self.assertGreaterEqual(len(seek_steps), 2)
        self.assertEqual(seek_steps[0]["arguments"]["moment_id"], "ev1")
        self.assertEqual(seek_steps[1]["arguments"]["moment_id"], "ev2")

    def test_oracle_sft_conversion_builds_branchy_verifier_trajectory_for_two_moment_anomaly(self):
        record = {
            "video_id": "sample_branchy_anom",
            "file_name": "sample_branchy_anom.mp4",
            "video_path": "data/sample_branchy_anom.mp4",
            "source_dataset": "MSAD",
            "source_split": "anomaly_training",
            "split": "train",
            "frame_index_base": 1,
            "video_meta": {"fps": 10.0, "width": 1280, "height": 720, "total_frames": 120, "duration_sec": 12.0},
            "scene": {"scenario": "shop"},
            "key_objects": ["person", "bag"],
            "label": {"is_anomaly": True, "category": "robbery", "severity": 4, "hard_normal": False},
            "temporal": {
                "anomaly_interval_frames": [41, 90],
                "precursor_interval_frames": [21, 40],
                "earliest_alert_frame": 41,
            },
            "evidence": {
                "evidence_moments": [
                    {
                        "moment_id": "ev1",
                        "start_frame": 21,
                        "end_frame": 40,
                        "role": "precursor",
                        "description": "suspicious approach",
                    },
                    {
                        "moment_id": "ev2",
                        "start_frame": 41,
                        "end_frame": 50,
                        "role": "trigger",
                        "description": "bag snatch",
                    },
                ]
            },
            "counterfactual": {"type": "remove_actor_interaction", "text": "No interaction, no robbery."},
            "language": {"summary": "A robbery happens.", "rationale": "The suspect approaches and snatches a bag."},
            "qa_pairs": [],
            "provenance": {"annotation_status": "qwen_preannotated"},
            "qwen_preannotation": {"model_name": "Qwen3-VL-32B-Instruct"},
        }

        converted = ctsa.convert_record(record, mode="oracle_sft")
        trajectory = list((converted.get("oracle_sft") or {}).get("trajectory") or [])
        verify_indices = [index for index, step in enumerate(trajectory) if step.get("tool") == "verify_hypothesis"]

        self.assertGreaterEqual(len(verify_indices), 3)
        self.assertTrue(
            any(
                step.get("tool") in {"scan_timeline", "seek_evidence"}
                for step in trajectory[verify_indices[0] + 1 : verify_indices[1]]
            )
        )
        self.assertTrue(
            any(
                step.get("tool") in {"scan_timeline", "seek_evidence"}
                for step in trajectory[verify_indices[1] + 1 : verify_indices[2]]
            )
        )

        verify_steps = [step for step in trajectory if step.get("tool") == "verify_hypothesis"]
        feedback_actions = [
            str((step.get("oracle_verifier_feedback") or {}).get("recommended_action") or "")
            for step in verify_steps
        ]
        self.assertEqual(feedback_actions[:3], ["revise_claim", "continue_search", "finalize"])
        self.assertEqual(verify_steps[0]["arguments"].get("selected_evidence_moment_ids"), ["ev1"])

        recommended_actions = _collect_verifier_recommended_actions(converted)
        self.assertEqual(recommended_actions[-1], "finalize")
        self.assertTrue(all(action != "finalize" for action in recommended_actions[:2]))


class ConvertToSaverAgentCliTests(unittest.TestCase):
    def test_cli_converts_jsonl_to_agent_train(self):
        sample = {
            "video_id": "cli_case",
            "file_name": "cli_case.mp4",
            "video_path": "data/cli_case.mp4",
            "source_dataset": "MSAD",
            "source_split": "anomaly_training",
            "split": "train",
            "frame_index_base": 1,
            "video_meta": {"fps": 5.0, "width": 1280, "height": 720, "total_frames": 50, "duration_sec": 10.0},
            "scene": {"scenario": "road"},
            "key_objects": ["car"],
            "label": {"is_anomaly": True, "category": "traffic_accident", "severity": 3, "hard_normal": False},
            "temporal": {
                "anomaly_interval_frames": [11, 40],
                "precursor_interval_frames": None,
                "earliest_alert_frame": 11,
            },
            "evidence": {
                "evidence_moments": [
                    {
                        "moment_id": "ev1",
                        "start_frame": 11,
                        "end_frame": 20,
                        "role": "trigger",
                        "description": "collision begins",
                    }
                ]
            },
            "counterfactual": {"type": "restore_safe_context", "text": "No collision if the road were clear."},
            "language": {"summary": "A crash occurs.", "rationale": "Two vehicles collide."},
            "qa_pairs": [],
            "provenance": {"annotation_status": "qwen_preannotated"},
            "qwen_preannotation": {"model_name": "Qwen3-VL-32B-Instruct"},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.jsonl"
            output_path = Path(tmpdir) / "output.jsonl"
            with input_path.open("w", encoding="utf-8") as f:
                f.write(json.dumps(sample) + "\n")

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "convert_to_saver_agent.py"),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--mode",
                    "agent_train",
                ],
                check=True,
            )

            with output_path.open(encoding="utf-8") as f:
                converted = json.loads(f.readline())

            self.assertEqual(converted["video_id"], "cli_case")
            self.assertEqual(converted["auto_completed"]["precursor_interval"], True)
            self.assertEqual(converted["structured_target"]["existence"], "anomaly")

    def test_cli_include_splits_filters_output_rows(self):
        train_sample = {
            "video_id": "cli_train_case",
            "file_name": "cli_train_case.mp4",
            "video_path": "data/cli_train_case.mp4",
            "source_dataset": "MSAD",
            "source_split": "anomaly_training",
            "split": "train",
            "frame_index_base": 1,
            "video_meta": {"fps": 5.0, "width": 1280, "height": 720, "total_frames": 50, "duration_sec": 10.0},
            "scene": {"scenario": "road"},
            "key_objects": ["car"],
            "label": {"is_anomaly": True, "category": "traffic_accident", "severity": 3, "hard_normal": False},
            "temporal": {"anomaly_interval_frames": [11, 40], "precursor_interval_frames": None, "earliest_alert_frame": 11},
            "evidence": {"evidence_moments": []},
            "counterfactual": {"type": "restore_safe_context", "text": "No collision if the road were clear."},
            "language": {"summary": "A crash occurs.", "rationale": "Two vehicles collide."},
            "qa_pairs": [],
            "provenance": {"annotation_status": "qwen_preannotated"},
            "qwen_preannotation": {"model_name": "Qwen3-VL-32B-Instruct"},
        }
        test_sample = {
            **train_sample,
            "video_id": "cli_test_case",
            "split": "test",
            "source_split": "anomaly_testing",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.jsonl"
            output_path = Path(tmpdir) / "output.jsonl"
            with input_path.open("w", encoding="utf-8") as f:
                f.write(json.dumps(test_sample) + "\n")
                f.write(json.dumps(train_sample) + "\n")

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "convert_to_saver_agent.py"),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--mode",
                    "agent_train",
                    "--include-splits",
                    "train",
                ],
                check=True,
            )

            with output_path.open(encoding="utf-8") as f:
                converted_rows = [json.loads(line) for line in f if line.strip()]

        self.assertEqual(len(converted_rows), 1)
        self.assertEqual(converted_rows[0]["video_id"], "cli_train_case")
        self.assertEqual(converted_rows[0]["split"], "train")

    def test_task_prompt_is_label_agnostic_between_anomaly_and_normal_records(self):
        anomaly_record = {
            "video_id": "sample_anomaly_prompt",
            "file_name": "sample_anomaly_prompt.mp4",
            "video_path": "data/sample_anomaly_prompt.mp4",
            "source_dataset": "MSAD",
            "source_split": "anomaly_training",
            "split": "train",
            "frame_index_base": 1,
            "video_meta": {"fps": 10.0, "width": 1280, "height": 720, "total_frames": 120, "duration_sec": 12.0},
            "scene": {"scenario": "shop"},
            "key_objects": ["person"],
            "label": {"is_anomaly": True, "category": "assault", "severity": 4, "hard_normal": False},
            "temporal": {
                "anomaly_interval_frames": [41, 90],
                "precursor_interval_frames": [21, 40],
                "earliest_alert_frame": 41,
            },
            "evidence": {
                "evidence_moments": [
                    {
                        "moment_id": "ev1",
                        "start_frame": 41,
                        "end_frame": 50,
                        "role": "trigger",
                        "description": "attack",
                    }
                ]
            },
            "counterfactual": {"type": "none", "text": None},
            "language": {"summary": "An assault happens.", "rationale": "A person attacks another person."},
            "qa_pairs": [],
            "provenance": {"annotation_status": "qwen_preannotated"},
            "qwen_preannotation": {"model_name": "Qwen3-VL-32B-Instruct"},
        }
        normal_record = {
            **anomaly_record,
            "video_id": "sample_normal_prompt",
            "file_name": "sample_normal_prompt.mp4",
            "video_path": "data/sample_normal_prompt.mp4",
            "label": {"is_anomaly": False, "category": "normal", "severity": 0, "hard_normal": False},
            "temporal": {
                "anomaly_interval_frames": None,
                "precursor_interval_frames": None,
                "earliest_alert_frame": None,
            },
            "evidence": {"evidence_moments": []},
            "language": {"summary": "Normal activity.", "rationale": "Nothing abnormal happens."},
        }

        anomaly_converted = ctsa.convert_record(anomaly_record, mode="agent_train")
        normal_converted = ctsa.convert_record(normal_record, mode="agent_train")

        self.assertEqual(
            anomaly_converted["agent_task"]["task_prompt"],
            normal_converted["agent_task"]["task_prompt"],
        )
        self.assertIn(
            "Determine whether the video contains an actionable anomaly or is normal",
            anomaly_converted["agent_task"]["task_prompt"],
        )


if __name__ == "__main__":
    unittest.main()
