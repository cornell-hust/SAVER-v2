import importlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path("/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saver_agent.environment import SaverEnvironmentState, SaverVideoInteraction
from saver_agent.config import RolloutTraceConfig, SaverAgentConfig


try:
    ADAPTER_MODULE = importlib.import_module("saver_agent.adapter")
except ModuleNotFoundError:
    ADAPTER_MODULE = None

try:
    ROLLOUT_MODULE = importlib.import_module("saver_agent.rollout")
except ModuleNotFoundError:
    ROLLOUT_MODULE = None


class SaverAgentAdapterTests(unittest.TestCase):
    def test_tool_observation_adapter_appends_timesearch_followup_prompt(self):
        self.assertIsNotNone(ADAPTER_MODULE, "saver_agent.adapter module is missing")

        adapter = ADAPTER_MODULE.TimeSearchRolloutAdapter()
        multimodal_cache = {
            "video": torch.zeros(4, 3, 2, 2),
            "embedding": None,
            "fps": 1.0,
            "duration": 4.0,
            "question": "Determine whether an anomaly exists.",
            "tool_io": {},
        }
        raw_tool_message = {
            "role": "tool",
            "name": "scan_timeline",
            "content": [
                {"type": "text", "text": "0.000s"},
                {"type": "image", "image": multimodal_cache["video"][0]},
                {"type": "text", "text": "1.000s"},
                {"type": "image", "image": multimodal_cache["video"][1]},
                {"type": "text", "text": "Scanned timeline window [0.000, 2.000] and selected 2 frames."},
            ],
        }

        adapted = adapter.adapt_tool_observation(raw_tool_message, multimodal_cache)

        self.assertEqual(adapted["role"], "tool")
        self.assertEqual(adapted["name"], "scan_timeline")
        self.assertEqual(sum(1 for item in adapted["content"] if item["type"] == "image"), 2)
        self.assertIn("Here are selected frames.", adapted["content"][-1]["text"])
        self.assertIn("If the frames provided above are sufficient", adapted["content"][-1]["text"])

    def test_verify_observation_prompt_pushes_finalize_when_verdict_is_complete(self):
        self.assertIsNotNone(ADAPTER_MODULE, "saver_agent.adapter module is missing")

        adapter = ADAPTER_MODULE.TimeSearchRolloutAdapter()
        multimodal_cache = {"question": "Determine whether an anomaly exists.", "duration": 8.0}
        tool_message = {
            "role": "tool",
            "name": "verify_hypothesis",
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "primary_status": "complete",
                            "alert_status": "justified",
                            "recommended_action": "finalize",
                            "claim": {"existence": "anomaly", "category": "assault"},
                        }
                    ),
                }
            ],
        }

        adapted = adapter.adapt_tool_observation(tool_message, multimodal_cache)

        self.assertIn("finalize_case", adapted["content"][-1]["text"])
        self.assertIn("<answer>", adapted["content"][-1]["text"])

    def test_finalize_case_observation_prompt_requests_terminal_answer(self):
        self.assertIsNotNone(ADAPTER_MODULE, "saver_agent.adapter module is missing")

        adapter = ADAPTER_MODULE.TimeSearchRolloutAdapter()
        multimodal_cache = {"question": "Determine whether an anomaly exists.", "duration": 8.0}
        tool_message = {
            "role": "tool",
            "name": "finalize_case",
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "status": "finalized",
                            "finalized_case": {"existence": "anomaly", "category": "assault"},
                        }
                    ),
                }
            ],
        }

        adapted = adapter.adapt_tool_observation(tool_message, multimodal_cache)

        self.assertIn("Output the final answer now", adapted["content"][-1]["text"])
        self.assertIn("<answer>", adapted["content"][-1]["text"])


class SaverAgentRolloutRunnerTests(unittest.TestCase):
    def setUp(self):
        self.item = {
            "video_id": "sample_rollout",
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "system prompt"}]},
                {"role": "user", "content": [{"type": "text", "text": "user prompt"}]},
            ],
            "multimodal_cache": {
                "video": torch.zeros(8, 3, 2, 2),
                "embedding": None,
                "fps": 1.0,
                "duration": 8.0,
                "question": "Determine whether an anomaly exists.",
                "preview_frames": torch.zeros(3, 3, 2, 2),
                "preview_timestamps": [0.0, 2.0, 4.0],
                "structured_target": {
                    "existence": "anomaly",
                    "category": "assault",
                    "severity": 4,
                    "anomaly_interval_sec": [0.0, 3.0],
                    "precursor_interval_sec": [0.0, 0.5],
                    "earliest_alert_sec": 0.0,
                    "counterfactual_type": "remove_actor_interaction",
                },
                "tool_io": {
                    "oracle_windows_sec": [
                        {"moment_id": "ev1", "role": "trigger", "window": [0.0, 3.0], "description": "trigger"},
                    ],
                    "finalize_case_schema": {"type": "object", "required": ["existence"]},
                },
            },
        }

    def test_runner_executes_replay_policy_and_returns_final_answer_payload(self):
        self.assertIsNotNone(ADAPTER_MODULE, "saver_agent.adapter module is missing")
        self.assertIsNotNone(ROLLOUT_MODULE, "saver_agent.rollout module is missing")

        runner = ROLLOUT_MODULE.SaverRolloutRunner(
            environment=SaverVideoInteraction(),
            adapter=ADAPTER_MODULE.TimeSearchRolloutAdapter(),
            max_turns=3,
        )
        policy = ROLLOUT_MODULE.ReplayPolicy(
            [
                '<think>scan</think><tool_call>{"name":"scan_timeline","arguments":{"start_sec":0.0,"end_sec":3.0,"num_frames":2}}</tool_call>',
                '<think>verify</think><tool_call>{"name":"verify_hypothesis","arguments":{"verification_mode":"final_check","claim":{"existence":"anomaly","category":"assault","earliest_alert_sec":0.0},"candidate_window_ids":["w0001"],"alert":{"decision":"hard_alert","alert_sec":0.1}}}</tool_call>',
                '<think>done</think><answer>{"existence":"normal","summary":"No anomaly found."}</answer>',
            ]
        )

        result = runner.run_episode(self.item, policy, initial_state=SaverEnvironmentState())

        self.assertEqual(result["terminated_reason"], "answered")
        self.assertEqual(result["final_answer"]["existence"], "normal")
        self.assertEqual(result["num_turns"], 3)
        self.assertEqual(len(result["turns"]), 3)
        self.assertEqual(result["turns"][0]["action"], "tool_call")
        self.assertEqual(result["turns"][0]["tool_name"], "scan_timeline")
        self.assertEqual(result["turns"][1]["tool_name"], "verify_hypothesis")
        self.assertEqual(result["turns"][1]["verifier_primary_status"], "incomplete")
        self.assertEqual(result["turns"][1]["verifier_alert_status"], "premature")
        self.assertEqual(result["state"]["visited_windows"][0]["kind"], "scan")
        self.assertEqual(
            result["state"]["last_claim"],
            {"existence": "anomaly", "category": "assault", "earliest_alert_sec": 0.0},
        )
        self.assertNotIn("severity", result["state"]["last_claim"])
        self.assertEqual(result["messages"][-1]["role"], "assistant")

    def test_runner_records_reward_and_verifier_friendly_trace_fields(self):
        self.assertIsNotNone(ADAPTER_MODULE, "saver_agent.adapter module is missing")
        self.assertIsNotNone(ROLLOUT_MODULE, "saver_agent.rollout module is missing")

        config = SaverAgentConfig(
            rollout_trace=RolloutTraceConfig(
                record_observation_content=True,
                record_state_deltas=True,
                record_message_history=True,
            )
        )
        runner = ROLLOUT_MODULE.SaverRolloutRunner(
            environment=SaverVideoInteraction(),
            adapter=ADAPTER_MODULE.TimeSearchRolloutAdapter(),
            max_turns=3,
            config=config,
        )
        policy = ROLLOUT_MODULE.ReplayPolicy(
            [
                '<think>scan</think><tool_call>{"name":"scan_timeline","arguments":{"start_sec":0.0,"end_sec":3.0,"num_frames":2}}</tool_call>',
                '<think>verify</think><tool_call>{"name":"verify_hypothesis","arguments":{"verification_mode":"final_check","claim":{"existence":"anomaly","category":"assault","earliest_alert_sec":0.0},"candidate_window_ids":["w0001"],"alert":{"decision":"hard_alert","alert_sec":0.1}}}</tool_call>',
                '<think>done</think><answer>{"existence":"normal","summary":"No anomaly found."}</answer>',
            ]
        )

        result = runner.run_episode(self.item, policy, initial_state=SaverEnvironmentState())

        self.assertEqual(result["config_snapshot"]["rollout_trace"]["record_state_deltas"], True)
        self.assertEqual(result["preview_trace"]["preview_frame_count"], 3)
        self.assertEqual(result["termination_trace"]["reason"], "answered")
        self.assertEqual(result["termination_trace"]["terminated_at_step"], 3)
        self.assertIn("assistant_response_raw", result["turns"][0])
        self.assertEqual(result["turns"][0]["assistant_action"], "tool_call")
        self.assertEqual(result["turns"][0]["parsed_tool_call"]["name"], "scan_timeline")
        self.assertEqual(result["turns"][0]["tool_image_count"], 2)
        self.assertEqual(len(result["turns"][0]["tool_timestamps"]), 2)
        self.assertIn("selected 2 frames", result["turns"][0]["tool_observation_summary"])
        self.assertEqual(result["turns"][0]["state_delta"]["new_visited_windows"][0]["kind"], "scan")
        self.assertEqual(result["turns"][0]["new_evidence_ids"], ["e0001"])
        self.assertEqual(result["turns"][1]["verifier_primary_status"], "incomplete")
        self.assertEqual(result["turns"][1]["verifier_alert_status"], "premature")
        self.assertEqual(result["turns"][1]["verifier_recommended_action"], "continue_search")
        self.assertLess(result["turns"][1]["verifier_derived_scores"]["sufficiency"], 0.6)
        self.assertEqual(result["termination_trace"]["latest_verifier_status"], "incomplete")
        self.assertEqual(result["termination_trace"]["verification_turn_count"], 1)

    def test_runner_records_counterfactual_anchor_metadata_for_alert_and_evidence(self):
        self.assertIsNotNone(ADAPTER_MODULE, "saver_agent.adapter module is missing")
        self.assertIsNotNone(ROLLOUT_MODULE, "saver_agent.rollout module is missing")

        config = SaverAgentConfig(
            rollout_trace=RolloutTraceConfig(
                record_observation_content=True,
                record_state_deltas=True,
                record_message_history=True,
            )
        )
        runner = ROLLOUT_MODULE.SaverRolloutRunner(
            environment=SaverVideoInteraction(),
            adapter=ADAPTER_MODULE.TimeSearchRolloutAdapter(),
            max_turns=5,
            config=config,
        )
        policy = ROLLOUT_MODULE.ReplayPolicy(
            [
                '<think>scan</think><tool_call>{"name":"scan_timeline","arguments":{"start_sec":0.0,"end_sec":3.0,"num_frames":2}}</tool_call>',
                '<think>alert</think><tool_call>{"name":"emit_alert","arguments":{"decision":"soft_alert","existence":"anomaly","category":"assault","alert_sec":0.5,"claim":{"existence":"anomaly","category":"assault","earliest_alert_sec":1.0}}}</tool_call>',
                '<think>verify</think><tool_call>{"name":"verify_hypothesis","arguments":{"verification_mode":"soft_alert_check","claim":{"existence":"anomaly","category":"assault","earliest_alert_sec":1.0},"candidate_window_ids":["w0001"],"alert":{"decision":"soft_alert","alert_sec":0.5}}}</tool_call>',
                '<think>finalize</think><tool_call>{"name":"finalize_case","arguments":{"existence":"anomaly","category":"assault"}}</tool_call>',
                '<think>done</think><answer>{"existence":"anomaly","category":"assault"}</answer>',
            ]
        )

        result = runner.run_episode(self.item, policy, initial_state=SaverEnvironmentState())

        verify_turn = result["turns"][2]
        finalize_turn = result["turns"][3]
        self.assertIn("observed_horizon_sec_before", verify_turn)
        self.assertIn("observed_horizon_sec_after", verify_turn)
        self.assertIn("latest_claim_after", verify_turn)
        self.assertIn("latest_alert_after", verify_turn)
        self.assertIn("selected_window_ids_after", verify_turn)
        self.assertIn("selected_evidence_ids_after", verify_turn)
        self.assertIn("counterfactual_anchor_tags", verify_turn)
        self.assertIn("evidence_anchor", verify_turn["counterfactual_anchor_tags"])
        self.assertIn("alert_anchor", verify_turn["counterfactual_anchor_tags"])
        self.assertIn("counterfactual_anchor_tags", finalize_turn)
        self.assertIn("evidence_anchor", finalize_turn["counterfactual_anchor_tags"])
        self.assertIn("counterfactual_anchor_summary", result)
        self.assertGreaterEqual(result["counterfactual_anchor_summary"]["num_alert_anchors"], 1)
        self.assertGreaterEqual(result["counterfactual_anchor_summary"]["num_evidence_anchors"], 1)
        self.assertIn("decision_turn_indices", result)

    def test_runner_records_search_counterfactual_anchor_metadata_for_search_turns(self):
        self.assertIsNotNone(ADAPTER_MODULE, "saver_agent.adapter module is missing")
        self.assertIsNotNone(ROLLOUT_MODULE, "saver_agent.rollout module is missing")

        config = SaverAgentConfig(
            rollout_trace=RolloutTraceConfig(
                record_observation_content=True,
                record_state_deltas=True,
                record_message_history=True,
            )
        )
        runner = ROLLOUT_MODULE.SaverRolloutRunner(
            environment=SaverVideoInteraction(),
            adapter=ADAPTER_MODULE.TimeSearchRolloutAdapter(),
            max_turns=3,
            config=config,
        )
        policy = ROLLOUT_MODULE.ReplayPolicy(
            [
                '<think>scan</think><tool_call>{"name":"scan_timeline","arguments":{"start_sec":0.0,"end_sec":3.0,"num_frames":2}}</tool_call>',
                '<think>finalize</think><tool_call>{"name":"finalize_case","arguments":{"existence":"anomaly","category":"assault"}}</tool_call>',
                '<think>done</think><answer>{"existence":"anomaly","category":"assault"}</answer>',
            ]
        )

        result = runner.run_episode(self.item, policy, initial_state=SaverEnvironmentState())

        search_turn = result["turns"][0]
        self.assertIn("counterfactual_anchor_tags", search_turn)
        self.assertIn("search_anchor", search_turn["counterfactual_anchor_tags"])
        self.assertEqual(search_turn["counterfactual_actual_search_branch"], "use_search")
        self.assertIn("counterfactual_anchor_summary", result)
        self.assertGreaterEqual(result["counterfactual_anchor_summary"]["num_search_anchors"], 1)
        self.assertIn(1, result["counterfactual_anchor_summary"]["search_anchor_turn_indices"])

    def test_runner_stops_at_max_turns_without_answer(self):
        self.assertIsNotNone(ADAPTER_MODULE, "saver_agent.adapter module is missing")
        self.assertIsNotNone(ROLLOUT_MODULE, "saver_agent.rollout module is missing")

        runner = ROLLOUT_MODULE.SaverRolloutRunner(
            environment=SaverVideoInteraction(),
            adapter=ADAPTER_MODULE.TimeSearchRolloutAdapter(),
            max_turns=1,
        )
        policy = ROLLOUT_MODULE.ReplayPolicy(
            [
                '<think>scan</think><tool_call>{"name":"scan_timeline","arguments":{"start_sec":0.0,"end_sec":3.0,"num_frames":2}}</tool_call>',
                '<think>done</think><answer>{"existence":"normal"}</answer>',
            ]
        )

        result = runner.run_episode(self.item, policy, initial_state=SaverEnvironmentState())

        self.assertEqual(result["terminated_reason"], "max_turns")
        self.assertIsNone(result["final_answer"])
        self.assertEqual(result["num_turns"], 1)

    def test_runner_appends_parse_error_observation_for_invalid_response(self):
        self.assertIsNotNone(ADAPTER_MODULE, "saver_agent.adapter module is missing")
        self.assertIsNotNone(ROLLOUT_MODULE, "saver_agent.rollout module is missing")

        runner = ROLLOUT_MODULE.SaverRolloutRunner(
            environment=SaverVideoInteraction(),
            adapter=ADAPTER_MODULE.TimeSearchRolloutAdapter(),
            max_turns=3,
        )
        policy = ROLLOUT_MODULE.ReplayPolicy(
            [
                "I will use scan_timeline to inspect the clip.",
                '<think>scan</think><tool_call>{"name":"scan_timeline","arguments":{"start_sec":0.0,"end_sec":3.0,"num_frames":2}}</tool_call>',
                '<think>done</think><answer>{"existence":"normal","summary":"No anomaly found."}</answer>',
            ]
        )

        result = runner.run_episode(self.item, policy, initial_state=SaverEnvironmentState())

        self.assertEqual(result["terminated_reason"], "answered")
        self.assertEqual(result["turns"][0]["action"], None)
        self.assertEqual(result["messages"][3]["role"], "tool")
        self.assertEqual(result["messages"][3]["name"], "parse_error")
        self.assertIn("invalid", result["turns"][0]["tool_observation_summary"].lower())
        self.assertEqual(result["turns"][1]["tool_name"], "scan_timeline")


class SaverAgentRolloutCliTests(unittest.TestCase):
    def test_cli_replays_responses_and_writes_json_output(self):
        sample = {
            "schema_version": "saver_agent.v1",
            "video_id": "cli_rollout",
            "video_path": "videos/cli_rollout.mp4",
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
            "structured_target": {"existence": "normal"},
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

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.jsonl"
            output_path = Path(tmpdir) / "rollout.json"
            with input_path.open("w", encoding="utf-8") as f:
                f.write(json.dumps(sample) + "\n")

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "run_saver_rollout.py"),
                    "--data",
                    str(input_path),
                    "--data-root",
                    "/dataset/root",
                    "--index",
                    "0",
                    "--output",
                    str(output_path),
                    "--response",
                    '<think>scan</think><tool_call>{"name":"scan_timeline","arguments":{"start_sec":0.0,"end_sec":3.0,"num_frames":2}}</tool_call>',
                    "--response",
                    '<think>done</think><answer>{"existence":"normal","summary":"No anomaly found."}</answer>',
                ],
                check=True,
            )

            with output_path.open("r", encoding="utf-8") as f:
                result = json.load(f)

        self.assertEqual(result["video_id"], "cli_rollout")
        self.assertEqual(result["terminated_reason"], "answered")
        self.assertEqual(result["final_answer"]["existence"], "normal")
        self.assertEqual(result["num_turns"], 2)

    def test_batch_cli_replays_responses_and_writes_jsonl_output(self):
        sample_a = {
            "schema_version": "saver_agent.v1",
            "video_id": "cli_batch_a",
            "video_path": "videos/cli_batch_a.mp4",
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
            "structured_target": {"existence": "normal"},
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
        sample_b = {
            **sample_a,
            "video_id": "cli_batch_b",
            "video_path": "videos/cli_batch_b.mp4",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.jsonl"
            output_path = Path(tmpdir) / "batch_rollouts.jsonl"
            with input_path.open("w", encoding="utf-8") as f:
                f.write(json.dumps(sample_a) + "\n")
                f.write(json.dumps(sample_b) + "\n")

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "batch_run_saver_rollout.py"),
                    "--data",
                    str(input_path),
                    "--data-root",
                    "/dataset/root",
                    "--start-index",
                    "0",
                    "--count",
                    "2",
                    "--output",
                    str(output_path),
                    "--response",
                    '<think>scan</think><tool_call>{"name":"scan_timeline","arguments":{"start_sec":0.0,"end_sec":3.0,"num_frames":2}}</tool_call>',
                    "--response",
                    '<think>done</think><answer>{"existence":"normal","summary":"No anomaly found."}</answer>',
                ],
                check=True,
            )

            results = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(len(results), 2)
        self.assertEqual([result["video_id"] for result in results], ["cli_batch_a", "cli_batch_b"])
        self.assertTrue(all(result["terminated_reason"] == "answered" for result in results))
        self.assertTrue(all(result["final_answer"]["existence"] == "normal" for result in results))
        self.assertTrue(all(result["num_turns"] == 2 for result in results))


if __name__ == "__main__":
    unittest.main()
