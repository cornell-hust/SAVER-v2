import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


ROOT = Path("/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saver_agent.environment import SaverEnvironmentState, SaverVideoInteraction
from saver_agent.environment import parse_actions_and_contents


class SaverAgentEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.env = SaverVideoInteraction()
        self.multimodal_cache = {
            "video": torch.zeros(8, 3, 2, 2),
            "embedding": None,
            "fps": 1.0,
            "duration": 8.0,
            "question": "Determine whether an anomaly exists.",
            "tool_io": {"finalize_case_schema": {"type": "object", "required": ["existence"]}},
        }

    def test_execute_predictions_runs_tool_call_and_keeps_episode_active(self):
        next_obs, dones, valid_actions, is_search, states = self.env.execute_predictions(
            [
                '<think>search</think><tool_call>{"name":"scan_timeline","arguments":{"start_sec":0.0,"end_sec":7.0,"num_frames":3}}</tool_call>'
            ],
            [self.multimodal_cache],
            [SaverEnvironmentState()],
            [True],
        )

        self.assertEqual(dones, [0])
        self.assertEqual(valid_actions, [1])
        self.assertEqual(is_search, [1])
        self.assertEqual(next_obs[0]["name"], "scan_timeline")
        self.assertEqual(len(states[0].visited_windows), 1)

    def test_execute_predictions_accepts_answer_and_marks_done(self):
        next_obs, dones, valid_actions, is_search, states = self.env.execute_predictions(
            ['<think>done</think><answer>{"existence":"normal"}</answer>'],
            [self.multimodal_cache],
            [SaverEnvironmentState()],
            [True],
        )

        self.assertEqual(next_obs, [None])
        self.assertEqual(dones, [1])
        self.assertEqual(valid_actions, [1])
        self.assertEqual(is_search, [0])
        self.assertEqual(len(states), 1)

    def test_execute_predictions_rejects_malformed_answer_and_keeps_episode_active(self):
        next_obs, dones, valid_actions, is_search, states = self.env.execute_predictions(
            ['<think>done</think><answer>not-json</answer>'],
            [self.multimodal_cache],
            [SaverEnvironmentState()],
            [True],
        )

        self.assertEqual(dones, [0])
        self.assertEqual(valid_actions, [0])
        self.assertEqual(is_search, [0])
        self.assertEqual(next_obs[0]["name"], "parse_error")
        self.assertIn("<answer>", next_obs[0]["content"][0]["text"])
        self.assertEqual(len(states), 1)

    def test_execute_predictions_invalid_tool_format_returns_parse_error_and_keeps_episode_active(self):
        next_obs, dones, valid_actions, is_search, states = self.env.execute_predictions(
            ["I will use scan_timeline to inspect the clip."],
            [self.multimodal_cache],
            [SaverEnvironmentState()],
            [True],
        )

        self.assertEqual(dones, [0])
        self.assertEqual(valid_actions, [0])
        self.assertEqual(is_search, [0])
        self.assertEqual(next_obs[0]["name"], "parse_error")
        self.assertIn("<tool_call>", next_obs[0]["content"][0]["text"])
        self.assertIn("verify_hypothesis", next_obs[0]["content"][0]["text"])
        self.assertEqual(len(states[0].visited_windows), 0)

    def test_parse_actions_accepts_unquoted_tool_name_mapping_payload(self):
        actions, contents = parse_actions_and_contents(
            ['<tool_call>{scan_timeline: {"start_time": "0.000s", "end_time": "7.000s"}}</tool_call>']
        )

        self.assertEqual(actions, ["tool_call"])
        self.assertEqual(contents[0]["function"]["name"], "scan_timeline")
        self.assertEqual(contents[0]["function"]["arguments"]["start_sec"], 0.0)
        self.assertEqual(contents[0]["function"]["arguments"]["end_sec"], 7.0)

    def test_parse_actions_prefers_terminal_answer_outside_think_block(self):
        actions, contents = parse_actions_and_contents(
            [
                (
                    '<think>first draft <answer>{"existence":"normal"}</answer></think>'
                    '<answer>{"existence":"anomaly","category":"assault"}</answer>'
                )
            ]
        )

        self.assertEqual(actions, ["answer"])
        self.assertEqual(contents[0], '{"existence":"anomaly","category":"assault"}')

    def test_parse_actions_rejects_malformed_answer_payload(self):
        actions, contents = parse_actions_and_contents(['<answer>not-json</answer>'])

        self.assertEqual(actions, ["invalid_answer"])
        self.assertEqual(contents[0], "not-json")

    def test_invalid_finalize_case_retry_message_mentions_required_schema_fields(self):
        multimodal_cache = {
            **self.multimodal_cache,
            "tool_io": {"finalize_case_schema": {"type": "object", "required": ["existence", "summary"]}},
        }

        next_obs, dones, valid_actions, is_search, _ = self.env.execute_predictions(
            [
                '<think>finalize</think><tool_call>{"name":"finalize_case","arguments":{"existence":"anomaly"}}</tool_call>'
            ],
            [multimodal_cache],
            [SaverEnvironmentState()],
            [True],
        )

        self.assertEqual(dones, [0])
        self.assertEqual(valid_actions, [0])
        self.assertEqual(is_search, [0])
        self.assertEqual(next_obs[0]["name"], "parse_error")
        retry_text = next_obs[0]["content"][0]["text"]
        self.assertIn("finalize_case", retry_text)
        self.assertIn("summary", retry_text)

    def test_invalid_verify_retry_message_uses_current_evidence_window_ids_without_hardcoded_assault_claim(self):
        next_obs, _, _, _, states = self.env.execute_predictions(
            [
                '<think>scan</think><tool_call>{"name":"scan_timeline","arguments":{"start_sec":0.0,"end_sec":7.0,"num_frames":3}}</tool_call>'
            ],
            [self.multimodal_cache],
            [SaverEnvironmentState()],
            [True],
        )
        self.assertEqual(next_obs[0]["name"], "scan_timeline")

        next_obs, _, _, _, states = self.env.execute_predictions(
            [
                '<think>seek</think><tool_call>{"name":"seek_evidence","arguments":{"query":"look for anomaly","start_sec":1.0,"end_sec":4.0,"num_frames":2}}</tool_call>'
            ],
            [self.multimodal_cache],
            states,
            [True],
        )
        self.assertEqual(next_obs[0]["name"], "seek_evidence")

        next_obs, dones, valid_actions, is_search, _ = self.env.execute_predictions(
            [
                (
                    '<think>verify</think><tool_call>{"name":"verify_hypothesis","arguments":'
                    '{"verification_mode":"final_check","verification_decision":"insufficient",'
                    '"recommended_action":"continue_search","sufficiency_score":0.2,"necessity_score":0.1,'
                    '"alertability_score":0.0,"counterfactual_faithfulness":0.3}}</tool_call>'
                )
            ],
            [self.multimodal_cache],
            states,
            [True],
        )

        self.assertEqual(dones, [0])
        self.assertEqual(valid_actions, [0])
        self.assertEqual(is_search, [0])
        self.assertEqual(next_obs[0]["name"], "parse_error")
        retry_text = next_obs[0]["content"][0]["text"]
        self.assertIn('"selected_window_ids":["w0002"]', retry_text)
        self.assertNotIn('"category":"assault"', retry_text)
        self.assertNotIn('"selected_window_ids":["w0001"]', retry_text)

    def test_execute_predictions_reraises_internal_tool_failures_instead_of_masking_them_as_parse_errors(self):
        with patch("saver_agent.environment.execute_tool_call", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                self.env.execute_predictions(
                    [
                        '<think>search</think><tool_call>{"name":"scan_timeline","arguments":{"start_sec":0.0,"end_sec":7.0,"num_frames":3}}</tool_call>'
                    ],
                    [self.multimodal_cache],
                    [SaverEnvironmentState()],
                    [True],
                )


if __name__ == "__main__":
    unittest.main()
