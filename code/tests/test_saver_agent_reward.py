import sys
import unittest
from pathlib import Path


ROOT = Path("/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saver_agent.reward import score_rollout_trace


def _make_rollout(primary_status: str, alert_status: str, sufficiency: float, necessity: float):
    return {
        "terminated_reason": "answered",
        "num_turns": 3,
        "final_answer": {"existence": "anomaly"},
        "state": {"finalized_case": {"existence": "anomaly"}},
        "turns": [
            {
                "tool_name": "verify_hypothesis",
                "verifier_primary_status": primary_status,
                "verifier_alert_status": alert_status,
                "verifier_recommended_action": "finalize" if primary_status == "complete" else "continue_search",
                "verifier_derived_scores": {
                    "sufficiency": sufficiency,
                    "necessity": necessity,
                    "consistency": 0.9,
                    "alertability": 0.8 if alert_status == "justified" else 0.3,
                    "counterfactual_faithfulness": (sufficiency + necessity) / 2.0,
                },
            }
        ],
    }


class SaverAgentRewardTests(unittest.TestCase):
    def test_reward_prefers_complete_verdict_over_incomplete(self):
        complete = score_rollout_trace(_make_rollout("complete", "justified", 0.8, 0.6))
        incomplete = score_rollout_trace(_make_rollout("incomplete", "premature", 0.3, 0.1))

        self.assertGreater(complete["total_reward"], incomplete["total_reward"])
        self.assertGreater(
            complete["components"]["self_verification_quality_reward"],
            incomplete["components"]["self_verification_quality_reward"],
        )

    def test_reward_penalizes_premature_alerts(self):
        justified = score_rollout_trace(_make_rollout("complete", "justified", 0.8, 0.6))
        premature = score_rollout_trace(_make_rollout("complete", "premature", 0.8, 0.6))

        self.assertGreater(justified["components"]["alert_reward"], premature["components"]["alert_reward"])
        self.assertGreater(justified["total_reward"], premature["total_reward"])

    def test_reward_penalizes_misaligned_verdict_most_strongly(self):
        redundant = score_rollout_trace(_make_rollout("redundant", "justified", 0.75, 0.1))
        misaligned = score_rollout_trace(_make_rollout("misaligned", "premature", 0.2, 0.0))

        self.assertLess(misaligned["components"]["verification_reward"], redundant["components"]["verification_reward"])
        self.assertLess(misaligned["total_reward"], redundant["total_reward"])

    def test_reward_marks_offline_reference_conditioned_fallback_explicitly(self):
        reward = score_rollout_trace(
            {
                "terminated_reason": "answered",
                "num_turns": 1,
                "final_answer": {"existence": "anomaly"},
                "state": {"finalized_case": {"existence": "anomaly"}},
                "offline_verifier": {
                    "primary_status": "complete",
                    "alert_status": "justified",
                    "derived_scores": {
                        "sufficiency": 0.8,
                        "necessity": 0.6,
                        "consistency": 0.9,
                        "alertability": 0.8,
                        "counterfactual_faithfulness": 0.7,
                    },
                },
            }
        )

        self.assertEqual(reward["verifier_source"], "offline_reference_conditioned")
        self.assertTrue(reward["uses_reference_conditioned_verifier"])

    def test_reward_uses_teacher_judge_signal_when_present(self):
        aligned = score_rollout_trace(
            {
                **_make_rollout("complete", "justified", 0.8, 0.6),
                "turns": [
                    {
                        **_make_rollout("complete", "justified", 0.8, 0.6)["turns"][0],
                        "self_verification_decision": "sufficient",
                        "self_verification_scores": {
                            "sufficiency": 0.8,
                            "necessity": 0.6,
                            "alertability": 0.8,
                            "counterfactual_faithfulness": 0.7,
                        },
                        "teacher_judge_scores": {
                            "sufficiency": 0.82,
                            "necessity": 0.58,
                            "alertability": 0.78,
                            "counterfactual_faithfulness": 0.72,
                        },
                        "teacher_judge_decision": "sufficient",
                    }
                ],
            }
        )
        misaligned = score_rollout_trace(
            {
                **_make_rollout("complete", "justified", 0.8, 0.6),
                "turns": [
                    {
                        **_make_rollout("complete", "justified", 0.8, 0.6)["turns"][0],
                        "self_verification_decision": "sufficient",
                        "self_verification_scores": {
                            "sufficiency": 0.8,
                            "necessity": 0.6,
                            "alertability": 0.8,
                            "counterfactual_faithfulness": 0.7,
                        },
                        "teacher_judge_scores": {
                            "sufficiency": 0.2,
                            "necessity": 0.1,
                            "alertability": 0.2,
                            "counterfactual_faithfulness": 0.2,
                        },
                        "teacher_judge_decision": "misaligned",
                    }
                ],
            }
        )

        self.assertIn("teacher_agreement_reward", aligned["components"])
        self.assertGreater(aligned["components"]["teacher_agreement_reward"], 0.0)
        self.assertLess(misaligned["components"]["teacher_agreement_reward"], 0.0)
        self.assertGreater(aligned["total_reward"], misaligned["total_reward"])

    def test_reward_does_not_give_decision_credit_for_malformed_answer_text(self):
        reward = score_rollout_trace(
            {
                "terminated_reason": "answered",
                "num_turns": 2,
                "final_answer": "not-json",
                "state": {"finalized_case": None},
                "turns": [],
            }
        )

        self.assertEqual(reward["components"]["decision_reward"], 0.0)

    def test_reward_penalizes_protocol_shortcuts_without_verify_and_finalize(self):
        reward = score_rollout_trace(
            {
                "terminated_reason": "answered",
                "num_turns": 1,
                "final_answer": {"existence": "anomaly", "category": "assault"},
                "state": {"finalized_case": None},
                "turns": [
                    {
                        "step_index": 1,
                        "action": "answer",
                        "valid_action": True,
                    }
                ],
            }
        )

        self.assertEqual(reward["components"]["decision_reward"], 0.0)
        self.assertLess(reward["components"]["protocol_reward"], 0.0)
        self.assertLess(reward["total_reward"], 0.0)

    def test_reward_penalizes_invalid_attempts(self):
        clean_reward = score_rollout_trace(_make_rollout("complete", "justified", 0.8, 0.6))
        retry_reward = score_rollout_trace(
            {
                **_make_rollout("complete", "justified", 0.8, 0.6),
                "num_invalid_attempts": 3,
                "invalid_attempts": [
                    {"step_index": 2, "action": None, "valid_action": False},
                    {"step_index": 2, "action": None, "valid_action": False},
                    {"step_index": 3, "action": None, "valid_action": False},
                ],
            }
        )

        self.assertLess(retry_reward["components"]["invalid_attempt_penalty"], 0.0)
        self.assertLess(retry_reward["total_reward"], clean_reward["total_reward"])


if __name__ == "__main__":
    unittest.main()
