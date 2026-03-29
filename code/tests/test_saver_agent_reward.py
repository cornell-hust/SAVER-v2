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
        self.assertGreater(complete["components"]["verification_reward"], incomplete["components"]["verification_reward"])

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


if __name__ == "__main__":
    unittest.main()
