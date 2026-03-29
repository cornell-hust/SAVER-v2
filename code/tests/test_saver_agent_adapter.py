import json
import sys
import unittest
from pathlib import Path


ROOT = Path("/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from saver_agent.adapter import TimeSearchRolloutAdapter


class SaverAgentAdapterTests(unittest.TestCase):
    def test_verify_hypothesis_finalize_hint_does_not_echo_structured_claim_json(self):
        adapter = TimeSearchRolloutAdapter()
        tool_message = {
            "role": "tool",
            "name": "verify_hypothesis",
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "verification_mode": "final_check",
                            "recommended_action": "finalize",
                            "claim": {
                                "existence": "anomaly",
                                "category": "robbery",
                                "severity": 4,
                            },
                        },
                        ensure_ascii=False,
                    ),
                }
            ],
        }

        adapted = adapter.adapt_tool_observation(
            tool_message,
            {"question": "Determine whether the video contains an actionable anomaly."},
        )

        final_hint = [
            item["text"]
            for item in adapted["content"]
            if item.get("type") == "text"
        ][-1]
        self.assertIn("Call finalize_case next", final_hint)
        self.assertNotIn("supported claim", final_hint)
        self.assertNotIn("severity", final_hint)
        self.assertNotIn("{", final_hint)


if __name__ == "__main__":
    unittest.main()
