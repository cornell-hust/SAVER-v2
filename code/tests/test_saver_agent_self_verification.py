import sys
import unittest
from pathlib import Path


ROOT = Path("/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saver_agent.self_verification import (
    build_self_verification_tool_schema,
    parse_self_verification_payload,
)


class SaverAgentSelfVerificationTests(unittest.TestCase):
    def test_build_self_verification_tool_schema_excludes_legacy_external_verifier_fields(self):
        schema = build_self_verification_tool_schema()
        properties = schema["properties"]

        self.assertIn("verification_decision", properties)
        self.assertIn("recommended_action", properties)
        self.assertIn("selected_window_ids", properties)
        self.assertIn("selected_evidence_moment_ids", properties)
        self.assertNotIn("allow_external_verifier_fallback", properties)
        self.assertNotIn("verifier_backend", properties)
        self.assertNotIn("primary_status", properties)
        self.assertNotIn("alert_status", properties)
        self.assertNotIn("failure_reasons", properties)
        self.assertNotIn("selected_evidence_ids", properties)
        self.assertNotIn("candidate_window_ids", properties)
        self.assertNotIn("verified_window_ids", properties)
        self.assertNotIn("best_effort_window_ids", properties)

    def test_build_self_verification_tool_schema_requires_compact_verdict_payload(self):
        schema = build_self_verification_tool_schema()

        self.assertEqual(
            schema["required"],
            [
                "verification_mode",
                "selected_window_ids",
                "verification_decision",
                "recommended_action",
                "sufficiency_score",
                "necessity_score",
                "alertability_score",
                "counterfactual_faithfulness",
            ],
        )
        self.assertEqual(schema["properties"]["selected_window_ids"]["minItems"], 1)

    def test_parse_self_verification_payload_accepts_complete_case(self):
        payload = {
            "claim": {"existence": "anomaly", "category": "assault"},
            "selected_window_ids": ["w0001", "w0002"],
            "selected_evidence_ids": ["e0001", "e0002"],
            "selected_evidence_moment_ids": ["ev1", "ev2"],
            "sufficiency_score": 0.82,
            "necessity_score": 0.63,
            "alertability_score": 0.74,
            "counterfactual_faithfulness": 0.71,
            "verification_decision": "sufficient",
            "recommended_action": "finalize",
            "rationale": "The selected evidence covers the decisive event and remains necessary.",
        }

        parsed = parse_self_verification_payload(payload)

        self.assertEqual(parsed["verification_decision"], "sufficient")
        self.assertEqual(parsed["recommended_action"], "finalize")
        self.assertEqual(parsed["primary_status"], "complete")
        self.assertEqual(parsed["verified_window_ids"], ["w0001", "w0002"])
        self.assertAlmostEqual(parsed["derived_scores"]["sufficiency"], 0.82, places=6)
        self.assertAlmostEqual(parsed["derived_scores"]["necessity"], 0.63, places=6)

    def test_parse_self_verification_payload_defaults_scores_and_normalizes_lists(self):
        parsed = parse_self_verification_payload(
            {
                "claim": {"existence": "anomaly", "category": "fire"},
                "selected_window_ids": "w0003",
                "verification_decision": "insufficient",
                "recommended_action": "continue_search",
            }
        )

        self.assertEqual(parsed["verified_window_ids"], ["w0003"])
        self.assertEqual(parsed["primary_status"], "incomplete")
        self.assertEqual(parsed["alert_status"], "not_applicable")
        self.assertEqual(parsed["derived_scores"]["sufficiency"], 0.0)
        self.assertEqual(parsed["derived_scores"]["necessity"], 0.0)

    def test_parse_self_verification_payload_accepts_legacy_verdict_fields_and_normalizes_mode_and_category(self):
        parsed = parse_self_verification_payload(
            {
                "verification_mode": "normal_check",
                "claim": {"existence": "anomaly", "category": "vehicle_collision"},
                "verified_window_ids": ["w0007"],
                "primary_status": "complete",
                "alert_status": "not_applicable",
                "derived_scores": {
                    "sufficiency": 0.73,
                    "necessity": 0.61,
                    "alertability": 0.0,
                    "counterfactual_faithfulness": 0.52,
                },
            }
        )

        self.assertEqual(parsed["verification_mode"], "final_check")
        self.assertEqual(parsed["verification_decision"], "sufficient")
        self.assertEqual(parsed["recommended_action"], "finalize")
        self.assertEqual(parsed["claim"]["category"], "traffic_accident")
        self.assertEqual(parsed["verified_window_ids"], ["w0007"])
        self.assertAlmostEqual(parsed["derived_scores"]["sufficiency"], 0.73, places=6)


if __name__ == "__main__":
    unittest.main()
