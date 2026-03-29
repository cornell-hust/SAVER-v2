import sys
import unittest
from pathlib import Path


ROOT = Path("/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saver_agent.schema import SaverEnvironmentState
from saver_agent.verifier import (
    run_counterfactual_verifier,
    score_alert_counterfactual_group,
    score_evidence_counterfactual_group,
    score_search_counterfactual_group,
)


class _FakeQwenVerifierRuntime:
    def __init__(self, view_scores):
        self.view_scores = view_scores
        self.calls = []

    def score_views(self, *, views, claim, verification_mode, question):
        self.calls.append(
            {
                "views": views,
                "claim": claim,
                "verification_mode": verification_mode,
                "question": question,
            }
        )
        return self.view_scores


def _make_multimodal_cache():
    return {
        "fps": 1.0,
        "duration": 10.0,
        "question": "Determine whether the video contains an actionable anomaly.",
        "structured_target": {
            "existence": "anomaly",
            "category": "assault",
            "severity": 4,
            "anomaly_interval_sec": [1.0, 8.0],
            "precursor_interval_sec": [0.0, 1.0],
            "earliest_alert_sec": 1.0,
            "counterfactual_type": "remove_actor_interaction",
        },
        "tool_io": {
            "oracle_windows_sec": [
                {"moment_id": "ev1", "role": "trigger", "window": [1.0, 4.0], "description": "trigger"},
                {"moment_id": "ev2", "role": "peak_action", "window": [4.0, 6.0], "description": "peak"},
                {"moment_id": "ev3", "role": "confirmation", "window": [6.0, 8.0], "description": "confirm"},
            ]
        },
    }


def _make_state():
    entries = [
        {
            "window_id": "w0001",
            "evidence_id": "e0001",
            "moment_id": "ev1",
            "kind": "evidence",
            "query": "possible assault",
            "start_sec": 1.0,
            "end_sec": 4.0,
            "selected_timestamps": [1.0, 2.0, 3.0],
            "selected_frame_count": 3,
        },
        {
            "window_id": "w0002",
            "evidence_id": "e0002",
            "moment_id": "ev2",
            "kind": "evidence",
            "query": "possible assault",
            "start_sec": 4.0,
            "end_sec": 6.0,
            "selected_timestamps": [4.0, 5.0],
            "selected_frame_count": 2,
        },
        {
            "window_id": "w0003",
            "evidence_id": "e0003",
            "moment_id": "ev3",
            "kind": "evidence",
            "query": "possible assault",
            "start_sec": 6.0,
            "end_sec": 8.0,
            "selected_timestamps": [6.0, 7.0],
            "selected_frame_count": 2,
        },
    ]
    return SaverEnvironmentState(
        visited_windows=list(entries),
        evidence_ledger=list(entries),
        next_evidence_id=4,
        next_window_id=4,
    )


def _make_partial_state(window_ids):
    full_state = _make_state()
    selected = [entry for entry in full_state.evidence_ledger if entry["window_id"] in set(window_ids)]
    return SaverEnvironmentState(
        visited_windows=list(selected),
        evidence_ledger=list(selected),
        next_evidence_id=len(selected) + 1,
        next_window_id=len(selected) + 1,
    )


def _make_normal_multimodal_cache():
    return {
        "fps": 1.0,
        "duration": 9.0,
        "question": "Determine whether the video is normal and whether more search is needed.",
        "structured_target": {},
        "tool_io": {},
    }


def _make_normal_state():
    entries = [
        {
            "window_id": "w0001",
            "evidence_id": "e0001",
            "moment_id": None,
            "kind": "evidence",
            "query": "normal check segment 1",
            "start_sec": 0.0,
            "end_sec": 3.0,
            "selected_timestamps": [0.0, 1.0, 2.0],
            "selected_frame_count": 3,
        },
        {
            "window_id": "w0002",
            "evidence_id": "e0002",
            "moment_id": None,
            "kind": "evidence",
            "query": "normal check segment 2",
            "start_sec": 3.0,
            "end_sec": 6.0,
            "selected_timestamps": [3.0, 4.0, 5.0],
            "selected_frame_count": 3,
        },
        {
            "window_id": "w0003",
            "evidence_id": "e0003",
            "moment_id": None,
            "kind": "evidence",
            "query": "normal check segment 3",
            "start_sec": 6.0,
            "end_sec": 9.0,
            "selected_timestamps": [6.0, 7.0, 8.0],
            "selected_frame_count": 3,
        },
    ]
    return SaverEnvironmentState(
        visited_windows=list(entries),
        evidence_ledger=list(entries),
        next_evidence_id=4,
        next_window_id=4,
    )


class SaverAgentVerifierTests(unittest.TestCase):
    def test_counterfactual_verifier_resolves_candidate_evidence_moment_ids_to_runtime_subset(self):
        verdict = run_counterfactual_verifier(
            state=_make_state(),
            multimodal_cache=_make_multimodal_cache(),
            verification_mode="final_check",
            claim={"existence": "anomaly", "category": "assault"},
            candidate_evidence_moment_ids=["ev2"],
            backend="heuristic",
            use_reference_supervision=False,
        )

        self.assertEqual(verdict["verified_window_ids"], ["w0002"])
        self.assertEqual(verdict["candidate_evidence_moment_ids"], ["ev2"])

    def test_counterfactual_verifier_can_disable_reference_supervision_for_online_use(self):
        verdict = run_counterfactual_verifier(
            state=_make_state(),
            multimodal_cache=_make_multimodal_cache(),
            verification_mode="final_check",
            claim={"existence": "anomaly", "category": "assault"},
            candidate_window_ids=["w0001"],
            backend="heuristic",
            use_reference_supervision=False,
        )

        self.assertEqual(verdict["claim"], {"existence": "anomaly", "category": "assault"})
        self.assertNotIn("severity", verdict["claim"])
        self.assertNotIn("anomaly_interval_sec", verdict["claim"])
        self.assertNotIn("counterfactual_type", verdict["claim"])

    def test_counterfactual_verifier_marks_complete_when_selected_subset_is_sufficient_and_necessary(self):
        verdict = run_counterfactual_verifier(
            state=_make_state(),
            multimodal_cache=_make_multimodal_cache(),
            verification_mode="final_check",
            claim={"existence": "anomaly", "category": "assault", "earliest_alert_sec": 1.0},
            candidate_window_ids=["w0001", "w0002"],
            alert={"decision": "hard_alert", "alert_sec": 1.2},
            backend="heuristic",
        )

        self.assertEqual(verdict["primary_status"], "complete")
        self.assertEqual(verdict["alert_status"], "justified")
        self.assertEqual(verdict["recommended_action"], "finalize")
        self.assertGreaterEqual(verdict["derived_scores"]["sufficiency"], 0.60)
        self.assertGreaterEqual(verdict["derived_scores"]["necessity"], 0.20)

    def test_counterfactual_verifier_marks_incomplete_when_subset_misses_decisive_support(self):
        verdict = run_counterfactual_verifier(
            state=_make_state(),
            multimodal_cache=_make_multimodal_cache(),
            verification_mode="soft_alert_check",
            claim={"existence": "anomaly", "category": "assault", "earliest_alert_sec": 1.0},
            candidate_window_ids=["w0003"],
            alert={"decision": "soft_alert", "alert_sec": 1.0},
            backend="heuristic",
        )

        self.assertEqual(verdict["primary_status"], "incomplete")
        self.assertEqual(verdict["recommended_action"], "continue_search")
        self.assertLess(verdict["derived_scores"]["sufficiency"], 0.55)

    def test_counterfactual_verifier_marks_redundant_when_drop_view_remains_strong(self):
        verdict = run_counterfactual_verifier(
            state=_make_state(),
            multimodal_cache=_make_multimodal_cache(),
            verification_mode="final_check",
            claim={"existence": "anomaly", "category": "assault", "earliest_alert_sec": 1.0},
            candidate_window_ids=["w0001"],
            alert={"decision": "hard_alert", "alert_sec": 1.5},
            backend="heuristic",
        )

        self.assertEqual(verdict["primary_status"], "redundant")
        self.assertEqual(verdict["recommended_action"], "refine_evidence")
        self.assertLess(verdict["derived_scores"]["necessity"], 0.20)

    def test_counterfactual_verifier_marks_alert_as_premature_when_prefix_is_insufficient(self):
        verdict = run_counterfactual_verifier(
            state=_make_state(),
            multimodal_cache=_make_multimodal_cache(),
            verification_mode="soft_alert_check",
            claim={"existence": "anomaly", "category": "assault", "earliest_alert_sec": 1.0},
            candidate_window_ids=["w0001", "w0002"],
            alert={"decision": "soft_alert", "alert_sec": 0.4},
            backend="heuristic",
        )

        self.assertEqual(verdict["alert_status"], "premature")
        self.assertLess(verdict["derived_scores"]["alertability"], 0.65)

    def test_counterfactual_verifier_marks_normal_claim_complete_after_broad_search_without_reference_targets(self):
        verdict = run_counterfactual_verifier(
            state=_make_normal_state(),
            multimodal_cache=_make_normal_multimodal_cache(),
            verification_mode="final_check",
            claim={"existence": "normal", "category": "normal"},
            candidate_window_ids=["w0001", "w0002", "w0003"],
            backend="heuristic",
            use_reference_supervision=False,
        )

        self.assertEqual(verdict["primary_status"], "complete")
        self.assertEqual(verdict["alert_status"], "not_applicable")
        self.assertEqual(verdict["recommended_action"], "finalize")
        self.assertGreaterEqual(verdict["derived_scores"]["sufficiency"], 0.60)

    def test_counterfactual_verifier_uses_qwen_runtime_when_requested(self):
        qwen_scores = {
            "full": {
                "exist_support": 0.95,
                "category_support": 0.95,
                "temporal_support": 0.95,
                "precursor_support": 0.60,
                "alert_support": 0.95,
                "counterfactual_support": 0.90,
                "overall_support": 0.92,
            },
            "keep": {
                "exist_support": 0.90,
                "category_support": 0.90,
                "temporal_support": 0.90,
                "precursor_support": 0.40,
                "alert_support": 0.90,
                "counterfactual_support": 0.85,
                "overall_support": 0.86,
            },
            "drop": {
                "exist_support": 0.10,
                "category_support": 0.10,
                "temporal_support": 0.10,
                "precursor_support": 0.10,
                "alert_support": 0.10,
                "counterfactual_support": 0.10,
                "overall_support": 0.10,
            },
            "alert_prefix": {
                "exist_support": 0.85,
                "category_support": 0.85,
                "temporal_support": 0.80,
                "precursor_support": 0.55,
                "alert_support": 0.88,
                "counterfactual_support": 0.82,
                "overall_support": 0.82,
            },
        }
        runtime = _FakeQwenVerifierRuntime(qwen_scores)
        multimodal_cache = _make_multimodal_cache()
        multimodal_cache["verifier_runtime"] = runtime

        verdict = run_counterfactual_verifier(
            state=_make_state(),
            multimodal_cache=multimodal_cache,
            verification_mode="final_check",
            claim={"existence": "anomaly", "category": "assault", "earliest_alert_sec": 1.0},
            candidate_window_ids=["w0003"],
            alert={"decision": "hard_alert", "alert_sec": 1.2},
            backend="qwen_self_verifier",
        )

        self.assertEqual(verdict["verifier_backend"], "qwen_self_verifier")
        self.assertEqual(verdict["primary_status"], "complete")
        self.assertEqual(verdict["alert_status"], "justified")
        self.assertEqual(len(runtime.calls), 1)

    def test_counterfactual_verifier_hybrid_backend_fuses_heuristic_and_qwen_scores(self):
        qwen_scores = {
            "full": {
                "exist_support": 1.0,
                "category_support": 1.0,
                "temporal_support": 1.0,
                "precursor_support": 1.0,
                "alert_support": 1.0,
                "counterfactual_support": 1.0,
                "overall_support": 1.0,
            },
            "keep": {
                "exist_support": 1.0,
                "category_support": 1.0,
                "temporal_support": 1.0,
                "precursor_support": 1.0,
                "alert_support": 1.0,
                "counterfactual_support": 1.0,
                "overall_support": 1.0,
            },
            "drop": {
                "exist_support": 0.0,
                "category_support": 0.0,
                "temporal_support": 0.0,
                "precursor_support": 0.0,
                "alert_support": 0.0,
                "counterfactual_support": 0.0,
                "overall_support": 0.0,
            },
            "alert_prefix": {
                "exist_support": 1.0,
                "category_support": 1.0,
                "temporal_support": 1.0,
                "precursor_support": 1.0,
                "alert_support": 1.0,
                "counterfactual_support": 1.0,
                "overall_support": 1.0,
            },
        }
        multimodal_cache = _make_multimodal_cache()
        multimodal_cache["verifier_runtime"] = _FakeQwenVerifierRuntime(qwen_scores)
        multimodal_cache["verifier_hybrid_alpha"] = 0.5

        heuristic_verdict = run_counterfactual_verifier(
            state=_make_state(),
            multimodal_cache=_make_multimodal_cache(),
            verification_mode="soft_alert_check",
            claim={"existence": "anomaly", "category": "assault", "earliest_alert_sec": 1.0},
            candidate_window_ids=["w0003"],
            alert={"decision": "soft_alert", "alert_sec": 1.0},
            backend="heuristic",
        )
        hybrid_verdict = run_counterfactual_verifier(
            state=_make_state(),
            multimodal_cache=multimodal_cache,
            verification_mode="soft_alert_check",
            claim={"existence": "anomaly", "category": "assault", "earliest_alert_sec": 1.0},
            candidate_window_ids=["w0003"],
            alert={"decision": "soft_alert", "alert_sec": 1.0},
            backend="hybrid",
        )

        expected_sufficiency = round((heuristic_verdict["derived_scores"]["sufficiency"] + 1.0) / 2.0, 6)
        self.assertEqual(hybrid_verdict["verifier_backend"], "hybrid")
        self.assertAlmostEqual(hybrid_verdict["derived_scores"]["sufficiency"], expected_sufficiency, places=6)

    def test_score_alert_counterfactual_group_returns_local_advantages_for_branch_comparison(self):
        multimodal_cache = _make_multimodal_cache()
        branches = [
            {
                "branch_type": "alert_now",
                "anchor_turn_index": 2,
                "state": _make_partial_state(["w0001"]),
                "claim": {"existence": "anomaly", "category": "assault", "earliest_alert_sec": 1.0},
                "alert": {"decision": "soft_alert", "alert_sec": 0.4},
            },
            {
                "branch_type": "defer_to_next_decision",
                "anchor_turn_index": 2,
                "state": _make_partial_state(["w0001", "w0002"]),
                "claim": {"existence": "anomaly", "category": "assault", "earliest_alert_sec": 1.0},
                "alert": {"decision": "hard_alert", "alert_sec": 1.2},
            },
            {
                "branch_type": "defer_to_final",
                "anchor_turn_index": 2,
                "state": _make_state(),
                "claim": {"existence": "anomaly", "category": "assault", "earliest_alert_sec": 1.0},
                "alert": {"decision": "hard_alert", "alert_sec": 1.8},
            },
        ]

        records = score_alert_counterfactual_group(
            branches=branches,
            multimodal_cache=multimodal_cache,
            verifier_backend="heuristic",
            use_reference_supervision=True,
        )

        self.assertEqual([record["branch_type"] for record in records], ["alert_now", "defer_to_next_decision", "defer_to_final"])
        self.assertTrue(all("branch_reward" in record for record in records))
        self.assertTrue(all("local_advantage" in record for record in records))
        self.assertAlmostEqual(sum(record["local_advantage"] for record in records), 0.0, places=5)
        self.assertLess(records[0]["local_advantage"], records[1]["local_advantage"])

    def test_score_evidence_counterfactual_group_scores_keep_drop_and_minimal_subset(self):
        records = score_evidence_counterfactual_group(
            state=_make_state(),
            multimodal_cache=_make_multimodal_cache(),
            claim={"existence": "anomaly", "category": "assault", "earliest_alert_sec": 1.0},
            selected_window_ids=["w0001", "w0002"],
            anchor_turn_index=3,
            alert={"decision": "hard_alert", "alert_sec": 1.2},
            verifier_backend="heuristic",
            use_reference_supervision=True,
            minimal_subset_window_ids=["w0002"],
        )

        self.assertEqual(
            [record["branch_type"] for record in records],
            ["full_ledger", "keep_selected", "drop_selected", "minimal_subset"],
        )
        self.assertTrue(all("branch_reward_components" in record for record in records))
        self.assertTrue(all("verifier_verdict" in record for record in records))
        self.assertAlmostEqual(sum(record["local_advantage"] for record in records), 0.0, places=5)
        keep_record = next(record for record in records if record["branch_type"] == "keep_selected")
        drop_record = next(record for record in records if record["branch_type"] == "drop_selected")
        self.assertGreater(keep_record["branch_reward"], drop_record["branch_reward"])

    def test_score_search_counterfactual_group_scores_skip_use_and_delta_only(self):
        records = score_search_counterfactual_group(
            state_before=_make_partial_state(["w0001"]),
            state_after=_make_partial_state(["w0001", "w0002"]),
            multimodal_cache=_make_multimodal_cache(),
            claim={"existence": "anomaly", "category": "assault", "earliest_alert_sec": 1.0},
            anchor_turn_index=2,
            alert={"decision": "hard_alert", "alert_sec": 1.2},
            verifier_backend="heuristic",
            use_reference_supervision=True,
        )

        self.assertEqual(
            [record["branch_type"] for record in records],
            ["skip_search", "use_search", "delta_only"],
        )
        self.assertTrue(all("branch_reward_components" in record for record in records))
        self.assertAlmostEqual(sum(record["local_advantage"] for record in records), 0.0, places=5)
        use_record = next(record for record in records if record["branch_type"] == "use_search")
        skip_record = next(record for record in records if record["branch_type"] == "skip_search")
        self.assertGreater(use_record["branch_reward"], skip_record["branch_reward"])


if __name__ == "__main__":
    unittest.main()
