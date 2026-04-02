import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path("/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from saver_agent.evaluation import RolloutEvaluationConfig, _resolve_proposal_device, run_rollout_evaluation
from saver_agent.runtime import DistributedRuntime


class _DummyDataset:
    def __init__(self, *args, **kwargs):
        self.items = [{"video_id": "eval_vid", "multimodal_cache": {}, "split": "val"}]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return dict(self.items[index])


class _DummyRunner:
    def __init__(self, *args, **kwargs):
        pass

    def run_episode(self, item, policy):
        return {"video_id": item["video_id"], "turns": [], "state": {}, "num_turns": 1}


class SaverAgentEvaluationTests(unittest.TestCase):
    def test_run_rollout_evaluation_initializes_torch_distributed_before_using_barriers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            data_path = tmpdir_path / "data.jsonl"
            data_path.write_text(
                json.dumps(
                    {
                        "video_id": "eval_vid",
                        "video_meta": {"fps": 1.0, "duration_sec": 4.0, "total_frames": 4},
                        "structured_target": {"existence": "normal", "category": "normal", "severity": 0},
                        "tool_io": {"oracle_windows_sec": []},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            def fake_save_rollout_records(records, output_path, **kwargs):
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with output_path.open("w", encoding="utf-8") as f:
                    for record in records:
                        f.write(json.dumps(record) + "\n")
                sibling_path = output_path.parent / "part.rank01-of-02.jsonl"
                sibling_path.write_text("", encoding="utf-8")

            with patch("saver_agent.evaluation.SaverAgentDataset", _DummyDataset), patch(
                "saver_agent.evaluation.SaverRolloutRunner", _DummyRunner
            ), patch("saver_agent.evaluation._serialize_result", side_effect=lambda result: dict(result)), patch(
                "saver_agent.evaluation.score_rollout_records",
                side_effect=lambda records, **kwargs: list(records),
            ), patch(
                "saver_agent.evaluation.save_rollout_records",
                side_effect=fake_save_rollout_records,
            ), patch(
                "saver_agent.evaluation.summarize_saver_metrics",
                return_value={"num_records": 1},
            ), patch(
                "saver_agent.evaluation.init_torch_distributed",
                create=True,
                return_value=True,
            ) as init_dist:
                run_rollout_evaluation(
                    policy=object(),
                    eval_config=RolloutEvaluationConfig(data_path=data_path),
                    output_dir=tmpdir_path,
                    epoch_index=0,
                    runtime=DistributedRuntime(rank=0, world_size=2, local_rank=0),
                )

        init_dist.assert_called_once()

    def test_resolve_proposal_device_falls_back_to_cuda_zero_when_local_rank_exceeds_visible_devices(self):
        runtime = DistributedRuntime(rank=2, world_size=4, local_rank=2)

        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.device_count", return_value=2):
            resolved = _resolve_proposal_device("", runtime=runtime)

        self.assertEqual(resolved, "cuda:0")

    def test_run_rollout_evaluation_keeps_default_main_eval_free_of_online_verifier_context(self):
        captured = {}

        class VerifierContextAwareRunner:
            def __init__(self, *args, **kwargs):
                pass

            def run_episode(self, item, policy):
                captured["multimodal_cache"] = dict(item.get("multimodal_cache") or {})
                return {"video_id": item["video_id"], "turns": [], "state": {}, "num_turns": 1}

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            data_path = tmpdir_path / "data.jsonl"
            data_path.write_text(
                json.dumps(
                    {
                        "video_id": "eval_vid",
                        "video_meta": {"fps": 1.0, "duration_sec": 4.0, "total_frames": 4},
                        "structured_target": {"existence": "normal", "category": "normal", "severity": 0},
                        "tool_io": {"oracle_windows_sec": []},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch("saver_agent.evaluation.SaverAgentDataset", _DummyDataset), patch(
                "saver_agent.evaluation.SaverRolloutRunner",
                VerifierContextAwareRunner,
            ), patch("saver_agent.evaluation._serialize_result", side_effect=lambda result: dict(result)), patch(
                "saver_agent.evaluation.score_rollout_records",
                side_effect=lambda records, **kwargs: list(records),
            ), patch(
                "saver_agent.evaluation.summarize_saver_metrics",
                return_value={"num_records": 1},
            ):
                run_rollout_evaluation(
                    policy=object(),
                    eval_config=RolloutEvaluationConfig(data_path=data_path),
                    output_dir=tmpdir_path,
                    epoch_index=0,
                    runtime=DistributedRuntime(),
                )

        self.assertNotIn("verifier_backend", captured["multimodal_cache"])
        self.assertNotIn("verifier_model_path", captured["multimodal_cache"])
        self.assertNotIn("allow_external_verifier_fallback", captured["multimodal_cache"])
        self.assertTrue(captured["multimodal_cache"].get("disable_external_verifier_fallback"))
        self.assertFalse(captured["multimodal_cache"].get("allow_legacy_verify_compatibility"))

    def test_run_rollout_evaluation_does_not_force_reference_reverify_by_default(self):
        captured = {}

        def fake_score_rollout_records(records, **kwargs):
            captured.update(kwargs)
            return list(records)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            data_path = tmpdir_path / "data.jsonl"
            data_path.write_text(
                json.dumps(
                    {
                        "video_id": "eval_vid",
                        "video_meta": {"fps": 1.0, "duration_sec": 4.0, "total_frames": 4},
                        "structured_target": {"existence": "normal", "category": "normal", "severity": 0},
                        "tool_io": {"oracle_windows_sec": []},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch("saver_agent.evaluation.SaverAgentDataset", _DummyDataset), patch(
                "saver_agent.evaluation.SaverRolloutRunner", _DummyRunner
            ), patch("saver_agent.evaluation._serialize_result", side_effect=lambda result: dict(result)), patch(
                "saver_agent.evaluation.score_rollout_records", side_effect=fake_score_rollout_records
            ), patch(
                "saver_agent.evaluation.summarize_saver_metrics", return_value={"num_records": 1}
            ):
                summary = run_rollout_evaluation(
                    policy=object(),
                    eval_config=RolloutEvaluationConfig(data_path=data_path),
                    output_dir=tmpdir_path,
                    epoch_index=0,
                    runtime=DistributedRuntime(),
                )

        self.assertEqual(summary["num_records"], 1)
        self.assertFalse(captured["force_reverify"])
        self.assertFalse(captured["attach_reference_offline_verifier"])

    def test_run_rollout_evaluation_attaches_proposal_runtime_when_configured(self):
        captured = {}

        class ProposalAwareRunner:
            def __init__(self, *args, **kwargs):
                pass

            def run_episode(self, item, policy):
                captured["proposal_runtime"] = item["multimodal_cache"].get("proposal_runtime")
                return {"video_id": item["video_id"], "turns": [], "state": {}, "num_turns": 1}

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            data_path = tmpdir_path / "data.jsonl"
            data_path.write_text(
                json.dumps(
                    {
                        "video_id": "eval_vid",
                        "video_meta": {"fps": 1.0, "duration_sec": 4.0, "total_frames": 4},
                        "structured_target": {"existence": "normal", "category": "normal", "severity": 0},
                        "tool_io": {"oracle_windows_sec": []},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch("saver_agent.evaluation.SaverAgentDataset", _DummyDataset), patch(
                "saver_agent.evaluation.SaverRolloutRunner",
                ProposalAwareRunner,
            ), patch("saver_agent.evaluation._serialize_result", side_effect=lambda result: dict(result)), patch(
                "saver_agent.evaluation._load_proposal_runtime",
                return_value="siglip_runtime",
            ), patch(
                "saver_agent.evaluation.score_rollout_records",
                side_effect=lambda records, **kwargs: list(records),
            ), patch(
                "saver_agent.evaluation.summarize_saver_metrics",
                return_value={"num_records": 1},
            ):
                run_rollout_evaluation(
                    policy=object(),
                    eval_config=RolloutEvaluationConfig(
                        data_path=data_path,
                        proposal_model_path="/models/siglip_base",
                    ),
                    output_dir=tmpdir_path,
                    epoch_index=0,
                    runtime=DistributedRuntime(),
                )

        self.assertEqual(captured["proposal_runtime"], "siglip_runtime")

    def test_run_rollout_evaluation_does_not_merge_stale_scored_shards_from_previous_runs(self):
        captured = {}

        def fake_summarize(records, **kwargs):
            captured["video_ids"] = [record.get("video_id") for record in records]
            return {"num_records": len(records)}

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            data_path = tmpdir_path / "data.jsonl"
            data_path.write_text(
                json.dumps(
                    {
                        "video_id": "eval_vid",
                        "video_meta": {"fps": 1.0, "duration_sec": 4.0, "total_frames": 4},
                        "structured_target": {"existence": "normal", "category": "normal", "severity": 0},
                        "tool_io": {"oracle_windows_sec": []},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            stale_dir = tmpdir_path / "rollout_eval" / "epoch_000" / "scored_shards"
            stale_dir.mkdir(parents=True, exist_ok=True)
            (stale_dir / "stale.jsonl").write_text(json.dumps({"video_id": "stale_vid"}) + "\n", encoding="utf-8")

            with patch("saver_agent.evaluation.SaverAgentDataset", _DummyDataset), patch(
                "saver_agent.evaluation.SaverRolloutRunner", _DummyRunner
            ), patch("saver_agent.evaluation._serialize_result", side_effect=lambda result: dict(result)), patch(
                "saver_agent.evaluation.score_rollout_records",
                side_effect=lambda records, **kwargs: list(records),
            ), patch(
                "saver_agent.evaluation.summarize_saver_metrics",
                side_effect=fake_summarize,
            ):
                summary = run_rollout_evaluation(
                    policy=object(),
                    eval_config=RolloutEvaluationConfig(data_path=data_path),
                    output_dir=tmpdir_path,
                    epoch_index=0,
                    runtime=DistributedRuntime(),
                )

        self.assertEqual(summary["num_records"], 1)
        self.assertEqual(captured["video_ids"], ["eval_vid"])


if __name__ == "__main__":
    unittest.main()
