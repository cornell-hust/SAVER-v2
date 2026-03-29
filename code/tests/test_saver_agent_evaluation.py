import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path("/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from saver_agent.evaluation import RolloutEvaluationConfig, run_rollout_evaluation
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


if __name__ == "__main__":
    unittest.main()
