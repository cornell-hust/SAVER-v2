import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path("/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


try:
    import build_frame_cache as build_frame_cache
except ModuleNotFoundError:
    build_frame_cache = None


def _write_test_video(path: Path, *, num_frames: int = 8, fps: float = 4.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (32, 32))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open test video writer for {path}")
    for frame_id in range(num_frames):
        frame = np.full((32, 32, 3), (frame_id * 20) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class BuildFrameCacheTests(unittest.TestCase):
    def _sample_record(self, *, split: str = "train") -> dict:
        return {
            "schema_version": "saver_agent.v1",
            "video_id": "sample_cache_video",
            "video_path": "videos/sample_cache_video.mp4",
            "split": split,
            "source_dataset": "MSAD",
            "frame_index_base": 1,
            "video_meta": {"fps": 4.0, "duration_sec": 2.0, "total_frames": 8},
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
                "allowed_tools": ["scan_timeline"],
                "initial_scan_window_frames": [1, 8],
                "initial_scan_window_sec": [0.0, 2.0],
                "oracle_windows_frames": [],
                "oracle_windows_sec": [],
                "finalize_case_schema": {"type": "object", "required": ["existence"]},
            },
            "auto_completed": {"precursor_interval": False},
        }

    def test_build_frame_cache_writes_cache_next_to_video(self):
        self.assertIsNotNone(build_frame_cache, "build_frame_cache.py is missing")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video_path = root / "videos" / "sample_cache_video.mp4"
            _write_test_video(video_path, num_frames=8, fps=4.0)
            data_path = root / "agent.jsonl"
            data_path.write_text(json.dumps(self._sample_record()) + "\n", encoding="utf-8")

            summary = build_frame_cache.build_frame_caches(
                data_path=data_path,
                data_root=root,
                include_splits="train",
                overwrite=False,
                progress_every=0,
            )

            cache_path = Path(str(video_path) + ".frame_cache")
            self.assertTrue(cache_path.exists())
            cache = torch.load(cache_path, map_location="cpu")
            self.assertIn("frame_tensor", cache)
            self.assertIn("frame_indices", cache)
            self.assertIn("fps", cache)
            self.assertEqual(int(cache["frame_tensor"].shape[0]), len(cache["frame_indices"]))
            self.assertGreater(float(cache["fps"]), 0.0)
            self.assertEqual(summary["written"], 1)
            self.assertEqual(summary["skipped_existing"], 0)

    def test_build_frame_cache_respects_split_and_skip_existing(self):
        self.assertIsNotNone(build_frame_cache, "build_frame_cache.py is missing")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_video = root / "videos" / "sample_cache_video.mp4"
            test_video = root / "videos" / "sample_cache_video_test.mp4"
            _write_test_video(train_video, num_frames=8, fps=4.0)
            _write_test_video(test_video, num_frames=8, fps=4.0)
            train_record = self._sample_record(split="train")
            test_record = dict(self._sample_record(split="test"))
            test_record["video_id"] = "sample_cache_video_test"
            test_record["video_path"] = "videos/sample_cache_video_test.mp4"
            data_path = root / "agent.jsonl"
            data_path.write_text(
                json.dumps(train_record) + "\n" + json.dumps(test_record) + "\n",
                encoding="utf-8",
            )

            first_summary = build_frame_cache.build_frame_caches(
                data_path=data_path,
                data_root=root,
                include_splits="train",
                overwrite=False,
                progress_every=0,
            )
            second_summary = build_frame_cache.build_frame_caches(
                data_path=data_path,
                data_root=root,
                include_splits="train",
                overwrite=False,
                progress_every=0,
            )

            train_cache = Path(str(train_video) + ".frame_cache")
            test_cache = Path(str(test_video) + ".frame_cache")
            self.assertTrue(train_cache.exists())
            self.assertFalse(test_cache.exists())
            self.assertEqual(first_summary["written"], 1)
            self.assertEqual(second_summary["skipped_existing"], 1)


if __name__ == "__main__":
    unittest.main()
