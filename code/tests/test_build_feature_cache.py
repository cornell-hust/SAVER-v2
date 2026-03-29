import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch


ROOT = Path("/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


try:
    import build_feature_cache as build_feature_cache
except ModuleNotFoundError:
    build_feature_cache = None


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


class _FakeFeatureEncoder:
    def encode_images(self, images):
        images = images.float()
        flat = images.flatten(start_dim=1)
        mean = flat.mean(dim=1)
        std = flat.std(dim=1)
        return torch.stack([mean, std], dim=1)


class _FakeFeatureEncoderWithModelOutput:
    def encode_images(self, images):
        images = images.float()
        flat = images.flatten(start_dim=1)
        mean = flat.mean(dim=1)
        std = flat.std(dim=1)
        features = torch.stack([mean, std], dim=1)
        return SimpleNamespace(pooler_output=features)


class BuildFeatureCacheTests(unittest.TestCase):
    def _sample_record(self) -> dict:
        return {
            "schema_version": "saver_agent.v1",
            "video_id": "sample_feature_video",
            "video_path": "videos/sample_feature_video.mp4",
            "split": "train",
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

    def test_build_feature_cache_writes_structured_cache_next_to_video(self):
        self.assertIsNotNone(build_feature_cache, "build_feature_cache.py is missing")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video_path = root / "videos" / "sample_feature_video.mp4"
            _write_test_video(video_path, num_frames=8, fps=4.0)
            frame_cache_path = Path(str(video_path) + ".frame_cache")
            torch.save(
                {
                    "frame_tensor": torch.randint(0, 255, (4, 3, 32, 32), dtype=torch.uint8),
                    "frame_indices": [0, 2, 4, 6],
                    "fps": 2.0,
                },
                frame_cache_path,
            )
            data_path = root / "agent.jsonl"
            data_path.write_text(json.dumps(self._sample_record()) + "\n", encoding="utf-8")

            summary = build_feature_cache.build_feature_caches(
                data_path=data_path,
                data_root=root,
                encoder=_FakeFeatureEncoder(),
                overwrite=False,
                progress_every=0,
            )

            cache_path = Path(str(video_path) + ".feature_cache")
            self.assertTrue(cache_path.exists())
            cache = torch.load(cache_path, map_location="cpu")
            self.assertEqual(cache["version"], "saver_feature_cache_v1")
            self.assertEqual(cache["frame_indices"], [0, 2, 4, 6])
            self.assertEqual(tuple(cache["embeddings"].shape), (4, 2))
            self.assertEqual(summary["written"], 1)

    def test_build_feature_cache_accepts_model_output_with_pooler_output(self):
        self.assertIsNotNone(build_feature_cache, "build_feature_cache.py is missing")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video_path = root / "videos" / "sample_feature_video.mp4"
            _write_test_video(video_path, num_frames=8, fps=4.0)
            frame_cache_path = Path(str(video_path) + ".frame_cache")
            torch.save(
                {
                    "frame_tensor": torch.randint(0, 255, (4, 3, 32, 32), dtype=torch.uint8),
                    "frame_indices": [0, 2, 4, 6],
                    "fps": 2.0,
                },
                frame_cache_path,
            )
            data_path = root / "agent.jsonl"
            data_path.write_text(json.dumps(self._sample_record()) + "\n", encoding="utf-8")

            summary = build_feature_cache.build_feature_caches(
                data_path=data_path,
                data_root=root,
                encoder=_FakeFeatureEncoderWithModelOutput(),
                overwrite=True,
                progress_every=0,
            )

            cache_path = Path(str(video_path) + ".feature_cache")
            self.assertTrue(cache_path.exists())
            cache = torch.load(cache_path, map_location="cpu")
            self.assertEqual(tuple(cache["embeddings"].shape), (4, 2))
            self.assertEqual(summary["written"], 1)
            self.assertEqual(summary["encode_failures"], 0)

    def test_build_feature_cache_does_not_log_per_file_skip_messages(self):
        self.assertIsNotNone(build_feature_cache, "build_feature_cache.py is missing")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video_path = root / "videos" / "sample_feature_video.mp4"
            _write_test_video(video_path, num_frames=8, fps=4.0)
            frame_cache_path = Path(str(video_path) + ".frame_cache")
            torch.save(
                {
                    "frame_tensor": torch.randint(0, 255, (4, 3, 32, 32), dtype=torch.uint8),
                    "frame_indices": [0, 2, 4, 6],
                    "fps": 2.0,
                },
                frame_cache_path,
            )
            data_path = root / "agent.jsonl"
            data_path.write_text(json.dumps(self._sample_record()) + "\n", encoding="utf-8")

            build_feature_cache.build_feature_caches(
                data_path=data_path,
                data_root=root,
                encoder=_FakeFeatureEncoder(),
                overwrite=False,
                progress_every=1,
            )
            with patch.object(build_feature_cache, "_print_progress") as mock_print_progress:
                summary = build_feature_cache.build_feature_caches(
                    data_path=data_path,
                    data_root=root,
                    encoder=_FakeFeatureEncoder(),
                    overwrite=False,
                    progress_every=1,
                )

            logged_messages = [call.args[0] for call in mock_print_progress.call_args_list]
            self.assertEqual(summary["skipped_existing"], 1)
            self.assertFalse(any("skip existing feature cache" in message for message in logged_messages))


if __name__ == "__main__":
    unittest.main()
