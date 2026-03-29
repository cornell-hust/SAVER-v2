import json
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path("/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saver_agent.dataset import SaverAgentDataset
from saver_agent.config import PromptConfig, PreviewConfig, SaverAgentConfig


def _write_test_video(path: Path, *, num_frames: int = 6, fps: float = 4.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (32, 32))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open test video writer for {path}")
    for frame_id in range(num_frames):
        frame = np.full((32, 32, 3), (frame_id * 20) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class SaverAgentDatasetTests(unittest.TestCase):
    def _sample_record(
        self,
        *,
        video_id: str,
        video_path: str,
        split: str = "train",
    ) -> dict:
        return {
            "schema_version": "saver_agent.v1",
            "video_id": video_id,
            "video_path": video_path,
            "split": split,
            "source_dataset": "MSAD",
            "frame_index_base": 1,
            "video_meta": {"fps": 4.0, "duration_sec": 1.5, "total_frames": 6},
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
                "initial_scan_window_frames": [1, 6],
                "initial_scan_window_sec": [0.0, 1.5],
                "oracle_windows_frames": [],
                "oracle_windows_sec": [],
                "finalize_case_schema": {"type": "object", "required": ["existence"]},
            },
            "auto_completed": {"precursor_interval": False},
        }

    def test_dataset_loads_agent_train_record_and_builds_messages(self):
        sample = {
            "schema_version": "saver_agent.v1",
            "video_id": "sample_001",
            "video_path": "videos/sample_001.mp4",
            "split": "train",
            "source_dataset": "MSAD",
            "frame_index_base": 1,
            "video_meta": {"fps": 10.0, "duration_sec": 12.0, "total_frames": 120},
            "scene": {"scenario": "shop"},
            "label": {"is_anomaly": True, "category": "robbery", "severity": 4, "hard_normal": False},
            "agent_task": {
                "task_type": "video_anomaly_search_alert_verify",
                "query_mode": "internal_hypothesis_generation",
                "task_prompt": "Find anomaly evidence and decide whether to alert.",
                "success_criteria": ["criterion_a", "criterion_b"],
            },
            "structured_target": {
                "existence": "anomaly",
                "category": "robbery",
                "severity": 4,
                "hard_normal": False,
                "anomaly_interval_sec": [4.0, 9.0],
                "precursor_interval_sec": [2.0, 4.0],
                "earliest_alert_sec": 4.0,
                "evidence_moment_ids": ["ev1"],
                "evidence_windows_sec": [{"moment_id": "ev1", "role": "trigger", "window_sec": [4.0, 5.0]}],
                "counterfactual_type": "remove_actor_interaction",
                "counterfactual_text": "No interaction, no robbery.",
                "summary": "A robbery happens.",
                "rationale": "A suspect approaches and snatches a bag.",
            },
            "tool_io": {
                "allowed_tools": ["scan_timeline", "seek_evidence", "emit_alert", "verify_hypothesis", "finalize_case"],
                "initial_scan_window_frames": [1, 120],
                "initial_scan_window_sec": [0.0, 12.0],
                "oracle_windows_frames": [],
                "oracle_windows_sec": [],
                "finalize_case_schema": {"type": "object", "required": ["existence"]},
            },
            "auto_completed": {"precursor_interval": False},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "agent.jsonl"
            with path.open("w", encoding="utf-8") as f:
                f.write(json.dumps(sample) + "\n")

            dataset = SaverAgentDataset(path, data_root="/dataset/root")
            item = dataset[0]

        self.assertEqual(len(dataset), 1)
        self.assertEqual(item["video_id"], "sample_001")
        self.assertEqual(item["video"], "/dataset/root/videos/sample_001.mp4")
        self.assertEqual(item["structured_target"]["existence"], "anomaly")
        self.assertEqual(item["tool_io"]["allowed_tools"][0], "scan_timeline")
        self.assertEqual(item["messages"][0]["role"], "system")
        self.assertEqual(item["messages"][1]["role"], "user")
        self.assertIn("Do not describe the tool call in plain English", item["messages"][0]["content"][0]["text"])
        self.assertIn('"name":"scan_timeline"', item["messages"][0]["content"][0]["text"])
        self.assertIn("Find anomaly evidence", item["messages"][1]["content"][0]["text"])
        self.assertEqual(item["multimodal_cache"]["duration"], 12.0)

    def test_dataset_resolves_legacy_data_prefix_to_existing_video_root(self):
        sample = {
            "schema_version": "saver_agent.v1",
            "video_id": "Assault_1",
            "video_path": "data/MSAD_anomaly_blur/Assault/Assault_1.mp4",
            "split": "train",
            "source_dataset": "MSAD",
            "frame_index_base": 1,
            "video_meta": {"fps": 4.0, "duration_sec": 1.5, "total_frames": 6},
            "scene": {"scenario": "street"},
            "label": {"is_anomaly": True, "category": "assault", "severity": 3, "hard_normal": False},
            "agent_task": {
                "task_type": "video_anomaly_search_alert_verify",
                "query_mode": "internal_hypothesis_generation",
                "task_prompt": "Find anomaly evidence.",
                "success_criteria": ["criterion_a"],
            },
            "structured_target": {"existence": "anomaly"},
            "tool_io": {
                "allowed_tools": ["scan_timeline"],
                "initial_scan_window_frames": [1, 6],
                "initial_scan_window_sec": [0.0, 1.5],
                "oracle_windows_frames": [],
                "oracle_windows_sec": [],
                "finalize_case_schema": {"type": "object", "required": ["existence"]},
            },
            "auto_completed": {"precursor_interval": False},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            actual_video = root / "Wmh" / "datasets" / "MSDA" / "MSAD_anomaly_blur" / "Assault" / "Assault_1.mp4"
            _write_test_video(actual_video)
            path = root / "agent.jsonl"
            with path.open("w", encoding="utf-8") as f:
                f.write(json.dumps(sample) + "\n")

            dataset = SaverAgentDataset(path, data_root=root)
            item = dataset[0]

        self.assertEqual(item["video"], str(actual_video))
        self.assertIsNotNone(item["multimodal_cache"]["video"])

    def test_dataset_summarizes_frame_cache_availability(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cached_video = root / "videos" / "cached.mp4"
            uncached_video = root / "videos" / "uncached.mp4"
            _write_test_video(cached_video)
            _write_test_video(uncached_video)
            torch.save(
                {
                    "frame_tensor": torch.zeros(2, 3, 32, 32, dtype=torch.uint8),
                    "frame_indices": [0, 3],
                    "fps": 2.0,
                },
                Path(str(cached_video) + ".frame_cache"),
            )
            data_path = root / "agent.jsonl"
            records = [
                self._sample_record(video_id="cached_case", video_path="videos/cached.mp4"),
                self._sample_record(video_id="uncached_case", video_path="videos/uncached.mp4"),
            ]
            data_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

            dataset = SaverAgentDataset(data_path, data_root=root)
            summary = dataset.summarize_frame_cache_status()
            message = dataset.format_frame_cache_status(prefix="train cache")

        self.assertEqual(summary["num_records"], 2)
        self.assertEqual(summary["num_cached_videos"], 1)
        self.assertEqual(summary["num_missing_frame_cache"], 1)
        self.assertEqual(summary["num_missing_video_files"], 0)
        self.assertEqual(summary["missing_examples"][0]["video_id"], "uncached_case")
        self.assertIn("train cache", message)
        self.assertIn("cached=1/2", message)
        self.assertIn("missing_frame_cache=1", message)

    def test_dataset_loads_frames_from_video_when_cache_missing(self):
        sample = {
            "schema_version": "saver_agent.v1",
            "video_id": "sample_002",
            "video_path": "videos/sample_002.mp4",
            "split": "train",
            "source_dataset": "MSAD",
            "frame_index_base": 1,
            "video_meta": {"fps": 4.0, "duration_sec": 1.5, "total_frames": 6},
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
                "initial_scan_window_frames": [1, 6],
                "initial_scan_window_sec": [0.0, 1.5],
                "oracle_windows_frames": [],
                "oracle_windows_sec": [],
                "finalize_case_schema": {"type": "object", "required": ["existence"]},
            },
            "auto_completed": {"precursor_interval": False},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            actual_video = root / "videos" / "sample_002.mp4"
            _write_test_video(actual_video)
            path = root / "agent.jsonl"
            with path.open("w", encoding="utf-8") as f:
                f.write(json.dumps(sample) + "\n")

            dataset = SaverAgentDataset(path, data_root=root)
            item = dataset[0]

        self.assertEqual(item["video"], str(actual_video))
        self.assertIsNotNone(item["multimodal_cache"]["video"])
        self.assertGreaterEqual(item["multimodal_cache"]["video"].shape[0], 2)
        self.assertGreater(item["multimodal_cache"]["fps"], 0.0)

    def test_dataset_prints_cache_warning_once_when_falling_back_to_raw_video(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            actual_video = root / "videos" / "sample_warning.mp4"
            _write_test_video(actual_video)
            data_path = root / "agent.jsonl"
            data_path.write_text(
                json.dumps(self._sample_record(video_id="sample_warning", video_path="videos/sample_warning.mp4")) + "\n",
                encoding="utf-8",
            )

            dataset = SaverAgentDataset(data_path, data_root=root)
            captured = io.StringIO()
            with redirect_stdout(captured):
                _ = dataset[0]
                _ = dataset[0]

        output = captured.getvalue()
        self.assertIn("[cache-warning]", output)
        self.assertIn("video_id=sample_warning", output)
        self.assertEqual(output.count("[cache-warning]"), 1)

    def test_dataset_builds_timesearch_style_preview_frames_in_user_message(self):
        sample = {
            "schema_version": "saver_agent.v1",
            "video_id": "sample_preview",
            "video_path": "videos/sample_preview.mp4",
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

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            actual_video = root / "videos" / "sample_preview.mp4"
            _write_test_video(actual_video, num_frames=8, fps=4.0)
            path = root / "agent.jsonl"
            with path.open("w", encoding="utf-8") as f:
                f.write(json.dumps(sample) + "\n")

            dataset = SaverAgentDataset(path, data_root=root)
            item = dataset[0]

        user_content = item["messages"][1]["content"]
        image_items = [entry for entry in user_content if entry["type"] == "image"]
        timestamp_items = [
            entry for entry in user_content if entry["type"] == "text" and entry["text"].endswith("s")
        ]

        self.assertGreaterEqual(len(image_items), 2)
        self.assertEqual(len(image_items), len(timestamp_items))
        self.assertIn("preview frames", user_content[-1]["text"].lower())
        self.assertIsNotNone(item["multimodal_cache"]["preview_frames"])
        self.assertEqual(len(item["multimodal_cache"]["preview_timestamps"]), len(image_items))

    def test_dataset_uses_configured_preview_sampling_and_prompt_template(self):
        sample = {
            "schema_version": "saver_agent.v1",
            "video_id": "sample_configured_preview",
            "video_path": "videos/sample_configured_preview.mp4",
            "split": "train",
            "source_dataset": "MSAD",
            "frame_index_base": 1,
            "video_meta": {"fps": 4.0, "duration_sec": 4.0, "total_frames": 16},
            "scene": {"scenario": "office"},
            "label": {"is_anomaly": False, "category": "normal", "severity": 0, "hard_normal": False},
            "agent_task": {
                "task_type": "video_anomaly_search_alert_verify",
                "query_mode": "internal_hypothesis_generation",
                "task_prompt": "Inspect the office clip.",
                "success_criteria": ["criterion_a"],
            },
            "structured_target": {"existence": "normal"},
            "tool_io": {
                "allowed_tools": ["scan_timeline"],
                "initial_scan_window_frames": [1, 16],
                "initial_scan_window_sec": [0.0, 4.0],
                "oracle_windows_frames": [],
                "oracle_windows_sec": [],
                "finalize_case_schema": {"type": "object", "required": ["existence"]},
            },
            "auto_completed": {"precursor_interval": False},
        }

        config = SaverAgentConfig(
            preview=PreviewConfig(num_preview_frames=6, preview_sampling_fps=0.5, max_preview_frames=6),
            prompt=PromptConfig(
                initial_user_template="Case: {video_id}\nTask: {task_prompt}\nCriteria:\n{criteria_text}",
                preview_instruction="Custom preview instruction.",
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            actual_video = root / "videos" / "sample_configured_preview.mp4"
            _write_test_video(actual_video, num_frames=16, fps=4.0)
            path = root / "agent.jsonl"
            with path.open("w", encoding="utf-8") as f:
                f.write(json.dumps(sample) + "\n")

            dataset = SaverAgentDataset(path, data_root=root, config=config)
            item = dataset[0]

        user_content = item["messages"][1]["content"]
        image_items = [entry for entry in user_content if entry["type"] == "image"]
        self.assertEqual(len(image_items), 2)
        self.assertIn("Case: sample_configured_preview", user_content[-1]["text"])
        self.assertIn("Custom preview instruction.", user_content[-1]["text"])

    def test_dataset_filters_records_by_split(self):
        base_sample = {
            "schema_version": "saver_agent.v1",
            "video_path": "videos/sample.mp4",
            "source_dataset": "MSAD",
            "frame_index_base": 1,
            "video_meta": {"fps": 10.0, "duration_sec": 1.0, "total_frames": 10},
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
                "initial_scan_window_frames": [1, 10],
                "initial_scan_window_sec": [0.0, 1.0],
                "oracle_windows_frames": [],
                "oracle_windows_sec": [],
                "finalize_case_schema": {"type": "object", "required": ["existence"]},
            },
            "auto_completed": {"precursor_interval": False},
        }
        test_sample = {
            **base_sample,
            "video_id": "test_case",
            "split": "test",
        }
        train_sample = {
            **base_sample,
            "video_id": "train_case",
            "split": "train",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "agent.jsonl"
            with path.open("w", encoding="utf-8") as f:
                f.write(json.dumps(test_sample) + "\n")
                f.write(json.dumps(train_sample) + "\n")

            dataset = SaverAgentDataset(path, data_root="/dataset/root", include_splits="train")
            item = dataset[0]

        self.assertEqual(len(dataset), 1)
        self.assertEqual(item["video_id"], "train_case")


if __name__ == "__main__":
    unittest.main()
