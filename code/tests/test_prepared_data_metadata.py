import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path("/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


try:
    import build_saver_data as build_saver_data
    import train_saver_sft as train_saver_sft
except ModuleNotFoundError:
    build_saver_data = None
    train_saver_sft = None


def _canonical_record(*, video_id: str, split: str, is_anomaly: bool) -> dict:
    category = "assault" if is_anomaly else "normal"
    evidence_moments = (
        [
            {
                "moment_id": "ev1",
                "start_frame": 9,
                "end_frame": 20,
                "role": "precursor",
                "description": "person approaches another person",
            },
            {
                "moment_id": "ev2",
                "start_frame": 21,
                "end_frame": 40,
                "role": "trigger",
                "description": "aggressive physical contact occurs",
            },
        ]
        if is_anomaly
        else []
    )
    return {
        "video_id": video_id,
        "file_name": f"{video_id}.mp4",
        "video_path": f"videos/{video_id}.mp4",
        "source_dataset": "MSAD",
        "source_split": split,
        "split": split,
        "frame_index_base": 1,
        "video_meta": {
            "fps": 10.0,
            "width": 1280,
            "height": 720,
            "total_frames": 100,
            "duration_sec": 10.0,
        },
        "scene": {"scenario": "corridor"},
        "key_objects": ["person"],
        "label": {
            "is_anomaly": is_anomaly,
            "category": category,
            "severity": 4 if is_anomaly else 0,
            "hard_normal": False,
        },
        "temporal": {
            "anomaly_interval_frames": [21, 40] if is_anomaly else None,
            "precursor_interval_frames": [9, 20] if is_anomaly else None,
            "earliest_alert_frame": 21 if is_anomaly else None,
        },
        "evidence": {"evidence_moments": evidence_moments},
        "counterfactual": {
            "type": "remove_actor_interaction" if is_anomaly else "none",
            "text": "No contact, no anomaly." if is_anomaly else "Normal activity remains normal.",
        },
        "language": {
            "summary": "An assault occurs." if is_anomaly else "Normal activity.",
            "rationale": "One person attacks another." if is_anomaly else "No anomaly is visible.",
        },
        "qa_pairs": [],
        "provenance": {"annotation_status": "unit_test"},
    }


class PreparedDataMetadataTests(unittest.TestCase):
    def test_build_saver_data_writes_metadata_sidecar_for_prepared_sft(self):
        self.assertIsNotNone(build_saver_data, "build_saver_data.py is missing")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "canonical.jsonl"
            runtime_train_output = root / "msad_saver_runtime_train.jsonl"
            runtime_test_output = root / "msad_saver_runtime_test.jsonl"
            sft_train_output = root / "msad_saver_sft_train.jsonl"

            records = [
                _canonical_record(video_id="Assault_1", split="train", is_anomaly=True),
                _canonical_record(video_id="Normal_1", split="test", is_anomaly=False),
            ]
            input_path.write_text(
                "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
                encoding="utf-8",
            )

            result = build_saver_data.main(
                [
                    "--input",
                    str(input_path),
                    "--runtime-train-output",
                    str(runtime_train_output),
                    "--runtime-test-output",
                    str(runtime_test_output),
                    "--sft-train-output",
                    str(sft_train_output),
                ]
            )

            self.assertEqual(result, 0)
            metadata_path = Path(str(sft_train_output) + ".meta.json")
            self.assertTrue(metadata_path.exists())
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertIn("schema_version", metadata)
            self.assertIn("preview", metadata)
            self.assertIn("prompt", metadata)

    def test_train_saver_sft_rejects_prepared_data_without_metadata(self):
        self.assertIsNotNone(train_saver_sft, "train_saver_sft.py is missing")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prepared_path = root / "prepared.jsonl"
            prepared_row = {
                "video_id": "Assault_1",
                "split": "train",
                "step_index": 1,
                "source": "oracle_sft",
                "target_action": "tool_call",
                "target_response": "<tool_call>{\"name\":\"scan_timeline\",\"arguments\":{}}</tool_call>",
                "messages": [],
                "sample_weight": 1.0,
                "tool_name": "scan_timeline",
                "proposal_supervision": {},
            }
            prepared_path.write_text(json.dumps(prepared_row, ensure_ascii=False) + "\n", encoding="utf-8")

            argv = [
                "train_saver_sft.py",
                "--prepared-data",
                str(prepared_path),
                "--include-splits",
                "train",
                "--dry-run",
            ]
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(ValueError, "metadata|meta"):
                    train_saver_sft.main()


if __name__ == "__main__":
    unittest.main()
