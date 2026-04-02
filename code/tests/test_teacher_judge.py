import sys
import unittest
from pathlib import Path


ROOT = Path("/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from saver_agent.teacher_judge import build_teacher_judge_package


class TeacherJudgePackageTests(unittest.TestCase):
    def test_build_teacher_judge_package_uses_selected_subset_for_keep_and_complement_for_drop(self):
        example = {
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "system"}]},
                {"role": "user", "content": [{"type": "text", "text": "user"}]},
                {
                    "role": "tool",
                    "name": "scan_timeline",
                    "arguments": {"start_sec": 0.0, "end_sec": 6.0, "purpose": "global_overview"},
                    "content": [
                        {"type": "text", "text": "0.500s"},
                        {
                            "type": "image",
                            "image_ref": {
                                "video_path": "/tmp/video.mp4",
                                "timestamp_sec": 0.5,
                                "sampled_frame_index": 0,
                            },
                        },
                    ],
                },
                {
                    "role": "tool",
                    "name": "seek_evidence",
                    "arguments": {
                        "start_sec": 1.0,
                        "end_sec": 2.0,
                        "moment_id": "m1",
                        "role": "precursor",
                        "query": "person on ground",
                    },
                    "content": [
                        {"type": "text", "text": "1.250s"},
                        {
                            "type": "image",
                            "image_ref": {
                                "video_path": "/tmp/video.mp4",
                                "timestamp_sec": 1.25,
                                "sampled_frame_index": 1,
                            },
                        },
                    ],
                },
                {
                    "role": "tool",
                    "name": "seek_evidence",
                    "arguments": {
                        "start_sec": 3.0,
                        "end_sec": 4.0,
                        "moment_id": "m2",
                        "role": "trigger",
                        "query": "physical struggle",
                    },
                    "content": [
                        {"type": "text", "text": "3.500s"},
                        {
                            "type": "image",
                            "image_ref": {
                                "video_path": "/tmp/video.mp4",
                                "timestamp_sec": 3.5,
                                "sampled_frame_index": 2,
                            },
                        },
                    ],
                },
            ],
            "target_response": (
                '<think>verify</think><tool_call>{"name":"verify_hypothesis","arguments":'
                '{"verification_mode":"full_keep_drop","claim":{"existence":"anomaly","category":"assault"},'
                '"selected_window_ids":["w0002"],"selected_evidence_moment_ids":["m1"],'
                '"verification_decision":"insufficient","recommended_action":"continue_search"}}'
                "</tool_call>"
            ),
        }

        package = build_teacher_judge_package(example, topk_frames_per_view=8)
        full_timestamps = [float(item["timestamp_sec"]) for item in package["views"]["full"]["images"]]
        keep_timestamps = [float(item["timestamp_sec"]) for item in package["views"]["keep"]["images"]]
        drop_timestamps = [float(item["timestamp_sec"]) for item in package["views"]["drop"]["images"]]

        self.assertIn(1.25, full_timestamps)
        self.assertIn(3.5, full_timestamps)
        self.assertEqual(keep_timestamps, [1.25])
        self.assertIn(3.5, drop_timestamps)
        self.assertNotIn(1.25, drop_timestamps)


if __name__ == "__main__":
    unittest.main()
