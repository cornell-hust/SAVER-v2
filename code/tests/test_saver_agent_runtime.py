import sys
import re
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch


ROOT = Path("/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


try:
    from saver_agent.runtime import (
        distributed_runtime_from_env,
        resolve_inference_device_map,
        resolve_shard_spec,
        runtime_log,
        shard_sequence,
        sharded_output_path,
    )
except ModuleNotFoundError:
    distributed_runtime_from_env = None
    resolve_inference_device_map = None
    resolve_shard_spec = None
    runtime_log = None
    shard_sequence = None
    sharded_output_path = None


class SaverAgentRuntimeTests(unittest.TestCase):
    def test_distributed_runtime_from_env_defaults_to_single_process(self):
        self.assertIsNotNone(distributed_runtime_from_env, "saver_agent/runtime.py is missing")

        runtime = distributed_runtime_from_env({})

        self.assertEqual(runtime.rank, 0)
        self.assertEqual(runtime.world_size, 1)
        self.assertEqual(runtime.local_rank, 0)
        self.assertFalse(runtime.is_distributed)
        self.assertTrue(runtime.is_main_process)

    def test_distributed_runtime_from_env_reads_torchrun_variables(self):
        self.assertIsNotNone(distributed_runtime_from_env, "saver_agent/runtime.py is missing")

        runtime = distributed_runtime_from_env({"RANK": "2", "WORLD_SIZE": "4", "LOCAL_RANK": "2"})

        self.assertEqual(runtime.rank, 2)
        self.assertEqual(runtime.world_size, 4)
        self.assertEqual(runtime.local_rank, 2)
        self.assertTrue(runtime.is_distributed)
        self.assertFalse(runtime.is_main_process)

    def test_resolve_shard_spec_uses_explicit_cli_values(self):
        self.assertIsNotNone(resolve_shard_spec, "resolve_shard_spec is missing")

        runtime = distributed_runtime_from_env({"RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0"})
        shard_spec = resolve_shard_spec(num_shards=4, shard_index=1, runtime=runtime)

        self.assertEqual(shard_spec.num_shards, 4)
        self.assertEqual(shard_spec.shard_index, 1)
        self.assertTrue(shard_spec.is_sharded)

    def test_resolve_shard_spec_falls_back_to_distributed_runtime(self):
        self.assertIsNotNone(resolve_shard_spec, "resolve_shard_spec is missing")

        runtime = distributed_runtime_from_env({"RANK": "3", "WORLD_SIZE": "4", "LOCAL_RANK": "1"})
        shard_spec = resolve_shard_spec(runtime=runtime)

        self.assertEqual(shard_spec.num_shards, 4)
        self.assertEqual(shard_spec.shard_index, 3)

    def test_shard_sequence_uses_strided_partitioning(self):
        self.assertIsNotNone(shard_sequence, "shard_sequence is missing")

        values = list(range(10))

        self.assertEqual(shard_sequence(values, num_shards=3, shard_index=0), [0, 3, 6, 9])
        self.assertEqual(shard_sequence(values, num_shards=3, shard_index=1), [1, 4, 7])
        self.assertEqual(shard_sequence(values, num_shards=3, shard_index=2), [2, 5, 8])

    def test_sharded_output_path_adds_rank_suffix_for_files(self):
        self.assertIsNotNone(sharded_output_path, "sharded_output_path is missing")

        output_path = sharded_output_path("/tmp/rollouts.jsonl", num_shards=4, shard_index=1)

        self.assertEqual(output_path, Path("/tmp/rollouts.shard01-of-04.jsonl"))

    def test_sharded_output_path_creates_subdirectory_for_directory_outputs(self):
        self.assertIsNotNone(sharded_output_path, "sharded_output_path is missing")

        output_path = sharded_output_path("/tmp/rollouts_dir", num_shards=4, shard_index=2)

        self.assertEqual(output_path, Path("/tmp/rollouts_dir/shard02-of-04"))

    def test_resolve_inference_device_map_pins_each_torchrun_rank_to_its_local_gpu(self):
        self.assertIsNotNone(resolve_inference_device_map, "resolve_inference_device_map is missing")

        runtime = distributed_runtime_from_env({"RANK": "1", "WORLD_SIZE": "4", "LOCAL_RANK": "1"})
        device_map = resolve_inference_device_map("auto", runtime=runtime)

        self.assertEqual(device_map, {"": 1})

    def test_resolve_inference_device_map_falls_back_to_gpu_zero_when_local_rank_exceeds_visible_devices(self):
        self.assertIsNotNone(resolve_inference_device_map, "resolve_inference_device_map is missing")

        runtime = distributed_runtime_from_env({"RANK": "2", "WORLD_SIZE": "4", "LOCAL_RANK": "2"})
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.device_count", return_value=2):
            device_map = resolve_inference_device_map("auto", runtime=runtime)

        self.assertEqual(device_map, {"": 0})

    def test_runtime_log_prefixes_timestamp_before_rank_prefix(self):
        self.assertIsNotNone(runtime_log, "runtime_log is missing")

        runtime = distributed_runtime_from_env({"RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0"})
        buffer = StringIO()
        with redirect_stdout(buffer):
            runtime_log("hello world", runtime=runtime)

        output = buffer.getvalue().strip()
        self.assertRegex(output, r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] \[main\] hello world$")

    def test_runtime_log_does_not_double_prefix_preformatted_runtime_lines(self):
        self.assertIsNotNone(runtime_log, "runtime_log is missing")

        runtime = distributed_runtime_from_env({"RANK": "0", "WORLD_SIZE": "4", "LOCAL_RANK": "0"})
        message = "[2026-04-02 17:16:27] [rank 0/4] rollout eval progress: 16/60 video_id=People_falling_32"
        buffer = StringIO()
        with redirect_stdout(buffer):
            runtime_log(message, runtime=runtime)

        output = buffer.getvalue().strip()
        self.assertEqual(output, message)


if __name__ == "__main__":
    unittest.main()
