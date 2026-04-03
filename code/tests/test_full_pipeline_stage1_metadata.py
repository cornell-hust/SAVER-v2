import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path("/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code")
SCRIPT = ROOT / "scripts/00_full_pipeline.sh"


class FullPipelineStage1MetadataTests(unittest.TestCase):
    def test_stage1_rebuilds_when_prepared_metadata_sidecars_are_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            data_utils_dir = base / "data_utils"
            data_utils_dir.mkdir(parents=True, exist_ok=True)
            experiment_base_dir = base / "runs"
            model_root = base / "models"
            (model_root / "qwen3-vl-8b-Instruct").mkdir(parents=True, exist_ok=True)

            for name in (
                "msad_saver_runtime_train.jsonl",
                "msad_saver_runtime_test.jsonl",
                "msad_saver_sft_train.jsonl",
            ):
                (data_utils_dir / name).write_text("{}\n", encoding="utf-8")

            fake_bin = base / "fake_bin"
            fake_bin.mkdir(parents=True, exist_ok=True)
            log_path = base / "command.log"

            fake_python = textwrap.dedent(
                """\
                #!/usr/bin/env python3
                import os
                import sys
                from pathlib import Path

                log_path = Path(os.environ["FAKE_CMD_LOG"])
                argv = sys.argv[1:]
                if argv:
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(" ".join(argv) + "\\n")

                if not argv:
                    raise SystemExit(0)

                script_name = Path(argv[0]).name
                args = argv[1:]

                def arg_value(flag: str) -> str:
                    try:
                        return args[args.index(flag) + 1]
                    except ValueError:
                        return ""

                def write_file(path_str: str, content: str) -> None:
                    if not path_str:
                        return
                    path = Path(path_str)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(content, encoding="utf-8")

                def write_complete_checkpoint(path_str: str) -> None:
                    if not path_str:
                        return
                    path = Path(path_str)
                    path.mkdir(parents=True, exist_ok=True)
                    (path / "adapter_config.json").write_text("{}", encoding="utf-8")
                    (path / "adapter_model.safetensors").write_text("", encoding="utf-8")
                    (path / "preprocessor_config.json").write_text("{}", encoding="utf-8")

                if script_name == "build_saver_data.py":
                    runtime_train = arg_value("--runtime-train-output")
                    runtime_test = arg_value("--runtime-test-output")
                    prepared = arg_value("--sft-train-output")
                    write_file(runtime_train, "{}\\n")
                    write_file(runtime_test, "{}\\n")
                    write_file(prepared, "{}\\n")
                    write_file(prepared + ".meta.json", "{\\\"schema_version\\\": 1, \\\"preview\\\": {}, \\\"prompt\\\": {}}")
                elif script_name == "train_saver_sft.py":
                    write_complete_checkpoint(arg_value("--output-dir"))
                elif script_name == "batch_run_saver_rollout.py":
                    write_file(arg_value("--output"), "{}\\n")
                elif script_name == "score_saver_rollout.py":
                    write_file(arg_value("--output"), "{}\\n")
                elif script_name == "summarize_saver_scores.py":
                    write_file(arg_value("--output"), "{}")
                elif script_name == "train_saver_rl.py":
                    write_complete_checkpoint(arg_value("--output-dir"))

                raise SystemExit(0)
                """
            )
            fake_python_path = fake_bin / "python"
            fake_python_path.write_text(fake_python, encoding="utf-8")
            fake_python_path.chmod(0o755)

            env = os.environ.copy()
            env.update(
                {
                    "PATH": f"{fake_bin}:{env['PATH']}",
                    "FAKE_CMD_LOG": str(log_path),
                    "DATA_ROOT": str(base),
                    "EXP_ROOT": str(base / "exp"),
                    "DATA_UTILS_DIR": str(data_utils_dir),
                    "EXPERIMENT_BASE_DIR": str(experiment_base_dir),
                    "MODEL_ROOT": str(model_root),
                    "SFT_NPROC_PER_NODE": "1",
                    "RL_NPROC_PER_NODE": "1",
                    "BUILD_FRAME_CACHE": "0",
                    "BUILD_FEATURE_CACHE": "0",
                    "TEACHER_JUDGE_ENABLE": "0",
                }
            )

            result = subprocess.run(
                ["bash", str(SCRIPT)],
                cwd=str(ROOT),
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            log_text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
            self.assertEqual(result.returncode, 0, msg=result.stdout + "\n" + result.stderr)
            self.assertIn("build_saver_data.py", log_text)
            self.assertTrue((data_utils_dir / "msad_saver_sft_train.jsonl.meta.json").exists())


if __name__ == "__main__":
    unittest.main()
