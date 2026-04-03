import os
import re
import stat
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path("/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code")
SFT_SCRIPT = ROOT / "scripts/02_train_sft_with_rollout_eval.sh"
ROLLOUT_SCRIPT = ROOT / "scripts/03_batch_rollout_score_summarize.sh"
RL_SCRIPT = ROOT / "scripts/04_train_rl.sh"


class ExperimentNameScriptTests(unittest.TestCase):
    maxDiff = None

    def _write_executable(self, path: Path, content: str) -> None:
        path.write_text(content, encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    def _touch_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")

    def _build_env(self, *, base: Path, fake_bin: Path, log_path: Path) -> dict[str, str]:
        exp_root = base / "exp"
        experiment_base_dir = base / "runs"
        model_root = base / "models"
        data_utils_dir = base / "data_utils"
        env = os.environ.copy()
        env.update(
            {
                "PATH": f"{fake_bin}:{env['PATH']}",
                "FAKE_CMD_LOG": str(log_path),
                "DATA_ROOT": str(base),
                "EXP_ROOT": str(exp_root),
                "DATA_UTILS_DIR": str(data_utils_dir),
                "EXPERIMENT_BASE_DIR": str(experiment_base_dir),
                "MODEL_ROOT": str(model_root),
                "EXP_NAME": "exp1",
                "NPROC_PER_NODE": "1",
            }
        )
        return env

    def _write_fake_python(self, fake_bin: Path) -> None:
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

            if script_name == "train_saver_sft.py":
                write_complete_checkpoint(arg_value("--output-dir"))
            elif script_name == "annotate_teacher_judge_sft.py":
                write_file(arg_value("--output"), "{}\\n")
            elif script_name == "batch_run_saver_rollout.py":
                write_file(arg_value("--output"), "{}\\n")
            elif script_name == "score_saver_rollout.py":
                write_file(arg_value("--output"), "{}\\n")
            elif script_name == "summarize_saver_scores.py":
                write_file(arg_value("--output"), "{}")

            raise SystemExit(0)
            """
        )
        self._write_executable(fake_bin / "python", fake_python)

    def test_common_experiment_defaults_to_code_ckpt_base_dir(self):
        helper = ROOT / "scripts" / "common_experiment.sh"
        command = textwrap.dedent(
            f"""\
            source "{helper}"
            configure_experiment_layout "{ROOT}" "/tmp/exp_root" "/tmp/exp_root/benchmark_annotations"
            printf '%s\\n' "$RUN_BASE_DIR"
            """
        )
        env = os.environ.copy()
        env["EXP_NAME"] = "exp1"
        env.pop("EXPERIMENT_BASE_DIR", None)
        env["PROMPT_EXP_NAME"] = "0"
        result = subprocess.run(
            ["bash", "-lc", command],
            cwd=str(ROOT),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stdout + "\n" + result.stderr)
        self.assertIn(str(ROOT / "ckpt" / "exp1"), result.stdout)

    def test_common_experiment_exposes_stage_specific_layout_defaults(self):
        helper = ROOT / "scripts" / "common_experiment.sh"
        command = textwrap.dedent(
            f"""\
            source "{helper}"
            EXP_NAME="exp1"
            PROMPT_EXP_NAME=0
            configure_experiment_layout "{ROOT}" "/tmp/exp_root" "/tmp/exp_root/data_utils"
            printf 'SFT=%s\\n' "$DEFAULT_SFT_CHECKPOINT_DIR"
            printf 'RL=%s\\n' "$DEFAULT_RL_CHECKPOINT_DIR"
            printf 'ROLLOUT=%s\\n' "$DEFAULT_ROLLOUT_ROOT"
            printf 'SFT_EVAL=%s\\n' "$DEFAULT_SFT_EVAL_DIR"
            printf 'PIPELINE_LOG=%s\\n' "$DEFAULT_PIPELINE_LOG_DIR"
            """
        )
        result = subprocess.run(
            ["bash", "-lc", command],
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stdout + "\n" + result.stderr)
        self.assertIn("SFT=" + str(ROOT / "ckpt" / "exp1" / "checkpoints" / "sft"), result.stdout)
        self.assertIn("RL=" + str(ROOT / "ckpt" / "exp1" / "checkpoints" / "rl"), result.stdout)
        self.assertIn("ROLLOUT=" + str(ROOT / "ckpt" / "exp1" / "eval" / "batch_rollout"), result.stdout)
        self.assertIn("SFT_EVAL=" + str(ROOT / "ckpt" / "exp1" / "eval" / "sft_epoch_end"), result.stdout)
        self.assertIn("PIPELINE_LOG=" + str(ROOT / "ckpt" / "exp1" / "logs" / "pipeline"), result.stdout)

    def test_configure_script_logging_prefixes_terminal_lines_with_timestamps(self):
        helper = ROOT / "scripts" / "common_experiment.sh"
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"
            command = textwrap.dedent(
                f"""\
                source "{helper}"
                DEFAULT_LOG_DIR="{log_dir}"
                SCRIPT_LOGGING_ENABLED=1
                configure_script_logging "timestamp_test"
                echo "hello timestamp"
                """
            )
            env = os.environ.copy()
            env["PROMPT_EXP_NAME"] = "0"
            result = subprocess.run(
                ["bash", "-lc", command],
                cwd=str(ROOT),
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stdout + "\n" + result.stderr)
            stdout_lines = [line for line in result.stdout.splitlines() if line.strip()]
            self.assertGreaterEqual(len(stdout_lines), 2, msg=result.stdout)
            self.assertRegex(stdout_lines[0], r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] \[main\] script log: .+timestamp_test\.\d{8}_\d{6}\.log$")
            self.assertRegex(stdout_lines[1], r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] hello timestamp$")

    def test_configure_script_logging_does_not_double_prefix_existing_timestamped_lines(self):
        helper = ROOT / "scripts" / "common_experiment.sh"
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"
            command = textwrap.dedent(
                f"""\
                source "{helper}"
                DEFAULT_LOG_DIR="{log_dir}"
                SCRIPT_LOGGING_ENABLED=1
                configure_script_logging "timestamp_passthrough_test"
                echo "[2026-04-03 09:19:58] [main] rollout progress: 115/240 dataset_index=114 video_id=Vandalism_12"
                """
            )
            env = os.environ.copy()
            env["PROMPT_EXP_NAME"] = "0"
            result = subprocess.run(
                ["bash", "-lc", command],
                cwd=str(ROOT),
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stdout + "\n" + result.stderr)
            stdout_lines = [line for line in result.stdout.splitlines() if line.strip()]
            self.assertGreaterEqual(len(stdout_lines), 2, msg=result.stdout)
            self.assertEqual(
                stdout_lines[1],
                "[2026-04-03 09:19:58] [main] rollout progress: 115/240 dataset_index=114 video_id=Vandalism_12",
            )

    def test_sft_script_uses_exp_name_for_prepared_data_and_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            data_utils_dir = base / "data_utils"
            run_base = base / "runs" / "exp1"
            prepared_path = data_utils_dir / "msad_saver_sft_train.jsonl"
            eval_path = data_utils_dir / "msad_saver_runtime_test.jsonl"
            model_dir = base / "models" / "qwen3-vl-8b-Instruct"
            fake_bin = base / "fake_bin"
            fake_bin.mkdir(parents=True, exist_ok=True)
            self._write_fake_python(fake_bin)
            self._touch_jsonl(prepared_path)
            self._touch_jsonl(eval_path)
            model_dir.mkdir(parents=True, exist_ok=True)
            log_path = base / "command.log"

            result = subprocess.run(
                ["bash", str(SFT_SCRIPT)],
                cwd=str(ROOT),
                env=self._build_env(base=base, fake_bin=fake_bin, log_path=log_path),
                text=True,
                capture_output=True,
                check=False,
            )

            log_text = log_path.read_text(encoding="utf-8")
            self.assertEqual(result.returncode, 0, msg=result.stdout + "\n" + result.stderr)
            self.assertIn(str(prepared_path), log_text)
            self.assertIn(f"--eval-data {eval_path}", log_text)
            self.assertIn("--eval-include-splits test", log_text)
            self.assertIn("--inline-rollout-eval", log_text)
            self.assertIn(str(run_base / "checkpoints" / "sft" / "saver_sft_qwen3vl_8b_eval_ddp"), log_text)
            self.assertIn(str(run_base / "logs" / "sft"), log_text)
            self.assertIn(str(run_base / "eval" / "sft_epoch_end"), log_text)

    def test_rollout_script_uses_exp_name_for_model_and_rollout_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            data_utils_dir = base / "data_utils"
            run_base = base / "runs" / "exp1"
            data_path = data_utils_dir / "msad_saver_runtime_test.jsonl"
            model_dir = base / "models" / "qwen3-vl-8b-Instruct"
            fake_bin = base / "fake_bin"
            fake_bin.mkdir(parents=True, exist_ok=True)
            self._write_fake_python(fake_bin)
            self._touch_jsonl(data_path)
            model_dir.mkdir(parents=True, exist_ok=True)
            log_path = base / "command.log"

            result = subprocess.run(
                ["bash", str(ROLLOUT_SCRIPT)],
                cwd=str(ROOT),
                env=self._build_env(base=base, fake_bin=fake_bin, log_path=log_path),
                text=True,
                capture_output=True,
                check=False,
            )

            log_text = log_path.read_text(encoding="utf-8")
            self.assertEqual(result.returncode, 0, msg=result.stdout + "\n" + result.stderr)
            self.assertIn(f"--data {data_path}", log_text)
            self.assertIn("--include-splits test", log_text)
            self.assertIn(str(run_base / "checkpoints" / "sft" / "saver_sft_qwen3vl_8b_eval_ddp"), log_text)
            self.assertIn(str(run_base / "eval" / "batch_rollout" / "sft_rollout_eval" / "rollouts.raw.jsonl"), log_text)
            self.assertIn(str(run_base / "eval" / "batch_rollout" / "sft_rollout_eval" / "summary.json"), log_text)
            self.assertIn(str(run_base / "logs" / "rollout"), log_text)

    def test_rl_script_uses_exp_name_for_model_reference_and_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            data_utils_dir = base / "data_utils"
            run_base = base / "runs" / "exp1"
            data_path = data_utils_dir / "msad_saver_runtime_train.jsonl"
            eval_path = data_utils_dir / "msad_saver_runtime_test.jsonl"
            model_dir = base / "models" / "qwen3-vl-8b-Instruct"
            fake_bin = base / "fake_bin"
            fake_bin.mkdir(parents=True, exist_ok=True)
            self._write_fake_python(fake_bin)
            self._touch_jsonl(data_path)
            self._touch_jsonl(eval_path)
            model_dir.mkdir(parents=True, exist_ok=True)
            log_path = base / "command.log"

            result = subprocess.run(
                ["bash", str(RL_SCRIPT)],
                cwd=str(ROOT),
                env=self._build_env(base=base, fake_bin=fake_bin, log_path=log_path),
                text=True,
                capture_output=True,
                check=False,
            )

            log_text = log_path.read_text(encoding="utf-8")
            expected_sft_dir = run_base / "checkpoints" / "sft" / "saver_sft_qwen3vl_8b_eval_ddp"
            expected_rl_dir = run_base / "checkpoints" / "rl" / "saver_cea_grpo_v1"
            self.assertEqual(result.returncode, 0, msg=result.stdout + "\n" + result.stderr)
            self.assertIn(f"--data {data_path}", log_text)
            self.assertIn(f"--eval-data {eval_path}", log_text)
            self.assertIn("--eval-include-splits test", log_text)
            self.assertIn(f"--model-path {expected_sft_dir}", log_text)
            self.assertIn(f"--reference-model-path {expected_sft_dir}", log_text)
            self.assertIn(f"--output-dir {expected_rl_dir}", log_text)
            self.assertIn(str(run_base / "logs" / "rl"), log_text)

    def test_sft_script_can_materialize_teacher_prepared_data_before_training(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            data_utils_dir = base / "data_utils"
            run_base = base / "runs" / "exp1"
            prepared_path = data_utils_dir / "msad_saver_sft_train.jsonl"
            eval_path = data_utils_dir / "msad_saver_runtime_test.jsonl"
            model_dir = base / "models" / "qwen3-vl-8b-Instruct"
            fake_bin = base / "fake_bin"
            fake_bin.mkdir(parents=True, exist_ok=True)
            self._write_fake_python(fake_bin)
            self._touch_jsonl(prepared_path)
            self._touch_jsonl(eval_path)
            model_dir.mkdir(parents=True, exist_ok=True)
            log_path = base / "command.log"

            env = self._build_env(base=base, fake_bin=fake_bin, log_path=log_path)
            env.update(
                {
                    "TEACHER_JUDGE_MODEL_PATH": "/tmp/fake_teacher",
                    "TEACHER_JUDGE_INPUT_MODE": "multimodal_visual",
                }
            )
            result = subprocess.run(
                ["bash", str(SFT_SCRIPT)],
                cwd=str(ROOT),
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            log_text = log_path.read_text(encoding="utf-8")
            self.assertEqual(result.returncode, 0, msg=result.stdout + "\n" + result.stderr)
            self.assertIn("annotate_teacher_judge_sft.py", log_text)
            self.assertIn("msad_saver_sft_train.teacher.jsonl", log_text)
            self.assertIn("--prepared-data", log_text)

    def test_sft_script_defaults_teacher_annotation_to_auto_mode_and_topk_views(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            data_utils_dir = base / "data_utils"
            run_base = base / "runs" / "exp1"
            prepared_path = data_utils_dir / "msad_saver_sft_train.jsonl"
            eval_path = data_utils_dir / "msad_saver_runtime_test.jsonl"
            model_dir = base / "models" / "qwen3-vl-8b-Instruct"
            fake_bin = base / "fake_bin"
            fake_bin.mkdir(parents=True, exist_ok=True)
            self._write_fake_python(fake_bin)
            self._touch_jsonl(prepared_path)
            self._touch_jsonl(eval_path)
            model_dir.mkdir(parents=True, exist_ok=True)
            log_path = base / "command.log"

            env = self._build_env(base=base, fake_bin=fake_bin, log_path=log_path)
            env.update(
                {
                    "TEACHER_JUDGE_MODEL_PATH": "/tmp/fake_teacher",
                }
            )
            result = subprocess.run(
                ["bash", str(SFT_SCRIPT)],
                cwd=str(ROOT),
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            log_text = log_path.read_text(encoding="utf-8")
            self.assertEqual(result.returncode, 0, msg=result.stdout + "\n" + result.stderr)
            self.assertIn("--input-mode auto", log_text)
            self.assertIn("--topk-frames-per-view 4", log_text)

    def test_rollout_script_passes_teacher_judge_and_summary_controls_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            data_utils_dir = base / "data_utils"
            run_base = base / "runs" / "exp1"
            data_path = data_utils_dir / "msad_saver_runtime_test.jsonl"
            model_dir = base / "models" / "qwen3-vl-8b-Instruct"
            fake_bin = base / "fake_bin"
            fake_bin.mkdir(parents=True, exist_ok=True)
            self._write_fake_python(fake_bin)
            self._touch_jsonl(data_path)
            model_dir.mkdir(parents=True, exist_ok=True)
            log_path = base / "command.log"

            env = self._build_env(base=base, fake_bin=fake_bin, log_path=log_path)
            env.update(
                {
                    "TEACHER_JUDGE_MODEL_PATH": "/tmp/fake_teacher",
                    "TEACHER_JUDGE_INPUT_MODE": "multimodal_visual",
                    "MAX_TEACHER_DISAGREEMENT_CASES": "9",
                }
            )
            result = subprocess.run(
                ["bash", str(ROLLOUT_SCRIPT)],
                cwd=str(ROOT),
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            log_text = log_path.read_text(encoding="utf-8")
            self.assertEqual(result.returncode, 0, msg=result.stdout + "\n" + result.stderr)
            self.assertIn("--teacher-judge-model-path /tmp/fake_teacher", log_text)
            self.assertIn("--teacher-judge-input-mode multimodal_visual", log_text)
            self.assertIn("--teacher-judge-topk-frames-per-view 4", log_text)
            self.assertIn("--max-teacher-disagreement-cases 9", log_text)

    def test_rl_script_passes_teacher_judge_configuration_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            data_utils_dir = base / "data_utils"
            run_base = base / "runs" / "exp1"
            data_path = data_utils_dir / "msad_saver_runtime_train.jsonl"
            eval_path = data_utils_dir / "msad_saver_runtime_test.jsonl"
            model_dir = base / "models" / "qwen3-vl-8b-Instruct"
            fake_bin = base / "fake_bin"
            fake_bin.mkdir(parents=True, exist_ok=True)
            self._write_fake_python(fake_bin)
            self._touch_jsonl(data_path)
            self._touch_jsonl(eval_path)
            model_dir.mkdir(parents=True, exist_ok=True)
            log_path = base / "command.log"

            env = self._build_env(base=base, fake_bin=fake_bin, log_path=log_path)
            env.update(
                {
                    "TEACHER_JUDGE_MODEL_PATH": "/tmp/fake_teacher",
                    "TEACHER_JUDGE_INPUT_MODE": "multimodal_visual",
                    "RL_TEACHER_JUDGE_LOCAL_ALPHA": "0.8",
                }
            )
            result = subprocess.run(
                ["bash", str(RL_SCRIPT)],
                cwd=str(ROOT),
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            log_text = log_path.read_text(encoding="utf-8")
            self.assertEqual(result.returncode, 0, msg=result.stdout + "\n" + result.stderr)
            self.assertIn("--teacher-judge-model-path /tmp/fake_teacher", log_text)
            self.assertIn("--teacher-judge-input-mode multimodal_visual", log_text)
            self.assertIn("--teacher-judge-local-alpha 0.8", log_text)


if __name__ == "__main__":
    unittest.main()
