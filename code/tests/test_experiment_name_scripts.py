import os
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
        env = os.environ.copy()
        env.update(
            {
                "PATH": f"{fake_bin}:{env['PATH']}",
                "FAKE_CMD_LOG": str(log_path),
                "DATA_ROOT": str(base),
                "EXP_ROOT": str(exp_root),
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

    def test_sft_script_uses_exp_name_for_prepared_data_and_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            exp_root = base / "exp"
            run_base = base / "runs" / "exp1"
            prepared_path = run_base / "train_artifacts" / "msad_saver_agent_train.prepared_sft.jsonl"
            eval_path = exp_root / "benchmark_annotations" / "msad_saver_oracle_sft_test.jsonl"
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
            self.assertIn(str(run_base / "checkpoints" / "saver_sft_qwen3vl_8b_eval_ddp"), log_text)

    def test_rollout_script_uses_exp_name_for_model_and_rollout_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            exp_root = base / "exp"
            run_base = base / "runs" / "exp1"
            data_path = exp_root / "benchmark_annotations" / "msad_saver_oracle_sft_test.jsonl"
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
            self.assertIn(str(run_base / "checkpoints" / "saver_sft_qwen3vl_8b_eval_ddp"), log_text)
            self.assertIn(str(run_base / "rollouts" / "sft_rollout_eval" / "rollouts.raw.jsonl"), log_text)
            self.assertIn(str(run_base / "rollouts" / "sft_rollout_eval" / "summary.json"), log_text)

    def test_rl_script_uses_exp_name_for_model_reference_and_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            exp_root = base / "exp"
            run_base = base / "runs" / "exp1"
            data_path = exp_root / "benchmark_annotations" / "msad_saver_agent_train.jsonl"
            eval_path = exp_root / "benchmark_annotations" / "msad_saver_oracle_sft_test.jsonl"
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
            expected_sft_dir = run_base / "checkpoints" / "saver_sft_qwen3vl_8b_eval_ddp"
            expected_rl_dir = run_base / "checkpoints" / "saver_cea_grpo_v1"
            self.assertEqual(result.returncode, 0, msg=result.stdout + "\n" + result.stderr)
            self.assertIn(f"--model-path {expected_sft_dir}", log_text)
            self.assertIn(f"--reference-model-path {expected_sft_dir}", log_text)
            self.assertIn(f"--output-dir {expected_rl_dir}", log_text)


if __name__ == "__main__":
    unittest.main()
