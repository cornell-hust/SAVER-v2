import os
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path("/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code")
SCRIPT = ROOT / "scripts/00_full_pipeline.sh"


class FullPipelineScriptTests(unittest.TestCase):
    maxDiff = None

    def _write_executable(self, path: Path, content: str) -> None:
        path.write_text(content, encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    def _write_complete_checkpoint(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "adapter_config.json").write_text("{}", encoding="utf-8")
        (path / "adapter_model.safetensors").write_text("", encoding="utf-8")
        (path / "preprocessor_config.json").write_text("{}", encoding="utf-8")

    def _touch_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")

    def _run_pipeline(
        self,
        *,
        sft_root_complete: bool = False,
        sft_checkpoint_step: int | None = None,
        raw_rollout_exists: bool = False,
        scored_rollout_exists: bool = False,
        summary_exists: bool = False,
        extra_env: dict[str, str] | None = None,
    ) -> tuple[subprocess.CompletedProcess[str], str, str]:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            exp_root = base / "exp"
            experiment_base_dir = Path((extra_env or {}).get("EXPERIMENT_BASE_DIR", str(base / "runs")))
            exp_name = str((extra_env or {}).get("EXP_NAME", "")).strip()
            annotation_dir = exp_root / "benchmark_annotations"
            if exp_name:
                run_base_dir = experiment_base_dir / exp_name
                artifact_dir = run_base_dir / "train_artifacts"
                checkpoint_dir = run_base_dir / "checkpoints"
                rollout_dir = run_base_dir / "rollouts" / "sft_rollout_eval"
            else:
                artifact_dir = exp_root / "train_artifacts"
                checkpoint_dir = exp_root / "checkpoints"
                rollout_dir = exp_root / "rollouts" / "sft_rollout_eval"
            model_root = base / "models"
            sft_output_dir = checkpoint_dir / "saver_sft_qwen3vl_8b_eval_ddp"
            raw_rollout_output = rollout_dir / "rollouts.raw.jsonl"
            scored_rollout_output = rollout_dir / "rollouts.scored.jsonl"
            rollout_summary_output = rollout_dir / "summary.json"

            annotation_dir.mkdir(parents=True, exist_ok=True)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            rollout_dir.mkdir(parents=True, exist_ok=True)
            (model_root / "qwen3-vl-8b-Instruct").mkdir(parents=True, exist_ok=True)

            self._touch_jsonl(annotation_dir / "msad_saver_agent_train.jsonl")
            self._touch_jsonl(annotation_dir / "msad_saver_oracle_sft.jsonl")
            self._touch_jsonl(artifact_dir / "msad_saver_agent_train.prepared_sft.jsonl")

            if sft_root_complete:
                self._write_complete_checkpoint(sft_output_dir)
            if sft_checkpoint_step is not None:
                self._write_complete_checkpoint(sft_output_dir / f"checkpoint-{sft_checkpoint_step}")

            if raw_rollout_exists:
                self._touch_jsonl(raw_rollout_output)
            if scored_rollout_exists:
                self._touch_jsonl(scored_rollout_output)
            if summary_exists:
                rollout_summary_output.write_text("{}", encoding="utf-8")

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

            env = os.environ.copy()
            env.update(
                {
                    "PATH": f"{fake_bin}:{env['PATH']}",
                    "FAKE_CMD_LOG": str(log_path),
                    "DATA_ROOT": str(base),
                    "EXP_ROOT": str(exp_root),
                    "EXPERIMENT_BASE_DIR": str(experiment_base_dir),
                    "MODEL_ROOT": str(model_root),
                    "SFT_NPROC_PER_NODE": "1",
                    "RL_NPROC_PER_NODE": "1",
                }
            )
            if extra_env:
                env.update(extra_env)

            result = subprocess.run(
                ["bash", str(SCRIPT)],
                cwd=str(ROOT),
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            return result, log_path.read_text(encoding="utf-8"), str(sft_output_dir)

    def test_full_pipeline_skips_sft_and_stage3_when_prior_artifacts_are_complete(self):
        result, log_text, sft_output_dir = self._run_pipeline(
            sft_checkpoint_step=1200,
            raw_rollout_exists=True,
            scored_rollout_exists=True,
            summary_exists=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stdout + "\n" + result.stderr)
        self.assertNotIn("train_saver_sft.py", log_text)
        self.assertNotIn("batch_run_saver_rollout.py", log_text)
        self.assertNotIn("score_saver_rollout.py", log_text)
        self.assertNotIn("summarize_saver_scores.py", log_text)
        self.assertIn("train_saver_rl.py", log_text)
        self.assertIn(f"--model-path {sft_output_dir}/checkpoint-1200", log_text)
        self.assertIn(f"--reference-model-path {sft_output_dir}/checkpoint-1200", log_text)

    def test_full_pipeline_skips_sft_but_resumes_stage3_before_rl_when_rollout_artifacts_missing(self):
        result, log_text, sft_output_dir = self._run_pipeline(
            sft_checkpoint_step=800,
            raw_rollout_exists=False,
            scored_rollout_exists=False,
            summary_exists=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stdout + "\n" + result.stderr)
        self.assertNotIn("train_saver_sft.py", log_text)
        self.assertIn("batch_run_saver_rollout.py", log_text)
        self.assertIn("score_saver_rollout.py", log_text)
        self.assertIn("summarize_saver_scores.py", log_text)
        self.assertIn("train_saver_rl.py", log_text)
        self.assertLess(log_text.index("batch_run_saver_rollout.py"), log_text.index("score_saver_rollout.py"))
        self.assertLess(log_text.index("score_saver_rollout.py"), log_text.index("summarize_saver_scores.py"))
        self.assertLess(log_text.index("summarize_saver_scores.py"), log_text.index("train_saver_rl.py"))
        self.assertIn(f"--model-path {sft_output_dir}/checkpoint-800", log_text)

    def test_full_pipeline_passes_sft_text_budget_flags_when_overridden(self):
        result, log_text, _ = self._run_pipeline(
            sft_root_complete=False,
            raw_rollout_exists=True,
            scored_rollout_exists=True,
            summary_exists=True,
            extra_env={
                "SFT_MAX_SEQ_LENGTH": "6144",
                "SFT_KEEP_RECENT_TEXT_MESSAGES": "16",
            },
        )

        self.assertEqual(result.returncode, 0, msg=result.stdout + "\n" + result.stderr)
        self.assertIn("train_saver_sft.py", log_text)
        self.assertIn("--max-seq-length 6144", log_text)
        self.assertIn("--keep-recent-text-messages 16", log_text)

    def test_full_pipeline_builds_caches_and_passes_proposal_runtime_when_enabled(self):
        result, log_text, _ = self._run_pipeline(
            sft_root_complete=False,
            raw_rollout_exists=False,
            scored_rollout_exists=False,
            summary_exists=False,
            extra_env={
                "PROPOSAL_MODEL_PATH": "/tmp/fake_siglip",
                "BUILD_FRAME_CACHE": "1",
                "BUILD_FEATURE_CACHE": "1",
            },
        )

        self.assertEqual(result.returncode, 0, msg=result.stdout + "\n" + result.stderr)
        self.assertIn("build_frame_cache.py", log_text)
        self.assertIn("build_feature_cache.py", log_text)
        self.assertIn("--proposal-model-path /tmp/fake_siglip", log_text)
        self.assertLess(log_text.index("build_frame_cache.py"), log_text.index("build_feature_cache.py"))
        self.assertLess(log_text.index("build_feature_cache.py"), log_text.index("train_saver_sft.py"))
        self.assertLess(log_text.index("train_saver_sft.py"), log_text.index("batch_run_saver_rollout.py"))

    def test_full_pipeline_uses_experiment_name_to_relocate_outputs(self):
        result, log_text, sft_output_dir = self._run_pipeline(
            sft_root_complete=False,
            raw_rollout_exists=False,
            scored_rollout_exists=False,
            summary_exists=False,
            extra_env={
                "EXP_NAME": "exp1",
            },
        )

        self.assertEqual(result.returncode, 0, msg=result.stdout + "\n" + result.stderr)
        self.assertIn("/runs/exp1/train_artifacts/msad_saver_agent_train.prepared_sft.jsonl", log_text)
        self.assertIn(f"--output-dir {sft_output_dir}", log_text)
        self.assertIn("/runs/exp1/rollouts/sft_rollout_eval/rollouts.raw.jsonl", log_text)
        self.assertIn("/runs/exp1/checkpoints/saver_cea_grpo_v1", log_text)


if __name__ == "__main__":
    unittest.main()
