import os
import re
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

    def _write_trainer_state(self, checkpoint_dir: Path, *, epoch: float, num_train_epochs: float = 2.0) -> None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "trainer_state.json").write_text(
            textwrap.dedent(
                f"""\
                {{
                  "epoch": {float(epoch)},
                  "global_step": 100,
                  "num_train_epochs": {float(num_train_epochs)}
                }}
                """
            ),
            encoding="utf-8",
        )

    def _touch_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")

    def _run_pipeline(
        self,
        *,
        sft_root_complete: bool = False,
        sft_checkpoint_step: int | None = None,
        sft_checkpoint_epoch: float | None = None,
        sft_resume_epoch: int | None = None,
        sft_resume_num_train_epochs: float = 2.0,
        sft_eval_epoch_exists: int | None = None,
        raw_rollout_exists: bool = False,
        scored_rollout_exists: bool = False,
        summary_exists: bool = False,
        teacher_model_exists: bool = False,
        extra_env: dict[str, str] | None = None,
    ) -> tuple[subprocess.CompletedProcess[str], str, str]:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            exp_root = base / "exp"
            data_utils_dir = base / "data_utils"
            experiment_base_dir = Path((extra_env or {}).get("EXPERIMENT_BASE_DIR", str(base / "runs")))
            exp_name = str((extra_env or {}).get("EXP_NAME", "")).strip()
            annotation_dir = data_utils_dir
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
            if teacher_model_exists:
                (model_root / "Qwen3-VL-32B-Instruct").mkdir(parents=True, exist_ok=True)

            self._touch_jsonl(annotation_dir / "msad_saver_agent_train.jsonl")
            self._touch_jsonl(annotation_dir / "msad_saver_oracle_sft.jsonl")
            self._touch_jsonl(data_utils_dir / "msad_saver_agent_train.prepared_sft.jsonl")

            if sft_root_complete:
                self._write_complete_checkpoint(sft_output_dir)
            if sft_checkpoint_step is not None:
                self._write_complete_checkpoint(sft_output_dir / f"checkpoint-{sft_checkpoint_step}")
                if sft_checkpoint_epoch is not None:
                    self._write_trainer_state(
                        sft_output_dir / f"checkpoint-{sft_checkpoint_step}",
                        epoch=sft_checkpoint_epoch,
                        num_train_epochs=sft_resume_num_train_epochs,
                    )
            if sft_resume_epoch is not None:
                resume_dir = sft_output_dir / "epoch_resume" / f"epoch_{sft_resume_epoch:03d}"
                self._write_complete_checkpoint(resume_dir)
                self._write_trainer_state(
                    resume_dir,
                    epoch=float(sft_resume_epoch),
                    num_train_epochs=sft_resume_num_train_epochs,
                )
            if sft_eval_epoch_exists is not None:
                eval_metrics_path = sft_output_dir / "rollout_eval" / f"epoch_{sft_eval_epoch_exists:03d}" / "metrics.json"
                eval_metrics_path.parent.mkdir(parents=True, exist_ok=True)
                eval_metrics_path.write_text("{}", encoding="utf-8")

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

                def write_epoch_resume_checkpoint(output_dir: str, epoch_index: int = 1) -> None:
                    if not output_dir:
                        return
                    checkpoint_dir = Path(output_dir) / "epoch_resume" / f"epoch_{int(epoch_index):03d}"
                    write_complete_checkpoint(str(checkpoint_dir))
                    (checkpoint_dir / "trainer_state.json").write_text(
                        "{\\"epoch\\": %s, \\"global_step\\": 100, \\"num_train_epochs\\": 1.0}" % float(epoch_index),
                        encoding="utf-8",
                    )

                def write_rollout_eval_metrics(output_dir: str, resume_path: str) -> None:
                    if not output_dir or not resume_path:
                        return
                    resume_dir = Path(resume_path)
                    epoch_index = 0
                    name = resume_dir.name
                    if name.startswith("epoch_"):
                        try:
                            epoch_index = int(name.split("_", 1)[1])
                        except Exception:
                            epoch_index = 0
                    if epoch_index <= 0:
                        epoch_index = 1
                    metrics_path = Path(output_dir) / "rollout_eval" / f"epoch_{epoch_index:03d}" / "metrics.json"
                    metrics_path.parent.mkdir(parents=True, exist_ok=True)
                    metrics_path.write_text("{}", encoding="utf-8")

                if script_name == "train_saver_sft.py":
                    if "--resume-rollout-eval-only" in args:
                        write_rollout_eval_metrics(arg_value("--output-dir"), arg_value("--resume-from-checkpoint"))
                    else:
                        write_complete_checkpoint(arg_value("--output-dir"))
                        write_epoch_resume_checkpoint(arg_value("--output-dir"))
                        if "--inline-rollout-eval" in args:
                            write_rollout_eval_metrics(arg_value("--output-dir"), str(Path(arg_value("--output-dir")) / "epoch_resume" / "epoch_001"))
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
        self.assertIn("--cea-local-verifier-backend self_teacher", log_text)

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

    def test_full_pipeline_replays_missing_sft_rollout_eval_from_epoch_resume_checkpoint(self):
        result, log_text, sft_output_dir = self._run_pipeline(
            sft_resume_epoch=1,
            sft_resume_num_train_epochs=1.0,
            raw_rollout_exists=True,
            scored_rollout_exists=True,
            summary_exists=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stdout + "\n" + result.stderr)
        self.assertIn("train_saver_sft.py", log_text)
        self.assertIn("--resume-rollout-eval-only", log_text)
        self.assertIn(f"--resume-from-checkpoint {sft_output_dir}/epoch_resume/epoch_001", log_text)
        self.assertNotIn("batch_run_saver_rollout.py", log_text)
        self.assertNotIn("score_saver_rollout.py", log_text)

    def test_full_pipeline_runs_external_eval_after_fresh_sft_training(self):
        result, log_text, sft_output_dir = self._run_pipeline(
            sft_root_complete=False,
            raw_rollout_exists=True,
            scored_rollout_exists=True,
            summary_exists=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stdout + "\n" + result.stderr)
        self.assertEqual(log_text.count("train_saver_sft.py"), 2)
        self.assertIn(f"--output-dir {sft_output_dir}", log_text)
        self.assertNotIn("--inline-rollout-eval", log_text)
        self.assertIn("--resume-rollout-eval-only", log_text)

    def test_full_pipeline_replays_missing_eval_then_resumes_remaining_sft_epochs(self):
        result, log_text, sft_output_dir = self._run_pipeline(
            sft_resume_epoch=1,
            sft_resume_num_train_epochs=2.0,
            raw_rollout_exists=True,
            scored_rollout_exists=True,
            summary_exists=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stdout + "\n" + result.stderr)
        self.assertGreaterEqual(log_text.count("train_saver_sft.py"), 2)
        self.assertIn("--resume-rollout-eval-only", log_text)
        self.assertIn(f"--resume-from-checkpoint {sft_output_dir}/epoch_resume/epoch_001", log_text)

    def test_full_pipeline_resumes_sft_training_from_latest_checkpoint(self):
        result, log_text, sft_output_dir = self._run_pipeline(
            sft_checkpoint_step=640,
            sft_checkpoint_epoch=0.5,
            raw_rollout_exists=True,
            scored_rollout_exists=True,
            summary_exists=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stdout + "\n" + result.stderr)
        self.assertIn("train_saver_sft.py", log_text)
        self.assertIn(f"--resume-from-checkpoint {sft_output_dir}/checkpoint-640", log_text)
        self.assertNotIn("--inline-rollout-eval", log_text)
        self.assertIn("--resume-rollout-eval-only", log_text)

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

    def test_full_pipeline_passes_eval_budget_flags_by_default(self):
        result, log_text, _ = self._run_pipeline(
            sft_root_complete=False,
            raw_rollout_exists=True,
            scored_rollout_exists=True,
            summary_exists=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stdout + "\n" + result.stderr)
        self.assertIn("--eval-max-new-tokens-per-turn 256", log_text)
        self.assertIn("--eval-max-total-images 24", log_text)
        self.assertNotIn("--inline-rollout-eval", log_text)

    def test_full_pipeline_can_reenable_inline_rollout_eval_explicitly(self):
        result, log_text, _ = self._run_pipeline(
            sft_root_complete=False,
            raw_rollout_exists=True,
            scored_rollout_exists=True,
            summary_exists=True,
            extra_env={
                "SFT_INLINE_ROLLOUT_EVAL": "1",
            },
        )

        self.assertEqual(result.returncode, 0, msg=result.stdout + "\n" + result.stderr)
        self.assertIn("--inline-rollout-eval", log_text)

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
        self.assertIn("/data_utils/msad_saver_agent_train.prepared_sft.jsonl", log_text)
        self.assertNotIn("/runs/exp1/train_artifacts/msad_saver_agent_train.prepared_sft.jsonl", log_text)
        self.assertIn(f"--output-dir {sft_output_dir}", log_text)
        self.assertIn("/runs/exp1/rollouts/sft_rollout_eval/rollouts.raw.jsonl", log_text)

    def test_full_pipeline_keeps_cache_summaries_in_data_utils_even_with_experiment_name(self):
        result, log_text, _ = self._run_pipeline(
            sft_root_complete=False,
            raw_rollout_exists=False,
            scored_rollout_exists=False,
            summary_exists=False,
            extra_env={
                "EXP_NAME": "exp1",
                "PROPOSAL_MODEL_PATH": "/tmp/fake_siglip",
                "BUILD_FRAME_CACHE": "1",
                "BUILD_FEATURE_CACHE": "1",
            },
        )

        self.assertEqual(result.returncode, 0, msg=result.stdout + "\n" + result.stderr)
        self.assertIn("/data_utils/frame_cache_summary.json", log_text)
        self.assertIn("/data_utils/feature_cache_summary.json", log_text)
        self.assertNotIn("/runs/exp1/train_artifacts/frame_cache_summary.json", log_text)
        self.assertNotIn("/runs/exp1/train_artifacts/feature_cache_summary.json", log_text)

    def test_full_pipeline_auto_enables_teacher_when_default_teacher_model_exists(self):
        result, log_text, _ = self._run_pipeline(
            sft_root_complete=False,
            raw_rollout_exists=False,
            scored_rollout_exists=False,
            summary_exists=False,
            teacher_model_exists=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stdout + "\n" + result.stderr)
        self.assertIn("Qwen3-VL-32B-Instruct", log_text)
        self.assertIn("--prepared-data", log_text)
        self.assertIn("msad_saver_agent_train.prepared_sft.teacher.jsonl", log_text)
        self.assertIn("score_saver_rollout.py", log_text)
        self.assertIn("--teacher-judge-model-path", log_text)
        self.assertIn("train_saver_rl.py", log_text)
        self.assertIn("--teacher-judge-local-alpha 0.5", log_text)

    def test_full_pipeline_runs_teacher_annotation_and_propagates_teacher_args_when_enabled(self):
        result, log_text, _ = self._run_pipeline(
            sft_root_complete=False,
            raw_rollout_exists=False,
            scored_rollout_exists=False,
            summary_exists=False,
            extra_env={
                "TEACHER_JUDGE_MODEL_PATH": "/tmp/fake_teacher",
                "TEACHER_JUDGE_INPUT_MODE": "multimodal_visual",
                "TEACHER_JUDGE_ATTN_IMPLEMENTATION": "flash_attention_3",
                "RL_TEACHER_JUDGE_LOCAL_ALPHA": "0.75",
                "MAX_TEACHER_DISAGREEMENT_CASES": "7",
            },
        )

        self.assertEqual(result.returncode, 0, msg=result.stdout + "\n" + result.stderr)
        self.assertIn("msad_saver_agent_train.prepared_sft.teacher.jsonl", log_text)
        self.assertIn("--teacher-judge-model-path /tmp/fake_teacher", log_text)
        self.assertIn("--teacher-judge-input-mode multimodal_visual", log_text)
        self.assertIn("--max-teacher-disagreement-cases 7", log_text)
        self.assertIn("--teacher-judge-local-alpha 0.75", log_text)

    def test_full_pipeline_creates_script_log_under_experiment_logs(self):
        result, _, _ = self._run_pipeline(
            sft_root_complete=False,
            raw_rollout_exists=True,
            scored_rollout_exists=True,
            summary_exists=True,
            extra_env={
                "EXP_NAME": "exp1",
            },
        )

        self.assertEqual(result.returncode, 0, msg=result.stdout + "\n" + result.stderr)
        match = re.search(r"script log: ([^\n]+\.log)", result.stdout)
        self.assertIsNotNone(match, msg=result.stdout)
        log_path = Path(match.group(1).strip())
        self.assertIn("/runs/exp1/logs/", str(log_path))


if __name__ == "__main__":
    unittest.main()
