import io
import json
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import types

import cv2
import numpy as np
import torch


ROOT = Path("/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


try:
    import saver_agent.training as training
except ModuleNotFoundError:
    training = None

from saver_agent.runtime import DistributedRuntime


class SaverAgentTrainingTests(unittest.TestCase):
    class _FakeTokenizer:
        def __init__(self):
            self.truncation_side = "right"
            self.pad_token_id = 0

        def __call__(self, text, add_special_tokens=False, return_attention_mask=False):
            tokens = str(text).split()
            return {"input_ids": [list(range(1, len(tokens) + 1))]}

    class _FakeProcessor:
        def __init__(self):
            self.tokenizer = SaverAgentTrainingTests._FakeTokenizer()

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            parts = []
            for message in messages:
                parts.append(str(message.get("role") or ""))
                for item in message.get("content", []):
                    if item.get("type") == "text":
                        parts.append(str(item.get("text") or ""))
                    elif item.get("type") == "image":
                        parts.append("<image>")
            if add_generation_prompt:
                parts.extend(["assistant", "<generation_prompt>"])
            return " ".join(part for part in parts if part)

        def __call__(
            self,
            *,
            text,
            padding=False,
            return_tensors="pt",
            truncation=False,
            max_length=None,
            images=None,
            videos=None,
        ):
            tokens = str(text).split()
            if truncation and max_length is not None and len(tokens) > int(max_length):
                if self.tokenizer.truncation_side == "left":
                    tokens = tokens[-int(max_length) :]
                else:
                    tokens = tokens[: int(max_length)]
            input_ids = torch.tensor([list(range(1, len(tokens) + 1))], dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    class _CountingProcessor(_FakeProcessor):
        def __init__(self):
            super().__init__()
            self.encode_call_count = 0
            self.template_call_count = 0

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            self.template_call_count += 1
            return super().apply_chat_template(messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt)

        def __call__(
            self,
            *,
            text,
            padding=False,
            return_tensors="pt",
            truncation=False,
            max_length=None,
            images=None,
            videos=None,
        ):
            self.encode_call_count += 1
            return super().__call__(
                text=text,
                padding=padding,
                return_tensors=return_tensors,
                truncation=truncation,
                max_length=max_length,
                images=images,
                videos=videos,
            )

    class _StrictMultimodalProcessor(_FakeProcessor):
        def __call__(
            self,
            *,
            text,
            padding=False,
            return_tensors="pt",
            truncation=False,
            max_length=None,
            images=None,
            videos=None,
        ):
            tokens = str(text).split()
            original_image_count = sum(1 for token in tokens if token == "<image>")
            if truncation and max_length is not None and len(tokens) > int(max_length):
                if self.tokenizer.truncation_side == "left":
                    tokens = tokens[-int(max_length) :]
                else:
                    tokens = tokens[: int(max_length)]
            retained_image_count = sum(1 for token in tokens if token == "<image>")
            if (images or videos) and retained_image_count != original_image_count:
                raise ValueError(
                    "Mismatch in `image` token count between text and `input_ids`. "
                    "Likely due to `truncation='max_length'`."
                )
            input_ids = torch.tensor([list(range(1, len(tokens) + 1))], dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    @staticmethod
    def _write_test_video(path: Path, *, num_frames: int = 4, fps: float = 4.0) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, fps, (32, 32))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open test video writer for {path}")
        for frame_id in range(num_frames):
            frame = np.full((32, 32, 3), (frame_id * 40) % 255, dtype=np.uint8)
            writer.write(frame)
        writer.release()

    def test_compute_masked_forward_kl_is_zero_for_identical_logits(self):
        self.assertIsNotNone(training, "saver_agent.training module is missing")

        logits = torch.tensor(
            [[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]],
            dtype=torch.float32,
        )
        labels = torch.tensor([[-100, 1, 1]], dtype=torch.long)

        kl = training.compute_masked_forward_kl(
            policy_logits=logits,
            reference_logits=logits.clone(),
            labels=labels,
        )

        self.assertAlmostEqual(float(kl), 0.0, places=6)

    def test_compute_masked_forward_kl_ignores_prompt_only_positions(self):
        self.assertIsNotNone(training, "saver_agent.training module is missing")

        policy_logits = torch.tensor(
            [[[0.0, 3.0], [0.0, 3.0], [0.0, 3.0]]],
            dtype=torch.float32,
        )
        reference_logits = policy_logits.clone()
        reference_logits[0, 0] = torch.tensor([3.0, 0.0], dtype=torch.float32)
        labels = torch.tensor([[-100, -100, 1]], dtype=torch.long)

        kl = training.compute_masked_forward_kl(
            policy_logits=policy_logits,
            reference_logits=reference_logits,
            labels=labels,
        )

        self.assertAlmostEqual(float(kl), 0.0, places=6)

    def test_compute_masked_forward_kl_is_positive_on_response_tokens(self):
        self.assertIsNotNone(training, "saver_agent.training module is missing")

        policy_logits = torch.tensor(
            [[[0.0, 3.0], [0.0, 3.0], [0.0, 3.0]]],
            dtype=torch.float32,
        )
        reference_logits = policy_logits.clone()
        reference_logits[0, 1] = torch.tensor([3.0, 0.0], dtype=torch.float32)
        labels = torch.tensor([[-100, -100, 1]], dtype=torch.long)

        kl = training.compute_masked_forward_kl(
            policy_logits=policy_logits,
            reference_logits=reference_logits,
            labels=labels,
        )

        self.assertGreater(float(kl), 0.0)

    def test_compute_masked_response_log_probs_averages_only_response_tokens(self):
        self.assertIsNotNone(training, "saver_agent.training module is missing")

        logits = torch.tensor(
            [[[0.0, 0.0], [0.0, 2.0], [2.0, 0.0]]],
            dtype=torch.float32,
        )
        labels = torch.tensor([[-100, 1, 0]], dtype=torch.long)

        log_probs = training.compute_masked_response_log_probs(
            logits=logits,
            labels=labels,
        )

        expected = torch.log_softmax(logits[:, :-1, :], dim=-1)
        expected_value = (expected[0, 0, 1] + expected[0, 1, 0]) / 2.0
        self.assertAlmostEqual(float(log_probs), float(expected_value), places=6)

    def test_compute_masked_response_token_log_probs_returns_tokenwise_values(self):
        self.assertIsNotNone(training, "saver_agent.training module is missing")

        logits = torch.tensor(
            [[[0.0, 0.0], [0.0, 2.0], [2.0, 0.0]]],
            dtype=torch.float32,
        )
        labels = torch.tensor([[-100, 1, 0]], dtype=torch.long)

        token_log_probs, response_mask = training.compute_masked_response_token_log_probs(
            logits=logits,
            labels=labels,
        )

        expected = torch.log_softmax(logits[:, :-1, :], dim=-1)
        self.assertTrue(torch.equal(response_mask, torch.tensor([[True, True]])))
        self.assertAlmostEqual(float(token_log_probs[0, 0]), float(expected[0, 0, 1]), places=6)
        self.assertAlmostEqual(float(token_log_probs[0, 1]), float(expected[0, 1, 0]), places=6)

    def test_create_trainer_applies_dataloader_knobs(self):
        self.assertIsNotNone(training, "saver_agent.training module is missing")

        class _FakeTrainingArguments:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        class _FakeTrainerCallback:
            pass

        class _FakeTrainer:
            def __init__(self, *args, **kwargs):
                self.args = kwargs["args"]
                self.train_dataset = kwargs["train_dataset"]
                self.data_collator = kwargs["data_collator"]

            def add_callback(self, callback):
                return callback

        class _ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 4)

            def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
                batch, seq = input_ids.shape
                logits = torch.zeros(batch, seq, 4, dtype=torch.float32)
                return SimpleNamespace(loss=torch.tensor(0.0, requires_grad=True), logits=logits)

        fake_transformers = SimpleNamespace(
            Trainer=_FakeTrainer,
            TrainingArguments=_FakeTrainingArguments,
            TrainerCallback=_FakeTrainerCallback,
        )
        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            trainer = training.create_trainer(
                model=_ToyModel(),
                processor=self._FakeProcessor(),
                train_dataset=training.WeightedExampleDataset(
                    [
                        {
                            "messages": [
                                {"role": "system", "content": [{"type": "text", "text": "system"}]},
                                {"role": "user", "content": [{"type": "text", "text": "user"}]},
                            ],
                            "target_response": "<answer>{}</answer>",
                        }
                    ]
                ),
                output_dir="/tmp/saver_training_test",
                learning_rate=1e-5,
                num_train_epochs=1.0,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                logging_steps=1,
                save_steps=10,
                save_total_limit=1,
                warmup_ratio=0.0,
                weight_decay=0.0,
                max_grad_norm=1.0,
                bf16=False,
                fp16=False,
                dataloader_num_workers=3,
                dataloader_prefetch_factor=4,
                dataloader_persistent_workers=True,
            )

        self.assertEqual(trainer.args.dataloader_num_workers, 3)
        self.assertEqual(trainer.args.dataloader_prefetch_factor, 4)
        self.assertTrue(trainer.args.dataloader_persistent_workers)

    def test_rollout_eval_callback_only_saves_epoch_resume_checkpoint(self):
        self.assertIsNotNone(training, "saver_agent.training module is missing")

        rollout_eval_config = training.RolloutEvaluationConfig(data_path="/tmp/eval.jsonl")
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.TrainerCallback = object
        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            callback = training._build_rollout_eval_callback(
                processor=object(),
                rollout_eval_config=rollout_eval_config,
            )
        callback.runtime = SimpleNamespace(
            is_main_process=True,
            is_distributed=False,
            rank=0,
            world_size=1,
            local_rank=0,
        )

        class _FakeModel:
            def __init__(self):
                self.training = True
                self.eval_calls = 0
                self.train_calls = 0

            def eval(self):
                self.training = False
                self.eval_calls += 1

            def train(self):
                self.training = True
                self.train_calls += 1

        model = _FakeModel()
        args = SimpleNamespace(output_dir="/tmp/sft_out")
        state = SimpleNamespace(epoch=1.0, global_step=12)
        control = SimpleNamespace()
        captured = {}

        def _fake_save(**kwargs):
            captured["epoch_index"] = kwargs["epoch_index"]
            captured["output_dir"] = kwargs["output_dir"]
            captured["optimizer"] = kwargs["optimizer"]
            captured["lr_scheduler"] = kwargs["lr_scheduler"]
            return Path("/tmp/sft_out/epoch_resume/epoch_001")

        with patch(
            "saver_agent.training.save_sft_epoch_resume_checkpoint",
            side_effect=_fake_save,
        ), patch(
            "saver_agent.qwen_policy.QwenGenerationPolicy.from_components",
            side_effect=AssertionError("in-process rollout eval should not build a policy"),
        ), patch(
            "saver_agent.training.run_rollout_eval_with_policy",
            side_effect=AssertionError("in-process rollout eval should be deferred to a separate process"),
        ), patch(
            "saver_agent.training.gc.collect",
            return_value=0,
        ), patch(
            "torch.cuda.is_available",
            return_value=False,
        ):
            returned = callback.on_epoch_end(
                args,
                state,
                control,
                model=model,
                optimizer="optimizer_state",
                lr_scheduler="scheduler_state",
            )

        self.assertIs(returned, control)
        self.assertEqual(captured["epoch_index"], 1)
        self.assertEqual(captured["output_dir"], "/tmp/sft_out")
        self.assertEqual(captured["optimizer"], "optimizer_state")
        self.assertEqual(captured["lr_scheduler"], "scheduler_state")
        self.assertEqual(model.eval_calls, 1)
        self.assertEqual(model.train_calls, 1)

    def test_rollout_eval_callback_runs_inline_eval_when_enabled(self):
        self.assertIsNotNone(training, "saver_agent.training module is missing")

        rollout_eval_config = training.RolloutEvaluationConfig(
            data_path="/tmp/eval.jsonl",
            inline_rollout_eval=True,
            policy_max_new_tokens=77,
            max_total_images=5,
        )
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.TrainerCallback = object
        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            callback = training._build_rollout_eval_callback(
                processor="processor_stub",
                rollout_eval_config=rollout_eval_config,
            )
        callback.runtime = SimpleNamespace(
            is_main_process=True,
            is_distributed=False,
            rank=0,
            world_size=1,
            local_rank=0,
        )

        class _FakeModel:
            def __init__(self):
                self.training = True
                self.eval_calls = 0
                self.train_calls = 0

            def eval(self):
                self.training = False
                self.eval_calls += 1

            def train(self):
                self.training = True
                self.train_calls += 1

        model = _FakeModel()
        args = SimpleNamespace(output_dir="/tmp/sft_out")
        state = SimpleNamespace(epoch=1.0, global_step=12)
        control = SimpleNamespace()
        captured = {}

        def _fake_save(**kwargs):
            captured["epoch_index"] = kwargs["epoch_index"]
            return Path("/tmp/sft_out/epoch_resume/epoch_001")

        def _fake_policy_from_components(**kwargs):
            captured["policy_kwargs"] = dict(kwargs)
            return "inline_policy"

        def _fake_run_rollout_eval_with_policy(policy, **kwargs):
            captured["policy"] = policy
            captured["eval_kwargs"] = dict(kwargs)
            return {"temporal_miou": 0.42}

        with patch(
            "saver_agent.training.save_sft_epoch_resume_checkpoint",
            side_effect=_fake_save,
        ), patch(
            "saver_agent.qwen_policy.QwenGenerationPolicy.from_components",
            side_effect=_fake_policy_from_components,
        ), patch(
            "saver_agent.training.run_rollout_eval_with_policy",
            side_effect=_fake_run_rollout_eval_with_policy,
        ), patch(
            "saver_agent.training.gc.collect",
            return_value=0,
        ), patch(
            "torch.cuda.is_available",
            return_value=False,
        ):
            returned = callback.on_epoch_end(
                args,
                state,
                control,
                model=model,
                optimizer="optimizer_state",
                lr_scheduler="scheduler_state",
            )

        self.assertIs(returned, control)
        self.assertEqual(captured["epoch_index"], 1)
        self.assertEqual(captured["policy"], "inline_policy")
        self.assertEqual(captured["policy_kwargs"]["model"], model)
        self.assertEqual(captured["policy_kwargs"]["processor"], "processor_stub")
        self.assertEqual(captured["policy_kwargs"]["max_new_tokens"], 77)
        self.assertEqual(captured["policy_kwargs"]["max_total_images"], 5)
        self.assertFalse(captured["policy_kwargs"]["do_sample"])
        self.assertEqual(captured["eval_kwargs"]["output_dir"], "/tmp/sft_out")
        self.assertEqual(captured["eval_kwargs"]["epoch_index"], 1)
        self.assertEqual(captured["eval_kwargs"]["epoch_value"], 1.0)
        self.assertIs(captured["eval_kwargs"]["rollout_eval_config"], rollout_eval_config)
        self.assertEqual(model.eval_calls, 1)
        self.assertEqual(model.train_calls, 1)

    def test_run_rollout_eval_from_checkpoint_uses_rank_local_policy_device_map(self):
        self.assertIsNotNone(training, "saver_agent.training module is missing")

        captured = {}
        runtime = DistributedRuntime(rank=2, world_size=4, local_rank=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "epoch_resume" / "epoch_001"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            rollout_eval_config = training.RolloutEvaluationConfig(
                data_path="/tmp/eval.jsonl",
                policy_max_new_tokens=77,
                max_total_images=5,
                max_image_side=640,
                max_image_pixels=307200,
            )

            def _fake_from_pretrained(
                checkpoint_path,
                *,
                torch_dtype,
                device_map,
                attn_implementation,
                max_new_tokens,
                max_total_images,
                max_image_side,
                max_image_pixels,
                do_sample,
            ):
                captured["checkpoint_path"] = str(checkpoint_path)
                captured["torch_dtype"] = torch_dtype
                captured["device_map"] = device_map
                captured["attn_implementation"] = attn_implementation
                captured["max_new_tokens"] = max_new_tokens
                captured["max_total_images"] = max_total_images
                captured["max_image_side"] = max_image_side
                captured["max_image_pixels"] = max_image_pixels
                captured["do_sample"] = do_sample
                return "policy_stub"

            with patch(
                "saver_agent.training.load_qwen_model_and_processor",
                side_effect=AssertionError("recovery eval should load policy directly onto the rank-local inference device"),
            ), patch(
                "saver_agent.qwen_policy.QwenGenerationPolicy.from_pretrained",
                side_effect=_fake_from_pretrained,
            ), patch(
                "saver_agent.training.run_rollout_eval_with_policy",
                return_value={"temporal_miou": 0.5},
            ) as mocked_run_rollout_eval_with_policy, patch(
                "saver_agent.training.gc.collect",
                return_value=0,
            ), patch(
                "torch.cuda.is_available",
                return_value=False,
            ):
                result = training.run_rollout_eval_from_checkpoint(
                    checkpoint_path=checkpoint_dir,
                    output_dir="/tmp/sft_out",
                    rollout_eval_config=rollout_eval_config,
                    epoch_index=1,
                    model_path="/models/base",
                    torch_dtype="bfloat16",
                    attn_implementation="flash_attention_2",
                    runtime=runtime,
                )

        self.assertEqual(result["temporal_miou"], 0.5)
        self.assertEqual(captured["checkpoint_path"], str(checkpoint_dir))
        self.assertEqual(captured["device_map"], {"": 2})
        self.assertEqual(captured["torch_dtype"], "bfloat16")
        self.assertEqual(captured["attn_implementation"], "flash_attention_2")
        self.assertEqual(captured["max_new_tokens"], 77)
        self.assertEqual(captured["max_total_images"], 5)
        self.assertEqual(captured["max_image_side"], 640)
        self.assertEqual(captured["max_image_pixels"], 307200)
        self.assertFalse(captured["do_sample"])
        self.assertEqual(mocked_run_rollout_eval_with_policy.call_args.args[0], "policy_stub")

    def test_write_rollout_eval_record_also_writes_human_readable_summary_log(self):
        self.assertIsNotNone(training, "saver_agent.training module is missing")

        runtime = DistributedRuntime(rank=0, world_size=1, local_rank=0)
        metrics = {
            "num_records": 240,
            "existence_ap": 0.81,
            "category_macro_f1": 0.12,
            "temporal_miou": 0.23,
            "precursor_miou": 0.11,
            "alert_utility": 0.52,
            "evidence_f1_at_3": 0.34,
            "protocol_compliance_rate": 0.27,
            "mean_num_turns": 8.5,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            training._write_rollout_eval_record(
                output_dir=tmpdir,
                metrics=metrics,
                epoch_value=1.0,
                runtime=runtime,
            )

            summary_path = Path(tmpdir) / "rollout_eval_summary.log"
            logs_summary_path = Path(tmpdir) / "logs" / "rollout_eval_summary.log"
            self.assertTrue(summary_path.exists())
            self.assertTrue(logs_summary_path.exists())

            summary_line = summary_path.read_text(encoding="utf-8").strip()
            self.assertRegex(
                summary_line,
                r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] epoch=1\.00 num_records=240 "
                r"existence_ap=0\.8100 category_macro_f1=0\.1200 temporal_miou=0\.2300 "
                r"precursor_miou=0\.1100 alert_utility=0\.5200 evidence_f1_at_3=0\.3400 "
                r"protocol_compliance_rate=0\.2700 mean_num_turns=8\.5000$",
            )
            self.assertEqual(summary_line, logs_summary_path.read_text(encoding="utf-8").strip())

            jsonl_record = json.loads((Path(tmpdir) / "rollout_eval_metrics.jsonl").read_text(encoding="utf-8").strip())
            self.assertEqual(jsonl_record["epoch"], 1.0)

    def test_write_rollout_eval_record_can_target_explicit_eval_output_dir_without_legacy_duplicates(self):
        self.assertIsNotNone(training, "saver_agent.training module is missing")

        runtime = DistributedRuntime(rank=0, world_size=1, local_rank=0)
        metrics = {
            "num_records": 16,
            "temporal_miou": 0.25,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "checkpoints" / "sft"
            eval_output_dir = Path(tmpdir) / "eval" / "sft_epoch_end"
            training._write_rollout_eval_record(
                output_dir=output_dir,
                eval_output_dir=eval_output_dir,
                metrics=metrics,
                epoch_value=1.0,
                runtime=runtime,
            )

            self.assertTrue((eval_output_dir / "rollout_eval_metrics.jsonl").exists())
            self.assertTrue((eval_output_dir / "rollout_eval_summary.log").exists())
            self.assertFalse((output_dir / "rollout_eval_metrics.jsonl").exists())
            self.assertFalse((output_dir / "logs" / "rollout_eval_metrics.jsonl").exists())

    def test_weighted_sft_uses_token_advantages_for_token_level_nll(self):
        self.assertIsNotNone(training, "saver_agent.training module is missing")

        class _FakeTrainingArguments:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        class _FakeTrainerCallback:
            pass

        class _FakeTrainer:
            def __init__(self, *args, **kwargs):
                self.args = kwargs["args"]
                self.train_dataset = kwargs["train_dataset"]
                self.data_collator = kwargs["data_collator"]

            def add_callback(self, callback):
                return callback

        logits = torch.tensor(
            [[[2.0, 0.0], [2.0, 0.0], [0.0, 0.0]]],
            dtype=torch.float32,
        )

        class _ToyModel(torch.nn.Module):
            def __init__(self, fixed_logits: torch.Tensor):
                super().__init__()
                self.fixed_logits = fixed_logits

            def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
                return SimpleNamespace(loss=None, logits=self.fixed_logits.clone())

        fake_transformers = SimpleNamespace(
            Trainer=_FakeTrainer,
            TrainingArguments=_FakeTrainingArguments,
            TrainerCallback=_FakeTrainerCallback,
        )
        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            trainer = training.create_trainer(
                model=_ToyModel(logits),
                processor=self._FakeProcessor(),
                train_dataset=training.WeightedExampleDataset(
                    [
                        {
                            "messages": [
                                {"role": "system", "content": [{"type": "text", "text": "system"}]},
                                {"role": "user", "content": [{"type": "text", "text": "user"}]},
                            ],
                            "target_response": "<answer>{}</answer>",
                        }
                    ]
                ),
                output_dir="/tmp/saver_training_test",
                learning_rate=1e-5,
                num_train_epochs=1.0,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                logging_steps=1,
                save_steps=10,
                save_total_limit=1,
                warmup_ratio=0.0,
                weight_decay=0.0,
                max_grad_norm=1.0,
                bf16=False,
                fp16=False,
                training_objective="weighted_sft",
            )

        inputs = {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            "labels": torch.tensor([[-100, 0, 1]], dtype=torch.long),
            "sample_weight": torch.tensor([1.0], dtype=torch.float32),
            "token_advantages": torch.tensor([[0.0, 1.0, 3.0]], dtype=torch.float32),
        }
        loss = trainer.compute_loss(_ToyModel(logits), dict(inputs))

        expected_token_nll = -torch.log_softmax(logits[:, :-1, :], dim=-1)
        expected = (
            float(expected_token_nll[0, 0, 0]) * 1.0 + float(expected_token_nll[0, 1, 1]) * 3.0
        ) / 4.0
        self.assertAlmostEqual(float(loss), expected, places=6)

    def test_collator_reuses_cached_budget_plan_for_same_feature_cache_key(self):
        self.assertIsNotNone(training, "saver_agent.training module is missing")

        processor = self._CountingProcessor()
        collator = training.SingleExampleMultimodalCollator(
            processor,
            max_seq_length=6,
        )
        base_feature = {
            "_feature_cache_key": "example-001",
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "system prompt"}]},
                {"role": "user", "content": [{"type": "text", "text": "inspect the clip"}]},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "old assistant reasoning tokens repeated repeated repeated"}],
                },
                {
                    "role": "tool",
                    "content": [{"type": "text", "text": "old tool observation tokens repeated repeated repeated"}],
                },
            ],
            "target_response": '<answer>{"existence":"normal"}</answer>',
        }

        first_batch = collator([dict(base_feature, sample_weight=1.0)])
        first_encode_calls = processor.encode_call_count
        first_template_calls = processor.template_call_count
        second_batch = collator([dict(base_feature, sample_weight=2.0)])

        self.assertGreaterEqual(first_encode_calls, 1)
        self.assertGreaterEqual(first_template_calls, 2)
        self.assertEqual(processor.encode_call_count, first_encode_calls + 1)
        self.assertEqual(processor.template_call_count, first_template_calls)
        self.assertEqual(len(collator._feature_plan_cache), 1)
        self.assertAlmostEqual(float(first_batch["sample_weight"].item()), 1.0, places=6)
        self.assertAlmostEqual(float(second_batch["sample_weight"].item()), 2.0, places=6)
        self.assertTrue(torch.equal(first_batch["input_ids"], second_batch["input_ids"]))

    def test_frame_reference_resolver_prints_cache_warning_once_when_falling_back_to_raw_video(self):
        self.assertIsNotNone(training, "saver_agent.training module is missing")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video_path = root / "videos" / "resolver_warning.mp4"
            self._write_test_video(video_path, num_frames=4, fps=4.0)
            resolver = training._FrameReferenceResolver()
            image_ref = {
                "video_path": str(video_path),
                "raw_frame_index": 1,
            }

            captured = io.StringIO()
            with redirect_stdout(captured):
                first_frame = resolver._resolve_image_ref(image_ref)
                second_frame = resolver._resolve_image_ref(image_ref)

        output = captured.getvalue()
        self.assertEqual(tuple(first_frame.shape), (3, 32, 32))
        self.assertEqual(tuple(second_frame.shape), (3, 32, 32))
        self.assertIn("[cache-warning]", output)
        self.assertIn(str(video_path), output)
        self.assertEqual(output.count("[cache-warning]"), 1)

    def test_frame_reference_resolver_evicts_old_frame_cache_entries(self):
        self.assertIsNotNone(training, "saver_agent.training module is missing")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video_a = root / "videos" / "resolver_a.mp4"
            video_b = root / "videos" / "resolver_b.mp4"
            self._write_test_video(video_a, num_frames=4, fps=4.0)
            self._write_test_video(video_b, num_frames=4, fps=4.0)

            resolver = training._FrameReferenceResolver(max_cached_videos=1)
            ref_a = {"video_path": str(video_a), "raw_frame_index": 0}
            ref_b = {"video_path": str(video_b), "raw_frame_index": 0}

            resolver._resolve_image_ref(ref_a)
            self.assertIn(str(video_a), resolver._frame_cache_tensors)
            self.assertEqual(len(resolver._frame_cache_tensors), 1)

            resolver._resolve_image_ref(ref_b)
            self.assertNotIn(str(video_a), resolver._frame_cache_tensors)
            self.assertIn(str(video_b), resolver._frame_cache_tensors)
            self.assertEqual(len(resolver._frame_cache_tensors), 1)

    def test_compute_grpo_surrogate_loss_uses_signed_advantages_and_clipping(self):
        self.assertIsNotNone(training, "saver_agent.training module is missing")

        policy_log_probs = torch.tensor([0.6, -0.4], dtype=torch.float32)
        old_log_probs = torch.tensor([0.0, 0.0], dtype=torch.float32)
        advantages = torch.tensor([1.0, -1.0], dtype=torch.float32)

        loss = training.compute_grpo_surrogate_loss(
            policy_log_probs=policy_log_probs,
            old_policy_log_probs=old_log_probs,
            advantages=advantages,
            clip_epsilon=0.2,
        )

        ratio = torch.exp(policy_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
        expected = -torch.mean(torch.minimum(ratio * advantages, clipped_ratio * advantages))
        self.assertAlmostEqual(float(loss), float(expected), places=6)

    def test_build_token_advantages_from_offsets_upweights_payload_over_think(self):
        self.assertIsNotNone(training, "saver_agent.training module is missing")

        response_text = '<think>inspect first</think><answer>{"existence":"normal"}</answer>'
        think_start = response_text.index("inspect first")
        think_end = think_start + len("inspect first")
        payload_start = response_text.index('{"existence":"normal"}')
        payload_end = payload_start + len('{"existence":"normal"}')

        token_advantages = training.build_token_advantages_from_offsets(
            response_text=response_text,
            offsets=[
                (think_start, think_end),
                (payload_start, payload_end),
            ],
            base_advantage=1.0,
            target_action="answer",
            tool_name=None,
        )

        self.assertEqual(len(token_advantages), 2)
        self.assertLess(token_advantages[0], token_advantages[1])
        self.assertAlmostEqual(sum(token_advantages) / 2.0, 1.0, places=6)

    def test_build_token_advantages_from_offsets_routes_alert_and_evidence_components_by_field(self):
        self.assertIsNotNone(training, "saver_agent.training module is missing")

        response_text = (
            '<tool_call>{"name":"verify_hypothesis","arguments":{"alert_sec":1.0,"candidate_window_ids":["w0001"]}}</tool_call>'
        )
        alert_start = response_text.index("alert_sec")
        alert_end = alert_start + len("alert_sec")
        evidence_start = response_text.index("candidate_window_ids")
        evidence_end = evidence_start + len("candidate_window_ids")

        alert_weighted = training.build_token_advantages_from_offsets(
            response_text=response_text,
            offsets=[
                (alert_start, alert_end),
                (evidence_start, evidence_end),
            ],
            base_advantage=1.0,
            target_action="tool_call",
            tool_name="verify_hypothesis",
            advantage_components={"global": 1.0, "alert_local": 2.0, "evidence_local": 0.0},
            turn_component_weights={"global": 1.0, "alert_local": 1.0, "evidence_local": 0.0},
        )
        evidence_weighted = training.build_token_advantages_from_offsets(
            response_text=response_text,
            offsets=[
                (alert_start, alert_end),
                (evidence_start, evidence_end),
            ],
            base_advantage=1.0,
            target_action="tool_call",
            tool_name="verify_hypothesis",
            advantage_components={"global": 1.0, "alert_local": 0.0, "evidence_local": 2.0},
            turn_component_weights={"global": 1.0, "alert_local": 0.0, "evidence_local": 1.0},
        )

        self.assertGreater(alert_weighted[0], alert_weighted[1])
        self.assertGreater(evidence_weighted[1], evidence_weighted[0])

    def test_build_token_advantages_from_offsets_routes_search_component_to_search_fields(self):
        self.assertIsNotNone(training, "saver_agent.training module is missing")

        response_text = (
            '<tool_call>{"name":"scan_timeline","arguments":{"start_sec":0.0,"end_sec":4.0,"query":"find assault","num_frames":4}}</tool_call>'
        )
        name_start = response_text.index("name")
        name_end = name_start + len("name")
        query_start = response_text.index("query")
        query_end = query_start + len("query")

        search_weighted = training.build_token_advantages_from_offsets(
            response_text=response_text,
            offsets=[
                (name_start, name_end),
                (query_start, query_end),
            ],
            base_advantage=1.0,
            target_action="tool_call",
            tool_name="scan_timeline",
            advantage_components={"global": 1.0, "search_local": 2.0},
            turn_component_weights={"global": 1.0, "search_local": 1.0},
        )

        self.assertGreater(search_weighted[1], search_weighted[0])

    def test_weighted_example_dataset_materializes_image_refs_from_video(self):
        self.assertIsNotNone(training, "saver_agent.training module is missing")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video_path = root / "videos" / "sample.mp4"
            self._write_test_video(video_path)

            dataset = training.WeightedExampleDataset(
                [
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "image_ref": {
                                            "video_path": str(video_path),
                                            "sampled_frame_index": 0,
                                            "raw_frame_index": 0,
                                            "timestamp_sec": 0.0,
                                        },
                                    }
                                ],
                            }
                        ],
                        "target_response": '<answer>{"existence":"normal"}</answer>',
                        "sample_weight": 1.0,
                    }
                ]
            )

            item = dataset[0]

        image_item = item["messages"][0]["content"][0]
        self.assertEqual(image_item["type"], "image")
        self.assertIn("image", image_item)
        self.assertNotIn("image_ref", image_item)
        self.assertIsInstance(image_item["image"], torch.Tensor)

    def test_weighted_example_dataset_preserves_advantage_field(self):
        self.assertIsNotNone(training, "saver_agent.training module is missing")

        dataset = training.WeightedExampleDataset(
            [
                {
                    "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
                    "target_response": '<answer>{"existence":"normal"}</answer>',
                    "sample_weight": -0.5,
                    "advantage": -0.5,
                }
            ]
        )

        item = dataset[0]

        self.assertAlmostEqual(float(item["advantage"]), -0.5, places=6)

    def test_validate_prepared_examples_materializes_image_refs(self):
        self.assertIsNotNone(training, "saver_agent.training module is missing")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video_path = root / "videos" / "sample.mp4"
            self._write_test_video(video_path)

            summary = training.validate_prepared_examples(
                [
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "image_ref": {
                                            "video_path": str(video_path),
                                            "sampled_frame_index": 0,
                                            "raw_frame_index": 0,
                                            "timestamp_sec": 0.0,
                                        },
                                    }
                                ],
                            }
                        ],
                        "target_response": '<answer>{"existence":"normal"}</answer>',
                        "sample_weight": 1.0,
                    }
                ],
                materialize_images=True,
                max_materialized_examples=1,
            )

        self.assertEqual(summary["num_errors"], 0)
        self.assertEqual(summary["num_image_refs"], 1)
        self.assertEqual(summary["materialized_examples"], 1)

    def test_prepare_messages_prunes_stale_text_history_when_requested(self):
        self.assertIsNotNone(training, "saver_agent.training module is missing")

        messages = [
            {"role": "system", "content": [{"type": "text", "text": "sys"}]},
            {"role": "user", "content": [{"type": "text", "text": "user"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "a1"}]},
            {"role": "tool", "content": [{"type": "text", "text": "t1"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "a2"}]},
            {"role": "tool", "content": [{"type": "text", "text": "t2"}]},
        ]

        prepared = training._prepare_messages(messages, keep_recent_text_messages=2)

        self.assertEqual([message["role"] for message in prepared], ["system", "user", "assistant", "tool"])
        texts = [message["content"][0]["text"] for message in prepared]
        self.assertEqual(texts, ["sys", "user", "a2", "t2"])

    def test_prepare_messages_drops_timestamp_text_when_capping_total_images(self):
        self.assertIsNotNone(training, "saver_agent.training module is missing")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "0.000s"},
                    {"type": "image", "image": torch.zeros(3, 8, 8)},
                    {"type": "text", "text": "1.000s"},
                    {"type": "image", "image": torch.ones(3, 8, 8)},
                    {"type": "text", "text": "2.000s"},
                    {"type": "image", "image": torch.full((3, 8, 8), 2)},
                    {"type": "text", "text": "decide next tool"},
                ],
            }
        ]

        prepared = training._prepare_messages(messages, max_total_images=2)

        text_items = [
            item.get("text")
            for item in prepared[0]["content"]
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        image_items = [
            item
            for item in prepared[0]["content"]
            if isinstance(item, dict) and item.get("type") == "image"
        ]
        self.assertEqual(len(image_items), 2)
        self.assertNotIn("0.000s", text_items)
        self.assertIn("1.000s", text_items)
        self.assertIn("2.000s", text_items)
        self.assertIn("decide next tool", text_items)

    def test_prepare_messages_drops_timestamp_text_when_pruning_stale_tool_images(self):
        self.assertIsNotNone(training, "saver_agent.training module is missing")

        messages = [
            {"role": "system", "content": [{"type": "text", "text": "sys"}]},
            {"role": "user", "content": [{"type": "text", "text": "user"}]},
            {
                "role": "tool",
                "content": [
                    {"type": "text", "text": "0.000s"},
                    {"type": "image", "image": torch.zeros(3, 8, 8)},
                    {"type": "text", "text": "older evidence"},
                ],
            },
            {
                "role": "tool",
                "content": [
                    {"type": "text", "text": "1.000s"},
                    {"type": "image", "image": torch.ones(3, 8, 8)},
                    {"type": "text", "text": "newer evidence"},
                ],
            },
        ]

        prepared = training._prepare_messages(messages, keep_recent_tool_image_messages=1)

        older_tool_texts = [
            item.get("text")
            for item in prepared[2]["content"]
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        newer_tool_texts = [
            item.get("text")
            for item in prepared[3]["content"]
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        self.assertEqual(older_tool_texts, ["older evidence"])
        self.assertIn("1.000s", newer_tool_texts)

    def test_drop_oldest_multimodal_item_removes_paired_timestamp_text(self):
        self.assertIsNotNone(training, "saver_agent.training module is missing")

        messages = [
            {
                "role": "tool",
                "content": [
                    {"type": "text", "text": "0.000s"},
                    {"type": "image", "image": torch.zeros(3, 8, 8)},
                    {"type": "text", "text": "1.000s"},
                    {"type": "image", "image": torch.ones(3, 8, 8)},
                    {"type": "text", "text": "notes"},
                ],
            }
        ]

        removed = training._drop_oldest_multimodal_item(messages)

        self.assertTrue(removed)
        text_items = [
            item.get("text")
            for item in messages[0]["content"]
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        self.assertNotIn("0.000s", text_items)
        self.assertIn("1.000s", text_items)

    def test_single_example_multimodal_collator_enforces_max_seq_length_and_keeps_response_labels(self):
        self.assertIsNotNone(training, "saver_agent.training module is missing")

        collator = training.SingleExampleMultimodalCollator(
            self._FakeProcessor(),
            max_seq_length=8,
            keep_recent_text_messages=0,
        )
        feature = {
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "sys one two"}]},
                {"role": "user", "content": [{"type": "text", "text": "user three four five six seven eight nine"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "old tool call"}]},
                {"role": "tool", "content": [{"type": "text", "text": "old tool observation"}]},
            ],
            "target_response": "resp_a resp_b resp_c",
            "sample_weight": 1.0,
        }

        batch = collator([feature])

        self.assertEqual(int(batch["input_ids"].shape[-1]), 8)
        labels = batch["labels"][0].tolist()
        self.assertEqual(labels[:-2], [-100] * 6)
        self.assertTrue(all(value != -100 for value in labels[-2:]))

    def test_single_example_multimodal_collator_avoids_qwen_mm_token_mismatch_under_budget(self):
        self.assertIsNotNone(training, "saver_agent.training module is missing")

        collator = training.SingleExampleMultimodalCollator(
            self._StrictMultimodalProcessor(),
            max_seq_length=9,
            keep_recent_text_messages=0,
        )
        feature = {
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "sys"}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "user one two three four five"},
                        {"type": "image", "image": torch.zeros(3, 8, 8)},
                        {"type": "image", "image": torch.zeros(3, 8, 8)},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": "old tool call"}]},
                {"role": "tool", "content": [{"type": "text", "text": "old tool observation"}]},
            ],
            "target_response": "resp_a resp_b",
            "sample_weight": 1.0,
        }

        batch = collator([feature])

        self.assertLessEqual(int(batch["input_ids"].shape[-1]), 9)
        labels = batch["labels"][0].tolist()
        self.assertTrue(any(value != -100 for value in labels))


if __name__ == "__main__":
    unittest.main()
