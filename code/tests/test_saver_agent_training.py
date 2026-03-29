import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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
