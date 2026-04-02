import json
import sys
import unittest
from pathlib import Path


ROOT = Path("/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from saver_agent.config import PromptConfig
from saver_agent.prompts import build_system_prompt, build_user_prompt
from saver_agent.tool_registry import get_tool_schemas


class SaverAgentPromptTests(unittest.TestCase):
    def test_build_system_prompt_accepts_real_tool_schemas_and_embeds_schema_payload(self):
        tool_schemas = get_tool_schemas()

        prompt = build_system_prompt(tool_schemas)

        self.assertIn("Do not invent tool names or argument keys.", prompt)
        self.assertIn('"name":"scan_timeline"', prompt)
        self.assertIn('"name":"seek_evidence"', prompt)
        self.assertIn('"parameters":{"type":"object"', prompt)
        self.assertIn('"required":["query","start_sec","end_sec"]', prompt)

    def test_build_system_prompt_mentions_verify_hypothesis_enum_from_schema(self):
        prompt = build_system_prompt(get_tool_schemas())

        self.assertIn("soft_alert_check", prompt)
        self.assertIn("hard_alert_check", prompt)
        self.assertIn("full_keep_drop", prompt)
        self.assertNotIn("allow_external_verifier_fallback", prompt)
        self.assertNotIn("verifier_backend", prompt)

    def test_build_system_prompt_requires_finalize_before_terminal_answer(self):
        prompt = build_system_prompt(get_tool_schemas())

        self.assertIn("Only output <answer> after finalize_case", prompt)

    def test_build_system_prompt_embeds_overridden_finalize_case_schema(self):
        override_schema = {
            "type": "object",
            "properties": {
                "existence": {"type": "string"},
                "summary": {"type": "string"},
                "counterfactual_type": {"type": "string"},
            },
            "required": ["existence", "summary", "counterfactual_type"],
        }

        prompt = build_system_prompt(get_tool_schemas(finalize_case_schema=override_schema))

        self.assertIn("summary", prompt)
        self.assertIn("counterfactual_type", prompt)

    def test_build_user_prompt_hides_raw_video_id_by_default(self):
        record = {
            "video_id": "Assault_1",
            "scene": {"scenario": "street"},
            "video_meta": {"duration_sec": 12.0},
            "agent_task": {
                "task_prompt": "Inspect the clip.",
                "success_criteria": ["criterion_a"],
            },
        }

        prompt = build_user_prompt(record)

        self.assertNotIn("Assault_1", prompt)
        self.assertIn("Case ID:", prompt)

    def test_build_user_prompt_uses_anonymized_case_id_even_in_video_id_template_slot(self):
        record = {
            "video_id": "Assault_1",
            "scene": {"scenario": "street"},
            "video_meta": {"duration_sec": 12.0},
            "agent_task": {
                "task_prompt": "Inspect the clip.",
                "success_criteria": ["criterion_a"],
            },
        }

        prompt = build_user_prompt(
            record,
            prompt_config=PromptConfig(
                initial_user_template="Case: {video_id}\nTask: {task_prompt}",
            ),
        )

        self.assertNotIn("Assault_1", prompt)
        self.assertIn("Case: case_", prompt)

    def test_build_user_prompt_does_not_expose_raw_video_id_placeholder(self):
        record = {
            "video_id": "Assault_1",
            "scene": {"scenario": "street"},
            "video_meta": {"duration_sec": 12.0},
            "agent_task": {
                "task_prompt": "Inspect the clip.",
                "success_criteria": ["criterion_a"],
            },
        }

        prompt = build_user_prompt(
            record,
            prompt_config=PromptConfig(
                initial_user_template="Raw: {raw_video_id}\nTask: {task_prompt}",
            ),
        )

        self.assertNotIn("Assault_1", prompt)
        self.assertIn("Raw: case_", prompt)


if __name__ == "__main__":
    unittest.main()
