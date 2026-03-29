import json
import sys
import unittest
from pathlib import Path


ROOT = Path("/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from saver_agent.prompts import build_system_prompt
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


if __name__ == "__main__":
    unittest.main()
