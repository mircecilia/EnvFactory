import json
import tempfile
import unittest
from pathlib import Path

from repro_1p7b.graph_frontier.mcpmark_openai_compat_proxy import clamp_request
from repro_1p7b.graph_frontier.mcpmark_pilot_runner import verified_meta


class MCPMarkPilotTest(unittest.TestCase):
    def test_transport_cap_preserves_messages_and_tools(self):
        request = {
            "model": "mcpmark-cpi3",
            "messages": [{"role": "user", "content": "question"}],
            "tools": [{"type": "function", "function": {"name": "read_file"}}],
            "temperature": 1.0,
            "max_tokens": 32768,
        }
        result = json.loads(clamp_request(json.dumps(request).encode(), 8192))
        self.assertEqual(result["max_tokens"], 8192)
        for key in ("model", "messages", "tools", "temperature"):
            self.assertEqual(result[key], request[key])

    def test_transport_does_not_raise_smaller_budget(self):
        result = json.loads(clamp_request(b'{"max_tokens":128}', 8192))
        self.assertEqual(result["max_tokens"], 128)

    def test_resume_requires_matching_verified_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "pilot" / "model" / "run-1" / "task" / "meta.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({
                "task_name": "desktop__project_management",
                "model_name": "openai/mcpmark-grpo",
                "execution_result": {"success": False},
            }))
            self.assertIsNotNone(
                verified_meta(root, "desktop__project_management", "openai/mcpmark-grpo")
            )
            self.assertIsNone(
                verified_meta(root, "desktop__project_management", "openai/mcpmark-cpi3")
            )


if __name__ == "__main__":
    unittest.main()
