import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_runtime.metrics import append_runtime_metric


class AgentMetricsTests(unittest.TestCase):
    def test_content_and_nonfinite_values_never_enter_performance_log(self):
        with tempfile.TemporaryDirectory() as root:
            append_runtime_metric({
                "kind": "sample", "turn_id": "turn-safe", "duration_ms": 12,
                "actual_prompt_tokens": None, "first_text_ms": float("nan"),
                "prompt": "private request", "text": "private answer",
                "arguments": {"secret": "private tool input"},
                "provider": "https://user:private-credential@example.invalid/path",
            }, root)
            raw = (Path(root) / ".vimax/logs/agent_performance.runtime.jsonl").read_text()
            self.assertNotIn("private", raw)
            record = json.loads(raw)
            self.assertIsNone(record["actual_prompt_tokens"])
            self.assertNotIn("first_text_ms", record)
            self.assertEqual(record["duration_ms"], 12)

    def test_retention_keeps_bounded_complete_json_records(self):
        with tempfile.TemporaryDirectory() as root, patch("agent_runtime.metrics.MAX_FILE_BYTES", 400):
            for number in range(50):
                append_runtime_metric({"kind": "turn", "turn_id": f"turn-{number}", "duration_ms": number}, root)
            files = list((Path(root) / ".vimax/logs").glob("agent_performance.runtime.jsonl*"))
            self.assertLessEqual(len(files), 2)
            records = []
            for file in files:
                self.assertLessEqual(file.stat().st_size, 400)
                records.extend(json.loads(line) for line in file.read_text().splitlines())
            self.assertIn("turn-49", [record["turn_id"] for record in records])
            self.assertNotIn("turn-0", [record["turn_id"] for record in records])

    def test_unwritable_log_is_nonfatal_and_diagnostic_is_bounded(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / ".vimax").write_text("not a directory")
            with patch("agent_runtime.metrics._warned", False), patch("agent_runtime.metrics.logging.warning") as warning:
                for _ in range(3):
                    append_runtime_metric({"kind": "turn", "duration_ms": 1}, root)
                self.assertEqual(warning.call_count, 1)
