"""Regression tests for small crash bugs in helper stringification paths."""

import time
import unittest

from agent_runtime.context_compactor import ContextCompactor
from interfaces.shot_description import ShotBriefDescription


class TestContextCompactorToolCallPreview(unittest.TestCase):
    def test_fallback_summary_handles_tool_call_messages(self):
        compactor = ContextCompactor(None, token_threshold=200, buffer_tokens=0, preserve_last_n=2, summary_max_chars=2000)
        messages = [
            {"role": "user", "content": "list the files"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {"name": "list_files", "arguments": "{}"}}]},
        ]
        summary = compactor._fallback_summary(messages, [], "", "test")
        self.assertIn("[tool calls]", summary)
        self.assertIn("list_files", summary)

    def test_fallback_summary_extracts_paths_from_mixed_content_quickly(self):
        compactor = ContextCompactor(None, token_threshold=200, buffer_tokens=0, preserve_last_n=2, summary_max_chars=2000)
        messages = [
            {"role": "user", "content": "z" * 40000},
            {"role": "assistant", "content": '{"artifact": ".working_dir/idea2video/script.json"}'},
            {"role": "user", "content": "re-render [frames](shots/3/frame.png) next."},
        ]
        started = time.perf_counter()
        summary = compactor._fallback_summary(messages, [], "", "test")
        elapsed = time.perf_counter() - started
        self.assertIn(".working_dir/idea2video/script.json", summary)
        self.assertIn("shots/3/frame.png", summary)
        # Scanning runs of word characters must not restart a failing match at
        # every offset; the quadratic version needed ~30s for this input.
        self.assertLess(elapsed, 5.0)


class TestShotBriefDescriptionStr(unittest.TestCase):
    def test_str_uses_existing_fields(self):
        shot = ShotBriefDescription(
            idx=0,
            is_last=False,
            cam_idx=1,
            visual_desc="<Alice> waves at the camera.",
            audio_desc="[Speaker] Alice (Happy): Hello!",
        )
        text = str(shot)
        self.assertIn("Shot 0", text)
        self.assertIn("Camera Index: 1", text)
        self.assertIn("<Alice> waves at the camera.", text)
        self.assertIn("[Speaker] Alice (Happy): Hello!", text)


if __name__ == "__main__":
    unittest.main()
