import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from agent_runtime.prompts import PromptBuilder
from agent_runtime.session_index import SessionIndex
from agent_runtime.tools import build_builtin_registry


class PromptBuilderTests(unittest.TestCase):
    def test_prompt_injects_context_and_tool_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "prompts").mkdir()
            (root / "prompts" / "agent.md").write_text("agent rules", encoding="utf-8")
            (root / "prompts" / "workflow.md").write_text("workflow rules", encoding="utf-8")
            index = SessionIndex(root)
            index.create(idea="cat")
            registry = build_builtin_registry(root, index)
            builder = PromptBuilder(root / "prompts", index, registry)
            messages = builder.build_messages("start")
            self.assertIn("Available tools", messages[0]["content"])
            self.assertIn("当前 working_dir 尚未完成结构化文本文件", messages[0]["content"])
            self.assertIn("read_file", messages[0]["content"])
            trace = builder.trace(builder.build_parts("start"))
            self.assertGreater(trace["total_estimated_tokens"], 0)


    def test_prompt_injects_compacted_session_summary_as_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "prompts").mkdir()
            (root / "prompts" / "agent.md").write_text("agent rules", encoding="utf-8")
            (root / "prompts" / "workflow.md").write_text("workflow rules", encoding="utf-8")
            index = SessionIndex(root)
            record = index.create(idea="cat")
            index.update_compaction(record["session_id"], {"summary": "## Reference Context Only\n- user wants moon cat", "compacted_message_count": 4, "preserved_message_count": 2})
            registry = build_builtin_registry(root, index)
            builder = PromptBuilder(root / "prompts", index, registry)
            message = builder.build_messages("continue")[0]["content"]
            self.assertIn("Session context summary", message)
            self.assertIn("reference context only", message)
            self.assertIn("user wants moon cat", message)
            trace = builder.trace(builder.build_parts("continue"))
            self.assertGreater(trace["totals"]["dynamic_tokens"], 0)

    def test_prompt_refreshes_live_approval_after_historical_refusal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "prompts").mkdir()
            (root / "prompts" / "agent.md").write_text("agent rules", encoding="utf-8")
            (root / "prompts" / "workflow.md").write_text("workflow rules", encoding="utf-8")
            index = SessionIndex(root)
            session = index.create(idea="cat")
            index.update_compaction(session["session_id"], {
                "summary": "The previous assistant said slot 0 stills were not accepted.",
            })
            working_dir = root / session["working_dir"]
            shot_dir = working_dir / "script2video" / "shots" / "0"
            shot_dir.mkdir(parents=True)
            (shot_dir / "shot_description.json").write_text("{}", encoding="utf-8")
            images = []
            for frame in ("first_frame.png", "last_frame.png"):
                path = shot_dir / "model" / frame
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(frame.encode("utf-8"))
                images.append(path)

            acceptance_path = working_dir / "render_acceptance.json"
            acceptance_path.write_text(json.dumps({"script2video": {}}), encoding="utf-8")
            registry = build_builtin_registry(root, index)
            builder = PromptBuilder(root / "prompts", index, registry)
            before_approval = builder.build_messages("continue")[0]["content"]
            self.assertIn('"keyframes slot 0": rendered, not accepted', before_approval)

            acceptance_path.write_text(json.dumps({
                "script2video": {
                    "shots": {"0": {"keyframes": {
                        "accepted_at": "2026-10-08T19:23:01",
                        "artifacts": [
                            {
                                "path": path.relative_to(working_dir).as_posix(),
                                "size": path.stat().st_size,
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            }
                            for path in images
                        ],
                    }}},
                },
            }), encoding="utf-8")
            after_approval = builder.build_messages("continue")[0]["content"]

            self.assertIn('"keyframes slot 0": accepted at "2026-10-08T19:23:01"', after_approval)

            images[0].write_bytes(b"x" * images[0].stat().st_size)
            after_artifact_change = builder.build_messages("continue")[0]["content"]
            self.assertIn('"keyframes slot 0": stale (accepted artifact fingerprint no longer matches)', after_artifact_change)

    def test_prompt_treats_novel_text_artifacts_as_text_stage_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "prompts").mkdir()
            (root / "prompts" / "agent.md").write_text("agent rules", encoding="utf-8")
            (root / "prompts" / "workflow.md").write_text("workflow rules", encoding="utf-8")
            index = SessionIndex(root)
            record = index.create(idea="novel")
            session_root = root / record["working_dir"] / "novel2video"
            (session_root / "novel").mkdir(parents=True)
            (session_root / "novel" / "novel_compressed.txt").write_text("compressed", encoding="utf-8")
            (session_root / "events").mkdir()
            (session_root / "events" / "event_0.json").write_text("{}", encoding="utf-8")
            (session_root / "relevant_chunks" / "event_0").mkdir(parents=True)
            (session_root / "relevant_chunks" / "event_0" / "chunk.txt").write_text("chunk", encoding="utf-8")
            (session_root / "scenes" / "event_0").mkdir(parents=True)
            (session_root / "scenes" / "event_0" / "scene_0.json").write_text("{}", encoding="utf-8")
            (session_root / "global_information" / "characters" / "novel_level").mkdir(parents=True)
            (session_root / "global_information" / "characters" / "novel_level" / "novel_characters_after_event_0.json").write_text("[]", encoding="utf-8")
            registry = build_builtin_registry(root, index)
            builder = PromptBuilder(root / "prompts", index, registry)
            messages = builder.build_messages("continue")
            self.assertIn("文本规划阶段已完成", messages[0]["content"])
            self.assertIn("novel2video/events/event_*.json: present", messages[0]["content"])
