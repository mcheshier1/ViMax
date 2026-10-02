import asyncio
import json
import tempfile
import unittest
from copy import deepcopy

from agent_runtime.context_compactor import ContextCompactor
from agent_runtime.llm import AssistantMessage
from agent_runtime.loop import AgentLoop
from agent_runtime.models import ToolCall, ToolResult
from agent_runtime.prompts import PromptBuilder
from agent_runtime.session_index import SessionIndex
from agent_runtime.tool_executor import ToolExecutor
from agent_runtime.tools import ToolArgumentSchema, ToolRegistry, ToolSpec


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)

    async def complete(self, messages, tools):
        return self.replies.pop(0)


class FailingLLM:
    async def complete(self, messages, tools):
        raise RuntimeError("provider returned invalid response shape")


class CapturingLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    async def complete(self, messages, tools):
        self.calls.append(deepcopy(messages))
        return self.replies.pop(0)


class CompactionAwareLLM(CapturingLLM):
    """CapturingLLM that answers the compactor's summarization prompts itself.

    Only sampling calls are captured in ``calls``, so tests can assert on the
    requests the loop actually sends to the model.
    """

    def __init__(self, replies):
        super().__init__(replies)
        self.summary_calls = 0

    async def complete(self, messages, tools):
        if any("compressing conversation history" in str(message.get("content", "")) for message in messages):
            self.summary_calls += 1
            return AssistantMessage(text="## Reference Context Only\nhandoff summary")
        return await super().complete(messages, tools)


class AgentLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_tool_call_finishes(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            registry = ToolRegistry([])
            loop = AgentLoop(index, PromptBuilder(f"{tmp}/prompts", index, registry), registry, ToolExecutor(registry, index), FakeLLM([AssistantMessage(text="done")]))
            events = [event async for event in loop.stream_events("hi")]
            self.assertEqual(events[-2]["type"], "done")
            turn_id = events[0]["turn_id"]
            self.assertTrue(all(event.get("turn_id") == turn_id for event in events))
            log_text = (index.logs_dir / "loop_history.jsonl").read_text(encoding="utf-8")
            self.assertIn("assistant_finished_without_tools", log_text)


    async def test_turn_record_follows_session_created_by_tool(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            old = index.create(idea="old")

            def create_actual(args):
                record = index.create(idea="actual")
                return ToolResult("create_actual", True, record["session_id"])

            registry = ToolRegistry([ToolSpec("create_actual", "Create actual session", create_actual, schema={})])
            llm = FakeLLM([AssistantMessage(tool_calls=[ToolCall(name="create_actual", arguments={})]), AssistantMessage(text="finished")])
            loop = AgentLoop(index, PromptBuilder(f"{tmp}/prompts", index, registry), registry, ToolExecutor(registry, index), llm)
            events = [event async for event in loop.stream_events("start new project")]
            active = index.active()
            self.assertNotEqual(active["session_id"], old["session_id"])
            self.assertEqual(len(index.get(active["session_id"])["recent_turn_records"]), 1)
            self.assertEqual(index.get(old["session_id"])["recent_turn_records"], [])
            self.assertEqual(events[-1]["session"]["active_session_id"], active["session_id"])


    async def test_tool_progress_streams_before_tool_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            release = asyncio.Event()

            async def slow_tool(args, runtime):
                runtime.emit_progress("started", stage="running")
                await release.wait()
                return ToolResult("slow_tool", True, "done")

            registry = ToolRegistry([ToolSpec("slow_tool", "Slow tool", slow_tool, schema={})])
            llm = FakeLLM([AssistantMessage(tool_calls=[ToolCall(name="slow_tool", arguments={})]), AssistantMessage(text="finished")])
            loop = AgentLoop(index, PromptBuilder(f"{tmp}/prompts", index, registry), registry, ToolExecutor(registry, index), llm)
            agen = loop.stream_events("start")
            seen = []
            while True:
                event = await asyncio.wait_for(anext(agen), timeout=1)
                seen.append(event["type"])
                if event["type"] == "tool_progress":
                    self.assertFalse(release.is_set())
                    break
            release.set()
            async for event in agen:
                seen.append(event["type"])
            self.assertLess(seen.index("tool_progress"), seen.index("tool_result"))


    async def test_final_tool_progress_is_drained_before_result_and_next_tool(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)

            async def quick_tool(args, runtime):
                runtime.emit_progress(f"{args['slot']}: starting", stage="starting")
                runtime.emit_progress(f"{args['slot']}: completed", stage="completed")
                return ToolResult("quick_tool", True, args["slot"])

            registry = ToolRegistry([ToolSpec(
                "quick_tool", "Quick tool", quick_tool,
                schema={"slot": ToolArgumentSchema(str, required=True)},
            )])
            llm = FakeLLM([
                AssistantMessage(tool_calls=[
                    ToolCall(name="quick_tool", arguments={"slot": "first"}),
                    ToolCall(name="quick_tool", arguments={"slot": "second"}),
                ]),
                AssistantMessage(text="finished"),
            ])
            loop = AgentLoop(index, PromptBuilder(f"{tmp}/prompts", index, registry),
                             registry, ToolExecutor(registry, index), llm)
            events = [event async for event in loop.stream_events("start")]
            observed = [
                event["progress"]["message"] if event["type"] == "tool_progress"
                else event["tool_result"]["content"]
                for event in events if event["type"] in {"tool_progress", "tool_result"}
            ]
            self.assertEqual(observed, [
                "first: starting", "first: completed", "first",
                "second: starting", "second: completed", "second",
            ])

    async def test_preflight_compact_summarizes_old_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            registry = ToolRegistry([])
            compactor = ContextCompactor(None, token_threshold=200, buffer_tokens=0, preserve_last_n=2, summary_max_chars=2000)
            loop = AgentLoop(index, PromptBuilder(f"{tmp}/prompts", index, registry), registry, ToolExecutor(registry, index), FakeLLM([AssistantMessage(text="after compact")]), compactor)
            loop.history = [
                {"role": "user", "content": "old request " + "x" * 1200},
                {"role": "assistant", "content": "old answer " + "y" * 1200},
                {"role": "user", "content": "recent request"},
                {"role": "assistant", "content": "recent answer"},
            ]
            events = [event async for event in loop.stream_events("continue")]
            self.assertIn("compact", [event.get("phase") for event in events if event["type"] == "status"])
            session = index.active()
            self.assertIn("Reference Context Only", session["compacted_summary"])
            self.assertGreaterEqual(session["compacted_turns"], 1)
            self.assertTrue(session["compaction_snapshots"])
            self.assertEqual(loop.history[0]["role"], "system")
            self.assertIn("after compact", loop.history[-1]["content"])
            self.assertNotIn("old request", index.memory_text())


    async def test_llm_sampling_error_yields_error_without_crashing_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            registry = ToolRegistry([])
            loop = AgentLoop(index, PromptBuilder(f"{tmp}/prompts", index, registry), registry, ToolExecutor(registry, index), FailingLLM())
            events = [event async for event in loop.stream_events("start")]
            self.assertTrue(any(event["type"] == "error" and event.get("metadata", {}).get("error_type") == "llm_sampling_failed" for event in events))
            self.assertEqual(events[-2]["type"], "done")
            self.assertEqual(events[-1]["type"], "session")
            self.assertEqual(index.active()["recent_turn_records"][-1]["status"], "failed")

    async def test_tool_call_continues_then_finishes(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)

            def hello(args):
                return ToolResult("hello", True, "hello result")

            registry = ToolRegistry([ToolSpec("hello", "Say hello", hello, schema={"name": ToolArgumentSchema(str, False, "x")})])
            llm = FakeLLM([AssistantMessage(tool_calls=[ToolCall(name="hello", arguments={})]), AssistantMessage(text="finished")])
            loop = AgentLoop(index, PromptBuilder(f"{tmp}/prompts", index, registry), registry, ToolExecutor(registry, index), llm)
            events = [event async for event in loop.stream_events("start")]
            self.assertTrue(any(event["type"] == "tool_result" for event in events))
            self.assertEqual(events[-2]["assistant"], "finished")

    async def test_transient_tool_images_reach_next_llm_turn_but_not_events_or_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            data_url = "data:image/jpeg;base64,ZmFrZS1pbWFnZQ=="

            def view(args):
                return ToolResult(
                    "view_image",
                    True,
                    "image loaded",
                    {"path": "idea2video/frame.png"},
                    model_content=[{"type": "image_url", "image_url": {"url": data_url, "detail": "high"}}],
                )

            registry = ToolRegistry([ToolSpec("view_image", "View image", view, schema={"path": ToolArgumentSchema(str, True)})])
            llm = CapturingLLM(
                [
                    AssistantMessage(tool_calls=[ToolCall(name="view_image", arguments={"path": "idea2video/frame.png"})]),
                    AssistantMessage(text="The frame is visible."),
                ]
            )
            loop = AgentLoop(index, PromptBuilder(f"{tmp}/prompts", index, registry), registry, ToolExecutor(registry, index), llm)
            events = [event async for event in loop.stream_events("inspect the frame")]

            image_messages = [message for message in llm.calls[1] if message.get("role") == "user" and isinstance(message.get("content"), list)]
            self.assertEqual(len(image_messages), 1)
            self.assertEqual(image_messages[0]["content"][1]["image_url"]["url"], data_url)
            self.assertNotIn(data_url, json.dumps(events))
            self.assertNotIn(data_url, json.dumps(loop.history))
            self.assertNotIn(data_url, (index.logs_dir / "tool_calls.jsonl").read_text(encoding="utf-8"))


    async def test_oversized_tool_result_is_truncated_for_llm_turn(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            big = "Z" * 50000

            def hello(args):
                return ToolResult("hello", True, big)

            registry = ToolRegistry([ToolSpec("hello", "Say hello", hello, schema={})])
            llm = CapturingLLM(
                [
                    AssistantMessage(tool_calls=[ToolCall(name="hello", arguments={})]),
                    AssistantMessage(text="done"),
                ]
            )
            loop = AgentLoop(index, PromptBuilder(f"{tmp}/prompts", index, registry), registry, ToolExecutor(registry, index), llm)
            events = [event async for event in loop.stream_events("go")]
            # The second LLM call is the one made after the tool runs.
            tool_messages = [message for message in llm.calls[1] if message.get("role") == "tool"]
            self.assertEqual(len(tool_messages), 1)
            self.assertIn("[truncated", tool_messages[0]["content"])
            # default VIMAX_MAX_TOOL_RESULT_CHARS cap is 20000
            self.assertLessEqual(len(tool_messages[0]["content"]), 20000 + 80)
            # the full result is still what the session/event stream records
            self.assertTrue(any(event["type"] == "tool_result" and event["tool_result"]["content"] == big for event in events))


    async def test_history_is_compacted_at_most_once_per_turn(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)

            def hello(args):
                return ToolResult("hello", True, "Z" * 40000)

            registry = ToolRegistry([ToolSpec("hello", "Say hello", hello, schema={})])
            replies = [AssistantMessage(tool_calls=[ToolCall(name="hello", arguments={})]) for _ in range(6)]
            replies.append(AssistantMessage(text="final"))
            llm = CompactionAwareLLM(replies)
            compactor = ContextCompactor(llm, token_threshold=8000, buffer_tokens=0, preserve_last_n=0, summary_max_chars=2000)
            loop = AgentLoop(index, PromptBuilder(f"{tmp}/prompts", index, registry), registry, ToolExecutor(registry, index), llm, compactor)
            loop.history = [
                {"role": "user", "content": "old " + "a" * 40000},
                {"role": "assistant", "content": "old ans " + "b" * 40000},
                {"role": "user", "content": "mid " + "c" * 40000},
                {"role": "assistant", "content": "mid ans " + "d" * 40000},
            ]
            events = [event async for event in loop.stream_events("generate")]

            # History is frozen for the duration of a turn, so re-summarizing it on
            # later samples would cost an LLM call per round without shrinking the
            # request. Exactly one compaction is meaningful.
            self.assertEqual([event["phase"] for event in events if event["type"] == "status" and event.get("phase") == "compact"], ["compact"])
            self.assertEqual(llm.summary_calls, 1)
            self.assertEqual(loop.history[0]["role"], "system")  # compaction replaced history head
            self.assertEqual(loop.history[-1]["content"], "final")
            target = compactor.compact_target_tokens()
            self.assertTrue(all(compactor.estimate_messages_tokens(call) < target for call in llm.calls))


    async def test_in_turn_tool_pressure_trims_oldest_rounds_without_recompacting(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            rounds = 6
            counter = {"round": 0}

            def hello(args):
                counter["round"] += 1
                return ToolResult("hello", True, f"round-{counter['round']} " + "Z" * 40000)

            registry = ToolRegistry([ToolSpec("hello", "Say hello", hello, schema={})])
            replies = [AssistantMessage(tool_calls=[ToolCall(name="hello", arguments={})]) for _ in range(rounds)]
            replies.append(AssistantMessage(text="final"))
            llm = CompactionAwareLLM(replies)
            compactor = ContextCompactor(llm, token_threshold=12000, buffer_tokens=0, preserve_last_n=2)
            loop = AgentLoop(index, PromptBuilder(f"{tmp}/prompts", index, registry), registry, ToolExecutor(registry, index), llm, compactor)
            # History stays small: all the pressure comes from the in-turn transcript.
            loop.history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]
            events = [event async for event in loop.stream_events("generate")]

            trim_phases = [event["phase"] for event in events if event["type"] == "status" and event.get("phase") == "trim"]
            self.assertGreaterEqual(len(trim_phases), 1)
            self.assertEqual(llm.summary_calls, 0)  # history was never the source of pressure
            target = compactor.compact_target_tokens()
            self.assertTrue(all(compactor.estimate_messages_tokens(call) < target for call in llm.calls))
            final_transcript = json.dumps(llm.calls[-1])
            self.assertIn(f"round-{rounds} ", final_transcript)  # newest round kept
            self.assertNotIn("round-1 ", final_transcript)  # oldest round dropped
            # the authoritative record still holds every tool result
            self.assertEqual(len([event for event in events if event["type"] == "tool_result"]), rounds)
