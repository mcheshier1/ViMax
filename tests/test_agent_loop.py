import asyncio
import json
import tempfile
import unittest
from copy import deepcopy
from unittest.mock import patch

from agent_runtime.context_compactor import ContextCompactor
from agent_runtime.llm import AssistantMessage, StreamComplete, StreamTextDelta
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

    async def stream(self, messages, tools):
        message = await self.complete(messages, tools)
        if message.text:
            yield StreamTextDelta(message.text)
        yield StreamComplete(message)


class FailingLLM(FakeLLM):
    def __init__(self):
        super().__init__([])

    async def complete(self, messages, tools):
        raise RuntimeError("provider returned invalid response shape")


class CapturingLLM(FakeLLM):
    def __init__(self, replies):
        super().__init__(replies)
        self.calls = []

    async def complete(self, messages, tools):
        self.calls.append(deepcopy(messages))
        return await super().complete(messages, tools)




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
    async def test_confirmed_render_rejects_invalid_or_unbounded_commands_without_tools_or_llm(self):
        from unittest.mock import Mock

        commands = [
            "/render",
            "/render not-json",
            "/render []",
            '/render {"render_mode":"script2video","stop_after":"video","redo_shots":[]}',
            '/render {"render_mode":"script2video","stop_after":"video","redo_shots":["7","7"]}',
            '/render {"render_mode":"script2video","stop_after":"video","redo_shots":["7"],"allow_unlocked":true}',
            '/render {"render_mode":"script2video","stop_after":"portraits","redo_shots":["7"]}',
        ]
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            registry = ToolRegistry([])
            llm = Mock()
            llm.stream.side_effect = AssertionError("Malformed render command must not reach the LLM")
            loop = AgentLoop(index, PromptBuilder(f"{tmp}/prompts", index, registry), registry, ToolExecutor(registry, index), llm)
            for command in commands:
                with self.subTest(command=command):
                    events = [event async for event in loop.stream_events(command)]
                    self.assertFalse(any(event["type"] == "tool_start" for event in events))
                    done = next(event for event in events if event["type"] == "done")
                    self.assertEqual(done["tool_results"], [])
                    self.assertIn("not submitted", done["assistant"])
            llm.stream.assert_not_called()

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



    async def test_streaming_sample_identity_and_authoritative_end_are_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            registry = ToolRegistry([])
            loop = AgentLoop(index, PromptBuilder(f"{tmp}/prompts", index, registry), registry, ToolExecutor(registry, index), FakeLLM([AssistantMessage(text="héllo")]))
            events = [event async for event in loop.stream_events("hi")]
            sample_id = f"{events[0]['turn_id']}-sample-0"
            self.assertEqual([event["type"] for event in events if event["type"].startswith("assistant_")], ["assistant_start", "assistant_end"])
            tokens = [event for event in events if event["type"] == "token"]
            self.assertEqual([(event["sample_id"], event["seq"], event["delta"]) for event in tokens], [(sample_id, 1, "héllo")])
            end = next(event for event in events if event["type"] == "assistant_end")
            self.assertEqual((end["text"], end["outcome"]), ("héllo", "final"))
            record = index.active()["recent_turn_records"][0]
            self.assertEqual(record["samples"][0]["sample_id"], sample_id)
            self.assertEqual(events[-2]["final_sample_id"], sample_id)
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

    async def test_image_pressure_preserves_all_observations_in_newest_multi_tool_round(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            observations = []

            def observe(args):
                image = f"data:image/jpeg;base64,{len(observations)}" + "A" * 6000
                observations.append(image)
                return ToolResult(
                    "observe", True, "reference observed",
                    model_content=[{"type": "image_url", "image_url": {"url": image, "detail": "high"}}],
                )

            registry = ToolRegistry([ToolSpec("observe", "Observe a reference", observe, schema={})])
            replies = [
                AssistantMessage(tool_calls=[
                    ToolCall(id=f"round-{round_number}-a", name="observe", arguments={}),
                    ToolCall(id=f"round-{round_number}-b", name="observe", arguments={}),
                ])
                for round_number in range(1, 4)
            ]
            llm = CapturingLLM([*replies, AssistantMessage(text="review complete")])
            compactor = ContextCompactor(None, token_threshold=6000, buffer_tokens=0)
            loop = AgentLoop(index, PromptBuilder(f"{tmp}/prompts", index, registry), registry, ToolExecutor(registry, index), llm, compactor)
            events = [event async for event in loop.stream_events("inspect the references")]

            for round_number, messages in enumerate(llm.calls[1:], start=1):
                images = [
                    block["image_url"]["url"]
                    for message in messages if isinstance(message.get("content"), list)
                    for block in message["content"] if block.get("type") == "image_url"
                ]
                self.assertEqual(images, observations[(round_number - 1) * 2:round_number * 2])
                tool_ids = [message["tool_call_id"] for message in messages if message["role"] == "tool"]
                self.assertEqual(tool_ids, [f"round-{round_number}-a", f"round-{round_number}-b"])
                self.assertLess(compactor.estimate_messages_tokens(messages), compactor.compact_target_tokens())
            self.assertEqual(len([event for event in events if event["type"] == "tool_result"]), 6)
            self.assertEqual(events[-2]["assistant"], "review complete")

    async def test_snapshot_failure_does_not_rewrite_a_persisted_turn_as_failed(self):
        for prompt in ("hi", "/compact"):
            with self.subTest(prompt=prompt), tempfile.TemporaryDirectory() as tmp:
                index = SessionIndex(tmp)
                registry = ToolRegistry([])
                loop = AgentLoop(index, PromptBuilder(f"{tmp}/prompts", index, registry),
                                 registry, ToolExecutor(registry, index),
                                 FakeLLM([AssistantMessage(text="saved answer")]))
                original_snapshot = index.snapshot
                record_path = index.workspace_root / ".vimax/logs/loop_history.jsonl"
                def fail_after_persistence():
                    if record_path.exists():
                        raise OSError("unavailable")
                    return original_snapshot()
                with patch.object(index, "snapshot", side_effect=fail_after_persistence):
                    events = [event async for event in loop.stream_events(prompt)]
                completed = [event for event in events if event["type"] == "done"]
                self.assertEqual(len(completed), 1)
                self.assertEqual(completed[0]["status"], "completed")
                records = [json.loads(line) for line in record_path.read_text().splitlines()]
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["status"], "completed")
                self.assertEqual(records[0]["final_assistant_text"], completed[0]["assistant"])
                self.assertTrue(any(event["type"] == "error" for event in events))

    async def test_first_text_is_delivered_while_provider_is_still_running(self):
        release = asyncio.Event()
        class HeldLLM:
            async def stream(self, messages, tools):
                yield StreamTextDelta("Live ")
                await release.wait()
                yield StreamTextDelta("answer")
                yield StreamComplete(AssistantMessage(text="Live answer", finish_reason="stop"))
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            registry = ToolRegistry([])
            loop = AgentLoop(index, PromptBuilder(f"{tmp}/prompts", index, registry), registry, ToolExecutor(registry, index), HeldLLM())
            events = loop.stream_events("hello")
            observed = []
            while True:
                event = await asyncio.wait_for(anext(events), 1)
                observed.append(event)
                if event["type"] == "token":
                    break
            self.assertFalse(release.is_set())
            self.assertFalse(any(event["type"] == "done" for event in observed))
            release.set()
            observed.extend([event async for event in events])
            self.assertEqual([event["delta"] for event in observed if event["type"] == "token"], ["Live ", "answer"])
            self.assertEqual(next(event["assistant"] for event in observed if event["type"] == "done"), "Live answer")

    async def test_manual_compaction_is_recorded_without_model_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            registry = ToolRegistry([])
            llm = CapturingLLM([])
            loop = AgentLoop(index, PromptBuilder(f"{tmp}/prompts", index, registry), registry, ToolExecutor(registry, index), llm)
            events = [event async for event in loop.stream_events("/compact")]
            self.assertEqual(llm.calls, [])
            self.assertEqual(next(event for event in events if event["type"] == "done")["status"], "completed")
            record = json.loads((index.logs_dir / "loop_history.jsonl").read_text().splitlines()[-1])
            self.assertEqual(record["raw_user_input"], "/compact")
            self.assertEqual(loop.history, [])

    async def test_partial_failure_is_not_saved_as_successful_model_history(self):
        from agent_runtime.llm import LLMStreamError
        class BrokenLLM:
            async def stream(self, messages, tools):
                yield StreamTextDelta("Incomplete evidence")
                raise LLMStreamError("disconnected")
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            registry = ToolRegistry([])
            loop = AgentLoop(index, PromptBuilder(f"{tmp}/prompts", index, registry), registry, ToolExecutor(registry, index), BrokenLLM())
            events = [event async for event in loop.stream_events("hello")]
            self.assertEqual(loop.history, [])
            terminal = [event for event in events if event["type"] == "done"]
            self.assertEqual([(event["status"], event["assistant"]) for event in terminal], [("failed", "")])
            record = json.loads((index.logs_dir / "loop_history.jsonl").read_text().splitlines()[-1])
            self.assertEqual(record["assistant_turns"][0]["text"], "Incomplete evidence")
            self.assertEqual(record["assistant_turns"][0]["outcome"], "interrupted")

    async def test_request_trace_includes_retained_history_and_tool_schemas(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            registry = ToolRegistry([ToolSpec("read", "Read state", lambda args: ToolResult("read", True, "data"), schema={})])
            llm = CapturingLLM([AssistantMessage(text="answer")])
            loop = AgentLoop(index, PromptBuilder(f"{tmp}/prompts", index, registry), registry, ToolExecutor(registry, index), llm)
            loop.history = [{"role": "user", "content": "history " * 1000}, {"role": "assistant", "content": "retained answer"}]
            events = [event async for event in loop.stream_events("new request")]
            trace = next(event["prompt_trace"] for event in events if event["type"] == "prompt_trace")
            self.assertGreater(trace["request_estimates"]["history_tokens_estimate"], 2000)
            self.assertGreater(trace["request_estimates"]["tool_tokens_estimate"], 0)
