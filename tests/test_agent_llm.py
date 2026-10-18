import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from openai import APIConnectionError

from agent_runtime.llm import LLMResponseShapeError, OpenAICompatibleLLM, StreamComplete, StreamTextDelta


class AsyncChunks:
    def __init__(self, chunks):
        self.chunks = iter(chunks)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            item = next(self.chunks)
            if isinstance(item, BaseException):
                raise item
            return item
        except StopIteration:
            raise StopAsyncIteration

    async def close(self):
        self.closed = True


def chunk(delta=None, finish=None, usage=None):
    choice = {"delta": delta or {}, "finish_reason": finish}
    return {"choices": [choice] if delta is not None or finish is not None else [], "usage": usage}


class AgentLLMTests(unittest.IsolatedAsyncioTestCase):
    def client(self, stream):
        llm = OpenAICompatibleLLM(model="test", base_url="https://example.invalid/v1", api_key="test-key")
        create = AsyncMock(return_value=stream)
        llm.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        return llm, create

    async def test_stream_yields_unicode_deltas_then_authoritative_message_and_usage(self):
        stream = AsyncChunks([
            chunk({"content": "hé"}),
            chunk({"content": "llo"}),
            chunk(finish="stop"),
            chunk(usage={"prompt_tokens": 7, "completion_tokens": 2}),
        ])
        llm, _ = self.client(stream)
        events = [item async for item in llm.stream([], [])]
        self.assertEqual([item.delta for item in events if isinstance(item, StreamTextDelta)], ["hé", "llo"])
        final = next(item.message for item in events if isinstance(item, StreamComplete))
        self.assertEqual(final.text, "héllo")
        self.assertEqual(final.finish_reason, "stop")
        self.assertEqual(final.usage["prompt_tokens"], 7)
        self.assertTrue(stream.closed)

    async def test_interleaved_fragmented_tool_calls_are_assembled_only_after_finish(self):
        stream = AsyncChunks([
            chunk({"tool_calls": [{"index": 1, "id": "call_b", "function": {"name": "se", "arguments": "{\"b\":"}}, {"index": 0, "id": "call_a", "function": {"name": "lo", "arguments": "{\"a\":"}}]}),
            chunk({"tool_calls": [{"index": 0, "function": {"name": "ok", "arguments": "1}"}}, {"index": 1, "function": {"name": "arch", "arguments": "2}"}}]}),
            chunk(finish="tool_calls"),
        ])
        llm, _ = self.client(stream)
        events = [item async for item in llm.stream([], [{"type": "function"}])]
        final = next(item.message for item in events if isinstance(item, StreamComplete))
        self.assertEqual([(call.id, call.name, call.arguments) for call in final.tool_calls], [("call_a", "look", {"a": 1}), ("call_b", "search", {"b": 2})])

    async def test_malformed_arguments_never_fall_back_to_toolless_request(self):
        stream = AsyncChunks([chunk({"tool_calls": [{"index": 0, "id": "call", "function": {"name": "run", "arguments": "{"}}]}), chunk(finish="tool_calls")])
        llm, create = self.client(stream)
        with self.assertRaisesRegex(LLMResponseShapeError, "malformed tool arguments"):
            async for _ in llm.stream([], [{"type": "function"}]):
                pass
        self.assertEqual(create.await_count, 1)
        self.assertTrue(create.await_args.kwargs.get("tools"))

    async def test_missing_finish_reason_is_truncation(self):
        from agent_runtime.llm import LLMStreamError
        llm, _ = self.client(AsyncChunks([chunk({"content": "partial"})]))
        with self.assertRaises(LLMStreamError) as caught:
            async for _ in llm.stream([], []):
                pass
        self.assertEqual(caught.exception.finish_reason, "truncated")

    async def test_length_termination_preserves_reason_after_visible_text(self):
        from agent_runtime.llm import LLMStreamError
        llm, create = self.client(AsyncChunks([chunk({"content": "unfinished"}), chunk(finish="length")]))
        with self.assertRaises(LLMStreamError) as caught:
            async for _ in llm.stream([], []):
                pass
        self.assertEqual(caught.exception.finish_reason, "length")
        self.assertEqual(create.await_count, 1)

    async def test_tool_arguments_with_normal_stop_are_not_executable(self):
        fragments = [chunk({"tool_calls": [{"index": 0, "id": "call", "function": {"name": "write_json", "arguments": "{}"}}]}), chunk(finish="stop")]
        llm, create = self.client(AsyncChunks(fragments))
        with self.assertRaises(LLMResponseShapeError):
            await llm.complete([], [{"type": "function"}])
        self.assertEqual(create.await_count, 1)

    async def test_transient_failure_before_text_retries_and_records_both_attempts(self):
        llm, create = self.client(AsyncChunks([]))
        create.side_effect = [APIConnectionError(request=httpx.Request("POST", "https://example.invalid")), AsyncChunks([chunk({"content": "recovered"}), chunk(finish="stop")])]
        with patch("agent_runtime.llm.asyncio.sleep", new=AsyncMock()):
            answer = await llm.complete([], [])
        self.assertEqual(answer.text, "recovered")
        self.assertEqual(create.await_count, 2)
        attempts = answer.request_metadata["attempts"]
        self.assertEqual([item["outcome"] for item in attempts], ["failed", "completed"])
        self.assertIsNone(attempts[0]["first_text_ms"])
        self.assertGreaterEqual(attempts[1]["completion_ms"], attempts[1]["first_text_ms"])

    async def test_transient_failure_after_text_never_replays_request(self):
        from agent_runtime.llm import LLMStreamError
        source = AsyncChunks([chunk({"content": "partial"}), APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))])
        llm, create = self.client(source)
        delivered = []
        with self.assertRaises(LLMStreamError):
            async for item in llm.stream([], []):
                if isinstance(item, StreamTextDelta):
                    delivered.append(item.delta)
        self.assertEqual(delivered, ["partial"])
        self.assertEqual(create.await_count, 1)
        self.assertTrue(source.closed)

    async def test_abandoning_stream_closes_provider_without_finishing_tools(self):
        source = AsyncChunks([chunk({"content": "partial"}), chunk(finish="stop")])
        llm, _ = self.client(source)
        events = llm.stream([], [])
        self.assertEqual((await anext(events)).delta, "partial")
        await events.aclose()
        self.assertTrue(source.closed)


if __name__ == "__main__":
    unittest.main()
