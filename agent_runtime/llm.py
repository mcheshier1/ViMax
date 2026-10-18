from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, AsyncIterator
from urllib.parse import urlsplit

from openai import APIConnectionError, APITimeoutError, AsyncOpenAI

from .config import llm_api_key, llm_base_url, llm_model
from .models import ToolCall

LLM_MAX_ATTEMPTS = 3
LLM_RETRY_BACKOFF_SECONDS = (1.0, 4.0)
LLM_REQUEST_TIMEOUT_SECONDS = 300.0


def _is_retryable_llm_error(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None)
    if status is not None:
        try:
            status = int(status)
        except (TypeError, ValueError):
            return False
        return status == 429 or status >= 500
    return isinstance(exc, (APIConnectionError, APITimeoutError))


class LLMResponseShapeError(RuntimeError):
    pass


class LLMStreamError(RuntimeError):
    def __init__(self, message: str, *, finish_reason: str = "error", request_metadata: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.finish_reason = finish_reason
        self.request_metadata = request_metadata or {}

@dataclass(slots=True)
class AssistantMessage:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_message: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    request_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StreamTextDelta:
    delta: str


@dataclass(slots=True)
class StreamComplete:
    message: AssistantMessage


class OpenAICompatibleLLM:
    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None) -> None:
        self.model = model or llm_model()
        self.base_url = base_url or llm_base_url()
        self.api_key = api_key or llm_api_key()
        if not self.api_key:
            raise RuntimeError("VIMAX_LLM_API_KEY is required for the agent LLM client")
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, timeout=LLM_REQUEST_TIMEOUT_SECONDS, max_retries=0)
        usage_option = os.environ.get("VIMAX_LLM_STREAM_INCLUDE_USAGE")
        host = _provider_label(self.base_url)
        self.include_usage = (usage_option.lower() in {"1", "true", "yes"}) if usage_option is not None else host in {"api.openai.com", "openrouter.ai"}

    async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> AssistantMessage:
        terminal: AssistantMessage | None = None
        async for item in self.stream(messages, tools):
            if isinstance(item, StreamComplete):
                terminal = item.message
        if terminal is None:
            raise LLMResponseShapeError("LLM stream ended without a terminal completion")
        return terminal

    async def stream(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> AsyncIterator[StreamTextDelta | StreamComplete]:
        attempts: list[dict[str, Any]] = []
        provider = _provider_label(self.base_url)
        for attempt in range(LLM_MAX_ATTEMPTS):
            published = False
            stream = None
            text_parts: list[str] = []
            calls: dict[int, dict[str, str]] = {}
            finish_reason: str | None = None
            usage: dict[str, Any] | None = None
            started = perf_counter()
            first_chunk_ms: float | None = None
            first_text_ms: float | None = None
            last_chunk_ms: float | None = None
            attempt_record: dict[str, Any] = {"attempt": attempt + 1, "provider": provider, "model": self.model}
            attempts.append(attempt_record)
            try:
                kwargs: dict[str, Any] = {"model": self.model, "messages": messages, "stream": True}
                if self.include_usage:
                    kwargs["stream_options"] = {"include_usage": True}
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"
                stream = await self.client.chat.completions.create(**kwargs)
                if not hasattr(stream, "__aiter__"):
                    raise LLMResponseShapeError("LLM provider did not return an asynchronous stream")
                async for chunk in stream:
                    elapsed = (perf_counter() - started) * 1000
                    if first_chunk_ms is None:
                        first_chunk_ms = elapsed
                    last_chunk_ms = elapsed
                    chunk_usage = _message_value(chunk, "usage")
                    if chunk_usage is not None:
                        usage = _dump_value(chunk_usage)
                    choices = _message_value(chunk, "choices") or []
                    for choice in choices:
                        reason = _message_value(choice, "finish_reason")
                        if reason is not None:
                            finish_reason = str(reason)
                        delta = _message_value(choice, "delta")
                        refusal = _message_value(delta, "refusal") if delta is not None else None
                        if refusal:
                            raise LLMStreamError("LLM provider refused the request", finish_reason="refusal")
                        content = _message_value(delta, "content") if delta is not None else None
                        if content is not None and not isinstance(content, str):
                            raise LLMResponseShapeError("LLM provider returned non-text answer content")
                        if isinstance(content, str) and content:
                            if first_text_ms is None:
                                first_text_ms = elapsed
                            text_parts.append(content)
                            published = True
                            yield StreamTextDelta(content)
                        for fragment in (_message_value(delta, "tool_calls") or []) if delta is not None else ():
                            index = _message_value(fragment, "index")
                            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                                raise LLMResponseShapeError("Tool-call stream fragment is missing its index")
                            call = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                            call_id = _message_value(fragment, "id")
                            if call_id:
                                call["id"] += str(call_id)
                            function = _message_value(fragment, "function") or {}
                            name = _message_value(function, "name")
                            args = _message_value(function, "arguments")
                            if name:
                                call["name"] += str(name)
                            if args:
                                call["arguments"] += str(args)
                if finish_reason is None:
                    raise LLMStreamError("LLM provider stream ended before a finish reason", finish_reason="truncated")
                if finish_reason in {"length", "content_filter", "refusal"}:
                    raise LLMStreamError(f"LLM provider ended with finish reason {finish_reason}", finish_reason=finish_reason)
                if finish_reason not in {"stop", "tool_calls"}:
                    raise LLMResponseShapeError("LLM provider returned an unsupported finish reason")
                tool_calls: list[ToolCall] = []
                for index in sorted(calls):
                    call = calls[index]
                    if not call["id"] or not call["name"] or not call["arguments"]:
                        raise LLMResponseShapeError("LLM provider returned an incomplete tool call")
                    try:
                        arguments = json.loads(call["arguments"])
                    except json.JSONDecodeError as exc:
                        raise LLMResponseShapeError("LLM provider returned malformed tool arguments") from exc
                    if not isinstance(arguments, dict):
                        raise LLMResponseShapeError("LLM tool arguments must be a JSON object")
                    tool_calls.append(ToolCall(id=call["id"], name=call["name"], arguments=arguments))
                if (finish_reason == "tool_calls") != bool(tool_calls):
                    raise LLMResponseShapeError("LLM finish reason and streamed tool calls are inconsistent")
                completed_ms = (perf_counter() - started) * 1000
                attempt_record.update({"first_chunk_ms": first_chunk_ms, "first_text_ms": first_text_ms, "last_chunk_ms": last_chunk_ms, "completion_ms": completed_ms, "outcome": "completed"})
                result = AssistantMessage("".join(text_parts), tool_calls, {"content": "".join(text_parts)}, finish_reason, _usage_summary(usage), {"provider": provider, "model": self.model, "attempt_count": attempt + 1, "attempts": attempts, "duration_ms": completed_ms})
                yield StreamComplete(result)
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                attempt_record.update({"first_chunk_ms": first_chunk_ms, "first_text_ms": first_text_ms, "last_chunk_ms": last_chunk_ms, "completion_ms": (perf_counter() - started) * 1000, "outcome": "failed", "error_class": type(exc).__name__})
                metadata = {"provider": provider, "model": self.model, "attempt_count": attempt + 1, "attempts": attempts}
                if published or isinstance(exc, LLMResponseShapeError) or attempt == LLM_MAX_ATTEMPTS - 1 or not _is_retryable_llm_error(exc):
                    if isinstance(exc, LLMStreamError):
                        exc.request_metadata = metadata
                        raise
                    if published:
                        raise LLMStreamError(f"LLM stream failed after text was published ({type(exc).__name__})", request_metadata=metadata) from exc
                    exc.request_metadata = metadata
                    raise
                delay = LLM_RETRY_BACKOFF_SECONDS[min(attempt, len(LLM_RETRY_BACKOFF_SECONDS) - 1)]
                attempt_record["backoff_ms"] = delay * 1000
                logging.warning("LLM call failed (%s); retrying in %.1fs (attempt %d/%d)", type(exc).__name__, delay, attempt + 1, LLM_MAX_ATTEMPTS)
            finally:
                if stream is not None:
                    close = getattr(stream, "close", None) or getattr(stream, "aclose", None)
                    if close is not None:
                        try:
                            result = close()
                            if hasattr(result, "__await__"):
                                await result
                        except Exception:
                            logging.debug("Failed to close LLM stream", exc_info=True)
            backoff_started = perf_counter()
            await asyncio.sleep(delay)
            attempt_record["backoff_ms"] = (perf_counter() - backoff_started) * 1000


def _message_value(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _dump_value(value: Any) -> Any:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return dict(value) if hasattr(value, "items") else value


def _provider_label(base_url: str) -> str:
    try:
        return urlsplit(base_url).hostname or "custom"
    except ValueError:
        return "custom"


def _usage_summary(usage: Any) -> dict[str, int | None] | None:
    if not isinstance(usage, dict):
        return None
    def count(value: Any) -> int | None:
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    return {
        "prompt_tokens": count(usage.get("prompt_tokens")),
        "completion_tokens": count(usage.get("completion_tokens")),
        "total_tokens": count(usage.get("total_tokens")),
        "cached_tokens": count(prompt_details.get("cached_tokens")) if isinstance(prompt_details, dict) else None,
        "reasoning_tokens": count(completion_details.get("reasoning_tokens")) if isinstance(completion_details, dict) else None,
    }
