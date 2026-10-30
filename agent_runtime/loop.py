from __future__ import annotations

import json
import asyncio
from contextlib import aclosing
import os
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, AsyncIterator

from .context_compactor import ContextCompactor, CompactionResult
from .llm import AssistantMessage, OpenAICompatibleLLM, StreamComplete, StreamTextDelta
from .models import ToolCall, ToolResult, TurnControl
from .prompts import PromptBuilder
from .session_index import SessionIndex
from .tool_executor import ToolExecutor
from .tools import ToolRegistry, build_builtin_registry

from .metrics import append_runtime_metric


MAX_TOOL_PASSES = 50


def _max_tool_result_chars() -> int:
    """Max characters of a single tool result replayed to the LLM.

    Tool outputs (e.g. large JSON artifacts or verbose tool dumps) are forwarded
    verbatim to the next LLM sample. That unbounded replay is what can blow past
    the model context window mid-turn. This caps the replayed copy only; the
    authoritative result is still stored via the session index / tool executor.
    """
    try:
        return max(0, int(os.environ.get("VIMAX_MAX_TOOL_RESULT_CHARS", "20000")))
    except ValueError:
        return 20000


def _clip_tool_content(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 60)].rstrip() + f"\n... [truncated for LLM context; full result stored separately]"


def _in_turn_round_starts(in_turn: list[dict[str, Any]]) -> list[int]:
    """Indices of the assistant messages that open each in-turn tool round."""
    return [index for index, message in enumerate(in_turn) if message.get("role") == "assistant" and message.get("tool_calls")]


def _drop_oldest_tool_round(in_turn: list[dict[str, Any]]) -> bool:
    """Drop the oldest complete tool round from the live in-turn transcript.

    A round is the assistant message that requested tools plus every tool result
    (and tool-provided image block) answering it, so removing a whole round never
    leaves a tool result without its assistant tool call. The newest round is
    always kept: it is the context the next sample is about to act on. Full
    results stay available in the session record and the event stream.
    """
    starts = _in_turn_round_starts(in_turn)
    if len(starts) < 2:
        return False
    del in_turn[: starts[-1]]
    return True


async def _confirmed_render_stream(call: ToolCall | None, error: str, result: ToolResult | None):
    """Dispatch one confirmed render, then report only its authoritative result."""
    if error:
        text = error
    elif result is None and call is not None:
        yield StreamComplete(AssistantMessage(tool_calls=[call], finish_reason="tool_calls"))
        return
    elif result is not None and not result.ok:
        text = f"Render request did not complete: {result.content}"
    elif result is not None:
        phase = str(result.metadata.get("phase") or "Render")
        text = f"{phase.capitalize()} render request finished. Review the outputs in Film."
        missing = result.metadata.get("missing")
        if missing:
            text += f" Some outputs are still missing: {json.dumps(missing, ensure_ascii=False)}."
    else:
        raise RuntimeError("Confirmed render has neither a tool call nor a result")
    yield StreamTextDelta(text)
    yield StreamComplete(AssistantMessage(text=text, finish_reason="stop"))


class AgentLoop:
    def __init__(self, session_index: SessionIndex, prompt_builder: PromptBuilder, tool_registry: ToolRegistry, tool_executor: ToolExecutor, llm: Any, context_compactor: ContextCompactor | None = None) -> None:
        self.session_index = session_index
        self.prompt_builder = prompt_builder
        self.tool_registry = tool_registry
        self.tool_executor = tool_executor
        self.llm = llm
        self.context_compactor = context_compactor or ContextCompactor(llm)
        self.history: list[dict[str, Any]] = []

    def _metric(self, record: dict[str, Any]) -> None:
        append_runtime_metric(record, self.session_index.workspace_root)

    def _record_requests(self, turn_id: str | None, sample_id: str, metadata: dict[str, Any], usage: dict[str, Any] | None = None) -> None:
        for attempt in metadata.get("attempts", []):
            self._metric({"kind": "provider_attempt", "turn_id": turn_id, "sample_id": sample_id, **attempt})
        if usage is not None:
            self._metric({
                "kind": "usage", "turn_id": turn_id, "sample_id": sample_id,
                "actual_prompt_tokens": usage.get("prompt_tokens"), "actual_completion_tokens": usage.get("completion_tokens"),
                "actual_cached_tokens": usage.get("cached_tokens"), "actual_reasoning_tokens": usage.get("reasoning_tokens"),
            })

    async def compact_history(self, *, reason: str = "manual", turn_id: str | None = None) -> str:
        if not self.history:
            return "No conversation history to compact."
        session = self.session_index.active() or self.session_index.create()
        result = await self.context_compactor.compact(
            self.history,
            previous_summary=str(session.get("compacted_summary", "") or ""),
            reason=reason,
        )
        self.history = [self.context_compactor.synthetic_summary_message(result.summary), *result.preserved_messages]
        self.session_index.update_compaction(session["session_id"], _compaction_record(result))
        self._metric({"kind": "compaction", "turn_id": turn_id, "duration_ms": result.duration_ms, "reason": reason, "mode": result.mode, "estimated_tokens_before": result.estimated_tokens_before, "estimated_tokens_after": result.estimated_tokens_after})
        self._record_requests(turn_id, f"{turn_id or 'manual'}-compaction", result.request_metadata, result.usage)
        return f"Compacted context {result.estimated_tokens_before} -> {result.estimated_tokens_after} ({result.mode})."

    async def stream_events(self, user_input: str) -> AsyncIterator[dict[str, Any]]:
        """Keep exceptional preparation/tool failures terminal and recoverable."""
        turn_id = ""
        terminal = False
        samples: dict[str, dict[str, Any]] = {}
        active_sample = ""
        started = perf_counter()
        try:
            async with aclosing(self._stream_turn(user_input)) as events:
                async for event in events:
                    turn_id = event.get("turn_id", turn_id)
                    sample_id = event.get("sample_id", "")
                    if event["type"] == "assistant_start":
                        active_sample = sample_id
                        samples[sample_id] = {"sample_id": sample_id, "parts": [], "seq": 0}
                    elif event["type"] == "token" and sample_id in samples:
                        samples[sample_id]["parts"].append(event["delta"])
                        samples[sample_id]["seq"] = event["seq"]
                    elif event["type"] == "assistant_end" and sample_id in samples:
                        samples[sample_id].update(text=event["text"], outcome=event["outcome"], finish_reason=event.get("finish_reason"))
                        samples[sample_id]["parts"].clear()
                        active_sample = ""
                    elif event["type"] == "done":
                        terminal = True
                    yield event
        except Exception as exc:
            if terminal:
                self._metric({"kind": "post_terminal_failure", "turn_id": turn_id, "error_class": type(exc).__name__})
                return
            records = []
            for sample in samples.values():
                text = sample["text"] if "text" in sample else "".join(sample["parts"])
                record = {"sample_id": sample["sample_id"], "text": text, "outcome": sample.get("outcome", "interrupted"), "finish_reason": sample.get("finish_reason", "error"), "tool_calls": []}
                records.append(record)
                if sample["sample_id"] == active_sample:
                    yield {"type": "assistant_end", "turn_id": turn_id, "sample_id": active_sample, "seq": sample["seq"], "text": text, "outcome": "interrupted", "finish_reason": "error"}
            try:
                session = self.session_index.active() or self.session_index.create()
                self.session_index.append_turn_record(session["session_id"], {"turn_id": turn_id, "status": "failed", "raw_user_input": user_input, "assistant_turns": records, "tool_rounds": [], "final_assistant_text": "", "created_at": datetime.now().isoformat(timespec="seconds")})
            except Exception:
                pass  # Terminal delivery must survive a persistence failure.
            metrics = {"duration_ms": (perf_counter() - started) * 1000, "sample_count": len(records), "status": "failed", "error_class": type(exc).__name__}
            self._metric({"kind": "turn", "turn_id": turn_id, **metrics})
            yield {"type": "error", "turn_id": turn_id, "message": f"Agent turn failed ({type(exc).__name__})"}
            yield {"type": "turn_metrics", "turn_id": turn_id, "metrics": metrics}
            yield {"type": "done", "turn_id": turn_id, "status": "failed", "assistant": "", "tool_results": []}

    async def _stream_turn(self, user_input: str) -> AsyncIterator[dict[str, Any]]:
        control = TurnControl()
        turn_started = perf_counter()
        yield {"type": "turn", "turn_id": control.turn_id, "turn": {"id": control.turn_id}}
        if user_input.strip() == "/compact":
            sample_id = f"{control.turn_id}-sample-0"
            yield {"type": "status", "turn_id": control.turn_id, "phase": "compact", "message": "Compacting context"}
            yield {"type": "assistant_start", "turn_id": control.turn_id, "sample_id": sample_id, "round": 0}
            message = await self.compact_history(reason="manual", turn_id=control.turn_id)
            yield {"type": "token", "turn_id": control.turn_id, "sample_id": sample_id, "seq": 1, "delta": message}
            yield {"type": "assistant_end", "turn_id": control.turn_id, "sample_id": sample_id, "seq": 1, "text": message, "outcome": "final", "finish_reason": "stop"}
            session = self.session_index.active() or self.session_index.create()
            persist_started = perf_counter()
            self.session_index.append_turn_record(session["session_id"], {"turn_id": control.turn_id, "status": "completed", "raw_user_input": "/compact", "assistant_turns": [{"sample_id": sample_id, "phase": "manual_compaction", "text": message, "outcome": "final", "finish_reason": "stop", "tool_calls": []}], "tool_rounds": [], "final_assistant_text": message, "final_sample_id": sample_id})
            persistence_ms = (perf_counter() - persist_started) * 1000
            snapshot_started = perf_counter()
            try:
                snapshot = self.session_index.snapshot()
            except Exception as exc:
                snapshot = None
                yield {"type": "error", "turn_id": control.turn_id, "message": "Could not refresh session state.", "metadata": {"error_type": type(exc).__name__}}
            metrics = {"duration_ms": (perf_counter() - turn_started) * 1000, "sample_count": 1, "tool_count": 0, "persistence_ms": persistence_ms, "snapshot_ms": (perf_counter() - snapshot_started) * 1000, "status": "completed"}
            self._metric({"kind": "turn", "turn_id": control.turn_id, **metrics})
            yield {"type": "turn_metrics", "turn_id": control.turn_id, "metrics": metrics}
            yield {"type": "done", "turn_id": control.turn_id, "status": "completed", "assistant": message, "final_sample_id": sample_id, "tool_results": []}
            if snapshot is not None:
                yield {"type": "session", "turn_id": control.turn_id, "session": snapshot}
            return
        command = user_input.strip().split(maxsplit=1)
        confirmed_render = bool(command and command[0] == "/render")
        render_call = None
        render_error = ""
        if confirmed_render:
            try:
                arguments = json.loads(command[1]) if len(command) == 2 else None
                allowed = {"render_mode", "redo_shots", "stop_after", "allow_unreviewed_redo"}
                if not isinstance(arguments, dict) or set(arguments) - allowed:
                    raise ValueError("Use /render with a JSON object containing only render_mode, redo_shots, stop_after, and optionally allow_unreviewed_redo.")
                if arguments.get("render_mode") not in {"script2video", "idea2video", "novel2video"}:
                    raise ValueError("A confirmed render must name its render_mode.")
                if arguments.get("stop_after") not in {"stills", "video"}:
                    raise ValueError("A confirmed render must stop after stills or video.")
                shots = arguments.get("redo_shots")
                if not isinstance(shots, list) or not shots or not all(isinstance(slot, str) and slot.strip() for slot in shots) or len(set(shots)) != len(shots):
                    raise ValueError("A confirmed render must name a nonempty, unique list of shot slot keys.")
                if "allow_unreviewed_redo" in arguments and not isinstance(arguments["allow_unreviewed_redo"], bool):
                    raise ValueError("allow_unreviewed_redo must be a boolean.")
                render_call = ToolCall(name="vimax_render_video", arguments=arguments)
            except (ValueError, TypeError) as exc:
                render_error = f"Render request was not submitted: {exc}"
        yield {"type": "status", "turn_id": control.turn_id, "phase": "preparing_context", "message": "Preparing context"}
        catalog_started = perf_counter()
        tool_schemas = self.tool_registry.list_function_tools()
        catalog_ms = (perf_counter() - catalog_started) * 1000
        prompt_started = perf_counter()
        parts = self.prompt_builder.build_parts(user_input)
        system = "\n\n".join(f"## {part.title}\n{part.body}" for part in parts if part.id != "request.user")
        prompt_ms = (perf_counter() - prompt_started) * 1000
        # Messages accumulated *inside this turn* (assistant tool-calls + tool
        # results + any tool-provided image blocks). Kept separate from
        # self.history so history can be compacted mid-turn without losing the
        # live in-turn messages. Rebuilt into the request on every sample.
        in_turn: list[dict[str, Any]] = []
        history_compacted = False

        # History and appended messages are unchanged between samples. Count them
        # once per turn rather than repeatedly scanning retained image Data URIs.
        budget_enabled = self.context_compactor.compact_target_tokens() > 0
        estimate_started = perf_counter()
        schema_tokens = _tool_schema_tokens(tool_schemas)
        history_estimate = user_estimate = system_estimate = 0
        estimate_ms = (perf_counter() - estimate_started) * 1000
        in_turn_token_counts: list[int] = []
        in_turn_tokens = 0

        def base_token_count() -> int:
            nonlocal history_estimate, user_estimate, system_estimate, estimate_ms
            started = perf_counter()
            history_estimate = self.context_compactor.estimate_messages_tokens(self.history)
            user_estimate = self.context_compactor.estimate_message_tokens({"role": "user", "content": user_input})
            system_estimate = self.context_compactor.estimate_message_tokens({"role": "system", "content": system})
            estimate_ms += (perf_counter() - started) * 1000
            return history_estimate + user_estimate + _prompt_tokens(parts) + schema_tokens if budget_enabled else 0

        base_tokens = base_token_count()

        def append_in_turn(message: dict[str, Any]) -> None:
            nonlocal in_turn_tokens, estimate_ms
            started = perf_counter()
            tokens = self.context_compactor.estimate_message_tokens(message)
            estimate_ms += (perf_counter() - started) * 1000
            in_turn.append(message)
            in_turn_token_counts.append(tokens)
            in_turn_tokens += tokens

        def over_budget(extra: list[dict[str, Any]] | None = None) -> bool:
            target = self.context_compactor.compact_target_tokens()
            return target > 0 and base_tokens + (in_turn_tokens if extra else 0) >= target

        if not confirmed_render and over_budget():
            yield {"type": "status", "turn_id": control.turn_id, "phase": "compact", "message": "Compacting context before sampling"}
            await self.compact_history(reason="token-pressure", turn_id=control.turn_id)
            history_compacted = True
            prompt_started = perf_counter()
            parts = self.prompt_builder.build_parts(user_input)
            system = "\n\n".join(f"## {part.title}\n{part.body}" for part in parts if part.id != "request.user")
            prompt_ms += (perf_counter() - prompt_started) * 1000
            base_tokens = base_token_count()
        assistant_turns: list[dict[str, Any]] = []
        tool_rounds: list[dict[str, Any]] = []
        transitions: list[dict[str, str]] = []
        all_tool_results: list[ToolResult] = []
        final_text = ""
        final_sample_id: str | None = None
        status = "completed"
        tool_round = 0
        sample_round = 0
        _max_result_chars = _max_tool_result_chars()

        while True:
            # Guard the request again here: tool output can push the live turn past
            # the model window while the loop runs. History is frozen for the
            # duration of a turn, so re-summarizing it on every sample cannot
            # shrink the request; once history is compacted, the residual pressure
            # is the in-turn tool transcript, which is trimmed round by round.
            if not confirmed_render and over_budget(in_turn):
                if not history_compacted and over_budget():
                    yield {"type": "status", "turn_id": control.turn_id, "phase": "compact", "message": "Compacting context before sampling"}
                    await self.compact_history(reason="token-pressure", turn_id=control.turn_id)
                    history_compacted = True
                    prompt_started = perf_counter()
                    parts = self.prompt_builder.build_parts(user_input)
                    system = "\n\n".join(f"## {part.title}\n{part.body}" for part in parts if part.id != "request.user")
                    prompt_ms += (perf_counter() - prompt_started) * 1000
                    base_tokens = base_token_count()
                while over_budget(in_turn):
                    previous_count = len(in_turn)
                    if not _drop_oldest_tool_round(in_turn):
                        break
                    del in_turn_token_counts[:previous_count - len(in_turn)]
                    in_turn_tokens = sum(in_turn_token_counts)
                    yield {"type": "status", "turn_id": control.turn_id, "phase": "trim", "message": "Dropped the oldest in-turn tool round to stay inside the model context window"}
            runtime_messages: list[dict[str, Any]] = [{"role": "system", "content": system}, *self.history, {"role": "user", "content": user_input}, *in_turn]
            yield {"type": "status", "turn_id": control.turn_id, "phase": "dispatching_render" if confirmed_render else "sampling_assistant", "message": "Processing confirmed render" if confirmed_render else "Sampling assistant"}
            sample_id = f"{control.turn_id}-sample-{sample_round}"
            sample_round += 1
            sample_text: list[str] = []
            sample_seq = 0
            sample_started = perf_counter()
            request_tokens_estimate = 0 if confirmed_render else system_estimate + history_estimate + user_estimate + in_turn_tokens + schema_tokens
            estimates = {"system_tokens_estimate": system_estimate, "history_tokens_estimate": history_estimate, "current_tokens_estimate": user_estimate + in_turn_tokens, "tool_tokens_estimate": schema_tokens}
            if not confirmed_render:
                trace = self.prompt_builder.trace(parts)
                trace["total_estimated_tokens"] = request_tokens_estimate
                trace["totals"]["total_tokens"] = request_tokens_estimate
                trace["request_estimates"] = estimates
                yield {"type": "prompt_trace", "turn_id": control.turn_id, "prompt_trace": trace}
                self._metric({"kind": "request", "turn_id": control.turn_id, "sample_id": sample_id, "input_tokens_estimate": request_tokens_estimate, **estimates})
            sample_record: StreamComplete | None = None
            yield {"type": "assistant_start", "turn_id": control.turn_id, "sample_id": sample_id, "round": sample_round - 1}
            try:
                source = _confirmed_render_stream(render_call, render_error, all_tool_results[-1] if all_tool_results else None) if confirmed_render else self.llm.stream(runtime_messages, tools=tool_schemas)
                async with aclosing(source) as stream:
                    async for item in stream:
                        if isinstance(item, StreamTextDelta):
                            sample_seq += 1
                            sample_text.append(item.delta)
                            if sample_seq == 1:
                                yield {"type": "status", "turn_id": control.turn_id, "phase": "responding", "message": "Responding"}
                            yield {"type": "token", "turn_id": control.turn_id, "sample_id": sample_id, "seq": sample_seq, "delta": item.delta}
                        elif isinstance(item, StreamComplete):
                            sample_record = item
                if sample_record is None:
                    raise RuntimeError("LLM stream completed without terminal assistant message")
                assistant = sample_record.message
                # The terminal assembled text is authoritative; emitted deltas are not replayed.
                text = assistant.text
                outcome = "tool_preamble" if assistant.tool_calls else "final"
                yield {"type": "assistant_end", "turn_id": control.turn_id, "sample_id": sample_id, "seq": sample_seq, "text": text, "outcome": outcome, "finish_reason": assistant.finish_reason or "stop"}
            except Exception as exc:
                text = "".join(sample_text)
                status = "failed"
                finish_reason = getattr(exc, "finish_reason", "error")
                outcome = "truncated" if finish_reason in {"length", "content_filter", "refusal", "truncated"} else ("interrupted" if text else "truncated")
                assistant_turns.append({"sample_id": sample_id, "phase": "initial" if tool_round == 0 else f"followup_{tool_round}", "text": text, "outcome": outcome, "finish_reason": finish_reason, "tool_calls": [], "incomplete": True, "input_tokens_estimate": request_tokens_estimate, "duration_ms": (perf_counter() - sample_started) * 1000})
                self._record_requests(control.turn_id, sample_id, getattr(exc, "request_metadata", {}))
                yield {"type": "assistant_end", "turn_id": control.turn_id, "sample_id": sample_id, "seq": sample_seq, "text": text, "outcome": outcome, "finish_reason": finish_reason}
                final_text = ""
                transitions.append(_transition("sampling_assistant", "finalizing_answer", "llm_sampling_failed"))
                yield {"type": "error", "turn_id": control.turn_id, "message": f"Agent LLM request failed ({type(exc).__name__})", "metadata": {"error_type": "llm_sampling_failed"}}
                break
            assistant_turns.append({"sample_id": sample_id, "phase": "initial" if tool_round == 0 else f"followup_{tool_round}", "text": text, "outcome": outcome, "finish_reason": assistant.finish_reason or "stop", "tool_calls": [call.as_dict() for call in assistant.tool_calls], "usage": assistant.usage, "request_metadata": assistant.request_metadata, "input_tokens_estimate": request_tokens_estimate, "duration_ms": (perf_counter() - sample_started) * 1000})
            self._record_requests(control.turn_id, sample_id, assistant.request_metadata, assistant.usage)
            if not assistant.tool_calls:
                transitions.append(_transition("sampling_assistant", "finalizing_answer", "assistant_finished_without_tools"))
                final_text = text
                final_sample_id = sample_id
                break
            transitions.append(_transition("sampling_assistant", "executing_tools", "assistant_requested_tools"))
            if tool_round >= MAX_TOOL_PASSES:
                status = "halted"
                final_text = "Tool loop halted after max tool passes."
                transitions.append(_transition("executing_tools", "finalizing_answer", "max_tool_passes_reached"))
                yield {"type": "error", "turn_id": control.turn_id, "message": final_text, "metadata": {"max_tool_passes": MAX_TOOL_PASSES}}
                break
            tool_round += 1
            yield {"type": "status", "turn_id": control.turn_id, "phase": "executing_tools", "message": f"Running tools (round {tool_round})"}
            append_in_turn({"role": "assistant", "content": assistant.text or "", "tool_calls": [_openai_tool_call(call) for call in assistant.tool_calls]})
            round_results: list[ToolResult] = []
            round_model_content: list[dict[str, Any]] = []

            for call in assistant.tool_calls:
                tool_started = perf_counter()
                yield {"type": "tool_start", "turn_id": control.turn_id, "tool": call.as_dict()}
                progress_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

                def on_progress(event: dict[str, Any]) -> None:
                    progress_queue.put_nowait(event)

                task = asyncio.create_task(self.tool_executor.execute(call, control, progress_callback=on_progress))
                try:
                    while not task.done():
                        progress_task = asyncio.create_task(progress_queue.get())
                        try:
                            ready, _ = await asyncio.wait(
                                (task, progress_task), return_when=asyncio.FIRST_COMPLETED,
                            )
                            if progress_task in ready:
                                yield progress_task.result()
                        finally:
                            if not progress_task.done():
                                progress_task.cancel()
                            await asyncio.gather(progress_task, return_exceptions=True)
                    while not progress_queue.empty():
                        yield progress_queue.get_nowait()
                    record = await task
                except BaseException as exc:
                    self._metric({"kind": "tool", "turn_id": control.turn_id, "tool_id": call.id, "tool_name": call.name, "duration_ms": (perf_counter() - tool_started) * 1000, "outcome": "interrupted", "error_class": type(exc).__name__})
                    raise
                finally:
                    if not task.done():
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                result = record.result
                self._metric({"kind": "tool", "turn_id": control.turn_id, "tool_id": call.id, "tool_name": call.name, "duration_ms": (perf_counter() - tool_started) * 1000, "outcome": "success" if result.ok else "failure"})
                round_results.append(result)
                all_tool_results.append(result)
                yield {"type": "tool_result", "turn_id": control.turn_id, "tool_result": result.as_dict()}
                tool_content = json.dumps(result.as_dict(), ensure_ascii=False)
                if len(tool_content) > _max_result_chars:
                    # Cap only the copy replayed to the LLM; the authoritative,
                    # full result is still stored by the tool executor / session.
                    tool_content = _clip_tool_content(tool_content, _max_result_chars)
                append_in_turn({"role": "tool", "tool_call_id": call.id, "name": result.name, "content": tool_content})
                if result.model_content:
                    round_model_content.extend(result.model_content)
            if round_model_content:
                append_in_turn(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Tool-provided image observation(s). Inspect these pixels as evidence for the active task; this is not a new user request.",
                            },
                            *round_model_content,
                        ],
                    }
                )
            tool_rounds.append({"tool_round": tool_round, "requested_tools": [call.as_dict() for call in assistant.tool_calls], "tool_results": [result.as_dict() for result in round_results]})
            transitions.append(_transition("executing_tools", "post_tool_decision", "tool_round_completed"))
            transitions.append(_transition("post_tool_decision", "sampling_assistant", "runtime_continuation_after_tools"))

        turn_record = {"turn_id": control.turn_id, "status": status, "raw_user_input": user_input, "assistant_turns": assistant_turns, "tool_rounds": tool_rounds, "transitions": transitions, "final_assistant_text": final_text, "final_sample_id": final_sample_id, "created_at": datetime.now().isoformat(timespec="seconds")}
        persistence_started = perf_counter()
        try:
            final_session = self.session_index.active() or self.session_index.create()
            self.session_index.append_turn_record(final_session["session_id"], turn_record)
        except Exception as exc:
            status = "failed"
            final_text = ""
            yield {"type": "error", "turn_id": control.turn_id, "message": "Could not persist the completed turn.", "metadata": {"error_type": type(exc).__name__}}
        if status == "completed":
            self.history.extend([{"role": "user", "content": user_input}, {"role": "assistant", "content": final_text}])
        persistence_ms = (perf_counter() - persistence_started) * 1000
        snapshot_started = perf_counter()
        try:
            snapshot = self.session_index.snapshot()
        except Exception as exc:
            snapshot = None
            yield {"type": "error", "turn_id": control.turn_id, "message": "Could not refresh session state.", "metadata": {"error_type": type(exc).__name__}}
        snapshot_ms = (perf_counter() - snapshot_started) * 1000
        def actual_total(key: str) -> int | None:
            values = [(sample.get("usage") or {}).get(key) for sample in assistant_turns]
            return sum(values) if values and all(isinstance(value, int) and not isinstance(value, bool) for value in values) else None
        metrics = {"duration_ms": (perf_counter() - turn_started) * 1000, "sample_count": len(assistant_turns), "tool_round_count": len(tool_rounds), "tool_count": len(all_tool_results), "input_tokens_estimate": sum(int(sample.get("input_tokens_estimate", 0)) for sample in assistant_turns), "actual_prompt_tokens": actual_total("prompt_tokens"), "actual_completion_tokens": actual_total("completion_tokens"), "actual_cached_tokens": actual_total("cached_tokens"), "actual_reasoning_tokens": actual_total("reasoning_tokens"), "catalog_ms": catalog_ms, "prompt_ms": prompt_ms, "estimate_ms": estimate_ms, "persistence_ms": persistence_ms, "snapshot_ms": snapshot_ms, "status": status}
        self._metric({"kind": "turn", "turn_id": control.turn_id, **metrics})
        for sample in assistant_turns:
            self._metric({"kind": "sample", "turn_id": control.turn_id, "sample_id": sample.get("sample_id"), "duration_ms": sample.get("duration_ms", 0), "input_tokens_estimate": sample.get("input_tokens_estimate", 0), "outcome": sample.get("outcome"), "finish_reason": sample.get("finish_reason")})
        yield {"type": "turn_metrics", "turn_id": control.turn_id, "metrics": metrics}
        yield {"type": "done", "turn_id": control.turn_id, "status": status, "assistant": final_text, "final_sample_id": final_sample_id, "tool_results": [result.as_dict() for result in all_tool_results]}
        if snapshot is not None:
            yield {"type": "session", "turn_id": control.turn_id, "session": snapshot}


def _compaction_record(result: CompactionResult) -> dict[str, Any]:
    return {
        "summary": result.summary,
        "preserved_message_count": len(result.preserved_messages),
        "compacted_message_count": result.compacted_message_count,
        "estimated_tokens_before": result.estimated_tokens_before,
        "estimated_tokens_after": result.estimated_tokens_after,
        "reason": result.reason,
        "mode": result.mode,
        "created_at": result.created_at,
    }


def _prompt_tokens(parts: list[Any]) -> int:
    return sum(max(1, len(str(getattr(part, "body", ""))) // 4) for part in parts)


def _tool_schema_tokens(tool_schemas: list[dict[str, Any]]) -> int:
    try:
        return max(0, len(json.dumps(tool_schemas, ensure_ascii=False, default=str)) // 4)
    except TypeError:
        return max(0, len(str(tool_schemas)) // 4)


def _transition(src: str, dst: str, reason: str) -> dict[str, str]:
    return {"from": src, "to": dst, "reason": reason}


def _openai_tool_call(call: ToolCall) -> dict[str, Any]:
    return {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)}}


def build_runtime(workspace_root: str | Path = ".", llm: Any | None = None, adapter_specs: list[Any] | None = None) -> AgentLoop:
    from .vimax_adapters import build_vimax_adapter_specs
    root = Path(workspace_root).resolve()
    session_index = SessionIndex(root)
    specs = adapter_specs if adapter_specs is not None else build_vimax_adapter_specs(root, session_index)
    registry = build_builtin_registry(root, session_index, specs)
    executor = ToolExecutor(registry, session_index)
    prompt_builder = PromptBuilder(root / "prompts", session_index, registry)
    resolved_llm = llm or OpenAICompatibleLLM()
    return AgentLoop(session_index, prompt_builder, registry, executor, resolved_llm, ContextCompactor(resolved_llm))
