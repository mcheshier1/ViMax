# Assistant real-time streaming and instrumentation plan

Status: implemented, verified, and deployed at an idle boundary on `feature/assistant-realtime-streaming`; evidence below.

## Decision and scope

Keep Python and the existing JSONL → Node → SSE → React transport. Stream model output end to end, measure where each turn waits, and make interrupted delivery recoverable. Instrumentation comes first so the streaming rollout has a baseline.

This work does not change context budgets, model selection, compaction scheduling, tool concurrency, rendering permissions, or paid-generation scope. Those decisions should follow measured results. Streaming reduces the delay between available model text and visible text; it cannot remove provider prefill, queueing, or compaction delays.

## Pre-implementation evidence

| Boundary | Baseline behavior | Consequence |
| --- | --- | --- |
| `agent_runtime/llm.py:52–92` | `complete()` awaits a request with `stream: False`. | No incremental model output reaches the loop. |
| `agent_runtime/loop.py:165–181` | Awaits the entire completion; emits one `token` event only for a response without tools. | The event name is misleading; tool-round prose is not shown live. |
| `main_agent.py:43–46` | JSONL events are printed with `flush=True`. | Existing stdout transport can carry real deltas. |
| `web/server.mjs:450–457,507–540,593–618` | Parses child lines and writes SSE immediately; reconnect sends bridge and queue snapshots only. No SSE event IDs or write-backpressure policy. | No mandatory transport replacement; partial answer recovery and slow-client handling are missing. |
| `web/agent-message-queue.mjs:59–75` | Binds a request to a turn and advances the queue only on matching `done`. | Stream completion must not release the queue; terminal turn completion must. |
| `web/src/events.ts:86–87,138–152,168–177` | Deltas append into one `assistant-${turnId}` message. `done.assistant` is used only if that message does not exist. | Multiple model passes would merge together; missed deltas would remain missing. |
| `web/src/App.tsx:160–190,202–262` | Buffers events during history loading and restores persisted turns; reconnect does not restore active assistant text. | Streaming needs identity- and revision-aware history/snapshot reconciliation. |
| `web/src/App.tsx:295–298,684–703` | Scroll effect runs on message changes; rows parse Markdown and are not memoized. | Per-token updates could repeatedly re-render old messages and trigger smooth scrolling. |
| `agent_runtime/prompts.py:41–55` | Prompt trace estimates prompt parts, not the full request history and tool schemas. | Current displayed token count is not full model-input usage. |
| `agent_runtime/loop.py:247–252`; `agent_runtime/session_index.py:220–235,297–301` | Final turn records and JSONL logs already exist. | Reuse final persistence; do not persist every token into sessions.json. |
| `ui/src/lineMapping.ts:11–59`; `main_agent.py:49–76` | Terminal consumers understand token events but not per-sample lifecycle or authoritative text repair. | Update all consumers of the shared event contract, not just the browser. |

Local, no-network observations:

- An actual `AgentLoop` with a delayed local LLM returned `turn` at 0.2 ms, sampling status at 1.7 ms, the entire `first second third` text in one event at 152.3 ms, then `done` at 160.2 ms. This demonstrates buffering, not production latency.
- A constructed production LLM client with a dummy key reported SDK `max_retries=2` and a 300-second request timeout. The application also has its own retry loop. Hidden SDK attempts must be eliminated from or exposed in timing accounting.
- Local compactor defaults produce a 160k estimated-token trigger. The web launcher injects a 900k context-window default; with the default ratio/buffer that implies a 790k trigger, unless overridden. Neither observation establishes the active process's effective environment or actual token usage.

## Shared invariants

1. Only complete, successfully terminated, validated tool calls may execute. Never act on streamed argument fragments.
2. Preserve existing tool authorization, approvals, scope checks, and sequential mutation order.
3. A model sample is not a user turn. A turn can contain several samples and tool rounds.
4. `done` is emitted once per terminal turn, after normal final-record persistence; `error` alone never advances the server queue.
5. Partial text is visible but explicitly incomplete on interruption. It is never silently promoted to a successful final answer.
6. A reconnect never restarts generation or tools. Missing text is repaired from authoritative state.
7. New telemetry contains timings, IDs, counts, safe provider/model labels, and error classes—not prompts, answers, tool arguments/results, images, credentials, or raw provider error bodies.

## 1. Establish instrumentation and baseline

### Correlation and clocks

Use the existing browser/server `messageId` and Python `turn_id`. Add `sample_id` for each model pass, `attempt` for actual HTTP attempts, and existing tool-call IDs for tool spans. Node maintains the request-to-turn mapping and enriches every event, not only `turn` and `done`.

Do not change stdin from plain prompts just to pass IDs: correlate Python records by `turn_id`, and join that ID to `messageId` at the bridge. Include a process/bridge epoch so IDs and snapshots cannot be confused after restart.

Use monotonic durations within each process (`perf_counter`, Node performance clock, browser `performance.now`). Wall-clock timestamps are for log ordering only. Do not subtract independent process clocks to claim precise transport latency.

### Spans and counters

| Metric | Measurement |
| --- | --- |
| Browser send → first visible text | Browser send mark through committed first nonempty assistant text; visible drawer/document only. Hidden tabs or unopened conversations are marked unavailable, not fast or slow. |
| Server queue wait | Message accepted → dispatched to child. |
| Dispatch → turn observed | Node dispatch → receipt of Python `turn`; includes child scheduling/stdio startup. |
| Prompt preparation | Tool catalog, prompt parts, and full-request token estimation separately from model time. |
| Compaction | Start/end, reason, mode, estimated before/after tokens; compaction model calls identified separately from conversational samples. |
| Provider attempts | Request start, first chunk, first nonempty displayable text, last chunk, completion/failure; attempt count and backoff duration. A role-only/reasoning/tool-argument chunk is not visible text. |
| Model input/output | Full-request estimates split by system/history/current turn/tools; actual prompt/completion/cache/reasoning token usage when returned. Missing actual usage is null, never estimated and relabeled as actual. |
| Tools | Start/end/outcome per call, including failed/denied tools; count tool rounds. |
| Finalization | Final record persistence and snapshot creation durations. |
| Delivery/render | Delta/event count, text bytes, disconnects, snapshot repairs, slow-client disconnects; browser receipt → committed-frame timing. Treat frame timing as a paint proxy, not proof of physical display. |
| Turn completion | Browser send → terminal observation; server accepted → terminal event; Python turn duration. Report failed/cancelled turns separately. |

Persist bounded summary records using the existing JSONL convention. The implementation uses `.vimax/logs/agent_performance.runtime.jsonl` for Python and `.vimax/logs/agent_performance.jsonl` for Node/browser summaries, so each rotating file has one process owner. Python writes sample/tool/turn summaries; Node writes bridge summaries; a small validated browser timing endpoint accepts only allowlisted numeric values and correlation IDs. Bound payloads, retention, and record size; no per-token disk writes. Telemetry-write failure must not fail a conversation; report a bounded diagnostic instead. One stable `turn_metrics` event can update the UI and support diagnostics without flooding the chat.

Make the application retry policy the sole owner of retries (`AsyncOpenAI(max_retries=0)` for this client); retain the existing bounded transient-error policy and record each attempt/backoff. Do not introduce new retries or extend timeouts in this refactor.

Capture short and long retained-history baselines, a tool-followup turn, and a compaction-triggering turn using isolated local fixtures. For later real-provider comparisons, use explicit user-authorized requests and record model/provider/context size. Never present fixture timings as provider performance.

## 2. Implement the provider stream adapter

Primary file: `agent_runtime/llm.py`; coverage: `tests/test_agent_llm.py`.

- Add one typed asynchronous sample stream that yields displayable text deltas and a terminal assembled `AssistantMessage` with finish reason, usage, and request metadata. Keep `complete()` as a collector over the same implementation for compaction and other non-display callers; do not maintain two response parsers.
- Request `stream=True`; consume the SDK asynchronous iterator and close it on success, failure, cancellation, and consumer exit.
- Assemble tool calls by their streamed index, retaining call IDs and concatenating name/argument fragments according to the provider protocol. Support several interleaved calls.
- Parse and validate complete arguments only after successful stream termination. Malformed JSON, missing call fields, premature EOF, or truncation cannot become `{}` or execute a tool. Surface a clear structured failure instead.
- Distinguish finish reasons, including normal stop, tool calls, length limit, and provider refusal/filtering. Do not mark truncated text as a normal successful answer.
- Capture optional final usage-only chunks without assuming `choices[0]` exists. Request usage only through supported provider options; absence is not failure. Never forward internal reasoning as answer text.
- Retry transient failures only before externally committed output. Once text has been published, do not silently replay the sample: preserve the partial answer and terminate it as interrupted. Tool fragments remain private and cannot execute after a failed stream.
- Remove streaming-path response-shape retries that silently drop the tool catalog. A malformed or unsupported stream must be an explicit error, not a hidden nonstreaming or tool-less fallback. This changes existing malformed-response fallback tests intentionally.

## 3. Give model samples a lifecycle in the agent loop

Primary files: `agent_runtime/loop.py`, `main_agent.py`, `agent_runtime/context_compactor.py`, and final turn-record handling in `session_index.py`.

Proposed shared event contract:

- `assistant_start {turn_id, sample_id, round}`
- `token {turn_id, sample_id, seq, delta}` — monotonically sequenced text deltas within a sample; not model-token counts.
- `assistant_end {turn_id, sample_id, text, outcome, finish_reason}` — authoritative sample text and outcome (`final`, `tool_preamble`, `interrupted`, or `truncated`).
- Existing tool events retain their identities and ordering.
- `done {turn_id, status, assistant, ...}` — terminal turn outcome; `assistant` remains the final answer, not concatenated preambles.
- `turn_metrics {turn_id, ...safe summary}` — instrumentation, not chat content.

Emit text as it arrives, including useful prose before tools. End the sample before starting tools. Render preambles as separate assistant segments, not part of the final answer. Do not expose tool argument fragments.

Retain full assembled sample text once for final persistence. Existing `assistant_turns` already records per-sample text; extend it with stable IDs/outcomes and have history reconstruction use those records. Do not append final text again after already streaming it. Preserve the current model-history policy in this work; interrupted fragments should be recorded for the UI, not treated as completed successful model history.

Normal failures must close the sample, record the partial/outcome, emit the error and terminal `done`, and keep the REPL usable. Manual `/compact` must follow the same sample/terminal display contract. CLI and TUI consumers must handle boundaries and repair without duplicate output. A hard process kill remains distinguishable from a persisted terminal failure.

## 4. Make the bridge resumable without a full event-log service

Primary files: `web/server.mjs`, `web/agent-message-queue.mjs`, `web/server-lib.mjs`.

Keep SSE. Add bridge epoch/event revision and a bounded per-session current-turn snapshot containing sample IDs, authoritative accumulated text, outcomes, active tool summaries, and terminal state. Register a subscriber and capture/send its snapshot atomically with respect to broadcast; subsequent events carry newer revisions. Older/duplicate deltas are ignored.

On initial connection and reconnect, send the queue state plus current-turn snapshot. Keep the most recent terminal snapshot until the next turn supersedes it. Reload persisted completed history on reconnect when needed, merging by canonical message identity; this also repairs turns completed entirely while disconnected. No unbounded replay buffer is required.

Give both live and restored messages the same ID derived from turn/sample. Reconcile `assistant_end`, terminal snapshots, and history by replacement rather than concatenation. Existing records lacking sample IDs must still load through a deterministic legacy-record projection; no old runtime protocol needs to be retained.

Honor SSE backpressure: when a subscriber cannot keep up past a bounded writable-buffer limit, close that subscriber and let it reconnect to a snapshot. Do not stall the child/provider or buffer unlimited deltas for one browser. Bound current-turn snapshot bytes as well; if the limit is reached, terminate that connection with an explicit recovery path to finalized history rather than silently truncating text. Stop/session-switch/process-exit clear or terminate the appropriate active snapshot, and late events from an old child cannot affect the new session.

Process-exit recovery: mark the browser's active sample interrupted even when no final record exists. Preserve any already-visible text; report that hard-kill partial text is not durable across a server restart. Durable per-delta journaling is intentionally out of scope.

## 5. Render incremental text efficiently and explain waiting

Primary files: `web/src/api.ts`, `types.ts`, `events.ts`, `App.tsx`; terminal equivalents: `ui/src/types.ts`, `lineMapping.ts`, and `cli.tsx`.

- Model assistant segments by turn/sample ID and explicit streaming/completed/interrupted state. `assistant_end` and snapshots replace text authoritatively, repairing gaps without duplication.
- Apply the first text promptly. Batch later deltas into one update per animation frame, with a bounded timer fallback for throttled/hidden documents; flush before sample-end, errors, terminal events, and session transitions. Never batch lifecycle ordering away.
- Memoize unchanged message rows so old Markdown is not parsed again for every chunk. Use stable Markdown component/plugin configuration. Keep unfinished Markdown safe through the existing renderer.
- Auto-scroll only while the reader is near the bottom; avoid repeated smooth-scroll animations during streaming. Preserve position when reading older messages.
- Show truthful phases: queued, preparing context, compacting, waiting for model, responding, running tool, interrupted. Existing `status` events are currently not rendered as phase state; wire them through explicitly.
- Collect visible-text latency from committed rendered segments, not merely SSE receipt. Flush metrics once per turn/visibility outcome rather than per chunk. A history restore or snapshot recovery must not be counted as a fresh model first token.
- Instrument the current long-chat render cost before deciding whether virtualization is necessary; no virtualization or global state rewrite by default.

## 6. Verification and rollout gates

Extend existing behavior suites (`tests/test_agent_llm.py`, `test_agent_loop.py`, `test_main_agent_cli.py`, `test_agent_session_index.py`; `web/src/events.test.ts`, bridge/queue tests, and terminal mapping tests). Add tests only for consumer-visible boundaries and state transitions:

1. First text arrives while the provider stream is deliberately held open; final text equals all fragments exactly once, including Unicode and fragmented Markdown.
2. Interleaved fragmented tool calls assemble correctly; no tool executes before successful stream completion. Invalid/truncated arguments never execute tools.
3. Tool preamble → tool → final answer remains separate and correctly ordered live and after history reload.
4. Failure before output follows bounded retries; failure after visible output never replays text or tools, preserves partial text, and releases the queue once.
5. Disconnect/reconnect mid-answer, after completion, and during history loading repairs text without duplication or loss. Duplicate revisions, stale snapshots, and old-child events cannot corrupt a newer sample/session.
6. Stop, process exit, malformed provider chunks, missing usage, length-limit termination, and client backpressure produce truthful outcomes without stuck busy state.
7. Prompt estimates cover the actual assembled request. Usage absence remains null; timing fields remain attributable; telemetry rejects content/oversized payloads and write failure does not break chat.

Smoke the real stack in an isolated temporary workspace using a local OpenAI-compatible HTTP server that emits delayed SSE chunks. Launch actual Python/Node/frontend, inspect it in Chromium, and hold the provider response open while verifying text is already visible. Exercise a real local tool call, reconnect, stop, and a slow-reader connection; no paid provider or real project mutation is needed.

Suggested initial local acceptance targets, not measured guarantees:

- First visible text within 150 ms of locally available provider text under an active visible tab and normal load; report p50/p95 and event/paint stages separately.
- No artificial per-character replay of an already completed response.
- Growing-answer updates stay bounded by render cadence; unchanged historical rows do not re-render per delta.
- Every normal terminal turn has correlated bridge/runtime timing summaries; missing provider usage or browser visibility is explicitly represented.
- Baseline versus streaming compares equal final content and tool effects, and separates time-to-first-text from total completion time.

Run targeted Python, web, and terminal suites plus TypeScript/build checks once after integration, then the real-stack browser smoke. Update existing assistant documentation to describe actual streaming, failure/reconnect behavior, metrics location, retention, and how to interpret timing fields. Deploy only when the live request queue is idle. Do not restart an active render for this rollout.

## Implementation order and dependency boundaries

1. Instrument the existing path and capture reproducible local baselines. Keep prompt/model/context behavior unchanged.
2. Define shared sample/metric/snapshot contracts and canonical message identities.
3. Implement provider adapter + Python sample lifecycle as one integration-owned slice; in parallel, implement reducer/rendering + terminal consumers and bridge timing/snapshot handling against that fixed contract.
4. Integrate and exercise failure/reconnect/tool semantics end to end. Remove obsolete one-shot token emission and malformed-response tool-less fallback behavior, updating affected callers and tests.
5. Verify, document, and deploy at an idle boundary. Compare measured latency stages before deciding on working-context budgets, background compaction, provider changes, or native-code optimization.

No language migration, new tracing service, WebSocket conversion, or paid-provider benchmark is required to complete this plan.

## Implementation and verification — 2026-10-18

Implemented on `feature/assistant-realtime-streaming`, retaining the Python → JSONL → Node → SSE → React architecture. Context thresholds, model selection, compaction scheduling, tool authorization, and sequential tool execution were not changed. Optional usage requests are provider-compatible and configurable; actual retention and field semantics are documented in [README.md](README.md#assistant-performance-diagnostics).

### Integrated checks

- Python: `uv run python -m unittest tests.test_agent_llm tests.test_agent_loop tests.test_agent_metrics tests.test_main_agent_cli tests.test_agent_session_index tests.test_agent_prompt_builder tests.test_agent_tools tests.test_robustness` — 76 tests passed.
- Web: `npm test` — 24 files, 355 tests passed; `npx tsc --noEmit` and `npm run build` passed. Vite reports a 513.37 kB minified JavaScript chunk-size warning; this is not a build failure.
- Terminal: `npm test` and `npx tsc --noEmit` passed. Actual Ink output was also exercised in a 120×40 PTY and decoded as terminal screen state: Unicode text was visible while the provider was held open, then one complete answer replaced the partial.

### Real-stack smoke

An isolated temporary workspace and local OpenAI-compatible HTTP SSE server exercised the actual Python CLI, Node bridge, production frontend in Chromium, and Ink terminal. No paid provider request or real project mutation was used.

- Text was visible on screen while the provider was deliberately held open and before any `done`. A real `read_file` call returned the fixture's bytes; no tool started before successful completion of fragmented arguments. Preamble, tool activity, final answer, and a queued follow-up retained their distinct identities.
- Mid-answer reload restored the partial without restarting the request. Holding an old history response at the HTTP response boundary while the stream finished did not roll back the final answer; canonical assistant IDs remained unique. Completed history retained preamble/final boundaries and tool activity.
- Premature connection close and length termination preserved partial/truncated labels, made only one provider request after visible output, and released the queue. Missing usage stayed null. Stop cancelled active/pending requests; a subsequent request completed normally, with the stopped request and partial kept in chronological position. SIGKILL of the isolated Python child marked its visible partial interrupted and cleared busy state.
- A 360,000-character answer exceeded the live snapshot buffer. The browser displayed its recovery notice and kept partial text; the running overflow snapshot was 388 bytes. Reconnection restored the exact finalized 360,000 characters from history.
- A real TCP SSE client with a 1,024-byte receive buffer stopped reading during an 8,640,000-character response. The bridge closed that slow subscriber after approximately 1.90 seconds, while the provider and draining observer continued. The observer reconnected four times across overflow/oversized-event boundaries and recovered the exact finalized response.
- Telemetry HTTP ingress returned 202 for a correlated numeric summary, 409 for duplicate/unknown correlations, 400 for content fields, and 413 for an oversized body. Making both performance-log append paths directories produced bounded write diagnostics while the real conversation still completed exactly once. Conversation persistence remained writable.
- With the retained 120-message history surface, 122 deltas produced 26 React commits. Across 1,534 historical-row comparisons, no historical Markdown child tree changed during streaming. Reader position at the top remained unchanged when an answer completed.

### Local latency measurements

Five warm local-provider requests in an active Chromium assistant drawer, with retained history and render diagnostics enabled:

| Browser interval | p50 | p95 |
| --- | ---: | ---: |
| First displayable SSE receipt → viewport-visible rendered-frame proxy | 12.2 ms | 16.4 ms |
| Send → first viewport-visible rendered-frame proxy | 111.4 ms | 114.1 ms |
| Send → terminal observation | 161.3 ms | 165.9 ms |

These are same-browser-clock measurements, not cross-process timestamp subtraction or physical-pixel timestamps. They satisfy the suggested 150 ms local delivery target for this fixture, not a production/provider TTFT guarantee.

The matched local loop workload used a simulated 40 ms sample: the original fake returned buffered text, while the streaming fake split the same text into three delayed fragments. Final text, conversational model-call counts, and local tool-result counts matched:

| Scenario | Buffered first text / done | Streaming first text / done | Conversational model calls |
| --- | ---: | ---: | ---: |
| Short | 45.45 / 47.81 ms | 21.22 / 55.43 ms | 1 |
| Long retained history | 61.13 / 64.17 ms | 33.74 / 67.68 ms | 1 |
| Tool follow-up | 89.33 / 92.09 ms | 19.01 / 99.96 ms | 2 |
| Forced compaction | 181.88 / 184.55 ms | 106.72 / 140.74 ms | 1 |

These synthetic single-run comparisons separate first text from total completion; they do not establish a provider speedup, compaction-cost improvement, or production latency distribution.

### Idle-boundary rollout

The existing port-4173 bridge reported `busy: false`, no active request, and zero pending messages immediately before replacement. It was restarted with `node server.mjs --dev`, preserving its development serving mode. The deployed assistant drawer loaded existing project history in Chromium; read-only SSE inspection confirmed a new epoch, increasing revisions, `stream_snapshot`, and an idle queue. No paid request was submitted against the deployed project.

Performance-log privacy checks covered the actual local smoke records: 19 normally completed bridge turns joined to runtime summaries; deliberately injected write-failure turns were treated as unavailable telemetry, not successful measurements. Fixture content and the rejected content-field sentinel were absent from the performance logs. Temporary provider/workspace/smoke scripts were removed after verification.
