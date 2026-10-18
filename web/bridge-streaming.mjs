import {mkdir, rename, rm, stat, appendFile} from 'node:fs/promises';
import path from 'node:path';
import {performance} from 'node:perf_hooks';
import {randomUUID} from 'node:crypto';

const MAX_SESSION_SNAPSHOTS = 32;
const MAX_SNAPSHOT_BYTES = 128 * 1024;
const MAX_SNAPSHOT_EVENT_BYTES = 192 * 1024;
const MAX_SAMPLES = 32;
const MAX_TOOL_STATES = 64;
const MAX_PERFORMANCE_FILE_BYTES = 1024 * 1024;
const MAX_PERFORMANCE_RECORD_BYTES = 8 * 1024;

const METRIC_TIMINGS = new Set([
  'firstVisibleTextMs', 'firstReceiptToCommitMs', 'completionMs', 'renderCommitMs',
]);
const METRIC_COUNTS = new Set(['deltaEvents', 'snapshotRepairs']);
const METRIC_FIELDS = new Set([...METRIC_TIMINGS, ...METRIC_COUNTS, 'visible']);
const RUNTIME_PERFORMANCE_FIELDS = new Set([
  'attempt', 'provider', 'model', 'duration_ms', 'first_chunk_ms', 'first_text_ms',
  'last_chunk_ms', 'completion_ms', 'backoff_ms', 'prompt_tokens', 'completion_tokens',
  'cache_tokens', 'reasoning_tokens', 'tool_name', 'tool_call_id', 'outcome', 'error_class',
  'request_tokens', 'input_tokens', 'output_tokens', 'total_tokens', 'cached_tokens',
  'prompt_estimate_tokens', 'tool_count', 'tool_rounds', 'event_count', 'prepare_ms',
  'compaction_ms', 'model_ms', 'usage_available', 'round', 'text_bytes', 'delta_events',
]);
const SAFE_PHASES = new Set([
  'preparing_context', 'compact', 'compacting', 'sampling_assistant', 'executing_tools',
  'responding', 'finalizing_answer', 'trim',
]);
const RUNTIME_LABEL_FIELDS = new Set(['provider', 'model', 'tool_name', 'outcome', 'error_class']);

function isRecord(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function safeIdentifier(value, maxLength = 256) {
  return typeof value === 'string' && value.length > 0 && value.length <= maxLength
    && /^[A-Za-z0-9_.:-]+$/.test(value);
}

function boundedToolName(value) {
  return typeof value === 'string' && /^[A-Za-z0-9_.-]{1,100}$/.test(value) ? value : 'tool';
}

function sampleById(snapshot, sampleId) {
  return snapshot.samples.find((sample) => sample.sample_id === sampleId);
}

function sampleView(snapshot) {
  return snapshot.samples.map(({sample_id, seq, text, round, outcome, finish_reason}) => ({
    sample_id, seq, text,
    ...(Number.isInteger(round) ? {round} : {}),
    ...(outcome ? {outcome} : {}),
    ...(finish_reason ? {finish_reason} : {}),
  }));
}

function snapshotView(epoch, revision, activeSessionId, snapshot) {
  return {
    type: 'stream_snapshot',
    epoch,
    revision,
    activeSessionId,
    ...(snapshot?.turn_id ? {turn_id: snapshot.turn_id} : {}),
    ...(snapshot?.messageId ? {messageId: snapshot.messageId} : {}),
    status: snapshot?.status || 'interrupted',
    samples: snapshot ? sampleView(snapshot) : [],
    tools: snapshot ? snapshot.tools.map((tool) => ({...tool})) : [],
    ...(snapshot?.phase ? {phase: snapshot.phase} : {}),
    historyRequired: Boolean(snapshot?.historyRequired),
  };
}

function safeSequence(value) {
  return Number.isSafeInteger(value) && value >= 0 && value <= 10_000_000;
}

export function createBridgeStreamState({
  epoch = randomUUID(),
  maxSessions = MAX_SESSION_SNAPSHOTS,
  maxSnapshotBytes = MAX_SNAPSHOT_BYTES,
} = {}) {
  let revision = 0;
  const sessions = new Map();

  function touch(sessionId, snapshot) {
    sessions.delete(sessionId);
    sessions.set(sessionId, snapshot);
    while (sessions.size > maxSessions) sessions.delete(sessions.keys().next().value);
  }

  function createTurn(event, sessionId, messageId) {
    const snapshot = {
      sessionId,
      turn_id: event.turn_id,
      messageId: messageId || null,
      status: 'running',
      samples: [],
      tools: [],
      phase: undefined,
      historyRequired: false,
      textBytes: 0,
      currentRound: null,
      observedTextBytes: 0,
      eventCount: 0,
      deltaEvents: 0,
      startedAt: performance.now(),
      finishedAt: null,
    };
    touch(sessionId, snapshot);
    return snapshot;
  }

  function overflow(snapshot) {
    snapshot.historyRequired = true;
    for (const sample of snapshot.samples) sample.text = '';
    snapshot.textBytes = 0;
  }

  function ensureSample(snapshot, event) {
    const id = typeof event.sample_id === 'string' && safeIdentifier(event.sample_id, 256) ? event.sample_id : '';
    if (!id) {
      overflow(snapshot);
      return null;
    }
    let sample = sampleById(snapshot, id);
    if (!sample && snapshot.samples.length < MAX_SAMPLES) {
      sample = {sample_id: id, seq: 0, text: ''};
      snapshot.samples.push(sample);
    } else if (!sample) {
      overflow(snapshot);
      return null;
    }
    return sample;
  }

  function upsertTool(snapshot, event) {
    const raw = event.tool || event.tool_result || {};
    const name = boundedToolName(raw.name || raw.requested_name);
    const id = safeIdentifier(raw.id, 256) ? raw.id : `${name}-${snapshot.tools.length + 1}`;
    let tool = snapshot.tools.find((item) => item.id === id);
    if (!tool) tool = [...snapshot.tools].reverse().find((item) => item.name === name && item.status === 'running');
    if (!tool) {
      if (snapshot.tools.length >= MAX_TOOL_STATES) snapshot.tools.shift();
      tool = {id, name, status: 'running'};
      if (Number.isInteger(snapshot.currentRound)) tool.round = snapshot.currentRound;
      snapshot.tools.push(tool);
    }
    if (event.type === 'tool_result') tool.status = raw.ok === false ? 'failed' : 'completed';
    if (event.type === 'tool_progress') {
      const stage = event.progress?.stage;
      if (typeof stage === 'string' && /^[A-Za-z0-9_-]{1,40}$/.test(stage)) tool.stage = stage;
    }
  }

  return {
    get epoch() { return epoch; },
    get revision() { return revision; },
    get snapshotCount() { return sessions.size; },
    observe(event) {
      const sessionId = event.activeSessionId;
      const turnId = event.turn_id;
      if (!safeIdentifier(sessionId, 256) || !safeIdentifier(turnId, 256)) return {accepted: true, snapshot: null};
      let snapshot = sessions.get(sessionId);
      if (snapshot?.turn_id === turnId && event.type === 'turn' && snapshot.status !== 'running') {
        return {accepted: false, snapshot};
      }
      if (!snapshot || snapshot.turn_id !== turnId) snapshot = createTurn(event, sessionId, event.messageId);
      if (event.messageId) snapshot.messageId = event.messageId;
      const wasHistoryRequired = snapshot.historyRequired;
      snapshot.eventCount = Math.min(10_000_000, snapshot.eventCount + 1);
      if (event.type === 'status' && SAFE_PHASES.has(event.phase)) {
        snapshot.phase = event.phase;
      } else if (event.type === 'assistant_start') {
        const sample = ensureSample(snapshot, event);
        if (Number.isInteger(event.round) && safeSequence(event.round)) {
          snapshot.currentRound = event.round;
          if (sample) sample.round = event.round;
        }
        if (sample && event.seq !== undefined && safeSequence(event.seq)) sample.seq = Math.max(sample.seq, event.seq);
      } else if (event.type === 'token') {
        const sample = ensureSample(snapshot, event);
        if (sample) {
          if (event.seq !== undefined && !safeSequence(event.seq)) return {accepted: false, snapshot};
          const seq = event.seq === undefined ? sample.seq + 1 : event.seq;
          if (!safeSequence(seq) || seq <= sample.seq) return {accepted: false, snapshot};
          if (seq !== sample.seq + 1) overflow(snapshot);
          sample.seq = seq;
          snapshot.deltaEvents = Math.min(10_000_000, snapshot.deltaEvents + 1);
          const delta = typeof event.delta === 'string' ? event.delta : '';
          const deltaBytes = Buffer.byteLength(delta, 'utf8');
          snapshot.observedTextBytes = Math.min(10_000_000, snapshot.observedTextBytes + deltaBytes);
          if (!snapshot.historyRequired && snapshot.textBytes + deltaBytes <= maxSnapshotBytes) {
            sample.text += delta;
            snapshot.textBytes += deltaBytes;
          } else if (!snapshot.historyRequired) {
            overflow(snapshot);
          }
        }
      } else if (event.type === 'assistant_end') {
        const sample = ensureSample(snapshot, event);
        if (sample) {
          if (event.seq !== undefined) {
            if (!safeSequence(event.seq) || event.seq < sample.seq) return {accepted: false, snapshot};
            sample.seq = Math.max(sample.seq, event.seq);
          }
          const text = typeof event.text === 'string' ? event.text : sample.text;
          const oldBytes = Buffer.byteLength(sample.text, 'utf8');
          const newBytes = Buffer.byteLength(text, 'utf8');
          if (!snapshot.historyRequired && snapshot.textBytes - oldBytes + newBytes <= maxSnapshotBytes) {
            sample.text = text;
            snapshot.textBytes += newBytes - oldBytes;
          } else if (!snapshot.historyRequired) {
            overflow(snapshot);
          }
          if (['final', 'tool_preamble', 'interrupted', 'truncated'].includes(event.outcome)) sample.outcome = event.outcome;
          if (typeof event.finish_reason === 'string') sample.finish_reason = event.finish_reason.slice(0, 64);
        }
      } else if (event.type === 'tool_start' || event.type === 'tool_progress' || event.type === 'tool_result') {
        upsertTool(snapshot, event);
      } else if (event.type === 'done') {
        snapshot.status = event.status === 'failed' ? 'failed' : event.status === 'halted' ? 'interrupted' : 'completed';
        for (const sample of snapshot.samples) {
          if (!sample.outcome && snapshot.status !== 'completed') sample.outcome = 'interrupted';
        }
        snapshot.finishedAt = performance.now();
        snapshot.phase = undefined;
      }
      touch(sessionId, snapshot);
      return {accepted: true, snapshot, becameOverflow: !wasHistoryRequired && snapshot.historyRequired};
    },
    enrich(event, activeSessionId, messageId = null) {
      revision += 1;
      return {
        ...event,
        activeSessionId: activeSessionId || '',
        messageId: messageId || null,
        epoch,
        revision,
      };
    },
    snapshotEvent(activeSessionId) {
      revision += 1;
      const snapshot = sessions.get(activeSessionId);
      let event = snapshotView(epoch, revision, activeSessionId || '', snapshot);
      if (snapshot && !snapshot.historyRequired
        && Buffer.byteLength(JSON.stringify(event), 'utf8') > MAX_SNAPSHOT_EVENT_BYTES) {
        overflow(snapshot);
        touch(activeSessionId, snapshot);
        event = snapshotView(epoch, revision, activeSessionId || '', snapshot);
      }
      return event;
    },
    interrupt(sessionId) {
      const snapshot = sessions.get(sessionId);
      if (!snapshot || snapshot.status !== 'running') return null;
      snapshot.status = 'interrupted';
      snapshot.phase = undefined;
      for (const sample of snapshot.samples) if (!sample.outcome) sample.outcome = 'interrupted';
      snapshot.finishedAt = performance.now();
      touch(sessionId, snapshot);
      return snapshotView(epoch, revision, sessionId, snapshot);
    },
    current(sessionId) {
      const snapshot = sessions.get(sessionId);
      return snapshot ? snapshotView(epoch, revision, sessionId, snapshot) : null;
    },
    completionSummary(sessionId, turnId) {
      const snapshot = sessions.get(sessionId);
      if (!snapshot || snapshot.turn_id !== turnId) return null;
      return {
        epoch,
        type: 'bridge_turn',
        turn_id: turnId,
        messageId: snapshot.messageId,
        sessionId,
        status: snapshot.status,
        duration_ms: Math.max(0, (snapshot.finishedAt || performance.now()) - snapshot.startedAt),
        event_count: snapshot.eventCount,
        delta_events: snapshot.deltaEvents,
        text_bytes: snapshot.observedTextBytes,
        history_required: snapshot.historyRequired,
      };
    },
  };
}

export function validateAgentMetricsPayload(body) {
  if (!isRecord(body)) return {ok: false, error: 'A JSON object body is required'};
  const allowed = new Set(['sessionId', 'messageId', 'turnId', 'metrics']);
  if (Object.keys(body).some((key) => !allowed.has(key))) return {ok: false, error: 'Unexpected telemetry field'};
  if (!safeIdentifier(body.sessionId) || !safeIdentifier(body.messageId)
    || body.turnId !== undefined && !safeIdentifier(body.turnId)) {
    return {ok: false, error: 'Valid sessionId, messageId, and optional turnId are required'};
  }
  if (!isRecord(body.metrics) || Object.keys(body.metrics).length === 0
    || Object.keys(body.metrics).some((key) => !METRIC_FIELDS.has(key))) {
    return {ok: false, error: 'Metrics must contain only supported numeric fields'};
  }
  const metrics = {};
  for (const [key, value] of Object.entries(body.metrics)) {
    if (METRIC_TIMINGS.has(key)) {
      if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value > 86_400_000) {
        return {ok: false, error: `Invalid ${key}`};
      }
    } else if (METRIC_COUNTS.has(key)) {
      if (!Number.isInteger(value) || value < 0 || value > 10_000_000) return {ok: false, error: `Invalid ${key}`};
    } else if (key === 'visible' && value !== 0 && value !== 1) {
      return {ok: false, error: 'Invalid visible'};
    }
    metrics[key] = value;
  }
  return {ok: true, value: {
    sessionId: body.sessionId,
    messageId: body.messageId,
    ...(body.turnId ? {turnId: body.turnId} : {}),
    metrics,
  }};
}

export function safeRuntimePerformanceRecord(event) {
  const source = isRecord(event?.record) ? event.record : {};
  const record = {type: 'runtime_performance'};
  if (safeIdentifier(event?.epoch, 256)) record.epoch = event.epoch;
  if (safeIdentifier(event?.turn_id, 256)) record.turn_id = event.turn_id;
  if (safeIdentifier(event?.activeSessionId, 256)) record.sessionId = event.activeSessionId;
  if (safeIdentifier(event?.messageId, 256)) record.messageId = event.messageId;
  if (safeIdentifier(event?.sample_id, 256)) record.sample_id = event.sample_id;
  for (const [key, value] of Object.entries(source)) {
    if (!RUNTIME_PERFORMANCE_FIELDS.has(key)) continue;
    if (typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 86_400_000) record[key] = value;
    else if (typeof value === 'boolean') record[key] = value;
    else if (typeof value === 'string' && RUNTIME_LABEL_FIELDS.has(key)
      && /^[A-Za-z0-9_.:/+-]{1,128}$/.test(value)) record[key] = value;
    else if (key === 'tool_call_id' && safeIdentifier(value, 128)) record[key] = value;
    else if (value === null) record[key] = null;
  }
  return record;
}

export function isSseBackpressureExceeded(writableLength, incomingBytes, maxBytes) {
  return writableLength + incomingBytes > maxBytes;
}

export class BoundedPerformanceLog {
  constructor(repoRoot, onFailure = () => {}, {maxBytes = MAX_PERFORMANCE_FILE_BYTES, maxRecordBytes = MAX_PERFORMANCE_RECORD_BYTES, maxPending = 128} = {}) {
    this.filePath = path.join(repoRoot, '.vimax', 'logs', 'agent_performance.jsonl');
    this.onFailure = onFailure;
    this.maxBytes = maxBytes;
    this.maxRecordBytes = maxRecordBytes;
    this.failureReported = false;
    this.pending = Promise.resolve();
    this.pendingCount = 0;
    this.maxPending = maxPending;
  }

  append(record) {
    if (this.pendingCount >= this.maxPending) return Promise.resolve(false);
    let line;
    try {
      line = `${JSON.stringify({...record, timestamp: new Date().toISOString()})}\n`;
    } catch {
      return Promise.resolve(false);
    }
    const lineBytes = Buffer.byteLength(line, 'utf8');
    if (lineBytes > this.maxRecordBytes || lineBytes > this.maxBytes) return Promise.resolve(false);
    this.pendingCount += 1;
    this.pending = this.pending.then(async () => {
      await mkdir(path.dirname(this.filePath), {recursive: true, mode: 0o700});
      let size = 0;
      try { size = (await stat(this.filePath)).size; } catch {}
      if (size + lineBytes > this.maxBytes) {
        const rotatedPath = `${this.filePath}.1`;
        await rm(rotatedPath, {force: true});
        if (size > this.maxBytes) {
          await rm(this.filePath, {force: true});
        } else {
          try { await rename(this.filePath, rotatedPath); } catch (error) {
            if (error?.code !== 'ENOENT') throw error;
          }
        }
      }
      await appendFile(this.filePath, line, {encoding: 'utf8', mode: 0o600});
      return true;
    }).catch(() => {
      if (!this.failureReported) {
        this.failureReported = true;
        try { this.onFailure(); } catch {}
      }
      return false;
    }).finally(() => {
      this.pendingCount -= 1;
    });
    return this.pending;
  }
}
