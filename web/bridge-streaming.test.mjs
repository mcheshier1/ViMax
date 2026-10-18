import {mkdtemp, readFile, rm, stat} from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {afterEach, describe, expect, it} from 'vitest';
import {
  BoundedPerformanceLog,
  createBridgeStreamState,
  isSseBackpressureExceeded,
  safeRuntimePerformanceRecord,
  validateAgentMetricsPayload,
} from './bridge-streaming.mjs';

const roots = [];

afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) => rm(root, {recursive: true, force: true})));
});

function observe(state, type, fields = {}) {
  return state.observe({type, activeSessionId: 'session-a', turn_id: 'turn-a', messageId: 'message-a', ...fields});
}

describe('bridge streaming state and telemetry', () => {
  it('correlates ordered sample deltas and repairs authoritative end text exactly', () => {
    const state = createBridgeStreamState({epoch: 'bridge-a'});
    const turn = state.enrich({type: 'turn', turn_id: 'turn-a'}, 'session-a', 'message-a');
    const status = state.enrich({type: 'status', turn_id: 'turn-a'}, 'session-a', 'message-a');
    expect(turn).toMatchObject({epoch: 'bridge-a', activeSessionId: 'session-a', messageId: 'message-a', revision: 1});
    expect(status.revision).toBe(2);

    observe(state, 'turn');
    observe(state, 'assistant_start', {sample_id: 'turn-a-sample-0', round: 0});
    expect(observe(state, 'token', {sample_id: 'turn-a-sample-0', seq: 1, delta: 'Hi 💡\n**'}).accepted).toBe(true);
    expect(observe(state, 'token', {sample_id: 'turn-a-sample-0', seq: 1, delta: 'duplicated'}).accepted).toBe(false);
    observe(state, 'token', {sample_id: 'turn-a-sample-0', seq: 2, delta: 'draft'});
    observe(state, 'assistant_end', {
      sample_id: 'turn-a-sample-0', seq: 2, text: 'Hi 💡\n**complete**', outcome: 'final', finish_reason: 'stop',
    });
    observe(state, 'done', {status: 'completed'});
    expect(observe(state, 'turn').accepted).toBe(false);

    expect(state.snapshotEvent('session-a')).toMatchObject({
      epoch: 'bridge-a', activeSessionId: 'session-a', turn_id: 'turn-a', messageId: 'message-a',
      status: 'completed', historyRequired: false,
      samples: [{sample_id: 'turn-a-sample-0', seq: 2, text: 'Hi 💡\n**complete**', outcome: 'final', finish_reason: 'stop'}],
    });
  });

  it('bounds per-session text snapshots and sends one explicit history recovery transition', () => {
    const state = createBridgeStreamState({epoch: 'bridge-b', maxSnapshotBytes: 5});
    observe(state, 'turn');
    observe(state, 'assistant_start', {sample_id: 'turn-a-sample-0'});
    observe(state, 'token', {sample_id: 'turn-a-sample-0', seq: 1, delta: 'hello'});
    expect(observe(state, 'token', {sample_id: 'turn-a-sample-0', seq: 2, delta: '!'})).toMatchObject({
      accepted: true, becameOverflow: true,
    });
    expect(state.snapshotEvent('session-a')).toMatchObject({historyRequired: true, status: 'running', samples: [{text: '', seq: 2}]});
    expect(observe(state, 'token', {sample_id: 'turn-a-sample-0', seq: 3, delta: 'still live'}).becameOverflow).toBe(false);
    observe(state, 'done', {status: 'completed'});
    expect(state.snapshotEvent('session-a')).toMatchObject({historyRequired: true, status: 'completed', samples: [{text: ''}]});
  });

  it('requests history recovery when a live delta sequence has a gap', () => {
    const state = createBridgeStreamState({epoch: 'bridge-gap'});
    observe(state, 'turn');
    observe(state, 'assistant_start', {sample_id: 'turn-a-sample-0'});
    expect(observe(state, 'token', {sample_id: 'turn-a-sample-0', seq: 2, delta: 'later'})).toMatchObject({
      accepted: true, becameOverflow: true,
    });
    expect(state.snapshotEvent('session-a')).toMatchObject({historyRequired: true, samples: [{text: '', seq: 2}]});
  });

  it('marks hard-exit samples interrupted while retaining bounded visible text', () => {
    const state = createBridgeStreamState({epoch: 'bridge-c'});
    observe(state, 'turn');
    observe(state, 'assistant_start', {sample_id: 'turn-a-sample-0'});
    observe(state, 'token', {sample_id: 'turn-a-sample-0', seq: 1, delta: 'visible partial'});
    state.interrupt('session-a');
    expect(state.snapshotEvent('session-a')).toMatchObject({
      status: 'interrupted', samples: [{text: 'visible partial', outcome: 'interrupted'}],
    });
  });

  it('leaves room for snapshot metadata under the SSE recovery-frame bound', () => {
    const state = createBridgeStreamState({epoch: 'bridge-cap'});
    observe(state, 'turn');
    observe(state, 'assistant_start', {sample_id: 'turn-a-sample-0'});
    observe(state, 'token', {sample_id: 'turn-a-sample-0', seq: 1, delta: 'x'.repeat(128 * 1024)});
    const snapshot = state.snapshotEvent('session-a');
    expect(Buffer.byteLength(JSON.stringify(snapshot))).toBeLessThan(256 * 1024);
  });

  it('clears escaped text that would exceed the recovery event wire budget', () => {
    const state = createBridgeStreamState({epoch: 'bridge-escaped'});
    observe(state, 'turn');
    observe(state, 'assistant_start', {sample_id: 'turn-a-sample-0'});
    observe(state, 'token', {sample_id: 'turn-a-sample-0', seq: 1, delta: '\n'.repeat(100_000)});
    expect(state.snapshotEvent('session-a')).toMatchObject({historyRequired: true, samples: [{text: ''}]});
  });

  it('stores compact tool state without snapshots of arguments, progress prose, or results', () => {
    const state = createBridgeStreamState({epoch: 'bridge-tools'});
    observe(state, 'turn');
    observe(state, 'assistant_start', {sample_id: 'turn-a-sample-0', round: 0});
    observe(state, 'tool_start', {tool: {id: 'call-1', name: 'render', arguments: {secret: 'private argument'}}});
    observe(state, 'tool_progress', {tool: {id: 'call-1', name: 'render'}, progress: {stage: 'encoding', message: 'private progress'}});
    observe(state, 'tool_result', {tool_result: {name: 'render', ok: true, content: 'private result'}});
    const snapshot = state.snapshotEvent('session-a');
    expect(snapshot.tools).toEqual([{id: 'call-1', name: 'render', status: 'completed', round: 0, stage: 'encoding'}]);
    expect(JSON.stringify(snapshot)).not.toContain('private');
  });

  it('accepts only allowlisted bounded browser metrics and safe runtime summaries', () => {
    const valid = validateAgentMetricsPayload({
      sessionId: 'session-a', messageId: 'message-a', turnId: 'turn-a',
      metrics: {firstVisibleTextMs: 150.5, completionMs: 400, deltaEvents: 12, snapshotRepairs: 1, visible: 1},
    });
    expect(valid).toMatchObject({ok: true, value: {metrics: {firstVisibleTextMs: 150.5, visible: 1}}});
    expect(validateAgentMetricsPayload({sessionId: 'session-a', messageId: 'message-a', metrics: {text: 'private answer'}}).ok).toBe(false);
    expect(validateAgentMetricsPayload({sessionId: 'session-a', messageId: 'message-a', content: 'private answer', metrics: {visible: 1}}).ok).toBe(false);
    expect(validateAgentMetricsPayload({sessionId: 'private answer', messageId: 'message-a', metrics: {visible: 1}}).ok).toBe(false);
    expect(validateAgentMetricsPayload({sessionId: 'session-a', messageId: 'message-a', metrics: {completionMs: 86_400_001}}).ok).toBe(false);
    expect(validateAgentMetricsPayload({sessionId: 'session-a', messageId: 'message-a', metrics: {deltaEvents: 10_000_001}}).ok).toBe(false);

    const safe = safeRuntimePerformanceRecord({
      type: 'performance', turn_id: 'turn-a', record: {provider: 'local', model: 'private answer', duration_ms: 42, answer: 'private answer', delta: 'private token'},
    });
    expect(safe).toMatchObject({type: 'runtime_performance', turn_id: 'turn-a', provider: 'local', duration_ms: 42});
    expect(JSON.stringify(safe)).not.toContain('private');
  });

  it('disconnect threshold bounds SSE writes and rotates bridge logs within injected limits', async () => {
    expect(isSseBackpressureExceeded(90, 10, 100)).toBe(false);
    expect(isSseBackpressureExceeded(91, 10, 100)).toBe(true);

    const root = await mkdtemp(path.join(os.tmpdir(), 'vimax-bridge-log-'));
    roots.push(root);
    const log = new BoundedPerformanceLog(root, () => {}, {maxBytes: 80, maxRecordBytes: 80});
    await Promise.all(Array.from({length: 8}, (_, index) => log.append({type: 'bridge_turn', index})));
    const current = await stat(log.filePath);
    const rotated = await stat(`${log.filePath}.1`);
    expect(current.size).toBeLessThanOrEqual(80);
    expect(rotated.size).toBeLessThanOrEqual(80);
    expect(JSON.parse((await readFile(log.filePath, 'utf8')).trim().split('\n').at(-1)).index).toBe(7);
  });

  it('drops excess telemetry work instead of growing an unbounded disk queue', async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), 'vimax-bridge-pending-'));
    roots.push(root);
    const log = new BoundedPerformanceLog(root, () => {}, {maxPending: 2});
    const accepted = await Promise.all(Array.from({length: 20}, (_, index) => log.append({type: 'bridge_turn', index})));
    expect(accepted.filter(Boolean)).toHaveLength(2);
    expect(accepted.filter((value) => value === false)).toHaveLength(18);
    const records = (await readFile(log.filePath, 'utf8')).trim().split('\n').map((line) => JSON.parse(line));
    expect(records.map((record) => record.index)).toEqual([0, 1]);
  });
});
