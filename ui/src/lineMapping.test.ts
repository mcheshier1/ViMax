import assert from 'node:assert/strict';
import {applyStreamEvent, createMappingState} from './lineMapping.js';
import type {MappingState, WorkspaceLine} from './types.js';

let lines: WorkspaceLine[] = [];
let state: MappingState = createMappingState();

lines = [...lines, {kind: 'user', text: 'start'}];
assert.deepEqual(lines[0], {kind: 'user', text: 'start'});

({lines, state} = applyStreamEvent(lines, state, {type: 'token', delta: 'hello'}));
({lines, state} = applyStreamEvent(lines, state, {type: 'token', delta: ' world'}));
assert.equal(lines.length, 2);
assert.deepEqual(lines[1], {kind: 'assistant', text: 'hello world'});

({lines, state} = applyStreamEvent(lines, state, {type: 'tool_start', tool: {name: 'vimax_narrative_planning'}}));
({lines, state} = applyStreamEvent(lines, state, {type: 'tool_progress', tool: {name: 'vimax_narrative_planning'}, progress: {stage: 'running', message: 'planning'}}));
({lines, state} = applyStreamEvent(lines, state, {type: 'tool_result', tool_result: {name: 'vimax_narrative_planning', ok: true}}));
assert.equal(lines.at(-3)?.kind, 'tool');
assert.equal(lines.at(-2)?.kind, 'tool');
assert.deepEqual(lines.at(-1), {kind: 'tool', status: 'done', text: 'tool vimax_narrative_planning done'});

({lines, state} = applyStreamEvent(lines, state, {type: 'tool_result', tool_result: {name: 'vimax_narrative_planning', ok: false, content: 'Developing story failed: Request timed out.'}}));
assert.deepEqual(lines.at(-1), {kind: 'tool', status: 'error', text: 'tool vimax_narrative_planning error: Developing story failed: Request timed out.'});

({lines, state} = applyStreamEvent(lines, state, {type: 'terminal', stream: 'stdout', line: 'render log'}));
assert.deepEqual(lines.at(-1), {kind: 'terminal', text: '[stdout]: render log'});

const lengthBeforeInternalEvents = lines.length;
({lines, state} = applyStreamEvent(lines, state, {type: 'turn', turn_id: 'turn-1'}));
({lines, state} = applyStreamEvent(lines, state, {type: 'status', phase: 'sampling_assistant', message: 'Sampling assistant'}));
({lines, state} = applyStreamEvent(lines, state, {type: 'session', session: {active_session_id: 's1', session: {session_id: 's1', stage: 'narrative_planned', working_dir: '.working_dir/s1'}}}));
assert.equal(lines.length, lengthBeforeInternalEvents);

({lines, state} = applyStreamEvent(lines, state, {type: 'error', message: 'bad'}));
assert.deepEqual(lines.at(-1), {kind: 'error', text: 'bad'});

({lines, state} = applyStreamEvent(lines, state, {type: 'done', assistant: 'hello world'}));
assert.equal(state.assistantStreaming, false);

// Samples stay separate across a tool round; authoritative ends repair missed deltas.
lines = [];
state = createMappingState();
({lines, state} = applyStreamEvent(lines, state, {type: 'assistant_start', sample_id: 'turn-a-sample-0'}));
({lines, state} = applyStreamEvent(lines, state, {type: 'token', sample_id: 'turn-a-sample-0', seq: 1, delta: 'Checking'}));
({lines, state} = applyStreamEvent(lines, state, {type: 'token', sample_id: 'turn-a-sample-0', seq: 1, delta: 'Checking'}));
({lines, state} = applyStreamEvent(lines, state, {type: 'assistant_end', sample_id: 'turn-a-sample-0', seq: 2, text: 'Checking local state.', outcome: 'tool_preamble'}));
({lines, state} = applyStreamEvent(lines, state, {type: 'tool_result', tool_result: {name: 'read_file', ok: true}}));
({lines, state} = applyStreamEvent(lines, state, {type: 'assistant_start', sample_id: 'turn-a-sample-1'}));
({lines, state} = applyStreamEvent(lines, state, {type: 'token', sample_id: 'turn-a-sample-1', seq: 1, delta: 'The result'}));
({lines, state} = applyStreamEvent(lines, state, {type: 'assistant_end', sample_id: 'turn-a-sample-1', seq: 2, text: 'The result is ready.', outcome: 'final'}));
({lines, state} = applyStreamEvent(lines, state, {type: 'done', final_sample_id: 'turn-a-sample-1', assistant: 'The result is ready.'}));
assert.deepEqual(lines.filter((line) => line.kind === 'assistant').map((line) => [line.text, line.outcome]), [
  ['Checking local state.', 'tool_preamble'], ['The result is ready.', 'final'],
]);
assert.equal(state.assistantStreaming, false);

// A failed stream retains its incomplete evidence without becoming a final answer.
({lines, state} = applyStreamEvent(lines, state, {type: 'assistant_start', sample_id: 'turn-b-sample-0'}));
({lines, state} = applyStreamEvent(lines, state, {type: 'token', sample_id: 'turn-b-sample-0', seq: 1, delta: 'Partial ☃'}));
({lines, state} = applyStreamEvent(lines, state, {type: 'assistant_end', sample_id: 'turn-b-sample-0', seq: 1, text: 'Partial ☃', outcome: 'interrupted'}));
({lines, state} = applyStreamEvent(lines, state, {type: 'done', status: 'failed', assistant: ''}));
assert.deepEqual(lines.at(-1), {kind: 'assistant', sampleId: 'turn-b-sample-0', seq: 1, text: 'Partial ☃', outcome: 'interrupted'});

console.log('lineMapping tests passed');
