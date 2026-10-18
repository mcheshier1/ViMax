import {describe, expect, it} from 'vitest';
import {applyAgentEvent, applyAgentQueueSnapshot, appendLocalUser, composeAgentPrompt, createChatState, humanize, mergeHistoryMessages, setUserDelivery} from './events';

describe('agent event mapping', () => {
  it('streams assistant text into one message', () => {
    let state = appendLocalUser(createChatState(), 'Make a short film');
    state = applyAgentEvent(state, {type: 'turn', turn_id: 'turn-1'});
    state = applyAgentEvent(state, {type: 'token', turn_id: 'turn-1', delta: 'Planning'});
    state = applyAgentEvent(state, {type: 'token', turn_id: 'turn-1', delta: ' ready'});
    state = applyAgentEvent(state, {type: 'done', turn_id: 'turn-1'});
    expect(state.messages.at(-1)).toMatchObject({role: 'assistant', text: 'Planning ready'});
    expect(state.busy).toBe(false);
  });

  it('keeps tool preambles separate and repairs each sample from its authoritative end', () => {
    let state = createChatState();
    state = applyAgentEvent(state, {type: 'assistant_start', turn_id: 'turn-2', sample_id: 'turn-2-sample-0', epoch: 'epoch-a', revision: 1});
    state = applyAgentEvent(state, {type: 'token', turn_id: 'turn-2', sample_id: 'turn-2-sample-0', seq: 1, delta: 'Searching', epoch: 'epoch-a', revision: 2});
    state = applyAgentEvent(state, {type: 'assistant_end', turn_id: 'turn-2', sample_id: 'turn-2-sample-0', seq: 1, text: 'Searching files…', outcome: 'tool_preamble', epoch: 'epoch-a', revision: 3});
    state = applyAgentEvent(state, {type: 'assistant_start', turn_id: 'turn-2', sample_id: 'turn-2-sample-1', epoch: 'epoch-a', revision: 4});
    state = applyAgentEvent(state, {type: 'token', turn_id: 'turn-2', sample_id: 'turn-2-sample-1', seq: 1, delta: 'Caf', epoch: 'epoch-a', revision: 5});
    state = applyAgentEvent(state, {type: 'token', turn_id: 'turn-2', sample_id: 'turn-2-sample-1', seq: 1, delta: 'duplicate', epoch: 'epoch-a', revision: 6});
    state = applyAgentEvent(state, {type: 'token', turn_id: 'turn-2', sample_id: 'turn-2-sample-1', seq: 2, delta: 'é **bold**', epoch: 'epoch-a', revision: 7});
    state = applyAgentEvent(state, {type: 'assistant_end', turn_id: 'turn-2', sample_id: 'turn-2-sample-1', seq: 2, text: 'Café **bold**', outcome: 'final', epoch: 'epoch-a', revision: 8});
    state = applyAgentEvent(state, {type: 'done', turn_id: 'turn-2', final_sample_id: 'turn-2-sample-1', assistant: 'Café **bold**', status: 'completed', epoch: 'epoch-a', revision: 9});
    expect(state.messages.filter((message) => message.role === 'assistant')).toEqual([
      expect.objectContaining({id: 'assistant-turn-2-sample-0', text: 'Searching files…', streamState: 'preamble'}),
      expect.objectContaining({id: 'assistant-turn-2-sample-1', text: 'Café **bold**', streamState: 'complete'}),
    ]);
  });

  it('keeps a halted turn explanation separate from an unfinished tool preamble', () => {
    let state = createChatState();
    state = applyAgentEvent(state, {type: 'assistant_start', turn_id: 'turn-halted', sample_id: 'turn-halted-sample-0', epoch: 'epoch-a', revision: 1});
    state = applyAgentEvent(state, {type: 'token', turn_id: 'turn-halted', sample_id: 'turn-halted-sample-0', seq: 1, delta: 'The tool call was requested', epoch: 'epoch-a', revision: 2});
    state = applyAgentEvent(state, {type: 'done', turn_id: 'turn-halted', assistant: 'Maximum tool passes reached.', status: 'halted', epoch: 'epoch-a', revision: 3});
    expect(state.messages.filter((message) => message.role === 'assistant')).toEqual([
      expect.objectContaining({id: 'assistant-turn-halted-sample-0', text: 'The tool call was requested', streamState: 'interrupted'}),
      expect.objectContaining({id: 'assistant-turn-halted', text: 'Maximum tool passes reached.', streamState: 'interrupted'}),
    ]);
  });

  it('repairs missing sequence gaps from snapshots and ignores stale epochs and revisions', () => {
    let state = createChatState();
    state = applyAgentEvent(state, {type: 'assistant_start', turn_id: 'turn-3', sample_id: 'sample-3', epoch: 'epoch-a', revision: 1});
    state = applyAgentEvent(state, {type: 'token', turn_id: 'turn-3', sample_id: 'sample-3', seq: 1, delta: 'A', epoch: 'epoch-a', revision: 2});
    state = applyAgentEvent(state, {type: 'token', turn_id: 'turn-3', sample_id: 'sample-3', seq: 3, delta: 'C', epoch: 'epoch-a', revision: 3});
    state = applyAgentEvent(state, {type: 'stream_snapshot', turn_id: 'turn-3', status: 'running', samples: [{sample_id: 'sample-3', seq: 3, text: 'ABC'}], epoch: 'epoch-a', revision: 4});
    state = applyAgentEvent(state, {type: 'token', turn_id: 'turn-3', sample_id: 'sample-3', seq: 4, delta: 'duplicate revision', epoch: 'epoch-a', revision: 4});
    state = applyAgentEvent(state, {type: 'status', turn_id: 'turn-3', phase: 'preparing_context', epoch: 'epoch-b', revision: 1});
    state = applyAgentEvent(state, {type: 'token', turn_id: 'turn-3', sample_id: 'sample-3', seq: 4, delta: 'stale', epoch: 'epoch-a', revision: 99});
    const message = state.messages.find((item) => item.id === 'assistant-sample-3');
    expect(message).toMatchObject({text: 'ABC', streamState: 'interrupted'});
    expect(state).toMatchObject({epoch: 'epoch-b', revision: 1, phase: 'preparing_context'});
  });

  it('preserves visible partial text when the bridge snapshot requires history recovery', () => {
    let state = createChatState();
    state = applyAgentEvent(state, {type: 'assistant_start', turn_id: 'turn-4', sample_id: 'sample-4', epoch: 'epoch-a', revision: 1});
    state = applyAgentEvent(state, {type: 'token', turn_id: 'turn-4', sample_id: 'sample-4', seq: 1, delta: 'Visible partial', epoch: 'epoch-a', revision: 2});
    state = applyAgentEvent(state, {type: 'stream_snapshot', turn_id: 'turn-4', status: 'running', historyRequired: true, samples: [], epoch: 'epoch-a', revision: 3});
    expect(state.messages.find((message) => message.id === 'assistant-sample-4')).toMatchObject({
      text: 'Visible partial',
      streamState: 'streaming',
      liveOutput: true,
    });
  });

  it('repairs an overflowed terminal response from persisted history without retaining a partial duplicate', () => {
    let state = applyAgentEvent(createChatState(), {type: 'token', turn_id: 'turn-overflow', sample_id: 'sample-overflow', seq: 1, delta: 'Partial', epoch: 'epoch-a', revision: 1});
    state = applyAgentEvent(state, {type: 'stream_snapshot', turn_id: 'turn-overflow', status: 'completed', historyRequired: true, samples: [], epoch: 'epoch-a', revision: 2});
    expect(state).toMatchObject({busy: false, recoveringHistory: true});
    expect(state.messages[0]).toMatchObject({text: 'Partial', streamState: 'interrupted'});
    const merged = mergeHistoryMessages(state.messages, [{id: 'assistant-sample-overflow', role: 'assistant', turnId: 'turn-overflow', text: 'Complete recovered response', streamState: 'complete'}], 3, true);
    expect(merged).toEqual([expect.objectContaining({text: 'Complete recovered response', streamState: 'complete', liveOutput: false})]);
  });

  it('cancels old queued deliveries on bridge restart and accepts immutable terminal history', () => {
    let state = appendLocalUser(createChatState(), 'Queued work', 'message-restart');
    state = applyAgentQueueSnapshot(state, {epoch: 'epoch-a', revision: 1, busy: true, active: {id: 'message-restart', text: 'Queued work'}});
    state = applyAgentEvent(state, {type: 'token', turn_id: 'turn-restart', sample_id: 'sample-restart', seq: 1, delta: 'Partial', epoch: 'epoch-a', revision: 2});
    state = applyAgentQueueSnapshot(state, {epoch: 'epoch-b', revision: 1, busy: false, active: null});
    expect(state.messages.find((message) => message.role === 'user')?.delivery).toBe('cancelled');
    const merged = mergeHistoryMessages(state.messages, [{id: 'assistant-sample-restart', role: 'assistant', text: 'Persisted final', streamState: 'complete'}], 2, false);
    expect(merged.filter((message) => message.role === 'assistant')).toEqual([expect.objectContaining({text: 'Persisted final', streamState: 'complete'})]);
  });

  it('orders snapshot tools between their preamble and following assistant sample', () => {
    const state = applyAgentEvent(createChatState(), {
      type: 'stream_snapshot',
      turn_id: 'turn-rounds',
      status: 'running',
      samples: [
        {sample_id: 'turn-rounds-sample-0', round: 0, seq: 1, text: 'Searching…', outcome: 'tool_preamble'},
        {sample_id: 'turn-rounds-sample-1', round: 1, seq: 1, text: 'Found it', outcome: 'final'},
      ],
      tools: [{id: 'call-1', round: 0, name: 'search_text', status: 'completed'}],
      epoch: 'epoch-rounds',
      revision: 1,
    });
    expect(state.messages.map((message) => message.id)).toEqual([
      'assistant-turn-rounds-sample-0',
      'activity-turn-rounds-call-1',
      'assistant-turn-rounds-sample-1',
    ]);
  });

  it('keeps a recovered failure beside its turn rather than after a later answer', () => {
    const state = applyAgentEvent(createChatState(), {type: 'error', turn_id: 'failed-turn', message: 'Provider disconnected'});
    const merged = mergeHistoryMessages(state.messages, [
      {id: 'old-answer', role: 'assistant', turnId: 'failed-turn', text: 'Partial', streamState: 'interrupted'},
      {id: 'new-answer', role: 'assistant', turnId: 'later-turn', text: 'Completed answer', streamState: 'complete'},
    ]);
    expect(merged.map((message) => [message.role, message.turnId])).toEqual([
      ['assistant', 'failed-turn'], ['error', 'failed-turn'], ['assistant', 'later-turn'],
    ]);
  });

  it('keeps the submitted user text when history contains the expanded model prompt', () => {
    const merged = mergeHistoryMessages(
      [{id: 'message-local', role: 'user', text: 'Revise the ending', delivery: 'done'}],
      [{id: 'message-local', role: 'user', text: 'Revise the ending\nFilm context: internal request context.'}],
    );
    expect(merged).toEqual([expect.objectContaining({text: 'Revise the ending', delivery: 'done'})]);
  });

  it('retains a stopped request and its partial answer before newer persisted work', () => {
    const current = [
      {id: 'stopped-request', role: 'user' as const, text: 'Stopped work', delivery: 'cancelled' as const},
      {id: 'stopped-answer', role: 'assistant' as const, text: 'Visible partial', streamState: 'interrupted' as const},
      {id: 'later-request', role: 'user' as const, text: 'Next request', delivery: 'done' as const},
      {id: 'later-answer', role: 'assistant' as const, text: 'Next answer', streamState: 'complete' as const},
    ];
    const merged = mergeHistoryMessages(current, current.slice(2));
    expect(merged.map((message) => message.id)).toEqual(['stopped-request', 'stopped-answer', 'later-request', 'later-answer']);
    expect(merged[0].delivery).toBe('cancelled');
    expect(merged[1]).toMatchObject({text: 'Visible partial', streamState: 'interrupted'});
  });

  it('merges recovered history without losing local queued users or active partial output', () => {
    const current = [
      {id: 'assistant-live', role: 'assistant' as const, text: 'Partial', turnId: 'turn-live', streamState: 'streaming' as const, revision: 12},
      {id: 'queued-1', role: 'user' as const, text: 'Follow-up', delivery: 'queued' as const},
    ];
    const merged = mergeHistoryMessages(current, [
      {id: 'user-old', role: 'user', text: 'Earlier'},
      {id: 'assistant-live', role: 'assistant', text: 'Old history'},
    ], 15, true);
    expect(merged).toEqual([
      {id: 'user-old', role: 'user', text: 'Earlier'},
      expect.objectContaining({id: 'assistant-live', text: 'Partial', streamState: 'streaming'}),
      expect.objectContaining({id: 'queued-1', text: 'Follow-up', delivery: 'queued'}),
    ]);
  });

  it('updates a running tool instead of appending every progress event', () => {
    let state = createChatState();
    state = applyAgentEvent(state, {type: 'tool_start', turn_id: 'turn-1', tool: {id: 'tool-1', name: 'vimax_render_video'}});
    state = applyAgentEvent(state, {type: 'tool_progress', turn_id: 'turn-1', tool: {name: 'vimax_render_video'}, progress: {stage: 'generate_frames', message: 'Generating frames'}});
    state = applyAgentEvent(state, {type: 'tool_result', turn_id: 'turn-1', tool_result: {name: 'vimax_render_video', ok: true}});
    expect(state.messages).toHaveLength(1);
    expect(state.messages[0]).toMatchObject({
      id: 'activity-turn-1-tool-1',
      turnId: 'turn-1',
      toolCallId: 'tool-1',
      tool: 'vimax_render_video',
      status: 'done',
      text: 'Completed',
    });
  });

  it('keeps each composer submission on one stdin line', () => {
    expect(composeAgentPrompt('first line\nsecond line')).toBe('first line second line');
    expect(composeAgentPrompt('Use these references', ['uploads/script.txt', 'uploads/look.png']))
      .toBe('Use these references <workspace_uploads>["uploads/script.txt","uploads/look.png"]</workspace_uploads>');
    expect(humanize('vimax_narrative_planning')).toBe('ViMax Narrative Planning');
  });

  it('advances queued user rows through FIFO running and done without duplicates', () => {
    let state = appendLocalUser(createChatState(), 'First follow-up', 'message-1');
    state = appendLocalUser(state, 'Second follow-up', 'message-2');
    state = applyAgentQueueSnapshot(state, {
      type: 'agent_queue',
      activeSessionId: 'project-1',
      busy: true,
      active: {id: 'message-current', text: 'Make a film'},
      pending: [
        {id: 'message-1', text: 'First follow-up'},
        {id: 'message-2', text: 'Second follow-up'},
      ],
    });
    expect(state.messages.find((message) => message.id === 'message-1')).toMatchObject({delivery: 'queued'});
    expect(state.messages.find((message) => message.id === 'message-2')).toMatchObject({delivery: 'queued'});
    state = applyAgentEvent(state, {type: 'done', turn_id: 'turn-current', messageId: 'message-current'});
    expect(state.messages.find((message) => message.id === 'message-current')).toMatchObject({delivery: 'done'});
    expect(state.busy).toBe(true);
    state = applyAgentQueueSnapshot(state, {
      type: 'agent_queue',
      activeSessionId: 'project-1',
      busy: true,
      active: {id: 'message-1', text: 'First follow-up'},
      pending: [{id: 'message-2', text: 'Second follow-up'}],
    });
    expect(state.messages.filter((message) => message.id === 'message-1')).toHaveLength(1);
    expect(state.messages.find((message) => message.id === 'message-1')).toMatchObject({delivery: 'running'});
    state = applyAgentEvent(state, {type: 'turn', turn_id: 'turn-1', messageId: 'message-1'});
    state = applyAgentEvent(state, {type: 'done', turn_id: 'turn-1', messageId: 'message-1'});
    expect(state.messages.find((message) => message.id === 'message-1')).toMatchObject({delivery: 'done'});
    expect(state.messages.find((message) => message.id === 'message-2')).toMatchObject({delivery: 'queued'});
    expect(state.busy).toBe(true);
    state = applyAgentQueueSnapshot(state, {
      type: 'agent_queue',
      activeSessionId: 'project-1',
      busy: true,
      active: {id: 'message-2', text: 'Second follow-up'},
      pending: [],
    });
    state = applyAgentEvent(state, {type: 'turn', turn_id: 'turn-2', messageId: 'message-2'});
    state = applyAgentEvent(state, {type: 'done', turn_id: 'turn-2', messageId: 'message-2'});
    expect(state.messages.find((message) => message.id === 'message-2')).toMatchObject({delivery: 'done'});
    expect(state.busy).toBe(false);
  });

  it('marks active and pending rows cancelled when the queue is cleared', () => {
    let state = applyAgentQueueSnapshot(createChatState(), {
      type: 'agent_queue',
      activeSessionId: 'project-1',
      busy: true,
      active: {id: 'active', text: 'Current request'},
      pending: [{id: 'pending', text: 'Follow-up'}],
    });
    state = applyAgentQueueSnapshot(state, {
      type: 'agent_queue',
      activeSessionId: 'project-1',
      busy: false,
      active: null,
      pending: [],
      cancelledIds: ['active', 'pending'],
    });
    expect(state.messages.map((message) => message.delivery)).toEqual(['cancelled', 'cancelled']);
    expect(state.busy).toBe(false);
  });

  it('does not advance an active queue on an error event alone', () => {
    let state = applyAgentQueueSnapshot(createChatState(), {
      type: 'agent_queue',
      activeSessionId: 'project-1',
      busy: true,
      active: {id: 'active', text: 'Current request'},
      pending: [{id: 'pending', text: 'Follow-up'}],
    });
    state = applyAgentEvent(state, {type: 'error', message: 'Transient runtime error'});
    expect(state.busy).toBe(true);
    expect(state.activeMessageId).toBe('active');
    expect(state.messages.find((message) => message.id === 'pending')).toMatchObject({delivery: 'queued'});
  });

  it('keeps the active request busy when a follow-up send fails', () => {
    let state = applyAgentQueueSnapshot(createChatState(), {
      type: 'agent_queue',
      activeSessionId: 'project-1',
      busy: true,
      active: {id: 'active', text: 'Current request'},
      pending: [],
    });
    state = appendLocalUser(state, 'Follow-up', 'failed');
    state = setUserDelivery(state, 'failed', 'error', 'Offline');
    expect(state.messages.find((message) => message.id === 'active')).toMatchObject({delivery: 'running'});
    expect(state.messages.find((message) => message.id === 'failed')).toMatchObject({delivery: 'error', deliveryError: 'Offline'});
    expect(state.busy).toBe(true);
  });

  it('hydrates a queued snapshot onto the matching history row', () => {
    const history = createChatState([{
      id: 'message-1',
      role: 'user',
      text: 'Follow-up from before refresh',
      turnId: 'turn-1',
    }]);
    const restored = applyAgentQueueSnapshot(history, {
      type: 'agent_queue',
      activeSessionId: 'project-1',
      busy: true,
      active: {id: 'message-current', text: 'Current request'},
      pending: [{id: 'message-1', text: 'Follow-up from before refresh'}],
    });
    expect(restored.messages.filter((message) => message.id === 'message-1')).toHaveLength(1);
    expect(restored.messages.find((message) => message.id === 'message-1')).toMatchObject({turnId: 'turn-1', delivery: 'queued'});
  });

  it('ends running tools when the agent process stops', () => {
    let state = applyAgentEvent(createChatState(), {
      type: 'tool_start',
      turn_id: 'turn-1',
      tool: {id: 'tool-1', name: 'vimax_render_video'},
    });
    state = applyAgentEvent(state, {
      type: 'bridge_status',
      status: 'stopped',
      message: 'Configuration updated',
    });
    expect(state.busy).toBe(false);
    expect(state.messages[0]).toMatchObject({
      tool: 'vimax_render_video',
      status: 'error',
      stage: 'interrupted',
      text: 'Configuration updated',
    });
  });
});
