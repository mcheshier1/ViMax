import {describe, expect, it} from 'vitest';
import {applyAgentEvent, applyAgentQueueSnapshot, appendLocalUser, composeAgentPrompt, createChatState, humanize, setUserDelivery} from './events';

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

  it('updates a running tool instead of appending every progress event', () => {
    let state = createChatState();
    state = applyAgentEvent(state, {type: 'tool_start', turn_id: 'turn-1', tool: {id: 'tool-1', name: 'vimax_render_video'}});
    state = applyAgentEvent(state, {type: 'tool_progress', turn_id: 'turn-1', tool: {name: 'vimax_render_video'}, progress: {stage: 'generate_frames', message: 'Generating frames'}});
    state = applyAgentEvent(state, {type: 'tool_result', turn_id: 'turn-1', tool_result: {name: 'vimax_render_video', ok: true}});
    expect(state.messages).toHaveLength(1);
    expect(state.messages[0]).toMatchObject({tool: 'vimax_render_video', status: 'done', text: 'Completed'});
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
