import {describe, expect, it} from 'vitest';
import {createAgentMessageQueue} from './agent-message-queue.mjs';

const message = (id, text = id) => ({id, text, input: `prompt:${text}`});

describe('agent message queue', () => {
  it('starts one request and promotes pending requests FIFO only on its matching done event', () => {
    const queue = createAgentMessageQueue();
    queue.reset('session-a');
    expect(queue.enqueue('session-a', message('one'))).toMatchObject({accepted: true, queued: false, started: true});
    expect(queue.enqueue('session-a', message('two'))).toMatchObject({accepted: true, queued: true, started: false});
    expect(queue.enqueue('session-a', message('three'))).toMatchObject({accepted: true, queued: true, started: false});
    expect(queue.snapshot()).toMatchObject({
      busy: true,
      active: {id: 'one', text: 'one'},
      pending: [{id: 'two', text: 'two'}, {id: 'three', text: 'three'}],
    });

    expect(queue.consumeEvent({type: 'turn', turn_id: 'turn-one'})).toMatchObject({messageId: 'one'});
    expect(queue.consumeEvent({type: 'error', turn_id: 'turn-one'})).toMatchObject({completed: false});
    expect(queue.snapshot().active).toEqual({id: 'one', text: 'one'});
    expect(queue.consumeEvent({type: 'done', turn_id: 'stale-turn'})).toMatchObject({completed: false});
    expect(queue.snapshot().pending).toHaveLength(2);

    expect(queue.consumeEvent({type: 'done', turn_id: 'turn-one'})).toMatchObject({messageId: 'one', completed: true, started: true});
    expect(queue.snapshot()).toMatchObject({active: {id: 'two', text: 'two'}, pending: [{id: 'three', text: 'three'}]});
    expect(queue.consumeEvent({type: 'turn', turn_id: 'turn-one'})).toMatchObject({messageId: null, completed: false});
    expect(queue.consumeEvent({type: 'done', turn_id: 'turn-one'})).toMatchObject({completed: false});
    expect(queue.snapshot().active).toEqual({id: 'two', text: 'two'});
    queue.consumeEvent({type: 'turn', turn_id: 'turn-two'});
    expect(queue.consumeEvent({type: 'done', turn_id: 'turn-two'})).toMatchObject({messageId: 'two', completed: true, started: true});
    queue.consumeEvent({type: 'turn', turn_id: 'turn-three'});
    expect(queue.consumeEvent({type: 'done', turn_id: 'turn-three'})).toMatchObject({messageId: 'three', completed: true, started: false});
    expect(queue.snapshot()).toEqual({type: 'agent_queue', activeSessionId: 'session-a', busy: false, active: null, pending: []});
  });

  it('cancels active and pending requests on stop or session switch without dispatching them elsewhere', () => {
    const queue = createAgentMessageQueue();
    queue.reset('session-a');
    queue.enqueue('session-a', message('active'));
    queue.enqueue('session-a', message('pending'));
    expect(queue.cancel()).toEqual({
      type: 'agent_queue',
      activeSessionId: 'session-a',
      busy: false,
      active: null,
      pending: [],
      cancelledIds: ['active', 'pending'],
    });
    expect(queue.enqueue('session-a', message('after-stop')).started).toBe(true);
    expect(queue.enqueue('session-a', message('second-after-stop')).queued).toBe(true);
    const switched = queue.reset('session-b');
    expect(switched).toEqual({
      type: 'agent_queue',
      activeSessionId: 'session-b',
      busy: false,
      active: null,
      pending: [],
      cancelledIds: ['after-stop', 'second-after-stop'],
    });
    expect(queue.enqueue('session-a', message('wrong-session'))).toEqual({accepted: false, queued: false, started: false});
    expect(queue.enqueue('session-b', message('new-project'))).toMatchObject({accepted: true, started: true});
    expect(queue.snapshot().active).toEqual({id: 'new-project', text: 'new-project'});
  });

  it('provides a complete initial SSE queue snapshot including the active session', () => {
    const queue = createAgentMessageQueue();
    queue.reset('session-a');
    expect(queue.snapshot()).toEqual({type: 'agent_queue', activeSessionId: 'session-a', busy: false, active: null, pending: []});
    queue.enqueue('session-a', message('first', 'original visible text'));
    expect(queue.snapshot()).toEqual({
      type: 'agent_queue',
      activeSessionId: 'session-a',
      busy: true,
      active: {id: 'first', text: 'original visible text'},
      pending: [],
    });
  });
  it('keeps a new-project requirement as the first active turn until it completes', () => {
    const queue = createAgentMessageQueue();
    queue.reset('');
    expect(queue.enqueue('', message('requirement', 'Create a short film'))).toMatchObject({accepted: true, queued: false, started: true});
    queue.setSessionId('created-session');
    expect(queue.snapshot()).toMatchObject({
      activeSessionId: 'created-session',
      active: {id: 'requirement', text: 'Create a short film'},
      pending: [],
    });
    queue.consumeEvent({type: 'turn', turn_id: 'requirement-turn'});
    expect(queue.consumeEvent({type: 'done', turn_id: 'requirement-turn'})).toMatchObject({messageId: 'requirement', completed: true});
    expect(queue.snapshot().busy).toBe(false);
  });

  it('cancels old-project follow-ups while preserving the running turn identity on a session change', () => {
    const queue = createAgentMessageQueue();
    queue.reset('session-a');
    queue.enqueue('session-a', message('running'));
    queue.enqueue('session-a', message('old-follow-up'));
    queue.consumeEvent({type: 'turn', turn_id: 'running-turn'});
    expect(queue.cancelPending()).toMatchObject({
      activeSessionId: 'session-a', active: {id: 'running'},
      pending: [], cancelledIds: ['old-follow-up'],
    });
    queue.setSessionId('session-b');
    expect(queue.enqueue('session-a', message('wrong-project')).accepted).toBe(false);
    expect(queue.consumeEvent({type: 'done', turn_id: 'running-turn'})).toMatchObject({
      messageId: 'running', sessionId: 'session-a', completed: true, started: false,
    });
    expect(queue.snapshot()).toMatchObject({activeSessionId: 'session-b', busy: false, pending: []});
  });

});
