export function createAgentMessageQueue() {
  let activeSessionId = '';
  let active = null;
  const completedTurnIds = new Set();
  const pending = [];

  function messageView(message) {
    return {id: message.id, text: message.text};
  }

  function snapshot(cancelledIds = []) {
    return {
      type: 'agent_queue',
      activeSessionId,
      busy: active !== null,
      active: active ? messageView(active) : null,
      pending: pending.map(messageView),
      ...(cancelledIds.length ? {cancelledIds} : {}),
    };
  }

  function cancel() {
    const cancelledIds = [
      ...(active ? [active.id] : []),
      ...pending.map((message) => message.id),
    ];
    active = null;
    pending.length = 0;
    return snapshot(cancelledIds);
  }

  return {
    reset(sessionId) {
      const state = cancel();
      activeSessionId = sessionId;
      return snapshot(state.cancelledIds || []);
    },

    setSessionId(sessionId) {
      activeSessionId = sessionId;
      return snapshot();
    },

    cancelPending() {
      const cancelledIds = pending.map((message) => message.id);
      pending.length = 0;
      return snapshot(cancelledIds);
    },

    enqueue(sessionId, message) {
      if (sessionId !== activeSessionId) return {accepted: false, queued: false, started: false};
      message = {...message, sessionId};
      const queued = active !== null;
      if (queued) pending.push(message);
      else active = {...message, turnId: null};
      return {accepted: true, queued, started: !queued};
    },

    consumeEvent(event) {
      if (event?.type === 'turn' && active && typeof event.turn_id === 'string' && event.turn_id) {
        if (completedTurnIds.has(event.turn_id) || (active.turnId && active.turnId !== event.turn_id)) {
          return {messageId: null, completed: false, started: false};
        }
        active.turnId = event.turn_id;
        return {messageId: active.id, sessionId: active.sessionId, completed: false, started: false};
      }
      if (event?.type !== 'done' || !active || !active.turnId || event.turn_id !== active.turnId) {
        return {messageId: null, completed: false, started: false};
      }
      const completed = active;
      completedTurnIds.add(completed.turnId);
      if (completedTurnIds.size > 1_000) completedTurnIds.delete(completedTurnIds.values().next().value);
      active = pending.shift() || null;
      if (active) active.turnId = null;
      return {messageId: completed.id, sessionId: completed.sessionId, completed: true, started: active !== null};
    },

    cancel,
    snapshot: () => snapshot(),
    get activeSessionId() { return activeSessionId; },
    get activeMessage() { return active; },
  };
}
