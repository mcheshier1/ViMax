import type {AgentEvent, ChatState, Message, StreamToolSnapshot} from './types';

export function createChatState(messages: Message[] = []): ChatState {
  return {messages, busy: false, turnId: '', promptTokens: 0, queueSupported: false, activeMessageId: '', epoch: '', revision: 0, phase: '', retiredEpochs: []};
}

export function appendLocalUser(state: ChatState, text: string, id = `local-user-${Date.now()}-${state.messages.length}`): ChatState {
  return {...state, messages: upsertUserMessage(state.messages, id, text, 'sending')};
}

export function setUserDelivery(state: ChatState, id: string, delivery: NonNullable<Message['delivery']>, deliveryError?: string): ChatState {
  return {
    ...state,
    messages: state.messages.map((message) => message.id === id ? {...message, delivery, deliveryError} : message),
  };
}

export function applyAgentQueueSnapshot(state: ChatState, event: AgentEvent): ChatState {
  const next = acceptRevision(state, event);
  if (!next) return state;
  const active = event.active || null;
  const pending = event.pending || [];
  const cancelledIds = new Set(event.cancelledIds || []);
  const messages = [...next.messages];
  const updateDelivery = (id: string, delivery: NonNullable<Message['delivery']>) => {
    const index = messages.findIndex((message) => message.id === id);
    if (index >= 0) messages[index] = {...messages[index], delivery, deliveryError: undefined};
  };
  const addOrUpdate = (item: {id: string; text: string}, delivery: 'running' | 'queued') => {
    const existing = messages.findIndex((message) => message.id === item.id);
    if (existing < 0) messages.push({id: item.id, role: 'user', text: item.text, delivery});
    else messages[existing] = {...messages[existing], role: 'user', text: messages[existing].text || item.text, delivery, deliveryError: undefined};
  };

  if (next.activeMessageId && next.activeMessageId !== active?.id && !cancelledIds.has(next.activeMessageId)) {
    updateDelivery(next.activeMessageId, 'done');
  }
  if (active) addOrUpdate(active, 'running');
  for (const item of pending) addOrUpdate(item, 'queued');
  for (const id of cancelledIds) updateDelivery(id, 'cancelled');
  return {...next, messages, busy: Boolean(event.busy), queueSupported: true, activeMessageId: active?.id || ''};
}

function upsertUserMessage(messages: Message[], id: string, text: string, delivery: NonNullable<Message['delivery']>) {
  const index = messages.findIndex((message) => message.id === id);
  if (index < 0) return [...messages, {id, role: 'user' as const, text, delivery}];
  return messages.map((message) => message.id === id
    ? {...message, role: 'user' as const, text: message.text || text, delivery}
    : message);
}

export function composeAgentPrompt(text: string, workspaceUploads: string[] = []) {
  const message = text.replace(/\s+/g, ' ').trim();
  const paths = workspaceUploads.map((path) => path.trim()).filter(Boolean);
  if (paths.length === 0) return message;
  return `${message} <workspace_uploads>${JSON.stringify(paths)}</workspace_uploads>`;
}

export function applyAgentEvent(state: ChatState, event: AgentEvent): ChatState {
  const accepted = acceptRevision(state, event);
  if (!accepted) return state;
  const turnId = event.turn_id || accepted.turnId || `turn-${Date.now()}`;
  switch (event.type) {
    case 'turn': {
      const activeMessageId = event.messageId || accepted.activeMessageId;
      const next = activeMessageId ? setUserDelivery(accepted, activeMessageId, 'running') : accepted;
      return {...next, busy: true, turnId, activeMessageId, recoveringHistory: false};
    }
    case 'status':
      return {...accepted, turnId, phase: event.phase || accepted.phase, busy: true};
    case 'prompt_trace': {
      const tokens = event.prompt_trace?.totals?.total_tokens
        ?? event.prompt_trace?.totals?.total_estimated_tokens
        ?? event.prompt_trace?.total_estimated_tokens
        ?? accepted.promptTokens;
      return {...accepted, promptTokens: Math.max(0, Math.round(tokens))};
    }
    case 'assistant_start':
      return upsertAssistant(accepted, {
        id: sampleMessageId(event.sample_id, turnId),
        role: 'assistant',
        turnId,
        sampleId: event.sample_id,
        round: event.round,
        text: '',
        streamState: 'streaming',
        seq: 0,
        epoch: event.epoch,
        revision: event.revision,
      }, false);
    case 'token':
      return appendAssistantDelta(accepted, event, turnId);
    case 'assistant_end':
      return upsertAssistant(accepted, {
        id: sampleMessageId(event.sample_id, turnId),
        role: 'assistant',
        turnId,
        sampleId: event.sample_id,
        round: event.round,
        text: typeof event.text === 'string' ? event.text : '',
        streamState: streamStateForOutcome(event.outcome),
        seq: event.seq,
        epoch: event.epoch,
        revision: event.revision,
      }, true);
    case 'stream_snapshot':
      return applyStreamSnapshot(accepted, event, turnId);
    case 'tool_start': {
      const name = event.tool?.name || event.tool?.requested_name || 'tool';
      return upsertActivity(accepted, {
        id: activityId(turnId, event.tool?.id, name),
        role: 'activity',
        round: event.round,
        turnId,
        toolCallId: event.tool?.id,
        tool: name,
        status: 'running',
        stage: 'starting',
        text: 'Starting',
      });
    }
    case 'tool_progress': {
      const name = event.tool?.name || event.tool?.requested_name || 'tool';
      return upsertLatestToolActivity(accepted, name, {
        role: 'activity',
        round: event.round,
        turnId,
        ...(event.tool?.id ? {toolCallId: event.tool.id} : {}),
        tool: name,
        status: 'running',
        stage: event.progress?.stage || 'running',
        text: event.progress?.message || humanize(event.progress?.stage || 'Running'),
      });
    }
    case 'tool_result': {
      const name = event.tool_result?.name || 'tool';
      const ok = event.tool_result?.ok !== false;
      return upsertLatestToolActivity(accepted, name, {
        role: 'activity',
        round: event.round,
        turnId,
        ...(event.tool_result?.id ? {toolCallId: event.tool_result.id} : {}),
        tool: name,
        status: ok ? 'done' : 'error',
        stage: ok ? 'completed' : 'failed',
        text: ok ? 'Completed' : cleanError(event.tool_result?.content || 'Tool failed'),
      });
    }
    case 'terminal':
      if (event.stream !== 'stderr') return accepted;
      return {
        ...accepted,
        messages: [...accepted.messages, {id: `terminal-${Date.now()}`, role: 'activity', status: 'error', tool: 'runtime', text: cleanError(event.line || 'Runtime error')}],
      };
    case 'error':
      return {
        ...accepted,
        busy: accepted.queueSupported ? accepted.busy : false,
        messages: [...accepted.messages, {id: `error-${turnId}-${Date.now()}`, role: 'error', turnId, text: event.message || 'Unknown agent error'}],
      };
    case 'done': {
      let messages = accepted.messages;
      const target = event.final_sample_id
        ? sampleMessageId(event.final_sample_id, turnId)
        : event.status === 'halted' ? undefined : findFinalMessageId(messages, turnId);
      if (typeof event.assistant === 'string' && event.assistant.length > 0) {
        const messageId = target || `assistant-${turnId}`;
        messages = upsertAssistantMessages(messages, {
          id: messageId,
          role: 'assistant',
          turnId,
          sampleId: event.final_sample_id,
          text: event.assistant,
          streamState: event.status === 'completed' ? 'complete' : 'interrupted',
          epoch: event.epoch,
          revision: event.revision,
        }, true);
      }
      messages = messages.map((message) => message.role === 'assistant' && message.turnId === turnId && message.streamState === 'streaming'
        ? {...message, streamState: event.status === 'completed' ? 'complete' : 'interrupted'}
        : message);
      const messageId = event.messageId || accepted.activeMessageId;
      const completed = messageId ? setUserDelivery({...accepted, messages}, messageId, 'done') : {...accepted, messages};
      const hasQueued = completed.messages.some((message) => message.role === 'user' && message.delivery === 'queued');
      return {
        ...completed,
        busy: accepted.queueSupported ? hasQueued : false,
        turnId,
        phase: '',
        activeMessageId: accepted.activeMessageId === messageId ? '' : accepted.activeMessageId,
      };
    }
    case 'bridge_status':
      if (event.status !== 'error' && event.status !== 'stopped') return accepted;
      return {
        ...accepted,
        busy: false,
        phase: '',
        messages: accepted.messages.map((message) => message.role === 'activity' && message.status === 'running'
          ? {...message, status: 'error', stage: 'interrupted', text: event.message || 'Generation stopped'}
          : message.role === 'assistant' && message.streamState === 'streaming'
            ? {...message, streamState: 'interrupted'}
            : message),
      };
    default:
      return accepted;
  }
}

function acceptRevision(state: ChatState, event: AgentEvent): ChatState | null {
  let next = state;
  if (event.epoch && state.epoch && event.epoch !== state.epoch) {
    if (state.retiredEpochs.includes(event.epoch)) return null;
    next = {
      ...state,
      epoch: event.epoch,
      revision: 0,
      phase: '',
      activeMessageId: '',
      busy: false,
      recoveringHistory: false,
      retiredEpochs: [...state.retiredEpochs.slice(-7), state.epoch],
      messages: state.messages.map((message) => message.role === 'assistant' && message.streamState === 'streaming'
        ? {...message, streamState: 'interrupted'}
        : message.role === 'activity' && message.status === 'running'
          ? {...message, status: 'error', stage: 'interrupted', text: 'Connection restarted'}
          : message.role === 'user' && ['sending', 'queued', 'running'].includes(message.delivery || '')
            ? {...message, delivery: 'cancelled'}
            : message),
    };
  } else if (event.epoch && !state.epoch) {
    next = {...state, epoch: event.epoch};
  }
  if (typeof event.revision === 'number') {
    if (event.epoch === next.epoch && event.revision <= next.revision) return null;
    if (!event.epoch && event.revision <= next.revision) return null;
    next = {...next, revision: event.revision};
  }
  return next;
}

function sampleMessageId(sampleId: string | undefined, turnId: string) {
  return sampleId ? `assistant-${sampleId}` : `assistant-${turnId}`;
}

function streamStateForOutcome(outcome: AgentEvent['outcome']): NonNullable<Message['streamState']> {
  if (outcome === 'tool_preamble') return 'preamble';
  if (outcome === 'interrupted') return 'interrupted';
  if (outcome === 'truncated') return 'truncated';
  return 'complete';
}

function upsertAssistant(state: ChatState, message: Message, authoritative: boolean): ChatState {
  return {...state, messages: upsertAssistantMessages(state.messages, message, authoritative)};
}

function upsertAssistantMessages(messages: Message[], incoming: Message, authoritative: boolean): Message[] {
  const index = messages.findIndex((message) => message.id === incoming.id);
  if (index < 0) return [...messages, incoming];
  const next = [...messages];
  const current = next[index];
  next[index] = {
    ...current,
    ...incoming,
    text: authoritative ? incoming.text : current.text || incoming.text,
    seq: incoming.seq ?? current.seq,
  };
  return next;
}

function appendAssistantDelta(state: ChatState, event: AgentEvent, turnId: string): ChatState {
  const delta = event.delta || '';
  if (!delta) return state;
  const id = sampleMessageId(event.sample_id, turnId);
  const index = state.messages.findIndex((message) => message.id === id);
  const current = index < 0 ? undefined : state.messages[index];
  if (typeof event.seq === 'number' && current?.seq !== undefined && event.seq <= current.seq) return state;
  if (typeof event.seq === 'number' && current?.seq !== undefined && event.seq > current.seq + 1) return state;
  if (typeof event.seq === 'number' && current?.seq === undefined && event.seq > 1) return state;
  const nextMessage: Message = {
    id, role: 'assistant', turnId, sampleId: event.sample_id,
    text: `${current?.text || ''}${delta}`,
    streamState: current?.streamState === 'interrupted' ? 'interrupted' : 'streaming',
    seq: event.seq ?? ((current?.seq || 0) + 1),
    liveOutput: true,
    epoch: event.epoch,
    revision: event.revision,
  };
  return upsertAssistant(state, nextMessage, true);
}

function applyStreamSnapshot(state: ChatState, event: AgentEvent, turnId: string): ChatState {
  let next: ChatState = {...state, turnId, phase: event.phase || state.phase, recoveringHistory: Boolean(event.historyRequired)};
  if (!event.historyRequired) {
    const samples = event.samples || [];
    const finalSampleId = event.final_sample_id || samples.at(-1)?.sample_id;
    for (const sample of samples) {
      const statusOutcome = event.status === 'interrupted' || event.status === 'failed' ? 'interrupted'
        : event.status === 'completed' ? sample.sample_id === finalSampleId ? 'final' : 'tool_preamble'
          : undefined;
      const outcome = sample.outcome || statusOutcome;
      next = upsertAssistant(next, {
        id: sampleMessageId(sample.sample_id, turnId),
        role: 'assistant', turnId, sampleId: sample.sample_id, text: sample.text,
        round: sample.round,
        liveOutput: false,
        streamState: outcome ? streamStateForOutcome(outcome) : 'streaming',
        seq: sample.seq, epoch: event.epoch, revision: event.revision,
      }, true);
    }
  }
  for (const tool of event.tools || []) next = applySnapshotTool(next, tool, turnId);
  const terminal = event.status === 'completed' || event.status === 'failed' || event.status === 'interrupted';
  const messages = orderTurnOutput(next.messages, turnId).map((message) => (
    terminal && event.historyRequired && message.turnId === turnId && message.streamState === 'streaming'
      ? {...message, streamState: 'interrupted' as const} : message
  ));
  return {...next, messages, busy: event.status === 'running' ? true : terminal ? false : next.busy, phase: terminal ? '' : next.phase};
}
function orderTurnOutput(messages: Message[], turnId: string): Message[] {
  const indexes: number[] = [];
  const rows: Message[] = [];
  for (let index = 0; index < messages.length; index += 1) {
    const message = messages[index];
    if (message.turnId === turnId && (message.role === 'assistant' || message.role === 'activity')) {
      indexes.push(index);
      rows.push(message);
    }
  }
  if (rows.length < 2 || rows.some((message) => message.round === undefined)) return messages;
  const ordered = [...rows].sort((left, right) => {
    const roundDifference = (left.round || 0) - (right.round || 0);
    if (roundDifference) return roundDifference;
    if (left.role === right.role) return 0;
    return left.role === 'assistant' ? -1 : 1;
  });
  const result = [...messages];
  indexes.forEach((index, position) => { result[index] = ordered[position]; });
  return result;
}

function applySnapshotTool(state: ChatState, tool: AgentEvent | StreamToolSnapshot, turnId: string): ChatState {
  if ('type' in tool && tool.type) {
    const {type: _type, epoch: _epoch, revision: _revision, ...toolEvent} = tool;
    return applyAgentEvent(state, {...toolEvent, turn_id: toolEvent.turn_id || turnId});
  }
  const summary = tool as StreamToolSnapshot;
  const name = summary.name || summary.requested_name || 'tool';
  const status = summary.status || 'running';
  const messageStatus = status === 'running' ? 'running' : status === 'completed' ? 'done' : 'error';
  return upsertActivity(state, {
    id: activityId(turnId, summary.id, name),
    round: summary.round,
    role: 'activity',
    turnId,
    toolCallId: summary.id,
    tool: name,
    status: messageStatus,
    stage: summary.stage || (status === 'running' ? 'running' : status === 'completed' ? 'completed' : 'failed'),
    text: summary.text || (status === 'running' ? 'Running' : status === 'completed' ? 'Completed' : 'Failed'),
  });
}

function findFinalMessageId(messages: Message[], turnId: string) {
  return [...messages].reverse().find((message) => message.role === 'assistant' && message.turnId === turnId && (message.streamState === 'streaming' || message.streamState === 'complete'))?.id;
}

/** Reconcile persisted history with live rows without dropping queued drafts or active partial output. */
export function mergeHistoryMessages(current: Message[], history: Message[], historyRevision?: number, sameEpoch = false): Message[] {
  const currentById = new Map(current.map((message) => [message.id, message]));
  const historyIds = new Set(history.map((message) => message.id));
  const merged = history.map((stored) => {
    const live = currentById.get(stored.id);
    if (!live) return stored;
    if (live.role === 'user') {
      return {...stored, text: live.text, delivery: live.delivery || stored.delivery, deliveryError: live.deliveryError};
    }
    if (live.role === 'assistant' && stored.streamState && stored.streamState !== 'streaming') {
      // Persisted samples are immutable terminal records, including across bridge restarts.
      if (sameEpoch && historyRevision !== undefined && (live.revision || 0) > historyRevision) return live;
      return {...stored, liveOutput: false};
    }
    if (live.role === 'assistant' && live.streamState === 'streaming') return live;
    if (live.role === 'assistant' && live.streamState === 'interrupted') {
      return sameEpoch && typeof historyRevision === 'number' && (live.revision || 0) <= historyRevision ? stored : live;
    }
    if (live.role === 'assistant' && live.revision !== undefined && (!sameEpoch || typeof historyRevision !== 'number' || live.revision > historyRevision)) return live;
    return stored;
  });
  for (const [position, live] of current.entries()) {
    if (historyIds.has(live.id)) continue;
    const retain = live.role === 'user' && ['sending', 'queued', 'running', 'cancelled', 'error'].includes(live.delivery || '')
      || live.role === 'assistant' && (live.streamState === 'streaming' || live.streamState === 'interrupted'
        || live.revision !== undefined && (!sameEpoch || typeof historyRevision !== 'number' || live.revision > historyRevision))
      || live.role === 'activity' && live.status === 'running'
      || live.role === 'error';
    if (retain) insertRecoveredMessage(merged, current, position);
  }
  return merged;
}

function insertRecoveredMessage(merged: Message[], current: Message[], position: number): void {
  const message = current[position];
  if (message.role === 'error' && message.turnId) {
    for (let index = merged.length - 1; index >= 0; index -= 1) {
      if (merged[index].turnId !== message.turnId) continue;
      merged.splice(index + 1, 0, message);
      return;
    }
  }
  // Keep unpersisted interrupted turns beside their local request, not after newer history.
  for (let index = position - 1; index >= 0; index -= 1) {
    const anchor = merged.findIndex((candidate) => candidate.id === current[index].id);
    if (anchor < 0) continue;
    merged.splice(anchor + 1, 0, message);
    return;
  }
  for (let index = position + 1; index < current.length; index += 1) {
    const anchor = merged.findIndex((candidate) => candidate.id === current[index].id);
    if (anchor < 0) continue;
    merged.splice(anchor, 0, message);
    return;
  }
  merged.push(message);
}

function upsertActivity(state: ChatState, message: Message): ChatState {
  const index = state.messages.findIndex((item) => item.id === message.id);
  if (index < 0) return {...state, messages: [...state.messages, message]};
  const messages = [...state.messages];
  messages[index] = {...messages[index], ...message};
  return {...state, messages};
}

function upsertLatestToolActivity(state: ChatState, tool: string, update: Omit<Message, 'id'>): ChatState {
  const messages = [...state.messages];
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const current = messages[index];
    const sameToolCall = typeof update.toolCallId === 'string'
      ? current.toolCallId === update.toolCallId
      : current.tool === tool && (!update.turnId || current.turnId === update.turnId);
    if (current.role === 'activity' && current.status === 'running' && sameToolCall) {
      messages[index] = {...current, ...update};
      return {...state, messages};
    }
  }
  const turnId = typeof update.turnId === 'string' ? update.turnId : '';
  const toolCallId = typeof update.toolCallId === 'string' ? update.toolCallId : undefined;
  const id = turnId ? activityId(turnId, toolCallId, tool) : `activity-${tool}-${Date.now()}`;
  return {...state, messages: [...messages, {id, ...update}]};
}

function activityId(turnId: string, toolId: string | undefined, name: string) {
  return `activity-${turnId}-${toolId || name}`;
}
export function humanize(value: string) {
  return value
    .replace(/[_-]+/g, ' ')
    .replace(/\b\w/g, (letter) => letter.toUpperCase())
    .replace(/\bVimax\b/g, 'ViMax');
}

function cleanError(value: string) {
  return value.replace(/\s+/g, ' ').trim();
}
