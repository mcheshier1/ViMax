import type {MappingState, StreamEvent, WorkspaceLine} from './types.js';

export function createMappingState(): MappingState {
  return {assistantStreaming: false};
}

export function appendUserLine(lines: WorkspaceLine[], text: string): {lines: WorkspaceLine[]; state: MappingState} {
  return {lines: [...lines, {kind: 'user', text}], state: createMappingState()};
}

export function applyStreamEvent(lines: WorkspaceLine[], state: MappingState, event: StreamEvent): {lines: WorkspaceLine[]; state: MappingState} {
  switch (event.type) {
    case 'turn':
    case 'status':
      return {lines, state};
    case 'assistant_start':
      return {lines, state: {assistantStreaming: true, sampleId: event.sample_id}};
    case 'token':
      return appendAssistantToken(lines, state, event);
    case 'assistant_end':
      return repairAssistant(lines, state, event.sample_id, event.text ?? '', event.outcome, event.seq);
    case 'tool_start':
      return append(lines, state, {kind: 'tool', status: 'running', text: `tool ${event.tool?.name ?? 'unknown'} started`});
    case 'tool_progress':
      return append(lines, state, {
        kind: 'tool',
        status: 'running',
        text: compactJoin([`tool ${event.tool?.name ?? 'unknown'}`, event.progress?.stage, event.progress?.message]),
      });
    case 'tool_result': {
      const result = event.tool_result ?? {};
      const ok = result.ok !== false;
      const name = result.name ?? 'unknown';
      const detail = ok ? '' : cleanToolError(result.content);
      return append(lines, state, {kind: 'tool', status: ok ? 'done' : 'error', text: detail ? `tool ${name} error: ${detail}` : `tool ${name} ${ok ? 'done' : 'error'}`});
    }
    case 'terminal':
      return append(lines, state, {kind: 'terminal', text: compactJoin([event.stream ? `[${event.stream}]` : '', event.line])});
    case 'session':
      return {lines, state};
    case 'done': {
      const repaired = event.assistant
        ? repairAssistant(lines, state, event.final_sample_id, event.assistant, 'final')
        : {lines, state};
      return {...repaired, state: createMappingState()};
    }
    case 'error':
      return append(lines, state, {kind: 'error', text: event.message ?? 'Unknown error'});
    default:
      return {lines, state};
  }
}

function append(lines: WorkspaceLine[], state: MappingState, line: WorkspaceLine): {lines: WorkspaceLine[]; state: MappingState} {
  return {lines: [...lines, line], state: {...state, assistantStreaming: false}};
}

function appendAssistantToken(lines: WorkspaceLine[], state: MappingState, event: StreamEvent): {lines: WorkspaceLine[]; state: MappingState} {
  const delta = event.delta ?? '';
  const sampleId = event.sample_id ?? state.sampleId;
  if (!delta) return {lines, state};
  const next = [...lines];
  const index = sampleId
    ? next.findIndex((line) => line.kind === 'assistant' && line.sampleId === sampleId)
    : state.assistantStreaming && next.at(-1)?.kind === 'assistant' ? next.length - 1 : -1;
  const previous = index >= 0 ? next[index] : undefined;
  if (previous?.kind === 'assistant') {
    if (event.seq !== undefined && previous.seq !== undefined && event.seq <= previous.seq) return {lines, state};
    next[index] = {...previous, text: previous.text + delta, ...(event.seq !== undefined ? {seq: event.seq} : {})};
  } else {
    next.push({kind: 'assistant', text: delta, ...(sampleId ? {sampleId} : {}), ...(event.seq !== undefined ? {seq: event.seq} : {})});
  }
  return {lines: next, state: {assistantStreaming: true, ...(sampleId ? {sampleId} : {})}};
}

function repairAssistant(
  lines: WorkspaceLine[], state: MappingState, sampleId: string | undefined, text: string,
  outcome: StreamEvent['outcome'], seq?: number,
): {lines: WorkspaceLine[]; state: MappingState} {
  let index = -1;
  for (let i = lines.length - 1; i >= 0; i -= 1) {
    const line = lines[i];
    if (line.kind === 'assistant' && (!sampleId || line.sampleId === sampleId)) {
      index = i;
      break;
    }
  }
  if (!text && index < 0) return {lines, state: createMappingState()};
  const next = [...lines];
  const previous = index >= 0 ? next[index] : undefined;
  const message: WorkspaceLine = {
    ...(previous?.kind === 'assistant' ? previous : {}),
    kind: 'assistant', text,
    ...(sampleId ? {sampleId} : {}),
    ...(outcome ? {outcome} : {}),
    ...(seq !== undefined ? {seq} : {}),
  };
  if (index >= 0) next[index] = message;
  else next.push(message);
  return {lines: next, state: createMappingState()};
}


function compactJoin(parts: Array<string | undefined | null>): string {
  return parts.map((part) => String(part ?? '').trim()).filter(Boolean).join(': ');
}


function cleanToolError(content: string | undefined): string {
  return String(content ?? '').replace(/\s+/g, ' ').trim();
}
