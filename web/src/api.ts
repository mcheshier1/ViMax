import type {AcceptanceUpdateRequest, AgentConfig, AgentEvent, Artifact, ContinuityPayload, JsonValue, Message, ProjectMetadata, ProjectUpdateRequest, ProjectUpdateResponse, RemovedShot, RenderAcceptance, SessionSummary, ShotPlan, ShotPlanUpdateRequest, WorkspaceUpload} from './types';
import type {FilmProgress, FilmSnapshot} from './filmSession';

export async function getSessions() {
  return request<{activeSessionId: string; sessions: SessionSummary[]}>('/api/sessions');
}

export async function readProject(sessionId: string) {
  return request<ProjectMetadata>(`/api/session?session=${encodeURIComponent(sessionId)}`);
}

export async function updateProject(payload: ProjectUpdateRequest) {
  return request<ProjectUpdateResponse>('/api/session', {method: 'PUT', body: JSON.stringify(payload)});
}

export async function deleteSession(sessionId: string) {
  return request<{activeSessionId: string; sessions: SessionSummary[]}>(`/api/sessions?session=${encodeURIComponent(sessionId)}`, {method: 'DELETE'});
}

export async function getAgentConfig() {
  return request<AgentConfig>('/api/config');
}

export async function saveAgentConfig(config: AgentConfig) {
  return request<AgentConfig>('/api/config', {method: 'PUT', body: JSON.stringify(config)});
}

export async function getHistory(sessionId: string) {
  return request<{messages: Message[]}>(`/api/history?session=${encodeURIComponent(sessionId)}`);
}

export async function getArtifacts(sessionId: string) {
  return request<{artifacts: Artifact[]}>(`/api/artifacts?session=${encodeURIComponent(sessionId)}`);
}

export async function readFilmSnapshot(sessionId: string, root = '', signal?: AbortSignal) {
  const query = new URLSearchParams({session: sessionId, root});
  return request<FilmSnapshot>(`/api/film?${query}`, {signal, cache: 'no-store'});
}

export async function readFilmProgress(sessionId: string, root = '', signal?: AbortSignal) {
  const query = new URLSearchParams({session: sessionId, root});
  return request<FilmProgress>(`/api/progress?${query}`, {signal, cache: 'no-store'});
}

/** Whether an agent is running, which is what makes a written render status live or stale. */
export async function readHealth() {
  return request<{ok: boolean; agentRunning: boolean; activeSessionId: string}>('/api/health');
}

export async function readAcceptance(sessionId: string, root = '') {
  return request<RenderAcceptance>(`/api/acceptance?session=${encodeURIComponent(sessionId)}&root=${encodeURIComponent(root)}`);
}

export async function updateAcceptance(payload: AcceptanceUpdateRequest) {
  return request<RenderAcceptance>('/api/acceptance', {method: 'PUT', body: JSON.stringify(payload)});
}

export async function uploadWorkspaceFile(sessionId: string, file: File) {
  const url = `/api/uploads?session=${encodeURIComponent(sessionId)}&name=${encodeURIComponent(file.name)}`;
  const response = await fetch(url, {
    method: 'POST',
    headers: {'Content-Type': file.type || 'application/octet-stream'},
    body: file,
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `Upload failed with HTTP ${response.status}`);
  return payload as {file: WorkspaceUpload};
}

export async function getJsonArtifact(artifact: Artifact): Promise<JsonValue> {
  const separator = artifact.url.includes('?') ? '&' : '?';
  const response = await fetch(`${artifact.url}${separator}updated=${encodeURIComponent(artifact.updatedAt)}`, {
    cache: 'no-store',
    headers: {Accept: 'application/json'},
  });
  const payload = await response.json();
  if (!response.ok) {
    const message = payload && typeof payload === 'object' && 'error' in payload ? String(payload.error) : `Request failed with HTTP ${response.status}`;
    throw new Error(message);
  }
  return payload as JsonValue;
}

/** A file the artifact list does not surface (render trails), read as text. */
export async function getTextArtifact(sessionId: string, relativePath: string): Promise<string> {
  const query = new URLSearchParams({session: sessionId, path: relativePath, updated: String(Date.now())});
  const response = await fetch(`/api/artifact?${query}`, {cache: 'no-store', headers: {Accept: 'text/plain'}});
  if (!response.ok) throw new Error(`Reading ${relativePath} failed with HTTP ${response.status}`);
  return response.text();
}

/** What one shot's frames describe and show, and the prompt they were drawn from. */
export async function readShotPlan(sessionId: string, root: string, slot: string) {
  const query = new URLSearchParams({session: sessionId, root, slot});
  return request<ShotPlan>(`/api/shot-plan?${query}`);
}

export async function updateShotPlan(payload: ShotPlanUpdateRequest) {
  return request<ShotPlan>('/api/shot-plan', {method: 'PUT', body: JSON.stringify(payload)});
}

/** Take a shot out of the film: its camera stops listing it and its files are set aside. */
export async function removeShot(payload: {sessionId: string; root: string; slot: string}) {
  return request<RenderAcceptance>('/api/shot-plan', {method: 'DELETE', body: JSON.stringify(payload)});
}

/** Every shot's plan at once, for a view that shows all of them. */
export async function readShotPlans(sessionId: string, root: string) {
  const query = new URLSearchParams({session: sessionId, root});
  return request<{root: string; plans: ShotPlan[]}>(`/api/shot-plans?${query}`);
}

/** The shots taken out of the film, with what is kept for each. */
export async function readRemovedShots(sessionId: string, root: string) {
  const query = new URLSearchParams({session: sessionId, root});
  return request<{root: string; removed: RemovedShot[]}>(`/api/removed-shots?${query}`);
}

/** Add a shot after another one, with its own frames when a coverage review suggested it. */
export async function createShot(payload: {
  sessionId: string;
  root: string;
  after: string;
  brief: string;
  /** The dialogue the new shot speaks; empty when it speaks none. */
  audioDesc?: string;
  ffDesc?: string;
  lfDesc?: string;
  ffVis?: number[];
  lfVis?: number[];
}) {
  return request<RenderAcceptance & {created: {slot: string; copiedFrom: string; copiedPlan: boolean}}>('/api/shot-plan', {method: 'POST', body: JSON.stringify(payload)});
}

/** The script-coverage review of a root, with whether it still describes the timeline. */
export async function readContinuity(sessionId: string, root: string) {
  return request<ContinuityPayload>(`/api/continuity?session=${encodeURIComponent(sessionId)}&root=${encodeURIComponent(root)}`);
}

/** Move a shot one place earlier or later in the film, or to a point named by a shot. */
export async function moveShot(payload: {sessionId: string; root: string; slot: string} & ({direction: 'earlier' | 'later'} | {after: string})) {
  return request<RenderAcceptance & {moved: {slot: string; after: string; camera: number; position: number}}>('/api/shot-move', {method: 'POST', body: JSON.stringify(payload)});
}

/** Put a removed shot back into the film. */
export async function restoreShot(payload: {sessionId: string; root: string; slot: string}) {
  return request<RenderAcceptance>('/api/shot-plan', {method: 'PATCH', body: JSON.stringify(payload)});
}

export async function startAgent(options: {sessionId?: string; newSession?: boolean; projectName?: string; style?: string; userRequirement?: string}) {
  return request<{ok: boolean}>('/api/agent/start', {method: 'POST', body: JSON.stringify(options)});
}

export async function sendMessage(text: string, sessionId: string) {
  return request<{ok: boolean}>('/api/messages', {method: 'POST', body: JSON.stringify({text, sessionId})});
}

export async function stopAgent() {
  return request<{ok: boolean}>('/api/agent/stop', {method: 'POST', body: '{}'});
}

export function subscribeToEvents(onEvent: (event: AgentEvent) => void, onConnection: (connected: boolean) => void) {
  const source = new EventSource('/api/events');
  source.onopen = () => onConnection(true);
  source.onerror = () => onConnection(false);
  source.onmessage = (message) => {
    try {
      onEvent(JSON.parse(message.data) as AgentEvent);
    } catch {
      onEvent({type: 'error', message: 'Received an invalid event from the local bridge'});
    }
  };
  return () => source.close();
}

async function request<T>(url: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(url, {
    ...init,
    headers: {'Content-Type': 'application/json', ...(init.headers || {})},
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `Request failed with HTTP ${response.status}`);
  return payload as T;
}
