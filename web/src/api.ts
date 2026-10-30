import type {AcceptanceUpdateRequest, AgentConfig, AgentEvent, Artifact, AssemblyCorrectionReport, AssetDeletionPlan, AssetDeletionRequest, AssetDeletionResponse, AssetsDeletionPlan, AssetsDeletionRequest, AssetsDeletionResponse, BrowserAgentMetricsPayload, ContinuityPayload, JsonValue, Message, PortraitMutationRequest, PortraitMutationResponse, PortraitOptions, ProjectMetadata, ProjectUpdateRequest, ProjectUpdateResponse, RemovedShot, RenderAcceptance, SessionSummary, SetReferenceAcceptanceRequest, SetReferenceInventory, SetReferenceSelectionRequest, SetReferenceUpdateRequest, ShotPlan, ShotPlanUpdateRequest, ShotRestartPlan, WorkspaceUpload} from './types';
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

export type HistoryResponse = {messages: Message[]; epoch?: string; revision?: number};

export async function getHistory(sessionId: string) {
  return request<HistoryResponse>(`/api/history?session=${encodeURIComponent(sessionId)}`, {cache: 'no-store'});
}

export async function postAgentMetrics(payload: BrowserAgentMetricsPayload) {
  const metrics: BrowserAgentMetricsPayload['metrics'] = {visible: payload.metrics.visible === 1 ? 1 : 0};
  for (const field of ['firstVisibleTextMs', 'firstReceiptToCommitMs', 'completionMs'] as const) {
    const value = payload.metrics[field];
    if (typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 86_400_000) metrics[field] = value;
  }
  for (const field of ['deltaEvents', 'snapshotRepairs'] as const) {
    const value = payload.metrics[field];
    if (typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 10_000_000) metrics[field] = Math.floor(value);
  }
  return request<{ok: boolean}>('/api/agent-metrics', {
    method: 'POST',
    body: JSON.stringify({
      sessionId: payload.sessionId,
      messageId: payload.messageId,
      ...(payload.turnId ? {turnId: payload.turnId} : {}),
      metrics,
    }),
  });
}


export async function getArtifacts(sessionId: string) {
  return request<{artifacts: Artifact[]}>(`/api/artifacts?session=${encodeURIComponent(sessionId)}`);
}

export async function getAssetDeletionPlan(sessionId: string, path: string) {
  const query = new URLSearchParams({session: sessionId, path});
  return request<AssetDeletionPlan>(`/api/asset-deletion?${query}`);
}

export async function deleteAsset(payload: AssetDeletionRequest) {
  return request<AssetDeletionResponse>('/api/artifacts', {method: 'DELETE', body: JSON.stringify(payload)});
}

export async function getAssetsDeletionPlan(sessionId: string, paths: string[]) {
  return request<AssetsDeletionPlan>('/api/assets/deletion-plan', {method: 'POST', body: JSON.stringify({sessionId, paths})});
}

export async function deleteAssets(payload: AssetsDeletionRequest) {
  return request<AssetsDeletionResponse>('/api/assets', {method: 'DELETE', body: JSON.stringify(payload)});
}

export async function getPortraitOptions(sessionId: string, sourcePath: string) {
  return request<PortraitOptions>(`/api/portrait-options?session=${encodeURIComponent(sessionId)}&path=${encodeURIComponent(sourcePath)}`);
}

export async function selectPortraitSet(payload: PortraitMutationRequest) {
  return request<PortraitMutationResponse>('/api/portrait-selection', {method: 'POST', body: JSON.stringify(payload)});
}

export async function approvePortraitSet(payload: PortraitMutationRequest) {
  return request<PortraitMutationResponse>('/api/portrait-approval', {method: 'POST', body: JSON.stringify(payload)});
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

export async function readSetReferences(sessionId: string, root = '') {
  return request<SetReferenceInventory>(`/api/set-references?session=${encodeURIComponent(sessionId)}&root=${encodeURIComponent(root)}`);
}

export async function updateSetReference(payload: SetReferenceUpdateRequest) {
  return request<SetReferenceInventory>('/api/set-references', {method: 'PUT', body: JSON.stringify(payload)});
}

export async function selectSetReference(payload: SetReferenceSelectionRequest) {
  return request<SetReferenceInventory>('/api/set-references/selection', {method: 'PUT', body: JSON.stringify(payload)});
}

export async function updateSetReferenceAcceptance(payload: SetReferenceAcceptanceRequest) {
  return request<SetReferenceInventory>('/api/set-references/acceptance', {method: 'PUT', body: JSON.stringify(payload)});
}

export async function applySetReferenceImage(payload: {
  sessionId: string; root: string; revision: string; id?: string; name?: string;
  sourcePath: string; pngBase64: string; crop?: {x: number; y: number; width: number; height: number};
}) {
  return request<SetReferenceInventory>('/api/set-reference-image', {method: 'POST', body: JSON.stringify(payload)});
}
export type AssembleRequest = {sessionId: string; root: string; revision?: string};
export type AssembleResponse = {path: string; reused: boolean; shotCount: number; correction: AssemblyCorrectionReport};

export async function assembleFilm(payload: AssembleRequest) {
  return request<AssembleResponse>('/api/assemble', {method: 'POST', body: JSON.stringify(payload)});
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

/** Prepare a reviewed predecessor video's actual ending locally, without generation. */
export async function prepareRenderedEnding(payload: {
  sessionId: string; root: string; slot: string; previousSlot: string; previousClipSha256: string; revision: string;
}) {
  return request<ShotPlan>('/api/rendered-ending', {method: 'POST', body: JSON.stringify(payload)});
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

/** Clear a shot's generated outputs and approvals while retaining its editable plan. */
export async function restartShot(payload: {sessionId: string; root: string; slot: string; revision: string}) {
  return request<{ok: true; restarted: string; root: string}>('/api/shot-restart', {method: 'POST', body: JSON.stringify(payload)});
}

/** Read the exact impact and revision for a shot restart confirmation. */
export async function getShotRestartPlan(payload: {sessionId: string; root: string; slot: string}) {
  const query = new URLSearchParams({session: payload.sessionId, root: payload.root, slot: payload.slot});
  return request<ShotRestartPlan>(`/api/shot-restart?${query}`);
}

export async function startAgent(options: {sessionId?: string; newSession?: boolean; projectName?: string; style?: string; userRequirement?: string}) {
  return request<{ok: boolean}>('/api/agent/start', {method: 'POST', body: JSON.stringify(options)});
}

export async function sendMessage(payload: {text: string; sessionId: string; messageId: string; displayText: string}) {
  return request<{ok: boolean; messageId: string; queued: boolean}>('/api/messages', {method: 'POST', body: JSON.stringify(payload)});
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
  let payload;
  try {
    payload = await response.json();
  } catch {
    throw new Error(`HTTP ${response.status} from ${url}: the web server returned an empty or invalid JSON response. If the server was recently updated, restart the web backend when idle and reload the page. Check the saved state before retrying; this response does not confirm whether the operation was saved.`);
  }
  if (!response.ok) throw new Error(payload.error || `Request failed with HTTP ${response.status}`);
  return payload as T;
}
