import {createReadStream, existsSync} from 'node:fs';
import {randomUUID} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import {createServer} from 'node:http';
import path from 'node:path';
import {spawn} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {performance} from 'node:perf_hooks';
import {readAgentConfig, saveAgentConfig} from './config-store.mjs';
import {closeMediaCache, serveArtifact, serveThumbnail} from './media-serving.mjs';
import {createAgentMessageQueue} from './agent-message-queue.mjs';
import {applySetReferenceImage, approvePortraitCharacter, artifactContentType, assembleFilm, deleteAsset, deleteAssets, deleteSession, getAssetDeletionPlan, getAssetsDeletionPlan, listSessionArtifacts, moveShot, readContinuityReview, readFilmProgress, readFilmSnapshot, readPortraitOptions, readProjectMetadata, readRemovedShots, readRenderAcceptance, readSessionHistory, readSessionState, readShotPlan, readShotPlans, removeShot, rerenderPrompt, restoreShot, selectPortraitCharacter, selectSetReference, storeWorkspaceUpload, updateProjectMetadata, updateRenderAcceptance, updateShotPlan, readSetReferences, updateSetReference, updateSetReferenceAcceptance} from './server-lib.mjs';
import {getShotRestartPlan, restartShot} from './server-lib.mjs';
import {BoundedPerformanceLog, createBridgeStreamState, isSseBackpressureExceeded, safeRuntimePerformanceRecord, validateAgentMetricsPayload} from './bridge-streaming.mjs';

const webRoot = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(webRoot, '..');
const isDev = process.argv.includes('--dev');
const host = process.env.VIMAX_WEB_HOST || '127.0.0.1';
const port = Number(process.env.VIMAX_WEB_PORT || 4173);
const configuredUploadLimit = Number(process.env.VIMAX_WEB_UPLOAD_MAX_BYTES || 100 * 1024 * 1024);
const uploadMaxBytes = Number.isFinite(configuredUploadLimit) && configuredUploadLimit > 0
  ? configuredUploadLimit
  : 100 * 1024 * 1024;
const subscribers = new Set();
let agentProcess = null;
let activeSessionId = '';
const agentQueue = createAgentMessageQueue();
const bridgeStreamState = createBridgeStreamState();
const performanceLog = new BoundedPerformanceLog(repoRoot, () => console.warn('Unable to persist bounded bridge performance telemetry'));
const turnContexts = new Map();
const messageSessions = new Map();
const messageTimings = new Map();
const recordedMetricKeys = new Set();
let vite = null;

const server = createServer(async (request, response) => {
  const url = new URL(request.url || '/', `http://${request.headers.host || `${host}:${port}`}`);
  try {
    if (url.pathname === '/api/events' && request.method === 'GET') {
      return openEventStream(request, response);
    }
    if (url.pathname === '/api/sessions' && request.method === 'GET') {
      return sendJson(response, 200, await readSessionState(repoRoot));
    }
    if (url.pathname === '/api/session' && request.method === 'GET') {
      const session = await readProjectMetadata(repoRoot, url.searchParams.get('session') || '');
      if (!session) return sendJson(response, 404, {error: 'Project not found'});
      return sendJson(response, 200, session);
    }
    if (url.pathname === '/api/session' && request.method === 'PUT') {
      const body = await readJsonBody(request);
      if (!body || typeof body !== 'object' || Array.isArray(body)) {
        return sendJson(response, 400, {error: 'A JSON object body is required'});
      }
      const project = await updateProjectMetadata(repoRoot, {
        sessionId: typeof body.sessionId === 'string' ? body.sessionId : '',
        projectName: body.projectName,
        idea: body.idea,
        userRequirement: body.userRequirement,
        style: body.style,
        invalidate: body.invalidate,
      });
      if (!project) return sendJson(response, 404, {error: 'Project not found'});
      const onActiveSession = project.session.sessionId === activeSessionId;
      // `invalidated` doubles as the candidate list on the unconfirmed call, so
      // the request flag decides whether artifacts were really deleted.
      let regenerationStarted = false;
      if (body.invalidate === true && project.invalidated.length) {
        // Deleting the artifacts is only half of a confirmed style change: the
        // render is what rebuilds them, and it skips whatever still exists. So
        // the agent is respawned on this project (which also makes it active,
        // since the user just confirmed the rebuild) and told to run it, rather
        // than leaving the project silently empty until someone asks again.
        await startAgent({sessionId: project.session.sessionId});
        if (agentProcess?.stdin.writable) {
          agentProcess.stdin.write(`${rerenderPrompt(project.session.style)}\n`);
          regenerationStarted = true;
        }
      } else if (project.changed.length && onActiveSession) {
        // The agent caches the session record, so a metadata edit that lands on
        // the running session must respawn it (mirrors PUT /api/config).
        stopAgent('config');
      }
      broadcast({type: 'sessions_changed', ...(await readSessionState(repoRoot))});
      return sendJson(response, 200, {...project, regenerationStarted});
    }
    if (url.pathname === '/api/config' && request.method === 'GET') {
      return sendJson(response, 200, await readAgentConfig(repoRoot));
    }
    if (url.pathname === '/api/config' && request.method === 'PUT') {
      const config = await saveAgentConfig(repoRoot, await readJsonBody(request));
      stopAgent('config');
      return sendJson(response, 200, config);
    }
    if (url.pathname === '/api/sessions' && request.method === 'DELETE') {
      const sessionId = url.searchParams.get('session') || '';
      const current = await readSessionState(repoRoot);
      if (!current.sessions.some((session) => session.sessionId === sessionId)) {
        return sendJson(response, 404, {error: 'Project not found'});
      }
      if (sessionId === activeSessionId) stopAgent('delete');
      const state = await deleteSession(repoRoot, sessionId);
      activeSessionId = state.activeSessionId;
      broadcast({type: 'sessions_changed', ...state});
      return sendJson(response, 200, state);
    }
    if (url.pathname === '/api/history' && request.method === 'GET') {
      const sessionId = url.searchParams.get('session') || '';
      const messages = await readSessionHistory(repoRoot, sessionId);
      return sendJson(response, 200, {
        messages: reconcileHistoryMessageIds(messages, sessionId),
        epoch: bridgeStreamState.epoch,
        revision: bridgeStreamState.revision,
        activeSessionId,
      });
    }
    if (url.pathname === '/api/artifacts' && request.method === 'GET') {
      return sendJson(response, 200, {artifacts: await listSessionArtifacts(repoRoot, url.searchParams.get('session') || '')});
    }
    if (url.pathname === '/api/asset-deletion' && request.method === 'GET') {
      const sessionId = url.searchParams.get('session') || '';
      const plan = await getAssetDeletionPlan(repoRoot, sessionId, url.searchParams.get('path') || '');
      return plan ? sendJson(response, 200, plan) : sendJson(response, 404, {error: 'Project not found'});
    }
    if (url.pathname === '/api/assets/deletion-plan' && request.method === 'POST') {
      let body;
      try { body = await readJsonBody(request); }
      catch (error) { return sendJson(response, 400, {error: error.message}); }
      const plan = await getAssetsDeletionPlan(repoRoot, body?.sessionId, body?.paths);
      return plan ? sendJson(response, 200, plan) : sendJson(response, 404, {error: 'Project not found'});
    }
    if (url.pathname === '/api/assets' && request.method === 'DELETE') {
      let body;
      try { body = await readJsonBody(request); }
      catch (error) { return sendJson(response, 400, {error: error.message}); }
      const result = await deleteAssets(repoRoot, body, () => agentQueue.hasWorkFor(body?.sessionId));
      return result ? sendJson(response, 200, result) : sendJson(response, 404, {error: 'Project not found'});
    }
    if (url.pathname === '/api/artifacts' && request.method === 'DELETE') {
      let body;
      try { body = await readJsonBody(request); }
      catch (error) { return sendJson(response, 400, {error: error.message}); }
      if (!body || typeof body !== 'object' || Array.isArray(body)
        || typeof body.sessionId !== 'string' || typeof body.path !== 'string' || typeof body.revision !== 'string') {
        return sendJson(response, 400, {error: 'sessionId, path, and revision are required'});
      }
      const result = await deleteAsset(repoRoot, body, () => agentQueue.hasWorkFor(body.sessionId));
      return result ? sendJson(response, 200, result) : sendJson(response, 404, {error: 'Project not found'});
    }
    if (url.pathname === '/api/portrait-options' && request.method === 'GET') {
      const options = await readPortraitOptions(
        repoRoot,
        url.searchParams.get('session') || '',
        url.searchParams.get('path') || '',
      );
      return options ? sendJson(response, 200, options) : sendJson(response, 404, {error: 'Project not found'});
    }
    if (url.pathname === '/api/portrait-selection' && request.method === 'POST') {
      let body;
      try { body = await readJsonBody(request); }
      catch (error) { return sendJson(response, 400, {error: error.message}); }
      if (!body || typeof body !== 'object' || Array.isArray(body)) {
        return sendJson(response, 400, {error: 'A JSON object body is required'});
      }
      const result = await selectPortraitCharacter(repoRoot, body);
      return result ? sendJson(response, 200, result) : sendJson(response, 404, {error: 'Project not found'});
    }
    if (url.pathname === '/api/portrait-approval' && request.method === 'POST') {
      let body;
      try { body = await readJsonBody(request); }
      catch (error) { return sendJson(response, 400, {error: error.message}); }
      if (!body || typeof body !== 'object' || Array.isArray(body)) {
        return sendJson(response, 400, {error: 'A JSON object body is required'});
      }
      const result = await approvePortraitCharacter(repoRoot, body);
      return result ? sendJson(response, 200, result) : sendJson(response, 404, {error: 'Project not found'});
    }
    if (url.pathname === '/api/set-references' && request.method === 'GET') {
      const payload = await readSetReferences(repoRoot, url.searchParams.get('session') || '', url.searchParams.get('root') || '');
      return payload ? sendJson(response, 200, payload) : sendJson(response, 404, {error: 'Project not found'});
    }
    if (url.pathname === '/api/set-references' && request.method === 'PUT') {
      let body;
      try { body = await readJsonBody(request); }
      catch (error) { return sendJson(response, 400, {error: error.message}); }
      if (!body || typeof body !== 'object' || Array.isArray(body)) return sendJson(response, 400, {error: 'A JSON object body is required'});
      const payload = await updateSetReference(repoRoot, body);
      return payload ? sendJson(response, 200, payload) : sendJson(response, 404, {error: 'Project not found'});
    }

    if (url.pathname === '/api/set-references/selection' && request.method === 'PUT') {
      let body;
      try { body = await readJsonBody(request); }
      catch (error) { return sendJson(response, 400, {error: error.message}); }
      if (!body || typeof body !== 'object' || Array.isArray(body)) return sendJson(response, 400, {error: 'A JSON object body is required'});
      const payload = await selectSetReference(repoRoot, body, () => agentQueue.hasWorkFor(body.sessionId));
      return payload ? sendJson(response, 200, payload) : sendJson(response, 404, {error: 'Project not found'});
    }
    if (url.pathname === '/api/set-reference-image' && request.method === 'POST') {
      let body;
      try { body = await readSmallJsonBody(request, 48 * 1024 * 1024); }
      catch (error) { return sendJson(response, error.statusCode || 400, {error: error.message}); }
      if (!body || typeof body !== 'object' || Array.isArray(body)) return sendJson(response, 400, {error: 'A JSON object body is required'});
      const payload = await applySetReferenceImage(repoRoot, body, () => agentQueue.hasWorkFor(body.sessionId));
      return payload ? sendJson(response, 200, payload) : sendJson(response, 404, {error: 'Project not found'});
    }
    if (url.pathname === '/api/set-references/acceptance' && request.method === 'PUT') {
      let body;
      try { body = await readJsonBody(request); }
      catch (error) { return sendJson(response, 400, {error: error.message}); }
      if (!body || typeof body !== 'object' || Array.isArray(body)) return sendJson(response, 400, {error: 'A JSON object body is required'});
      const payload = await updateSetReferenceAcceptance(repoRoot, body);
      return payload ? sendJson(response, 200, payload) : sendJson(response, 404, {error: 'Project not found'});
    }
    if (url.pathname === '/api/film' && request.method === 'GET') {
      const payload = await readFilmSnapshot(repoRoot, url.searchParams.get('session') || '', url.searchParams.get('root') || '');
      return payload ? sendJson(response, 200, payload) : sendJson(response, 404, {error: 'Project not found'});
    }
    if (url.pathname === '/api/progress' && request.method === 'GET') {
      const payload = await readFilmProgress(repoRoot, url.searchParams.get('session') || '', url.searchParams.get('root') || '');
      return payload
        ? sendJson(response, 200, {...payload, agentRunning: Boolean(agentProcess), activeSessionId})
        : sendJson(response, 404, {error: 'Project not found'});
    }
    if (url.pathname === '/api/continuity' && request.method === 'GET') {
      try {
        const payload = await readContinuityReview(repoRoot, url.searchParams.get('session') || '', url.searchParams.get('root') || '');
        return payload ? sendJson(response, 200, payload) : sendJson(response, 404, {error: 'Project not found'});
      } catch (error) {
        return sendJson(response, error.statusCode || 400, {error: error.message});
      }
    }

    if (url.pathname === '/api/shot-plans' && request.method === 'GET') {
      try {
        const payload = await readShotPlans(repoRoot, url.searchParams.get('session') || '', url.searchParams.get('root') || '');
        return payload ? sendJson(response, 200, payload) : sendJson(response, 404, {error: 'Project not found'});
      } catch (error) {
        return sendJson(response, error.statusCode || 400, {error: error.message});
      }
    }
    if (url.pathname === '/api/removed-shots' && request.method === 'GET') {
      const payload = await readRemovedShots(repoRoot, url.searchParams.get('session') || '', url.searchParams.get('root') || '');
      return payload ? sendJson(response, 200, payload) : sendJson(response, 404, {error: 'Project not found'});
    }
    if (url.pathname === '/api/shot-plan' && request.method === 'POST') {
      const body = await readJsonBody(request);
      try {
        const payload = await createShot(repoRoot, body);
        return payload ? sendJson(response, 201, payload) : sendJson(response, 404, {error: 'Project not found'});
      } catch (error) {
        return sendJson(response, error.statusCode || 400, {error: error.message});
      }
    }
    if (url.pathname === '/api/shot-plan' && request.method === 'PATCH') {
      const body = await readJsonBody(request);
      try {
        const payload = await restoreShot(repoRoot, body);
        return payload ? sendJson(response, 200, payload) : sendJson(response, 404, {error: 'Project not found'});
      } catch (error) {
        return sendJson(response, error.statusCode || 400, {error: error.message});
      }
    }
    if (url.pathname === '/api/shot-move' && request.method === 'POST') {
      try {
        const payload = await moveShot(repoRoot, await readJsonBody(request));
        return payload ? sendJson(response, 200, payload) : sendJson(response, 404, {error: 'Project not found'});
      } catch (error) {
        return sendJson(response, error.statusCode || 400, {error: error.message});
      }
    }

    if (url.pathname === '/api/shot-restart' && request.method === 'GET') {
      const payload = await getShotRestartPlan(repoRoot, {
        sessionId: url.searchParams.get('session') || '',
        root: url.searchParams.get('root') || '',
        slot: url.searchParams.get('slot') || '',
      });
      return payload ? sendJson(response, 200, payload) : sendJson(response, 404, {error: 'Project not found'});
    }
    if (url.pathname === '/api/shot-restart' && request.method === 'POST') {
      const body = await readJsonBody(request);
      const payload = await restartShot(repoRoot, body, () => agentQueue.hasWorkFor(body?.sessionId));
      return payload ? sendJson(response, 200, payload) : sendJson(response, 404, {error: 'Project not found'});
    }

    if (url.pathname === '/api/shot-plan' && request.method === 'GET') {
      try {
        const payload = await readShotPlan(repoRoot, url.searchParams.get('session') || '', url.searchParams.get('root') || '', url.searchParams.get('slot') || '');
        return payload ? sendJson(response, 200, payload) : sendJson(response, 404, {error: 'Project not found'});
      } catch (error) {
        return sendJson(response, error.statusCode || 400, {error: error.message});
      }
    }
    if (url.pathname === '/api/shot-plan' && request.method === 'DELETE') {
      const body = await readJsonBody(request);
      try {
        const payload = await removeShot(repoRoot, body);
        return payload ? sendJson(response, 200, payload) : sendJson(response, 404, {error: 'Project not found'});
      } catch (error) {
        return sendJson(response, error.statusCode || 400, {error: error.message});
      }
    }
    if (url.pathname === '/api/shot-plan' && request.method === 'PUT') {
      const body = await readJsonBody(request);
      try {
        const payload = await updateShotPlan(repoRoot, body);
        return payload ? sendJson(response, 200, payload) : sendJson(response, 404, {error: 'Project not found'});
      } catch (error) {
        return sendJson(response, error.statusCode || 400, {error: error.message});
      }
    }
    if (url.pathname === '/api/assemble' && request.method === 'POST') {
      let body;
      try { body = await readJsonBody(request); }
      catch (error) { return sendJson(response, 400, {error: error.message}); }
      if (!body || typeof body !== 'object' || Array.isArray(body)) {
        return sendJson(response, 400, {error: 'A JSON object body is required'});
      }
      try {
        const result = await assembleFilm(repoRoot, body);
        return result ? sendJson(response, 200, result) : sendJson(response, 404, {error: 'Project not found'});
      } catch (error) {
        return sendJson(response, error.statusCode || 500, {error: error.message});
      }
    }
    if (url.pathname === '/api/acceptance' && request.method === 'GET') {
      const acceptance = await readRenderAcceptance(repoRoot, url.searchParams.get('session') || '', url.searchParams.get('root') || '');
      if (!acceptance) return sendJson(response, 404, {error: 'Project not found'});
      return sendJson(response, 200, acceptance);
    }
    if (url.pathname === '/api/acceptance' && request.method === 'PUT') {
      const body = await readJsonBody(request);
      if (!body || typeof body !== 'object' || Array.isArray(body)) {
        return sendJson(response, 400, {error: 'A JSON object body is required'});
      }
      const acceptance = await updateRenderAcceptance(repoRoot, {
        sessionId: typeof body.sessionId === 'string' ? body.sessionId : '',
        root: typeof body.root === 'string' ? body.root : '',
        stage: body.stage,
        shot: body.shot,
        accepted: body.accepted,
        reason: body.reason,
      });
      if (!acceptance) return sendJson(response, 404, {error: 'Project not found'});
      return sendJson(response, 200, acceptance);
    }
    if (url.pathname === '/api/artifact' && (request.method === 'GET' || request.method === 'HEAD')) {
      return await serveArtifact(request, response, repoRoot, url.searchParams.get('session') || '', url.searchParams.get('path') || '');
    }
    if (url.pathname === '/api/thumbnail' && (request.method === 'GET' || request.method === 'HEAD')) {
      return await serveThumbnail(request, response, repoRoot, url.searchParams.get('session') || '', url.searchParams.get('path') || '', url.searchParams.get('width'));
    }
    if (url.pathname === '/api/uploads' && request.method === 'POST') {
      const sessionId = url.searchParams.get('session') || '';
      const fileName = url.searchParams.get('name') || '';
      const current = await readSessionState(repoRoot);
      if (!current.sessions.some((session) => session.sessionId === sessionId)) {
        return sendJson(response, 404, {error: 'Project not found'});
      }
      const declaredSize = Number(request.headers['content-length'] || 0);
      if (declaredSize > uploadMaxBytes) {
        return sendJson(response, 413, {error: `File exceeds the ${formatByteLimit(uploadMaxBytes)} upload limit`});
      }
      const data = await readBinaryBody(request, uploadMaxBytes);
      const file = await storeWorkspaceUpload(repoRoot, sessionId, fileName, data);
      return sendJson(response, 201, {file});
    }
    if (url.pathname === '/api/agent/start' && request.method === 'POST') {
      const body = await readJsonBody(request);
      const sessionId = typeof body.sessionId === 'string' ? body.sessionId : '';
      const projectName = typeof body.projectName === 'string' ? body.projectName.trim() : '';
      const style = typeof body.style === 'string' ? body.style.trim() : '';
      const userRequirement = typeof body.userRequirement === 'string' ? body.userRequirement.trim() : '';
      if (projectName.length > 64) {
        return sendJson(response, 400, {error: 'Project name must be 64 characters or fewer'});
      }
      await startAgent({newSession: body.newSession === true, sessionId, projectName, style, userRequirement});
      return sendJson(response, 200, {ok: true});
    }
    if (url.pathname === '/api/agent-metrics' && request.method === 'POST') {
      let body;
      try {
        body = await readSmallJsonBody(request, 8 * 1024);
      } catch (error) {
        return sendJson(response, error.statusCode || 400, {error: error.message || 'Invalid JSON body'});
      }
      const validated = validateAgentMetricsPayload(body);
      if (!validated.ok) return sendJson(response, 400, {error: validated.error});
      const metric = validated.value;
      const knownSessionId = lookupMessageSession(metric.messageId);
      const turnContext = metric.turnId ? lookupTurnContext(metric.turnId) : null;
      if (knownSessionId !== metric.sessionId
        || metric.turnId && (!turnContext || turnContext.sessionId !== metric.sessionId || turnContext.messageId !== metric.messageId)) {
        return sendJson(response, 409, {error: 'Unknown message correlation'});
      }
      const metricKey = JSON.stringify([metric.sessionId, metric.messageId]);
      if (recordedMetricKeys.has(metricKey)) return sendJson(response, 409, {error: 'Metrics already recorded for this turn'});
      recordedMetricKeys.add(metricKey);
      if (recordedMetricKeys.size > 1_000) recordedMetricKeys.delete(recordedMetricKeys.values().next().value);
      void performanceLog.append({
        type: 'browser_metrics',
        epoch: bridgeStreamState.epoch,
        sessionId: metric.sessionId,
        messageId: metric.messageId,
        ...(metric.turnId ? {turn_id: metric.turnId} : {}),
        metrics: metric.metrics,
      });
      return sendJson(response, 202, {ok: true});
    }

    if (url.pathname === '/api/messages' && request.method === 'POST') {
      const body = await readJsonBody(request);
      if (!body || typeof body !== 'object' || Array.isArray(body)) {
        return sendJson(response, 400, {error: 'A JSON object body is required'});
      }
      const text = typeof body.text === 'string' ? body.text.trim() : '';
      const sessionId = typeof body.sessionId === 'string' ? body.sessionId.trim() : '';
      if (!text) return sendJson(response, 400, {error: 'Message text is required'});
      if (!sessionId) return sendJson(response, 400, {error: 'Session id is required'});
      if (sessionId !== activeSessionId) return sendJson(response, 409, {error: 'The requested project is no longer active'});
      const child = agentProcess;
      if (!child?.stdin.writable) return sendJson(response, 409, {error: 'Agent is not running'});
      const messageId = typeof body.messageId === 'string' && body.messageId.trim() ? body.messageId : randomUUID();
      if (messageId.length > 256 || !/^[A-Za-z0-9_.:-]+$/.test(messageId)) {
        return sendJson(response, 400, {error: 'Message id must be a short text identifier'});
      }
      const displayText = typeof body.displayText === 'string' ? body.displayText : text;
      const queuedMessage = {id: messageId, text: displayText, input: text};
      const enqueueResult = agentQueue.enqueue(sessionId, queuedMessage);
      if (!enqueueResult.accepted) return sendJson(response, 409, {error: 'The requested project is no longer active'});
      rememberMessageSession(messageId, sessionId);
      rememberMessageTiming(messageId, sessionId);
      if (enqueueResult.started && !dispatchAgentMessage(child)) {
        return sendJson(response, 503, {error: 'Unable to send message to agent'});
      }
      broadcastQueueSnapshot();
      return sendJson(response, 202, {ok: true, messageId, queued: enqueueResult.queued});
    }
    if (url.pathname === '/api/agent/stop' && request.method === 'POST') {
      stopAgent('user');
      return sendJson(response, 200, {ok: true});
    }
    if (url.pathname === '/api/health' && request.method === 'GET') {
      return sendJson(response, 200, {ok: true, agentRunning: Boolean(agentProcess), activeSessionId});
    }
    if (url.pathname === '/assets/vimax.png' && request.method === 'GET') {
      response.writeHead(200, {'Content-Type': 'image/png', 'Cache-Control': 'public, max-age=3600'});
      createReadStream(path.join(repoRoot, 'assets', 'vimax.png')).pipe(response);
      return;
    }
    if (url.pathname.startsWith('/api/')) {
      return sendJson(response, 404, {error: `Unknown API endpoint: ${request.method} ${url.pathname}. If the web server was recently updated, restart it when idle and reload the page.`});
    }
    if (vite) {
      vite.middlewares(request, response, () => sendJson(response, 404, {error: 'Not found'}));
      return;
    }
    return serveProductionApp(response, url.pathname);
  } catch (error) {
    const status = Number(error?.statusCode) || 500;
    sendJson(response, status, {error: error instanceof Error ? error.message : String(error)});
  }
});

if (isDev) {
  vite = await (await import('vite')).createServer({
    root: webRoot,
    server: {middlewareMode: true, hmr: {server}},
    appType: 'spa',
  });
}

server.listen(port, host, () => {
  console.log(`ViMax Web: http://${host}:${port}`);
});

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);

async function startAgent({newSession, sessionId, projectName = '', style = '', userRequirement = ''}) {
  if (newSession && sessionId) throw new Error('Choose either a new or existing session');
  stopAgent('switch');
  const {command, args} = agentCommand();
  const sessionArgs = newSession
    ? ['--new-session', ...(projectName ? ['--new-session-name', projectName] : []), ...(style ? ['--new-session-style', style] : [])]
    : sessionId
      ? ['--session', sessionId]
      : [];
  activeSessionId = sessionId;
  agentQueue.reset(sessionId);
  // Inject ViMax agent defaults unless the user already overrode them in the
  // environment. Keeps the agent's request within the model context window:
  // VIMAX_CONTEXT_WINDOW_TOKENS raises the auto-compaction trigger to ~810k
  // (deepseek-v4-flash-0731 has a 1.31M window) and VIMAX_MAX_TOOL_RESULT_CHARS
  // caps the tool-output copy replayed to the LLM.
  const agentEnv = {...process.env};
  const setDefault = (name, value) => { if (!agentEnv[name]) agentEnv[name] = value; };
  setDefault('VIMAX_CONTEXT_WINDOW_TOKENS', '900000');
  setDefault('VIMAX_MAX_TOOL_RESULT_CHARS', '20000');
  const child = spawn(command, [...args, 'main_agent.py', '--jsonl', '--stdin-repl', ...sessionArgs], {
    cwd: repoRoot,
    env: agentEnv,
    stdio: ['pipe', 'pipe', 'pipe'],
  });
  agentProcess = child;
  let childStdoutBuffer = '';
  child.stdout.setEncoding('utf8');
  child.stdout.on('data', (chunk) => {
    if (agentProcess !== child) return;
    childStdoutBuffer += String(chunk);
    const lines = childStdoutBuffer.split(/\r?\n/);
    childStdoutBuffer = lines.pop() || '';
    for (const line of lines) consumeAgentLine(child, line);
  });
  child.stderr.setEncoding('utf8');
  child.stderr.on('data', (chunk) => {
    if (agentProcess !== child) return;
    for (const line of String(chunk).split(/\r?\n/)) {
      if (line.trim()) broadcast({type: 'terminal', stream: 'stderr', line, activeSessionId});
    }
  });
  child.stdin.on('error', (error) => handleAgentFailure(child, error));
  child.on('error', (error) => handleAgentFailure(child, error));
  child.on('exit', (code, signal) => {
    if (agentProcess !== child) return;
    interruptSession(activeSessionId);
    agentProcess = null;
    broadcast(agentQueue.cancel());
    broadcast({
      type: 'bridge_status',
      status: code === 0 || signal === 'SIGTERM' ? 'stopped' : 'error',
      message: signal ? `Agent stopped by ${signal}` : `Agent exited with code ${code ?? 0}`,
      activeSessionId,
    });
  });
  broadcast({type: 'bridge_status', status: 'starting', message: newSession ? 'Creating workspace' : 'Opening workspace', activeSessionId});

  // A brand-new project can carry its user requirement up front: the CLI has no
  // flag for it, so it becomes the agent's first turn (the session record picks
  // it up when narrative planning runs).
  if (newSession && userRequirement) {
    const requirement = {id: randomUUID(), text: userRequirement, input: userRequirement};
    agentQueue.enqueue(activeSessionId, requirement);
    rememberMessageSession(requirement.id, activeSessionId);
    rememberMessageTiming(requirement.id, activeSessionId);
    if (dispatchAgentMessage(child)) broadcastQueueSnapshot();
  } else {
    broadcastQueueSnapshot();
  }

  setTimeout(async () => {
    if (agentProcess !== child) return;
    try {
      const state = await readSessionState(repoRoot);
      if (agentProcess !== child) return;
      const nextSessionId = state.activeSessionId || sessionId || activeSessionId;
      changeAgentSession(nextSessionId);
      broadcast({type: 'sessions_changed', ...state, activeSessionId});
      broadcast({type: 'bridge_status', status: 'ready', message: 'Agent ready', activeSessionId});
    } catch (error) {
      if (agentProcess === child) broadcast({type: 'error', message: `Unable to read session state: ${error.message}`, activeSessionId});
    }
  }, 350);
}

function consumeAgentLine(child, line) {
  if (agentProcess !== child || !line.trim()) return;
  try {
    const parsed = JSON.parse(line);
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) throw new Error('Invalid event');
    const event = parsed;
    if (event.type === 'session') {
      const nextSessionId = event.session?.active_session_id || activeSessionId;
      changeAgentSession(nextSessionId);
    }
    const requestSessionId = agentQueue.activeMessage?.sessionId;
    const queueResult = agentQueue.consumeEvent(event);
    if (event.type === 'turn' && !queueResult.messageId) return;
    if (event.type === 'done' && !queueResult.completed) return;
    let context = event.turn_id ? lookupTurnContext(event.turn_id) : null;
    if (event.type === 'turn' && event.turn_id && queueResult.messageId && !context) {
      const observedAt = performance.now();
      const timing = lookupMessageTiming(queueResult.messageId);
      if (timing?.dispatchedAt !== null && timing?.dispatchedAt !== undefined) {
        timing.queueWaitMs = Math.max(0, timing.dispatchedAt - timing.acceptedAt);
        timing.dispatchToTurnMs = Math.max(0, observedAt - timing.dispatchedAt);
      }
      context = {
        messageId: queueResult.messageId,
        sessionId: queueResult.sessionId || requestSessionId || activeSessionId,
        queueWaitMs: timing?.queueWaitMs ?? null,
        dispatchToTurnMs: timing?.dispatchToTurnMs ?? null,
        seenAt: Date.now(),
      };
      rememberTurnContext(event.turn_id, context);
      rememberMessageSession(context.messageId, context.sessionId);
    } else if (event.turn_id && queueResult.messageId && !context) {
      context = {
        messageId: queueResult.messageId,
        sessionId: queueResult.sessionId || requestSessionId || activeSessionId,
        seenAt: Date.now(),
      };
      rememberTurnContext(event.turn_id, context);
      rememberMessageSession(context.messageId, context.sessionId);
    }
    const sessionId = event.type === 'session'
      ? activeSessionId
      : queueResult.sessionId || context?.sessionId || requestSessionId || activeSessionId;
    const messageId = queueResult.messageId || context?.messageId || null;
    const eventFields = {...event};
    delete eventFields.epoch;
    delete eventFields.revision;
    const correlatedEvent = {...eventFields, messageId, activeSessionId: sessionId};
    const observed = bridgeStreamState.observe(correlatedEvent);
    if (!observed.accepted) return;
    broadcast(correlatedEvent);
    if (event.type === 'performance') {
      void performanceLog.append(safeRuntimePerformanceRecord({...correlatedEvent, epoch: bridgeStreamState.epoch}));
    }
    if (event.type === 'done') {
      const summary = completionLogRecord(sessionId, event.turn_id);
      if (summary) void performanceLog.append(summary);
    }
    if (observed.becameOverflow) {
      broadcast(bridgeStreamState.snapshotEvent(sessionId));
      disconnectSubscribers();
    }
    if (event.type === 'session') {
      broadcastQueueSnapshot();
      readSessionState(repoRoot).then((state) => {
        if (agentProcess === child) broadcast({type: 'sessions_changed', ...state, activeSessionId});
      }).catch((error) => {
        if (agentProcess === child) broadcast({type: 'error', message: `Unable to read session state: ${error.message}`, activeSessionId});
      });
    }
    if (queueResult.completed) {
      if (queueResult.started && !dispatchAgentMessage(child)) return;
      broadcastQueueSnapshot();
    }
  } catch {
    broadcast({type: 'terminal', stream: 'stdout', line, activeSessionId});
  }
}

function changeAgentSession(sessionId) {
  if (sessionId === activeSessionId) return;
  interruptSession(activeSessionId);
  const cancelled = agentQueue.cancelPending();
  if (cancelled.cancelledIds?.length) broadcast(cancelled);
  activeSessionId = sessionId;
  broadcast(agentQueue.setSessionId(sessionId));
}

function dispatchAgentMessage(child) {
  if (agentProcess !== child) return false;
  const message = agentQueue.activeMessage;
  if (!message) return false;
  if (!child.stdin.writable) {
    handleAgentFailure(child, new Error('Agent stdin is not writable'));
    return false;
  }
  try {
    markMessageDispatched(message.id);
    child.stdin.write(`${message.input}\n`, (error) => {
      if (error) handleAgentFailure(child, error);
    });
    return true;
  } catch (error) {
    handleAgentFailure(child, error);
    return false;
  }
}

function handleAgentFailure(child, error) {
  if (agentProcess !== child) return;
  interruptSession(activeSessionId);
  agentProcess = null;
  broadcast(agentQueue.cancel());
  try {
    child.kill('SIGTERM');
  } catch {
    // The process may already have exited while its stream reported an error.
  }
  const message = `Agent process error: ${error?.message || String(error)}`;
  broadcast({type: 'error', message, activeSessionId});
  broadcast({type: 'bridge_status', status: 'error', message, activeSessionId});
}

function reconcileHistoryMessageIds(messages, sessionId) {
  return messages.map((message) => {
    if (message.role !== 'user' || !message.id.endsWith('-user')) return message;
    const turnId = message.id.slice(0, -5);
    const context = lookupTurnContext(turnId);
    const messageId = context?.sessionId === sessionId ? context.messageId : null;
    return {...message, id: messageId || message.id, turnId};
  });
}

const MAX_SSE_BUFFER_BYTES = 256 * 1024;

function openEventStream(request, response) {
  const openedAt = performance.now();
  response.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache, no-transform',
    Connection: 'keep-alive',
    'X-Accel-Buffering': 'no',
  });
  subscribers.add(response);
  const running = Boolean(agentProcess);
  writeSseEvent(response, {
    type: 'bridge_status',
    status: running ? 'ready' : 'idle',
    message: running ? 'Agent connected' : 'Agent idle',
    activeSessionId,
  });
  writeSseEvent(response, agentQueue.snapshot());
  writeSseEvent(response, bridgeStreamState.snapshotEvent(activeSessionId));
  const heartbeat = setInterval(() => {
    if (isSseBackpressureExceeded(response.writableLength || 0, 14, MAX_SSE_BUFFER_BYTES)) {
      closeSseSubscriber(response, false);
      return;
    }
    if (!response.write(': keepalive\n\n') && response.writableLength > MAX_SSE_BUFFER_BYTES) {
      closeSseSubscriber(response, false);
    }
  }, 15_000);
  response.on('close', () => {
    clearInterval(heartbeat);
    subscribers.delete(response);
    void performanceLog.append({type: 'sse_connection', epoch: bridgeStreamState.epoch, outcome: response.vimaxCloseReason || 'disconnected', duration_ms: performance.now() - openedAt});
  });
}

function broadcastQueueSnapshot() {
  broadcast(agentQueue.snapshot());
}

function broadcast(event) {
  const enriched = event.type === 'stream_snapshot'
    && event.epoch === bridgeStreamState.epoch && Number.isInteger(event.revision)
    ? event
    : bridgeStreamState.enrich(event, event.activeSessionId ?? activeSessionId, event.messageId ?? null);
  for (const subscriber of [...subscribers]) writeSseEvent(subscriber, enriched);
}

function writeSseEvent(response, event) {
  if (response.destroyed || response.writableEnded) {
    subscribers.delete(response);
    return false;
  }
  const enriched = event.epoch === bridgeStreamState.epoch && Number.isInteger(event.revision)
    ? event
    : bridgeStreamState.enrich(event, event.activeSessionId ?? activeSessionId, event.messageId ?? null);
  const payload = `id: ${enriched.revision}\ndata: ${JSON.stringify(enriched)}\n\n`;
  const payloadBytes = Buffer.byteLength(payload, 'utf8');
  if (isSseBackpressureExceeded(response.writableLength || 0, payloadBytes, MAX_SSE_BUFFER_BYTES)) {
    closeSseSubscriber(response, false, payloadBytes > MAX_SSE_BUFFER_BYTES ? 'event_too_large' : 'slow_subscriber');
    return false;
  }
  const ready = response.write(payload);
  if (!ready && response.writableLength > MAX_SSE_BUFFER_BYTES) {
    closeSseSubscriber(response, false);
    return false;
  }
  return true;
}

function closeSseSubscriber(response, graceful, reason = 'slow_subscriber') {
  subscribers.delete(response);
  if (response.destroyed || response.writableEnded) return;
  response.vimaxCloseReason = reason;
  if (graceful) response.end();
  else response.destroy();
}

function disconnectSubscribers() {
  for (const subscriber of [...subscribers]) closeSseSubscriber(subscriber, true, 'snapshot_overflow');
}

function interruptSession(sessionId) {
  const historyRequiredBefore = bridgeStreamState.current(sessionId)?.historyRequired || false;
  if (!bridgeStreamState.interrupt(sessionId)) return;
  const current = bridgeStreamState.current(sessionId);
  if (current?.turn_id) {
    const summary = completionLogRecord(sessionId, current.turn_id);
    if (summary) void performanceLog.append(summary);
  }
  const snapshot = bridgeStreamState.snapshotEvent(sessionId);
  broadcast(snapshot);
  if (!historyRequiredBefore && snapshot.historyRequired) disconnectSubscribers();
}

function rememberMessageSession(messageId, sessionId) {
  if (!messageId || !sessionId) return;
  messageSessions.delete(messageId);
  messageSessions.set(messageId, {sessionId, seenAt: Date.now()});
  pruneCorrelationMap(messageSessions);
}

function lookupMessageSession(messageId) {
  pruneCorrelationMap(messageSessions);
  return messageSessions.get(messageId)?.sessionId || null;
}
function rememberMessageTiming(messageId, sessionId) {
  if (!messageId) return;
  messageTimings.delete(messageId);
  messageTimings.set(messageId, {
    sessionId,
    acceptedAt: performance.now(),
    dispatchedAt: null,
    seenAt: Date.now(),
  });
  pruneCorrelationMap(messageTimings);
}

function markMessageDispatched(messageId) {
  const timing = lookupMessageTiming(messageId);
  if (!timing || timing.dispatchedAt !== null) return;
  timing.dispatchedAt = performance.now();
  timing.seenAt = Date.now();
  messageTimings.delete(messageId);
  messageTimings.set(messageId, timing);
  pruneCorrelationMap(messageTimings);
}

function lookupMessageTiming(messageId) {
  pruneCorrelationMap(messageTimings);
  return messageTimings.get(messageId) || null;
}

function completionLogRecord(sessionId, turnId) {
  const summary = bridgeStreamState.completionSummary(sessionId, turnId);
  if (!summary) return null;
  const context = lookupTurnContext(turnId);
  const timing = lookupMessageTiming(summary.messageId);
  return {
    ...summary,
    queue_wait_ms: context?.queueWaitMs ?? timing?.queueWaitMs ?? null,
    dispatch_to_turn_ms: context?.dispatchToTurnMs ?? timing?.dispatchToTurnMs ?? null,
    accepted_to_terminal_ms: timing ? Math.max(0, performance.now() - timing.acceptedAt) : null,
  };
}

function rememberTurnContext(turnId, context) {
  turnContexts.delete(turnId);
  turnContexts.set(turnId, {...context, seenAt: Date.now()});
  pruneCorrelationMap(turnContexts);
}

function lookupTurnContext(turnId) {
  pruneCorrelationMap(turnContexts);
  return turnContexts.get(turnId) || null;
}

function pruneCorrelationMap(map) {
  const cutoff = Date.now() - 24 * 60 * 60 * 1000;
  for (const [key, value] of map) {
    if (value.seenAt >= cutoff) break;
    map.delete(key);
  }
  while (map.size > 1_000) map.delete(map.keys().next().value);
}

function stopAgent(reason) {
  const child = agentProcess;
  interruptSession(activeSessionId);
  if (child) agentProcess = null;
  broadcast(agentQueue.cancel());
  if (!child) return;
  try {
    child.kill('SIGTERM');
  } catch {
    // The process may already have exited.
  }
  const message = reason === 'switch'
    ? 'Switching workspace'
    : reason === 'config'
      ? 'Configuration updated'
      : 'Generation stopped';
  broadcast({type: 'bridge_status', status: 'stopped', message, activeSessionId});
}


function agentCommand() {
  if (process.env.VIMAX_AGENT_COMMAND) {
    return {command: process.env.VIMAX_AGENT_COMMAND, args: splitArgs(process.env.VIMAX_AGENT_ARGS || '')};
  }
  const configuredPython = process.env.VIMAX_PYTHON_CMD;
  if (configuredPython) return {command: configuredPython, args: []};
  const bundledUv = process.env.VIMAX_UV_CMD || path.join(process.env.HOME || '', '.local', 'bin', 'uv');
  if (bundledUv && existsSync(bundledUv)) return {command: bundledUv, args: ['run', 'python']};
  const venvPython = path.join(repoRoot, '.venv', 'bin', 'python3');
  if (existsSync(venvPython)) return {command: venvPython, args: []};
  return {command: 'uv', args: ['run', 'python']};
}

function splitArgs(value) {
  return value.split(/\s+/).map((part) => part.trim()).filter(Boolean);
}

async function readJsonBody(request) {
  const chunks = [];
  for await (const chunk of request) chunks.push(chunk);
  if (!chunks.length) return {};
  const text = Buffer.concat(chunks).toString('utf8');
  if (text.length > 1_000_000) throw new Error('Request body is too large');
  return JSON.parse(text);
}

async function readSmallJsonBody(request, maxBytes) {
  const declaredLength = Number(request.headers['content-length'] || 0);
  if (declaredLength > maxBytes) {
    const error = new Error('Request body is too large');
    error.statusCode = 413;
    throw error;
  }
  const chunks = [];
  let size = 0;
  for await (const chunk of request) {
    size += chunk.length;
    if (size > maxBytes) {
      const error = new Error('Request body is too large');
      error.statusCode = 413;
      throw error;
    }
    chunks.push(chunk);
  }
  if (!size) return {};
  return JSON.parse(Buffer.concat(chunks, size).toString('utf8'));
}

async function readBinaryBody(request, maxBytes) {
  const chunks = [];
  let size = 0;
  for await (const chunk of request) {
    size += chunk.length;
    if (size > maxBytes) {
      const error = new Error(`File exceeds the ${formatByteLimit(maxBytes)} upload limit`);
      error.statusCode = 413;
      throw error;
    }
    chunks.push(chunk);
  }
  return Buffer.concat(chunks, size);
}

function formatByteLimit(bytes) {
  return `${Math.max(1, Math.round(bytes / (1024 * 1024)))} MB`;
}

function sendJson(response, status, payload) {
  if (response.writableEnded || response.destroyed) return;
  if (response.headersSent) {
    response.destroy();
    return;
  }
  response.writeHead(status, {'Content-Type': 'application/json; charset=utf-8'});
  response.end(JSON.stringify(payload));
}


async function serveProductionApp(response, pathname) {
  const requested = pathname === '/' ? 'index.html' : pathname.replace(/^\/+/, '');
  const candidate = path.resolve(webRoot, 'dist', requested);
  const distRoot = path.resolve(webRoot, 'dist');
  const safeCandidate = candidate.startsWith(`${distRoot}${path.sep}`) ? candidate : path.join(distRoot, 'index.html');
  const filePath = existsSync(safeCandidate) ? safeCandidate : path.join(distRoot, 'index.html');
  const body = await readFile(filePath);
  response.writeHead(200, {'Content-Type': artifactContentType(filePath)});
  response.end(body);
}

function shutdown() {
  stopAgent('shutdown');
  void Promise.allSettled([
    closeMediaCache(),
    performanceLog.pending,
    new Promise((resolve) => server.close(resolve)),
  ]).then(() => process.exit(0));
  setTimeout(() => process.exit(0), 1_000).unref();
}
