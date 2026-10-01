import {createReadStream, existsSync} from 'node:fs';
import {readFile} from 'node:fs/promises';
import {createServer} from 'node:http';
import path from 'node:path';
import {spawn} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {readAgentConfig, saveAgentConfig} from './config-store.mjs';
import {artifactContentType, createShot, deleteSession, listSessionArtifacts, moveShot, readContinuityReview, readFilmProgress, readFilmSnapshot, readProjectMetadata, readRemovedShots, readRenderAcceptance, readSessionHistory, readSessionState, readShotPlan, readShotPlans, removeShot, rerenderPrompt, restoreShot, storeWorkspaceUpload, updateProjectMetadata, updateRenderAcceptance, updateShotPlan} from './server-lib.mjs';
import {closeMediaCache, serveArtifact, serveThumbnail} from './media-serving.mjs';

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
      return sendJson(response, 200, {messages: await readSessionHistory(repoRoot, url.searchParams.get('session') || '')});
    }
    if (url.pathname === '/api/artifacts' && request.method === 'GET') {
      return sendJson(response, 200, {artifacts: await listSessionArtifacts(repoRoot, url.searchParams.get('session') || '')});
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
    if (url.pathname === '/api/messages' && request.method === 'POST') {
      const body = await readJsonBody(request);
      const text = String(body.text || '').trim();
      const sessionId = typeof body.sessionId === 'string' ? body.sessionId.trim() : '';
      if (!text) return sendJson(response, 400, {error: 'Message text is required'});
      if (!sessionId) return sendJson(response, 400, {error: 'Session id is required'});
      if (sessionId !== activeSessionId) return sendJson(response, 409, {error: 'The requested project is no longer active'});
      if (!agentProcess?.stdin.writable) return sendJson(response, 409, {error: 'Agent is not running'});
      agentProcess.stdin.write(`${text}\n`);
      return sendJson(response, 202, {ok: true});
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
  // A brand-new project can carry its user requirement up front: the CLI has no
  // flag for it, so it becomes the agent's first turn (the session record picks
  // it up when narrative planning runs).
  if (newSession && userRequirement) child.stdin.write(`${userRequirement}\n`);
  let childStdoutBuffer = '';
  broadcast({type: 'bridge_status', status: 'starting', message: newSession ? 'Creating workspace' : 'Opening workspace'});
  child.stdout.setEncoding('utf8');
  child.stdout.on('data', (chunk) => {
    if (agentProcess !== child) return;
    childStdoutBuffer += String(chunk);
    const lines = childStdoutBuffer.split(/\r?\n/);
    childStdoutBuffer = lines.pop() || '';
    for (const line of lines) consumeAgentLine(line);
  });
  child.stderr.setEncoding('utf8');
  child.stderr.on('data', (chunk) => {
    if (agentProcess !== child) return;
    for (const line of String(chunk).split(/\r?\n/)) {
      if (line.trim()) broadcast({type: 'terminal', stream: 'stderr', line});
    }
  });
  child.on('error', (error) => {
    if (agentProcess !== child) return;
    broadcast({type: 'error', message: `Agent process error: ${error.message}`});
  });
  child.on('exit', (code, signal) => {
    if (agentProcess !== child) return;
    agentProcess = null;
    broadcast({
      type: 'bridge_status',
      status: code === 0 || signal === 'SIGTERM' ? 'stopped' : 'error',
      message: signal ? `Agent stopped by ${signal}` : `Agent exited with code ${code ?? 0}`,
    });
  });
  setTimeout(async () => {
    if (agentProcess !== child) return;
    const state = await readSessionState(repoRoot);
    activeSessionId = state.activeSessionId || sessionId || activeSessionId;
    broadcast({type: 'sessions_changed', ...state, activeSessionId});
    broadcast({type: 'bridge_status', status: 'ready', message: 'Agent ready'});
  }, 350);
}

function consumeAgentLine(line) {
  if (!line.trim()) return;
  try {
    const event = JSON.parse(line);
    if (event.type === 'session') activeSessionId = event.session?.active_session_id || activeSessionId;
    broadcast(event);
    if (event.type === 'session') {
      readSessionState(repoRoot).then((state) => broadcast({type: 'sessions_changed', ...state}));
    }
  } catch {
    broadcast({type: 'terminal', stream: 'stdout', line});
  }
}

function openEventStream(request, response) {
  response.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache, no-transform',
    Connection: 'keep-alive',
    'X-Accel-Buffering': 'no',
  });
  response.write(`data: ${JSON.stringify({type: 'bridge_status', status: agentProcess ? 'ready' : 'idle', message: agentProcess ? 'Agent connected' : 'Agent idle'})}\n\n`);
  subscribers.add(response);
  const heartbeat = setInterval(() => response.write(': keepalive\n\n'), 15_000);
  request.on('close', () => {
    clearInterval(heartbeat);
    subscribers.delete(response);
  });
}

function broadcast(event) {
  const payload = `data: ${JSON.stringify(event)}\n\n`;
  for (const subscriber of subscribers) subscriber.write(payload);
}

function stopAgent(reason) {
  if (!agentProcess) return;
  const child = agentProcess;
  agentProcess = null;
  child.kill('SIGTERM');
  const message = reason === 'switch'
    ? 'Switching workspace'
    : reason === 'config'
      ? 'Configuration updated'
      : 'Generation stopped';
  broadcast({type: 'bridge_status', status: 'stopped', message});
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
    new Promise((resolve) => server.close(resolve)),
  ]).then(() => process.exit(0));
  setTimeout(() => process.exit(0), 1_000).unref();
}
