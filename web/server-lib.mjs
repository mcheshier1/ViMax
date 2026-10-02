import {createHash, randomUUID} from 'node:crypto';
import {createReadStream} from 'node:fs';
import {lstat, mkdir, open, readFile, readdir, realpath, rename, rm, stat, utimes, writeFile} from 'node:fs/promises';
import path from 'node:path';
import {readClipSettings} from './config-store.mjs';
import {withProjectWriteLock} from './project-lock.mjs';
import {assembleClips} from './film-assembly.mjs';

const IMAGE_EXTENSIONS = new Set(['.png', '.jpg', '.jpeg', '.webp', '.gif']);
const VIDEO_EXTENSIONS = new Set(['.mp4', '.webm', '.mov']);
const TEXT_EXTENSIONS = new Set(['.txt', '.md', '.json']);

const historyCache = new Map();
const historyReads = new Map();
const MAX_CACHED_HISTORIES = 16;

function copyHistoryMessages(messages) {
  return messages.map((message) => ({...message}));
}

function sameHistoryFile(left, right) {
  return left.dev === right.dev && left.ino === right.ino && left.size === right.size
    && left.mtimeNs === right.mtimeNs && left.ctimeNs === right.ctimeNs;
}

// All memoization is scoped to one read. A later request must see external writers,
// including same-size replacements whose approval checksum no longer matches.
function memoizeRead(operation) {
  const reads = new Map();
  return (key) => {
    if (!reads.has(key)) reads.set(key, operation(key));
    return reads.get(key);
  };
}

function createFilmReadContext() {
  return {
    entries: memoizeRead((directory) => readdir(directory, {withFileTypes: true})),
    stat: memoizeRead(stat),
    lstat: memoizeRead(lstat),
    json: memoizeRead(readJsonOptional),
    hash: memoizeRead(sha256File),
    scenes: new Map(),
  };
}

export async function readSessionState(repoRoot) {
  const fallback = {activeSessionId: '', sessions: []};
  try {
    const payload = JSON.parse(await readFile(path.join(repoRoot, '.vimax', 'sessions.json'), 'utf8'));
    const records = Object.values(payload.sessions ?? {})
      .filter((record) => record && typeof record === 'object')
      .map(sanitizeSession);
    records.sort((left, right) => right.updatedAt.localeCompare(left.updatedAt));
    return {
      activeSessionId: String(payload.active_session_id ?? ''),
      sessions: records,
    };
  } catch {
    return fallback;
  }
}

export async function deleteSession(repoRoot, sessionId) {
  assertSessionId(sessionId);
  const statePath = path.join(repoRoot, '.vimax', 'sessions.json');
  const payload = JSON.parse(await readFile(statePath, 'utf8'));
  const sessions = payload.sessions && typeof payload.sessions === 'object' ? payload.sessions : {};
  if (!sessions[sessionId]) throw new Error('Project not found');

  delete sessions[sessionId];
  const remaining = Object.values(sessions)
    .filter((record) => record && typeof record === 'object')
    .sort((left, right) => String(right.updated_at ?? right.created_at ?? '').localeCompare(String(left.updated_at ?? left.created_at ?? '')));
  if (payload.active_session_id === sessionId || !sessions[payload.active_session_id]) {
    payload.active_session_id = String(remaining[0]?.session_id ?? '');
  }
  payload.sessions = sessions;

  const temporaryPath = `${statePath}.${process.pid}.tmp`;
  await writeFile(temporaryPath, `${JSON.stringify(payload, null, 2)}\n`, {mode: 0o600});
  await rename(temporaryPath, statePath);
  await rm(resolveSessionRoot(repoRoot, sessionId), {recursive: true, force: true});
  await removeSessionLogRecords(repoRoot, sessionId);
  return readSessionState(repoRoot);
}

export async function readSessionHistory(repoRoot, sessionId) {
  assertSessionId(sessionId);
  const logPath = path.join(repoRoot, '.vimax', 'logs', 'loop_history.jsonl');
  const cacheKey = `${logPath}\0${sessionId}`;
  const pending = historyReads.get(cacheKey);
  if (pending) return copyHistoryMessages(await pending);

  const read = (async () => {
    let before;
    try {
      before = await stat(logPath, {bigint: true});
    } catch {
      historyCache.delete(cacheKey);
      return [];
    }
    const cached = historyCache.get(cacheKey);
    if (cached && sameHistoryFile(cached.metadata, before)) {
      historyCache.delete(cacheKey);
      historyCache.set(cacheKey, cached);
      return cached.messages;
    }

    try {
      const messages = [];
      let messageCount = 0;
      const appendMessage = (message) => {
        messages.push(message);
        messageCount += 1;
        if (messages.length > 120) messages.shift();
      };
      const consumeLine = (line) => {
        if (!line) return;
        let record;
        try {
          record = JSON.parse(line);
        } catch {
          return;
        }
        if (record.session_id !== sessionId || !record.raw_user_input) return;
        const turnId = String(record.turn_id || `turn-${messageCount}`);
        appendMessage({
          id: `${turnId}-user`,
          role: 'user',
          text: displayUserInput(record.raw_user_input),
          createdAt: String(record.created_at || record.timestamp || ''),
        });
        for (const round of Array.isArray(record.tool_rounds) ? record.tool_rounds : []) {
          for (const result of Array.isArray(round.tool_results) ? round.tool_results : []) {
            appendMessage({
              id: `${turnId}-tool-${messageCount}`,
              role: 'activity',
              text: historyToolResultText(result),
              tool: String(result.name || 'tool'),
              status: result.ok === false ? 'error' : 'done',
              stage: result.ok === false ? 'failed' : 'completed',
              createdAt: String(record.created_at || record.timestamp || ''),
            });
          }
        }
        if (record.final_assistant_text) {
          appendMessage({
            id: `${turnId}-assistant`,
            role: record.status === 'failed' ? 'error' : 'assistant',
            text: String(record.final_assistant_text),
            createdAt: String(record.created_at || record.timestamp || ''),
          });
        }
      };
      const input = createReadStream(logPath, {encoding: 'utf8'});
      const lineParts = [];
      const consumeChunk = (chunk) => {
        let start = 0;
        while (true) {
          const newline = chunk.indexOf('\n', start);
          if (newline === -1) {
            if (start < chunk.length) lineParts.push(chunk.slice(start));
            return;
          }
          let line = chunk.slice(start, newline);
          if (lineParts.length) {
            lineParts.push(line);
            line = lineParts.join('');
            lineParts.length = 0;
          }
          if (line.endsWith('\r')) line = line.slice(0, -1);
          consumeLine(line);
          start = newline + 1;
        }
      };
      try {
        for await (const chunk of input) consumeChunk(chunk);
        if (lineParts.length) consumeLine(lineParts.join(''));
      } finally {
        input.destroy();
      }
      let after;
      try {
        after = await stat(logPath, {bigint: true});
      } catch {
        return messages;
      }
      if (sameHistoryFile(before, after)) {
        historyCache.delete(cacheKey);
        historyCache.set(cacheKey, {metadata: after, messages});
        if (historyCache.size > MAX_CACHED_HISTORIES) {
          historyCache.delete(historyCache.keys().next().value);
        }
      }
      return messages;
    } catch {
      historyCache.delete(cacheKey);
      return [];
    }
  })();
  historyReads.set(cacheKey, read);
  try {
    return copyHistoryMessages(await read);
  } finally {
    if (historyReads.get(cacheKey) === read) historyReads.delete(cacheKey);
  }
}

export async function storeWorkspaceUpload(repoRoot, sessionId, fileName, data) {
  const sessionRoot = resolveSessionRoot(repoRoot, sessionId);
  const sessionInfo = await stat(sessionRoot);
  if (!sessionInfo.isDirectory()) throw new Error('Session workspace is not a directory');

  const safeName = validateUploadName(fileName);
  const uploadRoot = path.join(sessionRoot, 'uploads');
  await mkdir(uploadRoot, {recursive: true, mode: 0o700});
  const uploadRootInfo = await lstat(uploadRoot);
  if (!uploadRootInfo.isDirectory() || uploadRootInfo.isSymbolicLink()) {
    throw new Error('Workspace upload directory is not safe');
  }

  const extension = path.extname(safeName);
  const stem = safeName.slice(0, safeName.length - extension.length);
  const payload = Buffer.isBuffer(data) ? data : Buffer.from(data);
  for (let attempt = 1; attempt <= 1_000; attempt += 1) {
    const storedName = attempt === 1 ? safeName : `${stem} (${attempt})${extension}`;
    const destination = path.join(uploadRoot, storedName);
    try {
      await writeFile(destination, payload, {flag: 'wx', mode: 0o600});
      return {
        name: storedName,
        path: path.relative(sessionRoot, destination).split(path.sep).join('/'),
        size: payload.byteLength,
      };
    } catch (error) {
      if (error?.code === 'EEXIST') continue;
      throw error;
    }
  }
  throw new Error('Could not allocate a unique upload filename');
}

function historyToolResultText(result) {
  if (result.ok !== false) return 'Completed';
  const content = String(result.content || 'Tool failed').trim();
  try {
    const payload = JSON.parse(content);
    const detail = payload?.error ?? payload?.message;
    if (detail) return conciseText(detail);
  } catch {
    // The provider may return a plain-text error.
  }
  return conciseText(content);
}

function conciseText(value) {
  const text = String(value || 'Tool failed').replace(/\s+/g, ' ').trim();
  return text.length > 280 ? `${text.slice(0, 277)}…` : text;
}

function displayUserInput(value) {
  return String(value || '')
    .replace(/\s*<workspace_uploads>.*<\/workspace_uploads>\s*$/s, '')
    .trim();
}

export async function listSessionArtifacts(repoRoot, sessionId) {
  return collectSessionArtifacts(resolveSessionRoot(repoRoot, sessionId), sessionId, createFilmReadContext());
}

async function collectSessionArtifacts(sessionRoot, sessionId, context, referenced = []) {
  const artifacts = new Map();

  async function add(absolute) {
    if (!absolute.startsWith(`${sessionRoot}${path.sep}`) || artifacts.has(absolute)) return;
    const name = path.basename(absolute);
    const extension = path.extname(name).toLowerCase();
    const kind = IMAGE_EXTENSIONS.has(extension) ? 'image'
      : VIDEO_EXTENSIONS.has(extension) ? 'video'
        : TEXT_EXTENSIONS.has(extension) ? 'document' : null;
    if (!kind) return;
    const info = await context.stat(absolute).catch(() => null);
    if (!info?.isFile()) return;
    const relativePath = relativeArtifactPath(sessionRoot, absolute);
    artifacts.set(absolute, {
      path: relativePath,
      name,
      kind,
      size: info.size,
      updatedAt: info.mtime.toISOString(),
      url: `/api/artifact?session=${encodeURIComponent(sessionId)}&path=${encodeURIComponent(relativePath)}`,
    });
  }

  async function walk(directory) {
    const entries = await context.entries(directory).catch(() => []);
    for (const entry of entries) {
      if (artifacts.size >= 400) break;
      if (entry.name.startsWith('.')) continue;
      const absolute = path.join(directory, entry.name);
      if (entry.isDirectory()) {
        // Moviepy scratch debris is represented by the transition video beside it.
        if (entry.name !== 'cache') await walk(absolute);
      } else if (entry.isFile()) {
        await add(absolute);
      }
    }
  }

  await walk(sessionRoot);
  // Browsing is bounded, but the film must never lose a preview because an unrelated
  // folder filled that budget. These paths came from symlink-free stage discovery.
  for (const absolute of referenced) await add(absolute);
  return [...artifacts.values()].sort((left, right) => right.updatedAt.localeCompare(left.updatedAt));
}

export async function readProjectMetadata(repoRoot, sessionId) {
  const payload = await readSessionsPayload(repoRoot);
  const record = sessionRecord(payload, sessionId);
  if (!record) return null;
  return projectMetadata(repoRoot, sessionId, record);
}

export async function updateProjectMetadata(repoRoot, input = {}) {
  const body = input && typeof input === 'object' ? input : {};
  const fields = [
    ['projectName', 'project_name'],
    ['idea', 'idea'],
    ['userRequirement', 'user_requirement'],
    ['style', 'style'],
  ];
  for (const [field] of fields) {
    if (body[field] !== undefined && typeof body[field] !== 'string') {
      const error = new Error(`${field} must be a string`);
      error.statusCode = 400;
      throw error;
    }
  }

  const payload = await readSessionsPayload(repoRoot);
  const record = sessionRecord(payload, body.sessionId);
  if (!record) return null;

  const changed = [];
  for (const [field, key] of fields) {
    if (body[field] === undefined) continue;
    const value = body[field].trim();
    if (String(record[key] ?? '') !== value) {
      record[key] = value;
      changed.push(field);
    }
  }

  // A style change invalidates the artifacts that were generated with the old
  // style. Without an explicit confirmation we only report the paths (relative
  // to the session working dir) so the caller can warn before spending money.
  // The confirmation re-posts the same diff with `invalidate: true`, so the
  // style may already be persisted and `changed` may be empty: deletion is
  // keyed off the flag, not off a fresh diff.
  let invalidated = [];
  let requiresInvalidation = false;
  if (body.invalidate === true) {
    const derived = await listStyleDerivedArtifacts(repoRoot, body.sessionId);
    invalidated = await removeStyleDerivedArtifacts(repoRoot, body.sessionId, derived);
    // The render tool refuses to re-render while the manifest pins the old
    // style, so the confirmed change must move the manifest with the session.
    await updateRenderManifestStyle(repoRoot, record.working_dir, String(record.style ?? ''));
  } else if (changed.includes('style')) {
    requiresInvalidation = true;
    invalidated = await listStyleDerivedArtifacts(repoRoot, body.sessionId);
  }

  if (changed.length) {
    record.updated_at = new Date().toISOString();
    const statePath = path.join(repoRoot, '.vimax', 'sessions.json');
    const temporaryPath = `${statePath}.${process.pid}.tmp`;
    await writeFile(temporaryPath, `${JSON.stringify(payload, null, 2)}\n`, {mode: 0o600});
    await rename(temporaryPath, statePath);
  }

  return {
    session: await projectMetadata(repoRoot, body.sessionId, record),
    changed,
    invalidated,
    requiresInvalidation,
  };
}

/**
 * What the agent is told after a confirmed style change deleted the
 * style-derived artifacts. Deleting them only makes the rebuild possible:
 * the render is what actually regenerates them, and the render tool skips
 * anything that already exists, so without this instruction the project is
 * left with the old artifacts gone and nothing scheduled to replace them.
 */
export function rerenderPrompt(style) {
  const trimmed = String(style ?? '').trim();
  const styleClause = trimmed
    ? `The project style is now "${trimmed}".`
    : 'The project style is now empty: ask the user which style to use before rendering, and never choose one yourself.';
  return [
    'The project style changed on the project page, so the previously rendered portraits, keyframes, shot clips and final video were deleted and must be rebuilt.',
    styleClause,
    'Re-run the render for this session with `vimax_render_video` and stop after the portraits phase, so the new style can be approved before the rest is regenerated.',
  ].join(' ');
}

/**
 * What the last render actually left on disk, per render phase.
 *
 * The `stale` flags in the session record are not written by the render, so
 * they cannot answer "is this project rendered?" — only the files can. This is
 * also what makes a confirmed style change legible: the counts drop to zero
 * when its artifacts are deleted and climb back as the render rebuilds them.
 */
export async function readProjectArtifacts(repoRoot, sessionId) {
  const inventory = {portraits: 0, shots: 0, frames: 0, clips: 0, finalVideo: false};
  await collectRenderInventory(resolveSessionRoot(repoRoot, sessionId), inventory);
  return inventory;
}

async function collectRenderInventory(directory, inventory) {
  let entries;
  try {
    entries = await readdir(directory, {withFileTypes: true});
  } catch {
    return;
  }
  if (entries.some((entry) => entry.isFile() && entry.name === 'final_video.mp4')) inventory.finalVideo = true;
  for (const entry of entries) {
    if (!entry.isDirectory() || entry.name.startsWith('.')) continue;
    const absolute = path.join(directory, entry.name);
    if (entry.name === 'shots') {
      await collectShotInventory(absolute, inventory);
      continue;
    }
    if (entry.name === 'character_portraits') {
      await collectPortraitInventory(absolute, inventory);
      continue;
    }
    await collectRenderInventory(absolute, inventory);
  }
}

async function collectShotInventory(shotsDir, inventory) {
  let shots;
  try {
    shots = await readdir(shotsDir, {withFileTypes: true});
  } catch {
    return;
  }
  for (const shot of shots) {
    if (!shot.isDirectory() || shot.name.startsWith('.')) continue;
    inventory.shots += 1;
    const names = await collectFileNames(path.join(shotsDir, shot.name));
    if (names.includes('first_frame.png')) inventory.frames += 1;
    if (names.includes('video.mp4')) inventory.clips += 1;
  }
}

/** One portrait set per directory holding a `front.png`, at any depth (model-slug dirs or legacy). */
async function collectPortraitInventory(directory, inventory) {
  let entries;
  try {
    entries = await readdir(directory, {withFileTypes: true});
  } catch {
    return;
  }
  if (entries.some((entry) => entry.isFile() && entry.name === 'front.png')) {
    inventory.portraits += 1;
    return;
  }
  for (const entry of entries) {
    if (!entry.isDirectory() || entry.name.startsWith('.')) continue;
    await collectPortraitInventory(path.join(directory, entry.name), inventory);
  }
}

async function collectFileNames(directory) {
  const names = [];
  let entries;
  try {
    entries = await readdir(directory, {withFileTypes: true});
  } catch {
    return names;
  }
  for (const entry of entries) {
    if (entry.name.startsWith('.')) continue;
    if (entry.isFile()) {
      names.push(entry.name);
      continue;
    }
    // `isDirectory` is false for symlinks, so a link is never followed.
    if (entry.isDirectory()) names.push(...(await collectFileNames(path.join(directory, entry.name))));
  }
  return names;
}

// Style-derived artifacts only: portraits (with their registry), keyframes,
// clips/transition videos, and the concatenated final video. Text planning
// artifacts and `*_selector_output.json` prompt records are never listed.
export async function listStyleDerivedArtifacts(repoRoot, sessionId) {
  const sessionRoot = resolveSessionRoot(repoRoot, sessionId);
  const artifacts = [];
  await collectStyleDerivedArtifacts(sessionRoot, sessionRoot, artifacts);
  return artifacts.sort();
}

export function resolveArtifactPath(repoRoot, sessionId, relativePath) {
  const sessionRoot = resolveSessionRoot(repoRoot, sessionId);
  const candidate = path.resolve(sessionRoot, String(relativePath || ''));
  if (candidate === sessionRoot || !candidate.startsWith(`${sessionRoot}${path.sep}`)) {
    throw new Error('Artifact path escapes the active session');
  }
  return candidate;
}

async function readSessionsPayload(repoRoot) {
  try {
    const payload = JSON.parse(await readFile(path.join(repoRoot, '.vimax', 'sessions.json'), 'utf8'));
    return payload && typeof payload === 'object' ? payload : null;
  } catch {
    return null;
  }
}

function sessionRecord(payload, sessionId) {
  const id = String(sessionId || '');
  if (!/^[A-Za-z0-9][A-Za-z0-9-]{0,95}$/.test(id)) return null;
  const record = payload?.sessions?.[id];
  return record && typeof record === 'object' ? record : null;
}

async function projectMetadata(repoRoot, sessionId, record) {
  const workingDir = String(record.working_dir ?? '');
  return {
    sessionId: String(record.session_id ?? sessionId),
    projectName: String(record.project_name ?? ''),
    idea: String(record.idea ?? ''),
    userRequirement: String(record.user_requirement ?? ''),
    style: String(record.style ?? ''),
    stage: String(record.stage ?? 'created'),
    summary: String(record.summary ?? ''),
    workingDir,
    manifest: await readRenderManifest(repoRoot, workingDir),
    artifacts: await readProjectArtifacts(repoRoot, String(record.session_id ?? sessionId)),
  };
}

async function readRenderManifest(repoRoot, workingDir, context = createFilmReadContext()) {
  if (!workingDir) return null;
  try {
    const manifest = await context.json(path.resolve(repoRoot, workingDir, 'render_manifest.json'));
    return manifest && typeof manifest === 'object' && !Array.isArray(manifest) ? manifest : null;
  } catch {
    return null;
  }
}

async function updateRenderManifestStyle(repoRoot, workingDir, style) {
  if (!workingDir) return;
  const manifestPath = path.resolve(repoRoot, workingDir, 'render_manifest.json');
  let manifest;
  try {
    manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
  } catch {
    // No manifest yet: the render writes one with the new style.
    return;
  }
  if (!manifest || typeof manifest !== 'object' || Array.isArray(manifest)) return;
  if (String(manifest.style ?? '') === style) return;
  const temporaryPath = `${manifestPath}.${process.pid}.tmp`;
  await writeFile(temporaryPath, `${JSON.stringify({...manifest, style}, null, 2)}\n`, {mode: 0o600});
  await rename(temporaryPath, manifestPath);
}

async function collectStyleDerivedArtifacts(root, directory, artifacts) {
  let entries;
  try {
    entries = await readdir(directory, {withFileTypes: true});
  } catch {
    return;
  }
  for (const entry of entries) {
    if (entry.name.startsWith('.')) continue;
    const absolute = path.join(directory, entry.name);
    // `withFileTypes` reports symlinks as neither directories nor files, so a
    // link pointing out of the session is never followed or listed.
    if (entry.isDirectory()) {
      if (entry.name === 'character_portraits' || entry.name === 'shots') {
        await collectContainerEntries(root, absolute, entry.name === 'shots', artifacts);
        continue;
      }
      await collectStyleDerivedArtifacts(root, absolute, artifacts);
      continue;
    }
    if (entry.isFile() && entry.name === 'final_video.mp4') {
      artifacts.push(relativeArtifactPath(root, absolute));
    }
  }
}

/** Files under `shots/<n>/` the render consumes instead of produces. */
const SHOT_INPUT_FILES = new Set(['shot_description.json']);
const SHOT_INPUT_SUFFIX = '_selector_output.json';

function isShotInput(name) {
  return SHOT_INPUT_FILES.has(name) || name.endsWith(SHOT_INPUT_SUFFIX);
}

/**
 * Lists the style-derived entries of a render output container.
 *
 * `character_portraits/` holds one directory per portrait set — `<idx>_<name>`
 * before artifacts were grouped by model, `<model-slug>` after — so only its
 * directories are listed. `shots/<shot>/` holds keyframes, clips, camera images
 * and scratch directories, flat (legacy) or inside a model-slug directory, so
 * both its files and its directories are listed.
 *
 * Prompt and shot-description JSONs are render inputs and are never listed, and
 * `withFileTypes` reports symlinks as neither directory nor file, so a link
 * pointing out of the session is never followed or listed.
 */
async function collectContainerEntries(root, container, perShot, artifacts) {
  let entries;
  try {
    entries = await readdir(container, {withFileTypes: true});
  } catch {
    return;
  }
  for (const entry of entries) {
    if (!entry.isDirectory() || entry.name.startsWith('.')) continue;
    const absolute = path.join(container, entry.name);
    if (!perShot) {
      artifacts.push(relativeArtifactPath(root, absolute));
      continue;
    }
    let shotEntries;
    try {
      shotEntries = await readdir(absolute, {withFileTypes: true});
    } catch {
      continue;
    }
    for (const shotEntry of shotEntries) {
      if (shotEntry.name.startsWith('.')) continue;
      if (isShotInput(shotEntry.name)) continue;
      if (!shotEntry.isDirectory() && !shotEntry.isFile()) continue;
      artifacts.push(relativeArtifactPath(root, path.join(absolute, shotEntry.name)));
    }
  }
}

async function removeStyleDerivedArtifacts(repoRoot, sessionId, relativePaths) {
  const sessionRoot = resolveSessionRoot(repoRoot, sessionId);
  const removed = [];
  for (const relativePath of relativePaths) {
    const absolute = path.resolve(sessionRoot, relativePath);
    if (absolute === sessionRoot || !absolute.startsWith(`${sessionRoot}${path.sep}`)) {
      throw new Error('Style artifact path escapes the active session');
    }
    let info;
    try {
      info = await lstat(absolute);
    } catch {
      continue;
    }
    // Removing a symlink never recurses into its target.
    await rm(absolute, {recursive: !info.isSymbolicLink(), force: true});
    removed.push(relativePath);
  }
  return removed;
}

function relativeArtifactPath(root, absolute) {
  return path.relative(root, absolute).split(path.sep).join('/');
}

// Render acceptance (locks): a human reviews each render stage and locks what they
// accept, and the render refuses to advance past a stage that is not accepted. The
// lock file is shared with the Python render gate (agent_runtime/vimax_adapters.py),
// so this shape and the path rules below have to stay in step with it.
const ACCEPTANCE_FILENAME = 'render_acceptance.json';
const RENDER_ROOTS = new Set(['script2video', 'idea2video', 'novel2video']);
const STAGE_SCOPES = {portraits: 'session', keyframes: 'shot', clips: 'shot', final_video: 'session'};
const STAGE_ORDER = ['portraits', 'keyframes', 'clips', 'final_video'];
const KEYFRAME_NAMES = new Set(['first_frame.png', 'last_frame.png']);
// A rejection is a short human note, not a document; the cap keeps the lock file sane.
const REASON_MAX_LENGTH = 2000;

/** A clip's per-second price, by the model that renders it.
 *
 * Read from each model's own page (`from $0.126/second` for kling-v3.0-std with its native
 * audio, versus `$0.112/second` for the silent kling-video-o1 the sequence used to render
 * with). A model that is not listed is priced at the highest known rate: an estimate that is
 * wrong upward is a surprise, one that is wrong downward is a bill.
 */
const CLIP_USD_PER_SECOND_BY_MODEL = {
  'kwaivgi/kling-v3.0-std': 0.126,
  'kwaivgi/kling-v3.0-pro': 0.168,
  'kwaivgi/kling-video-o1': 0.112,
  'google/veo-3.1-lite': 0.05,
};

function clipUsdPerSecond(model, resolution) {
  const normalizedModel = String(model || '').trim();
  if (normalizedModel === 'heygen/heygen-video-1') {
    // Budget with the higher reference-duration SKU while its applicability remains unconfirmed.
    return resolution === '480p' ? 0.04 : 0.06;
  }
  const rates = Object.values(CLIP_USD_PER_SECOND_BY_MODEL);
  return CLIP_USD_PER_SECOND_BY_MODEL[normalizedModel] ?? Math.max(...rates);
}

/**
 * The acceptance state of one render root, derived from the files on disk and the
 * stored lock at call time. `stages` is always the four stages in order and every
 * slot state comes from disk (a stale file can never claim a lock).
 */
export async function readRenderAcceptance(repoRoot, sessionId, root = '') {
  const record = sessionRecord(await readSessionsPayload(repoRoot), sessionId);
  if (!record) return null;
  const workingDir = sessionWorkingDir(repoRoot, sessionId, record);
  const context = createFilmReadContext();
  return acceptancePayload(repoRoot, workingDir, await resolveRenderRoot(repoRoot, workingDir, root, context), context);
}

/** One coherent read budget for the film, without re-entering mutation locks. */
export async function readFilmSnapshot(repoRoot, sessionId, root = '') {
  const record = sessionRecord(await readSessionsPayload(repoRoot), sessionId);
  if (!record) return null;
  const workingDir = sessionWorkingDir(repoRoot, sessionId, record);
  const context = createFilmReadContext();
  const resolved = await resolveRenderRoot(repoRoot, workingDir, root, context);
  const discovery = await collectAcceptanceState(workingDir, resolved, context);
  const referenced = [
    ...discovery.portraits,
    ...discovery.finalVideo,
    ...discovery.shots.flatMap((shot) => [...shot.keyframes, ...shot.clips]),
  ].map((relative) => path.resolve(workingDir, relative));
  const [acceptance, plans, removed, continuity, artifacts] = await Promise.all([
    acceptancePayload(repoRoot, workingDir, resolved, context, discovery),
    shotPlansPayload(workingDir, resolved, discovery, context),
    listRemoved(workingDir, resolved, context),
    continuityPayload(repoRoot, workingDir, resolved, record, context, discovery),
    collectSessionArtifacts(resolveSessionRoot(repoRoot, sessionId), sessionId, context, referenced),
  ]);
  return {acceptance, plans, removed, continuity, artifacts};
}

/** Stitch the accepted active clips for a film, without involving the render agent. */
export async function assembleFilm(repoRoot, input = {}) {
  if (!isPlainObject(input) || typeof input.sessionId !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9-]{0,95}$/.test(input.sessionId)
    || typeof input.root !== 'string' || !input.root.trim()
    || (input.revision !== undefined && typeof input.revision !== 'string')) {
    throw acceptanceError('sessionId and root are required and must be valid; revision must be a string when supplied');
  }
  const sessionId = input.sessionId;
  const record = sessionRecord(await readSessionsPayload(repoRoot), sessionId);
  if (!record) return null;
  const workingDir = sessionWorkingDir(repoRoot, sessionId, record);
  return withProjectWriteLock(workingDir, async () => {
    const root = await resolveRenderRoot(repoRoot, workingDir, input.root);
    if (input.revision !== undefined) {
      const progress = await readFilmProgress(repoRoot, sessionId, root);
      if (progress?.revision !== input.revision) throw conflictError('Film changed since it was loaded; refresh and try assembling again');
    }
    const context = createFilmReadContext();
    const discovery = await collectAcceptanceState(workingDir, root, context);
    const order = discovery.order;
    if (!order.length) throw conflictError('There are no active shots to assemble');
    if (new Set(order).size !== order.length) throw conflictError('The camera tree contains duplicate active shots');
    const slots = new Map(discovery.shots.map((shot) => [shot.shot, shot]));
    const clips = [];
    const canonicalWorkingDir = await realpath(workingDir);
    for (const shot of order) {
      const found = slots.get(shot);
      if (!found || found.clips.length !== 1) throw conflictError(`Shot ${shot} is missing its current video clip`);
      const store = await readAcceptanceFile(workingDir, context);
      const shotRecord = store[root]?.shots?.[shot]?.clips;
      if (await slotState(workingDir, shotRecord, found.clips, context) !== 'accepted') {
        throw conflictError(`Shot ${shot} does not have a current accepted clip`);
      }
      const clipPath = await realpath(path.resolve(workingDir, found.clips[0]));
      if (!clipPath.startsWith(`${canonicalWorkingDir}${path.sep}`)) throw conflictError(`Shot ${shot} clip escapes the project`);
      clips.push(clipPath);
    }
    // Every active tree slot must correspond to a real shot container; scenes are assembled
    // in numeric scene order by filmOrder rather than flattened or alphabetically guessed.
    const signature = createHash('sha256');
    signature.update('vimax-local-assembly-v1\0');
    const inputHashes = [];
    for (let i = 0; i < order.length; i += 1) {
      const hash = await sha256File(clips[i]);
      inputHashes.push(hash);
      signature.update(order[i]); signature.update('\0');
      signature.update(hash); signature.update('\0');
    }
    const key = signature.digest('hex');
    const finalPath = path.join(workingDir, root, 'final_video.mp4');
    const cachePath = `${finalPath}.assembly.json`;
    const cache = await readJsonOptional(cachePath).catch(() => undefined);
    if (cache?.key === key && typeof cache.outputHash === 'string' && await isRegularFile(finalPath)
      && await sha256File(finalPath) === cache.outputHash) {
      return {path: relativeArtifactPath(workingDir, finalPath), reused: true, shotCount: order.length};
    }

    const temporaryPath = path.join(path.dirname(finalPath), `.final_video.${process.pid}.${randomUUID()}.tmp.mp4`);
    const oldFinalExists = await isRegularFile(finalPath);
    try {
      await assembleClips(clips, temporaryPath).catch((cause) => {
        const error = new Error(`Local FFmpeg assembly failed: ${cause.message}`, {cause});
        error.statusCode = 500;
        throw error;
      });
      const verifiedContext = createFilmReadContext();
      const verified = await collectAcceptanceState(workingDir, root, verifiedContext);
      if (JSON.stringify(verified.order) !== JSON.stringify(order)) throw conflictError('Film playback order changed during assembly');
      const verifiedStore = await readAcceptanceFile(workingDir, verifiedContext);
      const after = createHash('sha256').update('vimax-local-assembly-v1\0');
      for (let index = 0; index < order.length; index += 1) {
        const found = verified.shots.find((shot) => shot.shot === order[index]);
        if (!found || found.clips.length !== 1
          || await slotState(workingDir, verifiedStore[root]?.shots?.[order[index]]?.clips, found.clips, verifiedContext) !== 'accepted') {
          throw conflictError(`Shot ${order[index]} is no longer accepted`);
        }
        const hash = await sha256File(clips[index]);
        if (hash !== inputHashes[index]) throw conflictError('An input clip changed during assembly');
        after.update(order[index]); after.update('\0'); after.update(hash); after.update('\0');
      }
      if (after.digest('hex') !== key) throw conflictError('Input clips changed during assembly');
      const outputHash = await sha256File(temporaryPath);
      await withMutationTransaction(async (transaction) => {
        await transaction.remove(finalPath);
        await transaction.move(temporaryPath, finalPath);
        if (oldFinalExists && cache?.key !== key) {
          const rootStore = isPlainObject(verifiedStore[root]) ? verifiedStore[root] : {};
          if (isPlainObject(rootStore.final_video)) {
            rootStore.final_video = {...rootStore.final_video, invalidated_at: isoSeconds(), invalidation_reason: 'The assembled clip inputs changed; review the new final film.'};
            verifiedStore[root] = rootStore;
            await transaction.write(path.join(workingDir, ACCEPTANCE_FILENAME), `${JSON.stringify(verifiedStore, null, 2)}\n`);
          }
        }
        await transaction.write(cachePath, `${JSON.stringify({key, outputHash, shots: order})}\n`);
      });
      return {path: relativeArtifactPath(workingDir, finalPath), reused: false, shotCount: order.length};
    } finally {
      await rm(temporaryPath, {force: true}).catch(() => {});
    }
  });
}

const PROGRESS_READ_BYTES = 256 * 1024;
const PROGRESS_TRAIL_ROWS = 128;

/** Fast polling reads only bounded status/trail data and a fixed set of metadata. */
export async function readFilmProgress(repoRoot, sessionId, root = '') {
  const record = sessionRecord(await readSessionsPayload(repoRoot), sessionId);
  if (!record) return null;
  const workingDir = sessionWorkingDir(repoRoot, sessionId, record);
  const resolved = await resolveRenderRoot(repoRoot, workingDir, root);
  const [statusFile, trailFile, watched] = await Promise.all([
    readProgressFile(path.join(workingDir, 'render_status.json')),
    readProgressFile(path.join(workingDir, 'render_events.jsonl'), true),
    Promise.all(['render_manifest.json', ACCEPTANCE_FILENAME, CONTINUITY_FILENAME, resolved]
      .map(async (relative) => fileRevision(await stat(path.join(workingDir, relative)).catch(() => null)))),
  ]);
  const belongsToRoot = (value) => {
    const eventRoot = value?.render_mode || value?.root;
    return !eventRoot || eventRoot === resolved;
  };
  let status = progressObject(statusFile.text);
  if (!belongsToRoot(status)) status = null;
  const rows = trailFile.text.split('\n').map(progressObject).filter((row) => row && belongsToRoot(row));
  const lastStatus = rows.findLast((row) => typeof row.status === 'string');
  // A reader may overlap the producer's status-write / append pair. Prefer the newer
  // complete record, and use the trail if the status file is temporarily half-written.
  if (lastStatus && (!status || Date.parse(lastStatus.timestamp) > Date.parse(status.timestamp))) status = lastStatus;
  if (status && !rows.some((row) => row.timestamp === status.timestamp && row.status === status.status)) rows.push(status);
  const start = Math.max(0, rows.length - PROGRESS_TRAIL_ROWS);
  const recent = rows.slice(start);
  // Keep the latest decision before a busy tail so redraw/failure semantics survive
  // ordinary progress chatter. The byte limit remains the hard bound on log history.
  const isDecision = (row) => row.render_started || row.render_completed
    || row.awaiting_confirmation || row.redone_shots?.length
    || ['error', 'rendered', 'dependency_missing'].includes(row.status);
  if (start && !recent.some(isDecision)) {
    const decision = rows.slice(0, start).findLast(isDecision);
    if (decision) recent[0] = decision;
  }
  const trail = recent.map((row) => JSON.stringify(row)).join('\n');
  const timestamps = [statusFile.mtimeMs, trailFile.mtimeMs,
    Date.parse(String(status?.timestamp || '')), ...recent.map((row) => Date.parse(String(row.timestamp || '')))]
    .filter((value) => Number.isFinite(value) && value > 0);
  const lastProgressAt = timestamps.length ? new Date(Math.max(...timestamps)).toISOString() : '';
  // This revision is an invalidation hint, never an approval checksum. No media is
  // opened here; periodic snapshots discover writes that do not touch these markers.
  const revision = createHash('sha256').update(JSON.stringify([
    resolved, record.updated_at, record.stage, statusFile.revision, trailFile.revision,
    watched, status, trail,
  ])).digest('hex');
  return {status, trail, revision, lastProgressAt};
}

function progressObject(text) {
  try {
    const value = JSON.parse(text);
    return isPlainObject(value) ? value : null;
  } catch {
    return null;
  }
}

function fileRevision(info) {
  return info ? `${info.ino}:${info.size}:${info.mtimeMs}:${info.ctimeMs}` : '';
}

async function readProgressFile(filePath, tail = false) {
  let handle;
  try {
    handle = await open(filePath, 'r');
    const info = await handle.stat();
    const metadata = {revision: fileRevision(info), mtimeMs: info.mtimeMs};
    if (!info.isFile() || (!tail && info.size > PROGRESS_READ_BYTES)) return {...metadata, text: ''};
    const offset = tail ? Math.max(0, info.size - PROGRESS_READ_BYTES) : 0;
    const buffer = Buffer.alloc(Math.min(info.size, PROGRESS_READ_BYTES));
    let length = 0;
    while (length < buffer.length) {
      const {bytesRead} = await handle.read(buffer, length, buffer.length - length, offset + length);
      if (!bytesRead) break;
      length += bytesRead;
    }
    let text = buffer.toString('utf8', 0, length);
    if (tail) {
      // Discard a possible partial first row and any last row still being appended.
      if (offset) text = text.slice(text.indexOf('\n') + 1);
      text = text.slice(0, text.lastIndexOf('\n') + 1);
    }
    return {...metadata, text};
  } catch {
    return {text: '', revision: '', mtimeMs: 0};
  } finally {
    await handle?.close();
  }
}

/**
 * Accept, reject or release one slot of a render root and return the refreshed
 * payload. Accepting recomputes the size and sha256 from the files on disk at that
 * moment; rejecting with a reason replaces any acceptance for the slot; releasing
 * with no reason deletes the entry rather than storing it as empty.
 */
export async function updateRenderAcceptance(repoRoot, input = {}) {
  const sessionId = typeof input?.sessionId === 'string' ? input.sessionId : '';
  const record = sessionRecord(await readSessionsPayload(repoRoot), sessionId);
  if (!record) return null;
  const workingDir = sessionWorkingDir(repoRoot, sessionId, record);
  return withProjectWriteLock(workingDir, () => updateRenderAcceptanceUnlocked(repoRoot, input));
}

async function updateRenderAcceptanceUnlocked(repoRoot, input = {}) {
  const body = input && typeof input === 'object' ? input : {};
  const sessionId = typeof body.sessionId === 'string' ? body.sessionId : '';
  const record = sessionRecord(await readSessionsPayload(repoRoot), sessionId);
  if (!record) return null;

  if (typeof body.accepted !== 'boolean') throw acceptanceError('accepted must be a boolean');
  const stage = String(body.stage ?? '');
  const scope = STAGE_SCOPES[stage];
  if (!scope) throw acceptanceError(`Unknown stage: ${stage || '(missing)'}`);
  const reason = typeof body.reason === 'string' ? body.reason.trim() : '';

  const workingDir = sessionWorkingDir(repoRoot, sessionId, record);
  const root = await resolveRenderRoot(repoRoot, workingDir, body.root);
  const discovery = await collectAcceptanceState(workingDir, root);

  let shot = null;
  let paths;
  if (scope === 'session') {
    paths = stage === 'portraits' ? discovery.portraits : discovery.finalVideo;
  } else {
    shot = body.shot === undefined || body.shot === null ? '' : String(body.shot);
    const found = discovery.shots.find((entry) => entry.shot === shot);
    if (!found) throw acceptanceError(`Unknown shot: ${shot || '(missing)'}`);
    paths = found[stage];
  }

  const store = await readAcceptanceFile(workingDir);
  const slot = scope === 'session' ? null : shot;
  if (body.accepted) {
    if (!paths.length) throw acceptanceError(`Cannot accept ${stage}${shot === null ? '' : ` for shot ${shot}`}: no artifacts on disk`);
    recordSlot(store, root, slot, stage, {
      accepted_at: isoSeconds(),
      artifacts: await acceptanceArtifacts(workingDir, paths),
    });
  } else if (reason) {
    if (reason.length > REASON_MAX_LENGTH) throw acceptanceError(`reason must be at most ${REASON_MAX_LENGTH} characters`);
    // A rejection overwrites the slot's record, so it can never carry both an
    // acceptance and a rejection.
    recordSlot(store, root, slot, stage, {rejected_at: isoSeconds(), reason});
  } else {
    removeAcceptance(store, root, slot, stage);
  }
  await writeAcceptanceFile(workingDir, store);
  return acceptancePayload(repoRoot, workingDir, root);
}

async function acceptancePayload(repoRoot, workingDir, root, context = createFilmReadContext(), discovery = null) {
  const store = await readAcceptanceFile(workingDir, context);
  const rootStore = isPlainObject(store[root]) ? store[root] : {};
  const shotStore = isPlainObject(rootStore.shots) ? rootStore.shots : {};
  discovery ??= await collectAcceptanceState(workingDir, root, context);

  const stages = [];
  for (const stage of STAGE_ORDER) {
    const scope = STAGE_SCOPES[stage];
    let slots;
    if (scope === 'session') {
      const paths = stage === 'portraits' ? discovery.portraits : discovery.finalVideo;
      const slotRecord = rootStore[stage];
      slots = [slotPayload(null, await slotState(workingDir, slotRecord, paths, context), paths, slotRecord)];
    } else {
      slots = [];
      for (const shot of discovery.shots) {
        const paths = shot[stage];
        const slotRecord = isPlainObject(shotStore[shot.shot]) ? shotStore[shot.shot][stage] : undefined;
        slots.push(slotPayload(shot.shot, await slotState(workingDir, slotRecord, paths, context), paths, slotRecord));
      }
    }
    const accepted = slots.length > 0 && slots.every((slot) => slot.state === 'accepted');
    stages.push({stage, scope, state: stageState(slots), accepted, slots});
  }

  const keyframeSlots = stages.find((stage) => stage.stage === 'keyframes').slots;
  const clipSlots = stages.find((stage) => stage.stage === 'clips').slots;
  const keyframes = keyframeSlots.reduce((count, slot) => count + slot.artifacts.length, 0);
  const acceptedKeyframes = keyframeSlots
    .filter((slot) => slot.state === 'accepted')
    .reduce((count, slot) => count + slot.artifacts.length, 0);
  const clips = clipSlots.reduce((count, slot) => count + slot.artifacts.length, 0);
  const rejected = stages.reduce((count, entry) => count + entry.slots.filter((slot) => slot.state === 'rejected').length, 0);
  // One clip's length and its per-second price, not a running total: the timeline
  // multiplies them by the clips that do not exist yet.
  const clipSettings = await filmClipSettings(repoRoot, workingDir, context);
  return {
    root,
    stages,
    totals: {
      acceptedKeyframes,
      keyframes,
      clips,
      rejected,
      clipSeconds: clipSettings.seconds,
      clipCostUsd: clipUsdPerSecond(clipSettings.model, clipSettings.resolution),
      requiredFrameTypes: clipSettings.model === 'heygen/heygen-video-1' ? ['first_frame'] : ['first_frame', 'last_frame'],
    },
  };
}

function filmClipSettings(repoRoot, workingDir, context) {
  context.clipSettings ??= readRenderManifest(repoRoot, workingDir, context)
    .then((manifest) => readClipSettings(repoRoot, String(manifest?.video_model || '')));
  return context.clipSettings;
}

/** The payload slot: its state, plus the note against it, whether or not it was redrawn. */
function slotPayload(shot, state, artifacts, record) {
  const slot = {shot, state, artifacts};
  if (isPlainObject(record) && typeof record.invalidated_at === 'string' && record.invalidated_at !== '') {
    slot.invalidatedAt = record.invalidated_at;
    if (typeof record.invalidation_reason === 'string' && record.invalidation_reason) {
      slot.invalidationReason = record.invalidation_reason;
    }
  }
  if (state === 'rejected') {
    slot.rejectedAt = String(record.rejected_at);
    if (typeof record.reason === 'string') slot.reason = record.reason;
  } else if (isPlainObject(record) && typeof record.redone_at === 'string' && record.redone_at !== '') {
    // Redrawn in response to a note: the note stays attached to what replaced it.
    slot.redoneAt = record.redone_at;
    if (typeof record.reason === 'string' && record.reason) slot.note = record.reason;
  }
  return slot;
}

function stageState(slots) {
  if (!slots.length) return 'planned';
  if (slots.every((slot) => slot.state === 'accepted')) return 'accepted';
  if (slots.some((slot) => slot.state === 'rejected')) return 'rejected';
  if (slots.some((slot) => slot.state === 'stale')) return 'stale';
  if (slots.some((slot) => slot.state === 'rendered')) return 'rendered';
  return 'planned';
}

/** planned (nothing on disk) | rendered (artifacts, no lock) | accepted | rejected | stale. */
async function slotState(workingDir, record, paths, context) {
  if (isPlainObject(record) && typeof record.invalidated_at === 'string' && record.invalidated_at !== '') return 'stale';
  // A stored rejection outranks every other reading unless a later continuity change
  // explicitly invalidated the record.
  if (isRejected(record)) return 'rejected';
  if (!paths.length) return 'planned';
  if (!isPlainObject(record) || typeof record.accepted_at !== 'string' || record.accepted_at === '') return 'rendered';
  return (await acceptanceLockMatches(workingDir, record, context)) ? 'accepted' : 'stale';
}

/** A slot record carrying a rejection note rather than an acceptance. */
function isRejected(record) {
  return isPlainObject(record) && typeof record.rejected_at === 'string' && record.rejected_at !== '';
}

async function acceptanceLockMatches(workingDir, record, context) {
  if (!Array.isArray(record.artifacts) || record.artifacts.length === 0) return false;
  for (const artifact of record.artifacts) {
    if (!isPlainObject(artifact)) return false;
    const absolute = path.resolve(workingDir, String(artifact.path ?? ''));
    if (!absolute.startsWith(`${workingDir}${path.sep}`)) return false;
    let info;
    try {
      info = await context.stat(absolute);
    } catch {
      return false;
    }
    if (!info.isFile() || info.size !== artifact.size) return false;
    if (await context.hash(absolute) !== artifact.sha256) return false;
  }
  return true;
}

/** Mirror of `utils.text.safe_path_component`, so a model names the same directory on both sides. */
function safePathComponent(name) {
  const cleaned = String(name ?? '').replace(/[^\w\-. ]/g, '_').replace(/^\.+/, '').trim();
  return cleaned || 'unnamed';
}

/** The artifacts a stage shows are the ones the render in use produced.
 *
 * A model change leaves the previous model's artifacts where they were, so a slot that has
 * been through two video models holds two clips and both were counted against it — while
 * the film only ever holds the one under the model the manifest pins. When the pin names a
 * directory this prefers it; without one (or before it has anything) everything counts, so
 * a render that has not written its manifest yet still shows what is on disk.
 */
function currentModelArtifacts(paths, modelDir) {
  if (!modelDir) return paths;
  const marker = `/${modelDir}/`;
  const matching = paths.filter((item) => item.includes(marker));
  return matching.length > 0 ? matching : paths;
}

/** Every reviewable slot of a root, derived from the paths on disk. */
async function collectAcceptanceState(workingDir, root, context = createFilmReadContext()) {
  const rootDir = path.join(workingDir, root);
  let manifest = null;
  try {
    manifest = await context.json(path.join(workingDir, 'render_manifest.json'));
  } catch {
    manifest = null;
  }
  const imageModelDir = safePathComponent(manifest?.image_model ?? '');
  const videoModelDir = safePathComponent(manifest?.video_model ?? '');

  const portraits = [];
  await collectAcceptanceFiles(workingDir, path.join(rootDir, 'character_portraits'), (name) => name.toLowerCase().endsWith('.png'), portraits, context);
  portraits.sort();
  portraits.splice(0, portraits.length, ...currentModelArtifacts(portraits, imageModelDir));

  const finalVideo = [];
  const finalPath = path.join(rootDir, 'final_video.mp4');
  if (await isRegularFile(finalPath, context)) finalVideo.push(relativeArtifactPath(workingDir, finalPath));

  const shots = [];
  // Script mode keeps shots flat under `shots/`, so the slot is the directory name.
  // Idea mode nests them under `scene_<idx>/shots/`, where the same shot index appears
  // once per scene, so the slot is scene-qualified. Same keys, same rule, as the gate.
  const containers = await shotContainers(rootDir, context);
  for (const {container, prefix} of containers) {
    const entries = await context.entries(container).catch(() => []);
    for (const entry of entries) {
      if (!entry.isDirectory() || entry.name.startsWith('.')) continue;
      const shotDir = path.join(container, entry.name);
      // A shot is planned only once its description exists, exactly like the render gate.
      if (!(await isRegularFile(path.join(shotDir, 'shot_description.json'), context))) continue;
      const media = [];
      await collectAcceptanceFiles(workingDir, shotDir, (name) => KEYFRAME_NAMES.has(name) || name === 'video.mp4', media, context);
      const keyframes = media.filter((file) => KEYFRAME_NAMES.has(path.basename(file)));
      const clips = media.filter((file) => path.basename(file) === 'video.mp4');
      shots.push({
        shot: `${prefix}${entry.name}`,
        keyframes: currentModelArtifacts(keyframes, imageModelDir).sort(),
        clips: currentModelArtifacts(clips, videoModelDir).sort(),
      });
    }
  }
  const playing = await filmOrder(workingDir, root, context);
  const rank = new Map(playing.map((slot, index) => [slot, index]));
  shots.sort((left, right) => {
    const leftOrder = rank.get(left.shot);
    const rightOrder = rank.get(right.shot);
    if (leftOrder !== undefined || rightOrder !== undefined) {
      if (leftOrder === undefined) return 1;
      if (rightOrder === undefined) return -1;
      return leftOrder - rightOrder;
    }
    return compareShotNames(left.shot, right.shot);
  });

  return {portraits, finalVideo, shots, order: playing};
}

/** Where a render root keeps its shots: flat, or one directory per scene. */
async function shotContainers(rootDir, context = createFilmReadContext()) {
  const flat = path.join(rootDir, 'shots');
  const entries = await context.entries(rootDir).catch(() => []);
  if (entries.some((entry) => entry.isDirectory() && entry.name === 'shots')) {
    return [{container: flat, prefix: ''}];
  }
  return entries
    .filter((entry) => entry.isDirectory() && /^scene_\d+$/.test(entry.name))
    .sort((left, right) => Number(left.name.slice('scene_'.length)) - Number(right.name.slice('scene_'.length)))
    .map((entry) => ({container: path.join(rootDir, entry.name, 'shots'), prefix: `${entry.name}/`}));
}

async function collectAcceptanceFiles(base, directory, isArtifact, artifacts, context = createFilmReadContext()) {
  let entries;
  try {
    entries = await context.entries(directory);
  } catch {
    return;
  }
  for (const entry of entries) {
    if (entry.name.startsWith('.')) continue;
    const absolute = path.join(directory, entry.name);
    // `cache/` is moviepy scratch debris; whatever it holds is never a reviewable artifact.
    if (entry.isDirectory()) {
      if (entry.name !== 'cache') await collectAcceptanceFiles(base, absolute, isArtifact, artifacts, context);
      continue;
    }
    // `withFileTypes` reports symlinks as neither directory nor file, so a link out is never followed.
    if (entry.isFile() && isArtifact(entry.name)) artifacts.push(relativeArtifactPath(base, absolute));
  }
}

async function acceptanceArtifacts(workingDir, paths) {
  const artifacts = [];
  for (const relativePath of [...paths].sort()) {
    const absolute = path.resolve(workingDir, relativePath);
    const info = await stat(absolute);
    artifacts.push({path: relativePath, size: info.size, sha256: await sha256File(absolute)});
  }
  return artifacts;
}

/** Stores a slot's record (an acceptance or a rejection), replacing whatever was there. */
function recordSlot(store, root, shot, stage, entry) {
  const rootStore = isPlainObject(store[root]) ? store[root] : {};
  if (shot === null) {
    rootStore[stage] = entry;
  } else {
    const shotStore = isPlainObject(rootStore.shots) ? rootStore.shots : {};
    shotStore[shot] = {...(isPlainObject(shotStore[shot]) ? shotStore[shot] : {}), [stage]: entry};
    rootStore.shots = shotStore;
  }
  store[root] = rootStore;
}

function removeAcceptance(store, root, shot, stage) {
  const rootStore = store[root];
  if (!isPlainObject(rootStore)) return;
  if (shot === null) {
    delete rootStore[stage];
  } else {
    const shotStore = isPlainObject(rootStore.shots) ? rootStore.shots : {};
    if (isPlainObject(shotStore[shot])) {
      delete shotStore[shot][stage];
      if (!Object.keys(shotStore[shot]).length) delete shotStore[shot];
    }
    if (!Object.keys(shotStore).length) delete rootStore.shots;
  }
  if (!Object.keys(rootStore).length) delete store[root];
}

async function readAcceptanceFile(workingDir, context = createFilmReadContext()) {
  try {
    const payload = await context.json(path.join(workingDir, ACCEPTANCE_FILENAME));
    return isPlainObject(payload) ? payload : {};
  } catch {
    return {};
  }
}

async function writeAcceptanceFile(workingDir, store) {
  const filePath = path.join(workingDir, ACCEPTANCE_FILENAME);
  await writeAtomic(filePath, `${JSON.stringify(store, null, 2)}\n`, 0o600);
}

async function resolveRenderRoot(repoRoot, workingDir, requested, context = createFilmReadContext()) {
  const value = String(requested ?? '').trim();
  if (value) {
    if (!RENDER_ROOTS.has(value)) throw acceptanceError(`Unknown render root: ${value}`);
    return value;
  }
  const manifest = await readRenderManifest(repoRoot, workingDir, context);
  const mode = String(manifest?.render_mode ?? '').trim();
  return RENDER_ROOTS.has(mode) ? mode : 'script2video';
}

function sessionWorkingDir(repoRoot, sessionId, record) {
  const workingDir = String(record?.working_dir ?? '');
  return workingDir ? path.resolve(repoRoot, workingDir) : resolveSessionRoot(repoRoot, sessionId);
}

async function isRegularFile(filePath, context = createFilmReadContext()) {
  try {
    return (await context.stat(filePath)).isFile();
  } catch {
    return false;
  }
}

function isPlainObject(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function acceptanceError(message) {
  const error = new Error(message);
  error.statusCode = 400;
  return error;
}

function conflictError(message) {
  const error = acceptanceError(message);
  error.statusCode = 409;
  return error;
}

/** Timeline order: scene-qualified idea slots first by scene, then by shot number. */
function compareShotNames(left, right) {
  const [leftScene, leftShot] = splitShotKey(left);
  const [rightScene, rightShot] = splitShotKey(right);
  if (leftScene !== rightScene) return leftScene - rightScene;
  if (leftShot !== rightShot) return leftShot - rightShot;
  return left.localeCompare(right);
}

/** A slot key as (scene index, shot index); a flat script slot is scene 0. */
function splitShotKey(key) {
  const match = /^scene_(\d+)\/(\d+)$/.exec(String(key));
  return match ? [Number(match[1]), Number(match[2])] : [0, Number(key)];
}

function sha256File(filePath) {
  return new Promise((resolve, reject) => {
    const hash = createHash('sha256');
    const stream = createReadStream(filePath);
    stream.on('data', (chunk) => hash.update(chunk));
    stream.on('error', reject);
    stream.on('end', () => resolve(hash.digest('hex')));
  });
}

// Matches the Python gate's `datetime.now().isoformat(timespec="seconds")`.
function isoSeconds() {
  const now = new Date();
  const pad = (value) => String(value).padStart(2, '0');
  return `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}T${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}`;
}

export function artifactContentType(filePath) {
  const extension = path.extname(filePath).toLowerCase();
  return {
    '.html': 'text/html; charset=utf-8',
    '.js': 'text/javascript; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.svg': 'image/svg+xml',
    '.woff': 'font/woff',
    '.woff2': 'font/woff2',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.webp': 'image/webp',
    '.gif': 'image/gif',
    '.mp4': 'video/mp4',
    '.webm': 'video/webm',
    '.mov': 'video/quicktime',
    '.json': 'application/json; charset=utf-8',
    '.txt': 'text/plain; charset=utf-8',
    '.md': 'text/markdown; charset=utf-8',
  }[extension] ?? 'application/octet-stream';
}

function resolveSessionRoot(repoRoot, sessionId) {
  assertSessionId(sessionId);
  const workingRoot = path.resolve(repoRoot, '.working_dir');
  const candidate = path.resolve(workingRoot, sessionId);
  if (!candidate.startsWith(`${workingRoot}${path.sep}`)) {
    throw new Error('Session path escapes .working_dir');
  }
  return candidate;
}

function assertSessionId(sessionId) {
  if (!/^[A-Za-z0-9][A-Za-z0-9-]{0,95}$/.test(String(sessionId || ''))) {
    throw new Error('Invalid session id');
  }
}

function validateUploadName(fileName) {
  const value = String(fileName || '').normalize('NFC').trim();
  if (!value || value === '.' || value === '..') throw new Error('A valid filename is required');
  if (value.length > 180) throw new Error('Filename must be 180 characters or fewer');
  if (/[\\/\u0000-\u001f\u007f]/.test(value)) throw new Error('Filename contains unsupported characters');
  return value;
}

async function removeSessionLogRecords(repoRoot, sessionId) {
  const logsRoot = path.join(repoRoot, '.vimax', 'logs');
  let entries;
  try {
    entries = await readdir(logsRoot, {withFileTypes: true});
  } catch {
    return;
  }
  for (const entry of entries) {
    if (!entry.isFile() || path.extname(entry.name) !== '.jsonl') continue;
    const logPath = path.join(logsRoot, entry.name);
    const lines = (await readFile(logPath, 'utf8')).split(/\r?\n/).filter(Boolean);
    const retained = lines.filter((line) => {
      try {
        const record = JSON.parse(line);
        const recordSessionId = record.session_id
          ?? record.sessionId
          ?? record.session?.session_id
          ?? record.context?.session_id
          ?? record.metadata?.session_id;
        return recordSessionId !== sessionId;
      } catch {
        return true;
      }
    });
    if (retained.length === lines.length) continue;
    const temporaryPath = `${logPath}.${process.pid}.tmp`;
    await writeFile(temporaryPath, retained.length ? `${retained.join('\n')}\n` : '', {mode: 0o600});
    await rename(temporaryPath, logPath);
  }
}

function sanitizeSession(record) {
  return {
    sessionId: String(record.session_id ?? ''),
    projectName: String(record.project_name ?? ''),
    workingDir: String(record.working_dir ?? ''),
    stage: String(record.stage ?? 'created'),
    summary: String(record.summary ?? ''),
    idea: String(record.idea ?? ''),
    updatedAt: String(record.updated_at ?? record.created_at ?? ''),
    createdAt: String(record.created_at ?? ''),
    compactionTurns: Number(record.compacted_turns ?? 0),
  };
}

// ---------------------------------------------------------------------------
// Per-shot plan: what a shot's frames show, and the prompt each was drawn from.
//
// The plan is what the render follows: the frame descriptions are what the image model is
// asked for, and the visible-character lists decide whose portraits the reference selector
// is offered. A Timeline that cannot show or change them leaves the user able to re-run a
// shot but not to say what it should have been, which is how a shot gets redrawn five times
// with the same wrong answer.
async function withTimelineMutation(repoRoot, input, operation) {
  const sessionId = typeof input?.sessionId === 'string' ? input.sessionId : '';
  const record = sessionRecord(await readSessionsPayload(repoRoot), sessionId);
  if (!record) return null;
  return withProjectWriteLock(sessionWorkingDir(repoRoot, sessionId, record), operation);
}

function shotLocation(workingDir, root, slot) {
  const key = String(slot ?? '');
  const match = /^(scene_\d+)\/(\d+)$/.exec(key);
  if (!match && !/^\d+$/.test(key)) throw acceptanceError(`Unknown shot: ${key || '(missing)'}`);
  const container = path.join(workingDir, root, match ? match[1] : '');
  const local = match ? match[2] : key;
  return {
    key,
    local,
    scene: match ? match[1] : '',
    container,
    shotDir: path.join(container, 'shots', local),
    treePath: path.join(container, CAMERA_TREE_FILENAME),
    storyboardPath: path.join(container, STORYBOARD_FILENAME),
  };
}

const SHOT_DESCRIPTION_FILENAME = 'shot_description.json';
const STORYBOARD_FILENAME = 'storyboard.json';
const CAMERA_TREE_FILENAME = 'camera_tree.json';
const REMOVED_SHOTS_DIR = '.removed_shots';
const MAX_BRIEF_LENGTH = 1200;
const MAX_DESCRIPTION_LENGTH = 4000;
// Written by the agent's vimax_review_timeline tool; the web layer only reads it.
const CONTINUITY_FILENAME = 'continuity_review.json';

async function withMutationTransaction(operation) {
  const undo = [];
  const backups = [];
  const transaction = {
    async write(filePath, contents) {
      let previous = null;
      let previousMode;
      let previousTimes;
      try {
        const info = await lstat(filePath);
        if (!info.isFile()) throw acceptanceError(`Cannot replace non-file metadata: ${path.basename(filePath)}`);
        previous = await readFile(filePath);
        previousMode = info.mode & 0o777;
        previousTimes = {atime: info.atime, mtime: info.mtime};
      } catch (error) {
        if (error.code !== 'ENOENT') throw error;
      }
      await writeAtomic(filePath, contents);
      undo.push(async () => {
        if (previous === null) return rm(filePath, {force: true});
        await writeAtomic(filePath, previous, previousMode);
        await utimes(filePath, previousTimes.atime, previousTimes.mtime);
      });
    },
    async mkdir(directory) {
      await mkdir(directory);
      undo.push(() => rm(directory, {recursive: true, force: true}));
    },
    async move(from, to) {
      const parent = path.dirname(to);
      let parentCreated = false;
      try { await lstat(parent); }
      catch (error) {
        if (error.code !== 'ENOENT') throw error;
        await mkdir(parent, {recursive: true});
        parentCreated = true;
      }
      if (parentCreated) undo.push(() => rm(parent, {recursive: true, force: true}));
      try {
        await lstat(to);
        throw acceptanceError(`Destination already exists: ${path.basename(to)}`);
      } catch (error) {
        if (error.code !== 'ENOENT') throw error;
      }
      await rename(from, to);
      undo.push(() => rename(to, from));
    },
    async remove(filePath) {
      try { await lstat(filePath); } catch (error) { if (error.code === 'ENOENT') return; throw error; }
      const backup = `${filePath}.timeline-${randomUUID()}.bak`;
      await rename(filePath, backup);
      undo.push(() => rename(backup, filePath));
      backups.push(backup);
    },
  };

  try {
    const result = await operation(transaction);
    await Promise.all(backups.map((backup) => rm(backup, {recursive: true, force: true}).catch(() => {})));
    return result;
  } catch (cause) {
    const failures = [];
    for (const rollback of undo.reverse()) {
      try { await rollback(); } catch (error) { failures.push(error.message); }
    }
    if (cause.statusCode === undefined) cause.statusCode = 500;
    if (failures.length) {
      const rollbackError = new Error(`Timeline mutation failed: ${cause.message}; rollback was incomplete: ${failures.join('; ')}`, {cause});
      rollbackError.statusCode = 500;
      throw rollbackError;
    }
    throw cause;
  }
}

async function writeAtomic(filePath, contents, mode) {
  const directory = path.dirname(filePath);
  await mkdir(directory, {recursive: true});
  const temporary = path.join(directory, `.${path.basename(filePath)}.${process.pid}.${randomUUID()}.tmp`);
  try {
    await writeFile(temporary, contents, {flag: 'wx', ...(mode === undefined ? {} : {mode})});
    await rename(temporary, filePath);
  } catch (error) {
    await rm(temporary, {force: true}).catch(() => {});
    throw error;
  }
}

/**
 * The film's slots in playing order: camera by camera, in each camera's own order.
 *
 * A shot's number is its identity, not its place, so the shot directories on disk are not
 * the order the film plays in — the camera tree is. Anything comparing the timeline against
 * a review has to use this, or a review that is current reads as out of date.
 */
async function filmOrder(workingDir, root, context = createFilmReadContext()) {
  const rootDir = path.join(workingDir, root);
  const entries = await context.entries(rootDir).catch(() => []);
  const scenes = entries.filter((entry) => entry.isDirectory() && /^scene_\d+$/.test(entry.name))
    .sort((left, right) => Number(left.name.slice(6)) - Number(right.name.slice(6)));
  const containers = scenes.length ? scenes.map((entry) => ({name: entry.name, dir: path.join(rootDir, entry.name)}))
    : [{name: '', dir: rootDir}];
  const order = [];
  for (const {name, dir} of containers) {
    let cameras;
    try { cameras = await context.json(path.join(dir, CAMERA_TREE_FILENAME)); } catch { continue; }
    if (!Array.isArray(cameras)) continue;
    order.push(...cameras.slice().sort((left, right) => Number(left.idx) - Number(right.idx))
      .flatMap((camera) => (camera.active_shot_idxs || []).map((shot) => `${name ? `${name}/` : ''}${shot}`)));
  }
  return order;
}

/**
 * The script-coverage review of a root, and whether it still describes this timeline.
 *
 * Staleness is derived here rather than trusted: a review that lists different shots, or
 * that predates an edit to the plan, is not describing the film any more and says so.
 */
export async function readContinuityReview(repoRoot, sessionId, root = '') {
  const record = sessionRecord(await readSessionsPayload(repoRoot), sessionId);
  if (!record) return null;
  const workingDir = sessionWorkingDir(repoRoot, sessionId, record);
  const context = createFilmReadContext();
  const resolved = await resolveRenderRoot(repoRoot, workingDir, root, context);
  return continuityPayload(repoRoot, workingDir, resolved, record, context);
}

async function continuityPayload(repoRoot, workingDir, root, record, context, discovery = null) {
  let review = null;
  try {
    const payload = await context.json(path.join(workingDir, CONTINUITY_FILENAME));
    review = isPlainObject(payload) ? payload : null;
  } catch {
    review = null;
  }
  const playing = discovery?.order ?? await filmOrder(workingDir, root, context);
  const shots = playing.length ? playing
    : (discovery ?? await collectAcceptanceState(workingDir, root, context)).shots.map((entry) => String(entry.shot));
  const {stale, reason} = await reviewStaleness(workingDir, root, review, shots, String(record.user_requirement ?? ''), context);
  const clipSettings = await filmClipSettings(repoRoot, workingDir, context);
  return {
    root,
    review,
    stale,
    staleReason: reason,
    shots,
    clipSeconds: clipSettings.seconds,
  };
}

async function reviewStaleness(workingDir, root, review, shots, userRequirement, context) {
  if (!review) return {stale: true, reason: 'No review yet.'};
  if (review.root !== undefined && review.root !== root) return {stale: true, reason: 'The reviewed render root no longer matches.'};
  const reviewed = (Array.isArray(review.shots_reviewed) ? review.shots_reviewed : []).map(String);
  const added = shots.filter((shot) => !reviewed.includes(shot));
  const removed = reviewed.filter((shot) => !shots.includes(shot));
  if (added.length || removed.length) {
    const parts = [];
    if (added.length) parts.push(`shots ${added.join(', ')} added`);
    if (removed.length) parts.push(`shots ${removed.join(', ')} removed`);
    return {stale: true, reason: `The timeline changed since this review: ${parts.join(', ')}.`};
  }

  if (Object.hasOwn(review, 'input_files')) {
    if (review.root !== root) return {stale: true, reason: 'The review does not identify this render root.'};
    if (typeof review.user_requirement !== 'string' || review.user_requirement !== userRequirement) {
      return {stale: true, reason: 'The user requirement changed since this review.'};
    }
    if (!isPlainObject(review.input_files)) return {stale: true, reason: 'The review input fingerprints are invalid.'};
    const expected = await continuityInputPaths(workingDir, root, shots, context);
    const missing = [...expected].filter((relative) => !Object.hasOwn(review.input_files, relative));
    if (missing.length) return {stale: true, reason: `The review does not cover current inputs: ${missing.join(', ')}.`};
    const changed = [];
    for (const [relative, expectedHash] of Object.entries(review.input_files)) {
      if (expectedHash !== null && (typeof expectedHash !== 'string' || !/^[a-f\d]{64}$/i.test(expectedHash))) {
        return {stale: true, reason: `The review fingerprint for ${relative} is invalid.`};
      }
      const absolute = path.resolve(workingDir, relative);
      if (absolute === workingDir || !absolute.startsWith(`${workingDir}${path.sep}`)) {
        return {stale: true, reason: `The review input path ${relative} is invalid.`};
      }
      let current = null;
      try {
        const info = await context.lstat(absolute);
        if (!info.isFile()) { changed.push(relative); continue; }
        current = await context.hash(absolute);
      } catch (error) {
        if (error.code !== 'ENOENT') { changed.push(relative); continue; }
      }
      if (current !== expectedHash) changed.push(relative);
    }
    if (changed.length) return {stale: true, reason: `Inputs changed since this review: ${changed.join(', ')}.`};
    return {stale: false, reason: ''};
  }

  // Older reviews have no fingerprints. Keep their source-mtime contract, but watch the
  // actual root and scene inputs rather than guessing paths from a qualified shot identity.
  const reviewedAt = Date.parse(String(review.reviewed_at ?? ''));
  if (!Number.isFinite(reviewedAt)) return {stale: true, reason: 'The review has no timestamp.'};
  const watched = await continuityInputPaths(workingDir, root, shots, context);
  const newer = [];
  for (const relative of watched) {
    const info = await context.stat(path.join(workingDir, relative)).catch(() => null);
    if (info && info.mtimeMs > reviewedAt) newer.push(relative.slice(root.length + 1));
  }
  if (newer.length) return {stale: true, reason: `Edited since this review: ${newer.join(', ')}.`};
  return {stale: false, reason: ''};
}

async function continuityInputPaths(workingDir, root, shots, context) {
  const paths = new Set();
  const add = (absolute) => paths.add(path.relative(workingDir, absolute).split(path.sep).join('/'));
  const rootDir = path.join(workingDir, root);
  const scriptCandidates = root === 'script2video'
    ? ['script.txt']
    : root === 'idea2video'
      ? ['script.json', 'story.txt']
      : ['novel/novel_compressed.txt'];
  for (const candidate of scriptCandidates) add(path.join(rootDir, candidate));

  const containers = await shotContainers(rootDir, context);
  if (!containers.length) containers.push({container: path.join(rootDir, 'shots'), prefix: ''});
  for (const {container: shotsDir} of containers) {
    const sceneDir = path.dirname(shotsDir);
    for (const name of ['characters.json', CAMERA_TREE_FILENAME, STORYBOARD_FILENAME]) add(path.join(sceneDir, name));
    for (const shot of shots) {
      const location = shotLocation(workingDir, root, shot);
      if (location.container === sceneDir) add(path.join(location.shotDir, SHOT_DESCRIPTION_FILENAME));
    }
    await collectJsonInputs(shotsDir, add, context);
    const removedDir = path.join(sceneDir, REMOVED_SHOTS_DIR);
    const removed = await context.entries(removedDir).catch(() => []);
    for (const entry of removed) {
      if (!entry.isDirectory() || !/^\d+$/.test(entry.name)) continue;
      const kept = path.join(removedDir, entry.name);
      for (const name of ['placement.json', 'brief.json', SHOT_DESCRIPTION_FILENAME]) add(path.join(kept, name));
      await collectJsonInputs(kept, add, context);
    }
  }
  return paths;
}

async function collectJsonInputs(directory, add, context) {
  const entries = await context.entries(directory).catch(() => []);
  for (const entry of entries) {
    if (entry.name.startsWith('.')) continue;
    const absolute = path.join(directory, entry.name);
    if (entry.isDirectory()) await collectJsonInputs(absolute, add, context);
    else if (entry.isFile() && (isShotInput(entry.name) || entry.name === 'brief.json' || entry.name === 'placement.json')) add(absolute);
  }
}

/** Every shot's plan in one request: the reel shows all of them, not only the open ones. */
export async function readShotPlans(repoRoot, sessionId, root = '') {
  const record = sessionRecord(await readSessionsPayload(repoRoot), sessionId);
  if (!record) return null;
  const workingDir = sessionWorkingDir(repoRoot, sessionId, record);
  const context = createFilmReadContext();
  const resolved = await resolveRenderRoot(repoRoot, workingDir, root, context);
  const discovery = await collectAcceptanceState(workingDir, resolved, context);
  return {root: resolved, plans: await shotPlansPayload(workingDir, resolved, discovery, context)};
}

async function shotPlansPayload(workingDir, root, discovery, context) {
  const plans = [];
  for (const {shot} of discovery.shots) {
    try { plans.push(await shotPlanPayload(workingDir, root, shot, context)); }
    catch { /* A shot with no readable plan is simply not part of what can be shown. */ }
  }
  return plans;
}

/** The shots taken out of the film, with what is kept for them. */
export async function readRemovedShots(repoRoot, sessionId, root = '') {
  const record = sessionRecord(await readSessionsPayload(repoRoot), sessionId);
  if (!record) return null;
  const workingDir = sessionWorkingDir(repoRoot, sessionId, record);
  const resolved = await resolveRenderRoot(repoRoot, workingDir, root);
  return {root: resolved, removed: await listRemoved(workingDir, resolved)};
}

async function isDirectory(target) {
  try {
    return (await stat(target)).isDirectory();
  } catch {
    return false;
  }
}

async function listRemoved(workingDir, root, context = createFilmReadContext()) {
  const rootDir = path.join(workingDir, root);
  const containers = await shotContainers(rootDir, context);
  if (!containers.length) containers.push({container: path.join(rootDir, 'shots'), prefix: ''});
  const removed = [];
  for (const {container: shotsDir, prefix} of containers) {
    const sceneDir = path.dirname(shotsDir);
    const removedDir = path.join(sceneDir, REMOVED_SHOTS_DIR);
    const entries = await context.entries(removedDir).catch(() => []);
    for (const entry of entries) {
      if (!entry.isDirectory() || !/^\d+$/.test(entry.name)) continue;
      const kept = path.join(removedDir, entry.name);
      const files = await context.entries(kept).catch(() => []);
      removed.push({
        slot: `${prefix}${entry.name}`,
        files: files.filter((file) => file.isFile() || file.isDirectory()).length,
        hasBrief: files.some((file) => file.name === 'brief.json'),
      });
    }
  }
  return removed.sort((left, right) => compareShotNames(left.slot, right.slot));
}

/** The plan behind one shot, with the characters it can choose from. */
export async function readShotPlan(repoRoot, sessionId, root = '', slot = '') {
  const record = sessionRecord(await readSessionsPayload(repoRoot), sessionId);
  if (!record) return null;
  const workingDir = sessionWorkingDir(repoRoot, sessionId, record);
  const resolved = await resolveRenderRoot(repoRoot, workingDir, root);
  return shotPlanPayload(workingDir, resolved, slot);
}

/** Rewrite one shot's plan: its frame descriptions and the characters each frame shows. */
export async function updateShotPlan(repoRoot, input = {}) {
  return withTimelineMutation(repoRoot, input, () => updateShotPlanUnlocked(repoRoot, input));
}

async function updateShotPlanUnlocked(repoRoot, input = {}) {
  const body = input && typeof input === 'object' ? input : {};
  const sessionId = typeof body.sessionId === 'string' ? body.sessionId : '';
  const record = sessionRecord(await readSessionsPayload(repoRoot), sessionId);
  if (!record) return null;
  const workingDir = sessionWorkingDir(repoRoot, sessionId, record);
  const root = await resolveRenderRoot(repoRoot, workingDir, body.root);
  const slot = String(body.slot ?? '');
  const discovery = await collectAcceptanceState(workingDir, root);
  if (!discovery.shots.some((entry) => entry.shot === slot)) throw acceptanceError(`Unknown shot: ${slot || '(missing)'}`);

  const plan = await shotPlanPayload(workingDir, root, slot);
  const edits = {};
  for (const frame of ['ff', 'lf']) {
    if (typeof body[`${frame}Desc`] === 'string') {
      const description = body[`${frame}Desc`].trim();
      if (!description) throw acceptanceError(`${frame === 'ff' ? 'First' : 'Last'} frame description cannot be empty`);
      if (description.length > MAX_DESCRIPTION_LENGTH) throw acceptanceError(`Description must be at most ${MAX_DESCRIPTION_LENGTH} characters`);
      edits[`${frame}_desc`] = description;
    }
    if (Array.isArray(body[`${frame}Vis`])) {
      const wanted = [...new Set(body[`${frame}Vis`].map((value) => Number(value)))];
      const known = new Set(plan.characters.map((character) => character.idx));
      const unknown = wanted.filter((idx) => !known.has(idx));
      if (unknown.length) throw acceptanceError(`Unknown character index: ${unknown.join(', ')}`);
      edits[`${frame}_vis_char_idxs`] = wanted.sort((left, right) => left - right);
    }
  }
  if (!Object.keys(edits).length) throw acceptanceError('Nothing to save');

  const location = shotLocation(workingDir, root, slot);
  const target = path.join(location.shotDir, SHOT_DESCRIPTION_FILENAME);
  const current = JSON.parse(await readFile(target, 'utf8'));
  const sourceEndingChanged = ['lf_desc', 'lf_vis_char_idxs'].some(
    (key) => Object.hasOwn(edits, key) && JSON.stringify(current[key]) !== JSON.stringify(edits[key]),
  );
  await withMutationTransaction(async (transaction) => {
    await transaction.write(target, `${JSON.stringify({...current, ...edits}, null, 4)}\n`);
    await invalidateTimelineOutputs(transaction, workingDir, root, [slot], true, {
      continuitySourceSlots: sourceEndingChanged ? [slot] : [],
    });
  });
  return shotPlanPayload(workingDir, root, slot);
}

/**
 * Move a shot to a different point in the film.
 *
 * A shot's place is its camera's list, and the film plays camera by camera, so moving a shot
 * to a point another camera owns moves it into that camera. The camera decides which still
 * the shot is drawn from, so the plan and the brief move with it rather than being left
 * naming the camera it came from.
 *
 * The two things a removal refuses to break are refused here too: a camera left with no
 * shots has nothing to play, and a shot that a child camera moves away from carries that
 * camera's transition.
 */
export async function moveShot(repoRoot, input = {}) {
  return withTimelineMutation(repoRoot, input, () => moveShotUnlocked(repoRoot, input));
}

async function moveShotUnlocked(repoRoot, input = {}) {
  const body = input && typeof input === 'object' ? input : {};
  const sessionId = typeof body.sessionId === 'string' ? body.sessionId : '';
  const record = sessionRecord(await readSessionsPayload(repoRoot), sessionId);
  if (!record) return null;
  const workingDir = sessionWorkingDir(repoRoot, sessionId, record);
  const root = await resolveRenderRoot(repoRoot, workingDir, body.root);
  const slot = String(body.slot ?? '');
  const orderBefore = await filmOrder(workingDir, root);
  const location = shotLocation(workingDir, root, slot);
  const direction = String(body.direction ?? '');
  if (direction && !['earlier', 'later'].includes(direction)) throw acceptanceError(`Unknown direction: ${direction}`);

  let cameras;
  try { cameras = JSON.parse(await readFile(location.treePath, 'utf8')); }
  catch { throw acceptanceError('This sequence has no camera tree to change'); }
  if (!Array.isArray(cameras)) throw acceptanceError('This sequence has no camera tree to change');
  const prefix = location.scene ? `${location.scene}/` : '';
  const order = cameras.slice().sort((left, right) => Number(left.idx) - Number(right.idx))
    .flatMap((camera) => (camera.active_shot_idxs || []).map((local) => `${prefix}${local}`));
  const at = order.indexOf(slot);
  const after = direction
    ? direction === 'earlier'
      ? (at >= 2 ? order[at - 2] : '')
      : (at >= 0 && at + 1 < order.length ? order[at + 1] : '')
    : String(body.after ?? '');
  if (slot === after) throw acceptanceError('A shot cannot follow itself');
  if (direction && !after) {
    throw acceptanceError(`Shot ${slot} is already the film's ${direction === 'earlier' ? 'first' : 'last'} shot, so there is nowhere to move it`);
  }
  const targetLocation = shotLocation(workingDir, root, after);
  if (location.scene !== targetLocation.scene) throw acceptanceError('Shots cannot be moved between scenes');
  const owner = cameras.find((camera) => (camera.active_shot_idxs || []).map(String).includes(location.local));
  if (!owner) throw acceptanceError(`Shot ${slot || '(missing)'} is not part of a camera in this sequence`);
  const target = cameras.find((camera) => (camera.active_shot_idxs || []).map(String).includes(targetLocation.local));
  if (!target) throw acceptanceError(`Shot ${after || '(missing)'} is not part of the film, so ${slot} cannot follow it`);
  const leaving = owner !== target;
  const remaining = (owner.active_shot_idxs || []).filter((idx) => String(idx) !== location.local);
  if (!remaining.length) {
    throw acceptanceError(`Shot ${slot} is the only shot of camera ${owner.idx}; moving it would leave the camera with nothing`);
  }
  if (leaving) {
    const orphaned = cameras.find((camera) => String(camera.parent_shot_idx) === location.local && camera !== owner);
    if (orphaned) {
      throw acceptanceError(`Shot ${slot} is the shot camera ${orphaned.idx} moves away from; moving it would leave that camera without its transition`);
    }
  }
  const planPath = path.join(location.shotDir, SHOT_DESCRIPTION_FILENAME);
  let plan;
  try { plan = JSON.parse(await readFile(planPath, 'utf8')); }
  catch { throw acceptanceError(`Unknown shot: ${slot}`); }
  const storyboard = await readJsonOptional(location.storyboardPath);

  owner.active_shot_idxs = remaining;
  const ordered = (target.active_shot_idxs || []).filter((idx) => String(idx) !== location.local);
  const position = ordered.map(String).indexOf(targetLocation.local) + 1;
  target.active_shot_idxs = [...ordered.slice(0, position), Number(location.local), ...ordered.slice(position)];
  const changedPlan = {...plan, cam_idx: target.idx};
  const entry = Array.isArray(storyboard) && storyboard.find((row) => String(row?.idx) === location.local);
  if (entry) entry.cam_idx = target.idx;

  await withMutationTransaction(async (transaction) => {
    await transaction.write(location.treePath, `${JSON.stringify(cameras, null, 4)}\n`);
    await transaction.write(planPath, `${JSON.stringify(changedPlan, null, 4)}\n`);
    if (Array.isArray(storyboard)) await transaction.write(location.storyboardPath, `${JSON.stringify(storyboard, null, 4)}\n`);
    await invalidateTimelineOutputs(transaction, workingDir, root, [slot], leaving, {orderBefore});
    await normalizeTimelineEnding(transaction, workingDir, root);
  });

  return {...(await acceptancePayload(repoRoot, workingDir, root)), moved: {slot, after, camera: target.idx, position}};
}

/**
 * Take a shot out of the film.
 *
 * A shot is in the film because its camera lists it and its plan exists, so removing it
 * means both: the index leaves the camera's list, and the shot's directory moves out of
 * `shots/` where nothing enumerates it any more. Nothing is deleted — the frames and the
 * plan are kept under `.removed_shots/`, so a shot removed by mistake can be put back.
 */
export async function removeShot(repoRoot, input = {}) {
  return withTimelineMutation(repoRoot, input, () => removeShotUnlocked(repoRoot, input));
}

async function removeShotUnlocked(repoRoot, input = {}) {
  const body = input && typeof input === 'object' ? input : {};
  const sessionId = typeof body.sessionId === 'string' ? body.sessionId : '';
  const record = sessionRecord(await readSessionsPayload(repoRoot), sessionId);
  if (!record) return null;
  const workingDir = sessionWorkingDir(repoRoot, sessionId, record);
  const root = await resolveRenderRoot(repoRoot, workingDir, body.root);
  const slot = String(body.slot ?? '');
  const orderBefore = await filmOrder(workingDir, root);
  const location = shotLocation(workingDir, root, slot);
  let cameras;
  try { cameras = JSON.parse(await readFile(location.treePath, 'utf8')); }
  catch { throw acceptanceError('This sequence has no camera tree to change'); }
  if (!Array.isArray(cameras)) throw acceptanceError('This sequence has no camera tree to change');
  const owner = cameras.find((camera) => (camera.active_shot_idxs || []).map(String).includes(location.local));
  if (!owner) throw acceptanceError(`Shot ${slot} is not part of a camera in this sequence`);
  const orphaned = cameras.find((camera) => String(camera.parent_shot_idx) === location.local && camera !== owner);
  if (orphaned) {
    throw acceptanceError(`Shot ${slot} is the shot camera ${orphaned.idx} moves away from; removing it would leave that camera without its transition`);
  }
  const planPath = path.join(location.shotDir, SHOT_DESCRIPTION_FILENAME);
  let plan;
  try { plan = JSON.parse(await readFile(planPath, 'utf8')); }
  catch { throw acceptanceError(`Unknown shot: ${slot}`); }
  const from = location.shotDir;
  const to = path.join(location.container, REMOVED_SHOTS_DIR, location.local);
  const storyboard = await readJsonOptional(location.storyboardPath);
  if (storyboard !== undefined && !Array.isArray(storyboard)) throw acceptanceError('The storyboard is not a list');
  const brief = Array.isArray(storyboard) ? storyboard.find((entry) => String(entry?.idx) === location.local) : null;
  if (brief && await pathExists(path.join(from, 'brief.json'))) throw acceptanceError(`Shot ${slot} already has kept brief evidence`);

  const active = owner.active_shot_idxs || [];
  const position = active.map(String).indexOf(location.local);
  const placement = {
    camera: owner.idx,
    position,
    previous: position > 0 ? String(active[position - 1]) : null,
    next: position + 1 < active.length ? String(active[position + 1]) : null,
  };
  owner.active_shot_idxs = active.filter((idx) => String(idx) !== location.local);
  const remainingBriefs = brief ? storyboard.filter((entry) => String(entry?.idx) !== location.local) : null;
  await withMutationTransaction(async (transaction) => {
    await transaction.move(from, to);
    await transaction.write(path.join(to, 'placement.json'), `${JSON.stringify(placement, null, 4)}\n`);
    if (brief) {
      await transaction.write(path.join(to, 'brief.json'), `${JSON.stringify(brief, null, 4)}\n`);
      await transaction.write(location.storyboardPath, `${JSON.stringify(remainingBriefs, null, 4)}\n`);
    }
    await transaction.write(location.treePath, `${JSON.stringify(cameras, null, 4)}\n`);
    if (plan.is_last) await transaction.write(path.join(to, SHOT_DESCRIPTION_FILENAME), `${JSON.stringify({...plan, is_last: false}, null, 4)}\n`);
    await invalidateTimelineOutputs(transaction, workingDir, root, [slot], true, {orderBefore});
    await normalizeTimelineEnding(transaction, workingDir, root);
  });
  return acceptancePayload(repoRoot, workingDir, root);
}

/**
 * Put a removed shot back into the film.
 *
 * Three things went out, so three come back: the shot's directory, its brief in the
 * storyboard, and its place in its camera. The camera is the one it is put back into,
 * which is why the kept directory records it — a shot restored into the wrong camera would
 * change the film's shape rather than undo the change.
 */
export async function restoreShot(repoRoot, input = {}) {
  return withTimelineMutation(repoRoot, input, () => restoreShotUnlocked(repoRoot, input));
}

async function restoreShotUnlocked(repoRoot, input = {}) {
  const body = input && typeof input === 'object' ? input : {};
  const sessionId = typeof body.sessionId === 'string' ? body.sessionId : '';
  const record = sessionRecord(await readSessionsPayload(repoRoot), sessionId);
  if (!record) return null;
  const workingDir = sessionWorkingDir(repoRoot, sessionId, record);
  const root = await resolveRenderRoot(repoRoot, workingDir, body.root);
  const slot = String(body.slot ?? '');
  const orderBefore = await filmOrder(workingDir, root);
  const location = shotLocation(workingDir, root, slot);
  const kept = path.join(location.container, REMOVED_SHOTS_DIR, location.local);
  const live = location.shotDir;
  if (!(await isDirectory(kept))) throw acceptanceError(`Shot ${slot || '(missing)'} has not been removed from this sequence`);
  if (await pathExists(live)) throw acceptanceError(`Shot ${slot} already exists in this sequence`);

  const placement = await readJsonOptional(path.join(kept, 'placement.json'));
  let cameras;
  try { cameras = JSON.parse(await readFile(location.treePath, 'utf8')); }
  catch { throw acceptanceError('This sequence has no camera tree to change'); }
  if (!Array.isArray(cameras)) throw acceptanceError('This sequence has no camera tree to change');
  if (cameras.some((camera) => (camera.active_shot_idxs || []).map(String).includes(location.local))) {
    throw acceptanceError(`Shot ${slot} is already part of this sequence`);
  }
  const cameraIdx = Number.isFinite(Number(placement?.camera)) ? Number(placement.camera) : cameras[0]?.idx;
  const camera = cameras.find((entry) => Number(entry.idx) === Number(cameraIdx));
  if (!camera) throw acceptanceError(`Camera ${cameraIdx} is no longer in this sequence, so shot ${slot} has nowhere to go back to`);
  const active = [...(camera.active_shot_idxs || [])];
  let position = Number.isFinite(Number(placement?.position)) ? Number(placement.position) : active.length;
  const before = placement?.previous == null ? -1 : active.map(String).indexOf(String(placement.previous));
  const after = placement?.next == null ? -1 : active.map(String).indexOf(String(placement.next));
  if (before >= 0) position = before + 1;
  else if (after >= 0) position = after;
  position = Math.max(0, Math.min(position, active.length));
  active.splice(position, 0, Number(location.local));
  camera.active_shot_idxs = active;

  const briefPath = path.join(kept, 'brief.json');
  const brief = await readJsonOptional(briefPath);
  let storyboard = await readJsonOptional(location.storyboardPath);
  if (storyboard === undefined) storyboard = [];
  if (!Array.isArray(storyboard)) throw acceptanceError('The storyboard is not a list');
  const nextStoryboard = brief && typeof brief === 'object'
    ? [...storyboard.filter((entry) => String(entry?.idx) !== location.local), brief]
    : storyboard;

  await withMutationTransaction(async (transaction) => {
    await transaction.move(kept, live);
    await transaction.write(location.treePath, `${JSON.stringify(cameras, null, 4)}\n`);
    if (brief) await transaction.write(location.storyboardPath, `${JSON.stringify(nextStoryboard, null, 4)}\n`);
    await transaction.remove(path.join(live, 'placement.json'));
    if (brief) await transaction.remove(path.join(live, 'brief.json'));
    await invalidateTimelineOutputs(transaction, workingDir, root, [slot], true, {orderBefore});
    await normalizeTimelineEnding(transaction, workingDir, root);
  });
  return acceptancePayload(repoRoot, workingDir, root);
}

/**
 * Add a shot to the film, copying the plan of a shot that already exists.
 *
 * A shot needs a brief, a plan and a place in a camera. The brief is what the user wrote;
 * the plan starts as a copy of the reference shot's, because the frame descriptions are the
 * part that takes judgement and copying a shot that works is how a new one gets made. The
 * copy is stated in the response, not hidden: a shot that renders as a duplicate until it is
 * edited is worth saying out loud.
 */
export async function createShot(repoRoot, input = {}) {
  return withTimelineMutation(repoRoot, input, () => createShotUnlocked(repoRoot, input));
}

async function createShotUnlocked(repoRoot, input = {}) {
  const body = input && typeof input === 'object' ? input : {};
  const sessionId = typeof body.sessionId === 'string' ? body.sessionId : '';
  const record = sessionRecord(await readSessionsPayload(repoRoot), sessionId);
  if (!record) return null;
  const workingDir = sessionWorkingDir(repoRoot, sessionId, record);
  const root = await resolveRenderRoot(repoRoot, workingDir, body.root);
  const orderBefore = await filmOrder(workingDir, root);
  const after = String(body.after ?? '');
  const afterLocation = shotLocation(workingDir, root, after);
  const brief = String(body.brief ?? '').trim();
  if (!brief) throw acceptanceError('Say what happens in the new shot');
  if (brief.length > MAX_BRIEF_LENGTH) throw acceptanceError(`The description must be at most ${MAX_BRIEF_LENGTH} characters`);

  let cameras;
  try { cameras = JSON.parse(await readFile(afterLocation.treePath, 'utf8')); }
  catch { throw acceptanceError('This sequence has no camera tree to change'); }
  if (!Array.isArray(cameras)) throw acceptanceError('This sequence has no camera tree to change');
  const camera = cameras.find((entry) => (entry.active_shot_idxs || []).map(String).includes(afterLocation.local));
  if (!camera) throw acceptanceError(`Shot ${after} is not part of a camera, so there is nothing to add a shot after`);
  const reference = await shotPlanPayload(workingDir, root, after);
  const sourcePath = path.join(afterLocation.shotDir, SHOT_DESCRIPTION_FILENAME);
  const source = JSON.parse(await readFile(sourcePath, 'utf8'));

  const frames = {};
  const audioDesc = String(body.audioDesc ?? '').trim();
  const motionDesc = String(body.motionDesc ?? '').trim();
  if (audioDesc.length > MAX_BRIEF_LENGTH) throw acceptanceError(`The dialogue must be at most ${MAX_BRIEF_LENGTH} characters`);
  for (const frame of ['ff', 'lf']) {
    if (typeof body[`${frame}Desc`] === 'string') {
      const description = body[`${frame}Desc`].trim();
      if (!description) throw acceptanceError(`${frame === 'ff' ? 'First' : 'Last'} frame description cannot be empty`);
      if (description.length > MAX_DESCRIPTION_LENGTH) throw acceptanceError(`Description must be at most ${MAX_DESCRIPTION_LENGTH} characters`);
      frames[`${frame}_desc`] = description;
    }
    if (Array.isArray(body[`${frame}Vis`])) {
      const wanted = [...new Set(body[`${frame}Vis`].map((value) => Number(value)))];
      const unknown = wanted.filter((idx) => !reference.characters.some((character) => character.idx === idx));
      if (unknown.length) throw acceptanceError(`Unknown character index: ${unknown.join(', ')}`);
      frames[`${frame}_vis_char_idxs`] = wanted.sort((left, right) => left - right);
    }
  }

  const used = new Set((cameras.flatMap((entry) => entry.active_shot_idxs || [])).map(Number).filter(Number.isSafeInteger));
  const removedDir = path.join(afterLocation.container, REMOVED_SHOTS_DIR);
  for (const entry of await readdir(removedDir, {withFileTypes: true}).catch(() => [])) {
    if (!entry.isDirectory() || !/^\d+$/.test(entry.name)) continue;
    const id = Number(entry.name);
    if (Number.isSafeInteger(id)) used.add(id);
  }
  let localSlot = used.size ? Math.max(...used) + 1 : 0;
  let shotDir;
  while (Number.isSafeInteger(localSlot)) {
    shotDir = path.join(afterLocation.container, 'shots', String(localSlot));
    if (!used.has(localSlot) && !(await pathExists(shotDir))) break;
    localSlot += 1;
  }
  if (!Number.isSafeInteger(localSlot)) throw acceptanceError('No safe shot index remains in this sequence');
  const slot = `${afterLocation.scene ? `${afterLocation.scene}/` : ''}${localSlot}`;

  const position = (camera.active_shot_idxs || []).map(String).indexOf(afterLocation.local) + 1;
  camera.active_shot_idxs = [
    ...(camera.active_shot_idxs || []).slice(0, position),
    localSlot,
    ...(camera.active_shot_idxs || []).slice(position),
  ];
  let storyboard = await readJsonOptional(afterLocation.storyboardPath);
  if (storyboard === undefined) storyboard = [];
  if (!Array.isArray(storyboard)) throw acceptanceError('The storyboard is not a list');
  storyboard = [...storyboard, {idx: localSlot, is_last: false, cam_idx: camera.idx, visual_desc: brief, audio_desc: audioDesc}];
  const created = {...source, idx: localSlot, is_last: false, cam_idx: camera.idx, ...frames, audio_desc: audioDesc};
  if (motionDesc) created.motion_desc = motionDesc;

  await withMutationTransaction(async (transaction) => {
    await transaction.mkdir(shotDir);
    await transaction.write(path.join(shotDir, SHOT_DESCRIPTION_FILENAME), `${JSON.stringify(created, null, 4)}\n`);
    await transaction.write(afterLocation.treePath, `${JSON.stringify(cameras, null, 4)}\n`);
    await transaction.write(afterLocation.storyboardPath, `${JSON.stringify(storyboard, null, 4)}\n`);
    await invalidateTimelineOutputs(transaction, workingDir, root, [slot], true, {orderBefore});
    await normalizeTimelineEnding(transaction, workingDir, root);
  });
  return {
    ...(await acceptancePayload(repoRoot, workingDir, root)),
    created: {slot, copiedFrom: after, copiedPlan: Object.keys(frames).length === 0, plan: reference},
  };
}

async function invalidateTimelineOutputs(
  transaction,
  workingDir,
  root,
  shots,
  clearShotAcceptance = true,
  {orderBefore, continuitySourceSlots = []} = {},
) {
  const continuityEnabled = await frameContinuityEnabled(workingDir);
  const store = await readAcceptanceFile(workingDir);
  const before = JSON.stringify(store);
  const rootStore = isPlainObject(store[root]) ? store[root] : {};
  const shotStore = isPlainObject(rootStore.shots) ? rootStore.shots : {};
  const now = isoSeconds();

  if (continuityEnabled) {
    let invalidatedSlots = [];
    let playbackOrder = [];
    if (Array.isArray(orderBefore)) {
      playbackOrder = await filmOrder(workingDir, root);
      const changedSuffix = timelineOrderChangedSuffix(orderBefore, playbackOrder);
      invalidatedSlots = [
        ...changedSuffix,
        ...(changedSuffix.length || clearShotAcceptance ? shots : []),
      ];
    } else if (continuitySourceSlots.length) {
      playbackOrder = await filmOrder(workingDir, root);
      invalidatedSlots = continuitySuccessors(
        playbackOrder,
        continuitySourceSlots,
        new Set(shots.map(String)),
      );
    }
    const uniqueSlots = [...new Set(invalidatedSlots.map(String))];
    for (const slot of uniqueSlots) {
      const record = shotStore[slot];
      if (!isPlainObject(record)) continue;
      const reason = Array.isArray(orderBefore)
        ? 'The shot or film playback order changed; re-review this shot’s keyframes and clip for the continuous handoff.'
        : continuityInvalidationReason(playbackOrder, slot, continuitySourceSlots);
      for (const stage of ['keyframes', 'clips']) {
        if (isPlainObject(record[stage])) {
          record[stage] = {...record[stage], invalidated_at: now, invalidation_reason: reason};
        }
      }
      shotStore[slot] = record;
    }

    const finalRecord = rootStore.final_video;
    if (isPlainObject(finalRecord)) {
      rootStore.final_video = {
        ...finalRecord,
        invalidated_at: now,
        invalidation_reason: 'The continuous timeline changed; rebuild and review the final assembly.',
      };
    }
    if (Object.keys(shotStore).length) rootStore.shots = shotStore;
    if (Object.keys(rootStore).length) store[root] = rootStore;
  } else {
    removeAcceptance(store, root, null, 'final_video');
  }

  if (clearShotAcceptance && !(continuityEnabled && Array.isArray(orderBefore))) {
    for (const slot of shots) {
      removeAcceptance(store, root, slot, 'keyframes');
      removeAcceptance(store, root, slot, 'clips');
    }
  }
  if (JSON.stringify(store) !== before) {
    await transaction.write(path.join(workingDir, ACCEPTANCE_FILENAME), `${JSON.stringify(store, null, 2)}\n`);
  }

  const rootDir = path.join(workingDir, root);
  const films = new Set([path.join(rootDir, 'final_video.mp4')]);
  for (const slot of shots) {
    const location = shotLocation(workingDir, root, slot);
    if (location.scene) films.add(path.join(location.container, 'final_video.mp4'));
  }
  for (const film of films) await transaction.remove(film);
}

async function frameContinuityEnabled(workingDir) {
  try {
    const settings = await readJsonOptional(path.join(workingDir, 'frame_continuity.json'));
    return isPlainObject(settings) && settings.mode === 'chained_keyframes';
  } catch {
    return false;
  }
}

function timelineOrderChangedSuffix(before, after) {
  let firstChanged = 0;
  while (firstChanged < before.length && firstChanged < after.length && before[firstChanged] === after[firstChanged]) {
    firstChanged += 1;
  }
  return after.slice(firstChanged);
}

function continuitySuccessors(order, changedSlots, excluded = new Set()) {
  const changed = new Set(changedSlots.map(String));
  const firstChanged = order.reduce(
    (first, slot, index) => changed.has(slot) ? Math.min(first, index) : first,
    order.length,
  );
  return order.slice(firstChanged + 1).filter((slot) => !changed.has(slot) && !excluded.has(slot));
}

function continuityInvalidationReason(order, slot, changedSlots) {
  const at = order.indexOf(slot);
  const changed = new Set(changedSlots.map(String));
  const source = order.slice(0, at).reverse().find((candidate) => changed.has(candidate));
  return source
    ? `Shot ${source}’s last keyframe changed; re-review this shot’s keyframes and clip for the continuous handoff.`
    : 'An earlier keyframe ending changed; re-review this shot’s keyframes and clip.';
}

async function normalizeTimelineEnding(transaction, workingDir, root) {
  const order = await filmOrder(workingDir, root);
  const last = order.at(-1) || '';
  const rootDir = path.join(workingDir, root);
  const containers = await shotContainers(rootDir);
  if (!containers.length) containers.push({container: path.join(rootDir, 'shots'), prefix: ''});
  for (const {container: shotsDir, prefix} of containers) {
    const sceneDir = path.dirname(shotsDir);
    const treePath = path.join(sceneDir, CAMERA_TREE_FILENAME);
    const cameras = await readJsonOptional(treePath);
    if (cameras === undefined) continue;
    if (!Array.isArray(cameras)) throw acceptanceError('This sequence has no camera tree to change');
    const active = cameras.slice().sort((left, right) => Number(left.idx) - Number(right.idx))
      .flatMap((camera) => (camera.active_shot_idxs || []).map(String));
    for (const local of active) {
      const planPath = path.join(shotsDir, local, SHOT_DESCRIPTION_FILENAME);
      let plan;
      try { plan = JSON.parse(await readFile(planPath, 'utf8')); }
      catch (error) { if (error.code === 'ENOENT') continue; throw error; }
      const isLast = `${prefix}${local}` === last;
      if (plan.is_last !== isLast) await transaction.write(planPath, `${JSON.stringify({...plan, is_last: isLast}, null, 4)}\n`);
    }

    const storyboardPath = path.join(sceneDir, STORYBOARD_FILENAME);
    const storyboard = await readJsonOptional(storyboardPath);
    if (storyboard === undefined) continue;
    if (!Array.isArray(storyboard)) throw acceptanceError('The storyboard is not a list');
    const bySlot = new Map();
    for (const entry of storyboard) {
      const key = String(entry?.idx ?? '');
      const entries = bySlot.get(key) || [];
      entries.push(entry);
      bySlot.set(key, entries);
    }
    const normalized = [];
    const included = new Set();
    for (const local of active) {
      for (const [index, entry] of (bySlot.get(local) || []).entries()) {
        const isLast = `${prefix}${local}` === last && index === 0;
        normalized.push({...entry, is_last: isLast});
      }
      included.add(local);
    }
    for (const [local, entries] of bySlot) {
      if (included.has(local)) continue;
      normalized.push(...entries.map((entry) => ({...entry, is_last: false})));
    }
    if (JSON.stringify(normalized) !== JSON.stringify(storyboard)) {
      await transaction.write(storyboardPath, `${JSON.stringify(normalized, null, 4)}\n`);
    }
  }
}

async function readJsonOptional(filePath) {
  try { return JSON.parse(await readFile(filePath, 'utf8')); }
  catch (error) {
    if (error.code === 'ENOENT') return undefined;
    throw error;
  }
}

async function pathExists(target) {
  try { await lstat(target); return true; }
  catch (error) { if (error.code === 'ENOENT') return false; throw error; }
}

async function shotPlanPayload(workingDir, root, slot, context = createFilmReadContext()) {
  const location = shotLocation(workingDir, root, slot);
  const shotDir = location.shotDir;
  let description;
  try {
    description = await context.json(path.join(shotDir, SHOT_DESCRIPTION_FILENAME));
    if (!description) throw new Error('Missing shot description');
  } catch {
    throw acceptanceError(`Unknown shot: ${slot || '(missing)'}`);
  }
  const {characters, briefs} = await scenePlanMetadata(workingDir, location, context);
  const prompt = async (frame) => {
    try {
      const cached = await context.json(path.join(shotDir, `${frame}_selector_output.json`));
      return typeof cached?.sent_prompt === 'string' ? cached.sent_prompt : '';
    } catch { return ''; }
  };
  return {
    slot: String(slot), root,
    brief: briefs.get(String(location.local)) || '',
    characters,
    firstFrame: {description: String(description.ff_desc || ''), visible: [...(description.ff_vis_char_idxs || [])], prompt: await prompt('first_frame')},
    lastFrame: {description: String(description.lf_desc || ''), visible: [...(description.lf_vis_char_idxs || [])], prompt: await prompt('last_frame')},
    motionDescription: String(description.motion_desc || ''),
  };
}

function scenePlanMetadata(workingDir, location, context) {
  if (!context.scenes.has(location.container)) {
    context.scenes.set(location.container, (async () => {
      const characters = await readSessionCharacters(workingDir, location.container, context);
      const storyboard = await context.json(location.storyboardPath).catch(() => null);
      const briefs = new Map();
      for (const shot of Array.isArray(storyboard) ? storyboard : []) {
        const key = String(shot?.idx);
        if (!briefs.has(key)) briefs.set(key, typeof shot?.visual_desc === 'string' ? shot.visual_desc : '');
      }
      return {characters, briefs};
    })());
  }
  return context.scenes.get(location.container);
}

/** The characters a root can draw, from the plan beside its shots. */
async function readSessionCharacters(workingDir, root, context) {
  try {
    const payload = await context.json(path.isAbsolute(root) ? path.join(root, 'characters.json') : path.join(workingDir, root, 'characters.json'));
    if (!Array.isArray(payload)) return [];
    return payload
      .map((character) => ({idx: Number(character.idx), name: String(character.identifier_in_scene || '')}))
      .filter((character) => Number.isFinite(character.idx) && character.name)
      .sort((left, right) => left.idx - right.idx);
  } catch {
    return [];
  }
}
