import {createHash} from 'node:crypto';
import {createReadStream} from 'node:fs';
import {lstat, mkdir, readFile, readdir, rename, rm, stat, writeFile} from 'node:fs/promises';
import path from 'node:path';
import {readClipSettings} from './config-store.mjs';

const IMAGE_EXTENSIONS = new Set(['.png', '.jpg', '.jpeg', '.webp', '.gif']);
const VIDEO_EXTENSIONS = new Set(['.mp4', '.webm', '.mov']);
const TEXT_EXTENSIONS = new Set(['.txt', '.md', '.json']);

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
  try {
    const lines = (await readFile(logPath, 'utf8')).split(/\r?\n/).filter(Boolean);
    const messages = [];
    for (const line of lines) {
      let record;
      try {
        record = JSON.parse(line);
      } catch {
        continue;
      }
      if (record.session_id !== sessionId || !record.raw_user_input) continue;
      const turnId = String(record.turn_id || `turn-${messages.length}`);
      messages.push({
        id: `${turnId}-user`,
        role: 'user',
        text: displayUserInput(record.raw_user_input),
        createdAt: String(record.created_at || record.timestamp || ''),
      });
      for (const round of Array.isArray(record.tool_rounds) ? record.tool_rounds : []) {
        for (const result of Array.isArray(round.tool_results) ? round.tool_results : []) {
          messages.push({
            id: `${turnId}-tool-${messages.length}`,
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
        messages.push({
          id: `${turnId}-assistant`,
          role: record.status === 'failed' ? 'error' : 'assistant',
          text: String(record.final_assistant_text),
          createdAt: String(record.created_at || record.timestamp || ''),
        });
      }
    }
    return messages.slice(-120);
  } catch {
    return [];
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
  const sessionRoot = resolveSessionRoot(repoRoot, sessionId);
  const artifacts = [];

  async function walk(directory) {
    let entries;
    try {
      entries = await readdir(directory, {withFileTypes: true});
    } catch {
      return;
    }
    for (const entry of entries) {
      if (artifacts.length >= 400 || entry.name.startsWith('.')) continue;
      const absolute = path.join(directory, entry.name);
      if (entry.isDirectory()) {
        // `cache/` holds moviepy's scene-split debris from a transition render.
        // It is scratch, fully represented by the transition video beside it, so
        // it is not offered as a browsable artifact. Deletion still covers it
        // (see collectStyleDerivedArtifacts), which walks for cleanup, not browsing.
        if (entry.name === 'cache') continue;
        await walk(absolute);
        continue;
      }
      if (!entry.isFile()) continue;
      const extension = path.extname(entry.name).toLowerCase();
      const kind = IMAGE_EXTENSIONS.has(extension)
        ? 'image'
        : VIDEO_EXTENSIONS.has(extension)
          ? 'video'
          : TEXT_EXTENSIONS.has(extension)
            ? 'document'
            : null;
      if (!kind) continue;
      const info = await stat(absolute);
      const relativePath = path.relative(sessionRoot, absolute).split(path.sep).join('/');
      artifacts.push({
        path: relativePath,
        name: entry.name,
        kind,
        size: info.size,
        updatedAt: info.mtime.toISOString(),
        url: `/api/artifact?session=${encodeURIComponent(sessionId)}&path=${encodeURIComponent(relativePath)}`,
      });
    }
  }

  await walk(sessionRoot);
  return artifacts.sort((left, right) => right.updatedAt.localeCompare(left.updatedAt));
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

async function readRenderManifest(repoRoot, workingDir) {
  if (!workingDir) return null;
  try {
    const manifest = JSON.parse(await readFile(path.resolve(repoRoot, workingDir, 'render_manifest.json'), 'utf8'));
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

function clipUsdPerSecond(model) {
  const rates = Object.values(CLIP_USD_PER_SECOND_BY_MODEL);
  return CLIP_USD_PER_SECOND_BY_MODEL[String(model || '').trim()] ?? Math.max(...rates);
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
  return acceptancePayload(repoRoot, workingDir, await resolveRenderRoot(repoRoot, workingDir, root));
}

/**
 * Accept, reject or release one slot of a render root and return the refreshed
 * payload. Accepting recomputes the size and sha256 from the files on disk at that
 * moment; rejecting with a reason replaces any acceptance for the slot; releasing
 * with no reason deletes the entry rather than storing it as empty.
 */
export async function updateRenderAcceptance(repoRoot, input = {}) {
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

async function acceptancePayload(repoRoot, workingDir, root) {
  const store = await readAcceptanceFile(workingDir);
  const rootStore = isPlainObject(store[root]) ? store[root] : {};
  const shotStore = isPlainObject(rootStore.shots) ? rootStore.shots : {};
  const discovery = await collectAcceptanceState(workingDir, root);

  const stages = [];
  for (const stage of STAGE_ORDER) {
    const scope = STAGE_SCOPES[stage];
    let slots;
    if (scope === 'session') {
      const paths = stage === 'portraits' ? discovery.portraits : discovery.finalVideo;
      const slotRecord = rootStore[stage];
      slots = [slotPayload(null, await slotState(workingDir, slotRecord, paths), paths, slotRecord)];
    } else {
      slots = [];
      for (const shot of discovery.shots) {
        const paths = shot[stage];
        const slotRecord = isPlainObject(shotStore[shot.shot]) ? shotStore[shot.shot][stage] : undefined;
        slots.push(slotPayload(shot.shot, await slotState(workingDir, slotRecord, paths), paths, slotRecord));
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
  const clipSettings = await readClipSettings(repoRoot, String((await readRenderManifest(repoRoot, workingDir))?.video_model || ''));
  return {
    root,
    stages,
    totals: {
      acceptedKeyframes,
      keyframes,
      clips,
      rejected,
      clipSeconds: clipSettings.seconds,
      clipCostUsd: clipUsdPerSecond(clipSettings.model),
    },
  };
}

/** The payload slot: its state, plus the note against it, whether or not it was redrawn. */
function slotPayload(shot, state, artifacts, record) {
  const slot = {shot, state, artifacts};
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
async function slotState(workingDir, record, paths) {
  // A stored rejection outranks every other reading: the human's note is the point.
  if (isRejected(record)) return 'rejected';
  if (!paths.length) return 'planned';
  // A record without an acceptance — a rejection that was just redrawn, say — leaves the
  // new artifacts unreviewed: redrawing something is not accepting it.
  if (!isPlainObject(record) || typeof record.accepted_at !== 'string' || record.accepted_at === '') return 'rendered';
  return (await acceptanceLockMatches(workingDir, record)) ? 'accepted' : 'stale';
}

/** A slot record carrying a rejection note rather than an acceptance. */
function isRejected(record) {
  return isPlainObject(record) && typeof record.rejected_at === 'string' && record.rejected_at !== '';
}

async function acceptanceLockMatches(workingDir, record) {
  if (!Array.isArray(record.artifacts) || record.artifacts.length === 0) return false;
  for (const artifact of record.artifacts) {
    if (!isPlainObject(artifact)) return false;
    const absolute = path.resolve(workingDir, String(artifact.path ?? ''));
    if (!absolute.startsWith(`${workingDir}${path.sep}`)) return false;
    let info;
    try {
      info = await stat(absolute);
    } catch {
      return false;
    }
    if (!info.isFile() || info.size !== artifact.size) return false;
    if (await sha256File(absolute) !== artifact.sha256) return false;
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
async function collectAcceptanceState(workingDir, root) {
  const rootDir = path.join(workingDir, root);
  let manifest = null;
  try {
    manifest = JSON.parse(await readFile(path.join(workingDir, 'render_manifest.json'), 'utf8'));
  } catch {
    manifest = null;
  }
  const imageModelDir = safePathComponent(manifest?.image_model ?? '');
  const videoModelDir = safePathComponent(manifest?.video_model ?? '');

  const portraits = [];
  await collectAcceptanceFiles(workingDir, path.join(rootDir, 'character_portraits'), (name) => name.toLowerCase().endsWith('.png'), portraits);
  portraits.sort();
  portraits.splice(0, portraits.length, ...currentModelArtifacts(portraits, imageModelDir));

  const finalVideo = [];
  const finalPath = path.join(rootDir, 'final_video.mp4');
  if (await isRegularFile(finalPath)) finalVideo.push(relativeArtifactPath(workingDir, finalPath));

  const shots = [];
  // Script mode keeps shots flat under `shots/`, so the slot is the directory name.
  // Idea mode nests them under `scene_<idx>/shots/`, where the same shot index appears
  // once per scene, so the slot is scene-qualified. Same keys, same rule, as the gate.
  const containers = await shotContainers(rootDir);
  for (const {container, prefix} of containers) {
    const entries = await readdir(container, {withFileTypes: true}).catch(() => []);
    for (const entry of entries) {
      if (!entry.isDirectory() || entry.name.startsWith('.')) continue;
      const shotDir = path.join(container, entry.name);
      // A shot is planned only once its description exists, exactly like the render gate.
      if (!(await isRegularFile(path.join(shotDir, 'shot_description.json')))) continue;
      const keyframes = [];
      await collectAcceptanceFiles(workingDir, shotDir, (name) => KEYFRAME_NAMES.has(name), keyframes);
      const clips = [];
      await collectAcceptanceFiles(workingDir, shotDir, (name) => name === 'video.mp4', clips);
      shots.push({
        shot: `${prefix}${entry.name}`,
        keyframes: currentModelArtifacts(keyframes, imageModelDir).sort(),
        clips: currentModelArtifacts(clips, videoModelDir).sort(),
      });
    }
  }
  shots.sort((left, right) => compareShotNames(left.shot, right.shot));

  return {portraits, finalVideo, shots};
}

/** Where a render root keeps its shots: flat, or one directory per scene. */
async function shotContainers(rootDir) {
  const flat = path.join(rootDir, 'shots');
  const entries = await readdir(rootDir, {withFileTypes: true}).catch(() => []);
  if (entries.some((entry) => entry.isDirectory() && entry.name === 'shots')) {
    return [{container: flat, prefix: ''}];
  }
  return entries
    .filter((entry) => entry.isDirectory() && /^scene_\d+$/.test(entry.name))
    .sort((left, right) => Number(left.name.slice('scene_'.length)) - Number(right.name.slice('scene_'.length)))
    .map((entry) => ({container: path.join(rootDir, entry.name, 'shots'), prefix: `${entry.name}/`}));
}

async function collectAcceptanceFiles(base, directory, isArtifact, artifacts) {
  let entries;
  try {
    entries = await readdir(directory, {withFileTypes: true});
  } catch {
    return;
  }
  for (const entry of entries) {
    if (entry.name.startsWith('.')) continue;
    const absolute = path.join(directory, entry.name);
    // `cache/` is moviepy scratch debris; whatever it holds is never a reviewable artifact.
    if (entry.isDirectory()) {
      if (entry.name !== 'cache') await collectAcceptanceFiles(base, absolute, isArtifact, artifacts);
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

async function readAcceptanceFile(workingDir) {
  try {
    const payload = JSON.parse(await readFile(path.join(workingDir, ACCEPTANCE_FILENAME), 'utf8'));
    return isPlainObject(payload) ? payload : {};
  } catch {
    return {};
  }
}

async function writeAcceptanceFile(workingDir, store) {
  const filePath = path.join(workingDir, ACCEPTANCE_FILENAME);
  const temporaryPath = `${filePath}.${process.pid}.tmp`;
  await writeFile(temporaryPath, `${JSON.stringify(store, null, 2)}\n`, {mode: 0o600});
  await rename(temporaryPath, filePath);
}

async function resolveRenderRoot(repoRoot, workingDir, requested) {
  const value = String(requested ?? '').trim();
  if (value) {
    if (!RENDER_ROOTS.has(value)) throw acceptanceError(`Unknown render root: ${value}`);
    return value;
  }
  const manifest = await readRenderManifest(repoRoot, workingDir);
  const mode = String(manifest?.render_mode ?? '').trim();
  return RENDER_ROOTS.has(mode) ? mode : 'script2video';
}

function sessionWorkingDir(repoRoot, sessionId, record) {
  const workingDir = String(record?.working_dir ?? '');
  return workingDir ? path.resolve(repoRoot, workingDir) : resolveSessionRoot(repoRoot, sessionId);
}

async function isRegularFile(filePath) {
  try {
    return (await stat(filePath)).isFile();
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
const SHOT_DESCRIPTION_FILENAME = 'shot_description.json';
const STORYBOARD_FILENAME = 'storyboard.json';
const CAMERA_TREE_FILENAME = 'camera_tree.json';
const REMOVED_SHOTS_DIR = '.removed_shots';
const MAX_BRIEF_LENGTH = 1200;
const MAX_DESCRIPTION_LENGTH = 4000;
// Written by the agent's vimax_review_timeline tool; the web layer only reads it.
const CONTINUITY_FILENAME = 'continuity_review.json';

/**
 * The film's slots in playing order: camera by camera, in each camera's own order.
 *
 * A shot's number is its identity, not its place, so the shot directories on disk are not
 * the order the film plays in — the camera tree is. Anything comparing the timeline against
 * a review has to use this, or a review that is current reads as out of date.
 */
async function filmOrder(workingDir, root) {
  let cameras;
  try {
    cameras = JSON.parse(await readFile(path.join(workingDir, root, CAMERA_TREE_FILENAME), 'utf8'));
  } catch {
    return [];
  }
  if (!Array.isArray(cameras)) return [];
  return cameras
    .slice()
    .sort((left, right) => Number(left.idx) - Number(right.idx))
    .flatMap((camera) => (camera.active_shot_idxs || []).map(String));
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
  const resolved = await resolveRenderRoot(repoRoot, workingDir, root);
  let review = null;
  try {
    const payload = JSON.parse(await readFile(path.join(workingDir, CONTINUITY_FILENAME), 'utf8'));
    review = isPlainObject(payload) ? payload : null;
  } catch {
    review = null;
  }
  const discovery = await collectAcceptanceState(workingDir, resolved);
  const playing = await filmOrder(workingDir, resolved);
  // The camera tree is the film's order; the directories are the fallback for a root that has
  // not been grouped into cameras yet.
  const shots = playing.length ? playing : discovery.shots.map((entry) => String(entry.shot));
  const {stale, reason} = await reviewStaleness(workingDir, resolved, review, shots);
  const acceptance = await acceptancePayload(repoRoot, workingDir, resolved);
  return {
    root: resolved,
    review,
    stale,
    staleReason: reason,
    shots,
    clipSeconds: Number(acceptance.totals?.clipSeconds) || 0,
  };
}

async function reviewStaleness(workingDir, root, review, shots) {
  if (!review) return {stale: true, reason: 'No review yet.'};
  const reviewed = (Array.isArray(review.shots_reviewed) ? review.shots_reviewed : []).map(String);
  const added = shots.filter((shot) => !reviewed.includes(shot));
  const removed = reviewed.filter((shot) => !shots.includes(shot));
  if (added.length || removed.length) {
    const parts = [];
    if (added.length) parts.push(`shots ${added.join(', ')} added`);
    if (removed.length) parts.push(`shots ${removed.join(', ')} removed`);
    return {stale: true, reason: `The timeline changed since this review: ${parts.join(', ')}.`};
  }
  const reviewedAt = Date.parse(String(review.reviewed_at ?? ''));
  if (!Number.isFinite(reviewedAt)) return {stale: true, reason: 'The review has no timestamp.'};
  const watched = [
    CAMERA_TREE_FILENAME,
    STORYBOARD_FILENAME,
    ...shots.map((shot) => path.join('shots', shot, SHOT_DESCRIPTION_FILENAME)),
  ];
  const newer = [];
  for (const relative of watched) {
    const info = await stat(path.join(workingDir, root, relative)).catch(() => null);
    if (info && info.mtimeMs > reviewedAt) newer.push(relative);
  }
  if (newer.length) return {stale: true, reason: `Edited since this review: ${newer.join(', ')}.`};
  return {stale: false, reason: ''};
}

/** Every shot's plan in one request: the reel shows all of them, not only the open ones. */
export async function readShotPlans(repoRoot, sessionId, root = '') {
  const record = sessionRecord(await readSessionsPayload(repoRoot), sessionId);
  if (!record) return null;
  const workingDir = sessionWorkingDir(repoRoot, sessionId, record);
  const resolved = await resolveRenderRoot(repoRoot, workingDir, root);
  const discovery = await collectAcceptanceState(workingDir, resolved);
  const plans = [];
  for (const shot of discovery.shots) {
    try {
      plans.push(await shotPlanPayload(workingDir, resolved, shot.shot));
    } catch {
      // A shot with no readable plan is simply not part of what can be shown.
    }
  }
  return {root: resolved, plans};
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

async function listRemoved(workingDir, root) {
  const container = path.join(workingDir, root, REMOVED_SHOTS_DIR);
  const entries = await readdir(container, {withFileTypes: true}).catch(() => []);
  const removed = [];
  for (const entry of entries) {
    if (!entry.isDirectory() || entry.name.startsWith('.')) continue;
    const kept = path.join(container, entry.name);
    const files = await readdir(kept, {withFileTypes: true}).catch(() => []);
    removed.push({
      slot: entry.name,
      files: files.filter((file) => file.isFile() || file.isDirectory()).length,
      hasBrief: files.some((file) => file.name === 'brief.json'),
    });
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

  const target = path.join(workingDir, root, 'shots', slot, SHOT_DESCRIPTION_FILENAME);
  const current = JSON.parse(await readFile(target, 'utf8'));
  await writeFile(target, `${JSON.stringify({...current, ...edits}, null, 4)}\n`);
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
  const body = input && typeof input === 'object' ? input : {};
  const sessionId = typeof body.sessionId === 'string' ? body.sessionId : '';
  const record = sessionRecord(await readSessionsPayload(repoRoot), sessionId);
  if (!record) return null;
  const workingDir = sessionWorkingDir(repoRoot, sessionId, record);
  const root = await resolveRenderRoot(repoRoot, workingDir, body.root);
  const slot = String(body.slot ?? '');
  const direction = String(body.direction ?? '');
  if (direction && !['earlier', 'later'].includes(direction)) throw acceptanceError(`Unknown direction: ${direction}`);

  const treePath = path.join(workingDir, root, CAMERA_TREE_FILENAME);
  let tree;
  try {
    tree = JSON.parse(await readFile(treePath, 'utf8'));
  } catch {
    throw acceptanceError('This sequence has no camera tree to change');
  }
  const cameras = Array.isArray(tree) ? tree : [];
  // The film's own order decides what "earlier" means, so a caller cannot act on an order it
  // read a moment ago and have the shot land somewhere else. An explicit `after` is the same
  // request the create endpoint takes: place this shot after that one.
  const order = cameras
    .slice()
    .sort((left, right) => Number(left.idx) - Number(right.idx))
    .flatMap((camera) => (camera.active_shot_idxs || []).map(String));
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
  const owner = cameras.find((camera) => (camera.active_shot_idxs || []).map(String).includes(slot));
  if (!owner) throw acceptanceError(`Shot ${slot || '(missing)'} is not part of a camera in this sequence`);
  const target = cameras.find((camera) => (camera.active_shot_idxs || []).map(String).includes(after));
  if (!target) throw acceptanceError(`Shot ${after || '(missing)'} is not part of the film, so ${slot} cannot follow it`);
  const leaving = owner !== target;

  const remaining = (owner.active_shot_idxs || []).filter((idx) => String(idx) !== slot);
  if (!remaining.length) {
    throw acceptanceError(`Shot ${slot} is the only shot of camera ${owner.idx}; moving it would leave the camera with nothing`);
  }
  if (leaving) {
    const orphaned = cameras.find((camera) => String(camera.parent_shot_idx) === slot && camera !== owner);
    if (orphaned) {
      throw acceptanceError(`Shot ${slot} is the shot camera ${orphaned.idx} moves away from; moving it would leave that camera without its transition`);
    }
  }

  owner.active_shot_idxs = remaining;
  const ordered = (target.active_shot_idxs || []).filter((idx) => String(idx) !== slot);
  const position = ordered.map(String).indexOf(after) + 1;
  target.active_shot_idxs = [...ordered.slice(0, position), Number(slot), ...ordered.slice(position)];
  await writeFile(treePath, `${JSON.stringify(cameras, null, 4)}\n`);

  // Where it plays is which still its frames are drawn from, so the plan and the brief are
  // moved into the target camera instead of being left pointing at the one it came from.
  const planPath = path.join(workingDir, root, 'shots', slot, SHOT_DESCRIPTION_FILENAME);
  let wasLast = false;
  try {
    const plan = JSON.parse(await readFile(planPath, 'utf8'));
    wasLast = Boolean(plan.is_last);
    await writeFile(planPath, `${JSON.stringify({...plan, cam_idx: target.idx}, null, 4)}\n`);
  } catch {
    // A shot with no plan has nothing to keep in step.
  }
  try {
    const storyboard = JSON.parse(await readFile(path.join(workingDir, root, STORYBOARD_FILENAME), 'utf8'));
    if (Array.isArray(storyboard)) {
      const row = storyboard.find((entry) => String(entry?.idx) === slot);
      if (row) row.cam_idx = target.idx;
      await writeFile(path.join(workingDir, root, STORYBOARD_FILENAME), `${JSON.stringify(storyboard, null, 4)}\n`);
    }
  } catch {
    // Likewise.
  }

  // Moving the film's ending out of last place would otherwise leave the ending marked on a
  // shot that is no longer the end of the film.
  if (wasLast) {
    const lastCamera = cameras.slice().sort((left, right) => Number(right.idx) - Number(left.idx))[0];
    const finalSlot = (lastCamera?.active_shot_idxs || []).slice(-1)[0];
    if (finalSlot !== undefined && String(finalSlot) !== slot) await setLastShot(workingDir, root, finalSlot);
  }

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
  const body = input && typeof input === 'object' ? input : {};
  const sessionId = typeof body.sessionId === 'string' ? body.sessionId : '';
  const record = sessionRecord(await readSessionsPayload(repoRoot), sessionId);
  if (!record) return null;
  const workingDir = sessionWorkingDir(repoRoot, sessionId, record);
  const root = await resolveRenderRoot(repoRoot, workingDir, body.root);
  const slot = String(body.slot ?? '');
  const treePath = path.join(workingDir, root, CAMERA_TREE_FILENAME);
  let tree;
  try {
    tree = JSON.parse(await readFile(treePath, 'utf8'));
  } catch {
    throw acceptanceError('This sequence has no camera tree to change');
  }
  const cameras = Array.isArray(tree) ? tree : [];
  const owner = cameras.find((camera) => (camera.active_shot_idxs || []).map(String).includes(slot));
  if (!owner) throw acceptanceError(`Shot ${slot} is not part of a camera in this sequence`);
  // A camera may be left holding nothing: its shots are all under .removed_shots/, the render
  // skips a camera with no shots, and restoring one puts it back in the camera it left. What
  // cannot happen is taking away a shot another camera moves away from — that is its transition.
  const remaining = (owner.active_shot_idxs || []).filter((idx) => String(idx) !== slot);
  const orphaned = cameras.find((camera) => String(camera.parent_shot_idx) === slot && camera !== owner);
  if (orphaned) {
    throw acceptanceError(`Shot ${slot} is the shot camera ${orphaned.idx} moves away from; removing it would leave that camera without its transition`);
  }

  // Where it sat, read before the list changes: a shot put back goes back where it was.
  const position = (owner.active_shot_idxs || []).map(String).indexOf(slot);
  owner.active_shot_idxs = remaining;
  // The film is every shot's clip joined, so a shot leaving makes it stale — and the
  // concatenation is skipped while a film exists, which would keep the old cut on disk.
  await rm(path.join(workingDir, root, 'final_video.mp4'), {force: true}).catch(() => {});
  await writeFile(treePath, `${JSON.stringify(cameras, null, 4)}\n`);

  const from = path.join(workingDir, root, 'shots', slot);
  const to = path.join(workingDir, root, REMOVED_SHOTS_DIR, slot);
  await mkdir(path.dirname(to), {recursive: true});
  await rename(from, to).catch(() => {});
  await setAsideBrief(workingDir, root, slot, to);
  await writeFile(path.join(to, 'placement.json'), `${JSON.stringify({camera: owner.idx, position}, null, 4)}\n`);
  await markLastShot(workingDir, root, remaining);
  return acceptancePayload(repoRoot, workingDir, root);
}

/**
 * Move a removed shot's brief out of the storyboard, into the shot's kept directory.
 *
 * The storyboard is what a re-plan walks: a brief left behind is a shot that comes back,
 * empty, the next time the film is planned. The brief is kept with the shot rather than
 * dropped, so restoring is putting two things back instead of writing one again.
 */
async function setAsideBrief(workingDir, root, slot, keptDirectory) {
  const storyboardPath = path.join(workingDir, root, STORYBOARD_FILENAME);
  let storyboard;
  try {
    storyboard = JSON.parse(await readFile(storyboardPath, 'utf8'));
  } catch {
    return;
  }
  if (!Array.isArray(storyboard)) return;
  const brief = storyboard.find((entry) => String(entry?.idx) === slot);
  if (!brief) return;
  await writeFile(path.join(keptDirectory, 'brief.json'), `${JSON.stringify(brief, null, 4)}\n`);
  await writeFile(storyboardPath, `${JSON.stringify(storyboard.filter((entry) => String(entry?.idx) !== slot), null, 4)}\n`);
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
  const body = input && typeof input === 'object' ? input : {};
  const sessionId = typeof body.sessionId === 'string' ? body.sessionId : '';
  const record = sessionRecord(await readSessionsPayload(repoRoot), sessionId);
  if (!record) return null;
  const workingDir = sessionWorkingDir(repoRoot, sessionId, record);
  const root = await resolveRenderRoot(repoRoot, workingDir, body.root);
  const slot = String(body.slot ?? '');
  const kept = path.join(workingDir, root, REMOVED_SHOTS_DIR, slot);
  if (!(await isDirectory(kept))) throw acceptanceError(`Shot ${slot || '(missing)'} has not been removed from this sequence`);

  const placement = JSON.parse(await readFile(path.join(kept, 'placement.json'), 'utf8').catch(() => 'null'));
  const cameras = JSON.parse(await readFile(path.join(workingDir, root, CAMERA_TREE_FILENAME), 'utf8'));
  const cameraIdx = Number.isFinite(Number(placement?.camera)) ? Number(placement.camera) : cameras[0]?.idx;
  const camera = cameras.find((entry) => Number(entry.idx) === Number(cameraIdx));
  if (!camera) throw acceptanceError(`Camera ${cameraIdx} is no longer in this sequence, so shot ${slot} has nowhere to go back to`);
  const at = Number.isFinite(Number(placement?.position)) ? Number(placement.position) : (camera.active_shot_idxs || []).length;
  const active = [...(camera.active_shot_idxs || [])];
  active.splice(Math.max(0, Math.min(at, active.length)), 0, Number(slot));
  camera.active_shot_idxs = active;
  await writeFile(path.join(workingDir, root, CAMERA_TREE_FILENAME), `${JSON.stringify(cameras, null, 4)}\n`);

  await rename(kept, path.join(workingDir, root, 'shots', slot)).catch(() => {});
  await putBackBrief(workingDir, root, slot);
  // A shot coming back changes the film too: the same reason a removal drops it.
  await rm(path.join(workingDir, root, 'final_video.mp4'), {force: true}).catch(() => {});
  return acceptancePayload(repoRoot, workingDir, root);
}

/** Put a removed shot's brief back into the storyboard, in shot order. */
async function putBackBrief(workingDir, root, slot) {
  const storyboardPath = path.join(workingDir, root, STORYBOARD_FILENAME);
  let brief;
  try {
    brief = JSON.parse(await readFile(path.join(workingDir, root, 'shots', slot, 'brief.json'), 'utf8'));
  } catch {
    return;
  }
  let storyboard = [];
  try {
    const parsed = JSON.parse(await readFile(storyboardPath, 'utf8'));
    if (Array.isArray(parsed)) storyboard = parsed;
  } catch {
    storyboard = [];
  }
  const without = storyboard.filter((entry) => String(entry?.idx) !== slot);
  without.push(brief);
  without.sort((left, right) => Number(left.idx) - Number(right.idx));
  await writeFile(storyboardPath, `${JSON.stringify(without, null, 4)}\n`);
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
  const body = input && typeof input === 'object' ? input : {};
  const sessionId = typeof body.sessionId === 'string' ? body.sessionId : '';
  const record = sessionRecord(await readSessionsPayload(repoRoot), sessionId);
  if (!record) return null;
  const workingDir = sessionWorkingDir(repoRoot, sessionId, record);
  const root = await resolveRenderRoot(repoRoot, workingDir, body.root);
  const after = String(body.after ?? '');
  const brief = String(body.brief ?? '').trim();
  if (!brief) throw acceptanceError('Say what happens in the new shot');
  if (brief.length > MAX_BRIEF_LENGTH) throw acceptanceError(`The description must be at most ${MAX_BRIEF_LENGTH} characters`);

  const cameras = JSON.parse(await readFile(path.join(workingDir, root, CAMERA_TREE_FILENAME), 'utf8'));
  const camera = cameras.find((entry) => (entry.active_shot_idxs || []).map(String).includes(after));
  if (!camera) throw acceptanceError(`Shot ${after || '(missing)'} is not part of a camera, so there is nothing to add a shot after`);
  const reference = await shotPlanPayload(workingDir, root, after);

  // A shot suggested by a coverage review arrives with its own frames and dialogue, so
  // that adding it produces the proposed shot rather than a copy of its neighbour.
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

  // The next number nothing has held: a removed shot keeps its identity, so its number is
  // not handed to a new shot that would then share its kept files.
  const used = new Set(cameras.flatMap((entry) => entry.active_shot_idxs || []).map(Number));
  const removedDir = path.join(workingDir, root, REMOVED_SHOTS_DIR);
  for (const entry of await readdir(removedDir, {withFileTypes: true}).catch(() => [])) {
    if (entry.isDirectory() && /^\d+$/.test(entry.name)) used.add(Number(entry.name));
  }
  const slot = used.size ? Math.max(...used) + 1 : 0;

  // A camera's list is the order its shots play in, so the new shot is spliced where it
  // belongs and the list is left alone: sorting it would move the shot to the end of the film.
  const position = (camera.active_shot_idxs || []).map(String).indexOf(after) + 1;
  const wasLast = Boolean(JSON.parse(await readFile(path.join(workingDir, root, 'shots', after, SHOT_DESCRIPTION_FILENAME), 'utf8')).is_last);
  camera.active_shot_idxs = [...(camera.active_shot_idxs || []).slice(0, position), slot, ...(camera.active_shot_idxs || []).slice(position)];
  await writeFile(path.join(workingDir, root, CAMERA_TREE_FILENAME), `${JSON.stringify(cameras, null, 4)}\n`);

  const storyboardPath = path.join(workingDir, root, STORYBOARD_FILENAME);
  let storyboard = [];
  try {
    const parsed = JSON.parse(await readFile(storyboardPath, 'utf8'));
    if (Array.isArray(parsed)) storyboard = parsed;
  } catch {
    storyboard = [];
  }
  storyboard.push({idx: slot, is_last: false, cam_idx: camera.idx, visual_desc: brief, audio_desc: audioDesc});
  storyboard.sort((left, right) => Number(left.idx) - Number(right.idx));
  await writeFile(storyboardPath, `${JSON.stringify(storyboard, null, 4)}\n`);

  const source = JSON.parse(await readFile(path.join(workingDir, root, 'shots', after, SHOT_DESCRIPTION_FILENAME), 'utf8'));
  const shotDir = path.join(workingDir, root, 'shots', String(slot));
  await mkdir(shotDir, {recursive: true});
  // Nothing of the reference shot's script survives the copy: its dialogue and its motion
  // belong to the shot being copied from, and leaving them here makes the render play the
  // wrong lines and move for the wrong shot.
  const created = {...source, idx: slot, is_last: false, ...frames, audio_desc: audioDesc};
  if (motionDesc) created.motion_desc = motionDesc;
  await writeFile(path.join(shotDir, SHOT_DESCRIPTION_FILENAME), `${JSON.stringify(created, null, 4)}\n`);
  if (wasLast) {
    // The shot was added after the film's ending, so the film now ends with the new one.
    await writeFile(path.join(workingDir, root, 'shots', after, SHOT_DESCRIPTION_FILENAME), `${JSON.stringify({...source, is_last: false}, null, 4)}\n`);
    await setLastShot(workingDir, root, slot);
  }
  return {
    ...(await acceptancePayload(repoRoot, workingDir, root)),
    created: {slot: String(slot), copiedFrom: after, copiedPlan: Object.keys(frames).length === 0, plan: reference},
  };
}

/** Mark one shot as the film's ending, clearing the mark anywhere else it sits. */
async function setLastShot(workingDir, root, slot) {
  const cameras = JSON.parse(await readFile(path.join(workingDir, root, CAMERA_TREE_FILENAME), 'utf8'));
  for (const candidate of new Set(cameras.flatMap((camera) => camera.active_shot_idxs || []).map(Number))) {
    const file = path.join(workingDir, root, 'shots', String(candidate), SHOT_DESCRIPTION_FILENAME);
    let description;
    try {
      description = JSON.parse(await readFile(file, 'utf8'));
    } catch {
      continue;
    }
    const shouldBeLast = String(candidate) === String(slot);
    if (Boolean(description.is_last) !== shouldBeLast) {
      await writeFile(file, `${JSON.stringify({...description, is_last: shouldBeLast}, null, 4)}\n`);
    }
  }
}

/**
 * Keep the film's ending marked after a shot leaves it.
 *
 * The mark says where the story ends, which the camera tree does not encode — cameras
 * interleave and a shot's number is its identity, not its place. So a mark that is still
 * standing is left where the plan put it, and only a mark that was removed is put on the
 * highest-numbered shot that remains.
 */
async function markLastShot(workingDir, root, remainingInCamera) {
  const cameras = JSON.parse(await readFile(path.join(workingDir, root, CAMERA_TREE_FILENAME), 'utf8'));
  const present = new Set(cameras.flatMap((camera) => camera.active_shot_idxs || []).map(Number));
  for (const slot of present) {
    const file = path.join(workingDir, root, 'shots', String(slot), SHOT_DESCRIPTION_FILENAME);
    try {
      if (JSON.parse(await readFile(file, 'utf8')).is_last) return remainingInCamera;
    } catch {
      continue;
    }
  }
  if (present.size) await setLastShot(workingDir, root, Math.max(...present));
  return remainingInCamera;
}

async function shotPlanPayload(workingDir, root, slot) {
  const shotDir = path.join(workingDir, root, 'shots', String(slot));
  let description;
  try {
    description = JSON.parse(await readFile(path.join(shotDir, SHOT_DESCRIPTION_FILENAME), 'utf8'));
  } catch {
    // A slot with no plan is not a shot in this root: say so rather than failing obscurely.
    throw acceptanceError(`Unknown shot: ${slot || '(missing)'}`);
  }
  const characters = await readSessionCharacters(workingDir, root);
  const prompt = async (frame) => {
    const cache = path.join(shotDir, `${frame}_selector_output.json`);
    try {
      const cached = JSON.parse(await readFile(cache, 'utf8'));
      return typeof cached.sent_prompt === 'string' ? cached.sent_prompt : '';
    } catch {
      return '';
    }
  };
  return {
    slot: String(slot),
    root,
    // The brief is what the shot is for; the frame descriptions are how it is drawn. Both
    // are shown, and a new shot copies both from the shot it is added after.
    brief: await readBrief(workingDir, root, slot),
    characters,
    firstFrame: {description: String(description.ff_desc || ''), visible: [...(description.ff_vis_char_idxs || [])], prompt: await prompt('first_frame')},
    lastFrame: {description: String(description.lf_desc || ''), visible: [...(description.lf_vis_char_idxs || [])], prompt: await prompt('last_frame')},
    motionDescription: String(description.motion_desc || ''),
  };
}

/** The one-line brief a shot came from, as the storyboard holds it. */
async function readBrief(workingDir, root, slot) {
  try {
    const storyboard = JSON.parse(await readFile(path.join(workingDir, root, STORYBOARD_FILENAME), 'utf8'));
    const entry = Array.isArray(storyboard) ? storyboard.find((shot) => String(shot?.idx) === String(slot)) : null;
    return typeof entry?.visual_desc === 'string' ? entry.visual_desc : '';
  } catch {
    return '';
  }
}

/** The characters a root can draw, from the plan beside its shots. */
async function readSessionCharacters(workingDir, root) {
  try {
    const payload = JSON.parse(await readFile(path.join(workingDir, root, 'characters.json'), 'utf8'));
    if (!Array.isArray(payload)) return [];
    return payload
      .map((character) => ({idx: Number(character.idx), name: String(character.identifier_in_scene || '')}))
      .filter((character) => Number.isFinite(character.idx) && character.name)
      .sort((left, right) => left.idx - right.idx);
  } catch {
    return [];
  }
}
