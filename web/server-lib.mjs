import {createHash, randomUUID} from 'node:crypto';
import {createReadStream} from 'node:fs';
import {lstat, mkdir, open, readFile, readdir, realpath, rename, rm, stat, utimes, writeFile} from 'node:fs/promises';
import path from 'node:path';
import {crc32} from 'node:zlib';
import {readAgentConfig, readClipSettings} from './config-store.mjs';
import {withProjectWriteLock} from './project-lock.mjs';
import {assembleClips} from './film-assembly.mjs';

const IMAGE_EXTENSIONS = new Set(['.png', '.jpg', '.jpeg', '.webp', '.gif']);
const VIDEO_EXTENSIONS = new Set(['.mp4', '.webm', '.mov']);
const MEDIA_EXTENSIONS = new Set([...IMAGE_EXTENSIONS, ...VIDEO_EXTENSIONS]);
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
      const appendToolRound = (round, turnId, createdAt, roundNumber, stableIds) => {
        const requestedTools = Array.isArray(round?.requested_tools) ? round.requested_tools : [];
        for (const [resultIndex, result] of (Array.isArray(round?.tool_results) ? round.tool_results : []).entries()) {
          const callId = typeof requestedTools[resultIndex]?.id === 'string' && requestedTools[resultIndex].id
            ? requestedTools[resultIndex].id
            : '';
          appendMessage({
            id: callId
              ? `activity-${turnId}-${callId}`
              : stableIds
                ? `activity-${turnId}-tool-${roundNumber}-${resultIndex}`
                : `${turnId}-tool-${messageCount}`,
            role: 'activity',
            text: historyToolResultText(result),
            tool: String(result.name || requestedTools[resultIndex]?.name || 'tool'),
            status: result.ok === false ? 'error' : 'done',
            stage: result.ok === false ? 'failed' : 'completed',
            createdAt,
            turnId,
            round: roundNumber - 1,
            ...(callId ? {toolCallId: callId} : {}),
          });
        }
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
        const createdAt = String(record.created_at || record.timestamp || '');
        appendMessage({
          id: `${turnId}-user`,
          role: 'user',
          text: displayUserInput(record.raw_user_input),
          createdAt,
          turnId,
        });
        const assistantTurns = Array.isArray(record.assistant_turns) ? record.assistant_turns : [];
        const hasCanonicalSamples = assistantTurns.length > 0 && assistantTurns.every((sample) => (
          sample && typeof sample === 'object'
          && typeof sample.sample_id === 'string' && sample.sample_id.length > 0
        ));
        if (hasCanonicalSamples) {
          const rounds = Array.isArray(record.tool_rounds) ? record.tool_rounds : [];
          assistantTurns.forEach((sample, index) => {
            const sampleMatch = sample.sample_id.match(/-sample-(\d+)$/);
            const sampleRound = Number.isInteger(sample.round) ? sample.round : Number(sampleMatch?.[1] ?? index);
            const text = typeof sample.text === 'string' ? sample.text : '';
            if (text || sample.outcome) {
              appendMessage({
                id: `assistant-${sample.sample_id}`,
                role: 'assistant',
                text,
                turnId,
                sampleId: sample.sample_id,
                round: sampleRound,
                streamState: sample.outcome === 'tool_preamble' ? 'preamble'
                  : sample.outcome === 'interrupted' ? 'interrupted'
                    : sample.outcome === 'truncated' ? 'truncated' : 'complete',
                createdAt,
              });
            }
            const roundNumber = sampleRound + 1;
            const round = rounds.find((item, roundIndex) => (
              Number.isInteger(item?.tool_round) ? item.tool_round === roundNumber : roundIndex === sampleRound
            ));
            if (round) appendToolRound(round, turnId, createdAt, roundNumber, true);
          });
          const finalText = typeof record.final_assistant_text === 'string' ? record.final_assistant_text : '';
          const matchingFinalSample = assistantTurns.some((sample) => (
            sample.outcome === 'final' && sample.text === finalText
          ));
          if (finalText && !matchingFinalSample) {
            appendMessage({
              id: `assistant-${turnId}`,
              role: record.status === 'failed' ? 'error' : 'assistant',
              text: finalText,
              turnId,
              streamState: record.status === 'completed' ? 'complete' : 'interrupted',
              createdAt,
            });
          }
        } else {
          for (const [roundIndex, round] of (Array.isArray(record.tool_rounds) ? record.tool_rounds : []).entries()) {
            appendToolRound(round, turnId, createdAt, roundIndex + 1, false);
          }
          if (record.final_assistant_text) {
            appendMessage({
              id: `assistant-${turnId}`,
              role: record.status === 'failed' ? 'error' : 'assistant',
              text: String(record.final_assistant_text),
              turnId,
              streamState: record.status === 'failed' ? 'interrupted' : 'complete',
              createdAt,
            });
          }
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

export async function getAssetDeletionPlan(repoRoot, sessionId, requestedPath) {
  const plan = await readAssetDeletionPlan(repoRoot, sessionId, [requestedPath]);
  return plan ? {path: plan.paths[0], revision: plan.revision, impacts: plan.impacts} : null;
}

export async function getAssetsDeletionPlan(repoRoot, sessionId, requestedPaths) {
  if (typeof sessionId !== 'string') throw acceptanceError('sessionId is required');
  const plan = await readAssetDeletionPlan(repoRoot, sessionId, requestedPaths);
  return plan ? {paths: plan.paths, revision: plan.revision, impacts: plan.impacts} : null;
}

async function readAssetDeletionPlan(repoRoot, sessionId, requestedPaths) {
  validateAssetPaths(requestedPaths);
  const record = sessionRecord(await readSessionsPayload(repoRoot), sessionId);
  if (!record) return null;
  const workingDir = sessionWorkingDir(repoRoot, sessionId, record);
  return withProjectWriteLock(workingDir, async () => {
    return assetDeletionPlanUnlocked(workingDir, requestedPaths);
  });
}

export async function deleteAsset(repoRoot, input = {}, isBusy = () => false) {
  if (!isPlainObject(input) || typeof input.sessionId !== 'string'
    || typeof input.path !== 'string' || typeof input.revision !== 'string' || !input.revision) {
    throw acceptanceError('sessionId, path, and revision are required');
  }
  const result = await deleteAssets(repoRoot, {
    sessionId: input.sessionId,
    paths: [input.path],
    revision: input.revision,
  }, isBusy);
  return result && {deletedPath: result.deletedPaths[0], artifacts: result.artifacts};
}

export async function deleteAssets(repoRoot, input = {}, isBusy = () => false) {
  if (!isPlainObject(input) || typeof input.sessionId !== 'string'
    || typeof input.revision !== 'string' || !input.revision) {
    throw acceptanceError('sessionId, paths, and revision are required');
  }
  validateAssetPaths(input.paths);
  const record = sessionRecord(await readSessionsPayload(repoRoot), input.sessionId);
  if (!record) return null;
  const workingDir = sessionWorkingDir(repoRoot, input.sessionId, record);
  return withProjectWriteLock(workingDir, async () => {
    if (isBusy()) throw conflictError(`Wait for active and queued requests to finish before deleting ${input.paths.length === 1 ? 'an asset' : 'assets'}`);
    const plan = await assetDeletionPlanUnlocked(workingDir, input.paths);
    if (plan.revision !== input.revision) {
      throw conflictError('Assets or related metadata changed; refresh the deletion confirmation');
    }
    const changedAcceptance = invalidateAssetApprovals(
      plan.acceptance,
      plan.assetPlans,
      plan.continuityEnabled,
      plan.dependencyDirectories,
      plan.cameraTrees,
      plan.shotDescriptions,
    );
    if (isBusy()) throw conflictError(`A request started or was queued before ${input.paths.length === 1 ? 'the asset could' : 'the assets could'} be deleted; wait for requests to finish`);
    const registryUpdates = new Map();
    for (const asset of plan.assetPlans) {
      for (const reference of asset.registryRefs) {
        let registry = registryUpdates.get(reference.file);
        if (!registry) {
          registry = plan.registries.get(reference.file);
          registryUpdates.set(reference.file, registry);
        }
        const entry = registry[reference.character];
        if (isPlainObject(entry) && isPlainObject(entry[reference.view])
          && typeof entry[reference.view].path === 'string'
          && safeRegistryPortraitPath(workingDir, entry[reference.view].path) === asset.path) {
          delete entry[reference.view];
        }
      }
    }
    await withMutationTransaction(async (transaction) => {
      for (const asset of plan.assetPlans) {
        await assertSafeRegularFile(workingDir, path.resolve(workingDir, asset.path));
      }
      for (const [file, registry] of registryUpdates) {
        await transaction.write(file, `${JSON.stringify(registry, null, 2)}\n`, plan.registryBytes.get(file));
      }
      if (changedAcceptance) {
        await transaction.write(
          plan.acceptancePath,
          `${JSON.stringify(plan.acceptance, null, 2)}\n`,
          plan.acceptanceBytes,
        );
      }
      for (const asset of plan.assetPlans) {
        await transaction.remove(path.resolve(workingDir, asset.path));
      }
    });
    return {
      deletedPaths: plan.paths,
      artifacts: await collectSessionArtifacts(workingDir, input.sessionId, createFilmReadContext()),
    };
  });
}

function validateAssetPaths(requestedPaths) {
  if (!Array.isArray(requestedPaths) || requestedPaths.length === 0
    || requestedPaths.some((requestedPath) => typeof requestedPath !== 'string')) {
    throw acceptanceError('paths must be a non-empty array of session-relative asset paths');
  }
}

async function assetDeletionPlanUnlocked(workingDir, requestedPaths) {
  const paths = [...new Set(requestedPaths.map((requestedPath) => {
    const relative = safeSessionRelativePath(requestedPath);
    if (!MEDIA_EXTENSIONS.has(path.extname(relative).toLowerCase())) {
      throw acceptanceError('Only image and video assets can be deleted');
    }
    return relative;
  }))].sort();
  const assetPlans = [];
  for (const relative of paths) {
    const absolute = path.resolve(workingDir, relative);
    if (!isPathInside(workingDir, absolute)) throw acceptanceError('Asset path escapes the project');
    try { await assertSafeRegularFile(workingDir, absolute); }
    catch (error) {
      if (error.code === 'ENOENT') {
        const missing = new Error('Asset not found');
        missing.statusCode = 404;
        throw missing;
      }
      throw error;
    }
    const [info, hash] = await Promise.all([lstat(absolute), sha256File(absolute)]);
    assetPlans.push({path: relative, info, hash, registryRefs: [], portraitRoots: new Set(), activeShot: false, referencedShots: []});
  }

  const assetsByPath = new Map(assetPlans.map((asset) => [asset.path, asset]));
  const referencedShots = new Map(paths.map((relative) => [relative, new Map()]));
  const [manifestBytes, continuityBytes, frameContinuityBytes] = await Promise.all([
    readSafeMetadata(workingDir, path.join(workingDir, 'render_manifest.json')),
    readSafeMetadata(workingDir, path.join(workingDir, CONTINUITY_FILENAME)),
    readSafeMetadata(workingDir, path.join(workingDir, 'frame_continuity.json')),
  ]);
  const manifest = manifestBytes === null ? {} : parseJson(manifestBytes, 'Invalid render manifest');
  const dependencySnapshot = await listAssetDependencyFiles(workingDir);
  const dependencyFiles = dependencySnapshot.files;
  const registryHashes = [];
  const registries = new Map();
  const registryBytes = new Map();
  const cameraTrees = new Map();
  const shotDescriptions = new Map();
  for (const file of dependencyFiles) {
    const bytes = await readSafeMetadata(workingDir, file);
    if (bytes === null) continue;
    const relative = relativeArtifactPath(workingDir, file);
    registryHashes.push([relative, createHash('sha256').update(bytes).digest('hex')]);
    if (path.basename(file) === 'camera_tree.json') {
      try { cameraTrees.set(relative, JSON.parse(bytes.toString('utf8'))); }
      catch { cameraTrees.set(relative, null); }
      continue;
    }
    const shotDescription = shotDescriptionDependency(relative);
    if (shotDescription) {
      try { shotDescriptions.set(`${shotDescription.root}::${shotDescription.slot}`, JSON.parse(bytes.toString('utf8'))); }
      catch { shotDescriptions.set(`${shotDescription.root}::${shotDescription.slot}`, null); }
      continue;
    }
    if (/^(?:first|last)_frame_selector_output\.json$/.test(path.basename(file))) {
      const selector = parseJson(bytes, 'Invalid frame reference metadata');
      const sources = [
        ...(Array.isArray(selector?.reference_image_path_and_text_pairs)
          ? selector.reference_image_path_and_text_pairs.map((pair) => Array.isArray(pair) ? pair[0] : null) : []),
        selector?.source_frame?.path,
        selector?.continuity_source?.path,
      ];
      for (const source of sources) {
        if (typeof source !== 'string') continue;
        const asset = assetsByPath.get(safeRegistryPortraitPath(workingDir, source));
        if (!asset) continue;
        const shot = shotPathDependency(relative
          .replace(/(?:first|last)_frame_selector_output\.json$/, 'first_frame.png'));
        if (shot) referencedShots.get(asset.path).set(`${shot.root}::${shot.slot}`, {root: shot.root, slot: shot.slot});
      }
      continue;
    }
    if (path.basename(file) !== 'registry.json') continue;
    registryBytes.set(file, bytes);
    const registry = parseJson(bytes, 'Invalid portrait registry metadata');
    if (!isPlainObject(registry)) throw acceptanceError('Portrait registry metadata must be an object');
    registries.set(file, registry);
    for (const [character, record] of Object.entries(registry)) {
      if (!isPlainObject(record)) continue;
      for (const [view, value] of Object.entries(record)) {
        if (!isPlainObject(value) || typeof value.path !== 'string') continue;
        const asset = assetsByPath.get(safeRegistryPortraitPath(workingDir, value.path));
        if (asset) asset.registryRefs.push({file, character, view});
      }
    }
  }
  const acceptancePath = path.join(workingDir, ACCEPTANCE_FILENAME);
  const acceptanceBytes = await readSafeMetadata(workingDir, acceptancePath);
  const acceptance = acceptanceBytes === null ? {} : parseJson(acceptanceBytes, 'Invalid render approval metadata');
  if (!isPlainObject(acceptance)) throw acceptanceError('Render approval metadata must be an object');
  const manifestHash = manifestBytes === null ? null : createHash('sha256').update(manifestBytes).digest('hex');
  const continuityHash = continuityBytes === null ? null : createHash('sha256').update(continuityBytes).digest('hex');
  const frameContinuityHash = frameContinuityBytes === null ? null : createHash('sha256').update(frameContinuityBytes).digest('hex');
  let continuityEnabled = false;
  if (frameContinuityBytes !== null) {
    try {
      continuityEnabled = parseJson(frameContinuityBytes, 'Invalid frame continuity metadata')?.mode === 'chained_keyframes';
    } catch {
      continuityEnabled = false;
    }
  }
  const acceptanceHash = acceptanceBytes === null ? null : createHash('sha256').update(acceptanceBytes).digest('hex');
  const approvalIndex = indexAssetApprovalPaths(acceptance);
  const impacts = [];
  for (const asset of assetPlans) {
    asset.referencedShots = [...referencedShots.get(asset.path).values()];
    asset.portraitRoots = new Set(asset.registryRefs.filter(({file}) =>
      !manifest?.image_model || path.basename(path.dirname(file)) === safePathComponent(manifest.image_model))
      .map(({file}) => relativeArtifactPath(workingDir, file).split('/')[0]));
    asset.approvalPortraitRoots = approvalIndex.portraitRootsByPath.get(asset.path) || new Set();
    const selectedPortrait = asset.approvalPortraitRoots.size > 0 || asset.portraitRoots.size > 0;
    if (asset.registryRefs.length) {
      const count = asset.registryRefs.length;
      impacts.push(`${paths.length === 1 ? '' : `${asset.path}: `}Remove ${count} selected portrait reference${count === 1 ? '' : 's'}; the portrait set becomes incomplete${selectedPortrait ? ' and portrait approvals/dependent outputs will be invalidated' : ''}`);
    }
    if (approvalIndex.selectedPaths.has(asset.path)) {
      impacts.push(`${paths.length === 1 ? '' : `${asset.path}: `}Invalidate approvals that selected this media`);
    }
    const shotDependency = shotPathDependency(asset.path);
    const activeModel = shotDependency?.stages.includes('keyframes') ? manifest?.image_model : manifest?.video_model;
    asset.activeShot = Boolean(shotDependency && (
      !shotDependency.modelDirectory || !activeModel || shotDependency.modelDirectory === safePathComponent(activeModel)
      || approvalIndex.selectedPaths.has(asset.path)
    ));
    const linkedSuccessors = !continuityEnabled && asset.activeShot && shotDependency?.lastFrame
      ? linkedContinuitySuccessors(
        filmOrderFromDependencySnapshot(shotDependency.root, dependencySnapshot.directories, cameraTrees),
        [shotDependency.slot],
        shotReuseChoicesFromSnapshot(shotDependency.root, shotDescriptions),
      )
      : [];
    if (asset.activeShot) {
      const continuityDependents = (continuityEnabled && shotDependency.stages.includes('keyframes'))
        || linkedSuccessors.length > 0;
      impacts.push(`${paths.length === 1 ? '' : `${asset.path}: `}Invalidate ${shotDependency.stages.join(' and ')} approvals and dependent final video${continuityDependents ? ' plus continuity successors' : ''}`);
    }
    if (asset.referencedShots.length) {
      const count = asset.referencedShots.length;
      impacts.push(`${paths.length === 1 ? '' : `${asset.path}: `}Invalidate keyframe, clip, and final approvals for ${count} shot${count === 1 ? '' : 's'} with a recorded reference to this asset.`);
    }
  }
  const revision = createHash('sha256').update(JSON.stringify(paths.length === 1 ? {
    path: paths[0],
    size: assetPlans[0].info.size,
    mtimeMs: assetPlans[0].info.mtimeMs,
    ctimeMs: assetPlans[0].info.ctimeMs,
    hash: assetPlans[0].hash,
    registries: registryHashes,
    acceptance: acceptanceHash,
    manifest: manifestHash,
    continuity: continuityHash,
    frameContinuity: frameContinuityHash,
  } : {
    paths,
    assets: assetPlans.map(({path: assetPath, info, hash}) => ({
      path: assetPath,
      size: info.size,
      mtimeMs: info.mtimeMs,
      ctimeMs: info.ctimeMs,
      hash,
    })),
    registries: registryHashes,
    acceptance: acceptanceHash,
    manifest: manifestHash,
    continuity: continuityHash,
    frameContinuity: frameContinuityHash,
  })).digest('hex');
  return {
    paths,
    revision,
    impacts,
    assetPlans,
    registries,
    acceptance,
    acceptancePath,
    acceptanceBytes,
    registryBytes,
    continuityEnabled,
    dependencyDirectories: dependencySnapshot.directories,
    cameraTrees,
    shotDescriptions,
  };
}

async function listAssetDependencyFiles(root) {
  const found = [];
  const directories = [];
  async function walk(directory) {
    const entries = await readdir(directory, {withFileTypes: true}).catch(() => []);
    for (const entry of entries) {
      if (entry.name.startsWith('.')) continue;
      const file = path.join(directory, entry.name);
      if (entry.isDirectory()) {
        directories.push(relativeArtifactPath(root, file));
        await walk(file);
      } else if (entry.isFile() && (entry.name === 'camera_tree.json'
        || entry.name === SHOT_DESCRIPTION_FILENAME
        || /^(?:first|last)_frame_selector_output\.json$/.test(entry.name)
        || (entry.name === 'registry.json' && path.relative(root, file).split(path.sep).includes('character_portraits')))) {
        found.push(file);
      }
    }
  }
  await walk(root);
  return {files: found.sort(), directories};
}

function shotDescriptionDependency(relative) {
  const match = /^(script2video|idea2video|novel2video)\/(?:scenes\/)?(scene_\d+\/)?shots\/([^/]+)\/shot_description\.json$/i.exec(relative);
  if (!match) return null;
  return {root: match[1], slot: `${match[2] || ''}${match[3]}`};
}

function indexAssetApprovalPaths(store) {
  const selectedPaths = new Set();
  const portraitRootsByPath = new Map();
  function addPaths(record, paths) {
    if (!isPlainObject(record) || !Array.isArray(record.artifacts)) return;
    for (const artifact of record.artifacts) if (typeof artifact?.path === 'string') paths.add(artifact.path);
  }
  for (const [rootName, root] of Object.entries(store)) {
    if (!isPlainObject(root)) continue;
    const portraitPaths = new Set();
    addPaths(root.portraits, portraitPaths);
    if (isPlainObject(root.portrait_characters)) {
      for (const record of Object.values(root.portrait_characters)) addPaths(record, portraitPaths);
    }
    for (const relative of portraitPaths) {
      selectedPaths.add(relative);
      let roots = portraitRootsByPath.get(relative);
      if (!roots) {
        roots = new Set();
        portraitRootsByPath.set(relative, roots);
      }
      roots.add(rootName);
    }
    addPaths(root.final_video, selectedPaths);
    if (isPlainObject(root.shots)) {
      for (const shot of Object.values(root.shots)) if (isPlainObject(shot)) {
        addPaths(shot.keyframes, selectedPaths);
        addPaths(shot.clips, selectedPaths);
      }
    }
  }
  return {selectedPaths, portraitRootsByPath};
}

function shotPathDependency(relative) {
  const match = /^(script2video|idea2video|novel2video)\/(?:scenes\/)?(scene_\d+\/)?shots\/([^/]+)\/(?:([^/]+)\/)?([^/]+)$/i.exec(relative);
  if (!match) return null;
  const root = match[1];
  const slot = `${match[2] || ''}${match[3]}`;
  const file = match[5].toLowerCase();
  const modelDirectory = match[4] || '';
  const isKeyframe = KEYFRAME_NAMES.has(file) || /(?:^|\/)keyframes?\//i.test(relative);
  const isClip = VIDEO_EXTENSIONS.has(path.extname(file));
  if (!isKeyframe && !isClip) return null;
  return {root, slot, stages: isKeyframe ? ['keyframes', 'clips'] : ['clips'], modelDirectory, lastFrame: file === 'last_frame.png'};
}

function filmOrderFromDependencySnapshot(root, directories, cameraTrees) {
  const prefix = `${root}/`;
  const scenes = directories
    .filter((directory) => directory.startsWith(prefix) && /^scene_\d+$/.test(directory.slice(prefix.length)))
    .sort((left, right) => Number(left.slice(prefix.length + 6)) - Number(right.slice(prefix.length + 6)));
  const containers = scenes.length
    ? scenes.map((directory) => ({name: directory.slice(prefix.length), directory}))
    : [{name: '', directory: root}];
  const order = [];
  for (const {name, directory} of containers) {
    const cameras = cameraTrees.get(`${directory}/camera_tree.json`);
    if (!Array.isArray(cameras)) continue;
    order.push(...cameras.slice().sort((left, right) => Number(left.idx) - Number(right.idx))
      .flatMap((camera) => (camera.active_shot_idxs || []).map((shot) => `${name ? `${name}/` : ''}${shot}`)));
  }
  return order;
}

function invalidateAssetApprovals(store, assets, continuityEnabled, dependencyDirectories, cameraTrees, shotDescriptions) {
  let changed = false;
  const now = isoSeconds();
  const targetPaths = new Set(assets.map((asset) => asset.path));
  const dependenciesByRoot = new Map();
  const selectedPortraitRoots = new Set();
  for (const asset of assets) {
    for (const root of asset.portraitRoots) selectedPortraitRoots.add(root);
    for (const root of asset.approvalPortraitRoots) selectedPortraitRoots.add(root);
    const dependencies = [];
    const dependency = shotPathDependency(asset.path);
    if (asset.activeShot && dependency) dependencies.push(dependency);
    dependencies.push(...asset.referencedShots.map((shot) => ({...shot, stages: ['keyframes', 'clips'], lastFrame: true})));
    for (const item of dependencies) {
      let slots = dependenciesByRoot.get(item.root);
      if (!slots) {
        slots = new Map();
        dependenciesByRoot.set(item.root, slots);
      }
      let existing = slots.get(item.slot);
      if (!existing) {
        existing = {slot: item.slot, stages: new Set(), keyframes: false, lastFrame: false};
        slots.set(item.slot, existing);
      }
      for (const stage of item.stages) existing.stages.add(stage);
      if (item.stages.includes('keyframes')) existing.keyframes = true;
      existing.lastFrame ||= item.lastFrame;
    }
  }

  const orders = new Map();
  const reuseChoices = new Map();
  for (const [root, slots] of dependenciesByRoot) {
    if ([...slots.values()].some((item) => item.keyframes)) {
      orders.set(root, filmOrderFromDependencySnapshot(root, dependencyDirectories, cameraTrees));
      reuseChoices.set(root, shotReuseChoicesFromSnapshot(root, shotDescriptions));
    }
  }

  for (const [rootName, root] of Object.entries(store)) {
    if (!isPlainObject(root)) continue;
    let selectedPortrait = selectedPortraitRoots.has(rootName);
    if (isPlainObject(root.portrait_characters)) {
      for (const approval of Object.values(root.portrait_characters)) {
        if (!isPlainObject(approval) || !approvalHasAnyPath(approval, targetPaths)) continue;
        Object.assign(approval, {
          invalidated_at: now,
          invalidation_reason: 'A selected portrait was deleted; choose and review portraits again.',
        });
        selectedPortrait = true;
        changed = true;
      }
    }
    if (isPlainObject(root.portraits) && approvalHasAnyPath(root.portraits, targetPaths)) {
      Object.assign(root.portraits, {
        invalidated_at: now,
        invalidation_reason: 'A selected portrait was deleted; review the portrait set again.',
      });
      selectedPortrait = true;
      changed = true;
    }
    if (selectedPortrait) {
      if (isPlainObject(root.portraits)) {
        Object.assign(root.portraits, {
          invalidated_at: now,
          invalidation_reason: 'A selected portrait was deleted; review the portrait set again.',
        });
      }
      invalidatePortraitDependents(root);
      changed = true;
    }

    const slots = dependenciesByRoot.get(rootName);
    if (slots) {
      const order = orders.get(rootName) || [];
      const choices = reuseChoices.get(rootName) || new Map();
      for (const dependency of slots.values()) {
        const targetSlots = [dependency.slot];
        if (continuityEnabled && dependency.keyframes) {
          targetSlots.push(...continuitySuccessors(order, [dependency.slot], new Set([dependency.slot])));
        } else if (dependency.lastFrame && dependency.keyframes) {
          targetSlots.push(...linkedContinuitySuccessors(order, [dependency.slot], choices));
        }
        for (const slot of targetSlots) {
          const shot = root.shots?.[slot];
          if (!isPlainObject(shot)) continue;
          for (const stage of dependency.stages) if (isPlainObject(shot[stage])) {
            Object.assign(shot[stage], {
              invalidated_at: now,
              invalidation_reason: `A selected ${stage} asset was deleted.`,
            });
            changed = true;
          }
          if (slot !== dependency.slot) {
            for (const stage of ['keyframes', 'clips']) if (isPlainObject(shot[stage])) {
              Object.assign(shot[stage], {
                invalidated_at: now,
                invalidation_reason: 'An earlier keyframe was deleted; review continuous handoffs.',
              });
              changed = true;
            }
          }
        }
        if (isPlainObject(root.final_video)) {
          Object.assign(root.final_video, {
            invalidated_at: now,
            invalidation_reason: 'A selected shot asset was deleted; review the final video again.',
          });
          changed = true;
        }
      }
    }
    if (isPlainObject(root.shots)) {
      for (const record of Object.values(root.shots)) {
        if (!isPlainObject(record)) continue;
        for (const stage of ['keyframes', 'clips']) {
          if (!approvalHasAnyPath(record[stage], targetPaths)) continue;
          Object.assign(record[stage], {
            invalidated_at: now,
            invalidation_reason: `An approved ${stage} asset was deleted.`,
          });
          changed = true;
        }
      }
    }
    if (isPlainObject(root.final_video) && approvalHasAnyPath(root.final_video, targetPaths)) {
      Object.assign(root.final_video, {
        invalidated_at: now,
        invalidation_reason: 'An approved final video asset was deleted.',
      });
      changed = true;
    }
  }
  return changed;
}

function shotReuseChoicesFromSnapshot(root, shotDescriptions) {
  const prefix = `${root}::`;
  const choices = new Map();
  for (const [key, description] of shotDescriptions) {
    if (key.startsWith(prefix)) choices.set(key.slice(prefix.length), description?.reuse_previous_last_frame);
  }
  return choices;
}

function approvalHasAnyPath(record, paths) {
  return isPlainObject(record) && Array.isArray(record.artifacts)
    && record.artifacts.some((item) => paths.has(item?.path));
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
  await annotatePortraitArtifacts(sessionRoot, artifacts, context);
  return [...artifacts.values()].sort((left, right) => right.updatedAt.localeCompare(left.updatedAt));
}

async function annotatePortraitArtifacts(sessionRoot, artifacts, context) {
  const portraits = [...artifacts.entries()].filter(([, artifact]) =>
    /(?:^|\/)character_portraits\/.*\/(?:REMOVED_)?(?:front|side|back)\.png$/i.test(artifact.path));
  if (!portraits.length) return;
  const manifest = await context.json(path.join(sessionRoot, 'render_manifest.json')).catch(() => null);
  const root = PORTRAIT_WORKFLOW_ROOTS.has(manifest?.render_mode) ? manifest.render_mode : '';
  const model = manifest?.image_model ? safePathComponent(manifest.image_model) : '';
  const discovery = root && model ? await discoverPortraits(sessionRoot, root, model) : {
    characters: [], byCharacter: new Map(), portraits: [], complete: false,
  };
  const selected = new Map();
  const store = await readAcceptanceFile(sessionRoot, context);
  const rootStore = isPlainObject(store[root]) ? store[root] : {};
  const legacyAccepted = root && await legacyPortraitApproval(sessionRoot, discovery, rootStore.portraits);
  for (const character of discovery.characters) {
    const accepted = await portraitCharacterApproved(
      sessionRoot,
      rootStore.portrait_characters,
      character,
    ) || Boolean(legacyAccepted);
    for (const [view, relative] of Object.entries(character.pathsByView)) {
      const absolute = path.resolve(sessionRoot, relative);
      selected.set(absolute, {
        character: character.identifier,
        scope: character.scope,
        view,
        registryPath: relativeArtifactPath(sessionRoot, path.join(sessionRoot, character.scope, 'character_portraits', model, 'registry.json')),
        accepted,
      });
    }
  }
  for (const [absolute, artifact] of portraits) {
    const view = artifact.name.toLowerCase().replace(/^removed_/, '').replace('.png', '');
    const active = selected.get(absolute);
    artifact.portrait = {
      ...(active || {}),
      view,
      active: Boolean(active),
      accepted: Boolean(active?.accepted),
    };
    const directory = path.dirname(absolute);
    const standardPrompt = path.join(directory, 'prompts.json');
    const alternatives = [...artifacts.keys()].filter((candidate) =>
      path.dirname(candidate) === directory && /^prompts[^/]*\.json$/i.test(path.basename(candidate)));
    const prompt = artifacts.has(standardPrompt) ? standardPrompt : alternatives.length === 1 ? alternatives[0] : null;
    if (prompt) artifact.promptPath = relativeArtifactPath(sessionRoot, prompt);
  }
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
  const inventory = {portraits: 0, setReferences: 0, shots: 0, frames: 0, clips: 0, finalVideo: false};
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
    if (entry.name === 'set_references') {
      await collectSetReferenceInventory(absolute, inventory);
      continue;
    }
    await collectRenderInventory(absolute, inventory);
  }
}
async function collectSetReferenceInventory(directory, inventory) {
  const sets = await readdir(directory, {withFileTypes: true}).catch(() => []);
  for (const set of sets) {
    if (!set.isDirectory() || !SET_ID_PATTERN.test(set.name)) continue;
    const modelDirectories = await readdir(path.join(directory, set.name), {withFileTypes: true}).catch(() => []);
    for (const model of modelDirectories) {
      if (!model.isDirectory()) continue;
      const image = path.join(directory, set.name, model.name, 'reference.png');
      const info = await lstat(image).catch(() => null);
      if (info?.isFile()) inventory.setReferences += 1;
    }
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
      if (entry.name === 'set_references') {
        await collectSetReferenceArtifacts(root, absolute, artifacts);
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
async function collectSetReferenceArtifacts(root, directory, artifacts) {
  const ids = await readdir(directory, {withFileTypes: true}).catch(() => []);
  for (const id of ids) {
    if (!id.isDirectory() || !SET_ID_PATTERN.test(id.name)) continue;
    const models = await readdir(path.join(directory, id.name), {withFileTypes: true}).catch(() => []);
    for (const model of models) {
      if (!model.isDirectory()) continue;
      const modelDir = path.join(directory, id.name, model.name);
      for (const name of ['reference.png', 'provenance.json']) {
        const absolute = path.join(modelDir, name);
        const info = await lstat(absolute).catch(() => null);
        if (info?.isFile()) artifacts.push(relativeArtifactPath(root, absolute));
      }
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
const PORTRAIT_VIEWS = ['front', 'side', 'back'];
const PORTRAIT_WORKFLOW_ROOTS = new Set(['script2video', 'idea2video']);
const PNG_SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

function isCompletePng(bytes) {
  if (bytes.length < 57 || !bytes.subarray(0, PNG_SIGNATURE.length).equals(PNG_SIGNATURE)) return false;
  let offset = PNG_SIGNATURE.length;
  let hasImageData = false;
  while (offset + 12 <= bytes.length) {
    const length = bytes.readUInt32BE(offset);
    const end = offset + length + 12;
    if (end > bytes.length) return false;
    const type = bytes.toString('ascii', offset + 4, offset + 8);
    if (crc32(bytes.subarray(offset + 4, end - 4)) !== bytes.readUInt32BE(end - 4)) return false;
    if (offset === PNG_SIGNATURE.length
      && (type !== 'IHDR' || length !== 13 || !bytes.readUInt32BE(offset + 8) || !bytes.readUInt32BE(offset + 12))) return false;
    if (type === 'IDAT') hasImageData = true;
    if (type === 'IEND') return length === 0 && hasImageData && end === bytes.length;
    offset = end;
  }
  return false;
}

/**
 * Preview the sibling portrait set containing the clicked image. Paths returned
 * here are session-relative, just like Artifact.path; no disk paths cross the API.
 */
export async function readPortraitOptions(repoRoot, sessionId, sourcePath) {
  const record = sessionRecord(await readSessionsPayload(repoRoot), sessionId);
  if (!record) return null;
  const workingDir = sessionWorkingDir(repoRoot, sessionId, record);
  return portraitOptions(workingDir, sessionId, sourcePath);
}

/** Select one visible character's source trio as its current model-scoped portraits. */
export async function selectPortraitCharacter(repoRoot, input = {}) {
  const body = input && typeof input === 'object' ? input : {};
  const sessionId = typeof body.sessionId === 'string' ? body.sessionId : '';
  const record = sessionRecord(await readSessionsPayload(repoRoot), sessionId);
  if (!record) return null;
  const workingDir = sessionWorkingDir(repoRoot, sessionId, record);
  return withProjectWriteLock(workingDir, async () => {
    const request = portraitRequest(body);
    const current = await portraitWorkflow(workingDir, request.root, request.sourcePath);
    if (current.scope !== request.scope) throw acceptanceError('The source portrait is not in the requested workflow scope');
    const source = await portraitSourceSet(workingDir, request.sourcePath);
    const selectedCharacter = current.scopeState.characters.find((character) =>
      character.identifier_in_scene === request.character && character.is_visible === true);
    if (!selectedCharacter) throw acceptanceError(`Unknown visible character in ${request.scope}: ${request.character}`);
    const revision = await portraitRevision(workingDir, current, source);
    if (revision !== request.revision) throw conflictError('Portrait options changed; refresh and try again');
    if (!source.complete) throw acceptanceError('The selected source set must contain valid front, side, and back PNG images');
    if (!Number.isSafeInteger(selectedCharacter.idx) || selectedCharacter.idx < 0) {
      throw acceptanceError(`Character ${request.character} has an invalid index`);
    }

    const target = await canonicalPortraitSet(workingDir, current, selectedCharacter);
    const selectedKey = `${request.scope}::${request.character}`;
    const active = current.discovery.byCharacter.get(selectedKey);
    const targetMatches = await sameImageSet(workingDir, source.paths, target.paths);
    const activeMatches = active?.complete && samePaths(active.paths, target.pathsArray);
    if (targetMatches && activeMatches) {
      return {
        artifacts: await collectSessionArtifacts(workingDir, sessionId, createFilmReadContext()),
        options: await portraitOptions(workingDir, sessionId, target.paths.front),
      };
    }

    const acceptance = await readAcceptanceFile(workingDir);
    const priorRoot = isPlainObject(acceptance[current.root]) ? acceptance[current.root] : {};
    const legacyPortraits = priorRoot.portraits;
    const legacyApproved = await legacyPortraitApproval(workingDir, current.discovery, legacyPortraits);
    const nextStore = structuredClone(acceptance);
    const nextRoot = isPlainObject(nextStore[current.root]) ? nextStore[current.root] : {};
    const portraitRecords = isPlainObject(nextRoot.portrait_characters) ? nextRoot.portrait_characters : {};
    const oldHashes = active?.complete ? await imageSetHashes(workingDir, active.pathsByView) : null;
    const nextHashes = await imageSetHashes(workingDir, source.paths);
    const bytesChanged = !oldHashes || JSON.stringify(oldHashes) !== JSON.stringify(nextHashes);
    const canonicalChanged = !activeMatches;

    // Promote a valid legacy aggregate before a changed selection invalidates it.
    // Identical bytes remain approved, including the character being re-homed.
    if (legacyApproved) {
      for (const character of current.discovery.characters) {
        const key = `${character.scope}::${character.identifier}`;
        if ((key === selectedKey && bytesChanged) || !character.complete) continue;
        if (!await isExactPortraitApproval(workingDir, portraitRecords[key], character.paths)) {
          portraitRecords[key] = {
            accepted_at: String(legacyPortraits.accepted_at),
            artifacts: await acceptanceArtifacts(workingDir, character.paths),
          };
        }
      }
    }

    await withMutationTransaction(async (transaction) => {
      await ensureSafeParentDirectory(workingDir, target.directory, transaction);
      for (const view of PORTRAIT_VIEWS) {
        await transaction.write(
          path.resolve(workingDir, target.paths[view]),
          await readSafePortrait(workingDir, source.paths[view]),
        );
      }
      const prompts = await savedPortraitPrompts(workingDir, source.directory);
      if (prompts) await transaction.write(path.join(target.directory, 'prompts.json'), Buffer.from(prompts));
      else await transaction.remove(path.join(target.directory, 'prompts.json'));

      const provenance = await portraitProvenance(current.manifest, {
        idx: selectedCharacter.idx ?? null,
        identifier: selectedCharacter.identifier_in_scene,
        visible: selectedCharacter.is_visible === true,
        static_features: selectedCharacter.static_features ?? null,
        dynamic_features: selectedCharacter.dynamic_features ?? null,
      });
      await transaction.write(path.join(target.directory, 'provenance.json'), Buffer.from(JSON.stringify({
        fingerprint: provenance,
        complete: true,
      })));

      const registry = current.scopeState.registry;
      const priorEntry = isPlainObject(registry[request.character]) ? registry[request.character] : {};
      registry[request.character] = Object.fromEntries(PORTRAIT_VIEWS.map((view) => [view, {
        ...(isPlainObject(priorEntry[view]) ? priorEntry[view] : {}),
        path: path.resolve(workingDir, target.paths[view]),
      }]));
      await transaction.write(current.scopeState.registryPath, Buffer.from(`${JSON.stringify(registry, null, 4)}\n`));

      if (bytesChanged) {
        for (const character of current.discovery.characters) {
          if (character.paths.some((relative) => target.pathsArray.includes(relative))) {
            delete portraitRecords[`${character.scope}::${character.identifier}`];
          }
        }
        delete portraitRecords[selectedKey];
        delete nextRoot.portraits;
        invalidatePortraitDependents(nextRoot);
      } else if (canonicalChanged && active?.complete
        && await isExactPortraitApproval(workingDir, portraitRecords[selectedKey], active.paths)) {
        portraitRecords[selectedKey] = {
          accepted_at: portraitRecords[selectedKey].accepted_at,
          artifacts: await acceptanceArtifacts(workingDir, target.pathsArray),
        };
      }
      if (Object.keys(portraitRecords).length) nextRoot.portrait_characters = portraitRecords;
      else delete nextRoot.portrait_characters;
      nextStore[current.root] = nextRoot;

      const updated = await portraitWorkflow(workingDir, current.root, target.paths.front);
      await syncPortraitAggregate(workingDir, updated, nextStore);
      await transaction.write(
        path.join(workingDir, ACCEPTANCE_FILENAME),
        Buffer.from(`${JSON.stringify(nextStore, null, 2)}\n`),
      );
    });

    return {
      artifacts: await collectSessionArtifacts(workingDir, sessionId, createFilmReadContext()),
      options: await portraitOptions(workingDir, sessionId, target.paths.front),
    };
  });
}

/** Approve only the active trio for one visible character. */
export async function approvePortraitCharacter(repoRoot, input = {}) {
  const body = input && typeof input === 'object' ? input : {};
  const sessionId = typeof body.sessionId === 'string' ? body.sessionId : '';
  const record = sessionRecord(await readSessionsPayload(repoRoot), sessionId);
  if (!record) return null;
  const workingDir = sessionWorkingDir(repoRoot, sessionId, record);
  return withProjectWriteLock(workingDir, async () => {
    const request = portraitRequest(body);
    const current = await portraitWorkflow(workingDir, request.root, request.sourcePath);
    if (current.scope !== request.scope) throw acceptanceError('The source portrait is not in the requested workflow scope');
    const source = await portraitSourceSet(workingDir, request.sourcePath);
    const character = current.discovery.byCharacter.get(`${request.scope}::${request.character}`);
    if (!character?.visible) throw acceptanceError(`Unknown visible character in ${request.scope}: ${request.character}`);
    const revision = await portraitRevision(workingDir, current, source);
    if (revision !== request.revision) throw conflictError('Portrait options changed; refresh and try again');
    if (!source.complete || !character.complete || !samePaths(source.pathsArray, character.paths)) {
      throw acceptanceError('Approval requires the source front, side, and back trio to match the active selected trio');
    }

    const store = await readAcceptanceFile(workingDir);
    const rootStore = isPlainObject(store[current.root]) ? store[current.root] : {};
    const portraitRecords = isPlainObject(rootStore.portrait_characters) ? rootStore.portrait_characters : {};
    portraitRecords[`${request.scope}::${request.character}`] = {
      accepted_at: isoSeconds(),
      artifacts: await acceptanceArtifacts(workingDir, character.paths),
    };
    rootStore.portrait_characters = portraitRecords;
    store[current.root] = rootStore;
    await syncPortraitAggregate(workingDir, current, store);
    await withMutationTransaction((transaction) =>
      transaction.write(path.join(workingDir, ACCEPTANCE_FILENAME), `${JSON.stringify(store, null, 2)}\n`));
    return {
      artifacts: await collectSessionArtifacts(workingDir, sessionId, createFilmReadContext()),
      options: await portraitOptions(workingDir, sessionId, character.pathsByView.front),
    };
  });
}

function portraitRequest(body) {
  const root = typeof body.root === 'string' ? body.root : '';
  const scope = typeof body.scope === 'string' ? body.scope : '';
  const character = typeof body.character === 'string' ? body.character : '';
  const sourcePath = typeof body.sourcePath === 'string' ? body.sourcePath : '';
  const revision = typeof body.revision === 'string' ? body.revision : '';
  if (!root || !scope || !character || !sourcePath || !revision) {
    throw acceptanceError('root, scope, character, sourcePath, and revision are required');
  }
  return {root, scope, character, sourcePath, revision};
}

async function portraitOptions(workingDir, sessionId, sourcePath) {
  const workflow = await portraitWorkflow(workingDir, '', sourcePath);
  const source = await portraitSourceSet(workingDir, sourcePath);
  const revision = await portraitRevision(workingDir, workflow, source);
  const store = await readAcceptanceFile(workingDir);
  const legacyAccepted = await legacyPortraitApproval(workingDir, workflow.discovery, store[workflow.root]?.portraits);
  const characters = [];
  for (const item of workflow.scopeState.characters) {
    if (item?.is_visible !== true || typeof item.identifier_in_scene !== 'string') continue;
    const selected = workflow.discovery.byCharacter.get(`${workflow.scope}::${item.identifier_in_scene}`);
    const accepted = await portraitCharacterApproved(
      workingDir,
      store[workflow.root]?.portrait_characters,
      selected,
    ) || Boolean(legacyAccepted && selected?.complete);
    characters.push({
      identifier: item.identifier_in_scene,
      idx: item.idx ?? null,
      scope: workflow.scope,
      active: Boolean(selected?.complete),
      accepted,
    });
  }
  return {
    root: workflow.root,
    revision,
    sourcePath: source.sourcePath,
    paths: source.paths,
    complete: source.complete,
    characters,
  };
}

async function portraitWorkflow(workingDir, requestedRoot, sourcePath) {
  const manifest = await readPortraitJson(workingDir, path.join(workingDir, 'render_manifest.json'), 'render manifest');
  const root = requestedRoot || String(manifest.render_mode || '');
  if (!PORTRAIT_WORKFLOW_ROOTS.has(root)) {
    throw acceptanceError(`Portrait selection is unavailable for render root ${root || '(not pinned)'}`);
  }
  if (manifest.render_mode !== root) throw conflictError('The requested render root is no longer active');
  const model = typeof manifest.image_model === 'string' ? manifest.image_model : '';
  if (!model) throw acceptanceError('The active render has no pinned image model');
  const modelSlug = safePathComponent(model);
  const scope = portraitScopeFromPath(root, sourcePath);
  const scopeState = await readPortraitScope(workingDir, scope, modelSlug);
  if (!scopeState) throw acceptanceError(`Portrait workflow scope is unavailable: ${scope}`);
  const discovery = await discoverPortraits(workingDir, root, modelSlug);
  return {root, manifest, model, modelSlug, scope, scopeState, discovery};
}

function portraitScopeFromPath(root, sourcePath) {
  const relative = safeSessionRelativePath(sourcePath);
  const marker = '/character_portraits/';
  const markerAt = relative.indexOf(marker);
  if (markerAt < 0) throw acceptanceError('sourcePath must identify a portrait image');
  const scope = relative.slice(0, markerAt);
  const suffix = relative.slice(markerAt + marker.length).split('/');
  const validScope = root === 'script2video'
    ? scope === root
    : scope === root || /^idea2video\/scene_\d+$/.test(scope);
  if (!validScope || ![2, 3].includes(suffix.length)
    || !/^(?:REMOVED_)?(?:front|side|back)\.png$/i.test(suffix.at(-1))) {
    throw acceptanceError('sourcePath must identify a front, side, or back portrait in the active render scope');
  }
  return scope;
}

async function readPortraitScope(workingDir, scope, modelSlug) {
  if (!validPortraitScope(scope)) return null;
  const scopeDir = path.join(workingDir, scope);
  const charactersPath = path.join(scopeDir, 'characters.json');
  const charactersRaw = await readSafeMetadata(workingDir, charactersPath);
  const registryPath = path.join(scopeDir, 'character_portraits', modelSlug, 'registry.json');
  const registryRaw = await readSafeMetadata(workingDir, registryPath);
  let registry = {};
  if (registryRaw !== null) {
    registry = parseJson(registryRaw, `Invalid portrait registry in ${scope}`);
    if (!isPlainObject(registry)) throw acceptanceError(`Portrait registry in ${scope} must be an object`);
  }
  if (charactersRaw === null) {
    if (registryRaw === null) return null;
    return {scope, scopeDir, charactersPath, charactersRaw, registryPath, registryRaw, registry, characters: [], invalidCharacters: true};
  }
  const characters = parseJson(charactersRaw, `Invalid character metadata in ${scope}`);
  if (!Array.isArray(characters)) throw acceptanceError(`Character metadata in ${scope} must be a list`);
  return {scope, scopeDir, charactersPath, charactersRaw, registryPath, registryRaw, registry, characters};
}

function validPortraitScope(scope) {
  return scope === 'script2video' || scope === 'idea2video' || /^idea2video\/scene_\d+$/.test(scope);
}

async function discoverPortraits(workingDir, root, modelSlug) {
  const rootScope = await readPortraitScope(workingDir, root, modelSlug);
  const scopes = rootScope ? [rootScope] : [];
  if (root === 'idea2video') {
    const entries = await readdir(path.join(workingDir, root), {withFileTypes: true}).catch(() => []);
    const scenes = entries.filter((entry) => entry.isDirectory() && /^scene_\d+$/.test(entry.name))
      .sort((left, right) => Number(left.name.slice(6)) - Number(right.name.slice(6)));
    for (const entry of scenes) {
      const scene = await readPortraitScope(workingDir, `${root}/${entry.name}`, modelSlug);
      // Planned scenes share the global set until their renderer owns a registry.
      if (scene && scene.registryRaw !== null) scopes.push(scene);
    }
  }
  const characters = [];
  const inventoryCharacters = [];
  const byCharacter = new Map();
  const portraits = new Set();
  let complete = Boolean(rootScope) && scopes.length > 0 && !scopes.some((scope) => scope.invalidCharacters);
  for (const scopeState of scopes) {
    for (const item of scopeState.characters) {
      if (!item || typeof item.identifier_in_scene !== 'string' || !item.identifier_in_scene) continue;
      const visible = item.is_visible === true;
      const entry = isPlainObject(scopeState.registry[item.identifier_in_scene])
        ? scopeState.registry[item.identifier_in_scene] : {};
      const pathsByView = {};
      const viewStates = {};
      const valid = [];
      for (const view of PORTRAIT_VIEWS) {
        const stored = entry[view]?.path;
        if (typeof stored !== 'string' || !stored) {
          viewStates[view] = 'missing';
          valid.push(false);
          continue;
        }
        const relative = safeRegistryPortraitPath(workingDir, stored);
        if (!relative || !relative.toLowerCase().endsWith('.png')) {
          viewStates[view] = 'invalid';
          valid.push(false);
          continue;
        }
        pathsByView[view] = relative;
        let bytes;
        try {
          bytes = await readSafeMetadata(workingDir, path.resolve(workingDir, relative));
        } catch {
          viewStates[view] = 'invalid';
          valid.push(false);
          continue;
        }
        if (bytes !== null && visible) portraits.add(relative);
        const ready = Boolean(bytes && bytes.length > PNG_SIGNATURE.length
          && bytes.subarray(0, PNG_SIGNATURE.length).equals(PNG_SIGNATURE));
        viewStates[view] = bytes === null ? 'missing' : ready ? 'ready' : 'invalid';
        valid.push(ready);
      }
      const paths = PORTRAIT_VIEWS.map((view) => pathsByView[view]).filter(Boolean).sort();
      const isComplete = visible && valid.length === PORTRAIT_VIEWS.length && valid.every(Boolean);
      const character = {
        identifier: item.identifier_in_scene,
        idx: item.idx ?? null,
        scope: scopeState.scope,
        visible,
        description: [item.static_features, item.dynamic_features].filter((value) => typeof value === 'string' && value.trim()).join('\n'),
        paths,
        pathsByView,
        viewStates,
        complete: isComplete,
      };
      inventoryCharacters.push(character);
      if (visible) {
        characters.push(character);
        byCharacter.set(`${scopeState.scope}::${character.identifier}`, character);
      }
      if (visible && !isComplete) complete = false;
    }
  }
  return {scopes, characters, inventoryCharacters, byCharacter, portraits: [...portraits].sort(), complete};
}
async function portraitSourceSet(workingDir, sourcePath) {
  const sourceRelative = safeSessionRelativePath(sourcePath);
  const sourceFile = path.resolve(workingDir, sourceRelative);
  if (!isPathInside(workingDir, sourceFile)) throw acceptanceError('sourcePath escapes the project');
  const name = path.basename(sourceRelative);
  const match = /^(REMOVED_)?(front|side|back)\.png$/i.exec(name);
  if (!match) throw acceptanceError('sourcePath must identify a portrait PNG');
  const prefix = match[1] || '';
  const directory = path.dirname(sourceRelative);
  const paths = Object.fromEntries(PORTRAIT_VIEWS.map((view) => [view, path.join(directory, `${prefix}${view}.png`).split(path.sep).join('/')]));
  const complete = await Promise.all(PORTRAIT_VIEWS.map((view) => validPortraitPng(workingDir, paths[view])));
  return {
    sourcePath: sourceRelative,
    directory: path.resolve(workingDir, directory),
    paths,
    pathsArray: PORTRAIT_VIEWS.map((view) => paths[view]).sort(),
    complete: complete.every(Boolean),
  };
}

async function canonicalPortraitSet(workingDir, workflow, character) {
  const directory = path.join(
    workingDir,
    workflow.scope,
    'character_portraits',
    workflow.modelSlug,
    `${character.idx}_${pythonSafePathComponent(character.identifier ?? character.identifier_in_scene)}`,
  );
  const paths = Object.fromEntries(PORTRAIT_VIEWS.map((view) => [
    view,
    relativeArtifactPath(workingDir, path.join(directory, `${view}.png`)),
  ]));
  const pathsArray = PORTRAIT_VIEWS.map((view) => paths[view]).sort();
  return {directory, paths, pathsArray};
}

async function portraitRevision(workingDir, workflow, source) {
  const approvals = await readAcceptanceFile(workingDir);
  const rootStore = isPlainObject(approvals[workflow.root]) ? approvals[workflow.root] : {};
  const relevantApprovals = {
    portraits: rootStore.portraits ?? null,
    portrait_characters: rootStore.portrait_characters ?? null,
  };
  const sourceHashes = {};
  for (const [view, relative] of Object.entries(source.paths)) {
    const bytes = await readSafePortrait(workingDir, relative).catch(() => null);
    sourceHashes[view] = bytes ? createHash('sha256').update(bytes).digest('hex') : null;
  }
  const revisionScopes = new Map(workflow.discovery.scopes.map((scope) => [scope.scope, scope]));
  revisionScopes.set(workflow.scopeState.scope, workflow.scopeState);
  const scopes = [...revisionScopes.values()].sort((left, right) => left.scope.localeCompare(right.scope)).map((scope) => ({
    scope: scope.scope,
    characters: scope.charactersRaw === null ? null : createHash('sha256').update(scope.charactersRaw).digest('hex'),
    registry: scope.registryRaw === null ? null : createHash('sha256').update(scope.registryRaw).digest('hex'),
  }));
  const activeHashes = {};
  for (const relative of workflow.discovery.portraits) {
    const bytes = await readSafePortrait(workingDir, relative).catch(() => null);
    activeHashes[relative] = bytes ? createHash('sha256').update(bytes).digest('hex') : null;
  }
  const digestInput = {
    root: workflow.root,
    scope: workflow.scope,
    sourcePath: source.sourcePath,
    sourceHashes,
    activeHashes,
    scopes,
    style: workflow.manifest.style || '',
    model: workflow.modelSlug,
    renderMode: workflow.manifest.render_mode,
    approvals: relevantApprovals,
  };
  return createHash('sha256').update(JSON.stringify(digestInput)).digest('hex');
}

async function portraitProvenance(manifest, character) {
  const normalizedCharacter = {
    idx: character.idx ?? null,
    identifier_in_scene: character.identifier ?? null,
    is_visible: character.visible ?? null,
    static_features: character.static_features ?? null,
    dynamic_features: character.dynamic_features ?? null,
  };
  const fingerprintInput = {
    character: normalizedCharacter,
    style: manifest.style || '',
    model: safePathComponent(manifest.image_model || ''),
  };
  return createHash('sha256').update(pythonJson(fingerprintInput), 'utf8').digest('hex');
}

function pythonJson(value) {
  if (Array.isArray(value)) return `[${value.map(pythonJson).join(', ')}]`;
  if (isPlainObject(value)) {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}: ${pythonJson(value[key])}`).join(', ')}}`;
  }
  return JSON.stringify(value);
}

function pythonSafePathComponent(value) {
  return safePathComponent(value);
}

function safeSessionRelativePath(value) {
  const relative = String(value ?? '');
  if (!relative || relative.includes('\\') || path.posix.isAbsolute(relative)
    || path.posix.normalize(relative) !== relative || relative.split('/').some((part) => !part || part === '.' || part === '..')) {
    throw acceptanceError('Portrait paths must be normalized session-relative paths');
  }
  return relative;
}

function safeRegistryPortraitPath(workingDir, stored) {
  const absolute = path.isAbsolute(stored) ? path.resolve(stored)
    : stored.startsWith('.working_dir/') ? path.resolve(workingDir, '../..', stored)
      : path.resolve(workingDir, stored);
  if (!isPathInside(workingDir, absolute)) return null;
  return relativeArtifactPath(workingDir, absolute);
}

function isPathInside(root, absolute) {
  return absolute !== root && absolute.startsWith(`${root}${path.sep}`);
}

async function readSafeMetadata(workingDir, absolute) {
  if (!isPathInside(workingDir, absolute)) throw acceptanceError('Portrait metadata escapes the project');
  try {
    await assertSafeRegularFile(workingDir, absolute);
    return await readFile(absolute);
  } catch (error) {
    if (error.code === 'ENOENT') return null;
    throw error;
  }
}

async function readPortraitJson(workingDir, absolute, label) {
  try {
    await assertSafeRegularFile(workingDir, absolute);
    return parseJson(await readFile(absolute), `Invalid ${label}`);
  } catch (error) {
    if (error.code === 'ENOENT') throw acceptanceError(`The ${label} is unavailable`);
    throw error;
  }
}

function parseJson(value, message) {
  try { return JSON.parse(Buffer.isBuffer(value) ? value.toString('utf8') : String(value)); }
  catch { throw acceptanceError(message); }
}

async function validPortraitPng(workingDir, relative) {
  try {
    const bytes = await readSafePortrait(workingDir, relative);
    return bytes.length > PNG_SIGNATURE.length && bytes.subarray(0, PNG_SIGNATURE.length).equals(PNG_SIGNATURE);
  } catch {
    return false;
  }
}

async function readSafePortrait(workingDir, relative) {
  const normalized = safeSessionRelativePath(relative);
  const absolute = path.resolve(workingDir, normalized);
  if (!isPathInside(workingDir, absolute)) throw acceptanceError('Portrait path escapes the project');
  await assertSafeRegularFile(workingDir, absolute);
  const bytes = await readFile(absolute);
  if (bytes.length <= PNG_SIGNATURE.length || !bytes.subarray(0, PNG_SIGNATURE.length).equals(PNG_SIGNATURE)) {
    throw acceptanceError(`Portrait is not a nonempty PNG image: ${normalized}`);
  }
  return bytes;
}

async function assertSafeRegularFile(root, filePath) {
  const relative = path.relative(root, filePath);
  if (!relative || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) {
    throw acceptanceError('Portrait file escapes the project');
  }
  let current = root;
  const parts = relative.split(path.sep);
  for (let index = 0; index < parts.length; index += 1) {
    current = path.join(current, parts[index]);
    const info = await lstat(current);
    if (info.isSymbolicLink()) throw acceptanceError(`Symbolic links are not allowed in portrait paths: ${relative}`);
    if (index < parts.length - 1 ? !info.isDirectory() : !info.isFile()) {
      throw acceptanceError(`Portrait path is not a regular file: ${relative}`);
    }
  }
  const resolved = await realpath(filePath);
  if (!isPathInside(root, resolved)) throw acceptanceError('Portrait file escapes the project');
}

async function ensureSafeParentDirectory(root, directory, transaction) {
  const relative = path.relative(root, directory);
  if (!relative || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) {
    throw acceptanceError('Portrait destination escapes the project');
  }
  let current = root;
  for (const part of relative.split(path.sep)) {
    current = path.join(current, part);
    try {
      const info = await lstat(current);
      if (info.isSymbolicLink() || !info.isDirectory()) throw acceptanceError(`Unsafe portrait destination: ${path.relative(root, current)}`);
    } catch (error) {
      if (error.code !== 'ENOENT') throw error;
      await transaction.mkdir(current);
    }
  }
}

async function sameImageSet(workingDir, sourcePaths, targetPaths) {
  for (const view of PORTRAIT_VIEWS) {
    const source = await readSafePortrait(workingDir, sourcePaths[view]).catch(() => null);
    const target = await readSafePortrait(workingDir, targetPaths[view]).catch(() => null);
    if (!source || !target || !source.equals(target)) return false;
  }
  return true;
}

async function imageSetHashes(workingDir, pathsByView) {
  const hashes = {};
  for (const view of PORTRAIT_VIEWS) {
    const bytes = await readSafePortrait(workingDir, pathsByView[view]);
    hashes[view] = createHash('sha256').update(bytes).digest('hex');
  }
  return hashes;
}

function samePaths(left, right) {
  return Array.isArray(left) && Array.isArray(right)
    && JSON.stringify([...left].sort()) === JSON.stringify([...right].sort());
}

async function savedPortraitPrompts(workingDir, sourceDirectory) {
  const entries = await readdir(sourceDirectory, {withFileTypes: true}).catch(() => []);
  const standard = entries.find((entry) => entry.isFile() && entry.name === 'prompts.json');
  const alternatives = entries.filter((entry) => entry.isFile() && /^prompts[^/]*\.json$/i.test(entry.name));
  const selected = standard || (alternatives.length === 1 ? alternatives[0] : null);
  if (!selected) return null;
  const absolute = path.join(sourceDirectory, selected.name);
  try {
    const content = parseJson(await readSafeMetadata(workingDir, absolute), 'Invalid saved portrait prompts');
    if (!isPlainObject(content)) return null;
    const prompts = Object.fromEntries(PORTRAIT_VIEWS.filter((view) => typeof content[view] === 'string' && content[view])
      .map((view) => [view, content[view]]));
    return Object.keys(prompts).length ? `${JSON.stringify(prompts, null, 2)}\n` : null;
  } catch {
    return null;
  }
}

async function isExactPortraitApproval(workingDir, record, paths) {
  if (!Array.isArray(paths) || paths.length === 0 || !isPlainObject(record)
    || typeof record.accepted_at !== 'string' || !record.accepted_at
    || record.invalidated_at || record.rejected_at
    || !Array.isArray(record.artifacts) || record.artifacts.length !== paths.length) return false;
  const expected = [...paths].sort();
  const actual = record.artifacts.map((artifact) => artifact?.path).sort();
  if (!samePaths(expected, actual)) return false;
  return acceptanceLockMatches(workingDir, record, createFilmReadContext());
}
async function portraitCharacterApproved(workingDir, records, character) {
  if (!character?.complete || !isPlainObject(records)) return false;
  const ownKey = `${character.scope}::${character.identifier}`;
  if (await isExactPortraitApproval(workingDir, records[ownKey], character.paths)) return true;
  for (const [key, record] of Object.entries(records)) {
    if (key !== ownKey && key.endsWith(`::${character.identifier}`)
      && await isExactPortraitApproval(workingDir, record, character.paths)) return true;
  }
  return false;
}


async function legacyPortraitApproval(workingDir, discovery, record) {
  return discovery.complete && discovery.portraits.length > 0
    && await isExactPortraitApproval(workingDir, record, discovery.portraits);
}

async function syncPortraitAggregate(workingDir, workflow, store) {
  const rootStore = isPlainObject(store[workflow.root]) ? store[workflow.root] : {};
  const records = isPlainObject(rootStore.portrait_characters) ? rootStore.portrait_characters : {};
  if (!workflow.discovery.complete || !workflow.discovery.characters.length) {
    delete rootStore.portraits;
  } else {
    let allAccepted = true;
    for (const character of workflow.discovery.characters) {
      if (!await portraitCharacterApproved(workingDir, records, character)) {
        allAccepted = false;
        break;
      }
    }
    if (allAccepted) {
      rootStore.portraits = {
        accepted_at: isoSeconds(),
        artifacts: await acceptanceArtifacts(workingDir, workflow.discovery.portraits),
      };
    } else {
      delete rootStore.portraits;
    }
  }
  store[workflow.root] = rootStore;
}

function invalidatePortraitDependents(rootStore) {
  const now = isoSeconds();
  if (isPlainObject(rootStore.final_video)) {
    rootStore.final_video = {
      ...rootStore.final_video,
      invalidated_at: now,
      invalidation_reason: 'A selected character portrait changed; review the final video again.',
    };
  }
  if (isPlainObject(rootStore.shots)) {
    for (const shot of Object.values(rootStore.shots)) {
      if (!isPlainObject(shot)) continue;
      for (const stage of ['keyframes', 'clips']) {
        if (isPlainObject(shot[stage])) {
          shot[stage] = {
            ...shot[stage],
            invalidated_at: now,
            invalidation_reason: 'A selected character portrait changed; review this output again.',
          };
        }
      }
    }
  }
}

/** One coherent read budget for the film, without re-entering mutation locks. */
export async function readFilmSnapshot(repoRoot, sessionId, root = '') {
  const record = sessionRecord(await readSessionsPayload(repoRoot), sessionId);
  if (!record) return null;
  const workingDir = sessionWorkingDir(repoRoot, sessionId, record);
  const context = createFilmReadContext();
  const resolved = await resolveRenderRoot(repoRoot, workingDir, root, context);
  const discovery = await collectAcceptanceState(workingDir, resolved, context);
  const sets = await readSetReferences(repoRoot, sessionId, resolved);
  const setArtifacts = (sets?.sets ?? []).flatMap((set) => {
    const base = path.join(workingDir, resolved, 'set_references', set.id);
    return [path.join(base, 'set.json'), ...(['reference.png', 'provenance.json'].map((name) =>
      path.join(base, safePathComponent(sets.imageModel), name)))];
  });
  const referenced = [
    ...discovery.portraits,
    ...discovery.finalVideo,
    ...discovery.shots.flatMap((shot) => [...shot.keyframes, ...shot.clips]),
    ...setArtifacts,
  ].map((relative) => path.isAbsolute(relative) ? relative : path.resolve(workingDir, relative));
  const [acceptance, plans, removed, continuity, artifacts] = await Promise.all([
    acceptancePayload(repoRoot, workingDir, resolved, context, discovery),
    shotPlansPayload(workingDir, resolved, discovery, context, sets),
    listRemoved(workingDir, resolved, context),
    continuityPayload(repoRoot, workingDir, resolved, record, context, discovery),
    collectSessionArtifacts(resolveSessionRoot(repoRoot, sessionId), sessionId, context, referenced),
  ]);
  return {acceptance, plans, removed, continuity, artifacts, sets};
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
    if (stage === 'portraits') {
      if (!discovery.portraitComplete || !discovery.portraitCharacters.length) {
        throw acceptanceError('Cannot accept portraits: every visible character needs a complete active front, side, and back set');
      }
      const acceptedAt = isoSeconds();
      recordSlot(store, root, null, stage, {
        accepted_at: acceptedAt,
        artifacts: await acceptanceArtifacts(workingDir, paths),
      });
      const rootStore = isPlainObject(store[root]) ? store[root] : {};
      const portraitCharacters = isPlainObject(rootStore.portrait_characters) ? rootStore.portrait_characters : {};
      for (const character of discovery.portraitCharacters) {
        portraitCharacters[`${character.scope}::${character.identifier}`] = {
          accepted_at: acceptedAt,
          artifacts: await acceptanceArtifacts(workingDir, character.paths),
        };
      }
      rootStore.portrait_characters = portraitCharacters;
      store[root] = rootStore;
    } else {
      recordSlot(store, root, slot, stage, {
        accepted_at: isoSeconds(),
        artifacts: await acceptanceArtifacts(workingDir, paths),
      });
    }
  } else if (reason) {
    if (reason.length > REASON_MAX_LENGTH) throw acceptanceError(`reason must be at most ${REASON_MAX_LENGTH} characters`);
    recordSlot(store, root, slot, stage, {rejected_at: isoSeconds(), reason});
  } else {
    removeAcceptance(store, root, slot, stage);
    if (stage === 'portraits') {
      const rootStore = isPlainObject(store[root]) ? store[root] : {};
      delete rootStore.portrait_characters;
      if (Object.keys(rootStore).length) store[root] = rootStore;
      else delete store[root];
    }
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
      let state = await slotState(workingDir, slotRecord, paths, context);
      if (stage === 'portraits') {
        if (!discovery.portraitComplete) state = paths.length ? 'stale' : 'planned';
        else if (state === 'accepted' && !await legacyPortraitApproval(workingDir, {
          complete: discovery.portraitComplete,
          portraits: paths,
        }, slotRecord)) state = 'stale';
      }
      slots = [slotPayload(null, state, paths, slotRecord)];
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
    characters: discovery.characters,
    portraitImageModel: discovery.portraitImageModel,
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
  const cleaned = String(name ?? '').replace(/[^\p{L}\p{N}\p{M}\p{Pc}\-. ]/gu, '_').trim().replace(/^\.+/, '');
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
  const configured = manifest?.image_model ? null : await readAgentConfig(path.resolve(workingDir, '..', '..')).catch(() => null);
  const imageModel = manifest?.image_model || configured?.sections?.image?.model || '';
  const imageModelDir = safePathComponent(imageModel);
  const videoModelDir = safePathComponent(manifest?.video_model ?? '');

  const portraitState = PORTRAIT_WORKFLOW_ROOTS.has(root)
    ? await discoverPortraits(workingDir, root, imageModelDir)
    : {portraits: [], characters: [], inventoryCharacters: [], complete: false};
  const portraits = portraitState.portraits;

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

  const acceptance = await readAcceptanceFile(workingDir, context);
  const approvalRoot = isPlainObject(acceptance[root]) ? acceptance[root] : {};
  const approvalRecords = isPlainObject(approvalRoot.portrait_characters) ? approvalRoot.portrait_characters : {};
  const legacyApproved = await legacyPortraitApproval(workingDir, portraitState, approvalRoot.portraits);
  const characters = [];
  for (const character of portraitState.inventoryCharacters) {
    const key = `${character.scope}::${character.identifier}`;
    const record = approvalRecords[key];
    const approved = character.visible && (await portraitCharacterApproved(workingDir, approvalRecords, character)
      || Boolean(legacyApproved && character.complete));
    const rejected = Boolean(record?.rejected_at);
    const views = Object.fromEntries(PORTRAIT_VIEWS.map((view) => [view, {
      ...(character.pathsByView[view] ? {path: character.pathsByView[view]} : {}),
      state: character.viewStates[view],
    }]));
    characters.push({
      identifier: character.identifier,
      scope: character.scope,
      description: character.description,
      visible: character.visible,
      complete: character.complete,
      state: !character.visible ? 'planned' : rejected ? 'rejected' : approved ? 'accepted'
        : !character.paths.length ? 'planned'
          : character.complete ? 'rendered' : 'stale',
      views,
    });
  }
  return {
    portraits,
    portraitCharacters: portraitState.characters,
    portraitComplete: portraitState.complete,
    characters,
    portraitImageModel: imageModel,
    finalVideo,
    shots,
    order: playing,
  };
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
    async write(filePath, contents, previousSnapshot) {
      let previous = null;
      let previousMode;
      let previousTimes;
      try {
        const info = await lstat(filePath);
        if (!info.isFile()) throw acceptanceError(`Cannot replace non-file metadata: ${path.basename(filePath)}`);
        if (previousSnapshot === null) throw conflictError('Related metadata changed; refresh the deletion confirmation');
        previous = previousSnapshot === undefined ? await readFile(filePath) : previousSnapshot;
        previousMode = info.mode & 0o777;
        previousTimes = {atime: info.atime, mtime: info.mtime};
      } catch (error) {
        if (error.code !== 'ENOENT') throw error;
        if (previousSnapshot !== undefined && previousSnapshot !== null) {
          throw conflictError('Related metadata changed; refresh the deletion confirmation');
        }
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
async function filmPlaybackContext(workingDir, root, context = createFilmReadContext(), includeCameraOwners = true) {
  const rootDir = path.join(workingDir, root);
  const entries = await context.entries(rootDir).catch(() => []);
  const scenes = entries.filter((entry) => entry.isDirectory() && /^scene_\d+$/.test(entry.name))
    .sort((left, right) => Number(left.name.slice(6)) - Number(right.name.slice(6)));
  const containers = scenes.length ? scenes.map((entry) => ({name: entry.name, dir: path.join(rootDir, entry.name)}))
    : [{name: '', dir: rootDir}];
  const order = [];
  const cameraBySlot = includeCameraOwners ? new Map() : null;
  for (const {name, dir} of containers) {
    let cameras;
    try { cameras = await context.json(path.join(dir, CAMERA_TREE_FILENAME)); } catch { continue; }
    if (!Array.isArray(cameras)) continue;
    const orderedCameras = cameras.slice().sort((left, right) => Number(left.idx) - Number(right.idx));
    for (const [index, camera] of orderedCameras.entries()) {
      const owner = includeCameraOwners ? `${name}/${String(camera.idx ?? index)}` : '';
      for (const shot of camera.active_shot_idxs || []) {
        const slot = `${name ? `${name}/` : ''}${shot}`;
        order.push(slot);
        if (cameraBySlot) cameraBySlot.set(slot, owner);
      }
    }
  }
  return {order, cameraBySlot};
}

async function filmOrder(workingDir, root, context = createFilmReadContext()) {
  return (await filmPlaybackContext(workingDir, root, context, false)).order;
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
  if (typeof review.set_references_invalidated_at === 'string' && review.set_references_invalidated_at) {
    return {stale: true, reason: review.set_references_invalidation_reason || 'A set reference changed; review continuity again.'};
  }
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
  const setDirectory = path.join(rootDir, 'set_references');
  const setEntries = await context.entries(setDirectory).catch(() => []);
  for (const entry of setEntries) {
    if (!entry.isDirectory() || !SET_ID_PATTERN.test(entry.name)) continue;
    const setDir = path.join(setDirectory, entry.name);
    add(path.join(setDir, 'set.json'));
    const modelDirs = await context.entries(setDir).catch(() => []);
    for (const model of modelDirs) {
      if (!model.isDirectory() || model.name === 'set.json') continue;
      for (const name of ['reference.png', 'provenance.json']) add(path.join(setDir, model.name, name));
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
  const playback = await filmPlaybackContext(workingDir, resolved, context);
  const reuseContext = {playback, continuityEnabled: await frameContinuityEnabled(workingDir)};
  const discovery = await collectAcceptanceState(workingDir, resolved, context);
  const setInventory = await readSetReferences(repoRoot, sessionId, resolved);
  return {root: resolved, plans: await shotPlansPayload(workingDir, resolved, discovery, context, setInventory, reuseContext)};
}

async function shotPlansPayload(workingDir, root, discovery, context, setInventory, reuseContext) {
  const plans = [];
  for (const {shot} of discovery.shots) {
    try { plans.push(await shotPlanPayload(workingDir, root, shot, context, setInventory, reuseContext)); }
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
  const context = createFilmReadContext();
  const playback = await filmPlaybackContext(workingDir, resolved, context);
  const reuseContext = {playback, continuityEnabled: await frameContinuityEnabled(workingDir)};
  const inventory = await readSetReferences(repoRoot, sessionId, resolved);
  return shotPlanPayload(workingDir, resolved, slot, context, inventory, reuseContext);
}

/** Rewrite one shot's plan: guidance, frame descriptions, and visible characters. */
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
  const context = createFilmReadContext();
  const playback = await filmPlaybackContext(workingDir, root, context);
  const reuseContext = {playback, continuityEnabled: await frameContinuityEnabled(workingDir)};
  const discovery = await collectAcceptanceState(workingDir, root, context);
  if (!discovery.shots.some((entry) => entry.shot === slot)) throw acceptanceError(`Unknown shot: ${slot || '(missing)'}`);
  if (Object.hasOwn(body, 'reusePreviousLastFrame')) {
    if (typeof body.reusePreviousLastFrame !== 'boolean') {
      throw acceptanceError('reusePreviousLastFrame must be a boolean');
    }
    if (!Object.hasOwn(body, 'previousSlot')
      || (body.previousSlot !== null && typeof body.previousSlot !== 'string')) {
      throw acceptanceError('previousSlot must be a string or null when changing previous-frame reuse');
    }
    const index = playback.order.indexOf(slot);
    const actualPreviousSlot = index > 0 ? playback.order[index - 1] : null;
    if (body.previousSlot !== actualPreviousSlot) {
      throw conflictError('Film playback order changed; refresh the shot before changing previous-frame reuse');
    }
    if (body.reusePreviousLastFrame && actualPreviousSlot === null) {
      throw acceptanceError('The first shot in the film has no previous last frame to reuse');
    }
  }
  const setInventory = await readSetReferences(repoRoot, sessionId, root);
  if (Object.hasOwn(body, 'setIds')) {
    if (body.setRevision !== setInventory?.revision) throw conflictError('Set references changed; refresh before saving shot bindings');
    if (!Array.isArray(body.setIds) || !body.setIds.every((id) => typeof id === 'string') || new Set(body.setIds).size !== body.setIds.length) {
      throw acceptanceError('setIds must be a unique list of set identifiers');
    }
    const known = new Set((setInventory?.sets ?? []).map(({id}) => id));
    const unknown = body.setIds.filter((id) => !SET_ID_PATTERN.test(id) || !known.has(id));
    if (unknown.length) throw acceptanceError(`Unknown set identifier: ${unknown.join(', ')}`);
  }
  const plan = await shotPlanPayload(workingDir, root, slot, context, setInventory, reuseContext);

  const edits = {};
  if (Object.hasOwn(body, 'setIds')) edits.set_ids = [...body.setIds];
  if (Object.hasOwn(body, 'reusePreviousLastFrame')) {
    edits.reuse_previous_last_frame = body.reusePreviousLastFrame;
  }
  for (const field of ['brief', 'motionDesc']) {
    if (!Object.hasOwn(body, field)) continue;
    if (typeof body[field] !== 'string') throw acceptanceError(`${field === 'brief' ? 'Brief' : 'Motion description'} must be text`);
    const value = body[field].trim();
    const limit = field === 'brief' ? MAX_BRIEF_LENGTH : MAX_DESCRIPTION_LENGTH;
    const label = field === 'brief' ? 'Brief' : 'Motion description';
    if (!value) throw acceptanceError(`${label} cannot be empty`);
    if (value.length > limit) throw acceptanceError(`${label} must be at most ${limit} characters`);
    if (field === 'brief') edits.visual_desc = value;
    if (field === 'motionDesc') edits.motion_desc = value;
  }
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
  const storyboard = Object.hasOwn(body, 'brief') ? await readJsonOptional(location.storyboardPath) : null;
  if (Object.hasOwn(body, 'brief') && !Array.isArray(storyboard)) throw acceptanceError('The storyboard is not a list');
  const storyboardIndex = Array.isArray(storyboard) ? storyboard.findIndex((entry) => String(entry?.idx) === location.local) : -1;
  if (Object.hasOwn(body, 'brief') && storyboardIndex < 0) throw acceptanceError(`Shot ${slot} has no storyboard brief`);
  const briefTarget = path.join(location.shotDir, 'brief.json');
  const currentBrief = Object.hasOwn(body, 'brief') ? await readJsonOptional(briefTarget) : null;
  if (Object.hasOwn(body, 'brief')) {
    storyboard[storyboardIndex] = {...storyboard[storyboardIndex], visual_desc: body.brief.trim()};
    if (isPlainObject(currentBrief)) currentBrief.visual_desc = body.brief.trim();
  }
  const sourceEndingChanged = ['lf_desc', 'lf_vis_char_idxs'].some(
    (key) => Object.hasOwn(edits, key) && JSON.stringify(current[key]) !== JSON.stringify(edits[key]),
  );
  const effectiveSetIdsChanged = Object.hasOwn(body, 'setIds')
    && JSON.stringify(plan.setIds) !== JSON.stringify(body.setIds);
  const reuseChoiceChanged = Object.hasOwn(body, 'reusePreviousLastFrame')
    && current.reuse_previous_last_frame !== body.reusePreviousLastFrame;
  const invalidateOutputs = Object.keys(edits).some((key) => !['set_ids', 'reuse_previous_last_frame'].includes(key))
    || effectiveSetIdsChanged || reuseChoiceChanged;
  await withMutationTransaction(async (transaction) => {
    if (Object.keys(edits).length) await transaction.write(target, `${JSON.stringify({...current, ...edits}, null, 4)}\n`);
    if (Object.hasOwn(body, 'brief')) {
      await transaction.write(location.storyboardPath, `${JSON.stringify(storyboard, null, 4)}\n`);
      if (isPlainObject(currentBrief)) await transaction.write(briefTarget, `${JSON.stringify(currentBrief, null, 4)}\n`);
    }
    if (invalidateOutputs) {
      await invalidateTimelineOutputs(transaction, workingDir, root, [slot], true, {
        continuitySourceSlots: sourceEndingChanged || effectiveSetIdsChanged || reuseChoiceChanged ? [slot] : [],
        forceContinuity: reuseChoiceChanged,
        preserveMedia: reuseChoiceChanged,
      });
    }
  });
  const updatedInventory = await readSetReferences(repoRoot, sessionId, root);
  const updatedContext = createFilmReadContext();
  const updatedPlayback = await filmPlaybackContext(workingDir, root, updatedContext);
  return shotPlanPayload(workingDir, root, slot, updatedContext, updatedInventory, {
    playback: updatedPlayback,
    continuityEnabled: await frameContinuityEnabled(workingDir),
  });

}
/** Preview a shot reset against the exact outputs and dependency metadata on disk. */
export async function getShotRestartPlan(repoRoot, input = {}) {
  return withTimelineMutation(repoRoot, input, async () => {
    const record = sessionRecord(await readSessionsPayload(repoRoot), input.sessionId);
    const workingDir = sessionWorkingDir(repoRoot, input.sessionId, record);
    return shotRestartPlanUnlocked(repoRoot, workingDir, input);
  });
}

async function shotRestartPlanUnlocked(repoRoot, workingDir, input) {
  const root = await resolveRenderRoot(repoRoot, workingDir, input.root);
  const slot = String(input.slot ?? '');
  const location = shotLocation(workingDir, root, slot);
  await assertSafeRegularFile(workingDir, path.join(location.shotDir, SHOT_DESCRIPTION_FILENAME));
  if (!(await filmOrder(workingDir, root)).includes(slot)) {
    throw acceptanceError(`Shot ${slot} is not in the active film`);
  }
  const paths = [];
  async function collect(directory, preservePlan = false) {
    let entries;
    try { entries = await readdir(directory, {withFileTypes: true}); }
    catch (error) { if (error.code === 'ENOENT') return; throw error; }
    for (const entry of entries) {
      const file = path.join(directory, entry.name);
      if (entry.isSymbolicLink()) throw acceptanceError('Shot restart refuses symbolic links');
      if (entry.isDirectory()) await collect(file);
      else if (entry.isFile()) {
        await assertSafeRegularFile(workingDir, file);
        if (preservePlan && [SHOT_DESCRIPTION_FILENAME, 'brief.json'].includes(entry.name)) continue;
        paths.push(path.relative(workingDir, file).split(path.sep).join('/'));
      } else throw acceptanceError('Shot restart refuses non-regular files');
    }
  }
  await collect(location.shotDir, true);
  const backupShot = path.join(workingDir, '.redo_backup', path.relative(workingDir, location.shotDir));
  // A failed future redraw must not resurrect the outputs the user explicitly reset.
  try {
    const info = await lstat(backupShot);
    if (!info.isDirectory() || info.isSymbolicLink()) throw acceptanceError('Unsafe shot redraw backup');
    await collect(backupShot);
  } catch (error) { if (error.code !== 'ENOENT') throw error; }
  const films = new Set([path.join(workingDir, root, 'final_video.mp4')]);
  if (location.scene) films.add(path.join(location.container, 'final_video.mp4'));
  for (const film of [...films]) films.add(path.join(workingDir, '.redo_backup', path.relative(workingDir, film)));
  for (const film of films) {
    try {
      await assertSafeRegularFile(workingDir, film);
      paths.push(path.relative(workingDir, film).split(path.sep).join('/'));
    } catch (error) { if (error.code !== 'ENOENT') throw error; }
  }
  paths.sort();
  const metadata = [
    path.join(location.shotDir, SHOT_DESCRIPTION_FILENAME),
    path.join(location.shotDir, 'brief.json'),
    location.treePath, location.storyboardPath,
    path.join(workingDir, ACCEPTANCE_FILENAME),
    path.join(workingDir, 'render_manifest.json'),
    path.join(workingDir, 'frame_continuity.json'),
  ];
  const fingerprints = [];
  for (const file of [...paths.map((relative) => path.join(workingDir, relative)), ...metadata]) {
    try {
      await assertSafeRegularFile(workingDir, file);
      fingerprints.push([path.relative(workingDir, file), await sha256File(file)]);
    } catch (error) {
      if (error.code !== 'ENOENT') throw error;
      fingerprints.push([path.relative(workingDir, file), null]);
    }
  }
  const continuityEnabled = await frameContinuityEnabled(workingDir);
  const revision = createHash('sha256').update(JSON.stringify([
    input.sessionId, root, slot, paths, fingerprints, await filmOrder(workingDir, root),
  ])).digest('hex');
  return {sessionId: input.sessionId, root, slot, revision, paths, continuityEnabled};
}

export async function restartShot(repoRoot, input = {}, isBusy = () => false) {
  if (!isPlainObject(input) || typeof input.sessionId !== 'string'
    || typeof input.root !== 'string' || !input.root || typeof input.slot !== 'string' || !input.slot
    || typeof input.revision !== 'string' || !input.revision) {
    throw acceptanceError('sessionId, root, slot, and revision are required');
  }
  return withTimelineMutation(repoRoot, input, async () => {
    if (isBusy()) throw conflictError('Wait for active and queued requests to finish before restarting a shot');
    const record = sessionRecord(await readSessionsPayload(repoRoot), input.sessionId);
    const workingDir = sessionWorkingDir(repoRoot, input.sessionId, record);
    const plan = await shotRestartPlanUnlocked(repoRoot, workingDir, input);
    if (plan.revision !== input.revision) {
      throw conflictError('Shot outputs or related metadata changed; reopen the restart confirmation');
    }
    if (isBusy()) throw conflictError('A request started or was queued; wait before restarting a shot');
    await withMutationTransaction(async (transaction) => {
      for (const relative of plan.paths) await transaction.remove(path.join(workingDir, relative));
      await invalidateTimelineOutputs(transaction, workingDir, plan.root, [plan.slot], true, {
        continuitySourceSlots: [plan.slot],
      });
    });
    return {ok: true, restarted: plan.slot, root: plan.root};
  });
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
  const setInventory = await readSetReferences(repoRoot, sessionId, root);
  const reference = await shotPlanPayload(workingDir, root, after, createFilmReadContext(), setInventory);
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
  const created = {...source, idx: localSlot, is_last: false, cam_idx: camera.idx, set_ids: [...reference.setIds], ...frames, audio_desc: audioDesc};
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
  {orderBefore, continuitySourceSlots = [], forceContinuity = false, preserveMedia = false} = {},
) {
  const continuityEnabled = await frameContinuityEnabled(workingDir);
  const store = await readAcceptanceFile(workingDir);
  const before = JSON.stringify(store);
  const rootStore = isPlainObject(store[root]) ? store[root] : {};
  const shotStore = isPlainObject(rootStore.shots) ? rootStore.shots : {};
  const now = isoSeconds();
  let playbackOrder = [];
  let linkedInvalidatedSlots = [];
  if (!continuityEnabled && (Array.isArray(orderBefore) || continuitySourceSlots.length)) {
    playbackOrder = await filmOrder(workingDir, root);
    const choices = await timelineReuseChoices(workingDir, root, [...new Set([...playbackOrder, ...(orderBefore || [])])]);
    linkedInvalidatedSlots = Array.isArray(orderBefore)
      ? changedLinkedContinuityTargets(orderBefore, playbackOrder, choices)
      : linkedContinuitySuccessors(playbackOrder, continuitySourceSlots, choices);
  }

  const preserveApprovalContinuity = continuityEnabled || forceContinuity || linkedInvalidatedSlots.length > 0;
  let invalidatedSlots = [];
  if (continuityEnabled) {
    if (Array.isArray(orderBefore)) {
      playbackOrder = await filmOrder(workingDir, root);
      const changedSuffix = timelineOrderChangedSuffix(orderBefore, playbackOrder);
      invalidatedSlots = [
        ...changedSuffix,
        ...(changedSuffix.length || clearShotAcceptance ? shots : []),
      ];
    } else if (continuitySourceSlots.length) {
      playbackOrder = await filmOrder(workingDir, root);
      invalidatedSlots = [
        ...(forceContinuity ? shots : []),
        ...continuitySuccessors(
          playbackOrder,
          continuitySourceSlots,
          new Set(shots.map(String)),
        ),
      ];
    }
  } else if (forceContinuity) {
    invalidatedSlots = [...shots, ...linkedInvalidatedSlots];
  } else {
    invalidatedSlots = linkedInvalidatedSlots;
  }

  if (preserveApprovalContinuity) {
    const uniqueSlots = [...new Set(invalidatedSlots.map(String))];
    for (const slot of uniqueSlots) {
      const record = shotStore[slot];
      if (!isPlainObject(record)) continue;
      const reason = Array.isArray(orderBefore)
        ? 'The shot or film playback order changed; re-review this shot’s keyframes and clip for the continuous handoff.'
        : continuityInvalidationReason(playbackOrder, slot, continuitySourceSlots.length ? continuitySourceSlots : shots);
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

  if (clearShotAcceptance && !(continuityEnabled && Array.isArray(orderBefore)) && !forceContinuity) {
    for (const slot of shots) {
      removeAcceptance(store, root, slot, 'keyframes');
      removeAcceptance(store, root, slot, 'clips');
    }
  }
  if (JSON.stringify(store) !== before) {
    await transaction.write(path.join(workingDir, ACCEPTANCE_FILENAME), `${JSON.stringify(store, null, 2)}\n`);
  }

  if (!preserveMedia) {
    const rootDir = path.join(workingDir, root);
    const films = new Set([path.join(rootDir, 'final_video.mp4')]);
    for (const slot of shots) {
      const location = shotLocation(workingDir, root, slot);
      if (location.scene) films.add(path.join(location.container, 'final_video.mp4'));
    }
    for (const film of films) await transaction.remove(film);
  }
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
async function timelineReuseChoices(workingDir, root, slots) {
  const choices = new Map();
  for (const slot of new Set(slots.map(String))) {
    const location = shotLocation(workingDir, root, slot);
    const activePlanPath = path.join(location.shotDir, SHOT_DESCRIPTION_FILENAME);
    let bytes = await readSafeMetadata(workingDir, activePlanPath);
    if (bytes === null) {
      bytes = await readSafeMetadata(
        workingDir,
        path.join(location.container, REMOVED_SHOTS_DIR, location.local, SHOT_DESCRIPTION_FILENAME),
      );
    }
    const description = bytes === null ? null : JSON.parse(bytes.toString('utf8'));
    if (typeof description?.reuse_previous_last_frame === 'boolean') {
      choices.set(slot, description.reuse_previous_last_frame);
    }
  }
  return choices;
}

function linkedContinuitySuccessors(order, changedSlots, reuseChoices) {
  const changed = new Set(changedSlots.map(String));
  const invalidated = [];
  for (let index = 1; index < order.length; index += 1) {
    const previous = order[index - 1];
    const slot = order[index];
    if (changed.has(previous) && reuseChoices.get(slot) === true) {
      invalidated.push(slot);
      changed.add(slot);
    }
  }
  return invalidated;
}

function changedLinkedContinuityTargets(before, after, reuseChoices) {
  const previousByShot = (order) => new Map(order.map((slot, index) => [slot, index ? order[index - 1] : null]));
  const beforePrevious = previousByShot(before);
  const afterPrevious = previousByShot(after);
  const changedTargets = after.filter((slot) => reuseChoices.get(slot) === true
    && beforePrevious.get(slot) !== afterPrevious.get(slot));
  return [...new Set([
    ...changedTargets,
    ...linkedContinuitySuccessors(after, changedTargets, reuseChoices),
  ])];
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

async function shotPlanPayload(workingDir, root, slot, context = createFilmReadContext(), setInventory, reuseContext) {
  const location = shotLocation(workingDir, root, slot);
  const shotDir = location.shotDir;
  let description;
  try {
    description = await context.json(path.join(shotDir, SHOT_DESCRIPTION_FILENAME));
    if (!description) throw new Error('Missing shot description');
  } catch {
    throw acceptanceError(`Unknown shot: ${slot || '(missing)'}`);
  }
  const playback = reuseContext?.playback ?? await filmPlaybackContext(workingDir, root, context);
  const continuityEnabled = reuseContext?.continuityEnabled ?? await frameContinuityEnabled(workingDir);
  const index = playback.order.indexOf(String(slot));
  const previousSlot = index > 0 ? playback.order[index - 1] : null;
  const storedChoice = typeof description.reuse_previous_last_frame === 'boolean'
    ? description.reuse_previous_last_frame : null;
  const reusePreviousLastFrame = previousSlot !== null && (storedChoice
    ?? (continuityEnabled && playback.cameraBySlot.get(String(slot)) === playback.cameraBySlot.get(previousSlot)));
  const previousLastFrameAvailable = previousSlot
    ? await shotLastFrameAvailable(workingDir, root, previousSlot, context)
    : false;
  const {characters, briefs} = await scenePlanMetadata(workingDir, location, context);
  const shotBrief = await context.json(path.join(shotDir, 'brief.json')).catch(() => null);
  const brief = typeof shotBrief?.visual_desc === 'string'
    ? shotBrief.visual_desc
    : briefs.get(String(location.local)) || '';
  const prompt = async (frame) => {
    try {
      const cached = await context.json(path.join(shotDir, `${frame}_selector_output.json`));
      return typeof cached?.sent_prompt === 'string' ? cached.sent_prompt : '';
    } catch { return ''; }
  };
  const setShot = setInventory?.shots.find((shot) => shot.slot === String(slot));
  return {
    slot: String(slot), root,
    brief,
    characters,
    firstFrame: {description: String(description.ff_desc || ''), visible: [...(description.ff_vis_char_idxs || [])], prompt: await prompt('first_frame')},
    lastFrame: {description: String(description.lf_desc || ''), visible: [...(description.lf_vis_char_idxs || [])], prompt: await prompt('last_frame')},
    motionDescription: String(description.motion_desc || ''),
    reusePreviousLastFrame,
    previousSlot,
    previousLastFrameAvailable,
    sets: (setInventory?.sets ?? []).map(({id, name, state}) => ({id, name, accepted: state === 'accepted'})),
    setIds: setShot?.setIds ?? [],
    setRevision: setInventory?.revision ?? '',
  };
}

async function shotLastFrameAvailable(workingDir, root, slot, context) {
  const location = shotLocation(workingDir, root, slot);
  if (!isPathInside(workingDir, location.shotDir)) return false;
  let current = workingDir;
  for (const part of path.relative(workingDir, location.shotDir).split(path.sep)) {
    current = path.join(current, part);
    let info;
    try { info = await lstat(current); }
    catch (error) { if (error.code === 'ENOENT') return false; throw error; }
    if (!info.isDirectory() || info.isSymbolicLink()) return false;
  }
  const artifacts = [];
  await collectAcceptanceFiles(workingDir, location.shotDir, (name) => name === 'last_frame.png', artifacts, context);
  return artifacts.length > 0;
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
const SET_ID_PATTERN = /^[a-z0-9][a-z0-9_-]{0,63}$/;
const SET_REFERENCE_LIMIT = 4000;

/** Read the editable set plan and its approval state without creating files. */
export async function readSetReferences(repoRoot, sessionId, requestedRoot = '') {
  const record = sessionRecord(await readSessionsPayload(repoRoot), sessionId);
  if (!record) return null;
  const workingDir = sessionWorkingDir(repoRoot, sessionId, record);
  const context = createFilmReadContext();
  const root = await resolveRenderRoot(repoRoot, workingDir, requestedRoot, context);
  const discovery = await collectAcceptanceState(workingDir, root, context);
  const manifest = await readRenderManifest(repoRoot, workingDir, context) ?? {};
  const store = await readAcceptanceFile(workingDir, context);
  const setStore = isPlainObject(store[root]?.sets) ? store[root].sets : {};
  const directory = path.join(workingDir, root, 'set_references');
  let entries = [];
  try {
    if (await realpath(directory) !== path.resolve(directory)) throw acceptanceError('Set reference directory is unsafe');
    entries = await context.entries(directory);
  } catch {}
  const ids = entries.filter((entry) => entry.isDirectory() && SET_ID_PATTERN.test(entry.name))
    .map((entry) => entry.name).sort();
  const configured = manifest.image_model ? null : await readAgentConfig(repoRoot).catch(() => null);
  const model = String(manifest.image_model || configured?.sections?.image?.model || '');
  const modelSlug = model ? safePathComponent(model) : '';
  const style = String(manifest.style ?? record.style ?? '');
  const sets = [];
  for (const id of ids) {
    const setDir = path.join(directory, id);
    if (!await safeSetDirectory(workingDir, setDir)) continue;
    const descriptorPath = path.join(setDir, 'set.json');
    if (!await safeSetFile(workingDir, descriptorPath)) continue;
    const descriptor = await context.json(descriptorPath).catch(() => null);
    const descriptorKeys = isPlainObject(descriptor) ? Object.keys(descriptor).sort().join(',') : '';
    if (!isPlainObject(descriptor) || descriptorKeys !== 'description,id,name,shots'
      || descriptor.id !== id || typeof descriptor.name !== 'string' || typeof descriptor.description !== 'string'
      || !Array.isArray(descriptor.shots) || !descriptor.shots.every((slot) => typeof slot === 'string')
      || new Set(descriptor.shots).size !== descriptor.shots.length) continue;
    const specBytes = await readFile(descriptorPath).catch(() => Buffer.from(''));
    const specSha256 = createHash('sha256').update(specBytes).digest('hex');
    const imagePath = modelSlug ? `${root}/set_references/${id}/${modelSlug}/reference.png` : '';
    const canonicalImage = imagePath ? path.join(workingDir, imagePath) : '';
    const provenancePath = modelSlug ? path.join(setDir, modelSlug, 'provenance.json') : '';
    const approval = setStore[id];
    const provenanceExists = Boolean(provenancePath && await safeSetFile(workingDir, provenancePath));
    const provenance = provenanceExists ? await context.json(provenancePath).catch(() => null) : null;
    const currentProvenance = Boolean(modelSlug && provenanceExists
      && isPlainObject(provenance) && provenance.image_model === model && provenance.style === style
      && provenance.spec_sha256 === specSha256);
    const imageExists = Boolean(canonicalImage && await safeSetFile(workingDir, canonicalImage));
    let state = 'planned';
    if (imageExists) {
      state = isRejected(approval) ? 'rejected' : currentProvenance ? 'rendered' : 'stale';
      if (isPlainObject(approval) && approval.accepted_at) {
        const expectedPaths = [descriptorPath, canonicalImage, provenancePath].map((file) => path.relative(workingDir, file));
        const approvedPaths = Array.isArray(approval.artifacts) ? approval.artifacts.map((artifact) => artifact?.path) : [];
        const exactArtifacts = approvedPaths.length === 3 && new Set(approvedPaths).size === 3
          && approvedPaths.every((file) => expectedPaths.includes(file));
        state = !approval.invalidated_at && !isRejected(approval) && currentProvenance && exactArtifacts
          && await acceptanceLockMatches(workingDir, approval, context) ? 'accepted' : 'stale';
      }
    } else if (isRejected(approval)) state = 'rejected';
    else if (isPlainObject(approval) && approval.accepted_at) state = 'stale';
    const candidateInventory = await readSetCandidates(workingDir, setDir, model, modelSlug, style, specSha256, context);
    const revision = createHash('sha256').update(specBytes).update(JSON.stringify(approval ?? null))
      .update(imageExists ? await context.hash(canonicalImage).catch(() => '') : '')
      .update(provenanceExists ? await context.hash(provenancePath).catch(() => '') : '')
      .update(JSON.stringify(candidateInventory.fingerprint))
      .digest('hex');
    sets.push({
      id, name: descriptor.name, description: descriptor.description, shots: descriptor.shots,
      state, ...(imageExists ? {imagePath} : {}), imageModel: model,
      candidates: candidateInventory.candidates,
      ...(typeof provenance?.candidate_id === 'string' && /^[a-f0-9]{32}$/.test(provenance.candidate_id)
        ? {selectedCandidateId: provenance.candidate_id} : {}),
      revision,
    });
  }
  const rawShots = root === 'novel2video'
    ? await novelSetShotSlots(path.join(workingDir, root))
    : await setInventoryShotSlots(workingDir, root, discovery, context);
  const defaults = new Map(sets.map((set) => [set.id, set.shots]));
  const shots = rawShots.map(({slot, brief, setIdsOverride}) => ({
    slot,
    brief,
    setIds: setIdsOverride ?? sets.filter((set) => defaults.get(set.id)?.includes(slot)).map((set) => set.id),
  }));
  const activeShots = shots.map(({slot}) => slot);
  const knownSetIds = new Set(sets.map(({id}) => id));
  const unassignedShots = shots
    .filter(({setIds}) => !setIds.some((id) => knownSetIds.has(id)))
    .map(({slot}) => slot);
  const revision = createHash('sha256').update(JSON.stringify([
    root, model, style, sets.map(({id, revision: rev}) => [id, rev]),
    shots.map(({slot, setIds}, index) => [slot, setIds, rawShots[index].setIdsOverride]), activeShots, unassignedShots,
  ])).digest('hex');
  return {sessionId, root, revision, imageModel: model, style, sets, shots, unassignedShots};
}

function shotSetIdsOverride(description) {
  if (!isPlainObject(description) || !Object.hasOwn(description, 'set_ids') || description.set_ids === null) return null;
  return Array.isArray(description.set_ids) ? description.set_ids.filter((id) => typeof id === 'string') : [];
}

async function setInventoryShotSlots(workingDir, root, discovery, context) {
  const rows = [];
  for (const {shot: slot} of discovery.shots) {
    const location = shotLocation(workingDir, root, slot);
    const shotDir = location.shotDir;
    const description = await context.json(path.join(shotDir, SHOT_DESCRIPTION_FILENAME)).catch(() => null);
    const {briefs} = await scenePlanMetadata(workingDir, location, context);
    const shotBrief = await context.json(path.join(shotDir, 'brief.json')).catch(() => null);
    rows.push({
      slot,
      brief: typeof shotBrief?.visual_desc === 'string' ? shotBrief.visual_desc : briefs.get(location.local) || '',
      setIdsOverride: shotSetIdsOverride(description),
    });
  }
  return rows;
}

async function readSetCandidates(workingDir, setDir, model, modelSlug, style, specSha256, context) {
  const modelEntries = await context.entries(setDir).catch(() => []);
  const candidates = [];
  const fingerprint = [];
  for (const modelEntry of modelEntries) {
    if (modelEntry.name.startsWith('.') || !modelEntry.isDirectory()) continue;
    const modelDir = path.join(setDir, modelEntry.name);
    if (!await safeSetDirectory(workingDir, modelDir)) continue;
    const files = await context.entries(modelDir).catch(() => []);
    for (const file of files) {
      const match = /^candidate_([a-f0-9]{32})\.png$/.exec(file.name);
      if (!match || !file.isFile()) continue;
      const candidateId = match[1];
      const absoluteImage = path.join(modelDir, file.name);
      if (!await safeSetFile(workingDir, absoluteImage)) continue;
      const imagePath = path.relative(workingDir, absoluteImage).split(path.sep).join('/');
      const provenancePath = path.join(modelDir, `candidate_${candidateId}.json`);
      const provenanceExists = await safeSetFile(workingDir, provenancePath);
      const provenance = provenanceExists ? await context.json(provenancePath).catch(() => null) : null;
      const imageBytes = await readFile(absoluteImage).catch(() => null);
      let state = 'invalid';
      if (imageBytes && isCompletePng(imageBytes) && provenanceExists && isPlainObject(provenance)
        && provenance.candidate_id === candidateId
        && typeof provenance.image_model === 'string' && typeof provenance.style === 'string'
        && typeof provenance.spec_sha256 === 'string' && /^[a-f0-9]{64}$/.test(provenance.spec_sha256)
        && (provenance.prompt === undefined || typeof provenance.prompt === 'string')) {
        state = modelEntry.name === modelSlug && provenance.image_model === model && provenance.style === style
          && provenance.spec_sha256 === specSha256 ? 'ready' : 'stale';
      }
      const imageHash = await context.hash(absoluteImage).catch(() => '');
      const provenanceHash = provenanceExists ? await context.hash(provenancePath).catch(() => '') : '';
      fingerprint.push([candidateId, imagePath, state, imageHash, provenanceHash]);
      candidates.push({
        id: candidateId,
        imagePath,
        state,
        ...(typeof provenance?.prompt === 'string' ? {prompt: provenance.prompt} : {}),
      });
    }
  }
  candidates.sort((left, right) => left.imagePath.localeCompare(right.imagePath));
  fingerprint.sort((left, right) => `${left[1]}`.localeCompare(`${right[1]}`));
  return {candidates, fingerprint};
}

async function novelSetShotSlots(rootDir) {
  const videosDir = path.join(rootDir, 'videos');
  const found = [];
  async function walk(directory, depth) {
    if (depth > 12) return;
    const entries = await readdir(directory, {withFileTypes: true}).catch(() => []);
    for (const entry of entries) {
      if (entry.name.startsWith('.') || !entry.isDirectory()) continue;
      const child = path.join(directory, entry.name);
      const descriptionPath = path.join(child, SHOT_DESCRIPTION_FILENAME);
      if (await isRegularFile(descriptionPath)) {
        let description = {};
        try { description = JSON.parse(await readFile(descriptionPath, 'utf8')); } catch {}
        let slot = path.relative(rootDir, child).split(path.sep).join('/');
        slot = slot.replace(/\/shots\//g, '/');
        found.push({slot, brief: isPlainObject(description) ? String(description.brief ?? '') : '', setIdsOverride: shotSetIdsOverride(description)});
      }
      await walk(child, depth + 1);
    }
  }
  await walk(videosDir, 0);
  return found.sort((left, right) => left.slot.localeCompare(right.slot, undefined, {numeric: true}));
}

async function safeSetDirectory(workingDir, directory) {
  try {
    const info = await lstat(directory);
    const requested = path.resolve(directory);
    const absolute = await realpath(directory);
    return info.isDirectory() && absolute === requested && absolute.startsWith(`${workingDir}${path.sep}`);
  } catch {
    return false;
  }
}

async function safeSetFile(workingDir, filePath) {
  try {
    const info = await lstat(filePath);
    const requested = path.resolve(filePath);
    const absolute = await realpath(filePath);
    return info.isFile() && absolute === requested && absolute.startsWith(`${workingDir}${path.sep}`);
  } catch {
    return false;
  }
}
async function prepareSetDirectory(workingDir, root, id) {
  let current = workingDir;
  for (const component of [root, 'set_references', id]) {
    current = path.join(current, component);
    const info = await lstat(current).catch((error) => {
      if (error.code === 'ENOENT') return null;
      throw error;
    });
    if (info && (!info.isDirectory() || await realpath(current) !== path.resolve(current))) {
      throw acceptanceError('Set reference directory is unsafe');
    }
    if (!info) await mkdir(current);
  }
}

async function prepareSetModelDirectory(workingDir, root, id, modelSlug) {
  await prepareSetDirectory(workingDir, root, id);
  const directory = path.join(workingDir, root, 'set_references', id, modelSlug);
  const info = await lstat(directory).catch((error) => {
    if (error.code === 'ENOENT') return null;
    throw error;
  });
  if (info && (!info.isDirectory() || await realpath(directory) !== path.resolve(directory))) {
    throw acceptanceError('Set reference directory is unsafe');
  }
  if (!info) await mkdir(directory);
}
/** Create or edit a set descriptor, rejecting stale snapshots before any write. */
export async function updateSetReference(repoRoot, input = {}) {
  const body = isPlainObject(input) ? input : {};
  const sessionId = typeof body.sessionId === 'string' ? body.sessionId : '';
  const record = sessionRecord(await readSessionsPayload(repoRoot), sessionId);
  if (!record) return null;
  const workingDir = sessionWorkingDir(repoRoot, sessionId, record);
  return withProjectWriteLock(workingDir, async () => {
    const inventory = await readSetReferences(repoRoot, sessionId, body.root);
    if (!inventory) return null;
    if (body.revision !== inventory.revision) throw conflictError('Set references changed; refresh before saving');
    const {id, name, description, shots} = body;
    if (typeof id !== 'string' || !SET_ID_PATTERN.test(id)) throw acceptanceError('id must be a safe immutable set identifier');
    if (typeof name !== 'string' || !name.trim() || name.length > 160) throw acceptanceError('name must be 1–160 characters');
    if (typeof description !== 'string' || description.length > SET_REFERENCE_LIMIT) throw acceptanceError('description must be at most 4000 characters');
    if (!Array.isArray(shots) || !shots.every((slot) => typeof slot === 'string') || new Set(shots).size !== shots.length) {
      throw acceptanceError('shots must be a unique list of shot slots');
    }
    const active = new Set(inventory.shots.map(({slot}) => slot));
    if (shots.some((slot) => !active.has(slot))) throw acceptanceError('shots must name active shot slots');
    const descriptorPath = path.join(workingDir, inventory.root, 'set_references', id, 'set.json');
    const bytes = Buffer.from(`${JSON.stringify({id, name: name.trim(), description, shots}, null, 2)}\n`);
    await prepareSetDirectory(workingDir, inventory.root, id);
    const store = await readAcceptanceFile(workingDir);
    const rootStore = isPlainObject(store[inventory.root]) ? store[inventory.root] : {};
    const setRecords = isPlainObject(rootStore.sets) ? rootStore.sets : {};
    delete setRecords[id];
    if (Object.keys(setRecords).length) rootStore.sets = setRecords;
    else delete rootStore.sets;
    store[inventory.root] = rootStore;
    await withMutationTransaction(async (transaction) => {
      await transaction.write(descriptorPath, bytes);
      const updated = await readSetReferences(repoRoot, sessionId, inventory.root);
      const before = inventory.shots.filter((shot) => shot.setIds.includes(id)).map(({slot}) => slot);
      const after = (updated?.shots ?? []).filter((shot) => shot.setIds.includes(id)).map(({slot}) => slot);
      await invalidateSetBoundShots(workingDir, inventory.root, [...new Set([...before, ...after])], store, transaction);
      await transaction.write(path.join(workingDir, ACCEPTANCE_FILENAME), `${JSON.stringify(store, null, 2)}\n`);
    });
    return readSetReferences(repoRoot, sessionId, inventory.root);
  });
}

/** Select a current candidate as canonical without deleting its immutable source. */
export async function selectSetReference(repoRoot, input = {}, isBusy = () => false) {
  const body = isPlainObject(input) ? input : {};
  const sessionId = typeof body.sessionId === 'string' ? body.sessionId : '';
  const record = sessionRecord(await readSessionsPayload(repoRoot), sessionId);
  if (!record) return null;
  const workingDir = sessionWorkingDir(repoRoot, sessionId, record);
  return withProjectWriteLock(workingDir, async () => {
    if (isBusy()) throw conflictError('Wait for active and queued requests to finish before selecting a set image');
    const inventory = await readSetReferences(repoRoot, sessionId, body.root);
    if (!inventory) return null;
    if (body.revision !== inventory.revision) throw conflictError('Set references changed; refresh before selecting an image');
    if (typeof body.id !== 'string' || !SET_ID_PATTERN.test(body.id)) throw acceptanceError('id must be a safe set identifier');
    if (typeof body.candidateId !== 'string' || !/^[a-f0-9]{32}$/.test(body.candidateId)) {
      throw acceptanceError('candidateId must be a lowercase 32-character candidate identifier');
    }
    const set = inventory.sets.find((item) => item.id === body.id);
    if (!set) throw acceptanceError('Unknown set identifier');
    const candidate = set.candidates.find((item) => item.id === body.candidateId);
    if (!candidate || candidate.state !== 'ready') throw conflictError('Only a current valid candidate can be selected');
    const manifest = await readRenderManifest(repoRoot, workingDir);
    const model = String(manifest?.image_model ?? '');
    const modelSlug = model ? safePathComponent(model) : '';
    if (!modelSlug) throw conflictError('A current image model is required before selecting a candidate');
    const base = path.join(workingDir, inventory.root, 'set_references', set.id);
    const descriptorPath = path.join(base, 'set.json');
    const modelDir = path.join(base, modelSlug);
    const candidateImage = path.join(modelDir, `candidate_${body.candidateId}.png`);
    const candidateProvenance = path.join(modelDir, `candidate_${body.candidateId}.json`);
    if (candidate.imagePath !== path.relative(workingDir, candidateImage).split(path.sep).join('/')
      || !await safeSetFile(workingDir, descriptorPath)
      || !await safeSetDirectory(workingDir, modelDir)
      || !await safeSetFile(workingDir, candidateImage)
      || !await safeSetFile(workingDir, candidateProvenance)) {
      throw conflictError('Set candidate files are missing or unsafe');
    }
    const [descriptorBytes, imageBytes, provenanceBytes] = await Promise.all([
      readFile(descriptorPath), readFile(candidateImage), readFile(candidateProvenance),
    ]);
    let provenance;
    try { provenance = JSON.parse(provenanceBytes.toString('utf8')); }
    catch { throw conflictError('Set candidate provenance is malformed'); }
    if (!isPlainObject(provenance) || provenance.candidate_id !== body.candidateId
      || provenance.image_model !== model || provenance.style !== (manifest?.style ?? '')
      || provenance.spec_sha256 !== createHash('sha256').update(descriptorBytes).digest('hex')
      || (provenance.prompt !== undefined && typeof provenance.prompt !== 'string')
      || !isCompletePng(imageBytes)) {
      throw conflictError('Set candidate provenance or image is stale; generate a current candidate before selecting');
    }

    const canonicalImage = path.join(modelDir, 'reference.png');
    const canonicalProvenance = path.join(modelDir, 'provenance.json');
    const imageExists = await pathExists(canonicalImage);
    const provenanceExists = await pathExists(canonicalProvenance);
    if (imageExists && !await safeSetFile(workingDir, canonicalImage)
      || provenanceExists && !await safeSetFile(workingDir, canonicalProvenance)) {
      throw acceptanceError('Set reference files are unsafe');
    }
    const [previousImage, previousProvenance] = await Promise.all([
      imageExists ? readFile(canonicalImage) : null,
      provenanceExists ? readFile(canonicalProvenance) : null,
    ]);
    if (previousImage?.equals(imageBytes) && previousProvenance?.equals(provenanceBytes)) return inventory;

    const store = await readAcceptanceFile(workingDir);
    const rootStore = isPlainObject(store[inventory.root]) ? store[inventory.root] : {};
    const setRecords = isPlainObject(rootStore.sets) ? rootStore.sets : {};
    delete setRecords[set.id];
    if (Object.keys(setRecords).length) rootStore.sets = setRecords;
    else delete rootStore.sets;
    store[inventory.root] = rootStore;
    const boundShots = inventory.shots.filter((shot) => shot.setIds.includes(set.id)).map(({slot}) => slot);
    await prepareSetModelDirectory(workingDir, inventory.root, set.id, modelSlug);
    await withMutationTransaction(async (transaction) => {
      await transaction.write(canonicalImage, imageBytes);
      await transaction.write(canonicalProvenance, provenanceBytes);
      await invalidateSetBoundShots(workingDir, inventory.root, boundShots, store, transaction);
      await transaction.write(path.join(workingDir, ACCEPTANCE_FILENAME), `${JSON.stringify(store, null, 2)}\n`);
    });
    return readSetReferences(repoRoot, sessionId, inventory.root);
  });
}

export async function updateSetReferenceAcceptance(repoRoot, input = {}) {
  const body = isPlainObject(input) ? input : {};
  const sessionId = typeof body.sessionId === 'string' ? body.sessionId : '';
  const record = sessionRecord(await readSessionsPayload(repoRoot), sessionId);
  if (!record) return null;
  const workingDir = sessionWorkingDir(repoRoot, sessionId, record);
  return withProjectWriteLock(workingDir, async () => {
    const inventory = await readSetReferences(repoRoot, sessionId, body.root);
    if (!inventory) return null;
    if (body.revision !== inventory.revision) throw conflictError('Set references changed; refresh before reviewing');
    const set = inventory.sets.find((item) => item.id === body.id);
    if (!set) throw acceptanceError('Unknown set identifier');
    if (typeof body.accepted !== 'boolean') throw acceptanceError('accepted must be a boolean');
    const store = await readAcceptanceFile(workingDir);
    const rootStore = isPlainObject(store[inventory.root]) ? store[inventory.root] : {};
    const records = isPlainObject(rootStore.sets) ? rootStore.sets : {};
    if (body.accepted) {
      if (set.state !== 'rendered' && set.state !== 'accepted') throw conflictError('A current generated set image is required before acceptance');
      const manifest = await readRenderManifest(repoRoot, workingDir);
      const modelSlug = safePathComponent(manifest?.image_model ?? '');
      const base = path.join(workingDir, inventory.root, 'set_references', set.id);
      const descriptorPath = path.join(base, 'set.json');
      const provenancePath = path.join(base, modelSlug, 'provenance.json');
      const imagePath = path.join(base, modelSlug, 'reference.png');
      if (!await safeSetFile(workingDir, descriptorPath) || !await safeSetFile(workingDir, provenancePath) || !await safeSetFile(workingDir, imagePath)) {
        throw acceptanceError('Set reference files are missing or unsafe');
      }
      const descriptor = await readFile(descriptorPath);
      let provenance;
      try { provenance = JSON.parse(await readFile(provenancePath, 'utf8')); }
      catch { throw acceptanceError('Set reference provenance is malformed'); }
      if (!isPlainObject(provenance) || provenance.image_model !== manifest.image_model || provenance.style !== (manifest.style ?? '')
        || provenance.spec_sha256 !== createHash('sha256').update(descriptor).digest('hex')) {
        throw conflictError('Set image provenance is stale; regenerate this set before acceptance');
      }
      const png = await readFile(imagePath);
      if (!isCompletePng(png)) {
        throw acceptanceError('Set reference is not a valid PNG');
      }
      records[set.id] = {accepted_at: isoSeconds(), artifacts: await acceptanceArtifacts(workingDir,
        [descriptorPath, imagePath, provenancePath].map((file) => path.relative(workingDir, file)))};
    } else {
      if (body.reason !== undefined && typeof body.reason !== 'string') throw acceptanceError('reason must be a string');
      const reason = typeof body.reason === 'string' ? body.reason.trim() : '';
      if (reason.length > REASON_MAX_LENGTH) throw acceptanceError(`reason must be at most ${REASON_MAX_LENGTH} characters`);
      if (reason) records[set.id] = {rejected_at: isoSeconds(), reason};
      else delete records[set.id];
    }
    const boundShots = inventory.shots.filter((shot) => shot.setIds.includes(set.id)).map(({slot}) => slot);
    if (Object.keys(records).length) rootStore.sets = records;
    else delete rootStore.sets;
    store[inventory.root] = rootStore;
    await invalidateSetBoundShots(workingDir, inventory.root, boundShots, store);
    await writeAcceptanceFile(workingDir, store);
    return readSetReferences(repoRoot, sessionId, inventory.root);
  });
}

async function invalidateSetBoundShots(workingDir, root, shots, store, transaction) {
  const affected = [...new Set(shots.map(String))];
  if (!affected.length) return;
  const playbackOrder = await filmOrder(workingDir, root);
  const reuseChoices = await timelineReuseChoices(workingDir, root, playbackOrder);
  const linkedSuccessors = linkedContinuitySuccessors(playbackOrder, affected, reuseChoices);
  const rootStore = isPlainObject(store[root]) ? store[root] : {};
  const shotStore = isPlainObject(rootStore.shots) ? rootStore.shots : {};
  const now = isoSeconds();
  for (const slot of [...new Set([...affected, ...linkedSuccessors])]) {
    const record = isPlainObject(shotStore[slot]) ? shotStore[slot] : {};
    for (const stage of ['keyframes', 'clips']) {
      if (isPlainObject(record[stage])) record[stage] = {...record[stage], invalidated_at: now,
        invalidation_reason: affected.includes(slot)
          ? 'A bound set reference changed; re-review this shot.'
          : 'An earlier linked keyframe changed; re-review this shot.'};
    }
    if (Object.keys(record).length) shotStore[slot] = record;
  }
  if (Object.keys(shotStore).length) rootStore.shots = shotStore;
  if (Object.keys(rootStore).length) store[root] = rootStore;
  if (isPlainObject(rootStore.final_video)) {
    rootStore.final_video = {...rootStore.final_video, invalidated_at: now,
      invalidation_reason: 'A bound set reference changed; review the updated film assembly.'};
  }
  const continuityPath = path.join(workingDir, CONTINUITY_FILENAME);
  const review = await readJsonOptional(continuityPath);
  if (isPlainObject(review) && (!review.root || review.root === root)) {
    const bytes = `${JSON.stringify({
      ...review,
      set_references_invalidated_at: now,
      set_references_invalidation_reason: 'A bound set reference changed; review continuity again.',
    }, null, 2)}\n`;
    if (transaction) await transaction.write(continuityPath, bytes);
    else await writeAtomic(continuityPath, bytes);
  }
}
