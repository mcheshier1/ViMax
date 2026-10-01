import {constants} from 'node:fs';
import {mkdtemp, open, rename, rm, realpath, stat} from 'node:fs/promises';
import {createHash, randomUUID} from 'node:crypto';
import {spawn} from 'node:child_process';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {pipeline} from 'node:stream/promises';
import {artifactContentType, resolveArtifactPath} from './server-lib.mjs';

const thumbnailFormats = new Map([
  ['.png', 'png_pipe'], ['.jpg', 'jpeg_pipe'], ['.jpeg', 'jpeg_pipe'],
  ['.webp', 'webp_pipe'], ['.gif', 'gif'], ['.mp4', 'mov'],
  ['.mov', 'mov'], ['.webm', 'matroska'],
]);
const thumbnails = new Map();
const thumbnailQueue = [];
const thumbnailChildren = new Set();
const maxThumbnailWorkers = 2;
const maxThumbnailQueue = 64;
const maxCachedThumbnails = 512;
let thumbnailWorkers = 0;
let thumbnailDirectory;
let closing = false;

function mediaError(statusCode, message) {
  return Object.assign(new Error(message), {statusCode});
}

function fingerprint(info) {
  return `${info.dev}:${info.ino}:${info.size}:${info.mtimeNs}:${info.ctimeNs}`;
}

function digest(value) {
  return createHash('sha256').update(value).digest('hex');
}

async function openArtifact(repoRoot, sessionId, relativePath) {
  let candidate;
  try {
    candidate = resolveArtifactPath(repoRoot, sessionId, relativePath);
  } catch (error) {
    throw mediaError(400, error.message);
  }
  let handle;
  try {
    const workingRoot = await realpath(path.join(repoRoot, '.working_dir'));
    const [sessionRoot, filePath] = await Promise.all([
      realpath(path.join(repoRoot, '.working_dir', sessionId)), realpath(candidate),
    ]);
    // The lexical resolver rejects traversal; real paths also reject symlink escapes.
    if (sessionRoot !== path.join(workingRoot, sessionId) || !filePath.startsWith(`${sessionRoot}${path.sep}`)) {
      throw mediaError(403, 'Artifact path escapes the active session');
    }
    handle = await open(filePath, constants.O_RDONLY | (constants.O_NOFOLLOW || 0) | (constants.O_NONBLOCK || 0));
    const info = await handle.stat({bigint: true});
    if (!info.isFile()) throw mediaError(404, 'Artifact is not a regular file');
    return {handle, info, filePath};
  } catch (error) {
    await handle?.close();
    if (error.code === 'ENOENT' || error.code === 'ENOTDIR') throw mediaError(404, 'Artifact not found');
    if (['EACCES', 'EPERM', 'ELOOP'].includes(error.code)) throw mediaError(403, 'Artifact cannot be accessed');
    throw error;
  }
}

function matchesTag(value, etag, weak = false) {
  return value.split(',').some((tag) => {
    const candidate = tag.trim();
    return candidate === '*' || (weak ? candidate.replace(/^W\//, '') === etag : candidate === etag);
  });
}

function dateMatches(value, modified) {
  const date = Date.parse(value);
  return Number.isFinite(date) && modified <= date;
}

function byteRange(value, size) {
  // Multiple ranges are deliberately ignored (RFC 9110 permits a full 200).
  // Unknown units likewise do not change the selected representation.
  if (!value || !/^bytes=/i.test(value) || value.includes(',')) return null;
  const match = /^bytes=(\d*)-(\d*)$/i.exec(value.trim());
  if (!match || (!match[1] && !match[2]) || size === 0n) return false;
  let start;
  let end;
  if (!match[1]) {
    const suffix = BigInt(match[2]);
    if (suffix === 0n) return false;
    start = suffix >= size ? 0n : size - suffix;
    end = size - 1n;
  } else {
    start = BigInt(match[1]);
    end = match[2] ? BigInt(match[2]) : size - 1n;
    if (start >= size || end < start) return false;
    if (end >= size) end = size - 1n;
  }
  return {start: Number(start), end: Number(end)};
}

async function deliverFile(request, response, source, contentType) {
  const {handle, info} = source;
  try {
    if (response.destroyed) return;
    if (info.size > BigInt(Number.MAX_SAFE_INTEGER)) throw mediaError(500, 'Artifact is too large to stream safely');
    const etag = `"${digest(fingerprint(info))}"`;
    const modified = Math.floor(Number(info.mtimeNs / 1_000_000n) / 1000) * 1000;
    const headers = {
      'Content-Type': contentType,
      'Content-Length': String(info.size),
      'Accept-Ranges': 'bytes',
      'ETag': etag,
      'Last-Modified': new Date(modified).toUTCString(),
      'Cache-Control': 'private, max-age=0, must-revalidate',
      'X-Content-Type-Options': 'nosniff',
    };
    const incoming = request.headers;
    if ((incoming['if-match'] && !matchesTag(incoming['if-match'], etag)) ||
        (!incoming['if-match'] && incoming['if-unmodified-since'] &&
          Number.isFinite(Date.parse(incoming['if-unmodified-since'])) && !dateMatches(incoming['if-unmodified-since'], modified))) {
      response.writeHead(412, {...headers, 'Content-Length': '0'});
      response.end();
      return;
    }
    if (incoming['if-none-match'] ? matchesTag(incoming['if-none-match'], etag, true) :
      incoming['if-modified-since'] && dateMatches(incoming['if-modified-since'], modified)) {
      response.writeHead(304, headers);
      response.end();
      return;
    }
    const ifRange = incoming['if-range'];
    const rangeAllowed = !ifRange || ifRange === etag || (!ifRange.startsWith('"') && !ifRange.startsWith('W/') && Date.parse(ifRange) === modified);
    const range = request.method === 'GET' && rangeAllowed ? byteRange(incoming.range, info.size) : null;
    if (range === false) {
      response.writeHead(416, {...headers, 'Content-Range': `bytes */${info.size}`, 'Content-Length': '0'});
      response.end();
      return;
    }
    if (range) {
      headers['Content-Range'] = `bytes ${range.start}-${range.end}/${info.size}`;
      headers['Content-Length'] = String(range.end - range.start + 1);
    }
    response.writeHead(range ? 206 : 200, headers);
    if (request.method === 'HEAD' || info.size === 0n) {
      response.end();
      return;
    }
    // pipeline destroys the read stream when the client disconnects; the finally
    // closes the file descriptor for success, early abort, and read errors alike.
    await pipeline(handle.createReadStream({autoClose: false, ...(range || {})}), response);
  } catch (error) {
    if (error.code !== 'ERR_STREAM_PREMATURE_CLOSE' && error.code !== 'ECONNRESET') throw error;
  } finally {
    await handle.close();
  }
}

export async function serveArtifact(request, response, repoRoot, sessionId, relativePath) {
  const source = await openArtifact(repoRoot, sessionId, relativePath);
  return deliverFile(request, response, source, artifactContentType(source.filePath));
}

function thumbnailWidth(value) {
  if (value === null || value === '') return 480;
  if (!/^\d+$/.test(value) || !Number.isSafeInteger(Number(value)) || Number(value) < 1) {
    throw mediaError(400, 'Thumbnail width must be a positive integer');
  }
  return Math.max(64, Math.min(1280, Number(value)));
}

async function acquireThumbnailWorker(signal) {
  if (closing || signal.aborted) throw mediaError(499, 'Thumbnail request was cancelled');
  if (thumbnailWorkers < maxThumbnailWorkers) {
    thumbnailWorkers += 1;
    return;
  }
  if (thumbnailQueue.length >= maxThumbnailQueue) throw mediaError(503, 'Thumbnail queue is full; retry shortly');
  await new Promise((resolve, reject) => {
    const waiting = {resolve, reject, signal, abort: null};
    waiting.abort = () => {
      const index = thumbnailQueue.indexOf(waiting);
      if (index !== -1) thumbnailQueue.splice(index, 1);
      reject(mediaError(499, 'Thumbnail request was cancelled'));
    };
    signal.addEventListener('abort', waiting.abort, {once: true});
    thumbnailQueue.push(waiting);
  });
}

function releaseThumbnailWorker() {
  const waiting = thumbnailQueue.shift();
  if (waiting) {
    waiting.signal.removeEventListener('abort', waiting.abort);
    waiting.resolve();
  } else {
    thumbnailWorkers -= 1;
  }
}

function runThumbnail(source, destination, width, format, signal) {
  return new Promise((resolve, reject) => {
    if (signal.aborted) return reject(mediaError(499, 'Thumbnail request was cancelled'));
    // An inherited descriptor pins the authorized file, even during regeneration.
    const input = process.platform === 'linux' ? '/proc/self/fd/3' : process.platform === 'win32' ? source.filePath : '/dev/fd/3';
    const child = spawn(process.env.VIMAX_FFMPEG_CMD || 'ffmpeg', [
      '-nostdin', '-hide_banner', '-loglevel', 'error', '-threads', '1',
      '-filter_threads', '1', '-protocol_whitelist', 'file,pipe', '-f', format,
      '-i', input, '-map', '0:v:0', '-frames:v', '1',
      '-vf', `scale=w='min(iw,${width})':h='min(ih,${width * 2})':force_original_aspect_ratio=decrease,setsar=1`,
      '-threads', '1', '-c:v', 'mjpeg', '-q:v', '4', '-pix_fmt', 'yuvj420p',
      '-f', 'image2', '-update', '1', '-y', destination,
    ], {stdio: ['ignore', 'ignore', 'pipe', source.handle.fd], windowsHide: true});
    thumbnailChildren.add(child);
    let stderr = '';
    let timedOut = false;
    let spawnError;
    child.stderr.setEncoding('utf8');
    child.stderr.on('data', (chunk) => { stderr = (stderr + chunk).slice(-4096); });
    const abort = () => child.kill('SIGKILL');
    signal.addEventListener('abort', abort, {once: true});
    const timer = setTimeout(() => { timedOut = true; child.kill('SIGKILL'); }, 30_000);
    timer.unref();
    child.once('error', (error) => { spawnError = error; });
    child.once('close', (code) => {
      clearTimeout(timer);
      signal.removeEventListener('abort', abort);
      thumbnailChildren.delete(child);
      if (spawnError) {
        reject(mediaError(503, spawnError.code === 'ENOENT'
          ? 'Thumbnail generation requires ffmpeg on PATH (or VIMAX_FFMPEG_CMD)'
          : `Unable to launch thumbnail generator: ${spawnError.message}`));
      } else if (timedOut) {
        reject(mediaError(504, 'Thumbnail generation exceeded 30 seconds'));
      } else if (signal.aborted) {
        reject(mediaError(499, 'Thumbnail request was cancelled'));
      } else if (code !== 0) {
        reject(mediaError(422, `Cannot decode this artifact for a thumbnail${stderr.trim() ? `: ${stderr.trim()}` : ''}`));
      } else {
        resolve();
      }
    });
  });
}

async function buildThumbnail(source, key, width, format, signal) {
  let temporary;
  let acquired = false;
  try {
    await acquireThumbnailWorker(signal);
    acquired = true;
    thumbnailDirectory ||= mkdtemp(path.join(tmpdir(), 'vimax-thumbnails-'));
    const directory = await thumbnailDirectory;
    const destination = path.join(directory, `${key}.jpg`);
    temporary = path.join(directory, `${key}.${randomUUID()}.tmp`);
    await runThumbnail(source, temporary, width, format, signal);
    const current = await stat(source.filePath, {bigint: true});
    if (fingerprint(current) !== fingerprint(source.info)) throw mediaError(409, 'Artifact changed while its thumbnail was being generated; refresh to retry');
    const generated = await stat(temporary);
    if (!generated.isFile() || generated.size === 0) throw mediaError(422, 'Thumbnail generator produced no image');
    await rename(temporary, destination);
    return destination;
  } finally {
    if (temporary) await rm(temporary, {force: true}).catch(() => {});
    await source.handle.close();
    if (acquired) releaseThumbnailWorker();
  }
}

async function trimThumbnails() {
  for (const [key, entry] of thumbnails) {
    if (thumbnails.size <= maxCachedThumbnails) break;
    if (!entry.ready || entry.users) continue;
    thumbnails.delete(key);
    await rm(entry.filePath, {force: true}).catch(() => {});
  }
}

export async function serveThumbnail(request, response, repoRoot, sessionId, relativePath, requestedWidth) {
  const width = thumbnailWidth(requestedWidth);
  const source = await openArtifact(repoRoot, sessionId, relativePath);
  const format = thumbnailFormats.get(path.extname(source.filePath).toLowerCase());
  if (!format) {
    await source.handle.close();
    throw mediaError(415, 'Thumbnails support PNG, JPEG, WebP, GIF, MP4, MOV, and WebM artifacts');
  }
  const key = digest(`jpeg-v1:${source.filePath}:${fingerprint(source.info)}:${width}`);
  let entry = thumbnails.get(key);
  if (entry) {
    entry.users += 1;
    thumbnails.delete(key);
    thumbnails.set(key, entry);
    await source.handle.close();
  } else {
    entry = {users: 1, ready: false, controller: new AbortController(), filePath: null, promise: null};
    entry.promise = buildThumbnail(source, key, width, format, entry.controller.signal).then((filePath) => {
      entry.ready = true;
      entry.filePath = filePath;
      return filePath;
    }, (error) => {
      if (thumbnails.get(key) === entry) thumbnails.delete(key);
      throw error;
    });
    thumbnails.set(key, entry);
  }
  let disconnected;
  const aborted = new Promise((_, reject) => {
    disconnected = () => reject(mediaError(499, 'Thumbnail request was cancelled'));
    response.once('close', disconnected);
    if (response.destroyed) disconnected();
  });
  try {
    const filePath = await Promise.race([entry.promise, aborted]);
    response.removeListener('close', disconnected);
    if (response.destroyed) return;
    const handle = await open(filePath, constants.O_RDONLY);
    let info;
    try {
      info = await handle.stat({bigint: true});
    } catch (error) {
      await handle.close();
      throw error;
    }
    await deliverFile(request, response, {handle, info}, 'image/jpeg');
  } finally {
    response.removeListener('close', disconnected);
    entry.users -= 1;
    if (!entry.users && !entry.ready) {
      if (thumbnails.get(key) === entry) thumbnails.delete(key);
      entry.controller.abort();
    }
    await trimThumbnails();
  }
}

export async function closeMediaCache() {
  closing = true;
  for (const entry of thumbnails.values()) entry.controller.abort();
  for (const child of thumbnailChildren) child.kill('SIGKILL');
  await Promise.allSettled([...thumbnails.values()].map((entry) => entry.promise));
  thumbnails.clear();
  if (thumbnailDirectory) await rm(await thumbnailDirectory, {recursive: true, force: true});
}
