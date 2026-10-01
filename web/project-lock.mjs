import {randomUUID} from 'node:crypto';
import {mkdir, readFile, rm, writeFile} from 'node:fs/promises';
import path from 'node:path';

const queues = new Map();
const lockName = '.timeline-write-lock';
const reaperName = '.timeline-write-lock-reclaim';

export async function withProjectWriteLock(workingDir, operation) {
  const key = path.resolve(workingDir);
  const previous = queues.get(key) || Promise.resolve();
  let releaseQueue;
  const turn = new Promise((resolve) => { releaseQueue = resolve; });
  const queued = previous.then(() => turn);
  queues.set(key, queued);
  await previous;
  let lock;
  try {
    lock = await acquire(key);
    return await operation();
  } finally {
    if (lock) await release(lock);
    releaseQueue();
    if (queues.get(key) === queued) queues.delete(key);
  }
}

async function acquire(workingDir) {
  const directory = path.join(workingDir, lockName);
  const token = randomUUID();
  try {
    await mkdir(directory);
    try {
      await writeFile(path.join(directory, 'owner.json'), JSON.stringify({pid: process.pid, token}), {flag: 'wx', mode: 0o600});
    } catch (error) {
      await rm(directory, {recursive: true, force: true}).catch(() => {});
      throw error;
    }
    return {directory, token};
  } catch (error) {
    if (error.code !== 'EEXIST') throw error;
  }

  const staleOwner = await readOwner(directory);
  if (!staleOwner || isAlive(staleOwner.pid)) throw busyError();

  // A separate atomic guard serializes stale-owner recovery across Node and Python. The
  // second owner check occurs under this guard, so no reaper can remove a replacement lock.
  const reaper = path.join(workingDir, reaperName);
  const reaperToken = randomUUID();
  try {
    await mkdir(reaper);
  } catch (error) {
    if (error.code === 'EEXIST') throw busyError();
    throw error;
  }
  let guardMetadataWritten = false;
  try {
    await writeFile(path.join(reaper, 'owner.json'), JSON.stringify({pid: process.pid, token: reaperToken}), {flag: 'wx', mode: 0o600});
    guardMetadataWritten = true;
    const current = await readOwner(directory);
    if (!current || current.pid !== staleOwner.pid || current.token !== staleOwner.token || isAlive(current.pid)) throw busyError();
    await rm(directory, {recursive: true});
  } finally {
    if (guardMetadataWritten) await release({directory: reaper, token: reaperToken});
    else await rm(reaper, {recursive: true, force: true}).catch(() => {});
  }
  return acquire(workingDir);
}

async function readOwner(directory) {
  try {
    const owner = JSON.parse(await readFile(path.join(directory, 'owner.json'), 'utf8'));
    return Number.isSafeInteger(owner?.pid) && owner.pid > 0 && typeof owner.token === 'string' && owner.token
      ? {pid: owner.pid, token: owner.token}
      : null;
  } catch {
    return null;
  }
}

function isAlive(pid) {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    if (error.code === 'ESRCH') return false;
    if (error.code === 'EPERM') return true;
    throw error;
  }
}

async function release(lock) {
  try {
    const owner = await readOwner(lock.directory);
    if (owner?.pid === process.pid && owner.token === lock.token) await rm(lock.directory, {recursive: true});
  } catch { /* A replaced or damaged lock belongs to someone else. */ }
}

function busyError() {
  const error = new Error('This project is being changed by another writer');
  error.code = 'PROJECT_BUSY';
  error.statusCode = 409;
  return error;
}
