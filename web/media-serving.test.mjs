import {createServer} from 'node:http';
import {mkdtemp, mkdir, rename, rm, stat, symlink, utimes, writeFile} from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {afterEach, describe, expect, it} from 'vitest';
import {serveArtifact} from './media-serving.mjs';

const fixtures = [];

afterEach(async () => {
  for (const {server, root} of fixtures.splice(0)) {
    server.closeAllConnections();
    await new Promise((resolve) => server.close(resolve));
    await rm(root, {recursive: true, force: true});
  }
});

async function fixture() {
  const root = await mkdtemp(path.join(os.tmpdir(), 'vimax-media-'));
  const directory = path.join(root, '.working_dir', 'media-session');
  await mkdir(directory, {recursive: true});
  const file = path.join(directory, 'clip.mp4');
  await writeFile(file, '0123456789');
  const server = createServer((request, response) => {
    const url = new URL(request.url, 'http://local');
    serveArtifact(request, response, root, 'media-session', url.searchParams.get('path') || 'clip.mp4')
      .catch((error) => {
        if (response.headersSent || response.destroyed) return response.destroy();
        response.writeHead(error.statusCode || 500);
        response.end(error.message);
      });
  });
  fixtures.push({server, root});
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  return {root, directory, file, url: `http://127.0.0.1:${server.address().port}/`};
}

describe('original media delivery', () => {
  it('names final-film downloads after the project while preserving inline range playback', async () => {
    const {url, root, directory} = await fixture();
    await mkdir(path.join(root, '.vimax'));
    await writeFile(path.join(root, '.vimax', 'sessions.json'), JSON.stringify({
      sessions: {'media-session': {session_id: 'media-session', project_name: 'Claude’s café'}},
    }));
    await writeFile(path.join(directory, 'final_video.mp4'), '0123456789');
    const response = await fetch(`${url}?path=final_video.mp4`, {headers: {Range: 'bytes=2-4'}});
    expect(response.status).toBe(206);
    expect(response.headers.get('content-disposition')).toBe('inline; filename="Claude_s caf_.mp4"; filename*=UTF-8\'\'Claude%E2%80%99s%20caf%C3%A9.mp4');
    expect(await response.text()).toBe('234');
    const head = await fetch(`${url}?path=final_video.mp4`, {method: 'HEAD'});
    expect(head.headers.get('content-disposition')).toBe(response.headers.get('content-disposition'));
    expect(await head.text()).toBe('');
  });

  it('makes unsafe project-title characters safe and falls back to the session when unnamed', async () => {
    const {url, root, directory} = await fixture();
    await writeFile(path.join(directory, 'final_video.mp4'), '0123456789');
    const unnamed = await fetch(`${url}?path=final_video.mp4`);
    expect(unnamed.headers.get('content-disposition')).toBe('inline; filename="media-session.mp4"; filename*=UTF-8\'\'media-session.mp4');
    await unnamed.arrayBuffer();
    await mkdir(path.join(root, '.vimax'));
    await writeFile(path.join(root, '.vimax', 'sessions.json'), JSON.stringify({
      sessions: {'media-session': {session_id: 'media-session', project_name: 'Film/"title"\r\n'}},
    }));
    const unsafe = await fetch(`${url}?path=final_video.mp4`);
    expect(unsafe.headers.get('content-disposition')).toBe('inline; filename="Film--title-.mp4"; filename*=UTF-8\'\'Film--title-.mp4');
    expect(await unsafe.text()).toBe('0123456789');
  });

  it('serves suffix and open ranges without sending bytes outside the requested interval', async () => {
    const {url} = await fixture();
    const suffix = await fetch(url, {headers: {Range: 'bytes=-3'}});
    expect(suffix.status).toBe(206);
    expect(suffix.headers.get('content-range')).toBe('bytes 7-9/10');
    expect(suffix.headers.get('content-length')).toBe('3');
    expect(await suffix.text()).toBe('789');

    const open = await fetch(url, {headers: {Range: 'bytes=8-'}});
    expect(open.status).toBe(206);
    expect(await open.text()).toBe('89');

    const outside = await fetch(url, {headers: {Range: 'bytes=10-'}});
    expect(outside.status).toBe(416);
    expect(outside.headers.get('content-range')).toBe('bytes */10');
    expect(await outside.text()).toBe('');

    const head = await fetch(url, {method: 'HEAD', headers: {Range: 'bytes=1-2'}});
    expect(head.status).toBe(200);
    expect(head.headers.get('content-length')).toBe('10');
    expect(await head.text()).toBe('');
  });

  it('revalidates originals and refuses stale range validators after replacement', async () => {
    const {url, file} = await fixture();
    const original = await fetch(url);
    const etag = original.headers.get('etag');
    expect(await original.text()).toBe('0123456789');
    const unchanged = await fetch(url, {headers: {'If-None-Match': etag}});
    expect(unchanged.status).toBe(304);
    expect(await unchanged.text()).toBe('');

    const old = await stat(file);
    const replacement = `${file}.new`;
    await writeFile(replacement, 'abcdefghij');
    await utimes(replacement, old.atime, old.mtime);
    await rename(replacement, file);
    const changed = await fetch(url, {headers: {Range: 'bytes=2-4', 'If-Range': etag}});
    expect(changed.status).toBe(200);
    expect(changed.headers.get('etag')).not.toBe(etag);
    expect(await changed.text()).toBe('abcdefghij');
  });

  it('does not follow a session artifact symlink into unrelated files', async () => {
    const {url, root, directory} = await fixture();
    const privateFile = path.join(root, 'private.txt');
    await writeFile(privateFile, 'not an artifact');
    await symlink(privateFile, path.join(directory, 'linked.txt'));
    const response = await fetch(`${url}?path=linked.txt`);
    expect(response.status).toBe(403);
    expect(await response.text()).not.toContain('not an artifact');
  });
});
