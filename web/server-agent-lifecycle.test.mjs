import {existsSync} from 'node:fs';
import {mkdir, readFile, writeFile} from 'node:fs/promises';
import path from 'node:path';
import {describe, expect, it} from 'vitest';
import {startAgentServer} from './test/agent-server-fixture.mjs';

async function queueSnapshot(server) {
  const response = await server.request('/api/events');
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let text = '';
  try {
    while (true) {
      const chunk = await reader.read();
      if (chunk.done) throw new Error('Event stream ended before queue snapshot');
      text += decoder.decode(chunk.value, {stream: true});
      for (const line of text.split('\n').slice(0, -1)) {
        if (!line.startsWith('data: ')) continue;
        const event = JSON.parse(line.slice(6));
        if (event.type === 'agent_queue') return event;
      }
    }
  } finally {
    await reader.cancel();
  }
}

describe('agent replacement safety', () => {
  it('preserves active and queued work when creating, switching, or restarting; permits replacement after completion', async () => {
    const server = await startAgentServer();
    try {
      expect((await server.request('/api/agent/start', {sessionId: 'original'})).status).toBe(200);
      expect((await server.request('/api/messages', {sessionId: 'original', text: 'first', messageId: 'first'})).status).toBe(202);
      await expect.poll(() => existsSync(path.join(server.root, 'started-first'))).toBe(true);
      const pid = await readFile(path.join(server.root, 'started-first'), 'utf8');
      const queued = await server.request('/api/messages', {sessionId: 'original', text: 'second', messageId: 'second'});
      expect(await queued.json()).toMatchObject({queued: true});
      const configSave = await fetch(server.url + '/api/config', {method: 'PUT', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({sections: {video: {model: 'agnes-video-2.5', shot_duration_limits: {'agnes:agnes-video-2.5': 12}}}})});
      expect(configSave.status).toBe(409);
      expect(existsSync(path.join(server.root, 'configs', 'agent.local.yaml'))).toBe(false);

      for (const body of [{newSession: true, projectName: 'New project'}, {sessionId: 'other'}, {sessionId: 'original'}]) {
        const response = await server.request('/api/agent/start', body);
        expect(response.status).toBe(409);
        expect(await response.json()).toMatchObject({error: expect.stringContaining('active or queued work')});
        expect(await queueSnapshot(server)).toMatchObject({activeSessionId: 'original', active: {id: 'first'}, pending: [{id: 'second'}]});
        expect(existsSync(path.join(server.root, 'terminated-' + pid))).toBe(false);
      }

      await server.release('first');
      await expect.poll(() => existsSync(path.join(server.root, 'started-second'))).toBe(true);
      expect(await readFile(path.join(server.root, 'started-second'), 'utf8')).toBe(pid);
      expect((await server.request('/api/agent/start', {newSession: true, projectName: 'New project'})).status).toBe(409);
      await server.release('second');
      await expect.poll(async () => (await queueSnapshot(server)).busy).toBe(false);
      expect(existsSync(path.join(server.root, 'completed-first'))).toBe(true);
      expect(existsSync(path.join(server.root, 'completed-second'))).toBe(true);
      expect(existsSync(path.join(server.root, 'terminated-' + pid))).toBe(false);
      expect((await server.request('/api/agent/start', {sessionId: 'other'})).status).toBe(200);
      expect(await (await server.request('/api/health')).json()).toMatchObject({agentRunning: true, activeSessionId: 'other'});
    } finally {
      await server.close();
    }
  }, 20000);
  it('refuses deletion during active or queued work and preserves an unrelated active agent', async () => {
    const server = await startAgentServer();
    const deleteProject = (sessionId) => fetch(`${server.url}/api/sessions?session=${sessionId}`, {method: 'DELETE'});
    try {
      expect((await server.request('/api/agent/start', {sessionId: 'original'})).status).toBe(200);
      expect((await server.request('/api/messages', {sessionId: 'original', text: 'first', messageId: 'first'})).status).toBe(202);
      await expect.poll(() => existsSync(path.join(server.root, 'started-first'))).toBe(true);
      const pid = await readFile(path.join(server.root, 'started-first'), 'utf8');
      await server.request('/api/messages', {sessionId: 'original', text: 'second', messageId: 'second'});

      const refused = await deleteProject('other');
      expect(refused.status).toBe(409);
      expect(await refused.json()).toMatchObject({error: expect.stringContaining('active or queued work')});
      expect(await queueSnapshot(server)).toMatchObject({activeSessionId: 'original', active: {id: 'first'}, pending: [{id: 'second'}]});
      expect(existsSync(path.join(server.root, 'terminated-' + pid))).toBe(false);
      expect(existsSync(path.join(server.root, '.working_dir', 'other'))).toBe(true);

      await server.release('first');
      await expect.poll(() => existsSync(path.join(server.root, 'started-second'))).toBe(true);
      await server.release('second');
      await expect.poll(async () => (await queueSnapshot(server)).busy).toBe(false);
      const registryPath = path.join(server.root, '.vimax', 'sessions.json');
      const registry = JSON.parse(await readFile(registryPath, 'utf8'));
      registry.active_session_id = 'other';
      registry.sessions.fallback = {
        session_id: 'fallback',
        project_name: 'Stored fallback',
        working_dir: '.working_dir/fallback',
        updated_at: '2026-10-01T10:00:00',
      };
      await writeFile(registryPath, JSON.stringify(registry));

      const deletedOther = await deleteProject('other');
      expect(deletedOther.status).toBe(200);
      expect(await deletedOther.json()).toMatchObject({
        activeSessionId: 'fallback', sessions: [{sessionId: 'fallback'}, {sessionId: 'original'}],
      });
      expect(await (await server.request('/api/health')).json()).toMatchObject({agentRunning: true, activeSessionId: 'original'});
      expect(existsSync(path.join(server.root, '.working_dir', 'other'))).toBe(false);
      expect(existsSync(path.join(server.root, 'terminated-' + pid))).toBe(false);

      const deletedActive = await deleteProject('original');
      expect(deletedActive.status).toBe(200);
      expect(await deletedActive.json()).toMatchObject({activeSessionId: 'fallback', sessions: [{sessionId: 'fallback'}]});
      await expect.poll(() => existsSync(path.join(server.root, 'terminated-' + pid))).toBe(true);
      expect(await (await server.request('/api/health')).json()).toMatchObject({agentRunning: false, activeSessionId: 'fallback'});

      const deletedFallback = await deleteProject('fallback');
      expect(deletedFallback.status).toBe(200);
      expect(await deletedFallback.json()).toMatchObject({activeSessionId: '', sessions: []});
    } finally {
      await server.close();
    }
  }, 20000);
});
