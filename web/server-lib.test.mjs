import {mkdtemp, mkdir, readFile, rename, stat, writeFile} from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {afterEach, describe, expect, it} from 'vitest';
import {deleteSession, listSessionArtifacts, readSessionHistory, readSessionState, resolveArtifactPath, storeWorkspaceUpload, updateRenderAcceptance} from './server-lib.mjs';

const png = (payload) => Buffer.concat([
  Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
  Buffer.from(payload),
]);
const roots = [];

afterEach(async () => {
  const {rm} = await import('node:fs/promises');
  await Promise.all(roots.splice(0).map((root) => rm(root, {recursive: true, force: true})));
});

async function fixture() {
  const root = await mkdtemp(path.join(os.tmpdir(), 'vimax-web-'));
  roots.push(root);
  await mkdir(path.join(root, '.vimax', 'logs'), {recursive: true});
  await mkdir(path.join(root, '.working_dir', 'session-1', 'script2video', 'shots', '0'), {recursive: true});
  await writeFile(path.join(root, '.vimax', 'sessions.json'), JSON.stringify({
    active_session_id: 'session-1',
    sessions: {
      'session-1': {session_id: 'session-1', project_name: 'Ocean campaign', working_dir: '.working_dir/session-1', stage: 'rendering', updated_at: '2026-07-17T10:00:00'},
    },
  }));
  return root;
}

describe('web bridge state', () => {
  it('returns sanitized session records', async () => {
    const root = await fixture();
    const state = await readSessionState(root);
    expect(state.activeSessionId).toBe('session-1');
    expect(state.sessions[0]).toMatchObject({sessionId: 'session-1', projectName: 'Ocean campaign', stage: 'rendering'});
  });

  it('restores persisted turn history', async () => {
    const root = await fixture();
    await writeFile(path.join(root, '.vimax', 'logs', 'loop_history.jsonl'), `${JSON.stringify({
      session_id: 'session-1', turn_id: 'turn-1', raw_user_input: 'Make a film', final_assistant_text: 'Planning is ready', status: 'completed', tool_rounds: [],
    })}\n`);
    const history = await readSessionHistory(root, 'session-1');
    expect(history.map((message) => message.role)).toEqual(['user', 'assistant']);
  });

  it('retains the last 120 matching messages with original ordering and generated IDs', async () => {
    const root = await fixture();
    const logPath = path.join(root, '.vimax', 'logs', 'loop_history.jsonl');
    const rows = [];
    for (let turn = 0; turn <= 60; turn += 1) {
      rows.push(JSON.stringify({
        session_id: 'session-1',
        turn_id: `turn-${turn}`,
        raw_user_input: `prompt ${turn}`,
        final_assistant_text: `answer ${turn}`,
      }));
    }
    rows.splice(10, 0, '{malformed json', JSON.stringify({
      session_id: 'another-session', turn_id: 'ignored', raw_user_input: 'ignore me',
    }));
    rows.push(JSON.stringify({
      session_id: 'session-1',
      raw_user_input: 'prompt 61',
      tool_rounds: [{tool_results: [{name: 'render', ok: true}]}],
      final_assistant_text: 'answer 61',
    }));
    await writeFile(logPath, `${rows.join('\n')}\n`);

    const history = await readSessionHistory(root, 'session-1');
    expect(history).toHaveLength(120);
    expect(history[0]).toMatchObject({id: 'turn-2-assistant', role: 'assistant', text: 'answer 2'});
    expect(history.slice(-3).map((message) => message.id)).toEqual([
      'turn-122-user', 'turn-122-tool-123', 'turn-122-assistant',
    ]);
    expect(history.map((message) => message.text)).not.toContain('ignore me');
  });

  it('reuses unchanged history and invalidates cached projections on replacement, append, or removal', async () => {
    const root = await fixture();
    const logPath = path.join(root, '.vimax', 'logs', 'loop_history.jsonl');
    const initial = `${JSON.stringify({session_id: 'session-1', turn_id: 'turn-old', raw_user_input: 'first'})}\n`;
    await writeFile(logPath, initial);
    const first = await readSessionHistory(root, 'session-1');
    first[0].text = 'caller mutation';
    expect((await readSessionHistory(root, 'session-1'))[0].text).toBe('first');

    const replacementPath = `${logPath}.replacement`;
    await writeFile(replacementPath, `${JSON.stringify({session_id: 'session-1', turn_id: 'turn-new', raw_user_input: 'other'})}\n`);
    await rename(replacementPath, logPath);
    expect((await readSessionHistory(root, 'session-1'))[0].id).toBe('turn-new-user');
    await writeFile(logPath, `${JSON.stringify({session_id: 'session-1', turn_id: 'turn-now', raw_user_input: 'other'})}\n`);
    expect((await readSessionHistory(root, 'session-1'))[0].id).toBe('turn-now-user');

    await writeFile(logPath, `{not complete\n${JSON.stringify({
      session_id: 'session-1', turn_id: 'turn-appended', raw_user_input: 'later',
    })}\n`, {flag: 'a'});
    expect((await readSessionHistory(root, 'session-1')).map((message) => message.id)).toEqual([
      'turn-now-user', 'turn-appended-user',
    ]);
    await import('node:fs/promises').then(({rm}) => rm(logPath));
    expect(await readSessionHistory(root, 'session-1')).toEqual([]);
  });

  it('hides workspace upload metadata from restored user messages', async () => {
    const root = await fixture();
    await writeFile(path.join(root, '.vimax', 'logs', 'loop_history.jsonl'), `${JSON.stringify({
      session_id: 'session-1',
      turn_id: 'turn-upload',
      raw_user_input: 'Use this script <workspace_uploads>["uploads/script.txt"]</workspace_uploads>',
    })}\n`);
    const history = await readSessionHistory(root, 'session-1');
    expect(history[0].text).toBe('Use this script');
  });

  it('does not expose successful tool result payloads in restored history', async () => {
    const root = await fixture();
    await writeFile(path.join(root, '.vimax', 'logs', 'loop_history.jsonl'), `${JSON.stringify({
      session_id: 'session-1',
      turn_id: 'turn-tool',
      raw_user_input: 'Plan a video',
      tool_rounds: [{tool_results: [{
        name: 'vimax_narrative_planning',
        ok: true,
        content: JSON.stringify({session_id: 'session-1', working_dir: '.working_dir/session-1', generated: ['script.json']}),
      }]}],
    })}\n`);
    const history = await readSessionHistory(root, 'session-1');
    const activity = history.find((message) => message.role === 'activity');
    expect(activity).toMatchObject({text: 'Completed', status: 'done', stage: 'completed'});
    expect(JSON.stringify(activity)).not.toContain('working_dir');
  });

  it('lists media artifacts and blocks path traversal', async () => {
    const root = await fixture();
    await writeFile(path.join(root, '.working_dir', 'session-1', 'script2video', 'shots', '0', 'first_frame.png'), 'image');
    const artifacts = await listSessionArtifacts(root, 'session-1');
    expect(artifacts[0]).toMatchObject({kind: 'image', name: 'first_frame.png'});
    expect(() => resolveArtifactPath(root, 'session-1', '../../secrets')).toThrow(/escapes/);
  });

  it('keeps render scratch out of the browsable artifacts', async () => {
    const root = await fixture();
    const shot = path.join(root, '.working_dir', 'session-1', 'script2video', 'shots', '0', 'qwen_qwen-image-3');
    await mkdir(path.join(shot, 'cache'), {recursive: true});
    await writeFile(path.join(shot, 'first_frame.png'), 'image');
    await writeFile(path.join(shot, 'cache', 'transition-Scene-001.mp4'), 'scratch');

    const artifacts = await listSessionArtifacts(root, 'session-1');

    expect(artifacts.map((artifact) => artifact.name)).toEqual(['first_frame.png']);
  });

  it('stores uploads inside the session without overwriting matching names', async () => {
    const root = await fixture();
    const first = await storeWorkspaceUpload(root, 'session-1', 'script.txt', Buffer.from('first'));
    const second = await storeWorkspaceUpload(root, 'session-1', 'script.txt', Buffer.from('second'));
    expect(first).toMatchObject({name: 'script.txt', path: 'uploads/script.txt', size: 5});
    expect(second).toMatchObject({name: 'script (2).txt', path: 'uploads/script (2).txt', size: 6});
    expect(await readFile(path.join(root, '.working_dir', 'session-1', second.path), 'utf8')).toBe('second');
    await expect(storeWorkspaceUpload(root, 'session-1', '../escape.txt', Buffer.from('no'))).rejects.toThrow(/unsupported/);
  });

  it('distinguishes registry-selected portraits from backups and stale approvals', async () => {
    const repo = await fixture();
    const session = path.join(repo, '.working_dir', 'session-1');
    const portraits = path.join(session, 'script2video', 'character_portraits');
    const current = path.join(portraits, 'test_model', '0_Claude');
    const backup = path.join(portraits, '_backup');
    await Promise.all([mkdir(current, {recursive: true}), mkdir(backup, {recursive: true})]);
    await writeFile(path.join(session, 'render_manifest.json'), JSON.stringify({render_mode: 'script2video', image_model: 'test/model'}));
    await writeFile(path.join(session, 'script2video', 'characters.json'), JSON.stringify([{identifier_in_scene: 'Claude', is_visible: true}]));
    const entry = {};
    for (const view of ['front', 'side', 'back']) {
      const file = path.join(current, `${view}.png`);
      await writeFile(file, png(view));
      await writeFile(path.join(backup, `${view}.png`), png(`old ${view}`));
      entry[view] = {path: file};
    }
    const registry = path.join(portraits, 'test_model', 'registry.json');
    await writeFile(registry, JSON.stringify({Claude: entry}));
    await writeFile(path.join(backup, 'prompts_weathered.json'), JSON.stringify({front: 'Saved backup prompt'}));
    let artifacts = await listSessionArtifacts(repo, 'session-1');
    expect(artifacts.filter((artifact) => artifact.portrait?.active).map((artifact) => artifact.name).sort()).toEqual(['back.png', 'front.png', 'side.png']);
    expect(artifacts.find((artifact) => artifact.path.endsWith('_backup/front.png'))).toMatchObject({
      portrait: {active: false, accepted: false, view: 'front'}, promptPath: 'script2video/character_portraits/_backup/prompts_weathered.json',
    });
    await updateRenderAcceptance(repo, {sessionId: 'session-1', root: 'script2video', stage: 'portraits', accepted: true});
    artifacts = await listSessionArtifacts(repo, 'session-1');
    expect(artifacts.filter((artifact) => artifact.portrait?.accepted)).toHaveLength(3);
    await writeFile(entry.front.path, png('other'));
    artifacts = await listSessionArtifacts(repo, 'session-1');
    expect(artifacts.filter((artifact) => artifact.portrait?.accepted)).toEqual([]);
    expect(artifacts.filter((artifact) => artifact.portrait?.active)).toHaveLength(3);
    await writeFile(registry, JSON.stringify({}));
    artifacts = await listSessionArtifacts(repo, 'session-1');
    expect(artifacts.filter((artifact) => artifact.portrait?.active)).toEqual([]);
  });

  it('keeps partial selected paths active without accepting incomplete or invisible character sets', async () => {
    const repo = await fixture();
    const session = path.join(repo, '.working_dir', 'session-1');
    const directory = path.join(session, 'script2video/character_portraits/test_model');
    await mkdir(directory, {recursive: true});
    await writeFile(path.join(session, 'render_manifest.json'), JSON.stringify({render_mode: 'script2video', image_model: 'test/model'}));
    await writeFile(path.join(session, 'script2video', 'characters.json'), JSON.stringify([
      {identifier_in_scene: 'Claude', is_visible: true}, {identifier_in_scene: 'Offscreen', is_visible: false},
    ]));
    const registry = {};
    for (const character of ['Claude', 'Offscreen']) {
      registry[character] = {};
      const characterDir = path.join(directory, character);
      await mkdir(characterDir);
      for (const view of ['front', 'side', 'back']) {
        const file = path.join(characterDir, `${view}.png`);
        if (character !== 'Claude' || view !== 'back') await writeFile(file, png(view));
        registry[character][view] = {path: file};
      }
    }
    await writeFile(path.join(directory, 'registry.json'), JSON.stringify(registry));
    const artifacts = await listSessionArtifacts(repo, 'session-1');
    expect(artifacts.filter((artifact) => artifact.portrait?.active).map((artifact) => artifact.name).sort())
      .toEqual(['front.png', 'side.png']);
    expect(artifacts.filter((artifact) => artifact.portrait?.accepted)).toEqual([]);
  });

  it('deletes project state, artifacts, and matching log records', async () => {
    const root = await fixture();
    const logPath = path.join(root, '.vimax', 'logs', 'loop_history.jsonl');
    await writeFile(logPath, [
      JSON.stringify({session_id: 'session-1', raw_user_input: 'Delete me'}),
      JSON.stringify({session_id: 'session-2', raw_user_input: 'Keep me'}),
      '',
    ].join('\n'));

    const state = await deleteSession(root, 'session-1');
    expect(state).toEqual({activeSessionId: '', sessions: []});
    await expect(stat(path.join(root, '.working_dir', 'session-1'))).rejects.toThrow();
    expect(await readFile(logPath, 'utf8')).toContain('session-2');
    expect(await readFile(logPath, 'utf8')).not.toContain('session-1');
  });
});
