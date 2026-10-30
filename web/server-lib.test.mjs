import {mkdtemp, mkdir, readFile, rename, rm, stat, symlink, writeFile} from 'node:fs/promises';
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

  it('projects persisted assistant samples and tool results in chronological canonical order', async () => {
    const root = await fixture();
    await writeFile(path.join(root, '.vimax', 'logs', 'loop_history.jsonl'), `${JSON.stringify({
      session_id: 'session-1',
      turn_id: 'turn-streamed',
      raw_user_input: 'Make a film',
      status: 'completed',
      assistant_turns: [
        {sample_id: 'turn-streamed-sample-0', text: 'Preparing', outcome: 'tool_preamble', phase: 'initial', tool_calls: [{id: 'render-call'}]},
        {sample_id: 'turn-streamed-sample-1', text: 'Finished', outcome: 'final', phase: 'followup_1'},
      ],
      tool_rounds: [{tool_round: 1, requested_tools: [{id: 'render-call', name: 'render'}], tool_results: [{name: 'render', ok: true}]}],
      final_assistant_text: 'Finished',
    })}\n`);
    const history = await readSessionHistory(root, 'session-1');
    expect(history.map(({id, role}) => [id, role])).toEqual([
      ['turn-streamed-user', 'user'],
      ['assistant-turn-streamed-sample-0', 'assistant'],
      ['activity-turn-streamed-render-call', 'activity'],
      ['assistant-turn-streamed-sample-1', 'assistant'],
    ]);
    expect(history[1]).toMatchObject({turnId: 'turn-streamed', sampleId: 'turn-streamed-sample-0', streamState: 'preamble', text: 'Preparing'});
    expect(history[2]).toMatchObject({turnId: 'turn-streamed', toolCallId: 'render-call', tool: 'render', status: 'done'});
    expect(history[3].text).toBe('Finished');
  });

  it('retains a distinct turn-level final message when a halted turn has no final sample', async () => {
    const root = await fixture();
    await writeFile(path.join(root, '.vimax', 'logs', 'loop_history.jsonl'), `${JSON.stringify({
      session_id: 'session-1',
      turn_id: 'turn-limit',
      raw_user_input: 'Render a film',
      status: 'halted',
      assistant_turns: [
        {sample_id: 'turn-limit-sample-0', text: 'Preparing render', outcome: 'tool_preamble', phase: 'initial'},
      ],
      tool_rounds: [{tool_round: 1, tool_results: [{name: 'render', ok: true}]}],
      final_sample_id: null,
      final_assistant_text: 'Stopped after the maximum tool passes.',
    })}\n`);
    const history = await readSessionHistory(root, 'session-1');
    expect(history.map(({id, role}) => [id, role])).toEqual([
      ['turn-limit-user', 'user'],
      ['assistant-turn-limit-sample-0', 'assistant'],
      ['activity-turn-limit-tool-1-0', 'activity'],
      ['assistant-turn-limit', 'assistant'],
    ]);
    expect(history[3].text).toBe('Stopped after the maximum tool passes.');
  });

  it('retains bounded matching history in chronological order with distinct identities', async () => {
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
    expect(history[0]).toMatchObject({role: 'assistant', text: 'answer 2'});
    expect(history.slice(-3).map((message) => message.role)).toEqual(['user', 'activity', 'assistant']);
    expect(history.at(-3).text).toBe('prompt 61');
    expect(history.at(-1).text).toBe('answer 61');
    expect(new Set(history.map((message) => message.id)).size).toBe(history.length);
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

  it('deletes recorded and canonical project directories, including uploads, while preserving another project', async () => {
    const root = await fixture();
    const canonical = path.join(root, '.working_dir', 'session-1');
    const recorded = path.join(root, '.working_dir', 'recorded-session-1');
    const other = path.join(root, '.working_dir', 'session-2');
    await mkdir(path.join(recorded, 'uploads'), {recursive: true});
    await mkdir(path.join(canonical, 'uploads'), {recursive: true});
    await mkdir(path.join(other, 'uploads'), {recursive: true});
    await writeFile(path.join(recorded, 'uploads', 'story.txt'), 'recorded project upload');
    await writeFile(path.join(canonical, 'uploads', 'cover.png'), 'canonical project upload');
    await writeFile(path.join(other, 'uploads', 'keep.txt'), 'another project');

    const statePath = path.join(root, '.vimax', 'sessions.json');
    const payload = JSON.parse(await readFile(statePath, 'utf8'));
    payload.sessions['session-1'].working_dir = '.working_dir/recorded-session-1';
    payload.sessions['session-2'] = {
      session_id: 'session-2',
      project_name: 'Keep this project',
      working_dir: '.working_dir/session-2',
      updated_at: '2026-07-18T10:00:00',
    };
    await writeFile(statePath, JSON.stringify(payload));

    const logPath = path.join(root, '.vimax', 'logs', 'loop_history.jsonl');
    await writeFile(logPath, [
      JSON.stringify({session_id: 'session-1', raw_user_input: 'Delete me'}),
      JSON.stringify({session_id: 'session-2', raw_user_input: 'Keep me'}),
      '',
    ].join('\n'));

    const state = await deleteSession(root, 'session-1');
    expect(state.activeSessionId).toBe('session-2');
    expect(state.sessions.map((session) => session.sessionId)).toEqual(['session-2']);
    await expect(stat(canonical)).rejects.toThrow();
    await expect(stat(recorded)).rejects.toThrow();
    await expect(stat(path.join(other, 'uploads', 'keep.txt'))).resolves.toBeDefined();
    expect(await readFile(logPath, 'utf8')).toContain('session-2');
    expect(await readFile(logPath, 'utf8')).not.toContain('session-1');
  });

  it('rejects shared and escaping recorded directories before changing project state', async () => {
    const root = await fixture();
    const statePath = path.join(root, '.vimax', 'sessions.json');
    const original = JSON.parse(await readFile(statePath, 'utf8'));
    original.sessions['session-2'] = {
      session_id: 'session-2',
      project_name: 'Keep this project',
      working_dir: '.working_dir/recorded-session-2',
      updated_at: '2026-07-18T10:00:00',
    };
    const otherProject = path.join(root, '.working_dir', 'session-2');
    await mkdir(otherProject);
    const otherMarker = path.join(otherProject, 'keep.txt');
    await writeFile(otherMarker, 'preserve');
    const otherRecorded = path.join(root, '.working_dir', 'recorded-session-2');
    await mkdir(otherRecorded);
    const otherRecordedMarker = path.join(otherRecorded, 'keep.txt');
    await writeFile(otherRecordedMarker, 'preserve');
    const marker = path.join(root, '.working_dir', 'unrelated.txt');
    await writeFile(marker, 'preserve');
    const outside = await mkdtemp(path.join(os.tmpdir(), 'vimax-delete-outside-'));
    roots.push(outside);
    const outsideMarker = path.join(outside, 'keep.txt');
    await writeFile(outsideMarker, 'preserve');

    for (const workingDir of ['.working_dir', '.working_dir/session-2', outside]) {
      const payload = JSON.parse(JSON.stringify(original));
      payload.sessions['session-1'].working_dir = workingDir;
      await writeFile(statePath, JSON.stringify(payload));
      await expect(deleteSession(root, 'session-1')).rejects.toMatchObject({statusCode: 400});
      expect(await readFile(statePath, 'utf8')).toContain('"session-1"');
      await expect(stat(path.join(root, '.working_dir', 'session-1'))).resolves.toBeDefined();
      expect(await readFile(marker, 'utf8')).toBe('preserve');
      expect(await readFile(otherMarker, 'utf8')).toBe('preserve');
      expect(await readFile(otherRecordedMarker, 'utf8')).toBe('preserve');
      expect(await readFile(outsideMarker, 'utf8')).toBe('preserve');
    }
  });

  it('rejects symlinked project storage without touching the linked directory', async () => {
    const root = await fixture();
    const sessionRoot = path.join(root, '.working_dir', 'session-1');
    const outside = await mkdtemp(path.join(os.tmpdir(), 'vimax-delete-symlink-'));
    roots.push(outside);
    const marker = path.join(outside, 'keep.txt');
    await writeFile(marker, 'preserve');
    await rm(sessionRoot, {recursive: true});
    await symlink(outside, sessionRoot);

    await expect(deleteSession(root, 'session-1')).rejects.toMatchObject({statusCode: 400});
    expect(await readFile(marker, 'utf8')).toBe('preserve');
    expect(await readSessionState(root)).toMatchObject({activeSessionId: 'session-1'});
  });

  it('tolerates missing project directories and refuses a live project writer lock', async () => {
    const missingRoot = await fixture();
    await rm(path.join(missingRoot, '.working_dir', 'session-1'), {recursive: true});
    await expect(deleteSession(missingRoot, 'session-1')).resolves.toMatchObject({activeSessionId: '', sessions: []});

    const busyRoot = await fixture();
    const sessionRoot = path.join(busyRoot, '.working_dir', 'session-1');
    const lock = path.join(sessionRoot, '.timeline-write-lock');
    await mkdir(lock);
    await writeFile(path.join(lock, 'owner.json'), JSON.stringify({pid: process.pid, token: 'test-owner'}));
    await expect(deleteSession(busyRoot, 'session-1')).rejects.toMatchObject({code: 'PROJECT_BUSY', statusCode: 409});
    await expect(stat(sessionRoot)).resolves.toBeDefined();
    expect(await readSessionState(busyRoot)).toMatchObject({activeSessionId: 'session-1'});
  });
});
