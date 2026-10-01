import {mkdtemp, mkdir, readFile, rm, stat, utimes, writeFile} from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {afterEach, describe, expect, it} from 'vitest';
import {readFilmProgress, readFilmSnapshot, updateRenderAcceptance} from './server-lib.mjs';

const roots = [];
const SESSION = 'film-review';
const RENDER = 'script2video';

afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) => rm(root, {recursive: true, force: true})));
});

async function fixture() {
  const root = await mkdtemp(path.join(os.tmpdir(), 'vimax-film-'));
  roots.push(root);
  const working = path.join(root, '.working_dir', SESSION);
  const put = async (relative, value) => {
    const file = path.join(working, relative);
    await mkdir(path.dirname(file), {recursive: true});
    await writeFile(file, typeof value === 'string' ? value : JSON.stringify(value));
    return file;
  };
  await mkdir(path.join(root, '.vimax'), {recursive: true});
  await writeFile(path.join(root, '.vimax', 'sessions.json'), JSON.stringify({
    active_session_id: SESSION,
    sessions: {[SESSION]: {session_id: SESSION, project_name: 'Review', working_dir: `.working_dir/${SESSION}`}},
  }));
  await put('render_manifest.json', {render_mode: RENDER, image_model: 'qwen/qwen-image-3'});
  await put(`${RENDER}/characters.json`, [{idx: 0, identifier_in_scene: 'Keeper'}]);
  await put(`${RENDER}/camera_tree.json`, [{idx: 0, active_shot_idxs: [0]}]);
  await put(`${RENDER}/storyboard.json`, [{idx: 0, cam_idx: 0, visual_desc: 'Keeper lights the beacon.', audio_desc: ''}]);
  const planPath = await put(`${RENDER}/shots/0/shot_description.json`, {
    idx: 0, ff_desc: 'Keeper enters.', lf_desc: 'Keeper leaves.', ff_vis_char_idxs: [0], lf_vis_char_idxs: [0],
  });
  await put(`${RENDER}/shots/0/qwen_qwen-image-3/first_frame.png`, 'first-frame');
  const framePath = await put(`${RENDER}/shots/0/qwen_qwen-image-3/last_frame.png`, 'last-frame');
  return {root, working, put, planPath, framePath};
}

describe('film snapshots', () => {
  it('detects external same-size media replacement and plan edits between snapshots', async () => {
    const {root, planPath, framePath} = await fixture();
    await updateRenderAcceptance(root, {sessionId: SESSION, root: RENDER, stage: 'keyframes', shot: '0', accepted: true});
    const first = await readFilmSnapshot(root, SESSION);
    expect(first.acceptance.stages.find((stage) => stage.stage === 'keyframes').slots[0].state).toBe('accepted');
    expect(first.plans[0].firstFrame.description).toBe('Keeper enters.');

    // A writer outside the web server can replace bytes without changing size
    // or mtime. Request-scoped reuse must never become a persistent approval cache.
    const previous = await stat(framePath);
    await writeFile(framePath, 'LAST-frame');
    await utimes(framePath, previous.atime, previous.mtime);
    const plan = JSON.parse(await readFile(planPath, 'utf8'));
    await writeFile(planPath, JSON.stringify({...plan, ff_desc: 'Keeper climbs.'}));

    const second = await readFilmSnapshot(root, SESSION);
    expect(second.acceptance.stages.find((stage) => stage.stage === 'keyframes').slots[0].state).toBe('stale');
    expect(second.plans[0].firstFrame.description).toBe('Keeper climbs.');
    expect(first.plans[0].firstFrame.description).toBe('Keeper enters.');
  });

  it('resolves review media even when unrelated documents fill the asset-browser limit', async () => {
    const {root, put} = await fixture();
    for (let index = 0; index < 410; index++) await put(`aaa-documents/${String(index).padStart(3, '0')}.json`, '{}');
    const snapshot = await readFilmSnapshot(root, SESSION);
    const referenced = snapshot.acceptance.stages.flatMap((stage) => stage.slots.flatMap((slot) => slot.artifacts));
    const byPath = new Map(snapshot.artifacts.map((artifact) => [artifact.path, artifact]));
    const firstFrame = `${RENDER}/shots/0/qwen_qwen-image-3/first_frame.png`;
    expect(referenced).toContain(firstFrame);
    for (const relative of referenced) {
      expect(byPath.get(relative)).toMatchObject({path: relative, kind: 'image'});
      expect(new URL(byPath.get(relative).url, 'http://local').searchParams.get('session')).toBe(SESSION);
    }
  });
});

describe('film progress', () => {
  it('keeps redraw context across a bounded tail and tolerates interrupted status writes', async () => {
    const {root, put} = await fixture();
    const started = {timestamp: '2026-09-28T10:00:00Z', status: 'rendering', phase: 'video', render_mode: RENDER, render_started: true, redone_shots: ['0']};
    const rows = [started, ...Array.from({length: 180}, (_, index) => ({
      timestamp: new Date(Date.parse(started.timestamp) + (index + 1) * 1000).toISOString(),
      status: 'rendering', phase: 'video', render_mode: RENDER, progress: index,
    }))];
    const trail = rows.map((row) => JSON.stringify(row)).join('\n') + '\n';
    await put('render_status.json', '{"status":');
    await put('render_events.jsonl', trail + '{"unfinished":');
    const progress = await readFilmProgress(root, SESSION);
    expect(progress.status).toMatchObject({status: 'rendering', progress: 179});
    const recent = progress.trail.split('\n').map((row) => JSON.parse(row));
    expect(recent.at(-1)).toMatchObject({status: 'rendering', progress: 179});
    expect(recent[0]).toMatchObject({render_started: true, redone_shots: ['0']});
    expect((await readFilmProgress(root, SESSION)).revision).toBe(progress.revision);

    const failed = {...started, timestamp: '2026-09-28T10:04:00Z', status: 'error', error: 'Provider unavailable'};
    await put('render_events.jsonl', trail + JSON.stringify(failed) + '\n');
    const failure = await readFilmProgress(root, SESSION);
    expect(failure.status).toMatchObject({status: 'error', error: 'Provider unavailable'});
    expect(failure.revision).not.toBe(progress.revision);
  });
});
