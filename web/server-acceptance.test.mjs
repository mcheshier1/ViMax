import {createHash} from 'node:crypto';
import {mkdtemp, mkdir, readFile, rm, writeFile} from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {afterEach, describe, expect, it} from 'vitest';
import {readRenderAcceptance, updateRenderAcceptance} from './server-lib.mjs';

const roots = [];
const SESSION_ID = 'session-1';
const ROOT = 'script2video';

afterEach(async () => {
  delete process.env.VIMAX_OPENROUTER_VIDEO_DURATION;
  await Promise.all(roots.splice(0).map((root) => rm(root, {recursive: true, force: true})));
});

async function fixture() {
  const root = await mkdtemp(path.join(os.tmpdir(), 'vimax-acceptance-'));
  roots.push(root);
  const working = path.join(root, '.working_dir', SESSION_ID);
  await mkdir(path.join(working), {recursive: true});
  await mkdir(path.join(root, '.vimax'), {recursive: true});
  await writeFile(path.join(root, '.vimax', 'sessions.json'), JSON.stringify({
    active_session_id: SESSION_ID,
    sessions: {
      [SESSION_ID]: {
        session_id: SESSION_ID,
        project_name: 'Ocean campaign',
        working_dir: `.working_dir/${SESSION_ID}`,
        stage: 'stills_ready',
        updated_at: '2026-07-17T10:00:00',
      },
    },
  }));
  await writeFile(path.join(working, 'render_manifest.json'), JSON.stringify({
    image_model: 'qwen/qwen-image-3',
    video_model: 'kwaivgi/kling-video-o1',
    style: 'cinematic',
    render_mode: ROOT,
  }, null, 2));
  return {root, working};
}

async function put(working, relativePath, contents = 'x') {
  const absolute = path.join(working, relativePath);
  await mkdir(path.dirname(absolute), {recursive: true});
  await writeFile(absolute, contents);
}

async function planShot(working, index, {first = false, last = false, clip = false} = {}) {
  await put(working, `${ROOT}/shots/${index}/shot_description.json`, '{}');
  if (first) await put(working, `${ROOT}/shots/${index}/qwen_qwen-image-3/first_frame.png`, 'frame-first');
  if (last) await put(working, `${ROOT}/shots/${index}/qwen_qwen-image-3/last_frame.png`, 'frame-last');
  if (clip) await put(working, `${ROOT}/shots/${index}/kwaivgi_kling-video-o1/video.mp4`, 'clip');
}

const stage = (payload, name) => payload.stages.find((entry) => entry.stage === name);
const accept = (root, input) => updateRenderAcceptance(root, {sessionId: SESSION_ID, root: ROOT, ...input});
const storedAcceptance = async (working) => JSON.parse(await readFile(path.join(working, 'render_acceptance.json'), 'utf8'));

describe('render acceptance', () => {
  it('derives planned, rendered, accepted and stale per slot from disk', async () => {
    const {root, working} = await fixture();
    await planShot(working, 0);
    await planShot(working, 1, {first: true});
    await planShot(working, 2, {first: true, last: true});
    await planShot(working, 3, {first: true, last: true});

    await accept(root, {stage: 'keyframes', shot: '2', accepted: true});
    await accept(root, {stage: 'keyframes', shot: '3', accepted: true});
    // Same size, different bytes: only the sha256 can tell the artifact changed.
    await put(working, `${ROOT}/shots/3/qwen_qwen-image-3/last_frame.png`, 'frame-LAST');

    const keyframes = stage(await readRenderAcceptance(root, SESSION_ID, ROOT), 'keyframes');
    expect(keyframes.slots.map((slot) => [slot.shot, slot.state])).toEqual([
      ['0', 'planned'],
      ['1', 'rendered'],
      ['2', 'accepted'],
      ['3', 'stale'],
    ]);
    expect(keyframes).toMatchObject({scope: 'shot', accepted: false, state: 'stale'});
    expect(keyframes.slots[2].artifacts).toEqual([
      `${ROOT}/shots/2/qwen_qwen-image-3/first_frame.png`,
      `${ROOT}/shots/2/qwen_qwen-image-3/last_frame.png`,
    ]);
  });

  it('counts only the clip the pinned video model produced', async () => {
    const {root, working} = await fixture();
    await planShot(working, 0, {first: true, last: true, clip: true});
    await put(working, `${ROOT}/shots/0/kwaivgi_kling-v3.0-std/video.mp4`, 'clip for the new model');
    await writeFile(path.join(working, 'render_manifest.json'), JSON.stringify({
      image_model: 'qwen/qwen-image-3',
      video_model: 'kwaivgi/kling-v3.0-std',
      style: 'cinematic',
      render_mode: ROOT,
    }, null, 2));

    // A model change leaves the previous model's clip where it was, and both were counted
    // against the slot — while the film only ever holds the pinned model's.
    const clips = stage(await readRenderAcceptance(root, SESSION_ID, ROOT), 'clips');
    expect(clips.slots[0].artifacts).toEqual([`${ROOT}/shots/0/kwaivgi_kling-v3.0-std/video.mp4`]);
  });

  it('still shows the clip on disk when the pinned model has drawn none yet', async () => {
    const {root, working} = await fixture();
    await planShot(working, 0, {first: true, last: true, clip: true});
    await writeFile(path.join(working, 'render_manifest.json'), JSON.stringify({
      image_model: 'qwen/qwen-image-3',
      video_model: 'kwaivgi/kling-v3.0-std',
      style: 'cinematic',
      render_mode: ROOT,
    }, null, 2));

    const clips = stage(await readRenderAcceptance(root, SESSION_ID, ROOT), 'clips');
    expect(clips.slots[0].artifacts).toEqual([`${ROOT}/shots/0/kwaivgi_kling-video-o1/video.mp4`]);
  });

  it('records the file size and sha256 from disk when a slot is accepted', async () => {
    const {root, working} = await fixture();
    await planShot(working, 0, {first: true, last: true});

    const payload = await accept(root, {stage: 'keyframes', shot: '0', accepted: true});
    expect(stage(payload, 'keyframes').slots[0].state).toBe('accepted');
    expect(payload.totals).toEqual({acceptedKeyframes: 2, keyframes: 2, clips: 0, rejected: 0, clipSeconds: 8, clipCostUsd: 0.112});

    const digest = (value) => createHash('sha256').update(value).digest('hex');
    expect((await storedAcceptance(working)).script2video.shots['0'].keyframes.artifacts).toEqual([
      {path: `${ROOT}/shots/0/qwen_qwen-image-3/first_frame.png`, size: 11, sha256: digest('frame-first')},
      {path: `${ROOT}/shots/0/qwen_qwen-image-3/last_frame.png`, size: 10, sha256: digest('frame-last')},
    ]);
  });

  it('reads a redrawn slot as unreviewed, with the note that prompted the redraw', async () => {
    const {root, working} = await fixture();
    await planShot(working, 0, {first: true, last: true});
    // The shape the render writes when it redraws a rejected shot: the rejection is
    // replaced by the note under `redone_at`, and the new artifacts were never accepted.
    await writeFile(path.join(working, 'render_acceptance.json'), JSON.stringify({
      [ROOT]: {shots: {'0': {keyframes: {redone_at: '2026-07-17T11:00:00', reason: 'the backdrop is a studio white'}}}},
    }));

    const slot = stage(await readRenderAcceptance(root, SESSION_ID, ROOT), 'keyframes').slots[0];
    expect(slot.state).toBe('rendered');
    expect(slot.note).toBe('the backdrop is a studio white');
    expect(slot.redoneAt).toBe('2026-07-17T11:00:00');
    expect(slot).not.toHaveProperty('reason');
    expect(slot).not.toHaveProperty('rejectedAt');
  });

  it('numbers idea-mode shots by scene rather than as text', async () => {
    const {root, working} = await fixture();
    await writeFile(path.join(working, 'render_manifest.json'), JSON.stringify({render_mode: 'idea2video'}));
    for (const scene of [10, 2, 1]) {
      await put(working, `idea2video/scene_${scene}/shots/0/shot_description.json`, '{}');
      await put(working, `idea2video/scene_${scene}/shots/0/qwen_qwen-image-3/first_frame.png`, 'frame');
    }

    const payload = await readRenderAcceptance(root, SESSION_ID, 'idea2video');
    expect(stage(payload, 'keyframes').slots.map((slot) => slot.shot)).toEqual(['scene_1/0', 'scene_2/0', 'scene_10/0']);
    expect(payload.root).toBe('idea2video');
  });

  it('deletes the entry when a lock is released', async () => {
    const {root, working} = await fixture();
    await planShot(working, 0, {first: true, last: true});
    await accept(root, {stage: 'keyframes', shot: '0', accepted: true});
    expect(stage(await readRenderAcceptance(root, SESSION_ID, ROOT), 'keyframes').slots[0].state).toBe('accepted');

    await accept(root, {stage: 'keyframes', shot: '0', accepted: false});
    expect(await storedAcceptance(working)).toEqual({});
    expect(stage(await readRenderAcceptance(root, SESSION_ID, ROOT), 'keyframes').slots[0].state).toBe('rendered');
  });

  it('rejects an unknown stage, an unknown shot, a non-boolean and a slot with no artifacts', async () => {
    const {root, working} = await fixture();
    await planShot(working, 0);
    await planShot(working, 1, {first: true});
    const rejection = (input) => updateRenderAcceptance(root, {sessionId: SESSION_ID, root: ROOT, ...input}).catch((error) => error);

    expect((await rejection({stage: 'portraits', accepted: 'yes'})).statusCode).toBe(400);
    expect((await rejection({stage: 'storyboard', accepted: true})).statusCode).toBe(400);
    expect((await rejection({stage: 'keyframes', shot: '9', accepted: true})).statusCode).toBe(400);
    expect((await rejection({root: 'slide2video', stage: 'final_video', accepted: true})).statusCode).toBe(400);
    const empty = await rejection({stage: 'keyframes', shot: '0', accepted: true});
    expect(empty.statusCode).toBe(400);
    expect(empty.message).toMatch(/no artifacts on disk/);

    await expect(updateRenderAcceptance(root, {sessionId: 'unknown-session', stage: 'portraits', accepted: true})).resolves.toBeNull();
    await expect(readRenderAcceptance(root, 'unknown-session')).resolves.toBeNull();
  });

  it('never counts scratch cache directories', async () => {
    const {root, working} = await fixture();
    await put(working, `${ROOT}/shots/0/shot_description.json`, '{}');
    await put(working, `${ROOT}/shots/0/qwen_qwen-image-3/cache/first_frame.png`, 'debris');
    await put(working, `${ROOT}/shots/0/kwaivgi_kling-video-o1/cache/video.mp4`, 'debris');

    const payload = await readRenderAcceptance(root, SESSION_ID, ROOT);
    expect(stage(payload, 'keyframes').slots[0]).toMatchObject({state: 'planned', artifacts: []});
    expect(stage(payload, 'clips').slots[0]).toMatchObject({state: 'planned', artifacts: []});
    expect(payload.totals).toMatchObject({keyframes: 0, clips: 0});
  });

  it('keeps each render root scoped to its own acceptance', async () => {
    const {root, working} = await fixture();
    await put(working, `${ROOT}/final_video.mp4`, 'film');
    await put(working, 'idea2video/final_video.mp4', 'film');

    await accept(root, {stage: 'final_video', accepted: true});
    expect(stage(await readRenderAcceptance(root, SESSION_ID, ROOT), 'final_video')).toMatchObject({scope: 'session', state: 'accepted', accepted: true});
    expect(stage(await readRenderAcceptance(root, SESSION_ID, 'idea2video'), 'final_video').slots[0].state).toBe('rendered');
    expect(Object.keys(await storedAcceptance(working))).toEqual([ROOT]);
  });

  it('accepts a session-wide portraits slot', async () => {
    const {root, working} = await fixture();
    for (const view of ['front', 'side', 'back']) {
      await put(working, `${ROOT}/character_portraits/qwen_qwen-image-3/0_Claude/${view}.png`, view);
    }
    await put(working, `${ROOT}/character_portraits/qwen_qwen-image-3/registry.json`, '{}');

    const portraits = stage(await accept(root, {stage: 'portraits', accepted: true}), 'portraits');
    expect(portraits).toMatchObject({scope: 'session', state: 'accepted', accepted: true});
    expect(portraits.slots).toEqual([{
      shot: null,
      state: 'accepted',
      artifacts: [
        `${ROOT}/character_portraits/qwen_qwen-image-3/0_Claude/back.png`,
        `${ROOT}/character_portraits/qwen_qwen-image-3/0_Claude/front.png`,
        `${ROOT}/character_portraits/qwen_qwen-image-3/0_Claude/side.png`,
      ],
    }]);
  });

  it('prices the clips it finds and honours the configured clip duration', async () => {
    const {root, working} = await fixture();
    await planShot(working, 0, {clip: true});
    await accept(root, {stage: 'clips', shot: '0', accepted: true});

    expect((await readRenderAcceptance(root, SESSION_ID, ROOT)).totals)
      .toEqual({acceptedKeyframes: 0, keyframes: 0, clips: 1, rejected: 0, clipSeconds: 8, clipCostUsd: 0.112});

    process.env.VIMAX_OPENROUTER_VIDEO_DURATION = '5';
    expect((await readRenderAcceptance(root, SESSION_ID, ROOT)).totals).toMatchObject({clips: 1, clipSeconds: 5, clipCostUsd: 0.112});
  });

  it('prices a clip at the advertised rate of the model that renders it', async () => {
    const {root, working} = await fixture();
    await planShot(working, 0, {clip: true});
    // The fixture's manifest records the silent kling-video-o1, at $0.112/second.
    expect((await readRenderAcceptance(root, SESSION_ID, ROOT)).totals.clipCostUsd).toBe(0.112);

    await put(working, 'render_manifest.json', JSON.stringify({video_model: 'kwaivgi/kling-v3.0-std', render_mode: ROOT}));
    // kling-v3.0-std bills $0.126/second with its native audio, which is what an 8 second
    // clip of dialogue costs — and what the sequence is switched to.
    expect((await readRenderAcceptance(root, SESSION_ID, ROOT)).totals.clipCostUsd).toBe(0.126);

    await put(working, 'render_manifest.json', JSON.stringify({video_model: 'someone/unpriced-model', render_mode: ROOT}));
    // An unpriced model is estimated at the highest known rate rather than at nothing: an
    // estimate that is wrong upward is a surprise, one that is wrong downward is a bill.
    expect((await readRenderAcceptance(root, SESSION_ID, ROOT)).totals.clipCostUsd).toBe(0.168);
  });

  it('defaults the root to the manifest render mode', async () => {
    const {root, working} = await fixture();
    await put(working, `${ROOT}/final_video.mp4`, 'film');
    expect((await readRenderAcceptance(root, SESSION_ID)).root).toBe(ROOT);
    expect(stage(await readRenderAcceptance(root, SESSION_ID), 'final_video').slots[0].state).toBe('rendered');

    await rm(path.join(working, 'render_manifest.json'));
    expect((await readRenderAcceptance(root, SESSION_ID)).root).toBe(ROOT);
  });

  it('records a rejection that clears the acceptance on the same slot', async () => {
    const {root, working} = await fixture();
    await planShot(working, 0, {first: true, last: true});
    await accept(root, {stage: 'keyframes', shot: '0', accepted: true});
    // Same artifact set, different bytes: the lock is now stale, but the rejection outranks it.
    await put(working, `${ROOT}/shots/0/qwen_qwen-image-3/first_frame.png`, 'FRAME-first');

    const payload = await accept(root, {stage: 'keyframes', shot: '0', accepted: false, reason: 'her dress is blue here but pink in every other shot'});
    const keyframes = stage(payload, 'keyframes');
    expect(keyframes).toMatchObject({state: 'rejected', accepted: false});
    expect(keyframes.slots[0]).toMatchObject({
      shot: '0',
      state: 'rejected',
      reason: 'her dress is blue here but pink in every other shot',
    });
    expect(typeof keyframes.slots[0].rejectedAt).toBe('string');
    expect(keyframes.slots[0]).not.toHaveProperty('accepted_at');

    const stored = (await storedAcceptance(working)).script2video.shots['0'].keyframes;
    expect(stored).toEqual({rejected_at: expect.any(String), reason: 'her dress is blue here but pink in every other shot'});
    expect(stored).not.toHaveProperty('accepted_at');
  });

  it('keeps a rejection reason across a round trip through the file', async () => {
    const {root, working} = await fixture();
    await planShot(working, 0, {first: true, last: true});
    const reason = 'she is holding the cup in her left hand, not her right';
    const written = stage(await accept(root, {stage: 'keyframes', shot: '0', accepted: false, reason}), 'keyframes').slots[0];

    const reread = stage(await readRenderAcceptance(root, SESSION_ID, ROOT), 'keyframes').slots[0];
    expect(reread).toEqual(written);
    expect(reread).toMatchObject({state: 'rejected', reason, rejectedAt: written.rejectedAt});
  });

  it('refuses a rejection reason longer than the limit', async () => {
    const {root, working} = await fixture();
    await planShot(working, 0, {first: true});
    const refuse = (reason) => accept(root, {stage: 'keyframes', shot: '0', accepted: false, reason}).catch((error) => error);

    const tooLong = await refuse('x'.repeat(2001));
    expect(tooLong.statusCode).toBe(400);
    expect(tooLong.message).toMatch(/2000/);
    expect(await storedAcceptance(working).catch(() => ({}))).toEqual({});

    // Whitespace-only is empty after trimming, so it deletes instead of rejecting.
    await accept(root, {stage: 'keyframes', shot: '0', accepted: false, reason: '   '});
    expect(stage(await readRenderAcceptance(root, SESSION_ID, ROOT), 'keyframes').slots[0].state).toBe('rendered');
    const boundary = stage(await accept(root, {stage: 'keyframes', shot: '0', accepted: false, reason: 'x'.repeat(2000)}), 'keyframes').slots[0];
    expect(boundary.reason).toHaveLength(2000);
  });

  it('counts rejected slots in the totals across the payload', async () => {
    const {root, working} = await fixture();
    await planShot(working, 0, {first: true});
    await planShot(working, 1, {first: true});
    await accept(root, {stage: 'keyframes', shot: '0', accepted: true});
    await accept(root, {stage: 'keyframes', shot: '1', accepted: false, reason: 'wrong face'});

    const payload = await readRenderAcceptance(root, SESSION_ID, ROOT);
    expect(stage(payload, 'keyframes')).toMatchObject({state: 'rejected', accepted: false});
    expect(payload.totals.rejected).toBe(1);

    await accept(root, {stage: 'keyframes', shot: '0', accepted: false, reason: 'also wrong'});
    expect((await readRenderAcceptance(root, SESSION_ID, ROOT)).totals.rejected).toBe(2);
  });

  it('reads rejected even when the record still carries a valid acceptance', async () => {
    const {root, working} = await fixture();
    await planShot(working, 0, {first: true, last: true});
    await accept(root, {stage: 'keyframes', shot: '0', accepted: true});
    // A hand-edited or migrated file could hold both; the rejection must outrank the lock.
    const store = await storedAcceptance(working);
    Object.assign(store.script2video.shots['0'].keyframes, {rejected_at: '2026-09-20T00:00:00', reason: 'both present'});
    await writeFile(path.join(working, 'render_acceptance.json'), `${JSON.stringify(store, null, 2)}\n`);

    const slot = stage(await readRenderAcceptance(root, SESSION_ID, ROOT), 'keyframes').slots[0];
    expect(slot).toMatchObject({state: 'rejected', reason: 'both present', rejectedAt: '2026-09-20T00:00:00'});
  });

  it('clears a rejection when the slot is released with no reason', async () => {
    const {root, working} = await fixture();
    await planShot(working, 0, {first: true});
    await accept(root, {stage: 'keyframes', shot: '0', accepted: false, reason: 'not this one'});
    expect(stage(await readRenderAcceptance(root, SESSION_ID, ROOT), 'keyframes').slots[0].state).toBe('rejected');

    await accept(root, {stage: 'keyframes', shot: '0', accepted: false});
    expect(await storedAcceptance(working)).toEqual({});
    expect(stage(await readRenderAcceptance(root, SESSION_ID, ROOT), 'keyframes').slots[0]).toMatchObject({state: 'rendered'});
    expect(stage(await readRenderAcceptance(root, SESSION_ID, ROOT), 'keyframes').slots[0]).not.toHaveProperty('reason');
  });
});