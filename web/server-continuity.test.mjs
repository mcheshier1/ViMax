/**
 * The script-coverage review, and the shot a suggestion becomes.
 *
 * The review itself is written by the agent's vimax_review_timeline tool; what the server
 * owes is reading it, saying honestly whether it still describes the timeline, and turning a
 * suggested shot into a real one — with the suggestion's own frames and dialogue rather than
 * a copy of whatever shot it was placed after.
 */

import {mkdtemp, mkdir, readFile, rm, writeFile} from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {afterEach, describe, expect, it} from 'vitest';
import {createShot, moveShot, readContinuityReview} from './server-lib.mjs';

const roots = [];
const SESSION_ID = 'session-1';
const ROOT = 'script2video';

afterEach(async () => {
  delete process.env.VIMAX_OPENROUTER_VIDEO_DURATION;
  await Promise.all(roots.splice(0).map((root) => rm(root, {recursive: true, force: true})));
});

async function fixture() {
  const root = await mkdtemp(path.join(os.tmpdir(), 'vimax-continuity-'));
  roots.push(root);
  const working = path.join(root, '.working_dir', SESSION_ID);
  await mkdir(path.join(root, '.vimax'), {recursive: true});
  await writeFile(path.join(root, '.vimax', 'sessions.json'), JSON.stringify({
    active_session_id: SESSION_ID,
    sessions: {[SESSION_ID]: {session_id: SESSION_ID, project_name: 'Film', working_dir: `.working_dir/${SESSION_ID}`}},
  }));
  await mkdir(path.join(working, ROOT, 'shots', '2'), {recursive: true});
  await mkdir(path.join(working, ROOT, 'shots', '4'), {recursive: true});
  await writeFile(path.join(working, 'render_manifest.json'), JSON.stringify({render_mode: ROOT}));
  await writeFile(path.join(working, ROOT, 'characters.json'), JSON.stringify([
    {idx: 0, identifier_in_scene: 'Claude', static_features: 'heavyset', dynamic_features: 'orange shirt'},
    {idx: 1, identifier_in_scene: 'Wife', static_features: 'slim', dynamic_features: 'pink dress'},
    {idx: 2, identifier_in_scene: 'DeepSeek', static_features: 'younger man', dynamic_features: 'whale t-shirt'},
  ]));
  await writeFile(path.join(working, ROOT, 'camera_tree.json'), JSON.stringify([
    {idx: 2, active_shot_idxs: [2, 4], parent_cam_idx: 0, parent_shot_idx: 0},
  ]));
  await writeFile(path.join(working, ROOT, 'storyboard.json'), JSON.stringify([
    {idx: 2, is_last: false, cam_idx: 2, visual_desc: 'the Wife turns to face the couch', audio_desc: ''},
    {idx: 4, is_last: false, cam_idx: 2, visual_desc: 'the Wife looks irritated', audio_desc: ''},
  ]));
  for (const [slot, ff, lf] of [[2, 'The Wife turns.', 'The Wife faces the couch.'], [4, 'The Wife irritated.', 'The Wife looking away.']]) {
    await writeFile(path.join(working, ROOT, 'shots', String(slot), 'shot_description.json'), JSON.stringify({
      idx: slot, ff_desc: ff, lf_desc: lf, ff_vis_char_idxs: [1], lf_vis_char_idxs: [1], motion_desc: 'small', audio_desc: '',
    }, null, 4));
  }
  process.env.VIMAX_OPENROUTER_VIDEO_DURATION = '5';
  return {root, working};
}

const shotDescription = async (working, slot) =>
  JSON.parse(await readFile(path.join(working, ROOT, 'shots', String(slot), 'shot_description.json'), 'utf8'));
const storyboard = async (working) => JSON.parse(await readFile(path.join(working, ROOT, 'storyboard.json'), 'utf8'));
const cameras = async (working) => JSON.parse(await readFile(path.join(working, ROOT, 'camera_tree.json'), 'utf8'));
const writeReview = (working, review) => writeFile(path.join(working, 'continuity_review.json'), JSON.stringify(review));

const SUGGESTION = {
  sessionId: SESSION_ID,
  root: ROOT,
  after: '4',
  brief: 'In the living room, a medium-wide shot shows DeepSeek entering from the right and putting his arm around the Wife.',
  audioDesc: '[Sound Effect] Footsteps entering the room',
  ffDesc: 'DeepSeek entering from the right.',
  lfDesc: 'His arm around the Wife.',
  ffVis: [2, 1],
  lfVis: [1, 2],
};

describe('adding a suggested shot', () => {
  it('writes the suggestion’s own frames and dialogue rather than copying its neighbour', async () => {
    const {root, working} = await fixture();

    const payload = await createShot(root, SUGGESTION);

    expect(payload.created.copiedPlan).toBe(false);
    const created = await shotDescription(working, payload.created.slot);
    expect(created.ff_desc).toBe('DeepSeek entering from the right.');
    expect(created.lf_desc).toBe('His arm around the Wife.');
    expect(created.ff_vis_char_idxs).toEqual([1, 2]);
    // Fields the suggestion said nothing about still come from the reference shot.
    expect(created.motion_desc).toBe('small');
    const entry = (await storyboard(working)).find((row) => String(row.idx) === payload.created.slot);
    expect(entry.audio_desc).toBe('[Sound Effect] Footsteps entering the room');
    expect(entry.visual_desc).toBe(SUGGESTION.brief);
    // A camera's list is the order its shots play in, so it plays where it was put.
    const camera = (await cameras(working))[0];
    expect(camera.active_shot_idxs.map(String)).toEqual(['2', '4', payload.created.slot]);
  });

  it('still copies the neighbour when nothing described the new frames', async () => {
    const {root, working} = await fixture();

    const payload = await createShot(root, {sessionId: SESSION_ID, root: ROOT, after: '4', brief: 'another close-up of the Wife'});

    expect(payload.created.copiedPlan).toBe(true);
    const created = await shotDescription(working, payload.created.slot);
    expect(created.ff_desc).toBe('The Wife irritated.');
    expect(created.idx).toBe(Number(payload.created.slot));
  });

  it('refuses a character the root does not have', async () => {
    const {root} = await fixture();

    const refused = await createShot(root, {...SUGGESTION, ffVis: [7]}).catch((error) => error);

    expect(refused.message).toMatch(/Unknown character index: 7/);
  });
});

describe('reading a coverage review', () => {
  it('reports a timeline nobody has reviewed as out of date', async () => {
    const {root} = await fixture();

    const payload = await readContinuityReview(root, SESSION_ID, ROOT);

    expect(payload.review).toBeNull();
    expect(payload.stale).toBe(true);
    expect(payload.staleReason).toMatch(/No review yet/);
    // The shots come from disk, in the camera's playing order.
    expect(payload.shots).toEqual(['2', '4']);
    expect(payload.clipSeconds).toBe(5);
  });

  it('reports the shots in playing order, not in the order of their numbers', async () => {
    const {root, working} = await fixture();
    // a shot's number is its identity, not its place: this film plays 4 before 2
    await writeFile(path.join(working, ROOT, 'camera_tree.json'), JSON.stringify([
      {idx: 2, active_shot_idxs: [4, 2]},
      {idx: 3, active_shot_idxs: [5]},
    ]));

    const payload = await readContinuityReview(root, SESSION_ID, ROOT);

    expect(payload.shots).toEqual(['4', '2', '5']);
  });

  it('accepts a review that covers the shots the film has', async () => {
    const {root, working} = await fixture();
    await writeReview(working, {reviewed_at: new Date().toISOString(), shots_reviewed: [2, 4], beats: [], suggestions: []});

    const payload = await readContinuityReview(root, SESSION_ID, ROOT);

    expect(payload.stale).toBe(false);
    expect(payload.staleReason).toBe('');
  });

  it('reports a review as out of date once the timeline moves under it', async () => {
    const {root, working} = await fixture();
    await writeReview(working, {reviewed_at: new Date().toISOString(), shots_reviewed: [2, 4], beats: [], suggestions: []});
    await createShot(root, {sessionId: SESSION_ID, root: ROOT, after: '4', brief: 'a new closing shot'});

    const payload = await readContinuityReview(root, SESSION_ID, ROOT);

    expect(payload.stale).toBe(true);
    expect(payload.staleReason).toMatch(/shots \d+ added/);
  });

  it('reports a review as out of date once a plan is edited after it', async () => {
    const {root, working} = await fixture();
    await writeReview(working, {reviewed_at: '2026-09-22T09:00:00', shots_reviewed: [2, 4], beats: [], suggestions: []});
    await writeFile(
      path.join(working, ROOT, 'shots', '4', 'shot_description.json'),
      JSON.stringify({idx: 4, ff_desc: 'edited by hand', lf_desc: '', ff_vis_char_idxs: [1], lf_vis_char_idxs: [1], motion_desc: 'small', audio_desc: ''}),
    );

    const payload = await readContinuityReview(root, SESSION_ID, ROOT);

    expect(payload.stale).toBe(true);
    expect(payload.staleReason).toMatch(/Edited since this review/);
    expect(payload.staleReason).toMatch(/shots\/4\/shot_description.json/);
  });
});

describe('moving a shot in the film', () => {
  // The film plays camera by camera, so a shot moved to a point another camera owns changes
  // which still its frames are drawn from: the plan and the brief have to move with it.
  // Camera 3 moves away from shot 4, which is the transition a move must not break.
  async function moveFixture() {
    const {root, working} = await fixture();
    await writeFile(path.join(working, ROOT, 'camera_tree.json'), JSON.stringify([
      {idx: 2, active_shot_idxs: [2, 4], parent_cam_idx: 0, parent_shot_idx: 0},
      {idx: 3, active_shot_idxs: [5], parent_cam_idx: 2, parent_shot_idx: 4},
    ]));
    await mkdir(path.join(working, ROOT, 'shots', '5'), {recursive: true});
    await writeFile(path.join(working, ROOT, 'shots', '5', 'shot_description.json'), JSON.stringify({
      idx: 5, ff_desc: 'The walk-off.', lf_desc: 'Out of frame.', ff_vis_char_idxs: [1], lf_vis_char_idxs: [1], motion_desc: 'they leave', audio_desc: '',
    }, null, 4));
    await writeFile(path.join(working, ROOT, 'storyboard.json'), JSON.stringify([
      {idx: 2, is_last: false, cam_idx: 2, visual_desc: 'the Wife turns', audio_desc: ''},
      {idx: 4, is_last: false, cam_idx: 2, visual_desc: 'the Wife looks irritated', audio_desc: ''},
      {idx: 5, is_last: true, cam_idx: 3, visual_desc: 'the walk-off', audio_desc: ''},
    ]));
    return {root, working};
  }

  const cameraLists = async (working) => {
    const cameras = JSON.parse(await readFile(path.join(working, ROOT, 'camera_tree.json'), 'utf8'));
    return Object.fromEntries(cameras.map((camera) => [camera.idx, (camera.active_shot_idxs || []).map(String)]));
  };

  it('reorders within a camera without changing which camera the shot plays in', async () => {
    const {root, working} = await moveFixture();

    const payload = await moveShot(root, {sessionId: SESSION_ID, root: ROOT, slot: '2', after: '4'});

    expect(payload.moved).toMatchObject({slot: '2', after: '4', camera: 2});
    expect(await cameraLists(working)).toEqual({2: ['4', '2'], 3: ['5']});
  });

  it('moves a shot into the camera that owns the point it is moved to', async () => {
    const {root, working} = await moveFixture();

    const payload = await moveShot(root, {sessionId: SESSION_ID, root: ROOT, slot: '2', after: '5'});

    expect(payload.moved).toMatchObject({slot: '2', after: '5', camera: 3});
    expect(await cameraLists(working)).toEqual({2: ['4'], 3: ['5', '2']});
    // Both the plan and the brief name the camera the shot now plays in.
    expect((await shotDescription(working, 2)).cam_idx).toBe(3);
    expect((await storyboard(working)).find((row) => String(row.idx) === '2').cam_idx).toBe(3);
  });

  it('refuses a move that would leave a camera with nothing to play', async () => {
    const {root} = await moveFixture();

    // camera 3 holds only shot 5
    const refused = await moveShot(root, {sessionId: SESSION_ID, root: ROOT, slot: '5', after: '2'}).catch((error) => error);

    expect(refused.message).toMatch(/only shot of camera 3/);
  });

  it('refuses a move that would leave a camera without its transition', async () => {
    const {root} = await moveFixture();

    // camera 3 moves away from shot 4, so shot 4 cannot leave camera 2
    const refused = await moveShot(root, {sessionId: SESSION_ID, root: ROOT, slot: '4', after: '5'}).catch((error) => error);

    expect(refused.message).toMatch(/camera 3 moves away from/);
  });

  it('refuses a shot following itself, or following a shot that is not in the film', async () => {
    const {root} = await moveFixture();

    const itself = await moveShot(root, {sessionId: SESSION_ID, root: ROOT, slot: '2', after: '2'}).catch((error) => error);
    const missing = await moveShot(root, {sessionId: SESSION_ID, root: ROOT, slot: '2', after: '99'}).catch((error) => error);

    expect(itself.message).toMatch(/cannot follow itself/);
    expect(missing.message).toMatch(/is not part of the film/);
  });

  it('steps a shot a place at a time using the film\'s own order', async () => {
    const {root, working} = await moveFixture();

    const payload = await moveShot(root, {sessionId: SESSION_ID, root: ROOT, slot: '2', direction: 'later'});

    // the film plays 2 then 4, so stepping 2 later puts it after 4 within the same camera
    expect(payload.moved).toMatchObject({slot: '2', after: '4', camera: 2});
    expect(await cameraLists(working)).toEqual({2: ['4', '2'], 3: ['5']});
  });

  it('refuses to step a shot past either end of the film', async () => {
    const {root} = await moveFixture();

    const first = await moveShot(root, {sessionId: SESSION_ID, root: ROOT, slot: '2', direction: 'earlier'}).catch((error) => error);
    const last = await moveShot(root, {sessionId: SESSION_ID, root: ROOT, slot: '5', direction: 'later'}).catch((error) => error);

    expect(first.message).toMatch(/already the film's first shot/);
    expect(last.message).toMatch(/already the film's last shot/);
  });

  it('refuses a direction it does not know', async () => {
    const {root} = await moveFixture();

    const refused = await moveShot(root, {sessionId: SESSION_ID, root: ROOT, slot: '2', direction: 'sideways'}).catch((error) => error);

    expect(refused.message).toMatch(/Unknown direction: sideways/);
  });
});
