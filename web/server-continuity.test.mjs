/**
 * The script-coverage review, and the shot a suggestion becomes.
 *
 * The review itself is written by the agent's vimax_review_timeline tool; what the server
 * owes is reading it, saying honestly whether it still describes the timeline, and turning a
 * suggested shot into a real one — with the suggestion's own frames and dialogue rather than
 * a copy of whatever shot it was placed after.
 */

import {createHash} from 'node:crypto';
import {mkdtemp, mkdir, readFile, rm, writeFile} from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {afterEach, describe, expect, it} from 'vitest';
import {createShot, moveShot, readContinuityReview, readRemovedShots, readRenderAcceptance, readShotPlan, readShotPlans, removeShot, restoreShot, updateRenderAcceptance, updateShotPlan} from './server-lib.mjs';

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
    sessions: {[SESSION_ID]: {session_id: SESSION_ID, project_name: 'Film', working_dir: `.working_dir/${SESSION_ID}`, user_requirement: ''}},
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

async function ideaFixture() {
  const root = await mkdtemp(path.join(os.tmpdir(), 'vimax-idea-timeline-'));
  const working = path.join(root, '.working_dir', SESSION_ID);
  await mkdir(working, {recursive: true});
  await mkdir(path.join(root, '.vimax'), {recursive: true});
  await writeFile(path.join(root, '.vimax', 'sessions.json'), JSON.stringify({
    active_session_id: SESSION_ID,
    sessions: {[SESSION_ID]: {session_id: SESSION_ID, working_dir: `.working_dir/${SESSION_ID}`, user_requirement: ''}},
  }));
  await writeFile(path.join(working, 'render_manifest.json'), JSON.stringify({render_mode: 'idea2video'}));
  await mkdir(path.join(working, 'idea2video'), {recursive: true});
  await writeFile(path.join(working, 'idea2video', 'characters.json'), JSON.stringify([{idx: 0, identifier_in_scene: 'Global character'}]));
  for (const [scene, slots, character] of [
    ['scene_0', [4, 2], {idx: 0, identifier_in_scene: 'Scene Zero'}],
    ['scene_1', [2], {idx: 7, identifier_in_scene: 'Scene One'}],
  ]) {
    const directory = path.join(working, 'idea2video', scene);
    await mkdir(path.join(directory, 'shots'), {recursive: true});
    await writeFile(path.join(directory, 'characters.json'), JSON.stringify([character]));
    await writeFile(path.join(directory, 'camera_tree.json'), JSON.stringify([{idx: 0, active_shot_idxs: slots}]));
    await writeFile(path.join(directory, 'storyboard.json'), JSON.stringify(
      slots.map((idx) => ({idx, is_last: false, cam_idx: 0, visual_desc: `${scene} shot ${idx}`, audio_desc: ''})),
    ));
    for (const idx of slots) {
      await mkdir(path.join(directory, 'shots', String(idx)), {recursive: true});
      await writeFile(path.join(directory, 'shots', String(idx), 'shot_description.json'), JSON.stringify({
        idx, ff_desc: `${scene} first ${idx}`, lf_desc: `${scene} last ${idx}`,
        ff_vis_char_idxs: [character.idx], lf_vis_char_idxs: [character.idx], motion_desc: 'move', audio_desc: '',
      }));
    }
  }
  process.env.VIMAX_OPENROUTER_VIDEO_DURATION = '5';
  return {root, working};
}

async function fingerprints(working, paths) {
  const result = {};
  for (const relative of paths) {
    try {
      result[relative] = createHash('sha256').update(await readFile(path.join(working, relative))).digest('hex');
    } catch (error) {
      if (error.code !== 'ENOENT') throw error;
      result[relative] = null;
    }
  }
  return result;
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
  it('serializes simultaneous adds so both distinct briefs survive', async () => {
    const {root, working} = await fixture();

    const results = await Promise.all([
      createShot(root, {sessionId: SESSION_ID, root: ROOT, after: '4', brief: 'first simultaneous addition'}),
      createShot(root, {sessionId: SESSION_ID, root: ROOT, after: '4', brief: 'second simultaneous addition'}),
    ]);

    const slots = results.map((result) => result.created.slot);
    expect(new Set(slots).size).toBe(2);
    const tree = await cameras(working);
    expect(tree[0].active_shot_idxs.map(String)).toHaveLength(4);
    expect((await storyboard(working)).filter((row) => slots.includes(String(row.idx))).map((row) => row.visual_desc).sort()).toEqual([
      'first simultaneous addition', 'second simultaneous addition',
    ]);
    for (const slot of slots) expect((await shotDescription(working, slot)).idx).toBe(Number(slot));
  });

  it('invalidates accepted assemblies on create and reorder and keeps endings canonical', async () => {
    const {root, working} = await fixture();
    const finalFilm = path.join(working, ROOT, 'final_video.mp4');
    await writeFile(finalFilm, 'old film');
    await updateRenderAcceptance(root, {sessionId: SESSION_ID, root: ROOT, stage: 'final_video', accepted: true});

    const created = await createShot(root, {sessionId: SESSION_ID, root: ROOT, after: '4', brief: 'new ending'});

    expect(await readFile(finalFilm).catch(() => null)).toBeNull();
    let acceptance = await readRenderAcceptance(root, SESSION_ID, ROOT);
    expect(acceptance.stages.find((stage) => stage.stage === 'final_video').slots[0].state).toBe('planned');
    const createdPlan = await shotDescription(working, created.created.slot);
    expect(createdPlan.is_last).toBe(true);
    expect((await storyboard(working)).filter((row) => row.is_last).map((row) => String(row.idx))).toEqual([created.created.slot]);

    await writeFile(finalFilm, 'old film after add');
    await updateRenderAcceptance(root, {sessionId: SESSION_ID, root: ROOT, stage: 'final_video', accepted: true});
    await moveShot(root, {sessionId: SESSION_ID, root: ROOT, slot: '2', after: created.created.slot});

    expect(await readFile(finalFilm).catch(() => null)).toBeNull();
    acceptance = await readRenderAcceptance(root, SESSION_ID, ROOT);
    expect(acceptance.stages.find((stage) => stage.stage === 'final_video').slots[0].state).toBe('planned');
    const endingDescriptions = await Promise.all(['2', '4', created.created.slot].map((slot) => shotDescription(working, slot)));
    expect(endingDescriptions.filter((plan) => plan.is_last).map((plan) => String(plan.idx))).toEqual(['2']);
    expect((await storyboard(working)).filter((row) => row.is_last).map((row) => String(row.idx))).toEqual(['2']);
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
  it('stales reviews when the root, source script, or captured requirement changes', async () => {
    const {root, working} = await fixture();
    const scriptPath = path.join(working, ROOT, 'script.txt');
    await writeFile(scriptPath, 'source script');
    const inputs = await fingerprints(working, [
      `${ROOT}/script.txt`,
      `${ROOT}/characters.json`,
      `${ROOT}/camera_tree.json`,
      `${ROOT}/storyboard.json`,
      `${ROOT}/shots/2/shot_description.json`,
      `${ROOT}/shots/4/shot_description.json`,
    ]);
    const review = {
      root: ROOT,
      user_requirement: '',
      input_files: inputs,
      reviewed_at: new Date().toISOString(),
      shots_reviewed: ['2', '4'],
      beats: [],
      suggestions: [],
    };
    await writeReview(working, review);
    expect((await readContinuityReview(root, SESSION_ID, ROOT)).stale).toBe(false);

    await writeFile(scriptPath, 'changed source script');
    const scriptChange = await readContinuityReview(root, SESSION_ID, ROOT);
    expect(scriptChange.stale).toBe(true);
    expect(scriptChange.staleReason).toMatch(/script2video\/script.txt/);

    await writeFile(scriptPath, 'source script');
    const sessionsPath = path.join(root, '.vimax', 'sessions.json');
    const sessions = JSON.parse(await readFile(sessionsPath, 'utf8'));
    sessions.sessions[SESSION_ID].user_requirement = 'keep the ending quiet';
    await writeFile(sessionsPath, JSON.stringify(sessions));
    const requirementChange = await readContinuityReview(root, SESSION_ID, ROOT);
    expect(requirementChange.stale).toBe(true);
    expect(requirementChange.staleReason).toMatch(/user requirement changed/);
  });

  it('rejects a fingerprinted review belonging to another render root', async () => {
    const {root, working} = await fixture();
    await writeReview(working, {
      root: 'idea2video', user_requirement: '', input_files: {},
      reviewed_at: new Date().toISOString(), shots_reviewed: ['2', '4'], beats: [], suggestions: [],
    });

    const payload = await readContinuityReview(root, SESSION_ID, ROOT);

    expect(payload.stale).toBe(true);
    expect(payload.staleReason).toMatch(/render root/);
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

describe('idea scene timeline storage', () => {
  const sceneFile = (working, scene, ...parts) => path.join(working, 'idea2video', scene, ...parts);

  it('reads, updates, adds, removes, lists, and restores scene-local shots', async () => {
    const {root, working} = await ideaFixture();
    const listed = await readShotPlans(root, SESSION_ID, 'idea2video');

    expect(listed.plans.map((plan) => plan.slot)).toEqual(['scene_0/4', 'scene_0/2', 'scene_1/2']);
    expect((await readShotPlan(root, SESSION_ID, 'idea2video', 'scene_0/2')).characters.map((character) => character.name)).toEqual(['Scene Zero']);
    expect((await readShotPlan(root, SESSION_ID, 'idea2video', 'scene_1/2')).characters.map((character) => character.name)).toEqual(['Scene One']);

    await updateShotPlan(root, {sessionId: SESSION_ID, root: 'idea2video', slot: 'scene_0/2', ffDesc: 'Updated scene-zero opening'});
    expect(JSON.parse(await readFile(sceneFile(working, 'scene_0', 'shots', '2', 'shot_description.json'), 'utf8')).ff_desc)
      .toBe('Updated scene-zero opening');
    const created = await createShot(root, {sessionId: SESSION_ID, root: 'idea2video', after: 'scene_0/4', brief: 'A scene-zero insert'});
    expect(created.created.slot).toBe('scene_0/5');
    expect(JSON.parse(await readFile(sceneFile(working, 'scene_0', 'camera_tree.json'), 'utf8'))[0].active_shot_idxs).toEqual([4, 5, 2]);
    await expect(moveShot(root, {
      sessionId: SESSION_ID, root: 'idea2video', slot: 'scene_0/2', after: 'scene_1/2',
    })).rejects.toThrow(/cannot be moved between scenes/);

    await removeShot(root, {sessionId: SESSION_ID, root: 'idea2video', slot: created.created.slot});
    const removed = await readRemovedShots(root, SESSION_ID, 'idea2video');
    expect(removed.removed.map((entry) => entry.slot)).toEqual(['scene_0/5']);
    await restoreShot(root, {sessionId: SESSION_ID, root: 'idea2video', slot: created.created.slot});
    expect(JSON.parse(await readFile(sceneFile(working, 'scene_0', 'camera_tree.json'), 'utf8'))[0].active_shot_idxs).toEqual([4, 5, 2]);
    expect(JSON.parse(await readFile(sceneFile(working, 'scene_0', 'storyboard.json'), 'utf8')).map((row) => row.idx)).toEqual([4, 5, 2]);
  });
});
