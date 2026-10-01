import {existsSync} from 'node:fs';
import {mkdtemp, mkdir, readFile, rm, writeFile} from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {afterEach, describe, expect, it, vi} from 'vitest';
import {createShot, readRemovedShots, readShotPlan, readRenderAcceptance, removeShot, restoreShot, updateRenderAcceptance, updateShotPlan} from './server-lib.mjs';

const storageTestControl = vi.hoisted(() => ({failAcceptanceWrite: false}));
vi.mock('node:fs/promises', async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    writeFile: (filePath, ...args) => {
      const target = String(filePath);
      if (storageTestControl.failAcceptanceWrite && target.includes('.render_acceptance.json.') && target.endsWith('.tmp')) {
        const error = new Error('injected acceptance metadata failure');
        error.code = 'EIO';
        throw error;
      }
      return actual.writeFile(filePath, ...args);
    },
  };
});
const roots = [];
const SESSION_ID = 'session-1';
const ROOT = 'script2video';

afterEach(async () => {
  storageTestControl.failAcceptanceWrite = false;
  await Promise.all(roots.splice(0).map((root) => rm(root, {recursive: true, force: true})));
});

async function fixture() {
  const root = await mkdtemp(path.join(os.tmpdir(), 'vimax-shot-plan-'));
  roots.push(root);
  const working = path.join(root, '.working_dir', SESSION_ID);
  await mkdir(path.join(root, '.vimax'), {recursive: true});
  await writeFile(path.join(root, '.vimax', 'sessions.json'), JSON.stringify({
    active_session_id: SESSION_ID,
    sessions: {[SESSION_ID]: {session_id: SESSION_ID, project_name: 'Film', working_dir: `.working_dir/${SESSION_ID}`, stage: 'stills_ready', updated_at: '2026-09-21T09:00:00'}},
  }));
  await mkdir(working, {recursive: true});
  await writeFile(path.join(working, 'render_manifest.json'), JSON.stringify({render_mode: ROOT}));
  await mkdir(path.join(working, ROOT, 'shots', '7'), {recursive: true});
  await writeFile(path.join(working, ROOT, 'characters.json'), JSON.stringify([
    {idx: 0, identifier_in_scene: 'Claude', static_features: 'heavyset', dynamic_features: 'orange shirt'},
    {idx: 1, identifier_in_scene: 'Wife', static_features: 'slim', dynamic_features: 'pink dress'},
    {idx: 2, identifier_in_scene: 'DeepSeek', static_features: 'younger man', dynamic_features: 'whale t-shirt'},
  ]));
  // The shape the render wrote for this shot: the description is about DeepSeek and the Wife
  // while the visible list says Claude and Wife.
  await writeFile(path.join(working, ROOT, 'shots', '7', 'shot_description.json'), JSON.stringify({
    idx: 7, ff_desc: 'DeepSeek enters from the right.', lf_desc: 'DeepSeek beside the Wife.',
    ff_vis_char_idxs: [0], lf_vis_char_idxs: [0, 1], motion_desc: 'he walks in', audio_desc: 'none',
  }, null, 4));
  await writeFile(path.join(working, ROOT, 'shots', '7', 'last_frame_selector_output.json'), JSON.stringify({sent_prompt: 'the prompt that was sent'}));
  await writeFile(path.join(working, ROOT, 'storyboard.json'), JSON.stringify([{idx: 7, is_last: false, cam_idx: 3, visual_desc: 'the shot being removed', audio_desc: ''}]));
  return {root, working};
}

const read = (root) => readShotPlan(root, SESSION_ID, ROOT, '7');
const update = (root, input) => updateShotPlan(root, {sessionId: SESSION_ID, root: ROOT, slot: '7', ...input});

describe('shot plans', () => {
  it('reads a shot with the characters it can choose from and the prompt it was drawn from', async () => {
    const {root} = await fixture();

    const plan = await read(root);

    expect(plan.characters.map((character) => character.name)).toEqual(['Claude', 'Wife', 'DeepSeek']);
    expect(plan.brief).toBe('the shot being removed');
    expect(plan.firstFrame).toMatchObject({description: 'DeepSeek enters from the right.', visible: [0], prompt: ''});
    expect(plan.lastFrame).toMatchObject({description: 'DeepSeek beside the Wife.', visible: [0, 1], prompt: 'the prompt that was sent'});
  });

  it('writes back which characters each frame shows, and what it describes', async () => {
    const {root, working} = await fixture();

    const plan = await update(root, {ffVis: [0, 1, 2], lfDesc: 'Claude on the couch with the Wife standing beside him.'});

    expect(plan.firstFrame.visible).toEqual([0, 1, 2]);
    expect(plan.lastFrame.description).toBe('Claude on the couch with the Wife standing beside him.');
    const stored = JSON.parse(await readFile(path.join(working, ROOT, 'shots', '7', 'shot_description.json'), 'utf8'));
    expect(stored.ff_vis_char_idxs).toEqual([0, 1, 2]);
    expect(stored.lf_desc).toBe('Claude on the couch with the Wife standing beside him.');
    // Everything the edit did not touch stays as the render wrote it.
    expect(stored.motion_desc).toBe('he walks in');
    expect(stored.ff_desc).toBe('DeepSeek enters from the right.');
  });

  it('refuses a plan edit that would leave the render nothing to work with', async () => {
    const {root} = await fixture();
    const refused = (input) => update(root, input).catch((error) => error);

    expect((await refused({ffDesc: '   '})).message).toMatch(/cannot be empty/);
    expect((await refused({ffVis: [9]})).message).toMatch(/Unknown character index: 9/);
    expect((await refused({})).message).toMatch(/Nothing to save/);
    expect((await readShotPlan(root, SESSION_ID, ROOT, '99').catch((error) => error)).message).toMatch(/Unknown shot: 99/);
    await expect(readShotPlan(root, 'unknown-session', ROOT, '7')).resolves.toBeNull();
  });
});

describe('removing a shot from the film', () => {
  async function withCameras(working) {
    await writeFile(path.join(working, ROOT, 'camera_tree.json'), JSON.stringify([
      {idx: 0, active_shot_idxs: [0, 3, 6, 9, 12], parent_shot_idx: null},
      {idx: 2, active_shot_idxs: [2, 4, 5, 8, 10], parent_shot_idx: 0},
      {idx: 3, active_shot_idxs: [7, 11], parent_shot_idx: 0},
    ], null, 4));
  }

  async function withShots(working, slots) {
    for (const slot of slots) {
      const dir = path.join(working, ROOT, 'shots', String(slot));
      await mkdir(dir, {recursive: true});
      await writeFile(path.join(dir, 'shot_description.json'), JSON.stringify({idx: Number(slot), ff_desc: 'f', lf_desc: 'l', ff_vis_char_idxs: [], lf_vis_char_idxs: [], is_last: false}));
      await mkdir(path.join(dir, 'qwen_qwen-image-3'), {recursive: true});
      await writeFile(path.join(dir, 'qwen_qwen-image-3', 'first_frame.png'), 'frame');
    }
  }

  const remove = (root, slot) => removeShot(root, {sessionId: SESSION_ID, root: ROOT, slot});

  it('takes the shot out of its camera and out of everything the timeline lists', async () => {
    const {root, working} = await fixture();
    await withCameras(working);
    await withShots(working, [0, 3, 6, 9, 12]);

    const payload = await remove(root, '6');

    const tree = JSON.parse(await readFile(path.join(working, ROOT, 'camera_tree.json'), 'utf8'));
    expect(tree.find((camera) => camera.idx === 0).active_shot_idxs).toEqual([0, 3, 9, 12]);
    // Its frames are kept, out of the way, so the removal can be undone.
    expect(existsSync(path.join(working, ROOT, 'shots', '6'))).toBe(false);
    expect(await readFile(path.join(working, ROOT, '.removed_shots', '6', 'qwen_qwen-image-3', 'first_frame.png'), 'utf8')).toBe('frame');
    expect(payload.stages.find((stage) => stage.stage === 'keyframes').slots.map((slot) => slot.shot)).not.toContain('6');
  });

  it('takes the shot\u2019s brief out of the storyboard, so a re-plan cannot bring it back', async () => {
    const {root, working} = await fixture();
    await withCameras(working);
    await withShots(working, [0, 3, 6, 9, 12]);
    await writeFile(path.join(working, ROOT, 'storyboard.json'), JSON.stringify([
      {idx: 0, is_last: false, cam_idx: 0, visual_desc: 'first'},
      {idx: 6, is_last: false, cam_idx: 0, visual_desc: 'the shot being removed'},
      {idx: 12, is_last: true, cam_idx: 0, visual_desc: 'last'},
    ], null, 4));

    await remove(root, '6');

    const storyboard = JSON.parse(await readFile(path.join(working, ROOT, 'storyboard.json'), 'utf8'));
    expect(storyboard.map((entry) => entry.idx)).toEqual([0, 12]);
    // Kept with the shot, so restoring is putting two things back rather than rewriting one.
    const brief = JSON.parse(await readFile(path.join(working, ROOT, '.removed_shots', '6', 'brief.json'), 'utf8'));
    expect(brief.visual_desc).toBe('the shot being removed');
  });

  it('moves the film\u2019s ending onto the shot that now ends it', async () => {
    const {root, working} = await fixture();
    await withCameras(working);
    // Every shot a camera lists, so the film's ending is a shot that exists.
    await withShots(working, [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    const mark = (slot, value) => writeFile(path.join(working, ROOT, 'shots', String(slot), 'shot_description.json'),
      JSON.stringify({idx: slot, ff_desc: 'f', lf_desc: 'l', ff_vis_char_idxs: [], lf_vis_char_idxs: [], is_last: value}));
    await mark(12, true);

    await remove(root, '12');

    expect(JSON.parse(await readFile(path.join(working, ROOT, 'shots', '11', 'shot_description.json'), 'utf8')).is_last).toBe(true);
  });

  it('refuses a removal that would leave a camera without its transition', async () => {
    const {root, working} = await fixture();
    await withCameras(working);
    await withShots(working, [0, 3, 7, 11]);
    const refused = (slot) => remove(root, slot).catch((error) => error);

    // Shots 2 and 3 move away from shot 0: removing it would leave them without a transition.
    expect((await refused('0')).message).toMatch(/without its transition/);
    expect((await refused('99')).message).toMatch(/not part of a camera/);
    // A camera's first shot is removable: the next one takes over its opening.
    expect((await remove(root, '7')).stages).toBeDefined();
  });

  it('lets a camera be emptied, and takes the stale film with it', async () => {
    const {root, working} = await fixture();
    await withCameras(working);
    await withShots(working, [7, 11]);
    await writeFile(path.join(working, ROOT, 'final_video.mp4'), 'film');

    // camera 3 holds only shots 7 and 11, and nothing moves away from either of them
    await remove(root, '7');
    const payload = await remove(root, '11');

    expect(payload.stages).toBeDefined();
    const cameras = JSON.parse(await readFile(path.join(working, ROOT, 'camera_tree.json'), 'utf8'));
    expect(cameras.find((camera) => camera.idx === 3).active_shot_idxs).toEqual([]);
    // the film was every shot's clip joined, and cannot outlive the shot that left it
    expect(existsSync(path.join(working, ROOT, 'final_video.mp4'))).toBe(false);
  });
});

describe('adding and restoring shots', () => {
  const remove = (root, slot) => removeShot(root, {sessionId: SESSION_ID, root: ROOT, slot});

  async function withFilm(working, slots) {
    await writeFile(path.join(working, ROOT, 'camera_tree.json'), JSON.stringify([
      {idx: 0, active_shot_idxs: [0, 3], parent_shot_idx: null},
      {idx: 2, active_shot_idxs: slots, parent_shot_idx: 0},
    ], null, 4));
    await writeFile(path.join(working, ROOT, 'storyboard.json'), JSON.stringify(
      slots.map((slot) => ({idx: slot, is_last: false, cam_idx: 2, visual_desc: `shot ${slot}`, audio_desc: ''})), null, 4));
    for (const slot of slots) {
      const dir = path.join(working, ROOT, 'shots', String(slot));
      await mkdir(path.join(dir, 'qwen_qwen-image-3'), {recursive: true});
      await writeFile(path.join(dir, 'shot_description.json'), JSON.stringify({idx: slot, ff_desc: `first ${slot}`, lf_desc: `last ${slot}`, ff_vis_char_idxs: [0], lf_vis_char_idxs: [0, 1], motion_desc: 'm', audio_desc: 'none', is_last: false}));
      await writeFile(path.join(dir, 'qwen_qwen-image-3', 'first_frame.png'), 'frame');
    }
  }

  it('adds a shot after another one, copying its plan and saying so', async () => {
    const {root, working} = await fixture();
    await withFilm(working, [2, 4]);

    const payload = await createShot(root, {sessionId: SESSION_ID, root: ROOT, after: '2', brief: 'the Wife sets the table behind him'});

    expect(payload.created.copiedFrom).toBe('2');
    const tree = JSON.parse(await readFile(path.join(working, ROOT, 'camera_tree.json'), 'utf8'));
    // Spliced where it plays, not sorted to the end: a camera's list is the film's order.
    expect(tree.find((camera) => camera.idx === 2).active_shot_idxs).toEqual([2, 5, 4]);
    // The new shot's plan starts as the reference's, and its brief is what the user wrote.
    const plan = JSON.parse(await readFile(path.join(working, ROOT, 'shots', '5', 'shot_description.json'), 'utf8'));
    expect(plan.ff_desc).toBe('first 2');
    expect(plan.idx).toBe(5);
    const storyboard = JSON.parse(await readFile(path.join(working, ROOT, 'storyboard.json'), 'utf8'));
    expect(storyboard.find((entry) => entry.idx === 5).visual_desc).toBe('the Wife sets the table behind him');
    expect(payload.stages.find((stage) => stage.stage === 'keyframes').slots.map((slot) => slot.shot)).toContain('5');
  });

  it('refuses to add a shot without a description, or after a shot that is not in a camera', async () => {
    const {root, working} = await fixture();
    await withFilm(working, [2, 4]);
    const refused = (input) => createShot(root, {sessionId: SESSION_ID, root: ROOT, ...input}).catch((error) => error);

    expect((await refused({after: '2', brief: '  '})).message).toMatch(/Say what happens/);
    expect((await refused({after: '99', brief: 'x'})).message).toMatch(/nothing to add a shot after/);
  });

  it('puts a removed shot back into its camera, with its brief and its frames', async () => {
    const {root, working} = await fixture();
    await withFilm(working, [2, 4]);

    await remove(root, '4');
    const payload = await restoreShot(root, {sessionId: SESSION_ID, root: ROOT, slot: '4'});

    const tree = JSON.parse(await readFile(path.join(working, ROOT, 'camera_tree.json'), 'utf8'));
    expect(tree.find((camera) => camera.idx === 2).active_shot_idxs).toEqual([2, 4]);
    expect(JSON.parse(await readFile(path.join(working, ROOT, 'shots', '4', 'shot_description.json'), 'utf8')).ff_desc).toBe('first 4');
    expect(await readFile(path.join(working, ROOT, 'shots', '4', 'qwen_qwen-image-3', 'first_frame.png'), 'utf8')).toBe('frame');
    const storyboard = JSON.parse(await readFile(path.join(working, ROOT, 'storyboard.json'), 'utf8'));
    expect(storyboard.map((entry) => entry.idx)).toEqual([2, 4]);
    expect(payload.stages.find((stage) => stage.stage === 'keyframes').slots.map((slot) => slot.shot)).toContain('4');
  });

  it('lists what has been removed, and refuses to restore something that was not', async () => {
    const {root, working} = await fixture();
    await withFilm(working, [2, 4]);
    await remove(root, '4');

    const listed = await readRemovedShots(root, SESSION_ID, ROOT);

    expect(listed.removed.map((entry) => entry.slot)).toEqual(['4']);
    expect(listed.removed[0].hasBrief).toBe(true);
    await expect(restoreShot(root, {sessionId: SESSION_ID, root: ROOT, slot: '9'})).rejects.toThrow(/has not been removed/);
  });
  it('restores adjacent removals to their original order and keeps one canonical ending', async () => {
    const {root, working} = await fixture();
    await withFilm(working, [0, 1, 2]);
    await writeFile(path.join(working, ROOT, 'camera_tree.json'), JSON.stringify([
      {idx: 0, active_shot_idxs: [0, 1, 2], parent_shot_idx: null},
    ]));
    const storyboardPath = path.join(working, ROOT, 'storyboard.json');
    const storyboard = JSON.parse(await readFile(storyboardPath, 'utf8'));
    await writeFile(storyboardPath, JSON.stringify(storyboard.map((row) => ({...row, is_last: row.idx === 2}))));
    const endingPath = path.join(working, ROOT, 'shots', '2', 'shot_description.json');
    const ending = JSON.parse(await readFile(endingPath, 'utf8'));
    await writeFile(endingPath, JSON.stringify({...ending, is_last: true}));

    await remove(root, '0');
    await remove(root, '1');
    await restoreShot(root, {sessionId: SESSION_ID, root: ROOT, slot: '0'});
    await restoreShot(root, {sessionId: SESSION_ID, root: ROOT, slot: '1'});

    const cameras = JSON.parse(await readFile(path.join(working, ROOT, 'camera_tree.json'), 'utf8'));
    expect(cameras[0].active_shot_idxs).toEqual([0, 1, 2]);
    const descriptions = await Promise.all([0, 1, 2].map((slot) => readFile(
      path.join(working, ROOT, 'shots', String(slot), 'shot_description.json'), 'utf8',
    ).then((text) => JSON.parse(text))));
    expect(descriptions.filter((description) => description.is_last).map((description) => description.idx)).toEqual([2]);
    const endingRows = JSON.parse(await readFile(storyboardPath, 'utf8')).filter((row) => row.is_last);
    expect(endingRows.map((row) => row.idx)).toEqual([2]);
  });

  it('rejects a removed-directory collision without changing the live shot or timeline', async () => {
    const {root, working} = await fixture();
    await withFilm(working, [2, 4]);
    const treePath = path.join(working, ROOT, 'camera_tree.json');
    const storyboardPath = path.join(working, ROOT, 'storyboard.json');
    const originalTree = await readFile(treePath, 'utf8');
    const originalStoryboard = await readFile(storyboardPath, 'utf8');
    const collision = path.join(working, ROOT, '.removed_shots', '4');
    await mkdir(collision, {recursive: true});
    await writeFile(path.join(collision, 'sentinel.txt'), 'kept');

    await expect(remove(root, '4')).rejects.toThrow(/Destination already exists/);

    expect(await readFile(treePath, 'utf8')).toBe(originalTree);
    expect(await readFile(storyboardPath, 'utf8')).toBe(originalStoryboard);
    expect(existsSync(path.join(working, ROOT, 'shots', '4'))).toBe(true);
    expect(await readFile(path.join(collision, 'sentinel.txt'), 'utf8')).toBe('kept');
  });

  it('rolls back a removal when acceptance metadata cannot be replaced', async () => {
    const {root, working} = await fixture();
    await withFilm(working, [2, 4]);
    await writeFile(path.join(working, ROOT, 'final_video.mp4'), 'old film');
    await updateRenderAcceptance(root, {sessionId: SESSION_ID, root: ROOT, stage: 'final_video', accepted: true});
    const treePath = path.join(working, ROOT, 'camera_tree.json');
    const storyboardPath = path.join(working, ROOT, 'storyboard.json');
    const originalTree = await readFile(treePath, 'utf8');
    const originalStoryboard = await readFile(storyboardPath, 'utf8');
    storageTestControl.failAcceptanceWrite = true;

    await expect(remove(root, '4')).rejects.toThrow(/injected acceptance metadata failure/);

    storageTestControl.failAcceptanceWrite = false;
    expect(await readFile(treePath, 'utf8')).toBe(originalTree);
    expect(await readFile(storyboardPath, 'utf8')).toBe(originalStoryboard);
    expect(existsSync(path.join(working, ROOT, 'shots', '4'))).toBe(true);
    expect(existsSync(path.join(working, ROOT, '.removed_shots', '4'))).toBe(false);
    expect(await readFile(path.join(working, ROOT, 'final_video.mp4'), 'utf8')).toBe('old film');
    const acceptance = await readRenderAcceptance(root, SESSION_ID, ROOT);
    expect(acceptance.stages.find((stage) => stage.stage === 'final_video').slots[0].state).toBe('accepted');
  });
});
