import {createHash} from 'node:crypto';
import {mkdirSync, rmSync} from 'node:fs';
import {mkdtemp, mkdir, readFile, rm, symlink, writeFile} from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {afterEach, describe, expect, it} from 'vitest';
import {deleteAsset, deleteAssets, getAssetDeletionPlan, getAssetsDeletionPlan} from './server-lib.mjs';
import {createAgentMessageQueue} from './agent-message-queue.mjs';

const roots = [];
const sessionId = 'session-1';
const renderRoot = 'script2video';

afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) => rm(root, {recursive: true, force: true})));
});

async function fixture() {
  const repo = await mkdtemp(path.join(os.tmpdir(), 'vimax-asset-delete-'));
  roots.push(repo);
  const working = path.join(repo, '.working_dir', sessionId);
  await mkdir(working, {recursive: true});
  await mkdir(path.join(repo, '.vimax'));
  await writeFile(path.join(repo, '.vimax', 'sessions.json'), JSON.stringify({sessions: {
    [sessionId]: {session_id: sessionId, working_dir: `.working_dir/${sessionId}`},
  }}));
  return {repo, working};
}

async function put(root, relative, data = 'media') {
  const file = path.join(root, relative);
  await mkdir(path.dirname(file), {recursive: true});
  await writeFile(file, data);
  return file;
}

const approval = (relative) => ({accepted_at: '2026-01-01T00:00:00', artifacts: [{path: relative, size: 5, sha256: createHash('sha256').update('media').digest('hex')}]});

async function readAcceptance(working) {
  return JSON.parse(await readFile(path.join(working, 'render_acceptance.json'), 'utf8'));
}

describe('confirmed asset deletion backend', () => {
  it.each([false, true])('preserves media when queued work arrives during deletion preflight (bulk=%s)', async (bulk) => {
    const {repo, working} = await fixture();
    const paths = bulk ? ['archive/one.png', 'archive/two.png'] : ['archive/one.png'];
    for (const target of paths) await put(working, target);
    const plan = await getAssetsDeletionPlan(repo, sessionId, paths);
    const queue = createAgentMessageQueue();
    queue.reset(sessionId);
    let preflightStarted = false;
    const isBusy = () => {
      if (preflightStarted) queue.enqueue(sessionId, {id: 'new-work', text: 'New work', input: 'New work'});
      preflightStarted = true;
      return queue.hasWorkFor(sessionId);
    };
    const request = bulk
      ? deleteAssets(repo, {sessionId, paths, revision: plan.revision}, isBusy)
      : deleteAsset(repo, {sessionId, path: paths[0], revision: plan.revision}, isBusy);
    await expect(request).rejects.toMatchObject({statusCode: 409});
    for (const target of paths) expect(await readFile(path.join(working, target), 'utf8')).toBe('media');
    queue.cancel();
    await deleteAssets(repo, {sessionId, paths, revision: plan.revision}, () => queue.hasWorkFor(sessionId));
    for (const target of paths) await expect(readFile(path.join(working, target))).rejects.toMatchObject({code: 'ENOENT'});
  });

  it('deletes only the confirmed archived media and preserves live approvals and prompts', async () => {
    const {repo, working} = await fixture();
    const target = 'script2video/character_portraits/archive/front.png';
    await put(working, target);
    await put(working, 'script2video/character_portraits/archive/prompts.json', '{"front":"keep"}');
    await put(working, 'script2video/character_portraits/active/front.png');
    const previous = {script2video: {portraits: approval('script2video/character_portraits/active/front.png')}};
    await writeFile(path.join(working, 'render_acceptance.json'), JSON.stringify(previous));
    const plan = await getAssetDeletionPlan(repo, sessionId, target);
    expect(plan).toMatchObject({path: target, impacts: []});
    const result = await deleteAsset(repo, {sessionId, path: target, revision: plan.revision});
    expect(result.deletedPath).toBe(target);
    await expect(readFile(path.join(working, target))).rejects.toMatchObject({code: 'ENOENT'});
    expect(await readFile(path.join(working, 'script2video/character_portraits/archive/prompts.json'), 'utf8')).toContain('keep');
    expect(await readFile(path.join(working, 'script2video/character_portraits/active/front.png'), 'utf8')).toBe('media');
    expect(await readAcceptance(working)).toEqual(previous);
  });

  it('removes the matching selected view across shared registries and invalidates portrait and film approvals only', async () => {
    const {repo, working} = await fixture();
    const target = 'script2video/character_portraits/model-a/1_hero/front.png';
    await put(working, target);
    await put(working, 'script2video/character_portraits/model-a/1_hero/side.png');
    await put(working, 'script2video/character_portraits/model-b/1_hero/front.png');
    for (const model of ['model-a', 'model-b']) {
      const file = `script2video/character_portraits/${model}/registry.json`;
      await put(working, file, JSON.stringify({hero: {front: {path: model === 'model-a' ? target : path.join(working, target)}, side: {path: 'script2video/character_portraits/model-a/1_hero/side.png'}}}));
    }
    const unrelated = approval('script2video/character_portraits/model-a/1_hero/side.png');
    const initial = {script2video: {
      portrait_characters: {one: approval(target), other: unrelated},
      portraits: approval(target),
      shots: {0: {keyframes: approval('script2video/shots/0/first_frame.png'), clips: approval('script2video/shots/0/video.mp4')}},
      final_video: approval('script2video/final_video.mp4'),
    }};
    await writeFile(path.join(working, 'render_acceptance.json'), JSON.stringify(initial));
    const plan = await getAssetDeletionPlan(repo, sessionId, target);
    await deleteAsset(repo, {sessionId, path: target, revision: plan.revision});
    for (const model of ['model-a', 'model-b']) {
      const registry = JSON.parse(await readFile(path.join(working, `script2video/character_portraits/${model}/registry.json`), 'utf8'));
      expect(registry.hero.front).toBeUndefined();
      expect(registry.hero.side).toMatchObject({path: 'script2video/character_portraits/model-a/1_hero/side.png'});
    }
    const approvals = await readAcceptance(working);
    expect(approvals.script2video.portrait_characters.one.invalidated_at).toBeTruthy();
    expect(approvals.script2video.portrait_characters.other).toEqual(unrelated);
    expect(approvals.script2video.portraits.invalidated_at).toBeTruthy();
    expect(approvals.script2video.shots[0].keyframes.invalidated_at).toBeTruthy();
    expect(approvals.script2video.shots[0].clips.invalidated_at).toBeTruthy();
    expect(approvals.script2video.final_video.invalidated_at).toBeTruthy();
  });

  it('invalidates approved keyframes, their clip, and final without deleting neighboring media', async () => {
    const {repo, working} = await fixture();
    const target = 'script2video/shots/0/first_frame.png';
    await put(working, target);
    await put(working, 'script2video/shots/0/last_frame.png');
    await put(working, 'script2video/shots/0/model/video.mp4');
    const keep = approval('script2video/shots/0/model/video.mp4');
    await writeFile(path.join(working, 'render_acceptance.json'), JSON.stringify({script2video: {
      shots: {0: {keyframes: approval(target), clips: keep}, 1: {keyframes: approval('unrelated'), clips: approval('unrelated')}} ,
      final_video: approval('script2video/final_video.mp4'),
    }}));
    const plan = await getAssetDeletionPlan(repo, sessionId, target);
    await deleteAsset(repo, {sessionId, path: target, revision: plan.revision});
    const approvals = await readAcceptance(working);
    expect(approvals.script2video.shots[0].keyframes.invalidated_at).toBeTruthy();
    expect(approvals.script2video.shots[0].clips.invalidated_at).toBeTruthy();
    expect(approvals.script2video.shots[1]).toEqual({keyframes: approval('unrelated'), clips: approval('unrelated')});
    expect(approvals.script2video.final_video.invalidated_at).toBeTruthy();
    await expect(readFile(path.join(working, 'script2video/shots/0/last_frame.png'))).resolves.toBeTruthy();
    await expect(readFile(path.join(working, 'script2video/shots/0/model/video.mp4'))).resolves.toBeTruthy();
  });
  it('invalidates linked playback successors after a last-frame deletion with global chaining off', async () => {
    const {repo, working} = await fixture();
    const order = [1, 2, 3, 4];
    await put(working, 'script2video/camera_tree.json', JSON.stringify([
      {idx: 0, active_shot_idxs: order},
    ]));
    for (const slot of order) {
      await put(working, `script2video/shots/${slot}/shot_description.json`, JSON.stringify({
        idx: slot,
        reuse_previous_last_frame: [2, 3].includes(slot),
      }));
    }
    const target = 'script2video/shots/1/model/last_frame.png';
    await put(working, target);
    const original = {script2video: {
      shots: Object.fromEntries(order.map((slot) => [slot, {
        keyframes: approval(`script2video/shots/${slot}/first_frame.png`),
        clips: approval(`script2video/shots/${slot}/video.mp4`),
      }])),
      final_video: approval('script2video/final_video.mp4'),
    }};
    await put(working, 'render_acceptance.json', JSON.stringify(original));

    const plan = await getAssetDeletionPlan(repo, sessionId, target);
    await deleteAsset(repo, {sessionId, path: target, revision: plan.revision});

    const approvals = (await readAcceptance(working)).script2video;
    for (const slot of ['1', '2', '3']) {
      expect(approvals.shots[slot].keyframes.invalidated_at).toBeTruthy();
      expect(approvals.shots[slot].clips.invalidated_at).toBeTruthy();
    }
    expect(approvals.shots[4]).toEqual(original.script2video.shots[4]);
    expect(approvals.final_video.invalidated_at).toBeTruthy();
    await expect(readFile(path.join(working, target))).rejects.toMatchObject({code: 'ENOENT'});
  });

  it('refuses changed reference metadata and invalidates shots referencing an archived source', async () => {
    const {repo, working} = await fixture();
    const target = 'archive/geometry.png';
    await put(working, target);
    const frame = 'script2video/shots/0/current-model/first_frame.png';
    const clip = 'script2video/shots/0/current-video/video.mp4';
    const final = 'script2video/final_video.mp4';
    for (const file of [frame, clip, final]) await put(working, file);
    await put(working, 'render_acceptance.json', JSON.stringify({script2video: {
      shots: {0: {keyframes: approval(frame), clips: approval(clip)}},
      final_video: approval(final),
    }}));
    const stale = await getAssetDeletionPlan(repo, sessionId, target);
    const selector = 'script2video/shots/0/first_frame_selector_output.json';
    const saved = JSON.stringify({reference_image_path_and_text_pairs: [[path.join(working, target), 'Geometry source']]});
    await put(working, selector, saved);
    await expect(deleteAsset(repo, {sessionId, path: target, revision: stale.revision})).rejects.toMatchObject({statusCode: 409});
    expect(await readFile(path.join(working, target), 'utf8')).toBe('media');
    const fresh = await getAssetDeletionPlan(repo, sessionId, target);
    await deleteAsset(repo, {sessionId, path: target, revision: fresh.revision});
    const approvals = await readAcceptance(working);
    expect(approvals.script2video.shots[0].keyframes.invalidated_at).toBeTruthy();
    expect(approvals.script2video.shots[0].clips.invalidated_at).toBeTruthy();
    expect(approvals.script2video.final_video.invalidated_at).toBeTruthy();
    for (const file of [frame, clip, final]) expect(await readFile(path.join(working, file), 'utf8')).toBe('media');
    expect(await readFile(path.join(working, selector), 'utf8')).toBe(saved);
  });

  it('keeps current-model approvals when deleting an unselected old-model keyframe', async () => {
    const {repo, working} = await fixture();
    const target = 'script2video/shots/0/old-model/first_frame.png';
    await put(working, target);
    const current = 'script2video/shots/0/current-model/first_frame.png';
    const clip = 'script2video/shots/0/current-video/video.mp4';
    const final = 'script2video/final_video.mp4';
    for (const file of [current, clip, final]) await put(working, file);
    await put(working, 'render_manifest.json', JSON.stringify({image_model: 'current-model', video_model: 'current-video'}));
    const previous = {script2video: {
      shots: {0: {keyframes: approval(current), clips: approval(clip)}},
      final_video: approval(final),
    }};
    await put(working, 'render_acceptance.json', JSON.stringify(previous));
    const plan = await getAssetDeletionPlan(repo, sessionId, target);
    await deleteAsset(repo, {sessionId, path: target, revision: plan.revision});
    await expect(readFile(path.join(working, target))).rejects.toMatchObject({code: 'ENOENT'});
    expect(await readAcceptance(working)).toEqual(previous);
    for (const file of [current, clip, final]) expect(await readFile(path.join(working, file), 'utf8')).toBe('media');
  });

  it('rejects stale bytes or approval metadata without mutating any file', async () => {
    const {repo, working} = await fixture();
    const target = 'archive/frame.jpg';
    const file = await put(working, target, 'first');
    const plan = await getAssetDeletionPlan(repo, sessionId, target);
    await writeFile(file, 'second');
    await expect(deleteAsset(repo, {sessionId, path: target, revision: plan.revision})).rejects.toMatchObject({statusCode: 409});
    expect(await readFile(file, 'utf8')).toBe('second');
    const fresh = await getAssetDeletionPlan(repo, sessionId, target);
    await writeFile(path.join(working, 'render_acceptance.json'), JSON.stringify({script2video: {portraits: approval('elsewhere')}}));
    await expect(deleteAsset(repo, {sessionId, path: target, revision: fresh.revision})).rejects.toMatchObject({statusCode: 409});
    expect(await readFile(file, 'utf8')).toBe('second');
  });

  it('rejects a busy session and malformed deletion bodies without changing the media', async () => {
    const {repo, working} = await fixture();
    const target = 'archive/busy.webp';
    const file = await put(working, target);
    const plan = await getAssetDeletionPlan(repo, sessionId, target);
    await expect(deleteAsset(repo, {sessionId, path: target, revision: plan.revision}, () => true))
      .rejects.toMatchObject({statusCode: 409});
    await expect(deleteAsset(repo, {sessionId, path: target}))
      .rejects.toMatchObject({statusCode: 400});
    expect(await readFile(file, 'utf8')).toBe('media');
  });

  it('rejects unsafe paths, symlink ancestors, non-media and absent files', async () => {
    const {repo, working} = await fixture();
    const outside = await put(path.dirname(working), 'outside/photo.png');
    await symlink(path.dirname(outside), path.join(working, 'linked'));
    await expect(getAssetDeletionPlan(repo, sessionId, '../outside/photo.png')).rejects.toMatchObject({statusCode: 400});
    await expect(getAssetDeletionPlan(repo, sessionId, 'linked/photo.png')).rejects.toMatchObject({statusCode: 400});
    await put(working, 'archive/prompts.json', '{}');
    await expect(getAssetDeletionPlan(repo, sessionId, 'archive/prompts.json')).rejects.toMatchObject({statusCode: 400});
    await expect(getAssetDeletionPlan(repo, sessionId, 'archive/missing.png')).rejects.toMatchObject({statusCode: 404});
    expect(await readFile(outside, 'utf8')).toBe('media');
  });
  it('plans a mixed archive and active batch once, canonicalizes duplicates, and preserves survivors', async () => {
    const {repo, working} = await fixture();
    const front = 'archive/portrait-front.png';
    const back = 'archive/portrait-back.png';
    const retired = 'archive/retired.mp4';
    const active = 'script2video/shots/0/current-model/first_frame.png';
    const clip = 'script2video/shots/0/current-video/video.mp4';
    const final = 'script2video/final_video.mp4';
    const survivor = 'archive/keep.png';
    for (const target of [front, back, retired, active, clip, final, survivor]) await put(working, target);
    await put(working, 'render_manifest.json', JSON.stringify({image_model: 'current-model', video_model: 'current-video'}));
    const registryPath = 'script2video/character_portraits/current-model/registry.json';
    const promptsPath = 'script2video/character_portraits/current-model/prompts.json';
    await put(working, registryPath, JSON.stringify({
      hero: {front: {path: front}, back: {path: back}},
      villain: {front: {path: survivor}},
    }));
    await put(working, promptsPath, JSON.stringify({front: 'keep these prompts'}));
    await put(working, 'render_acceptance.json', JSON.stringify({script2video: {
      portraits: approval(front),
      portrait_characters: {hero: approval(back)},
      shots: {0: {keyframes: approval(active), clips: approval(clip)}},
      final_video: approval(final),
    }}));

    const paths = [active, retired, back, front, active];
    const plan = await getAssetsDeletionPlan(repo, sessionId, paths);
    expect(plan.paths).toEqual([back, front, retired, active].sort());
    expect(plan.impacts.some((impact) => impact.includes(`${active}: Invalidate keyframes and clips`))).toBe(true);
    expect(plan.impacts.some((impact) => impact.includes(`${front}: Remove 1 selected portrait reference`))).toBe(true);
    expect(plan.impacts.some((impact) => impact.includes(`${back}: Remove 1 selected portrait reference`))).toBe(true);

    const result = await deleteAssets(repo, {sessionId, paths, revision: plan.revision});
    expect(result.deletedPaths).toEqual(plan.paths);
    for (const target of plan.paths) await expect(readFile(path.join(working, target))).rejects.toMatchObject({code: 'ENOENT'});
    for (const target of [clip, final, survivor]) {
      expect(await readFile(path.join(working, target), 'utf8')).toBe('media');
    }
    const registry = JSON.parse(await readFile(path.join(working, registryPath), 'utf8'));
    expect(registry.hero).toEqual({});
    expect(registry.villain.front.path).toBe(survivor);
    expect(await readFile(path.join(working, promptsPath), 'utf8')).toContain('keep these prompts');
    const approvals = await readAcceptance(working);
    expect(approvals.script2video.portraits.invalidated_at).toBeTruthy();
    expect(approvals.script2video.portrait_characters.hero.invalidated_at).toBeTruthy();
    expect(approvals.script2video.shots[0].keyframes.invalidated_at).toBeTruthy();
    expect(approvals.script2video.shots[0].clips.invalidated_at).toBeTruthy();
    expect(approvals.script2video.final_video.invalidated_at).toBeTruthy();
  });

  it('refuses a stale later batch member before deleting any earlier member', async () => {
    const {repo, working} = await fixture();
    const first = 'archive/a.png';
    const later = 'archive/b.png';
    const firstFile = await put(working, first);
    const laterFile = await put(working, later);
    const plan = await getAssetsDeletionPlan(repo, sessionId, [later, first, later]);
    expect(plan.paths).toEqual([first, later]);

    await writeFile(laterFile, 'changed');
    await expect(deleteAssets(repo, {sessionId, paths: [first, later, later], revision: plan.revision}))
      .rejects.toMatchObject({statusCode: 409});
    expect(await readFile(firstFile, 'utf8')).toBe('media');
    expect(await readFile(laterFile, 'utf8')).toBe('changed');
  });

  it('refuses stale approval and registry metadata for a batch without deleting media', async () => {
    const {repo, working} = await fixture();
    const first = 'archive/portrait.png';
    const second = 'archive/other.png';
    const firstFile = await put(working, first);
    const secondFile = await put(working, second);
    const registryPath = 'script2video/character_portraits/model-a/registry.json';
    await put(working, registryPath, JSON.stringify({hero: {front: {path: first}}}));
    await put(working, 'render_acceptance.json', JSON.stringify({script2video: {portraits: approval('elsewhere.png')}}));

    const approvalPlan = await getAssetsDeletionPlan(repo, sessionId, [first, second]);
    await writeFile(path.join(working, 'render_acceptance.json'), JSON.stringify({script2video: {portraits: approval(first)}}));
    await expect(deleteAssets(repo, {sessionId, paths: [first, second], revision: approvalPlan.revision}))
      .rejects.toMatchObject({statusCode: 409});
    expect(await readFile(firstFile, 'utf8')).toBe('media');
    expect(await readFile(secondFile, 'utf8')).toBe('media');

    const registryPlan = await getAssetsDeletionPlan(repo, sessionId, [first, second]);
    const updatedRegistry = {hero: {front: {path: second}}};
    await writeFile(path.join(working, registryPath), JSON.stringify(updatedRegistry));
    await expect(deleteAssets(repo, {sessionId, paths: [first, second], revision: registryPlan.revision}))
      .rejects.toMatchObject({statusCode: 409});
    expect(await readFile(firstFile, 'utf8')).toBe('media');
    expect(await readFile(secondFile, 'utf8')).toBe('media');
    expect(JSON.parse(await readFile(path.join(working, registryPath), 'utf8'))).toEqual(updatedRegistry);
  });

  it('refuses missing, unsafe, and busy batch members without partial deletion', async () => {
    const {repo, working} = await fixture();
    const first = 'archive/a.png';
    const second = 'archive/b.png';
    const firstFile = await put(working, first);
    const secondFile = await put(working, second);
    const missingPlan = await getAssetsDeletionPlan(repo, sessionId, [first, second]);
    await rm(secondFile);
    await expect(deleteAssets(repo, {sessionId, paths: [first, second], revision: missingPlan.revision}))
      .rejects.toMatchObject({statusCode: 404});
    expect(await readFile(firstFile, 'utf8')).toBe('media');

    await put(working, second);
    const unsafePlan = await getAssetsDeletionPlan(repo, sessionId, [first, second]);
    const outside = await put(path.dirname(working), 'outside/linked.png');
    await rm(secondFile);
    await symlink(outside, secondFile);
    await expect(deleteAssets(repo, {sessionId, paths: [first, second], revision: unsafePlan.revision}))
      .rejects.toMatchObject({statusCode: 400});
    expect(await readFile(firstFile, 'utf8')).toBe('media');

    await rm(secondFile);
    await put(working, second);
    const busyPlan = await getAssetsDeletionPlan(repo, sessionId, [first, second]);
    await expect(deleteAssets(repo, {sessionId, paths: [first, second], revision: busyPlan.revision}, () => true))
      .rejects.toMatchObject({statusCode: 409});
    expect(await readFile(firstFile, 'utf8')).toBe('media');
    expect(await readFile(path.join(working, second), 'utf8')).toBe('media');
  });
  it('rolls back registry writes when approval replacement fails mid-transaction', async () => {
    const {repo, working} = await fixture();
    const first = 'archive/portrait.png';
    const second = 'archive/other.png';
    const firstFile = await put(working, first);
    const secondFile = await put(working, second);
    const registryPath = path.join(working, 'script2video/character_portraits/model-a/registry.json');
    const registryBytes = JSON.stringify({hero: {front: {path: first}}, keep: {front: {path: second}}});
    const acceptancePath = path.join(working, 'render_acceptance.json');
    const acceptanceBytes = JSON.stringify({script2video: {portraits: approval(first)}});
    await put(working, 'script2video/character_portraits/model-a/registry.json', registryBytes);
    await put(working, 'render_acceptance.json', acceptanceBytes);
    const plan = await getAssetsDeletionPlan(repo, sessionId, [first, second]);
    let busyChecks = 0;

    await expect(deleteAssets(repo, {sessionId, paths: [first, second], revision: plan.revision}, () => {
      busyChecks += 1;
      if (busyChecks === 2) {
        rmSync(acceptancePath);
        mkdirSync(acceptancePath);
      }
      return false;
    })).rejects.toMatchObject({statusCode: 400});

    expect(await readFile(registryPath, 'utf8')).toBe(registryBytes);
    expect(await readFile(firstFile, 'utf8')).toBe('media');
    expect(await readFile(secondFile, 'utf8')).toBe('media');
    rmSync(acceptancePath, {recursive: true});
    await writeFile(acceptancePath, acceptanceBytes);
  });
});
