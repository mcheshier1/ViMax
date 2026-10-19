import {createHash} from 'node:crypto';
import {mkdtemp, mkdir, readFile, rm, writeFile} from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {afterEach, describe, expect, it, vi} from 'vitest';
import {applySetReferenceImage, readSetReferences, selectSetReference, updateSetReference, updateSetReferenceAcceptance} from './server-lib.mjs';

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
const SESSION_ID = 'set-project';
const ROOT = 'script2video';
const MODEL = 'qwen/qwen-image-3';
const MODEL_DIR = 'qwen_qwen-image-3';
const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAACAAAAASCAIAAAC1qksFAAAAI0lEQVR4nGP8z0BbwERj80ctGLVg1IJRC0YtGLWAgYGBgQEATC4BI51ZZZUAAAAASUVORK5CYII=', 'base64');

afterEach(async () => {
  storageTestControl.failAcceptanceWrite = false;
  await Promise.all(roots.splice(0).map((root) => rm(root, {recursive: true, force: true})));
});

async function fixture() {
  const repo = await mkdtemp(path.join(os.tmpdir(), 'vimax-set-references-'));
  roots.push(repo);
  const working = path.join(repo, '.working_dir', SESSION_ID);
  await mkdir(path.join(repo, '.vimax'), {recursive: true});
  await mkdir(working, {recursive: true});
  await writeFile(path.join(repo, '.vimax', 'sessions.json'), JSON.stringify({sessions: {[SESSION_ID]: {
    session_id: SESSION_ID, project_name: 'Western', working_dir: `.working_dir/${SESSION_ID}`, stage: 'stills_ready', updated_at: '2026-07-17T10:00:00',
  }}}));
  await writeFile(path.join(working, 'render_manifest.json'), JSON.stringify({render_mode: ROOT, image_model: MODEL, style: 'cinematic'}));
  for (const slot of ['0', '1']) {
    const shot = path.join(working, ROOT, 'shots', slot);
    await mkdir(shot, {recursive: true});
    await writeFile(path.join(shot, 'shot_description.json'), JSON.stringify({brief: `Shot ${slot}`}));
  }
  return {repo, working};
}

it('offers the configured model and project style before any media phase has run', async () => {
  const {repo, working} = await fixture();
  await rm(path.join(working, 'render_manifest.json'));
  const sessionsPath = path.join(repo, '.vimax', 'sessions.json');
  const sessions = JSON.parse(await readFile(sessionsPath, 'utf8'));
  sessions.sessions[SESSION_ID].style = 'painted western';
  await writeFile(sessionsPath, JSON.stringify(sessions));
  await mkdir(path.join(repo, 'configs'));
  await writeFile(path.join(repo, 'configs', 'agent.local.yaml'), JSON.stringify({image: {model: MODEL}}));
  const inventory = await readSetReferences(repo, SESSION_ID, ROOT);
  expect(inventory.imageModel).toBe(MODEL);
  expect(inventory.style).toBe('painted western');
});

async function saveSet(repo, inventory, overrides = {}) {
  return updateSetReference(repo, {
    sessionId: SESSION_ID, root: ROOT, id: 'saloon', name: 'Saloon', description: 'Empty timber room, bar on the west wall; warm lanterns from the right.',
    shots: ['0'], revision: inventory.revision, ...overrides,
  });
}

async function writeCandidate(working, candidateId, image = png, overrides = {}) {
  const base = path.join(working, ROOT, 'set_references', 'saloon');
  const descriptor = await readFile(path.join(base, 'set.json'));
  const directory = path.join(base, MODEL_DIR);
  await mkdir(directory, {recursive: true});
  const provenance = Buffer.from(`${JSON.stringify({
    image_model: MODEL,
    style: 'cinematic',
    spec_sha256: createHash('sha256').update(descriptor).digest('hex'),
    candidate_id: candidateId,
    prompt: 'A low timber saloon, lantern light from the right.',
    ...overrides,
  }, null, 2)}\n`);
  const imagePath = path.join(directory, `candidate_${candidateId}.png`);
  const provenancePath = path.join(directory, `candidate_${candidateId}.json`);
  await writeFile(imagePath, image);
  await writeFile(provenancePath, provenance);
  return {image, provenance, imagePath, provenancePath};
}

async function artifactLock(working, relativePath) {
  const bytes = await readFile(path.join(working, relativePath));
  return {path: relativePath, size: bytes.length, sha256: createHash('sha256').update(bytes).digest('hex')};
}

describe('set reference inventory and approval', () => {
  it('creates an assigned set under a fresh revision and rejects stale writes', async () => {
    const {repo} = await fixture();
    const initial = await readSetReferences(repo, SESSION_ID, ROOT);
    expect(initial.unassignedShots).toEqual(['0', '1']);
    const updated = await saveSet(repo, initial);
    expect(updated.sets.map(({id, shots, state}) => ({id, shots, state}))).toEqual([{id: 'saloon', shots: ['0'], state: 'planned'}]);
    expect(updated.unassignedShots).toEqual(['1']);
    await expect(saveSet(repo, initial)).rejects.toMatchObject({statusCode: 409});
  });

  it('distinguishes explicit empty bindings from missing and null bindings', async () => {
    const {repo, working} = await fixture();
    const initial = await readSetReferences(repo, SESSION_ID, ROOT);
    expect(initial.shots.map(({slot, setBindingExplicit}) => [slot, setBindingExplicit])).toEqual([
      ['0', false], ['1', false],
    ]);

    await writeFile(path.join(working, ROOT, 'shots', '0', 'shot_description.json'), JSON.stringify({brief: 'Shot 0', set_ids: null}));
    await writeFile(path.join(working, ROOT, 'shots', '1', 'shot_description.json'), JSON.stringify({brief: 'Shot 1', set_ids: []}));
    const explicitEmpty = await readSetReferences(repo, SESSION_ID, ROOT);
    expect(explicitEmpty.revision).not.toBe(initial.revision);
    expect(explicitEmpty.shots.map(({slot, setIds, setBindingExplicit}) => [slot, setIds, setBindingExplicit])).toEqual([
      ['0', [], false], ['1', [], true],
    ]);
    expect(explicitEmpty.unassignedShots).toEqual(['0']);

    const withDefaults = await saveSet(repo, explicitEmpty);
    expect(withDefaults.shots.map(({slot, setIds, setBindingExplicit}) => [slot, setIds, setBindingExplicit])).toEqual([
      ['0', ['saloon'], false], ['1', [], true],
    ]);
    expect(withDefaults.unassignedShots).toEqual([]);

    await writeFile(path.join(working, ROOT, 'shots', '1', 'shot_description.json'), JSON.stringify({brief: 'Shot 1'}));
    const missingOverride = await readSetReferences(repo, SESSION_ID, ROOT);
    expect(missingOverride.revision).not.toBe(withDefaults.revision);
    expect(missingOverride.shots[1]).toMatchObject({setIds: [], setBindingExplicit: false});
    expect(missingOverride.unassignedShots).toEqual(['1']);
  });

  it('keeps unknown and malformed overrides from becoming valid empty bindings', async () => {
    const {repo, working} = await fixture();
    const shotPath = path.join(working, ROOT, 'shots', '1', 'shot_description.json');
    await writeFile(shotPath, JSON.stringify({brief: 'Shot 1', set_ids: ['missing-set']}));
    const unknown = await readSetReferences(repo, SESSION_ID, ROOT);
    expect(unknown.shots[1]).toMatchObject({setIds: ['missing-set'], setBindingExplicit: true});
    expect(unknown.unassignedShots).toEqual(['0', '1']);

    await writeFile(shotPath, JSON.stringify({brief: 'Shot 1', set_ids: ['bad id']}));
    await expect(readSetReferences(repo, SESSION_ID, ROOT)).rejects.toMatchObject({statusCode: 400});
  });

  it('accepts only the current descriptor/image/provenance and invalidates approval after an edit', async () => {
    const {repo, working} = await fixture();
    let inventory = await readSetReferences(repo, SESSION_ID, ROOT);
    inventory = await saveSet(repo, inventory, {shots: ['0', '1']});
    const descriptorPath = path.join(working, ROOT, 'set_references', 'saloon', 'set.json');
    const descriptor = await readFile(descriptorPath);
    const imageDir = path.join(working, ROOT, 'set_references', 'saloon', MODEL_DIR);
    await mkdir(imageDir, {recursive: true});
    await writeFile(path.join(imageDir, 'reference.png'), png);
    await writeFile(path.join(imageDir, 'provenance.json'), JSON.stringify({image_model: MODEL, style: 'cinematic', spec_sha256: createHash('sha256').update(descriptor).digest('hex')}));
    inventory = await readSetReferences(repo, SESSION_ID, ROOT);
    const accepted = await updateSetReferenceAcceptance(repo, {sessionId: SESSION_ID, root: ROOT, id: 'saloon', revision: inventory.revision, accepted: true});
    expect(accepted.sets[0].state).toBe('accepted');

    const edited = await saveSet(repo, accepted, {description: 'Different room layout.'});
    expect(edited.sets[0].state).toBe('stale');
  });

  it('refuses a truncated PNG even when its provenance matches the current descriptor', async () => {
    const {repo, working} = await fixture();
    let inventory = await saveSet(repo, await readSetReferences(repo, SESSION_ID, ROOT));
    const base = path.join(working, ROOT, 'set_references', 'saloon');
    const imageDir = path.join(base, MODEL_DIR);
    await mkdir(imageDir, {recursive: true});
    await writeFile(path.join(imageDir, 'reference.png'), png.subarray(0, png.length - 5));
    await writeFile(path.join(imageDir, 'provenance.json'), JSON.stringify({
      image_model: MODEL, style: 'cinematic',
      spec_sha256: createHash('sha256').update(await readFile(path.join(base, 'set.json'))).digest('hex'),
    }));
    inventory = await readSetReferences(repo, SESSION_ID, ROOT);
    await expect(updateSetReferenceAcceptance(repo, {
      sessionId: SESSION_ID, root: ROOT, id: 'saloon', revision: inventory.revision, accepted: true,
    })).rejects.toMatchObject({statusCode: 400});
  });

  it('rejects unsafe slot bindings and does not follow a set image symlink', async () => {
    const {repo, working} = await fixture();
    const initial = await readSetReferences(repo, SESSION_ID, ROOT);
    await expect(saveSet(repo, initial, {shots: ['../../outside']})).rejects.toMatchObject({statusCode: 400});
    const inventory = await saveSet(repo, initial);
    const imageDir = path.join(working, ROOT, 'set_references', 'saloon', MODEL_DIR);
    await mkdir(imageDir, {recursive: true});
    const outside = path.join(working, 'outside.png');
    await writeFile(outside, png);
    await import('node:fs/promises').then(({symlink}) => symlink(outside, path.join(imageDir, 'reference.png')));
    await expect(updateSetReferenceAcceptance(repo, {sessionId: SESSION_ID, root: ROOT, id: 'saloon', revision: inventory.revision, accepted: true}))
      .rejects.toMatchObject({statusCode: 409});
  });

  it('uses effective shot overrides when editing descriptor defaults', async () => {
    const {repo, working} = await fixture();
    let inventory = await saveSet(repo, await readSetReferences(repo, SESSION_ID, ROOT), {shots: ['0', '1']});
    await writeFile(path.join(working, ROOT, 'shots', '1', 'shot_description.json'), JSON.stringify({brief: 'Shot 1', set_ids: []}));
    const untouched = {accepted_at: 'reviewed', artifacts: []};
    await writeFile(path.join(working, 'render_acceptance.json'), JSON.stringify({
      [ROOT]: {
        sets: {saloon: {accepted_at: 'approved'}},
        shots: {
          '0': {keyframes: {...untouched}},
          '1': {keyframes: {...untouched}},
        },
      },
    }));
    inventory = await readSetReferences(repo, SESSION_ID, ROOT);
    expect(inventory.shots.map(({slot, setIds}) => [slot, setIds])).toEqual([['0', ['saloon']], ['1', []]]);

    const updated = await saveSet(repo, inventory, {description: 'Revised saloon', shots: ['1']});

    expect(updated.sets[0].shots).toEqual(['1']);
    expect(updated.shots.map(({slot, setIds}) => [slot, setIds])).toEqual([['0', []], ['1', []]]);
    expect(JSON.parse(await readFile(path.join(working, ROOT, 'shots', '1', 'shot_description.json'), 'utf8')).set_ids).toEqual([]);
    const acceptance = JSON.parse(await readFile(path.join(working, 'render_acceptance.json'), 'utf8'))[ROOT];
    expect(acceptance.shots['0'].keyframes.invalidated_at).toBeTruthy();
    expect(acceptance.shots['1']).toEqual({keyframes: untouched});
  });

  it('lists ready, stale, and invalid candidates and rejects stale or busy selections', async () => {
    const {repo, working} = await fixture();
    let inventory = await saveSet(repo, await readSetReferences(repo, SESSION_ID, ROOT));
    const readyId = 'a'.repeat(32);
    const staleId = 'b'.repeat(32);
    const invalidId = 'c'.repeat(32);
    await writeCandidate(working, readyId);
    await writeCandidate(working, staleId, png, {style: 'old-style'});
    await writeCandidate(working, invalidId, png.subarray(0, png.length - 4));

    inventory = await readSetReferences(repo, SESSION_ID, ROOT);

    expect(inventory.sets[0].candidates.map(({id, state}) => [id, state])).toEqual([
      [readyId, 'ready'], [staleId, 'stale'], [invalidId, 'invalid'],
    ]);
    await expect(selectSetReference(repo, {
      sessionId: SESSION_ID, root: ROOT, id: 'saloon', candidateId: staleId, revision: inventory.revision,
    })).rejects.toMatchObject({statusCode: 409});
    await writeCandidate(working, readyId, png, {prompt: 'A changed prompt.'});
    await expect(selectSetReference(repo, {
      sessionId: SESSION_ID, root: ROOT, id: 'saloon', candidateId: readyId, revision: inventory.revision,
    })).rejects.toMatchObject({statusCode: 409});
    const refreshed = await readSetReferences(repo, SESSION_ID, ROOT);
    await expect(selectSetReference(repo, {
      sessionId: SESSION_ID, root: ROOT, id: 'saloon', candidateId: readyId, revision: refreshed.revision,
    }, () => true)).rejects.toMatchObject({statusCode: 409});
  });

  it('copies a candidate transactionally and invalidates bound shots only', async () => {
    const {repo, working} = await fixture();
    let inventory = await saveSet(repo, await readSetReferences(repo, SESSION_ID, ROOT), {shots: ['0', '1']});
    await writeFile(path.join(working, ROOT, 'shots', '1', 'shot_description.json'), JSON.stringify({
      brief: 'Shot 1', set_ids: [], reuse_previous_last_frame: true,
    }));
    await writeFile(path.join(working, ROOT, 'camera_tree.json'), JSON.stringify([{idx: 0, active_shot_idxs: [0, 1]}]));
    const candidateId = 'd'.repeat(32);
    const candidate = await writeCandidate(working, candidateId);
    await writeFile(path.join(working, 'continuity_review.json'), JSON.stringify({root: ROOT, summary: 'reviewed'}));
    const untouched = {accepted_at: 'reviewed', artifacts: []};
    await writeFile(path.join(working, 'render_acceptance.json'), JSON.stringify({
      [ROOT]: {
        sets: {saloon: {accepted_at: 'approved'}},
        shots: {
          '0': {keyframes: {...untouched}, clips: {...untouched}},
          '1': {keyframes: {...untouched}, clips: {...untouched}},
        },
        final_video: {...untouched},
      },
    }));
    inventory = await readSetReferences(repo, SESSION_ID, ROOT);
    expect(inventory.sets[0].shots).toEqual(['0', '1']);
    expect(inventory.shots.map(({slot, setIds}) => [slot, setIds])).toEqual([['0', ['saloon']], ['1', []]]);

    const selected = await selectSetReference(repo, {
      sessionId: SESSION_ID, root: ROOT, id: 'saloon', candidateId, revision: inventory.revision,
    });

    const canonicalDir = path.join(working, ROOT, 'set_references', 'saloon', MODEL_DIR);
    expect(await readFile(path.join(canonicalDir, 'reference.png'))).toEqual(candidate.image);
    expect(await readFile(path.join(canonicalDir, 'provenance.json'))).toEqual(candidate.provenance);
    expect(await readFile(candidate.imagePath)).toEqual(candidate.image);
    expect(await readFile(candidate.provenancePath)).toEqual(candidate.provenance);
    expect(selected.sets[0]).toMatchObject({state: 'rendered', selectedCandidateId: candidateId});
    const acceptance = JSON.parse(await readFile(path.join(working, 'render_acceptance.json'), 'utf8'))[ROOT];
    expect(acceptance.shots['0'].keyframes.invalidated_at).toBeTruthy();
    expect(acceptance.shots['0'].clips.invalidated_at).toBeTruthy();
    expect(acceptance.shots['1'].keyframes).toEqual(untouched);
    expect(acceptance.shots['1'].clips).toEqual(untouched);
    expect(acceptance.final_video.invalidated_at).toBeTruthy();
    expect(JSON.parse(await readFile(path.join(working, 'continuity_review.json'), 'utf8')).set_references_invalidated_at).toBeTruthy();
    const approved = await updateSetReferenceAcceptance(repo, {
      sessionId: SESSION_ID, root: ROOT, id: 'saloon', revision: selected.revision, accepted: true,
    });
    expect(approved.sets[0].state).toBe('accepted');
  });

  it('preserves approval when the same candidate bytes and provenance are selected again', async () => {
    const {repo, working} = await fixture();
    let inventory = await saveSet(repo, await readSetReferences(repo, SESSION_ID, ROOT));
    const candidateId = 'e'.repeat(32);
    const candidate = await writeCandidate(working, candidateId);
    const canonicalDir = path.join(working, ROOT, 'set_references', 'saloon', MODEL_DIR);
    await writeFile(path.join(canonicalDir, 'reference.png'), candidate.image);
    await writeFile(path.join(canonicalDir, 'provenance.json'), candidate.provenance);
    const descriptorRelative = `${ROOT}/set_references/saloon/set.json`;
    const imageRelative = `${ROOT}/set_references/saloon/${MODEL_DIR}/reference.png`;
    const provenanceRelative = `${ROOT}/set_references/saloon/${MODEL_DIR}/provenance.json`;
    await writeFile(path.join(working, 'render_acceptance.json'), JSON.stringify({
      [ROOT]: {sets: {saloon: {
        accepted_at: 'approved',
        artifacts: await Promise.all([descriptorRelative, imageRelative, provenanceRelative].map((file) => artifactLock(working, file))),
      }}},
    }));
    inventory = await readSetReferences(repo, SESSION_ID, ROOT);
    expect(inventory.sets[0]).toMatchObject({state: 'accepted', selectedCandidateId: candidateId});
    const before = await readFile(path.join(working, 'render_acceptance.json'), 'utf8');

    const selected = await selectSetReference(repo, {
      sessionId: SESSION_ID, root: ROOT, id: 'saloon', candidateId, revision: inventory.revision,
    });

    expect(selected.sets[0].state).toBe('accepted');
    expect(await readFile(path.join(working, 'render_acceptance.json'), 'utf8')).toBe(before);
  });

  it('rolls back canonical copies when approval metadata cannot be replaced', async () => {
    const {repo, working} = await fixture();
    await saveSet(repo, await readSetReferences(repo, SESSION_ID, ROOT));
    const candidateId = 'f'.repeat(32);
    const candidate = await writeCandidate(working, candidateId);
    const inventory = await readSetReferences(repo, SESSION_ID, ROOT);
    const acceptanceBefore = await readFile(path.join(working, 'render_acceptance.json'));
    storageTestControl.failAcceptanceWrite = true;

    await expect(selectSetReference(repo, {
      sessionId: SESSION_ID, root: ROOT, id: 'saloon', candidateId, revision: inventory.revision,
    })).rejects.toMatchObject({code: 'EIO'});

    expect(await readFile(path.join(working, 'render_acceptance.json'))).toEqual(acceptanceBefore);
    const canonicalDir = path.join(working, ROOT, 'set_references', 'saloon', MODEL_DIR);
    await expect(readFile(path.join(canonicalDir, 'reference.png'))).rejects.toMatchObject({code: 'ENOENT'});
    await expect(readFile(path.join(canonicalDir, 'provenance.json'))).rejects.toMatchObject({code: 'ENOENT'});
    expect(await readFile(candidate.imagePath)).toEqual(candidate.image);
    expect(await readFile(candidate.provenancePath)).toEqual(candidate.provenance);
  });

  it('applies reviewed pixels as an accepted source import and invalidates bound reviews', async () => {
    const {repo, working} = await fixture();
    let inventory = await saveSet(repo, await readSetReferences(repo, SESSION_ID, ROOT), {shots: ['0', '1']});
    const oldCanonical = path.join(working, ROOT, 'set_references', 'saloon', MODEL_DIR);
    await mkdir(oldCanonical, {recursive: true});
    const descriptor = await readFile(path.join(working, ROOT, 'set_references', 'saloon', 'set.json'));
    await writeFile(path.join(oldCanonical, 'reference.png'), png);
    await writeFile(path.join(oldCanonical, 'provenance.json'), JSON.stringify({
      image_model: MODEL, style: 'cinematic', spec_sha256: createHash('sha256').update(descriptor).digest('hex'),
    }));
    const source = path.join(working, 'uploads', 'western-room.png');
    await mkdir(path.dirname(source), {recursive: true});
    await writeFile(source, png);
    const untouched = {accepted_at: 'reviewed', artifacts: []};
    await writeFile(path.join(working, 'render_acceptance.json'), JSON.stringify({
      [ROOT]: {sets: {saloon: {rejected_at: 'rejected'}}, shots: {
        '0': {keyframes: {...untouched}, clips: {...untouched}},
      }, final_video: {...untouched}},
    }));
    inventory = await readSetReferences(repo, SESSION_ID, ROOT);
    const applied = await applySetReferenceImage(repo, {
      sessionId: SESSION_ID, root: ROOT, revision: inventory.revision, id: 'saloon',
      sourcePath: 'uploads/western-room.png', pngBase64: png.toString('base64'), crop: {x: 1, y: 2, width: 12, height: 9},
    });
    const imagePath = path.join(working, ROOT, 'set_references', 'saloon', MODEL_DIR, 'reference.png');
    const provenance = JSON.parse(await readFile(path.join(path.dirname(imagePath), 'provenance.json'), 'utf8'));
    const acceptance = JSON.parse(await readFile(path.join(working, 'render_acceptance.json'), 'utf8'))[ROOT];
    expect(await readFile(imagePath)).toEqual(png);
    expect(provenance.source_import).toMatchObject({
      source_path: 'uploads/western-room.png', source_sha256: createHash('sha256').update(png).digest('hex'),
      crop: {x: 1, y: 2, width: 12, height: 9},
    });
    expect(applied.sets[0].state).toBe('accepted');
    expect(applied.sets[0].candidates).toHaveLength(1);
    expect(applied.sets[0].candidates[0]).toMatchObject({state: 'ready', prompt: 'Not generated; preserved previous canonical reference.'});
    expect(acceptance.sets.saloon).toMatchObject({accepted_at: expect.any(String), artifacts: expect.arrayContaining([
      expect.objectContaining({path: `${ROOT}/set_references/saloon/${MODEL_DIR}/reference.png`}),
    ])});
    expect(acceptance.sets.saloon.rejected_at).toBeUndefined();
    expect(acceptance.shots['0'].keyframes.invalidated_at).toBeTruthy();
    expect(acceptance.shots['0'].clips.invalidated_at).toBeTruthy();
    expect(acceptance.final_video.invalidated_at).toBeTruthy();
  });

  it('invalidates only bound reviews in global chained mode and retains media', async () => {
    const {repo, working} = await fixture();
    const slots = ['0', '1', '2', '3', '4', '5'];
    for (const slot of slots.slice(2)) {
      const shot = path.join(working, ROOT, 'shots', slot);
      await mkdir(shot, {recursive: true});
      await writeFile(path.join(shot, 'shot_description.json'), JSON.stringify({brief: `Shot ${slot}`}));
    }
    await writeFile(path.join(working, ROOT, 'shots', '3', 'shot_description.json'), JSON.stringify({
      brief: 'Shot 3', reuse_previous_last_frame: false,
    }));
    await writeFile(path.join(working, ROOT, 'shots', '5', 'shot_description.json'), JSON.stringify({
      brief: 'Shot 5', set_ids: [],
    }));
    await writeFile(path.join(working, ROOT, 'camera_tree.json'), JSON.stringify([
      {idx: 0, active_shot_idxs: [4, 1, 0, 3, 2, 5]},
    ]));
    await writeFile(path.join(working, 'frame_continuity.json'), JSON.stringify({mode: 'chained_keyframes'}));
    const retainedMedia = path.join(working, ROOT, 'shots', '3', 'first_frame.png');
    await writeFile(retainedMedia, png);
    let inventory = await saveSet(repo, await readSetReferences(repo, SESSION_ID, ROOT));
    const acceptance = {accepted_at: 'reviewed', artifacts: [{path: 'media.png'}]};
    const shots = Object.fromEntries(slots.map((slot) => [slot, {
      keyframes: {...acceptance}, clips: {...acceptance},
    }]));
    await writeFile(path.join(working, 'render_acceptance.json'), JSON.stringify({
      [ROOT]: {sets: {saloon: {accepted_at: 'reviewed'}}, shots, final_video: {...acceptance}},
    }));
    const candidateId = 'e'.repeat(32);
    await writeCandidate(working, candidateId);
    inventory = await readSetReferences(repo, SESSION_ID, ROOT);
    await selectSetReference(repo, {sessionId: SESSION_ID, root: ROOT, id: 'saloon', candidateId, revision: inventory.revision});
    const updated = JSON.parse(await readFile(path.join(working, 'render_acceptance.json'), 'utf8'))[ROOT];
    expect(updated.shots['0'].keyframes.invalidated_at).toBeTruthy();
    expect(updated.shots['0'].clips.invalidated_at).toBeTruthy();
    for (const slot of ['1', '2', '3', '4', '5']) expect(updated.shots[slot]).toEqual(shots[slot]);
    expect(await readFile(retainedMedia)).toEqual(png);
    expect(updated.final_video.invalidated_at).toBeTruthy();
  });

  it('keeps non-chained set invalidation limited to bound shots', async () => {
    const {repo, working} = await fixture();
    const slots = ['0', '1', '2', '3'];
    for (const slot of slots.slice(2)) {
      const shot = path.join(working, ROOT, 'shots', slot);
      await mkdir(shot, {recursive: true});
      await writeFile(path.join(shot, 'shot_description.json'), JSON.stringify({brief: `Shot ${slot}`}));
    }
    await writeFile(path.join(working, ROOT, 'camera_tree.json'), JSON.stringify([
      {idx: 0, active_shot_idxs: [0, 1, 2, 3]},
    ]));
    let inventory = await saveSet(repo, await readSetReferences(repo, SESSION_ID, ROOT));
    const approval = {accepted_at: 'reviewed', artifacts: []};
    const shots = Object.fromEntries(slots.map((slot) => [slot, {keyframes: {...approval}, clips: {...approval}}]));
    await writeFile(path.join(working, 'render_acceptance.json'), JSON.stringify({[ROOT]: {shots, final_video: {...approval}}}));
    await writeCandidate(working, 'a'.repeat(32));
    inventory = await readSetReferences(repo, SESSION_ID, ROOT);
    await selectSetReference(repo, {sessionId: SESSION_ID, root: ROOT, id: 'saloon', candidateId: 'a'.repeat(32), revision: inventory.revision});
    const updated = JSON.parse(await readFile(path.join(working, 'render_acceptance.json'), 'utf8'))[ROOT];
    expect(updated.shots['0'].keyframes.invalidated_at).toBeTruthy();
    for (const slot of ['1', '2', '3']) expect(updated.shots[slot]).toEqual(shots[slot]);
  });

  it('creates a safely identified unbound set and rejects stale or busy applies', async () => {
    const {repo, working} = await fixture();
    const source = path.join(working, 'source.jpg');
    await writeFile(source, Buffer.from([0xff, 0xd8, 0xff, 0xd9]));
    const initial = await readSetReferences(repo, SESSION_ID, ROOT);
    const apply = (revision, busy) => applySetReferenceImage(repo, {
      sessionId: SESSION_ID, root: ROOT, revision, name: 'New location',
      sourcePath: 'source.jpg', pngBase64: png.toString('base64'),
    }, busy);
    await expect(apply(initial.revision, () => true)).rejects.toMatchObject({statusCode: 409});
    await expect(apply('stale', () => false)).rejects.toMatchObject({statusCode: 409});
    expect((await readSetReferences(repo, SESSION_ID, ROOT)).sets).toHaveLength(0);
    const created = await apply(initial.revision, () => false);
    expect(created.sets).toHaveLength(1);
    expect(created.sets[0]).toMatchObject({name: 'New location', shots: [], state: 'accepted'});
    expect(created.sets[0].id).toMatch(/^[0-9a-f-]{36}$/);
    expect(created.shots.map(({setIds}) => setIds)).toEqual([[], []]);
  });

  it('rolls back imported pixels and a new descriptor if acceptance persistence fails', async () => {
    const {repo, working} = await fixture();
    const source = path.join(working, 'source.png');
    await writeFile(source, png);
    await writeFile(path.join(working, 'render_acceptance.json'), '{}\n');
    const inventory = await readSetReferences(repo, SESSION_ID, ROOT);
    const acceptanceBefore = await readFile(path.join(working, 'render_acceptance.json'));
    storageTestControl.failAcceptanceWrite = true;
    await expect(applySetReferenceImage(repo, {
      sessionId: SESSION_ID, root: ROOT, revision: inventory.revision, name: 'Temporary',
      sourcePath: 'source.png', pngBase64: png.toString('base64'),
    })).rejects.toMatchObject({code: 'EIO'});
    expect(await readFile(path.join(working, 'render_acceptance.json'))).toEqual(acceptanceBefore);
    expect((await readSetReferences(repo, SESSION_ID, ROOT)).sets).toEqual([]);
    await expect(readFile(path.join(working, ROOT, 'set_references'))).rejects.toMatchObject({code: 'ENOENT'});
  });
});
