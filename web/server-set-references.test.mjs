import {createHash} from 'node:crypto';
import {mkdtemp, mkdir, readFile, rm, writeFile} from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {afterEach, describe, expect, it, vi} from 'vitest';
import {readSetReferences, selectSetReference, updateSetReference, updateSetReferenceAcceptance} from './server-lib.mjs';

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

  it('copies a candidate transactionally and invalidates bound shots plus linked successors', async () => {
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
    expect(acceptance.shots['1'].keyframes).toMatchObject({accepted_at: 'reviewed', invalidated_at: expect.any(String)});
    expect(acceptance.shots['1'].clips).toMatchObject({accepted_at: 'reviewed', invalidated_at: expect.any(String)});
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
});
