import {createHash} from 'node:crypto';
import {mkdtemp, mkdir, readFile, rm, symlink, writeFile} from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {afterEach, describe, expect, it} from 'vitest';
import {approvePortraitCharacter, readPortraitOptions, readRenderAcceptance, selectPortraitCharacter, updateRenderAcceptance} from './server-lib.mjs';

const SESSION_ID = 'session-portrait';
const MODEL = 'qwen/qwen-image-3';
const MODEL_SLUG = 'qwen_qwen-image-3';
const roots = [];
const png = (payload) => Buffer.concat([
  Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
  Buffer.from(payload),
]);

afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) => rm(root, {recursive: true, force: true})));
});

async function fixture({root = 'script2video', characters = [character(0, 'Ada'), character(1, 'Bo')]} = {}) {
  const repo = await mkdtemp(path.join(os.tmpdir(), 'vimax-portrait-selection-'));
  roots.push(repo);
  const working = path.join(repo, '.working_dir', SESSION_ID);
  await mkdir(working, {recursive: true});
  await mkdir(path.join(repo, '.vimax'), {recursive: true});
  await writeFile(path.join(repo, '.vimax', 'sessions.json'), JSON.stringify({
    active_session_id: SESSION_ID,
    sessions: {
      [SESSION_ID]: {
        session_id: SESSION_ID,
        project_name: 'Portrait selection',
        working_dir: `.working_dir/${SESSION_ID}`,
        stage: 'stills_ready',
        updated_at: '2026-09-01T00:00:00',
      },
    },
  }));
  await writeFile(path.join(working, 'render_manifest.json'), JSON.stringify({
    image_model: MODEL,
    video_model: 'kwaivgi/kling-video-o1',
    style: 'noir',
    render_mode: root,
  }));
  await mkdir(path.join(working, root), {recursive: true});
  await writeFile(path.join(working, root, 'characters.json'), JSON.stringify(characters));
  const current = {};
  for (const item of characters.filter((entry) => entry.is_visible)) {
    const directory = path.join(working, root, 'character_portraits', MODEL_SLUG, `${item.idx}_${item.identifier_in_scene.replaceAll(' ', '_')}`);
    const paths = {};
    for (const view of ['front', 'side', 'back']) {
      const relative = path.relative(working, path.join(directory, `${view}.png`)).split(path.sep).join('/');
      await put(working, relative, png(`${item.identifier_in_scene} current ${view}`));
      paths[view] = path.join(working, relative);
    }
    current[item.identifier_in_scene] = Object.fromEntries(Object.entries(paths).map(([view, image]) => [view, {path: image}]));
  }
  const registry = path.join(working, root, 'character_portraits', MODEL_SLUG, 'registry.json');
  await writeFile(registry, JSON.stringify(current));
  return {repo, working, root, characters, registry, current};
}

function character(idx, identifier_in_scene) {
  return {
    idx,
    identifier_in_scene,
    is_visible: true,
    static_features: `${identifier_in_scene} with a salt-weathered face`,
    dynamic_features: null,
  };
}

async function put(working, relative, bytes) {
  const target = path.join(working, relative);
  await mkdir(path.dirname(target), {recursive: true});
  await writeFile(target, bytes);
  return target;
}

function relative(working, absolute) {
  return path.relative(working, absolute).split(path.sep).join('/');
}

async function portraitSet(working, directory, content, prefix = '') {
  const paths = {};
  for (const view of ['front', 'side', 'back']) {
    paths[view] = `${directory}/${prefix}${view}.png`;
    await put(working, paths[view], png(`${content} ${view}`));
  }
  return paths;
}

async function lockArtifacts(working, paths) {
  const artifacts = [];
  for (const file of [...paths].sort()) {
    const bytes = await readFile(path.join(working, file));
    artifacts.push({path: file, size: bytes.length, sha256: createHash('sha256').update(bytes).digest('hex')});
  }
  return artifacts;
}

function setArray(paths) {
  return ['front', 'side', 'back'].map((view) => paths[view]).sort();
}

function stage(payload, name) {
  return payload.stages.find((entry) => entry.stage === name);
}

describe('existing portrait selection and approval', () => {
  it('selects only the clicked REMOVED_ sibling trio, preserves unaffected approvals, and reuses the renderer fingerprint', async () => {
    const {repo, working, root, registry} = await fixture();
    const ada = setArray(await activePathMap(root, 'Ada'));
    const bo = setArray(await activePathMap(root, 'Bo'));
    const source = await portraitSet(working, `${root}/character_portraits/_backups/0_Ada`, 'backup', 'REMOVED_');
    const sourceDirectory = path.join(working, root, 'character_portraits', '_backups', '0_Ada');
    await writeFile(path.join(sourceDirectory, 'front.png'), png('must not mix front'));
    await writeFile(path.join(sourceDirectory, 'side.png'), png('must not mix side'));
    await writeFile(path.join(sourceDirectory, 'back.png'), png('must not mix back'));
    await writeFile(path.join(sourceDirectory, 'prompts_weathered.json'), JSON.stringify({front: 'Saved front prompt'}));

    const frame = `${root}/shots/0/${MODEL_SLUG}/first_frame.png`;
    const finalVideo = `${root}/final_video.mp4`;
    await put(working, frame, png('keep frame'));
    await put(working, finalVideo, Buffer.from('keep final video'));
    const acceptance = {
      [root]: {
        portraits: {accepted_at: 'reviewed', artifacts: await lockArtifacts(working, [...ada, ...bo])},
        portrait_characters: {
          [`${root}::Ada`]: {accepted_at: 'reviewed', artifacts: await lockArtifacts(working, ada)},
          [`${root}::Bo`]: {accepted_at: 'reviewed', artifacts: await lockArtifacts(working, bo)},
        },
        shots: {0: {
          keyframes: {accepted_at: 'reviewed', artifacts: await lockArtifacts(working, [frame])},
          clips: {accepted_at: 'reviewed', artifacts: [{path: `${root}/shots/0/video.mp4`, size: 0, sha256: '0'.repeat(64)}]},
        }},
        final_video: {accepted_at: 'reviewed', artifacts: await lockArtifacts(working, [finalVideo])},
      },
    };
    await writeFile(path.join(working, 'render_acceptance.json'), `${JSON.stringify(acceptance, null, 2)}\n`);

    const clicked = source.side;
    const options = await readPortraitOptions(repo, SESSION_ID, clicked);
    expect(options).toMatchObject({
      root,
      sourcePath: clicked,
      complete: true,
      paths: {
        front: `${root}/character_portraits/_backups/0_Ada/REMOVED_front.png`,
        side: clicked,
        back: `${root}/character_portraits/_backups/0_Ada/REMOVED_back.png`,
      },
      characters: [
        {identifier: 'Ada', idx: 0, scope: root, active: true, accepted: true},
        {identifier: 'Bo', idx: 1, scope: root, active: true, accepted: true},
      ],
    });

    const selected = await selectPortraitCharacter(repo, {
      sessionId: SESSION_ID,
      root,
      scope: root,
      character: 'Ada',
      sourcePath: clicked,
      revision: options.revision,
    });
    const canonical = `${root}/character_portraits/${MODEL_SLUG}/0_Ada`;
    expect(selected.options).toMatchObject({
      sourcePath: `${canonical}/front.png`,
      paths: {
        front: `${canonical}/front.png`,
        side: `${canonical}/side.png`,
        back: `${canonical}/back.png`,
      },
      complete: true,
      characters: [
        {identifier: 'Ada', active: true, accepted: false},
        {identifier: 'Bo', active: true, accepted: true},
      ],
    });
    expect(await readFile(path.join(working, `${canonical}/front.png`))).toEqual(await readFile(path.join(working, source.front)));
    expect(await readFile(path.join(sourceDirectory, 'REMOVED_side.png'))).toEqual(png('backup side'));
    expect(JSON.parse(await readFile(path.join(working, `${canonical}/prompts.json`), 'utf8')))
      .toEqual({front: 'Saved front prompt'});
    const writtenRegistry = JSON.parse(await readFile(registry, 'utf8'));
    expect(writtenRegistry.Ada.front.path).toBe(path.join(working, `${canonical}/front.png`));
    const provenance = JSON.parse(await readFile(path.join(working, `${canonical}/provenance.json`), 'utf8'));
    const fingerprintInput = '{"character": {"dynamic_features": null, "identifier_in_scene": "Ada", "idx": 0, "is_visible": true, "static_features": "Ada with a salt-weathered face"}, "model": "qwen_qwen-image-3", "style": "noir"}';
    expect(provenance).toEqual({
      fingerprint: createHash('sha256').update(fingerprintInput).digest('hex'),
      complete: true,
    });

    const afterSelection = JSON.parse(await readFile(path.join(working, 'render_acceptance.json'), 'utf8'))[root];
    expect(afterSelection.portrait_characters[`${root}::Ada`]).toBeUndefined();
    expect(afterSelection.portrait_characters[`${root}::Bo`]).toEqual(acceptance[root].portrait_characters[`${root}::Bo`]);
    expect(afterSelection.portraits).toBeUndefined();
    expect(afterSelection.shots['0'].keyframes.invalidated_at).toEqual(expect.any(String));
    expect(afterSelection.shots['0'].clips.invalidated_at).toEqual(expect.any(String));
    expect(afterSelection.final_video.invalidated_at).toEqual(expect.any(String));
    expect(await readFile(path.join(working, frame))).toEqual(png('keep frame'));
    expect(await readFile(path.join(working, finalVideo), 'utf8')).toBe('keep final video');

    const approvalOptions = selected.options;
    const approved = await approvePortraitCharacter(repo, {
      sessionId: SESSION_ID,
      root,
      scope: root,
      character: 'Ada',
      sourcePath: approvalOptions.sourcePath,
      revision: approvalOptions.revision,
    });
    const stored = JSON.parse(await readFile(path.join(working, 'render_acceptance.json'), 'utf8'))[root];
    expect(stored.portrait_characters[`${root}::Ada`].artifacts.map((item) => item.path).sort())
      .toEqual([`${canonical}/back.png`, `${canonical}/front.png`, `${canonical}/side.png`]);
    expect(stored.portrait_characters[`${root}::Bo`]).toEqual(acceptance[root].portrait_characters[`${root}::Bo`]);
    expect(stored.portraits.artifacts.map((item) => item.path).sort()).toEqual([
      ...[`${canonical}/back.png`, `${canonical}/front.png`, `${canonical}/side.png`],
      ...bo,
    ].sort());
    expect(approved.options.characters.find((item) => item.identifier === 'Ada').accepted).toBe(true);
  });
  it('leaves approvals unchanged when the exact active bytes are reselected', async () => {
    const {repo, working, root} = await fixture({characters: [character(0, 'Ada')]});
    const paths = setArray(await activePathMap(root, 'Ada'));
    const artifacts = await lockArtifacts(working, paths);
    const acceptance = {
      [root]: {
        portraits: {accepted_at: 'reviewed', artifacts},
        portrait_characters: {
          [`${root}::Ada`]: {accepted_at: 'reviewed', artifacts},
        },
      },
    };
    const acceptancePath = path.join(working, 'render_acceptance.json');
    const originalAcceptance = `${JSON.stringify(acceptance, null, 2)}\n`;
    await writeFile(acceptancePath, originalAcceptance);
    const sourcePath = `${root}/character_portraits/${MODEL_SLUG}/0_Ada/front.png`;
    const options = await readPortraitOptions(repo, SESSION_ID, sourcePath);
    const result = await selectPortraitCharacter(repo, {
      sessionId: SESSION_ID, root, scope: root, character: 'Ada', sourcePath, revision: options.revision,
    });
    expect(result.options.characters).toEqual([{identifier: 'Ada', idx: 0, scope: root, active: true, accepted: true}]);
    expect(await readFile(acceptancePath, 'utf8')).toBe(originalAcceptance);
  });

  it('refuses source image symlinks even when their targets are valid PNGs', async () => {
    const {repo, working, root} = await fixture({characters: [character(0, 'Ada')]});
    const source = await portraitSet(working, `${root}/character_portraits/_backups/0_Ada`, 'source');
    const outside = path.join(repo, 'outside.png');
    const linked = path.join(working, source.front);
    await writeFile(outside, png('outside project'));
    await rm(linked);
    await symlink(outside, linked);
    const options = await readPortraitOptions(repo, SESSION_ID, source.side);
    expect(options.complete).toBe(false);
    await expect(selectPortraitCharacter(repo, {
      sessionId: SESSION_ID, root, scope: root, character: 'Ada', sourcePath: source.side, revision: options.revision,
    })).rejects.toMatchObject({statusCode: 400});
    expect(await readFile(outside)).toEqual(png('outside project'));
  });


  it('rejects a stale revision and incomplete or undecodable source trios without writing', async () => {
    const {repo, working, root} = await fixture({characters: [character(0, 'Ada')]});
    const source = await portraitSet(working, 'script2video/character_portraits/_backups/0_Ada', 'source');
    const clicked = source.side;
    const options = await readPortraitOptions(repo, SESSION_ID, clicked);
    await writeFile(path.join(working, source.front), png('changed after preview'));
    await expect(selectPortraitCharacter(repo, {
      sessionId: SESSION_ID, root, scope: root, character: 'Ada', sourcePath: clicked, revision: options.revision,
    })).rejects.toMatchObject({statusCode: 409});

    await writeFile(path.join(working, source.front), Buffer.from('not a png'));
    const refreshed = await readPortraitOptions(repo, SESSION_ID, clicked);
    expect(refreshed.complete).toBe(false);
    await expect(selectPortraitCharacter(repo, {
      sessionId: SESSION_ID, root, scope: root, character: 'Ada', sourcePath: clicked, revision: refreshed.revision,
    })).rejects.toMatchObject({statusCode: 400});
    expect(JSON.parse(await readFile(path.join(working, 'script2video/character_portraits/qwen_qwen-image-3/registry.json'), 'utf8')).Ada.front.path)
      .toContain('0_Ada/front.png');
  });

  it('uses Idea global portraits first and blocks stage acceptance when a registered scene is missing a visible set', async () => {
    const {repo, working, root} = await fixture({root: 'idea2video', characters: [character(0, 'Ada')]});
    const scene = 'idea2video/scene_0';
    await put(working, `${scene}/characters.json`, JSON.stringify([character(0, 'Ada')]));
    const source = `idea2video/character_portraits/${MODEL_SLUG}/0_Ada/front.png`;
    const options = await readPortraitOptions(repo, SESSION_ID, source);
    expect(options.root).toBe('idea2video');
    expect(options.characters).toEqual([{identifier: 'Ada', idx: 0, scope: 'idea2video', active: true, accepted: false}]);

    const accepted = await updateRenderAcceptance(repo, {
      sessionId: SESSION_ID, root: 'idea2video', stage: 'portraits', accepted: true,
    });
    expect(stage(accepted, 'portraits')).toMatchObject({state: 'accepted', accepted: true});
    await put(working, `${scene}/character_portraits/${MODEL_SLUG}/registry.json`, '{}');
    expect(stage(await readRenderAcceptance(repo, SESSION_ID, 'idea2video'), 'portraits').state).toBe('stale');
    await expect(updateRenderAcceptance(repo, {
      sessionId: SESSION_ID, root: 'idea2video', stage: 'portraits', accepted: true,
    })).rejects.toMatchObject({statusCode: 400});
  });
});

async function activePathMap(root, identifier) {
  const canonical = `${root}/character_portraits/${MODEL_SLUG}/${identifier === 'Ada' ? '0' : '1'}_${identifier}`;
  return Object.fromEntries(['front', 'side', 'back'].map((view) => [view, `${canonical}/${view}.png`]));
}
