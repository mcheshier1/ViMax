import {mkdtemp, mkdir, readFile, rm, stat, symlink, writeFile} from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {afterEach, describe, expect, it} from 'vitest';
import {listStyleDerivedArtifacts, readProjectMetadata, rerenderPrompt, updateProjectMetadata} from './server-lib.mjs';

const roots = [];
const SESSION_ID = 'session-1';
const STYLE_DERIVED = [
  'script2video/character_portraits/qwen_qwen-image-3',
  'script2video/final_video.mp4',
  'script2video/shots/0/qwen_qwen-image-3',
  'script2video/shots/0/x-ai_grok-imagine-video',
];
const KEPT_ARTIFACTS = [
  'script.txt',
  'characters.json',
  'storyboard.json',
  'shots/0/shot_description.json',
  'shots/0/first_frame_selector_output.json',
];

afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) => rm(root, {recursive: true, force: true})));
});

async function fixture() {
  const root = await mkdtemp(path.join(os.tmpdir(), 'vimax-project-'));
  roots.push(root);
  const working = path.join(root, '.working_dir', SESSION_ID, 'script2video');
  await mkdir(path.join(working, 'character_portraits', 'qwen_qwen-image-3', '0_Cat'), {recursive: true});
  await mkdir(path.join(working, 'shots', '0', 'qwen_qwen-image-3'), {recursive: true});
  await mkdir(path.join(working, 'shots', '0', 'x-ai_grok-imagine-video'), {recursive: true});
  await writeFile(path.join(working, 'character_portraits', 'qwen_qwen-image-3', 'registry.json'), '{}');
  await writeFile(path.join(working, 'character_portraits', 'qwen_qwen-image-3', '0_Cat', 'front.png'), 'portrait');
  await writeFile(path.join(working, 'shots', '0', 'qwen_qwen-image-3', 'first_frame.png'), 'frame');
  await writeFile(path.join(working, 'shots', '0', 'x-ai_grok-imagine-video', 'video.mp4'), 'clip');
  for (const artifact of KEPT_ARTIFACTS) await writeFile(path.join(working, artifact), 'planning');
  await writeFile(path.join(working, 'final_video.mp4'), 'video');

  await mkdir(path.join(root, '.vimax'), {recursive: true});
  await writeFile(path.join(root, '.vimax', 'sessions.json'), JSON.stringify({
    active_session_id: SESSION_ID,
    sessions: {
      [SESSION_ID]: {
        session_id: SESSION_ID,
        project_name: 'Ocean campaign',
        idea: 'A cat sails a boat',
        user_requirement: '16:9, no dialogue',
        style: 'cartoon',
        working_dir: `.working_dir/${SESSION_ID}`,
        stage: 'rendering',
        summary: 'Rendering the clips',
        stale: {frames: false, clips: true},
        custom_field: 'preserved',
        created_at: '2026-07-16T09:00:00',
        updated_at: '2026-07-17T10:00:00',
      },
    },
  }));
  await writeFile(path.join(root, '.working_dir', SESSION_ID, 'render_manifest.json'), JSON.stringify({
    image_model: 'qwen/qwen-image-3',
    video_model: 'x-ai/grok-imagine-video',
    style: 'cartoon',
    created_at: '2026-07-17T10:00:00',
  }, null, 2));
  return {root, working, session: path.join(root, '.working_dir', SESSION_ID), manifest: path.join(root, '.working_dir', SESSION_ID, 'render_manifest.json')};
}

describe('project metadata', () => {
  it('reads the project shape with its manifest and stale flags', async () => {
    const {root} = await fixture();
    expect(await readProjectMetadata(root, SESSION_ID)).toEqual({
      sessionId: SESSION_ID,
      projectName: 'Ocean campaign',
      idea: 'A cat sails a boat',
      userRequirement: '16:9, no dialogue',
      style: 'cartoon',
      stage: 'rendering',
      summary: 'Rendering the clips',
      workingDir: `.working_dir/${SESSION_ID}`,
      manifest: {
        image_model: 'qwen/qwen-image-3',
        video_model: 'x-ai/grok-imagine-video',
        style: 'cartoon',
        created_at: '2026-07-17T10:00:00',
      },
      artifacts: {portraits: 1, shots: 1, frames: 1, clips: 1, finalVideo: true},
    });
    expect(await readProjectMetadata(root, 'unknown-session')).toBeNull();
  });

  it('reports a null manifest when it is missing or malformed', async () => {
    const {root, manifest} = await fixture();
    await writeFile(manifest, '{not json');
    expect((await readProjectMetadata(root, SESSION_ID)).manifest).toBeNull();
    await rm(manifest);
    expect((await readProjectMetadata(root, SESSION_ID)).manifest).toBeNull();
  });

  it('lists only style-derived artifacts that exist on disk', async () => {
    const {root} = await fixture();
    expect(await listStyleDerivedArtifacts(root, SESSION_ID)).toEqual(STYLE_DERIVED);
  });

  it('persists a style change but touches no files without invalidate', async () => {
    const {root, session, manifest} = await fixture();
    const result = await updateProjectMetadata(root, {sessionId: SESSION_ID, style: 'photorealistic cinematic'});

    expect(result.changed).toEqual(['style']);
    expect(result.requiresInvalidation).toBe(true);
    expect(result.invalidated).toEqual(STYLE_DERIVED);
    expect(result.session.style).toBe('photorealistic cinematic');
    expect((await readProjectMetadata(root, SESSION_ID)).style).toBe('photorealistic cinematic');

    for (const artifact of STYLE_DERIVED) {
      await expect(stat(path.join(session, artifact))).resolves.toBeDefined();
    }
    expect(JSON.parse(await readFile(manifest, 'utf8')).style).toBe('cartoon');

    const payload = JSON.parse(await readFile(path.join(root, '.vimax', 'sessions.json'), 'utf8'));
    expect(payload.sessions[SESSION_ID]).toMatchObject({custom_field: 'preserved', project_name: 'Ocean campaign', style: 'photorealistic cinematic'});
  });

  it('removes exactly the style-derived artifacts and repins the manifest on confirmation', async () => {
    const {root, working, session, manifest} = await fixture();
    const pending = await updateProjectMetadata(root, {sessionId: SESSION_ID, style: 'photorealistic cinematic'});
    // The confirm re-posts the same diff, so the style is already persisted.
    const confirmed = await updateProjectMetadata(root, {sessionId: SESSION_ID, style: 'photorealistic cinematic', invalidate: true});

    expect(confirmed.changed).toEqual([]);
    expect(confirmed.requiresInvalidation).toBe(false);
    expect(confirmed.invalidated).toEqual(pending.invalidated);

    for (const artifact of STYLE_DERIVED) {
      await expect(stat(path.join(session, artifact))).rejects.toThrow();
    }
    for (const artifact of KEPT_ARTIFACTS) {
      await expect(stat(path.join(working, artifact))).resolves.toBeDefined();
    }
    expect(JSON.parse(await readFile(manifest, 'utf8'))).toEqual({
      image_model: 'qwen/qwen-image-3',
      video_model: 'x-ai/grok-imagine-video',
      style: 'photorealistic cinematic',
      created_at: '2026-07-17T10:00:00',
    });
    expect((await readProjectMetadata(root, SESSION_ID)).manifest.style).toBe('photorealistic cinematic');
    // The Project page reports render output from disk, so the invalidation is
    // visible there instead of still being described by the stale flags.
    expect((await readProjectMetadata(root, SESSION_ID)).artifacts).toEqual({portraits: 0, shots: 1, frames: 0, clips: 0, finalVideo: false});
  });

  it('never follows a symlink out of the session working dir', async () => {
    const {root, working} = await fixture();
    const outside = await mkdtemp(path.join(os.tmpdir(), 'vimax-outside-'));
    roots.push(outside);
    await writeFile(path.join(outside, 'keep.txt'), 'outside');
    await symlink(outside, path.join(working, 'character_portraits', 'escape-link'), 'dir');

    expect(await listStyleDerivedArtifacts(root, SESSION_ID)).not.toContain('script2video/character_portraits/escape-link');
    const result = await updateProjectMetadata(root, {sessionId: SESSION_ID, invalidate: true});
    expect(result.invalidated).not.toContain('script2video/character_portraits/escape-link');
    expect(await readFile(path.join(outside, 'keep.txt'), 'utf8')).toBe('outside');
  });

  it('reports a no-op update without rewriting the record', async () => {
    const {root} = await fixture();
    const result = await updateProjectMetadata(root, {
      sessionId: SESSION_ID,
      projectName: ' Ocean campaign ',
      idea: 'A cat sails a boat',
      userRequirement: '16:9, no dialogue',
      style: 'cartoon',
    });
    expect(result.changed).toEqual([]);
    expect(result.invalidated).toEqual([]);
    expect(result.requiresInvalidation).toBe(false);

    const payload = JSON.parse(await readFile(path.join(root, '.vimax', 'sessions.json'), 'utf8'));
    expect(payload.sessions[SESSION_ID].updated_at).toBe('2026-07-17T10:00:00');
    expect(await updateProjectMetadata(root, {sessionId: 'unknown-session', style: 'x'})).toBeNull();
  });

  it('lists and removes artifacts written before they were grouped by model', async () => {
    const {root, working} = await fixture();
    const legacy = [
      'character_portraits/0_Cat',
      'shots/0/cache',
      'shots/0/first_frame.png',
      'shots/0/last_frame.png',
      'shots/0/video.mp4',
      'shots/0/transition_video_from_shot_0.mp4',
      'shots/0/new_camera_1.png',
    ];
    await mkdir(path.join(working, 'character_portraits', '0_Cat'), {recursive: true});
    await writeFile(path.join(working, 'character_portraits', '0_Cat', 'front.png'), 'portrait');
    await mkdir(path.join(working, 'shots', '0', 'cache'), {recursive: true});
    await writeFile(path.join(working, 'shots', '0', 'cache', 'segment.mp4'), 'debris');
    for (const name of ['first_frame.png', 'last_frame.png', 'video.mp4', 'transition_video_from_shot_0.mp4', 'new_camera_1.png']) {
      await writeFile(path.join(working, 'shots', '0', name), name);
    }

    const listed = await listStyleDerivedArtifacts(root, SESSION_ID);
    for (const artifact of [...STYLE_DERIVED, ...legacy.map((entry) => `script2video/${entry}`)]) {
      expect(listed).toContain(artifact);
    }
    // Prompts and shot descriptions are render inputs, whatever the layout.
    for (const kept of ['shots/0/shot_description.json', 'shots/0/first_frame_selector_output.json']) {
      expect(listed).not.toContain(`script2video/${kept}`);
    }

    const result = await updateProjectMetadata(root, {sessionId: SESSION_ID, invalidate: true});
    expect(result.invalidated.sort()).toEqual([...listed].sort());
    for (const artifact of [...legacy, 'final_video.mp4']) {
      await expect(stat(path.join(working, artifact))).rejects.toThrow();
    }
    expect(await readFile(path.join(working, 'shots', '0', 'first_frame_selector_output.json'), 'utf8')).toBe('planning');
  });

  it('rejects non-string fields', async () => {
    const {root} = await fixture();
    await expect(updateProjectMetadata(root, {sessionId: SESSION_ID, style: 42})).rejects.toThrow(/must be a string/);
    await expect(updateProjectMetadata(root, {sessionId: SESSION_ID, idea: {}})).rejects.toThrow(/must be a string/);
  });
});

describe('rerender prompt', () => {
  it('quotes the new style and asks for a render that stops at the portraits phase', () => {
    const prompt = rerenderPrompt('photorealistic cinematic live action');
    expect(prompt).toContain('photorealistic cinematic live action');
    expect(prompt).toContain('vimax_render_video');
    expect(prompt).toContain('portraits');
  });

  it('tells the agent to ask for a style instead of inventing one when it is unset', () => {
    const prompt = rerenderPrompt('   ');
    expect(prompt).not.toContain('""');
    expect(prompt).toMatch(/ask the user/i);
  });
});
