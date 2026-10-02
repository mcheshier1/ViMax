import {execFile as execFileCallback} from 'node:child_process';
import {promisify} from 'node:util';
import {mkdtemp, mkdir, readFile, rm, writeFile} from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {afterEach, describe, expect, it} from 'vitest';
import {assembleFilm, readRenderAcceptance, updateRenderAcceptance} from './server-lib.mjs';

const roots = [];
const sessionId = 'assembly-project';
const renderRoot = 'script2video';
const stage = (acceptance, name) => acceptance.stages.find((item) => item.stage === name);

afterEach(async () => {
  delete process.env.VIMAX_FFMPEG_CMD;
  delete process.env.VIMAX_FFPROBE_CMD;
  delete process.env.VIMAX_ASSEMBLY_FAIL;
  await Promise.all(roots.splice(0).map((root) => rm(root, {recursive: true, force: true})));
});

async function fixture() {
  const repo = await mkdtemp(path.join(os.tmpdir(), 'vimax-assembly-'));
  roots.push(repo);
  const working = path.join(repo, '.working_dir', sessionId);
  const rootDir = path.join(working, renderRoot);
  await mkdir(path.join(repo, '.vimax'), {recursive: true});
  await mkdir(rootDir, {recursive: true});
  await writeFile(path.join(repo, '.vimax', 'sessions.json'), JSON.stringify({sessions: {[sessionId]: {
    session_id: sessionId, working_dir: `.working_dir/${sessionId}`, updated_at: '2026-08-01T00:00:00',
  }}}));
  await writeFile(path.join(working, 'render_manifest.json'), JSON.stringify({render_mode: renderRoot, video_model: 'kwaivgi/kling-video-o1'}));
  await writeFile(path.join(rootDir, 'camera_tree.json'), JSON.stringify([{idx: 0, active_shot_idxs: [1, 0]}]));
  for (const shot of ['0', '1']) {
    const directory = path.join(rootDir, 'shots', shot, 'kwaivgi_kling-video-o1');
    await mkdir(directory, {recursive: true});
    await writeFile(path.join(rootDir, 'shots', shot, 'shot_description.json'), '{}');
    await writeFile(path.join(directory, 'video.mp4'), `clip-${shot}`);
  }
  const ffmpeg = path.join(repo, 'fake-ffmpeg');
  const ffprobe = path.join(repo, 'fake-ffprobe');
  await writeFile(ffmpeg, '#!/bin/sh\nif [ "$VIMAX_ASSEMBLY_FAIL" = "1" ]; then exit 9; fi\nfor output do :; done\nprintf assembled > "$output"\n');
  await writeFile(ffprobe, '#!/bin/sh\nprintf \'{"streams":[{"codec_type":"video","duration":"1"}],"format":{"duration":"1"}}\\n\'\n');
  await Promise.all([ffmpeg, ffprobe].map((file) => import('node:fs/promises').then(({chmod}) => chmod(file, 0o755))));
  process.env.VIMAX_FFMPEG_CMD = ffmpeg;
  process.env.VIMAX_FFPROBE_CMD = ffprobe;
  const acceptClip = (shot) => updateRenderAcceptance(repo, {sessionId, root: renderRoot, stage: 'clips', shot, accepted: true});
  return {repo, working, rootDir, acceptClip};
}

describe('film assembly', () => {
  it('requires every active clip to be accepted and current', async () => {
    const {repo, working} = await fixture();
    await expect(assembleFilm(repo, {sessionId, root: renderRoot})).rejects.toMatchObject({statusCode: 409});
    const clip = path.join(working, renderRoot, 'shots', '0', 'kwaivgi_kling-video-o1', 'video.mp4');
    await updateRenderAcceptance(repo, {sessionId, root: renderRoot, stage: 'clips', shot: '0', accepted: true});
    await updateRenderAcceptance(repo, {sessionId, root: renderRoot, stage: 'clips', shot: '1', accepted: true});
    await writeFile(clip, 'changed after approval');
    await expect(assembleFilm(repo, {sessionId, root: renderRoot})).rejects.toMatchObject({statusCode: 409});
    await writeFile(clip, 'clip-0');
    await updateRenderAcceptance(repo, {sessionId, root: renderRoot, stage: 'clips', shot: '0', accepted: true});
    await rm(clip);
    await expect(assembleFilm(repo, {sessionId, root: renderRoot})).rejects.toMatchObject({statusCode: 409});
  });

  it('reuses identical output and preserves approval on failure', async () => {
    const {repo, working, rootDir, acceptClip} = await fixture();
    await acceptClip('0');
    await acceptClip('1');
    const assembled = await assembleFilm(repo, {sessionId, root: renderRoot});
    expect(assembled).toEqual({path: `${renderRoot}/final_video.mp4`, reused: false, shotCount: 2});
    expect(await assembleFilm(repo, {sessionId, root: renderRoot})).toMatchObject({reused: true, shotCount: 2});

    await updateRenderAcceptance(repo, {sessionId, root: renderRoot, stage: 'final_video', accepted: true});
    const finalPath = path.join(rootDir, 'final_video.mp4');
    const previousBytes = await readFile(finalPath);
    await writeFile(path.join(working, renderRoot, 'shots', '0', 'kwaivgi_kling-video-o1', 'video.mp4'), 'new accepted clip bytes');
    await acceptClip('0');
    process.env.VIMAX_ASSEMBLY_FAIL = '1';
    await expect(assembleFilm(repo, {sessionId, root: renderRoot})).rejects.toBeTruthy();
    expect(await readFile(finalPath)).toEqual(previousBytes);
    expect(stage(await readRenderAcceptance(repo, sessionId, renderRoot), 'final_video').slots[0].state).toBe('accepted');
    delete process.env.VIMAX_ASSEMBLY_FAIL;
    await assembleFilm(repo, {sessionId, root: renderRoot});
    expect(stage(await readRenderAcceptance(repo, sessionId, renderRoot), 'final_video').slots[0].state).toBe('stale');
    expect(stage(await readRenderAcceptance(repo, sessionId, renderRoot), 'clips').slots.map((slot) => slot.state)).toEqual(['accepted', 'accepted']);
  });

  it('restores the previous final and approval if publishing metadata fails', async () => {
    const {repo, working, rootDir, acceptClip} = await fixture();
    await acceptClip('0');
    await acceptClip('1');
    await assembleFilm(repo, {sessionId, root: renderRoot});
    await updateRenderAcceptance(repo, {sessionId, root: renderRoot, stage: 'final_video', accepted: true});
    const finalPath = path.join(rootDir, 'final_video.mp4');
    const priorFinal = await readFile(finalPath);
    await writeFile(path.join(rootDir, 'shots', '0', 'kwaivgi_kling-video-o1', 'video.mp4'), 'new accepted input');
    await acceptClip('0');
    const priorApproval = await readFile(path.join(working, 'render_acceptance.json'));
    await rm(`${finalPath}.assembly.json`);
    await mkdir(`${finalPath}.assembly.json`);
    await expect(assembleFilm(repo, {sessionId, root: renderRoot})).rejects.toThrow();
    expect(await readFile(finalPath)).toEqual(priorFinal);
    expect(await readFile(path.join(working, 'render_acceptance.json'))).toEqual(priorApproval);
  });

  it('preserves playback order and audio across different sizes and a silent clip', async () => {
    const {repo, working, rootDir, acceptClip} = await fixture();
    delete process.env.VIMAX_FFMPEG_CMD;
    delete process.env.VIMAX_FFPROBE_CMD;
    const execFile = promisify(execFileCallback);
    const clipPath = (shot) => path.join(working, renderRoot, 'shots', shot, 'kwaivgi_kling-video-o1', 'video.mp4');
    await execFile('ffmpeg', ['-v', 'error', '-y', '-f', 'lavfi', '-i', 'color=c=red:s=320x180:r=24:d=1', '-threads', '1', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', clipPath('0')]);
    await execFile('ffmpeg', ['-v', 'error', '-y', '-f', 'lavfi', '-i', 'color=c=blue:s=640x360:r=30:d=1', '-f', 'lavfi', '-i', 'sine=frequency=440:duration=1', '-threads', '1', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest', clipPath('1')]);
    await acceptClip('0');
    await acceptClip('1');
    await assembleFilm(repo, {sessionId, root: renderRoot});
    const finalPath = path.join(rootDir, 'final_video.mp4');
    const {stdout} = await execFile('ffprobe', ['-v', 'error', '-show_streams', '-show_format', '-of', 'json', finalPath]);
    const media = JSON.parse(stdout);
    expect(Number(media.format.duration)).toBeCloseTo(2, 1);
    expect(media.streams.find((stream) => stream.codec_type === 'video')).toMatchObject({width: 1280, height: 720});
    const pixelAt = async (seconds) => {
      const {stdout: bytes} = await execFile('ffmpeg', ['-v', 'error', '-ss', String(seconds), '-i', finalPath, '-frames:v', '1', '-vf', 'scale=1:1', '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-'], {encoding: 'buffer'});
      return [...bytes];
    };
    const blue = await pixelAt(.25);
    const red = await pixelAt(1.25);
    expect(blue[2]).toBeGreaterThan(200);
    expect(blue[0]).toBeLessThan(20);
    expect(red[0]).toBeGreaterThan(200);
    expect(red[2]).toBeLessThan(20);
    const amplitudeAt = async (seconds) => {
      const {stdout: samples} = await execFile('ffmpeg', ['-v', 'error', '-ss', String(seconds), '-i', finalPath, '-t', '0.1', '-vn', '-ac', '1', '-ar', '8000', '-f', 'f32le', '-'], {encoding: 'buffer'});
      let sum = 0;
      for (let offset = 0; offset < samples.length; offset += 4) sum += samples.readFloatLE(offset) ** 2;
      return Math.sqrt(sum / (samples.length / 4));
    };
    expect(await amplitudeAt(.25)).toBeGreaterThan(.03);
    expect(await amplitudeAt(1.25)).toBeLessThan(.001);
  }, 30_000);
});
