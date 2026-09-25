import {mkdtemp, mkdir, readFile, rm, writeFile} from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {afterEach, describe, expect, it} from 'vitest';
import {readAgentConfig, readClipSettings, saveAgentConfig} from './config-store.mjs';

const roots = [];

afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) => rm(root, {recursive: true, force: true})));
});

async function fixture() {
  const root = await mkdtemp(path.join(os.tmpdir(), 'vimax-config-'));
  roots.push(root);
  await mkdir(path.join(root, 'configs'), {recursive: true});
  await writeFile(path.join(root, 'configs', 'agent.local.yaml'), [
    'llm:',
    '  model_provider: openai',
    '  model: existing-model',
    '  base_url: https://example.test/v1',
    '  api_key: secret-value',
    '',
  ].join('\n'));
  return root;
}

describe('agent config store', () => {
  it('reads the clip length from the config, which outranks the environment', async () => {
    const root = await fixture();
    await writeFile(path.join(root, 'configs', 'agent.local.yaml'), ['video:', '  model: some/video-model', '  clip_seconds: 8', ''].join('\n'));
    // A duration exported into the environment is invisible from the project and set every
    // clip of a sequence to 5 seconds while its dialogue needed 8, so the file wins.
    process.env.VIMAX_OPENROUTER_VIDEO_DURATION = '5';
    try {
      expect(await readClipSettings(root)).toEqual({seconds: 8, model: 'some/video-model'});
    } finally {
      delete process.env.VIMAX_OPENROUTER_VIDEO_DURATION;
    }
  });

  it('falls back to the environment, then to eight seconds', async () => {
    const root = await fixture();
    process.env.VIMAX_OPENROUTER_VIDEO_DURATION = '10';
    try {
      expect((await readClipSettings(root)).seconds).toBe(10);
    } finally {
      delete process.env.VIMAX_OPENROUTER_VIDEO_DURATION;
    }
    expect((await readClipSettings(root)).seconds).toBe(8);
  });

  it('keeps a clip length through a save from the settings page', async () => {
    const root = await fixture();
    await writeFile(path.join(root, 'configs', 'agent.local.yaml'), ['video:', '  model: some/video-model', '  clip_seconds: 8', ''].join('\n'));

    await saveAgentConfig(root, {sections: {video: {model: 'other/video-model'}}});

    expect((await readClipSettings(root)).seconds).toBe(8);
  });

  it('never returns stored API keys', async () => {
    const root = await fixture();
    const config = await readAgentConfig(root);
    expect(config.sections.llm).toMatchObject({model: 'existing-model', api_key: '', has_api_key: true});
    expect(JSON.stringify(config)).not.toContain('secret-value');
  });

  it('keeps a stored key when a blank key is saved', async () => {
    const root = await fixture();
    await saveAgentConfig(root, {sections: {llm: {model: 'new-model', api_key: ''}}});
    const saved = await readFile(path.join(root, 'configs', 'agent.local.yaml'), 'utf8');
    expect(saved).toContain('model: new-model');
    expect(saved).toContain('api_key: secret-value');
  });

  it('rejects invalid base URLs', async () => {
    const root = await fixture();
    await expect(saveAgentConfig(root, {sections: {llm: {base_url: 'file:///tmp/key'}}})).rejects.toThrow(/http/);
  });
});
