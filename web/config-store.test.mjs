import {mkdtemp, mkdir, readFile, rm, writeFile} from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {afterEach, describe, expect, it} from 'vitest';
import {parse, stringify} from 'yaml';
import {readAgentConfig, readClipSettings, saveAgentConfig} from './config-store.mjs';
import {credentialProvider} from './provider-credentials.mjs';

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

const PROVIDER_CATEGORIES = [
  {name: 'llm', providerField: 'model_provider'},
  {name: 'image'},
  {name: 'video', providerField: 'provider'},
  {name: 'embedding', providerField: 'model_provider'},
  {name: 'reranker'},
];

function providerFields(category, provider, model = `${provider}-${category.name}`) {
  const fields = {
    model,
    base_url: provider === 'agnes' ? 'https://apihub.agnes-ai.com/v1' : 'https://openrouter.ai/api/v1',
  };
  if (category.providerField) fields[category.providerField] = provider;
  return fields;
}

async function writeConfig(root, payload) {
  await writeFile(path.join(root, 'configs', 'agent.local.yaml'), stringify(payload));
}

async function readStoredConfig(root) {
  return parse(await readFile(path.join(root, 'configs', 'agent.local.yaml'), 'utf8'));
}

async function writeLegacyProvider(root, category) {
  await writeConfig(root, {
    [category.name]: {
      ...providerFields(category, 'openrouter'),
      api_key: 'openrouter-secret',
    },
  });
}


describe('agent config store', () => {
  it('reads the clip length from the config, which outranks the environment', async () => {
    const root = await fixture();
    await writeFile(path.join(root, 'configs', 'agent.local.yaml'), ['video:', '  model: some/video-model', '  clip_seconds: 8', ''].join('\n'));
    // A duration exported into the environment is invisible from the project and set every
    // clip of a sequence to 5 seconds while its dialogue needed 8, so the file wins.
    process.env.VIMAX_OPENROUTER_VIDEO_DURATION = '5';
    try {
      expect(await readClipSettings(root)).toEqual({seconds: 8, model: 'some/video-model', resolution: ''});
    } finally {
      delete process.env.VIMAX_OPENROUTER_VIDEO_DURATION;
    }
  });

  it('falls back to the environment, then to eight seconds', async () => {
    const root = await fixture();
    process.env.VIMAX_OPENROUTER_VIDEO_DURATION = '5';
    try {
      expect((await readClipSettings(root)).seconds).toBe(5);
      expect((await readAgentConfig(root)).sections.video).toMatchObject({clip_seconds: '', effective_clip_seconds: '5'});
    } finally {
      delete process.env.VIMAX_OPENROUTER_VIDEO_DURATION;
    }
    expect((await readClipSettings(root)).seconds).toBe(8);
    expect((await readAgentConfig(root)).sections.video).toMatchObject({clip_seconds: '', effective_clip_seconds: '8'});
  });

  it('keeps a clip length through a save from the settings page', async () => {
    const root = await fixture();
    await writeFile(path.join(root, 'configs', 'agent.local.yaml'), ['video:', '  model: some/video-model', '  clip_seconds: 8', ''].join('\n'));

    await saveAgentConfig(root, {sections: {video: {model: 'other/video-model'}}});

    expect((await readClipSettings(root)).seconds).toBe(8);
  });
  it('keeps editing and generation model choices independent across partial and full saves', async () => {
    const root = await fixture();
    await writeConfig(root, {
      image: {
        model: 'bytedance-seed/seedream-5-0-flash',
        base_url: 'https://openrouter.ai/api/v1',
        api_keys: {openrouter: 'image-secret'},
      },
    });
    const initial = await readAgentConfig(root);
    expect(initial.sections.image.editing_model).toBe('black-forest-labs/flux-3-image-klein');
    await saveAgentConfig(root, initial);
    await saveAgentConfig(root, {sections: {image: {editing_model: 'custom/editor'}}});
    expect((await readStoredConfig(root)).image).toMatchObject({
      model: 'bytedance-seed/seedream-5-0-flash',
      editing_model: 'custom/editor',
      api_keys: {openrouter: 'image-secret'},
    });
    await saveAgentConfig(root, {sections: {image: {model: 'qwen/qwen-image-3'}}});
    expect((await readAgentConfig(root)).sections.image).toMatchObject({
      model: 'qwen/qwen-image-3',
      editing_model: 'custom/editor',
      has_api_key: true,
    });
  });


  it('round-trips image resolution choices without changing video resolution', async () => {
    const root = await fixture();
    const envResolution = process.env.VIMAX_IMAGE_RESOLUTION;
    delete process.env.VIMAX_IMAGE_RESOLUTION;
    await writeConfig(root, {
      image: {
        model: 'bytedance-seed/seedream-5-0-flash',
        base_url: 'https://openrouter.ai/api/v1',
        api_keys: {openrouter: 'openrouter-secret', agnes: 'agnes-secret'},
      },
      video: {model: 'some/video-model', resolution: '480p'},
    });
    try {
      expect((await readAgentConfig(root)).sections.image).toMatchObject({resolution: '', effective_resolution: '1K'});
      for (const resolution of ['1K', '2K', 'auto']) {
        await saveAgentConfig(root, {sections: {image: {resolution}}});
        expect((await readAgentConfig(root)).sections.image).toMatchObject({resolution, effective_resolution: resolution});
        expect((await readAgentConfig(root)).sections.video.resolution).toBe('480p');
        expect((await readStoredConfig(root)).image.resolution).toBe(resolution);
        expect((await readStoredConfig(root)).image.api_keys).toEqual({openrouter: 'openrouter-secret', agnes: 'agnes-secret'});
      }
    } finally {
      if (envResolution === undefined) delete process.env.VIMAX_IMAGE_RESOLUTION;
      else process.env.VIMAX_IMAGE_RESOLUTION = envResolution;
    }
  });

  it('uses the image resolution environment override ahead of file and Seedream defaults', async () => {
    const root = await fixture();
    const envResolution = process.env.VIMAX_IMAGE_RESOLUTION;
    await writeConfig(root, {
      image: {model: 'bytedance-seed/seedream-5-0-flash', base_url: 'https://openrouter.ai/api/v1', resolution: '2K'},
    });
    process.env.VIMAX_IMAGE_RESOLUTION = 'auto';
    try {
      expect((await readAgentConfig(root)).sections.image).toMatchObject({resolution: '2K', effective_resolution: 'auto'});
    } finally {
      if (envResolution === undefined) delete process.env.VIMAX_IMAGE_RESOLUTION;
      else process.env.VIMAX_IMAGE_RESOLUTION = envResolution;
    }
  });

  it('preserves a numeric clip duration when the full settings form saves an image provider change', async () => {
    const root = await fixture();
    await writeFile(path.join(root, 'configs', 'agent.local.yaml'), ['video:', '  model: some/video-model', '  clip_seconds: 8', ''].join('\n'));
    const config = await readAgentConfig(root);
    config.sections.image.model = 'agnes-image-2.5-flash';
    config.sections.image.base_url = 'https://apihub.agnes-ai.com/v1';
    process.env.VIMAX_OPENROUTER_VIDEO_DURATION = '5';
    try {
      await saveAgentConfig(root, config);
      expect(await readClipSettings(root)).toMatchObject({seconds: 8, model: 'some/video-model'});
      expect((await readAgentConfig(root)).sections.image.model).toBe('agnes-image-2.5-flash');
    } finally {
      delete process.env.VIMAX_OPENROUTER_VIDEO_DURATION;
    }
  });

  it('migrates and preserves distinct provider credentials for every settings category', async () => {
    for (const category of PROVIDER_CATEGORIES) {
      const root = await fixture();
      await writeLegacyProvider(root, category);

      const beforeSave = await readAgentConfig(root);
      expect(beforeSave.sections[category.name]).toMatchObject({
        api_key: '',
        has_api_key: true,
        configured_api_key_providers: ['openrouter'],
      });

      const switchedToAgnes = await saveAgentConfig(root, {
        sections: {[category.name]: {...providerFields(category, 'agnes'), api_key: 'agnes-secret'}},
      });
      expect(switchedToAgnes.sections[category.name]).toMatchObject({
        api_key: '',
        has_api_key: true,
        configured_api_key_providers: ['openrouter', 'agnes'],
      });
      expect(JSON.stringify(switchedToAgnes)).not.toContain('agnes-secret');
      expect(JSON.stringify(switchedToAgnes)).not.toContain('openrouter-secret');

      const blankSave = await saveAgentConfig(root, {
        sections: {[category.name]: {...providerFields(category, 'agnes'), api_key: '   '}},
      });
      expect(blankSave.sections[category.name].has_api_key).toBe(true);

      const returnedToOpenRouter = await saveAgentConfig(root, {
        sections: {[category.name]: {...providerFields(category, 'openrouter'), api_key: ''}},
      });
      expect(returnedToOpenRouter.sections[category.name]).toMatchObject({
        api_key: '',
        has_api_key: true,
        configured_api_key_providers: ['openrouter', 'agnes'],
      });
      const reloaded = await readAgentConfig(root);
      expect(reloaded.sections[category.name].has_api_key).toBe(true);
      expect(JSON.stringify(reloaded)).not.toContain('agnes-secret');
      expect(JSON.stringify(reloaded)).not.toContain('openrouter-secret');

      const modelOnlyChange = await saveAgentConfig(root, {
        sections: {[category.name]: {model: 'changed-within-openrouter'}},
      });
      expect(modelOnlyChange.sections[category.name]).toMatchObject({
        model: 'changed-within-openrouter',
        has_api_key: true,
      });
      const backToAgnes = await saveAgentConfig(root, {
        sections: {[category.name]: {...providerFields(category, 'agnes'), api_key: ''}},
      });
      expect(backToAgnes.sections[category.name].has_api_key).toBe(true);

      const stored = await readStoredConfig(root);
      expect(stored[category.name].api_keys.openrouter).toBe('openrouter-secret');
      expect(stored[category.name].api_keys.agnes).toBe('agnes-secret');
    }
  });

  it('keeps credentials scoped to their settings category', async () => {
    const root = await fixture();
    await writeConfig(root, {});
    for (const provider of ['openrouter', 'agnes']) {
      const sections = Object.fromEntries(PROVIDER_CATEGORIES.map((category) => [
        category.name,
        {...providerFields(category, provider), api_key: `${category.name}-${provider}-secret`},
      ]));
      await saveAgentConfig(root, {sections});
    }
    const sections = Object.fromEntries(PROVIDER_CATEGORIES.map((category) => [
      category.name,
      {...providerFields(category, 'openrouter'), api_key: ''},
    ]));
    const response = await saveAgentConfig(root, {sections});
    for (const category of PROVIDER_CATEGORIES) {
      expect(response.sections[category.name].has_api_key).toBe(true);
    }
    const stored = await readStoredConfig(root);
    for (const category of PROVIDER_CATEGORIES) {
      expect(stored[category.name].api_keys.openrouter).toBe(`${category.name}-openrouter-secret`);
      expect(stored[category.name].api_keys.agnes).toBe(`${category.name}-agnes-secret`);
    }
  });

  it('serializes concurrent saves so both provider credential banks survive', async () => {
    const root = await fixture();
    await writeConfig(root, {});
    await Promise.all([
      saveAgentConfig(root, {sections: {llm: {
        base_url: 'https://openrouter.ai/api/v1',
        api_key: 'concurrent-openrouter-secret',
      }}}),
      saveAgentConfig(root, {sections: {llm: {
        base_url: 'https://apihub.agnes-ai.com/v1',
        api_key: 'concurrent-agnes-secret',
      }}}),
    ]);

    const stored = await readStoredConfig(root);
    expect(stored.llm.api_keys.openrouter).toBe('concurrent-openrouter-secret');
    expect(stored.llm.api_keys.agnes).toBe('concurrent-agnes-secret');
    const openrouter = await saveAgentConfig(root, {sections: {llm: {base_url: 'https://openrouter.ai/api/v1'}}});
    expect(openrouter.sections.llm.has_api_key).toBe(true);
    const agnes = await saveAgentConfig(root, {sections: {llm: {base_url: 'https://apihub.agnes-ai.com/v1'}}});
    expect(agnes.sections.llm.has_api_key).toBe(true);
  });

  it('isolates unknown endpoint credentials by origin and path but ignores query, hash, and userinfo', async () => {
    const root = await fixture();
    await writeConfig(root, {});
    const endpoint = 'https://user:password@custom.example:8443/v1///?token=one#first';
    const first = await saveAgentConfig(root, {sections: {llm: {base_url: endpoint, api_key: 'custom-secret'}}});
    expect(first.sections.llm.has_api_key).toBe(true);

    const otherPath = await saveAgentConfig(root, {
      sections: {llm: {base_url: 'https://custom.example:8443/v2?token=two#second'}},
    });
    expect(otherPath.sections.llm.has_api_key).toBe(false);
    expect(otherPath.sections.llm.configured_api_key_providers).toEqual(['endpoint:https://custom.example:8443/v1']);

    const returned = await saveAgentConfig(root, {
      sections: {llm: {base_url: 'https://different-user:other@custom.example:8443/v1/?token=three#third'}},
    });
    expect(returned.sections.llm.has_api_key).toBe(true);
    expect(JSON.stringify(returned)).not.toContain('custom-secret');
  });

  it('updates all supplied key drafts, ignores inherited map keys, and rejects invalid updates before writing', async () => {
    const root = await fixture();
    await writeConfig(root, {});
    const drafts = Object.create({inherited: 'inherited-secret'});
    Object.defineProperty(drafts, '__proto__', {value: 'prototype-secret', enumerable: true});
    drafts.constructor = 'constructor-secret';
    drafts.openrouter = 'openrouter-draft-secret';
    drafts.agnes = 'agnes-draft-secret';
    const saved = await saveAgentConfig(root, {
      sections: {llm: {base_url: 'https://apihub.agnes-ai.com/v1', api_key_updates: drafts}},
    });
    expect(saved.sections.llm.configured_api_key_providers).toEqual(expect.arrayContaining(['__proto__', 'constructor', 'openrouter', 'agnes']));
    expect(saved.sections.llm.has_api_key).toBe(true);
    expect(JSON.stringify(saved)).not.toContain('prototype-secret');
    expect(JSON.stringify(saved)).not.toContain('constructor-secret');
    expect(JSON.stringify(saved)).not.toContain('openrouter-draft-secret');
    expect(JSON.stringify(saved)).not.toContain('agnes-draft-secret');
    const stored = await readStoredConfig(root);
    expect(stored.llm.api_keys.openrouter).toBe('openrouter-draft-secret');
    expect(stored.llm.api_keys.agnes).toBe('agnes-draft-secret');
    expect(stored.llm.api_keys.inherited).toBeUndefined();
    const openrouter = await saveAgentConfig(root, {sections: {llm: {base_url: 'https://openrouter.ai/api/v1'}}});
    expect(openrouter.sections.llm.has_api_key).toBe(true);
    const agnes = await saveAgentConfig(root, {sections: {llm: {base_url: 'https://apihub.agnes-ai.com/v1'}}});
    expect(agnes.sections.llm.has_api_key).toBe(true);

    const before = await readAgentConfig(root);
    await expect(saveAgentConfig(root, {
      sections: {llm: {model: 'must-not-be-written', api_key_updates: {valid: 'secret', invalid: 42}}},
    })).rejects.toThrow(/must be a string/);
    expect(await readAgentConfig(root)).toEqual(before);
  });

  it('uses provider domains and safe endpoint fallbacks for credential identity', () => {
    expect(credentialProvider({base_url: 'https://proxy.openrouter.ai/v1'})).toBe('openrouter');
    expect(credentialProvider({base_url: 'https://api.agnes-ai.com'})).toBe('agnes');
    expect(credentialProvider({base_url: 'https://api.yunwu.ai'})).toBe('yunwu');
    expect(credentialProvider({base_url: 'https://api.ltx.io'})).toBe('ltx');
    expect(credentialProvider({base_url: 'https://api.openai.com'})).toBe('openai');
    expect(credentialProvider({base_url: 'https://api.anthropic.com'})).toBe('anthropic');
    expect(credentialProvider({base_url: 'https://api.cohere.ai'})).toBe('cohere');
    expect(credentialProvider({base_url: 'https://api.jina.ai'})).toBe('jina');
    expect(credentialProvider({base_url: 'https://u:p@custom.example/v1///?token=x#h'})).toBe('endpoint:https://custom.example/v1');
    expect(credentialProvider({base_url: 'https://custom.example/v1/../v2'})).toBe('endpoint:https://custom.example/v1/../v2');
    expect(credentialProvider({base_url: 'https://custom.example:443/v1'})).toBe('endpoint:https://custom.example/v1');
    expect(credentialProvider({base_url: 'not a URL', model_provider: ' OpenAI '})).toBe('provider:openai');
    expect(credentialProvider({provider: ' Local Provider '})).toBe('provider:local provider');
    expect(credentialProvider({})).toBe('provider:default');
  });

  it('never returns stored API keys', async () => {
    const root = await fixture();
    const config = await readAgentConfig(root);
    expect(config.sections.llm).toMatchObject({
      model: 'existing-model',
      api_key: '',
      has_api_key: true,
      configured_api_key_providers: ['endpoint:https://example.test/v1'],
    });
    expect(JSON.stringify(config)).not.toContain('secret-value');
  });

  it('persists video provider and generation options without exposing its key', async () => {
    const root = await fixture();
    await saveAgentConfig(root, {sections: {video: {
      provider: 'ltx', model: 'ltx-2-5-fast', base_url: 'https://api.ltx.io',
      resolution: '1440p', clip_seconds: '10', generate_audio: 'false', api_key: 'video-secret',
    }}});

    const config = await readAgentConfig(root);
    expect(config.sections.video).toMatchObject({
      provider: 'ltx', model: 'ltx-2-5-fast', resolution: '1440p',
      clip_seconds: '10', generate_audio: 'false', api_key: '', has_api_key: true,
      configured_api_key_providers: ['ltx'],
    });
    expect(JSON.stringify(config)).not.toContain('video-secret');
    expect(await readClipSettings(root)).toMatchObject({seconds: 10, model: 'ltx-2-5-fast'});
  });

  it('does not migrate a stale active-key mirror into an already scoped provider bank', async () => {
    const root = await fixture();
    const configPath = path.join(root, 'configs', 'agent.local.yaml');
    await writeFile(configPath, stringify({
      image: {
        model: 'agnes-image-2.5-flash',
        base_url: 'https://apihub.agnes-ai.com/v1',
        api_key: 'router-secret',
        api_keys: {openrouter: 'router-secret'},
      },
    }));
    expect((await readAgentConfig(root)).sections.image.has_api_key).toBe(false);
    await saveAgentConfig(root, {sections: {image: {api_key: ''}}});
    expect((await readAgentConfig(root)).sections.image.has_api_key).toBe(false);
    const saved = parse(await readFile(configPath, 'utf8'));
    expect(saved.image.api_key).toBe('');
    expect(saved.image.api_keys).toEqual({openrouter: 'router-secret'});
  });

  it('rejects invalid base URLs', async () => {
    const root = await fixture();
    await expect(saveAgentConfig(root, {sections: {llm: {base_url: 'file:///tmp/key'}}})).rejects.toThrow(/http/);
  });
});
