import {readFile, rename, writeFile} from 'node:fs/promises';
import path from 'node:path';
import {parse, stringify} from 'yaml';
import {credentialProvider} from './provider-credentials.mjs';

const SECTION_FIELDS = {
  llm: ['model_provider', 'model', 'base_url'],
  image: ['model', 'editing_model', 'base_url', 'resolution'],
  video: ['provider', 'model', 'base_url', 'resolution', 'clip_seconds', 'generate_audio'],
  embedding: ['model_provider', 'model', 'base_url'],
  reranker: ['model', 'base_url'],
};

const saveQueues = new Map();

// Eight seconds is the shortest length that carries a spoken line without rushing it.
const DEFAULT_CLIP_SECONDS = 8;

export async function readAgentConfig(repoRoot) {
  const {payload} = await loadConfig(repoRoot);
  return publicConfig(payload, effectiveClipSeconds(payload), effectiveImageResolution(payload));
}

/**
 * The clip settings the film is rendered with: seconds each clip runs for, and the model
 * that renders it.
 *
 * The configured values win over the environment, unlike the model settings: a clip length
 * belongs to the film being made, while a duration exported into a shell or a supervisor's
 * environment cannot be seen from the project at all. One left set to 5 that way cut every
 * clip of a sequence whose dialogue needed eight seconds, and nothing on screen said why.
 * Mirrors `agent_runtime.config.video_clip_seconds`.
 */
export async function readClipSettings(repoRoot, renderedWith = '') {
  const {payload} = await loadConfig(repoRoot);
  const section = payload.video && typeof payload.video === 'object' ? payload.video : {};
  const seconds = effectiveClipSeconds(payload);
  const configured = typeof section.model === 'string' ? section.model.trim() : '';
  // The configured model is what the next clip is rendered with; the manifest's record is
  // what the clips on disk were rendered with, and it is all there is when no config can be
  // read. The configured value wins, because a model change re-renders the clips.
  const resolution = typeof section.resolution === 'string' ? section.resolution.trim() : '';
  return {seconds, model: configured || String(renderedWith || '').trim(), resolution};
}

function effectiveClipSeconds(payload) {
  const section = payload.video && typeof payload.video === 'object' ? payload.video : {};
  return usableSeconds(section.clip_seconds) ?? usableSeconds(process.env.VIMAX_OPENROUTER_VIDEO_DURATION) ?? DEFAULT_CLIP_SECONDS;
}

function usableSeconds(value) {
  const seconds = Number(value);
  return Number.isFinite(seconds) && seconds > 0 ? Math.floor(seconds) : null;
}

export async function saveAgentConfig(repoRoot, input) {
  if (!input || typeof input !== 'object' || !isRecord(input.sections)) {
    throw new Error('Configuration sections are required');
  }

  const updatesBySection = new Map();
  for (const [section, fields] of Object.entries(SECTION_FIELDS)) {
    if (!Object.hasOwn(input.sections, section)) continue;
    const update = input.sections[section];
    if (!isRecord(update)) continue;
    const prepared = {fields: Object.create(null), apiKeyUpdates: [], apiKey: ''};
    for (const field of fields) {
      if (Object.hasOwn(update, field)) {
        prepared.fields[field] = validatedValue(update[field], `${section}.${field}`, 2_048);
      }
    }
    if (Object.hasOwn(update, 'api_key') && typeof update.api_key === 'string' && update.api_key.trim()) {
      prepared.apiKey = validatedValue(update.api_key, `${section}.api_key`, 8_192);
    }
    if (Object.hasOwn(update, 'api_key_updates')) {
      if (!isRecord(update.api_key_updates)) throw new Error(`${section}.api_key_updates must be a mapping`);
      for (const [identity, key] of Object.entries(update.api_key_updates)) {
        if (!identity.trim() || identity !== identity.trim() || identity.length > 2_048) {
          throw new Error(`${section}.api_key_updates contains an invalid provider identity`);
        }
        prepared.apiKeyUpdates.push([
          identity,
          validatedValue(key, `${section}.api_key_updates.${identity}`, 8_192),
        ]);
      }
    }
    updatesBySection.set(section, prepared);
  }

  const configPath = path.resolve(repoRoot, 'configs', 'agent.local.yaml');
  const previous = saveQueues.get(configPath) || Promise.resolve();
  const operation = previous.catch(() => {}).then(() => saveConfigFile(configPath, updatesBySection));
  saveQueues.set(configPath, operation);
  try {
    return await operation;
  } finally {
    if (saveQueues.get(configPath) === operation) saveQueues.delete(configPath);
  }
}

async function saveConfigFile(configPath, updatesBySection) {
  const {payload} = await loadConfigFile(configPath);
  for (const [section, prepared] of updatesBySection) {
    const source = Object.hasOwn(payload, section) && isRecord(payload[section]) ? payload[section] : {};
    const current = Object.assign(Object.create(null), source);
    const previousIdentity = credentialProvider(current);
    const apiKeys = copyApiKeys(current.api_keys);
    const legacyKey = typeof current.api_key === 'string' && current.api_key.trim() ? current.api_key : '';
    if (!Object.hasOwn(current, 'api_keys') && legacyKey) apiKeys[previousIdentity] = legacyKey;

    for (const [field, value] of Object.entries(prepared.fields)) current[field] = value;
    const selectedIdentity = credentialProvider(current);
    for (const [identity, key] of prepared.apiKeyUpdates) {
      if (key) apiKeys[identity] = key;
    }
    if (prepared.apiKey) apiKeys[selectedIdentity] = prepared.apiKey;
    current.api_keys = apiKeys;
    current.api_key = Object.hasOwn(apiKeys, selectedIdentity) ? apiKeys[selectedIdentity] : '';
    payload[section] = current;
  }

  const temporaryPath = `${configPath}.${process.pid}.tmp`;
  await writeFile(temporaryPath, stringify(payload, {lineWidth: 0}), {mode: 0o600});
  await rename(temporaryPath, configPath);
  return publicConfig(payload, effectiveClipSeconds(payload), effectiveImageResolution(payload));
}

function isRecord(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function copyApiKeys(value) {
  const result = Object.create(null);
  if (!isRecord(value)) return result;
  for (const [identity, key] of Object.entries(value)) {
    if (typeof key === 'string' && key.trim()) result[identity] = key;
  }
  return result;
}

async function loadConfig(repoRoot) {
  return loadConfigFile(path.join(repoRoot, 'configs', 'agent.local.yaml'));
}

async function loadConfigFile(configPath) {
  let text = '';
  try {
    text = await readFile(configPath, 'utf8');
  } catch (error) {
    if (error?.code !== 'ENOENT') throw error;
  }
  const payload = text ? parse(text) : {};
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    throw new Error('configs/agent.local.yaml must be a YAML mapping');
  }
  return {payload};
}

function publicConfig(payload, effectiveSeconds, effectiveResolution) {
  const sections = {};
  for (const [section, fields] of Object.entries(SECTION_FIELDS)) {
    const source = Object.hasOwn(payload, section) && isRecord(payload[section]) ? payload[section] : {};
    const result = {};
    for (const field of fields) {
      const value = Object.hasOwn(source, field) ? source[field] : undefined;
      result[field] = typeof value === 'string' ? value : typeof value === 'boolean' || (field === 'clip_seconds' && typeof value === 'number') ? String(value) : '';
    }
    if (section === 'video' && !result.provider) {
      const baseUrl = String(source.base_url || '').toLowerCase();
      result.provider = baseUrl.includes('agnes-ai.com') ? 'agnes'
        : baseUrl.includes('ltx.io') ? 'ltx'
          : baseUrl.includes('yunwu.ai') ? 'yunwu' : 'openrouter';
    }
    const apiKeys = copyApiKeys(source.api_keys);
    const selectedIdentity = credentialProvider(source);
    const hasScopedMap = Object.hasOwn(source, 'api_keys') && isRecord(source.api_keys);
    const legacyKey = typeof source.api_key === 'string' && source.api_key.trim() ? source.api_key : '';
    if (!hasScopedMap && legacyKey) apiKeys[selectedIdentity] = legacyKey;
    result.api_key = '';
    result.has_api_key = Boolean(Object.hasOwn(apiKeys, selectedIdentity) && apiKeys[selectedIdentity].trim());
    result.configured_api_key_providers = Object.keys(apiKeys);
    sections[section] = result;
    if (section === 'image') result.effective_resolution = effectiveResolution;
    if (section === 'image' && !result.editing_model) result.editing_model = 'black-forest-labs/flux-3-image-klein';
    if (section === 'video') result.effective_clip_seconds = String(effectiveSeconds);
  }
  return {sections};
}

function effectiveImageResolution(payload) {
  const source = Object.hasOwn(payload, 'image') && isRecord(payload.image) ? payload.image : {};
  const environment = process.env.VIMAX_IMAGE_RESOLUTION?.trim();
  if (environment) return environment;
  const configured = typeof source.resolution === 'string' ? source.resolution.trim() : '';
  if (configured) return configured;
  return (process.env.VIMAX_IMAGE_MODEL?.trim() || source.model) === 'bytedance-seed/seedream-5-0-flash' ? '1K' : 'auto';
}

function validatedValue(value, label, maxLength) {
  if (typeof value !== 'string') throw new Error(`${label} must be a string`);
  const normalized = value.trim();
  if (normalized.length > maxLength) throw new Error(`${label} is too long`);
  if (label.endsWith('.base_url') && normalized && !/^https?:\/\//i.test(normalized)) {
    throw new Error(`${label} must use http:// or https://`);
  }
  return normalized;
}
