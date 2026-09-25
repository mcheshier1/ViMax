import {readFile, rename, writeFile} from 'node:fs/promises';
import path from 'node:path';
import {parse, stringify} from 'yaml';

const SECTION_FIELDS = {
  llm: ['model_provider', 'model', 'base_url'],
  image: ['model', 'base_url'],
  video: ['model', 'base_url'],
  embedding: ['model_provider', 'model', 'base_url'],
  reranker: ['model', 'base_url'],
};

// Eight seconds is the shortest length that carries a spoken line without rushing it.
const DEFAULT_CLIP_SECONDS = 8;

export async function readAgentConfig(repoRoot) {
  const {payload} = await loadConfig(repoRoot);
  return publicConfig(payload);
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
  const seconds = usableSeconds(section.clip_seconds) ?? usableSeconds(process.env.VIMAX_OPENROUTER_VIDEO_DURATION) ?? DEFAULT_CLIP_SECONDS;
  const configured = typeof section.model === 'string' ? section.model.trim() : '';
  // The configured model is what the next clip is rendered with; the manifest's record is
  // what the clips on disk were rendered with, and it is all there is when no config can be
  // read. The configured value wins, because a model change re-renders the clips.
  return {seconds, model: configured || String(renderedWith || '').trim()};
}

function usableSeconds(value) {
  const seconds = Number(value);
  return Number.isFinite(seconds) && seconds > 0 ? Math.floor(seconds) : null;
}

export async function saveAgentConfig(repoRoot, input) {
  if (!input || typeof input !== 'object' || !input.sections || typeof input.sections !== 'object') {
    throw new Error('Configuration sections are required');
  }
  const {configPath, payload} = await loadConfig(repoRoot);
  for (const [section, fields] of Object.entries(SECTION_FIELDS)) {
    const update = input.sections[section];
    if (!update || typeof update !== 'object') continue;
    const current = payload[section] && typeof payload[section] === 'object' ? payload[section] : {};
    for (const field of fields) {
      if (!(field in update)) continue;
      current[field] = validatedValue(update[field], `${section}.${field}`, 2_048);
    }
    if (typeof update.api_key === 'string' && update.api_key.trim()) {
      current.api_key = validatedValue(update.api_key, `${section}.api_key`, 8_192);
    }
    payload[section] = current;
  }
  const temporaryPath = `${configPath}.${process.pid}.tmp`;
  await writeFile(temporaryPath, stringify(payload, {lineWidth: 0}), {mode: 0o600});
  await rename(temporaryPath, configPath);
  return publicConfig(payload);
}

async function loadConfig(repoRoot) {
  const configPath = path.join(repoRoot, 'configs', 'agent.local.yaml');
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
  return {configPath, payload};
}

function publicConfig(payload) {
  const sections = {};
  for (const [section, fields] of Object.entries(SECTION_FIELDS)) {
    const source = payload[section] && typeof payload[section] === 'object' ? payload[section] : {};
    const result = {};
    for (const field of fields) result[field] = typeof source[field] === 'string' ? source[field] : '';
    result.api_key = '';
    result.has_api_key = Boolean(typeof source.api_key === 'string' && source.api_key.trim());
    sections[section] = result;
  }
  return {sections};
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
