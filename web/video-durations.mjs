import DURATION_CATALOG from '../configs/video-duration-models.json' with {type: 'json'};

const DEFAULT_DURATION_SECONDS = 8;
const KLING_O1_MODEL = 'kwaivgi/kling-video-o1';

/**
 * The active model's duration rules and the saved maximum for this provider/model pair.
 * A configured maximum enables explicit shot overrides; the global default never needs one.
 */
export function videoDurationPolicy(videoSection = {}) {
  const section = isRecord(videoSection) ? videoSection : {};
  const provider = selectedProvider(section);
  const model = typeof section.model === 'string' ? section.model.trim() : '';
  const key = `${provider}:${model}`;
  const supported = catalogDurations(section, key);
  const modelDefault = defaultForModel(key, supported);
  const explicitDefault = section.clip_seconds !== undefined && section.clip_seconds !== null && String(section.clip_seconds).trim() !== '';
  const inheritedDefault = positiveWholeSeconds(section.effective_clip_seconds);
  const defaultSeconds = explicitDefault
    ? Number(section.clip_seconds)
    : inheritedDefault !== null && (supported === null || supported.includes(inheritedDefault)) ? inheritedDefault : modelDefault;
  const registry = isRecord(section.shot_duration_limits) ? section.shot_duration_limits : {};
  const hasSavedMaximum = Object.hasOwn(registry, key);
  const savedMaximum = hasSavedMaximum ? registry[key] : undefined;
  const maximumIsValid = Number.isSafeInteger(savedMaximum) && savedMaximum > 0
    && savedMaximum >= defaultSeconds
    && (supported === null || supported.includes(savedMaximum));
  const configured = hasSavedMaximum && maximumIsValid;

  if (supported !== null) {
    const allowedSeconds = configured ? supported.filter((seconds) => seconds <= savedMaximum) : supported;
    return {
      provider,
      model,
      configured,
      defaultSeconds,
      minSeconds: allowedSeconds[0],
      maxSeconds: allowedSeconds.at(-1),
      allowedSeconds,
    };
  }

  return {
    provider,
    model,
    configured,
    defaultSeconds,
    minSeconds: 1,
    maxSeconds: configured ? savedMaximum : defaultSeconds,
    allowedSeconds: null,
  };
}

/** Return an empty string when an explicit shot duration is permitted. */
export function validateShotDuration(seconds, policy) {
  if (!Number.isSafeInteger(seconds) || seconds <= 0) return 'Shot duration must be a positive whole number of seconds.';
  if (!isRecord(policy) || policy.configured !== true) {
    return 'Configure a maximum duration for the active video model in Settings before setting a shot duration.';
  }
  if (seconds > policy.maxSeconds) return `Shot duration must not exceed the configured maximum of ${policy.maxSeconds} seconds.`;
  if (Array.isArray(policy.allowedSeconds) && !policy.allowedSeconds.includes(seconds)) {
    return `Shot duration must be one of the supported values: ${policy.allowedSeconds.join(', ')} seconds.`;
  }
  return '';
}

/** Validate a model maximum before persisting it; max must include the default duration. */
export function validateVideoDurationLimit(seconds, videoSection = {}) {
  if (!Number.isSafeInteger(seconds) || seconds <= 0) {
    return 'Maximum shot duration must be a positive whole number of seconds.';
  }
  const section = isRecord(videoSection) ? videoSection : {};
  const provider = selectedProvider(section);
  const model = typeof section.model === 'string' ? section.model.trim() : '';
  const key = `${provider}:${model}`;
  const supported = catalogDurations(section, key);
  const policy = videoDurationPolicy(section);
  if (seconds < policy.defaultSeconds) {
    return `Maximum shot duration must be at least the default duration of ${policy.defaultSeconds} seconds.`;
  }
  if (supported !== null && !supported.includes(seconds)) {
    return `Maximum shot duration must be one of the supported values: ${supported.join(', ')} seconds.`;
  }
  return '';
}

function defaultForModel(key, supported) {
  if (supported === null) return DEFAULT_DURATION_SECONDS;
  if (supported.includes(DEFAULT_DURATION_SECONDS)) return DEFAULT_DURATION_SECONDS;
  if (key === `openrouter:${KLING_O1_MODEL}` && supported.includes(10)) return 10;
  return supported[0];
}
function catalogDurations(section, key) {
  const supported = DURATION_CATALOG[key] ?? null;
  if (supported === null || !key.startsWith('ltx:') || !key.endsWith('-fast')) return supported;
  const resolution = typeof section.resolution === 'string' ? section.resolution.trim().toLowerCase() : '';
  return resolution === '1440p' || resolution === '4k' ? supported.filter((seconds) => seconds <= 10) : supported;
}

function selectedProvider(section) {
  const provider = typeof section.provider === 'string' ? section.provider.trim() : '';
  if (provider) return provider;
  const baseUrl = typeof section.base_url === 'string' ? section.base_url.toLowerCase() : '';
  if (baseUrl.includes('agnes-ai.com')) return 'agnes';
  if (baseUrl.includes('ltx.io')) return 'ltx';
  if (baseUrl.includes('yunwu.ai')) return 'yunwu';
  return 'openrouter';
}

function positiveWholeSeconds(value) {
  if (typeof value !== 'number' && typeof value !== 'string') return null;
  if (typeof value === 'string' && !value.trim()) return null;
  const seconds = typeof value === 'number' ? value : Number(value);
  return Number.isSafeInteger(seconds) && seconds > 0 ? seconds : null;
}

function isRecord(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}
