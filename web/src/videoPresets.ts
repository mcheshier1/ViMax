export const HEYGEN_VIDEO_MODEL = 'heygen/heygen-video-1';

export const VIDEO_PROVIDER_PRESETS = {
  openrouter: {
    label: 'OpenRouter',
    baseUrl: 'https://openrouter.ai/api/v1',
    defaultModel: 'google/veo-3.1-lite',
    resolution: '720p',
    models: ['google/veo-3.1-lite', 'kwaivgi/kling-v3.0-std', 'kwaivgi/kling-v3.0-pro', 'kwaivgi/kling-video-o1', HEYGEN_VIDEO_MODEL],
  },
  yunwu: {
    label: 'Yunwu',
    baseUrl: 'https://yunwu.ai',
    defaultModel: 'veo3.1-fast',
    resolution: '720p',
    models: ['veo3.1-fast'],
  },
  agnes: {
    label: 'Agnes AI',
    baseUrl: 'https://apihub.agnes-ai.com/v1',
    defaultModel: 'agnes-video-2.5',
    resolution: '720p',
    models: ['agnes-video-2.5', 'agnes-video-2.5-flash'],
  },
  ltx: {
    label: 'LTX',
    baseUrl: 'https://api.ltx.io',
    defaultModel: 'ltx-2-5-pro',
    resolution: '1080p',
    models: ['ltx-2-5-pro', 'ltx-2-5-fast', 'ltx-2-3-pro', 'ltx-2-3-fast'],
  },
} as const;

export type VideoProvider = keyof typeof VIDEO_PROVIDER_PRESETS;

export function videoProviderPreset(provider: string) {
  return VIDEO_PROVIDER_PRESETS[provider as VideoProvider];
}

export function isHeygenVideoModel(model: string | undefined): boolean {
  return model === HEYGEN_VIDEO_MODEL;
}

export function videoModelSelection(provider: string, model: string, resolution: string, duration: string) {
  if (isHeygenVideoModel(model)) {
    const seconds = Number(duration);
    const normalizedSeconds = Number.isInteger(seconds) && seconds >= 5 && seconds <= 15 ? seconds : 8;
    return {
      resolution: resolution === '480p' || resolution === '768p' ? resolution : '480p',
      clip_seconds: String(normalizedSeconds),
    };
  }
  const preset = videoProviderPreset(provider);
  if (provider === 'openrouter' && preset?.models.some((entry) => entry === model)) {
    const seconds = Number(duration);
    const supportedDuration = model === 'google/veo-3.1-lite'
      ? seconds === 4 || seconds === 6 || seconds === 8
      : model === 'kwaivgi/kling-video-o1'
        ? seconds === 5 || seconds === 10
        : Number.isInteger(seconds) && seconds >= 3 && seconds <= 15;
    return {
      resolution: resolution === '480p' || resolution === '768p' ? preset.resolution : resolution,
      clip_seconds: supportedDuration ? duration : model === 'kwaivgi/kling-video-o1' ? '10' : '8',
    };
  }
  return {resolution, clip_seconds: duration};
}

export function validHeygenVideoSettings(model: string | undefined, resolution: string | undefined, duration: string | undefined): boolean {
  if (!isHeygenVideoModel(model)) return true;
  const seconds = Number(duration);
  return (resolution === '480p' || resolution === '768p') && Number.isInteger(seconds) && seconds >= 5 && seconds <= 15;
}
