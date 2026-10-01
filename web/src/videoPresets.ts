export const VIDEO_PROVIDER_PRESETS = {
  openrouter: {
    label: 'OpenRouter',
    baseUrl: 'https://openrouter.ai/api/v1',
    defaultModel: 'google/veo-3.1-lite',
    resolution: '720p',
    models: ['google/veo-3.1-lite', 'kwaivgi/kling-v3.0-std', 'kwaivgi/kling-v3.0-pro', 'kwaivgi/kling-video-o1'],
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
