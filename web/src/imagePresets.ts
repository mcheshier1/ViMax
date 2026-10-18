export const IMAGE_PROVIDER_PRESETS = {
  openrouter: {
    label: 'OpenRouter',
    baseUrl: 'https://openrouter.ai/api/v1',
    defaultModel: 'qwen/qwen-image-3',
    models: [
      {id: 'qwen/qwen-image-3', label: 'Qwen Image 3'},
      {id: 'bytedance-seed/seedream-5-0-flash', label: 'ByteDance Seedream 5.0 Flash'},
    ],
  },
  yunwu: {
    label: 'Yunwu / current',
    baseUrl: 'https://yunwu.ai',
    defaultModel: 'gemini-3.1-flash-image-preview',
    models: [],
  },
  agnes: {
    label: 'Agnes AI',
    baseUrl: 'https://apihub.agnes-ai.com/v1',
    defaultModel: 'agnes-image-2.5-flash',
    models: [
      {id: 'agnes-image-2.5-flash', label: 'Agnes Image 2.5 Flash (recommended)'},
      {id: 'agnes-image-2.1-flash', label: 'Agnes Image 2.1 Flash'},
      {id: 'agnes-image-2.0-flash', label: 'Agnes Image 2.0 Flash'},
    ],
  },
} as const;

export type ImageProvider = keyof typeof IMAGE_PROVIDER_PRESETS;

export function imageProviderFromBaseUrl(baseUrl: string): ImageProvider {
  const normalized = baseUrl.toLowerCase();
  if (normalized.includes('openrouter.ai')) return 'openrouter';
  if (normalized.includes('agnes-ai.com')) return 'agnes';
  return 'yunwu';
}

export function imageProviderPreset(provider: string) {
  return IMAGE_PROVIDER_PRESETS[provider as ImageProvider];
}

export function imageResolutionTiers(provider: string, model: string): string[] {
  if (provider === 'openrouter' && (model === 'qwen/qwen-image-3' || model === 'bytedance-seed/seedream-5-0-flash')) {
    return ['1K', '2K'];
  }
  if (provider === 'agnes' && /agnes-image-2\.[15]-flash/i.test(model)) return ['1K', '2K', '3K', '4K'];
  return [];
}

export function defaultImageResolution(provider: string, model: string): string {
  return provider === 'openrouter' && model === 'bytedance-seed/seedream-5-0-flash' ? '1K' : 'auto';
}

export function compatibleImageResolution(current: string, provider: string, model: string): string {
  if (current === 'auto' || current === '') return current || defaultImageResolution(provider, model);
  if (imageResolutionTiers(provider, model).includes(current)) return current;
  return defaultImageResolution(provider, model);
}
