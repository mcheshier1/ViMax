import {describe, expect, it} from 'vitest';
import {compatibleImageResolution, defaultImageResolution, imageResolutionTiers} from './imagePresets';

describe('image resolution presets', () => {
  it('retains a tier shared by the newly selected image model', () => {
    expect(compatibleImageResolution('2K', 'openrouter', 'qwen/qwen-image-3')).toBe('2K');
    expect(compatibleImageResolution('2K', 'agnes', 'agnes-image-2.5-flash')).toBe('2K');
  });

  it('uses the new model default when a tier is unsupported', () => {
    expect(compatibleImageResolution('4K', 'openrouter', 'qwen/qwen-image-3')).toBe('auto');
    expect(compatibleImageResolution('2K', 'agnes', 'agnes-image-2.0-flash')).toBe('auto');
    expect(compatibleImageResolution('2K', 'yunwu', 'gemini-3.1-flash-image-preview')).toBe('auto');
  });

  it('offers tiers only for models with a documented resolution menu', () => {
    expect(imageResolutionTiers('openrouter', 'bytedance-seed/seedream-5-0-flash')).toEqual(['1K', '2K']);
    expect(imageResolutionTiers('agnes', 'agnes-image-2.0-flash')).toEqual([]);
    expect(defaultImageResolution('openrouter', 'bytedance-seed/seedream-5-0-flash')).toBe('1K');
  });
});
