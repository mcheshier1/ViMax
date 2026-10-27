import {describe, expect, it} from 'vitest';
import {validateShotDuration, validateVideoDurationLimit, videoDurationPolicy} from './video-durations.mjs';

describe('video duration policy', () => {
  it('defaults to eight seconds where supported and gates explicit overrides until configured', () => {
    const section = {provider: 'openrouter', model: 'google/veo-3.1-lite', clip_seconds: '8'};
    const policy = videoDurationPolicy(section);

    expect(policy).toMatchObject({
      provider: 'openrouter', model: 'google/veo-3.1-lite', configured: false,
      defaultSeconds: 8, minSeconds: 4, maxSeconds: 8, allowedSeconds: [4, 6, 8],
    });
    expect(validateShotDuration(6, policy)).toMatch(/Configure a maximum/);
    expect(validateVideoDurationLimit(9, section)).toMatch(/supported values/);
  });

  it('filters a configured maximum and enforces supported boundaries without rounding', () => {
    const policy = videoDurationPolicy({
      provider: 'openrouter', model: 'kwaivgi/kling-v3.0-std', clip_seconds: '8',
      shot_duration_limits: {'openrouter:kwaivgi/kling-v3.0-std': 10},
    });

    expect(policy).toMatchObject({configured: true, defaultSeconds: 8, minSeconds: 3, maxSeconds: 10});
    expect(policy.allowedSeconds).toEqual([3, 4, 5, 6, 7, 8, 9, 10]);
    expect(validateShotDuration(3, policy)).toBe('');
    expect(validateShotDuration(10, policy)).toBe('');
    expect(validateShotDuration(11, policy)).toMatch(/maximum of 10/);
    expect(validateShotDuration(8.5, policy)).toMatch(/positive whole number/);
  });

  it('isolates limits by provider and model and rejects a maximum below the default', () => {
    const limits = {'openrouter:kwaivgi/kling-v3.0-std': 10};
    const otherModel = videoDurationPolicy({
      provider: 'openrouter', model: 'kwaivgi/kling-v3.0-pro', clip_seconds: '8',
      shot_duration_limits: limits,
    });
    expect(otherModel.configured).toBe(false);
    expect(videoDurationPolicy({
      provider: 'openrouter', model: 'kwaivgi/kling-video-o1',
    }).defaultSeconds).toBe(10);

    const belowDefault = {provider: 'openrouter', model: 'google/veo-3.1-lite', clip_seconds: '8'};
    expect(validateVideoDurationLimit(6, belowDefault)).toMatch(/at least the default duration of 8/);
    expect(videoDurationPolicy({...belowDefault, shot_duration_limits: {'openrouter:google/veo-3.1-lite': 6}}).configured)
      .toBe(false);
  });

  it('supports provider-specific discrete and fixed-duration catalogs', () => {
    expect(videoDurationPolicy({provider: 'agnes', model: 'agnes-video-2.5'}).allowedSeconds)
      .toEqual([4, 5, 6, 7, 8, 9, 10, 11, 12]);
    expect(videoDurationPolicy({provider: 'agnes', model: 'agnes-video-2.5-flash'}).allowedSeconds)
      .toEqual([4, 5, 6, 7, 8, 9, 10, 11, 12]);
    expect(videoDurationPolicy({provider: 'yunwu', model: 'veo3.1-fast'})).toMatchObject({
      configured: false, defaultSeconds: 8, minSeconds: 8, maxSeconds: 8, allowedSeconds: [8],
    });
    expect(videoDurationPolicy({provider: 'ltx', model: 'ltx-2-5-fast'}).allowedSeconds)
      .toEqual([6, 8, 10, 12, 14, 16, 18, 20]);
    expect(videoDurationPolicy({
      provider: 'ltx', model: 'ltx-2-5-fast', resolution: '1440p',
    }).allowedSeconds).toEqual([6, 8, 10]);
    const highResolution = videoDurationPolicy({
      provider: 'ltx', model: 'ltx-2-5-fast', resolution: '4K', clip_seconds: '12',
    });
    expect(validateShotDuration(highResolution.defaultSeconds, {...highResolution, configured: true})).toMatch(/maximum/);
    expect(videoDurationPolicy({provider: 'ltx', model: 'ltx-2-5-pro'}).allowedSeconds)
      .toEqual([6, 8, 10]);
  });

  it('treats custom models as integer ranges bounded by their saved maximum', () => {
    const custom = {provider: 'openrouter', model: 'vendor/private-model', clip_seconds: '8'};
    const unconfigured = videoDurationPolicy(custom);
    expect(unconfigured).toMatchObject({configured: false, minSeconds: 1, maxSeconds: 8, allowedSeconds: null});
    expect(validateShotDuration(9, unconfigured)).toMatch(/Configure a maximum/);

    const configured = videoDurationPolicy({
      ...custom, shot_duration_limits: {'openrouter:vendor/private-model': 10},
    });
    expect(configured).toMatchObject({configured: true, minSeconds: 1, maxSeconds: 10, allowedSeconds: null});
    expect(validateShotDuration(9, configured)).toBe('');
    expect(validateShotDuration(11, configured)).toMatch(/maximum of 10/);
  });
});
