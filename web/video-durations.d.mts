import type {ConfigSection, VideoDurationPolicy} from './src/types';

export function videoDurationPolicy(videoSection?: Partial<ConfigSection>): VideoDurationPolicy;
export function validateShotDuration(seconds: unknown, policy: VideoDurationPolicy): string;
