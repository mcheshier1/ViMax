import {describe, expect, it} from 'vitest';
import {describeArtifacts, describeInvalidation, describeStyleMismatch, diffProjectFields, needsStyleInvalidation, toProjectFields, type ProjectFields} from './projectMetadata';

const saved: ProjectFields = {
  projectName: 'Moon cat',
  idea: 'A cat explores the moon',
  userRequirement: 'For children, at most three scenes',
  style: 'photorealistic cinematic live action',
};

const stylePaths = [
  'script2video/character_portraits/qwen_qwen-image-3',
  'script2video/shots/0/x-ai_grok-imagine-video',
  'script2video/final_video.mp4',
];

describe('project field diffing', () => {
  it('sends no request payload when the form is unchanged', () => {
    expect(diffProjectFields('s1', saved, {...saved})).toBeUndefined();
  });

  it('treats whitespace-only edits as unchanged', () => {
    expect(diffProjectFields('s1', saved, {...saved, projectName: '  Moon cat  ', style: `${saved.style}\n`})).toBeUndefined();
  });

  it('sends only the trimmed fields that changed', () => {
    const request = diffProjectFields('s1', saved, {...saved, projectName: '  Moon cat II  '});
    expect(request).toEqual({sessionId: 's1', projectName: 'Moon cat II'});
    expect(needsStyleInvalidation(Object.keys(request || {}))).toBe(false);
    expect(describeInvalidation({changed: ['projectName'], invalidated: [], requiresInvalidation: false})).toBeUndefined();
  });

  it('keeps a style edit in the same payload as the other edited fields', () => {
    const request = diffProjectFields('s1', saved, {...saved, style: '  watercolor  ', idea: 'A cat sails the moon'});
    expect(request).toEqual({sessionId: 's1', idea: 'A cat sails the moon', style: 'watercolor'});
    expect(needsStyleInvalidation(Object.keys(request || {}))).toBe(true);
  });

  it('reads the editable fields off a project record', () => {
    expect(toProjectFields(saved)).toEqual(saved);
  });
});

describe('style invalidation confirmation', () => {
  it('surfaces the artifacts a confirmed style change would regenerate', () => {
    const confirmation = describeInvalidation({changed: ['style'], invalidated: stylePaths, requiresInvalidation: true});
    expect(confirmation?.paths).toEqual(stylePaths);
    expect(confirmation?.heading).toMatch(/regenerates these/i);
    expect(confirmation?.detail).toMatch(/script, characters, storyboard, and camera files are kept/i);
  });

  it('does not confirm a change the server already applied', () => {
    expect(describeInvalidation({changed: ['style'], invalidated: stylePaths, requiresInvalidation: false})).toBeUndefined();
  });

  it('does not confirm a change that left the style alone', () => {
    expect(describeInvalidation({changed: ['idea'], invalidated: [], requiresInvalidation: true})).toBeUndefined();
  });
});

describe('manifest style mismatch', () => {
  it('names both styles when the render is behind the project', () => {
    const notice = describeStyleMismatch('watercolor', 'photorealistic cinematic live action');
    expect(notice).toContain('photorealistic cinematic live action');
    expect(notice).toContain('watercolor');
  });

  it('stays quiet when the render matches or has not started', () => {
    expect(describeStyleMismatch('watercolor', 'watercolor')).toBeUndefined();
    expect(describeStyleMismatch('watercolor', undefined)).toBeUndefined();
    expect(describeStyleMismatch('watercolor', '   ')).toBeUndefined();
  });
});

describe('render output on disk', () => {
  it('reports a fully rendered project as ready', () => {
    const checkpoints = describeArtifacts({portraits: 3, shots: 14, frames: 14, clips: 14, finalVideo: true});
    expect(checkpoints.map((checkpoint) => checkpoint.state)).toEqual(['ready', 'ready', 'ready', 'ready']);
  });

  it('shows a partially rendered project as in progress', () => {
    const checkpoints = describeArtifacts({portraits: 1, shots: 14, frames: 9, clips: 0, finalVideo: false});
    expect(checkpoints).toEqual([
      {label: 'Portraits', detail: '1 ready', state: 'ready'},
      {label: 'Keyframes', detail: '9/14', state: 'partial'},
      {label: 'Clips', detail: '0/14', state: 'missing'},
      {label: 'Final video', detail: 'Missing', state: 'missing'},
    ]);
  });

  it('reports a style change as missing artifacts until the render rebuilds them', () => {
    const checkpoints = describeArtifacts({portraits: 0, shots: 14, frames: 0, clips: 0, finalVideo: false});
    expect(checkpoints.every((checkpoint) => checkpoint.state === 'missing')).toBe(true);
    expect(checkpoints[0].detail).toBe('None');
  });

  it('says so when no shots have been planned yet', () => {
    const checkpoints = describeArtifacts({portraits: 0, shots: 0, frames: 0, clips: 0, finalVideo: false});
    expect(checkpoints[1].detail).toBe('No shots planned');
  });
});
