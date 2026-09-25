import {describe, expect, it} from 'vitest';
import {activeRenderCheckpoint, deriveStoryboardReadiness, extractStoryboardPreviews, formatStructuredValue, friendlyArtifactTitle, friendlyFieldLabel, isArtifactPathField, isJsonArtifact, relatedVisualArtifacts, sortVisuals, visualArtifactTitle, visualFamilyCounts, visualPromptSource, visualPromptText, visualRole, visibleVisuals} from './artifactPresentation';
import type {Artifact} from './types';

const artifact: Artifact = {
  path: 'idea2video/scene_0/storyboard.json',
  name: 'storyboard.json',
  kind: 'document',
  size: 1200,
  updatedAt: '2026-07-19T00:00:00Z',
  url: '/api/artifact',
};

describe('artifact presentation', () => {
  it('extracts only storyboard descriptions', () => {
    const previews = extractStoryboardPreviews([
      {idx: 0, visual_desc: 'Wide beach shot', audio_desc: 'Waves'},
      {idx: 1, visual_desc: 'Close portrait', audio_desc: 'Dialogue'},
    ], artifact.path);
    expect(previews.map((preview) => preview.description)).toEqual(['Wide beach shot', 'Close portrait']);
  });

  it('creates readable artifact and field labels', () => {
    expect(friendlyArtifactTitle(artifact)).toBe('Scene 1 · Storyboard');
    expect(friendlyFieldLabel('motion_desc')).toBe('Motion');
    expect(friendlyFieldLabel('custom_camera_note')).toBe('Custom Camera Note');
  });

  it('presents indexes and booleans for non-technical readers', () => {
    expect(formatStructuredValue(0, 'cam_idx')).toBe('1');
    expect(formatStructuredValue([0, 2], 'ff_vis_char_idxs')).toBe('1, 3');
    expect(formatStructuredValue(false, 'is_last')).toBe('No');
  });

  it('identifies filesystem metadata that should stay hidden in artifacts', () => {
    expect(isArtifactPathField('path')).toBe(true);
    expect(isArtifactPathField('image_path')).toBe(true);
    expect(isArtifactPathField('working_dir')).toBe(true);
    expect(isArtifactPathField('reference_image_path_and_text_pairs')).toBe(true);
    expect(isArtifactPathField('visual_desc')).toBe(false);
  });

  it('keeps internal render status out of the artifact document list', () => {
    expect(isJsonArtifact(makeArtifact('render_status.json'))).toBe(false);
    expect(isJsonArtifact(makeArtifact('idea2video/script.json'))).toBe(true);
  });

  it('marks a complete storyboard package ready to render', () => {
    const readiness = deriveStoryboardReadiness([
      artifact,
      makeArtifact('idea2video/scene_0/camera_tree.json'),
      makeArtifact('idea2video/scene_0/shots/0/shot_description.json'),
      makeArtifact('idea2video/scene_0/shots/1/shot_description.json'),
      makeArtifact('idea2video/scene_0/shots/2/shot_description.json'),
    ], 3);
    expect(readiness.readyToRender).toBe(true);
    expect(readiness.overall).toBe('ready');
  });

  it('keeps incomplete shot files out of render-ready state', () => {
    const readiness = deriveStoryboardReadiness([
      artifact,
      makeArtifact('idea2video/scene_0/shots/0/shot_description.json'),
    ], 3);
    expect(readiness.readyToRender).toBe(false);
    expect(readiness.overall).toBe('partial');
    expect(readiness.shotDescriptions.status).toBe('partial');
    expect(readiness.cameraPlans.status).toBe('missing');
  });

  it('marks an empty project as missing', () => {
    expect(deriveStoryboardReadiness([], 0).overall).toBe('missing');
  });

  it('keeps render lights inactive before rendering starts', () => {
    const readiness = deriveStoryboardReadiness([
      artifact,
      makeArtifact('idea2video/scene_0/camera_tree.json'),
      makeArtifact('idea2video/scene_0/shots/0/shot_description.json'),
    ], 1);
    expect(readiness.readyToRender).toBe(true);
    expect(readiness.render.started).toBe(false);
    expect(readiness.render.frames.status).toBe('inactive');
    expect(readiness.render.clips.status).toBe('inactive');
    expect(readiness.render.finalVideo.status).toBe('inactive');
  });

  it('tracks partial and completed render output separately', () => {
    const readiness = deriveStoryboardReadiness([
      artifact,
      makeArtifact('idea2video/scene_0/shots/0/qwen_qwen-image-3/first_frame.png'),
      makeArtifact('idea2video/scene_0/shots/0/x-ai_grok-imagine-video/video.mp4'),
      makeArtifact('idea2video/scene_0/shots/1/x-ai_grok-imagine-video/video.mp4'),
      makeArtifact('idea2video/final_video.mp4'),
    ], 2);
    expect(readiness.render.started).toBe(true);
    expect(readiness.render.frames.status).toBe('partial');
    expect(readiness.render.clips.status).toBe('ready');
    expect(readiness.render.finalVideo.status).toBe('ready');
  });

  it('maps live render stages to the light that should animate', () => {
    expect(activeRenderCheckpoint('frame_start')).toBe('frames');
    expect(activeRenderCheckpoint('video_clip_waiting_for_frames')).toBe('clips');
    expect(activeRenderCheckpoint('concat_start')).toBe('finalVideo');
  });

  it('pairs shot documents with visual output from the same shot', () => {
    const shotDocument = makeArtifact('idea2video/scene_0/shots/1/shot_description.json');
    const visuals = relatedVisualArtifacts(shotDocument, [
      makeMediaArtifact('idea2video/scene_0/shots/0/qwen_qwen-image-3/first_frame.png', 'image'),
      makeMediaArtifact('idea2video/scene_0/shots/1/qwen_qwen-image-3/first_frame.png', 'image'),
      makeMediaArtifact('idea2video/scene_0/shots/1/x-ai_grok-imagine-video/video.mp4', 'video'),
      makeMediaArtifact('idea2video/scene_1/shots/1/qwen_qwen-image-3/first_frame.png', 'image'),
    ]);
    expect(visuals.map((item) => item.path)).toEqual([
      'idea2video/scene_0/shots/1/qwen_qwen-image-3/first_frame.png',
      'idea2video/scene_0/shots/1/x-ai_grok-imagine-video/video.mp4',
    ]);
  });

  it('pairs scene planning documents with every visual in that scene', () => {
    const visuals = relatedVisualArtifacts(artifact, [
      makeMediaArtifact('idea2video/scene_0/shots/0/qwen_qwen-image-3/first_frame.png', 'image'),
      makeMediaArtifact('idea2video/scene_0/shots/1/x-ai_grok-imagine-video/video.mp4', 'video'),
      makeMediaArtifact('idea2video/scene_1/shots/0/qwen_qwen-image-3/first_frame.png', 'image'),
    ]);
    expect(visuals).toHaveLength(2);
  });

  it('pairs character documents with generated portraits', () => {
    const visuals = relatedVisualArtifacts(makeArtifact('idea2video/characters.json'), [
      makeMediaArtifact('idea2video/character_portraits/qwen_qwen-image-3/0_Claude/front.png', 'image'),
      makeMediaArtifact('idea2video/scene_0/shots/0/qwen_qwen-image-3/first_frame.png', 'image'),
    ]);
    expect(visuals.map((item) => item.path)).toEqual(['idea2video/character_portraits/qwen_qwen-image-3/0_Claude/front.png']);
  });

  it('recognises a portrait that sits under the image model slug', () => {
    const visuals = relatedVisualArtifacts(makeArtifact('idea2video/characters.json'), [
      makeMediaArtifact('idea2video/character_portraits/qwen_qwen-image-3/0_Claude/front.png', 'image'),
      makeMediaArtifact('idea2video/character_portraits/qwen_qwen-image-3/0_Claude/side.png', 'image'),
      makeMediaArtifact('idea2video/character_portraits/qwen_qwen-image-3/0_Claude/back.png', 'image'),
    ]);
    expect(visuals.map((item) => item.path)).toEqual([
      'idea2video/character_portraits/qwen_qwen-image-3/0_Claude/back.png',
      'idea2video/character_portraits/qwen_qwen-image-3/0_Claude/front.png',
      'idea2video/character_portraits/qwen_qwen-image-3/0_Claude/side.png',
    ]);
  });

  it('pairs the model-scoped portrait registry with its own slug only', () => {
    const registry = makeArtifact('idea2video/character_portraits/qwen_qwen-image-3/registry.json');
    const visuals = relatedVisualArtifacts(registry, [
      makeMediaArtifact('idea2video/character_portraits/qwen_qwen-image-3/0_Claude/front.png', 'image'),
      makeMediaArtifact('idea2video/character_portraits/meta_muse-image/0_Claude/front.png', 'image'),
      makeMediaArtifact('idea2video/scene_0/shots/0/qwen_qwen-image-3/first_frame.png', 'image'),
    ]);
    expect(visuals.map((item) => item.path)).toEqual(['idea2video/character_portraits/qwen_qwen-image-3/0_Claude/front.png']);
  });

  it('ignores a stray registry file outside the portrait tree', () => {
    const visuals = relatedVisualArtifacts(makeArtifact('idea2video/registry.json'), [
      makeMediaArtifact('idea2video/character_portraits/qwen_qwen-image-3/0_Claude/front.png', 'image'),
    ]);
    expect(visuals).toEqual([]);
  });

  it('counts keyframes and clips that live under model slug directories', () => {
    const readiness = deriveStoryboardReadiness([
      artifact,
      makeArtifact('idea2video/scene_0/shots/3/qwen_qwen-image-3/first_frame.png'),
      makeArtifact('idea2video/scene_0/shots/3/x-ai_grok-imagine-video/video.mp4'),
    ], 4);
    expect(readiness.render.frames).toMatchObject({status: 'partial', count: 1, expected: 4});
    expect(readiness.render.clips).toMatchObject({status: 'partial', count: 1, expected: 4});
    expect(readiness.render.started).toBe(true);
  });

  it('pairs a shot document with its clip under the video model slug', () => {
    const shotDocument = makeArtifact('idea2video/scene_0/shots/3/shot_description.json');
    const visuals = relatedVisualArtifacts(shotDocument, [
      makeMediaArtifact('idea2video/scene_0/shots/3/x-ai_grok-imagine-video/video.mp4', 'video'),
      makeMediaArtifact('idea2video/scene_0/shots/2/x-ai_grok-imagine-video/video.mp4', 'video'),
    ]);
    expect(visuals.map((item) => item.path)).toEqual(['idea2video/scene_0/shots/3/x-ai_grok-imagine-video/video.mp4']);
  });

  it('makes frame and clip totals match a fully rendered sequence', () => {
    const readiness = deriveStoryboardReadiness([
      artifact,
      makeArtifact('idea2video/scene_0/shots/0/qwen_qwen-image-3/first_frame.png'),
      makeArtifact('idea2video/scene_0/shots/1/qwen_qwen-image-3/first_frame.png'),
      makeArtifact('idea2video/scene_0/shots/0/x-ai_grok-imagine-video/video.mp4'),
      makeArtifact('idea2video/scene_0/shots/1/x-ai_grok-imagine-video/video.mp4'),
      makeArtifact('idea2video/scene_0/shots/0/x-ai_grok-imagine-video/transition_video_from_shot_0.mp4'),
      makeArtifact('idea2video/scene_0/shots/0/x-ai_grok-imagine-video/new_camera_0.png'),
      makeArtifact('idea2video/final_video.mp4'),
    ], 2);
    expect(readiness.render.frames.status).toBe('ready');
    expect(readiness.render.clips.status).toBe('ready');
    expect(readiness.render.finalVideo.status).toBe('ready');
  });
});

function makeArtifact(path: string): Artifact {
  const name = path.split('/').at(-1) || path;
  return {...artifact, path, name};
}

function makeMediaArtifact(path: string, kind: 'image' | 'video'): Artifact {
  return {...makeArtifact(path), kind};
}

describe('visual roles', () => {
  // Paths taken from a real session: model-scoped render output.
  const projectPaths = [
    'script2video/final_video.mp4',
    'script2video/shots/0/qwen_qwen-image-3/first_frame.png',
    'script2video/shots/0/x-ai_grok-imagine-video/video.mp4',
    'script2video/shots/1/x-ai_grok-imagine-video/transition_video_from_shot_9.mp4',
    'script2video/shots/1/x-ai_grok-imagine-video/new_camera_1.png',
    'script2video/shots/1/x-ai_grok-imagine-video/cache/transition_video_from_shot_9-Scene-001.mp4',
    'script2video/character_portraits/qwen_qwen-image-3/0_Claude/front.png',
  ];

  it('classifies every visual in a rendered session', () => {
    expect(projectPaths.map((path) => visualRole(path)?.family)).toEqual([
      'final', 'review', 'final', 'camera', 'camera', 'scratch', 'review',
    ]);
  });

  it('reads the shot, the model and the role out of the path', () => {
    expect(visualRole('script2video/shots/12/qwen_qwen-image-3/last_frame.png')).toMatchObject({family: 'review', kind: 'Last frame', shot: 12, model: 'qwen_qwen-image-3'});
    expect(visualRole('script2video/shots/1/x-ai_grok-imagine-video/transition_video_from_shot_9.mp4')).toMatchObject({family: 'camera', kind: 'Camera move from Shot 10', shot: 1});
    expect(visualRole('script2video/shots/1/x-ai_grok-imagine-video/new_camera_1.png')).toMatchObject({family: 'camera', kind: 'New camera angle'});
  });

  it('names portraits from the character directory, with or without a model slug', () => {
    expect(visualRole('script2video/character_portraits/qwen_qwen-image-3/1_Wife/side.png')).toMatchObject({family: 'review', kind: 'Wife · Side portrait', model: 'qwen_qwen-image-3'});
    // Sessions rendered before artifacts were grouped by model keep the flat layout.
    expect(visualRole('script2video/character_portraits/1_Wife/side.png')).toMatchObject({family: 'review', kind: 'Wife · Side portrait', model: undefined});
  });

  it('labels visuals with their shot, one-based like the rest of the UI', () => {
    expect(visualArtifactTitle('script2video/shots/0/x-ai_grok-imagine-video/video.mp4')).toBe('Shot 1 · Clip');
    expect(visualArtifactTitle('script2video/shots/0/qwen_qwen-image-3/first_frame.png')).toBe('Shot 1 · First frame');
    expect(visualArtifactTitle('script2video/shots/1/x-ai_grok-imagine-video/transition_video_from_shot_9.mp4')).toBe('Shot 2 · Camera move from Shot 10');
    expect(visualArtifactTitle('script2video/final_video.mp4')).toBe('Final video');
    expect(visualArtifactTitle('idea2video/scene_0/shots/3/x-ai_grok-imagine-video/video.mp4')).toBe('Scene 1 · Shot 4 · Clip');
  });

  it('sorts a grid into production order rather than render time', () => {
    const unsorted = [
      makeMediaArtifact('script2video/shots/2/x-ai_grok-imagine-video/video.mp4', 'video'),
      makeMediaArtifact('script2video/final_video.mp4', 'video'),
      makeMediaArtifact('script2video/shots/0/x-ai_grok-imagine-video/video.mp4', 'video'),
      makeMediaArtifact('script2video/shots/1/qwen_qwen-image-3/first_frame.png', 'image'),
      makeMediaArtifact('script2video/character_portraits/qwen_qwen-image-3/0_Claude/front.png', 'image'),
      makeMediaArtifact('script2video/shots/1/qwen_qwen-image-3/first_frame.png', 'image'),
    ];

    expect(sortVisuals(unsorted).map((item) => visualArtifactTitle(item.path))).toEqual([
      'Final video',
      'Shot 1 · Clip',
      'Shot 3 · Clip',
      'Claude · Front portrait',
      'Shot 2 · First frame',
      'Shot 2 · First frame',
    ]);
  });

  it('counts the families that are worth filtering, and never scratch', () => {
    const media = projectPaths.map((path) => makeMediaArtifact(path, path.endsWith('.mp4') ? 'video' : 'image'));

    expect(visualFamilyCounts(media).map((entry) => `${entry.label}:${entry.count}`)).toEqual(['Final cut:2', 'Review:2', 'Camera work:2']);
    expect(visibleVisuals(media, 'all')).toHaveLength(6);
    expect(visibleVisuals(media, 'camera').map((item) => item.name)).toEqual(['transition_video_from_shot_9.mp4', 'new_camera_1.png']);
  });
});

describe('prompt provenance', () => {
  it('points a keyframe at the shot prompt that produced it', () => {
    expect(visualPromptSource('script2video/shots/3/qwen_qwen-image-3/first_frame.png'))
      .toEqual({documentPath: 'script2video/shots/3/first_frame_selector_output.json', field: 'frame_selector_output'});
    expect(visualPromptSource('script2video/shots/3/qwen_qwen-image-3/last_frame.png'))
      .toEqual({documentPath: 'script2video/shots/3/last_frame_selector_output.json', field: 'frame_selector_output'});
  });

  it('points a clip at its shot description, which holds the prompt it was sent', () => {
    expect(visualPromptSource('script2video/shots/3/x-ai_grok-imagine-video/video.mp4'))
      .toEqual({documentPath: 'script2video/shots/3/shot_description.json', field: 'clip_prompt'});
  });

  it('has no prompt to show for artifacts the render does not record one for', () => {
    // Portraits record theirs beside the images; camera work and the concatenated film do not.
    expect(visualPromptSource('script2video/shots/3/x-ai_grok-imagine-video/transition_video_from_shot_2.mp4')).toBeUndefined();
    expect(visualPromptSource('script2video/shots/3/x-ai_grok-imagine-video/new_camera_3.png')).toBeUndefined();
    expect(visualPromptSource('script2video/final_video.mp4')).toBeUndefined();
  });

  it('reads a frame prompt as the model received it: references then the prompt', () => {
    const document = {
      reference_image_path_and_text_pairs: [
        ['/ref/0.png', 'A front view portrait of Claude.'],
        ['/ref/1.png', 'The setting of the previous shot.'],
      ],
      sent_prompt: 'A close-up of Claude. Image 0 for appearance, Image 1 for the setting.',
    };

    expect(visualPromptText(document, 'frame_selector_output')).toBe(
      'Image 0: A front view portrait of Claude.\nImage 1: The setting of the previous shot.\nA close-up of Claude. Image 0 for appearance, Image 1 for the setting.',
    );
  });

  it('falls back to the selector prompt when no sent prompt was recorded', () => {
    const document = {reference_image_path_and_text_pairs: [['/ref/0.png', 'Claude.']], text_prompt: 'Draw the room.'};

    expect(visualPromptText(document, 'frame_selector_output')).toBe('Image 0: Claude.\nDraw the room.');
  });

  it('reads a clip prompt as the shot motion and audio description', () => {
    expect(visualPromptText({motion_desc: 'Static camera, she turns.', audio_desc: 'Wife: "We need to talk."'}, 'clip_prompt'))
      .toBe('Static camera, she turns.\nWife: "We need to talk."');
  });

  it('returns nothing rather than inventing a prompt', () => {
    expect(visualPromptText('not an object', 'frame_selector_output')).toBe('');
    expect(visualPromptText({}, 'clip_prompt')).toBe('');
  });
});

describe('portrait prompts', () => {
  it('points a portrait at the prompts file beside it', () => {
    expect(visualPromptSource('script2video/character_portraits/qwen_qwen-image-3/0_Claude/front.png'))
      .toEqual({documentPath: 'script2video/character_portraits/qwen_qwen-image-3/0_Claude/prompts.json', field: 'portrait_prompt'});
  });

  it('reads the prompt for the view the file is', () => {
    const document = {front: 'a front-view portrait…', side: 'a side-view portrait…', back: 'a back-view portrait…'};

    expect(visualPromptText(document, 'portrait_prompt', 'script2video/character_portraits/qwen_qwen-image-3/0_Claude/side.png'))
      .toBe('a side-view portrait…');
    expect(visualPromptText(document, 'portrait_prompt', 'script2video/character_portraits/qwen_qwen-image-3/0_Claude/back.png'))
      .toBe('a back-view portrait…');
  });

  it('returns nothing when the prompt for that view was not recorded', () => {
    expect(visualPromptText({front: 'only the front'}, 'portrait_prompt', '…/0_Claude/back.png')).toBe('');
  });
});
