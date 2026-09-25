import type {Artifact, JsonValue} from './types';

export type StoryboardPreview = {
  id: string;
  description: string;
};

export type ReadinessStatus = 'ready' | 'partial' | 'missing' | 'inactive';

export type StoryboardReadiness = {
  overall: ReadinessStatus;
  readyToRender: boolean;
  storyboards: {status: ReadinessStatus; count: number};
  shotDescriptions: {status: ReadinessStatus; count: number; expected: number};
  cameraPlans: {status: ReadinessStatus; count: number; expected: number};
  render: {
    started: boolean;
    frames: {status: ReadinessStatus; count: number; expected: number};
    clips: {status: ReadinessStatus; count: number; expected: number};
    finalVideo: {status: ReadinessStatus; count: number; expected: number};
  };
};

export type RenderCheckpoint = 'frames' | 'clips' | 'finalVideo';

const FIELD_LABELS: Record<string, string> = {
  idx: 'Number',
  is_last: 'Final shot',
  cam_idx: 'Camera',
  visual_desc: 'Visual description',
  visual_description: 'Visual description',
  audio_desc: 'Audio',
  description: 'Description',
  ff_desc: 'First frame',
  lf_desc: 'Last frame',
  motion_desc: 'Motion',
  variation_type: 'Transition type',
  variation_reason: 'Transition notes',
  ff_vis_char_idxs: 'Characters in first frame',
  lf_vis_char_idxs: 'Characters in last frame',
  character_idx: 'Character',
  character_id: 'Character',
  shot_idx: 'Shot',
  scene_idx: 'Scene',
  camera_idx: 'Camera',
};

const FILE_TITLES: Record<string, string> = {
  'camera_tree.json': 'Camera plan',
  'characters.json': 'Characters',
  'script.json': 'Script',
  'shot_description.json': 'Shot description',
  'storyboard.json': 'Storyboard',
};

export function isJsonArtifact(artifact: Artifact): boolean {
  const name = artifact.name.toLowerCase();
  return name.endsWith('.json') && name !== 'render_status.json';
}

export function isStoryboardArtifact(artifact: Artifact): boolean {
  return artifact.name.toLowerCase() === 'storyboard.json';
}

export function relatedVisualArtifacts(documentArtifact: Artifact, artifacts: Artifact[]): Artifact[] {
  const media = artifacts.filter((artifact) => artifact.kind === 'image' || artifact.kind === 'video');
  const documentPath = normalizeArtifactPath(documentArtifact.path);
  const documentDirectory = parentArtifactPath(documentPath);
  const shotDirectory = documentPath.match(/^(.*\/shots\/\d+)(?:\/|$)/i)?.[1];
  const sceneDirectory = documentPath.match(/^(.*\/scene_\d+)(?:\/|$)/i)?.[1];
  const portraitRegistryDirectory = documentPath.match(/^(.*\/character_portraits\/[^/]+)\/registry\.json$/i)?.[1];
  const workflowRoot = documentPath.split('/')[0] || '';
  const name = documentArtifact.name.toLowerCase();

  const related = media.filter((artifact) => {
    const mediaPath = normalizeArtifactPath(artifact.path);
    if (shotDirectory) return mediaPath.startsWith(`${shotDirectory}/`);
    if (sceneDirectory) return mediaPath.startsWith(`${sceneDirectory}/`);
    if (name === 'characters.json') {
      return mediaPath.startsWith(`${workflowRoot}/character_portraits/`);
    }
    if (portraitRegistryDirectory) {
      return mediaPath.startsWith(`${portraitRegistryDirectory}/`);
    }
    if (name === 'script.json') {
      return mediaPath.startsWith(`${workflowRoot}/scene_`) || mediaPath === `${workflowRoot}/final_video.mp4`;
    }
    return parentArtifactPath(mediaPath) === documentDirectory;
  });

  return related.sort((left, right) => left.path.localeCompare(right.path, undefined, {numeric: true}));
}

export function friendlyFieldLabel(key: string): string {
  if (FIELD_LABELS[key]) return FIELD_LABELS[key];
  return key
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

export function isArtifactPathField(key: string): boolean {
  const normalized = key.trim().toLowerCase();
  return normalized.split('_').some((part) => part === 'path' || part === 'dir' || part === 'directory');
}

export function friendlyArtifactTitle(artifact: Artifact): string {
  const baseTitle = FILE_TITLES[artifact.name.toLowerCase()]
    || artifact.name.replace(/\.json$/i, '').replace(/[_-]+/g, ' ').replace(/\b\w/g, (character) => character.toUpperCase());
  const sceneMatch = artifact.path.match(/(?:^|\/)scene_(\d+)(?:\/|$)/i);
  const shotMatch = artifact.path.match(/(?:^|\/)shots\/(\d+)(?:\/|$)/i);
  const context = [];
  if (sceneMatch) context.push(`Scene ${Number(sceneMatch[1]) + 1}`);
  if (shotMatch) context.push(`Shot ${Number(shotMatch[1]) + 1}`);
  return context.length ? `${context.join(' · ')} · ${baseTitle}` : baseTitle;
}

export function structuredRecordTitle(value: JsonValue, index: number, artifact: Artifact): string {
  if (isJsonObject(value)) {
    for (const key of ['name', 'title', 'character_name', 'scene_title']) {
      const candidate = value[key];
      if (typeof candidate === 'string' && candidate.trim()) return candidate.trim();
    }
    const explicitIndex = value.idx;
    if (typeof explicitIndex === 'number') return `${recordNoun(artifact)} ${explicitIndex + 1}`;
  }
  return `${recordNoun(artifact)} ${index + 1}`;
}

/**
 * What a rendered visual is for.
 *
 * The render writes every artifact to a path that already states its role
 * (`<mode>/shots/<shot>/<model-slug>/<file>`, `character_portraits/<slug>/<character>/<view>.png`),
 * so role is derived from the path rather than tracked separately: it stays
 * correct for sessions rendered before artifacts were grouped by model, and it
 * needs no server state.
 *
 * `scratch` is the one family that is never shown: `cache/` holds moviepy's
 * scene-split debris from a transition render, which is fully represented by the
 * transition video beside it.
 */
export type VisualFamily = 'final' | 'review' | 'camera' | 'other' | 'scratch';

export type VisualRole = {
  family: VisualFamily;
  /** What the visual is, for the tile label. */
  kind: string;
  /** Sort key within the family, in production order. */
  order: number;
  /** Zero-based shot index, when the visual belongs to a shot. */
  shot?: number;
  /** Model slug that produced it, when the path carries one. */
  model?: string;
};

/** Families offered as filters, in the order they are shown. */
export const VISUAL_FAMILIES: {family: VisualFamily; label: string; hint: string}[] = [
  {family: 'final', label: 'Final cut', hint: 'The shot clips and the concatenated video: what ships.'},
  {family: 'review', label: 'Review', hint: 'Portraits and keyframes — the material the render gates ask you to approve.'},
  {family: 'camera', label: 'Camera work', hint: 'Camera moves and the angles derived from them while building a shot. They are not in the final cut.'},
  {family: 'other', label: 'Other', hint: 'Visuals that do not match a known render path.'},
];

const PORTRAIT_VIEWS = ['front', 'side', 'back'];

export function visualRole(path: string): VisualRole | undefined {
  const normalized = normalizeArtifactPath(path);
  const name = normalized.split('/').pop() || '';
  const lower = name.toLowerCase();
  if (normalized.split('/').includes('cache')) {
    return {family: 'scratch', kind: 'Scratch', order: 0};
  }
  const shotMatch = normalized.match(/(?:^|\/)shots\/(\d+)\//i);
  const shot = shotMatch ? Number(shotMatch[1]) : undefined;
  const model = normalized.match(/(?:^|\/)shots\/\d+\/([^/]+)\//i)?.[1];
  if (lower === 'final_video.mp4') {
    return {family: 'final', kind: 'Final video', order: 0};
  }
  if (shot !== undefined) {
    if (lower === 'video.mp4') {
      return {family: 'final', kind: 'Clip', order: 100 + shot, shot, model};
    }
    if (lower === 'first_frame.png' || lower === 'last_frame.png') {
      return {family: 'review', kind: lower === 'first_frame.png' ? 'First frame' : 'Last frame', order: 1000 + shot * 10 + (lower === 'first_frame.png' ? 0 : 1), shot, model};
    }
    const transition = lower.match(/^transition_video_from_shot_(\d+)\.mp4$/);
    if (transition) {
      return {family: 'camera', kind: `Camera move from Shot ${Number(transition[1]) + 1}`, order: shot * 100 + Number(transition[1]), shot, model};
    }
    const newCamera = lower.match(/^new_camera_(\d+)\.png$/);
    if (newCamera) {
      return {family: 'camera', kind: 'New camera angle', order: shot * 100 + 50 + Number(newCamera[1]), shot, model};
    }
  }
  const portrait = portraitRole(normalized, lower);
  if (portrait) return portrait;
  return {family: 'other', kind: '', order: 0, shot, model};
}

/** `character_portraits/<character>/<view>.png`, with or without a `<model-slug>` level. */
function portraitRole(normalized: string, lower: string): VisualRole | undefined {
  const view = PORTRAIT_VIEWS.find((candidate) => lower === `${candidate}.png`);
  if (!view) return undefined;
  const segments = normalized.split('/');
  const portraitsAt = segments.lastIndexOf('character_portraits');
  if (portraitsAt < 0 || portraitsAt + 2 > segments.length) return undefined;
  // `<idx>_<name>` is the character directory, whether or not a model slug sits above it.
  const characterSegment = segments.slice(portraitsAt + 1, -1).find((segment) => /^\d+_.+/.test(segment));
  if (!characterSegment) return undefined;
  const [index, ...rest] = characterSegment.split('_');
  const name = rest.join('_');
  const model = segments.length - portraitsAt > 3 ? segments[portraitsAt + 1] : undefined;
  return {
    family: 'review',
    kind: `${name} · ${view[0].toUpperCase()}${view.slice(1)} portrait`,
    order: Number(index) * 10 + PORTRAIT_VIEWS.indexOf(view),
    model,
  };
}

/** Tile label for a visual artifact. */
export function visualArtifactTitle(path: string): string {
  const role = visualRole(path);
  const kind = role?.kind || (path.split('/').pop() || '')
    .replace(/\.[^.]+$/, '')
    .replace(/[_-]+/g, ' ')
    .replace(/\b\w/g, (character) => character.toUpperCase());
  const context: string[] = [];
  const scene = path.match(/(?:^|\/)scene_(\d+)(?:\/|$)/i);
  if (scene) context.push(`Scene ${Number(scene[1]) + 1}`);
  if (role?.shot !== undefined) context.push(`Shot ${role.shot + 1}`);
  return context.length > 0 ? `${context.join(' · ')} · ${kind}` : kind;
}

export type VisualPromptSource = {
  /** Path of the document that holds what was sent to the model. */
  documentPath: string;
  /** How the prompt text is assembled from that document. */
  field: 'frame_selector_output' | 'clip_prompt' | 'portrait_prompt';
};

/**
 * Where to read the prompt that produced a visual.
 *
 * The render records what it sent where it computes it: a keyframe's prompt lives
 * in the shot's `<frame>_selector_output.json`, and a clip's prompt is the shot's
 * motion and audio description. Portraits, camera moves and derived angles build
 * their prompts in memory and keep no record, so they have none to show.
 */
export function visualPromptSource(path: string): VisualPromptSource | undefined {
  const normalized = normalizeArtifactPath(path);
  const role = visualRole(normalized);
  if (role?.family === 'review' && role.kind.includes('portrait')) {
    // Portrait prompts sit beside the portraits they produced, one file per character.
    return {documentPath: `${normalized.slice(0, normalized.lastIndexOf('/'))}/prompts.json`, field: 'portrait_prompt'};
  }
  if (role?.shot === undefined) return undefined;
  const shotDir = normalized.slice(0, normalized.lastIndexOf('/'));
  const shotRoot = shotDir.slice(0, shotDir.lastIndexOf('/'));
  const name = normalized.split('/').pop()?.toLowerCase() || '';
  if (name === 'first_frame.png' || name === 'last_frame.png') {
    return {documentPath: `${shotRoot}/${name.replace('.png', '')}_selector_output.json`, field: 'frame_selector_output'};
  }
  if (name === 'video.mp4') {
    return {documentPath: `${shotRoot}/shot_description.json`, field: 'clip_prompt'};
  }
  return undefined;
}

/** The prompt text a saved document holds, as it was sent to the model. */
export function visualPromptText(document: JsonValue, field: VisualPromptSource['field'], path = ''): string {
  if (!isJsonObject(document)) return '';
  if (field === 'portrait_prompt') {
    const view = (path.split('/').pop() || '').replace(/\.[^.]+$/, '').toLowerCase();
    const prompt = document[view];
    return typeof prompt === 'string' ? prompt : '';
  }
  if (field === 'clip_prompt') {
    return [document.motion_desc, document.audio_desc]
      .filter((part): part is string => typeof part === 'string' && part.trim().length > 0)
      .join('\n');
  }
  const prompt = typeof document.sent_prompt === 'string'
    ? document.sent_prompt
    : typeof document.text_prompt === 'string'
      ? document.text_prompt
      : '';
  const references = Array.isArray(document.reference_image_path_and_text_pairs) ? document.reference_image_path_and_text_pairs : [];
  const lines = references.map((pair, index) => {
    const description = Array.isArray(pair) && typeof pair[1] === 'string' ? pair[1] : '';
    return `Image ${index}: ${description}`;
  });
  return [...lines, prompt].filter((line) => line.trim().length > 0).join('\n');
}

/** Visuals in the order a viewer reads them: final cut, then what it was built from. */
export function sortVisuals<T extends {path: string}>(artifacts: T[]): T[] {
  const rank: Record<VisualFamily, number> = {final: 0, review: 1, camera: 2, other: 3, scratch: 4};
  return [...artifacts].sort((left, right) => {
    const leftRole = visualRole(left.path);
    const rightRole = visualRole(right.path);
    const byFamily = rank[leftRole?.family ?? 'other'] - rank[rightRole?.family ?? 'other'];
    if (byFamily !== 0) return byFamily;
    const byOrder = (leftRole?.order ?? 0) - (rightRole?.order ?? 0);
    if (byOrder !== 0) return byOrder;
    return left.path.localeCompare(right.path, undefined, {numeric: true});
  });
}

/** Families present in a session, with counts, for the filter chips. */
export function visualFamilyCounts<T extends {path: string}>(artifacts: T[]): {family: VisualFamily; label: string; hint: string; count: number}[] {
  const totals: Record<VisualFamily, number> = {final: 0, review: 0, camera: 0, other: 0, scratch: 0};
  for (const artifact of artifacts) {
    totals[visualRole(artifact.path)?.family ?? 'other'] += 1;
  }
  // Scratch is filtered out of the listing rather than offered as a view.
  totals.scratch = 0;
  return VISUAL_FAMILIES
    .filter((entry) => totals[entry.family] > 0)
    .map((entry) => ({...entry, count: totals[entry.family]}));
}

/** Visuals to show, shot-ordered, for one family or all of them. Scratch is never shown. */
export function visibleVisuals<T extends {path: string}>(artifacts: T[], family: VisualFamily | 'all'): T[] {
  const shown = artifacts.filter((artifact) => visualRole(artifact.path)?.family !== 'scratch');
  return sortVisuals(family === 'all' ? shown : shown.filter((artifact) => (visualRole(artifact.path)?.family ?? 'other') === family));
}

export function formatStructuredValue(value: JsonValue, key = ''): string {
  if (value === null) return 'Not specified';
  if (typeof value === 'boolean') return value ? 'Yes' : 'No';
  if (typeof value === 'number') return isIndexKey(key) ? String(value + 1) : String(value);
  if (typeof value === 'string') return value.trim() || 'Not specified';
  if (Array.isArray(value) && value.every(isJsonPrimitive)) {
    if (value.length === 0) return 'None';
    return value.map((item) => typeof item === 'number' && isIndexListKey(key) ? item + 1 : formatStructuredValue(item)).join(', ');
  }
  return '';
}

export function extractStoryboardPreviews(document: JsonValue, sourcePath: string): StoryboardPreview[] {
  const previews: StoryboardPreview[] = [];

  function visit(value: JsonValue) {
    if (Array.isArray(value)) {
      value.forEach(visit);
      return;
    }
    if (!isJsonObject(value)) return;
    const description = ['visual_desc', 'visual_description', 'description']
      .map((key) => value[key])
      .find((candidate) => typeof candidate === 'string' && candidate.trim());
    if (typeof description === 'string') {
      previews.push({id: `${sourcePath}:${previews.length}`, description: description.trim()});
      return;
    }
    for (const key of ['storyboards', 'storyboard', 'shots', 'scenes', 'items']) {
      const nested = value[key];
      if (nested !== undefined) visit(nested);
    }
  }

  visit(document);
  return previews;
}

export function deriveStoryboardReadiness(artifacts: Artifact[], storyboardCount: number): StoryboardReadiness {
  const storyboardFiles = artifacts.filter(isStoryboardArtifact).length;
  const shotDescriptions = artifacts.filter((artifact) => artifact.name.toLowerCase() === 'shot_description.json').length;
  const cameraPlans = artifacts.filter((artifact) => artifact.name.toLowerCase() === 'camera_tree.json').length;
  const frames = artifacts.filter((artifact) => artifact.name.toLowerCase() === 'first_frame.png').length;
  const clips = artifacts.filter((artifact) => artifact.name.toLowerCase() === 'video.mp4').length;
  const finalVideos = artifacts.filter((artifact) => artifact.name.toLowerCase() === 'final_video.mp4').length;
  const storyboards = checkpointStatus(storyboardCount, Math.max(storyboardFiles, 1));
  const shots = checkpointStatus(shotDescriptions, storyboardCount);
  const cameras = checkpointStatus(cameraPlans, storyboardFiles);
  const readyToRender = storyboardCount > 0
    && storyboards === 'ready'
    && shots === 'ready'
    && cameras === 'ready';
  const hasPlanningOutput = storyboardCount > 0 || shotDescriptions > 0 || cameraPlans > 0;

  return {
    overall: readyToRender ? 'ready' : hasPlanningOutput ? 'partial' : 'missing',
    readyToRender,
    storyboards: {status: storyboards, count: storyboardCount},
    shotDescriptions: {status: shots, count: shotDescriptions, expected: storyboardCount},
    cameraPlans: {status: cameras, count: cameraPlans, expected: storyboardFiles},
    render: {
      started: frames > 0 || clips > 0 || finalVideos > 0,
      frames: {status: renderCheckpointStatus(frames, storyboardCount), count: frames, expected: storyboardCount},
      clips: {status: renderCheckpointStatus(clips, storyboardCount), count: clips, expected: storyboardCount},
      finalVideo: {status: renderCheckpointStatus(finalVideos, 1), count: finalVideos, expected: 1},
    },
  };
}

export function activeRenderCheckpoint(stage: string): RenderCheckpoint {
  const normalized = stage.toLowerCase();
  if (normalized.includes('video_clip')) return 'clips';
  if (normalized.includes('concat') || normalized.includes('final_video') || normalized === 'render_done') return 'finalVideo';
  return 'frames';
}

export function isJsonObject(value: JsonValue): value is {[key: string]: JsonValue} {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

export function isJsonPrimitive(value: JsonValue): value is string | number | boolean | null {
  return value === null || ['string', 'number', 'boolean'].includes(typeof value);
}

function recordNoun(artifact: Artifact): string {
  const name = artifact.name.toLowerCase();
  if (name === 'storyboard.json' || name === 'shot_description.json') return 'Shot';
  if (name === 'characters.json') return 'Character';
  if (name === 'script.json') return 'Scene';
  return 'Item';
}

function isIndexKey(key: string): boolean {
  return key === 'idx' || key.endsWith('_idx') || key.endsWith('_index');
}

function isIndexListKey(key: string): boolean {
  return key.endsWith('_idxs') || key.endsWith('_indices');
}

function checkpointStatus(count: number, expected: number): ReadinessStatus {
  if (expected <= 0 || count <= 0) return 'missing';
  return count >= expected ? 'ready' : 'partial';
}

function renderCheckpointStatus(count: number, expected: number): ReadinessStatus {
  if (count <= 0) return 'inactive';
  return expected > 0 && count >= expected ? 'ready' : 'partial';
}

function normalizeArtifactPath(path: string): string {
  return path.replace(/\\/g, '/').replace(/^\.\//, '');
}

function parentArtifactPath(path: string): string {
  const separator = path.lastIndexOf('/');
  return separator >= 0 ? path.slice(0, separator) : '';
}
