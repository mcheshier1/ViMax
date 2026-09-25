import type {ProjectArtifacts, ProjectMetadata, ProjectUpdateRequest, ProjectUpdateResponse} from './types';

/** Editable project fields, in the order the server reports them in `changed`. */
export const PROJECT_FIELDS = ['projectName', 'idea', 'userRequirement', 'style'] as const;

export type ProjectFieldKey = (typeof PROJECT_FIELDS)[number];

export type ProjectFields = Record<ProjectFieldKey, string>;

/** Snapshot of a project's editable fields, used as the form baseline. */
export function toProjectFields(project: Pick<ProjectMetadata, ProjectFieldKey>): ProjectFields {
  return {
    projectName: project.projectName,
    idea: project.idea,
    userRequirement: project.userRequirement,
    style: project.style,
  };
}

/**
 * Trimmed request body holding only the fields that changed, or `undefined`
 * when the form still matches the saved project so nothing is sent.
 */
export function diffProjectFields(sessionId: string, baseline: ProjectFields, draft: ProjectFields): ProjectUpdateRequest | undefined {
  const changed = PROJECT_FIELDS.filter((field) => draft[field].trim() !== baseline[field].trim());
  if (changed.length === 0) return undefined;
  const request: ProjectUpdateRequest = {sessionId};
  for (const field of changed) request[field] = draft[field].trim();
  return request;
}

/** Style is the only editable field whose change invalidates rendered artifacts. */
export function needsStyleInvalidation(changed: readonly string[]): boolean {
  return changed.includes('style');
}

export type InvalidationConfirmation = {
  /** Artifact paths, relative to the session working dir, that a confirmed save deletes. */
  paths: string[];
  heading: string;
  detail: string;
};

/**
 * The confirmation a style change needs before its artifacts are deleted, or
 * `undefined` when the save is safe to apply on its own.
 */
export function describeInvalidation(response: Pick<ProjectUpdateResponse, 'changed' | 'invalidated' | 'requiresInvalidation'>): InvalidationConfirmation | undefined {
  if (!response.requiresInvalidation || !needsStyleInvalidation(response.changed)) return undefined;
  return {
    paths: [...response.invalidated],
    heading: 'Changing the style regenerates these',
    detail: 'Portraits, keyframes, shot clips, and the final video are deleted and rebuilt with the new style. Script, characters, storyboard, and camera files are kept.',
  };
}

export type ArtifactCheckpoint = {
  label: string;
  detail: string;
  state: 'ready' | 'partial' | 'missing';
};

/**
 * Render output as it exists on disk right now, phase by phase.
 *
 * Deliberately derived from the files rather than from the session record's
 * `stale` flags: the render never writes those, so they cannot tell whether a
 * project is rendered — and a confirmed style change shows up here immediately
 * as missing artifacts, then fills back in as the render rebuilds them.
 */
export function describeArtifacts(artifacts: ProjectArtifacts): ArtifactCheckpoint[] {
  const {portraits, shots, frames, clips, finalVideo} = artifacts;
  const ofShots = (count: number) => (shots > 0 ? `${count}/${shots}` : 'No shots planned');
  return [
    {label: 'Portraits', detail: portraits > 0 ? `${portraits} ready` : 'None', state: portraits > 0 ? 'ready' : 'missing'},
    {label: 'Keyframes', detail: ofShots(frames), state: shotState(frames, shots)},
    {label: 'Clips', detail: ofShots(clips), state: shotState(clips, shots)},
    {label: 'Final video', detail: finalVideo ? 'Ready' : 'Missing', state: finalVideo ? 'ready' : 'missing'},
  ];
}

function shotState(count: number, total: number): ArtifactCheckpoint['state'] {
  if (total <= 0 || count <= 0) return 'missing';
  return count >= total ? 'ready' : 'partial';
}

/** Plain-language warning when the last render used a different style than the project now records. */
export function describeStyleMismatch(style: string, manifestStyle?: string): string | undefined {
  const current = style.trim();
  const rendered = (manifestStyle || '').trim();
  if (!rendered || !current || rendered === current) return undefined;
  return `The last render used the style “${rendered}”. This project now records “${current}”, so the next render uses the new style.`;
}
