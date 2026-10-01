import type {Artifact} from './types';

/** Originals are reserved for the selected viewer and explicit downloads. */
export function artifactUrl(artifact: Artifact): string {
  const url = new URL(artifact.url, 'http://vimax.local');
  url.searchParams.set('updated', artifact.updatedAt);
  return `${url.pathname}${url.search}`;
}

/** The server versions its derivative cache from the source file itself. */
export function thumbnailUrl(artifact: Artifact, width = 480): string {
  const url = new URL(artifact.url, 'http://vimax.local');
  url.pathname = '/api/thumbnail';
  url.searchParams.set('width', String(width));
  url.searchParams.set('v', artifact.updatedAt);
  return `${url.pathname}${url.search}`;
}
