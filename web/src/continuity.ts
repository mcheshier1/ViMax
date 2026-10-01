/**
 * Presentation logic for the script-coverage review.
 *
 * The review itself is written by the agent's vimax_review_timeline tool; this turns it
 * into what the card shows and into the request that adds a suggested shot. It is pure so
 * that the shapes the browser and the server have to agree on — where a suggestion sits,
 * and the create request it becomes — are decided in one place and tested without a browser.
 */

import type {ContinuityBeat, ContinuityCheck, ContinuityPayload, ContinuityReview, ContinuitySuggestion, ShotPlan} from './types';
import {slotName} from './timeline';

export type {ContinuityBeat, ContinuityCheck, ContinuityPayload, ContinuityReview, ContinuitySuggestion};

/** How much of the script the film still plays. */
export function coverageCounts(review: ContinuityReview | null): {covered: number; partial: number; missing: number; total: number} {
  const beats = review?.beats ?? [];
  return {
    covered: beats.filter((beat) => beat.status === 'covered').length,
    partial: beats.filter((beat) => beat.status === 'partial').length,
    missing: beats.filter((beat) => beat.status === 'missing').length,
    total: beats.length,
  };
}

/** The beats the film does not fully play, with what is missing ahead of what is partial. */
export function gapBeats(review: ContinuityReview | null): ContinuityBeat[] {
  const rank: Record<ContinuityBeat['status'], number> = {missing: 0, partial: 1, covered: 2};
  return [...(review?.beats ?? [])]
    .filter((beat) => beat.status !== 'covered')
    .sort((left, right) => rank[left.status] - rank[right.status]);
}

/** The findings that name a problem; the reassuring ones are not worth the room. */
export function warnChecks(review: ContinuityReview | null): ContinuityCheck[] {
  return (review?.checks ?? []).filter((check) => check.status === 'warn');
}

/**
 * Whether the review is saying something worth acting on.
 *
 * A film-level card that is always open pushes the shots it is about off the screen, and
 * one that is always shut hides the gaps. So it opens itself exactly when there is
 * something to do: no review yet, a review the timeline has moved past, or a beat the film
 * does not play.
 */
export function coverageNeedsAttention(payload: ContinuityPayload | null): boolean {
  if (!payload?.review || payload.stale) return true;
  const counts = coverageCounts(payload.review);
  return counts.missing + counts.partial > 0 || warnChecks(payload.review).length > 0;
}

/** Where a suggestion sits, in the names the reel gives shots. */
export function suggestionPosition(suggestion: ContinuitySuggestion): string {
  return `after ${slotName('keyframes', String(suggestion.after_shot))}`;
}

/**
 * The characters a suggestion names, as plan indices.
 *
 * The create endpoint checks these against the root's characters and refuses an unknown
 * one, so a name the plan does not know is dropped rather than sent as a bad index.
 */
export function suggestionCharacterIndices(suggestion: ContinuitySuggestion, plans: ShotPlan[]): number[] {
  const known: Record<string, number> = {};
  for (const plan of plans) {
    for (const character of plan.characters) known[character.name.toLowerCase()] = character.idx;
  }
  const indices = (suggestion.characters ?? [])
    .map((name) => known[String(name).toLowerCase()])
    .filter((idx) => typeof idx === 'number');
  return [...new Set(indices)].sort((left, right) => left - right);
}

/** Whether a suggestion can be added: it needs a shot description and a shot to follow. */
export function suggestionAddable(suggestion: ContinuitySuggestion, shots: string[]): boolean {
  return Boolean(String(suggestion.visual_desc || '').trim()) && shots.includes(String(suggestion.after_shot));
}

/** The shot sizes a suggestion's own description can name, longest first so "close-up" does not win inside "extreme close-up". */
const SHOT_SIZES = ['extreme close-up', 'medium close-up', 'close-up', 'medium-wide', 'medium shot', 'wide shot', 'establishing shot', 'full shot'];

/**
 * The shot size a suggestion describes, so a close-up is visible at a glance.
 *
 * The direction decides what the review proposes — this only reports what it proposed, which
 * is what makes a suggestion worth checking before it is added.
 */
export function suggestionFraming(suggestion: ContinuitySuggestion): string {
  const text = `${suggestion.visual_desc || ''} ${suggestion.frames?.first || ''}`.toLowerCase();
  return SHOT_SIZES.find((size) => text.includes(size)) || '';
}

/**
 * Where the suggestion at this position goes, once the ones before it have been added.
 *
 * Suggestions that fill one gap all name the same shot to follow, so adding them in order
 * would put each new shot in front of the one before it — the reverse of the order the review
 * proposed, which is the order the story needs. Each follows the shot just created for the
 * previous suggestion when they share an anchor.
 */
export function suggestionAnchor(index: number, suggestions: ContinuitySuggestion[], created: string[]): string {
  const suggestion = suggestions[index];
  for (let previousIndex = index - 1; previousIndex >= 0; previousIndex -= 1) {
    if (String(suggestions[previousIndex].after_shot) === String(suggestion.after_shot) && created[previousIndex] !== undefined) {
      return String(created[previousIndex]);
    }
  }
  return String(suggestion.after_shot);
}

/** The create request that turns a suggested shot into a real one. */
export function addSuggestionRequest(sessionId: string, root: string, suggestion: ContinuitySuggestion, plans: ShotPlan[], after?: string) {
  const visible = suggestionCharacterIndices(suggestion, plans);
  const fallback = String(suggestion.visual_desc || '').trim();
  const first = String(suggestion.frames?.first || '').trim();
  const last = String(suggestion.frames?.last || '').trim();
  return {
    sessionId,
    root,
    // Adding several suggestions for one gap, one at a time, would put each new shot in front
    // of the last: they all name the same anchor. The caller passes where this one goes.
    after: after ?? String(suggestion.after_shot),
    brief: fallback,
    audioDesc: suggestion.audio_desc,
    motionDesc: suggestion.motion_desc,
    // A frame the review did not describe falls back to the shot description rather than
    // being sent empty, which the endpoint refuses.
    ffDesc: first || fallback,
    lfDesc: last || first || fallback,
    ffVis: visible,
    lfVis: visible,
  };
}

/** The instruction that asks the agent to review the timeline against the script. */
export function reviewRequestText(root = ''): string {
  const args = root ? `: render_mode=${JSON.stringify(root)}.` : '.';
  return (
    `Run vimax_review_timeline with exactly these arguments and no others${args} ` +
    'Then tell me which of the script’s beats the film no longer plays, and which shots would fix them.'
  );
}

/** How long the film runs, which is what the runtime finding compares against the request. */
export function runtimeLine(shots: number, clipSeconds: number): string {
  if (!shots || !clipSeconds) return '';
  return `${shots} shots x ${clipSeconds}s = ${shots * clipSeconds}s`;
}

/** What the card's header says about the review it is showing. */
export function reviewStatusLine(payload: ContinuityPayload | null, options: {asking?: boolean} = {}): string {
  if (options.asking) return 'Reviewing the timeline against the script…';
  const review = payload?.review;
  if (!review) return 'Not checked against the script yet.';
  const when = String(review.reviewed_at || '').slice(11, 16);
  if (payload?.stale) {
    const reason = String(payload.staleReason || '').trim();
    return reason ? `Reviewed at ${when}, but ${reason.charAt(0).toLowerCase()}${reason.slice(1)}` : `Reviewed at ${when}, and out of date.`;
  }
  return `Reviewed at ${when}, against the timeline as it is now.`;
}
