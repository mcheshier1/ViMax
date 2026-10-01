/**
 * Presentation logic for the render timeline.
 *
 * The timeline answers "what has been generated, what have I accepted, and what may run
 * next". Every slot state comes from the server, which derives it from the artifacts on
 * disk at call time, so nothing here has to interpret a stored flag.
 */

import type {AcceptanceSlot, AcceptanceStage, AcceptanceState} from './types';

export type {AcceptanceSlot, AcceptanceStage, AcceptanceState};

export const STAGE_LABELS: Record<string, {label: string; hint: string}> = {
  portraits: {
    label: 'Portraits',
    hint: 'Who the characters are. Every keyframe is drawn from these, so the style is decided here.',
  },
  keyframes: {
    label: 'Keyframes',
    hint: 'The stills each clip is drawn between. Clips may not run until these are accepted.',
  },
  clips: {
    label: 'Clips',
    hint: 'The rendered shots. This is the first phase that costs real money per shot.',
  },
  final_video: {
    label: 'Final video',
    hint: 'The shots concatenated in order.',
  },
};

export const STATE_LABELS: Record<AcceptanceState, string> = {
  planned: 'Not generated',
  rendered: 'Waiting for review',
  accepted: 'Accepted',
  rejected: 'Rejected',
  stale: 'Changed since accepted',
};

export function stageTitle(stage: AcceptanceStage, index: number): string {
  return `${index + 1}. ${STAGE_LABELS[stage.stage]?.label || stage.stage}`;
}

export function stageCounts(stage: AcceptanceStage): {accepted: number; total: number} {
  return {accepted: stage.slots.filter((slot) => slot.state === 'accepted').length, total: stage.slots.length};
}

/** Slots the next phase is waiting on, in the order the timeline shows them. */
export function blockingSlots(stage: AcceptanceStage): AcceptanceSlot[] {
  return stage.slots.filter((slot) => slot.state !== 'accepted');
}

/**
 * A frame whose description names a character it does not list as visible.
 *
 * The list decides whose portraits the reference selector is offered, so a character named
 * in the description but missing from it is drawn from the text alone — a DeepSeek who is
 * not DeepSeek, which is the failure this flag exists to catch before a render is paid for.
 * Descriptions name characters either as `<Name>` or in brackets, so both are read.
 */
export function planInconsistencies(plan: {characters: {idx: number; name: string}[]; firstFrame: {description: string; visible: number[]}; lastFrame: {description: string; visible: number[]}}): string[] {
  const byName = new Map(plan.characters.map((character) => [character.name.toLowerCase(), character]));
  const found: string[] = [];
  for (const [key, label] of [['firstFrame', 'first frame'], ['lastFrame', 'last frame']] as const) {
    const frame = plan[key];
    const named = new Set<string>();
    for (const match of frame.description.matchAll(/<([^<>]+)>|\(([^()]+)\)/g)) {
      const name = (match[1] || match[2] || '').trim();
      const character = byName.get(name.toLowerCase());
      if (character && !frame.visible.includes(character.idx)) named.add(character.name);
    }
    for (const name of named) found.push(`the ${label} describes ${name} but does not list them as visible`);
  }
  return found;
}

/**
 * Whether a row is worth opening on its own. Accepted shots collapse to a line, and so does
 * a shot nothing has been drawn for; anything with a note, a rejection or a plan that
 * contradicts itself opens, because that is what the page is for.
 */
export function rowNeedsAttention(state: string, options: {noted?: boolean; flagged?: boolean} = {}): boolean {
  if (options.noted || options.flagged) return true;
  // Only an accepted shot collapses. Everything else is either waiting for a review or has not
  // been drawn yet — and a shot nobody has drawn is the one whose frames are worth reading,
  // because those frames are the prompt the image model will be asked for.
  return state !== 'accepted';
}

/** The characters a frame shows, as chips the user can toggle. */
export function characterChips(characters: {idx: number; name: string}[], visible: number[]): {idx: number; name: string; on: boolean}[] {
  return characters.map((character) => ({...character, on: visible.includes(character.idx)}));
}

/** A visibility list with one character turned on or off, kept in the plan's own order. */
export function toggleCharacter(visible: number[], idx: number, on: boolean): number[] {
  const next = on ? [...visible, idx] : visible.filter((value) => value !== idx);
  return [...new Set(next)].sort((left, right) => left - right);
}

/**
 * Whether a saved plan differs from what the card is showing, so Save can stay quiet until
 * there is something to save.
 */
export function planDirty(plan: {firstFrame: {description: string; visible: number[]}; lastFrame: {description: string; visible: number[]}}, draft: {ffDesc: string; lfDesc: string; ffVis: number[]; lfVis: number[]}): boolean {
  return draft.ffDesc.trim() !== plan.firstFrame.description.trim()
    || draft.lfDesc.trim() !== plan.lastFrame.description.trim()
    || draft.ffVis.join(',') !== plan.firstFrame.visible.join(',')
    || draft.lfVis.join(',') !== plan.lastFrame.visible.join(',');
}

/** The name the Timeline gives a shot, from its slot key. */
export function slotName(stage: string, shot: string | null | undefined): string {
  if (shot === null || shot === undefined) return STAGE_LABELS[stage]?.label || stage;
  // Idea mode nests shots under scenes, so a slot is scene-qualified there.
  const scene = /^scene_(\d+)\/(.+)$/.exec(String(shot));
  if (scene) return `Scene ${Number(scene[1]) + 1} · Shot ${Number(scene[2]) + 1}`;
  return `Shot ${Number(shot) + 1}`;
}

/**
 * A stage's slots in the order the film plays them.
 *
 * A shot's number is its identity, not its place, so listing slots by number puts a shot
 * added in the middle of the film at the bottom of the page and makes a reorder look like it
 * did nothing. Anything the film does not play — the session-wide slots, or a shot with no
 * camera — keeps its place at the end.
 */
export function slotsInFilmOrder(slots: AcceptanceSlot[], order: string[]): AcceptanceSlot[] {
  const position: Record<string, number> = {};
  order.forEach((slot, index) => {
    position[slot] = index;
  });
  return [...slots].sort((left, right) => {
    const leftAt = position[String(left.shot ?? '')];
    const rightAt = position[String(right.shot ?? '')];
    if (leftAt === undefined) return rightAt === undefined ? 0 : 1;
    if (rightAt === undefined) return -1;
    return leftAt - rightAt;
  });
}

/**
 * The instruction to render one shot's clip, for a shot that has none yet.
 *
 * This is the per-shot render a redraw uses, with `allow_unreviewed_redo` because the guard
 * that flag lifts exists to stop a *person* confusing the Timeline's one-based "Shot 3" with
 * the zero-based slot key. A button on the row cannot be confused about which shot it is.
 */
export function renderClipText(root: string, slot: string): string {
  return (
    `Run vimax_render_video with exactly these arguments and no others: render_mode=${JSON.stringify(root)}, ` +
    `redo_shots=[${JSON.stringify(String(slot))}], stop_after="video", allow_unreviewed_redo=true. ` +
    "Render just this shot's clip and tell me when it is on disk."
  );
}

/** Where a shot plays in the film, in the words the reel uses for its neighbours. */
export function filmPosition(order: string[], slot: string): string {
  const position = order.indexOf(slot);
  if (position < 0) return '';
  const previous = position > 0 ? slotName('keyframes', order[position - 1]) : '';
  const total = `${position + 1} of ${order.length}`;
  return previous ? `plays ${total}, after ${previous}` : `plays ${total}, first in the film`;
}

/**
 * The shot a move should place this one after, or null when it is already at that end.
 *
 * The film plays camera by camera, so this is the whole film's order, not one camera's:
 * moving across a camera boundary is a move into that camera, which the endpoint handles.
 * A shot cannot go before the film's first shot, so there is nothing to ask for at the top.
 */
export function moveTarget(order: string[], slot: string, direction: 'earlier' | 'later'): string | null {
  const position = order.indexOf(slot);
  if (position < 0) return null;
  if (direction === 'earlier') return position >= 2 ? order[position - 2] : null;
  return position + 1 < order.length ? order[position + 1] : null;
}

export function slotTitle(stage: string, slot: AcceptanceSlot): string {
  return slotName(stage, slot.shot);
}

/** The slot key the render's redo argument takes, which is the shot as stored. */
export function slotKey(slot: AcceptanceSlot): string {
  return slot.shot === null || slot.shot === undefined ? '' : String(slot.shot);
}

/**
 * Whether a slot can be redrawn: it has to have artifacts to replace and no live
 * acceptance. A slot nobody has rendered yet has nothing to redraw.
 */
export function canRedo(stage: string, slot: AcceptanceSlot): boolean {
  return slot.artifacts.length > 0 && slot.state !== 'accepted' && REDOABLE_STAGES.has(stage);
}

/** The stages a redo can redraw: keyframes and clips, one shot at a time. */
export const REDOABLE_STAGES = new Set(['keyframes', 'clips']);

/** The phase a redraw of a stage runs, matching the render gate's vocabulary. */
export function redoPhase(stage: string): 'stills' | 'video' | null {
  if (stage === 'keyframes') return 'stills';
  if (stage === 'clips') return 'video';
  return null;
}

/**
 * Whether anything on disk is waiting for a human. The Timeline is the surface that
 * exists to answer this, so it is the tab a render should open on.
 */
export function needsReview(stages: AcceptanceStage[]): boolean {
  return stages.some((stage) => stage.slots.some((slot) => slot.artifacts.length > 0 && slot.state !== 'accepted'));
}

/** Slots with nothing on disk yet: what running a phase again would render from scratch. */
export function missingSlots(stage: AcceptanceStage | undefined): number {
  return stage ? stage.slots.filter((slot) => slot.state === 'planned').length : 0;
}

/** A keyframe is two files; a shot drawn part-way has one, and a phase re-run finishes it. */
const KEYFRAME_FILES = 2;

/**
 * Keyframes that are not all there: never drawn, or drawn part-way and then abandoned.
 *
 * A render that dies mid-shot leaves exactly that, and counting only never-drawn shots would
 * leave it with no way to be finished.
 */
export function missingKeyframes(stage: AcceptanceStage | undefined): number {
  return stage ? stage.slots.filter((slot) => slot.artifacts.length < KEYFRAME_FILES).length : 0;
}

/** The cost of those clips, in the words the user needs before spending. */
export function clipCostLine(remaining: number, clipSeconds: number, clipCostUsd: number): string {
  if (remaining <= 0) return 'No clips to render';
  const cost = (remaining * clipSeconds * clipCostUsd).toFixed(2);
  return `${remaining} clip${remaining === 1 ? '' : 's'} at ${clipSeconds}s ≈ $${cost}`;
}

export type Rejection = {title: string; reason: string; shot: string; stage: string};

/** Rejected slots with the guidance typed against them, in timeline order. */
export function rejections(stages: AcceptanceStage[]): Rejection[] {
  return stages.flatMap((stage) => stage.slots
    .filter((slot) => slot.state === 'rejected')
    .map((slot) => ({title: slotTitle(stage.stage, slot), reason: (slot.reason || '').trim(), shot: slotKey(slot), stage: stage.stage})));
}

/** Rejections that can be sent to the shot-scoped redraw tool. */
export function redoableRejections(items: Rejection[]): Rejection[] {
  return items.filter((item) => REDOABLE_STAGES.has(item.stage) && item.shot.trim().length > 0);
}

/**
 * The instruction that turns typed guidance into a redraw. The reasons are quoted
 * verbatim: they are the user's words about what went wrong, and the point of recording
 * them is that the regeneration answers them instead of guessing again. The shots are
 * named explicitly so a redraw cannot quietly spread into shots nobody asked about.
 */
export function redoRequestText(items: Rejection[], root = ''): string {
  const redoable = redoableRejections(items);
  if (redoable.length === 0) return '';
  const stages = [...new Set(redoable.map((item) => item.stage))];
  const phase = redoOrderedPhase(stages);
  // The slot key is stated beside the name on the card: the Timeline numbers shots from
  // one and the keys are shot directories numbered from zero, so "Shot 3" is slot "2".
  const lines = redoable.map((item) => `- ${item.title} (slot key "${item.shot}"): ${item.reason || 'no reason given'}`);
  return [
    `Redraw exactly these shots${root ? ` in ${root}` : ''} in ${phase}, and no others:`,
    ...lines,
    '',
    // A redraw asked for from a card names the slot itself, so the guard that exists to stop a
    // person confusing the card's "Shot 3" with slot 3 has nothing to protect here — and it has
    // been refusing an honest redraw because some *other* shot carries a rejection note.
    `Run vimax_render_video with exactly these arguments and no others: ${root ? `render_mode="${root}", ` : ''}${redoArgument(redoable)}, stop_after="${phase}", allow_unreviewed_redo=true.`,
    `The render clears those shots\' artifacts and puts each note into the prompt it redraws from, so state the problem in your own words above if the note is thin.`,
    `Every other shot and every accepted artifact stays as it is. Return the ${phase} outputs for review; do not begin another generation phase.`,
    'Do not change render_mode: a redraw never moves the sequence to another root.',
  ].join('\n');
}

/**
 * The instruction to redraw one slot, from the card the user is looking at. It is the
 * same request as the header's, scoped to the slot whose button was pressed.
 */
export function redoOneText(root: string, stage: string, slot: AcceptanceSlot, note = ''): string {
  return redoRequestText([{
    title: slotTitle(stage, slot),
    // A slot redrawn before carries its note as the reason it was redrawn: the request has
    // to quote that, or the second redraw repeats the first one's mistake. A note typed
    // against the plan takes precedence, because it is the newer thing said about this shot.
    reason: (note || slot.reason || slot.note || '').trim(),
    shot: slotKey(slot),
    stage,
  }], root);
}

/** `redo_shots=["2"]`, the explicit form of the same request. */
export function redoArgument(items: Rejection[]): string {
  return `redo_shots=[${redoableRejections(items).map((item) => JSON.stringify(item.shot)).join(', ')}]`;
}

/**
 * The phase a mixed set of redraws has to stop at: keyframes are redrawn before the
 * clips animated from them, and both need the user's eyes before anything else is spent.
 */
export function redoOrderedPhase(stages: string[]): 'stills' | 'video' {
  return stages.some((stage) => redoPhase(stage) === 'stills') ? 'stills' : 'video';
}

export type RenderActivity = 'idle' | 'running' | 'stalled' | 'interrupted' | 'finished' | 'failed' | 'blocked';

/**
 * How long a render may go without writing progress before it is reported as stalled. A
 * single image generation can take minutes, and a retry adds more: calling a live render
 * dead too early is its own kind of lie, so the window is generous.
 */
export const RENDER_STALL_MINUTES = 10;

/**
 * What a render is actually doing, from the status file plus the two things the file
 * cannot know: whether the agent it ran in still exists, and when it last made progress.
 * A status file says "rendering" forever once a render dies, so neither reading is
 * trusted on its own.
 */
export function renderActivity(status: unknown, options: {agentRunning?: boolean; lastProgressAt?: string; now?: Date} = {}): RenderActivity {
  if (!isStatus(status)) return 'idle';
  const state = String(status.status || '');
  if (state === 'rendering') {
    // A phase that stopped to ask for review is finished, not working: the render writes
    // `rendering` for the whole phase and `awaiting_confirmation` when it wants a human.
    if (isStatus(status) && status.awaiting_confirmation) return 'finished';
    if (options.agentRunning === false) return 'interrupted';
    // Silence is counted from the render's own start, so writes from before it began do
    // not read as progress it made.
    const silent = minutesSince(silenceSince(status, options.lastProgressAt), options.now ?? new Date());
    if (silent !== undefined && silent >= RENDER_STALL_MINUTES) return 'stalled';
    return 'running';
  }
  if (state === 'error') return 'failed';
  if (state === 'rendered') return 'finished';
  if (state === 'dependency_missing') return 'blocked';
  return 'idle';
}

/** The later of the render's start and its last write, as an ISO timestamp. */
function silenceSince(status: Record<string, unknown>, lastProgressAt?: string): string | undefined {
  const times = [lastProgressAt, status.timestamp]
    .map((value) => new Date(String(value || '')).getTime())
    .filter((value) => Number.isFinite(value));
  return times.length ? new Date(Math.max(...times)).toISOString() : undefined;
}

/** Whole minutes since a timestamp, or undefined when there is no usable one. */
function minutesSince(timestamp: string | undefined, now: Date): number | undefined {
  const value = new Date(String(timestamp || ''));
  if (!timestamp || Number.isNaN(value.getTime())) return undefined;
  return Math.max(0, Math.round((now.getTime() - value.getTime()) / 60000));
}

export type Redraw = {stage: string; slots: string[]};

/**
 * A render that failed, and the shots it cleared before it did. A failed render is the one
 * thing the review surface must never be quiet about: the artifacts it was replacing may be
 * gone, and only the person looking at the timeline can decide what to do about it.
 */
export type RenderFailure = {stage: string; slots: string[]; reason: string};

export function failedRedraw(events: unknown): RenderFailure | null {
  if (!Array.isArray(events)) return null;
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const row = events[index];
    if (!isStatus(row)) continue;
    if (row.status === 'dependency_missing' || row.error_type === 'dependency_missing') return null;
    if (row.status === 'error') {
      const slots = Array.isArray(row.redone_shots) ? row.redone_shots.map(String) : [];
      const reason = String(row.error || row.error_type || 'the render gave no reason').trim();
      return {stage: String(row.phase || '') === 'video' ? 'clips' : 'keyframes', slots, reason: reason.length > 300 ? `${reason.slice(0, 300)}…` : reason};
    }
    // Older producers/history rows start a redraw by naming its shots, without render_started.
    if (row.render_started || (row.status === 'rendering' && Array.isArray(row.redone_shots) && row.redone_shots.length > 0) || row.status === 'rendered' || row.awaiting_confirmation) return null;
  }
  return null;
}

/**
 * What a running render is redrawing, from the trail it leaves behind. Only the newest
 * decision counts, and only while nothing after it has ended the render: a trail that says
 * "redrawing slot 2" describes the present only until the next row contradicts it.
 *
 * The stage matters: a shot has a card in the keyframes lane and another in the clips
 * lane, and only the lane whose phase is running is being redrawn.
 */
export function activeRedraw(events: unknown): Redraw | null {
  if (!Array.isArray(events)) return null;
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const row = events[index];
    if (!isStatus(row)) continue;
    if (row.status === 'error' || row.status === 'rendered' || row.status === 'dependency_missing' || row.error_type === 'dependency_missing' || row.awaiting_confirmation) return null;
    const slots = row.redone_shots;
    if (row.render_started) {
      return Array.isArray(slots) && slots.length
        ? {stage: String(row.phase || '') === 'video' ? 'clips' : 'keyframes', slots: slots.map(String)}
        : null;
    }
    if (Array.isArray(slots) && slots.length) {
      return {stage: String(row.phase || '') === 'video' ? 'clips' : 'keyframes', slots: slots.map(String)};
    }
  }
  return null;
}

/**
 * Whether the render a click asked for has already ended.
 *
 * The waiting marker covers the gap between the click and the render taking that slot
 * over, and an active redraw for the slot clears it. A render that never shows up as one
 * — a whole-phase pass names no slots to redraw — leaves the card saying it is still
 * waiting for a render that finished some time ago.
 *
 * Only rows written after the click count, so the previous render's ending cannot clear a
 * request the click has just made.
 */
export function askedRenderFinished(events: unknown, askedAt: number): boolean {
  if (!Array.isArray(events)) return false;
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const row = events[index];
    if (!isStatus(row)) continue;
    const at = Date.parse(String(row.timestamp || ''));
    if (Number.isNaN(at) || at < askedAt) continue;
    if (row.status === 'error' || row.status === 'rendered' || row.status === 'dependency_missing' || row.error_type === 'dependency_missing' || row.awaiting_confirmation) return true;
  }
  return false;
}

/**
 * What the render is doing, in the words the surface that starts it has to show. A
 * render takes minutes, and without this the Timeline shows nothing at all while it runs.
 */
export function renderStatusLine(status: unknown, options: {now?: Date; agentRunning?: boolean; lastProgressAt?: string} = {}): string {
  const now = options.now ?? new Date();
  const where = `${clockTime(status && (status as Record<string, unknown>).timestamp)}, ${sinceWhen(status && (status as Record<string, unknown>).timestamp, now)}`;
  switch (renderActivity(status, options)) {
    case 'running': {
      const phase = isStatus(status) ? String(status.phase || '') : '';
      return `Rendering ${phase || 'the next phase'} — started ${where}`;
    }
    case 'interrupted':
      return `A render started ${where} and stopped without finishing — the agent is not running`;
    case 'stalled': {
      const silent = minutesSince(isStatus(status) ? silenceSince(status, options.lastProgressAt) : undefined, now);
      return `A render started at ${clockTime(isStatus(status) ? status.timestamp : '')} has written nothing for ${silent} min — it may have stopped`;
    }
    case 'failed': {
      const errorType = isStatus(status) ? String(status.error_type || '') : '';
      const reason = isStatus(status) ? String(status.error || status.error_type || '').trim() : '';
      return `Render stopped${errorType ? ` (${errorType})` : ''}${reason ? `: ${reason.slice(0, 200)}` : ''}`;
    }
    case 'finished':
      return `Last render finished ${clockTime(isStatus(status) ? status.timestamp : '')}`;
    case 'blocked':
      return 'Render is blocked: planning artifacts are missing';
    default:
      return '';
  }
}

function isStatus(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** `15:23`, the local clock time the status was written. */
function clockTime(timestamp: unknown): string {
  const parts = /(\d{2}):(\d{2})/.exec(String(timestamp || ''));
  return parts ? `${parts[1]}:${parts[2]}` : 'unknown';
}

/** How long ago that was, in the coarsest unit that is still honest. */
function sinceWhen(timestamp: unknown, now: Date): string {
  const value = new Date(String(timestamp || ''));
  if (Number.isNaN(value.getTime())) return 'at an unknown time';
  const minutes = Math.max(0, Math.round((now.getTime() - value.getTime()) / 60000));
  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes} min ago`;
  return `${Math.round(minutes / 60)} h ago`;
}

/**
 * What the agent is asked to do once a stage is accepted.
 *
 * The root is named, and the arguments are closed, because a session can hold an abandoned
 * plan beside the one being worked in: left to resolve the root itself the agent has picked
 * the other one, and the render only stops it after it has already started.
 */
export function renderRequestText(stage: 'stills' | 'video', root = ''): string {
  const mode = root ? `render_mode=${JSON.stringify(root)}, ` : '';
  if (stage === 'stills') {
    return `I have accepted the portraits in the Timeline. Run vimax_render_video with exactly these arguments and no others: ${mode}stop_after="stills", then show me the keyframes.`;
  }
  return `I have accepted every keyframe in the Timeline. Run vimax_render_video with exactly these arguments and no others: ${mode}stop_after="video", to render the clips and the final video.`;
}