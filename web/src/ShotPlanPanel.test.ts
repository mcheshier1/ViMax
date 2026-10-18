import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';
import {hasFilmDrafts, hasSessionFilmDrafts, reconcileFilmDrafts} from './ShotPlanPanel';
import type {AcceptanceStage, ShotPlan} from './types';

const fields = {ffDesc: 'Empty saloon', lfDesc: 'Empty saloon at dusk', ffVis: [], lfVis: [], brief: 'Exterior', motionDesc: 'Static camera', setIds: ['saloon']};
const key = (slot: string) => `vimax:film-draft:${sessionId}:script2video:${encodeURIComponent(slot)}`;
let records: Map<string, string>;
let sessionId: string;
let sequence = 0;

beforeEach(() => {
  sessionId = `draft-protection-${++sequence}`;
  records = new Map();
  vi.stubGlobal('localStorage', {
    get length() {return records.size;},
    key: (index: number) => [...records.keys()][index] ?? null,
    getItem: (name: string) => records.get(name) ?? null,
    setItem: (name: string, value: string) => records.set(name, value),
    removeItem: (name: string) => records.delete(name),
  });
});
afterEach(() => vi.unstubAllGlobals());

function store(slot: string, patch = {}, notes = {}) {
  const value = JSON.stringify({base: fields, plan: {...fields, ...patch}, notes, revision: 1});
  records.set(key(slot), value);
  return value;
}

function savedPlan(slot: string): ShotPlan {
  return {slot, root: 'script2video', brief: fields.brief, characters: [],
    firstFrame: {description: fields.ffDesc, visible: fields.ffVis, prompt: ''},
    lastFrame: {description: fields.lfDesc, visible: fields.lfVis, prompt: ''},
    motionDescription: fields.motionDesc, sets: [], setIds: fields.setIds, setRevision: 'saved',
    reusePreviousLastFrame: false, previousSlot: slot === '0' ? null : '0', previousLastFrameAvailable: false};
}

function savedFeedback(stage: AcceptanceStage['stage'], slot: string | null, reason: string): AcceptanceStage {
  return {stage, scope: slot === null ? 'session' : 'shot', state: 'rejected', accepted: false,
    slots: [{shot: slot, state: 'rejected', artifacts: [], reason}]};
}

describe('persisted shot draft protection', () => {
  it('does not warn or block generation for an unchanged persisted record', () => {
    store('0');
    expect(hasSessionFilmDrafts(sessionId)).toBe(false);
    expect(hasFilmDrafts(sessionId, 'script2video', ['0'])).toBe(false);
    expect(records.has(key('0'))).toBe(false);
  });

  it('keeps genuine unselected edits while pruning adjacent clean records', () => {
    const changed = store('0', {ffDesc: 'A wide view of the entrance'});
    store('1');
    store('2');
    expect(hasSessionFilmDrafts(sessionId)).toBe(true);
    expect(hasFilmDrafts(sessionId, 'script2video', ['0'])).toBe(true);
    expect(records.get(key('0'))).toBe(changed);
    expect(records.has(key('1'))).toBe(false);
    expect(records.has(key('2'))).toBe(false);
    expect(hasSessionFilmDrafts('another-project')).toBe(false);
  });

  it('preserves pending feedback even when the plan itself is unchanged', () => {
    const changed = store('0', {}, {keyframes: 'The entrance is too narrow'});
    expect(hasFilmDrafts(sessionId, 'script2video', ['0'])).toBe(true);
    expect(records.get(key('0'))).toBe(changed);
  });
  it('reconciles saved guidance and feedback on an unselected shot after reload', () => {
    records.set(key('1'), JSON.stringify({base: {...fields, ffDesc: 'Earlier description'}, plan: fields,
      notes: {keyframes: 'Already saved feedback'}, revision: 1}));
    reconcileFilmDrafts(sessionId, 'script2video', [savedPlan('0'), savedPlan('1')],
      [savedFeedback('keyframes', '1', 'Already saved feedback')]);
    expect(hasSessionFilmDrafts(sessionId)).toBe(false);
    expect(records.has(key('1'))).toBe(false);
  });

  it('does not mistake an unchanged feedback snapshot for edits when the saved plan changes', () => {
    store('1', {}, {keyframes: 'Already saved'});
    const updated = savedPlan('1');
    updated.firstFrame.description = 'The server has a newer view';
    reconcileFilmDrafts(sessionId, 'script2video', [updated], [savedFeedback('keyframes', '1', 'Already saved')]);
    expect(hasSessionFilmDrafts(sessionId)).toBe(false);
  });

  it('clears only already-saved feedback while keeping genuine guidance and other-stage feedback', () => {
    store('1', {ffDesc: 'A genuine unsaved entrance edit'}, {keyframes: 'Already saved', clips: 'Needs slower motion'});
    reconcileFilmDrafts(sessionId, 'script2video', [savedPlan('1')],
      [savedFeedback('keyframes', '1', 'Already saved'), savedFeedback('clips', '1', '')]);
    expect(hasSessionFilmDrafts(sessionId)).toBe(true);
    const retained = JSON.parse(records.get(key('1'))!);
    expect(retained.plan.ffDesc).toBe('A genuine unsaved entrance edit');
    expect(retained.notes).toEqual({clips: 'Needs slower motion'});
  });

  it('reconciles session-wide saved feedback without discarding unknown shot drafts', () => {
    store('@portraits', {}, {portraits: 'Already saved portrait feedback'});
    const unknown = store('not-in-snapshot', {ffDesc: 'A genuine draft outside this snapshot'});
    reconcileFilmDrafts(sessionId, 'script2video', [savedPlan('0')],
      [savedFeedback('portraits', null, 'Already saved portrait feedback')]);
    expect(records.has(key('@portraits'))).toBe(false);
    expect(records.get(key('not-in-snapshot'))).toBe(unknown);
    expect(hasSessionFilmDrafts(sessionId)).toBe(true);
  });
  it('keeps an unsaved continuity choice across reconciliation and clears it only after saving', () => {
    const changed = store('1', {reusePreviousLastFrame: true, previousSlot: '0'});
    reconcileFilmDrafts(sessionId, 'script2video', [savedPlan('1')], []);
    expect(hasFilmDrafts(sessionId, 'script2video', ['1'])).toBe(true);
    expect(records.get(key('1'))).toBe(changed);
    reconcileFilmDrafts(sessionId, 'script2video', [{...savedPlan('1'), reusePreviousLastFrame: true}], []);
    expect(hasFilmDrafts(sessionId, 'script2video', ['1'])).toBe(false);
  });

  it('preserves a continuity decision when timeline ordering changes its source', () => {
    records.set(key('1'), JSON.stringify({
      base: {...fields, reusePreviousLastFrame: false, previousSlot: '0'},
      plan: {...fields, reusePreviousLastFrame: true, previousSlot: '0'},
      notes: {}, revision: 1,
    }));
    const moved = {...savedPlan('1'), reusePreviousLastFrame: true, previousSlot: '2'};
    reconcileFilmDrafts(sessionId, 'script2video', [moved], []);
    expect(hasFilmDrafts(sessionId, 'script2video', ['1'])).toBe(true);
    expect(JSON.parse(records.get(key('1'))!).plan.previousSlot).toBe('0');
  });
});
