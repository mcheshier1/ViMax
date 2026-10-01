import {describe, expect, it} from 'vitest';
import {addSuggestionRequest, coverageCounts, coverageNeedsAttention, gapBeats, reviewRequestText, reviewStatusLine, runtimeLine, suggestionAddable, suggestionAnchor, suggestionCharacterIndices, suggestionFraming, suggestionPosition, warnChecks} from './continuity';
import type {ContinuityPayload, ContinuityReview, ContinuitySuggestion, ShotPlan} from './types';

const suggestion: ContinuitySuggestion = {
  id: 's3',
  after_shot: 4,
  title: 'DeepSeek Enters',
  visual_desc: 'In the living room, a medium-wide shot shows DeepSeek entering from the right.',
  audio_desc: '[Sound Effect] Footsteps entering the room',
  motion_desc: 'Static camera. DeepSeek walks in from the right.',
  characters: ['DeepSeek', 'Wife'],
  rationale: 'The entrance was cut with its shot.',
  frames: {first: 'DeepSeek entering from the right.', last: 'His arm around the Wife.'},
};

const plans: ShotPlan[] = [
  {
    slot: '4',
    root: 'script2video',
    brief: '',
    characters: [{idx: 0, name: 'Claude'}, {idx: 1, name: 'Wife'}, {idx: 2, name: 'DeepSeek'}],
    firstFrame: {description: '', visible: [], prompt: ''},
    lastFrame: {description: '', visible: [], prompt: ''},
    motionDescription: '',
  },
];

const review: ContinuityReview = {
  reviewed_at: '2026-09-22T10:33:10',
  root: 'script2video',
  shots_reviewed: [0, 3, 1],
  summary: '',
  beats: [
    {index: 0, text: 'Wife: "we need to talk"', status: 'covered', covered_by: [0], note: ''},
    {index: 1, text: 'Wife: "I have met someone new"', status: 'missing', covered_by: [], note: ''},
    {index: 2, text: 'DeepSeek enters', status: 'partial', covered_by: [], note: 'the action was cut'},
  ],
  suggestions: [suggestion],
  checks: [
    {id: 'dialogue_uncovered', status: 'warn', message: 'WIFE: "Claude, I have met someone new."', shots: []},
    {id: 'character_absent', status: 'ok', message: 'Every character appears.', shots: []},
  ],
  notes: '',
};

function payload(overrides: Partial<ContinuityPayload> = {}): ContinuityPayload {
  return {root: 'script2video', review, stale: false, staleReason: '', shots: ['0', '1', '2', '3', '4'], clipSeconds: 5, ...overrides};
}

describe('script coverage review', () => {
  it('counts what the film still plays', () => {
    expect(coverageCounts(review)).toEqual({covered: 1, partial: 1, missing: 1, total: 3});
    expect(coverageCounts(null)).toEqual({covered: 0, partial: 0, missing: 0, total: 0});
  });

  it('lists the gaps worst first', () => {
    expect(gapBeats(review).map((beat) => beat.status)).toEqual(['missing', 'partial']);
  });

  it('shows only the findings that name a problem', () => {
    expect(warnChecks(review).map((check) => check.id)).toEqual(['dialogue_uncovered']);
  });

  it('opens itself only when there is something to act on', () => {
    // a missing beat, a timeline nobody has reviewed, and one it has moved past all count
    expect(coverageNeedsAttention(payload())).toBe(true);
    expect(coverageNeedsAttention(payload({review: null}))).toBe(true);
    expect(coverageNeedsAttention(payload({stale: true, staleReason: 'shots [5] removed'}))).toBe(true);
    const whole = {...review, beats: review.beats.map((beat) => ({...beat, status: 'covered' as const})), checks: []};
    expect(coverageNeedsAttention(payload({review: whole}))).toBe(false);
  });

  it('names a suggestion position the way the reel names shots', () => {
    // slot key 4 is the shot the Timeline calls Shot 5
    expect(suggestionPosition(suggestion)).toBe('after Shot 5');
  });

  it('resolves suggested characters to plan indices and drops names the plan does not know', () => {
    expect(suggestionCharacterIndices(suggestion, plans)).toEqual([1, 2]);
    expect(suggestionCharacterIndices({...suggestion, characters: ['Ghost', 'Wife', 'Wife']}, plans)).toEqual([1]);
    expect(suggestionCharacterIndices({...suggestion, characters: []}, plans)).toEqual([]);
  });

  it('builds the create request a suggestion becomes', () => {
    const request = addSuggestionRequest('session-1', 'script2video', suggestion, plans);
    expect(request).toEqual({
      sessionId: 'session-1',
      root: 'script2video',
      after: '4',
      brief: suggestion.visual_desc,
      audioDesc: '[Sound Effect] Footsteps entering the room',
      motionDesc: 'Static camera. DeepSeek walks in from the right.',
      ffDesc: 'DeepSeek entering from the right.',
      lfDesc: 'His arm around the Wife.',
      ffVis: [1, 2],
      lfVis: [1, 2],
    });
  });

  it('falls back to the shot description when the review described no frames', () => {
    const bare = {...suggestion, frames: {first: '', last: ''}};
    const request = addSuggestionRequest('session-1', 'script2video', bare, plans);
    // the endpoint refuses an empty frame description, so the shot description stands in
    expect(request.ffDesc).toBe(suggestion.visual_desc);
    expect(request.lfDesc).toBe(suggestion.visual_desc);
  });

  it('names the shot size a suggestion describes, so a close-up is visible at a glance', () => {
    expect(suggestionFraming(suggestion)).toBe('medium-wide');
    expect(suggestionFraming({...suggestion, visual_desc: 'A close-up of the Wife.', frames: {first: '', last: ''}})).toBe('close-up');
    expect(suggestionFraming({...suggestion, visual_desc: 'An extreme close-up of her eyes.', frames: {first: '', last: ''}})).toBe('extreme close-up');
    expect(suggestionFraming({...suggestion, visual_desc: 'nothing named here', frames: {first: '', last: ''}})).toBe('');
  });

  it('chains suggestions that fill one gap, so they land in the order proposed', () => {
    const first = {...suggestion, id: 's1', after_shot: 4};
    const second = {...suggestion, id: 's2', after_shot: 4};

    expect(suggestionAnchor(0, [first, second], [])).toBe('4');
    // the second follows the shot the first created, rather than the anchor they share
    expect(suggestionAnchor(1, [first, second], ['21'])).toBe('21');
    // a different anchor is left alone, as is one whose predecessor was not added
    expect(suggestionAnchor(1, [first, {...second, after_shot: 8}], ['21'])).toBe('8');
    expect(suggestionAnchor(1, [first, second], [])).toBe('4');
  });

  it('chains a repeated anchor past intervening suggestions for another gap', () => {
    const suggestions = [
      {...suggestion, id: 's1', after_shot: 4},
      {...suggestion, id: 's2', after_shot: 8},
      {...suggestion, id: 's3', after_shot: 4},
    ];

    expect(suggestionAnchor(2, suggestions, ['21', '22'])).toBe('21');
  });

  it('only offers to add a suggestion that has a description and a place to go', () => {
    expect(suggestionAddable(suggestion, ['0', '4'])).toBe(true);
    expect(suggestionAddable({...suggestion, visual_desc: '  '}, ['0', '4'])).toBe(false);
    expect(suggestionAddable(suggestion, ['0', '1'])).toBe(false);
  });

  it('asks the agent for a review of the root', () => {
    expect(reviewRequestText('script2video')).toContain('vimax_review_timeline');
    expect(reviewRequestText('script2video')).toContain('render_mode="script2video"');
    expect(reviewRequestText()).not.toContain('render_mode');
  });

  it('says how long the film runs', () => {
    expect(runtimeLine(10, 5)).toBe('10 shots x 5s = 50s');
    expect(runtimeLine(0, 5)).toBe('');
    expect(runtimeLine(10, 0)).toBe('');
  });

  it('describes the review it is showing', () => {
    expect(reviewStatusLine(payload())).toContain('Reviewed at 10:33');
    expect(reviewStatusLine(payload(), {asking: true})).toContain('Reviewing');
    expect(reviewStatusLine(payload({review: null}))).toContain('Not checked');
    const stale = reviewStatusLine(payload({stale: true, staleReason: 'The timeline changed since this review: shots [5] removed.'}));
    expect(stale).toContain('but the timeline changed since this review');
  });
});
