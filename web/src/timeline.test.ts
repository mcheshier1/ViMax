import {describe, expect, it} from 'vitest';
import {activeRedraw, askedRenderFinished, blockingSlots, canRedo, characterChips, clipCostLine, failedRedraw, filmPosition, missingKeyframes, missingSlots, moveTarget, needsReview, planDirty, planInconsistencies, redoOneText, redoPhase, redoRequestText, rejections, renderActivity, renderClipText, renderRequestText, renderStatusLine, rowNeedsAttention, slotName, slotsInFilmOrder, stageCounts, stageTitle, slotTitle, toggleCharacter} from './timeline';
import type {AcceptanceSlot, AcceptanceStage, AcceptanceState} from './types';

describe('where a shot plays', () => {
  // The reel lists shots by number; the film plays them camera by camera, and these are the
  // two ends of that difference.
  const order = ['0', '3', '9', '12', '1', '2', '4', '14', '13', '8'];

  it('says where a shot plays, in the names the reel gives its neighbours', () => {
    expect(filmPosition(order, '13')).toBe('plays 9 of 10, after Shot 15');
    expect(filmPosition(order, '0')).toBe('plays 1 of 10, first in the film');
    expect(filmPosition(order, '99')).toBe('');
  });

  it('moves a shot one place at a time, and not past either end', () => {
    expect(moveTarget(order, '13', 'earlier')).toBe('4');
    expect(moveTarget(order, '13', 'later')).toBe('8');
    // the film's first and last shots have nowhere to go
    expect(moveTarget(order, '0', 'earlier')).toBeNull();
    expect(moveTarget(order, '8', 'later')).toBeNull();
    expect(moveTarget(order, '99', 'later')).toBeNull();
  });

  it('lists a stage in the order the film plays, not in the order of the numbers', () => {
    const slots = ['0', '1', '2', '13', '8'].map((shot) => ({shot, state: 'planned' as const, artifacts: []}));

    const listed = slotsInFilmOrder(slots, order);

    // in this film slot 13 plays before slot 8, so the listing is not the number order
    expect(listed.map((slot) => slot.shot)).toEqual(['0', '1', '2', '13', '8']);
  });

  it('keeps anything the film does not play at the end of the list', () => {
    const slots = [
      {shot: '13', state: 'planned' as const, artifacts: []},
      {shot: null, state: 'rendered' as const, artifacts: []},
      {shot: '8', state: 'planned' as const, artifacts: []},
    ];

    const listed = slotsInFilmOrder(slots, order);

    expect(listed.map((slot) => slot.shot)).toEqual(['13', '8', null]);
  });
});

function stage(overrides: Partial<AcceptanceStage> = {}): AcceptanceStage {
  return {stage: 'keyframes', scope: 'shot', state: 'rendered', accepted: false, slots: [], ...overrides};
}

describe('timeline stages', () => {
  it('numbers the stages in the order a render runs them', () => {
    expect(stageTitle(stage({stage: 'portraits'}), 0)).toBe('1. Portraits');
    expect(stageTitle(stage({stage: 'keyframes'}), 1)).toBe('2. Keyframes');
    expect(stageTitle(stage({stage: 'clips'}), 2)).toBe('3. Clips');
    expect(stageTitle(stage({stage: 'final_video'}), 3)).toBe('4. Final video');
  });

  it('counts what is accepted out of what exists', () => {
    const counted = stageCounts(stage({slots: [
      {shot: '0', state: 'accepted', artifacts: []},
      {shot: '1', state: 'rendered', artifacts: []},
      {shot: '2', state: 'planned', artifacts: []},
    ]}));

    expect(counted).toEqual({accepted: 1, total: 3});
  });

  it('reports the slots the next phase is waiting on', () => {
    const waiting = blockingSlots(stage({slots: [
      {shot: '0', state: 'accepted', artifacts: []},
      {shot: '1', state: 'stale', artifacts: []},
      {shot: '2', state: 'planned', artifacts: []},
    ]}));

    expect(waiting.map((slot) => slot.shot)).toEqual(['1', '2']);
  });

  it('names a shot slot one-based, like the rest of the UI', () => {
    expect(slotTitle('keyframes', {shot: '0', state: 'rendered', artifacts: []})).toBe('Shot 1');
    expect(slotTitle('keyframes', {shot: '12', state: 'rendered', artifacts: []})).toBe('Shot 13');
  });

  it('names a session-wide slot after its stage', () => {
    expect(slotTitle('portraits', {shot: null, state: 'rendered', artifacts: []})).toBe('Portraits');
  });
});

describe('the cost of the next phase', () => {
  it('counts the clips that do not exist yet', () => {
    const clips = stage({stage: 'clips', slots: [
      {shot: '0', state: 'planned', artifacts: []},
      {shot: '1', state: 'planned', artifacts: []},
      {shot: '2', state: 'rendered', artifacts: ['script2video/shots/2/x/video.mp4']},
    ]});

    expect(missingSlots(clips)).toBe(2);
  });

  it('counts a keyframe drawn part-way as missing too', () => {
    // a render that died mid-shot leaves the first frame and no last frame
    const keyframes = stage({stage: 'keyframes', slots: [
      {shot: '0', state: 'accepted', artifacts: ['a.png', 'b.png']},
      {shot: '1', state: 'rendered', artifacts: ['a.png']},
      {shot: '2', state: 'planned', artifacts: []},
    ]});

    expect(missingKeyframes(keyframes)).toBe(2);
    expect(missingSlots(keyframes)).toBe(1);
  });

  it('states that cost the way a person reads it', () => {
    expect(clipCostLine(13, 5, 0.112)).toBe('13 clips at 5s ≈ $7.28');
    expect(clipCostLine(1, 5, 0.112)).toBe('1 clip at 5s ≈ $0.56');
  });

  it('says so when there is nothing to render', () => {
    expect(clipCostLine(0, 5, 0.112)).toBe('No clips to render');
  });
});

describe('the request sent to the agent', () => {
  it('asks for the phase the acceptance unlocks, naming the tool and phase', () => {
    expect(renderRequestText('video')).toContain('stop_after="video"');
    expect(renderRequestText('stills')).toContain('stop_after="stills"');
    // the root is named, because a session can hold an abandoned plan beside the current one
    expect(renderRequestText('stills', 'script2video')).toContain('render_mode="script2video"');
    expect(renderRequestText('video', 'script2video')).toContain('render_mode="script2video"');
    expect(renderRequestText('video')).toContain('accepted every keyframe');
  });

  it('asks for one shot\'s clip on its own, so clips can be reviewed a shot at a time', () => {
    const text = renderClipText('script2video', '4');

    // the same per-shot render a redraw uses, closed to other shots and to other phases
    expect(text).toContain('render_mode="script2video"');
    expect(text).toContain('redo_shots=["4"]');
    expect(text).toContain('stop_after="video"');
    // the wrong-shot guard is about a person reading the reel's one-based names, not a button
    expect(text).toContain('allow_unreviewed_redo=true');
    expect(text).toContain('exactly these arguments and no others');
  });
});
describe('turning rejections into a re-render', () => {
  it('collects rejected slots with the guidance typed against them', () => {
    const stages: AcceptanceStage[] = [
      stage({stage: 'keyframes', slots: [
        {shot: '0', state: 'accepted', artifacts: []},
        {shot: '5', state: 'rejected', artifacts: [], reason: 'her dress is blue here but pink everywhere else', rejectedAt: 'now'},
        {shot: '6', state: 'rejected', artifacts: [], reason: '   '},
      ]}),
    ];

    expect(rejections(stages)).toEqual([
      {title: 'Shot 6', reason: 'her dress is blue here but pink everywhere else', shot: '5', stage: 'keyframes'},
      {title: 'Shot 7', reason: '', shot: '6', stage: 'keyframes'},
    ]);
  });

  it('quotes the reasons verbatim in the instruction', () => {
    const text = redoRequestText([{title: 'Shot 6', reason: 'her dress is blue here but pink everywhere else', shot: '5', stage: 'keyframes'}]);

    expect(text).toContain('- Shot 6 (slot key "5"): her dress is blue here but pink everywhere else');
    expect(text).toContain('Redraw exactly these shots in stills, and no others');
    expect(text).toContain('Every other shot and every accepted artifact stays as it is');
  });

  it('names the shots it is allowed to touch, so a redraw cannot spread', () => {
    const items = [
      {title: 'Shot 6', reason: 'her dress is blue', shot: '5', stage: 'keyframes'},
      {title: 'Shot 7', reason: 'the room is a different room', shot: '11', stage: 'clips'},
    ];

    expect(redoRequestText(items)).toContain('redo_shots=["5", "11"]');
    // Keyframes are redrawn before the clips animated from them, whatever the set holds.
    expect(redoRequestText(items)).toContain('stop_after="stills"');
  });

  it('redraws clips alone when only clips were rejected', () => {
    expect(redoRequestText([{title: 'Shot 2', reason: 'the ceiling sags', shot: '1', stage: 'clips'}])).toContain('stop_after="video"');
  });

  it('says so rather than inventing a note when none was typed', () => {
    expect(redoRequestText([{title: 'Shot 3', reason: '', shot: '2', stage: 'keyframes'}])).toContain('- Shot 3 (slot key "2"): no reason given');
  });

  it('asks for nothing when nothing was rejected', () => {
    expect(redoRequestText([])).toBe('');
  });
});

describe('a slot that can be redrawn', () => {
  const slot = (state: AcceptanceState, artifacts: string[], shot = '2') =>
    ({shot, state, artifacts}) as AcceptanceSlot;

  it('offers a redraw only where there are artifacts to replace and no live acceptance', () => {
    expect(canRedo('keyframes', slot('rendered', ['shots/2/qwen/first_frame.png']))).toBe(true);
    expect(canRedo('keyframes', slot('rejected', ['shots/2/qwen/first_frame.png']))).toBe(true);
    expect(canRedo('keyframes', slot('stale', ['shots/2/qwen/first_frame.png']))).toBe(true);
    expect(canRedo('keyframes', slot('accepted', ['shots/2/qwen/first_frame.png']))).toBe(false);
    expect(canRedo('keyframes', slot('planned', []))).toBe(false);
    expect(canRedo('clips', slot('rendered', ['shots/2/kling/video.mp4']))).toBe(true);
  });

  it('does not offer one where a redraw has nothing to redraw from', () => {
    expect(canRedo('portraits', {shot: null, state: 'rendered', artifacts: ['character_portraits/x/0_A/front.png']} as AcceptanceSlot)).toBe(false);
    expect(canRedo('final_video', {shot: null, state: 'rendered', artifacts: ['final_video.mp4']} as AcceptanceSlot)).toBe(false);
  });

  it('maps a stage to the phase that redraws it', () => {
    expect(redoPhase('keyframes')).toBe('stills');
    expect(redoPhase('clips')).toBe('video');
    expect(redoPhase('portraits')).toBeNull();
  });
});

describe('slot titles', () => {
  it('numbers a flat script slot', () => {
    expect(slotTitle('keyframes', {shot: '5', state: 'rendered', artifacts: []})).toBe('Shot 6');
  });

  it('names the scene for an idea slot, which nests shots under scenes', () => {
    expect(slotTitle('keyframes', {shot: 'scene_1/3', state: 'rendered', artifacts: []})).toBe('Scene 2 · Shot 4');
  });
});

describe('redrawing one slot from its card', () => {
  it('asks for that shot alone, with the note against it', () => {
    const text = redoOneText('script2video', 'keyframes', {shot: '4', state: 'rejected', artifacts: ['shots/4/first_frame.png'], reason: 'the door is open here and shut everywhere else'});

    expect(text).toContain('Redraw exactly these shots in script2video in stills, and no others');
    // The card's title and the slot key are both stated: one is 1-based, the other is not.
    expect(text).toContain('Shot 5 (slot key "4"): the door is open here and shut everywhere else');
    expect(text).toContain('render_mode="script2video"');
    // The note guard is lifted: this request names the slot itself, and the guard exists for a
    // person reading the card's 1-based name as a slot key.
    expect(text).toContain('Run vimax_render_video with exactly these arguments and no others: render_mode="script2video", redo_shots=["4"], stop_after="stills", allow_unreviewed_redo=true.');
    expect(text).toContain('a redraw never moves the sequence to another root');
  });

  it('quotes the note a redraw was last asked for, not just a live rejection', () => {
    const text = redoOneText('script2video', 'keyframes', {
      shot: '2', state: 'rendered', artifacts: ['shots/2/first_frame.png'],
      note: 'Wife should match the studio shot.', redoneAt: '2026-09-20T20:00:00',
    });

    expect(text).toContain('Shot 3 (slot key "2"): Wife should match the studio shot.');
  });

  it('redraws a clip-only slot with the video phase', () => {
    expect(redoOneText('script2video', 'clips', {shot: '4', state: 'rendered', artifacts: ['shots/4/video.mp4']})).toContain('stop_after="video"');
  });
});

describe('what the render is doing', () => {
  // The status was written at 15:23:41, so these are that clock plus a number of seconds.
  const at = (seconds: number) => new Date(2026, 8, 20, 15, 23, 41 + seconds);

  it('says what is rendering and how long it has been at it', () => {
    const line = renderStatusLine({status: 'rendering', phase: 'stills', timestamp: '2026-09-20T15:23:41'}, {now: at(240), agentRunning: true});

    expect(line).toBe('Rendering stills — started 15:23, 4 min ago');
  });

  it('does not round a fresh render up to a minute', () => {
    expect(renderStatusLine({status: 'rendering', phase: 'stills', timestamp: '2026-09-20T15:23:41'}, {now: at(10), agentRunning: true})).toContain('just now');
  });

  it('reports a stopped render with the reason the render gave', () => {
    const line = renderStatusLine({status: 'error', error_type: 'redo_wrong_root', error: 'A redraw stays inside the sequence\'s own root.'}, {now: at(60)});

    expect(line).toContain('Render stopped (redo_wrong_root)');
    expect(line).toContain('A redraw stays inside the sequence');
  });

  it('reads a phase that stopped for review as finished, not as still working', () => {
    const waiting = {status: 'rendering', phase: 'stills', awaiting_confirmation: 'video', timestamp: '2026-09-20T16:14:57'};

    expect(renderActivity(waiting, {agentRunning: true, now: at(240)})).toBe('finished');
    expect(renderStatusLine(waiting, {now: at(240)})).toBe('Last render finished 16:14');
  });

  it('does not claim a render is running when the agent it ran in is gone', () => {
    const status = {status: 'rendering', phase: 'stills', timestamp: '2026-09-20T15:23:41'};

    expect(renderStatusLine(status, {now: at(240), agentRunning: false}))
      .toBe('A render started 15:23, 4 min ago and stopped without finishing — the agent is not running');
    expect(renderActivity(status, {agentRunning: false, now: at(240)})).toBe('interrupted');
  });

  it('reads a render as running while the agent is, and lets the caller say it does not know', () => {
    expect(renderActivity({status: 'rendering'}, {agentRunning: true})).toBe('running');
    expect(renderActivity({status: 'rendering'}, {agentRunning: true, lastProgressAt: '2026-09-20T15:23:41', now: at(60)})).toBe('running');
    expect(renderActivity({status: 'rendering'})).toBe('running');
    expect(renderActivity({status: 'rendered'}, {agentRunning: false})).toBe('finished');
    expect(renderActivity({status: 'error'}, {agentRunning: false})).toBe('failed');
    expect(renderActivity({status: 'dependency_missing'})).toBe('blocked');
    expect(renderActivity(undefined, {agentRunning: false})).toBe('idle');
  });

  it('reports a render that has written no progress for minutes as stalled', () => {
    const status = {status: 'rendering', phase: 'stills', timestamp: '2026-09-20T15:23:41'};
    const stale = {agentRunning: true, lastProgressAt: '2026-09-20T15:23:41', now: at(600)};

    expect(renderActivity(status, stale)).toBe('stalled');
    expect(renderStatusLine(status, stale)).toBe('A render started at 15:23 has written nothing for 10 min — it may have stopped');
    // The trail moves as the render works, and then it is running again.
    expect(renderActivity(status, {...stale, lastProgressAt: '2026-09-20T15:33:00'})).toBe('running');

    // A write from before the render began is not progress this render made.
    expect(renderActivity(status, {...stale, lastProgressAt: '2026-09-19T23:56:00'})).toBe('stalled');
    expect(renderActivity(status, {...stale, lastProgressAt: undefined})).toBe('stalled');
  });

  it('shows nothing when there is no status to show, or nothing recognisable', () => {
    expect(renderStatusLine(undefined)).toBe('');
    expect(renderStatusLine('rendering')).toBe('');
    expect(renderStatusLine({status: 'whatever'})).toBe('');
  });
});

describe('what is waiting for a human', () => {
  const stage = (name: AcceptanceStage['stage'], slots: AcceptanceSlot[]): AcceptanceStage =>
    ({stage: name, scope: 'shot', state: 'rendered', accepted: false, slots});

  it('is the tab a render opens on when anything rendered is unreviewed', () => {
    expect(needsReview([stage('keyframes', [{shot: '2', state: 'rendered', artifacts: ['shots/2/first_frame.png']}])])).toBe(true);
    expect(needsReview([stage('keyframes', [{shot: '2', state: 'rejected', artifacts: ['shots/2/first_frame.png']}])])).toBe(true);
    expect(needsReview([stage('keyframes', [{shot: '2', state: 'accepted', artifacts: ['shots/2/first_frame.png']}])])).toBe(false);
  });

  it('is not, when nothing has been rendered yet', () => {
    expect(needsReview([stage('keyframes', [{shot: '2', state: 'planned', artifacts: []}])])).toBe(false);
    expect(needsReview([])).toBe(false);
  });
});

describe('a render that is redrawing a shot', () => {
  it('marks the shot the trail says is being redrawn', () => {
    const trail = [
      {timestamp: '2026-09-20T16:13:45', status: 'rendering', phase: 'stills', redone_shots: ['2'], cleared: 3},
      {timestamp: '2026-09-20T16:13:47', status: 'rendering', phase: 'stills'},
    ];

    expect(activeRedraw(trail)).toEqual({stage: 'keyframes', slots: ['2']});
  });

  it('stops marking it once the render ends, however it ends', () => {
    const redraw = {status: 'rendering', phase: 'stills', redone_shots: ['2']};

    expect(activeRedraw([redraw, {status: 'rendering', awaiting_confirmation: 'video'}])).toBeNull();
    expect(activeRedraw([redraw, {status: 'error', error_type: 'acceptance_required'}])).toBeNull();
    expect(activeRedraw([redraw, {status: 'rendered'}])).toBeNull();
  });

  it('marks nothing during an ordinary render, and survives junk in the trail', () => {
    expect(activeRedraw([{status: 'rendering', phase: 'stills'}])).toBeNull();
    expect(activeRedraw([])).toBeNull();
    expect(activeRedraw(undefined)).toBeNull();
    expect(activeRedraw([null, 'nonsense', redrawWithNothing])).toBeNull();
  });

  it('marks the newest redraw when a render redrew one shot after another', () => {
    expect(activeRedraw([
      {status: 'rendering', redone_shots: ['1']},
      {status: 'rendering', awaiting_confirmation: 'video'},
      {status: 'rendering', phase: 'stills', redone_shots: ['4']},
    ])).toEqual({stage: 'keyframes', slots: ['4']});
  });

  it('marks a clip redraw in the clips lane, not the keyframes lane', () => {
    expect(activeRedraw([{status: 'rendering', phase: 'video', redone_shots: ['3']}]))
      .toEqual({stage: 'clips', slots: ['3']});
  });
});

const redrawWithNothing = {status: 'rendering', redone_slots: ['9']};

describe('a render that failed', () => {
  it('reports it with the shots it had cleared, in the lane it was drawing', () => {
    const trail = [
      {timestamp: '2026-09-20T19:26:55', status: 'rendering', phase: 'stills', redone_shots: ['2']},
      {status: 'error', phase: 'stills', redone_shots: ['2'], error: 'HTTP 402: Insufficient credits', error_type: 'render_failed'},
    ];

    expect(failedRedraw(trail)).toEqual({stage: 'keyframes', slots: ['2'], reason: 'HTTP 402: Insufficient credits'});
  });

  it('stops reporting it once a later render succeeds', () => {
    const failed = {status: 'error', phase: 'stills', redone_shots: ['2'], error: 'Insufficient credits'};

    expect(failedRedraw([failed, {status: 'rendering', phase: 'stills', redone_shots: ['2']}])).toBeNull();
    expect(failedRedraw([failed, {status: 'rendering', awaiting_confirmation: 'video'}])).toBeNull();
    expect(failedRedraw([failed, {status: 'rendered'}])).toBeNull();
  });

  it('reports a failure that cleared nothing, and never invents one', () => {
    expect(failedRedraw([{status: 'error', error_type: 'dependency_missing'}])?.slots).toEqual([]);
    expect(failedRedraw([{status: 'rendering', phase: 'stills'}])).toBeNull();
    expect(failedRedraw([])).toBeNull();
    expect(failedRedraw(undefined)).toBeNull();
  });

  it('ends the waiting a click asked for when that render finishes', () => {
    // A whole-phase render names no slots to redraw, so nothing ever shows up as an active
    // redraw for the slot the click named and its card said "Waiting for the render" for good.
    const askedAt = Date.parse('2026-09-23T17:40:00');
    const ended = [
      {timestamp: '2026-09-23T17:39:00', status: 'rendered', phase: 'stills'},
      {timestamp: '2026-09-23T17:41:08', status: 'rendering', phase: 'video'},
      {timestamp: '2026-09-23T17:44:15', status: 'rendered', phase: 'video'},
    ];
    const stillRunning = [
      {timestamp: '2026-09-23T17:39:00', status: 'rendered', phase: 'stills'},
      {timestamp: '2026-09-23T17:41:08', status: 'rendering', phase: 'video'},
    ];

    expect(askedRenderFinished(ended, askedAt)).toBe(true);
    expect(askedRenderFinished(stillRunning, askedAt)).toBe(false);
    // The render before the click ending is not the one the click asked for.
    expect(askedRenderFinished([{timestamp: '2026-09-23T17:39:00', status: 'rendered'}], askedAt)).toBe(false);
    expect(askedRenderFinished([{timestamp: '2026-09-23T17:44:15', status: 'error', error: 'HTTP 402'}], askedAt)).toBe(true);
    expect(askedRenderFinished([], askedAt)).toBe(false);
    expect(askedRenderFinished(undefined, askedAt)).toBe(false);
  });

  it('names a shot the way the card does', () => {
    expect(slotName('keyframes', '2')).toBe('Shot 3');
    expect(slotName('keyframes', 'scene_1/0')).toBe('Scene 2 · Shot 1');
  });
});

describe('a shot plan the user can change', () => {
  const characters = [{idx: 0, name: 'Claude'}, {idx: 1, name: 'Wife'}, {idx: 2, name: 'DeepSeek'}];

  it('shows each character as on or off for the frame', () => {
    expect(characterChips(characters, [0, 2])).toEqual([
      {idx: 0, name: 'Claude', on: true},
      {idx: 1, name: 'Wife', on: false},
      {idx: 2, name: 'DeepSeek', on: true},
    ]);
  });

  it('toggles a character without disturbing the order or repeating one', () => {
    expect(toggleCharacter([2], 0, true)).toEqual([0, 2]);
    expect(toggleCharacter([0, 2], 2, false)).toEqual([0]);
    expect(toggleCharacter([0], 0, true)).toEqual([0]);
    expect(toggleCharacter([0, 1], 1, false)).toEqual([0]);
  });

  it('knows when there is nothing to save', () => {
    const plan = {firstFrame: {description: 'a', visible: [0]}, lastFrame: {description: 'b', visible: [0, 1]}};
    const same = {ffDesc: 'a', lfDesc: 'b', ffVis: [0], lfVis: [0, 1]};

    expect(planDirty(plan, same)).toBe(false);
    expect(planDirty(plan, {...same, ffDesc: 'a '})).toBe(false);
    expect(planDirty(plan, {...same, ffDesc: 'changed'})).toBe(true);
    expect(planDirty(plan, {...same, lfVis: [0, 1, 2]})).toBe(true);
  });
});

describe('a plan that contradicts itself', () => {
  const characters = [{idx: 0, name: 'Claude'}, {idx: 1, name: 'Wife'}, {idx: 2, name: 'DeepSeek'}];
  const plan = (firstDesc: string, firstVisible: number[], lastDesc = 'x', lastVisible: number[] = [0]) =>
    ({characters, firstFrame: {description: firstDesc, visible: firstVisible}, lastFrame: {description: lastDesc, visible: lastVisible}});

  it('catches the shot whose frame describes a character it does not list', () => {
    // Shot 8 of the live session, as the render wrote it.
    const flagged = planInconsistencies(plan(
      'In the living room, a medium-wide shot captures <DeepSeek> entering from the right.',
      [0],
      '…the younger, fitter man with glasses standing beside the woman…',
      [0, 1],
    ));

    expect(flagged).toEqual(['the first frame describes DeepSeek but does not list them as visible']);
  });

  it('reads a character named in brackets too, which is how the planner writes them', () => {
    expect(planInconsistencies(plan('A wide shot. Claude (Claude) sits, the woman (Wife) stands.', [0])))
      .toEqual(['the first frame describes Wife but does not list them as visible']);
  });

  it('says nothing when the frame lists everyone it names', () => {
    expect(planInconsistencies(plan('<DeepSeek> enters beside <Wife>.', [1, 2], 'Everyone stands.', [0, 1, 2]))).toEqual([]);
    expect(planInconsistencies(plan('A shot of the room with nobody in it.', []))).toEqual([]);
  });
});

describe('which rows open on their own', () => {
  it('opens anything with something said about it, or a plan that contradicts itself', () => {
    expect(rowNeedsAttention('accepted', {noted: true})).toBe(true);
    expect(rowNeedsAttention('accepted', {flagged: true})).toBe(true);
    expect(rowNeedsAttention('rendered', {})).toBe(true);
    expect(rowNeedsAttention('rejected', {})).toBe(true);
    // A shot nobody has drawn is the one whose frames are worth reading before they are paid
    // for, so it opens rather than collapsing like an accepted one.
    expect(rowNeedsAttention('planned', {})).toBe(true);
  });

  it('collapses only a shot that is finished', () => {
    expect(rowNeedsAttention('accepted', {})).toBe(false);
    // A planned shot used to collapse too, on the reasoning that there was nothing to look at.
    // There is: its frames are the prompt the image model will be asked for.
  });
});
