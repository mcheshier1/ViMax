import {useCallback, useEffect, useMemo, useState} from 'react';
import {AlertTriangle, Check, ChevronDown, ChevronRight, ChevronUp, Copy, FileJson, Film, Image as ImageIcon, Lock, Play, Plus, RefreshCw, ThumbsDown, Trash2, Undo2, Unlock, Video, ZoomIn} from 'lucide-react';
import {createShot, getArtifacts, getJsonArtifact, getTextArtifact, moveShot, readAcceptance, readContinuity, readHealth, readRemovedShots, readShotPlans, removeShot, restoreShot, updateAcceptance} from './api';
import {visualPromptSource} from './artifactPresentation';
import {MediaPreviewDialog} from './ArtifactViews';
import {ShotPlanPanel} from './ShotPlanPanel';
import {addSuggestionRequest, coverageCounts, coverageNeedsAttention, gapBeats, reviewRequestText, reviewStatusLine, runtimeLine, suggestionAddable, suggestionAnchor, suggestionFraming, suggestionPosition, warnChecks} from './continuity';
import type {ContinuityPayload, ContinuitySuggestion} from './continuity';
import {activeRedraw, askedRenderFinished, blockingSlots, canRedo, clipCostLine, failedRedraw, filmPosition, missingKeyframes as keyframesMissing, missingSlots, moveTarget, planInconsistencies, redoOneText, redoRequestText, rejections, renderActivity, renderClipText, renderRequestText, renderStatusLine, rowNeedsAttention, slotName, slotsInFilmOrder, stageCounts, stageTitle, STAGE_LABELS, STATE_LABELS, slotTitle} from './timeline';
import type {Redraw, RenderActivity, RenderFailure} from './timeline';
import type {AcceptanceSlot, AcceptanceStage, Artifact, JsonValue, RemovedShot, RenderAcceptance, SessionSummary, ShotPlan} from './types';

/**
 * The review surface for a render, laid out as a reel: one shot per row, its frames on the
 * left, everything about it on the right. Accepting writes a lock the render respects, so a
 * phase cannot spend money on material nobody has looked at.
 */
export function TimelineView({session, artifacts, onAskAgent}: {
  session?: SessionSummary;
  artifacts: Artifact[];
  onAskAgent: (text: string) => Promise<void>;
}) {
  const sessionId = session?.sessionId || '';
  const [payload, setPayload] = useState<RenderAcceptance>();
  const [plans, setPlans] = useState<Map<string, ShotPlan>>(new Map());
  const [removed, setRemoved] = useState<RemovedShot[]>([]);
  const [continuity, setContinuity] = useState<ContinuityPayload>();
  const [busy, setBusy] = useState('');
  const [preview, setPreview] = useState<Artifact>();
  const [rejecting, setRejecting] = useState<{stage: AcceptanceStage; slot: AcceptanceSlot} | null>(null);
  const [reason, setReason] = useState('');
  const [error, setError] = useState('');
  const [asked, setAsked] = useState('');
  const [renderStatus, setRenderStatus] = useState<JsonValue>();
  const [agentRunning, setAgentRunning] = useState<boolean>();
  const [redraw, setRedraw] = useState<Redraw | null>(null);
  const [failure, setFailure] = useState<RenderFailure | null>(null);
  const [watchNonce, setWatchNonce] = useState(0);
  // The slot a click just asked for. The agent takes a while to reach the render, and the
  // card has to say something for all of it, not only once the render starts.
  const [askedSlot, setAskedSlot] = useState<{stage: string; slot: string; at: number} | null>(null);
  const [lastProgressAt, setLastProgressAt] = useState('');

  const refresh = useCallback(async () => {
    if (!sessionId) return;
    try {
      const loaded = await readAcceptance(sessionId);
      setPayload(loaded);
      setError('');
      const root = loaded?.root || '';
      const [planList, removedList, continuityPayload] = await Promise.all([
        readShotPlans(sessionId, root).catch(() => undefined),
        readRemovedShots(sessionId, root).catch(() => undefined),
        readContinuity(sessionId, root).catch(() => undefined),
      ]);
      setPlans(new Map((planList?.plans || []).map((plan) => [plan.slot, plan])));
      setRemoved(removedList?.removed || []);
      setContinuity(continuityPayload);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }, [sessionId]);

  useEffect(() => {
    setPayload(undefined);
    setAsked('');
    void refresh();
  }, [refresh]);

  const byPath = useMemo(() => new Map(artifacts.map((artifact) => [artifact.path, artifact])), [artifacts]);
  const statusArtifact = artifacts.find((artifact) => artifact.path.endsWith('render_status.json'));
  const payloadRoot = payload?.root || '';
  // The film's slots in playing order, which is the order the rows are shown in.
  const filmOrder = continuity?.shots ?? [];
  const statusLine = renderStatusLine(renderStatus, {agentRunning, lastProgressAt});
  const activity = renderActivity(renderStatus, {agentRunning, lastProgressAt});
  const rendering = activity === 'running';

  // A render started by this page is only visible if the page keeps looking for it, so this
  // never stops: a render begins while the page is open, and a page open on an idle session
  // has to notice that. The cadence slows when nothing is running, and an ask restarts it.
  useEffect(() => {
    if (!statusArtifact) return;
    const startedAt = Date.now();
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    const read = () => {
      void Promise.all([
        getJsonArtifact(statusArtifact),
        readHealth().catch(() => undefined),
        getArtifacts(sessionId).catch(() => undefined),
        getTextArtifact(sessionId, 'render_events.jsonl').catch(() => ''),
      ])
        .then(([document, health, listing, trail]) => {
          if (cancelled) return;
          const progress = latestRenderedAt(listing?.artifacts, payloadRoot);
          setRenderStatus(document);
          setAgentRunning(health?.agentRunning);
          setLastProgressAt(progress);
          const state = renderActivity(document, {agentRunning: health?.agentRunning, lastProgressAt: progress});
          const rows = parseRenderTrail(trail);
          // The marker follows the trail, not the activity: a redraw that is slow, stalled or
          // dead is still the most important thing on that card, and the badge says which.
          const running = activeRedraw(rows);
          setRedraw(running);
          setFailure(failedRedraw(rows));
          // The render has taken over this slot, so the click's own waiting state ends —
          // and so does a render that has already finished, which a whole-phase pass never
          // shows up as an active redraw for the slot.
          setAskedSlot((current) => {
            if (!current) return current;
            if (running?.stage === current.stage && running.slots.includes(current.slot)) return null;
            return askedRenderFinished(rows, current.at) ? null : current;
          });
          if (state === 'running') {
            // Frames land one at a time, so the cards are pulled in again as they do.
            void refresh();
            timer = setTimeout(read, 4000);
          } else if (askedSlot) {
            // Waiting on the render a click just asked for: look often enough that it
            // shows up as it starts rather than a slow tick later.
            timer = setTimeout(read, 3000);
          } else {
            if (Date.now() - startedAt > 12000) void refresh();
            timer = setTimeout(read, 12000);
          }
        })
        .catch(() => setTimeout(read, 12000));
    };
    read();
    return () => { cancelled = true; clearTimeout(timer); };
  }, [statusArtifact?.path, statusArtifact?.updatedAt, refresh, sessionId, payloadRoot, watchNonce, askedSlot]);

  async function setAccepted(stage: AcceptanceStage, slot: AcceptanceSlot | undefined, accepted: boolean) {
    if (!sessionId) return;
    const key = `${stage.stage}:${slot?.shot ?? 'all'}`;
    setBusy(key);
    setError('');
    try {
      setPayload(await updateAcceptance({sessionId, root: payload?.root || '', stage: stage.stage, shot: slot?.shot ?? undefined, accepted}));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy('');
    }
  }

  async function acceptAll(stage: AcceptanceStage, accepted: boolean) {
    for (const slot of stage.slots) {
      if (slot.state === 'planned' || slot.state === (accepted ? 'accepted' : 'rendered')) continue;
      await setAccepted(stage, slot, accepted);
    }
  }

  async function reject() {
    if (!rejecting || !sessionId) return;
    const {stage, slot} = rejecting;
    const key = `${stage.stage}:${slot.shot ?? 'all'}`;
    setBusy(key);
    setError('');
    try {
      setPayload(await updateAcceptance({sessionId, root: payload?.root || '', stage: stage.stage, shot: slot.shot ?? undefined, accepted: false, reason}));
      setRejecting(null);
      setReason('');
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy('');
    }
  }

  async function ask(text: string, key = 'ask', done = 'Asked the agent — it is working on it now.', slot?: {stage: string; slot: string}) {
    setBusy(key);
    setAsked('');
    setAskedSlot(slot ? {...slot, at: Date.now()} : null);
    try {
      await onAskAgent(text);
      setAsked(done);
      // Watch for the render this may have just started from the first moment, not the
      // next tick of a slow cadence.
      setWatchNonce((current) => current + 1);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy('');
    }
  }

  /** Take a shot out of the film, or put one back, and pull the reel in again either way. */
  async function mutate(action: () => Promise<unknown>, key: string) {
    setBusy(key);
    setError('');
    try {
      await action();
      await refresh();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy('');
    }
  }

  if (!sessionId) return <div className="artifacts-empty"><Film size={24} /><strong>Select a project</strong><span>The render timeline appears here</span></div>;
  if (!payload) return <div className="artifact-document-state">{error ? <span className="is-error">{error}</span> : 'Loading timeline…'}</div>;

  const totals = payload.totals;
  const keyframes = payload.stages.find((stage) => stage.stage === 'keyframes');
  const clipsStage = payload.stages.find((stage) => stage.stage === 'clips');
  const waiters = keyframes ? blockingSlots(keyframes) : [];
  const keyframeSlots = keyframes?.slots ?? [];
  const keyframesAccepted = keyframeSlots.length > 0 && waiters.length === 0;
  const remainingClips = missingSlots(clipsStage);
  // Shots added since the last run have nothing on disk, so they have nothing to accept:
  // they need drawing first, and a phase re-run draws exactly them.
  const missingKeyframes = keyframesMissing(keyframes);
  const rejected = rejections(payload.stages);
  const editingNote = rejecting?.slot.state === 'rejected';
  const clipsReady = keyframesAccepted && remainingClips > 0;
  // "The end" is where the film actually ends, not the highest-numbered shot: a shot appended
  // to the film goes after whatever plays last.
  const lastShot = filmOrder.slice(-1)[0] ?? (keyframeSlots.length ? keyframeSlots[keyframeSlots.length - 1].shot : null);

  return (
    <section className="timeline-view">
      <header className="reel-summary">
        <div className="reel-summary-state">
          <strong>{totals.acceptedKeyframes}/{totals.keyframes} keyframes accepted</strong>
          <span>{keyframeSlots.length} shots · {totals.clipSeconds}s clips · {clipCostLine(remainingClips, totals.clipSeconds, totals.clipCostUsd)}</span>
          {statusLine && (
            <p className={`timeline-render-status${rendering ? ' is-active' : ''}`} role="status">
              {rendering ? <RefreshCw size={12} className="is-spinning" /> : <Film size={12} />} {statusLine}
            </p>
          )}
        </div>
        <div className="reel-summary-actions">
          {rejected.length > 0 && (
            <button className="timeline-action" onClick={() => void ask(redoRequestText(rejected, payload.root))} disabled={Boolean(busy)}>
              <RefreshCw size={14} /> Redo {rejected.length} rejected shot{rejected.length === 1 ? '' : 's'}
            </button>
          )}
          {clipsReady ? (
            <button className="timeline-action is-primary" onClick={() => void ask(renderRequestText('video', payload.root))} disabled={Boolean(busy)}>
              <Play size={14} /> Render clips · {clipCostLine(remainingClips, totals.clipSeconds, totals.clipCostUsd)}
            </button>
          ) : missingKeyframes > 0 ? (
            <button className="timeline-action is-primary" onClick={() => void ask(renderRequestText('stills', payload.root))} disabled={Boolean(busy)}>
              <ImageIcon size={14} /> Render {missingKeyframes} missing keyframe{missingKeyframes === 1 ? '' : 's'}
            </button>
          ) : keyframesAccepted && remainingClips === 0 ? (
            <span className="timeline-blocked"><Check size={13} /> Every clip is rendered — review them below</span>
          ) : (
            <span className="timeline-blocked">
              <Lock size={13} /> {waiters.length === 1 ? '1 shot' : `${waiters.length} shots`} still to accept before clips
            </span>
          )}
          <button className="timeline-action" onClick={() => void refresh()} disabled={Boolean(busy)}>
            <RefreshCw size={14} /> Refresh
          </button>
        </div>
      </header>

      {error && <p className="timeline-error" role="alert">{error}</p>}
      {failure && (
        <p className="timeline-failure" role="alert">
          <AlertTriangle size={13} />
          <span>
            <strong>
              {failure.slots.length
                ? `The last render failed while redrawing ${failure.slots.map((slot) => slotName(failure.stage, slot)).join(', ')}.`
                : 'The last render failed.'}
            </strong>{' '}
            {failure.reason}
          </span>
        </p>
      )}
      {asked && <p className="timeline-notice" role="status">{asked}</p>}

      <CoverageSection
        payload={continuity}
        plans={plans}
        busy={busy}
        onReview={() => void ask(
          reviewRequestText(payload.root),
          'review',
          'Asked the agent to review the timeline against the script. The gaps and suggested shots appear here when the review is written.',
        )}
        onAdd={(suggestion) => void mutate(
          () => createShot(addSuggestionRequest(sessionId, payload.root, suggestion, [...plans.values()])),
          `add-suggestion:${suggestion.id}`,
        )}
        onAddAll={(suggestions) => void mutate(async () => {
          // One at a time, each following the shot the previous one created: suggestions that
          // fill a single gap all name the same anchor, so adding them in parallel would land
          // them in the reverse of the order the review proposed.
          const created: string[] = [];
          for (const [index, suggestion] of suggestions.entries()) {
            const made = await createShot(addSuggestionRequest(
              sessionId,
              payload.root,
              suggestion,
              [...plans.values()],
              suggestionAnchor(index, suggestions, created),
            ));
            created.push(made.created.slot);
          }
        }, 'add-all')}
      />

      {payload.stages.map((stage, index) => {
        const counts = stageCounts(stage);
        const waitersHere = stage.slots.filter((slot) => slot.state !== 'accepted');
        const note = editingNote && rejecting?.stage.stage === stage.stage ? rejecting.slot : null;
        return (
          <section className="reel-section" key={stage.stage}>
            <header className="reel-section-head">
              <h2>{stageTitle(stage, index)}</h2>
              <span className="reel-section-hint">{STAGE_LABELS[stage.stage]?.hint}</span>
              <span className="reel-section-count">{counts.accepted}/{counts.total} accepted</span>
              <div className="reel-section-actions">
                {waitersHere.length > 1 && (
                  <button className="timeline-action" onClick={() => void acceptAll(stage, true)} disabled={Boolean(busy)}>
                    <Check size={13} /> Accept all waiting
                  </button>
                )}
                {counts.accepted > 0 && (
                  <button className="timeline-action is-quiet" onClick={() => void acceptAll(stage, false)} disabled={Boolean(busy)}>
                    <Unlock size={13} /> Unlock all
                  </button>
                )}
                {stage.stage === 'portraits' && stage.slots[0]?.state === 'accepted' && (
                  <button className="timeline-action is-primary" onClick={() => void ask(renderRequestText('stills', payload.root))} disabled={Boolean(busy)}>
                    <Play size={13} /> Render keyframes
                  </button>
                )}
                {stage.stage === 'keyframes' && missingKeyframes > 0 && (
                  <button className="timeline-action is-primary" onClick={() => void ask(renderRequestText('stills', payload.root))} disabled={Boolean(busy)}>
                    <ImageIcon size={13} /> Render {missingKeyframes} missing keyframe{missingKeyframes === 1 ? '' : 's'}
                  </button>
                )}
                {stage.stage === 'keyframes' && keyframesAccepted && remainingClips > 0 && (
                  <button className="timeline-action is-primary" onClick={() => void ask(renderRequestText('video', payload.root))} disabled={Boolean(busy)}>
                    <Play size={13} /> Render clips · {clipCostLine(remainingClips, totals.clipSeconds, totals.clipCostUsd)}
                  </button>
                )}
                {note && (
                  <span className="reel-section-note">rejecting {slotTitle(stage.stage, note)}</span>
                )}
              </div>
            </header>
            {stage.slots.length === 0 ? (
              <p className="timeline-empty">Nothing here yet.</p>
            ) : (
              slotsInFilmOrder(stage.slots, filmOrder).map((slot) => (
                <ReelRow
                  key={`${stage.stage}:${slot.shot ?? 'all'}`}
                  stage={stage}
                  slot={slot}
                  plan={slot.shot === null || slot.shot === undefined ? undefined : plans.get(String(slot.shot))}
                  byPath={byPath}
                  busy={busy}
                  rendering={rendering}
                  sessionId={sessionId}
                  root={payload.root}
                  redrawState={redraw && redraw.stage === stage.stage && redraw.slots.includes(String(slot.shot)) ? activity : null}
                  waiting={Boolean(askedSlot && askedSlot.stage === stage.stage && askedSlot.slot === String(slot.shot))}
                  onToggle={(accepted) => void setAccepted(stage, slot, accepted)}
                  onExamine={setPreview}
                  onReject={() => { setRejecting({stage, slot}); setReason(slot.reason || ''); }}
                  onRedo={(note) => void ask(
                    redoOneText(payload.root, stage.stage, slot, note),
                    `redo:${stage.stage}:${slot.shot ?? 'all'}`,
                    `Asked the agent to redraw ${slotTitle(stage.stage, slot)}. It reports back here when the new frames are on disk.`,
                    {stage: stage.stage, slot: String(slot.shot)},
                  )}
                  onRemove={() => void mutate(() => removeShot({sessionId, root: payload.root, slot: String(slot.shot)}), `remove:${slot.shot}`)}
                  onDuplicate={(brief) => void mutate(() => createShot({sessionId, root: payload.root, after: String(slot.shot), brief}), `add:${slot.shot}`)}
                  onPlanChanged={() => void refresh()}
                  filmOrder={filmOrder}
                  onMove={(direction) => void mutate(
                    () => moveShot({sessionId, root: payload.root, slot: String(slot.shot), direction}),
                    `move:${slot.shot}`,
                  )}
                  onRenderClip={stage.stage === 'clips' && slot.shot != null && slot.artifacts.length === 0 && keyframesAccepted ? () => void ask(
                    renderClipText(payload.root, String(slot.shot)),
                    `clip:${slot.shot}`,
                    `Asked the agent to render the clip for ${slotTitle(stage.stage, slot)}. It appears here when it is on disk.`,
                  ) : undefined}
                  oneClipCost={clipCostLine(1, totals.clipSeconds, totals.clipCostUsd)}
                />
              ))
            )}
            {stage.stage === 'keyframes' && lastShot !== null && lastShot !== undefined && (
              <AddShotRow
                busy={busy}
                disabled={rendering}
                onCreate={(brief) => void mutate(() => createShot({sessionId, root: payload.root, after: String(lastShot), brief}), 'add:end')}
              />
            )}
          </section>
        );
      })}

      {removed.length > 0 && (
        <section className="reel-section">
          <header className="reel-section-head">
            <h2>Removed shots</h2>
            <span className="reel-section-hint">out of the film, kept on disk, restorable</span>
            <span className="reel-section-count">{removed.length}</span>
          </header>
          {removed.map((entry) => (
            <article className="reel-row is-removed" key={entry.slot}>
              <div className="reel-strip">
                <div className="reel-strip-missing">removed</div>
                <p className="reel-strip-meta"><span>{entry.files} file{entry.files === 1 ? '' : 's'}</span><span>slot {entry.slot}</span></p>
              </div>
              <div className="reel-body">
                <header className="reel-row-head">
                  <h3>{slotName('keyframes', entry.slot)}</h3>
                  <span className="timeline-state">removed</span>
                  <span className="reel-row-hint">
                    {entry.hasBrief ? 'its brief and frames are kept' : 'its frames are kept'}
                  </span>
                  <div className="reel-controls">
                    <button
                      className="timeline-action"
                      disabled={Boolean(busy)}
                      onClick={() => void mutate(() => restoreShot({sessionId, root: payload.root, slot: entry.slot}), `restore:${entry.slot}`)}
                    >
                      <Undo2 size={13} /> Restore
                    </button>
                  </div>
                </header>
              </div>
            </article>
          ))}
        </section>
      )}

      {rejecting && (
        <div className="media-preview-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && setRejecting(null)}>
          <section className="timeline-reject-dialog" role="dialog" aria-modal="true" aria-label={`Reject ${slotTitle(rejecting.stage.stage, rejecting.slot)}`}>
            <header>
              <div>
                <strong>What is wrong with {slotTitle(rejecting.stage.stage, rejecting.slot)}?</strong>
                <span>
                  {editingNote
                    ? 'Your note is loaded below — change it, and it is what the redraw answers.'
                    : 'Written down against the shot, and quoted when it is regenerated.'}
                </span>
              </div>
            </header>
            <textarea
              autoFocus
              value={reason}
              onChange={(event) => setReason(event.target.value)}
              placeholder="e.g. her dress is blue here but pink in every other shot; the backdrop is a plain studio white; he is facing the wrong way"
              rows={4}
            />
            <footer>
              <button className="timeline-action" onClick={() => setRejecting(null)}>Cancel</button>
              <button className="timeline-action is-danger" onClick={() => void reject()} disabled={Boolean(busy) || !reason.trim()}>
                <ThumbsDown size={14} /> {editingNote ? 'Save note' : 'Reject with this note'}
              </button>
            </footer>
          </section>
        </div>
      )}
      {preview && (
        <MediaPreviewDialog
          artifact={preview}
          promptDocument={byPath.get(visualPromptSource(preview.path)?.documentPath || '')}
          onClose={() => setPreview(undefined)}
        />
      )}
    </section>
  );
}

/** One shot: its frames on the left, everything about it on the right. */
function ReelRow({stage, slot, plan, byPath, busy, rendering, sessionId, root, redrawState, waiting, onToggle, onExamine, onReject, onRedo, onRemove, onDuplicate, onPlanChanged, filmOrder, onMove, onRenderClip, oneClipCost}: {
  stage: AcceptanceStage;
  slot: AcceptanceSlot;
  plan?: ShotPlan;
  byPath: Map<string, Artifact>;
  /** The film's slots in playing order, which is what the position and the moves are about. */
  filmOrder: string[];
  onMove: (direction: 'earlier' | 'later') => void;
  /** Set only when this shot's clip can be rendered on its own, which needs its frames accepted. */
  onRenderClip?: () => void;
  /** What one clip costs, for the button's tooltip. */
  oneClipCost: string;
  busy: string;
  rendering: boolean;
  sessionId: string;
  root: string;
  redrawState: RenderActivity | null;
  waiting: boolean;
  onToggle: (accepted: boolean) => void;
  onExamine: (artifact: Artifact) => void;
  onReject: () => void;
  onRedo: (note?: string) => void;
  onRemove: () => void;
  onDuplicate: (brief: string) => void;
  onPlanChanged: () => void;
}) {
  const key = `${stage.stage}:${slot.shot ?? 'all'}`;
  const locked = slot.state === 'accepted';
  const redoing = busy === `redo:${key}`;
  const shotScoped = slot.shot !== null && slot.shot !== undefined;
  // A keyframe is two files. A render that died mid-shot leaves one, and there is nothing
  // complete to review, so it cannot be accepted until a run finishes it.
  const framesIncomplete = stage.stage === 'keyframes' && shotScoped && slot.artifacts.length > 0 && slot.artifacts.length < 2;
  // The plan's *warning* belongs to the keyframes lane: the clips lane would repeat a complaint
  // about frames it does not hold. The plan itself is editable from any lane — a clip's frames
  // are still the prompt a redraw will be asked for.
  const shownPlan = stage.stage === 'keyframes' ? plan : undefined;
  const flagged = shownPlan ? planInconsistencies(shownPlan) : [];
  const noted = Boolean(slot.reason || slot.note);
  const [open, setOpen] = useState(rowNeedsAttention(slot.state, {noted, flagged: flagged.length > 0}));
  const [removing, setRemoving] = useState(false);
  useEffect(() => {
    if (noted || flagged.length) setOpen(true);
  }, [noted, flagged.length]);

  return (
    <article className={`reel-row is-${slot.state}${open ? ' is-open' : ''}`}>
      <div className="reel-strip">
        {slot.artifacts.length > 0 ? (
          <div className="reel-strip-frames">
            {slot.artifacts.map((path) => {
              const known = byPath.get(path);
              const isVideo = path.endsWith('.mp4');
              const label = path.split('/').pop() || path;
              return (
                <figure key={path} title={path}>
                  {!known ? (
                    <span className="timeline-thumb-missing" />
                  ) : (
                    <button className="timeline-thumb" onClick={() => onExamine(known)} aria-label={`Examine ${label}`}>
                      {isVideo
                        ? <video src={url(known)} muted playsInline preload="metadata" />
                        : <img src={url(known)} alt={label} loading="lazy" />}
                      <i><ZoomIn size={14} /></i>
                    </button>
                  )}
                  <figcaption>{isVideo ? <Video size={11} /> : <ImageIcon size={11} />}{label}</figcaption>
                </figure>
              );
            })}
          </div>
        ) : (
          <div className="reel-strip-missing">{slot.state === 'planned' ? 'nothing yet' : 'no files'}</div>
        )}
        <p className="reel-strip-meta">
          <span>{framesIncomplete ? `${slot.artifacts.length} of 2 frames` : `${slot.artifacts.length} file${slot.artifacts.length === 1 ? '' : 's'}`}</span>
          {shotScoped && <span>slot {slot.shot}</span>}
        </p>
      </div>

      <div className="reel-body">
        <header className="reel-row-head">
          <button className="reel-toggle" onClick={() => setOpen((value) => !value)} aria-expanded={open}>
            {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
            <h3>{slotTitle(stage.stage, slot)}</h3>
          </button>
          {redrawState || waiting ? (
            <span className={`timeline-state is-redrawing${redrawState === 'interrupted' ? ' is-stopped' : ''}`}>
              <RefreshCw size={11} className={redrawState === 'interrupted' ? undefined : 'is-spinning'} />
              {redrawState === 'interrupted' ? 'Redraw stopped' : redrawState === 'stalled' ? 'Redrawing, no progress' : waiting && !redrawState ? 'Waiting for the render' : 'Redrawing'}
            </span>
          ) : (
            <span className={`timeline-state is-${slot.state}`}>{STATE_LABELS[slot.state]}</span>
          )}
          {flagged.map((flag) => (
            <span className="reel-flag" key={flag}><AlertTriangle size={11} /> {flag}</span>
          ))}
          {!flagged.length && shownPlan && !open && <span className="reel-row-hint">plan is consistent</span>}
          {shotScoped && stage.stage === 'keyframes' && filmOrder.length > 0 && (
            <span className="reel-row-hint">{filmPosition(filmOrder, String(slot.shot))}</span>
          )}
          {/* A shot nothing has been drawn for has no frames to show, so its brief is the
              only thing that says what was added. */}
          {slot.artifacts.length === 0 && plan?.brief && <span className="reel-row-brief">{plan.brief}</span>}
          <div className="reel-controls">
            <button
              className="timeline-action"
              onClick={() => onToggle(!locked)}
              disabled={slot.state === 'planned' || busy === key || (framesIncomplete && !locked)}
              title={framesIncomplete ? 'This shot has one of its two frames; render it again to finish it before accepting' : undefined}
            >
              {locked ? <><Lock size={13} /> Accepted</> : <><Unlock size={13} /> Accept</>}
            </button>
            <button className="timeline-action" onClick={onReject} disabled={slot.state === 'planned' || busy === key || Boolean(redrawState)}>
              <ThumbsDown size={13} /> {slot.state === 'rejected' ? 'Edit note' : 'Reject'}
            </button>
            {stage.stage === 'keyframes' && shotScoped && (
              <button className="timeline-action" onClick={() => setRemoving(true)} disabled={Boolean(busy) || rendering} title="Take this shot out of the film">
                <Trash2 size={13} /> Remove
              </button>
            )}
            {stage.stage === 'keyframes' && shotScoped && filmOrder.includes(String(slot.shot)) && (
              <>
                <button
                  className="timeline-action is-quiet"
                  onClick={() => onMove('earlier')}
                  disabled={Boolean(busy) || rendering || !moveTarget(filmOrder, String(slot.shot), 'earlier')}
                  title="Move this shot one place earlier in the film"
                >
                  <ChevronUp size={13} /> Earlier
                </button>
                <button
                  className="timeline-action is-quiet"
                  onClick={() => onMove('later')}
                  disabled={Boolean(busy) || rendering || !moveTarget(filmOrder, String(slot.shot), 'later')}
                  title="Move this shot one place later in the film"
                >
                  <ChevronDown size={13} /> Later
                </button>
              </>
            )}
            {onRenderClip && (
              <button
                className="timeline-action is-primary"
                onClick={onRenderClip}
                disabled={Boolean(busy) || rendering}
                title={`Render just this shot's clip · ${oneClipCost}`}
              >
                <Video size={13} /> Render this clip
              </button>
            )}
            {canRedo(stage.stage, slot) && (
              <button className="timeline-action is-primary" onClick={() => onRedo()} disabled={Boolean(busy) || rendering || waiting} title="Redraw just this shot">
                <RefreshCw size={13} className={redoing || redrawState || waiting ? 'is-spinning' : undefined} /> {redoing ? 'Asking…' : 'Redraw'}
              </button>
            )}
          </div>
        </header>

        {slot.state === 'rejected' && slot.reason && (
          <p className="timeline-reason"><ThumbsDown size={12} /> {slot.reason}</p>
        )}
        {slot.state !== 'rejected' && slot.note && (
          <p className="timeline-note"><RefreshCw size={12} /> Redrawn to fix: {slot.note}</p>
        )}
        {(redrawState || waiting) && (
          <p className="timeline-redraw-note">
            <RefreshCw size={12} className={redrawState === 'interrupted' ? undefined : 'is-spinning'} />
            {redrawState === 'interrupted'
              ? ' The render stopped before it finished this shot. Redraw it again to replace what is missing.'
              : redrawState === 'stalled'
                ? ' The render has written nothing for a while — it may have stalled.'
                : ' The render is drawing this shot again now — the new frames appear here as they land.'}
          </p>
        )}

        {removing && (
          <div className="plan-remove-confirm" role="alertdialog" aria-label={`Remove ${slotTitle(stage.stage, slot)}`}>
            <p>
              Take {slotTitle(stage.stage, slot)} out of the film? Its camera stops listing it, so it is not rendered and not
              counted in the runtime. Its frames and brief are kept under <code>.removed_shots/</code>, so this can be undone.
            </p>
            <div className="plan-actions">
              <button className="timeline-action" onClick={() => setRemoving(false)}>Keep it</button>
              <button className="timeline-action is-danger" onClick={() => { setRemoving(false); onRemove(); }} disabled={Boolean(busy)}>
                <Trash2 size={13} /> Remove the shot
              </button>
            </div>
          </div>
        )}

        {open && shotScoped && (
          <ShotPlanPanel
            sessionId={sessionId}
            root={root}
            slot={String(slot.shot)}
            stage={stage.stage}
            plan={plan}
            onRedraw={(note) => onRedo(note)}
            onChanged={onPlanChanged}
          />
        )}
        {open && shotScoped && stage.stage === 'keyframes' && (
          <footer className="reel-foot">
            <button
              className="timeline-action"
              disabled={Boolean(busy) || rendering}
              title="Add a shot after this one, starting from this shot's plan"
              onClick={() => onDuplicate(plan?.brief || '')}
            >
              <Copy size={13} /> Duplicate as a new shot
            </button>
            <span className="reel-row-hint">the new shot starts as a copy of this one; edit its frames before rendering it</span>
          </footer>
        )}
        {open && !shotScoped && (
          <p className="reel-row-hint">The whole sequence in one slot: accept it, and the render may move on.</p>
        )}
      </div>
    </article>
  );
}

/**
 * The script-coverage review: what the film no longer plays, and the shots that would fix it.
 *
 * The review is the agent's; this shows it and turns a suggested shot into the same create
 * request the reel's own "add a shot" uses, so a suggestion is added the way any shot is.
 */
function CoverageSection({payload, plans, busy, onReview, onAdd, onAddAll}: {
  payload?: ContinuityPayload;
  plans: Map<string, ShotPlan>;
  busy: string;
  onReview: () => void;
  onAdd: (suggestion: ContinuitySuggestion) => void;
  onAddAll: (suggestions: ContinuitySuggestion[]) => void;
}) {
  const review = payload?.review ?? null;
  const counts = coverageCounts(review);
  const gaps = gapBeats(review);
  const warnings = warnChecks(review);
  const shots = payload?.shots ?? [];
  const runtime = runtimeLine(shots.length, payload?.clipSeconds ?? 0);
  const asking = busy === 'review';
  // This card is about the whole film, so it sits above the shots it is about. It opens
  // itself when there is something to act on, and stays open once the user opens it.
  const [opened, setOpened] = useState(false);
  // Edits to a suggestion, and which one is open: what gets added is what the card shows,
  // so a suggestion can be rewritten before it becomes a shot.
  const [drafts, setDrafts] = useState<Record<string, ContinuitySuggestion>>({});
  const [openEditor, setOpenEditor] = useState<string | null>(null);
  const open = opened || coverageNeedsAttention(payload ?? null);
  // A review describes the timeline it was written against. Adding shots for gaps it named is
  // only right while that timeline is still the one on screen.
  const stale = payload?.stale ?? false;
  // Bulk add uses what the card shows, so a suggestion edited in place is added as edited.
  const suggestions = (review?.suggestions ?? []).map((item) => drafts[item.id] ?? item);
  return (
    <section className="reel-section coverage-section">
      <header className="reel-section-head">
        <button className="reel-toggle" onClick={() => setOpened((value) => !value)} aria-expanded={open}>
          {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          <h2>Script coverage</h2>
        </button>
        <span className="reel-section-hint">{reviewStatusLine(payload ?? null, {asking})}</span>
        {review && counts.total > 0 && <span className="reel-section-count">{counts.covered}/{counts.total} beats</span>}
        <div className="reel-section-actions">
          {runtime && <span className="coverage-runtime">{runtime}</span>}
          <button className="timeline-action" onClick={onReview} disabled={Boolean(busy)}>
            {asking ? <RefreshCw size={13} className="is-spinning" /> : <FileJson size={13} />} {review ? 'Review again' : 'Review timeline'}
          </button>
        </div>
      </header>

      {open && (
        <>
          {!review && (
            <p className="timeline-empty">
              Nothing has checked this timeline against the script yet. Reviewing lists the beats the film
              no longer plays, and proposes shots to fix them.
            </p>
          )}

          {warnings.length > 0 && (
            <ul className="coverage-checks">
              {warnings.map((check) => (
                <li className="coverage-check" key={`${check.id}:${check.message}`}>
                  <AlertTriangle size={12} />
                  <span>{check.message}</span>
                </li>
              ))}
            </ul>
          )}

          {gaps.length > 0 && (
            <div className="coverage-gaps">
              {gaps.map((beat) => (
                <article className={`coverage-gap is-${beat.status}`} key={`${beat.index}:${beat.text}`}>
                  <span className="coverage-gap-status">{beat.status}</span>
                  <p className="coverage-gap-text">{beat.text}</p>
                  <p className="coverage-gap-where">
                    {beat.covered_by.length
                      ? `played by ${beat.covered_by.map((shot) => slotName('keyframes', String(shot))).join(', ')}`
                      : 'no shot plays this'}
                    {beat.note ? ` — ${beat.note}` : ''}
                  </p>
                </article>
              ))}
            </div>
          )}

          {review && review.suggestions.length > 0 && (
            <div className="coverage-suggestions">
              <div className="coverage-suggestions-head">
                <h3>Suggested shots</h3>
                <button
                  className="timeline-action is-primary"
                  onClick={() => onAddAll(suggestions)}
                  disabled={Boolean(busy) || stale || suggestions.some((item) => !suggestionAddable(item, shots))}
                  title={stale
                    ? 'The timeline changed since this review — review it again before filling its gaps'
                    : 'Add every suggested shot, in the order proposed'}
                >
                  <Plus size={13} /> Add all {suggestions.length}
                </button>
              </div>
              {review.suggestions.map((suggestion) => {
                // The draft is the suggestion until it is edited: what gets added is what the
                // card shows, so the frame descriptions can be rewritten before the shot exists.
                const draft = drafts[suggestion.id] ?? suggestion;
                const addable = suggestionAddable(draft, shots);
                const editing = openEditor === suggestion.id;
                const editDraft = (patch: Partial<ContinuitySuggestion>) =>
                  setDrafts((current) => ({...current, [suggestion.id]: {...draft, ...patch}}));
                return (
                  <article className="coverage-suggestion" key={suggestion.id}>
                    <header className="coverage-suggestion-head">
                      <strong>{suggestion.title || 'Suggested shot'}</strong>
                      <span className="coverage-suggestion-where">{suggestionPosition(draft)}</span>
                      {draft.characters.length > 0 && (
                        <span className="coverage-suggestion-chars">{draft.characters.join(', ')}</span>
                      )}
                      {suggestionFraming(draft) && <span className="coverage-suggestion-size">{suggestionFraming(draft)}</span>}
                      <div className="coverage-suggestion-actions">
                        <button className="timeline-action is-quiet" onClick={() => setOpenEditor(editing ? null : suggestion.id)}>
                          {editing ? 'Close' : 'Edit'}
                        </button>
                        <button
                          className="timeline-action is-primary"
                          onClick={() => onAdd(draft)}
                          disabled={Boolean(busy) || !addable || stale}
                          title={stale
                            ? 'The timeline changed since this review — review it again before filling its gaps'
                            : addable ? 'Add this shot to the film' : 'A shot needs a description, and a shot to sit after that the film still has.'}
                        >
                          <Plus size={13} /> Add shot
                        </button>
                      </div>
                    </header>
                    {editing ? (
                      <div className="coverage-editor">
                        <label className="coverage-field">
                          <span>What the shot shows</span>
                          <textarea rows={2} value={draft.visual_desc} onChange={(event) => editDraft({visual_desc: event.target.value})} />
                        </label>
                        <label className="coverage-field">
                          <span>First frame — what the image is drawn from</span>
                          <textarea rows={3} value={draft.frames.first} onChange={(event) => editDraft({frames: {...draft.frames, first: event.target.value}})} />
                        </label>
                        <label className="coverage-field">
                          <span>Last frame — what the image is drawn from</span>
                          <textarea rows={3} value={draft.frames.last} onChange={(event) => editDraft({frames: {...draft.frames, last: event.target.value}})} />
                        </label>
                        <label className="coverage-field">
                          <span>Dialogue</span>
                          <textarea rows={2} value={draft.audio_desc} onChange={(event) => editDraft({audio_desc: event.target.value})} />
                        </label>
                        {!addable && <p className="coverage-notes">A shot needs a description and a shot to sit after.</p>}
                      </div>
                    ) : (
                      <>
                        <p className="coverage-suggestion-text">{draft.visual_desc}</p>
                        {draft.audio_desc && <p className="coverage-suggestion-audio">{draft.audio_desc}</p>}
                        {draft.rationale && <p className="coverage-suggestion-why">{draft.rationale}</p>}
                      </>
                    )}
                  </article>
                );
              })}
            </div>
          )}

          {review?.notes && <p className="coverage-notes">{review.notes}</p>}
        </>
      )}
    </section>
  );
}

/** Add a shot at the end of the film. */
function AddShotRow({busy, disabled, onCreate}: {busy: string; disabled: boolean; onCreate: (brief: string) => void}) {
  const [brief, setBrief] = useState('');
  return (
    <article className="reel-row is-insert">
      <div className="reel-strip">
        <div className="reel-strip-missing">new shot</div>
        <p className="reel-strip-meta"><span>no files yet</span></p>
      </div>
      <div className="reel-body">
        <header className="reel-row-head">
          <h3><Plus size={13} /> Add a shot at the end</h3>
          <span className="reel-row-hint">it copies the last shot's plan, then you edit its frames</span>
          <div className="reel-controls">
            <button className="timeline-action is-primary" disabled={Boolean(busy) || disabled || !brief.trim()} onClick={() => { onCreate(brief.trim()); setBrief(''); }}>
              <Plus size={13} /> Add shot
            </button>
          </div>
        </header>
        <input
          className="plan-note"
          value={brief}
          onChange={(event) => setBrief(event.target.value)}
          placeholder="What happens in it, in a sentence"
          aria-label="What happens in the new shot"
        />
        <p className="reel-row-hint">Only the new shot is rendered — nothing already accepted is redrawn.</p>
      </div>
    </article>
  );
}

/** The render's trail, one JSON row per line; a half-written row is skipped. */
function parseRenderTrail(text: string): unknown[] {
  return text.split('\n').filter((line) => line.trim()).flatMap((line) => {
    try {
      return [JSON.parse(line)];
    } catch {
      return [];
    }
  });
}

/**
 * When the render last produced anything: the newest write under the root it renders into.
 * A render holds no open handle the browser can see, so what it has written is the
 * evidence that it is still working.
 */
function latestRenderedAt(artifacts: Artifact[] | undefined, root: string): string {
  if (!root) return '';
  return (artifacts || [])
    .filter((artifact) => artifact.path.startsWith(`${root}/`))
    .reduce((newest, artifact) => (artifact.updatedAt > newest ? artifact.updatedAt : newest), '');
}

function url(artifact: Artifact): string {
  const separator = artifact.url.includes('?') ? '&' : '?';
  return `${artifact.url}${separator}updated=${encodeURIComponent(artifact.updatedAt)}`;
}
