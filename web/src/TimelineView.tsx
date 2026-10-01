import {useEffect, useMemo, useRef, useState} from 'react';
import {AlertTriangle, Check, ChevronDown, ChevronRight, ChevronUp, Copy, FileJson, Film, Image as ImageIcon, Lock, Plus, RefreshCw, Trash2, Undo2, Unlock, ZoomIn} from 'lucide-react';
import {activeRedraw, askedRenderFinished, failedRedraw, filmPosition, moveTarget, planInconsistencies, redoRequestText, renderActivity, renderStatusLine, slotName, slotsInFilmOrder, STATE_LABELS} from './timeline';
import {ShotPlanPanel, filmDraftToken, hasFilmDrafts, hasSessionFilmDrafts} from './ShotPlanPanel';
import type {ShotPlanHandle} from './ShotPlanPanel';
import {addSuggestionRequest, coverageCounts, coverageNeedsAttention, gapBeats, reviewRequestText, reviewStatusLine, runtimeLine, suggestionAddable, suggestionAnchor, suggestionFraming, suggestionPosition, warnChecks} from './continuity';
import type {ContinuityPayload, ContinuitySuggestion} from './continuity';
import {createShot, getAgentConfig, moveShot, removeShot, restoreShot, saveAgentConfig, updateAcceptance} from './api';
import type {AcceptanceSlot, AcceptanceStageName, Artifact, ConfigSection, SessionSummary, ShotPlan} from './types';
import type {FilmSelection, FilmSessionState} from './filmSession';
import {artifactUrl, thumbnailUrl} from './media';
import {VIDEO_PROVIDER_PRESETS, videoProviderPreset} from './videoPresets';
import './film-workbench.css';

type Filter = 'all' | 'review' | 'changes' | 'missing';
type PreviewMode = 'first' | 'last' | 'clip';
type FilmUi = {selection: string; filter: Filter; preview: PreviewMode; scroll: number};
type ReelShot = {slot: string; frames?: AcceptanceSlot; clip?: AcceptanceSlot; plan?: ShotPlan};
type Generation = {phase: 'stills' | 'video'; scope: 'selected' | 'missing' | 'changes'; slots: string[]; revision: string; drafts: string};
const filters: {key: Filter; label: string}[] = [{key: 'all', label: 'All'}, {key: 'review', label: 'Needs review'}, {key: 'changes', label: 'Needs changes'}, {key: 'missing', label: 'Missing'}];

type TimelineViewProps = {
  session?: SessionSummary;
  artifacts: Artifact[];
  film: FilmSessionState;
  active?: boolean;
  onAskAgent: (text: string, restartAgent?: boolean) => Promise<void>;
  onSelectionChange?: (selection: FilmSelection | null) => void;
};

export function TimelineView(props: TimelineViewProps) {
  return <FilmWorkbench key={props.session?.sessionId || 'empty'} {...props} />;
}

function FilmWorkbench({session, artifacts, film, active = true, onAskAgent, onSelectionChange}: TimelineViewProps) {
  const sessionId = session?.sessionId || '';
  const uiKey = `vimax:film-ui:${sessionId}`;
  const [ui, setUi] = useState<FilmUi>(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(uiKey) || 'null') as FilmUi | null;
      if (saved) return {selection: typeof saved.selection === 'string' ? saved.selection : '', filter: filters.some((item) => item.key === saved.filter) ? saved.filter : 'all', preview: ['first', 'last', 'clip'].includes(saved.preview) ? saved.preview : 'first', scroll: Number(saved.scroll) || 0};
    } catch { /* Session remains usable if storage is unavailable. */ }
    return {selection: '', filter: 'all', preview: 'first', scroll: 0};
  });
  const uiRef = useRef(ui);
  const reelRef = useRef<HTMLDivElement>(null);
  const inspector = useRef<ShotPlanHandle>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const mounted = useRef(true);
  const action = useRef(false);
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [dirty, setDirty] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [generation, setGeneration] = useState<Generation | null>(null);
  const generationRef = useRef(generation);
  generationRef.current = generation;
  const liveProgress = useRef(film.progress);
  liveProgress.current = film.progress;
  const [asked, setAsked] = useState<{slots: string[]; stage: string; at: number} | null>(null);
  const [attemptAt, setAttemptAt] = useState<number | null>(null);
  const [videoSettings, setVideoSettings] = useState<ConfigSection>();
  const savedVideoSettings = useRef<ConfigSection>();
  const settingsRead = useRef(0);
  const [imageModel, setImageModel] = useState('');
  const [reviewModel, setReviewModel] = useState('');
  const [settingsError, setSettingsError] = useState('');
  const [settingsLoading, setSettingsLoading] = useState(true);
  const [newBrief, setNewBrief] = useState('');
  const payload = film.data?.acceptance;
  const root = payload?.root || '';
  const plans = useMemo(() => new Map((film.data?.plans || []).map((plan) => [plan.slot, plan])), [film.data?.plans]);
  const byPath = useMemo(() => new Map(artifacts.map((artifact) => [artifact.path, artifact])), [artifacts]);
  const frameStage = payload?.stages.find((stage) => stage.stage === 'keyframes');
  const clipStage = payload?.stages.find((stage) => stage.stage === 'clips');
  const frameMap = useMemo(() => new Map((frameStage?.slots || []).map((slot) => [String(slot.shot), slot])), [frameStage]);
  const clipMap = useMemo(() => new Map((clipStage?.slots || []).map((slot) => [String(slot.shot), slot])), [clipStage]);
  const order = useMemo(() => {
    const active = new Map<string, AcceptanceSlot>();
    for (const slot of [...(frameStage?.slots || []), ...(clipStage?.slots || [])]) if (slot.shot != null) active.set(String(slot.shot), slot);
    return slotsInFilmOrder([...active.values()], film.data?.continuity.shots || []).map((slot) => String(slot.shot));
  }, [frameStage, clipStage, film.data?.continuity.shots]);
  const rows = useMemo<ReelShot[]>(() => order.map((slot) => ({slot, frames: frameMap.get(slot), clip: clipMap.get(slot), plan: plans.get(slot)})), [order, frameMap, clipMap, plans]);
  const matches = (row: ReelShot, filter: Filter) => {
    if (filter === 'all') return true;
    if (filter === 'missing') return (row.frames?.artifacts.length || 0) < 2 || !row.clip?.artifacts.length;
    if (filter === 'changes') return [row.frames, row.clip].some((slot) => slot?.state === 'rejected' || slot?.state === 'stale')
      || Boolean(row.plan && planInconsistencies(row.plan).length);
    return row.frames?.state === 'rendered' && row.frames.artifacts.length >= 2
      || row.clip?.state === 'rendered' && row.clip.artifacts.length > 0;
  };
  const filtered = rows.filter((row) => matches(row, ui.filter));
  const selected = rows.find((row) => row.slot === ui.selection);
  const sessionStage = payload?.stages.find((stage) => `@${stage.stage}` === ui.selection && stage.scope === 'session');
  const reviewStage: AcceptanceStageName = sessionStage?.stage || (ui.preview === 'clip' ? 'clips' : 'keyframes');
  const reviewSlot = sessionStage?.slots[0] || (ui.preview === 'clip' ? selected?.clip : selected?.frames);
  const previewPaths = sessionStage ? sessionStage.slots.flatMap((slot) => slot.artifacts) : ui.preview === 'clip' ? selected?.clip?.artifacts || [] : selected?.frames?.artifacts || [];
  const [sessionMedia, setSessionMedia] = useState(0);
  const selectedPath = sessionStage
    ? previewPaths[Math.min(sessionMedia, Math.max(0, previewPaths.length - 1))]
    : ui.preview === 'clip' ? previewPaths[0]
      : previewPaths.find((path) => path.endsWith(ui.preview === 'last' ? '/last_frame.png' : '/first_frame.png'));
  const selectedArtifact = byPath.get(selectedPath || '');
  const progress = film.progress;
  const trail = useMemo(() => parseRenderTrail(progress?.trail || '').filter((row) => attemptAt === null || renderWrittenAfter(row, attemptAt)), [progress?.trail, attemptAt]);
  const redraw = useMemo(() => activeRedraw(trail), [trail]);
  const failure = useMemo(() => failedRedraw(trail), [trail]);
  const agentRunning = progress?.activeSessionId === sessionId ? progress.agentRunning : false;
  const currentStatus = attemptAt === null || renderWrittenAfter(progress?.status, attemptAt) ? progress?.status : undefined;
  const activity = renderActivity(currentStatus, {agentRunning, lastProgressAt: progress?.lastProgressAt});
  const statusLine = renderStatusLine(currentStatus, {agentRunning, lastProgressAt: progress?.lastProgressAt});
  const rendering = activity === 'running';
  const waiting = Boolean(asked);
  const frozen = Boolean(busy) || rendering || waiting;
  const incomplete = reviewStage === 'keyframes' && (reviewSlot?.artifacts.length || 0) < 2;
  const locked = reviewSlot?.state === 'accepted';
  const selectedIdentity = `${sessionId}:${root}:${selected?.slot || ''}`;

  function remember(patch: Partial<FilmUi>) {
    const next = {...uiRef.current, ...patch};
    uiRef.current = next;
    setUi(next);
    try { localStorage.setItem(uiKey, JSON.stringify(next)); } catch { /* Only the optional view preference is lost. */ }
  }
  function leaveDraft(): boolean {
    return !inspector.current?.dirty() || window.confirm('Keep this unsaved draft and switch views? It stays with this shot and project. Choose Cancel to keep editing; use Discard draft to remove it.');
  }
  function select(slot: string) {
    if (slot === ui.selection || !leaveDraft()) return;
    setDirty(false);
    setSessionMedia(0);
    remember({selection: slot});
  }
  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);
  async function loadSettings() {
    const request = ++settingsRead.current;
    setSettingsLoading(true);
    setSettingsError('');
    try {
      const config = await getAgentConfig();
      if (!mounted.current || request !== settingsRead.current) return;
      savedVideoSettings.current = config.sections.video;
      setVideoSettings(config.sections.video);
      setImageModel(config.sections.image.model);
      setReviewModel(config.sections.llm.model);
    } catch (reason) {
      if (mounted.current && request === settingsRead.current) setSettingsError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      if (mounted.current && request === settingsRead.current) setSettingsLoading(false);
    }
  }
  useEffect(() => {
    if (sessionId) void loadSettings();
    else setSettingsLoading(false);
  }, [sessionId]);
  useEffect(() => {
    if (!active) {
      videoRef.current?.pause();
      setExpanded(false);
    }
  }, [active]);
  useEffect(() => {
    if (!payload || selected || sessionStage || !rows.length) return;
    remember({selection: rows[0].slot});
  }, [payload, selected, sessionStage, rows]);
  useEffect(() => {
    onSelectionChange?.(selected ? {root, slot: selected.slot, label: slotName('keyframes', selected.slot)} : null);
  }, [selected?.slot, root, onSelectionChange]);
  useEffect(() => {
    if (reelRef.current) reelRef.current.scrollTop = uiRef.current.scroll;
  }, [Boolean(payload)]);
  useEffect(() => {
    if (asked && (askedRenderFinished(trail, asked.at) || (redraw?.stage === asked.stage && asked.slots.some((slot) => redraw.slots.includes(slot))))) setAsked(null);
  }, [trail, redraw, asked]);
  useEffect(() => {
    const guard = (event: BeforeUnloadEvent) => {
      if (!hasSessionFilmDrafts(sessionId)) return;
      event.preventDefault(); event.returnValue = '';
    };
    window.addEventListener('beforeunload', guard);
    return () => window.removeEventListener('beforeunload', guard);
  }, [sessionId]);
  useEffect(() => {
    if (!expanded) return;
    const escape = (event: KeyboardEvent) => { if (event.key === 'Escape') setExpanded(false); };
    window.addEventListener('keydown', escape);
    return () => window.removeEventListener('keydown', escape);
  }, [expanded]);
  useEffect(() => {
    if (!generation && !expanded) return;
    const previousFocus = document.activeElement as HTMLElement | null;
    const dialog = document.querySelector<HTMLElement>(generation ? '.film-generation' : '.film-preview.is-expanded');
    if (!dialog) return;
    const focusable = () => [...dialog.querySelectorAll<HTMLElement>('button:not(:disabled), select:not(:disabled), textarea:not(:disabled), input:not(:disabled), [tabindex="0"]')];
    focusable()[0]?.focus();
    const trap = (event: KeyboardEvent) => {
      if (event.key !== 'Tab') return;
      const controls = focusable();
      const first = controls[0];
      const last = controls[controls.length - 1];
      if (!first || !last) { event.preventDefault(); return; }
      if (event.shiftKey && (document.activeElement === first || !dialog.contains(document.activeElement))) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && (document.activeElement === last || !dialog.contains(document.activeElement))) { event.preventDefault(); first.focus(); }
    };
    document.addEventListener('keydown', trap);
    return () => { document.removeEventListener('keydown', trap); previousFocus?.focus(); };
  }, [Boolean(generation), expanded]);

  async function mutate(key: string, operation: () => Promise<unknown>): Promise<boolean> {
    if (action.current || !sessionId || !payload) return false;
    action.current = true; setBusy(key); setError('');
    try {
      await operation();
      if (!mounted.current) return false;
      await film.refresh();
      return mounted.current;
    } catch (reason) {
      if (mounted.current) setError(reason instanceof Error ? reason.message : String(reason));
      return false;
    } finally {
      action.current = false;
      if (mounted.current) setBusy('');
    }
  }
  async function accept(accepted: boolean, next = false) {
    if (!reviewSlot || (accepted && incomplete) || dirty) return;
    let nextSlot = '';
    const succeeded = await mutate('accept', async () => {
      const updated = await updateAcceptance({sessionId, root, stage: reviewStage, shot: reviewSlot.shot ?? undefined, accepted});
      if (next && selected) {
        const stage = updated.stages.find((stage) => stage.stage === reviewStage);
        const index = order.indexOf(selected.slot);
        nextSlot = [...order.slice(index + 1), ...order.slice(0, index)].find((slot) => {
          const candidate = stage?.slots.find((item) => String(item.shot) === slot);
          return candidate && candidate.state === 'rendered' && candidate.artifacts.length >= (reviewStage === 'keyframes' ? 2 : 1);
        }) || '';
      }
    });
    if (succeeded && nextSlot) { setDirty(false); remember({selection: nextSlot}); }
    else if (succeeded && next) setNotice(`No more rendered ${reviewStage === 'clips' ? 'clips' : 'frame pairs'} waiting for review.`);
  }
  async function saveFeedback(note: string) {
    if (!reviewSlot) throw new Error('The review slot is no longer available.');
    const succeeded = await mutate('feedback', () => updateAcceptance({sessionId, root, stage: reviewStage, shot: reviewSlot.shot ?? undefined, accepted: false, reason: note}));
    if (!succeeded) throw new Error('Feedback was not saved. Your local draft is kept.');
  }
  function generationSlots(phase: Generation['phase'], scope: Generation['scope']): string[] {
    return rows.filter((row) => {
      const target = phase === 'stills' ? row.frames : row.clip;
      if (target?.state === 'accepted') return false;
      if (phase === 'stills' && row.clip?.state === 'rejected' && row.frames?.state !== 'rejected') return false;
      if (scope === 'selected') return row.slot === selected?.slot;
      if (scope === 'missing') return (target?.artifacts.length || 0) < (phase === 'stills' ? 2 : 1);
      return target?.state === 'rejected' || target?.state === 'stale';
    }).map((row) => row.slot);
  }
  function openGeneration(phase: Generation['phase'], scope: Generation['scope']) {
    if (!generation) void loadSettings();
    const slots = generationSlots(phase, scope);
    setGeneration({phase, scope, slots, revision: progress?.revision || '', drafts: filmDraftToken(sessionId, root, slots)});
  }
  async function submitGeneration() {
    const decision = generation;
    if (!decision || !decision.slots.length || frozen || settingsLoading || settingsError || (decision.phase === 'video' ? !videoSettings?.model : !imageModel)) return;
    if (hasFilmDrafts(sessionId, root, decision.slots)) { setError('Save or discard each targeted shot’s local draft before generating. No request was sent.'); return; }
    if (decision.revision !== (progress?.revision || '') || decision.drafts !== filmDraftToken(sessionId, root, decision.slots)) {
      setError('The film or a draft changed while you reviewed this decision. Review the refreshed scope and submit again.');
      openGeneration(decision.phase, decision.scope); return;
    }
    const frameGate = decision.phase === 'video' && decision.slots.some((slot) => frameMap.get(slot)?.state !== 'accepted');
    const portraitGate = decision.phase === 'stills' && payload?.stages.some((stage) => stage.stage === 'portraits' && stage.slots.some((slot) => slot.artifacts.length > 0 && slot.state !== 'accepted'));
    if (frameGate || portraitGate) { setError(frameGate ? 'Accept both frames for every targeted shot before rendering its clip.' : 'Review and accept portraits before generating frames.'); return; }
    const video = {...videoSettings};
    const stage = decision.phase === 'video' ? 'clips' : 'keyframes';
    const text = redoRequestText(decision.slots.map((shot) => ({shot, stage, title: slotName(stage, shot), reason: (stage === 'clips' ? clipMap : frameMap).get(shot)?.reason || (stage === 'clips' ? clipMap : frameMap).get(shot)?.note || ''})), root)
      + `\nUse the configured ${decision.phase === 'video' ? `video model ${JSON.stringify(video.model)} from provider ${JSON.stringify(video.provider || '')}` : `image model ${JSON.stringify(imageModel)}`}. Stop after the stated phase; do not retry a failure or begin another paid phase without a new explicit decision.`;
    const succeeded = await mutate('generate', async () => {
      const config = await getAgentConfig();
      if (!mounted.current || generationRef.current !== decision || decision.revision !== (liveProgress.current?.revision || '') || decision.drafts !== filmDraftToken(sessionId, root, decision.slots)) throw new Error('The generation decision changed. Review it again.');
      if (decision.phase === 'stills' && config.sections.image.model !== imageModel) {
        setImageModel(config.sections.image.model);
        throw new Error('The global image model changed. Review its updated name and submit again.');
      }
      let restart = false;
      if (decision.phase === 'video') {
        const previous = config.sections.video;
        const baseline = savedVideoSettings.current;
        if (baseline && ['provider', 'model', 'base_url', 'resolution', 'effective_clip_seconds'].some((key) => previous[key as keyof ConfigSection] !== baseline[key as keyof ConfigSection])) {
          savedVideoSettings.current = previous;
          setVideoSettings(previous);
          throw new Error('Global video settings changed. Review the updated selection before submitting again.');
        }
        restart = previous.provider !== video.provider || previous.model !== video.model || previous.base_url !== video.base_url || previous.resolution !== video.resolution;
        if (restart) await saveAgentConfig({...config, sections: {...config.sections, video: {...previous, provider: video.provider, model: video.model || previous.model, base_url: video.base_url || previous.base_url, resolution: video.resolution}}});
        savedVideoSettings.current = {...previous, ...video};
      }
      if (!mounted.current || generationRef.current !== decision || decision.revision !== (liveProgress.current?.revision || '') || decision.drafts !== filmDraftToken(sessionId, root, decision.slots)) throw new Error('The film or a draft changed. No generation request was sent.');
      const at = Date.now();
      setNotice('');
      setAttemptAt(at);
      setAsked({slots: decision.slots, stage, at});
      try { await onAskAgent(text, restart); } catch (reason) { if (mounted.current) { setAsked(null); setAttemptAt(null); } throw reason; }
      film.wake();
    });
    if (succeeded) setGeneration(null);
  }
  async function reviewCoverage() {
    if (!window.confirm(`Review this film against the script using ${reviewModel || 'the configured language model'}? This may make a paid language-model call; its cost is unknown. It will not generate media.`)) return;
    await mutate('review', async () => { await onAskAgent(reviewRequestText(root) + '\nReview only. Do not generate media or retry automatically.'); film.wake(); });
  }

  if (!sessionId) return <div className="artifacts-empty"><Film size={24} /><strong>Select a project</strong><span>Your film workbench appears here.</span></div>;
  if (!payload) return <div className="artifact-document-state">{film.error || 'Loading film…'}{film.error && <button onClick={() => void film.refresh().catch(() => {})}>Retry loading</button>}</div>;
  const preset = videoProviderPreset(videoSettings?.provider || '');
  const models = [...(videoSettings?.model && !preset?.models.some((model) => model === videoSettings.model) ? [videoSettings.model] : []), ...(preset?.models || [])];
  const missingCount = rows.filter((row) => matches(row, 'missing')).length;
  const targetHasDraft = generation ? hasFilmDrafts(sessionId, root, generation.slots) : false;
  const generationBlocked = generation?.phase === 'video' ? generation.slots.some((slot) => frameMap.get(slot)?.state !== 'accepted') : payload.stages.some((stage) => stage.stage === 'portraits' && stage.slots.some((slot) => slot.artifacts.length > 0 && slot.state !== 'accepted'));
  const phaseModel = generation?.phase === 'video' ? videoSettings?.model : imageModel;
  const changedVideoModel = videoSettings?.provider !== savedVideoSettings.current?.provider || videoSettings?.model !== savedVideoSettings.current?.model;
  const generationCost = generation?.phase === 'video' && payload.totals.clipCostUsd > 0
    ? `${changedVideoModel ? 'Cost for this new selection is unknown. ' : ''}Saved-settings reference: ≈$${(generation.slots.length * payload.totals.clipSeconds * payload.totals.clipCostUsd).toFixed(2)} for ${generation.slots.length} clip${generation.slots.length === 1 ? '' : 's'} at ${payload.totals.clipSeconds}s. Provider/model changes and other charges can change the actual price.`
    : 'Cost unknown. Image, reference-selection, and provider charges are not quoted by this workbench.';

  return <section className="timeline-view film-workbench">
    <header className="film-toolbar">
      <div><h2>Film</h2><span>{rows.length} shots · {payload.totals.acceptedKeyframes}/{payload.totals.keyframes} frames accepted · {missingCount} incomplete</span></div>
      <div className="film-toolbar-actions">
        <button className="timeline-action" disabled={Boolean(busy)} onClick={() => void film.refresh().catch(() => {})} aria-label="Refresh film"><RefreshCw size={14} />Refresh</button>
        <button className="timeline-action is-primary" disabled={frozen || !rows.length} onClick={() => openGeneration(ui.preview === 'clip' ? 'video' : 'stills', selected ? 'selected' : 'missing')}><Plus size={14} />Generate…</button>
      </div>
    </header>
    {statusLine && <p className={`timeline-render-status${rendering ? ' is-active' : ''}`} role="status">{rendering && <RefreshCw size={12} className="is-spinning" />}{statusLine}</p>}
    {asked && <p className="timeline-render-status is-active" role="status"><RefreshCw size={12} className="is-spinning" />Generation request sent — waiting for the agent to start {asked.stage === 'clips' ? 'clips' : 'frames'} for {asked.slots.map((slot) => slotName('keyframes', slot)).join(', ')}. No automatic retry. <button className="timeline-action" onClick={() => window.dispatchEvent(new CustomEvent('film-open-assistant'))}>Open Assistant</button> <button className="timeline-action" onClick={() => { if (window.confirm('Stop tracking this request? This does not cancel an agent or render. Check Assistant before submitting again to avoid duplicate charges.')) setAsked(null); }}>Clear waiting notice</button></p>}
    {failure && <p className="timeline-failure" role="alert"><AlertTriangle size={14} /><span><strong>{attemptAt === null ? 'The last render failed' : 'This generation failed'}{failure.slots.length ? ` for ${failure.slots.map((slot) => slotName(failure.stage, slot)).join(', ')}` : ''}.</strong> {failure.reason} No automatic retry has been requested.</span><button className="timeline-action" onClick={() => window.dispatchEvent(new CustomEvent('film-open-assistant'))}>Open Assistant</button></p>}
    {(activity === 'stalled' || activity === 'interrupted') && <p className="film-warning">{activity === 'stalled' ? 'The render has stopped writing progress; it may be stalled.' : 'The render was interrupted.'} Inspect the agent response before making another generation decision. <button className="timeline-action" onClick={() => window.dispatchEvent(new CustomEvent('film-open-assistant'))}>Open Assistant</button></p>}
    {(error || film.error) && <p className="timeline-error" role="alert">{error || film.error}</p>}
    {notice && <p className="timeline-notice" role="status">{notice}</p>}
    {!rows.length && <div className="film-empty"><Film size={28} /><h3>Your film starts with a plan</h3><p>Open Assistant to develop your idea and create a script and shot plan. Nothing is generated from this screen automatically.</p><button className="timeline-action is-primary" onClick={() => window.dispatchEvent(new CustomEvent('film-open-assistant'))}>Open Assistant</button></div>}
    <div className="film-layout">
      <aside className="film-reel" aria-label="Film order">
        <div className="film-filters" aria-label="Filter shots">{filters.map((filter) => <button key={filter.key} aria-pressed={ui.filter === filter.key} onClick={() => remember({filter: filter.key})}>{filter.label}<span>{rows.filter((row) => matches(row, filter.key)).length}</span></button>)}</div>
        <div className="film-reel-scroll" ref={reelRef} onScroll={(event) => {
          const scroll = event.currentTarget.scrollTop;
          uiRef.current = {...uiRef.current, scroll};
          try { localStorage.setItem(uiKey, JSON.stringify(uiRef.current)); } catch { /* Optional view preference. */ }
        }}>
          {filtered.map((row) => {
            const thumb = byPath.get(row.frames?.artifacts[0] || row.clip?.artifacts[0] || '');
            const active = redraw?.slots.includes(row.slot) && (activity === 'running' || activity === 'stalled' || activity === 'interrupted');
            return <button key={row.slot} className={`film-shot${ui.selection === row.slot ? ' is-selected' : ''}`} aria-current={ui.selection === row.slot ? 'true' : undefined} onClick={() => select(row.slot)}>
              <span className="film-shot-thumb">{thumb ? <img src={thumbnailUrl(thumb, 240)} loading="lazy" alt="" /> : <ImageIcon size={22} />}</span>
              <span className="film-shot-content"><strong>{slotName('keyframes', row.slot)}<small>{order.indexOf(row.slot) + 1}/{order.length}</small></strong><span className="film-shot-brief">{row.plan?.brief || row.plan?.firstFrame.description || 'No description yet'}</span><span className="film-shot-states"><span className={`is-${row.frames?.state || 'planned'}`}>Frames: {(row.frames?.artifacts.length || 0) < 2 ? `${row.frames?.artifacts.length || 0}/2 missing` : STATE_LABELS[row.frames!.state]}</span><span className={`is-${row.clip?.state || 'planned'}`}>Clip: {row.clip ? STATE_LABELS[row.clip.state] : 'Missing'}</span></span>{active && <span className="film-shot-progress">{activity === 'stalled' ? 'No recent progress' : activity === 'interrupted' ? 'Interrupted' : 'Regenerating'}</span>}{asked?.slots.includes(row.slot) && <span className="film-shot-progress">Waiting for agent</span>}</span>
            </button>;
          })}
          {!filtered.length && rows.length > 0 && <p className="timeline-empty">No shots match this filter. Your selected shot stays open.</p>}
        </div>
        <details className="film-sequence-review"><summary>Portraits & final film</summary>{payload.stages.filter((stage) => stage.scope === 'session').map((stage) => <button key={stage.stage} className={`film-sequence-button${ui.selection === `@${stage.stage}` ? ' is-selected' : ''}`} onClick={() => select(`@${stage.stage}`)}>{stage.stage === 'portraits' ? 'Character portraits' : 'Final film'}<span className={`timeline-state is-${stage.state}`}>{STATE_LABELS[stage.state]}</span></button>)}</details>
        {Boolean(film.data?.removed.length) && <details className="film-removed"><summary>Removed shots ({film.data?.removed.length})</summary><p>Kept on disk and excluded from the film.</p>{film.data?.removed.map((entry) => <div key={entry.slot}><span>{slotName('keyframes', entry.slot)} · {entry.files} files</span><button className="timeline-action" disabled={frozen} onClick={() => void mutate('restore', () => restoreShot({sessionId, root, slot: entry.slot}))}><Undo2 size={12} />Restore</button></div>)}</details>}
      </aside>
      <main className="film-review" aria-label="Selected shot review">
        {selected || sessionStage ? <>
          <header className="film-selection-head"><div><h3>{selected ? slotName('keyframes', selected.slot) : sessionStage?.stage === 'portraits' ? 'Character portraits' : 'Final film'}</h3><span>{selected ? filmPosition(order, selected.slot) : 'Sequence-wide review and approval'}</span></div>{selected && <div className="film-preview-tabs" role="group" aria-label="Preview media">{([{key: 'first', label: 'First frame'}, {key: 'last', label: 'Last frame'}, {key: 'clip', label: 'Clip'}] as const).map((tab) => <button key={tab.key} aria-pressed={ui.preview === tab.key} onClick={() => { if (tab.key === ui.preview || leaveDraft()) remember({preview: tab.key}); }}>{tab.label}</button>)}</div>}</header>
          <div className={`film-preview${expanded ? ' is-expanded' : ''}`} role={expanded ? 'dialog' : undefined} aria-modal={expanded ? true : undefined} aria-label={expanded ? 'Expanded media viewer' : 'Media viewer'}>
            {selectedArtifact ? selectedArtifact.kind === 'video' ? <video ref={videoRef} key={artifactUrl(selectedArtifact)} src={artifactUrl(selectedArtifact)} poster={thumbnailUrl(selectedArtifact)} controls playsInline preload="metadata" /> : <img key={artifactUrl(selectedArtifact)} src={artifactUrl(selectedArtifact)} alt={selected ? `${slotName('keyframes', selected.slot)} ${ui.preview === 'last' ? 'last' : 'first'} frame` : selectedArtifact.name} /> : <div className="film-preview-missing"><ImageIcon size={30} /><span>{ui.preview === 'clip' ? 'No clip yet' : 'This frame has not been generated'}</span></div>}
            {selectedArtifact && <button className="film-expand" aria-label={expanded ? 'Close expanded preview' : 'Expand preview'} onClick={() => setExpanded((value) => !value)}><ZoomIn size={15} />{expanded ? 'Close' : 'Expand'}</button>}
          </div>
          {sessionStage && previewPaths.length > 1 && <div className="film-session-thumbs">{previewPaths.map((path, index) => { const artifact = byPath.get(path); return artifact ? <button key={path} aria-label={`View ${artifact.name}`} aria-pressed={index === sessionMedia} onClick={() => setSessionMedia(index)}><img src={thumbnailUrl(artifact, 160)} loading="lazy" alt={artifact.name} /></button> : null; })}</div>}
          <div className="film-review-actions">
            <span className={`timeline-state is-${reviewSlot?.state || 'planned'}`}>{reviewSlot ? STATE_LABELS[reviewSlot.state] : 'Missing'}</span>
            <button className="timeline-action" disabled={frozen || !reviewSlot?.artifacts.length} onClick={() => inspector.current?.focusFeedback()}>Needs changes</button>
            <button className="timeline-action" disabled={frozen || dirty || !reviewSlot?.artifacts.length || (incomplete && !locked)} onClick={() => void accept(!locked)}>
              {locked ? <><Lock size={13} />Unlock {reviewStage === 'clips' ? 'clip' : reviewStage === 'keyframes' ? 'frames' : 'approval'}</> : <><Unlock size={13} />Accept {reviewStage === 'clips' ? 'clip' : reviewStage === 'keyframes' ? 'frames' : 'approval'}</>}
            </button>
            {selected && <button className="timeline-action is-primary" disabled={frozen || dirty || !reviewSlot?.artifacts.length || incomplete || locked} onClick={() => void accept(true, true)}><Check size={13} />Accept & next</button>}
          </div>
          {incomplete && <p className="film-warning">Both first and last frames must exist before this shot can be accepted.</p>}
          {reviewSlot?.reason && <p className="timeline-reason">Needs changes: {reviewSlot.reason}</p>}
          {reviewSlot?.note && !reviewSlot.reason && <p className="timeline-note">Previous take regenerated to fix: {reviewSlot.note}</p>}
          {selected?.plan && planInconsistencies(selected.plan).map((flag) => <p className="film-warning" key={flag}><AlertTriangle size={13} />{flag}</p>)}
          {sessionStage && <ShotPlanPanel
            key={`${sessionId}:${root}:${sessionStage.stage}`} ref={inspector}
            sessionId={sessionId} root={root} slot={`@${sessionStage.stage}`} stage={sessionStage.stage}
            feedback={reviewSlot?.reason || ''} accepted={locked} disabled={frozen} canRegenerate={false}
            onChanged={film.refresh} onSaveFeedback={saveFeedback}
            onRegenerate={() => openGeneration('stills', 'missing')} onDirtyChange={setDirty}
          />}
          {selected && <>
            <ShotPlanPanel key={selectedIdentity} ref={inspector} sessionId={sessionId} root={root} slot={selected.slot} stage={reviewStage} plan={selected.plan} feedback={reviewSlot?.reason || ''} accepted={selected.frames?.state === 'accepted' || selected.clip?.state === 'accepted'} disabled={frozen} canRegenerate={!frozen && (!locked || dirty)} onChanged={film.refresh} onSaveFeedback={saveFeedback} onRegenerate={() => openGeneration(ui.preview === 'clip' ? 'video' : 'stills', 'selected')} onDirtyChange={setDirty} />
            <details className="film-structure"><summary>Shot structure</summary><p>Order changes invalidate the final film. New shots copy the selected plan; removed media is kept for restoration.</p><div className="plan-actions"><button className="timeline-action" disabled={frozen || !moveTarget(order, selected.slot, 'earlier')} onClick={() => { if (leaveDraft()) void mutate('move', () => moveShot({sessionId, root, slot: selected.slot, direction: 'earlier'})); }}><ChevronUp size={13} />Earlier</button><button className="timeline-action" disabled={frozen || !moveTarget(order, selected.slot, 'later')} onClick={() => { if (leaveDraft()) void mutate('move', () => moveShot({sessionId, root, slot: selected.slot, direction: 'later'})); }}><ChevronDown size={13} />Later</button><button className="timeline-action" disabled={frozen || dirty} onClick={() => void mutate('duplicate', () => createShot({sessionId, root, after: selected.slot, brief: selected.plan?.brief || ''}))}><Copy size={13} />Duplicate</button><button className="timeline-action is-danger" disabled={frozen} onClick={() => { if (window.confirm(`Remove ${slotName('keyframes', selected.slot)} from the film? Its brief, frames and local draft are kept. This invalidates the final film.`)) void mutate('remove', () => removeShot({sessionId, root, slot: selected.slot})); }}><Trash2 size={13} />Remove</button></div><label className="film-feedback"><span>New shot after this one</span><input value={newBrief} onChange={(event) => setNewBrief(event.target.value)} placeholder="What happens in it?" /></label><button className="timeline-action" disabled={frozen || dirty || !newBrief.trim()} onClick={() => void mutate('add', () => createShot({sessionId, root, after: selected.slot, brief: newBrief.trim()})).then((saved) => { if (saved) setNewBrief(''); })}><Plus size={13} />Add shot</button></details>
          </>}
        </> : <div className="film-preview-missing">Select a shot to review its frames, clip and plan.</div>}
      </main>
    </div>
    <CoverageSection payload={film.data?.continuity} plans={plans} busy={frozen ? busy || 'rendering' : ''} onReview={() => void reviewCoverage()} onAdd={(suggestion) => void mutate('add-suggestion', () => createShot(addSuggestionRequest(sessionId, root, suggestion, [...plans.values()])))} onAddAll={(suggestions) => void mutate('add-all', async () => { const created: string[] = []; for (const [index, suggestion] of suggestions.entries()) { if (!mounted.current) return; const made = await createShot(addSuggestionRequest(sessionId, root, suggestion, [...plans.values()], suggestionAnchor(index, suggestions, created))); created.push(made.created.slot); } })} />
    {generation && <div className="film-generation-backdrop" onKeyDown={(event) => { if (event.key === 'Escape' && !busy) setGeneration(null); }}><section className="film-generation" role="dialog" aria-modal="true" aria-labelledby="film-generation-title"><header><h3 id="film-generation-title">Generation decision</h3><button className="timeline-action" disabled={Boolean(busy)} onClick={() => setGeneration(null)}>Cancel</button></header>
      <div className="film-generation-fields"><label><span>Phase</span><select autoFocus value={generation.phase} disabled={Boolean(busy)} onChange={(event) => openGeneration(event.target.value as Generation['phase'], generation.scope)}><option value="stills">First & last frames · stills</option><option value="video">Clips · video</option></select></label><label><span>Scope</span><select value={generation.scope} disabled={Boolean(busy)} onChange={(event) => openGeneration(generation.phase, event.target.value as Generation['scope'])}><option value="selected" disabled={!selected}>Selected shot</option><option value="missing">Missing in this phase</option><option value="changes">Needs changes in this phase</option></select></label></div>
      <p><strong>Exactly {generation.slots.length} shot{generation.slots.length === 1 ? '' : 's'}</strong>: {generation.slots.map((slot) => `${slotName('keyframes', slot)} [${slot}]`).join(', ') || 'Nothing eligible. Accepted takes are protected.'}</p><p className="plan-hint">Root: <code>{root}</code> · phase: <code>{generation.phase}</code>. {generation.phase === 'stills' ? 'Replaces selected frames and invalidates their clips and final approval.' : 'Replaces selected clips and may rebuild the final assembly; no other clips are requested.'}</p>
      {generation.phase === 'stills' && selected?.clip?.state === 'rejected' && selected.frames?.state !== 'rejected' && <p className="film-warning">This shot has clip feedback, so the renderer currently targets its clip. Save feedback against the frames first if you intend to redraw frames instead.</p>}
      {generation.phase === 'video' ? <div className="film-generation-fields"><label><span>Video provider</span><select value={videoSettings?.provider || ''} disabled={Boolean(busy) || settingsLoading} onChange={(event) => { const chosen = videoProviderPreset(event.target.value); setVideoSettings((current) => current ? {...current, provider: event.target.value, ...(chosen ? {base_url: chosen.baseUrl, model: chosen.defaultModel, resolution: chosen.resolution} : {})} : current); }}>{videoSettings?.provider && !preset && <option value={videoSettings.provider}>{videoSettings.provider}</option>}{Object.entries(VIDEO_PROVIDER_PRESETS).map(([value, entry]) => <option key={value} value={value}>{entry.label}</option>)}</select></label><label><span>Video model</span><select value={videoSettings?.model || ''} disabled={Boolean(busy) || settingsLoading} onChange={(event) => setVideoSettings((current) => current ? {...current, model: event.target.value} : current)}>{models.map((model) => <option key={model} value={model}>{model}</option>)}</select></label></div> : <p>Image model: <strong>{imageModel || 'Loading configuration…'}</strong></p>}
      <p className="film-warning">{generationCost}</p><p className="plan-hint">Video selection is saved to global configuration when you submit, affecting future renders in every project. Credentials are unchanged. Image model is configured globally in Settings. Failed requests are not automatically retried by this workbench.</p>
      {generationBlocked && <p className="timeline-error">{generation.phase === 'video' ? 'Accept both frames for each targeted shot first.' : 'Accept character portraits first.'}</p>}{targetHasDraft && <p className="timeline-error">One or more targeted shots have protected local drafts. Cancel and save or discard them first.</p>}{settingsError && <p className="timeline-error">{settingsError}</p>}
      {error && <p className="timeline-error" role="alert">{error}</p>}
      <button className="timeline-action is-primary" disabled={frozen || settingsLoading || Boolean(settingsError) || !phaseModel || !generation.slots.length || generationBlocked || targetHasDraft} onClick={() => void submitGeneration()}><RefreshCw size={14} />{busy === 'generate' ? 'Submitting…' : `Generate ${generation.phase} · ${generation.slots.length} shot${generation.slots.length === 1 ? '' : 's'} · ${phaseModel || 'model unavailable'}`}</button>
    </section></div>}
  </section>;
}

function renderWrittenAfter(value: unknown, at: number): boolean {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  const timestamp = Date.parse(String((value as Record<string, unknown>).timestamp || ''));
  return Number.isFinite(timestamp) && timestamp >= at;
}

function parseRenderTrail(text: string): unknown[] {
  return text.split('\n').filter((line) => line.trim()).flatMap((line) => { try { return [JSON.parse(line)]; } catch { return []; } });
}

function CoverageSection({payload, busy, onReview, onAdd, onAddAll}: {
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
  const [open, setOpen] = useState(false);
  const [drafts, setDrafts] = useState<Record<string, ContinuitySuggestion>>({});
  const [openEditor, setOpenEditor] = useState<string | null>(null);
  const suggestions = (review?.suggestions || []).map((suggestion) => drafts[suggestion.id] || suggestion);
  const stale = payload?.stale ?? false;
  return <section className="reel-section coverage-section film-coverage">
    <header className="reel-section-head">
      <button className="reel-toggle" aria-expanded={open} onClick={() => setOpen((value) => !value)}>{open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}<h2>Script coverage</h2></button>
      <span className="reel-section-hint">{reviewStatusLine(payload ?? null, {asking: busy === 'review'})}</span>
      {review && <span className="reel-section-count">{counts.covered}/{counts.total} beats</span>}
      {runtime && <span className="coverage-runtime">{runtime}</span>}
      <button className="timeline-action" disabled={Boolean(busy)} onClick={onReview}><FileJson size={13} />{review ? 'Review again…' : 'Review coverage…'}</button>
    </header>
    {!open && coverageNeedsAttention(payload ?? null) && <p className="film-warning"><AlertTriangle size={13} />{gaps.length} coverage gaps · {warnings.length} warnings{stale ? ' · review is out of date' : ''}. Expand to inspect; no suggested shot is added automatically.</p>}
    {open && <div className="film-coverage-content">
      {!review && <p className="timeline-empty">Review the film against its script to identify uncovered beats and propose real shots. This is an explicit language-model request, not media generation.</p>}
      {review?.summary && <p>{review.summary}</p>}
      {stale && <p className="film-warning">{payload?.staleReason || 'The film changed since this review.'} Review again before adding suggested shots.</p>}
      {warnings.length > 0 && <ul className="coverage-checks">{warnings.map((check) => <li className="coverage-check" key={`${check.id}:${check.message}`}><AlertTriangle size={12} /><span>{check.message}</span></li>)}</ul>}
      {gaps.length > 0 && <div className="coverage-gaps">{gaps.map((beat) => <article className={`coverage-gap is-${beat.status}`} key={`${beat.index}:${beat.text}`}><span className="coverage-gap-status">{beat.status}</span><p className="coverage-gap-text">{beat.text}</p><p className="coverage-gap-where">{beat.covered_by.length ? `Played by ${beat.covered_by.map((shot) => slotName('keyframes', String(shot))).join(', ')}` : 'No shot plays this'}{beat.note ? ` — ${beat.note}` : ''}</p></article>)}</div>}
      {suggestions.length > 0 && <div className="coverage-suggestions"><div className="coverage-suggestions-head"><h3>Suggested shots</h3><button className="timeline-action" disabled={Boolean(busy) || stale || suggestions.some((suggestion) => !suggestionAddable(suggestion, shots))} onClick={() => onAddAll(suggestions)}><Plus size={13} />Add all {suggestions.length}</button></div>
        {suggestions.map((suggestion) => {
          const editing = openEditor === suggestion.id;
          const edit = (patch: Partial<ContinuitySuggestion>) => setDrafts((current) => ({...current, [suggestion.id]: {...suggestion, ...patch}}));
          return <article className="coverage-suggestion" key={suggestion.id}><header className="coverage-suggestion-head"><strong>{suggestion.title || 'Suggested shot'}</strong><span className="coverage-suggestion-where">{suggestionPosition(suggestion)}</span>{suggestionFraming(suggestion) && <span>{suggestionFraming(suggestion)}</span>}<span>{suggestion.characters.join(', ')}</span><div className="coverage-suggestion-actions"><button className="timeline-action" onClick={() => setOpenEditor(editing ? null : suggestion.id)}>{editing ? 'Close editor' : 'Edit suggestion'}</button><button className="timeline-action" disabled={Boolean(busy) || stale || !suggestionAddable(suggestion, shots)} onClick={() => onAdd(suggestion)}><Plus size={13} />Add shot</button></div></header>
            {editing ? <div className="coverage-editor"><label className="coverage-field"><span>What the shot shows</span><textarea rows={2} value={suggestion.visual_desc} onChange={(event) => edit({visual_desc: event.target.value})} /></label><label className="coverage-field"><span>First frame description</span><textarea rows={3} value={suggestion.frames.first} onChange={(event) => edit({frames: {...suggestion.frames, first: event.target.value}})} /></label><label className="coverage-field"><span>Last frame description</span><textarea rows={3} value={suggestion.frames.last} onChange={(event) => edit({frames: {...suggestion.frames, last: event.target.value}})} /></label><label className="coverage-field"><span>Dialogue</span><textarea rows={2} value={suggestion.audio_desc} onChange={(event) => edit({audio_desc: event.target.value})} /></label></div> : <><p className="coverage-suggestion-text">{suggestion.visual_desc}</p>{suggestion.audio_desc && <p className="coverage-suggestion-audio">{suggestion.audio_desc}</p>}{suggestion.rationale && <p className="coverage-suggestion-why">{suggestion.rationale}</p>}</>}
          </article>;
        })}
      </div>}
      {review?.notes && <p className="coverage-notes">{review.notes}</p>}
    </div>}
  </section>;
}
