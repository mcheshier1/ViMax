import {useEffect, useMemo, useRef, useState} from 'react';
import {RefreshCw} from 'lucide-react';
import {getAgentConfig, readAcceptance} from './api';
import type {FilmCharacter} from './types';
import type {FilmProgress} from './filmSession';
import {hasSessionFilmDrafts} from './ShotPlanPanel';

type Props = {
  sessionId: string;
  root: string;
  revision: string;
  characters: FilmCharacter[];
  imageModel: string;
  portraitImageModel: string;
  progress?: FilmProgress;
  disabled?: boolean;
  openRequest?: number;
  onAskAgent: (text: string) => Promise<void>;
  onRefresh: () => Promise<void>;
};

export function CharacterPortraitDecision({sessionId, root, revision, characters, imageModel, portraitImageModel, progress, disabled = false, openRequest = 0, onAskAgent, onRefresh}: Props) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [sent, setSent] = useState(false);
  const submittedAt = useRef(0);
  const incomplete = useMemo(() => characters.filter((character) => character.visible && !character.complete), [characters]);
  const inventoryIdentity = JSON.stringify(incomplete);
  const latest = useRef({sessionId, root, revision, inventoryIdentity, imageModel, portraitImageModel, disabled});
  latest.current = {sessionId, root, revision, inventoryIdentity, imageModel, portraitImageModel, disabled};
  const mounted = useRef(true);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  useEffect(() => { if (openRequest > 0) setOpen(true); }, [openRequest]);
  const modelMismatch = Boolean(incomplete.length && imageModel && imageModel !== portraitImageModel);
  const invalidSets = incomplete.filter((character) => Object.values(character.views).some((view) => view.state === 'invalid'));
  const supportedRoot = /(?:^|[\\/])script2video(?:[\\/]|$)/i.test(root);
  const unsupportedRoot = Boolean(incomplete.length && !supportedRoot);
  const active = progress?.activeSessionId === sessionId && progress.agentRunning;
  const progressStatus = progress?.status && typeof progress.status === 'object' && !Array.isArray(progress.status)
    ? progress.status as Record<string, unknown>
    : undefined;
  const progressStatusValue = typeof progress?.status === 'string' ? progress.status : progressStatus?.status;
  const renderError = progressStatus?.error || progressStatus?.error_type || (['error', 'failed'].includes(String(progressStatusValue || '')) ? progressStatusValue : undefined);
  const statusAt = Date.parse(String(progressStatus?.timestamp || ''));
  const newProgress = sent && Number.isFinite(statusAt) && statusAt >= submittedAt.current && progressStatus?.phase === 'portraits';
  const renderFinished = newProgress && ['rendered', 'done', 'awaiting_confirmation'].includes(String(progressStatusValue || '').toLowerCase());
  const pending = sent && !renderFinished && !(newProgress && renderError);
  const blocked = disabled || busy || pending || !incomplete.length || !imageModel || modelMismatch || invalidSets.length > 0 || unsupportedRoot;

  useEffect(() => {
    if (sent && newProgress && renderError) setError(`Portrait generation reported an error: ${String(renderError)}`);
  }, [sent, newProgress, renderError]);

  async function submit() {
    if (!open || blocked || !sessionId || !root) return;
    const confirmed = {sessionId, root, revision, inventoryIdentity, imageModel};
    const verifyFresh = () => {
      const current = latest.current;
      if (!mounted.current || current.disabled || confirmed.sessionId !== current.sessionId || confirmed.root !== current.root || confirmed.revision !== current.revision || confirmed.inventoryIdentity !== current.inventoryIdentity || confirmed.imageModel !== current.imageModel || current.imageModel !== current.portraitImageModel) throw new Error('The project, character inventory, model, or busy state changed while you reviewed this decision. Reopen it before submitting.');
    };
    setBusy(true);
    setError('');
    setSent(false);
    try {
      if (hasSessionFilmDrafts(confirmed.sessionId)) throw new Error('Save or discard all local film drafts before requesting portraits. No request was sent.');
      verifyFresh();
      const config = await getAgentConfig();
      verifyFresh();
      if (config.sections.image.model !== confirmed.imageModel) throw new Error('The configured image model changed. Review its updated name before submitting.');
      if (hasSessionFilmDrafts(confirmed.sessionId)) throw new Error('A local film draft changed. Save or discard it before requesting portraits. No request was sent.');
      verifyFresh();
      const current = await readAcceptance(confirmed.sessionId, confirmed.root);
      verifyFresh();
      const currentIncomplete = (current.characters || []).filter((character) => character.visible && !character.complete);
      if (JSON.stringify(currentIncomplete) !== confirmed.inventoryIdentity || current.portraitImageModel !== confirmed.imageModel) throw new Error('The required portrait sets changed. Refresh the project and review the decision again.');
      const prompt = `Generate missing required character portraits for session ${JSON.stringify(confirmed.sessionId)}, root ${JSON.stringify(confirmed.root)}, using configured image model ${JSON.stringify(confirmed.imageModel)}.\n`
        + `Required incomplete sets: ${JSON.stringify(currentIncomplete)}.\n`
        + 'Use vimax_render_video with stop_after="portraits". Reconcile required missing sets using normal cache/provenance rules; reuse surviving valid views and complete unchanged sets. Do not use redo, force, or only_shots flags. Do not change the script, character descriptions, style, model, or selected portrait sets. An incomplete trio may need rebuilding under normal provenance rules. Stop after portraits and report the result for review and approval. Do not generate shot frames, clips, or a final video, and do not automatically retry a failure or begin another paid phase. Exact provider charges are unknown; this decision authorizes only the stated portrait phase.';
      submittedAt.current = Date.now();
      await onAskAgent(prompt);
      if (!mounted.current) return;
      setSent(true);
      setOpen(false);
      try { await onRefresh(); } catch (reason) { if (mounted.current) setError(reason instanceof Error ? reason.message : String(reason)); }
    } catch (reason) {
      if (mounted.current) setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      if (mounted.current) setBusy(false);
    }
  }

  return <>
    <button type="button" className="timeline-action is-primary" disabled={blocked} onClick={() => { setError(''); setOpen(true); }}>
      {busy ? 'Submitting portraits…' : `Generate missing portraits · ${incomplete.length}`}
    </button>
    {unsupportedRoot && <p className="timeline-error" role="status">Portrait-only recovery is unavailable for this workflow root. It is supported only for Script2Video; other workflows require their broader workflow-specific render.</p>}
    {modelMismatch && <p className="timeline-error" role="status">Missing-only portrait recovery is blocked because the configured image model differs from the portrait inventory model ({portraitImageModel || 'unknown'}). Use the recorded model or follow the workflow's broader model migration.</p>}
    {invalidSets.map((character) => <p key={`${character.scope}:${character.identifier}`} className="timeline-error" role="status">Missing-only recovery is blocked for {character.identifier}: invalid portrait views must be deleted or replaced with a valid trio first.</p>)}
    {sent && !error && <p className="timeline-render-status" role="status"><RefreshCw size={12} className={active ? 'is-spinning' : undefined} />{renderFinished ? incomplete.length ? `Portrait rendering finished; ${incomplete.length} required set${incomplete.length === 1 ? '' : 's'} remain incomplete. Inspect the result before another decision.` : 'Portrait rendering finished; review and approve the sets before generating frames or clips.' : active ? 'Portrait request sent; agent connected, waiting for portrait render progress.' : 'Portrait request sent; waiting for agent/render progress.'}</p>}
    {pending && <button className="timeline-action" disabled={busy || disabled} onClick={() => { if (window.confirm('Stop tracking this portrait request? This does not cancel the agent or render. Check Assistant before submitting again to avoid duplicate charges.')) setSent(false); }}>Clear waiting notice</button>}
    {error && !open && <p className="timeline-error" role="alert">{error}</p>}
    {open && <div className="film-generation-backdrop" onKeyDown={(event) => { if (event.key === 'Escape' && !busy) setOpen(false); }}><section className="film-generation character-portrait-decision" role="dialog" aria-modal="true" aria-labelledby="portrait-decision-title">
      <header><h3 id="portrait-decision-title">Portrait generation decision</h3><button className="timeline-action" disabled={busy} onClick={() => setOpen(false)}>Cancel</button></header>
      <p><strong>Configured image model:</strong> {imageModel || 'Unavailable'} · existing inventory model: {portraitImageModel || 'Unknown'}</p>
      <p>All incomplete visible required sets (global reconciliation; no individual-character targeting):</p>
      {incomplete.length ? <ul>{incomplete.map((character) => <li key={`${character.scope}:${character.identifier}`}><strong>{character.identifier}</strong> · {character.scope} — front {character.views.front.state}, side {character.views.side.state}, back {character.views.back.state}</li>)}</ul> : <p>No incomplete visible character portrait sets.</p>}
      <p className="film-warning">The pipeline reuses valid existing views without forced redraw, but can regenerate an entire incomplete trio; character/style/model provenance changes may also regenerate complete-looking sets. Invalid PNG views cannot be repaired by this missing-only request; delete the invalid asset or select a valid trio first. Provider charges may apply; exact cost is unknown. This request stops after portraits. Review and approve them later before generating frames or clips. Portrait approval does not block portrait rendering.</p>
      {modelMismatch && <p className="timeline-error" role="alert">The configured image model differs from the model recorded for this portrait inventory. Missing-only recovery is blocked to avoid a model migration.</p>}
      {invalidSets.map((character) => <p key={`${character.scope}:${character.identifier}`} className="timeline-error" role="alert">Cannot recover {character.identifier} through missing-only generation while it has an invalid view. Delete the invalid asset or select a valid trio first.</p>)}
      {hasSessionFilmDrafts(sessionId) && <p className="timeline-error">Protected unsaved film drafts exist. Save or discard them before sending.</p>}
      {error && <p className="timeline-error" role="alert">{error}</p>}
      <button type="button" className="timeline-action is-primary" disabled={blocked || hasSessionFilmDrafts(sessionId)} onClick={() => void submit()}>{busy ? 'Submitting…' : `Confirm portrait-only request · ${incomplete.length} sets`}</button>
    </section></div>}
  </>;
}
