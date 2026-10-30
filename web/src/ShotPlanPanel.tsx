import {forwardRef, useEffect, useImperativeHandle, useRef, useState} from 'react';
import {AlertTriangle, MapPin, Save, User} from 'lucide-react';
import {prepareRenderedEnding, updateShotPlan} from './api';
import {characterChips, planDirty, slotName, toggleCharacter} from './timeline';
import type {AcceptanceStage, PlanFrame, ShotPlan} from './types';
import {validateShotDuration} from '../video-durations.mjs';

// Older stored drafts contain only frame fields; absent guidance stays at its saved value.
type PlanDraft = {ffDesc: string; lfDesc: string; ffVis: number[]; lfVis: number[]; brief?: string; motionDesc?: string; setIds?: string[]; reusePreviousLastFrame?: boolean; previousSlot?: string | null; durationSeconds?: number | null};
type StoredDraft = {plan: PlanDraft; base: PlanDraft; notes: Record<string, string>; revision: number};
const memoryDrafts = new Map<string, StoredDraft>();
const draftKey = (session: string, root: string, slot: string) => `vimax:film-draft:${encodeURIComponent(session)}:${encodeURIComponent(root)}:${encodeURIComponent(slot)}`;
const fields = (plan: ShotPlan): PlanDraft => ({ffDesc: plan.firstFrame.description, lfDesc: plan.lastFrame.description, ffVis: plan.firstFrame.visible, lfVis: plan.lastFrame.visible, brief: plan.brief, motionDesc: plan.motionDescription, setIds: plan.setIds, reusePreviousLastFrame: plan.reusePreviousLastFrame, previousSlot: plan.previousSlot, durationSeconds: plan.durationOverrideSeconds});
function changedFromBase(value: StoredDraft): boolean {
  return planDirty({
    firstFrame: {description: value.base.ffDesc, visible: value.base.ffVis},
    lastFrame: {description: value.base.lfDesc, visible: value.base.lfVis},
    brief: value.base.brief ?? '', motionDescription: value.base.motionDesc ?? '', setIds: value.base.setIds,
    reusePreviousLastFrame: value.base.reusePreviousLastFrame, previousSlot: value.base.previousSlot,
    durationOverrideSeconds: value.base.durationSeconds,
  }, value.plan);
}
function readDraft(key: string): StoredDraft | undefined {
  const cached = memoryDrafts.get(key);
  if (cached) return cached;
  try {
    const value = JSON.parse(localStorage.getItem(key) || 'null') as StoredDraft | null;
    if (value && typeof value.plan?.ffDesc === 'string' && typeof value.plan?.lfDesc === 'string' && Array.isArray(value.plan.ffVis) && Array.isArray(value.plan.lfVis)
      && typeof value.base?.ffDesc === 'string' && typeof value.base?.lfDesc === 'string' && Array.isArray(value.base.ffVis) && Array.isArray(value.base.lfVis) && value.notes) {
      const changed = changedFromBase(value) || Object.keys(value.notes).length > 0;
      if (!changed) {storeDraft(key); return undefined;}
      memoryDrafts.set(key, value);
      return value;
    }
  } catch { /* A denied storage permission must not prevent editing. */ }
  return undefined;
}
function storeDraft(key: string, value?: StoredDraft): boolean {
  if (value) memoryDrafts.set(key, value); else memoryDrafts.delete(key);
  try {
    if (value) localStorage.setItem(key, JSON.stringify(value)); else localStorage.removeItem(key);
    return true;
  } catch { return false; }
}
/** Reconcile every saved shot, including drafts outside the mounted inspector. */
export function reconcileFilmDrafts(session: string, root: string, plans: ShotPlan[], stages: AcceptanceStage[]): void {
  function reconcile(slot: string, plan?: ShotPlan) {
    const key = draftKey(session, root, slot);
    const stored = readDraft(key);
    if (!stored) return;
    let notes = stored.notes;
    for (const [stageName, value] of Object.entries(stored.notes)) {
      const savedStage = stages.find((stage) => stage.stage === stageName);
      const saved = savedStage?.slots.find((entry) => entry.shot === (slot.startsWith('@') ? null : slot));
      if (saved && typeof value === 'string' && value.trim() === (saved.reason || '').trim()) {
        if (notes === stored.notes) notes = {...stored.notes};
        delete notes[stageName];
      }
    }
    if ((!plan || !changedFromBase(stored) || !planDirty(plan, stored.plan)) && !Object.keys(notes).length) storeDraft(key);
    else if (notes !== stored.notes) storeDraft(key, {...stored, notes});
  }
  for (const plan of plans) reconcile(plan.slot, plan);
  for (const stage of stages) if (stage.scope === 'session') reconcile(`@${stage.stage}`);
}

/** Captured with a generation decision; a changed draft requires another explicit decision. */
export function filmDraftToken(session: string, root: string, slots: string[]): string {
  return JSON.stringify(slots.map((slot) => readDraft(draftKey(session, root, slot)) ?? null));
}
export function hasFilmDrafts(session: string, root: string, slots: string[]): boolean {
  return slots.some((slot) => Boolean(readDraft(draftKey(session, root, slot))));
}
export function hasSessionFilmDrafts(session: string): boolean {
  const prefix = `vimax:film-draft:${encodeURIComponent(session)}:`;
  for (const key of memoryDrafts.keys()) if (key.startsWith(prefix)) return true;
  try {
    for (let index = localStorage.length - 1; index >= 0; index--) {
      const key = localStorage.key(index);
      if (key?.startsWith(prefix) && readDraft(key)) return true;
    }
  } catch { /* In-memory drafts still participate in navigation guards. */ }
  return false;
}
export type ShotPlanHandle = {dirty: () => boolean; save: () => Promise<boolean>; focusFeedback: () => void};

/** The sole shot editor. Drafts live independently of selection, preview mode, or mounting. */
export const ShotPlanPanel = forwardRef<ShotPlanHandle, {
  sessionId: string;
  root: string;
  slot: string;
  stage: string;
  plan?: ShotPlan;
  feedback: string;
  accepted: boolean;
  disabled: boolean;
  canRegenerate: boolean;
  onChanged: () => Promise<void>;
  onSaveFeedback: (note: string) => Promise<void>;
  onRegenerate: (phase?: 'stills' | 'video') => void;
  onDirtyChange: (dirty: boolean) => void;
}>(function ShotPlanPanel({sessionId, root, slot, stage, plan, feedback, accepted, disabled, canRegenerate, onChanged, onSaveFeedback, onRegenerate, onDirtyChange}, ref) {
  const key = draftKey(sessionId, root, slot);
  const [draft, setDraft] = useState<StoredDraft>(() => readDraft(key) ?? {
    plan: plan ? fields(plan) : {ffDesc: '', lfDesc: '', ffVis: [], lfVis: []},
    base: plan ? fields(plan) : {ffDesc: '', lfDesc: '', ffVis: [], lfVis: []},
    notes: {}, revision: 0,
  });
  const [error, setError] = useState('');
  const [continuityNotice, setContinuityNotice] = useState('');
  const [storageFailed, setStorageFailed] = useState(false);
  const [saving, setSaving] = useState(false);
  const [preparingEnding, setPreparingEnding] = useState(false);
  const [editing, setEditing] = useState(() => Boolean(Object.keys(readDraft(key)?.notes || {}).length));
  const feedbackRef = useRef<HTMLTextAreaElement>(null);
  const focusFeedback = useRef(false);
  const draftRef = useRef(draft);
  const current = useRef({key, stage, plan, feedback, disabled, onRegenerate});
  current.current = {key, stage, plan, feedback, disabled, onRegenerate};
  const mounted = useRef(true);
  const operation = useRef(false);
  const bindingsReady = Boolean(plan && Array.isArray(plan.sets) && Array.isArray(plan.setIds) && typeof plan.setRevision === 'string');
  const planChanged = Boolean(plan && draft && planDirty(plan, draft.plan));
  const guidanceChanged = Boolean(plan && planDirty(plan, {...draft.plan, durationSeconds: undefined}));
  const durationOnly = planChanged && !guidanceChanged && !Object.keys(draft.notes).length;
  const durationValue = draft.plan.durationSeconds === undefined ? plan?.durationOverrideSeconds ?? null : draft.plan.durationSeconds;
  const durationError = durationValue !== null && plan?.durationPolicy ? validateShotDuration(durationValue, plan.durationPolicy) : '';
  const note = draft?.notes[stage] ?? feedback;
  const noteChanged = note !== feedback;
  const dirty = planChanged || Boolean(draft && Object.keys(draft.notes).length);
  const externalChange = Boolean(planChanged && plan && draft && JSON.stringify(fields(plan)) !== JSON.stringify({...fields(plan), ...draft.base}));

  function commit(next: StoredDraft, savedPlan = plan) {
    const hasChanges = Boolean(savedPlan && planDirty(savedPlan, next.plan)) || Object.keys(next.notes).length > 0;
    draftRef.current = next;
    setDraft(next);
    setStorageFailed(!storeDraft(key, hasChanges ? next : undefined));
    onDirtyChange(hasChanges);
    window.dispatchEvent(new CustomEvent('film-drafts-change', {detail: {sessionId, dirty: hasSessionFilmDrafts(sessionId)}}));
  }
  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);
  useEffect(() => {
    if (operation.current) return;
    const stored = readDraft(key);
    if (stored && (!plan || planDirty(plan, stored.plan) || Object.keys(stored.notes).length > 0)) {
      if (stored !== draftRef.current) {draftRef.current = stored; setDraft(stored);}
      return;
    }
    if (stored) storeDraft(key);
    const savedFields = plan ? fields(plan) : draftRef.current.base;
    const next = {plan: savedFields, base: savedFields, notes: {}, revision: (draftRef.current?.revision ?? 0) + 1};
    draftRef.current = next;
    setDraft(next);
  }, [key, plan, feedback, stage]);
  useEffect(() => {
    onDirtyChange(dirty);
    queueMicrotask(() => {
      if (mounted.current) window.dispatchEvent(new CustomEvent('film-drafts-change', {detail: {sessionId, dirty: hasSessionFilmDrafts(sessionId)}}));
    });
  }, [dirty, onDirtyChange, sessionId]);
  useEffect(() => {
    const guard = (event: BeforeUnloadEvent) => {
      if (!readDraft(key)) return;
      event.preventDefault();
      event.returnValue = '';
    };
    window.addEventListener('beforeunload', guard);
    return () => window.removeEventListener('beforeunload', guard);
  }, [key]);
  useEffect(() => {
    if (!editing || !focusFeedback.current) return;
    focusFeedback.current = false;
    feedbackRef.current?.focus();
    feedbackRef.current?.scrollIntoView({block: 'nearest'});
  }, [editing]);

  function edit(patch: Partial<PlanDraft>) {
    if (!draftRef.current) return;
    commit({...draftRef.current, plan: {...draftRef.current.plan, ...patch}, revision: draftRef.current.revision + 1});
  }
  function editNote(value: string) {
    if (!draftRef.current) return;
    const notes = {...draftRef.current.notes};
    if (value === feedback) delete notes[stage]; else notes[stage] = value;
    commit({...draftRef.current, notes, revision: draftRef.current.revision + 1});
  }
  async function save(mode: 'plan' | 'feedback' | 'all'): Promise<boolean> {
    const submitted = draftRef.current;
    const owner = current.current;
    if (!submitted || owner.disabled || operation.current) return false;
    const changed = mode !== 'feedback' && Boolean(owner.plan && planDirty(owner.plan, submitted.plan));
    const changesGuidance = Boolean(owner.plan && planDirty(owner.plan, {...submitted.plan, durationSeconds: undefined}));
    const durationChanged = submitted.plan.durationSeconds !== undefined && submitted.plan.durationSeconds !== owner.plan?.durationOverrideSeconds;
    if (changed && durationChanged && submitted.plan.durationSeconds !== null && owner.plan?.durationPolicy) {
      const invalidDuration = validateShotDuration(submitted.plan.durationSeconds, owner.plan.durationPolicy);
      if (invalidDuration) { setError(invalidDuration); return false; }
    }
    if (changed && (!owner.plan || !Array.isArray(owner.plan.sets) || !Array.isArray(owner.plan.setIds) || typeof owner.plan.setRevision !== 'string')) {
      setError('Shot set bindings are unavailable. Restart the updated bridge when idle before saving this plan.');
      return false;
    }
    const continuityChanged = typeof submitted.plan.reusePreviousLastFrame === 'boolean' && submitted.plan.reusePreviousLastFrame !== owner.plan?.reusePreviousLastFrame;
    if (changed && submitted.plan.reusePreviousLastFrame && submitted.plan.previousSlot !== undefined && submitted.plan.previousSlot !== owner.plan?.previousSlot) {
      setError(owner.plan?.previousSlot === null
        ? 'This shot is now first in the timeline and cannot inherit a previous frame. Discard this draft or restore its timeline position before saving.'
        : 'The previous shot changed while this draft was open. Turn the continuity checkbox off and on again to confirm its new source, or discard the draft.');
      return false;
    }
    if (changed && !window.confirm(changesGuidance
      ? `${accepted ? 'This shot has accepted material. ' : ''}Saving shot guidance (brief, motion, frame descriptions, characters, sets or continuity) unlocks its frames, invalidates its clip and final-film approval and dependent continuity approvals, and makes the coverage review stale. No generation runs. Save this plan?`
      : 'Save this shot’s duration? Only its clip and final-film approval are invalidated. Frames, other shots, and all existing media are preserved. No generation runs.')) return false;
    if (changed && externalChange && !window.confirm('The saved plan changed while this draft was open. Replace those newer shot guidance fields with your draft?')) return false;
    operation.current = true;
    setSaving(true);
    setError('');
    try {
      const saved = changed ? await updateShotPlan({
        sessionId, root, slot, ...(changesGuidance ? submitted.plan : {}),
        revision: owner.plan?.revision,
        durationSeconds: durationChanged ? submitted.plan.durationSeconds : undefined,
        brief: submitted.plan.brief?.trim() !== owner.plan?.brief.trim() ? submitted.plan.brief : undefined,
        motionDesc: submitted.plan.motionDesc?.trim() !== owner.plan?.motionDescription.trim() ? submitted.plan.motionDesc : undefined,
        setIds: submitted.plan.setIds && (submitted.plan.setIds.length !== owner.plan?.setIds.length || submitted.plan.setIds.some((id) => !owner.plan?.setIds.includes(id))) ? submitted.plan.setIds : undefined,
        setRevision: owner.plan?.setRevision,
        reusePreviousLastFrame: continuityChanged ? submitted.plan.reusePreviousLastFrame : undefined,
        previousSlot: continuityChanged ? owner.plan?.previousSlot : undefined,
      }) : owner.plan;
      if (!mounted.current || current.current.key !== key) return false;
      if (draftRef.current?.revision !== submitted.revision) {
        await onChanged();
        setError('The submitted plan was saved, but you edited again while saving. Your newer draft is kept. Save it before generating.');
        return false;
      }
      const carriedNotes = changed && owner.feedback && submitted.notes[owner.stage] === undefined
        ? {...submitted.notes, [owner.stage]: owner.feedback} : submitted.notes;
      let next = changed && saved ? {...submitted, plan: fields(saved), base: fields(saved), notes: carriedNotes} : submitted;
      const feedbackToSave = mode !== 'plan' ? next.notes[owner.stage] : undefined;
      commit(next, saved);
      if (feedbackToSave !== undefined) {
        const savedNote = feedbackToSave.trim();
        if (!savedNote) throw new Error('Write feedback before saving it, or discard the empty feedback draft.');
        await onSaveFeedback(savedNote);
        if (!mounted.current || current.current.key !== key || current.current.stage !== owner.stage || draftRef.current?.revision !== submitted.revision) return false;
        const notes = {...next.notes};
        delete notes[owner.stage];
        next = {...next, notes};
        commit(next, saved);
      }
      if (feedbackToSave === undefined) await onChanged();
      return mounted.current && current.current.key === key && current.current.stage === owner.stage && draftRef.current?.revision === submitted.revision;
    } catch (reason) {
      if (mounted.current && current.current.key === key) setError(reason instanceof Error ? reason.message : String(reason));
      return false;
    } finally {
      operation.current = false;
      if (mounted.current) setSaving(false);
    }
  }
  async function prepareEnding() {
    const owner = current.current;
    const source = owner.plan;
    if (!source || owner.disabled || operation.current || dirty || !source.reusePreviousLastFrame
      || !source.previousSlot || !source.previousClipAvailable || !source.previousClipAccepted || !source.previousClipSha256) return;
    if (!window.confirm(`Replace ${slotName('keyframes', slot)}’s first frame with ${slotName('clips', source.previousSlot)}’s actual rendered ending? This is local and free. Its frames, clip and final-film approvals will need review again. Existing last-frame and video files are kept; no generation starts.`)) return;
    operation.current = true;
    setSaving(true);
    setError('');
    setContinuityNotice('');
    setPreparingEnding(true);
    try {
      const prepared = await prepareRenderedEnding({
        sessionId, root, slot, previousSlot: source.previousSlot,
        previousClipSha256: source.previousClipSha256, revision: source.revision,
      });
      if (!mounted.current || current.current.key !== owner.key) return;
      commit({plan: fields(prepared), base: fields(prepared), notes: {}, revision: draftRef.current.revision + 1}, prepared);
      await onChanged();
      if (mounted.current && current.current.key === owner.key) setContinuityNotice('Actual rendered ending prepared. Review and accept this shot’s frames before generating its clip. No generation was started.');
    } catch (reason) {
      if (mounted.current && current.current.key === owner.key) setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      operation.current = false;
      if (mounted.current) setSaving(false);
      if (mounted.current) setPreparingEnding(false);
    }
  }

  useImperativeHandle(ref, () => ({
    dirty: () => Boolean(readDraft(key)),
    save: () => save('all'),
    focusFeedback: () => {
      if (editing) {
        feedbackRef.current?.focus();
        feedbackRef.current?.scrollIntoView({block: 'nearest'});
      } else {
        focusFeedback.current = true;
        setEditing(true);
      }
    },
  }));
  return <section className="shot-plan film-inspector-plan" aria-label="Shot inspector">
    {externalChange && <p className="film-warning"><AlertTriangle size={14} /> The saved plan changed. Your local draft is protected; saving will require confirmation.</p>}
    {plan && !bindingsReady && <p className="plan-error" role="alert">Shot set bindings are unavailable. Restart the updated bridge when idle before editing this plan.</p>}
    {plan && bindingsReady && <section className="film-plan-fields" aria-label="Shot guidance">
      <div className="plan-frame film-shot-duration" role="group" aria-label="Shot duration">
        <label><strong>Shot duration (seconds)</strong>
          {plan.durationPolicy?.allowedSeconds ? <select aria-label="Shot duration (seconds)" value={durationValue === null ? '@default' : String(durationValue)} disabled={disabled || saving || !plan.durationPolicy.configured} onChange={(event) => edit({durationSeconds: event.target.value === '@default' ? null : Number(event.target.value)})}>
            <option value="@default">Default · {plan.durationPolicy.defaultSeconds}s</option>
            {durationValue !== null && !plan.durationPolicy.allowedSeconds.includes(durationValue) && <option value={durationValue}>{durationValue}s · unsupported for this model</option>}
            {plan.durationPolicy.allowedSeconds.map((seconds) => <option key={seconds} value={seconds}>{seconds}s</option>)}
          </select> : <input aria-label="Shot duration (seconds)" type="number" min={plan.durationPolicy?.minSeconds || 1} max={plan.durationPolicy?.maxSeconds} step={1} value={durationValue ?? ''} placeholder={`Default · ${plan.durationSeconds}s`} disabled={disabled || saving || !plan.durationPolicy?.configured} onChange={(event) => edit({durationSeconds: event.target.value === '' ? null : Number(event.target.value)})} />}
        </label>
        <p className="plan-hint">{plan.durationPolicy?.configured ? `Model: ${plan.durationPolicy.model} · configured maximum ${plan.durationPolicy.maxSeconds}s. Save plan to apply; generation is a separate decision.` : 'Configure and save this video model’s maximum shot duration in Settings → Video generation before changing shot lengths. Default: 8 seconds where supported.'}</p>
        {!plan.durationPolicy?.configured && <button className="timeline-action" onClick={() => window.dispatchEvent(new CustomEvent('film-open-settings'))}>Configure model duration limits…</button>}
        {durationError && <p className="plan-error" role="alert">{durationError}</p>}
      </div>
      <GuidanceField label="Shot brief" value={draft.plan.brief ?? plan.brief} savedValue={plan.brief} disabled={disabled} saving={saving}
        onChange={(brief) => edit({brief})} onCancel={() => edit({brief: plan.brief})} />
      {(['firstFrame', 'lastFrame'] as const).map((part) => <FramePlan
        key={part} label={part === 'firstFrame' ? 'First frame' : 'Last frame'} frame={plan[part]} characters={plan.characters}
        description={part === 'firstFrame' ? draft.plan.ffDesc : draft.plan.lfDesc}
        visible={part === 'firstFrame' ? draft.plan.ffVis : draft.plan.lfVis} disabled={disabled} saving={saving}
        sets={plan.sets} setIds={draft.plan.setIds ?? plan.setIds} savedSetIds={plan.setIds}
        continuity={part === 'firstFrame' && typeof plan.previousSlot === 'string' ? {
          previousSlot: plan.previousSlot, available: plan.previousClipAvailable, accepted: plan.previousClipAccepted,
          status: plan.renderedEndingStatus, reason: plan.renderedEndingReason,
          referenceView: !plan.reusePreviousLastFrame && plan.renderedEndingStatus !== 'off',
          prepareDisabled: dirty || plan.renderedEndingStatus === 'current', preparing: preparingEnding, onPrepare: () => void prepareEnding(),
          checked: draft.plan.reusePreviousLastFrame ?? plan.reusePreviousLastFrame,
          onChange: (reusePreviousLastFrame) => edit({reusePreviousLastFrame, previousSlot: plan.previousSlot}),
        } : undefined}
        onToggleSet={(id, on) => edit({setIds: on ? [...new Set([...(draft.plan.setIds ?? plan.setIds), id])].sort() : (draft.plan.setIds ?? plan.setIds).filter((value) => value !== id)})}
        onDescription={(value) => edit(part === 'firstFrame' ? {ffDesc: value} : {lfDesc: value})}
        onToggle={(idx, on) => edit(part === 'firstFrame' ? {ffVis: toggleCharacter(draft.plan.ffVis, idx, on)} : {lfVis: toggleCharacter(draft.plan.lfVis, idx, on)})}
        onCancel={() => edit({...(part === 'firstFrame' ? {ffDesc: plan.firstFrame.description, ffVis: plan.firstFrame.visible, reusePreviousLastFrame: plan.reusePreviousLastFrame, previousSlot: plan.previousSlot} : {lfDesc: plan.lastFrame.description, lfVis: plan.lastFrame.visible}), setIds: plan.setIds})}
      />)}
      <GuidanceField label="Motion description" value={draft.plan.motionDesc ?? plan.motionDescription} savedValue={plan.motionDescription} disabled={disabled} saving={saving}
        onChange={(motionDesc) => edit({motionDesc})} onCancel={() => edit({motionDesc: plan.motionDescription})} />
      {planChanged && <p className="film-warning">{guidanceChanged ? 'Saving shot guidance unlocks frames and invalidates downstream clip/final approval. It does not regenerate anything.' : 'Saving duration invalidates only this shot’s clip and final-film approval. Frames and other shots are preserved; nothing regenerates.'}</p>}
      <button className="timeline-action" disabled={!planChanged || saving || disabled || Boolean(durationError)} onClick={() => void save('plan')}><Save size={13} />{saving ? 'Saving…' : 'Save plan'}</button>
    </section>}
    <details className="film-inspector-disclosure" open={editing} onToggle={(event) => setEditing(event.currentTarget.open)}>
      <summary>Edit feedback{Object.keys(draft.notes).length > 0 && <span>Unsaved feedback</span>}</summary>
      <label className="film-feedback">
        <span>Feedback for {stage === 'clips' ? 'clip' : stage === 'keyframes' ? 'frames' : stage === 'portraits' ? 'portraits' : 'final film'}</span>
        <textarea ref={feedbackRef} rows={3} value={note} disabled={disabled} onChange={(event) => editNote(event.target.value)} placeholder="What should change in the next take?" />
      </label>
      <button className="timeline-action" disabled={!note.trim() || !noteChanged || saving || disabled} onClick={() => void save('feedback')}>Save feedback</button>
    </details>
    <div className="plan-actions">
      {(stage === 'keyframes' || stage === 'clips') && <button className="timeline-action is-primary" disabled={!canRegenerate || saving || disabled || !bindingsReady || Boolean(durationError)} onClick={() => void save('all').then((saved) => { if (saved && mounted.current) current.current.onRegenerate(durationOnly ? 'video' : undefined); })}>{durationOnly ? 'Save & generate clip…' : 'Save & regenerate…'}</button>}
      {dirty && <button className="timeline-action" disabled={saving} onClick={() => {
        if (!window.confirm('Discard this shot’s unsaved guidance, character and set choices, and feedback?')) return;
        const restored = plan ? fields(plan) : draft.base;
        commit({plan: restored, base: restored, notes: {}, revision: draft.revision + 1});
        setError('');
      }}>Discard draft</button>}
    </div>
    <p className="plan-hint">{dirty ? 'Local draft kept for this project' : 'Saved'} · never saves on blur. Feedback marks this take “Needs changes”; regeneration requires a separate scope, model and cost decision.</p>
    {Object.keys(draft.notes).some((key) => key !== stage) && <p className="film-warning">There is also unsaved feedback in the other preview stage. Open that stage to save or discard it before generating.</p>}
    {storageFailed && <p className="plan-error" role="alert">Browser storage is unavailable. Your draft is kept in this tab only; save before reloading.</p>}
    {continuityNotice && <p className="plan-hint" role="status">{continuityNotice}</p>}
    {error && <p className="plan-error" role="alert">{error}</p>}
  </section>;
});

function FramePlan({label, frame, characters, sets, setIds, savedSetIds, description, visible, disabled, saving, continuity, onDescription, onToggle, onToggleSet, onCancel}: {
  label: string; frame: PlanFrame; characters: {idx: number; name: string}[]; sets: ShotPlan['sets']; setIds: string[]; savedSetIds: string[]; description: string; visible: number[]; disabled: boolean; saving: boolean;
  onDescription: (value: string) => void; onToggle: (idx: number, on: boolean) => void; onToggleSet: (id: string, on: boolean) => void; onCancel: () => void;
  continuity?: {previousSlot: string; available: boolean; accepted: boolean; checked: boolean; referenceView: boolean; status: ShotPlan['renderedEndingStatus']; reason: string; prepareDisabled: boolean; preparing: boolean; onPrepare: () => void; onChange: (checked: boolean) => void};
}) {
  const changed = description !== frame.description || visible.length !== frame.visible.length || visible.some((idx) => !frame.visible.includes(idx))
    || setIds.length !== savedSetIds.length || setIds.some((id) => !savedSetIds.includes(id));
  const [editing, setEditing] = useState(changed);
  const descriptionRef = useRef<HTMLTextAreaElement>(null);
  useEffect(() => { if (editing) descriptionRef.current?.focus(); }, [editing]);
  return <div className="plan-frame" role="group" aria-label={`${label} description, characters and sets`}>
    <header><strong>{label} description, characters & sets</strong>
      {editing ? <button className="timeline-action" disabled={disabled || saving} onClick={() => { onCancel(); setEditing(false); }}>Cancel {label.toLowerCase()} edits</button>
        : <button className="timeline-action" disabled={disabled || saving} onClick={() => setEditing(true)}>Edit {label.toLowerCase()}</button>}
    </header>
    {continuity && <div className="plan-frame-continuity">
      <label><input type="checkbox" checked={continuity.checked} disabled={disabled || saving} onChange={(event) => continuity.onChange(event.target.checked)} />
        <span>Use previous shot’s rendered ending as this shot’s first frame</span>
      </label>
      <p className="plan-hint">Source: {slotName('clips', continuity.previousSlot)}’s rendered video, not its planned last-frame still. When enabled, the normalized final displayed frame is copied exactly; the first-frame description does not change that image.</p>
      {(continuity.checked || continuity.status !== 'off') && <p className="plan-hint">{continuity.reason}</p>}
      {!continuity.checked && continuity.referenceView && <p className="plan-hint">Project continuity uses the rendered ending as a reference for a new camera view. Enable the checkbox to copy that ending exactly instead.</p>}
      {continuity.checked && !continuity.available && <p className="film-warning">Render the previous shot’s clip first. A planned last-frame still is not a rendered ending.</p>}
      {continuity.checked && continuity.available && !continuity.accepted && <p className="film-warning">Review and accept the previous shot’s current clip before using its ending.</p>}
      {continuity.checked && <button className="timeline-action" disabled={disabled || saving || continuity.prepareDisabled || !continuity.available || !continuity.accepted} onClick={continuity.onPrepare}>{continuity.preparing ? 'Preparing rendered ending…' : 'Prepare from rendered ending · local, free'}</button>}
      {continuity.checked && continuity.prepareDisabled && continuity.status !== 'current' && <p className="plan-hint">Save or discard local edits before preparing the first frame.</p>}
    </div>}
    <div className="plan-characters">
      {characterChips(characters, visible).map((character) => <button key={character.idx} className={`plan-character${character.on ? ' is-on' : ''}`} aria-pressed={character.on} disabled={disabled || !editing} onClick={() => onToggle(character.idx, !character.on)}><User size={11} />{character.name}</button>)}
      {!characters.length && <span className="plan-hint">No characters in this sequence</span>}
    </div>
    <div className="plan-characters" role="group" aria-label={`${label} set references`}>
      {sets.map((set) => <button key={set.id} className={`plan-character plan-set${setIds.includes(set.id) ? ' is-on' : ''}`} aria-pressed={setIds.includes(set.id)} disabled={disabled || saving || !editing} title={`${set.id} · ${set.accepted ? 'Approved reference' : 'Approve a reference in Assets → Sets before generation'}`} onClick={() => onToggleSet(set.id, !setIds.includes(set.id))}><MapPin size={11} />{set.name}{!set.accepted && <AlertTriangle size={11} />}</button>)}
      {!sets.length && <span className="plan-hint">Prepare locations in Assets → Sets.</span>}
    </div>
    {sets.length > 0 && <p className="plan-hint">Set buttons bind both frames of this shot. Refer to the selected set by name in the description; mentioning a name alone does not bind its image.</p>}
    {editing ? <textarea ref={descriptionRef} rows={4} aria-label={`${label} description`} value={description} disabled={disabled} onChange={(event) => onDescription(event.target.value)} />
      : <p className="plan-frame-description">{description || 'No description yet.'}</p>}
    {frame.prompt ? <details className="plan-prompt"><summary>Prompt sent for this frame</summary><p className="plan-hint">This recorded provider prompt is read-only. Edit the frame description above; regeneration builds a new prompt from the saved guidance.</p><pre>{frame.prompt}</pre></details> : <p className="plan-hint">No prompt recorded for this frame.</p>}
  </div>;
}

function GuidanceField({label, value, savedValue, disabled, saving, onChange, onCancel}: {
  label: string; value: string; savedValue: string; disabled: boolean; saving: boolean;
  onChange: (value: string) => void; onCancel: () => void;
}) {
  const [editing, setEditing] = useState(value !== savedValue);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  useEffect(() => { if (editing) inputRef.current?.focus(); }, [editing]);
  return <div className="plan-frame" role="group" aria-label={label}>
    <header><strong>{label}</strong>
      {editing ? <button className="timeline-action" disabled={disabled || saving} onClick={() => { onCancel(); setEditing(false); }}>Cancel {label.toLowerCase()} edits</button>
        : <button className="timeline-action" disabled={disabled || saving} onClick={() => setEditing(true)}>Edit {label.toLowerCase()}</button>}
    </header>
    {editing ? <textarea ref={inputRef} rows={4} aria-label={label} value={value} disabled={disabled} onChange={(event) => onChange(event.target.value)} />
      : <p className="plan-frame-description">{value || 'No description yet.'}</p>}
  </div>;
}
