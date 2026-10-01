import {forwardRef, useEffect, useImperativeHandle, useRef, useState} from 'react';
import {AlertTriangle, Save, User} from 'lucide-react';
import {updateShotPlan} from './api';
import {characterChips, planDirty, toggleCharacter} from './timeline';
import type {PlanFrame, ShotPlan} from './types';

type PlanDraft = {ffDesc: string; lfDesc: string; ffVis: number[]; lfVis: number[]};
type StoredDraft = {plan: PlanDraft; base: PlanDraft; notes: Record<string, string>; revision: number};
const memoryDrafts = new Map<string, StoredDraft>();
const draftKey = (session: string, root: string, slot: string) => `vimax:film-draft:${encodeURIComponent(session)}:${encodeURIComponent(root)}:${encodeURIComponent(slot)}`;
const fields = (plan: ShotPlan): PlanDraft => ({ffDesc: plan.firstFrame.description, lfDesc: plan.lastFrame.description, ffVis: plan.firstFrame.visible, lfVis: plan.lastFrame.visible});
function readDraft(key: string): StoredDraft | undefined {
  const cached = memoryDrafts.get(key);
  if (cached) return cached;
  try {
    const value = JSON.parse(localStorage.getItem(key) || 'null') as StoredDraft | null;
    if (value && typeof value.plan?.ffDesc === 'string' && typeof value.plan?.lfDesc === 'string' && Array.isArray(value.plan.ffVis) && Array.isArray(value.plan.lfVis) && value.base && value.notes) {
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
    for (let index = 0; index < localStorage.length; index++) {
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
  onRegenerate: () => void;
  onDirtyChange: (dirty: boolean) => void;
}>(function ShotPlanPanel({sessionId, root, slot, stage, plan, feedback, accepted, disabled, canRegenerate, onChanged, onSaveFeedback, onRegenerate, onDirtyChange}, ref) {
  const key = draftKey(sessionId, root, slot);
  const [draft, setDraft] = useState<StoredDraft>(() => readDraft(key) ?? {
    plan: plan ? fields(plan) : {ffDesc: '', lfDesc: '', ffVis: [], lfVis: []},
    base: plan ? fields(plan) : {ffDesc: '', lfDesc: '', ffVis: [], lfVis: []},
    notes: {}, revision: 0,
  });
  const [error, setError] = useState('');
  const [storageFailed, setStorageFailed] = useState(false);
  const [saving, setSaving] = useState(false);
  const [editing, setEditing] = useState(() => Boolean(readDraft(key)));
  const feedbackRef = useRef<HTMLTextAreaElement>(null);
  const focusFeedback = useRef(false);
  const draftRef = useRef(draft);
  const current = useRef({key, stage, plan, feedback, disabled, onRegenerate});
  current.current = {key, stage, plan, feedback, disabled, onRegenerate};
  const mounted = useRef(true);
  const operation = useRef(false);
  const planChanged = Boolean(plan && draft && planDirty(plan, draft.plan));
  const note = draft?.notes[stage] ?? feedback;
  const noteChanged = note !== feedback;
  const dirty = planChanged || Boolean(draft && Object.keys(draft.notes).length);
  const externalChange = Boolean(planChanged && plan && draft && JSON.stringify(fields(plan)) !== JSON.stringify(draft.base));

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
    if (!plan || readDraft(key) || operation.current) return;
    const next = {plan: fields(plan), base: fields(plan), notes: {}, revision: (draftRef.current?.revision ?? 0) + 1};
    draftRef.current = next;
    setDraft(next);
  }, [key, plan]);
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
    if (changed && !window.confirm(`${accepted ? 'This shot has accepted material. ' : ''}Saving frame descriptions or characters unlocks its frames, invalidates its clip and final-film approval, and makes the coverage review stale. No generation runs. Save this plan?`)) return false;
    if (changed && externalChange && !window.confirm('The saved plan changed while this draft was open. Replace those newer descriptions and character choices with your draft?')) return false;
    operation.current = true;
    setSaving(true);
    setError('');
    try {
      const saved = changed ? await updateShotPlan({sessionId, root, slot, ...submitted.plan}) : owner.plan;
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
  return <details className="film-inspector-disclosure" open={editing} onToggle={(event) => setEditing(event.currentTarget.open)}>
    <summary>Edit plan & feedback{dirty && <span>Unsaved draft</span>}</summary>
    <section className="shot-plan film-inspector-plan" aria-label="Shot inspector">
      {plan?.brief && <p className="plan-brief"><span>Shot brief</span>{plan.brief}</p>}
      {externalChange && <p className="film-warning"><AlertTriangle size={14} /> The saved plan changed. Your local draft is protected; saving will require confirmation.</p>}
      <label className="film-feedback">
        <span>Feedback for {stage === 'clips' ? 'clip' : stage === 'keyframes' ? 'frames' : stage === 'portraits' ? 'portraits' : 'final film'}</span>
        <textarea ref={feedbackRef} rows={3} value={note} disabled={disabled} onChange={(event) => editNote(event.target.value)} placeholder="What should change in the next take?" />
      </label>
      {plan && <details className="film-plan-fields" open={planChanged || undefined}>
        <summary>Frame descriptions & characters</summary>
        {(['firstFrame', 'lastFrame'] as const).map((part) => <FramePlan
          key={part} label={part === 'firstFrame' ? 'First frame' : 'Last frame'} frame={plan[part]} characters={plan.characters}
          description={part === 'firstFrame' ? draft.plan.ffDesc : draft.plan.lfDesc}
          visible={part === 'firstFrame' ? draft.plan.ffVis : draft.plan.lfVis} disabled={disabled}
          onDescription={(value) => edit(part === 'firstFrame' ? {ffDesc: value} : {lfDesc: value})}
          onToggle={(idx, on) => edit(part === 'firstFrame' ? {ffVis: toggleCharacter(draft.plan.ffVis, idx, on)} : {lfVis: toggleCharacter(draft.plan.lfVis, idx, on)})}
        />)}
        {plan.motionDescription && <details className="plan-prompt"><summary>Motion description</summary><p>{plan.motionDescription}</p></details>}
        {planChanged && <p className="film-warning">Saving the plan unlocks frames and invalidates downstream clip/final approval. It does not regenerate anything.</p>}
        <button className="timeline-action" disabled={!planChanged || saving || disabled} onClick={() => void save('plan')}><Save size={13} />{saving ? 'Saving…' : 'Save plan'}</button>
      </details>}
      <div className="plan-actions">
        <button className="timeline-action" disabled={!note.trim() || !noteChanged || saving || disabled} onClick={() => void save('feedback')}>Save feedback</button>
        {(stage === 'keyframes' || stage === 'clips') && <button className="timeline-action is-primary" disabled={!canRegenerate || saving || disabled} onClick={() => void save('all').then((saved) => { if (saved && mounted.current) current.current.onRegenerate(); })}>Save & regenerate…</button>}
        {dirty && <button className="timeline-action" disabled={saving} onClick={() => {
          if (!window.confirm('Discard this shot’s unsaved descriptions, character choices, and feedback?')) return;
          const restored = plan ? fields(plan) : draft.base;
          commit({plan: restored, base: restored, notes: {}, revision: draft.revision + 1});
          setError('');
        }}>Discard draft</button>}
      </div>
      <p className="plan-hint">{dirty ? 'Local draft kept for this project' : 'Saved'} · never saves on blur. Feedback marks this take “Needs changes”; regeneration requires a separate scope, model and cost decision.</p>
      {Object.keys(draft.notes).some((key) => key !== stage) && <p className="film-warning">There is also unsaved feedback in the other preview stage. Open that stage to save or discard it before generating.</p>}
      {storageFailed && <p className="plan-error" role="alert">Browser storage is unavailable. Your draft is kept in this tab only; save before reloading.</p>}
      {error && <p className="plan-error" role="alert">{error}</p>}
    </section>
  </details>;
});

function FramePlan({label, frame, characters, description, visible, disabled, onDescription, onToggle}: {
  label: string; frame: PlanFrame; characters: {idx: number; name: string}[]; description: string; visible: number[]; disabled: boolean;
  onDescription: (value: string) => void; onToggle: (idx: number, on: boolean) => void;
}) {
  return <div className="plan-frame">
    <header><strong>{label}</strong><div className="plan-characters">
      {characterChips(characters, visible).map((character) => <button key={character.idx} className={`plan-character${character.on ? ' is-on' : ''}`} aria-pressed={character.on} disabled={disabled} onClick={() => onToggle(character.idx, !character.on)}><User size={11} />{character.name}</button>)}
      {!characters.length && <span className="plan-hint">No characters in this sequence</span>}
    </div></header>
    <textarea rows={3} aria-label={`${label} description`} value={description} disabled={disabled} onChange={(event) => onDescription(event.target.value)} />
    {frame.prompt ? <details className="plan-prompt"><summary>Prompt sent for this frame</summary><pre>{frame.prompt}</pre></details> : <p className="plan-hint">No prompt recorded for this frame.</p>}
  </div>;
}
