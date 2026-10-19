import {useEffect, useMemo, useState} from 'react';
import {Check, ExternalLink, ImagePlus, Plus, RefreshCw, Save, X} from 'lucide-react';
import {getAgentConfig, readSetReferences, selectSetReference, updateSetReference, updateSetReferenceAcceptance} from './api';
import type {SetReference, SetReferenceInventory} from './types';
import type {AssetCreationRequest} from './AssetCreationDialog';
import {hasFilmDrafts} from './ShotPlanPanel';

type Draft = {id: string; name: string; description: string; shots: string[]};

export function SetReferencesPanel({sessionId, root, revision, disabled, onGenerate, onPrepare, onCreateAsset, onRefresh}: {
  sessionId: string; root: string; revision: string; disabled: boolean; onGenerate: (instruction: string) => Promise<void>; onPrepare: (instruction: string) => Promise<void>; onCreateAsset?: (request: AssetCreationRequest) => void; onRefresh?: (sessionId: string) => void | Promise<void>;
}) {
  const [inventory, setInventory] = useState<SetReferenceInventory | null>(null);
  const [drafts, setDrafts] = useState<Record<string, Draft>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [requestNotice, setRequestNotice] = useState('');
  const [initialized, setInitialized] = useState(false);
  const candidateSupport = Boolean(inventory && inventory.sets.every((set) => Array.isArray(set.candidates)) && inventory.shots.every((shot) => Array.isArray(shot.setIds)));
  useEffect(() => {setError(''); setRequestNotice('');}, [sessionId, root]);
  useEffect(() => {
    let current = true;
    setInitialized(false);
    void readSetReferences(sessionId, root).then((value) => {
      if (!current) return;
      setInventory(value);
      const saved = loadDrafts(sessionId, value.root);
      setDrafts({...Object.fromEntries(value.sets.map((set) => [set.id, draftOf(set)])),
        ...Object.fromEntries(Object.entries(saved).filter(([id, draft]) => isDirty(draft, value.sets.find((set) => set.id === id))))});
      setInitialized(true);
    }).catch((reason) => {
      if (current) {setError(reason instanceof Error ? reason.message : String(reason)); setInitialized(true);}
    });
    return () => {current = false;};
  }, [sessionId, root, revision, disabled]);
  const dirty = useMemo(() => Object.values(drafts).some((draft) => isDirty(draft, inventory?.sets.find((set) => set.id === draft.id))), [drafts, inventory]);
  useEffect(() => {
    if (!initialized) return;
    try { sessionStorage.setItem(`vimax:set-drafts:${sessionId}:${root}`, JSON.stringify(drafts)); } catch {}
  }, [drafts, initialized, sessionId, root]);
  useEffect(() => {
    if (!dirty) return;
    const guard = (event: BeforeUnloadEvent) => {event.preventDefault(); event.returnValue = '';};
    window.addEventListener('beforeunload', guard);
    return () => window.removeEventListener('beforeunload', guard);
  }, [dirty]);
  const refresh = async () => {
    const value = await readSetReferences(sessionId, root);
    setInventory(value);
    setDrafts((previous) => ({
      ...Object.fromEntries(value.sets.map((set) => [set.id, previous[set.id] && isDirty(previous[set.id], inventory?.sets.find((item) => item.id === set.id)) ? previous[set.id] : draftOf(set)])),
      ...Object.fromEntries(Object.entries(previous).filter(([key]) => !value.sets.some((set) => set.id === key))),
    }));
  };
  const add = () => {
    const id = `set_${crypto.randomUUID().replaceAll('-', '')}`;
    setDrafts((old) => ({...old, [id]: {id, name: '', description: '', shots: []}}));
  };
  const updateDraft = (id: string, patch: Partial<Draft>) => setDrafts((old) => ({...old, [id]: {...old[id], ...patch}}));
  const save = async (id: string) => {
    if (!inventory) return;
    const draft = drafts[id];
    setBusy(true); setError('');
    try {
      const value = await updateSetReference({...draft, sessionId, root: inventory.root, revision: inventory.revision});
      setInventory(value);
      setDrafts((previous) => ({
        ...Object.fromEntries(value.sets.map((set) => [set.id,
          set.id === id ? draftOf(set) : previous[set.id] && isDirty(previous[set.id], inventory.sets.find((item) => item.id === set.id)) ? previous[set.id] : draftOf(set)])),
        ...Object.fromEntries(Object.entries(previous).filter(([key]) => !value.sets.some((set) => set.id === key))),
      }));
      await onRefresh?.(sessionId);
    } catch (reason) {setError(reason instanceof Error ? reason.message : String(reason));}
    finally {setBusy(false);}
  };
  const review = async (set: SetReference, accepted: boolean, reason?: string) => {
    if (!inventory) return;
    setBusy(true); setError('');
    try {
      setInventory(await updateSetReferenceAcceptance({sessionId, root: inventory.root, id: set.id, revision: inventory.revision, accepted, reason}));
      await onRefresh?.(sessionId);
    } catch (cause) {setError(cause instanceof Error ? cause.message : String(cause));}
    finally {setBusy(false);}
  };
  const useCandidate = async (set: SetReference, candidateId: string) => {
    if (!inventory) return;
    setBusy(true); setError('');
    try {
      setInventory(await selectSetReference({sessionId, root: inventory.root, id: set.id, candidateId, revision: inventory.revision}));
      await onRefresh?.(sessionId);
    } catch (cause) {setError(cause instanceof Error ? cause.message : String(cause));}
    finally {setBusy(false);}
  };
  const submitRequest = async (generate: boolean, setId?: string) => {
    if (!inventory || generate && !candidateSupport || busy || disabled) return;
    if (generate ? !setId || isDirty(drafts[setId], inventory.sets.find((set) => set.id === setId)) : dirty) return;
    setBusy(true); setError('');
    setRequestNotice(generate ? 'Submitting set-image request…' : 'Submitting text-only set-planning request…');
    try {
      const latest = await readSetReferences(sessionId, root);
      const config = await getAgentConfig();
      if (latest.revision !== inventory.revision || config.sections.image.model !== inventory.imageModel) {
        setInventory(latest);
        throw new Error('Set references or the configured image model changed. Refresh and review the decision again.');
      }
      const chosen = latest.sets.filter((set) => set.id === setId && set.state !== 'accepted');
      if (!generate) {
        if (hasFilmDrafts(sessionId, latest.root, latest.shots.map((shot) => shot.slot))) throw new Error('Save or discard shot drafts before preparing a set plan; text planning can change their assignments.');
        await onPrepare(`Prepare set references for render root ${latest.root}. Use text planning only: create descriptors with id, name, description, exact active shot slots. Do not generate any images or other media.`);
        setRequestNotice('Text-only planning request sent. Follow Activity for progress or errors; no images were requested.');
      } else {
        if (chosen.length !== 1) throw new Error('This set is unavailable or accepted. Unlock its reference before requesting another image.');
        const redo = chosen.filter((set) => Boolean(set.imagePath || set.candidates?.length)).map((set) => set.id);
        const set = chosen[0];
        await onGenerate(`Run vimax_render_video(stop_after="sets", set_ids=${JSON.stringify([set.id])}, set_candidate_count=1, redo_sets=${JSON.stringify(redo)}). Generate exactly one new alternative candidate image for ${set.name} (${set.id}) only. Total image requests: 1. Image model: ${latest.imageModel || 'unknown'}; style: ${latest.style || 'unknown'}. Provider cost is unknown; this is explicit authorization for this single image only. Do not generate images for any other set, choose or approve a candidate, or advance to shots or video.`);
        setRequestNotice(`Request sent for 1 image for ${set.name}. Follow Activity for progress or errors. Candidates refresh automatically; choose Use, then Approve.`);
        try {await onRefresh?.(sessionId);}
        catch (cause) {setError(`The request was sent, but refreshing Assets failed: ${cause instanceof Error ? cause.message : String(cause)}. Check Activity before requesting again.`);}
      }
    } catch (cause) {setRequestNotice(''); setError(cause instanceof Error ? cause.message : String(cause));}
    finally {setBusy(false);}
  };
  const activeSets = inventory?.sets ?? [];
  return <section className="set-reference-panel" aria-label="Set references">
    <header><div><span className="eyebrow">Location continuity</span><h2>Sets</h2><p>Approved location images are required before generating any shot images.</p></div><button type="button" className="icon-button" onClick={() => void refresh().catch((e) => setError(String(e)))} disabled={!initialized || busy} aria-label="Refresh set references"><RefreshCw size={16}/></button></header>
    {error && !inventory && <p className="inline-error" role="alert">{error}</p>}
    {!initialized && <p role="status">Loading set references…</p>}
    {inventory && !candidateSupport && <p className="inline-error" role="alert">Set candidate support is unavailable. Restart the updated bridge when idle before requesting images.</p>}
    {inventory && <>
      <div className="set-reference-toolbar"><span>Image model: <b>{inventory.imageModel || 'Not pinned'}</b> · Style: <b>{inventory.style || 'Not set'}</b></span><button type="button" onClick={add} disabled={busy || disabled}><Plus size={14}/> Add set</button></div>
      {inventory.unassignedShots.length > 0 && <p className="set-reference-warning">Unassigned shots: {inventory.unassignedShots.join(', ')}. Bind each shot to at least one set before shot-image generation.</p>}
      {!activeSets.length && <div className="set-reference-empty"><p>No set plan exists yet. Prepare creates a text-only plan and does not generate media.</p><button type="button" onClick={() => void submitRequest(false)} disabled={disabled || busy || dirty}>Prepare set plan (text only)</button></div>}
      <div className="set-reference-list">{[...activeSets, ...Object.values(drafts).filter((draft) => !activeSets.some((set) => set.id === draft.id)).map((draft) => ({...draft, state: 'planned' as const, revision: inventory.revision, candidates: []}))].map((set: SetReference) => {
        const draft = drafts[set.id] ?? draftOf(set as SetReference);
        const changed = isDirty(draft, activeSets.find((item) => item.id === set.id));
        const imageUrl = set.imagePath ? `/api/artifact?session=${encodeURIComponent(sessionId)}&path=${encodeURIComponent(set.imagePath)}&revision=${encodeURIComponent(inventory.revision)}` : '';
        return <article className="set-reference-card" key={set.id}>
          <div className="set-reference-canonical">{imageUrl ? <a href={imageUrl} target="_blank" rel="noreferrer" aria-label={`Open selected reference for ${draft.name || set.id}`}><img className="set-reference-preview" src={imageUrl} alt={`${draft.name || set.id} selected location reference`} /><span><ExternalLink size={13}/> Selected set reference</span></a> : <div className="set-reference-placeholder"><ImagePlus size={22}/><span>No reference image selected</span></div>}</div>
          <div className="set-reference-fields"><div className="set-reference-title"><label>Set name<input value={draft.name} onChange={(event) => updateDraft(set.id, {name: event.target.value})} placeholder="Saloon" maxLength={160}/></label><span className={`set-reference-state state-${set.state}`}>{set.state}</span></div>
            <label>Description<textarea value={draft.description} onChange={(event) => updateDraft(set.id, {description: event.target.value})} placeholder="Room layout, architecture, props, and lighting direction" maxLength={4000} rows={3}/></label>
            <fieldset><legend>Default shot assignments</legend><p className="plan-hint">Set buttons in the shot editor override these defaults for that shot.</p><div className="set-reference-shots">{inventory.shots.map((shot) => <label key={shot.slot}><input type="checkbox" checked={draft.shots.includes(shot.slot)} onChange={(event) => updateDraft(set.id, {shots: event.target.checked ? [...draft.shots, shot.slot] : draft.shots.filter((slot) => slot !== shot.slot)})}/><span>{shot.slot}</span>{shot.brief && <small>{shot.brief}</small>}</label>)}</div></fieldset>
            {set.candidates?.length ? <div className="set-candidate-grid" aria-label={`${draft.name || set.id} image candidates`}>{set.candidates.map((candidate) => {
              const candidateUrl = `/api/artifact?session=${encodeURIComponent(sessionId)}&path=${encodeURIComponent(candidate.imagePath)}`;
              const current = set.selectedCandidateId === candidate.id;
              return <article className={`set-candidate ${current ? 'is-current' : ''}`} key={candidate.id}>
                <a href={candidateUrl} target="_blank" rel="noreferrer" aria-label={`Open candidate image ${candidate.id} for ${draft.name || set.id}`}><img src={candidateUrl} alt={`${draft.name || set.id} candidate${candidate.state !== 'ready' ? ` (${candidate.state})` : ''}`} loading="lazy"/><span><ExternalLink size={12}/> Open full image · {candidate.state}</span></a>
                {candidate.prompt && <details><summary>Image prompt</summary><p>{candidate.prompt}</p></details>}
                {current && <strong>Selected reference</strong>}
                <button type="button" onClick={() => void useCandidate(set, candidate.id)} disabled={busy || disabled || changed || candidate.state !== 'ready' || current}>Use this image for {draft.name || set.id}</button>
              </article>;
            })}</div> : <p className="set-candidate-empty">No alternatives yet. Image generation only starts after you explicitly authorize a request below.</p>}
            <p className="plan-hint">Use chooses one image; Approve is separate. Choosing a different image clears affected approvals but keeps existing media. Neither action generates images.</p>
            <div className="set-reference-actions"><button type="button" onClick={() => void save(set.id)} disabled={busy || disabled || !changed || !draft.name.trim()}><Save size={14}/> Save</button>
              {onCreateAsset && activeSets.some((item) => item.id === set.id) && <button type="button" onClick={() => onCreateAsset({type: 'set', destination: {id: set.id, name: set.name || set.id}})}>Generate alternative…</button>}
              {activeSets.some((item) => item.id === set.id) && <button type="button" onClick={() => void submitRequest(true, set.id)} disabled={busy || disabled || !candidateSupport || changed || set.state === 'accepted'}><ImagePlus size={14}/> Request 1 image for {draft.name || set.id}</button>}
              {set.state === 'rendered' && !changed && <button type="button" onClick={() => void review(set as SetReference, true)} disabled={busy || disabled}><Check size={14}/> Approve {draft.name || 'reference'}</button>}
              {set.state === 'accepted' && <button type="button" onClick={() => void review(set as SetReference, false)} disabled={busy || disabled}><X size={14}/> Unlock</button>}
              {(set.state === 'rendered' || set.state === 'accepted') && !changed && <button type="button" onClick={() => {const reason = window.prompt('Why reject this set reference?'); if (reason?.trim()) void review(set as SetReference, false, reason);}} disabled={busy || disabled}>Reject</button>}
            </div>
            <p className="plan-hint">Each request generates one new alternative for this set only. Provider cost is unknown. Existing candidates and the selected image are retained.{set.state === 'accepted' ? ' Unlock this reference before requesting another image.' : changed ? ' Save this set’s edits before requesting its image.' : ''}</p>
          </div>
        </article>;
      })}</div>
      {requestNotice && <p className="timeline-render-status" role="status" aria-live="polite">{requestNotice}</p>}
      {error && <p className="inline-error" role="alert">{error}</p>}
    </>}
  </section>;
}
function draftOf(set: SetReference): Draft {return {id: set.id, name: set.name, description: set.description, shots: [...set.shots]};}
function same(a: string[], b: string[]) {return a.length === b.length && a.every((value, index) => value === b[index]);}
function isDirty(draft?: Draft, source?: SetReference) {return Boolean(draft && (!source || draft.name !== source.name || draft.description !== source.description || !same(draft.shots, source.shots)));}
function loadDrafts(sessionId: string, root: string): Record<string, Draft> {
  try {
    const payload = JSON.parse(sessionStorage.getItem(`vimax:set-drafts:${sessionId}:${root}`) || '{}');
    if (!payload || typeof payload !== 'object' || Array.isArray(payload)) return {};
    return Object.fromEntries(Object.entries(payload).filter(([, value]) => {
      const draft = value as Draft;
      return draft && typeof draft.id === 'string' && typeof draft.name === 'string'
        && typeof draft.description === 'string' && Array.isArray(draft.shots)
        && draft.shots.every((slot) => typeof slot === 'string');
    })) as Record<string, Draft>;
  } catch {
    return {};
  }
}
