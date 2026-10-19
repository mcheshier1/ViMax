import {useEffect, useMemo, useRef, useState} from 'react';
import {ArrowUpRight, FileJson, Film, Files, Image as ImageIcon, Video, X} from 'lucide-react';
import {approvePortraitSet, deleteAsset, deleteAssets, getAgentConfig, getAssetDeletionPlan, getAssetsDeletionPlan, getJsonArtifact, getPortraitOptions, selectPortraitSet} from './api';
import {
  extractStoryboardPreviews,
  formatStructuredValue,
  friendlyArtifactTitle,
  friendlyFieldLabel,
  isJsonArtifact,
  isJsonObject,
  isJsonPrimitive,
  isArtifactPathField,
  isStoryboardArtifact,
  sortVisuals,
  structuredRecordTitle,
  visualArtifactTitle,
  visualFamilyCounts,
  visualPromptSource,
  visualPromptText,
  visualRole,
  visibleVisuals,
  type VisualFamily,
} from './artifactPresentation';
import {artifactUrl, thumbnailUrl} from './media';
import type {Artifact, AssetDeletionPlan, AssetsDeletionPlan, FilmCharacter, JsonValue, PortraitMutationResponse, PortraitOptions, SessionSummary} from './types';
import type {FilmProgress} from './filmSession';
import {CharacterPortraitDecision} from './CharacterPortraitDecision';
import {SetReferencesPanel} from './SetReferencesPanel';
import type {AssetCreationRequest} from './AssetCreationDialog';

export function ArtifactsView({session, artifacts, onRefresh, characters = [], root = '', revision = '', portraitImageModel = '', imageModel = '', progress, disabled = false, onAskAgent, onCreateAsset}: {
  session?: SessionSummary;
  artifacts: Artifact[];
  onRefresh?: (sessionId: string) => void | Promise<void>;
  characters?: FilmCharacter[];
  root?: string;
  revision?: string;
  portraitImageModel?: string;
  imageModel?: string;
  progress?: FilmProgress;
  disabled?: boolean;
  onAskAgent: (text: string) => Promise<void>;
  onCreateAsset?: (request: AssetCreationRequest) => void;
}) {
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [previewArtifact, setPreviewArtifact] = useState<Artifact>();
  const mediaArtifacts = useMemo(() => sortVisuals(artifacts.filter((artifact) => artifact.kind === 'image' || artifact.kind === 'video')), [artifacts]);
  const jsonArtifacts = useMemo(() => artifacts.filter(isJsonArtifact).sort((left, right) => left.path.localeCompare(right.path, undefined, {numeric: true})), [artifacts]);
  return (
    <section className="artifacts-view workbench-assets">
      <header className="workbench-page-heading"><div><span>Media library</span><h1>Assets</h1></div><span className="artifact-file-count">{mediaArtifacts.length} visuals · {jsonArtifacts.length} documents</span>{session && onCreateAsset && <button className="workbench-primary" type="button" onClick={() => onCreateAsset({})}><ArrowUpRight size={15}/>Generate asset…</button>}</header>
      {!session ? <ArtifactsEmpty title="Select a project" detail="Your images, clips, and source files will appear here." /> : <>
        <SetReferencesPanel sessionId={session.sessionId} root={root} revision={revision} disabled={disabled} onGenerate={onAskAgent} onPrepare={onAskAgent} onCreateAsset={onCreateAsset} onRefresh={onRefresh} />
        <CharacterInventory characters={characters} artifacts={mediaArtifacts} sessionId={session.sessionId} root={root} revision={revision} portraitImageModel={portraitImageModel} imageModel={imageModel} progress={progress} disabled={disabled} onAskAgent={onAskAgent} onCreateAsset={onCreateAsset} onRefresh={() => onRefresh?.(session.sessionId)} onPreview={setPreviewArtifact} />
        <VisualArtifacts key={session.sessionId} session={session} artifacts={mediaArtifacts} onPreview={setPreviewArtifact} onRefresh={() => onRefresh?.(session.sessionId)} />
        <details className="assets-advanced" open={advancedOpen} onToggle={(event) => setAdvancedOpen(event.currentTarget.open)}>
          <summary><FileJson size={16} /><span>Advanced · raw documents</span><small>{jsonArtifacts.length} files</small></summary>
          {advancedOpen && <DocumentBrowser artifacts={jsonArtifacts} raw />}
        </details>
      </>}
      {previewArtifact && <MediaPreviewDialog key={`${session?.sessionId || ''}:${previewArtifact.path}`} session={session} artifact={previewArtifact} artifacts={mediaArtifacts} promptDocument={artifacts.find((candidate) => candidate.path === (previewArtifact.promptPath || visualPromptSource(previewArtifact.path)?.documentPath))} onRefresh={() => onRefresh?.(session?.sessionId || '')} onClose={() => setPreviewArtifact(undefined)} onCreateAsset={(request) => { setPreviewArtifact(undefined); onCreateAsset?.(request); }} />}
    </section>
  );
}

function CharacterInventory({characters, artifacts, sessionId, root, revision, portraitImageModel, imageModel, progress, disabled, onAskAgent, onCreateAsset, onRefresh, onPreview}: {
  characters: FilmCharacter[];
  artifacts: Artifact[];
  sessionId: string;
  root: string;
  revision: string;
  portraitImageModel: string;
  imageModel: string;
  progress?: FilmProgress;
  disabled: boolean;
  onAskAgent: (text: string) => Promise<void>;
  onCreateAsset?: (request: AssetCreationRequest) => void;
  onRefresh: () => void | Promise<void>;
  onPreview: (artifact: Artifact) => void;
}) {
  const [configuredImageModel, setConfiguredImageModel] = useState(imageModel);
  useEffect(() => {
    let cancelled = false;
    setConfiguredImageModel(imageModel);
    void getAgentConfig().then((config) => {
      if (!cancelled) setConfiguredImageModel(config.sections.image.model);
    }).catch(() => {});
    return () => { cancelled = true; };
  }, [sessionId, imageModel]);
  const byPath = Object.fromEntries(artifacts.map((artifact) => [artifact.path, artifact]));
  return <section className="character-inventory" aria-label="Characters">
    <header className="workbench-page-heading"><div><span>Character plan</span><h2>Characters</h2></div><span className="artifact-file-count">{characters.length} planned</span></header>
    <CharacterPortraitDecision sessionId={sessionId} root={root} revision={revision} characters={characters} imageModel={configuredImageModel} portraitImageModel={portraitImageModel} progress={progress} disabled={disabled} onAskAgent={onAskAgent} onRefresh={async () => { await onRefresh(); }} />
    {!characters.length ? <p className="artifact-document-state">No character inventory is available yet.</p> : <div className="character-inventory-list">
      {characters.map((character) => <article className="character-inventory-card" key={`${character.scope}:${character.identifier}`}>
        <header><div><h3>{character.identifier}</h3><span>{character.scope}</span></div><span className={`timeline-state is-${character.state}`}>{character.visible ? character.state : 'No portrait required'}</span></header>
        {onCreateAsset && <button type="button" className="workbench-secondary" onClick={() => onCreateAsset({type: 'character', destination: {id: `${character.scope}::${character.identifier}`, name: character.identifier}})}>Generate alternative…</button>}
        <details className="character-description"><summary>Character description</summary><p>{character.description || 'No description provided.'}</p></details>
        {!character.visible ? <p className="plan-hint">Planned but not visible in the film; portraits are not required.</p> : <>
          <div className="character-portrait-views">{(['front', 'side', 'back'] as const).map((view) => {
            const inventoryView = character.views[view];
            const artifact = inventoryView.path ? byPath[inventoryView.path] : undefined;
            return <button key={view} type="button" className={`character-portrait-view state-${inventoryView.state}`} disabled={!artifact} onClick={() => artifact && onPreview(artifact)} aria-label={`${character.identifier} ${view}: ${inventoryView.state}${artifact ? ', preview' : ''}`}>
              {artifact ? <img src={thumbnailUrl(artifact, 180)} loading="lazy" alt={`${character.identifier} ${view} portrait`} /> : <span className="character-portrait-placeholder">{inventoryView.state}</span>}
              <strong>{view}</strong><small>{inventoryView.state}</small>
            </button>;
          })}</div>
          <div className="character-inventory-status"><span>Full set: {character.complete ? 'Complete' : 'Incomplete'}</span><span>Approval: {character.state === 'accepted' ? 'Accepted' : character.state === 'rejected' ? 'Rejected' : character.state === 'stale' ? 'Stale' : character.state === 'rendered' ? 'Needs review' : 'Not rendered'}</span></div>
        </>}
      </article>)}
    </div>}
  </section>;
}

export function ScriptView({session, artifacts, onPlan}: {session?: SessionSummary; artifacts: Artifact[]; onPlan: () => void}) {
  const documents = useMemo(() => artifacts.filter((artifact) => artifact.name.toLowerCase() === 'script.json' || isStoryboardArtifact(artifact)).sort((left, right) => {
    const scriptFirst = Number(right.name.toLowerCase() === 'script.json') - Number(left.name.toLowerCase() === 'script.json');
    return scriptFirst || left.path.localeCompare(right.path, undefined, {numeric: true});
  }), [artifacts]);
  return (
    <section className="artifacts-view workbench-script">
      <header className="workbench-page-heading"><div><span>Story & direction</span><h1>Script</h1></div><span className="artifact-file-count">{documents.length} story documents</span></header>
      {!session ? <ArtifactsEmpty title="Select a project" detail="Read the story and storyboard alongside your film." /> : documents.length ? <DocumentBrowser artifacts={documents} /> : <div className="script-empty"><ArtifactsEmpty title="Every film begins with a story" detail="Develop a script and storyboard with the Assistant, then review them here." /><button className="workbench-primary" onClick={onPlan}>Plan with Assistant</button></div>}
    </section>
  );
}

function DocumentBrowser({artifacts, raw = false}: {artifacts: Artifact[]; raw?: boolean}) {
  const [selectedPath, setSelectedPath] = useState('');
  const selected = artifacts.find((artifact) => artifact.path === selectedPath) || artifacts[0];
  const identity = selected ? `${selected.url}\u0000${selected.updatedAt}` : '';
  const [result, setResult] = useState<{identity: string; document?: JsonValue; error?: string}>();
  useEffect(() => {
    if (!selected) return;
    let cancelled = false;
    void getJsonArtifact(selected).then((document) => {
      if (!cancelled) setResult({identity, document});
    }).catch((error) => {
      if (!cancelled) setResult({identity, error: error instanceof Error ? error.message : String(error)});
    });
    return () => { cancelled = true; };
  }, [identity]);
  const loaded = result?.identity === identity ? result : undefined;
  const storyboards = useMemo(() => selected && loaded?.document !== undefined && isStoryboardArtifact(selected)
    ? extractStoryboardPreviews(loaded.document, selected.path)
    : [], [selected, loaded]);
  if (!selected) return <ArtifactsEmpty title="No documents yet" detail="Source files appear after planning." />;
  return (
    <div className={`artifact-browser ${raw ? 'raw-document-browser' : 'script-document-browser'}`}>
      <nav className="artifact-document-list" aria-label={raw ? 'Source documents' : 'Script and storyboards'}>
        <div className="artifact-document-list-title"><span>{raw ? 'Source files' : 'Story documents'}</span><span>{artifacts.length}</span></div>
        {artifacts.map((artifact) => <button key={artifact.path} className={artifact.path === selected.path ? 'is-selected' : ''} onClick={() => setSelectedPath(artifact.path)} title={artifact.path}>
          <FileJson size={16} /><span><strong>{friendlyArtifactTitle(artifact)}</strong><small>{raw ? artifact.path : formatBytes(artifact.size)}</small></span>
        </button>)}
      </nav>
      <article className="artifact-document">
        <header><div><span>{raw ? 'Raw JSON' : isStoryboardArtifact(selected) ? 'Storyboard' : 'Screenplay'}</span><h2>{friendlyArtifactTitle(selected)}</h2></div><a className="document-original-link" href={artifactUrl(selected)} target="_blank" rel="noreferrer">Open original</a></header>
        <div className="artifact-structured-content">
          {!loaded ? <div className="artifact-document-state" role="status">Loading document…</div> : loaded.error ? <div className="artifact-document-state is-error" role="alert">{loaded.error}</div> : loaded.document !== undefined ? (
            raw ? <pre className="artifact-raw-document">{JSON.stringify(loaded.document, null, 2)}</pre>
              : storyboards.length ? <div className="script-storyboards">{storyboards.map((preview, index) => <section key={preview.id}><span>Shot {index + 1}</span><p>{preview.description}</p></section>)}</div>
                : <StructuredDocument value={loaded.document} artifact={selected} />
          ) : null}
        </div>
      </article>
    </div>
  );
}

function VisualArtifacts({session, artifacts, onPreview, onRefresh}: {session: SessionSummary; artifacts: Artifact[]; onPreview: (artifact: Artifact) => void; onRefresh: () => void}) {
  const [family, setFamily] = useState<VisualFamily | 'all' | 'active'>('all');
  const [selectedPaths, setSelectedPaths] = useState<string[]>([]);
  const [deletionPlan, setDeletionPlan] = useState<AssetsDeletionPlan>();
  const [deletionOpen, setDeletionOpen] = useState(false);
  const [deletionError, setDeletionError] = useState('');
  const [busy, setBusy] = useState(false);
  const alive = useRef(true);
  const deletionRequest = useRef(0);
  const sessionId = session.sessionId;
  const families = useMemo(() => visualFamilyCounts(artifacts), [artifacts]);
  const everything = useMemo(() => visibleVisuals(artifacts, 'all'), [artifacts]);
  const activeCount = artifacts.filter((artifact) => artifact.portrait?.active).length;
  const shown = useMemo(() => family === 'active'
    ? everything.filter((artifact) => artifact.portrait?.active)
    : family === 'all' ? everything : visibleVisuals(artifacts, family), [artifacts, family, everything]);
  const activeHint = families.find((entry) => entry.family === family)?.hint;
  const shownPaths = shown.map((artifact) => artifact.path);
  const shownPathSet = useMemo(() => new Set(shownPaths), [shownPaths]);
  const selectedSet = useMemo(() => new Set(selectedPaths), [selectedPaths]);
  const hiddenSelectedCount = selectedPaths.filter((path) => !shownPathSet.has(path)).length;
  const blocked = busy || deletionOpen;

  useEffect(() => {
    alive.current = true;
    return () => { alive.current = false; };
  }, []);
  useEffect(() => {
    const inventory = new Set(artifacts.map((artifact) => artifact.path));
    if (selectedPaths.some((path) => !inventory.has(path))) {
      deletionRequest.current += 1;
      setSelectedPaths((paths) => paths.filter((path) => inventory.has(path)));
      setDeletionPlan(undefined);
      setDeletionError('The asset inventory changed. Refresh the deletion plan before continuing.');
    }
  }, [artifacts, selectedPaths]);

  const inspectDeletion = async () => {
    if (busy || !selectedPaths.length) return;
    setDeletionOpen(true);
    setDeletionPlan(undefined);
    setDeletionError('');
    setBusy(true);
    const request = ++deletionRequest.current;
    try {
      const plan = await getAssetsDeletionPlan(sessionId, selectedPaths);
      if (alive.current && request === deletionRequest.current) setDeletionPlan(plan);
    } catch (reason) {
      if (alive.current && request === deletionRequest.current) setDeletionError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      if (alive.current) setBusy(false);
    }
  };
  const confirmDeletion = async () => {
    if (!deletionPlan || busy) return;
    setBusy(true);
    setDeletionError('');
    try {
      await deleteAssets({sessionId, paths: deletionPlan.paths, revision: deletionPlan.revision});
      if (!alive.current) return;
      setSelectedPaths([]);
      setDeletionOpen(false);
      setDeletionPlan(undefined);
      onRefresh();
    } catch (reason) {
      if (alive.current) {
        setDeletionError(reason instanceof Error ? reason.message : String(reason));
        setDeletionPlan(undefined);
      }
    } finally {
      if (alive.current) setBusy(false);
    }
  };
  const togglePath = (path: string) => setSelectedPaths((paths) => paths.includes(path) ? paths.filter((item) => item !== path) : [...paths, path]);

  return (
    <section className="artifact-visuals">
      <div className="artifact-view-switcher artifact-family-filter" role="tablist" aria-label="Visual families">
        <button role="tab" aria-selected={family === 'all'} className={family === 'all' ? 'is-selected' : ''} disabled={blocked} onClick={() => setFamily('all')}>
          Everything<span className="artifact-family-count">{everything.length}</span>
        </button>
        {families.map((entry) => (
          <button key={entry.family} role="tab" aria-selected={family === entry.family} className={family === entry.family ? 'is-selected' : ''} disabled={blocked} onClick={() => setFamily(entry.family)} title={entry.hint}>
            {entry.label}<span className="artifact-family-count">{entry.count}</span>
          </button>
        ))}
        <button role="tab" aria-selected={family === 'active'} className={family === 'active' ? 'is-selected' : ''} disabled={blocked} onClick={() => setFamily('active')} title="Portraits referenced by the current render registry">
          Active portraits<span className="artifact-family-count">{activeCount}</span>
        </button>
      </div>
      {family === 'active' && <p className="artifact-family-hint">Active means referenced by the current render registry; accepted means approved for this exact image. These are independent.</p>}
      {activeHint && <p className="artifact-family-hint">{activeHint}</p>}
      <div className="assets-bulk-toolbar" aria-label="Bulk asset selection">
        <span aria-live="polite">{selectedPaths.length} selected{hiddenSelectedCount > 0 ? ` · ${hiddenSelectedCount} hidden by filter` : ''}</span>
        <button className="workbench-secondary" disabled={blocked || shown.length === 0 || shownPaths.every((path) => selectedSet.has(path))} onClick={() => setSelectedPaths((paths) => [...new Set([...paths, ...shownPaths])])}>Select all shown</button>
        <button className="workbench-secondary" disabled={blocked || selectedPaths.length === 0} onClick={() => setSelectedPaths([])}>Clear selection</button>
        <button className="timeline-action is-danger" disabled={blocked || selectedPaths.length === 0} onClick={() => void inspectDeletion()}>Delete selected</button>
      </div>
      {deletionOpen && <div className="asset-deletion-confirmation assets-bulk-confirmation" role="group" aria-label="Confirm bulk asset deletion">
        <h3>Delete these assets permanently?</h3>
        <p>This action is irreversible. All selected media files will be deleted together, or none will be deleted. Other files and unrelated prompts will be preserved.</p>
        <ul className="assets-bulk-paths">{selectedPaths.map((path) => <li key={path}><code>{path}</code></li>)}</ul>
        {deletionError && <p className="is-error" role="alert">{deletionError}</p>}
        {!deletionPlan && !deletionError && <p role="status">Loading deletion details…</p>}
        {deletionPlan && <>
          <h4>Combined impacts</h4>
          {deletionPlan.impacts.length ? <ul>{deletionPlan.impacts.map((impact, index) => <li key={`${index}:${impact}`}>{impact}</li>)}</ul> : <p>No dependent approvals or registry references are affected.</p>}
        </>}
        <div className="asset-deletion-actions">
          <button className="workbench-secondary" disabled={busy} onClick={() => { setDeletionOpen(false); setDeletionPlan(undefined); setDeletionError(''); }}>Cancel</button>
          {deletionError && !deletionPlan && <button className="workbench-secondary" disabled={busy} onClick={() => void inspectDeletion()}>Refresh deletion plan</button>}
          <button className="timeline-action is-danger" disabled={!deletionPlan || busy} onClick={() => void confirmDeletion()}>{busy ? 'Working…' : `Delete ${selectedPaths.length} assets permanently`}</button>
        </div>
      </div>}
      {shown.length > 0 ? (
        <div className="render-grid">
          {shown.map((artifact) => (
            <article key={artifact.path} className={`render-item ${artifact.portrait?.active ? 'is-active-portrait' : ''} ${selectedSet.has(artifact.path) ? 'is-bulk-selected' : ''}`}>
              <label className="asset-select-control">
                <input type="checkbox" checked={selectedSet.has(artifact.path)} disabled={blocked} onChange={() => togglePath(artifact.path)} aria-label={`Select asset ${artifact.path}`} />
                <span>Select</span>
              </label>
              <button className="render-media" disabled={blocked} onClick={() => onPreview(artifact)} aria-label={`Preview ${visualArtifactTitle(artifact.path)} (${artifact.path})`}>
                <img src={thumbnailUrl(artifact)} alt={visualArtifactTitle(artifact.path)} loading="lazy" decoding="async" />
                <i>{artifact.kind === 'video' ? <Video size={15} /> : <ImageIcon size={15} />}</i>
              </button>
              <span className="render-copy">
                <strong>{visualArtifactTitle(artifact.path)}</strong>
                {artifact.portrait && <span className="portrait-badges">{artifact.portrait.active && <b>Active</b>}{artifact.portrait.accepted && <b>Accepted</b>}</span>}
                <small>{artifact.path}<VisualModel path={artifact.path} /></small>
              </span>
            </article>
          ))}
        </div>
      ) : (
        <div className="renders-empty">
          <Film size={24} />
          <strong>{family === 'active' ? 'No active portraits' : 'No visual artifacts yet'}</strong>
          <span>{family === 'active' ? 'No complete portrait sets are referenced by the current render registry. Existing backups are not automatically selected.' : 'Images and videos appear here during rendering'}</span>
        </div>
      )}
    </section>
  );
}


function VisualModel({path}: {path: string}) {
  const model = visualRole(path)?.model;
  if (!model) return null;
  return <em className="visual-model" title={`Produced by ${model}`}>{model}</em>;
}

export function MediaPreviewDialog({session, artifact, artifacts, promptDocument, onRefresh, onClose, onCreateAsset}: {session?: SessionSummary; artifact: Artifact; artifacts: Artifact[]; promptDocument?: Artifact; onRefresh: () => void; onClose: () => void; onCreateAsset?: (request: AssetCreationRequest) => void}) {
  const [prompt, setPrompt] = useState('');
  const [promptError, setPromptError] = useState('');
  const [promptLoaded, setPromptLoaded] = useState(false);
  const [options, setOptions] = useState<PortraitOptions>();
  const [optionsError, setOptionsError] = useState('');
  const [selectedIdentifier, setSelectedIdentifier] = useState('');
  const [appliedIdentifier, setAppliedIdentifier] = useState('');
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState('');
  const [deletionPlan, setDeletionPlan] = useState<AssetDeletionPlan>();
  const [deletionError, setDeletionError] = useState('');
  const [deletionOpen, setDeletionOpen] = useState(false);
  const alive = useRef(true);
  const currentArtifact = artifacts.find((candidate) => candidate.path === artifact.path) || artifact;
  const source = visualPromptSource(artifact.path);
  const promptPath = artifact.promptPath || source?.documentPath;
  const resolvedPromptDocument = promptDocument?.path === promptPath ? promptDocument : undefined;
  const isPortrait = source?.field === 'portrait_prompt' || artifact.portrait !== undefined;

  useEffect(() => {
    alive.current = true;
    return () => { alive.current = false; };
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !busy) {
        if (deletionOpen) setDeletionOpen(false);
        else onClose();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [onClose, busy, deletionOpen]);

  useEffect(() => {
    let cancelled = false;
    setPrompt('');
    setPromptError('');
    setPromptLoaded(false);
    if (!promptPath || !resolvedPromptDocument || !source) {
      setPromptLoaded(true);
      return () => { cancelled = true; };
    }
    void getJsonArtifact(resolvedPromptDocument)
      .then((document) => !cancelled && setPrompt(visualPromptText(document, source.field, artifact.path)))
      .catch((reason) => !cancelled && setPromptError(reason instanceof Error ? reason.message : String(reason)))
      .finally(() => !cancelled && setPromptLoaded(true));
    return () => { cancelled = true; };
  }, [promptPath, resolvedPromptDocument?.path, resolvedPromptDocument?.updatedAt, source?.field, artifact.path]);

  useEffect(() => {
    if (!session || !isPortrait) return;
    let cancelled = false;
    setOptions(undefined);
    setOptionsError('');
    setSelectedIdentifier('');
    setAppliedIdentifier('');
    void getPortraitOptions(session.sessionId, artifact.path).then((result) => {
      if (cancelled) return;
      setOptions(result);
      const folderMatch = artifact.path.match(/(?:^|\/)(\d+)_([^/]+)\/(?:REMOVED_)?(?:front|side|back)\.png$/i)?.[2];
      const hints = [artifact.portrait?.character, folderMatch].filter((item): item is string => Boolean(item));
      const matches = result.characters.filter((character) => hints.includes(character.identifier));
      if (matches.length === 1) setSelectedIdentifier(portraitCharacterKey(matches[0]));
    }).catch((reason) => {
      if (!cancelled) setOptionsError(reason instanceof Error ? reason.message : String(reason));
    });
    return () => { cancelled = true; };
  }, [session?.sessionId, artifact.path, isPortrait]);

  const chosen = options?.characters.find((character) => portraitCharacterKey(character) === selectedIdentifier);
  const canApprove = Boolean(chosen && appliedIdentifier === selectedIdentifier && options?.complete);
  const mutation = async (approve: boolean) => {
    if (!session || !options || !chosen || busy) return;
    const identity = {sessionId: session.sessionId, root: options.root, scope: chosen.scope, character: chosen.identifier, sourcePath: options.sourcePath, revision: options.revision};
    if (!approve) setAppliedIdentifier('');
    setBusy(true);
    setActionError('');
    try {
      const response: PortraitMutationResponse = approve ? await approvePortraitSet(identity) : await selectPortraitSet(identity);
      if (!alive.current) return;
      setOptions(response.options);
      setAppliedIdentifier(selectedIdentifier);
      onRefresh();
    } catch (reason) {
      if (alive.current) setActionError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      if (alive.current) setBusy(false);
    }
  };

  const inspectDeletion = async () => {
    if (!session || busy) return;
    setDeletionOpen(true);
    setDeletionPlan(undefined);
    setDeletionError('');
    setBusy(true);
    try {
      const plan = await getAssetDeletionPlan(session.sessionId, artifact.path);
      if (alive.current) setDeletionPlan(plan);
    } catch (reason) {
      if (alive.current) setDeletionError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      if (alive.current) setBusy(false);
    }
  };

  const confirmDeletion = async () => {
    if (!session || !deletionPlan || busy) return;
    setBusy(true);
    setDeletionError('');
    try {
      await deleteAsset({sessionId: session.sessionId, path: deletionPlan.path, revision: deletionPlan.revision});
      if (!alive.current) return;
      onRefresh();
      onClose();
    } catch (reason) {
      if (alive.current) {
        setDeletionError(reason instanceof Error ? reason.message : String(reason));
        setDeletionPlan(undefined);
      }
    } finally {
      if (alive.current) setBusy(false);
    }
  };

  const viewPaths = options ? [
    ['Front', options.paths.front],
    ['Side', options.paths.side],
    ['Back', options.paths.back],
  ] as const : [];

  return (
    <div className="media-preview-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && !busy && onClose()}>
      <section className={`media-preview-dialog ${isPortrait ? 'has-portrait-controls' : ''}`} role="dialog" aria-modal="true" aria-label={`Preview ${visualArtifactTitle(artifact.path)}`}>
        <header>
          <div>
            <strong>{visualArtifactTitle(artifact.path)}</strong>
            <span>{formatBytes(artifact.size)}<VisualModel path={artifact.path} /></span>
            <code>{artifact.path}</code>
            <a href={artifactUrl(artifact)} target="_blank" rel="noreferrer">Open original</a>
          </div>
          {currentArtifact.portrait && <span className="portrait-badges">{currentArtifact.portrait.active && <b>Active</b>}{currentArtifact.portrait.accepted && <b>Accepted</b>}</span>}
          <button className="icon-button" onClick={onClose} aria-label="Close preview" disabled={busy}><X size={18} /></button>
        </header>
        <section className="asset-deletion-area" aria-label="Delete asset">
          <button className="timeline-action is-danger" onClick={() => void inspectDeletion()} disabled={!session || busy} aria-label={`Delete asset ${artifact.path}`}>Delete asset</button>
          {deletionOpen && <div className="asset-deletion-confirmation" role="group" aria-label="Confirm asset deletion">
            <h3>Delete this asset permanently?</h3>
            <p>This action is irreversible. Only this media file will be deleted. Other files, sets, shots, and prompts will be preserved.</p>
            <code>{deletionPlan?.path || artifact.path}</code>
            {deletionError && <p className="is-error" role="alert">{deletionError}</p>}
            {!deletionPlan && !deletionError && <p role="status">Loading deletion details…</p>}
            {deletionPlan && <>
              <h4>Impacts</h4>
              {deletionPlan.impacts.length ? <ul>{deletionPlan.impacts.map((impact, index) => <li key={`${index}:${impact}`}>{impact}</li>)}</ul> : <p>No dependent approvals or registry references are affected.</p>}
            </>}
            <div className="asset-deletion-actions">
              <button className="workbench-secondary" disabled={busy} onClick={() => { setDeletionOpen(false); setDeletionPlan(undefined); setDeletionError(''); }}>Cancel</button>
              <button className="timeline-action is-danger" disabled={!deletionPlan || busy} onClick={() => void confirmDeletion()}>{busy ? 'Working…' : 'Delete asset permanently'}</button>
            </div>
          </div>}
        </section>
        {artifact.kind === 'image' && session && onCreateAsset && <button type="button" className="workbench-secondary" onClick={() => { onClose(); onCreateAsset({source: artifact}); }}>Generate asset from this image…</button>}
        <div className="media-preview-stage">
          {artifact.kind === 'image'
            ? <img src={artifactUrl(artifact)} alt={visualArtifactTitle(artifact.path)} />
            : <video src={artifactUrl(artifact)} controls autoPlay playsInline preload="metadata" />}
        </div>
        {isPortrait && <section className="portrait-set-controls" aria-label="Portrait set selection and approval">
          <h3>Choose a complete portrait set</h3>
          {!session ? <p>Select a project to manage portrait sets.</p> : optionsError ? <p className="is-error" role="alert">{optionsError}</p>
            : !options ? <p role="status">Loading portrait options…</p> : <>
              <div className="portrait-trio">
                {viewPaths.map(([label, path]) => {
                  const candidate = artifacts.find((item) => item.path === path);
                  return <figure key={label}>
                    {candidate ? <img src={thumbnailUrl(candidate)} alt={`${label} portrait`} /> : <div className="portrait-thumbnail-empty">No preview available</div>}
                    <figcaption><strong>{label}</strong><small>{path}</small></figcaption>
                  </figure>;
                })}
              </div>
              <label className="portrait-character-picker">Current character
                <select value={selectedIdentifier} disabled={busy} onChange={(event) => { setSelectedIdentifier(event.target.value); setAppliedIdentifier(''); setActionError(''); }}>
                  <option value="">Choose a character</option>
                  {options.characters.map((character) => <option key={portraitCharacterKey(character)} value={portraitCharacterKey(character)}>{character.identifier} · {character.scope}{character.active ? ' · current' : ''}</option>)}
                </select>
              </label>
              {!options.complete && <p role="status">This source does not have a complete front, side, and back set; choose another portrait before using or approving it.</p>}
              <p className="portrait-set-warning">Using this set copies these views and their saved prompts into the selected character’s current portrait location. If the image bytes change, that character’s approval and dependent keyframe, clip, and final approvals are invalidated; unrelated media is kept.</p>
              <div className="portrait-set-actions">
                <button className="workbench-secondary" disabled={!chosen || !options.complete || busy} onClick={() => void mutation(false)}>Use this set for {chosen?.identifier || 'character'}</button>
                <button className="workbench-primary" disabled={!canApprove || busy} onClick={() => void mutation(true)}>Approve {chosen?.identifier || 'character'}’s set</button>
              </div>
              {actionError && <p className="is-error" role="alert">{actionError}</p>}
              {chosen && <p className="portrait-approval-state" role="status">{chosen.accepted ? 'This character’s active set is approved.' : appliedIdentifier === selectedIdentifier ? 'Set selected for this character; approval is still required.' : 'Select this set before approving it.'}</p>}
            </>}
        </section>}
        <section className="media-preview-prompt">
          <header><span>Prompt</span><small>{promptPath || 'Not recorded for this artifact'}</small></header>
          {promptError ? <p className="is-error">{promptError}</p>
            : !promptLoaded ? <p>Loading prompt…</p>
              : prompt ? <pre>{prompt}</pre>
                : <p>{resolvedPromptDocument ? 'No prompt was recorded for this view in the saved document.' : 'No saved prompt is available for this artifact.'}</p>}
        </section>
      </section>
    </div>
  );
}

function portraitCharacterKey(character: {identifier: string; scope: string}) {
  return `${character.scope}\u0000${character.identifier}`;
}


function StructuredDocument({value, artifact}: {value: JsonValue; artifact: Artifact}) {
  if (Array.isArray(value)) {
    if (value.length === 0) return <div className="structured-empty">This document is empty</div>;
    if (value.every(isJsonPrimitive)) return <StructuredField label="Contents" value={formatStructuredValue(value)} />;
    return (
      <div className="structured-records">
        {value.map((record, index) => (
          <StructuredRecord key={index} title={structuredRecordTitle(record, index, artifact)} value={record} depth={0} />
        ))}
      </div>
    );
  }
  if (isJsonObject(value)) return <StructuredRecord title="Overview" value={value} depth={0} />;
  return <StructuredField label="Contents" value={formatStructuredValue(value)} />;
}

function StructuredRecord({title, value, depth}: {title: string; value: JsonValue; depth: number}) {
  if (!isJsonObject(value)) return <StructuredField label={title} value={formatStructuredValue(value)} />;
  const entries = Object.entries(value).filter(([key]) => !isArtifactPathField(key));
  const fields = entries.filter(([, entryValue]) => isJsonPrimitive(entryValue) || isPrimitiveArray(entryValue));
  const nested = entries.filter(([, entryValue]) => !isJsonPrimitive(entryValue) && !isPrimitiveArray(entryValue));
  return (
    <section className={`structured-record depth-${Math.min(depth, 3)}`}>
      <header><h3>{title}</h3></header>
      {fields.length > 0 && (
        <div className="structured-fields">
          {fields.map(([key, entryValue]) => (
            <StructuredField key={key} label={friendlyFieldLabel(key)} value={formatStructuredValue(entryValue, key)} />
          ))}
        </div>
      )}
      {nested.map(([key, entryValue]) => (
        <StructuredNested key={key} label={friendlyFieldLabel(key)} value={entryValue} depth={depth + 1} />
      ))}
    </section>
  );
}

function StructuredNested({label, value, depth}: {label: string; value: JsonValue; depth: number}) {
  if (depth > 5) return <StructuredField label={label} value="Additional structured details" />;
  if (Array.isArray(value)) {
    if (value.length === 0) return <StructuredField label={label} value="None" />;
    return (
      <section className="structured-nested">
        <h4>{label}</h4>
        {value.map((item, index) => <StructuredRecord key={index} title={`${label} ${index + 1}`} value={item} depth={depth} />)}
      </section>
    );
  }
  return <StructuredRecord title={label} value={value} depth={depth} />;
}

function StructuredField({label, value}: {label: string; value: string}) {
  return (
    <div className={`structured-field ${value.length > 100 ? 'is-long' : ''}`}>
      <span>{label}</span>
      <p>{value}</p>
    </div>
  );
}

function ArtifactsEmpty({title, detail}: {title: string; detail: string}) {
  return (
    <div className="artifacts-empty">
      <Files size={24} />
      <strong>{title}</strong>
      <span>{detail}</span>
    </div>
  );
}

function isPrimitiveArray(value: JsonValue): value is Array<string | number | boolean | null> {
  return Array.isArray(value) && value.every(isJsonPrimitive);
}


function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

