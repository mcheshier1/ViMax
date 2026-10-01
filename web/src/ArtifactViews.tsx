import {useEffect, useMemo, useState} from 'react';
import {FileJson, Film, Files, Image as ImageIcon, Video, X} from 'lucide-react';
import {getJsonArtifact} from './api';
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
import type {Artifact, JsonValue, SessionSummary} from './types';


export function ArtifactsView({session, artifacts}: {session?: SessionSummary; artifacts: Artifact[]}) {
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [previewArtifact, setPreviewArtifact] = useState<Artifact>();
  const mediaArtifacts = useMemo(() => sortVisuals(artifacts.filter((artifact) => artifact.kind === 'image' || artifact.kind === 'video')), [artifacts]);
  const jsonArtifacts = useMemo(() => artifacts.filter(isJsonArtifact).sort((left, right) => left.path.localeCompare(right.path, undefined, {numeric: true})), [artifacts]);
  return (
    <section className="artifacts-view workbench-assets">
      <header className="workbench-page-heading"><div><span>Media library</span><h1>Assets</h1></div><span className="artifact-file-count">{mediaArtifacts.length} visuals · {jsonArtifacts.length} documents</span></header>
      {!session ? <ArtifactsEmpty title="Select a project" detail="Your images, clips, and source files will appear here." /> : <>
        <VisualArtifacts artifacts={mediaArtifacts} onPreview={setPreviewArtifact} />
        <details className="assets-advanced" open={advancedOpen} onToggle={(event) => setAdvancedOpen(event.currentTarget.open)}>
          <summary><FileJson size={16} /><span>Advanced · raw documents</span><small>{jsonArtifacts.length} files</small></summary>
          {advancedOpen && <DocumentBrowser artifacts={jsonArtifacts} raw />}
        </details>
      </>}
      {previewArtifact && <MediaPreviewDialog key={previewArtifact.path} artifact={previewArtifact} promptDocument={artifacts.find((candidate) => candidate.path === visualPromptSource(previewArtifact.path)?.documentPath)} onClose={() => setPreviewArtifact(undefined)} />}
    </section>
  );
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

function VisualArtifacts({artifacts, onPreview}: {artifacts: Artifact[]; onPreview: (artifact: Artifact) => void}) {
  const [family, setFamily] = useState<VisualFamily | 'all'>('all');
  const families = useMemo(() => visualFamilyCounts(artifacts), [artifacts]);
  const everything = useMemo(() => visibleVisuals(artifacts, 'all'), [artifacts]);
  const shown = useMemo(() => (family === 'all' ? everything : visibleVisuals(artifacts, family)), [artifacts, family, everything]);
  const activeHint = families.find((entry) => entry.family === family)?.hint;
  return (
    <section className="artifact-visuals">
      {families.length > 1 && (
        <div className="artifact-view-switcher artifact-family-filter" role="tablist" aria-label="Visual families">
          <button role="tab" aria-selected={family === 'all'} className={family === 'all' ? 'is-selected' : ''} onClick={() => setFamily('all')}>
            Everything<span className="artifact-family-count">{everything.length}</span>
          </button>
          {families.map((entry) => (
            <button
              key={entry.family}
              role="tab"
              aria-selected={family === entry.family}
              className={family === entry.family ? 'is-selected' : ''}
              onClick={() => setFamily(entry.family)}
              title={entry.hint}
            >
              {entry.label}<span className="artifact-family-count">{entry.count}</span>
            </button>
          ))}
        </div>
      )}
      {activeHint && <p className="artifact-family-hint">{activeHint}</p>}
      {shown.length > 0 ? (
        <div className="render-grid">
          {shown.map((artifact) => (
            <article key={artifact.path} className="render-item">
              <button className="render-media" onClick={() => onPreview(artifact)} aria-label={`Preview ${visualArtifactTitle(artifact.path)}`}>
                <img src={thumbnailUrl(artifact)} alt={visualArtifactTitle(artifact.path)} loading="lazy" decoding="async" />
                <i>{artifact.kind === 'video' ? <Video size={15} /> : <ImageIcon size={15} />}</i>
              </button>
              <span className="render-copy">
                <strong>{visualArtifactTitle(artifact.path)}</strong>
                <small>{formatBytes(artifact.size)}<VisualModel path={artifact.path} /></small>
              </span>
            </article>
          ))}
        </div>
      ) : (
        <div className="renders-empty">
          <Film size={24} />
          <strong>No visual artifacts yet</strong>
          <span>Images and videos appear here during rendering</span>
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

export function MediaPreviewDialog({artifact, promptDocument, onClose}: {artifact: Artifact; promptDocument?: Artifact; onClose: () => void}) {
  const [prompt, setPrompt] = useState('');
  const [promptError, setPromptError] = useState('');
  const source = visualPromptSource(artifact.path);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [onClose]);

  useEffect(() => {
    let cancelled = false;
    setPrompt('');
    setPromptError('');
    if (!promptDocument || !source) return () => { cancelled = true; };
    void getJsonArtifact(promptDocument)
      .then((document) => !cancelled && setPrompt(visualPromptText(document, source.field, artifact.path)))
      .catch((reason) => !cancelled && setPromptError(reason instanceof Error ? reason.message : String(reason)));
    return () => { cancelled = true; };
  }, [promptDocument?.path, promptDocument?.updatedAt, source?.documentPath, source?.field, artifact.path]);

  return (
    <div className="media-preview-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <section className="media-preview-dialog" role="dialog" aria-modal="true" aria-label={`Preview ${visualArtifactTitle(artifact.path)}`}>
        <header>
          <div>
            <strong>{visualArtifactTitle(artifact.path)}</strong>
            <span>{formatBytes(artifact.size)}<VisualModel path={artifact.path} /></span>
          </div>
          <button className="icon-button" onClick={onClose} aria-label="Close preview"><X size={18} /></button>
        </header>
        <div className="media-preview-stage">
          {artifact.kind === 'image'
            ? <img src={artifactUrl(artifact)} alt={visualArtifactTitle(artifact.path)} />
            : <video src={artifactUrl(artifact)} controls autoPlay playsInline preload="metadata" />}
        </div>
        <section className="media-preview-prompt">
          <header><span>Prompt</span><small>{source?.documentPath || 'Not recorded for this artifact'}</small></header>
          {promptError
            ? <p className="is-error">{promptError}</p>
            : prompt
              ? <pre>{prompt}</pre>
              : <p>{promptDocument && source ? 'Loading prompt…' : 'No saved prompt is available for this artifact.'}</p>}
        </section>
      </section>
    </div>
  );
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

