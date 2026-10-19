import {useCallback, useEffect, useMemo, useRef, useState} from 'react';
import {AlertTriangle, Check, ChevronLeft, ImagePlus, Upload, X} from 'lucide-react';
import {applySetReferenceImage, getAgentConfig, readSetReferences, uploadWorkspaceFile} from './api';
import {artifactUrl, thumbnailUrl} from './media';
import type {Artifact, FilmCharacter, SetReference, SetReferenceInventory} from './types';
import './asset-creation.css';

export type AssetCreationRequest = {source?: Artifact; type?: 'set' | 'character'; destination?: {id: string; name: string}};
type Props = {sessionId: string; projectName: string; root: string; artifacts: Artifact[]; characters: FilmCharacter[]; request: AssetCreationRequest; onClose: () => void; onRefresh: (sessionId: string) => Promise<void>};
type LocalImage = {file: File; url: string; width: number; height: number};
type Source = {kind: 'artifact'; artifact: Artifact} | {kind: 'upload'; image: LocalImage};
type Crop = {x: number; y: number; width: number; height: number};
const DEFAULT_MODEL = 'black-forest-labs/flux-3-image-klein';
const prompts = {set: 'Remove all people from the selected image while preserving the exact viewpoint, layout, and lighting.', character: 'Isolate the selected subject as a character portrait on a neutral background.'};
const initialCrop: Crop = {x: 0, y: 0, width: 100, height: 100};

export function AssetCreationDialog({sessionId, projectName, root, artifacts, characters, request, onClose, onRefresh}: Props) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const closeRef = useRef(onClose);
  closeRef.current = onClose;
  const fileRef = useRef<HTMLInputElement>(null);
  const [source, setSource] = useState<Source | null>(() => request.source?.kind === 'image' ? {kind: 'artifact', artifact: request.source} : null);
  const [type, setType] = useState<'set' | 'character'>(request.type || 'set');
  const [name, setName] = useState(() => request.destination?.name || request.source?.name.replace(/\.[^.]+$/, '') || '');
  const [destination, setDestination] = useState(request.destination?.id || 'new');
  const [view, setView] = useState('other');
  const [mode, setMode] = useState<'as-is' | 'generate'>('as-is');
  const [promptDrafts, setPromptDrafts] = useState<Record<'set' | 'character', string>>(prompts);
  const [model, setModel] = useState(DEFAULT_MODEL);
  const [customModel, setCustomModel] = useState('');
  const [aspect, setAspect] = useState('auto');
  const [resolution, setResolution] = useState('auto');
  const [advanced, setAdvanced] = useState(false);
  const [cropEnabled, setCropEnabled] = useState(false);
  const [crop, setCrop] = useState<Crop>(initialCrop);
  const [localImage, setLocalImage] = useState<LocalImage | null>(null);
  const [inventory, setInventory] = useState<SetReference[]>([]);
  const [inventoryRevision, setInventoryRevision] = useState('');
  const [inventoryShots, setInventoryShots] = useState<SetReferenceInventory['shots']>([]);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState('');
  const [success, setSuccess] = useState('');
  const savingRef = useRef(false);
  const [loadingDestinations, setLoadingDestinations] = useState(true);
  const [inventoryError, setInventoryError] = useState('');
  const [uploadError, setUploadError] = useState('');
  const [validation, setValidation] = useState('');
  const [step, setStep] = useState<'setup' | 'review'>('setup');
  const [dragging, setDragging] = useState(false);
  const [configError, setConfigError] = useState('');
  const [sourceReady, setSourceReady] = useState(false);
  const [sourceLoading, setSourceLoading] = useState(false);
  const [savedModel, setSavedModel] = useState(DEFAULT_MODEL);
  const [imageDims, setImageDims] = useState<{width: number; height: number} | null>(null);
  const [previewError, setPreviewError] = useState('');
  const previousFocus = useRef<HTMLElement | null>(null);
  const imageGeneration = useRef(0);
  const modelChanged = useRef(false);
  const mounted = useRef(true);
  const pendingUrl = useRef<string | null>(null);

  useEffect(() => {
    mounted.current = true;
    previousFocus.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const first = dialogRef.current?.querySelector<HTMLElement>('button, input, select, textarea, a[href], [tabindex="0"]');
    first?.focus();
    const keydown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {event.preventDefault(); if (!savingRef.current) closeRef.current();}
      if (event.key !== 'Tab' || !dialogRef.current) return;
      const focusable = [...dialogRef.current.querySelectorAll<HTMLElement>('button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea:not(:disabled), a[href], [tabindex="0"]')].filter((node) => node.offsetParent !== null);
      if (!focusable.length) {event.preventDefault(); return;}
      const firstNode = focusable[0], lastNode = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === firstNode) {event.preventDefault(); lastNode.focus();}
      else if (!event.shiftKey && document.activeElement === lastNode) {event.preventDefault(); firstNode.focus();}
    };
    document.addEventListener('keydown', keydown);
    return () => {mounted.current = false; imageGeneration.current += 1; if (pendingUrl.current) URL.revokeObjectURL(pendingUrl.current); pendingUrl.current = null; document.removeEventListener('keydown', keydown); previousFocus.current?.focus();};
  }, []);
  useEffect(() => {
    let active = true;
    setLoadingDestinations(true); setInventoryError(''); setConfigError('');
    void getAgentConfig().then((config) => {
      if (!active) return;
      const defaultModel = config.sections.image.editing_model?.trim() || DEFAULT_MODEL;
      setSavedModel(defaultModel);
      if (!modelChanged.current) setModel(defaultModel);
    }).catch((error: unknown) => {if (active) setConfigError(error instanceof Error ? error.message : String(error));});
    void readSetReferences(sessionId, root).then((refs) => {if (active) {setInventory(refs.sets); setInventoryRevision(refs.revision); setInventoryShots(refs.shots);}})
      .catch((error: unknown) => {if (active) setInventoryError(error instanceof Error ? error.message : String(error));})
      .finally(() => {if (active) setLoadingDestinations(false);});
    return () => {active = false;};
  }, [sessionId, root]);
  useEffect(() => () => {if (localImage) URL.revokeObjectURL(localImage.url);}, [localImage]);

  const sourceUrl = source?.kind === 'upload' ? source.image.url : source?.kind === 'artifact' ? artifactUrl(source.artifact) : '';
  const sourceLabel = source?.kind === 'upload' ? source.image.file.name : source?.kind === 'artifact' ? source.artifact.name : '';
  const sourceKey = source?.kind === 'upload' ? source.image.url : source?.kind === 'artifact' ? source.artifact.path : '';
  const destinations = type === 'set' ? inventory.map((set) => ({id: set.id, name: set.name || set.id})) : characters.map((character) => ({id: `${character.scope}::${character.identifier}`, name: `${character.identifier} · ${character.scope}`}));
  const chosenDestination = destinations.find((item) => item.id === destination);
  const selectSourceArtifact = (artifact: Artifact) => {
    imageGeneration.current += 1;
    if (pendingUrl.current) URL.revokeObjectURL(pendingUrl.current);
    pendingUrl.current = null;
    setSource({kind: 'artifact', artifact}); setLocalImage(null); setName((current) => current || artifact.name.replace(/\.[^.]+$/, ''));
    setUploadError(''); setPreviewError(''); setImageDims(null); setCrop(initialCrop); setCropEnabled(false); setSourceReady(false); setSourceLoading(true);
  };
  const loadFile = useCallback((file?: File) => {
    if (!file) return;
    setUploadError('');
    if (!file.type.startsWith('image/')) {imageGeneration.current += 1; if (pendingUrl.current) URL.revokeObjectURL(pendingUrl.current); pendingUrl.current = null; setSourceLoading(false); setUploadError('Choose a supported image file.'); return;}
    const generation = ++imageGeneration.current;
    if (pendingUrl.current) URL.revokeObjectURL(pendingUrl.current);
    setSourceLoading(true); setSourceReady(false); setPreviewError('');
    const url = URL.createObjectURL(file);
    pendingUrl.current = url;
    const image = new Image();
    image.onload = () => {
      if (generation !== imageGeneration.current || !mounted.current) {URL.revokeObjectURL(url); return;}
      pendingUrl.current = null;
      const entry = {file, url, width: image.naturalWidth, height: image.naturalHeight};
      setLocalImage(entry); setSource({kind: 'upload', image: entry}); setName((current) => current || file.name.replace(/\.[^.]+$/, ''));
      setImageDims({width: image.naturalWidth, height: image.naturalHeight}); setPreviewError(''); setCrop(initialCrop); setCropEnabled(false);
      setSourceReady(true); setSourceLoading(false);
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      if (generation === imageGeneration.current && mounted.current) {pendingUrl.current = null; setUploadError('This image could not be opened. Try another file.'); setSourceLoading(false);}
    };
    image.src = url;
  }, []);
  const selectType = (next: 'set' | 'character') => {if (next === type) return; setType(next); setDestination('new'); setValidation('');};
  const changeDestination = (value: string) => {setDestination(value); const target = destinations.find((entry) => entry.id === value); if (target) setName(target.name.split(' · ')[0]);};
  const currentPrompt = promptDrafts[type];
  const changePrompt = (value: string) => setPromptDrafts((drafts) => ({...drafts, [type]: value}));
  const currentModel = model === 'custom' ? customModel.trim() || 'custom model not set' : model;
  const visibleArtifacts = useMemo(() => artifacts.filter((artifact) => artifact.kind === 'image'), [artifacts]);
  const changeModel = (next: string) => {
    modelChanged.current = true; setModel(next);
    const options = next === 'black-forest-labs/flux-3-image-klein' ? ['auto', '768', '1K', '1.5K', '2K', '4K'] : ['auto', '1K', '2K'];
    setResolution((current) => options.includes(current) ? current : 'auto');
  };
  const reloadDestinations = async () => {
    setLoadingDestinations(true); setInventoryError('');
    try {
      const refs = await readSetReferences(sessionId, root);
      setInventory(refs.sets); setInventoryRevision(refs.revision); setInventoryShots(refs.shots);
      setSaveError('');
    } catch (error) {
      setInventoryError(error instanceof Error ? error.message : String(error));
    } finally {setLoadingDestinations(false);}
  };
  const reviewSetup = () => {
    if (!source || !sourceReady || sourceLoading || previewError) {setValidation('Wait for a valid source image to finish loading.'); return;}
    if (!name.trim()) {setValidation('Enter a name for this asset.'); return;}
    if (destination !== 'new' && !chosenDestination) {setValidation('Choose an available destination, or create a new asset.'); return;}
    if (mode === 'generate' && !currentPrompt.trim()) {setValidation('Add instructions for the generated image.'); return;}
    if (mode === 'generate' && model === 'custom' && !customModel.trim()) {setValidation('Enter a model name for the custom OpenRouter model.'); return;}
    setValidation(''); setStep('review');
  };
  const applyImage = async () => {
    if (savingRef.current || type !== 'set' || mode !== 'as-is' || !source || !inventoryRevision || loadingDestinations) return;
    savingRef.current = true; setSaving(true); setSaveError(''); setSuccess('');
    try {
      const image = new Image();
      image.src = sourceUrl;
      await image.decode();
      const canvas = document.createElement('canvas');
      const sx = cropEnabled ? Math.round(image.naturalWidth * crop.x / 100) : 0;
      const sy = cropEnabled ? Math.round(image.naturalHeight * crop.y / 100) : 0;
      const sw = cropEnabled ? Math.max(1, Math.round(image.naturalWidth * crop.width / 100)) : image.naturalWidth;
      const sh = cropEnabled ? Math.max(1, Math.round(image.naturalHeight * crop.height / 100)) : image.naturalHeight;
      canvas.width = sw; canvas.height = sh;
      const context = canvas.getContext('2d');
      if (!context) throw new Error('Could not prepare the PNG image.');
      context.drawImage(image, sx, sy, sw, sh, 0, 0, sw, sh);
      const blob = await new Promise<Blob>((resolve, reject) => canvas.toBlob((value) => value ? resolve(value) : reject(new Error('Could not encode the PNG image.')), 'image/png'));
      const pngBase64 = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader(); reader.onload = () => resolve(String(reader.result).split(',')[1] || ''); reader.onerror = () => reject(new Error('Could not read the PNG image.')); reader.readAsDataURL(blob);
      });
      if (!pngBase64) throw new Error('Could not encode the PNG image.');
      const sourcePath = source.kind === 'artifact' ? source.artifact.path : (await uploadWorkspaceFile(sessionId, source.image.file)).file.path;
      const result = await applySetReferenceImage({sessionId, root, revision: inventoryRevision, ...(destination === 'new' ? {name: name.trim()} : {id: destination}), sourcePath, pngBase64, ...(cropEnabled ? {crop: {x: crop.x, y: crop.y, width: crop.width, height: crop.height}} : {})});
      const appliedSet = destination === 'new' ? result.sets.find((set) => set.name === name.trim()) : result.sets.find((set) => set.id === destination);
      const affectedSet = destination === 'new' ? appliedSet?.id : destination;
      const affectedShots = inventoryShots.filter((shot) => affectedSet && shot.setIds.includes(affectedSet)).map((shot) => shot.slot);
      setInventory(result.sets); setInventoryRevision(result.revision);
      let notice = `Applied image to ${appliedSet?.name || name}.`;
      try {await onRefresh(sessionId);}
      catch {notice = `Saved ${appliedSet?.name || name}, but refreshing Assets failed. Close and reopen Assets to see the persisted reference.`;}
      setStep('review');
      if (affectedShots.length) notice += ` Affected shots: ${affectedShots.join(', ')}; linked successor/final film approvals were invalidated. Existing media was retained; nothing was regenerated.`;
      setSuccess(notice);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setSaveError(`Could not apply image: ${message}. ${/stale|revision|changed|refresh/i.test(message) ? 'Return to setup and reload destinations before trying again; your image selection and settings are retained.' : ''}`);
    } finally {savingRef.current = false; setSaving(false);}
  };
  const imageWidth = imageDims?.width || (source?.kind === 'upload' ? source.image.width : 0);
  const imageHeight = imageDims?.height || (source?.kind === 'upload' ? source.image.height : 0);
  const cropDescription = cropEnabled ? `${Math.round(crop.x)}% left, ${Math.round(crop.y)}% top; ${Math.round(crop.width)}% × ${Math.round(crop.height)}% of source` : type === 'set' ? 'Full image (default for sets)' : 'Full image';
  const destinationDescription = destination === 'new' ? `New ${type}` : `Existing ${type}: ${chosenDestination?.name || name}`;
  const generatedMode = mode === 'generate';
  return <div className="asset-create-backdrop" onMouseDown={(event) => {if (event.target === event.currentTarget && !saving) onClose();}}>

    <section className="asset-create-dialog" role="dialog" aria-modal="true" aria-labelledby="asset-create-title" ref={dialogRef}>
      <header className="asset-create-header"><div><span className="asset-create-eyebrow">{projectName || 'Project'} · asset setup</span><h2 id="asset-create-title">{step === 'setup' ? 'Create an asset' : 'Review asset setup'}</h2></div><button type="button" className="asset-create-icon-button" onClick={() => !saving && onClose()} disabled={saving} aria-label="Close asset setup"><X size={18}/></button></header>
      <div className="asset-create-notice" role="note"><AlertTriangle size={17}/><p>{step === 'review' && type === 'set' && !generatedMode ? <><strong>Explicit Apply saves this image as the set reference and accepts the set.</strong> No model calls or shot generation occur.</> : <><strong>Review only.</strong> Character creation and paid image generation are not implemented; no model calls or shot generation occur.</>}</p></div>
      {saveError && <p className="asset-create-error asset-create-validation" role="alert">{saveError}</p>}
      {success && <p className="asset-create-cost asset-create-validation" role="status">{success}</p>}
      {step === 'setup' ? <>
        <div className="asset-create-body">
          <div className="asset-create-source-column">
            <section className="asset-create-panel"><div className="asset-create-section-title"><div><span className="asset-create-step">01</span><h3>Choose a source image</h3></div><button type="button" className="asset-create-secondary" onClick={() => fileRef.current?.click()}><Upload size={15}/> Upload</button></div>
              <input ref={fileRef} className="asset-create-file-input" type="file" accept="image/*" aria-label="Choose image from device" onChange={(event) => {loadFile(event.currentTarget.files?.[0]); event.currentTarget.value = '';}}/>
              <div className={`asset-create-drop ${dragging ? 'is-dragging' : ''}`} onDragOver={(event) => {event.preventDefault(); setDragging(true);}} onDragLeave={() => setDragging(false)} onDrop={(event) => {event.preventDefault(); setDragging(false); loadFile(event.dataTransfer.files[0]);}}>
                {visibleArtifacts.length ? <div className="asset-create-project-images" aria-label="Project images">{visibleArtifacts.map((artifact) => <button key={artifact.path} type="button" className={`asset-create-thumb ${source?.kind === 'artifact' && source.artifact.path === artifact.path ? 'is-selected' : ''}`} aria-pressed={source?.kind === 'artifact' && source.artifact.path === artifact.path} onClick={() => selectSourceArtifact(artifact)} title={artifact.name}><img src={thumbnailUrl(artifact, 160)} alt=""/><span>{artifact.name}</span></button>)}</div> : <span className="asset-create-empty">No project images are available.</span>}
                <div className="asset-create-drop-hint"><ImagePlus size={16}/><span>Drop an image here or choose one from your device</span></div>
              </div>
              {uploadError && <p className="asset-create-error" role="alert">{uploadError}</p>}
            </section>
            <section className="asset-create-panel"><div className="asset-create-section-title"><div><span className="asset-create-step">02</span><h3>Asset details</h3></div></div>
              <div className="asset-create-field"><label>Asset type</label><div className="asset-create-segmented"><button type="button" aria-pressed={type === 'set'} onClick={() => selectType('set')}>Set</button><button type="button" aria-pressed={type === 'character'} onClick={() => selectType('character')}>Character</button></div></div>
              <label className="asset-create-field">Name<input value={name} onChange={(event) => setName(event.target.value)} placeholder={type === 'set' ? 'e.g. Rooftop café' : 'e.g. Mira'} /></label>
              <label className="asset-create-field">Destination<select value={destination} onChange={(event) => changeDestination(event.target.value)}><option value="new">Create a new {type}</option>{destinations.map((entry) => <option key={entry.id} value={entry.id}>Existing · {entry.name}</option>)}</select></label>
              {loadingDestinations && <small className="asset-create-muted">Loading existing destinations…</small>}
              <button type="button" className="asset-create-secondary" onClick={() => void reloadDestinations()} disabled={loadingDestinations}>Reload destinations</button>
              {inventoryError && <p className="asset-create-muted">Could not load existing sets: {inventoryError}</p>}
              {configError && <p className="asset-create-muted">Could not load the saved editing model; FLUX default remains available. {configError}</p>}
              {type === 'character' && <label className="asset-create-field">Character view<select value={view} onChange={(event) => setView(event.target.value)}><option value="other">Other reference</option><option value="front">Front</option><option value="side">Side</option><option value="back">Back</option></select></label>}
              {type === 'character' && destination !== 'new' && <p className="asset-create-muted">This is one portrait view only—not a complete or approved trio.</p>}
            </section>
          </div>
          <div className="asset-create-edit-column">
            <section className="asset-create-panel"><div className="asset-create-section-title"><div><span className="asset-create-step">03</span><h3>Source and framing</h3></div></div>
              {source ? <><div className="asset-create-preview" key={sourceKey}><div className="asset-create-image-box"><img src={sourceUrl} alt={`Selected source: ${sourceLabel}`} onLoad={(event) => {setImageDims({width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight}); setSourceReady(true); setSourceLoading(false); setPreviewError('');}} onError={() => {setPreviewError('The selected image could not be previewed. Choose another image.'); setSourceReady(false); setSourceLoading(false);}} />{cropEnabled && <div className="asset-create-crop-outline" style={{left: `${crop.x}%`, top: `${crop.y}%`, width: `${crop.width}%`, height: `${crop.height}%`}} aria-label="Crop rectangle"/>}</div></div>
                <div className="asset-create-source-meta"><strong title={sourceLabel}>{sourceLabel}</strong><span>{imageWidth && imageHeight ? `${imageWidth} × ${imageHeight}px` : 'Loading image dimensions…'}</span><a href={sourceUrl} target="_blank" rel="noreferrer">View original image</a></div>
                <label className="asset-create-check"><input type="checkbox" checked={cropEnabled} onChange={(event) => setCropEnabled(event.target.checked)}/> Use a crop / subject rectangle</label>
                {cropEnabled && <div className="asset-create-crop-controls">{(['x', 'y', 'width', 'height'] as const).map((key) => <label key={key}>{key === 'x' ? 'Left' : key === 'y' ? 'Top' : key === 'width' ? 'Width' : 'Height'} <input type="range" min="0" max="100" value={crop[key]} onChange={(event) => setCrop((current) => {const value = Number(event.target.value); if (key === 'x') return {...current, x: Math.min(value, 100 - current.width)}; if (key === 'y') return {...current, y: Math.min(value, 100 - current.height)}; if (key === 'width') return {...current, width: Math.max(10, Math.min(value, 100 - current.x))}; return {...current, height: Math.max(10, Math.min(value, 100 - current.y))};})}/><output>{Math.round(crop[key])}%</output></label>)}</div>}
                {cropEnabled && <div className="asset-create-crop-result"><span>Crop preview</span><div className="asset-create-crop-preview" style={{aspectRatio: imageWidth && imageHeight ? `${imageWidth * crop.width} / ${imageHeight * crop.height}` : undefined}}><img src={sourceUrl} alt="Preview of selected crop" style={{width: `${10000 / crop.width}%`, height: `${10000 / crop.height}%`, left: `${-crop.x * 100 / crop.width}%`, top: `${-crop.y * 100 / crop.height}%`}}/></div></div>}
              </> : <div className="asset-create-no-source"><ImagePlus size={24}/><span>Select a project image or upload from your device</span></div>}
            </section>
            <section className="asset-create-panel"><div className="asset-create-section-title"><div><span className="asset-create-step">04</span><h3>How to use this image</h3></div></div>
              <div className="asset-create-mode-options"><label className={mode === 'as-is' ? 'is-selected' : ''}><input type="radio" name="asset-mode" checked={mode === 'as-is'} onChange={() => setMode('as-is')}/><span><strong>{source?.kind === 'upload' ? 'Use uploaded image as-is' : 'Use image as-is'}</strong><small>No instructions or model call. No model cost.</small></span></label><label className={mode === 'generate' ? 'is-selected' : ''}><input type="radio" name="asset-mode" checked={mode === 'generate'} onChange={() => setMode('generate')}/><span><strong>Generate from image</strong><small>Prepare editable instructions; one image request would be paid.</small></span></label></div>
              {generatedMode && <><label className="asset-create-field">Instructions<textarea rows={4} value={currentPrompt} onChange={(event) => changePrompt(event.target.value)} placeholder="Describe the image you want…"/></label><div className="asset-create-provider"><span>Provider</span><strong>OpenRouter</strong><small>Per-request choice only; global settings stay unchanged.</small></div><label className="asset-create-field">Image model<select value={model} onChange={(event) => changeModel(event.target.value)}><option value={savedModel}>Saved editing model · {savedModel}</option><option value="black-forest-labs/flux-3-image-klein">FLUX.3 Image Klein</option><option value="krea/k3-edit">Krea K3 Edit</option><option value="qwen/qwen-image-2.1-turbo">Qwen Image 2.1 Turbo</option><option value="custom">Custom OpenRouter model…</option></select></label>{model === 'custom' && <label className="asset-create-field">Custom model<input value={customModel} onChange={(event) => setCustomModel(event.target.value)} placeholder="provider/model-id"/></label>}
                <button type="button" className="asset-create-advanced-toggle" aria-expanded={advanced} onClick={() => setAdvanced((value) => !value)}>Advanced controls {advanced ? '−' : '+'}</button>{advanced && <div className="asset-create-advanced"><label className="asset-create-field">Aspect ratio<select value={aspect} onChange={(event) => setAspect(event.target.value)}>{['auto', '1:1', '3:2', '2:3', '4:3', '3:4', '16:9', '9:16'].map((option) => <option key={option}>{option}</option>)}</select></label><label className="asset-create-field">Resolution<select value={resolution} onChange={(event) => setResolution(event.target.value)}><option value="auto">Provider default</option>{(model === 'black-forest-labs/flux-3-image-klein' ? ['768', '1K', '1.5K', '2K', '4K'] : model === 'krea/k3-edit' || model === 'qwen/qwen-image-2.1-turbo' ? ['1K', '2K'] : ['1K', '2K', '4K']).map((option) => <option key={option}>{option}</option>)}</select></label></div>}</>}
            </section>
          </div>
        </div>
        {validation && <p className="asset-create-error asset-create-validation" role="alert">{validation}</p>}
        <footer className="asset-create-footer"><span>Review only · no asset is saved or generated</span><div><button type="button" className="asset-create-secondary" onClick={onClose}>Close</button><button type="button" className="asset-create-primary" onClick={reviewSetup} disabled={Boolean(source && (!sourceReady || sourceLoading || previewError))}>Review asset setup <ChevronLeft size={15} className="asset-create-flip"/></button></div></footer>
      </> : <>
        <div className="asset-create-review-body"><div className="asset-create-review-image">{source && <img src={sourceUrl} alt={`Review source: ${sourceLabel}`} />}{cropEnabled && source && <div className="asset-create-review-crop" style={{aspectRatio: imageWidth && imageHeight ? `${imageWidth * crop.width} / ${imageHeight * crop.height}` : undefined}}><img src={sourceUrl} alt="Selected crop preview" style={{width: `${10000 / crop.width}%`, height: `${10000 / crop.height}%`, left: `${-crop.x * 100 / crop.width}%`, top: `${-crop.y * 100 / crop.height}%`}}/></div>}<span>{sourceLabel} · {imageWidth} × {imageHeight}px</span></div>
          <div className="asset-create-summary"><div className="asset-create-summary-heading"><Check size={17}/><strong>Proposed setup</strong></div><dl><dt>Asset</dt><dd>{name} · {type}</dd><dt>Destination</dt><dd>{destinationDescription}</dd>{type === 'character' && <><dt>Character view</dt><dd>{view}</dd></>}<dt>Source</dt><dd>{sourceLabel}</dd><dt>Framing</dt><dd>{cropDescription}</dd><dt>Mode</dt><dd>{generatedMode ? 'Generate from image' : source?.kind === 'upload' ? 'Use uploaded image as-is' : 'Use image as-is'}</dd></dl>
            {generatedMode ? <div className="asset-create-review-instructions"><strong>Exact instructions</strong><p>{currentPrompt}</p><dl><dt>Provider</dt><dd>OpenRouter</dd><dt>Model</dt><dd>{currentModel}</dd><dt>Aspect ratio</dt><dd>{aspect}</dd><dt>Resolution</dt><dd>{resolution === 'auto' ? 'Provider default' : resolution}</dd></dl></div> : <p className="asset-create-cost">Zero model calls · zero model cost.</p>}
            {generatedMode && <p className="asset-create-cost">One paid image would be requested if generation were implemented; no request was sent.</p>}
            {type === 'set' && !generatedMode && <p className="asset-create-no-result">Explicit Apply accepts this set. Shots bound to the existing set: {inventoryShots.filter((shot) => destination !== 'new' && shot.setIds.includes(destination)).map((shot) => shot.slot).join(', ') || 'none'}. Bound shots and linked successor/final film approvals are cleared; media is retained and nothing is regenerated.</p>}
          </div>
        </div>
        <footer className="asset-create-footer"><button type="button" className="asset-create-secondary" onClick={() => !saving && !success && setStep('setup')} disabled={saving || Boolean(success)}><ChevronLeft size={15}/> Edit setup</button><button type="button" className="asset-create-secondary" onClick={() => !saving && onClose()} disabled={saving}>{success ? 'Done' : 'Close'}</button>{type === 'set' && !generatedMode && !success && <button type="button" className="asset-create-primary" onClick={() => void applyImage()} disabled={saving || loadingDestinations || !inventoryRevision || !source || !sourceReady}>{saving ? 'Applying…' : `Apply image to ${destination === 'new' ? 'new set' : chosenDestination?.name || 'set'}`}</button>}</footer>
      </>}
    </section>
  </div>;
}
