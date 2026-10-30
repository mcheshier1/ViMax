export type SessionSummary = {
  sessionId: string;
  projectName: string;
  workingDir: string;
  stage: string;
  summary: string;
  idea: string;
  updatedAt: string;
  createdAt: string;
  compactionTurns: number;
};

export type ProjectManifest = {
  image_model?: string;
  video_model?: string;
  style?: string;
};

export type ProjectMetadata = {
  sessionId: string;
  projectName: string;
  idea: string;
  userRequirement: string;
  style: string;
  stage: string;
  summary: string;
  workingDir: string;
  manifest: ProjectManifest | null;
  /** Render output found on disk, per phase: what a style change deletes and a render rebuilds. */
  artifacts: ProjectArtifacts;
};

export type ProjectArtifacts = {
  /** Portrait sets (one per character) present on disk. */
  portraits: number;
  /** Generated location reference images present on disk. */
  setReferences: number;
  /** Shot directories found on disk. */
  shots: number;
  /** Shots with a rendered first keyframe. */
  frames: number;
  /** Shots with a rendered clip. */
  clips: number;
  finalVideo: boolean;
};

/** A render slot's state, derived from the files on disk and the lock at call time. */
export type AcceptanceState = 'planned' | 'rendered' | 'accepted' | 'rejected' | 'stale';

export type AcceptanceStageName = 'portraits' | 'keyframes' | 'clips' | 'final_video';

export type AcceptanceSlot = {
  /** Shot index for shot-scoped stages, null for the session-wide stages. */
  shot: string | null;
  state: AcceptanceState;
  /** Artifact paths relative to the session working dir. */
  artifacts: string[];
  /** The human's note, present only while the slot is rejected. */
  reason?: string;
  /** When the rejection was recorded, present only while the slot is rejected. */
  rejectedAt?: string;
  /** When the artifacts on disk were last regenerated in response to a rejection. */
  redoneAt?: string;
  /** The note that prompted that regeneration, kept on what replaced the rejected take. */
  note?: string;
};

export type AcceptanceStage = {
  stage: AcceptanceStageName;
  scope: 'session' | 'shot';
  state: AcceptanceState;
  /** True only when every slot of the stage is accepted. */
  accepted: boolean;
  slots: AcceptanceSlot[];
};

export type AcceptanceTotals = {
  acceptedKeyframes: number;
  keyframes: number;
  clips: number;
  /** Required frame artifact types for each shot, in generation order. */
  requiredFrameTypes: Array<'first_frame' | 'last_frame'>;
  /** Rejected slots across the whole payload. */
  rejected: number;
  clipSeconds: number;
  clipCostUsd: number;
};

export type CharacterPortraitViewState = {
  path?: string;
  state: 'missing' | 'invalid' | 'ready';
};

export type FilmCharacter = {
  identifier: string;
  scope: string;
  description: string;
  visible: boolean;
  complete: boolean;
  state: AcceptanceState;
  views: {
    front: CharacterPortraitViewState;
    side: CharacterPortraitViewState;
    back: CharacterPortraitViewState;
  };
};

export type RenderAcceptance = {
  root: string;
  stages: AcceptanceStage[];
  totals: AcceptanceTotals;
  characters: FilmCharacter[];
  portraitImageModel: string;
  chainedKeyframes?: boolean;
};

export type ShotRestartPlan = {
  sessionId: string;
  root: string;
  slot: string;
  revision: string;
  paths: string[];
  continuityEnabled: boolean;
};
export type SetReferenceCandidate = {
  id: string;
  imagePath: string;
  state: 'ready' | 'stale' | 'invalid';
  prompt?: string;
};

export type SetReference = {
  id: string;
  name: string;
  description: string;
  shots: string[];
  state: AcceptanceState;
  imagePath?: string;
  imageModel?: string;
  candidates: SetReferenceCandidate[];
  selectedCandidateId?: string;
  revision: string;
};

export type SetReferenceShot = {slot: string; brief: string; setIds: string[]; setBindingExplicit: boolean};

export type SetReferenceInventory = {
  sessionId: string;
  root: string;
  revision: string;
  imageModel: string;
  style: string;
  sets: SetReference[];
  shots: SetReferenceShot[];
  unassignedShots: string[];
};

export type SetReferenceSelectionRequest = {
  sessionId: string;
  root: string;
  id: string;
  candidateId: string;
  revision: string;
};

export type SetReferenceUpdateRequest = {
  sessionId: string;
  root: string;
  id: string;
  name: string;
  description: string;
  shots: string[];
  revision: string;
};

export type SetReferenceAcceptanceRequest = {
  sessionId: string;
  root: string;
  id: string;
  revision: string;
  accepted: boolean;
  reason?: string;
};

export type VideoDurationPolicy = {
  provider: string;
  model: string;
  configured: boolean;
  defaultSeconds: number;
  minSeconds: number;
  maxSeconds: number;
  allowedSeconds: number[] | null;
};

export type ShotPlan = {
  slot: string;
  root: string;
  revision: string;
  /** The one-line brief the shot came from. */
  brief: string;
  characters: {idx: number; name: string}[];
  firstFrame: PlanFrame;
  lastFrame: PlanFrame;
  motionDescription: string;
  durationSeconds: number;
  durationOverrideSeconds: number | null;
  durationPolicy: VideoDurationPolicy;
  sets: Array<{id: string; name: string; accepted: boolean}>;
  setIds: string[];
  setRevision: string;
  reusePreviousLastFrame: boolean;
  previousSlot: string | null;
  previousClipAvailable: boolean;
  previousClipAccepted: boolean;
  previousClipSha256: string | null;
  renderedEndingStatus: 'off' | 'missing_clip' | 'unreviewed_clip' | 'not_prepared' | 'legacy_planned' | 'current' | 'stale';
  renderedEndingReason: string;
};

export type PlanFrame = {
  description: string;
  /** Indices of the characters this frame shows. */
  visible: number[];
  /** The prompt the frame was last drawn from, as it was sent. Empty until it is drawn. */
  prompt: string;
};

export type ShotPlanUpdateRequest = {
  sessionId: string;
  root: string;
  slot: string;
  revision?: string;
  ffDesc?: string;
  lfDesc?: string;
  ffVis?: number[];
  lfVis?: number[];
  brief?: string;
  motionDesc?: string;
  setIds?: string[];
  setRevision?: string;
  reusePreviousLastFrame?: boolean;
  previousSlot?: string | null;
  durationSeconds?: number | null;
};
/** One row of the render's trail: either a phase decision or a pipeline progress note. */
export type RenderEvent = {
  timestamp?: string;
  status?: string;
  phase?: string;
  /** Shots a redraw cleared, which are the ones being regenerated right now. */
  redone_shots?: string[];
  error_type?: string;
  error?: string;
  awaiting_confirmation?: string;
};


/** A shot taken out of the film, and what is kept for it. */
export type RemovedShot = {slot: string; files: number; hasBrief: boolean};

/** A story beat the review placed against the timeline. */
export type ContinuityBeat = {
  index: number;
  text: string;
  status: 'covered' | 'partial' | 'missing';
  /** Shot numbers that play this beat; empty means nothing does. */
  covered_by: number[];
  note: string;
};

/** A shot the review proposes for a beat that nothing covers. */
export type ContinuitySuggestion = {
  id: string;
  /** Slot key of the shot it should follow. */
  after_shot: number;
  title: string;
  visual_desc: string;
  /** The dialogue the shot speaks, verbatim from the script or from the cut shot. */
  audio_desc: string;
  /** What moves in the shot, in the film's own style. */
  motion_desc: string;
  characters: string[];
  rationale: string;
  frames: {first: string; last: string};
};

/** A finding computed from the timeline itself, with no model involved. */
export type ContinuityCheck = {
  id: string;
  status: 'ok' | 'warn';
  message: string;
  shots: number[];
};

/** The script-coverage review of one render root, as the agent recorded it. */
export type ContinuityReview = {
  reviewed_at: string;
  root: string;
  session_id?: string;
  shots_reviewed: number[];
  summary: string;
  beats: ContinuityBeat[];
  suggestions: ContinuitySuggestion[];
  checks: ContinuityCheck[];
  notes: string;
};

/** The review, plus what only the server can derive: whether it still describes the film. */
export type ContinuityPayload = {
  root: string;
  review: ContinuityReview | null;
  stale: boolean;
  staleReason: string;
  /** Slot keys of the film's shots, in playing order. */
  shots: string[];
  /** Default duration for shots without a saved per-shot override. */
  clipSeconds: number;
};


export type AcceptanceUpdateRequest = {
  sessionId: string;
  root?: string;
  stage: AcceptanceStageName;
  shot?: string;
  accepted: boolean;
  /** Rejection note; non-empty with accepted:false records a rejection. */
  reason?: string;
};

export type ProjectUpdateRequest = {
  sessionId: string;
  projectName?: string;
  idea?: string;
  userRequirement?: string;
  style?: string;
  invalidate?: boolean;
};

export type ProjectUpdateResponse = {
  session: ProjectMetadata;
  changed: string[];
  invalidated: string[];
  requiresInvalidation: boolean;
  /** True when a confirmed style change also respawned the agent to rebuild the artifacts. */
  regenerationStarted?: boolean;
};

export type AssemblyLightingReport = {
  status: 'corrected' | 'unchanged' | 'needs_review';
  reason: string;
  gains: [number, number, number];
  previous_slot: string | null;
  [evidence: string]: unknown;
};

export type AssemblyCorrectionReport = {
  version: number;
  canvas: {width: number; height: number; fps: number};
  crop_zoom: number;
  clips: Array<{
    slot: string;
    status: 'corrected' | 'unchanged' | 'needs_review';
    reason: string;
    duration_seconds: number;
    lighting?: AssemblyLightingReport;
  }>;
};

export type ConfigSection = {
  model_provider?: string;
  model: string;
  editing_model?: string;
  base_url: string;
  api_key: string;
  has_api_key: boolean;
  configured_api_key_providers?: string[];
  api_key_suffixes?: Record<string, string>;
  api_key_updates?: Record<string, string>;
  provider?: string;
  resolution?: string;
  effective_resolution?: string;
  clip_seconds?: string;
  effective_clip_seconds?: string;
  shot_duration_limits?: Record<string, number>;
  generate_audio?: string;
};

export type AgentConfig = {
  sections: Record<'llm' | 'image' | 'video' | 'embedding' | 'reranker', ConfigSection>;
};

export type ArtifactPortrait = {
  active: boolean;
  accepted: boolean;
  character?: string;
  view: 'front' | 'side' | 'back';
  registryPath?: string;
};

export type Artifact = {
  path: string;
  name: string;
  kind: 'image' | 'video' | 'document';
  size: number;
  updatedAt: string;
  url: string;
  portrait?: ArtifactPortrait;
  promptPath?: string;
};
export type AssetDeletionPlan = {
  path: string;
  revision: string;
  impacts: string[];
};

export type AssetsDeletionPlan = {
  paths: string[];
  revision: string;
  impacts: string[];
};

export type AssetDeletionResponse = {
  deletedPath: string;
  artifacts: Artifact[];
};

export type AssetsDeletionResponse = {
  deletedPaths: string[];
  artifacts: Artifact[];
};

export type AssetDeletionRequest = {
  sessionId: string;
  path: string;
  revision: string;
};

export type AssetsDeletionRequest = {
  sessionId: string;
  paths: string[];
  revision: string;
};
export type PortraitCharacterOption = {
  identifier: string;
  idx: number;
  scope: string;
  active: boolean;
  accepted: boolean;
};

export type PortraitOptions = {
  root: string;
  revision: string;
  sourcePath: string;
  paths: {front: string; side: string; back: string};
  complete: boolean;
  characters: PortraitCharacterOption[];
};

export type PortraitMutationResponse = {
  artifacts: Artifact[];
  options: PortraitOptions;
};

export type PortraitMutationRequest = {
  sessionId: string;
  root: string;
  scope: string;
  character: string;
  sourcePath: string;
  revision: string;
};


export type WorkspaceUpload = {
  name: string;
  path: string;
  size: number;
};

export type JsonPrimitive = string | number | boolean | null;

export type JsonValue = JsonPrimitive | JsonValue[] | {[key: string]: JsonValue};

export type AssistantStreamState = 'streaming' | 'complete' | 'preamble' | 'interrupted' | 'truncated';

export type Message = {
  id: string;
  role: 'user' | 'assistant' | 'activity' | 'error';
  text: string;
  createdAt?: string;
  turnId?: string;
  sampleId?: string;
  round?: number;
  streamState?: AssistantStreamState;
  seq?: number;
  epoch?: string;
  revision?: number;
  liveOutput?: boolean;
  tool?: string;
  toolCallId?: string;
  status?: 'running' | 'done' | 'error';
  stage?: string;
  delivery?: 'sending' | 'queued' | 'running' | 'done' | 'cancelled' | 'error';
  deliveryError?: string;
};

export type AgentQueueMessage = {id: string; text: string};

export type StreamSnapshotSample = {
  sample_id: string;
  round?: number;
  seq: number;
  text: string;
  outcome?: 'final' | 'tool_preamble' | 'interrupted' | 'truncated';
  finish_reason?: string;
};

export type StreamToolSnapshot = {
  id?: string;
  round?: number;
  name?: string;
  requested_name?: string;
  status?: 'running' | 'completed' | 'failed';
  stage?: string;
  text?: string;
};

export type AgentEvent = {
  type?: string;
  turn_id?: string;
  sample_id?: string;
  round?: number;
  seq?: number;
  outcome?: StreamSnapshotSample['outcome'];
  finish_reason?: string;
  text?: string;
  final_sample_id?: string;
  epoch?: string;
  revision?: number;
  messageId?: string;
  delta?: string;
  message?: string;
  phase?: string;
  status?: string;
  stream?: string;
  line?: string;
  assistant?: string;
  activeSessionId?: string;
  busy?: boolean;
  active?: AgentQueueMessage | null;
  pending?: AgentQueueMessage[];
  cancelledIds?: string[];
  sessions?: SessionSummary[];
  historyRequired?: boolean;
  samples?: StreamSnapshotSample[];
  tools?: Array<AgentEvent | StreamToolSnapshot>;
  metrics?: Record<string, unknown>;
  tool?: {id?: string; name?: string; requested_name?: string};
  progress?: {stage?: string; message?: string; metadata?: Record<string, unknown>};
  tool_result?: {id?: string; name?: string; ok?: boolean; content?: string; metadata?: Record<string, unknown>};
  session?: {
    active_session_id?: string;
    session?: {
      session_id?: string;
      working_dir?: string;
      stage?: string;
      summary?: string;
    } | null;
  };
  prompt_trace?: {
    total_estimated_tokens?: number;
    totals?: {total_tokens?: number; total_estimated_tokens?: number};
  };
};

export type ChatState = {
  messages: Message[];
  busy: boolean;
  turnId: string;
  promptTokens: number;
  queueSupported: boolean;
  activeMessageId: string;
  epoch: string;
  revision: number;
  phase: string;
  retiredEpochs: string[];
  recoveringHistory?: boolean;
};

export type BrowserAgentMetrics = {
  firstVisibleTextMs?: number;
  firstReceiptToCommitMs?: number;
  completionMs?: number;
  deltaEvents?: number;
  snapshotRepairs?: number;
  visible: 0 | 1;
};

export type BrowserAgentMetricsPayload = {
  sessionId: string;
  messageId: string;
  turnId?: string;
  metrics: BrowserAgentMetrics;
};
