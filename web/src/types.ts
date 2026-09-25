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
  /** Rejected slots across the whole payload. */
  rejected: number;
  clipSeconds: number;
  clipCostUsd: number;
};

export type RenderAcceptance = {
  root: string;
  stages: AcceptanceStage[];
  totals: AcceptanceTotals;
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

/** What one shot's frames describe and show, with the characters the root can draw. */
export type ShotPlan = {
  slot: string;
  root: string;
  /** The one-line brief the shot came from. */
  brief: string;
  characters: {idx: number; name: string}[];
  firstFrame: PlanFrame;
  lastFrame: PlanFrame;
  motionDescription: string;
};

export type PlanFrame = {
  description: string;
  /** Indices of the characters this frame shows. */
  visible: number[];
  /** The prompt the frame was last drawn from, as it was sent. Empty until it is drawn. */
  prompt: string;
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
  /** Seconds one clip runs, which is what the runtime finding multiplies. */
  clipSeconds: number;
};

export type ShotPlanUpdateRequest = {
  sessionId: string;
  root: string;
  slot: string;
  ffDesc?: string;
  lfDesc?: string;
  ffVis?: number[];
  lfVis?: number[];
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

export type ConfigSection = {
  model_provider?: string;
  model: string;
  base_url: string;
  api_key: string;
  has_api_key: boolean;
};

export type AgentConfig = {
  sections: Record<'llm' | 'image' | 'video' | 'embedding' | 'reranker', ConfigSection>;
};

export type Artifact = {
  path: string;
  name: string;
  kind: 'image' | 'video' | 'document';
  size: number;
  updatedAt: string;
  url: string;
};

export type WorkspaceUpload = {
  name: string;
  path: string;
  size: number;
};

export type JsonPrimitive = string | number | boolean | null;

export type JsonValue = JsonPrimitive | JsonValue[] | {[key: string]: JsonValue};

export type Message = {
  id: string;
  role: 'user' | 'assistant' | 'activity' | 'error';
  text: string;
  createdAt?: string;
  tool?: string;
  status?: 'running' | 'done' | 'error';
  stage?: string;
};

export type AgentEvent = {
  type?: string;
  turn_id?: string;
  delta?: string;
  message?: string;
  phase?: string;
  status?: string;
  stream?: string;
  line?: string;
  assistant?: string;
  activeSessionId?: string;
  sessions?: SessionSummary[];
  tool?: {id?: string; name?: string; requested_name?: string};
  progress?: {stage?: string; message?: string; metadata?: Record<string, unknown>};
  tool_result?: {name?: string; ok?: boolean; content?: string; metadata?: Record<string, unknown>};
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
};
