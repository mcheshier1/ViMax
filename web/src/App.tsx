import {memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState} from 'react';
import {
  ArrowUp,
  Brain,
  Braces,
  CircleStop,
  Clock3,
  FilePenLine,
  FileText,
  Film,
  Folder,
  FolderPlus,
  Files,
  Image as ImageIcon,
  MessageSquare,
  Moon,
  Palette,
  Activity,
  Plus,
  RefreshCw,
  Save,
  Search,
  Settings,
  ListChecks,
  Terminal,
  Trash2,
  Sun,
  Wrench,
  X,
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type {Components} from 'react-markdown';
import {deleteSession, getAgentConfig, getHistory, getSessions, postAgentMetrics, readProject, saveAgentConfig, sendMessage, startAgent, stopAgent, subscribeToEvents, updateProject, uploadWorkspaceFile} from './api';
import {ArtifactsView, ScriptView} from './ArtifactViews';
import {applyAgentEvent, applyAgentQueueSnapshot, appendLocalUser, composeAgentPrompt, createChatState, humanize, mergeHistoryMessages, setUserDelivery} from './events';
import {describeArtifacts, describeInvalidation, describeStyleMismatch, diffProjectFields, toProjectFields, type InvalidationConfirmation, type ProjectFields} from './projectMetadata';
import {matchingSlashCommands, shouldShowSlashCommands, type SlashCommandMatch} from './slashCommands';
import {applyTheme, resolveTheme, THEME_STORAGE_KEY, type Theme} from './theme';
import type {AgentConfig, AgentEvent, BrowserAgentMetricsPayload, ChatState, ConfigSection, Message, ProjectMetadata, ProjectUpdateRequest, ProjectUpdateResponse, SessionSummary, WorkspaceUpload} from './types';
import {createDeltaBatcher, type DeltaBatcher} from './streaming';
import {IMAGE_PROVIDER_PRESETS, compatibleImageResolution, defaultImageResolution, imageProviderFromBaseUrl, imageProviderPreset, imageResolutionTiers} from './imagePresets';
import {isHeygenVideoModel, validHeygenVideoSettings, VIDEO_PROVIDER_PRESETS, videoModelSelection, videoProviderPreset} from './videoPresets';
import {credentialProvider} from '../provider-credentials.mjs';
import {validateShotDuration, videoDurationPolicy} from '../video-durations.mjs';
import {AssetCreationDialog, type AssetCreationRequest} from './AssetCreationDialog';
import {TimelineView} from './TimelineView';
import {useFilmSession, type FilmSelection} from './filmSession';
import './workbench-shell.css';

const CONTEXT_TARGET = 160_000;

type WorkspaceView = 'film' | 'script' | 'assets';
type UtilityView = 'project' | 'settings' | null;
type AssistantScope = 'film' | 'shot';
const EMPTY_CHAT = createChatState();
const EMPTY_UPLOADS: WorkspaceUpload[] = [];

type TimedEvent = {sessionId: string; event: AgentEvent};
type BrowserTurnTiming = {
  sessionId: string;
  messageId: string;
  turnId?: string;
  sentAt: number;
  firstReceiptAt?: number;
  firstVisibleAt?: number;
  firstReceiptToCommitMs?: number;
  deltaEvents: number;
  snapshotRepairs: number;
  lastSeqBySample: Map<string, number>;
  lastSnapshotRevision?: number;
  lastSnapshotEpoch?: string;
};

const MARKDOWN_PLUGINS = [remarkGfm];
const MARKDOWN_COMPONENTS: Components = {
  a: ({node: _node, ...props}) => <a {...props} target="_blank" rel="noreferrer" />,
};

export default function App() {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [selectedSessionId, setSelectedSessionId] = useState('');
  const [workspaceView, setWorkspaceView] = useState<WorkspaceView>('film');
  const [utility, setUtility] = useState<UtilityView>(null);
  const [drawer, setDrawer] = useState<'assistant' | 'activity' | null>(null);
  const [selection, setSelection] = useState<FilmSelection | null>(null);
  const [assistantScope, setAssistantScope] = useState<AssistantScope>('film');
  const [chats, setChats] = useState<Record<string, ChatState>>({});
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [uploads, setUploads] = useState<Record<string, WorkspaceUpload[]>>({});
  const [uploadingSession, setUploadingSession] = useState('');
  const [sendingSession, setSendingSession] = useState('');
  const [queueBridgeReady, setQueueBridgeReady] = useState(false);
  const [historyLoading, setHistoryLoading] = useState('');
  const [documentVisible, setDocumentVisible] = useState(document.visibilityState === 'visible');
  const [theme, setTheme] = useState<Theme>(() => resolveTheme(document.documentElement.dataset.theme, false));
  const [loadError, setLoadError] = useState('');
  const [newProjectOpen, setNewProjectOpen] = useState(false);
  const [newProjectName, setNewProjectName] = useState('');
  const [newProjectStyle, setNewProjectStyle] = useState('');
  const [newProjectRequirement, setNewProjectRequirement] = useState('');
  const [newProjectError, setNewProjectError] = useState('');
  const [creatingProject, setCreatingProject] = useState(false);
  const [pendingDelete, setPendingDelete] = useState<SessionSummary>();
  const [deleting, setDeleting] = useState(false);
  const [assetCreationRequest, setAssetCreationRequest] = useState<{sessionId: string; request: AssetCreationRequest}>();
  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const selectedSessionIdRef = useRef(selectedSessionId);
  selectedSessionIdRef.current = selectedSessionId;
  const agentSessionRef = useRef('');
  const agentRunningRef = useRef(false);
  const sendingRef = useRef(false);
  const dirtyFilmSessions = useRef(new Set<string>());
  const loadedHistory = useRef(new Set<string>());
  const historyRequests = useRef(new Map<string, Promise<void>>());
  const queueSnapshots = useRef(new Map<string, AgentEvent>());
  const snapshotRecoveryRequests = useRef(new Map<string, string>());
  const busySessions = useRef(new Set<string>());
  const bridgeLifecycleObserved = useRef(false);
  const scrollNearBottom = useRef(true);
  const turnTimingById = useRef(new Map<string, BrowserTurnTiming>());
  const turnTimingByMessage = useRef(new Map<string, BrowserTurnTiming>());
  const reportedTurns = useRef(new Set<string>());
  const film = useFilmSession(selectedSessionId);
  const filmRef = useRef(film);
  filmRef.current = film;
  const artifacts = film.data?.artifacts || [];
  const chat = chats[selectedSessionId] || EMPTY_CHAT;
  const draft = drafts[selectedSessionId] || '';
  const workspaceUploads = uploads[selectedSessionId] || EMPTY_UPLOADS;
  const uploadingFiles = uploadingSession === selectedSessionId && Boolean(selectedSessionId);
  const sending = Boolean(sendingSession);
  const busy = chat.busy;
  const projectCreationBlocked = !queueBridgeReady || sending || Object.values(chats).some((state) => state.busy);
  const queuedCount = chat.messages.filter((message) => message.role === 'user' && message.delivery === 'queued').length;
  const selectedSession = sessions.find((session) => session.sessionId === selectedSessionId);
  const slashMatches = useMemo(() => matchingSlashCommands(draft), [draft]);
  const showSlashCommands = shouldShowSlashCommands(draft, busy, chat.queueSupported);
  const contextPercent = Math.min(100, Math.round((chat.promptTokens / CONTEXT_TARGET) * 100));

  const updateChat = useCallback((sessionId: string, update: (current: ChatState) => ChatState) => {
    setChats((current) => ({...current, [sessionId]: update(current[sessionId] || createChatState())}));
  }, []);

  const commitDeltaBatch = useCallback((batch: TimedEvent[]) => {
    const grouped = new Map<string, AgentEvent[]>();
    for (const item of batch) {
      const events = grouped.get(item.sessionId) || [];
      events.push(item.event);
      grouped.set(item.sessionId, events);
    }
    for (const [sessionId, events] of grouped) {
      updateChat(sessionId, (current) => events.reduce((state, event) => applyAgentEvent(state, event), current));
    }
  }, [updateChat]);
  const deltaBatcher = useRef<DeltaBatcher<TimedEvent> | null>(null);
  if (!deltaBatcher.current) deltaBatcher.current = createDeltaBatcher(commitDeltaBatch);

  function setDraft(value: string) {
    setDrafts((current) => ({...current, [selectedSessionId]: value}));
  }

  function confirmLeaveFilm() {
    return !dirtyFilmSessions.current.has(selectedSessionIdRef.current) || window.confirm('This project has unsaved shot edits. Switch projects? Your local drafts will be kept, not discarded.');
  }

  function changeDestination(view: WorkspaceView) {
    setWorkspaceView(view);
    try {
      window.localStorage.setItem(`vimax-view:${selectedSessionId}`, view);
    } catch {
      // Navigation remains available without browser storage.
    }
  }

  function changeProject(sessionId: string) {
    if (sessionId !== selectedSessionId && confirmLeaveFilm()) selectSession(sessionId);
  }

  function selectSession(sessionId: string) {
    deltaBatcher.current?.flush();
    scrollNearBottom.current = true;
    selectedSessionIdRef.current = sessionId;
    setSelectedSessionId(sessionId);
    let destination: WorkspaceView = 'film';
    try {
      const saved = window.localStorage.getItem(`vimax-view:${sessionId}`);
      if (saved === 'script' || saved === 'assets') destination = saved;
      window.localStorage.setItem('vimax-project', sessionId);
    } catch {
      // A new or storage-restricted project opens directly in Film.
    }
    setWorkspaceView(destination);
    setAssetCreationRequest(undefined);
    setSelection(null);
    setAssistantScope('film');
    setLoadError('');
    if (fileInputRef.current) fileInputRef.current.value = '';
  }

  const refreshSessions = useCallback(async () => {
    const state = await getSessions();
    setSessions(state.sessions);
    return state;
  }, []);

  const ensureHistory = useCallback(function ensureHistoryRequest(sessionId: string, force = false): Promise<void> {
    if (!sessionId || !force && loadedHistory.current.has(sessionId)) return Promise.resolve();
    const pending = historyRequests.current.get(sessionId);
    if (pending) return force ? pending.then(() => ensureHistoryRequest(sessionId, true)) : pending;
    if (force) loadedHistory.current.delete(sessionId);
    setHistoryLoading(sessionId);
    const request = getHistory(sessionId).then((history) => {
      updateChat(sessionId, (current) => {
        const sameEpoch = Boolean(history.epoch && current.epoch && history.epoch === current.epoch);
        const restored = {
          ...current,
          messages: mergeHistoryMessages(current.messages, history.messages, history.revision, sameEpoch),
          recoveringHistory: Boolean(current.recoveringHistory && current.busy),
        };
        const snapshot = queueSnapshots.current.get(sessionId);
        return snapshot ? applyAgentQueueSnapshot(restored, snapshot) : restored;
      });
      loadedHistory.current.add(sessionId);
    }).finally(() => {
      historyRequests.current.delete(sessionId);
      setHistoryLoading((current) => current === sessionId ? '' : current);
    });
    historyRequests.current.set(sessionId, request);
    return request;
  }, [updateChat]);

  const displayVisible = documentVisible && drawer === 'assistant';
  const displayVisibleRef = useRef(displayVisible);
  displayVisibleRef.current = displayVisible;
  const reportTurnTiming = useCallback((timing: BrowserTurnTiming, completedAt?: number) => {
    if (!timing.turnId || reportedTurns.current.has(timing.turnId)) return;
    const turnId = timing.turnId;
    reportedTurns.current.add(turnId);
    const visible = timing.firstVisibleAt !== undefined;
    const metrics: BrowserAgentMetricsPayload['metrics'] = {
      visible: visible ? 1 : 0,
      deltaEvents: timing.deltaEvents,
      snapshotRepairs: timing.snapshotRepairs,
      ...(timing.firstVisibleAt !== undefined ? {
        firstVisibleTextMs: timing.firstVisibleAt - timing.sentAt,
        firstReceiptToCommitMs: timing.firstReceiptToCommitMs,
      } : {}),
      ...(completedAt !== undefined ? {completionMs: completedAt - timing.sentAt} : {}),
    };
    void postAgentMetrics({sessionId: timing.sessionId, messageId: timing.messageId, turnId: timing.turnId, metrics}).catch(() => {});
    turnTimingById.current.delete(turnId);
    turnTimingByMessage.current.delete(timing.messageId);
    if (reportedTurns.current.size > 512) {
      const oldest = reportedTurns.current.values().next().value;
      if (oldest) reportedTurns.current.delete(oldest);
    }
  }, []);
  const trackTurnEvent = useCallback((sessionId: string, event: AgentEvent) => {
    let timing = event.messageId ? turnTimingByMessage.current.get(event.messageId) : undefined;
    if (!timing && event.turn_id) timing = turnTimingById.current.get(event.turn_id);
    if (timing && event.turn_id && timing.turnId !== event.turn_id) {
      timing.turnId = event.turn_id;
      turnTimingById.current.set(event.turn_id, timing);
    }
    if (timing && timing.sessionId === sessionId) {
      if (event.type === 'token') {
        const sampleId = event.sample_id || 'legacy';
        const lastSeq = timing.lastSeqBySample.get(sampleId);
        const duplicate = typeof event.seq === 'number' && lastSeq !== undefined && event.seq <= lastSeq;
        if (!duplicate) {
          if (typeof event.seq === 'number') timing.lastSeqBySample.set(sampleId, event.seq);
          timing.deltaEvents += 1;
          if (event.delta?.trim() && timing.firstReceiptAt === undefined) timing.firstReceiptAt = performance.now();
        }
      } else if (event.type === 'stream_snapshot' && ((event.samples?.length || 0) > 0 || event.historyRequired)) {
        if (timing.lastSnapshotEpoch !== event.epoch || timing.lastSnapshotRevision !== event.revision) {
          timing.snapshotRepairs += 1;
          timing.lastSnapshotEpoch = event.epoch;
          timing.lastSnapshotRevision = event.revision;
        }
      }
      if (event.type === 'done') {
        const finishedTiming = timing;
        const completedAt = performance.now();
        window.setTimeout(() => reportTurnTiming(finishedTiming, completedAt), 100);
      }
    }
    if (timing && timing.sessionId === sessionId && event.type === 'stream_snapshot' && ['completed', 'failed', 'interrupted'].includes(event.status || '')) {
        const completedTiming = timing;
        const completedAt = event.status === 'completed' ? performance.now() : undefined;
        window.setTimeout(() => reportTurnTiming(completedTiming, completedAt), 100);
      }
    if (event.type === 'bridge_status' && (event.status === 'stopped' || event.status === 'error')) {
      for (const active of turnTimingById.current.values()) {
        if (active.sessionId === sessionId) window.setTimeout(() => reportTurnTiming(active), 250);
      }
    }
  }, [reportTurnTiming]);
  const onAssistantCommit = useCallback((message: Message, sessionId: string) => {
    if (message.role !== 'assistant' || !message.liveOutput || !message.text.trim() || sessionId !== selectedSessionIdRef.current || !displayVisibleRef.current) return;
    const timing = message.turnId ? turnTimingById.current.get(message.turnId) : undefined;
    if (!timing || timing.sessionId !== sessionId || timing.firstReceiptAt === undefined || timing.firstVisibleAt !== undefined) return;
    const committedAt = performance.now();
    timing.firstVisibleAt = committedAt;
    timing.firstReceiptToCommitMs = committedAt - timing.firstReceiptAt;
  }, []);

  useEffect(() => {
    applyTheme(theme);
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, theme);
    } catch {
      // Theme still applies when persistence is unavailable.
    }
  }, [theme]);

  useEffect(() => subscribeToEvents((event) => {
    if (event.type === 'sessions_changed') {
      deltaBatcher.current?.flush();
      setSessions(event.sessions || []);
      return;
    }
    if (event.type === 'agent_queue') {
      deltaBatcher.current?.flush();
      setQueueBridgeReady(true);
      const sessionId = event.activeSessionId || '';
      if (!sessionId) return;
      bridgeLifecycleObserved.current = true;
      queueSnapshots.current.set(sessionId, event);
      if (event.busy) busySessions.current.add(sessionId);
      else busySessions.current.delete(sessionId);
      if (event.busy) agentRunningRef.current = true;
      agentSessionRef.current = sessionId;
      updateChat(sessionId, (current) => applyAgentQueueSnapshot(current, event));
      if (selectedSessionIdRef.current === sessionId) filmRef.current.wake();
      return;
    }
    if (event.type === 'session') {
      const sessionId = event.activeSessionId || event.session?.active_session_id || event.session?.session?.session_id || '';
      if (sessionId) agentSessionRef.current = sessionId;
      void refreshSessions().catch(() => {});
    }
    const explicitSessionId = event.activeSessionId || event.session?.active_session_id || event.session?.session?.session_id || '';
    if (event.type === 'bridge_status') {
      bridgeLifecycleObserved.current = true;
      if (explicitSessionId) agentSessionRef.current = explicitSessionId;
      if (event.status === 'ready' || event.status === 'starting') agentRunningRef.current = true;
      if (event.status === 'idle' || event.status === 'stopped' || event.status === 'error') agentRunningRef.current = false;
    }
    const sessionId = explicitSessionId || agentSessionRef.current;
    if (!sessionId) return;
    trackTurnEvent(sessionId, event);
    if (event.type === 'turn') busySessions.current.add(sessionId);
    if (event.type === 'done') {
      const snapshot = queueSnapshots.current.get(sessionId);
      if (snapshot?.busy || snapshot?.pending?.length) busySessions.current.add(sessionId);
      else busySessions.current.delete(sessionId);
    }
    if (event.type === 'bridge_status' && (event.status === 'stopped' || event.status === 'error')) busySessions.current.delete(sessionId);
    if (event.type === 'token') {
      const key = `${sessionId}:${event.sample_id || event.turn_id || 'legacy'}`;
      deltaBatcher.current?.push({sessionId, event}, key);
      return;
    }
    deltaBatcher.current?.flush();
    updateChat(sessionId, (current) => applyAgentEvent(current, event));
    const terminalSnapshot = event.type === 'stream_snapshot' && ['completed', 'failed', 'interrupted'].includes(event.status || '');
    if (event.type === 'done' || event.type === 'stream_snapshot' && (event.historyRequired || terminalSnapshot)) {
      const recoveryVersion = `${event.epoch || ''}:${event.turn_id || event.messageId || ''}:${event.status || 'history-required'}`;
      if (snapshotRecoveryRequests.current.get(sessionId) !== recoveryVersion) {
        snapshotRecoveryRequests.current.set(sessionId, recoveryVersion);
        void ensureHistory(sessionId, true).catch((error) => {
          if (selectedSessionIdRef.current === sessionId) setLoadError(error instanceof Error ? error.message : String(error));
        });
      }
    }
    if (selectedSessionIdRef.current === sessionId && ['tool_start', 'tool_progress', 'tool_result', 'done', 'session'].includes(event.type || '')) filmRef.current.wake();
  }, (connected) => {
    if (connected) {
      const sessionsToReload = new Set([...loadedHistory.current, selectedSessionIdRef.current].filter(Boolean));
      for (const sessionId of sessionsToReload) {
        void ensureHistory(sessionId, true).catch((error) => {
          if (selectedSessionIdRef.current === sessionId) setLoadError(error instanceof Error ? error.message : String(error));
        });
      }
      return;
    }
    deltaBatcher.current?.flush();
    setQueueBridgeReady(false);
    queueSnapshots.current.clear();
    setChats((current) => {
      let next: Record<string, ChatState> | undefined;
      for (const [sessionId, state] of Object.entries(current)) {
        if (!state.queueSupported) continue;
        next ||= {...current};
        next[sessionId] = {...state, queueSupported: false};
      }
      return next || current;
    });
  }), [ensureHistory, refreshSessions, trackTurnEvent, updateChat]);

  useEffect(() => {
    let cancelled = false;
    void refreshSessions().then((state) => {
      if (cancelled || selectedSessionIdRef.current) return;
      let previous = '';
      try {
        previous = window.localStorage.getItem('vimax-project') || '';
      } catch {
        // The server's active project remains the fallback.
      }
      const sessionId = state.sessions.some((session) => session.sessionId === previous)
        ? previous : state.activeSessionId || state.sessions[0]?.sessionId || '';
      if (sessionId) selectSession(sessionId);
    }).catch((error) => !cancelled && setLoadError(error instanceof Error ? error.message : String(error)));
    return () => { cancelled = true; };
  }, [refreshSessions]);

  useEffect(() => {
    if (!film.progress || bridgeLifecycleObserved.current) return;
    agentSessionRef.current = film.progress.activeSessionId;
    agentRunningRef.current = film.progress.agentRunning;
  }, [film.progress]);

  useEffect(() => {
    if (!drawer || !selectedSessionId) return;
    const sessionId = selectedSessionId;
    void ensureHistory(sessionId).catch((error) => {
      if (selectedSessionIdRef.current === sessionId) setLoadError(error instanceof Error ? error.message : String(error));
    });
  }, [drawer, selectedSessionId, ensureHistory]);

  useLayoutEffect(() => {
    const element = scrollRef.current;
    if (element && scrollNearBottom.current) element.scrollTo({top: element.scrollHeight, behavior: 'auto'});
  }, [chat.messages, chat.busy, drawer]);

  useEffect(() => {
    const onVisibilityChange = () => {
      deltaBatcher.current?.flush();
      setDocumentVisible(document.visibilityState === 'visible');
    };
    const flushPending = () => deltaBatcher.current?.flush();
    document.addEventListener('visibilitychange', onVisibilityChange);
    window.addEventListener('pagehide', flushPending);
    return () => {
      document.removeEventListener('visibilitychange', onVisibilityChange);
      window.removeEventListener('pagehide', flushPending);
      flushPending();
    };
  }, []);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = '0px';
    textarea.style.height = `${Math.min(168, Math.max(28, textarea.scrollHeight))}px`;
  }, [draft, drawer]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      if (utility) setUtility(null);
      else setDrawer(null);
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [utility]);

  useEffect(() => {
    const onDraftChange = (event: Event) => {
      const detail = (event as CustomEvent<{sessionId: string; dirty: boolean}>).detail;
      if (!detail?.sessionId) return;
      if (detail.dirty) dirtyFilmSessions.current.add(detail.sessionId);
      else dirtyFilmSessions.current.delete(detail.sessionId);
    };
    const onOpenAssistant = () => {
      setDrawer('assistant');
      setAssistantScope('film');
    };
    const onOpenSettings = () => setUtility('settings');
    window.addEventListener('film-drafts-change', onDraftChange);
    window.addEventListener('film-open-assistant', onOpenAssistant);
    window.addEventListener('film-open-settings', onOpenSettings);
    return () => {
      window.removeEventListener('film-drafts-change', onDraftChange);
      window.removeEventListener('film-open-assistant', onOpenAssistant);
      window.removeEventListener('film-open-settings', onOpenSettings);
    };
  }, []);

  function openNewProjectDialog() {
    if (!confirmLeaveFilm()) return;
    setNewProjectName('');
    setNewProjectStyle('');
    setNewProjectRequirement('');
    setNewProjectError('');
    setNewProjectOpen(true);
  }

  async function waitForNewSession(previousSessionIds: Set<string>, initiatingSessionId: string) {
    const deadline = Date.now() + 10_000;
    while (Date.now() < deadline) {
      if (selectedSessionIdRef.current !== initiatingSessionId) throw new Error('The selected project changed before the new project was ready');
      const state = await refreshSessions();
      if (state.activeSessionId && !previousSessionIds.has(state.activeSessionId)) return state.activeSessionId;
      await new Promise<void>((resolve) => window.setTimeout(resolve, 100));
    }
    throw new Error('The new project did not finish creating');
  }

  async function newProject() {
    const projectName = newProjectName.trim();
    if (!projectName || creatingProject || projectCreationBlocked) return;
    const initiatingSessionId = selectedSessionId;
    const previousSessionIds = new Set(sessions.map((session) => session.sessionId));
    setCreatingProject(true);
    setNewProjectError('');
    try {
      await startAgent({newSession: true, projectName, style: newProjectStyle.trim(), userRequirement: newProjectRequirement.trim()});
      const sessionId = await waitForNewSession(previousSessionIds, initiatingSessionId);
      agentSessionRef.current = sessionId;
      agentRunningRef.current = true;
      selectSession(sessionId);
      setWorkspaceView('film');
      setDrawer('assistant');
      setNewProjectOpen(false);
    } catch (error) {
      setNewProjectError(error instanceof Error ? error.message : String(error));
    } finally {
      setCreatingProject(false);
    }
  }

  /** Timeline instructions already name their scope. Only the composer adds its chosen context. */
  async function askAgent(
    text: string,
    attachments: string[] = [],
    restartAgent = false,
    scope?: AssistantScope,
    options: {messageId?: string; sessionId?: string; chosenShot?: FilmSelection | null} = {},
  ) {
    if (!text.trim()) return;
    if (!queueBridgeReady) throw new Error('Wait for the queue-enabled bridge to connect. An older web server needs an idle restart.');
    if (sendingRef.current) throw new Error('Wait for the current message to finish sending');
    const sessionId = options.sessionId || selectedSessionId;
    if (!sessionId) throw new Error('Create or select a project before sending a message');
    if (selectedSessionIdRef.current !== sessionId) throw new Error('The selected project changed before sending');
    const chosenShot = scope === 'shot' ? options.chosenShot ?? selection : null;
    if (scope === 'shot' && !chosenShot) throw new Error('Select a shot in Film or choose Whole film');
    const messageId = options.messageId || crypto.randomUUID();
    const root = chosenShot?.root || film.data?.acceptance.root || '';
    sendingRef.current = true;
    setSendingSession(sessionId);
    try {
      await ensureHistory(sessionId);
      if (selectedSessionIdRef.current !== sessionId) throw new Error('The selected project changed before sending');
      const queuedSnapshot = queueSnapshots.current.get(sessionId);
      const currentChat = chats[sessionId] || EMPTY_CHAT;
      const queueSupported = queueSnapshots.current.has(sessionId) || currentChat.queueSupported;
      const serverBusy = currentChat.busy || busySessions.current.has(sessionId) || queuedSnapshot?.busy === true;
      const activeSessionId = agentSessionRef.current;
      const activeQueue = activeSessionId ? queueSnapshots.current.get(activeSessionId) : undefined;
      const activeChat = activeSessionId ? chats[activeSessionId] : undefined;
      if (activeSessionId && activeSessionId !== sessionId && (busySessions.current.has(activeSessionId) || activeQueue?.busy || activeChat?.busy)) {
        throw new Error('Another project is still running. Wait for it to finish before switching the agent to this project.');
      }
      if (serverBusy && !queueSupported) throw new Error('This running bridge does not support safe message queueing yet');
      if (!serverBusy && (!agentRunningRef.current || agentSessionRef.current !== sessionId || restartAgent)) {
        await startAgent({sessionId});
        agentSessionRef.current = sessionId;
        agentRunningRef.current = true;
      }
      if (selectedSessionIdRef.current !== sessionId) throw new Error('The selected project changed before sending');
      const context = scope && !text.trimStart().startsWith('/')
        ? ` Film context: ${JSON.stringify({session: sessionId, root, scope: scope === 'shot' ? 'selected-shot' : 'whole-film', ...(chosenShot ? {slot: chosenShot.slot, label: chosenShot.label} : {})})}. ${chosenShot ? 'Limit changes to this selected shot unless I explicitly request broader changes.' : 'This request concerns the whole film.'}`
        : '';
      updateChat(sessionId, (current) => appendLocalUser(current, text, messageId));
      const timing: BrowserTurnTiming = {
        sessionId,
        messageId,
        sentAt: performance.now(),
        deltaEvents: 0,
        snapshotRepairs: 0,
        lastSeqBySample: new Map(),
      };
      turnTimingByMessage.current.set(messageId, timing);
      const response = await sendMessage({
        text: composeAgentPrompt(text + context, attachments),
        sessionId,
        messageId,
        displayText: text,
      });
      updateChat(sessionId, (current) => {
        const message = current.messages.find((item) => item.id === messageId);
        if (message?.delivery !== 'sending') return current;
        return setUserDelivery(current, messageId, response.queued ? 'queued' : 'running');
      });
      filmRef.current.wake();
    } catch (error) {
      const failedTiming = turnTimingByMessage.current.get(messageId);
      if (failedTiming && !failedTiming.turnId) turnTimingByMessage.current.delete(messageId);
      updateChat(sessionId, (current) => {
        const message = current.messages.find((item) => item.id === messageId);
        if (message?.delivery !== 'sending') return current;
        return setUserDelivery(current, messageId, 'error', error instanceof Error ? error.message : String(error));
      });
      throw error;
    } finally {
      sendingRef.current = false;
      setSendingSession('');
    }
  }

  async function submit() {
    const text = draft.trim();
    const sessionId = selectedSessionId;
    const submittedDraft = drafts[sessionId] || '';
    const submittedUploads = workspaceUploads;
    const attachmentPaths = submittedUploads.map((file) => file.path);
    const scope = assistantScope;
    const chosenShot = scope === 'shot' ? selection : null;
    if (!queueBridgeReady || !text || !sessionId || sending || uploadingFiles || scope === 'shot' && !chosenShot) return;
    if (busy && !chat.queueSupported) return;
    const messageId = crypto.randomUUID();
    setLoadError('');
    try {
      await askAgent(text, attachmentPaths, false, scope, {messageId, sessionId, chosenShot});
      setDrafts((current) => current[sessionId] === submittedDraft ? {...current, [sessionId]: ''} : current);
      setUploads((current) => ({
        ...current,
        [sessionId]: (current[sessionId] || []).filter((file) => !submittedUploads.includes(file)),
      }));
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : String(error));
    }
  }

  async function uploadFiles(files: FileList | null) {
    const sessionId = selectedSessionId;
    if (!files?.length || !sessionId || uploadingSession) return;
    setUploadingSession(sessionId);
    setLoadError('');
    let uploadedAny = false;
    try {
      for (const file of Array.from(files)) {
        const result = await uploadWorkspaceFile(sessionId, file);
        uploadedAny = true;
        setUploads((current) => ({...current, [sessionId]: [...(current[sessionId] || []).filter((item) => item.path !== result.file.path), result.file]}));
      }
    } catch (error) {
      if (selectedSessionIdRef.current === sessionId) setLoadError(error instanceof Error ? error.message : String(error));
    } finally {
      if (fileInputRef.current) fileInputRef.current.value = '';
      if (uploadedAny && selectedSessionIdRef.current === sessionId) {
        void filmRef.current.refresh().catch((error) => {
          if (selectedSessionIdRef.current === sessionId) setLoadError(error instanceof Error ? error.message : String(error));
        });
      }
      setUploadingSession('');
      textareaRef.current?.focus();
    }
  }

  async function stop() {
    try {
      await stopAgent();
      agentRunningRef.current = false;
      busySessions.current.delete(agentSessionRef.current);
      updateChat(agentSessionRef.current, (current) => applyAgentEvent(current, {type: 'bridge_status', status: 'stopped'}));
      filmRef.current.wake();
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : String(error));
    }
  }

  async function confirmDelete() {
    if (!pendingDelete || deleting) return;
    const sessionId = pendingDelete.sessionId;
    setDeleting(true);
    setLoadError('');
    try {
      const state = await deleteSession(sessionId);
      setSessions(state.sessions);
      setPendingDelete(undefined);
      loadedHistory.current.delete(sessionId);
      queueSnapshots.current.delete(sessionId);
      snapshotRecoveryRequests.current.delete(sessionId);
      busySessions.current.delete(sessionId);
      if (selectedSessionIdRef.current === sessionId) {
        selectSession(state.activeSessionId || state.sessions[0]?.sessionId || '');
        setUtility(null);
      }
      if (agentSessionRef.current === sessionId) agentRunningRef.current = false;
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : String(error));
    } finally {
      setDeleting(false);
    }
  }

  function openPlanningAssistant() {
    setDrawer('assistant');
    setAssistantScope('film');
    if (!draft.trim()) setDraft('Help me plan this film. Develop the script and storyboard from my idea; do not generate paid images or video yet.');
    textareaRef.current?.focus();
  }

  const assistantMessages = chat.messages.filter((message) => message.role !== 'activity');
  const activityMessages = chat.messages.filter((message) => message.role === 'activity' || message.role === 'error');

  return (
    <div className="workbench-shell">
      <header className="workbench-topbar">
        <strong className="workbench-brand">ViMax</strong>
        <div className="workbench-project-switcher">
          <select value={selectedSessionId} onChange={(event) => changeProject(event.target.value)} aria-label="Project" disabled={creatingProject}>
            {!selectedSessionId && <option value="">Select a project</option>}
            {sessions.map((session) => <option key={session.sessionId} value={session.sessionId}>{sessionTitle(session)}</option>)}
          </select>
          <button className="icon-button" onClick={openNewProjectDialog} aria-label="New project" title="New project"><Plus size={17} /></button>
        </div>
        <nav className="workbench-destinations" aria-label="Project views">
          <button className={workspaceView === 'film' ? 'is-active' : ''} aria-current={workspaceView === 'film' ? 'page' : undefined} onClick={() => changeDestination('film')}><Film size={16} />Film</button>
          <button className={workspaceView === 'script' ? 'is-active' : ''} aria-current={workspaceView === 'script' ? 'page' : undefined} onClick={() => changeDestination('script')}><FileText size={16} />Script</button>
          <button className={workspaceView === 'assets' ? 'is-active' : ''} aria-current={workspaceView === 'assets' ? 'page' : undefined} onClick={() => changeDestination('assets')}><Files size={16} />Assets</button>
        </nav>
        <div className="workbench-utilities">
          <button className="icon-button" onClick={() => setUtility('project')} disabled={!selectedSessionId} aria-label="Project details" title="Project details"><Palette size={17} /></button>
          <button className="icon-button" onClick={() => setUtility('settings')} aria-label="Settings" title="Settings"><Settings size={17} /></button>
          <ThemeToggle theme={theme} onToggle={() => setTheme((current) => current === 'dark' ? 'light' : 'dark')} />
          <button className={`workbench-utility-button ${drawer === 'activity' ? 'is-active' : ''}`} onClick={() => setDrawer((current) => current === 'activity' ? null : 'activity')} aria-expanded={drawer === 'activity'}><Activity size={16} /><span>Activity</span>{chat.busy && <i className="workbench-busy-dot" />}</button>
          <button className={`workbench-assistant-button ${drawer === 'assistant' ? 'is-active' : ''}`} onClick={() => setDrawer((current) => current === 'assistant' ? null : 'assistant')} aria-expanded={drawer === 'assistant'}><MessageSquare size={16} /><span>Assistant</span></button>
        </div>
      </header>
      {loadError && <div className="workbench-error inline-error" role="alert"><span>{loadError}</span><button onClick={() => setLoadError('')} aria-label="Dismiss error"><X size={15} /></button></div>}
      <div className={`workbench-body ${drawer ? 'has-drawer' : ''}`}>
        <main className="workbench-main">
          <div className="workbench-destination film-destination" hidden={workspaceView !== 'film'}>
            {!selectedSessionId ? (
              <section className="workbench-welcome">
                <Film size={32} /><h1>Your film starts here</h1><p>Plan a story, shape each shot, and review the film in one place.</p>
                <button className="workbench-primary" onClick={openNewProjectDialog}><Plus size={16} />Create a project</button>
              </section>
            ) : (
              <>
                {!film.loading && film.data && film.data.plans.length === 0 && <div className="workbench-planning-prompt"><div><strong>Give your film a starting point</strong><span>Use the Assistant to develop a script and shot plan before generating media.</span></div><button onClick={openPlanningAssistant}><MessageSquare size={15} />Plan with Assistant</button></div>}

                <TimelineView key={selectedSessionId} session={selectedSession} artifacts={artifacts} film={film} active={workspaceView === 'film' && !utility} requestBusy={chat.busy || sending || !queueBridgeReady} onAskAgent={(text, restartAgent) => askAgent(text, [], restartAgent)} onSelectionChange={setSelection} onCreateAsset={(request) => setAssetCreationRequest({sessionId: selectedSessionId, request})} />
              </>
            )}
          </div>
          {workspaceView === 'script' && <ScriptView key={selectedSessionId} session={selectedSession} artifacts={artifacts} onPlan={openPlanningAssistant} />}
          {workspaceView === 'assets' && <ArtifactsView key={selectedSessionId} session={selectedSession} artifacts={artifacts} characters={film.data?.acceptance.characters || []} root={film.data?.acceptance.root || ''} revision={film.progress?.revision || ''} portraitImageModel={film.data?.acceptance.portraitImageModel || ''} progress={film.progress} disabled={chat.busy || sending || !queueBridgeReady} onAskAgent={(text) => {setDrawer('activity'); return askAgent(text, [], false, 'film', {sessionId: selectedSessionId});}} onCreateAsset={(request) => setAssetCreationRequest({sessionId: selectedSessionId, request})} onRefresh={async (sessionId) => {
            if (selectedSessionIdRef.current !== sessionId) throw new Error('The selected project changed before refreshing assets.');
            try { await filmRef.current.refresh(); } catch (error) {
              if (selectedSessionIdRef.current === sessionId) setLoadError(error instanceof Error ? error.message : String(error));
              throw error;
            }
          }} />}
        </main>
        {drawer && <aside className={`workbench-drawer ${drawer === 'assistant' ? 'assistant-drawer' : 'activity-drawer'}`} aria-label={drawer === 'assistant' ? 'Assistant' : 'Activity'}>
          <header className="workbench-drawer-heading"><div><strong>{drawer === 'assistant' ? 'Assistant' : 'Activity'}</strong><span>{selectedSession ? sessionTitle(selectedSession) : 'No project selected'}</span></div><button className="icon-button" onClick={() => setDrawer(null)} aria-label={`Close ${drawer}`}><X size={18} /></button></header>
          {drawer === 'assistant' ? <>
            <div className="assistant-scope" aria-label="Assistant scope">
              <span>Work on</span><div role="group" aria-label="Request scope">
                <button aria-pressed={assistantScope === 'film'} onClick={() => setAssistantScope('film')}>Whole film</button>
                <button aria-pressed={assistantScope === 'shot'} onClick={() => setAssistantScope('shot')} disabled={!selection} title={selection ? `${selection.root} · ${selection.slot}` : 'Select a shot in Film first'}>Selected shot</button>
              </div>
              <small>{assistantScope === 'shot' ? selection?.label || 'No shot selected — choose a shot or Whole film' : 'Story, structure, and film-wide changes'}</small>
            </div>
            <div className="conversation" ref={scrollRef} onScroll={(event) => {
              const element = event.currentTarget;
              scrollNearBottom.current = element.scrollHeight - element.scrollTop - element.clientHeight <= 120;
            }}>
              {historyLoading === selectedSessionId && selectedSessionId && <p className="assistant-loading" role="status">Loading conversation…</p>}
              {chat.recoveringHistory && <p className="assistant-loading" role="status">Live recovery buffer exceeded. Your partial response is preserved; the full response will load from history when this turn finishes.</p>}
              {!assistantMessages.length && <div className="assistant-intro"><MessageSquare size={23} /><h2>{selectedSessionId ? 'A creative partner, on request' : 'Create a project to begin'}</h2><p>{selectedSessionId ? 'Describe an idea, ask for a revision, or select a shot to keep the request focused. Nothing runs until you send.' : 'Keep your script, shots, and conversation together in a project.'}</p>{selectedSessionId ? <button onClick={openPlanningAssistant}>Draft a planning request</button> : <button onClick={openNewProjectDialog}>Create project</button>}</div>}
              <div className="message-stream">{assistantMessages.map((message) => <MessageRow key={message.id} message={message} sessionId={selectedSessionId} canMeasure={displayVisible} onAssistantCommit={onAssistantCommit} />)}{busy && <ThinkingRow messages={chat.messages} phase={chat.phase} />}</div>
            </div>
            <div className="composer-zone">
              {showSlashCommands && <SlashCommandMenu matches={slashMatches} contextPercent={contextPercent} onSelect={(command) => {setDraft(command); textareaRef.current?.focus();}} />}
              {!queueBridgeReady && <div className="assistant-queue-note is-unavailable" role="status">
                <strong>Waiting for the queue-enabled bridge</strong>
                <span>If this persists after reconnecting, restart the web server once the current request finishes. Your draft stays editable.</span>
              </div>}
              {busy && queueBridgeReady && <div className={`assistant-queue-note ${chat.queueSupported ? '' : 'is-unavailable'}`} role="status" aria-live="polite">
                {chat.queueSupported
                  ? <><strong>{queuedCount ? `${queuedCount} message${queuedCount === 1 ? '' : 's'} queued` : 'Agent working'}</strong><span>Queued messages are answered in order after the current request finishes.</span></>
                  : <><strong>Safe queueing is not available yet</strong><span>This bridge has not confirmed queue support. Keep editing and wait for its queue snapshot before sending.</span></>}
              </div>}
              {queuedCount > 0 && !busy && <div className="assistant-queue-note" role="status"><strong>{queuedCount} queued</strong><span>Answers wait until the current request finishes.</span></div>}
              <div className={`composer ${busy ? 'is-busy' : ''}`}>
                <textarea ref={textareaRef} value={draft} onChange={(event) => setDraft(event.target.value)} onKeyDown={(event) => {
                  if (event.key === 'Tab' && slashMatches[0]) {event.preventDefault(); setDraft(slashMatches[0].name); return;}
                  if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {event.preventDefault(); void submit();}
                }} placeholder={assistantScope === 'shot' ? 'What should change in this shot?' : 'Describe an idea or a film-wide change'} aria-label="Message ViMax" disabled={!selectedSessionId} rows={2} />
                {(workspaceUploads.length > 0 || uploadingFiles) && <div className="composer-attachments" aria-live="polite">
                  {workspaceUploads.map((file) => <span className="composer-attachment" key={file.path} title={file.path}><FileText size={13} /><span>{file.name}</span><button type="button" onClick={() => setUploads((current) => ({...current, [selectedSessionId]: (current[selectedSessionId] || []).filter((item) => item.path !== file.path)}))} aria-label={`Remove ${file.name} from this message`}><X size={12} /></button></span>)}
                  {uploadingFiles && <span className="composer-uploading">Uploading…</span>}
                </div>}
                <div className="composer-controls">
                  <input ref={fileInputRef} className="composer-file-input" type="file" multiple onChange={(event) => void uploadFiles(event.currentTarget.files)} tabIndex={-1} />
                  <button type="button" className="composer-add" onClick={() => fileInputRef.current?.click()} disabled={!selectedSessionId || Boolean(uploadingSession)} aria-label="Upload files to workspace" title="Upload reference files"><Plus size={19} /></button>
                  <span className="assistant-composer-hint">{assistantScope === 'shot' ? 'Selected shot' : 'Whole film'}</span><div className="composer-spacer" />
                  <button className={`send-button ${(busy || sending) ? 'queue' : ''}`} onClick={() => void submit()} disabled={!queueBridgeReady || !selectedSessionId || !draft.trim() || uploadingFiles || sending || assistantScope === 'shot' && !selection || busy && !chat.queueSupported} aria-label={sending ? 'Sending message' : busy ? 'Queue message' : 'Send message'} title={sending ? 'Sending message' : busy ? 'Queue message' : 'Send message'}>
                    <ArrowUp size={19} />{busy && !sending && <span>Queue message</span>}{sending && <span>Sending…</span>}
                  </button>
                  {busy && <button className="send-button stop" onClick={() => void stop()} aria-label="Stop generation" title="Stop generation"><CircleStop size={18} /></button>}
                </div>
              </div>
              <small className="assistant-draft-note">Drafts stay with their project.</small>
            </div>
          </> : <div className="workbench-activity-content">
            <div className="workbench-activity-summary"><span>{busy ? 'Working' : film.progress?.agentRunning && film.progress.activeSessionId === selectedSessionId ? 'Agent connected' : 'No active request'}</span>{(busy || film.progress?.agentRunning && film.progress.activeSessionId === selectedSessionId) && <button onClick={() => void stop()}><CircleStop size={14} />Stop</button>}</div>
            {historyLoading === selectedSessionId && selectedSessionId && <p role="status">Loading activity…</p>}
            {activityMessages.map((message) => <MessageRow key={message.id} message={message} />)}
            {!activityMessages.length && !film.progress?.trail && <p className="assistant-loading">Planning and generation activity will appear here.</p>}
            {film.progress?.trail && <details className="activity-render-trail"><summary>Render log</summary><pre>{film.progress.trail}</pre></details>}
          </div>}
        </aside>}
      </div>
      {utility && <div className="workbench-utility-backdrop" onMouseDown={(event) => event.target === event.currentTarget && setUtility(null)}>
        <section className="workbench-utility-panel" role="dialog" aria-modal="true" aria-label={utility === 'project' ? 'Project details' : 'Settings'}>
          <div className="workbench-utility-heading"><strong>{utility === 'project' ? 'Project details' : 'Settings'}</strong>{utility === 'project' && selectedSession && <button className="workbench-delete" onClick={() => setPendingDelete(selectedSession)}><Trash2 size={14} />Delete project</button>}<button className="icon-button" onClick={() => setUtility(null)} aria-label="Close utility"><X size={18} /></button></div>
          {utility === 'project' ? <ProjectView key={selectedSessionId} sessionId={selectedSessionId} onChanged={() => {
            void refreshSessions().catch((error) => setLoadError(String(error)));
            void film.refresh().catch((error) => setLoadError(String(error)));
            film.wake();
          }} /> : <SettingsView agentBusy={projectCreationBlocked} onSaved={film.refresh} />}
        </section>
      </div>}
      <DeleteProjectDialog session={pendingDelete} deleting={deleting} onCancel={() => !deleting && setPendingDelete(undefined)} onConfirm={() => void confirmDelete()} />
      {assetCreationRequest && assetCreationRequest.sessionId === selectedSessionId && selectedSession && <AssetCreationDialog key={assetCreationRequest.sessionId} sessionId={selectedSessionId} projectName={sessionTitle(selectedSession)} root={film.data?.acceptance.root || ''} artifacts={artifacts} characters={film.data?.acceptance.characters || []} request={assetCreationRequest.request} onClose={() => setAssetCreationRequest(undefined)} onRefresh={async (sessionId) => {
        if (selectedSessionIdRef.current !== sessionId) throw new Error('The selected project changed before refreshing assets.');
        await filmRef.current.refresh();
      }} />}
      <NewProjectDialog open={newProjectOpen} name={newProjectName} style={newProjectStyle} requirement={newProjectRequirement} error={newProjectError} creating={creatingProject} blocked={projectCreationBlocked}
        onNameChange={(value) => {setNewProjectName(value); setNewProjectError('');}} onStyleChange={setNewProjectStyle} onRequirementChange={setNewProjectRequirement}
        onCancel={() => {if (!creatingProject) {setNewProjectOpen(false); setNewProjectError('');}}} onConfirm={() => void newProject()} />
    </div>
  );
}

function ThemeToggle({theme, onToggle}: {theme: Theme; onToggle: () => void}) {
  const dark = theme === 'dark';
  return (
    <button
      type="button"
      className="icon-button theme-toggle"
      onClick={onToggle}
      aria-label={dark ? 'Use light mode' : 'Use dark mode'}
      aria-pressed={dark}
      title={dark ? 'Light mode' : 'Dark mode'}
    >
      {dark ? <Sun size={18} /> : <Moon size={18} />}
    </button>
  );
}

const MessageRow = memo(function MessageRow({message, sessionId = '', canMeasure = false, onAssistantCommit}: {
  message: Message;
  sessionId?: string;
  canMeasure?: boolean;
  onAssistantCommit?: (message: Message, sessionId: string) => void;
}) {
  const rowRef = useRef<HTMLElement>(null);
  useLayoutEffect(() => {
    if (!canMeasure || !onAssistantCommit || message.role !== 'assistant' || !message.liveOutput || !message.text.trim()) return;
    const frame = requestAnimationFrame(() => {
      const row = rowRef.current;
      const conversation = row?.closest('.conversation');
      if (!row || !conversation) return;
      const bounds = row.getBoundingClientRect();
      const viewport = conversation.getBoundingClientRect();
      if (bounds.bottom > viewport.top && bounds.top < viewport.bottom && bounds.right > viewport.left && bounds.left < viewport.right) {
        onAssistantCommit(message, sessionId);
      }
    });
    return () => cancelAnimationFrame(frame);
  }, [canMeasure, message, onAssistantCommit, sessionId]);
  if (message.role === 'activity') return <ActivityRow message={message} />;
  const deliveryLabel = message.delivery === 'sending' ? 'Sending…'
    : message.delivery === 'queued' ? 'Queued'
      : message.delivery === 'running' ? 'In progress'
        : message.delivery === 'done' ? 'Answered'
          : message.delivery === 'cancelled' ? 'Cancelled'
            : message.delivery === 'error' ? 'Failed to send'
              : '';
  const streamLabel = message.streamState === 'preamble' ? 'Tool preamble'
    : message.streamState === 'streaming' ? 'Responding'
      : message.streamState === 'interrupted' ? 'Interrupted · partial response'
        : message.streamState === 'truncated' ? 'Response cut short'
          : '';
  return (
    <article ref={rowRef} className={`message-row role-${message.role}${message.streamState ? ` stream-${message.streamState}` : ''}`}>
      <div className="message-body">
        <ReactMarkdown remarkPlugins={MARKDOWN_PLUGINS} components={MARKDOWN_COMPONENTS}>
          {message.text}
        </ReactMarkdown>
        {message.role === 'assistant' && streamLabel && <small className={`message-stream-state stream-state-${message.streamState}`}>{streamLabel}</small>}
        {message.role === 'user' && deliveryLabel && <small className={`message-delivery delivery-${message.delivery}`}>{deliveryLabel}</small>}
        {message.role === 'user' && message.deliveryError && <small className="message-delivery-error">{message.deliveryError}</small>}
      </div>
    </article>
  );
});

function ActivityRow({message}: {message: Message}) {
  const stage = message.stage ? humanize(message.stage) : '';
  const detail = stage.toLowerCase() === message.text.toLowerCase() ? message.text : [stage, message.text].filter(Boolean).join(' · ');
  const toolKind = activityToolKind(message.tool);
  return (
    <div className={`activity-row status-${message.status || 'done'}`}>
      <span className={`activity-indicator tool-${toolKind}`}><ActivityToolIcon tool={message.tool} /></span>
      <div>
        <strong>{humanize(message.tool || 'Workflow')}</strong>
        <span>{detail}</span>
      </div>
    </div>
  );
}

function ActivityToolIcon({tool}: {tool?: string}) {
  const name = (tool || '').toLowerCase();
  const props = {size: 13, strokeWidth: 1.8};
  if (name.includes('narrative_planning') || name.includes('novel_planning')) return <FilePenLine {...props} />;
  if (name.includes('render_video')) return <Film {...props} />;
  if (name === 'view_image' || name.includes('image')) return <ImageIcon {...props} />;
  if (name === 'read_json' || name === 'write_json') return <Braces {...props} />;
  if (name === 'read_file' || name === 'write_file') return <FileText {...props} />;
  if (name === 'list_files' || name === 'glob_files') return <Folder {...props} />;
  if (name === 'search_text') return <Search {...props} />;
  if (name.startsWith('memory_')) return <Brain {...props} />;
  if (name.startsWith('todo_')) return <ListChecks {...props} />;
  if (name === 'run_shell') return <Terminal {...props} />;
  if (name === 'sleep') return <Clock3 {...props} />;
  return <Wrench {...props} />;
}

function activityToolKind(tool?: string) {
  const name = (tool || '').toLowerCase();
  if (name.includes('narrative_planning') || name.includes('novel_planning')) return 'planning';
  if (name.includes('render_video')) return 'render';
  if (name === 'view_image' || name.includes('image')) return 'image';
  if (name.startsWith('memory_')) return 'memory';
  if (name.startsWith('todo_')) return 'todo';
  if (name === 'run_shell') return 'shell';
  if (name === 'sleep') return 'time';
  return 'file';
}

function ThinkingRow({messages, phase}: {messages: Message[]; phase: string}) {
  const running = [...messages].reverse().find((message) => message.role === 'activity' && message.status === 'running');
  const responding = messages.some((message) => message.role === 'assistant' && message.streamState === 'streaming' && message.text);
  const phaseLabel = running ? `${humanize(running.tool || 'ViMax')} · ${running.text}`
    : responding ? 'Responding'
      : phase === 'compact' ? 'Compacting context'
        : phase === 'preparing_context' ? 'Preparing context'
          : phase === 'sampling_assistant' ? 'Waiting for model'
            : phase === 'executing_tools' ? 'Running tool'
              : phase === 'responding' ? 'Responding'
                : 'Preparing context';
  return (
    <div className="thinking-row">
      <span className="thinking-mark"><i /><i /><i /></span>
      <span>{phaseLabel}</span>
    </div>
  );
}

function SlashCommandMenu({matches, contextPercent, onSelect}: {matches: SlashCommandMatch[]; contextPercent: number; onSelect: (command: string) => void}) {
  return (
    <div className="slash-command-menu" role="listbox" aria-label="Slash commands">
      {matches.length > 0 ? matches.map((command) => (
        <button key={command.name} role="option" aria-selected="false" onMouseDown={(event) => event.preventDefault()} onClick={() => onSelect(command.name)}>
          <code><span><b>{command.matchedPrefix}</b><span>{command.unmatchedSuffix}</span></span>{command.name === '/compact' && <em>{contextPercent}%</em>}</code>
          <small>{command.description}</small>
        </button>
      )) : <span className="slash-command-empty">No matching commands</span>}
    </div>
  );
}

const CONFIG_SECTIONS: Array<{key: keyof AgentConfig['sections']; title: string; description: string}> = [
  {key: 'llm', title: 'Agent LLM', description: 'Planning, tool selection, and conversation'},
  {key: 'image', title: 'Image generation', description: 'Characters, keyframes, and shot frames'},
  {key: 'video', title: 'Video generation', description: 'Shot clips and final video'},
  {key: 'embedding', title: 'Embedding', description: 'Optional novel retrieval'},
  {key: 'reranker', title: 'Reranker', description: 'Optional novel retrieval ranking'},
];
function redactConfig(config: AgentConfig): AgentConfig {
  return {
    ...config,
    sections: Object.fromEntries(Object.entries(config.sections).map(([section, value]) => [
      section,
      {
        ...value,
        api_key: '',
        api_key_updates: undefined,
        has_api_key: Array.isArray(value.configured_api_key_providers)
          ? value.configured_api_key_providers.includes(credentialProvider(value))
          : value.has_api_key,
      },
    ])) as AgentConfig['sections'],
  };
}


function SettingsView({agentBusy, onSaved}: {agentBusy: boolean; onSaved: () => Promise<void>}) {
  const [config, setConfig] = useState<AgentConfig>();
  const [drafts, setDrafts] = useState<Partial<Record<keyof AgentConfig['sections'], Record<string, string>>>>({});
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [status, setStatus] = useState('');

  useEffect(() => {
    let cancelled = false;
    void getAgentConfig()
      .then((payload) => !cancelled && setConfig(redactConfig(payload)))
      .catch((error) => !cancelled && setStatus(error instanceof Error ? error.message : String(error)))
      .finally(() => !cancelled && setLoading(false));
    return () => { cancelled = true; };
  }, []);

  function update(section: keyof AgentConfig['sections'], field: keyof ConfigSection, value: string) {
    if (saving) return;
    setStatus('');
    if (field === 'api_key') {
      setDrafts((current) => ({
        ...current,
        [section]: {...current[section], [credentialProvider(config?.sections[section] || {})]: value},
      }));
      setConfig((current) => {
        if (!current) return current;
        return {...current, sections: {...current.sections, [section]: {...current.sections[section], api_key: value}}};
      });
      return;
    }
    setConfig((current) => {
      if (!current) return current;
      const sectionValue = current.sections[section];
      const nextSection = {...sectionValue, [field]: value};
      if (section === 'video' && field === 'shot_duration_limits') {
        const limits = {...sectionValue.shot_duration_limits};
        const key = `${sectionValue.provider || 'openrouter'}:${sectionValue.model}`;
        if (value === '') delete limits[key]; else limits[key] = Number(value);
        nextSection.shot_duration_limits = limits;
      }
      if (section === 'video' && field === 'model') {
        Object.assign(nextSection, videoModelSelection(String(sectionValue.provider || 'openrouter'), value, sectionValue.resolution || '', sectionValue.clip_seconds || sectionValue.effective_clip_seconds || '8'));
      }
      if (section === 'image' && field === 'model') {
        const provider = imageProviderFromBaseUrl(sectionValue.base_url);
        nextSection.resolution = compatibleImageResolution(sectionValue.resolution || defaultImageResolution(provider, sectionValue.model), provider, value);
      }
      const identity = credentialProvider(nextSection);
      nextSection.api_key = drafts[section]?.[identity] || '';
      nextSection.has_api_key = (nextSection.configured_api_key_providers || []).includes(identity);
      return {sections: {...current.sections, [section]: nextSection}};
    });
  }

  function selectImageProvider(provider: string) {
    if (saving) return;
    const preset = imageProviderPreset(provider);
    if (!preset) return;
    setStatus('');
    setConfig((current) => {
      if (!current) return current;
      const image = current.sections.image;
      const nextImage = {...image, base_url: preset.baseUrl, model: preset.defaultModel};
      nextImage.resolution = compatibleImageResolution(image.resolution || defaultImageResolution(imageProviderFromBaseUrl(image.base_url), image.model), provider, preset.defaultModel);
      const identity = credentialProvider(nextImage);
      nextImage.api_key = drafts.image?.[identity] || '';
      nextImage.has_api_key = (nextImage.configured_api_key_providers || []).includes(identity);
      return {...current, sections: {...current.sections, image: nextImage}};
    });
  }

  function selectVideoProvider(provider: string) {
    if (saving) return;
    const preset = videoProviderPreset(provider);
    setStatus('');
    setConfig((current) => {
      if (!current) return current;
      const video = current.sections.video;
      const nextVideo = preset
        ? {...video, provider, base_url: preset.baseUrl, model: preset.defaultModel, resolution: preset.resolution, clip_seconds: '8'}
        : {...video, provider};
      const identity = credentialProvider(nextVideo);
      nextVideo.api_key = drafts.video?.[identity] || '';
      nextVideo.has_api_key = (nextVideo.configured_api_key_providers || []).includes(identity);
      return {...current, sections: {...current.sections, video: nextVideo}};
    });
  }

  async function save() {
    if (!config || saving || agentBusy || !validVideoSettings) return;
    setSaving(true);
    setStatus('');
    const request: AgentConfig = {
      ...config,
      sections: Object.fromEntries(Object.entries(config.sections).map(([section, value]) => [
        section,
        {
          ...value,
          api_key: drafts[section as keyof AgentConfig['sections']]?.[credentialProvider(value)] || '',
          api_key_updates: Object.fromEntries(Object.entries(drafts[section as keyof AgentConfig['sections']] || {}).filter(([, secret]) => secret)),
        },
      ])) as AgentConfig['sections'],
    };
    try {
      setConfig(redactConfig(await saveAgentConfig(request)));
      setDrafts({});
      await onSaved();
      setStatus('Saved');
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setSaving(false);
    }
  }

  const videoSettings = config?.sections.video;
  const durationPolicy = videoSettings ? videoDurationPolicy(videoSettings) : undefined;
  const knownDurationPolicy = videoSettings ? videoDurationPolicy({...videoSettings, shot_duration_limits: {}}) : undefined;
  const configuredMaximum = videoSettings?.shot_duration_limits?.[`${videoSettings.provider || 'openrouter'}:${videoSettings.model}`];
  const defaultSeconds = Number(videoSettings?.clip_seconds || videoSettings?.effective_clip_seconds || '8');
  const validDuration = Number.isInteger(defaultSeconds) && defaultSeconds > 0
    && (!knownDurationPolicy?.allowedSeconds || knownDurationPolicy.allowedSeconds.includes(defaultSeconds))
    && (configuredMaximum === undefined || Number.isInteger(configuredMaximum) && configuredMaximum >= defaultSeconds
      && (!knownDurationPolicy?.allowedSeconds || knownDurationPolicy.allowedSeconds.includes(configuredMaximum))
      && durationPolicy !== undefined && !validateShotDuration(defaultSeconds, durationPolicy));
  const validVideoSettings = validDuration && (videoSettings?.provider === 'openrouter' ? validHeygenVideoSettings(videoSettings.model, videoSettings.resolution, String(defaultSeconds)) : true);
  const credentialsSupported = Boolean(config && CONFIG_SECTIONS.every(({key}) => Array.isArray(config.sections[key].configured_api_key_providers)));
  if (loading) return <div className="settings-loading">Loading configuration…</div>;
  if (!config) return <div className="settings-loading is-error">{status || 'Configuration unavailable'}</div>;
  const imageResolutionSupported = Object.hasOwn(config.sections.image, 'resolution');
  return (
    <section className="settings-view">
      <header>
        <div><span>Local configuration</span><h1>Settings</h1></div>
        <div className="settings-save-group">
          {status && <span className={status === 'Saved' ? 'is-saved' : 'is-error'}>{status}</span>}
          <button className="settings-save" onClick={() => void save()} disabled={saving || agentBusy || !validVideoSettings || !credentialsSupported || !imageResolutionSupported}>
            <Save size={15} />{saving ? 'Saving…' : 'Save'}
          </button>
          {!credentialsSupported && <span className="is-error">Restart the web server after the active request finishes, then reload settings to enable provider-specific key saving.</span>}
          {!imageResolutionSupported && <span className="is-error">Reload settings after the backend reloads to enable image resolution saving.</span>}
          {agentBusy && <span className="is-error">The agent is busy or connecting. Wait before saving model settings; the current generation will not be interrupted.</span>}
        </div>
      </header>
      <div className="settings-sections">
        {CONFIG_SECTIONS.map((definition) => (
          <ConfigSectionEditor
            key={definition.key}
            definition={definition}
            value={config.sections[definition.key]}
            onChange={(field, value) => definition.key === 'image' && field === 'provider' ? selectImageProvider(value) : definition.key === 'video' && field === 'provider' ? selectVideoProvider(value) : update(definition.key, field, value)}
            validVideoSettings={definition.key !== 'video' || validVideoSettings}
            disabled={saving}
            imageResolutionSupported={definition.key !== 'image' || imageResolutionSupported}
          />
        ))}
      </div>
    </section>
  );
}

const OPENROUTER_IMAGE_MODELS = IMAGE_PROVIDER_PRESETS.openrouter.models;
const IMAGE_EDITING_MODELS = [
  {id: 'black-forest-labs/flux-3-image-klein', label: 'Black Forest Labs: FLUX 3 Image Klein'},
  {id: 'krea/k3-edit', label: 'Krea K3 Edit'},
  {id: 'qwen/qwen-image-2.1-turbo', label: 'Qwen Image 2.1 Turbo'},
];

function ConfigSectionEditor({definition, value, onChange, validVideoSettings, disabled, imageResolutionSupported}: {
  definition: {key: keyof AgentConfig['sections']; title: string; description: string};
  value: ConfigSection;
  onChange: (field: keyof ConfigSection, value: string) => void;
  validVideoSettings: boolean;
  disabled: boolean;
  imageResolutionSupported: boolean;
}) {
  const heygen = definition.key === 'video' && value.provider === 'openrouter' && isHeygenVideoModel(value.model);
  const duration = value.clip_seconds || value.effective_clip_seconds || '8';
  const modelDurationPolicy = videoDurationPolicy({...value, shot_duration_limits: {}});
  const durationLimit = value.shot_duration_limits?.[`${value.provider || 'openrouter'}:${value.model}`];
  const resolutionOptions = heygen ? ['480p', '768p'] : value.provider === 'agnes' ? ['720p', '1080p', '1K', '2K'] : value.provider === 'ltx' ? ['720p', '1080p', '1440p', '4k'] : ['480p', '720p', '1080p'];
  const imageProvider = definition.key === 'image' ? imageProviderFromBaseUrl(value.base_url) : undefined;
  const openrouterImage = imageProvider === 'openrouter';
  const agnesImage = imageProvider === 'agnes';
  const imageTiers = definition.key === 'image' ? imageResolutionTiers(imageProvider || '', value.model) : [];
  const customOpenRouterImage = openrouterImage && imageTiers.length === 0;
  const imageResolution = value.resolution || value.effective_resolution || defaultImageResolution(imageProvider || '', value.model);
  const [customImageModel, setCustomImageModel] = useState('');
  const [customEditingModel, setCustomEditingModel] = useState(false);
  const editingModelIsCustom = customEditingModel || !IMAGE_EDITING_MODELS.some((model) => model.id === value.editing_model);
  const imageModels = imageProvider === 'agnes' ? IMAGE_PROVIDER_PRESETS.agnes.models : OPENROUTER_IMAGE_MODELS;
  const imageModelIsCustom = customImageModel === value.model || !imageModels.some((model) => model.id === value.model);
  return (
    <section className="config-section">
      <header><h2>{definition.title}</h2><p>{definition.description}</p></header>
      <fieldset className="config-fields" disabled={disabled} style={{border: 0, padding: 0, margin: 0, minWidth: 0}}>
        {value.model_provider !== undefined && (
          <label><span>Model provider</span><input value={value.model_provider} onChange={(event) => onChange('model_provider', event.target.value)} /></label>
        )}
        {definition.key === 'image' && (
          <label><span>Provider</span>
            <select value={imageProvider} onChange={(event) => onChange('provider' as keyof ConfigSection, event.target.value)}>
              {Object.entries(IMAGE_PROVIDER_PRESETS).map(([provider, preset]) => <option key={provider} value={provider}>{preset.label}</option>)}
            </select>
          </label>
        )}
        {(openrouterImage || agnesImage) ? <>
          <label><span>Model</span>
            <select value={imageModelIsCustom ? '@custom' : value.model} onChange={(event) => {
              const custom = event.target.value === '@custom';
              setCustomImageModel(custom ? value.model : '');
              if (!custom) onChange('model', event.target.value);
            }}>
              {imageModels.map((model) => <option key={model.id} value={model.id}>{model.label}</option>)}
              <option value="@custom">Custom model…</option>
            </select>
          </label>
          {imageModelIsCustom && <label><span>Custom model ID</span><input value={value.model} onChange={(event) => onChange('model', event.target.value)} /></label>}
        </> : <label><span>Model</span><input value={value.model} list={definition.key === 'video' && value.provider === 'openrouter' ? 'openrouter-video-models' : undefined} onChange={(event) => onChange('model', event.target.value)} />
          {definition.key === 'video' && value.provider === 'openrouter' && <datalist id="openrouter-video-models">{VIDEO_PROVIDER_PRESETS.openrouter.models.map((model) => <option key={model} value={model} />)}</datalist>}
        </label>}
        {definition.key === 'image' && <>
          <label><span>Image editing model</span>
            <select value={editingModelIsCustom ? '@custom' : value.editing_model} onChange={(event) => {
              const custom = event.target.value === '@custom';
              setCustomEditingModel(custom);
              if (!custom) onChange('editing_model', event.target.value);
            }}>
              {IMAGE_EDITING_MODELS.map((model) => <option key={model.id} value={model.id}>{model.label}</option>)}
              <option value="@custom">Custom model…</option>
            </select>
          </label>
          {editingModelIsCustom && <label><span>Custom editing model ID</span><input value={value.editing_model || ''} onChange={(event) => onChange('editing_model', event.target.value)} /></label>}
          <p className="config-field-wide">Separate OpenRouter model choice for image editing. Does not change the model used to generate portraits or keyframes.</p>
        </>}
        {definition.key === 'video' && (
          <label><span>Provider</span>
            <select value={value.provider || 'openrouter'} onChange={(event) => onChange('provider', event.target.value)}>
              <option value="openrouter">OpenRouter</option><option value="yunwu">Yunwu</option>
              <option value="agnes">Agnes AI</option><option value="ltx">LTX</option>
            </select>
          </label>
        )}
        <label className="config-field-wide"><span>Base URL</span><input value={value.base_url} onChange={(event) => onChange('base_url', event.target.value)} inputMode="url" /></label>
        {definition.key === 'image' && (
          <>
            <label><span>Resolution</span>
              {customOpenRouterImage ? (
                <>
                  <input list="custom-openrouter-image-resolutions" value={imageResolution} disabled={!imageResolutionSupported} onChange={(event) => onChange('resolution', event.target.value)} />
                  <datalist id="custom-openrouter-image-resolutions"><option value="auto" /><option value="1K" /><option value="2K" /></datalist>
                </>
              ) : (
                <select value={imageResolution} disabled={!imageResolutionSupported} onChange={(event) => onChange('resolution', event.target.value)}>
                  {imageResolution !== 'auto' && !imageTiers.includes(imageResolution) && <option value={imageResolution}>{imageResolution} (unsupported; choose a listed value)</option>}
                  <option value="auto">Provider default</option>
                  {imageTiers.map((tier) => <option key={tier} value={tier}>{tier}</option>)}
                </select>
              )}
            </label>
            {imageResolution !== 'auto' && !imageTiers.includes(imageResolution) && !customOpenRouterImage && <p className="film-warning config-field-wide">This resolution is unsupported for the selected model. Choose Provider default.</p>}
            <p className="config-field-wide">Applies to new portraits and keyframes only; existing assets are not resized or regenerated.</p>
            {value.effective_resolution && <p className="config-field-wide">Active saved resolution: {value.effective_resolution === 'auto' ? 'Provider default' : value.effective_resolution}. VIMAX_IMAGE_RESOLUTION overrides the saved choice when set.</p>}
            {customOpenRouterImage && <p className="film-warning config-field-wide">Custom OpenRouter model resolutions are checked against the provider catalog; unsupported tiers are rejected before generation.</p>}
          </>
        )}
        {definition.key === 'video' && (
          <>
            {value.provider !== 'yunwu' && (
              <>
                <label><span>Resolution</span>
                  <select value={value.resolution || (heygen ? '' : value.provider === 'ltx' ? '1080p' : '720p')} onChange={(event) => onChange('resolution', event.target.value)}>
                    {!value.resolution && heygen && <option value="">Resolution not configured — choose one</option>}
                    {value.resolution && !resolutionOptions.includes(value.resolution) && <option value={value.resolution}>{value.resolution} (unsupported; choose a listed value)</option>}
                    {resolutionOptions.map((resolution) => <option key={resolution} value={resolution}>{resolution}</option>)}
                  </select>
                </label>
                <label><span>Default shot duration (seconds)</span>{modelDurationPolicy.allowedSeconds ? <select value={duration} onChange={(event) => onChange('clip_seconds', event.target.value)}>
                  {!modelDurationPolicy.allowedSeconds.includes(Number(duration)) && <option value={duration}>{duration}s · unsupported</option>}
                  {modelDurationPolicy.allowedSeconds.map((seconds: number) => <option key={seconds} value={seconds}>{seconds}s</option>)}
                </select> : <input type="number" min={1} step={1} value={duration} onChange={(event) => onChange('clip_seconds', event.target.value)} />}</label>
              </>
            )}
            <label><span>Maximum shot duration for this model (seconds)</span>{modelDurationPolicy.allowedSeconds ? <select aria-label="Maximum shot duration for this model (seconds)" value={durationLimit === undefined ? '' : String(durationLimit)} onChange={(event) => onChange('shot_duration_limits', event.target.value)}>
              <option value="">Not configured · shot overrides disabled</option>
              {durationLimit !== undefined && !modelDurationPolicy.allowedSeconds.includes(durationLimit) && <option value={durationLimit}>{durationLimit}s · unsupported</option>}
              {modelDurationPolicy.allowedSeconds.map((seconds: number) => <option key={seconds} value={seconds}>{seconds}s</option>)}
            </select> : <input aria-label="Maximum shot duration for this model (seconds)" type="number" min={1} step={1} value={durationLimit ?? ''} placeholder="Configure to enable shot overrides" onChange={(event) => onChange('shot_duration_limits', event.target.value)} />}</label>
            <p className="config-field-wide">Save a maximum for {value.model || 'the selected model'} to enable individual shot lengths in Film. Limits are remembered separately for each provider/model. {modelDurationPolicy.allowedSeconds ? `Supported durations: ${modelDurationPolicy.allowedSeconds.join(', ')} seconds.` : 'For custom models, verify supported durations with the provider before configuring a maximum.'} Existing clips are not regenerated.</p>
            {!validVideoSettings && <p className="film-warning config-field-wide">Choose a supported default shot duration and a model maximum at least as long as the default. HeyGen also requires a supported resolution.</p>}
            {heygen && <p className="film-warning config-field-wide">HeyGen conditions the clip on the first frame only; the final pose is not constrained. Native audio is provided by the model and cannot be toggled via OpenRouter.</p>}
            {value.provider !== 'agnes' && value.provider !== 'yunwu' && !heygen && (
              <label><span>Generate audio</span>
                <select value={value.generate_audio || 'true'} onChange={(event) => onChange('generate_audio', event.target.value)}>
                  <option value="true">On</option><option value="false">Off</option>
                </select>
              </label>
            )}
          </>
        )}
        <label className="config-field-wide">
          <span>API key <i className={value.has_api_key ? 'is-configured' : ''}>{value.has_api_key ? 'Configured' : 'Not configured'}</i></span>
          <input type="password" value={value.api_key} onChange={(event) => onChange('api_key', event.target.value)} placeholder={value.has_api_key ? 'Leave blank to keep current key' : 'Enter API key'} autoComplete="off" />
          <small>Saved keys are remembered per provider.</small>
        </label>
      </fieldset>
    </section>
  );
}


function ProjectView({sessionId, onChanged}: {sessionId: string; onChanged: () => void}) {
  const [project, setProject] = useState<ProjectMetadata>();
  const [form, setForm] = useState<ProjectFields>();
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [status, setStatus] = useState('');
  const [confirmation, setConfirmation] = useState<{request: ProjectUpdateRequest; prompt: InvalidationConfirmation}>();

  useEffect(() => {
    setConfirmation(undefined);
    setStatus('');
    if (!sessionId) {
      setProject(undefined);
      setForm(undefined);
      setLoading(false);
      return;
    }
    let cancelled = false;
    setLoading(true);
    void readProject(sessionId)
      .then((payload) => {
        if (cancelled) return;
        setProject(payload);
        setForm(toProjectFields(payload));
      })
      .catch((error) => !cancelled && setStatus(error instanceof Error ? error.message : String(error)))
      .finally(() => !cancelled && setLoading(false));
    return () => { cancelled = true; };
  }, [sessionId]);

  function applyResponse(response: ProjectUpdateResponse) {
    setProject(response.session);
    setForm(toProjectFields(response.session));
    onChanged();
  }

  async function save() {
    if (!project || !form || saving) return;
    const request = diffProjectFields(project.sessionId, toProjectFields(project), form);
    if (!request) {
      setStatus('No changes to save');
      return;
    }
    setSaving(true);
    setStatus('');
    try {
      const response = await updateProject(request);
      applyResponse(response);
      const prompt = describeInvalidation(response);
      if (prompt) {
        // The fields are already saved; the confirmation only gates the deletion.
        setConfirmation({request: {...request, invalidate: true}, prompt});
        setStatus('Saved · regeneration not confirmed yet');
      } else {
        setStatus('Saved');
      }
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setSaving(false);
    }
  }

  async function confirmInvalidation() {
    if (!confirmation || saving) return;
    setSaving(true);
    setStatus('');
    try {
      const response = await updateProject(confirmation.request);
      applyResponse(response);
      setConfirmation(undefined);
      const count = `${response.invalidated.length} artifact path${response.invalidated.length === 1 ? '' : 's'} removed`;
      setStatus(response.regenerationStarted ? `Saved · ${count}; the agent is rebuilding them` : `Saved · ${count} for regeneration`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setSaving(false);
    }
  }

  if (!sessionId) return <div className="settings-loading">Create or select a project to see its metadata</div>;
  if (loading) return <div className="settings-loading">Loading project…</div>;
  if (!project || !form) return <div className="settings-loading is-error">{status || 'Project unavailable'}</div>;

  const manifest = project.manifest;
  const styleMismatch = describeStyleMismatch(project.style, manifest?.style);
  const readiness = describeArtifacts(project.artifacts);
  const renderedCount = readiness.filter((checkpoint) => checkpoint.state === 'ready').length;
  const dirty = Boolean(diffProjectFields(project.sessionId, toProjectFields(project), form));

  return (
    <section className="settings-view project-view">
      <header>
        <div><span>Project metadata</span><h1>{project.projectName || 'Untitled project'}</h1></div>
        <div className="settings-save-group">
          {dirty && <span className="project-dirty">Unsaved changes</span>}
          {status && <span className={status.startsWith('Saved') ? 'is-saved' : ''}>{status}</span>}
          <button className="settings-save" onClick={() => void save()} disabled={saving || !dirty}>
            <Save size={15} />{saving ? 'Saving…' : 'Save'}
          </button>
        </div>
      </header>
      {styleMismatch && <p className="project-notice" role="status">{styleMismatch}</p>}
      <div className="settings-sections">
        <section className="config-section">
          <header><h2>Editable</h2><p>The style is interpolated into the character portrait prompt and inherited by every later frame.</p></header>
          <div className="config-fields">
            <label>
              <span>Project name</span>
              <input value={form.projectName} onChange={(event) => setForm({...form, projectName: event.target.value})} placeholder="Untitled video" maxLength={64} disabled={saving} />
            </label>
            <label>
              <span>Style</span>
              <input value={form.style} onChange={(event) => setForm({...form, style: event.target.value})} placeholder="photorealistic cinematic live action" disabled={saving} />
            </label>
            <label className="config-field-wide">
              <span>Idea</span>
              <input value={form.idea} onChange={(event) => setForm({...form, idea: event.target.value})} placeholder="What the video should be about" disabled={saving} />
            </label>
            <label className="config-field-wide">
              <span>User requirement <i>Optional</i></span>
              <input value={form.userRequirement} onChange={(event) => setForm({...form, userRequirement: event.target.value})} placeholder="Constraints the agent must follow" disabled={saving} />
            </label>
          </div>
        </section>
        <section className="config-section">
          <header><h2>Session</h2><p>Identity and progress of the agent session behind this project.</p></header>
          <div className="structured-fields">
            <ReadOnlyField label="Session id" value={project.sessionId} />
            <ReadOnlyField label="Working dir" value={project.workingDir} />
            <ReadOnlyField label="Stage" value={stageLabel(project.stage)} />
            <ReadOnlyField label="Summary" value={project.summary} long />
          </div>
        </section>
        <section className="config-section">
          <header><h2>Render</h2><p>Models and style pinned by the last render, plus what is still valid.</p></header>
          <div className="project-render">
            <div className="structured-fields">
              <ReadOnlyField label="Pinned image model" value={manifest?.image_model || 'Not pinned yet'} />
              <ReadOnlyField label="Pinned video model" value={manifest?.video_model || 'Not pinned yet'} />
              <ReadOnlyField label="Style in render manifest" value={manifest?.style || 'No render yet'} long />
            </div>
            <div className="project-readiness">
              <span className="project-readiness-title">
                On disk now
                <i>{renderedCount}/{readiness.length} rendered</i>
              </span>
              <div className="render-checkpoints">
                {readiness.map((checkpoint) => (
                  <div className="render-checkpoint" key={checkpoint.label}>
                    <i className={`status-light is-${checkpoint.state === 'ready' ? 'ready' : checkpoint.state === 'partial' ? 'partial' : 'missing'}`} />
                    <span>{checkpoint.label}</span>
                    <small>{checkpoint.detail}</small>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>
      </div>
      <StyleInvalidationDialog
        open={Boolean(confirmation)}
        prompt={confirmation?.prompt}
        saving={saving}
        onCancel={() => {
          if (saving) return;
          setConfirmation(undefined);
          setStatus('Saved · existing artifacts kept');
        }}
        onConfirm={() => void confirmInvalidation()}
      />
    </section>
  );
}

function ReadOnlyField({label, value, long = false}: {label: string; value: string; long?: boolean}) {
  return (
    <div className={`structured-field ${long ? 'is-long' : ''}`}>
      <span>{label}</span>
      <p>{value || '—'}</p>
    </div>
  );
}

function StyleInvalidationDialog({open, prompt, saving, onCancel, onConfirm}: {
  open: boolean;
  prompt?: InvalidationConfirmation;
  saving: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  useEffect(() => {
    if (!open) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !saving) onCancel();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [onCancel, open, saving]);

  if (!open || !prompt) return null;
  return (
    <div className="dialog-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onCancel()}>
      <section className="project-dialog invalidation-dialog" role="dialog" aria-modal="true" aria-labelledby="invalidation-title">
        <span className="dialog-icon"><RefreshCw size={18} /></span>
        <div className="dialog-copy">
          <h2 id="invalidation-title">{prompt.heading}</h2>
          <p>{prompt.paths.length === 1 ? '1 artifact path is deleted and rebuilt on the next render:' : `${prompt.paths.length} artifact paths are deleted and rebuilt on the next render:`}</p>
          <ul className="invalidation-paths">
            {prompt.paths.map((artifactPath) => <li key={artifactPath}><code>{artifactPath}</code></li>)}
          </ul>
          <p>{prompt.detail}</p>
        </div>
        <div className="dialog-actions">
          <button onClick={onCancel} disabled={saving} autoFocus>Keep artifacts</button>
          <button className="danger" onClick={onConfirm} disabled={saving}>{saving ? 'Regenerating…' : 'Regenerate'}</button>
        </div>
      </section>
    </div>
  );
}

function DeleteProjectDialog({session, deleting, onCancel, onConfirm}: {
  session?: SessionSummary;
  deleting: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  useEffect(() => {
    if (!session) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !deleting) onCancel();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [deleting, onCancel, session]);

  if (!session) return null;
  return (
    <div className="dialog-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onCancel()}>
      <section className="delete-dialog" role="dialog" aria-modal="true" aria-labelledby="delete-project-title">
        <span className="dialog-icon"><Trash2 size={18} /></span>
        <div className="dialog-copy">
          <h2 id="delete-project-title">Delete project?</h2>
          <p><strong>{sessionTitle(session)}</strong> and its generated files will be permanently removed.</p>
        </div>
        <div className="dialog-actions">
          <button onClick={onCancel} disabled={deleting} autoFocus>Cancel</button>
          <button className="danger" onClick={onConfirm} disabled={deleting}>{deleting ? 'Deleting…' : 'Delete'}</button>
        </div>
      </section>
    </div>
  );
}

function NewProjectDialog({open, name, style, requirement, error, creating, blocked, onNameChange, onStyleChange, onRequirementChange, onCancel, onConfirm}: {
  open: boolean;
  name: string;
  style: string;
  requirement: string;
  error: string;
  creating: boolean;
  blocked: boolean;
  onNameChange: (value: string) => void;
  onStyleChange: (value: string) => void;
  onRequirementChange: (value: string) => void;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  useEffect(() => {
    if (!open) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !creating) onCancel();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [creating, onCancel, open]);

  if (!open) return null;
  return (
    <div className="dialog-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onCancel()}>
      <form className="project-dialog" role="dialog" aria-modal="true" aria-labelledby="new-project-title" onSubmit={(event) => {
        event.preventDefault();
        onConfirm();
      }}>
        <span className="dialog-icon is-create"><FolderPlus size={18} /></span>
        <div className="dialog-copy">
          <h2 id="new-project-title">Create a new project</h2>
          <p>Name the workspace and choose the style its renders must follow.</p>
        </div>
        <label className="project-name-field">
          <span>Project name</span>
          <input
            value={name}
            onChange={(event) => onNameChange(event.target.value)}
            placeholder="Untitled video"
            maxLength={64}
            autoFocus
            disabled={creating}
          />
          {error && <small role="alert">{error}</small>}
        </label>
        <label className="project-name-field">
          <span>Style</span>
          <input
            value={style}
            onChange={(event) => onStyleChange(event.target.value)}
            placeholder="photorealistic cinematic live action"
            disabled={creating}
          />
          <small className="field-hint">Applied to the character portraits and inherited by every frame. Leave blank to let the agent ask you for one.</small>
        </label>
        <label className="project-name-field">
          <span>User requirement <i className="field-tag">Optional</i></span>
          <input
            value={requirement}
            onChange={(event) => onRequirementChange(event.target.value)}
            placeholder="e.g. for children, at most three scenes"
            disabled={creating}
          />
        </label>
        {blocked && <p className="film-warning" role="status">Project creation is unavailable while the agent is busy or connecting. Wait for active and queued requests to finish. Browsing existing projects does not interrupt generation.</p>}
        <div className="dialog-actions">
          <button type="button" onClick={onCancel} disabled={creating}>Cancel</button>
          <button type="submit" className="primary" disabled={creating || blocked || !name.trim()}>{creating ? 'Creating…' : 'Create'}</button>
        </div>
      </form>
    </div>
  );
}

function sessionTitle(session?: SessionSummary) {
  if (!session) return 'New video';
  if (session.projectName) return session.projectName;
  const source = session.idea || session.summary;
  if (source) return source.length > 38 ? `${source.slice(0, 38).trim()}…` : source;
  return session.sessionId.replace(/^\d{8}-\d{6}-?/, '') || 'Untitled video';
}

function stageLabel(stage: string) {
  const labels: Record<string, string> = {
    created: 'Created',
    narrative_planning: 'Planning',
    narrative_planned: 'Plan ready',
    novel_planning: 'Planning novel',
    novel_planned: 'Novel ready',
    rendering: 'Rendering',
    rendered: 'Rendered',
    error: 'Needs attention',
  };
  return labels[stage] || humanize(stage || 'Created');
}

