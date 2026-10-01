import {useCallback, useEffect, useRef, useState} from 'react';
import {readFilmProgress, readFilmSnapshot} from './api';
import {renderActivity} from './timeline';
import type {Artifact, ContinuityPayload, JsonValue, RemovedShot, RenderAcceptance, ShotPlan} from './types';

export type FilmSnapshot = {
  acceptance: RenderAcceptance;
  plans: ShotPlan[];
  removed: RemovedShot[];
  continuity: ContinuityPayload;
  artifacts: Artifact[];
};

export type FilmProgress = {
  status: JsonValue | null;
  trail: string;
  revision: string;
  lastProgressAt: string;
  agentRunning: boolean;
  activeSessionId: string;
};

export type FilmSelection = {root: string; slot: string; label: string};

export type FilmSessionState = {
  data?: FilmSnapshot;
  progress?: FilmProgress;
  loading: boolean;
  error: string;
  refresh: () => Promise<void>;
  wake: () => void;
};

type SessionView = {
  owner: string;
  data?: FilmSnapshot;
  progress?: FilmProgress;
  loading: boolean;
  snapshotError: string;
  progressError: string;
};

type Coordinator = {owner: string; refresh: () => Promise<void>; wake: () => void};
const ACTIVE_DELAY = 3_000;
const IDLE_DELAY = 12_000;
const SNAPSHOT_MAX_AGE = 30_000;

/** One owner for the film, assets, and live status. Heavy reads never overlap. */
export function useFilmSession(sessionId: string): FilmSessionState {
  const [view, setView] = useState<SessionView>({owner: '', loading: false, snapshotError: '', progressError: ''});
  const coordinator = useRef<Coordinator>();

  useEffect(() => {
    const controller = new AbortController();
    let disposed = false;
    let timer: number | undefined;
    let inFlight: Promise<void> | undefined;
    let queuedRefresh: Promise<void> | undefined;
    let polling = false;
    let pollAgain = false;
    let lastPollStartedAt = 0;
    let latestProgress: FilmProgress | undefined;
    let snapshotRevision: string | undefined;
    let lastSnapshotAt = 0;
    let activeUntil = 0;
    let root = '';
    let hasSnapshot = false;
    let lastMediaWrite = '';

    const withMediaProgress = (progress: FilmProgress): FilmProgress => {
      return Date.parse(lastMediaWrite) > (Date.parse(progress.lastProgressAt) || 0)
        ? {...progress, lastProgressAt: lastMediaWrite}
        : progress;
    };

    const publish = (patch: Partial<SessionView>) => {
      if (!disposed) setView((current) => current.owner === sessionId ? {...current, ...patch} : current);
    };
    setView({owner: sessionId, loading: Boolean(sessionId), snapshotError: '', progressError: ''});

    const snapshot = (): Promise<void> => {
      if (disposed || !sessionId) return Promise.resolve();
      if (inFlight) return inFlight;
      const revisionAtStart = latestProgress?.revision;
      if (!hasSnapshot) publish({loading: true});
      inFlight = readFilmSnapshot(sessionId, '', controller.signal)
        .then((data) => {
          if (disposed) return;
          root = data.acceptance.root;
          lastMediaWrite = '';
          for (const artifact of data.artifacts) {
            if (artifact.kind !== 'document' && artifact.path.startsWith(`${root}/`)
              && Date.parse(artifact.updatedAt) > (Date.parse(lastMediaWrite) || 0)) lastMediaWrite = artifact.updatedAt;
          }
          if (latestProgress) latestProgress = withMediaProgress(latestProgress);
          hasSnapshot = true;
          snapshotRevision = revisionAtStart;
          lastSnapshotAt = Date.now();
          publish({data, progress: latestProgress, snapshotError: ''});
        })
        .catch((error: unknown) => {
          if (disposed) return;
          publish({snapshotError: error instanceof Error ? error.message : String(error)});
          throw error;
        })
        .finally(() => {
          inFlight = undefined;
          publish({loading: false});
        });
      return inFlight;
    };

    // A mutation may finish while an older snapshot is in flight. Queue one fresh
    // read behind it rather than satisfying the mutation with pre-save data.
    const refresh = (): Promise<void> => {
      if (disposed || !sessionId) return Promise.resolve();
      if (queuedRefresh) return queuedRefresh;
      if (!inFlight) return snapshot();
      queuedRefresh = inFlight.catch(() => undefined)
        .then(() => snapshot())
        .finally(() => { queuedRefresh = undefined; });
      return queuedRefresh;
    };

    const schedule = (delay: number) => {
      window.clearTimeout(timer);
      timer = undefined;
      if (!disposed && sessionId && document.visibilityState !== 'hidden') {
        timer = window.setTimeout(() => { void poll(); }, delay);
      }
    };

    const poll = async () => {
      if (disposed || !sessionId || document.visibilityState === 'hidden') return;
      if (polling) {
        pollAgain = true;
        return;
      }
      polling = true;
      lastPollStartedAt = Date.now();
      try {
        try {
          const progress = await readFilmProgress(sessionId, root, controller.signal);
          if (disposed) return;
          latestProgress = withMediaProgress(progress);
          publish({progress: latestProgress, progressError: ''});
        } catch (error) {
          if (disposed) return;
          publish({progressError: `Live updates unavailable: ${error instanceof Error ? error.message : String(error)}`});
        }
        if (!hasSnapshot || latestProgress?.revision !== snapshotRevision || Date.now() - lastSnapshotAt >= SNAPSHOT_MAX_AGE) {
          await (queuedRefresh || snapshot()).catch(() => undefined);
        }
      } finally {
        polling = false;
        const active = latestProgress && renderActivity(latestProgress.status ?? undefined, {
          agentRunning: latestProgress.activeSessionId === sessionId && latestProgress.agentRunning,
          lastProgressAt: latestProgress.lastProgressAt,
        }) === 'running';
        const delay = pollAgain ? Math.max(0, lastPollStartedAt + ACTIVE_DELAY - Date.now())
          : active || Date.now() < activeUntil ? ACTIVE_DELAY : IDLE_DELAY;
        pollAgain = false;
        schedule(delay);
      }
    };

    const wake = () => {
      activeUntil = Date.now() + 60_000;
      if (polling) pollAgain = true;
      else schedule(Math.max(0, lastPollStartedAt + ACTIVE_DELAY - Date.now()));
    };
    const onVisibility = () => {
      if (document.visibilityState === 'hidden') {
        window.clearTimeout(timer);
        timer = undefined;
      } else {
        schedule(0);
      }
    };
    const current = {owner: sessionId, refresh, wake};
    coordinator.current = current;
    document.addEventListener('visibilitychange', onVisibility);
    void poll();
    return () => {
      disposed = true;
      controller.abort();
      window.clearTimeout(timer);
      document.removeEventListener('visibilitychange', onVisibility);
      if (coordinator.current === current) coordinator.current = undefined;
    };
  }, [sessionId]);

  const refresh = useCallback(async () => {
    const current = coordinator.current;
    if (current?.owner === sessionId) await current.refresh();
  }, [sessionId]);
  const wake = useCallback(() => {
    const current = coordinator.current;
    if (current?.owner === sessionId) current.wake();
  }, [sessionId]);
  const scoped = view.owner === sessionId ? view : undefined;
  return {
    data: scoped?.data,
    progress: scoped?.progress,
    loading: scoped?.loading ?? Boolean(sessionId),
    error: scoped?.snapshotError || scoped?.progressError || '',
    refresh,
    wake,
  };
}
