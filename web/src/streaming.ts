export type DeltaBatcher<T> = {
  push(value: T, key: string): void;
  flush(): void;
};

export function createDeltaBatcher<T>(commit: (batch: T[]) => void, options: {
  fallbackMs?: number;
  maxBatchSize?: number;
  requestFrame?: (callback: FrameRequestCallback) => number;
  cancelFrame?: (id: number) => void;
} = {}): DeltaBatcher<T> {
  const fallbackMs = options.fallbackMs ?? 100;
  const maxBatchSize = Math.max(1, options.maxBatchSize ?? 256);
  const requestFrame = options.requestFrame || (typeof requestAnimationFrame === 'function' ? requestAnimationFrame : undefined);
  const cancelFrame = options.cancelFrame || (typeof cancelAnimationFrame === 'function' ? cancelAnimationFrame : undefined);
  const seenKeys = new Set<string>();
  let pending: T[] = [];
  let frameId: number | undefined;
  let timerId: ReturnType<typeof setTimeout> | undefined;

  const cancelScheduled = () => {
    if (frameId !== undefined && cancelFrame) cancelFrame(frameId);
    if (timerId !== undefined) clearTimeout(timerId);
    frameId = undefined;
    timerId = undefined;
  };

  const flush = () => {
    if (pending.length === 0) {
      cancelScheduled();
      return;
    }
    const batch = pending;
    pending = [];
    cancelScheduled();
    commit(batch);
  };

  const schedule = () => {
    if (frameId === undefined && requestFrame) frameId = requestFrame(() => flush());
    if (timerId === undefined) timerId = setTimeout(() => flush(), fallbackMs);
  };

  return {
    push(value, key) {
      if (!seenKeys.has(key)) {
        seenKeys.add(key);
        if (seenKeys.size > 1024) {
          const oldest = seenKeys.values().next().value;
          if (oldest) seenKeys.delete(oldest);
        }
        flush();
        commit([value]);
        return;
      }
      pending.push(value);
      if (pending.length >= maxBatchSize) flush();
      else schedule();
    },
    flush,
  };
}
