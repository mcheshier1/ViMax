import {afterEach, describe, expect, it, vi} from 'vitest';
import {createDeltaBatcher} from './streaming';

afterEach(() => {
  vi.useRealTimers();
});

describe('assistant delta batching', () => {
  it('commits the first delta immediately and batches later deltas on a frame', () => {
    const batches: string[][] = [];
    let frame: FrameRequestCallback | undefined;
    const batcher = createDeltaBatcher((batch: string[]) => batches.push(batch), {
      fallbackMs: 80,
      requestFrame: (callback) => { frame = callback; return 1; },
      cancelFrame: () => { frame = undefined; },
    });

    batcher.push('first', 'sample-1');
    expect(batches).toEqual([['first']]);
    batcher.push('second', 'sample-1');
    batcher.push('third', 'sample-1');
    expect(batches).toEqual([['first']]);
    frame?.(0);
    expect(batches).toEqual([['first'], ['second', 'third']]);
  });

  it('flushes lifecycle boundaries and bounds a throttled batch with a timer', () => {
    vi.useFakeTimers();
    const batches: string[][] = [];
    let frame: FrameRequestCallback | undefined;
    const batcher = createDeltaBatcher((batch: string[]) => batches.push(batch), {
      fallbackMs: 80,
      maxBatchSize: 2,
      requestFrame: (callback) => { frame = callback; return 1; },
      cancelFrame: () => { frame = undefined; },
    });

    batcher.push('visible', 'sample-2');
    batcher.push('queued-1', 'sample-2');
    batcher.flush();
    expect(batches).toEqual([['visible'], ['queued-1']]);

    batcher.push('queued-2', 'sample-2');
    batcher.push('bounded', 'sample-2');
    expect(batches.at(-1)).toEqual(['queued-2', 'bounded']);
    batcher.push('timer-fallback', 'sample-2');
    vi.advanceTimersByTime(80);
    expect(batches.at(-1)).toEqual(['timer-fallback']);
    expect(frame).toBeUndefined();
  });

  it('treats each sample first delta as immediate even when another sample is pending', () => {
    const batches: string[][] = [];
    let frame: FrameRequestCallback | undefined;
    const batcher = createDeltaBatcher((batch: string[]) => batches.push(batch), {
      requestFrame: (callback) => { frame = callback; return 1; },
      cancelFrame: () => { frame = undefined; },
    });
    batcher.push('first-A', 'sample-A');
    batcher.push('later-A', 'sample-A');
    batcher.push('first-B', 'sample-B');
    expect(batches).toEqual([['first-A'], ['later-A'], ['first-B']]);
    batcher.flush();
  });
});
