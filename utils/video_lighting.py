"""Conservative RGB exposure matching from stable pixels around clip joins."""

from __future__ import annotations

import cv2
import numpy as np


_MIN_PIXELS = 1200


def estimate_gains(previous: list[np.ndarray], incoming: list[np.ndarray],
                   previous_gain: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> dict:
    """Estimate one constant RGB multiplier for an incoming clip, or explain a skip.

    Frames must already have passed through the assembly geometry filter and be RGB.
    The estimator uses pixels temporally stable in both endpoint samples, rejects
    poor structural agreement, and never attempts semantic scene understanding.
    """
    neutral = [1.0, 1.0, 1.0]
    if len(previous) < 3 or len(incoming) < 3:
        return _skip("Too few endpoint samples for stable-background selection")
    if any(a.shape != b.shape or a.ndim != 3 or a.shape[2] != 3 for a, b in zip(previous, incoming)):
        return _skip("Endpoint geometry differs; lighting comparison is unreliable")
    # Work at bounded resolution; all samples are the same canonical assembly canvas.
    size = (min(320, previous[0].shape[1]), min(180, previous[0].shape[0]))
    a = np.stack([cv2.resize(frame, size, interpolation=cv2.INTER_AREA).astype(np.float32) for frame in previous])
    b = np.stack([cv2.resize(frame, size, interpolation=cv2.INTER_AREA).astype(np.float32) for frame in incoming])
    a *= np.asarray(previous_gain, dtype=np.float32)[None, None, None, :]
    # Constant portions of each clip are candidates; remove pixels near black/white,
    # saturated colors, and high-gradient moving/detail regions.
    ma = a.mean(axis=0)
    mb = b.mean(axis=0)
    temporal_stable = (a.std(axis=0).max(axis=2) < 8.0) & (b.std(axis=0).max(axis=2) < 8.0)
    stable = temporal_stable.copy()
    gray_a = cv2.cvtColor(np.clip(ma, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(np.clip(mb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges_a = cv2.Sobel(gray_a, cv2.CV_32F, 1, 0, ksize=3) ** 2 + cv2.Sobel(gray_a, cv2.CV_32F, 0, 1, ksize=3) ** 2
    edges_b = cv2.Sobel(gray_b, cv2.CV_32F, 1, 0, ksize=3) ** 2 + cv2.Sobel(gray_b, cv2.CV_32F, 0, 1, ksize=3) ** 2
    stable &= (edges_a < 1800) & (edges_b < 1800)
    stable &= (ma.min(axis=2) > 12) & (ma.max(axis=2) < 242)
    stable &= (mb.min(axis=2) > 12) & (mb.max(axis=2) < 242)
    stable &= ((ma.max(axis=2) - ma.min(axis=2)) < 170) & ((mb.max(axis=2) - mb.min(axis=2)) < 170)
    if int(stable.sum()) < _MIN_PIXELS or stable.mean() < 0.045:
        return _skip("No sufficiently broad, stable, unsaturated background region")
    # A rank-structure check is invariant to exposure but guards camera cuts / changed scenes.
    xa, xb = gray_a[stable].astype(np.float32), gray_b[stable].astype(np.float32)
    if xa.std() < 8 or xb.std() < 8 or np.corrcoef(xa, xb)[0, 1] < 0.82:
        return _skip("Structural mismatch or camera cut at join")
    # Fit on alternating cells of a real 4x3 spatial grid, then validate on the
    # held-out cells. The candidate background must span at least three cells.
    yy, xx = np.indices(stable.shape)
    rows = yy * 3 // stable.shape[0]
    cols = xx * 4 // stable.shape[1]
    tile = rows * 4 + cols
    present_tiles = np.unique(tile[stable])
    training = stable & ((rows + cols) % 2 == 0)
    heldout = stable & ((rows + cols) % 2 == 1)
    if len(present_tiles) < 3 or int(training.sum()) < _MIN_PIXELS // 2 or int(heldout.sum()) < _MIN_PIXELS // 2:
        return _skip("Stable background lacks spatial coverage for held-out validation")
    av, bv = ma[training], mb[training]
    valid = (av.min(axis=1) > 18) & (bv.min(axis=1) > 18)
    av, bv = av[valid], bv[valid]
    if len(av) < _MIN_PIXELS // 3:
        return _skip("Too few reliable background pixels")
    ratios = np.clip(av / np.maximum(bv, 1), 0.5, 2.0)
    raw = np.median(ratios, axis=0)
    if float(np.max(np.abs(raw - 1))) < 0.015:
        return {"status": "unchanged", "reason": "Estimated global lighting difference is below the correction threshold", "gains": neutral,
                "stable_fraction": float(stable.mean()), "brightness_gain": 1.0}
    hold_a, hold_b = ma[heldout], mb[heldout]
    hold_valid = (hold_a.min(axis=1) > 18) & (hold_b.min(axis=1) > 18)
    hold_a, hold_b = hold_a[hold_valid], hold_b[hold_valid]
    if len(hold_a) < _MIN_PIXELS // 3:
        return _skip("Too few held-out background pixels")
    base_error = float(np.mean(np.abs(hold_a - hold_b)))
    corrected = hold_b * raw[None, :]
    corrected_error = float(np.mean(np.abs(hold_a - corrected)))
    luma = np.asarray([0.2126, 0.7152, 0.0722])
    base_bias = float(np.mean((hold_a - hold_b) @ luma))
    corrected_bias = float(np.mean((hold_a - corrected) @ luma))
    if corrected_error >= base_error * 0.98 or (
            abs(base_bias) > 0.5 and abs(corrected_bias) >= abs(base_bias) * 0.98):
        return _skip("Lighting estimate did not improve held-out color and brightness error")
    held_residual = np.median(np.abs(hold_a - corrected) / np.maximum(hold_a, 24), axis=0)
    if float(np.max(held_residual)) > 0.12:
        return _skip("Lighting estimate failed held-out spatial validation")
    # Guard finer independent regions, including stationary textured pixels
    # excluded from fitting. Coarse aggregate cells can hide a harmed background
    # behind improvements in a large foreground or shadow.
    validation = temporal_stable & (ma.min(axis=2) > 18) & (mb.min(axis=2) > 18)
    validation &= (ma.max(axis=2) < 235) & (mb.max(axis=2) < 235)
    fine_cells = (yy * 6 // stable.shape[0]) * 8 + xx * 8 // stable.shape[1]
    for cell in np.unique(fine_cells[validation]):
        region = validation & (fine_cells == cell)
        if int(region.sum()) < 100:
            continue
        region_a, region_b = ma[region], mb[region]
        original_error = np.mean(np.abs(region_a - region_b), axis=0)
        adjusted_error = np.mean(np.abs(region_a - region_b * raw), axis=0)
        original_bias = abs(float(np.mean((region_a - region_b) @ luma)))
        adjusted_bias = abs(float(np.mean((region_a - region_b * raw) @ luma)))
        if np.any(adjusted_error > original_error + 0.5) or adjusted_bias > original_bias + 0.5:
            return _skip("Lighting correction would worsen a separate stable background region")
    original_pixels = 0
    before_clipped = 0
    after_clipped = 0
    for frame in incoming:
        pixels = frame.astype(np.float32, copy=False)
        original_pixels += pixels.shape[0] * pixels.shape[1]
        before_clipped += int(np.count_nonzero(np.any(pixels >= 250.0, axis=2)))
        after_clipped += int(np.count_nonzero(np.any(pixels * raw[None, None, :] >= 250.0, axis=2)))
    if original_pixels and (after_clipped - before_clipped) / original_pixels > 0.001:
        return _skip("Estimated gain would clip more than 0.1% additional incoming pixels")
    brightness = float(np.dot(raw, [0.2126, 0.7152, 0.0722]))
    wb = raw / max(brightness, 1e-9)
    if not 0.92 <= brightness <= 1.08:
        return _skip("Brightness difference exceeds the conservative correction range")
    if np.any(wb < 0.97) or np.any(wb > 1.03):
        return _skip("White-balance difference exceeds the conservative correction range")
    gains = raw.tolist()
    # Prior samples already include the previous clip's gain, making ``raw`` the
    # absolute correction needed to match the common identity reference.
    if np.any(raw < 0.90) or np.any(raw > 1.10):
        return _skip("Required absolute correction exceeds bounds; possible cumulative drift")
    return {"status": "corrected", "reason": "Bounded constant RGB gain matched on stable background",
            "gains": [float(x) for x in gains], "stable_fraction": float(stable.mean()),
            "brightness_gain": brightness}


def _skip(reason: str) -> dict:
    return {"status": "needs_review", "reason": reason, "gains": [1.0, 1.0, 1.0]}
