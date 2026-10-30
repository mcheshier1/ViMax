"""Local, keyframe-anchored geometric correction for assembled clips."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
if __package__:
    from .video_lighting import estimate_gains
else:
    from video_lighting import estimate_gains

ASSEMBLY_VERSION = 4
_MAX_SCALE_DELTA = 0.04
_MAX_OFFSET_X = 20.0
_MAX_OFFSET_Y = 20.0
_MAX_CROP_ZOOM = 1.05
_CANVAS_W = 1280
_CANVAS_H = 720
RENDERED_ENDING_NORMALIZATION = "rendered-end-v1"


def _executable(env_name: str, fallback: str) -> str:
    return os.environ.get(env_name, fallback)


def _run(args: list[str], *, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(args, check=True, stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
                          stderr=subprocess.PIPE, text=False)


def _probe(path: str) -> dict:
    result = _run([_executable("VIMAX_FFPROBE_CMD", "ffprobe"), "-v", "error", "-protocol_whitelist", "file,pipe", "-show_streams",
                   "-show_format", "-of", "json", path])
    info = json.loads(result.stdout)
    video = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError(f"No video stream in {path}")
    duration = _number(video.get("duration")) or _number(info.get("format", {}).get("duration"))
    if not duration or duration <= 0:
        raise ValueError(f"Cannot determine video duration: {path}")
    fps = _fraction(video.get("avg_frame_rate")) or _fraction(video.get("r_frame_rate"))
    nominal_fps = _fraction(video.get("r_frame_rate"))
    variable_rate = bool(fps and nominal_fps and abs(fps / nominal_fps - 1) > 0.01)
    sar = _fraction(video.get("sample_aspect_ratio")) or 1.0
    rotation = _number(video.get("tags", {}).get("rotate")) or 0.0
    for side in video.get("side_data_list", []):
        if "rotation" in side:
            rotation = float(side["rotation"])
    width = int(video["width"])
    height = int(video["height"])
    display_aspect = width * sar / height
    if abs(rotation) % 180 == 90:
        display_aspect = 1 / display_aspect
    return {"duration": duration, "audio": any(s.get("codec_type") == "audio" for s in info.get("streams", [])),
            "width": width, "height": height, "fps": fps, "display_aspect": display_aspect,
            "variable_rate": variable_rate, "sar": sar, "rotation": rotation}


def _number(value) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _fraction(value) -> float | None:
    try:
        numerator, denominator = str(value).replace(":", "/").split("/", 1)
        denominator = float(denominator)
        result = float(numerator) / denominator
        return result if denominator and math.isfinite(result) and result > 0 else None
    except (ValueError, ZeroDivisionError):
        return None


def _image(path: str) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)


def _frame(path: str, seconds: float) -> np.ndarray:
    args = [_executable("VIMAX_FFMPEG_CMD", "ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error",
            "-y", "-ss", f"{max(0.0, seconds):.6f}", "-protocol_whitelist", "file,pipe", "-i", path, "-vf",
            "scale=trunc(iw*sar/2)*2:ih:flags=bicubic,setsar=1", "-frames:v", "1", "-f", "image2pipe",
            "-vcodec", "png", "pipe:1"]
    result = _run(args)
    from io import BytesIO
    with Image.open(BytesIO(result.stdout)) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)


def _canonical(gray: np.ndarray) -> np.ndarray:
    return cv2.resize(gray, (_CANVAS_W, _CANVAS_H), interpolation=cv2.INTER_AREA if gray.shape[1] > _CANVAS_W else cv2.INTER_CUBIC)


def _match(source: np.ndarray, reference: np.ndarray) -> tuple[float, float, float, float] | None:
    sift = cv2.SIFT_create(nfeatures=1500)
    kp_a, desc_a = sift.detectAndCompute(source, None)
    kp_b, desc_b = sift.detectAndCompute(reference, None)
    if desc_a is None or desc_b is None or len(kp_a) < 40 or len(kp_b) < 40:
        return None
    pairs = cv2.BFMatcher().knnMatch(desc_a, desc_b, k=2)
    good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.7 * pair[1].distance]
    if len(good) < 40:
        return None
    src = np.float32([kp_a[m.queryIdx].pt for m in good])
    dst = np.float32([kp_b[m.trainIdx].pt for m in good])
    affine, inliers = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0,
                                            maxIters=2000, confidence=0.99, refineIters=10)
    if affine is None or inliers is None:
        return None
    mask = inliers.ravel().astype(bool)
    if int(mask.sum()) < 40 or int(mask.sum()) / len(good) < 0.50:
        return None
    if np.ptp(src[mask, 0]) < source.shape[1] * 0.20 or np.ptp(src[mask, 1]) < source.shape[0] * 0.08:
        return None
    x, y = src[mask].T
    u, v = dst[mask].T
    sx, tx = np.polyfit(x, u, 1)
    sy, ty = np.polyfit(y, v, 1)
    if not np.all(np.isfinite([sx, sy, tx, ty])):
        return None
    residual = np.sqrt((sx*x + tx - u)**2 + (sy*y + ty - v)**2)
    if np.median(residual) > 2.5:
        return None
    return float(sx), float(sy), float(tx), float(ty)


def _image_aspect(path: str) -> float:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        return image.width / image.height


def _fit_clip(
    clip: dict,
    metadata: dict,
    *,
    actual_last_frame: np.ndarray | None = None,
) -> tuple[list[tuple[float, float, float, float]] | None, str, float | None]:
    refs = (clip.get("first_frame"), clip.get("last_frame"))
    if not any(refs):
        return None, "No keyframe anchors; preserved source aspect ratio", None
    if not metadata["fps"] or metadata.get("variable_rate"):
        return None, "Variable or unknown frame rate; correction not safely frame-indexed", None
    try:
        ref_aspects = [_image_aspect(ref) for ref in refs if ref]
        if len(ref_aspects) == 2 and abs(ref_aspects[0] / ref_aspects[1] - 1) > 0.01:
            return None, "Keyframe references have different aspect ratios; correction not safely determined", None
        ref_aspect = ref_aspects[0]
        if abs(metadata["display_aspect"] / ref_aspect - 1) > _MAX_SCALE_DELTA:
            return None, "Source and keyframe aspect ratios differ beyond the bounded correction", None
        fits = []
        for index, ref in enumerate(refs):
            if not ref:
                fits.append(None)
                continue
            if index == 1 and actual_last_frame is not None:
                src = _canonical(cv2.cvtColor(actual_last_frame, cv2.COLOR_RGB2GRAY))
            else:
                seconds = 0 if index == 0 else max(0, metadata["duration"] - 1 / metadata["fps"])
                src = _canonical(_frame(clip["path"], seconds))
            target = _canonical(_image(ref))
            fits.append(_match(src, target))
    except (OSError, ValueError, subprocess.SubprocessError, cv2.error):
        return None, "Could not decode or register supplied keyframe anchor", None
    usable = [fit for fit in fits if fit is not None]
    if len(usable) != len([ref for ref in refs if ref]):
        return None, "Supplied keyframe anchor has insufficient reliable visual matches", None
    if not usable:
        return None, "Supplied keyframe anchors have no reliable visual matches", None
    valid_fits = [fit for fit in fits if fit is not None]
    for sx, sy, tx, ty in valid_fits:
        if abs(sx - 1) > _MAX_SCALE_DELTA or abs(sy - 1) > _MAX_SCALE_DELTA or abs(tx) > _MAX_OFFSET_X or abs(ty) > _MAX_OFFSET_Y:
            return None, "Required geometry exceeds the conservative per-clip correction bounds", None
    if not refs[0] or not refs[1]:
        return None, "Only one keyframe anchor is available; end-to-end correction is unavailable", None
    changed = any(abs(sx - 1) > 0.002 or abs(sy - 1) > 0.002 or abs(tx) > 1.5 or abs(ty) > 1.5
                  for sx, sy, tx, ty in valid_fits)
    return valid_fits, "Keyframe-anchored bounded geometry correction" if changed else "Keyframe geometry already matches", ref_aspect

def _expr(start: float, end: float, frame_count: int) -> str:
    delta = end - start
    last = max(1, frame_count - 1)
    return f"({start:.10f}+({delta:.10f})*if(lte(on,0),0,if(gte(on,{last}),1,pow(on/{last},2)*(3-2*on/{last}))))"


def _perspective(fits: list[tuple[float, float, float, float]], region_w: int, region_h: int,
                 frame_count: int) -> str:
    start, end = fits
    terms = [(sx, sy, tx * region_w / _CANVAS_W, ty * region_h / _CANVAS_H) for sx, sy, tx, ty in (start, end)]
    sx = _expr(terms[0][0], terms[1][0], frame_count)
    sy = _expr(terms[0][1], terms[1][1], frame_count)
    tx = _expr(terms[0][2], terms[1][2], frame_count)
    ty = _expr(terms[0][3], terms[1][3], frame_count)
    coords = [("x0", tx), ("y0", ty),
              ("x1", f"{tx}+{region_w}*{sx}"), ("y1", ty),
              ("x2", tx), ("y2", f"{ty}+{region_h}*{sy}"),
              ("x3", f"{tx}+{region_w}*{sx}"), ("y3", f"{ty}+{region_h}*{sy}")]
    options = ":".join(f"{name}='{expr}'" for name, expr in coords)
    return f"perspective={options}:sense=destination:eval=frame:interpolation=cubic"


def _region_size(aspect: float, width: int, height: int) -> tuple[int, int]:
    if aspect >= width / height:
        region_w = width
        region_h = max(2, round(width / aspect / 2) * 2)
    else:
        region_h = height
        region_w = max(2, round(height * aspect / 2) * 2)
    return region_w, region_h

def _fit_at_time(fits, info, seconds):
    if not fits or len(fits) == 1:
        return fits[0] if fits else None
    frame_count = max(2, round(info["duration"] * info["fps"]))
    progress = min(1.0, max(0.0, seconds * info["fps"] / (frame_count - 1)))
    progress = progress * progress * (3 - 2 * progress)
    return tuple(start + (end - start) * progress for start, end in zip(fits[0], fits[-1]))


def _geometry_filter(fits, info, region_w, region_h, width, height, zoom, *, sample_seconds=None) -> str:
    normalize_display = "scale=trunc(iw*sar/2)*2:ih:flags=bicubic,setsar=1"
    geometric = f"{normalize_display},scale={region_w}:{region_h}:flags=bicubic"
    if fits:
        frame_count = max(2, round(info["duration"] * info["fps"]))
        selected = fits if sample_seconds is None else [_fit_at_time(fits, info, sample_seconds)] * 2
        geometric += f",{_perspective(selected, region_w, region_h, frame_count)}"
    scaled_width = max(width, int(math.ceil(width * zoom / 2) * 2))
    scaled_height = max(height, int(math.ceil(height * zoom / 2) * 2))
    return (geometric + f",pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black"
            f",scale={scaled_width}:{scaled_height}:flags=bicubic,crop={width}:{height}:(iw-ow)/2:(ih-oh)/2")


def _lighting_samples(clip, fits, region, info, width, height, zoom, *, ending):
    """Decode three endpoint samples, applying the same geometry as the output graph."""
    from io import BytesIO
    samples = []
    fps = info["fps"]
    if not fps or info.get("variable_rate"):
        raise ValueError("Lighting sampling requires a known constant frame rate")
    frame_count = round(info["duration"] * fps)
    if frame_count < 3:
        raise ValueError("Lighting sampling requires three distinct frames")
    step = max(1, min(round(0.20 * fps), (frame_count - 1) // 2))
    indices = ([frame_count - 1 - 2 * step, frame_count - 1 - step, frame_count - 1]
               if ending else [0, step, 2 * step])
    for index in indices:
        # Seek just before the desired presentation timestamp, never beyond the
        # final frame due to timestamp rounding or a short sample window.
        seconds = max(0.0, (index - 0.25) / fps)
        sample_time = index / fps
        geometry = _geometry_filter(fits, info, *region, width, height, zoom, sample_seconds=sample_time)
        args = [_executable("VIMAX_FFMPEG_CMD", "ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error",
                "-ss", f"{seconds:.9f}", "-protocol_whitelist", "file,pipe", "-i", clip["path"],
                "-vf", geometry + ",format=rgb24", "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "pipe:1"]
        result = _run(args)
        with Image.open(BytesIO(result.stdout)) as image:
            samples.append(np.asarray(image.convert("RGB")).copy())
    return samples


def _needed_zoom(normalized, width: int, height: int) -> float:
    required = 1.0
    for clip, fits, _, (region_w, region_h) in normalized:
        if not fits:
            continue
        margin_x = (width - region_w) / 2
        margin_y = (height - region_h) / 2
        for sx, sy, tx, ty in fits:
            left = margin_x + tx * region_w / _CANVAS_W
            right = left + region_w * sx
            top = margin_y + ty * region_h / _CANVAS_H
            bottom = top + region_h * sy
            half_w, half_h = region_w / 2, region_h / 2
            required = max(required, half_w / max(1e-9, width/2 - left), half_w / max(1e-9, right - width/2),
                           half_h / max(1e-9, height/2 - top), half_h / max(1e-9, bottom - height/2))
    if any(fits and any(abs(sx-1) > 0.002 or abs(sy-1) > 0.002 or abs(tx) > 1.5 or abs(ty) > 1.5 for sx, sy, tx, ty in fits)
           for _, fits, _, _ in normalized):
        required += 2.0 / min(width, height)
    return required
 
 


def assemble_video_files(clips, output_path, *, width=1280, height=720, fps=30, codec="libx264", preset="medium") -> dict:
    """Assemble clips on a fixed canvas and conservatively correct reliable anchors.

    Output is built beside the requested destination and atomically replaced only
    after FFmpeg succeeds and the completed media probes successfully.
    """
    if not clips:
        raise ValueError("At least one clip is required")
    if width <= 0 or height <= 0 or fps <= 0:
        raise ValueError("width, height and fps must be positive")
    normalized = []
    metadata = []
    statuses = []
    for index, item in enumerate(clips):
        path = os.path.abspath(os.fspath(item["path"]))
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        clip = {**item, "path": path, "slot": str(item.get("slot", index))}
        info = _probe(path)
        metadata.append(info)
        fits, reason, anchor_aspect = _fit_clip(clip, info)
        aspect = anchor_aspect or info["display_aspect"]
        region = _region_size(aspect, width, height)
        if not any((clip.get("first_frame"), clip.get("last_frame"))):
            status = "unchanged"
        elif fits is None and "Only one keyframe anchor" in reason:
            status = "unchanged"
        elif fits is None:
            status = "needs_review"
        elif "already matches" in reason and abs(info["display_aspect"] / aspect - 1) <= 0.001:
            status = "unchanged"
        elif "already matches" in reason:
            status = "corrected"
            reason = "Keyframe baseline aspect correction; affine geometry already matches"
        else:
            status = "corrected"
        statuses.append({"slot": clip["slot"], "status": status, "reason": reason,
                         "duration_seconds": float(info["duration"])})
        normalized.append((clip, fits, aspect, region))
    zoom = _needed_zoom(normalized, width, height)
    if zoom > _MAX_CROP_ZOOM:
        zoom = 1.0
        adjusted = []
        for index, (clip, fits, aspect, region) in enumerate(normalized):
            if fits and statuses[index]["status"] == "corrected":
                statuses[index]["status"] = "needs_review"
                statuses[index]["reason"] = "Common crop required for reliable corrections exceeds the 5% limit"
                fits = None
            adjusted.append((clip, fits, aspect, region))
        normalized = adjusted

    # Lighting is estimated only after normalization/crop decisions are final.
    lighting = [{"status": "unchanged", "reason": "Identity clip establishes the lighting reference",
                 "gains": [1.0, 1.0, 1.0], "previous_slot": None}]
    cumulative_gain = np.ones(3, dtype=np.float64)
    for index in range(1, len(normalized)):
        previous, incoming = normalized[index - 1], normalized[index]
        try:
            before = _lighting_samples(previous[0], previous[1], previous[3], metadata[index - 1],
                                       width, height, zoom, ending=True)
            after = _lighting_samples(incoming[0], incoming[1], incoming[3], metadata[index],
                                      width, height, zoom, ending=False)
            result = estimate_gains(before, after, tuple(cumulative_gain))
        except (OSError, ValueError, subprocess.SubprocessError, cv2.error):
            result = {"status": "needs_review", "reason": "Could not decode endpoint samples for lighting analysis",
                      "gains": [1.0, 1.0, 1.0]}
        result["previous_slot"] = previous[0]["slot"]
        lighting.append(result)
        if result["status"] in ("corrected", "unchanged"):
            cumulative_gain = np.asarray(result["gains"], dtype=np.float64)
        else:
            cumulative_gain = np.ones(3, dtype=np.float64)

    filters = []
    concat = []
    for idx, ((clip, fits, aspect, region), info) in enumerate(zip(normalized, metadata)):
        geometric = _geometry_filter(fits, info, *region, width, height, zoom)
        gains = lighting[idx]["gains"]
        gain_filter = ""
        if any(abs(gain - 1.0) > 1e-9 for gain in gains):
            gain_filter = (f",format=gbrpf32le,colorchannelmixer=rr={gains[0]:.8f}"
                           f":gg={gains[1]:.8f}:bb={gains[2]:.8f}")
        filters.append(f"[{idx}:v:0]setpts=PTS-STARTPTS,{geometric}{gain_filter},setsar=1,fps={fps},format=yuv420p[v{idx}]")
        duration = info["duration"]
        if info["audio"]:
            filters.append(f"[{idx}:a:0]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,apad,atrim=duration={duration:.9f},asetpts=PTS-STARTPTS[a{idx}]")
        else:
            filters.append(f"anullsrc=channel_layout=stereo:sample_rate=48000,atrim=duration={duration:.9f},asetpts=PTS-STARTPTS[a{idx}]")
        concat.append(f"[v{idx}][a{idx}]")
    for index, status in enumerate(statuses):
        status["lighting"] = lighting[index]
    filters.append(f"{''.join(concat)}concat=n={len(clips)}:v=1:a=1[outv][outa]")
    destination = os.path.abspath(os.fspath(output_path))
    os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
    fd, staging = tempfile.mkstemp(prefix=f".{Path(destination).name}.", suffix=".tmp.mp4", dir=os.path.dirname(destination) or ".")
    os.close(fd)
    try:
        args = [_executable("VIMAX_FFMPEG_CMD", "ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error", "-filter_complex_threads", "1", "-y"]
        for clip, _, _, _ in normalized:
            args.extend(["-threads", "1", "-protocol_whitelist", "file,pipe", "-i", clip["path"]])
        args += ["-filter_complex", ";".join(filters), "-map", "[outv]", "-map", "[outa]", "-threads", "2", "-c:v", codec,
                 "-preset", preset, "-crf", "20", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-f", "mp4", staging]
        _run(args, capture=False)
        out_info = _probe(staging)
        if out_info["width"] != width or out_info["height"] != height:
            raise RuntimeError("Rendered output dimensions did not match the requested canvas")
        os.replace(staging, destination)
    except BaseException:
        try: os.unlink(staging)
        except FileNotFoundError: pass
        raise
    return {"version": ASSEMBLY_VERSION, "canvas": {"width": width, "height": height, "fps": fps},
            "crop_zoom": float(zoom), "clips": statuses}

def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_file(value, label: str) -> str:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError(f"{label} must be a local file path")
    path = os.fspath(value)
    if not isinstance(path, str) or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", path):
        raise ValueError(f"{label} must be a plain local file path")
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path


def _rendered_ending_frame(video_path: str) -> np.ndarray:
    """Decode every displayed frame but retain only the final one on disk."""
    with tempfile.TemporaryDirectory(prefix="vimax-rendered-ending-") as directory:
        # Avoid PNG-compressing every intermediate frame; the caller encodes only the ending.
        decoded = os.path.join(directory, "last.ppm")
        args = [
            _executable("VIMAX_FFMPEG_CMD", "ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error",
            "-xerror", "-err_detect", "explode", "-autorotate", "1", "-protocol_whitelist", "file,pipe",
            "-i", video_path, "-map", "0:v:0", "-an", "-sn", "-dn", "-vf",
            "scale=trunc(iw*sar/2)*2:ih:flags=bicubic,setsar=1", "-fps_mode", "passthrough",
            "-threads", "1", "-c:v", "ppm", "-f", "image2", "-update", "1", decoded,
        ]
        try:
            _run(args)
        except subprocess.CalledProcessError as exc:
            raise ValueError(f"Could not decode the final displayed frame of {video_path}") from exc
        if not os.path.isfile(decoded) or os.path.getsize(decoded) == 0:
            raise ValueError(f"Video contains no decodable displayed frames: {video_path}")
        try:
            with Image.open(decoded) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image.load()
                frame = np.asarray(image)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Could not decode the final displayed frame of {video_path}") from exc
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.shape[0] <= 0 or frame.shape[1] <= 0:
            raise ValueError(f"Could not decode the final displayed frame of {video_path}")
        return frame.copy()


def _resize_for_region(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)


def _canonical_rendered_ending(frame: np.ndarray, video_path: str, width: int, height: int,
                               first_frame: str | None, last_frame: str | None) -> np.ndarray:
    source_height, source_width = frame.shape[:2]
    source_aspect = source_width / source_height
    aspect = source_aspect
    fits = None
    if first_frame and last_frame:
        try:
            clip = {"path": video_path, "first_frame": first_frame, "last_frame": last_frame}
            metadata = _probe(video_path)
            fits, _, anchor_aspect = _fit_clip(clip, metadata, actual_last_frame=frame)
            if fits is not None and anchor_aspect is not None:
                region = _region_size(anchor_aspect, width, height)
                if _needed_zoom([(clip, fits, anchor_aspect, region)], width, height) > _MAX_CROP_ZOOM:
                    fits = None
                aspect = anchor_aspect
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError, cv2.error):
            fits = None
            aspect = source_aspect
    region_width, region_height = _region_size(aspect, width, height)
    region_width = min(width, max(1, region_width))
    region_height = min(height, max(1, region_height))
    normalized = _resize_for_region(frame, region_width, region_height)
    if fits is not None:
        sx, sy, tx, ty = fits[-1]
        transform = np.float32([
            [sx, 0, tx * region_width / _CANVAS_W],
            [0, sy, ty * region_height / _CANVAS_H],
        ])
        normalized = cv2.warpAffine(normalized, transform, (region_width, region_height),
                                    flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(0, 0, 0))
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    left = (width - region_width) // 2
    top = (height - region_height) // 2
    canvas[top:top + region_height, left:left + region_width] = normalized
    return canvas


def _cached_rendered_ending(metadata_path: str, image_path: str, expected: dict) -> dict | None:
    try:
        with open(metadata_path, "r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
        if not isinstance(metadata, dict) or any(metadata.get(key) != value for key, value in expected.items()):
            return None
        image_hash = metadata.get("sha256")
        if not isinstance(image_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", image_hash):
            return None
        if _sha256_file(image_path) != image_hash:
            return None
        with Image.open(image_path) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size != (expected["width"], expected["height"]):
                return None
            image.verify()
        return {**expected, "sha256": image_hash}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _atomic_bytes(path: str, contents: bytes) -> str:
    directory = os.path.dirname(path) or "."
    fd, staging = tempfile.mkstemp(prefix=f".{Path(path).name}.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(contents)
            output.flush()
            os.fsync(output.fileno())
        return staging
    except BaseException:
        try:
            os.unlink(staging)
        except FileNotFoundError:
            pass
        raise


def prepare_rendered_ending(video_path, *, first_frame=None, last_frame=None,
                            width=1280, height=720) -> dict:
    """Cache the actual final displayed video frame on the canonical canvas."""
    if (not isinstance(width, int) or isinstance(width, bool) or not isinstance(height, int)
            or isinstance(height, bool) or width <= 0 or height <= 0):
        raise ValueError("width and height must be positive integers")
    source = _local_file(video_path, "video_path")
    first = _local_file(first_frame, "first_frame") if first_frame is not None else None
    last = _local_file(last_frame, "last_frame") if last_frame is not None else None
    source_hash = _sha256_file(source)
    first_hash = _sha256_file(first) if first else None
    last_hash = _sha256_file(last) if last else None
    image_path = source + ".rendered-ending.png"
    metadata_path = source + ".rendered-ending.json"
    expected = {
        "path": image_path,
        "clip_path": source,
        "clip_sha256": source_hash,
        "normalization": RENDERED_ENDING_NORMALIZATION,
        "normalization_inputs": {
            "first_frame_sha256": first_hash,
            "last_frame_sha256": last_hash,
        },
        "width": width,
        "height": height,
    }
    cached = _cached_rendered_ending(metadata_path, image_path, expected)
    if cached is not None and _sha256_file(source) == source_hash:
        if (not first or _sha256_file(first) == first_hash) and (not last or _sha256_file(last) == last_hash):
            return cached

    frame = _rendered_ending_frame(source)
    normalized = _canonical_rendered_ending(frame, source, width, height, first, last)
    if _sha256_file(source) != source_hash:
        raise ValueError(f"Video changed while preparing rendered ending: {source}")
    if (first and _sha256_file(first) != first_hash) or (last and _sha256_file(last) != last_hash):
        raise ValueError("A keyframe anchor changed while preparing rendered ending")
    fd, image_staging = tempfile.mkstemp(prefix=f".{Path(image_path).name}.", suffix=".tmp",
                                         dir=os.path.dirname(image_path) or ".")
    os.close(fd)
    metadata_staging = None
    try:
        with open(image_staging, "wb") as output:
            Image.fromarray(normalized).save(output, format="PNG")
            output.flush()
            os.fsync(output.fileno())
        result = {**expected, "sha256": _sha256_file(image_staging)}
        metadata_staging = _atomic_bytes(
            metadata_path,
            (json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"),
        )
        if _sha256_file(source) != source_hash:
            raise ValueError(f"Video changed while preparing rendered ending: {source}")
        if (first and _sha256_file(first) != first_hash) or (last and _sha256_file(last) != last_hash):
            raise ValueError("A keyframe anchor changed while preparing rendered ending")
        os.replace(image_staging, image_path)
        os.replace(metadata_staging, metadata_path)
        return result
    except BaseException:
        for staging in (image_staging, metadata_staging):
            if staging:
                try:
                    os.unlink(staging)
                except FileNotFoundError:
                    pass
        raise


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output")
    mode.add_argument("--rendered-ending", metavar="VIDEO_PATH")
    args = parser.parse_args()
    payload = json.load(sys.stdin)
    if args.rendered_ending is not None:
        report = prepare_rendered_ending(
            args.rendered_ending,
            **{key: payload[key] for key in ("first_frame", "last_frame", "width", "height") if key in payload},
        )
    else:
        report = assemble_video_files(payload["clips"], args.output, **{key: payload[key] for key in ("width", "height", "fps", "codec", "preset") if key in payload})
    print(json.dumps(report, separators=(",", ":")))


if __name__ == "__main__":
    main()
