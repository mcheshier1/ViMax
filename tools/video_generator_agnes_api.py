"""Agnes AI asynchronous video generation API adapter."""

import asyncio
import os
import math
import random
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import aiohttp

from interfaces.video_output import VideoOutput
from utils.image import image_path_to_b64
from tools.video_duration import resolve_duration_seconds


VIDEO_CREATE_BACKOFF_INITIAL = 3.0
VIDEO_STATUS_INTERVAL = 1.5
VIDEO_STATUS_BACKOFF_MAX = 60.0


class _AgnesCapacityError(RuntimeError):
    def __init__(self, message: str, retry_after: float = 0):
        super().__init__(message)
        self.retry_after = retry_after


class VideoGeneratorAgnesAPI:
    def __init__(self, api_key: str, model: str = "agnes-video-2.5", base_url: str = "https://apihub.agnes-ai.com/v1", clip_seconds: int = 8, resolution: str = "720p"):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._clip_seconds = clip_seconds
        self.resolution = resolution

    @property
    def clip_seconds(self) -> int:
        return self._clip_seconds

    async def generate_single_video(self, prompt: str = "", reference_image_paths=None, aspect_ratio: str = "16:9", **kwargs) -> VideoOutput:
        paths = reference_image_paths or []
        allowed_durations = range(4, 13) if self.model in {"agnes-video-2.5", "agnes-video-2.5-flash"} else None
        duration = resolve_duration_seconds(
            kwargs.get("duration_seconds"),
            self.clip_seconds,
            allowed=allowed_durations,
        )
        payload = {
            "model": self.model,
            "prompt": prompt,
            "mode": "keyframe" if paths else "text",
            "seconds": str(duration),
            "size": _resolution(self.resolution),
            "aspect_ratio": aspect_ratio,
        }
        if paths:
            payload["first_frame"] = image_path_to_b64(paths[0])
            if len(paths) > 1:
                payload["last_frame"] = image_path_to_b64(paths[-1])
        progress = kwargs.get("progress")
        if progress:
            progress("video_create", f"Creating Agnes video with {self.model}", {"model": self.model, "duration": duration, "resolution": payload["size"]})
        request_timeout = float(os.environ.get("VIMAX_VIDEO_REQUEST_TIMEOUT_SECONDS", "60"))
        query_timeout = float(os.environ.get("VIMAX_VIDEO_QUERY_TIMEOUT_SECONDS", "0"))
        timeout = aiohttp.ClientTimeout(total=request_timeout)
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        async with aiohttp.ClientSession(timeout=timeout) as session:
            attempt = 0
            create_backoff = VIDEO_CREATE_BACKOFF_INITIAL
            while True:
                attempt += 1
                try:
                    async with session.post(f"{self.base_url}/videos", json=payload, headers=headers) as response:
                        create_data = await _json_response(response, "Agnes video create")
                    break
                except _AgnesCapacityError as exc:
                    delay = max(random.uniform(create_backoff / 2, create_backoff), exc.retry_after)
                    create_backoff = min(create_backoff * 2, VIDEO_STATUS_BACKOFF_MAX)
                    metadata = {"model": self.model, "attempt": attempt + 1, "retry_delay_seconds": delay}
                    if progress:
                        progress("video_create_retry", f"Agnes queue is full or rate-limited; retrying submission in {delay:g}s (attempt {attempt + 1}, no attempt limit)", metadata)
                    await _wait_with_progress(delay, progress, metadata)
            video_id = create_data.get("video_id")
            if not video_id:
                raise RuntimeError(f"Agnes video create response missing video_id: {create_data}")
            loop = asyncio.get_running_loop()
            deadline = loop.time() + query_timeout if query_timeout > 0 else None
            delay = VIDEO_STATUS_INTERVAL
            backoff = VIDEO_STATUS_INTERVAL
            pending_delay = VIDEO_STATUS_INTERVAL
            while deadline is None or loop.time() < deadline:
                remaining = deadline - loop.time() if deadline is not None else None
                await _wait_with_progress(min(delay, remaining) if remaining is not None else delay, progress, {"model": self.model, "video_id": video_id})
                if deadline is not None and loop.time() >= deadline:
                    break
                remaining = deadline - loop.time() if deadline is not None else None
                status_seconds = request_timeout
                if remaining is not None:
                    status_seconds = min(request_timeout, remaining) if request_timeout > 0 else remaining
                status_timeout = aiohttp.ClientTimeout(total=status_seconds)
                try:
                    async with session.get(f"{self.base_url.rsplit('/v1', 1)[0]}/agnesapi", params={"video_id": video_id, "model_name": self.model}, headers=headers, timeout=status_timeout) as response:
                        if response.status == 429 or response.status >= 500:
                            backoff = min(backoff * 2, VIDEO_STATUS_BACKOFF_MAX)
                            delay = max(random.uniform(backoff / 2, backoff), _retry_after_seconds(response.headers.get("Retry-After")) or 0)
                            if progress:
                                progress("video_status_retry", f"Agnes status check temporarily unavailable (HTTP {response.status}); retrying the same job in {delay:g}s", {"model": self.model, "video_id": video_id, "retry_delay_seconds": delay, "remaining_seconds": max(0, deadline - loop.time()) if deadline is not None else None})
                            continue
                        status = await _json_response(response, "Agnes video status")
                except (aiohttp.ClientConnectionError, aiohttp.ClientPayloadError, asyncio.TimeoutError) as exc:
                    backoff = min(backoff * 2, VIDEO_STATUS_BACKOFF_MAX)
                    delay = random.uniform(backoff / 2, backoff)
                    if progress:
                        progress("video_status_retry", f"Agnes status connection interrupted; retrying the same job in {delay:g}s", {"model": self.model, "video_id": video_id, "retry_delay_seconds": delay, "error": type(exc).__name__})
                    continue
                delay = VIDEO_STATUS_INTERVAL
                backoff = VIDEO_STATUS_INTERVAL
                state = status.get("status")
                if progress:
                    progress("video_status", f"Agnes video generation status: {state}", {"model": self.model, "video_id": video_id, "progress": status.get("progress")})
                if state == "completed":
                    url = status.get("url")
                    if not url:
                        raise RuntimeError(f"Agnes completed response missing video URL: {status}")
                    download_backoff = VIDEO_CREATE_BACKOFF_INITIAL
                    while True:
                        try:
                            async with session.get(url) as response:
                                if response.status == 429 or response.status >= 500:
                                    retry_after = _retry_after_seconds(response.headers.get("Retry-After")) or 0
                                    raise _AgnesCapacityError(f"HTTP {response.status}", retry_after)
                                if response.status >= 400:
                                    raise RuntimeError(f"Agnes video download failed with HTTP {response.status}")
                                data = await response.read()
                            break
                        except (_AgnesCapacityError, aiohttp.ClientConnectionError, aiohttp.ClientPayloadError, asyncio.TimeoutError) as exc:
                            delay = max(random.uniform(download_backoff / 2, download_backoff), getattr(exc, "retry_after", 0))
                            download_backoff = min(download_backoff * 2, VIDEO_STATUS_BACKOFF_MAX)
                            metadata = {"model": self.model, "video_id": video_id, "retry_delay_seconds": delay}
                            if progress:
                                progress("video_download_retry", f"Agnes result download interrupted; retrying the completed job's download in {delay:g}s", metadata)
                            await _wait_with_progress(delay, progress, metadata)
                    if progress:
                        progress("video_completed", "Agnes video generation completed", {"model": self.model, "video_id": video_id})
                    return VideoOutput(fmt="bytes", ext="mp4", data=data)
                if state in {"failed", "cancelled", "expired"}:
                    raise RuntimeError(f"Agnes video generation {state}: {status.get('error') or status}")
                pending_delay = min(pending_delay * 2, VIDEO_STATUS_BACKOFF_MAX)
                delay = pending_delay
        raise RuntimeError(f"Agnes video generation timed out after {query_timeout:g}s for video {video_id}")

async def _wait_with_progress(delay: float, progress, metadata: dict) -> None:
    while delay > 0:
        interval = min(delay, VIDEO_STATUS_BACKOFF_MAX)
        await asyncio.sleep(interval)
        delay -= interval
        if delay > 0 and progress:
            progress("video_wait", f"Still waiting for Agnes; next request in {delay:g}s", {**metadata, "remaining_wait_seconds": delay})



def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            seconds = (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
        except (ValueError, TypeError, OverflowError):
            return None
    return max(0, seconds) if math.isfinite(seconds) else None


async def _json_response(response, operation: str) -> dict:
    if response.status == 429:
        raise _AgnesCapacityError(f"{operation} rate-limited (HTTP 429)", _retry_after_seconds(response.headers.get("Retry-After")) or 0)
    try:
        payload = await response.json(content_type=None)
    except (aiohttp.ClientConnectionError, aiohttp.ClientPayloadError, asyncio.TimeoutError):
        raise
    except Exception as error:
        text = await response.text()
        raise RuntimeError(f"{operation} returned invalid JSON: {text[:1000]}") from error
    if response.status >= 400:
        message = f"{operation} failed with HTTP {response.status}: {payload}"
        if _is_capacity_503(response.status, payload):
            raise _AgnesCapacityError(message, _retry_after_seconds(response.headers.get("Retry-After")) or 0)
        raise RuntimeError(message)
    return payload


def _is_capacity_503(status: int, payload) -> bool:
    if status != 503 or not isinstance(payload, dict):
        return False
    details = [str(payload.get("code", "")), str(payload.get("message", ""))]
    error = payload.get("error")
    if isinstance(error, dict):
        details.extend(str(error.get(key, "")) for key in ("code", "message", "type", "detail"))
    else:
        details.append(str(error or ""))
    normalized = " ".join(details).lower().replace("_", " ").replace("-", " ")
    return any(marker in normalized for marker in ("capacity", "queue full", "queue is full", "overloaded"))


def _resolution(value: str) -> str:
    normalized = value.strip().upper()
    if normalized in {"720P", "1080P", "1K", "2K"}:
        return normalized
    raise ValueError(f"Unsupported Agnes resolution: {value}; use 720p, 1080p, 1K, or 2K")
