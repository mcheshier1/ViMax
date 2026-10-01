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


MAX_VIDEO_CREATE_ATTEMPTS = 3
VIDEO_CREATE_RETRY_DELAYS = (2, 4)
VIDEO_STATUS_INTERVAL = 1.5
VIDEO_STATUS_BACKOFF_MAX = 60.0


class _AgnesCapacityError(RuntimeError):
    pass


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
        payload = {
            "model": self.model,
            "prompt": prompt,
            "mode": "keyframe" if paths else "text",
            "seconds": str(self.clip_seconds),
            "size": _resolution(self.resolution),
            "aspect_ratio": aspect_ratio,
        }
        if paths:
            payload["first_frame"] = image_path_to_b64(paths[0])
            if len(paths) > 1:
                payload["last_frame"] = image_path_to_b64(paths[-1])
        progress = kwargs.get("progress")
        if progress:
            progress("video_create", f"Creating Agnes video with {self.model}", {"model": self.model, "duration": self.clip_seconds, "resolution": payload["size"]})
        request_timeout = float(os.environ.get("VIMAX_VIDEO_REQUEST_TIMEOUT_SECONDS", "60"))
        query_timeout = float(os.environ.get("VIMAX_VIDEO_QUERY_TIMEOUT_SECONDS", "600"))
        timeout = aiohttp.ClientTimeout(total=request_timeout)
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for attempt in range(1, MAX_VIDEO_CREATE_ATTEMPTS + 1):
                try:
                    async with session.post(f"{self.base_url}/videos", json=payload, headers=headers) as response:
                        create_data = await _json_response(response, "Agnes video create")
                    break
                except _AgnesCapacityError:
                    if attempt == MAX_VIDEO_CREATE_ATTEMPTS:
                        raise
                    delay = VIDEO_CREATE_RETRY_DELAYS[attempt - 1]
                    if progress:
                        progress(
                            "video_create_retry",
                            f"Agnes is at capacity; retrying video creation ({attempt + 1}/{MAX_VIDEO_CREATE_ATTEMPTS})",
                            {"model": self.model, "attempt": attempt + 1, "max_attempts": MAX_VIDEO_CREATE_ATTEMPTS, "retry_delay_seconds": delay},
                        )
                    await asyncio.sleep(delay)
            video_id = create_data.get("video_id")
            if not video_id:
                raise RuntimeError(f"Agnes video create response missing video_id: {create_data}")
            loop = asyncio.get_running_loop()
            deadline = loop.time() + query_timeout if query_timeout > 0 else None
            delay = VIDEO_STATUS_INTERVAL
            backoff = VIDEO_STATUS_INTERVAL
            while deadline is None or loop.time() < deadline:
                remaining = deadline - loop.time() if deadline is not None else None
                await asyncio.sleep(min(delay, remaining) if remaining is not None else delay)
                if deadline is not None and loop.time() >= deadline:
                    break
                remaining = deadline - loop.time() if deadline is not None else None
                status_seconds = request_timeout
                if remaining is not None:
                    status_seconds = min(request_timeout, remaining) if request_timeout > 0 else remaining
                status_timeout = aiohttp.ClientTimeout(total=status_seconds)
                async with session.get(f"{self.base_url.rsplit('/v1', 1)[0]}/agnesapi", params={"video_id": video_id, "model_name": self.model}, headers=headers, timeout=status_timeout) as response:
                    if response.status == 429:
                        backoff = min(backoff * 2, VIDEO_STATUS_BACKOFF_MAX)
                        delay = max(random.uniform(backoff / 2, backoff), _retry_after_seconds(response.headers.get("Retry-After")) or 0)
                        if progress:
                            remaining = max(0, deadline - loop.time()) if deadline is not None else None
                            message = f"Agnes status queries rate-limited (HTTP 429); next check in {delay:g}s"
                            if remaining is not None and delay >= remaining:
                                message = "Agnes status queries rate-limited (HTTP 429); waiting until the query timeout, with no further check"
                            progress("video_status_retry", message, {"model": self.model, "video_id": video_id, "retry_delay_seconds": delay, "remaining_seconds": remaining})
                        continue
                    status = await _json_response(response, "Agnes video status")
                delay = VIDEO_STATUS_INTERVAL
                backoff = VIDEO_STATUS_INTERVAL
                state = status.get("status")
                if progress:
                    progress("video_status", f"Agnes video generation status: {state}", {"model": self.model, "video_id": video_id, "progress": status.get("progress")})
                if state == "completed":
                    url = status.get("url")
                    if not url:
                        raise RuntimeError(f"Agnes completed response missing video URL: {status}")
                    async with session.get(url) as response:
                        data = await response.read()
                        if response.status >= 400:
                            raise RuntimeError(f"Agnes video download failed with HTTP {response.status}: {data[:500]!r}")
                    if progress:
                        progress("video_completed", "Agnes video generation completed", {"model": self.model, "video_id": video_id})
                    return VideoOutput(fmt="bytes", ext="mp4", data=data)
                if state in {"failed", "cancelled", "expired"}:
                    raise RuntimeError(f"Agnes video generation {state}: {status.get('error') or status}")
        raise RuntimeError(f"Agnes video generation timed out after {query_timeout:g}s for video {video_id}")


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
    try:
        payload = await response.json(content_type=None)
    except Exception as error:
        text = await response.text()
        raise RuntimeError(f"{operation} returned invalid JSON: {text[:1000]}") from error
    if response.status >= 400:
        message = f"{operation} failed with HTTP {response.status}: {payload}"
        if _is_capacity_503(response.status, payload):
            raise _AgnesCapacityError(message)
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
    return any(marker in normalized for marker in ("capacity", "queue full", "overloaded"))


def _resolution(value: str) -> str:
    normalized = value.strip().upper()
    if normalized in {"720P", "1080P", "1K", "2K"}:
        return normalized
    raise ValueError(f"Unsupported Agnes resolution: {value}; use 720p, 1080p, 1K, or 2K")
