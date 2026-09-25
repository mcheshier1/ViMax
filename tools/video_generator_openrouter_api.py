import asyncio
import logging
import os
from typing import Any, List
from urllib.parse import urljoin

import aiohttp

from interfaces.video_output import VideoOutput
from utils.image import image_path_to_b64


def _env_int(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.environ.get(name, str(default))))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _emit_progress(progress, stage: str, message: str, metadata: dict | None = None) -> None:
    if progress is not None:
        progress(stage, message, metadata or {})


class VideoGeneratorOpenRouterAPI:
    def __init__(
        self,
        api_key: str,
        model: str = "google/veo-3.1-lite",
        base_url: str = "https://openrouter.ai/api/v1",
        http_referer: str = "",
        app_title: str = "ViMax",
        clip_seconds: int | None = None,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.http_referer = http_referer
        self.app_title = app_title
        self._clip_seconds = clip_seconds

    @property
    def clip_seconds(self) -> int:
        """Seconds each clip renders as, which is what sets the runtime of a plan."""
        if self._clip_seconds is not None:
            return self._clip_seconds
        return _env_int("VIMAX_OPENROUTER_VIDEO_DURATION", 8)

    async def supports_last_frame(self) -> bool:
        """Whether this model accepts an end keyframe, so a clip can be bracketed."""
        capabilities = await _model_capabilities(
            self.base_url,
            self.model,
            headers=self._headers(),
            timeout=aiohttp.ClientTimeout(total=60),
            hard_timeout_seconds=60,
        )
        return "last_frame" in ((capabilities or {}).get("supported_frame_images") or [])

    async def generate_single_video(
        self,
        prompt: str = "",
        reference_image_paths: List[str] = [],
        aspect_ratio: str = "16:9",
        **kwargs,
    ) -> VideoOutput:
        progress = kwargs.get("progress")
        request_timeout_seconds = _env_float("VIMAX_VIDEO_REQUEST_TIMEOUT_SECONDS", 60.0)
        query_timeout_seconds = _env_float("VIMAX_VIDEO_QUERY_TIMEOUT_SECONDS", 600.0)
        poll_interval_seconds = _env_float("VIMAX_VIDEO_POLL_INTERVAL_SECONDS", 10.0)
        duration = self.clip_seconds
        resolution = os.environ.get("VIMAX_OPENROUTER_VIDEO_RESOLUTION", "720p")
        generate_audio = _env_bool("VIMAX_OPENROUTER_GENERATE_AUDIO", True)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "duration": duration,
            "resolution": resolution,
            "generate_audio": generate_audio,
        }
        frame_images = _frame_images(reference_image_paths)
        if frame_images:
            payload["frame_images"] = frame_images

        headers = self._headers()
        timeout = aiohttp.ClientTimeout(total=request_timeout_seconds)
        capabilities = await _model_capabilities(
            self.base_url,
            self.model,
            headers=headers,
            timeout=timeout,
            hard_timeout_seconds=request_timeout_seconds,
        )
        trimmed = _enforce_capabilities(payload, capabilities, model=self.model)
        if trimmed:
            _emit_progress(progress, "video_options_trimmed", " ".join(trimmed), {"model": self.model, "trimmed": trimmed})
        sent_frame_count = len(payload.get("frame_images") or [])
        _emit_progress(progress, "video_create", f"Creating OpenRouter video generation task with {self.model}", {"model": self.model, "duration": duration, "resolution": resolution, "frame_count": sent_frame_count})

        create_status, create_payload = await _post_json(
            f"{self.base_url}/videos",
            headers=headers,
            payload=payload,
            timeout=timeout,
            hard_timeout_seconds=request_timeout_seconds,
        )
        if create_status >= 400:
            raise RuntimeError(f"OpenRouter video create failed with HTTP {create_status}: {create_payload}")
        job_id = create_payload.get("id")
        polling_url = create_payload.get("polling_url")
        if not job_id or not polling_url:
            raise RuntimeError(f"OpenRouter video create response missing id or polling_url: {create_payload}")
        _emit_progress(progress, "video_task_created", "OpenRouter video generation task created", {"model": self.model, "job_id": job_id, "status": create_payload.get("status")})

        poll_url = _absolute_url(self.base_url, polling_url)
        deadline = asyncio.get_running_loop().time() + query_timeout_seconds if query_timeout_seconds > 0 else None
        last_status = create_payload.get("status")
        last_payload = create_payload
        while deadline is None or asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(poll_interval_seconds)
            # Asking about a running job is idempotent, so a blip here is waited out rather
            # than thrown: failing the render would abandon a video that is still being
            # generated (and paid for), and retrying the whole call would pay for a second.
            poll_status, poll_payload = await _poll_with_retries(
                poll_url, headers=headers, timeout=timeout, hard_timeout_seconds=request_timeout_seconds,
            )
            if poll_status >= 400:
                raise RuntimeError(f"OpenRouter video poll failed with HTTP {poll_status}: {poll_payload}")
            last_payload = poll_payload
            status = poll_payload.get("status")
            last_status = status
            _emit_progress(progress, "video_status", f"OpenRouter video generation status: {status}", {"model": self.model, "job_id": job_id, "status": status})

            if status == "completed":
                urls = poll_payload.get("unsigned_urls") or []
                if urls:
                    content_url = urls[0]
                else:
                    content_url = f"{self.base_url}/videos/{job_id}/content?index=0"
                _emit_progress(progress, "video_download_start", "Downloading OpenRouter video output", {"model": self.model, "job_id": job_id})
                download_status, data = await _get_bytes(
                    content_url,
                    headers=headers if _needs_authorization(content_url) else {},
                    timeout=timeout,
                    hard_timeout_seconds=request_timeout_seconds,
                )
                if download_status >= 400:
                    raise RuntimeError(f"OpenRouter video content download failed with HTTP {download_status}: {data[:500]!r}")
                _emit_progress(progress, "video_completed", "OpenRouter video generation completed and downloaded", {"model": self.model, "job_id": job_id})
                return VideoOutput(fmt="bytes", ext="mp4", data=data)
            if status in {"failed", "cancelled", "expired"}:
                raise RuntimeError(f"OpenRouter video generation {status} for job {job_id}: {poll_payload.get('error') or poll_payload}")

        raise RuntimeError(f"OpenRouter video generation timed out after {query_timeout_seconds:g}s for job {job_id}; last_status={last_status}; last_payload={last_payload}")

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.app_title:
            headers["X-OpenRouter-Title"] = self.app_title
        return headers


async def _poll_with_retries(url: str, *, headers: dict, timeout, hard_timeout_seconds: float, attempts: int = 3):
    """Read a job's status, tolerating a transient failure.

    A polling request asks a question; asking it again costs nothing and cannot start a
    second job, so a single 5xx or dropped connection is not allowed to abandon one.
    """
    last_exc: BaseException | None = None
    for attempt in range(attempts):
        try:
            status, payload = await _get_json(url, headers=headers, timeout=timeout, hard_timeout_seconds=hard_timeout_seconds)
            if status < 500:
                return status, payload
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            last_exc = exc
        if attempt + 1 < attempts:
            await asyncio.sleep(2 * (attempt + 1))
    if last_exc is not None:
        raise last_exc
    return status, payload


def _frame_images(reference_image_paths: List[str]) -> list[dict]:
    if len(reference_image_paths) > 2:
        raise ValueError("OpenRouter video generation supports at most first and last frame images")
    frame_types = ["first_frame", "last_frame"]
    return [
        {
            "type": "image_url",
            "image_url": {"url": image_path_to_b64(path, mime=True)},
            "frame_type": frame_types[index],
        }
        for index, path in enumerate(reference_image_paths)
    ]


_MODEL_CAPABILITIES: dict[tuple[str, str], dict[str, Any] | None] = {}

# Request fields validated against the catalogue. Values outside the advertised
# set are a caller configuration error, so they raise instead of being rewritten.
_VALIDATED_FIELDS = (
    ("duration", "supported_durations"),
    ("resolution", "supported_resolutions"),
    ("aspect_ratio", "supported_aspect_ratios"),
)


async def _model_capabilities(
    base_url: str,
    model: str,
    *,
    headers: dict[str, str],
    timeout: aiohttp.ClientTimeout,
    hard_timeout_seconds: float,
) -> dict[str, Any] | None:
    """Cached catalogue entry describing what the model's integration accepts.

    OpenRouter serves this from ``GET /videos/models`` and validates create
    requests against it, so a request that ignores it fails with an opaque 400
    (e.g. a ``last_frame`` the provider integration does not implement). Returns
    ``None`` when the catalogue cannot be read, leaving the request unvalidated
    rather than guessing.
    """
    cache_key = (base_url, model)
    if cache_key in _MODEL_CAPABILITIES:
        return _MODEL_CAPABILITIES[cache_key]
    try:
        status, payload = await _get_json(
            f"{base_url}/videos/models",
            headers=headers,
            timeout=timeout,
            hard_timeout_seconds=hard_timeout_seconds,
        )
    except Exception:
        # Transient failure: leave the cache empty so the next clip retries.
        return None
    if status >= 400 or not isinstance(payload, dict):
        return None
    entry = next((item for item in payload.get("data") or [] if isinstance(item, dict) and item.get("id") == model), None)
    _MODEL_CAPABILITIES[cache_key] = entry
    return entry


def _enforce_capabilities(payload: dict[str, Any], capabilities: dict[str, Any] | None, *, model: str) -> list[str]:
    """Drop request options the model's provider integration rejects.

    Returns human-readable notes for everything that was dropped so the caller can
    surface the degradation instead of silently changing the output.
    """
    if capabilities is None:
        return []
    trimmed: list[str] = []

    supported_frames = capabilities.get("supported_frame_images")
    allowed_frames = {str(frame) for frame in supported_frames} if isinstance(supported_frames, list) else set()
    requested_frames = payload.get("frame_images") or []
    if requested_frames:
        kept = [frame for frame in requested_frames if frame.get("frame_type") in allowed_frames]
        dropped = [str(frame.get("frame_type")) for frame in requested_frames if frame.get("frame_type") not in allowed_frames]
        if kept:
            payload["frame_images"] = kept
        else:
            payload.pop("frame_images", None)
        if dropped:
            supported = sorted(allowed_frames) or ["none"]
            if kept:
                kept_types = [str(frame.get("frame_type")) for frame in kept]
                trimmed.append(f"{model} does not support frame_images {dropped} (supported: {supported}); sending {kept_types} only.")
            else:
                trimmed.append(f"{model} does not support frame_images (supported: {supported}); generating from the prompt only.")

    if "generate_audio" in payload and not capabilities.get("generate_audio"):
        payload.pop("generate_audio")
        trimmed.append(f"{model} does not advertise audio output; letting the provider default apply.")

    for field, support_key in _VALIDATED_FIELDS:
        allowed_values = capabilities.get(support_key)
        requested = payload.get(field)
        if requested is None or not isinstance(allowed_values, list) or not allowed_values:
            continue
        if requested not in allowed_values:
            raise ValueError(f"{model} does not support {field}={requested!r}; supported values: {allowed_values}")

    return trimmed


def _absolute_url(base_url: str, url: str) -> str:
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return urljoin(f"{base_url.rstrip('/')}/", url.lstrip("/"))


def _needs_authorization(url: str) -> bool:
    return url.startswith("https://openrouter.ai/api/")


async def _post_json(url: str, *, headers: dict[str, str], payload: dict, timeout: aiohttp.ClientTimeout, hard_timeout_seconds: float) -> tuple[int, dict]:
    async def request() -> tuple[int, dict]:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=payload) as response:
                return response.status, await response.json(content_type=None)

    return await asyncio.wait_for(request(), timeout=hard_timeout_seconds + 5)


async def _get_json(url: str, *, headers: dict[str, str], timeout: aiohttp.ClientTimeout, hard_timeout_seconds: float) -> tuple[int, dict]:
    async def request() -> tuple[int, dict]:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=headers) as response:
                return response.status, await response.json(content_type=None)

    return await asyncio.wait_for(request(), timeout=hard_timeout_seconds + 5)


async def _get_bytes(url: str, *, headers: dict[str, str], timeout: aiohttp.ClientTimeout, hard_timeout_seconds: float) -> tuple[int, bytes]:
    async def request() -> tuple[int, bytes]:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=headers) as response:
                return response.status, await response.read()

    return await asyncio.wait_for(request(), timeout=hard_timeout_seconds + 5)
