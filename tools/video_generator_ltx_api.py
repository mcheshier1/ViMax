"""LTX synchronous video generation API adapter."""

import os

import aiohttp

from interfaces.video_output import VideoOutput
from utils.image import image_path_to_b64


class VideoGeneratorLTXAPI:
    def __init__(self, api_key: str, model: str = "ltx-2-5-pro", base_url: str = "https://api.ltx.io", clip_seconds: int = 8, resolution: str = "720p", generate_audio: bool = True):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._clip_seconds = clip_seconds
        self.resolution = resolution
        self.generate_audio = generate_audio

    @property
    def clip_seconds(self) -> int:
        return self._clip_seconds

    async def generate_single_video(self, prompt: str = "", reference_image_paths=None, aspect_ratio: str = "16:9", **kwargs) -> VideoOutput:
        paths = reference_image_paths or []
        endpoint = "image-to-video" if paths else "text-to-video"
        payload = {
            "prompt": prompt,
            "model": self.model,
            "duration": self.clip_seconds,
            "resolution": _resolution(self.resolution, aspect_ratio),
            "generate_audio": self.generate_audio,
        }
        if paths:
            payload["image_uri"] = image_path_to_b64(paths[0])
            if len(paths) > 1:
                payload["last_frame_uri"] = image_path_to_b64(paths[-1])
        progress = kwargs.get("progress")
        if progress:
            progress("video_create", f"Creating LTX video with {self.model}", {"model": self.model, "duration": self.clip_seconds, "resolution": payload["resolution"]})
        timeout_seconds = float(os.environ.get("VIMAX_VIDEO_REQUEST_TIMEOUT_SECONDS", "600"))
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{self.base_url}/v1/{endpoint}", json=payload, headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}) as response:
                data = await response.read()
                if response.status >= 400:
                    raise RuntimeError(f"LTX video generation failed with HTTP {response.status}: {data[:1000].decode('utf-8', errors='replace')}")
                content_type = response.headers.get("Content-Type", "")
                if "video/" not in content_type and not data.startswith(b"\x00\x00\x00"):
                    raise RuntimeError(f"LTX video response was not an MP4 (Content-Type: {content_type})")
        if progress:
            progress("video_completed", "LTX video generation completed", {"model": self.model})
        return VideoOutput(fmt="bytes", ext="mp4", data=data)


def _resolution(value: str, aspect_ratio: str) -> str:
    value = value.strip().lower()
    if "x" in value:
        return value
    sizes = {"720p": (1280, 720), "1080p": (1920, 1080), "1440p": (2560, 1440), "4k": (3840, 2160)}
    if value not in sizes:
        raise ValueError(f"Unsupported LTX resolution: {value}")
    width, height = sizes[value]
    if aspect_ratio in {"9:16", "3:4"}:
        return f"{height}x{width}"
    if aspect_ratio == "1:1":
        side = min(width, height)
        return f"{side}x{side}"
    return f"{width}x{height}"
