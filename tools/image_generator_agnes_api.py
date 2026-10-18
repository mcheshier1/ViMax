from __future__ import annotations

import base64
import os
from io import BytesIO
from typing import Any
from urllib.parse import urlparse

import aiohttp
from PIL import Image

from interfaces.image_output import ImageOutput
from tools.image_orientation import ensure_not_portrait, landscape_guard_requested
from utils.image import image_path_to_b64
from utils.rate_limiter import RateLimiter


class AgnesImageAPIError(RuntimeError):
    def __init__(self, status_code: int, payload: Any, *, model: str) -> None:
        self.status_code = status_code
        self.payload = payload
        self.model = model
        super().__init__(f"Agnes image API returned HTTP {status_code} for {model}: {payload}")


def _request_timeout_seconds() -> float:
    raw = os.environ.get("VIMAX_IMAGE_REQUEST_TIMEOUT_SECONDS", "300")
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 300.0


class ImageGeneratorAgnesAPI:
    """Generate images with Agnes AI's synchronous image endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str = "agnes-image-2.5-flash",
        base_url: str = "https://apihub.agnes-ai.com/v1",
        rate_limiter: RateLimiter | None = None,
        resolution: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.rate_limiter = rate_limiter
        self.resolution = resolution

    async def generate_single_image(
        self,
        prompt: str,
        reference_image_paths: list[str] | None = None,
        aspect_ratio: str = "16:9",
        **kwargs: Any,
    ) -> ImageOutput:
        if self.rate_limiter is not None:
            await self.rate_limiter.acquire()

        enforce_landscape = landscape_guard_requested(
            size=kwargs.get("size"),
            aspect_ratio=aspect_ratio,
            enforce_landscape=kwargs.get("enforce_landscape", True),
            allow_portrait=kwargs.get("allow_portrait", False),
        )
        request_prompt = prompt
        if enforce_landscape:
            ratio = aspect_ratio or "16:9"
            request_prompt = f"{prompt}\n\nComposition requirement: create a landscape image with an approximate {ratio} aspect ratio; the width must be greater than the height."
        size = kwargs.get("size")
        resolution = kwargs.get("resolution", self.resolution)
        if not size and resolution not in (None, "", "auto"):
            if self.model == "agnes-image-2.0-flash":
                raise ValueError(f"{self.model} does not support explicit image resolution {resolution}")
            tiers = {"1K": "1K", "2K": "2K", "3K": "3K", "4K": "4K"}
            try:
                size = tiers[resolution.upper()]
            except (AttributeError, KeyError):
                raise ValueError(f"Unsupported Agnes image resolution: {resolution}") from None
        if not size:
            size = "1024x768" if self.model == "agnes-image-2.0-flash" else "1K"
        body: dict[str, Any] = {"model": self.model, "prompt": request_prompt, "size": size}
        if self.model != "agnes-image-2.0-flash":
            body["ratio"] = aspect_ratio or "16:9"
        references = list(reference_image_paths or [])
        if references:
            body["extra_body"] = {"image": [self._reference_data(path) for path in references], "response_format": "b64_json"}
        else:
            body["return_base64"] = True

        timeout_seconds = _request_timeout_seconds()
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{self.base_url}/images/generations",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=body,
            ) as response:
                try:
                    payload = await response.json(content_type=None)
                except (ValueError, aiohttp.ContentTypeError):
                    payload = await response.text()
                if response.status >= 400:
                    raise AgnesImageAPIError(response.status, payload, model=self.model)

        image, extension = await self._decode_response(payload, timeout)
        if enforce_landscape:
            ensure_not_portrait(image)
        return ImageOutput(fmt="pil", ext=extension, data=image)

    @staticmethod
    def _reference_data(reference: str) -> str:
        if reference.startswith("data:") or urlparse(reference).scheme in {"http", "https"}:
            return reference
        return image_path_to_b64(reference, mime=True)

    async def _decode_response(self, payload: Any, timeout: aiohttp.ClientTimeout) -> tuple[Image.Image, str]:
        data = payload.get("data") if isinstance(payload, dict) else None
        item = data[0] if isinstance(data, list) and data and isinstance(data[0], dict) else None
        if item is None:
            raise ValueError("Agnes image response missing data[0]")
        encoded = item.get("b64_json")
        if isinstance(encoded, str) and encoded:
            if encoded.startswith("data:") and "," in encoded:
                encoded = encoded.split(",", 1)[1]
            try:
                raw = base64.b64decode(encoded, validate=True)
                with Image.open(BytesIO(raw)) as opened:
                    opened.load()
                    image = opened.copy()
            except Exception as exc:
                raise ValueError("Agnes image response contained invalid image data") from exc
            return image, _extension(item.get("mime_type") or item.get("media_type") or "image/png")

        url = item.get("url")
        if not isinstance(url, str) or not url:
            raise ValueError("Agnes image response missing data[0].b64_json or data[0].url")
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                response.raise_for_status()
                raw = await response.read()
        try:
            with Image.open(BytesIO(raw)) as opened:
                opened.load()
                image = opened.copy()
        except Exception as exc:
            raise ValueError("Agnes image URL returned invalid image data") from exc
        return image, _extension(response.headers.get("Content-Type", "image/png").split(";", 1)[0])


def _extension(media_type: str) -> str:
    return {"image/jpeg": "jpg", "image/webp": "webp"}.get(media_type.lower(), "png")
