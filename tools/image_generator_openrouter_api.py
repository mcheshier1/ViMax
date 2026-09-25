from __future__ import annotations

import asyncio
import base64
import json
import os
import re
from io import BytesIO
from typing import Any, List

import aiohttp
from PIL import Image
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from interfaces.image_output import ImageOutput
from tools.image_orientation import ensure_not_portrait, landscape_guard_requested
from utils.image import image_path_to_b64
from utils.rate_limiter import RateLimiter
from utils.retry import after_func


class OpenRouterImageAPIError(RuntimeError):
    def __init__(self, status_code: int, payload: Any, *, model: str = "") -> None:
        self.status_code = status_code
        self.payload = payload
        self.model = model
        detail = f"OpenRouter image generation with model {model} failed with HTTP {status_code}: {payload}" if model else f"OpenRouter image generation failed with HTTP {status_code}: {payload}"
        super().__init__(detail)


def _request_timeout_seconds() -> float:
    raw = os.environ.get("VIMAX_IMAGE_REQUEST_TIMEOUT_SECONDS", "300")
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 300.0


def _provider_fault(payload: Any) -> str:
    """The provider OpenRouter names when its upstream, not the request, refused to serve it."""
    if not isinstance(payload, dict):
        return ""
    error = payload.get("error")
    if not isinstance(error, dict):
        return ""
    metadata = error.get("metadata")
    if not isinstance(metadata, dict):
        return ""
    return str(metadata.get("provider_name") or "")


def _is_retryable_image_error(exc: BaseException) -> bool:
    if isinstance(exc, OpenRouterImageAPIError):
        if exc.status_code in {408, 409, 425, 429} or exc.status_code >= 500:
            return True
        # A 400 is usually the request being wrong, and retrying only asks the same wrong
        # question again. But when the body names the provider that failed, OpenRouter took
        # the request and the provider would not serve it: Alibaba answers qwen-image-3 with
        # "InternalError.Algo: Invalid request" for a prompt and set of references that
        # succeed on the very next attempt, so giving up on the first costs a whole redraw
        # to save a minute. An image call has no side effect, so the retry is free.
        return bool(_provider_fault(exc.payload))
    if isinstance(exc, (aiohttp.ClientError, asyncio.TimeoutError)):
        return True
    return isinstance(exc, ValueError) and "portrait-oriented" in str(exc)


class ImageGeneratorOpenRouterAPI:
    """Generate images through OpenRouter's dedicated Images API."""

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-image-2",
        base_url: str = "https://openrouter.ai/api/v1",
        quality: str = "auto",
        background: str = "auto",
        output_compression: int | None = None,
        rate_limiter: RateLimiter | None = None,
        http_referer: str = "",
        app_title: str = "ViMax",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.quality = quality
        self.background = background
        self.output_compression = output_compression
        self.rate_limiter = rate_limiter
        self.http_referer = http_referer
        self.app_title = app_title

    async def prepare(self) -> None:
        """Warm the model catalogue so the reference limit is known before references are built.

        The pipeline reads `reference_limit` when it assembles a frame's references, which
        happens before any image is requested. Without this the catalogue is only fetched by
        the first request, so the first frame of every run is assembled as if the model took
        any number of references — and one over the limit fails, as a refusal or as an upstream
        502 that carries nothing to learn from.
        """
        await _model_capabilities(
            self.base_url,
            self.model,
            headers=self._headers(),
            timeout=aiohttp.ClientTimeout(total=_request_timeout_seconds()),
            hard_timeout_seconds=_request_timeout_seconds(),
        )

    @property
    def reference_limit(self) -> int | None:
        """Reference images this model accepts, as far as that is known yet.

        Answered from the cached catalogue entry: unknown before ``prepare`` runs, in which
        case the caller is unbounded rather than wrongly capped.
        """
        return reference_limit_for(self.base_url, self.model, _MODEL_CAPABILITIES.get((self.base_url, self.model)))

    @retry(
        # An image call has no side effect, so retrying it is free, and the upstream
        # providers behind OpenRouter fail in bursts that outlast a ten second backoff:
        # giving up early costs a whole redraw (minutes and money) to avoid a minute of
        # waiting. Four attempts, doubling to a minute apart.
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        retry=retry_if_exception(_is_retryable_image_error),
        after=after_func,
        reraise=True,
    )
    async def generate_single_image(
        self,
        prompt: str,
        reference_image_paths: List[str] | None = None,
        aspect_ratio: str | None = "16:9",
        **kwargs: Any,
    ) -> ImageOutput:
        references = list(reference_image_paths or [])
        if self.rate_limiter is not None:
            await self.rate_limiter.acquire()

        enforce_landscape = landscape_guard_requested(
            size=kwargs.get("size"),
            aspect_ratio=aspect_ratio,
            enforce_landscape=kwargs.get("enforce_landscape", True),
            allow_portrait=kwargs.get("allow_portrait", False),
        )
        request_prompt = _prompt_with_landscape_requirement(prompt, aspect_ratio) if enforce_landscape else prompt
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": request_prompt,
            "n": 1,
        }
        headers = self._headers()
        timeout = aiohttp.ClientTimeout(total=_request_timeout_seconds())
        capabilities = await _model_capabilities(
            self.base_url,
            self.model,
            headers=headers,
            timeout=timeout,
            hard_timeout_seconds=_request_timeout_seconds(),
        )
        if capabilities is None:
            # No catalogue: keep sending the OpenAI-only parameters this API documents.
            if self.model.startswith("openai/"):
                payload["quality"] = kwargs.get("quality", self.quality)
                payload["background"] = kwargs.get("background", self.background)
        else:
            payload.update(_supported_options(capabilities, self, kwargs, aspect_ratio))
        compression = kwargs.get("output_compression", self.output_compression)
        if compression is not None and (capabilities is None or _supports(capabilities, "output_compression")):
            payload["output_compression"] = compression
        offered = len(references)
        if references:
            # Trim before sending: a provider refused for too many references
            # names the limit it enforces, and one it refused for is remembered.
            references = _trim_references(references, reference_limit_for(self.base_url, self.model, capabilities))
            if references:
                payload["input_references"] = _reference_payload(references)

        progress = kwargs.get("progress")
        _emit_progress(
            progress,
            "image_generation",
            f"Generating image with {self.model}",
            {"model": self.model, "reference_count": len(references)},
        )
        if offered and not references and capabilities is not None:
            # A model that takes no references draws the frame from the prompt alone: the
            # characters and the camera still are gone, and every frame invents them again.
            # That is a different film from the one the plan describes, so say so.
            _emit_progress(
                progress,
                "references_dropped",
                f"{self.model} takes no reference images, so this frame was drawn without them — the characters and the framing come from the prompt alone.",
                {"model": self.model, "reference_count": 0, "offered_references": offered},
            )
        status, response = await _post_json(
            f"{self.base_url}/images",
            headers=headers,
            payload=payload,
            timeout=timeout,
        )
        if status >= 400:
            learned = _learn_reference_limit(self.base_url, self.model, response)
            if learned is not None and len(references) > learned:
                # The model accepts fewer references than the catalogue promised.
                # Retry within the limit it named rather than failing the frame.
                references = _trim_references(references, learned)
                payload["input_references"] = _reference_payload(references)
                _emit_progress(
                    progress,
                    "image_reference_limit",
                    f"{self.model} accepts {learned} reference images; retrying with {len(references)}",
                    {"model": self.model, "reference_count": len(references), "reference_limit": learned},
                )
                status, response = await _post_json(
                    f"{self.base_url}/images",
                    headers=headers,
                    payload=payload,
                    timeout=timeout,
                )
        if status >= 400:
            raise OpenRouterImageAPIError(status, response, model=self.model)

        image, extension = _decode_image_response(response)
        if enforce_landscape:
            ensure_not_portrait(image)
        _emit_progress(
            progress,
            "image_completed",
            "OpenRouter image generation completed",
            {"model": self.model, "width": image.width, "height": image.height},
        )
        return ImageOutput(fmt="pil", ext=extension, data=image)

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


def _prompt_with_landscape_requirement(prompt: str, aspect_ratio: str | None) -> str:
    ratio = aspect_ratio or "16:9"
    return f"{prompt}\n\nComposition requirement: create a landscape image with an approximate {ratio} aspect ratio; the width must be greater than the height."


_MODEL_CAPABILITIES: dict[tuple[str, str], dict[str, Any] | None] = {}


async def _model_capabilities(
    base_url: str,
    model: str,
    *,
    headers: dict[str, str],
    timeout: aiohttp.ClientTimeout,
    hard_timeout_seconds: float,
) -> dict[str, Any] | None:
    """Cached catalogue entry describing which request options the model accepts.

    OpenRouter serves this from ``GET /images/models``. Models differ widely —
    ``meta/muse-image`` advertises no optional parameters at all and rejects
    anything beyond the base request, while others cap reference images at 3 or 4
    — so the payload has to follow the catalogue instead of per-provider guesses.
    Returns ``None`` when it cannot be read, leaving the caller's defaults in place.
    """
    cache_key = (base_url, model)
    if cache_key in _MODEL_CAPABILITIES:
        return _MODEL_CAPABILITIES[cache_key]
    try:
        status, payload = await _get_json(
            f"{base_url}/images/models",
            headers=headers,
            timeout=timeout,
            hard_timeout_seconds=hard_timeout_seconds,
        )
    except Exception:
        # Transient failure: leave the cache empty so the next request retries.
        return None
    if status >= 400 or not isinstance(payload, dict):
        return None
    entry = next((item for item in payload.get("data") or [] if isinstance(item, dict) and item.get("id") == model), None)
    _MODEL_CAPABILITIES[cache_key] = entry
    return entry


def _supported_parameters(capabilities: dict[str, Any]) -> dict[str, Any]:
    parameters = capabilities.get("supported_parameters")
    return parameters if isinstance(parameters, dict) else {}


def _supports(capabilities: dict[str, Any], name: str) -> bool:
    return name in _supported_parameters(capabilities)


def _enum_accepts(spec: Any, value: str | None) -> bool:
    if value is None or not isinstance(spec, dict):
        return False
    values = spec.get("values")
    return isinstance(values, list) and value in values


def _supported_options(capabilities: dict[str, Any], generator: Any, kwargs: dict[str, Any], aspect_ratio: str | None) -> dict[str, Any]:
    """The optional request fields this model advertises, without the rest.

    Value-level: a parameter is only sent when the catalogue lists the exact value.
    xAI's image endpoint, for instance, advertises ``quality`` but rejects ``auto``
    ("Accepted: low, medium"), and an unlisted value makes the request unroutable.
    """
    parameters = _supported_parameters(capabilities)
    options: dict[str, Any] = {}
    for name, value in (
        ("quality", kwargs.get("quality", generator.quality)),
        ("background", kwargs.get("background", generator.background)),
    ):
        if _enum_accepts(parameters.get(name), value):
            options[name] = value
    if _enum_accepts(parameters.get("aspect_ratio"), aspect_ratio):
        # Without this some providers return a square canvas and ignore the
        # landscape wording in the prompt.
        options["aspect_ratio"] = aspect_ratio
    return options


# A provider that refuses a request names the limit it enforces, e.g. "Model
# 'qwen-image-3.0-distill' supports 0~3 image content items. Got 4 image items."
_PROVIDER_REFERENCE_LIMIT = re.compile(r"supports\s+\d+\s*[~-]\s*(\d+)\s+image content items", re.IGNORECASE)

# The catalogue's `input_references.max` is an upper bound, not a promise: Qwen's
# entry advertises 4 while the model rejects more than 3. A revealed limit is
# remembered per model and enforced from then on.
_LEARNED_REFERENCE_LIMITS: dict[tuple[str, str], int] = {}


def _advertised_reference_limit(capabilities: dict[str, Any] | None) -> int | None:
    if capabilities is None:
        return None
    spec = _supported_parameters(capabilities).get("input_references")
    if isinstance(spec, dict) and isinstance(spec.get("max"), int):
        return spec["max"]
    return None


def reference_limit_for(base_url: str, model: str, capabilities: dict[str, Any] | None = None) -> int | None:
    """How many reference images this model accepts, as far as that is known.

    The narrower of the advertised limit and any limit a provider refusal
    revealed. ``None`` means unbounded or not known yet, in which case callers
    keep every reference and the generator trims if the provider says otherwise.
    """
    if capabilities is None:
        capabilities = _MODEL_CAPABILITIES.get((base_url, model))
    limits = [
        limit
        for limit in (_LEARNED_REFERENCE_LIMITS.get((base_url, model)), _advertised_reference_limit(capabilities))
        if isinstance(limit, int)
    ]
    return min(limits) if limits else None


def _learn_reference_limit(base_url: str, model: str, payload: Any) -> int | None:
    """Remember the limit a refusal revealed, so the next request is accepted."""
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    match = _PROVIDER_REFERENCE_LIMIT.search(text)
    if not match:
        return None
    limit = int(match.group(1))
    _LEARNED_REFERENCE_LIMITS[(base_url, model)] = limit
    return limit


def _trim_references(references: List[str], limit: int | None) -> List[str]:
    """Keep only the references the model accepts.

    Reference positions address the prompt's ``Image N`` mentions, so the excess
    is dropped from the end: every kept image keeps the index it was written with.
    """
    if limit is None or len(references) <= limit:
        return references
    return references[: max(limit, 0)]


def _reference_payload(references: List[str]) -> List[dict[str, Any]]:
    return [{"type": "image_url", "image_url": {"url": image_path_to_b64(path, mime=True)}} for path in references]


def _decode_image_response(payload: Any) -> tuple[Image.Image, str]:
    data = payload.get("data") if isinstance(payload, dict) else None
    item = data[0] if isinstance(data, list) and data and isinstance(data[0], dict) else None
    encoded = item.get("b64_json") if item else None
    if not isinstance(encoded, str) or not encoded:
        raise ValueError(f"OpenRouter image response missing data[0].b64_json: {payload}")
    if encoded.startswith("data:"):
        encoded = encoded.split(",", 1)[-1]
    try:
        raw = base64.b64decode(encoded, validate=True)
        with Image.open(BytesIO(raw)) as opened:
            opened.load()
            image = opened.copy()
    except Exception as exc:
        raise ValueError("OpenRouter image response contained invalid image data") from exc
    media_type = item.get("media_type", "image/png")
    extension = {"image/jpeg": "jpg", "image/webp": "webp"}.get(media_type, "png")
    return image, extension


def _emit_progress(progress: Any, stage: str, message: str, metadata: dict[str, Any]) -> None:
    if progress is not None:
        progress(stage, message, metadata)


async def _post_json(
    url: str,
    *,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: aiohttp.ClientTimeout,
) -> tuple[int, Any]:
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, headers=headers, json=payload) as response:
            text = await response.text()
            try:
                body = json.loads(text)
            except json.JSONDecodeError:
                body = {"message": text}
            return response.status, body


async def _get_json(
    url: str,
    *,
    headers: dict[str, str],
    timeout: aiohttp.ClientTimeout,
    hard_timeout_seconds: float,
) -> tuple[int, Any]:
    async def request() -> tuple[int, Any]:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=headers) as response:
                text = await response.text()
                try:
                    body = json.loads(text)
                except json.JSONDecodeError:
                    body = {"message": text}
                return response.status, body

    return await asyncio.wait_for(request(), timeout=hard_timeout_seconds + 5)
