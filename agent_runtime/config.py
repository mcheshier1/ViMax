from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

DEFAULT_LLM_MODEL = "gpt-5.5"
DEFAULT_LLM_MODEL_PROVIDER = "openai"
DEFAULT_LLM_BASE_URL = "https://yunwu.ai/v1"
DEFAULT_IMAGE_MODEL = "gemini-3.1-flash-image-preview"
DEFAULT_IMAGE_BASE_URL = "https://yunwu.ai"
DEFAULT_VIDEO_MODEL = "veo3.1-fast"
DEFAULT_VIDEO_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_VIDEO_PROVIDER = "openrouter"
DEFAULT_VIDEO_RESOLUTION = "720p"
DEFAULT_VIDEO_GENERATE_AUDIO = True
# Eight seconds is the shortest length that carries a spoken line without rushing it.
DEFAULT_VIDEO_CLIP_SECONDS = 8
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_MODEL_PROVIDER = "openai"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


@lru_cache(maxsize=4)
def load_agent_config(workspace_root: str | Path = ".") -> dict[str, Any]:
    path = Path(workspace_root).resolve() / "configs" / "agent.local.yaml"
    if not path.exists():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Invalid configs/agent.local.yaml: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("configs/agent.local.yaml must be a YAML mapping")
    return payload


def config_value(section: str, key: str, env_names: list[str], default: str = "", workspace_root: str | Path = ".") -> str:
    for env_name in env_names:
        value = os.environ.get(env_name)
        if value:
            return value
    section_payload = load_agent_config(workspace_root).get(section, {})
    if isinstance(section_payload, dict):
        value = section_payload.get(key)
        if isinstance(value, str) and value:
            return value
    return default


def _credential_provider(section: dict[str, Any]) -> str:
    base_url = str(section.get("base_url") or "").strip()
    try:
        parsed = urlsplit(base_url)
        hostname = (parsed.hostname or "").lower()
        if parsed.scheme.lower() in {"http", "https"} and hostname:
            port = parsed.port
            for domain, provider in (
                ("openrouter.ai", "openrouter"), ("agnes-ai.com", "agnes"),
                ("yunwu.ai", "yunwu"), ("ltx.io", "ltx"), ("openai.com", "openai"),
                ("anthropic.com", "anthropic"), ("cohere.com", "cohere"),
                ("cohere.ai", "cohere"), ("jina.ai", "jina"),
            ):
                if hostname == domain or hostname.endswith("." + domain):
                    return provider
            host = f"[{hostname}]" if ":" in hostname else hostname
            if port is not None and port != (443 if parsed.scheme.lower() == "https" else 80):
                host += f":{port}"
            return f"endpoint:{parsed.scheme.lower()}://{host}{parsed.path.rstrip('/')}"
    except ValueError:
        pass
    model_provider = str(section.get("model_provider") or "").strip().lower()
    provider = str(section.get("provider") or "").strip().lower()
    return f"provider:{model_provider or provider or 'default'}"


def _scoped_api_key(section: str, workspace_root: str | Path) -> str | None:
    """Resolve migrated provider credentials without replaying another provider's key."""
    source = load_agent_config(workspace_root).get(section, {})
    if not isinstance(source, dict) or "api_keys" not in source:
        return None
    keys = source["api_keys"]
    if not isinstance(keys, dict):
        raise RuntimeError(f"{section}.api_keys must be a mapping")
    env_names = [f"VIMAX_{section.upper()}_API_KEY"]
    if section == "llm":
        env_names.append("VIMAX_API_KEY")
    for name in env_names:
        if os.environ.get(name):
            return os.environ[name]
    selected = dict(source)
    if os.environ.get(f"VIMAX_{section.upper()}_BASE_URL"):
        selected["base_url"] = os.environ[f"VIMAX_{section.upper()}_BASE_URL"]
    if os.environ.get(f"VIMAX_{section.upper()}_MODEL_PROVIDER"):
        selected["model_provider"] = os.environ[f"VIMAX_{section.upper()}_MODEL_PROVIDER"]
    value = keys.get(_credential_provider(selected))
    if isinstance(value, str) and value.strip():
        return value.strip()
    if section in {"image", "video"}:
        # Preserve shared-LLM-key setups only when both effective endpoints use
        # the same provider. An OpenRouter key must not fall through to Agnes.
        base_url = image_base_url(workspace_root) if section == "image" else video_base_url(workspace_root)
        if _credential_provider({"base_url": base_url}) == _credential_provider({"base_url": llm_base_url(workspace_root)}):
            return llm_api_key(workspace_root)
    return ""


def llm_model(workspace_root: str | Path = ".") -> str:
    return config_value("llm", "model", ["VIMAX_LLM_MODEL"], DEFAULT_LLM_MODEL, workspace_root)


def llm_model_provider(workspace_root: str | Path = ".") -> str:
    return config_value("llm", "model_provider", ["VIMAX_LLM_MODEL_PROVIDER"], DEFAULT_LLM_MODEL_PROVIDER, workspace_root)


def llm_base_url(workspace_root: str | Path = ".") -> str:
    return config_value("llm", "base_url", ["VIMAX_LLM_BASE_URL"], DEFAULT_LLM_BASE_URL, workspace_root)


def llm_api_key(workspace_root: str | Path = ".") -> str:
    scoped = _scoped_api_key("llm", workspace_root)
    return scoped if scoped is not None else config_value("llm", "api_key", ["VIMAX_LLM_API_KEY", "VIMAX_API_KEY"], "", workspace_root)


def image_model(workspace_root: str | Path = ".") -> str:
    return config_value("image", "model", ["VIMAX_IMAGE_MODEL"], DEFAULT_IMAGE_MODEL, workspace_root)


def image_base_url(workspace_root: str | Path = ".") -> str:
    return config_value("image", "base_url", ["VIMAX_IMAGE_BASE_URL"], DEFAULT_IMAGE_BASE_URL, workspace_root)


def image_resolution(workspace_root: str | Path = ".") -> str:
    configured = config_value("image", "resolution", ["VIMAX_IMAGE_RESOLUTION"], "", workspace_root).strip()
    if configured:
        return configured
    return "1K" if image_model(workspace_root) == "bytedance-seed/seedream-5-0-flash" else "auto"


def image_api_key(workspace_root: str | Path = ".") -> str:
    scoped = _scoped_api_key("image", workspace_root)
    return scoped if scoped is not None else config_value("image", "api_key", ["VIMAX_IMAGE_API_KEY", "VIMAX_LLM_API_KEY", "VIMAX_API_KEY"], llm_api_key(workspace_root), workspace_root)



def embedding_model(workspace_root: str | Path = ".") -> str:
    return config_value("embedding", "model", ["VIMAX_EMBEDDING_MODEL"], DEFAULT_EMBEDDING_MODEL, workspace_root)


def embedding_model_provider(workspace_root: str | Path = ".") -> str:
    return config_value("embedding", "model_provider", ["VIMAX_EMBEDDING_MODEL_PROVIDER"], DEFAULT_EMBEDDING_MODEL_PROVIDER, workspace_root)


def embedding_base_url(workspace_root: str | Path = ".") -> str:
    return config_value("embedding", "base_url", ["VIMAX_EMBEDDING_BASE_URL"], "", workspace_root)


def embedding_api_key(workspace_root: str | Path = ".") -> str:
    scoped = _scoped_api_key("embedding", workspace_root)
    return scoped if scoped is not None else config_value("embedding", "api_key", ["VIMAX_EMBEDDING_API_KEY"], "", workspace_root)


def reranker_model(workspace_root: str | Path = ".") -> str:
    return config_value("reranker", "model", ["VIMAX_RERANKER_MODEL"], DEFAULT_RERANKER_MODEL, workspace_root)


def reranker_base_url(workspace_root: str | Path = ".") -> str:
    return config_value("reranker", "base_url", ["VIMAX_RERANKER_BASE_URL"], "", workspace_root)


def reranker_api_key(workspace_root: str | Path = ".") -> str:
    scoped = _scoped_api_key("reranker", workspace_root)
    return scoped if scoped is not None else config_value("reranker", "api_key", ["VIMAX_RERANKER_API_KEY"], "", workspace_root)


def video_model(workspace_root: str | Path = ".") -> str:
    return config_value("video", "model", ["VIMAX_VIDEO_MODEL"], DEFAULT_VIDEO_MODEL, workspace_root)


def video_clip_seconds(workspace_root: str | Path = ".") -> int:
    """Seconds each clip renders as, which is what the dialogue has to fit in.

    The configured value wins over the environment here, unlike the model settings: a clip
    length belongs to the film being made, while a duration exported into a shell or a
    supervisor's environment cannot be seen from the project at all. One left set to 5 by a
    process that started this workspace cut every clip of a sequence whose dialogue needed
    eight seconds, and nothing on any screen said why.
    """
    section_payload = load_agent_config(workspace_root).get("video", {})
    raw: Any = section_payload.get("clip_seconds") if isinstance(section_payload, dict) else None
    if raw in (None, ""):
        raw = os.environ.get("VIMAX_OPENROUTER_VIDEO_DURATION", "")
    if raw in (None, ""):
        return DEFAULT_VIDEO_CLIP_SECONDS
    try:
        return max(1, int(float(raw)))
    except (TypeError, ValueError):
        return DEFAULT_VIDEO_CLIP_SECONDS


@lru_cache(maxsize=1)
def _video_duration_models() -> dict[str, list[int]]:
    """Load the shared provider/model duration catalog."""
    path = Path(__file__).resolve().parents[1] / "configs" / "video-duration-models.json"
    try:
        payload = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RuntimeError(f"Video duration catalog {path} is missing") from exc
    try:
        catalog = json.loads(payload)
    except ValueError as exc:
        raise RuntimeError(f"Invalid video duration catalog {path}: {exc}") from exc
    if not isinstance(catalog, dict):
        raise RuntimeError(f"Video duration catalog {path} must be a JSON object")
    normalized: dict[str, list[int]] = {}
    for key, values in catalog.items():
        if (
            not isinstance(key, str)
            or not isinstance(values, list)
            or not values
            or any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in values)
            or values != sorted(set(values))
        ):
            raise RuntimeError(f"Invalid duration entry for {key!r} in {path}")
        normalized[key] = values
    return normalized


def video_duration_policy(workspace_root: str | Path = ".") -> dict[str, Any]:
    """Resolve the active model's default and explicit shot-duration bounds."""
    provider = video_provider(workspace_root)
    model = video_model(workspace_root)
    key = f"{provider}:{model}"
    catalog = _video_duration_models()
    supported = [8] if provider == "yunwu" else catalog.get(key)
    if provider == "ltx" and model.endswith("-fast") and supported:
        resolution = video_resolution(workspace_root).strip().lower().replace(" ", "")
        if resolution in {"1440p", "4k", "2560x1440", "3840x2160"}:
            supported = [seconds for seconds in supported if seconds <= 10]
    section = load_agent_config(workspace_root).get("video", {})
    section = section if isinstance(section, dict) else {}
    limits = section.get("shot_duration_limits", {})
    if not isinstance(limits, dict):
        raise ValueError("video.shot_duration_limits must be a mapping")
    configured = key in limits
    configured_max = limits.get(key)
    if configured and (
        not isinstance(configured_max, int)
        or isinstance(configured_max, bool)
        or configured_max < 1
    ):
        raise ValueError(f"video.shot_duration_limits[{key!r}] must be a positive whole number")
    if provider == "yunwu" and configured and configured_max != 8:
        raise ValueError(f"{key} supports fixed 8-second clips only")
    if supported and configured and configured_max not in supported:
        raise ValueError(
            f"Configured shot duration maximum {configured_max}s is unsupported by {key}; "
            f"supported durations: {supported}"
        )

    allowed = [seconds for seconds in supported if seconds <= configured_max] if supported and configured else supported
    if allowed == []:
        raise ValueError(f"Configured shot duration maximum for {key} is below its minimum supported duration")
    saved_default = section.get("clip_seconds")
    environment_default = os.environ.get("VIMAX_OPENROUTER_VIDEO_DURATION")
    explicit_default = saved_default not in (None, "") or environment_default not in (None, "")
    if saved_default not in (None, ""):
        if isinstance(saved_default, bool) or not isinstance(saved_default, (int, str)):
            raise ValueError("video.clip_seconds must be a positive whole number")
        try:
            parsed_default = float(saved_default)
        except ValueError as exc:
            raise ValueError("video.clip_seconds must be a positive whole number") from exc
        if not parsed_default.is_integer() or parsed_default < 1:
            raise ValueError("video.clip_seconds must be a positive whole number")
        raw_default = int(parsed_default)
    elif environment_default not in (None, ""):
        try:
            parsed_default = float(environment_default)
        except ValueError as exc:
            raise ValueError("VIMAX_OPENROUTER_VIDEO_DURATION must be a positive whole number") from exc
        if not parsed_default.is_integer() or parsed_default < 1:
            raise ValueError("VIMAX_OPENROUTER_VIDEO_DURATION must be a positive whole number")
        raw_default = int(parsed_default)
    else:
        raw_default = DEFAULT_VIDEO_CLIP_SECONDS
    if supported and raw_default not in supported and explicit_default:
        raise ValueError(
            f"Configured default duration {raw_default}s is unsupported by {key}; "
            f"supported durations: {supported}"
        )
    if provider == "yunwu":
        default_seconds = 8
    elif supported and raw_default not in supported:
        default_seconds = 8 if 8 in supported else supported[-1]
    else:
        default_seconds = raw_default
    if configured and default_seconds > configured_max:
        raise ValueError(
            f"Configured maximum {configured_max}s is below the default duration {default_seconds}s for {key}"
        )

    return {
        "provider": provider,
        "model": model,
        "configured": configured,
        "defaultSeconds": default_seconds,
        "minSeconds": min(allowed) if allowed else 1,
        "maxSeconds": configured_max if configured else (max(allowed) if allowed else 0),
        "allowedSeconds": allowed,
    }


def validate_video_shot_duration(seconds: Any, policy: dict[str, Any]) -> str:
    """Return a validation error for an explicit shot override, or an empty string."""
    if not isinstance(seconds, int) or isinstance(seconds, bool) or seconds < 1:
        return "Shot duration must be a positive whole number of seconds."
    if not policy.get("configured"):
        return f"Configure a maximum duration for {policy.get('provider')}:{policy.get('model')} in video settings before setting a shot duration."
    allowed = policy.get("allowedSeconds")
    if isinstance(allowed, list):
        if seconds not in allowed:
            return (
                f"{seconds}s is not supported by {policy.get('provider')}:{policy.get('model')} "
                f"with the configured maximum; allowed durations: {allowed}."
            )
    elif seconds > int(policy.get("maxSeconds") or 0):
        return f"{seconds}s exceeds the configured maximum of {policy.get('maxSeconds')}s."
    if seconds < int(policy.get("minSeconds") or 1):
        return f"{seconds}s is below the minimum supported duration of {policy.get('minSeconds')}s."
    return ""


def video_base_url(workspace_root: str | Path = ".") -> str:
    return config_value("video", "base_url", ["VIMAX_VIDEO_BASE_URL"], DEFAULT_VIDEO_BASE_URL, workspace_root)


def video_api_key(workspace_root: str | Path = ".") -> str:
    scoped = _scoped_api_key("video", workspace_root)
    return scoped if scoped is not None else config_value("video", "api_key", ["VIMAX_VIDEO_API_KEY", "VIMAX_LLM_API_KEY", "VIMAX_API_KEY"], llm_api_key(workspace_root), workspace_root)


def video_resolution(workspace_root: str | Path = ".") -> str:
    return config_value("video", "resolution", ["VIMAX_OPENROUTER_VIDEO_RESOLUTION"], DEFAULT_VIDEO_RESOLUTION, workspace_root)


def video_generate_audio(workspace_root: str | Path = ".") -> bool:
    value = load_agent_config(workspace_root).get("video", {})
    if isinstance(value, dict):
        configured = value.get("generate_audio")
        if isinstance(configured, bool):
            return configured
        if isinstance(configured, str) and configured.strip():
            return configured.strip().lower() in {"1", "true", "yes", "on"}
    raw = os.environ.get("VIMAX_OPENROUTER_GENERATE_AUDIO")
    if raw is not None:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return DEFAULT_VIDEO_GENERATE_AUDIO


def api_provider_from_base_url(base_url: str) -> str:
    normalized = base_url.strip().lower()
    if "openrouter.ai" in normalized:
        return "openrouter"
    if "yunwu.ai" in normalized:
        return "yunwu"
    if "agnes-ai.com" in normalized:
        return "agnes"
    if "ltx.io" in normalized:
        return "ltx"
    return ""


def video_provider(workspace_root: str | Path = ".") -> str:
    """Choose a video API adapter from explicit provider or configured base URL."""
    value = load_agent_config(workspace_root).get("video", {})
    configured = value.get("provider", "") if isinstance(value, dict) else ""
    return str(configured or api_provider_from_base_url(video_base_url(workspace_root))).strip().lower()


