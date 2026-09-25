from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from utils.text import safe_path_component

# Render phases, in execution order. Each phase is idempotent: artifacts that
# already exist are reused, so re-running a phase to revise it is cheap.
RENDER_PHASES = ("portraits", "stills", "video")

# Cheapest phase, and the one that needs a style decision before anything else is
# rendered from it. Callers opt in to spending more by asking for a later phase.
DEFAULT_RENDER_PHASE = "portraits"


def normalize_phase(value: str | None, default: str = DEFAULT_RENDER_PHASE) -> str:
    """Validate a requested phase, falling back to ``default`` when unspecified.

    Pipelines default to the cheapest phase so a caller that forgets to opt in
    cannot spend money on video generation by accident; callers that mean to render
    the whole sequence pass ``"video"`` explicitly.
    """
    phase = str(value or default).strip().lower()
    if phase not in RENDER_PHASES:
        raise ValueError(f"stop_after must be one of {list(RENDER_PHASES)}, got {value!r}")
    return phase


@dataclass(slots=True)
class RenderOutcome:
    """What a pipeline run produced, and what the caller should review next.

    Pipelines stop at the requested phase and report the artifacts the next phase
    would consume, so an agent can show them to the user and ask before spending
    money on video generation.
    """

    phase: str
    style: str = ""
    image_model: str = ""
    video_model: str = ""
    stills: List[str] = field(default_factory=list)
    awaiting_confirmation: str = ""
    final_video_path: str = ""

    def as_dict(self) -> dict:
        return {
            "phase": self.phase,
            "style": self.style,
            "image_model": self.image_model,
            "video_model": self.video_model,
            "stills": list(self.stills),
            "awaiting_confirmation": self.awaiting_confirmation,
            "final_video_path": self.final_video_path,
        }


class ModelScopedArtifacts:
    """Artifact paths grouped by the model that produced them.

    A sequence rendered by two different image models looks stitched together, so
    every generated artifact lives under the slug of the model that produced it.
    Prompts and structured text are inputs rather than model output and stay
    outside those directories.

    Requires ``working_dir``, ``image_generator`` and ``video_generator`` on the host
    class, plus a ``style`` argument to ``render_outcome``.
    """

    working_dir: str
    image_generator: Any
    video_generator: Any

    @property
    def image_model_slug(self) -> str:
        return safe_path_component(str(getattr(self.image_generator, "model", "") or "unknown-image-model"))

    @property
    def video_model_slug(self) -> str:
        return safe_path_component(str(getattr(self.video_generator, "model", "") or "unknown-video-model"))

    def portraits_dir(self) -> str:
        """Portraits grouped by the image model that drew them."""
        return os.path.join(self.working_dir, "character_portraits", self.image_model_slug)

    def portraits_registry_path(self) -> str:
        return os.path.join(self.portraits_dir(), "registry.json")

    def shot_image_dir(self, shot_idx: int) -> str:
        return os.path.join(self.working_dir, "shots", f"{shot_idx}", self.image_model_slug)

    def shot_video_dir(self, shot_idx: int) -> str:
        return os.path.join(self.working_dir, "shots", f"{shot_idx}", self.video_model_slug)

    def frame_path(self, shot_idx: int, frame_type: str) -> str:
        return os.path.join(self.shot_image_dir(shot_idx), f"{frame_type}.png")

    def selector_output_path(self, shot_idx: int, frame_type: str) -> str:
        """Prompts are inputs, not model output, so they stay out of the model directory."""
        return os.path.join(self.working_dir, "shots", f"{shot_idx}", f"{frame_type}_selector_output.json")

    def load_selector_output(self, shot_idx: int, frame_type: str) -> Optional[Dict[str, Any]]:
        """Cached reference selection for a frame, or ``None`` when it must be recomputed.

        A cached selection embeds absolute paths to the portraits and frames it
        references. Those artifacts are style- and model-scoped, so changing the
        style (or the image model) deletes the files a stale cache points at, and
        reusing it fails inside the image generator with a missing file. A cache
        whose references have gone is therefore discarded here, which makes the
        render rebuild the selection instead of dying on it.
        """
        path = self.selector_output_path(shot_idx, frame_type)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                cached = json.load(handle)
        except (OSError, ValueError):
            return None
        if not isinstance(cached, dict):
            return None
        references = cached.get("reference_image_path_and_text_pairs")
        if not isinstance(references, list):
            return None
        for reference in references:
            reference_path = reference[0] if isinstance(reference, (list, tuple)) and reference else None
            if not isinstance(reference_path, str) or not os.path.exists(reference_path):
                return None
        return cached

    def clip_path(self, shot_idx: int) -> str:
        return os.path.join(self.shot_video_dir(shot_idx), "video.mp4")

    def render_outcome(self, phase: str, style: str, *, stills: Optional[List[str]] = None, awaiting: str = "", final_video_path: str = "") -> RenderOutcome:
        return RenderOutcome(
            phase=phase,
            style=style,
            image_model=str(getattr(self.image_generator, "model", "") or ""),
            video_model=str(getattr(self.video_generator, "model", "") or ""),
            stills=list(stills or []),
            awaiting_confirmation=awaiting,
            final_video_path=final_video_path,
        )

    def portrait_stills(self, character_portraits_registry: Optional[Dict[str, Dict[str, Dict[str, str]]]]) -> List[str]:
        """Portrait images the user reviews to confirm the rendered style."""
        stills: List[str] = []
        for views in (character_portraits_registry or {}).values():
            for item in (views or {}).values():
                path = str((item or {}).get("path") or "")
                if path and os.path.exists(path):
                    stills.append(path)
        return stills