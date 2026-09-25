"""Script coverage: does the timeline still tell the script's story?

The timeline is just a list of shots. Nothing checked it against the script, so
shots can be cut or redrawn until a dialogue line, a character, or a whole beat
is gone and the film still renders happily.

Everything here is pure: paths and rows are passed in, nothing is read from
disk. The adapter assembles the evidence, calls the model for the narrative
pass, and persists the result; the web layer recomputes the deterministic
checks live so the review card is useful before any review has ever run.
"""

from __future__ import annotations

import datetime
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Artifact name, beside render_manifest.json / render_acceptance.json.
REVIEW_FILENAME = "continuity_review.json"

BEAT_STATUSES = ("covered", "partial", "missing")
CHECK_STATUSES = ("ok", "warn")

# A script line is "SPEAKER (stage): \"line\"", "SPEAKER narrates. \"line\"" or a
# line followed by a <stage direction>. All three forms occur in real scripts and
# a naive SPEAKER: "..." pattern silently drops the other two.
_SPEAKER_LINE = re.compile(
    r'^(?P<speaker>[A-Z][A-Za-z]*)\b(?P<pre>[^"]*?)[":\s]*"(?P<line>[^"]+)"\s*(?:<[^>]*>)?\s*$'
)
# "[Speaker] Claude (Arrogantly): \"...\"" and "[Speaker] Wife: \"...\"" both occur —
# the stage direction is optional, and a removed shot's brief often omits it.
_AUDIO_PREFIX = re.compile(r"^\[(?:speaker|sound effect)\]\s*", re.IGNORECASE)
_AUDIO_ATTRIBUTION = re.compile(r"^[A-Za-z .'\-]+(?:\((?:[^)]*)\))?\s*:\s*")
_CHARACTER_TAG = re.compile(r"<([A-Za-z0-9_]+)>")
_RUNTIME_RANGE = re.compile(r"(\d+)\s*(?:-|to|–)\s*(\d+)\s*(?:second|sec|s)\b", re.IGNORECASE)

# Below this a script line is too short to match by containment without matching
# incidentally ("I do" inside some other sentence), so it must match exactly.
MIN_MATCHABLE_LINE = 8


def _line_covered(needle: str, audio: str) -> bool:
    """Whether a normalized script line is the line a shot speaks."""
    if not needle or not audio:
        return False
    return needle in audio if len(needle) >= MIN_MATCHABLE_LINE else needle == audio


def normalize_dialogue(value: str) -> str:
    """Reduce a script line or an ``audio_desc`` to comparable words."""
    text = _AUDIO_PREFIX.sub("", str(value or "").strip())
    text = _AUDIO_ATTRIBUTION.sub("", text)
    text = text.strip().strip("\"'\u201c\u201d ")
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


def script_dialogue(script: str) -> List["DialogueLine"]:
    """Every spoken line in the script, in order. Stage directions are skipped."""
    lines: List[DialogueLine] = []
    for raw in (line.strip() for line in str(script or "").splitlines()):
        if not raw or raw.startswith("<"):
            continue
        match = _SPEAKER_LINE.match(raw)
        if match:
            lines.append(DialogueLine(speaker=match.group("speaker"), text=match.group("line").strip()))
    return lines


@dataclass(slots=True)
class DialogueLine:
    speaker: str
    text: str

    def as_dict(self) -> Dict[str, str]:
        return {"speaker": self.speaker, "text": self.text}


@dataclass(slots=True)
class CoverageGap:
    """A script line no shot speaks, and where it used to live if it was cut."""

    speaker: str
    text: str
    removed_slots: List[int] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {"speaker": self.speaker, "text": self.text, "removed_slots": list(self.removed_slots)}


@dataclass(slots=True)
class Check:
    """A deterministic finding. ``warn`` is a problem, ``ok`` is reassurance."""

    id: str
    status: str
    message: str
    shots: List[int] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "status": self.status, "message": self.message, "shots": list(self.shots)}


def film_order(camera_tree: Sequence[Dict[str, Any]]) -> List[int]:
    """The shots in the order they play: camera by camera, in each camera's order.

    This is deliberately not sorted by shot number — a shot's number is its
    identity, and the camera tree carries the film's order.
    """
    order: List[int] = []
    for camera in sorted(camera_tree or [], key=lambda cam: cam.get("idx", 0)):
        for shot_idx in camera.get("active_shot_idxs") or []:
            order.append(int(shot_idx))
    return order


def _audio_by_idx(rows_by_idx: Dict[int, Dict[str, Any]], order: Sequence[int]) -> Dict[int, str]:
    return {idx: normalize_dialogue(rows_by_idx.get(idx, {}).get("audio_desc") or "") for idx in order}


def dialogue_coverage(
    lines: Sequence[DialogueLine],
    rows_by_idx: Dict[int, Dict[str, Any]],
    order: Sequence[int],
    removed_rows_by_idx: Optional[Dict[int, Dict[str, Any]]] = None,
) -> List[CoverageGap]:
    """Script lines no active shot speaks.

    Matching is by containment after normalization, because a shot's
    ``audio_desc`` carries the speaker and stage direction around the same words.
    """
    spoken = _audio_by_idx(rows_by_idx, order)
    removed = {
        int(idx): normalize_dialogue(row.get("audio_desc") or "")
        for idx, row in (removed_rows_by_idx or {}).items()
    }
    gaps: List[CoverageGap] = []
    for line in lines:
        needle = normalize_dialogue(line.text)
        if not needle:
            continue
        if any(_line_covered(needle, audio) for audio in spoken.values()):
            continue
        cut_with = [idx for idx, audio in removed.items() if _line_covered(needle, audio)]
        gaps.append(CoverageGap(speaker=line.speaker, text=line.text, removed_slots=sorted(cut_with)))
    return gaps


def characters_in_shots(
    rows_by_idx: Dict[int, Dict[str, Any]],
    order: Sequence[int],
    plan_rows_by_idx: Optional[Dict[int, Dict[str, Any]]] = None,
    characters: Sequence[Dict[str, Any]] = (),
) -> List[str]:
    """Names of characters any active shot shows, by brief tag or plan index.

    The brief's ``<Tag>`` marks are planner notation and a shot may also list a
    character only in its plan's visible-character indices, so both are read.
    """
    names: set[str] = set()
    for idx in order:
        visual = str((rows_by_idx.get(idx) or {}).get("visual_desc") or "")
        names.update(tag.lower() for tag in _CHARACTER_TAG.findall(visual))

    by_character_idx = {
        int(character.get("idx", -1)): str(character.get("identifier_in_scene") or "")
        for character in characters or []
        if isinstance(character, dict)
    }
    for idx in order:
        plan = (plan_rows_by_idx or {}).get(idx) or {}
        for key in ("ff_vis_char_idxs", "lf_vis_char_idxs"):
            for character_idx in plan.get(key) or []:
                name = by_character_idx.get(int(character_idx))
                if name:
                    names.add(name.lower())
    return sorted(names)


def absent_characters(
    characters: Sequence[Dict[str, Any]],
    rows_by_idx: Dict[int, Dict[str, Any]],
    order: Sequence[int],
    script: str = "",
    plan_rows_by_idx: Optional[Dict[int, Dict[str, Any]]] = None,
) -> List[str]:
    """Characters the script uses who appear in no shot.

    A character can be used without speaking — the script has DeepSeek enter the
    room and put his arm around the Wife — so the whole script is searched, not
    just the dialogue. Characters marked not visible are off-screen by design,
    and a character the script never mentions is not a gap.
    """
    shown = set(characters_in_shots(rows_by_idx, order, plan_rows_by_idx, characters))
    script_text = str(script or "").lower()
    absent: List[str] = []
    for character in characters or []:
        name = str(character.get("identifier_in_scene") or "").strip()
        if not name or character.get("is_visible") is False:
            continue
        if normalize_dialogue(name) in shown or name.lower() in shown:
            continue
        if re.search(rf"\b{re.escape(name.lower())}\b", script_text):
            absent.append(name)
    return absent


def empty_cameras(camera_tree: Sequence[Dict[str, Any]]) -> List[int]:
    """Cameras that hold no shots, so their coverage is unreachable."""
    return [int(cam.get("idx", 0)) for cam in camera_tree or [] if not (cam.get("active_shot_idxs") or [])]


def parse_runtime_range(requirement: str) -> Optional[Tuple[int, int]]:
    """The runtime the user asked for, e.g. "45-60 second comedic short"."""
    match = _RUNTIME_RANGE.search(str(requirement or ""))
    if not match:
        return None
    low, high = int(match.group(1)), int(match.group(2))
    return (low, high) if low <= high else (high, low)


def run_checks(
    *,
    script: str,
    requirement: str,
    characters: Sequence[Dict[str, Any]],
    camera_tree: Sequence[Dict[str, Any]],
    rows_by_idx: Dict[int, Dict[str, Any]],
    removed_rows_by_idx: Optional[Dict[int, Dict[str, Any]]] = None,
    plan_rows_by_idx: Optional[Dict[int, Dict[str, Any]]] = None,
    clip_seconds: int = 0,
) -> List[Check]:
    """Every deterministic finding about the timeline, in reporting order."""
    order = film_order(camera_tree)
    lines = script_dialogue(script)
    checks: List[Check] = []

    gaps = dialogue_coverage(lines, rows_by_idx, order, removed_rows_by_idx)
    if gaps:
        for gap in gaps:
            cut = f" (cut with slot {gap.removed_slots[0]})" if gap.removed_slots else ""
            checks.append(
                Check(
                    id="dialogue_uncovered",
                    status="warn",
                    message=f'{gap.speaker}: "{gap.text}"{cut}',
                    shots=[],
                )
            )
    else:
        checks.append(
            Check(id="dialogue_uncovered", status="ok", message=f"All {len(lines)} script lines are spoken by a shot.")
        )

    absent = absent_characters(characters, rows_by_idx, order, script, plan_rows_by_idx)
    if absent:
        checks.append(
            Check(id="character_absent", status="warn", message=f"No shot shows {', '.join(absent)}.", shots=[])
        )
    else:
        checks.append(Check(id="character_absent", status="ok", message="Every character the script uses appears in a shot."))

    wanted = parse_runtime_range(requirement)
    if wanted and clip_seconds > 0:
        seconds = len(order) * clip_seconds
        low, high = wanted
        if low <= seconds <= high:
            checks.append(
                Check(id="runtime", status="ok", message=f"{seconds}s, inside the requested {low}-{high}s.", shots=order)
            )
        else:
            checks.append(
                Check(
                    id="runtime",
                    status="warn",
                    message=f"{seconds}s of the requested {low}-{high}s ({len(order)} shots x {clip_seconds}s).",
                    shots=order,
                )
            )

    empty = empty_cameras(camera_tree)
    if empty:
        listed = ", ".join(str(camera) for camera in empty)
        checks.append(
            Check(
                id="camera_empty",
                status="warn",
                # A camera is a group of shots sharing a framing; empty means all of its shots
                # were taken out of the film. Nothing plays in it and the render skips it, so
                # this is a hole in the plan rather than a fault — and a reversible one.
                message=(
                    f"Camera {listed} holds no shots. A camera is a group of shots sharing a framing, and every shot in this one "
                    f"has been removed: nothing plays in it, the render skips it, and restoring a removed shot puts it back."
                ),
                shots=[],
            )
        )
    return checks


def staleness(
    review: Dict[str, Any],
    *,
    active_shots: Sequence[int],
    plan_mtimes: Dict[str, float],
) -> Tuple[bool, str]:
    """Whether a stored review still describes this timeline, and why not.

    Derived at call time rather than stored, so a review can never claim to be
    current after the timeline moved under it.
    """
    if not review:
        return True, "No review yet."
    reviewed = [int(idx) for idx in review.get("shots_reviewed") or []]
    if sorted(reviewed) != sorted(int(idx) for idx in active_shots):
        added = sorted(set(int(idx) for idx in active_shots) - set(reviewed))
        removed = sorted(set(reviewed) - set(int(idx) for idx in active_shots))
        parts = []
        if added:
            parts.append(f"shots {added} added")
        if removed:
            parts.append(f"shots {removed} removed")
        return True, "The timeline changed since this review: " + ", ".join(parts) + "."
    reviewed_at = str(review.get("reviewed_at") or "")
    newer = sorted(name for name, mtime in plan_mtimes.items() if reviewed_at and mtime > _timestamp(reviewed_at))
    if newer:
        return True, "Edited since this review: " + ", ".join(newer) + "."
    return False, ""


def _timestamp(value: str) -> float:
    try:
        return datetime.datetime.fromisoformat(value).timestamp()
    except ValueError:
        return 0.0


def build_evidence(
    *,
    script: str,
    requirement: str,
    characters: Sequence[Dict[str, Any]],
    camera_tree: Sequence[Dict[str, Any]],
    rows_by_idx: Dict[int, Dict[str, Any]],
    removed_rows_by_idx: Optional[Dict[int, Dict[str, Any]]] = None,
    plan_rows_by_idx: Optional[Dict[int, Dict[str, Any]]] = None,
    checks: Optional[Sequence[Check]] = None,
    clip_seconds: int = 0,
) -> Dict[str, Any]:
    """The bundle the narrative pass reasons over.

    Shots are listed in playing order with their dialogue, and the cut shots are
    listed separately: a beat is often missing precisely because its shot was
    removed, and the model cannot see that from the active timeline alone.

    Each shot also carries the frame descriptions it is drawn from, so a suggested
    shot can be written at the film's own level of detail and keep the appearance
    its neighbours establish.
    """
    order = film_order(camera_tree)
    plans = plan_rows_by_idx or {}
    return {
        "script": str(script or "").strip(),
        "requirement": str(requirement or "").strip(),
        "clip_seconds": clip_seconds,
        "characters": [
            {
                "name": character.get("identifier_in_scene"),
                "static_features": character.get("static_features"),
                "dynamic_features": character.get("dynamic_features"),
            }
            for character in characters or []
        ],
        "timeline": [
            {
                "shot": idx,
                "camera": (rows_by_idx.get(idx) or {}).get("cam_idx"),
                "visual": (rows_by_idx.get(idx) or {}).get("visual_desc"),
                "audio": (rows_by_idx.get(idx) or {}).get("audio_desc"),
                "motion": (plans.get(idx) or {}).get("motion_desc"),
                "frames": {
                    "first": (plans.get(idx) or {}).get("ff_desc"),
                    "last": (plans.get(idx) or {}).get("lf_desc"),
                },
            }
            for idx in order
        ],
        "cut_shots": [
            {
                "shot": int(idx),
                "camera": row.get("cam_idx"),
                "visual": row.get("visual_desc"),
                "audio": row.get("audio_desc"),
            }
            for idx, row in sorted((removed_rows_by_idx or {}).items())
        ],
        "deterministic_findings": [check.as_dict() for check in checks or []],
    }


def _anchor(after: Any, active_shots: Sequence[int]) -> int:
    """The shot a suggestion follows, snapped to one the film actually has.

    A review naturally anchors a suggestion to the shot the beat was cut from, which is no
    longer in the timeline — and an anchor the film does not have leaves the suggestion
    impossible to add. Snapping it to the shot that now precedes the gap puts the shot back
    where it was cut from rather than at the end of the film.
    """
    shots = sorted(int(idx) for idx in active_shots)
    if not shots:
        return 0
    value = int(after) if str(after).lstrip("-").isdigit() else shots[-1]
    if value in shots:
        return value
    earlier = [shot for shot in shots if shot < value]
    return earlier[-1] if earlier else shots[0]


def normalize_review(payload: Any, *, root: str, active_shots: Sequence[int], reviewed_at: str) -> Dict[str, Any]:
    """Validate a model's review into the stored shape, dropping what does not fit.

    The model proposes beats and suggestions; the identity of the review (which
    shots it covered, when, and for which root) is ours, not its.
    """
    source = payload if isinstance(payload, dict) else {}
    beats: List[Dict[str, Any]] = []
    for index, beat in enumerate(source.get("beats") or []):
        if not isinstance(beat, dict) or not str(beat.get("text") or "").strip():
            continue
        status = str(beat.get("status") or "").strip().lower()
        covered_by = [int(idx) for idx in beat.get("covered_by") or [] if str(idx).lstrip("-").isdigit()]
        beats.append(
            {
                "index": index,
                "text": str(beat.get("text")).strip(),
                "status": status if status in BEAT_STATUSES else ("covered" if covered_by else "missing"),
                "covered_by": covered_by,
                "note": str(beat.get("note") or "").strip(),
            }
        )

    suggestions: List[Dict[str, Any]] = []
    for position, suggestion in enumerate(source.get("suggestions") or []):
        if not isinstance(suggestion, dict) or not str(suggestion.get("visual_desc") or "").strip():
            continue
        after = suggestion.get("after_shot")
        frames = suggestion.get("frames") if isinstance(suggestion.get("frames"), dict) else {}
        suggestions.append(
            {
                "id": str(suggestion.get("id") or f"s{position + 1}"),
                "after_shot": _anchor(after, active_shots),
                "title": str(suggestion.get("title") or "").strip(),
                "visual_desc": str(suggestion.get("visual_desc")).strip(),
                "audio_desc": str(suggestion.get("audio_desc") or "").strip(),
                "motion_desc": str(suggestion.get("motion_desc") or "").strip(),
                "characters": [str(name).strip() for name in suggestion.get("characters") or [] if str(name).strip()],
                "rationale": str(suggestion.get("rationale") or "").strip(),
                "frames": {
                    "first": str(frames.get("first") or "").strip(),
                    "last": str(frames.get("last") or "").strip(),
                },
            }
        )

    return {
        "reviewed_at": reviewed_at,
        "root": root,
        "shots_reviewed": [int(idx) for idx in active_shots],
        "summary": str(source.get("summary") or "").strip(),
        "beats": beats,
        "suggestions": suggestions,
    }


def review_path(working_dir: str) -> str:
    return os.path.join(working_dir, REVIEW_FILENAME)


def load_review(working_dir: str) -> Dict[str, Any]:
    """The stored review, or an empty dict when there is none or it is unreadable."""
    path = review_path(working_dir)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def write_review(working_dir: str, review: Dict[str, Any]) -> str:
    path = review_path(working_dir)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(review, handle, indent=2)
    return path
