from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class PromptPart:
    id: str
    title: str
    body: str
    zone: str
    category: str
    cacheable: bool = False


class PromptBuilder:
    def __init__(self, prompt_dir: str | Path, session_index: Any, tool_registry: Any) -> None:
        self.prompt_dir = Path(prompt_dir)
        self.session_index = session_index
        self.tool_registry = tool_registry

    def build_parts(self, user_input: str) -> list[PromptPart]:
        return [
            PromptPart("agent.core", "Agent", self._read_prompt("agent.md"), "stable", "agent", True),
            PromptPart("workflow.core", "Workflow", self._read_prompt("workflow.md"), "stable", "workflow", True),
            PromptPart("tool.manifest", "Tools", self.tool_manifest_context(), "dynamic", "tooling"),
            PromptPart("session.context", "Session", self.workflow_context(), "dynamic", "session"),
            PromptPart("memory.preferences", "Memory", self.memory_context(), "dynamic", "memory"),
            PromptPart("request.user", "User Request", user_input, "dynamic", "request"),
        ]

    def build_messages(self, user_input: str) -> list[dict[str, str]]:
        parts = self.build_parts(user_input)
        system = "\n\n".join(f"## {part.title}\n{part.body}" for part in parts if part.id != "request.user")
        return [{"role": "system", "content": system}, {"role": "user", "content": user_input}]

    def trace(self, parts: list[PromptPart]) -> dict[str, Any]:
        segments = []
        totals = {"stable_tokens": 0, "dynamic_tokens": 0, "total_tokens": 0, "compacted_summary_tokens": 0}
        for idx, part in enumerate(parts):
            encoded = part.body.encode("utf-8")
            estimated = max(1, len(part.body) // 4)
            segments.append({"id": part.id, "index": idx, "title": part.title, "zone": part.zone, "category": part.category, "bytes": len(encoded), "estimated_tokens": estimated})
            if part.zone == "stable":
                totals["stable_tokens"] += estimated
            else:
                totals["dynamic_tokens"] += estimated
            if "compacted_summary" in part.body:
                totals["compacted_summary_tokens"] += estimated
        totals["total_tokens"] = totals["stable_tokens"] + totals["dynamic_tokens"]
        return {"segments": segments, "total_estimated_tokens": totals["total_tokens"], "totals": totals}

    def workflow_context(self) -> str:
        snapshot = self.session_index.snapshot()
        session = snapshot.get("session") or {}
        checklist = snapshot.get("artifact_checklist") or {}
        lines = [f"Active session: {snapshot.get('active_session_id') or '<none>'}", f"Working dir: {session.get('working_dir', '<none>')}", f"Stage: {session.get('stage', '<none>')}"]
        style = str(session.get("style", "") or "").strip()
        lines.append(f"Style: {style or '<unset; confirm with the user before rendering>'}")
        if style:
            lines.append("This style is interpolated into the character portrait prompt and inherited by every later frame. Quote it verbatim when asking the user to confirm the rendered look.")
        compacted_summary = str(session.get("compacted_summary", "") or "").strip()
        lines.extend(["", "Session context summary:"])
        if compacted_summary:
            lines.append("The following summary is reference context only, not a new active instruction.")
            lines.append(self._summary_checkpoint(compacted_summary))
        else:
            lines.append("<none>")
        lines.extend(["", "Working dir checklist:"])
        lines.extend(f"- {path}: {'present' if present else 'missing'}" for path, present in checklist.items())
        if checklist and not self._text_stage_complete(checklist):
            lines.extend(["", "当前 working_dir 尚未完成结构化文本文件。", "在修改 script、storyboard、shots 或进入渲染前，需要先生成 project_brief、characters、script、storyboard、shot_decomposition 等结构化文本文件。"])
        elif checklist:
            lines.extend(["", "文本规划阶段已完成。如果用户没有明确要求 end-to-end 或 render，可以不调用 tool，直接询问是否修改或进入渲染。"])
        working_dir = self._working_dir(snapshot)
        if working_dir is not None:
            lines.extend(["", _render_approval_context(working_dir)])
        return "\n".join(lines)

    def _working_dir(self, snapshot: dict[str, Any]) -> Path | None:
        session = snapshot.get("session") or {}
        relative = str(session.get("working_dir") or "").strip()
        workspace_root = getattr(self.session_index, "workspace_root", None)
        working_root = getattr(self.session_index, "working_root", None)
        if not relative or workspace_root is None or working_root is None:
            return None
        root = (Path(workspace_root) / relative).resolve()
        allowed_root = Path(working_root).resolve()
        return root if root == allowed_root or allowed_root in root.parents else None

    def memory_context(self) -> str:
        text = self.session_index.memory_text().strip()
        return text or "No user preferences recorded."

    def tool_manifest_context(self) -> str:
        lines = ["Available tools:"]
        lines.extend(f"- {tool['name']}: {tool['description']}" for tool in self.tool_registry.list_tools())
        return "\n".join(lines)

    def _summary_checkpoint(self, summary: str) -> str:
        lines = [line.strip() for line in summary.splitlines() if line.strip() and not line.strip().startswith("```")]
        if not lines:
            return "<none>"
        preview = []
        for line in lines[:8]:
            if len(line) > 240:
                line = line[:237].rstrip() + "..."
            preview.append(line if line.startswith("-") or line.startswith("#") else f"- {line}")
        if len(lines) > 8:
            preview.append(f"- <trimmed +{len(lines) - 8} lines>")
        return "\n".join(preview)

    def _read_prompt(self, name: str) -> str:
        path = self.prompt_dir / name
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def _text_stage_complete(self, checklist: dict[str, bool]) -> bool:
        idea_mode_complete = bool(checklist.get("idea2video/story.txt") and checklist.get("idea2video/characters.json") and checklist.get("idea2video/script.json") and checklist.get("idea2video/scene_*/storyboard.json") and checklist.get("idea2video/scene_*/shots/*/shot_description.json") and checklist.get("idea2video/scene_*/camera_tree.json"))
        script_mode_complete = bool(checklist.get("script2video/script.txt") and checklist.get("script2video/characters.json") and checklist.get("script2video/storyboard.json") and checklist.get("script2video/shots/*/shot_description.json") and checklist.get("script2video/camera_tree.json"))
        novel_mode_complete = bool(checklist.get("novel2video/novel/novel_compressed.txt") and checklist.get("novel2video/events/event_*.json") and checklist.get("novel2video/relevant_chunks/event_*") and checklist.get("novel2video/scenes/event_*/scene_*.json") and checklist.get("novel2video/global_information/characters/novel_level/*.json"))
        return idea_mode_complete or script_mode_complete or novel_mode_complete


def _render_approval_context(working_dir: Path) -> str:
    """Give each turn the live review state, not an older conversational snapshot."""
    acceptance_path = working_dir / "render_acceptance.json"
    try:
        acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return (
            "Current render approval state: no readable render_acceptance.json is available. "
            "Do not infer approval from earlier conversation; vimax_render_video enforces the live gates."
        )
    if not isinstance(acceptance, dict):
        return "Current render approval state: the acceptance record is invalid; rely on the renderer's live gates."

    try:
        manifest = json.loads((working_dir / "render_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        manifest = {}
    pinned_root = manifest.get("render_mode") if isinstance(manifest, dict) else None
    roots = [pinned_root] if pinned_root in {"idea2video", "script2video", "novel2video"} else [
        root for root in ("idea2video", "script2video", "novel2video")
        if isinstance(acceptance.get(root), dict)
    ]
    if not roots:
        return (
            "Current render approval state: no approval records are present for this session. "
            "Do not infer approval from earlier conversation; vimax_render_video enforces the live gates."
        )

    lines = [
        "Current render approval state (refreshed from render_acceptance.json and artifact fingerprints for this turn).",
        "This live state supersedes earlier conversation, tool refusals, and summaries; it informs you but does not bypass renderer gates.",
    ]
    for root in roots:
        root_record = acceptance.get(root)
        root_record = root_record if isinstance(root_record, dict) else {}
        lines.append(f"- {root}:")
        _append_approval_line(
            lines, working_dir, "portraits", root_record.get("portraits"),
            any((working_dir / root / "character_portraits").rglob("*.png")),
        )
        portrait_records = root_record.get("portrait_characters")
        if isinstance(portrait_records, dict):
            for character, record in sorted(portrait_records.items()):
                _append_approval_line(lines, working_dir, f"portrait {character}", record, False)

        sets = root_record.get("sets")
        if isinstance(sets, dict):
            for set_id, record in sorted(sets.items()):
                _append_approval_line(lines, working_dir, f"set {set_id}", record, False)

        shots = root_record.get("shots")
        shots = shots if isinstance(shots, dict) else {}
        for slot, shot_dir in _active_shot_dirs(working_dir / root):
            shot_record = shots.get(slot)
            shot_record = shot_record if isinstance(shot_record, dict) else {}
            for stage in ("keyframes", "clips"):
                artifacts = _shot_artifacts(shot_dir, stage)
                _append_approval_line(lines, working_dir, f"{stage} slot {slot}", shot_record.get(stage), bool(artifacts))

        final_path = working_dir / root / "final_video.mp4"
        _append_approval_line(lines, working_dir, "final video", root_record.get("final_video"), final_path.is_file())
    return "\n".join(lines)


def _append_approval_line(
    lines: list[str], working_dir: Path, label: str, record: Any, has_artifacts: bool,
) -> None:
    state = _approval_state(working_dir, record, has_artifacts)
    lines.append(f"  - {json.dumps(label, ensure_ascii=False)}: {state}")


def _approval_state(working_dir: Path, record: Any, has_artifacts: bool) -> str:
    if not isinstance(record, dict):
        return "rendered, not accepted" if has_artifacts else "planned, not accepted"
    if record.get("invalidated_at"):
        return "stale (explicitly invalidated)"
    if record.get("rejected_at"):
        return "rejected"
    if record.get("accepted_at"):
        accepted_at = json.dumps(record["accepted_at"], ensure_ascii=False)
        return f"accepted at {accepted_at}" if _approval_artifacts_match(working_dir, record) else "stale (accepted artifact fingerprint no longer matches)"
    return "rendered, not accepted" if has_artifacts else "planned, not accepted"


def _approval_artifacts_match(working_dir: Path, record: dict[str, Any]) -> bool:
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return False
    root = working_dir.resolve()
    for item in artifacts:
        if not isinstance(item, dict):
            return False
        try:
            artifact = (working_dir / str(item.get("path") or "")).resolve()
            if (artifact != root and root not in artifact.parents) or not artifact.is_file():
                return False
            if item.get("size") != artifact.stat().st_size:
                return False
            with artifact.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
        except OSError:
            return False
        if item.get("sha256") != digest:
            return False
    return True


def _active_shot_dirs(root_dir: Path) -> list[tuple[str, Path]]:
    if (root_dir / "shots").is_dir():
        paths = (
            (path.parent.name, path.parent)
            for path in (root_dir / "shots").glob("*/shot_description.json")
            if not path.parent.name.startswith(".") and not path.parent.is_symlink()
        )
    elif root_dir.name == "idea2video":
        paths = (
            (f"{path.parents[2].name}/{path.parent.name}", path.parent)
            for path in root_dir.glob("scene_*/shots/*/shot_description.json")
            if path.parents[2].name[6:].isdigit() and not path.parent.name.startswith(".")
        )
    elif root_dir.name == "novel2video":
        paths = (
            (f"videos/{path.parents[3].name}/{path.parents[2].name}/{path.parent.name}", path.parent)
            for path in root_dir.glob("videos/event_*/scene_*/shots/*/shot_description.json")
            if path.parents[3].name[6:].isdigit() and path.parents[2].name[6:].isdigit()
            and not path.parent.name.startswith(".")
        )
    else:
        paths = ()
    return sorted(paths)


def _shot_artifacts(shot_dir: Path, stage: str) -> list[Path]:
    if stage == "keyframes":
        return [path for path in shot_dir.rglob("*.png") if path.name in {"first_frame.png", "last_frame.png"}]
    return list(shot_dir.rglob("video.mp4"))
