from __future__ import annotations

import asyncio
import hashlib
import traceback
from datetime import datetime, timezone
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import json
import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any


from pipelines.render_contract import DEFAULT_RENDER_PHASE, RENDER_PHASES, PortraitGenerationError, normalize_phase
from pipelines.set_references import (
    accepted_set_references,
    generate_set_references,
    plan_set_specs,
    shot_directories,
    validate_set_candidate_count,
    validate_set_reference_generation,
)

from .config import (
    api_provider_from_base_url,
    embedding_api_key,
    embedding_base_url,
    embedding_model,
    embedding_model_provider,
    image_api_key,
    image_base_url,
    image_model,
    image_resolution,
    llm_api_key,
    llm_base_url,
    llm_model,
    llm_model_provider,
    reranker_api_key,
    reranker_base_url,
    reranker_model,
    video_api_key,
    video_base_url,
    video_duration_policy,
    video_generate_audio,
    video_model,
    video_resolution,
    validate_video_shot_duration,
)
from .models import ToolResult
from .tools import ToolArgumentSchema, ToolRuntimeContext, ToolSpec

from utils.project_lock import ProjectBusyError, project_write_lock
from utils.text import safe_path_component

if TYPE_CHECKING:
    from interfaces import CharacterInScene
    from pipelines.novel2movie_pipeline import Novel2MoviePipeline
    from tools.image_generator_nanobanana_yunwu_api import ImageGeneratorNanobananaYunwuAPI
    from tools.image_generator_openrouter_api import ImageGeneratorOpenRouterAPI
    from tools.image_generator_agnes_api import ImageGeneratorAgnesAPI
    from tools.reranker_bge_silicon_api import RerankerBgeSiliconapi


class _UnavailableGenerator:
    async def generate_single_image(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Image generator is not available in narrative planning mode")

    async def generate_single_video(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Video generator is not available in narrative planning mode")


def build_vimax_adapter_specs(workspace_root: str | Path, session_index: Any) -> list[ToolSpec]:
    adapter = ViMaxAdapters(Path(workspace_root), session_index)
    return [
        ToolSpec(
            name="vimax_narrative_planning",
            description=(
                "Create or revise ViMax structured text artifacts for the active session. "
                "Idea mode writes story, characters, script, and scene-level storyboard/shot_decomposition/camera_tree under idea2video/scene_<idx>/. "
                "Script mode writes characters, storyboard, shot_decomposition, and camera_tree under script2video/. "
                "Pass the active session_id from prompt context when the user is working in the selected project. An empty active session is initialized in place; a different source on a non-empty session creates a new session instead of overwriting existing artifacts. If idea/script/revision_target are omitted and the active session has an idea, continue that session and fill missing structured text artifacts. "
                "It does not generate keyframes, video clips, or final video. Call this before revising storyboard/shots when those artifacts do not exist. "
                "render_mode picks the root to plan: \"script2video\" or \"idea2video\". A session can hold both — an abandoned first attempt beside the one being worked in — and planning would otherwise re-enter the idea root, so pass it whenever the session has more than one. The chosen root is pinned for the later render. "
                "revision_target takes precedence over idea/script and render_mode: revise that existing session-relative artifact without replanning or switching sessions. revision_instruction is required. Unchanged content (including JSON formatting-only changes) returns ok=false with error_type=revision_noop and leaves the file and stale dependencies unchanged. "
            ),
            handler=adapter.vimax_narrative_planning,
            schema={
                "session_id": ToolArgumentSchema(str, required=False, default=""),
                "idea": ToolArgumentSchema(str, required=False, default=""),
                "script": ToolArgumentSchema(str, required=False, default=""),
                "user_requirement": ToolArgumentSchema(str, required=False, default=""),
                "style": ToolArgumentSchema(str, required=False, default=""),
                "revision_target": ToolArgumentSchema(str, required=False, default=""),
                "revision_instruction": ToolArgumentSchema(str, required=False, default=""),
                "render_mode": ToolArgumentSchema(str, required=False, default=""),
            },
        ),
        ToolSpec(
            name="vimax_novel_planning",
            description=(
                "Create ViMax structured text artifacts from a novel or novel excerpt. "
                "This writes novel2video/novel, events, relevant_chunks, scenes, and global_information text artifacts. "
                "Use this when the user provides long prose, a novel excerpt, or asks for novel-to-video planning. Pass the active session_id when the user is working in a selected empty project. "
                "It does not generate character portraits, scene videos, or final video."
            ),
            handler=adapter.vimax_novel_planning,
            schema={
                "session_id": ToolArgumentSchema(str, required=False, default=""),
                "novel_text": ToolArgumentSchema(str, required=True),
                "user_requirement": ToolArgumentSchema(str, required=False, default=""),
                "style": ToolArgumentSchema(str, required=False, default=""),
            },
        ),
        ToolSpec(
            name="vimax_render_video",
            description=(
                "Render a ViMax session in four gated phases, selected with stop_after. "
                "\"portraits\" renders character portraits and stops for style/direction review. "
                "\"sets\" generates only the explicitly selected, character-free set references and stops so they can be reviewed and accepted before stills. It can run independently of portrait approval. "
                "\"stills\" renders the keyframes video generation will consume and stops for explicit approval before video clips. "
                "\"video\" renders clips and the final concatenated video. "
                "Portrait approval gates stills; every assigned set reference must be accepted before any missing or redrawn shot image, even with allow_unlocked=true; an explicitly empty set_ids binding requires no set reference. Accepted keyframes gate video. allow_unlocked applies only to existing portrait/keyframe gates, never set references. "
                "Default is \"portraits\". Each phase reuses artifacts that already exist. "
                "Changing the image model mid-sequence is refused unless allow_model_change=true is passed with explicit user approval. "
                "render_mode picks which root renders: \"script2video\", \"idea2video\" or \"novel2video\". The chosen root is pinned in render_manifest.json. "
                "Shot redraws use redo_shots with zero-based slot keys. redo_rejected=true redraws every rejected shot. A redraw is restricted to its shots and phase. "
                "Use set_ids to explicitly select set references for stop_after=\"sets\". set_candidate_count selects 1–4 alternative candidates per set (default 1) and values other than 1 require stop_after=\"sets\". redo_sets requests new alternatives, preserving existing candidates and canonical references; accepted sets must be unlocked before new candidates can be generated. Set generation never redraws portraits or shot frames. "
                "This checks that structured text artifacts exist before rendering and reports missing dependencies instead of pretending render started."
            ),
            handler=adapter.vimax_render_video,
            schema={
                "session_id": ToolArgumentSchema(str, required=False, default=""),
                "stop_after": ToolArgumentSchema(str, required=False, default=DEFAULT_RENDER_PHASE),
                "render_mode": ToolArgumentSchema(str, required=False, default=""),
                "allow_model_change": ToolArgumentSchema(bool, required=False, default=False),
                "allow_unlocked": ToolArgumentSchema(bool, required=False, default=False),
                "redo_shots": ToolArgumentSchema(list, required=False, default=[]),
                "redo_rejected": ToolArgumentSchema(bool, required=False, default=False),
                "allow_unreviewed_redo": ToolArgumentSchema(bool, required=False, default=False),
                "set_ids": ToolArgumentSchema(list, required=False, default=[]),
                "redo_sets": ToolArgumentSchema(list, required=False, default=[]),
                "set_candidate_count": ToolArgumentSchema(int, required=False, default=1),
            },
        ),
        ToolSpec(
            name="vimax_review_timeline",
            description=(
                "Check a ViMax timeline against its script and record what is missing. "
                "It reads the script, the shots in playing order, the shots that were cut, and the characters, and reports: script dialogue no shot speaks (naming the removed slot that used to carry it), characters the script uses that no shot shows, and the runtime against what the user asked for. A narrative pass then adds the story's beats with the shots that cover them and proposes new shots for the gaps. "
                "It writes continuity_review.json at the session root; the Timeline reads that to show the gaps and offer each suggested shot as a shot the user can add with one click. "
                "Call it when the user asks whether the film still tells the script's story, asks what is missing, or after shots have been cut or rewritten. "
                "render_mode picks which root is reviewed; pass it when the session holds more than one. "
                "It never changes the timeline — it reports and suggests, and the user adds a suggested shot themselves."
            ),
            handler=adapter.vimax_review_timeline,
            schema={
                "session_id": ToolArgumentSchema(str, required=False, default=""),
                "render_mode": ToolArgumentSchema(str, required=False, default=""),
            },
        ),
    ]


def _project_locked(method: Any) -> Any:
    async def locked(self: "ViMaxAdapters", args: dict[str, Any], runtime: ToolRuntimeContext | None = None) -> ToolResult:
        name = method.__name__
        if name == "vimax_narrative_planning" and str(args.get("revision_target") or "").strip():
            # An artifact revision stays in the selected session, even with source arguments.
            args = {**args, "idea": "", "script": ""}
        session_id = str(args.get("session_id", "") or "").strip()
        explicit_session_id = bool(session_id)
        call_args = args

        # Resolve the target without touching an existing session. A conflicting
        # writer must be rejected before set_active(), metadata, status, or artifacts
        # are changed. A brand-new generated session has no prior project lock; explicit
        # unknown IDs can still be locked at their deterministic directory before create.
        if name in {"vimax_narrative_planning", "vimax_novel_planning"}:
            if not session_id:
                session_id = str((self.session_index.active() or {}).get("session_id") or "")
            candidate = self.session_index.get(session_id) if session_id else None
            if name == "vimax_narrative_planning":
                source = str(args.get("idea", "") or "").strip() or str(args.get("script", "") or "").strip()
            else:
                source = str(args.get("novel_text", "") or "").strip()
            if candidate is None and explicit_session_id and not (name == "vimax_novel_planning" and not source):
                session_id = self.session_index._normalize_session_id(session_id)
                working_dir = self.session_index._working_dir_for_id(session_id)
                call_args = {**args, "session_id": session_id}
                try:
                    with project_write_lock(working_dir):
                        return await method(self, call_args, runtime)
                except ProjectBusyError as exc:
                    return ToolResult(name, False, str(exc), {"error_type": "project_busy", "session_id": session_id})
            if candidate is None or (source and _is_new_source_for_session(candidate, source)):
                if name == "vimax_novel_planning" and not source:
                    return await method(self, args, runtime)
                session = self._resolve_session(
                    session_id,
                    idea=str(args.get("idea", "") or "").strip() if name == "vimax_narrative_planning" else str(args.get("novel_text", "") or "").strip(),
                    script=str(args.get("script", "") or "").strip() if name == "vimax_narrative_planning" else "",
                    user_requirement=str(args.get("user_requirement", "") or "").strip(),
                    style=str(args.get("style", "") or "").strip(),
                )
                session_id = str(session["session_id"])
                call_args = {**args, "session_id": session_id}
        elif name in {"vimax_render_video", "vimax_review_timeline"} and not session_id:
            session_id = str((self.session_index.active() or {}).get("session_id") or "")

        if not session_id:
            return await method(self, call_args, runtime)
        working_dir = self.session_index.working_dir(session_id)
        try:
            with project_write_lock(working_dir):
                return await method(self, call_args, runtime)
        except ProjectBusyError as exc:
            return ToolResult(name, False, str(exc), {"error_type": "project_busy", "session_id": session_id})
    return locked


class ViMaxAdapters:
    def __init__(self, workspace_root: Path, session_index: Any) -> None:
        self.workspace_root = workspace_root.resolve()
        self.session_index = session_index

    @_project_locked
    async def vimax_narrative_planning(self, args: dict[str, Any], runtime: ToolRuntimeContext | None = None) -> ToolResult:
        idea = str(args.get("idea", "") or "").strip()
        script = str(args.get("script", "") or "").strip()
        user_requirement = str(args.get("user_requirement", "") or "").strip()
        requested_style = str(args.get("style", "") or "").strip()
        style = requested_style
        session = self._resolve_session(str(args.get("session_id", "") or ""), idea=idea, script=script, user_requirement=user_requirement, style=requested_style)
        session_id = session["session_id"]
        working_dir = self.session_index.working_dir(session_id)
        revision_target = str(args.get("revision_target") or "").strip()
        if revision_target:
            return await self._revise_narrative_artifact(
                session_id, working_dir, revision_target,
                str(args.get("revision_instruction") or "").strip(), runtime,
            )
        idea_dir = working_dir / "idea2video"
        script_dir = working_dir / "script2video"
        idea_dir.mkdir(parents=True, exist_ok=True)
        script_dir.mkdir(parents=True, exist_ok=True)

        # Which root this planning fills in. A session that was started from an idea
        # and later worked in script mode keeps both, and planning would otherwise
        # re-enter the idea root — the same trap the render has.
        try:
            planning_mode = _resolve_planning_mode(
                str(args.get("render_mode", "") or "").strip(),
                _read_render_manifest(working_dir),
                existing_script=_read_text(script_dir / "script.txt"),
            )
        except ValueError as exc:
            return ToolResult("vimax_narrative_planning", False, str(exc), {"error_type": "invalid_render_mode", "session_id": session_id})
        if planning_mode == "script2video":
            # Plan from a script: the argument, or the script already in the root.
            script = script or _read_text(script_dir / "script.txt")
            idea = ""
            user_requirement = user_requirement or str(session.get("user_requirement") or "").strip()

        if not idea and not script:
            session_idea = str(session.get("idea") or "").strip()
            if session_idea:
                idea = session_idea
                user_requirement = user_requirement or str(session.get("user_requirement") or "").strip()
                style = requested_style or str(session.get("style") or "").strip() or "Cinematic, coherent, 16:9"
            else:
                return ToolResult("vimax_narrative_planning", False, "Provide `idea`, `script`, a revision target, or an active session with an existing idea for narrative planning.", {"error_type": "missing_input", "session_id": session_id})

        style = style or str(session.get("style") or "").strip() or "Cinematic, coherent, 16:9"
        self._update_session_metadata(session_id, idea="", user_requirement="", style=style)

        try:
            from pipelines.idea2video_pipeline import Idea2VideoPipeline
            from pipelines.script2video_pipeline import Script2VideoPipeline

            self.session_index.update_stage(session_id, "narrative_planning", "Generating structured text artifacts")
            if runtime:
                runtime.emit_progress("Starting narrative planning", stage="starting", metadata={"session_id": session_id})
                await asyncio.sleep(0)
            generated_before = self.session_index.artifact_checklist(session_id)
            if runtime:
                runtime.emit_progress("Initializing bounded chat model", stage="initializing_llm", metadata={"session_id": session_id, "timeout_seconds": _llm_request_timeout_seconds(), "max_tokens": _narrative_max_tokens()})
                await asyncio.sleep(0)
            chat_model = _build_chat_model()
            if runtime:
                runtime.emit_progress("Bounded chat model initialized", stage="chat_model_ready", metadata={"session_id": session_id})
                await asyncio.sleep(0)
            dummy = _UnavailableGenerator()
            # Do not globally redirect stdout/stderr while the JSONL CLI is streaming events.
            # The adapter exposes pipeline progress through explicit tool_progress events instead.
            if idea:
                idea_pipeline = Idea2VideoPipeline(chat_model=chat_model, image_generator=dummy, video_generator=dummy, working_dir=str(idea_dir))
                if runtime:
                    runtime.emit_progress("Idea pipeline initialized", stage="idea_pipeline_ready", metadata={"session_id": session_id})
                    await asyncio.sleep(0)
                story = await _run_planning_step(
                    "Developing story from user idea",
                    "develop_story",
                    idea_pipeline.develop_story(idea=idea, user_requirement=user_requirement, quiet=True),
                    runtime,
                    {"session_id": session_id},
                )
                characters = await _run_planning_step(
                    "Extracting characters from story",
                    "extract_characters",
                    idea_pipeline.extract_characters(story=story, quiet=True),
                    runtime,
                    {"session_id": session_id},
                )
                scene_scripts = await _run_planning_step(
                    "Writing scene scripts from story",
                    "write_script",
                    idea_pipeline.write_script_based_on_story(story=story, user_requirement=user_requirement, quiet=True),
                    runtime,
                    {"session_id": session_id},
                )
                for idx, scene_script in enumerate(scene_scripts if isinstance(scene_scripts, list) else [scene_scripts]):
                    scene_dir = idea_dir / f"scene_{idx}"
                    scene_text = scene_script if isinstance(scene_script, str) else json.dumps(scene_script, ensure_ascii=False, indent=2)
                    script_pipeline = Script2VideoPipeline(chat_model=chat_model, image_generator=dummy, video_generator=dummy, working_dir=str(scene_dir))
                    await _run_planning_step(
                        f"Planning scene {idx} storyboard and shots",
                        "plan_scene",
                        script_pipeline.plan_text_artifacts(script=scene_text, user_requirement=user_requirement, style=style, characters=characters, progress=_pipeline_progress(runtime, session_id, scene_index=idx), quiet=True),
                        runtime,
                        {"session_id": session_id, "scene_index": idx},
                    )
            else:
                if script:
                    (script_dir / "script.txt").write_text(script, encoding="utf-8")
                script_pipeline = Script2VideoPipeline(chat_model=chat_model, image_generator=dummy, video_generator=dummy, working_dir=str(script_dir))
                if runtime:
                    runtime.emit_progress("Script pipeline initialized", stage="script_pipeline_ready", metadata={"session_id": session_id})
                    await asyncio.sleep(0)
                await _run_planning_step(
                    "Planning storyboard and shots from provided script",
                    "plan_script",
                    script_pipeline.plan_text_artifacts(script=script, user_requirement=user_requirement, style=style, progress=_pipeline_progress(runtime, session_id), quiet=True),
                    runtime,
                    {"session_id": session_id},
                )
            # Composite roots have all their child shot plans by now; only then may
            # the shared location plan be derived from complete coverage.
            set_specs = await plan_set_specs(chat_model, working_dir / ("idea2video" if idea else "script2video"))
        except Exception as exc:
            self.session_index.update_stage(session_id, "error", f"Narrative planning failed: {exc}")
            checklist = self.session_index.artifact_checklist(session_id)
            payload = {
                "session_id": session_id,
                "working_dir": str(working_dir.relative_to(self.workspace_root)),
                "error_type": "recoverable_planning_step_failed",
                "retryable": True,
                "error": str(exc),
                "present": [path for path, present in checklist.items() if present],
                "missing": [path for path, present in checklist.items() if not present],
            }
            if runtime:
                runtime.emit_progress("Narrative planning failed; partial artifacts were kept", stage="planning_failed", metadata=payload)
            return ToolResult("vimax_narrative_planning", False, f"Narrative planning failed: {exc}", payload)

        # Pin the root the plan belongs to, so a later render defaults to it.
        _pin_render_mode(working_dir, "idea2video" if idea else "script2video")
        checklist = self.session_index.artifact_checklist(session_id)
        generated = [path for path, present in checklist.items() if present and not generated_before.get(path)]
        reused = [path for path, present in checklist.items() if present and generated_before.get(path)]
        ready_for_render = _ready_for_render(checklist)
        self.session_index.update_stage(session_id, "narrative_planned", "Structured text planning complete" if ready_for_render else "Structured text planning partially complete")
        if runtime:
            runtime.emit_progress("Narrative planning complete", stage="completed", metadata={"ready_for_render": ready_for_render})
        payload = {
            "session_id": session_id,
            "working_dir": str(working_dir.relative_to(self.workspace_root)),
            "generated": generated,
            "reused": reused,
            "missing": [path for path, present in checklist.items() if not present],
            "ready_for_render": ready_for_render,
            "sets": [
                {"id": spec["id"], "name": spec["name"], "description": spec["description"], "shots": list(spec["shots"])}
                for spec in set_specs.values()
            ],
        }
        return ToolResult("vimax_narrative_planning", True, json.dumps(payload, ensure_ascii=False, indent=2), payload)

    async def _revise_narrative_artifact(self, session_id: str, working_dir: Path, revision_target: str, revision_instruction: str, runtime: ToolRuntimeContext | None = None) -> ToolResult:
        if not revision_instruction:
            self.session_index.update_stage(session_id, "error", "Revision failed: missing revision_instruction")
            return ToolResult("vimax_narrative_planning", False, "revision_instruction is required when revision_target is provided.", {"error_type": "missing_revision_instruction", "session_id": session_id, "revision_target": revision_target})
        try:
            target_path = _resolve_artifact_path(working_dir, revision_target)
        except ValueError as exc:
            self.session_index.update_stage(session_id, "error", f"Revision failed: {exc}")
            return ToolResult("vimax_narrative_planning", False, str(exc), {"error_type": "invalid_revision_target", "session_id": session_id, "revision_target": revision_target})
        if not target_path.exists():
            self.session_index.update_stage(session_id, "error", f"Revision failed: target does not exist: {revision_target}")
            return ToolResult("vimax_narrative_planning", False, f"Revision target does not exist: {revision_target}", {"error_type": "dependency_missing", "session_id": session_id, "revision_target": revision_target})
        try:
            self.session_index.update_stage(session_id, "narrative_planning", "Revising structured text artifact")
            if runtime:
                runtime.emit_progress("Revising structured text artifact", stage="revising", metadata={"session_id": session_id, "revision_target": revision_target})
            chat_model = _build_chat_model()
            before = target_path.read_text(encoding="utf-8")
            revised = await _revise_artifact_with_llm(chat_model, target_path.relative_to(working_dir).as_posix(), before, revision_instruction)
            unchanged = revised == before
            if target_path.suffix == ".json":
                try:
                    revised_payload = json.loads(revised)
                except json.JSONDecodeError as exc:
                    self.session_index.update_stage(session_id, "error", f"Revision failed: invalid JSON output: {exc}")
                    return ToolResult("vimax_narrative_planning", False, f"Revision output was not valid JSON: {exc}", {"error_type": "invalid_revision_json", "session_id": session_id, "revision_target": revision_target})
                try:
                    unchanged = revised_payload == json.loads(before)
                except json.JSONDecodeError:
                    # A valid replacement can repair a malformed existing artifact.
                    unchanged = False
                revised = json.dumps(revised_payload, ensure_ascii=False, indent=2)
            if unchanged:
                message = "Revision produced no content changes; the artifact was left unchanged."
                self.session_index.update_stage(session_id, "error", message)
                return ToolResult("vimax_narrative_planning", False, message, {
                    "error_type": "revision_noop", "session_id": session_id,
                    "revision_target": target_path.relative_to(working_dir).as_posix(),
                    "revised": [],
                })
            target_path.write_text(revised, encoding="utf-8")
        except Exception as exc:
            self.session_index.update_stage(session_id, "error", f"Revision failed: {exc}")
            raise
        revision_target_path = target_path.relative_to(working_dir).as_posix()
        revision_plan = _revisioned_shot_plan(revision_target_path)
        if revision_plan is not None and _continuity_enabled(working_dir, revision_plan[0]):
            root, slot = revision_plan
            reason = f"Shot {slot} plan changed; regenerate and review its keyframes and clip."
            _mark_continuity_records(working_dir, root, [slot], reason)
            try:
                old_plan = json.loads(before)
            except json.JSONDecodeError:
                old_plan = None
            new_plan = json.loads(revised)
            ending_changed = (
                not isinstance(old_plan, dict) or not isinstance(new_plan, dict)
                or any(
                    old_plan.get(key) != new_plan.get(key)
                    for key in ("lf_desc", "lf_vis_char_idxs", "reuse_previous_last_frame")
                )
            )
            if ending_changed:
                _invalidate_continuity_dependents(working_dir, root, [slot])

        stale = _stale_keys_for_revision(target_path.relative_to(working_dir).as_posix())
        if stale:
            self.session_index.mark_stale(session_id, stale)
        self.session_index.append_log("revisions", {"session_id": session_id, "target": target_path.relative_to(working_dir).as_posix(), "instruction": revision_instruction, "stale": stale, "before_preview": before[:500], "after_preview": revised[:500]})
        checklist = self.session_index.artifact_checklist(session_id)
        ready_for_render = _ready_for_render(checklist)
        self.session_index.update_stage(session_id, "narrative_planned" if ready_for_render else "narrative_planning", "Revised structured text artifact")
        payload = {
            "session_id": session_id,
            "working_dir": str(working_dir.relative_to(self.workspace_root)),
            "generated": [],
            "reused": [path for path, present in checklist.items() if present],
            "revised": [target_path.relative_to(working_dir).as_posix()],
            "missing": [path for path, present in checklist.items() if not present],
            "stale": stale,
            "ready_for_render": ready_for_render,
            "revision_target": target_path.relative_to(working_dir).as_posix(),
        }
        return ToolResult("vimax_narrative_planning", True, json.dumps(payload, ensure_ascii=False, indent=2), payload)

    @_project_locked
    async def vimax_novel_planning(self, args: dict[str, Any], runtime: ToolRuntimeContext | None = None) -> ToolResult:
        novel_text = str(args.get("novel_text", "") or "").strip()
        user_requirement = str(args.get("user_requirement", "") or "").strip()
        style = str(args.get("style", "") or "").strip() or "Cinematic, coherent, 16:9"
        if not novel_text:
            return ToolResult("vimax_novel_planning", False, "novel_text is required for novel planning.", {"error_type": "missing_input"})

        session_id_arg = str(args.get("session_id", "") or "").strip()
        session = self._resolve_session(session_id_arg, idea=novel_text, script="", user_requirement=user_requirement, style=style)
        session_id = session["session_id"]
        working_dir = self.session_index.working_dir(session_id)
        novel_dir = working_dir / "novel2video"
        novel_dir.mkdir(parents=True, exist_ok=True)
        generated_before = self.session_index.artifact_checklist(session_id)

        try:
            self.session_index.update_stage(session_id, "novel_planning", "Generating novel structured text artifacts")
            if runtime:
                runtime.emit_progress("Starting novel planning", stage="starting", metadata={"session_id": session_id})
                await asyncio.sleep(0)
            chat_model = _build_chat_model()
            pipeline = _build_novel_pipeline(novel_dir)
            await _run_planning_step(
                "Planning novel structured text artifacts",
                "novel_plan_text_artifacts",
                pipeline.plan_text_artifacts(
                    novel_text=novel_text,
                    user_requirement=user_requirement,
                    style=style,
                    progress=_pipeline_progress(runtime, session_id),
                    quiet=True,
                ),
                runtime,
                {"session_id": session_id},
            )
            dummy = _UnavailableGenerator()
            await _run_planning_step(
                "Planning novel scene shots without rendering images",
                "novel_plan_scene_artifacts",
                pipeline.plan_scene_artifacts(
                    chat_model,
                    image_generator=dummy,
                    video_generator=dummy,
                    user_requirement=user_requirement,
                    style=style,
                    progress=_pipeline_progress(runtime, session_id),
                    quiet=True,
                ),
                runtime,
                {"session_id": session_id},
            )
            set_specs = await _run_planning_step(
                "Planning character-free set descriptions",
                "novel_plan_set_specs",
                plan_set_specs(chat_model, novel_dir),
                runtime,
                {"session_id": session_id},
            )
        except Exception as exc:
            self.session_index.update_stage(session_id, "error", f"Novel planning failed: {exc}")
            return ToolResult("vimax_novel_planning", False, str(exc), {"error_type": "exception", "session_id": session_id})

        checklist = self.session_index.artifact_checklist(session_id)
        generated = [path for path, present in checklist.items() if path.startswith("novel2video/") and present and not generated_before.get(path)]
        reused = [path for path, present in checklist.items() if path.startswith("novel2video/") and present and generated_before.get(path)]
        missing = [path for path, present in checklist.items() if path.startswith("novel2video/") and not present]
        ready = _novel_text_ready(checklist)
        self.session_index.update_stage(session_id, "novel_planned" if ready else "novel_planning", "Novel structured text planning complete" if ready else "Novel structured text planning partially complete")
        if runtime:
            runtime.emit_progress("Novel planning complete", stage="completed", metadata={"session_id": session_id, "ready_for_scene_render": False})
        payload = {
            "session_id": session_id,
            "working_dir": str(working_dir.relative_to(self.workspace_root)),
            "generated": generated,
            "reused": reused,
            "missing": missing,
            "sets": [
                {"id": spec["id"], "name": spec["name"], "description": spec["description"], "shots": list(spec["shots"])}
                for spec in set_specs.values()
            ],
            "ready_for_scene_render": False,
        }
        return ToolResult("vimax_novel_planning", True, json.dumps(payload, ensure_ascii=False, indent=2), payload)

    @_project_locked
    async def vimax_render_video(self, args: dict[str, Any], runtime: ToolRuntimeContext | None = None) -> ToolResult:
        session_id = str(args.get("session_id", "") or "").strip()
        session = self.session_index.get(session_id) if session_id else self.session_index.active()
        if session is None:
            return ToolResult("vimax_render_video", False, "No active session to render.", {"error_type": "missing_session"})
        session_id = session["session_id"]
        checklist = self.session_index.artifact_checklist(session_id)
        missing = _missing_render_dependencies(checklist)
        working_dir = self.session_index.working_dir(session_id)
        if missing:
            payload = {"error_type": "dependency_missing", "missing": missing, "session_id": session_id}
            _write_render_status(working_dir, status="dependency_missing", payload=payload)
            return ToolResult("vimax_render_video", False, f"Dependency missing: {', '.join(missing)}", payload)

        try:
            stop_after = normalize_phase(args.get("stop_after") or DEFAULT_RENDER_PHASE)
        except ValueError as exc:
            return ToolResult("vimax_render_video", False, str(exc), {"error_type": "invalid_phase", "session_id": session_id})
        allow_model_change = bool(args.get("allow_model_change", False))
        try:
            render_mode = _resolve_render_mode(checklist, _read_render_manifest(working_dir), str(args.get("render_mode", "") or "").strip())
        except ValueError as exc:
            return ToolResult("vimax_render_video", False, str(exc), {"error_type": "invalid_render_mode", "session_id": session_id})
        set_ids = args.get("set_ids") or []
        redo_sets = args.get("redo_sets") or []
        set_candidate_count = args.get("set_candidate_count", 1)
        shot_redo_requested = bool(args.get("redo_shots") or args.get("redo_rejected", False))
        try:
            validate_set_candidate_count(set_candidate_count)
        except ValueError as exc:
            return ToolResult("vimax_render_video", False, str(exc), {"error_type": "invalid_set_candidate_count", "session_id": session_id})
        if stop_after == "sets" and shot_redo_requested:
            return ToolResult("vimax_render_video", False, "The sets phase cannot be combined with shot redraws.", {"error_type": "invalid_phase_arguments", "session_id": session_id, "phase": stop_after})
        if stop_after != "sets" and (set_ids or redo_sets or set_candidate_count != 1):
            return ToolResult("vimax_render_video", False, "set_ids, redo_sets, and non-default set_candidate_count are only valid with stop_after='sets'.", {"error_type": "invalid_phase_arguments", "session_id": session_id, "phase": stop_after})
        # A redo is resolved here, before anything is deleted or spent, so a mistyped
        # shot number cannot clear artifacts the render then fails to replace.
        redo = {} if stop_after == "sets" else _resolve_redo(working_dir, render_mode, args)
        if redo.get("error_type"):
            payload = {**redo, "session_id": session_id, "render_mode": render_mode}
            return ToolResult("vimax_render_video", False, str(redo["error"]), payload)
        # A redraw is shot-scoped, so it can never justify moving the sequence: the shots
        # being redrawn live in the root the sequence is already rendering from. Without
        # this, a redraw meant for one root re-renders the other, which nobody asked for.
        pinned_mode = str(_read_render_manifest(working_dir).get("render_mode") or "")
        if redo and pinned_mode and pinned_mode != render_mode:
            payload = {
                "error_type": "redo_wrong_root",
                "retryable": False,
                "session_id": session_id,
                "render_mode": render_mode,
                "recorded_render_mode": pinned_mode,
                "error": (
                    f"A redraw stays inside the sequence's own root, and this sequence renders from {pinned_mode}/. "
                    f"The shots in that Timeline are the ones that can be redrawn: pass render_mode={pinned_mode!r} with their slot keys. "
                    f"Redrawing one shot is not a reason to move the whole sequence to {render_mode}/."
                ),
            }
            self.session_index.update_stage(session_id, "error", payload["error"])
            _write_render_status(working_dir, status="error", payload=payload)
            return ToolResult("vimax_render_video", False, payload["error"], payload)
        if redo and normalize_phase(stop_after) != redo["phase"] and RENDER_PHASES.index(stop_after) < RENDER_PHASES.index(redo["phase"]):
            # Deleting a shot's artifacts and then stopping before the phase that redraws
            # them would leave the shot empty, so the redo's own phase is the floor.
            stop_after = redo["phase"]
        removed: list[str] = []
        tally: dict[str, Any] = {}
        try:
            from pipelines.idea2video_pipeline import Idea2VideoPipeline
            from pipelines.script2video_pipeline import Script2VideoPipeline

            chat_model = _build_chat_model()
            image_generator = _build_image_generator()
            video_generator = _build_video_generator(self.workspace_root)
            style = str(session.get("style", "") or "")
            # A redraw names shots, and this run works on those shots only.
            only_shots = list(redo.get("slots") or {}) if redo else None
            script_only_shots = (
                [int(slot) for slot in only_shots if str(slot).lstrip("-").isdigit()]
                if only_shots is not None and render_mode == "script2video"
                else only_shots
            )

            sequence_refusal = _enforce_render_sequence(
                working_dir,
                image_generator=image_generator,
                video_generator=video_generator,
                style=style,
                allow_model_change=allow_model_change,
                render_mode=render_mode,
                commit=False,
            )
            if sequence_refusal is not None:
                self.session_index.update_stage(session_id, "error", sequence_refusal["error"])
                _write_render_status(working_dir, status="error", payload={**sequence_refusal, "session_id": session_id})
                return ToolResult("vimax_render_video", False, sequence_refusal["error"], sequence_refusal)
            novel_pipeline = None
            if render_mode == "novel2video":
                novel_pipeline = _build_novel_render_pipeline(working_dir / "novel2video", chat_model, image_generator, video_generator)
                if stop_after in {"sets", "stills", "video"}:
                    await novel_pipeline.plan_scene_artifacts(
                        chat_model,
                        image_generator=image_generator,
                        video_generator=video_generator,
                        user_requirement=str(session.get("user_requirement", "")),
                        style=style,
                        progress=_pipeline_progress(runtime, session_id, tally=tally),
                        quiet=True,
                        require_existing=True,
                    )

            if stop_after != "sets":
                try:
                    _preflight_duration_overrides(
                        working_dir / render_mode,
                        only_shots,
                        getattr(video_generator, "duration_policy", None),
                    )
                except ValueError as exc:
                    refusal = {
                        "error_type": "invalid_shot_duration",
                        "retryable": False,
                        "render_mode": render_mode,
                        "error": str(exc),
                    }
                    return ToolResult("vimax_render_video", False, str(exc), refusal)
            # Validate the entire selected image scope before cutover, status changes,
            # destructive shot redo, or any media request. This gate is mandatory even
            # when the user has opted out of portrait/keyframe approval.
            if stop_after == "sets":
                try:
                    validate_set_reference_generation(
                        working_dir / render_mode,
                        image_generator,
                        style,
                        set_ids=set_ids,
                        redo_sets=redo_sets,
                        set_candidate_count=set_candidate_count,
                    )
                except (RuntimeError, ValueError) as exc:
                    refusal = {
                        "error_type": "set_generation_refused",
                        "retryable": False,
                        "render_mode": render_mode,
                        "stage": "sets",
                        "error": str(exc),
                    }
                    self.session_index.update_stage(session_id, "error", refusal["error"])
                    _write_render_status(working_dir, status="error", payload={**refusal, "session_id": session_id})
                    return ToolResult("vimax_render_video", False, refusal["error"], refusal)
            else:
                acceptance = None if bool(args.get("allow_unlocked", False)) else _acceptance_refusal(working_dir, render_mode, stop_after, only_shots)
                if acceptance is not None:
                    self.session_index.update_stage(session_id, "error", acceptance["error"])
                    _write_render_status(working_dir, status="error", payload={**acceptance, "session_id": session_id})
                    return ToolResult("vimax_render_video", False, acceptance["error"], acceptance)
                supports_last_frame = getattr(video_generator, "supports_last_frame", None)
                requires_last_frame = True if supports_last_frame is None else bool(await supports_last_frame())
                set_acceptance = _set_image_acceptance_refusal(
                    working_dir,
                    render_mode,
                    stop_after,
                    str(getattr(image_generator, "model", "") or "unknown-image-model"),
                    style,
                    only_shots,
                    redo,
                    requires_last_frame=requires_last_frame,
                )
                if set_acceptance is not None:
                    self.session_index.update_stage(session_id, "error", set_acceptance["error"])
                    _write_render_status(working_dir, status="error", payload={**set_acceptance, "session_id": session_id})
                    return ToolResult("vimax_render_video", False, set_acceptance["error"], set_acceptance)
                rendered_ending_refusal = await _rendered_ending_preflight_refusal(
                    working_dir,
                    render_mode,
                    stop_after,
                    image_generator,
                    only_shots,
                    redo,
                )
                if rendered_ending_refusal is not None:
                    self.session_index.update_stage(session_id, "error", rendered_ending_refusal["error"])
                    _write_render_status(
                        working_dir, status="error",
                        payload={**rendered_ending_refusal, "session_id": session_id},
                    )
                    return ToolResult(
                        "vimax_render_video",
                        False,
                        rendered_ending_refusal["error"],
                        rendered_ending_refusal,
                    )

            model_change = _enforce_render_sequence(
                working_dir,
                image_generator=image_generator,
                video_generator=video_generator,
                style=style,
                allow_model_change=allow_model_change,
                render_mode=render_mode,
            )
            if model_change is not None:
                self.session_index.update_stage(session_id, "error", model_change["error"])
                _write_render_status(working_dir, status="error", payload={**model_change, "session_id": session_id})
                return ToolResult("vimax_render_video", False, model_change["error"], model_change)

            self.session_index.update_stage(session_id, "rendering", f"Rendering {stop_after} phase")
            _write_render_status(working_dir, status="rendering", payload={
                "session_id": session_id, "render_mode": render_mode, "phase": stop_after,
                "redone_shots": sorted(redo.get("slots", {})), "set_ids": list(set_ids),
            })
            revision_notes = _revision_notes_for(working_dir, render_mode, redo)
            if redo:
                ending_changes = [
                    slot for slot, stage in redo["slots"].items()
                    if stage in {"keyframes", "clips"}
                ]
                if ending_changes:
                    _invalidate_continuity_dependents(working_dir, render_mode, ending_changes)
                removed = _clear_redo_targets(working_dir, render_mode, redo["slots"])
                _mark_slots_redone(working_dir, render_mode, redo["slots"], revision_notes)
            if runtime:
                runtime.emit_progress(f"Starting {stop_after} render phase", stage="rendering", metadata={"session_id": session_id, "stop_after": stop_after, "redone_shots": sorted(redo.get("slots", {}))})
            if render_mode == "idea2video":
                idea_pipeline = Idea2VideoPipeline(chat_model=chat_model, image_generator=image_generator, video_generator=video_generator, working_dir=str(working_dir / "idea2video"))
                with _suppress_pipeline_output():
                    outcome = await idea_pipeline(idea=str(session.get("idea", "")), user_requirement=str(session.get("user_requirement", "")), style=style, quiet=True, stop_after=stop_after, revision_notes=revision_notes, progress=_pipeline_progress(runtime, session_id, tally=tally), only_shots=only_shots, set_ids=set_ids, redo_sets=redo_sets, set_candidate_count=set_candidate_count)
                return self._render_outcome_result(runtime, session_id, working_dir, outcome, render_mode="idea2video", redo=redo, removed=removed, tally=tally)
            if render_mode == "script2video":
                script_dir = working_dir / "script2video"
                script_text = _load_script_text(working_dir)
                characters = _load_characters(script_dir / "characters.json")
                pipeline = Script2VideoPipeline(chat_model=chat_model, image_generator=image_generator, video_generator=video_generator, working_dir=str(script_dir))
                with _suppress_pipeline_output():
                    outcome = await pipeline(script=script_text, user_requirement=str(session.get("user_requirement", "")), style=style, characters=characters, quiet=True, progress=_pipeline_progress(runtime, session_id, tally=tally), stop_after=stop_after, revision_notes=revision_notes, only_shots=script_only_shots, set_ids=set_ids, redo_sets=redo_sets, set_candidate_count=set_candidate_count)
                return self._render_outcome_result(runtime, session_id, working_dir, outcome, render_mode="script2video", redo=redo, removed=removed, tally=tally)
            if render_mode == "novel2video":
                with _suppress_pipeline_output():
                    render_result = await novel_pipeline.render_video_artifacts(
                        style=style,
                        user_requirement=str(session.get("user_requirement", "")),
                        quiet=True,
                        progress=_pipeline_progress(runtime, session_id, tally=tally),
                        set_candidate_count=set_candidate_count,
                        stop_after=stop_after,
                        set_ids=set_ids,
                        redo_sets=redo_sets,
                        chat_model=chat_model,
                    )
                if isinstance(render_result, RenderOutcome):
                    return self._render_outcome_result(runtime, session_id, working_dir, render_result, render_mode="novel2video", redo=redo, removed=removed, tally=tally)
                scene_videos_dir = Path(render_result["scene_videos_dir"])
                payload = {
                    "session_id": session_id,
                    "render_mode": "novel2video",
                    "phase": render_result.get("phase", "video"),
                    "phase_gating": True,
                    "render_started": True,
                    "render_completed": True,
                    "scene_render_completed": True,
                    "final_video_path": None,
                    "scene_videos_dir": str(scene_videos_dir.relative_to(self.workspace_root)),
                    "scene_video_dirs": [str(Path(path).relative_to(self.workspace_root)) for path in render_result.get("scene_video_dirs", [])],
                    "scene_count": render_result.get("scene_count", 0),
                    "generated": tally.get("generated", 0),
                    "reused": tally.get("reused", 0),
                    "missing": [],
                }
                self.session_index.update_stage(session_id, "novel_scene_rendered", "Novel scene videos rendered")
                _write_render_status(working_dir, status="rendered", payload=payload)
                return ToolResult("vimax_render_video", True, json.dumps(payload, ensure_ascii=False, indent=2), payload)
        except Exception as exc:
            unwrapped = _unwrap_retry_error(exc)
            error_text = _sanitize_error_text(str(unwrapped))
            wrapped_error_text = _sanitize_error_text(str(exc))
            restored = _restore_redo_backup(working_dir, render_mode)
            # A render that fails on a bare message ("list index out of range") costs the
            # person reading it the chance to see where: the frames of the traceback are
            # carried in the payload, where the trail keeps them.
            where = traceback.format_exc()
            self.session_index.update_stage(session_id, "error", f"Render failed: {error_text}")
            checklist = self.session_index.artifact_checklist(session_id)
            portrait_missing = []
            if isinstance(unwrapped, PortraitGenerationError):
                portrait_root = Path(unwrapped.working_dir) if unwrapped.working_dir else working_dir / render_mode
                for missing_path in unwrapped.missing:
                    absolute = (portrait_root / missing_path).resolve()
                    portrait_missing.append(absolute.relative_to(working_dir.resolve()).as_posix())
            payload = {
                "error_type": "render_failed",
                "retryable": _is_retryable_render_error(unwrapped),
                "session_id": session_id,
                **({"redone_shots": sorted(redo.get("slots", {})), "cleared": len(removed), "cleared_paths": removed} if redo else {}),
                **({"restored": restored} if restored else {}),
                "error": error_text,
                "wrapped_error": wrapped_error_text,
                "present": [path for path, present in checklist.items() if present],
                "missing": portrait_missing if isinstance(unwrapped, PortraitGenerationError) else [path for path, present in checklist.items() if not present],
                **({
                    "phase": "portraits",
                    "render_mode": render_mode,
                    "render_started": True,
                    "render_completed": False,
                    "awaiting_confirmation": None,
                    "generated": tally.get("generated", 0),
                    "reused": tally.get("reused", 0),
                    "failed_characters": {name: _sanitize_error_text(error) for name, error in unwrapped.failures.items()},
                } if isinstance(unwrapped, PortraitGenerationError) else {}),
            }
            _write_render_status(working_dir, status="error", payload=payload)
            if runtime:
                detail = (
                    f" The redraw was rolled back: {len(restored)} cleared artifact(s) were put back, so the shot still holds what it held before."
                    if restored
                    else " Partial artifacts were kept."
                )
                runtime.emit_progress(f"Render failed.{detail}", stage="render_failed", metadata=payload)
            return ToolResult(
                "vimax_render_video",
                False,
                f"Render failed: {error_text}"
                + (f"\nThe redraw was rolled back — {len(restored)} artifact(s) it had cleared were restored, so nothing was lost. Fix the cause and redraw again." if restored else ""),
                payload,
            )
        payload = {"error_type": "dependency_missing", "session_id": session_id}
        _write_render_status(working_dir, status="dependency_missing", payload=payload)
        return ToolResult("vimax_render_video", False, "No render mode matched current session.", payload)

    @_project_locked
    async def vimax_review_timeline(self, args: dict[str, Any], runtime: ToolRuntimeContext | None = None) -> ToolResult:
        """Check the timeline against the script, and record what is missing.

        The deterministic checks always run. The narrative pass needs the language
        model and is skipped with a note when it is unavailable, so the exact gaps
        are still recorded rather than the whole review failing.
        """
        session_id = str(args.get("session_id", "") or "").strip()
        if not session_id:
            session_id = str((self.session_index.active() or {}).get("session_id") or "")
        if not session_id:
            return ToolResult("vimax_review_timeline", False, "No active session to review.", {"error_type": "missing_session"})

        working_dir = self.session_index.working_dir(session_id)
        try:
            render_mode = _resolve_render_mode(
                self.session_index.artifact_checklist(session_id),
                _read_render_manifest(working_dir),
                str(args.get("render_mode", "") or ""),
            )
        except ValueError as error:
            return ToolResult(
                "vimax_review_timeline",
                False,
                str(error),
                {"error_type": "invalid_render_mode", "session_id": session_id},
            )

        root_dir = working_dir / render_mode
        script = _load_root_script(working_dir, render_mode)
        if not script.strip():
            return ToolResult(
                "vimax_review_timeline",
                False,
                f"No script found for {render_mode}, so the timeline cannot be checked against it.",
                {"error_type": "missing_script", "session_id": session_id, "render_mode": render_mode},
            )
        from pipelines.continuity_review import build_evidence, film_order, normalize_review, run_checks, write_review

        camera_tree = _read_json_list(root_dir / "camera_tree.json")
        active_shots = film_order(camera_tree)
        if not active_shots:
            return ToolResult(
                "vimax_review_timeline",
                False,
                f"{render_mode} has no shots to review.",
                {"error_type": "no_shots", "session_id": session_id, "render_mode": render_mode},
            )

        characters_path = root_dir / "characters.json"
        characters = [character.model_dump() for character in _load_characters(characters_path)] if characters_path.exists() else []
        rows_by_idx = _storyboard_by_idx(root_dir / "storyboard.json")
        removed_rows_by_idx = _removed_rows_by_idx(root_dir)
        plan_rows_by_idx = _plan_rows_by_idx(root_dir)
        clip_seconds = _clip_seconds(self.workspace_root)
        requirement = str((self.session_index.get(session_id) or {}).get("user_requirement") or "")

        checks = run_checks(
            script=script,
            requirement=requirement,
            characters=characters,
            camera_tree=camera_tree,
            rows_by_idx=rows_by_idx,
            removed_rows_by_idx=removed_rows_by_idx,
            plan_rows_by_idx=plan_rows_by_idx,
            clip_seconds=clip_seconds,
        )
        warnings = [check for check in checks if check.status == "warn"]
        if runtime:
            runtime.emit_progress(
                f"Checked {len(active_shots)} shots against the script: {len(warnings)} finding(s).",
                stage="review_checked",
                metadata={"session_id": session_id, "render_mode": render_mode},
            )

        reviewed_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        input_files = _coverage_input_files(working_dir, root_dir, render_mode)
        notes = ""
        try:
            chat_model = _build_chat_model()
            evidence = build_evidence(
                script=script,
                requirement=requirement,
                characters=characters,
                camera_tree=camera_tree,
                rows_by_idx=rows_by_idx,
                removed_rows_by_idx=removed_rows_by_idx,
                plan_rows_by_idx=plan_rows_by_idx,
                checks=checks,
                clip_seconds=clip_seconds,
            )
            answer = await _complete_with_chat_model(chat_model, _timeline_review_prompt(evidence))
            review = normalize_review(
                _parse_review_payload(answer), root=render_mode, active_shots=active_shots, reviewed_at=reviewed_at
            )
        except Exception as exc:
            notes = (
                f"The narrative pass did not run: {_sanitize_error_text(str(exc))}. "
                "The findings below are exact and unaffected — re-run the review for the beats and suggested shots."
            )
            review = normalize_review({}, root=render_mode, active_shots=active_shots, reviewed_at=reviewed_at)

        stored = {
            **review,
            "session_id": session_id,
            "checks": [check.as_dict() for check in checks],
            "notes": notes,
            "input_files": input_files,
            "user_requirement": requirement,
        }
        review_path = str(Path(write_review(str(working_dir), stored)).relative_to(self.workspace_root))

        gaps = [check for check in warnings if check.id == "dialogue_uncovered"]
        missing_beats = [beat for beat in review["beats"] if beat["status"] != "covered"]
        lines = [
            f"Script coverage for {render_mode}: {len(active_shots)} shots, "
            f"{len(review['beats'])} beat(s) judged, {len(gaps)} uncovered dialogue line(s)."
        ]
        lines.extend(f"  missing dialogue: {gap.message}" for gap in gaps)
        lines.extend(
            f"  {beat['status']} beat: {beat['text']}" + (f" — {beat['note']}" if beat["note"] else "")
            for beat in missing_beats
        )
        lines.extend(
            f"  suggested shot after {suggestion['after_shot']}: {suggestion['title'] or suggestion['visual_desc'][:80]}"
            for suggestion in review["suggestions"]
        )
        if notes:
            lines.append(notes)
        lines.append(f"Recorded in {review_path}; the Timeline shows these gaps and can add each suggested shot.")

        payload = {
            "session_id": session_id,
            "render_mode": render_mode,
            "reviewed_at": review["reviewed_at"],
            "gaps": [gap.message for gap in gaps],
            "warnings": [check.as_dict() for check in warnings],
            "beats": review["beats"],
            "suggestions": review["suggestions"],
            "notes": notes,
            "review_path": review_path,
        }
        return ToolResult("vimax_review_timeline", True, "\n".join(lines), payload)

    def _render_outcome_result(self, runtime: ToolRuntimeContext | None, session_id: str, working_dir: Path, outcome: Any, *, render_mode: str, redo: dict[str, Any] | None = None, removed: list[str] | None = None, tally: dict[str, Any] | None = None) -> ToolResult:
        """Report a render phase, and what the user must confirm before the next one.

        The artifacts a redraw held are only dropped here, once the phase has run: a phase
        that raises instead keeps them for the rollback.
        """
        stills = [str(Path(path).relative_to(self.workspace_root)) for path in outcome.stills]
        set_references = [str(Path(path).relative_to(self.workspace_root)) for path in outcome.set_references]
        generated = (tally or {}).get("generated", 0)
        reused = (tally or {}).get("reused", 0)
        reused_set_paths: list[str] = []
        if outcome.phase == "sets":
            generated_set_paths = {
                str(Path(path))
                for path in (tally or {}).get("generated_set_candidate_paths", [])
            }
            generated = sum(str(Path(path)) in generated_set_paths for path in outcome.set_references)
            reused_set_paths = [
                str(Path(path).relative_to(self.workspace_root))
                for path in outcome.set_references
                if str(Path(path)) not in generated_set_paths
            ]
            reused = len(reused_set_paths)
            if runtime:
                for path in reused_set_paths:
                    runtime.emit_progress(
                        f"Reusing set reference: {Path(path).parent.parent.name}",
                        stage="set_reference_exists",
                        metadata={"session_id": session_id, "phase": "sets", "set_id": Path(path).parent.parent.name, "path": path},
                    )
        warnings = list((tally or {}).get("warnings") or [])
        payload = {
            "generated": generated,
            "reused": reused,
            **({"redone_shots": sorted((redo or {}).get("slots", {})), "cleared": len(removed or [])} if redo else {}),
            **({"warnings": warnings} if warnings else {}),
            "session_id": session_id,
            "render_mode": render_mode,
            "phase": outcome.phase,
            "style": outcome.style,
            "image_model": outcome.image_model,
            "video_model": outcome.video_model,
            "stills": stills,
            "set_references": set_references,
            "awaiting_confirmation": outcome.awaiting_confirmation,
            "render_started": True,
            "render_completed": outcome.phase == "video" and bool(outcome.final_video_path),
            "final_video_path": str(Path(outcome.final_video_path).relative_to(self.workspace_root)) if outcome.final_video_path else None,
            "missing": [],
        }
        if outcome.awaiting_confirmation:
            next_phase = outcome.awaiting_confirmation
            review = {
                "portraits": "the character portraits and the style/direction they establish",
                "sets": "the character-free set/location references and their layout continuity",
                "stills": "the keyframes that video generation will animate",
            }.get(outcome.phase, "the generated images")
            review_count = len(set_references) if outcome.phase == "sets" else len(stills)
            content = (
                f"{outcome.phase} phase complete: {review_count} image(s) ready for review, rendered with style {outcome.style!r} "
                f"and image model {outcome.image_model!r}.\n"
                f"Show the user {review}, then ask whether to continue to the {next_phase} phase. "
                f"Only call vimax_render_video again with stop_after=\"{next_phase}\" after the user approves. "
                f"If they want changes, revise the prompt or plan and re-run this phase instead."
            )
            if removed and any("transition_video_from_shot_" in path for path in removed):
                content += (
                    "\nThis shot's first frame is taken from the transition video that leads into it, so the redraw "
                    "re-rendered that video too — a video-model call, not just an image. Its first frame is now the new "
                    "camera still."
                )
            if outcome.phase == "sets" and generated == 0:
                content += "\nEvery selected set reference was reused; no image generation was needed."
            elif generated == 0 and redo:
                content += (
                    "\nWARNING: this run regenerated nothing — every artifact the redraw was meant to replace is still "
                    "the one on disk. Check that the shot keys you passed exist and that the phase redraws their stage."
                )
            elif generated == 0 and not redo:
                content += (
                    "\nNote: this phase regenerated nothing, because every artifact it produces is already on disk. If the "
                    "user asked for a shot to be redrawn, pass redo_shots=[...] with the slot keys from the Timeline; a plain "
                    "phase run skips what exists."
                )
            self.session_index.update_stage(session_id, f"{outcome.phase}_ready", f"{outcome.phase.capitalize()} ready for review")
            _write_render_status(working_dir, status="rendering", payload={**payload, "awaiting_confirmation": next_phase})
            if runtime:
                runtime.emit_progress(
                    f"{outcome.phase.capitalize()} ready for review",
                    stage=f"{outcome.phase}_ready",
                    metadata={**payload, "awaiting_confirmation": next_phase},
                )
            if redo:
                _discard_redo_backup(working_dir)
            return ToolResult("vimax_render_video", True, content, payload)
        if outcome.phase == "video" and not outcome.final_video_path:
            content = "Video rendering made progress, but at least one scene is incomplete; the final film is not ready yet."
            self.session_index.update_stage(session_id, "rendering", "Video clips rendered; final film is not complete")
            _write_render_status(working_dir, status="rendering", payload=payload)
            if runtime:
                runtime.emit_progress(content, stage="video_progress", metadata=payload)
            if redo:
                _discard_redo_backup(working_dir)
            return ToolResult("vimax_render_video", True, content, payload)


        self.session_index.update_stage(session_id, "rendered", "Final video rendered")
        _write_render_status(working_dir, status="rendered", payload=payload)
        if runtime:
            runtime.emit_progress("Render complete", stage="rendered", metadata=payload)
        if redo:
            _discard_redo_backup(working_dir)
        return ToolResult("vimax_render_video", True, json.dumps(payload, ensure_ascii=False, indent=2), payload)

    def _resolve_session(self, session_id: str, *, idea: str, script: str, user_requirement: str, style: str) -> dict[str, Any]:
        requested_source = idea or script
        if session_id:
            session = self.session_index.get(session_id)
            if session is None:
                session = self.session_index.create(idea=requested_source, user_requirement=user_requirement, style=style, session_id=session_id)
            elif requested_source and _is_new_source_for_session(session, requested_source):
                session = self.session_index.create(idea=requested_source, user_requirement=user_requirement, style=style)
            else:
                self.session_index.set_active(session_id)
        else:
            if requested_source:
                active = self.session_index.active()
                if active is not None and self._session_is_empty(active):
                    session = self.session_index.set_active(active["session_id"])
                else:
                    session = self.session_index.create(idea=requested_source, user_requirement=user_requirement, style=style)
            else:
                session = self.session_index.active() or self.session_index.create(idea=requested_source, user_requirement=user_requirement, style=style)
        self._update_session_metadata(session["session_id"], idea=requested_source, user_requirement=user_requirement, style=style)
        return self.session_index.get(session["session_id"]) or session

    def _session_is_empty(self, session: dict[str, Any]) -> bool:
        if str(session.get("idea") or "").strip():
            return False
        session_id = str(session.get("session_id") or "").strip()
        if not session_id:
            return False
        return not any(self.session_index.artifact_checklist(session_id).values())

    def _update_session_metadata(self, session_id: str, *, idea: str, user_requirement: str, style: str) -> None:
        data = self.session_index.load()
        record = data.get("sessions", {}).get(session_id)
        if not isinstance(record, dict):
            return
        if idea and not record.get("idea"):
            record["idea"] = idea
        if user_requirement:
            record["user_requirement"] = user_requirement
        if style:
            record["style"] = style
        self.session_index.save(data)


class _DiscardStream:
    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        pass


_PIPELINE_OUTPUT_SINK = _DiscardStream()


@contextmanager
def _suppress_pipeline_output():
    previous_disable_level = logging.root.manager.disable
    logging.disable(logging.WARNING)
    try:
        with redirect_stdout(_PIPELINE_OUTPUT_SINK), redirect_stderr(_PIPELINE_OUTPUT_SINK):
            yield
    finally:
        logging.disable(previous_disable_level)


def _narrative_step_timeout_seconds() -> float:
    raw = os.environ.get("VIMAX_NARRATIVE_STEP_TIMEOUT_SECONDS", "900")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 900.0


async def _run_planning_step(
    message: str,
    stage: str,
    awaitable: Any,
    runtime: ToolRuntimeContext | None,
    metadata: dict[str, Any] | None = None,
) -> Any:
    timeout_seconds = _narrative_step_timeout_seconds()
    event_metadata = dict(metadata or {})
    event_metadata["timeout_seconds"] = timeout_seconds
    if runtime:
        runtime.emit_progress(message, stage=stage, metadata=event_metadata)
        await asyncio.sleep(0)
    try:
        with _suppress_pipeline_output():
            if timeout_seconds <= 0:
                return await awaitable
            return await asyncio.wait_for(awaitable, timeout=timeout_seconds)
    except asyncio.TimeoutError as exc:
        raise RuntimeError(f"{message} timed out after {timeout_seconds:g}s") from exc
    except Exception as exc:
        cause = _unwrap_retry_error(exc)
        detail = _sanitize_error_text(str(cause))
        raise RuntimeError(f"{message} failed: {type(cause).__name__}: {detail}") from cause


def _is_new_source_for_session(session: dict[str, Any], requested_source: str) -> bool:
    current = str(session.get("idea") or "").strip()
    requested = requested_source.strip()
    if not current or not requested:
        return False
    return current != requested


def _llm_request_timeout_seconds() -> float:
    raw = os.environ.get("VIMAX_LLM_REQUEST_TIMEOUT_SECONDS", "300")
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 300.0


def _narrative_max_tokens() -> int:
    raw = os.environ.get("VIMAX_NARRATIVE_MAX_TOKENS", "16384")
    try:
        return max(256, int(raw))
    except ValueError:
        return 16384


# Progress stages the pipeline uses for artifacts it drew, and for ones it reused. A phase
# that regenerated nothing is a no-op, and reporting it as success is how a redraw that
# never happened looks like one.
_GENERATED_STAGES = ("frame_done", "video_clip_done", "character_portrait_front_done", "character_portrait_side_done", "character_portrait_back_done", "set_reference_done")
_REUSED_STAGES = ("frame_exists", "video_clip_exists", "character_portrait_front_exists", "character_portrait_side_exists", "character_portrait_back_exists", "set_reference_exists")


def _pipeline_progress(runtime: ToolRuntimeContext | None, session_id: str, *, scene_index: int | None = None, tally: dict[str, Any] | None = None):
    """The pipeline's progress sink.

    A tally is still kept when there is no runtime to report to: whether a phase actually
    drew anything is a fact about the run, not about who was listening, and reading it as
    zero would have the render claim it regenerated nothing when it did.
    """
    if runtime is None and tally is None:
        return None

    def emit(stage: str, message: str, metadata: dict[str, Any] | None = None) -> None:
        if tally is not None:
            if stage in _GENERATED_STAGES:
                tally["generated"] = tally.get("generated", 0) + 1
                if stage == "set_reference_done" and metadata and metadata.get("path"):
                    tally.setdefault("generated_set_candidate_paths", []).append(str(metadata["path"]))
            elif stage in _REUSED_STAGES:
                tally["reused"] = tally.get("reused", 0) + 1
            elif stage == "references_dropped":
                # The render carried on without the character references. That belongs in the
                # result, not only in the progress stream: the frames are not the ones the plan
                # describes, and whoever reads the trail has to be able to see why.
                tally.setdefault("warnings", []).append(str(message))
        if runtime is None:
            return
        payload = dict(metadata or {})
        payload["session_id"] = session_id
        if scene_index is not None:
            payload["scene_index"] = scene_index
        runtime.emit_progress(message, stage=stage, metadata=payload)

    return emit


def _build_chat_model() -> Any:
    from langchain.chat_models import init_chat_model

    api_key = llm_api_key()
    if not api_key:
        raise RuntimeError("VIMAX_LLM_API_KEY or configs/agent.local.yaml llm.api_key is required for narrative planning")
    return init_chat_model(
        model=llm_model(),
        model_provider=llm_model_provider(),
        api_key=api_key,
        base_url=llm_base_url(),
        timeout=_llm_request_timeout_seconds(),
        max_retries=0,
        max_completion_tokens=_narrative_max_tokens(),
    )


def _build_image_generator() -> ImageGeneratorNanobananaYunwuAPI | ImageGeneratorOpenRouterAPI | ImageGeneratorAgnesAPI:
    from tools.image_generator_nanobanana_yunwu_api import ImageGeneratorNanobananaYunwuAPI
    from tools.image_generator_openrouter_api import ImageGeneratorOpenRouterAPI

    api_key = image_api_key()
    if not api_key:
        raise RuntimeError("VIMAX_IMAGE_API_KEY, VIMAX_LLM_API_KEY, or configs/agent.local.yaml image/llm api_key is required for image generation")
    model = image_model()
    base_url = image_base_url()
    resolution = image_resolution()
    if api_provider_from_base_url(base_url) == "openrouter":
        return ImageGeneratorOpenRouterAPI(api_key=api_key, model=model, base_url=base_url, resolution=resolution)
    if api_provider_from_base_url(base_url) == "agnes":
        from tools.image_generator_agnes_api import ImageGeneratorAgnesAPI

        return ImageGeneratorAgnesAPI(api_key=api_key, model=model, base_url=base_url, resolution=resolution)
    return ImageGeneratorNanobananaYunwuAPI(api_key=api_key, model=model, base_url=base_url, resolution=resolution)


def _build_video_generator(workspace_root: str | Path = "."):
    policy = video_duration_policy(workspace_root)
    api_key = video_api_key(workspace_root)
    if not api_key:
        raise RuntimeError("VIMAX_VIDEO_API_KEY, VIMAX_LLM_API_KEY, or configs/agent.local.yaml video/llm api_key is required for video generation")
    model = video_model(workspace_root)
    base_url = video_base_url(workspace_root)
    provider = policy["provider"]
    common = {
        "api_key": api_key,
        "model": model,
        "base_url": base_url,
        "clip_seconds": policy["defaultSeconds"],
    }
    if provider == "openrouter":
        from tools.video_generator_openrouter_api import VideoGeneratorOpenRouterAPI

        generator = VideoGeneratorOpenRouterAPI(
            **common,
            resolution=video_resolution(workspace_root),
            generate_audio=video_generate_audio(workspace_root),
        )
    elif provider == "yunwu":
        from tools.video_generator_veo_yunwu_api import VideoGeneratorVeoYunwuAPI

        generator = VideoGeneratorVeoYunwuAPI(api_key=api_key, t2v_model=model, ff2v_model=model, base_url=base_url)
    elif provider == "agnes":
        from tools.video_generator_agnes_api import VideoGeneratorAgnesAPI

        generator = VideoGeneratorAgnesAPI(**common, resolution=video_resolution(workspace_root))
    elif provider == "ltx":
        from tools.video_generator_ltx_api import VideoGeneratorLTXAPI

        generator = VideoGeneratorLTXAPI(
            **common,
            resolution=video_resolution(workspace_root),
            generate_audio=video_generate_audio(workspace_root),
        )
    else:
        raise RuntimeError(f"Unsupported video provider '{provider}' for base URL: {base_url}")
    generator.duration_policy = policy
    return generator


class _IdentityRewriter:
    async def __call__(self, prompt: str) -> str:
        return prompt


def _build_embedding_model() -> Any:
    from langchain_openai import OpenAIEmbeddings

    api_key = embedding_api_key()
    base_url = embedding_base_url()
    provider = embedding_model_provider().strip().lower()
    if not api_key or not base_url:
        raise RuntimeError("VIMAX_EMBEDDING_API_KEY or configs/agent.local.yaml embedding api_key/base_url is required for novel planning")
    if provider != "openai":
        raise RuntimeError(f"Unsupported embedding model_provider: {provider}")
    return OpenAIEmbeddings(model=embedding_model(), api_key=api_key, base_url=base_url)


def _build_reranker() -> RerankerBgeSiliconapi:
    from tools.reranker_bge_silicon_api import RerankerBgeSiliconapi

    api_key = reranker_api_key()
    base_url = reranker_base_url()
    if not api_key or not base_url:
        raise RuntimeError("VIMAX_RERANKER_API_KEY or configs/agent.local.yaml reranker api_key/base_url is required for novel planning")
    return RerankerBgeSiliconapi(api_key=api_key, base_url=base_url, model=reranker_model())


def _build_novel_pipeline(working_dir: Path) -> Novel2MoviePipeline:
    from agents.event_extractor import EventExtractor
    from agents.global_information_planner import GlobalInformationPlanner
    from agents.novel_compressor import NovelCompressor
    from agents.scene_extractor import SceneExtractor
    from pipelines.novel2movie_pipeline import Novel2MoviePipeline

    api_key = llm_api_key()
    if not api_key:
        raise RuntimeError("VIMAX_LLM_API_KEY or configs/agent.local.yaml llm.api_key is required for novel planning")
    base_url = llm_base_url()
    model = llm_model()
    dummy = _UnavailableGenerator()
    return Novel2MoviePipeline(
        novel_compressor=NovelCompressor(api_key=api_key, base_url=base_url, chat_model=model),
        event_extractor=EventExtractor(api_key=api_key, base_url=base_url, chat_model=model),
        embeddings=_build_embedding_model(),
        rerank_model=_build_reranker(),
        scene_extractor=SceneExtractor(api_key=api_key, base_url=base_url, chat_model=model),
        global_information_planner=GlobalInformationPlanner(api_key=api_key, base_url=base_url, chat_model=model),
        image_generator=dummy,
        rewriter=_IdentityRewriter(),
        script2video_pipeline=dummy,
        working_dir=str(working_dir),
    )


def _build_novel_render_pipeline(working_dir: Path, chat_model: Any, image_generator: Any, video_generator: Any) -> Novel2MoviePipeline:
    from agents.event_extractor import EventExtractor
    from agents.global_information_planner import GlobalInformationPlanner
    from agents.novel_compressor import NovelCompressor
    from agents.scene_extractor import SceneExtractor
    from pipelines.novel2movie_pipeline import Novel2MoviePipeline
    from pipelines.script2video_pipeline import Script2VideoPipeline

    api_key = llm_api_key()
    if not api_key:
        raise RuntimeError("VIMAX_LLM_API_KEY or configs/agent.local.yaml llm.api_key is required for novel rendering")
    base_url = llm_base_url()
    model = llm_model()
    script_pipeline = Script2VideoPipeline(chat_model=chat_model, image_generator=image_generator, video_generator=video_generator, working_dir=str(working_dir / "videos"))
    return Novel2MoviePipeline(
        novel_compressor=NovelCompressor(api_key=api_key, base_url=base_url, chat_model=model),
        event_extractor=EventExtractor(api_key=api_key, base_url=base_url, chat_model=model),
        embeddings=_build_embedding_model(),
        rerank_model=_build_reranker(),
        scene_extractor=SceneExtractor(api_key=api_key, base_url=base_url, chat_model=model),
        global_information_planner=GlobalInformationPlanner(api_key=api_key, base_url=base_url, chat_model=model),
        image_generator=image_generator,
        rewriter=_IdentityRewriter(),
        script2video_pipeline=script_pipeline,
        working_dir=str(working_dir),
    )


def _unwrap_retry_error(exc: Exception) -> Exception:
    from tenacity import RetryError

    if isinstance(exc, RetryError):
        try:
            return exc.last_attempt.exception() or exc
        except Exception:
            return exc
    return exc


def _is_retryable_render_error(exc: Exception) -> bool:
    text = str(exc).lower()
    if isinstance(exc, AttributeError):
        return False
    if isinstance(exc, FileNotFoundError) or "no such file or directory" in text:
        # A missing reference image or input file is still missing on the next
        # attempt: retrying only re-fails and burns an agent round trip.
        return False
    if "http 400" in text:
        # The provider rejected the request itself (content filter, unsupported
        # option). The generators already retry transient statuses internally, so
        # an identical re-render cannot succeed.
        return False
    if "http 403" in text or "key limit exceeded" in text or "quota" in text:
        return False
    if "http 402" in text or "insufficient credits" in text:
        # The account is out of credit: the same call fails until someone adds more.
        return False
    return True


def _sanitize_error_text(text: str) -> str:
    sanitized = text
    for marker in ("Bearer ", "token "):
        if marker in sanitized:
            prefix, rest = sanitized.split(marker, 1)
            key_id = []
            for char in rest:
                if char.isalnum() or char in "-_":
                    key_id.append(char)
                    continue
                break
            sanitized = prefix + marker + "<redacted>" + rest[len(key_id):]
    if "sk-" in sanitized:
        prefix, rest = sanitized.split("sk-", 1)
        token = []
        for char in rest:
            if char.isalnum() or char in "-_":
                token.append(char)
                continue
            break
        sanitized = prefix + "sk-<redacted>" + rest[len(token):]
    return sanitized


def _write_render_status(working_dir: Path, *, status: str, payload: dict[str, Any]) -> None:
    working_dir.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "status": status,
        **payload,
    }
    (working_dir / "render_status.json").write_text(json.dumps(event, ensure_ascii=False, indent=2), encoding="utf-8")
    with (working_dir / "render_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


RENDER_MANIFEST_FILENAME = "render_manifest.json"


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _pin_render_mode(working_dir: Path, render_mode: str) -> None:
    """Record which root a session's plan lives in, so renders stop guessing."""
    manifest = _read_render_manifest(working_dir)
    if not render_mode or str(manifest.get("render_mode") or "") == render_mode:
        return
    manifest["render_mode"] = render_mode
    (working_dir / RENDER_MANIFEST_FILENAME).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_planning_mode(requested: str, manifest: dict[str, Any], *, existing_script: str) -> str:
    """Which root narrative planning should fill in.

    An explicit choice wins, then the root the sequence is pinned to, and only then
    what the session already holds: a script in the script root means the session is
    being worked in script mode, whatever idea it may have started from.
    """
    known = ("idea2video", "script2video")
    if requested:
        if requested not in known:
            raise ValueError(f"render_mode must be one of {list(known)}, got {requested!r}")
        return requested
    pinned = str(manifest.get("render_mode") or "")
    if pinned in known:
        return pinned
    return "script2video" if existing_script else "idea2video"


def _resolve_render_mode(checklist: dict[str, bool], manifest: dict[str, Any], requested: str) -> str:
    """Which root this render writes to.

    A session can hold more than one complete plan — an abandoned first attempt
    beside the one being worked in — and the roots are interchangeable to the
    readiness check, so picking the first ready one silently renders the wrong
    plan (and spends money on it). An explicit choice wins, then the root the
    sequence is already pinned to, and only then the first ready root.
    """
    ready = {
        "idea2video": _idea_mode_ready(checklist),
        "script2video": _script_mode_ready(checklist),
        "novel2video": _novel_mode_ready(checklist),
    }
    if requested:
        if requested not in ready:
            raise ValueError(f"render_mode must be one of {sorted(ready)}, got {requested!r}")
        if not ready[requested]:
            raise ValueError(f"render_mode={requested!r} was requested but that root has no plan yet: {_missing_render_dependencies(checklist)}")
        return requested
    pinned = str(manifest.get("render_mode") or "")
    if pinned and ready.get(pinned):
        return pinned
    for mode in ("idea2video", "script2video", "novel2video"):
        if ready[mode]:
            return mode
    raise ValueError("No render mode matched current session.")


def _read_render_manifest(working_dir: Path) -> dict[str, Any]:
    path = working_dir / RENDER_MANIFEST_FILENAME
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}



def _discard_image_model_outputs(working_dir: Path, render_mode: str, old_model: str) -> None:
    """Remove generated images for a cut-over model without deleting set descriptors."""
    root_dir = working_dir / render_mode if render_mode else working_dir
    model_slug = safe_path_component(old_model)

    def remove(path: Path) -> None:
        if path.is_symlink() or not path.is_dir():
            path.unlink(missing_ok=True)
        else:
            shutil.rmtree(path)

    portrait_dir = root_dir / "character_portraits" / model_slug
    if portrait_dir.exists() or portrait_dir.is_symlink():
        remove(portrait_dir)
    for shot_dir in shot_directories(root_dir).values():
        image_dir = shot_dir / model_slug
        if image_dir.exists() or image_dir.is_symlink():
            remove(image_dir)

    set_root = root_dir / "set_references"
    if set_root.is_dir() and not set_root.is_symlink():
        for set_dir in set_root.iterdir():
            if not set_dir.is_dir() or set_dir.is_symlink():
                continue
            image_dir = set_dir / model_slug
            if image_dir.exists() or image_dir.is_symlink():
                remove(image_dir)

    store = _read_acceptance(working_dir)
    root_record = store.get(render_mode) if isinstance(store.get(render_mode), dict) else {}
    set_records = root_record.get("sets") if isinstance(root_record.get("sets"), dict) else {}
    timestamp = datetime.now(timezone.utc).isoformat()
    changed = False
    for record in set_records.values():
        if isinstance(record, dict) and (record.get("accepted_at") or record.get("artifacts")):
            record["invalidated_at"] = timestamp
            record["invalidated_reason"] = "image_model_changed"
            changed = True
    if changed:
        _acceptance_path(working_dir).write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")


def _enforce_render_sequence(
    working_dir: Path,
    *,
    image_generator: Any,
    video_generator: Any,
    style: str,
    allow_model_change: bool,
    render_mode: str = "",
    commit: bool = True,
):
    """Keep one image model and one style per render sequence.

    Artifacts are reused by ``os.path.exists``, so rendering a sequence with a
    second image model would stitch two looks together, and rendering it under a
    new style would silently keep stills drawn in the old one. The model
    directories keep two models' output apart on disk; this keeps the sequence
    itself consistent. Returns an error payload when the caller must decide, else
    records what the sequence is pinned to and returns None.
    """
    manifest = _read_render_manifest(working_dir)
    recorded_mode = str(manifest.get("render_mode") or "")
    if recorded_mode and render_mode and recorded_mode != render_mode and not allow_model_change:
        return {
            "error_type": "render_mode_changed",
            "retryable": False,
            "render_mode": render_mode,
            "recorded_render_mode": recorded_mode,
            "error": (
                f"This sequence renders from {recorded_mode}/ but the render was asked to use {render_mode}/. "
                f"A session can hold the plan of an abandoned attempt beside the current one, and rendering both into one sequence "
                f"would mix them. Pass render_mode={recorded_mode!r}. Moving the sequence to {render_mode}/ is a decision for the user "
                f"(allow_model_change=true), never a way around one refused call."
            ),
        }
    recorded_model = str(manifest.get("image_model") or "")
    configured_model = str(getattr(image_generator, "model", "") or "")
    recorded_style = str(manifest.get("style") or "")
    if recorded_model and configured_model and recorded_model != configured_model and not allow_model_change:
        return {
            "error_type": "image_model_changed",
            "retryable": False,
            "image_model": configured_model,
            "recorded_image_model": recorded_model,
            "error": (
                f"This sequence was rendered with image model {recorded_model!r} but the configured image model is {configured_model!r}. "
                f"One image model per sequence keeps the stills visually consistent. "
                f"Set image.model back to {recorded_model!r}, start a new session, or pass allow_model_change=true to render the remaining artifacts with {configured_model!r}."
            ),
        }
    if recorded_style and style and recorded_style != style:
        return {
            "error_type": "style_changed",
            "retryable": False,
            "style": style,
            "recorded_style": recorded_style,
            "error": (
                f"This sequence was rendered with style {recorded_style!r}, but the project style is now {style!r}. "
                f"Portraits, set references, keyframes and clips drawn under the old style cannot be reused. "
                f"Change the style from the project page, which discards and regenerates the affected artifacts, "
                f"or set the style back to {recorded_style!r}."
            ),
        }
    # Validate refusal conditions before planning or approval gates. The committing
    # pass below performs all filesystem cutovers only after every gate has passed.
    if not commit:
        return None
    if recorded_model and configured_model and recorded_model != configured_model and allow_model_change:
        _discard_image_model_outputs(working_dir, render_mode, recorded_model)
    # A different video model renders the clips again, so the old concatenated film
    # cannot remain current. Do this only after all refusal checks have passed.
    recorded_video_model = str(manifest.get("video_model") or "")
    configured_video_model = str(getattr(video_generator, "model", "") or "")
    if recorded_video_model and configured_video_model and recorded_video_model != configured_video_model:
        film = (working_dir / render_mode if render_mode else working_dir) / "final_video.mp4"
        film.unlink(missing_ok=True)
    updated = {
        "image_model": configured_model or recorded_model,
        "video_model": str(getattr(video_generator, "model", "") or ""),
        "style": style or recorded_style,
        "render_mode": render_mode or recorded_mode,
        "clip_seconds": int(getattr(video_generator, "clip_seconds", 0) or manifest.get("clip_seconds") or 0),
        "created_at": str(manifest.get("created_at") or datetime.now().isoformat(timespec="seconds")),
    }
    if updated != manifest:
        (working_dir / RENDER_MANIFEST_FILENAME).write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
    return None


def _write_characters_if_missing(path: Path, characters: list[CharacterInScene]) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([character.model_dump() for character in characters], ensure_ascii=False, indent=2), encoding="utf-8")


def _load_characters(path: Path) -> list[CharacterInScene]:
    from interfaces import CharacterInScene
    return [CharacterInScene.model_validate(item) for item in json.loads(path.read_text(encoding="utf-8"))]


def _load_script_text(working_dir: Path) -> str:
    script_text = working_dir / "script2video" / "script.txt"
    if script_text.exists():
        return script_text.read_text(encoding="utf-8")
    idea_script = working_dir / "idea2video" / "script.json"
    if idea_script.exists():
        payload = json.loads(idea_script.read_text(encoding="utf-8"))
        return json.dumps(payload, ensure_ascii=False, indent=2) if not isinstance(payload, str) else payload
    story = working_dir / "idea2video" / "story.txt"
    if story.exists():
        return story.read_text(encoding="utf-8")
    return ""


def _resolve_artifact_path(working_dir: Path, revision_target: str) -> Path:
    rel = Path(revision_target)
    if rel.is_absolute():
        raise ValueError(f"revision_target must be relative to session working_dir: {revision_target}")
    path = (working_dir / rel).resolve()
    if path != working_dir and working_dir not in path.parents:
        raise ValueError(f"revision_target escapes session working_dir: {revision_target}")
    return path


async def _complete_with_chat_model(chat_model: Any, prompt: str) -> str:
    """One text completion, unwrapped from whatever shape the client returns."""
    if hasattr(chat_model, "ainvoke"):
        response = await chat_model.ainvoke(prompt)
    elif hasattr(chat_model, "invoke"):
        response = chat_model.invoke(prompt)
    else:
        raise RuntimeError("chat_model does not support invoke/ainvoke")
    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = "".join(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in content)
    return _strip_markdown_fences(str(content).strip())


async def _revise_artifact_with_llm(chat_model: Any, target: str, current_text: str, instruction: str) -> str:
    prompt = (
        "Revise this ViMax structured artifact exactly as requested. "
        "Return only the complete replacement file content, with no Markdown fences or explanation. "
        "If the file is JSON, preserve valid JSON and the existing schema shape.\n\n"
        "When writing shot descriptions or frame/motion prompts, use existing canonical character names, "
        "not appearance-based labels or repeated profiles. Use concise present-tense, filmable action, "
        "clear location/framing, and named speaker cues with verbatim dialogue; "
        "parentheticals are for essential delivery/action only. Preserve untouched content.\n\n"
        f"Target: {target}\n"
        f"Revision instruction: {instruction}\n\n"
        "Current file content:\n"
        f"{current_text}"
    )
    return await _complete_with_chat_model(chat_model, prompt)


def _strip_markdown_fences(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _load_root_script(working_dir: Path, render_mode: str) -> str:
    """The script of one root, never another root's.

    ``_load_script_text`` falls back across roots, so a session holding both an
    abandoned idea-mode attempt and a script-mode film would hand the review the
    wrong script — and checking the film against its own script is the point.
    """
    if render_mode == "script2video":
        return _read_text(working_dir / "script2video" / "script.txt")
    idea_script = working_dir / "idea2video" / "script.json"
    if idea_script.exists():
        try:
            payload = json.loads(idea_script.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ""
        return payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, indent=2)
    return _read_text(working_dir / "idea2video" / "story.txt")


def _coverage_input_files(working_dir: Path, root_dir: Path, render_mode: str) -> dict[str, str | None]:
    """Fingerprint every source file the persisted coverage review can depend on."""
    candidates = {
        root_dir / "characters.json",
        root_dir / "camera_tree.json",
        root_dir / "storyboard.json",
    }
    if render_mode == "script2video":
        candidates.add(root_dir / "script.txt")
    else:
        # script.json is preferred by _load_root_script; story.txt is its fallback.
        candidates.update({root_dir / "script.json", root_dir / "story.txt"})

    scene_dirs = sorted(
        path for path in root_dir.glob("scene_*")
        if path.is_dir() and path.name.removeprefix("scene_").isdigit()
    )
    for scene_dir in scene_dirs:
        candidates.update({scene_dir / "camera_tree.json", scene_dir / "storyboard.json"})
        shot_root = scene_dir / "shots"
        if shot_root.is_dir():
            candidates.update(
                shot_dir / "shot_description.json"
                for shot_dir in shot_root.iterdir()
                if shot_dir.is_dir()
            )

    flat_shots = root_dir / "shots"
    if flat_shots.is_dir():
        candidates.update(
            shot_dir / "shot_description.json"
            for shot_dir in flat_shots.iterdir()
            if shot_dir.is_dir()
        )

    removed_root = root_dir / ".removed_shots"
    if removed_root.is_dir():
        candidates.update(
            slot_dir / "brief.json"
            for slot_dir in removed_root.iterdir()
            if slot_dir.is_dir()
        )

    fingerprints: dict[str, str | None] = {}
    for path in sorted(candidates):
        relative = path.relative_to(working_dir).as_posix()
        try:
            fingerprints[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            fingerprints[relative] = None
    return fingerprints


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    """A JSON array of objects, or nothing when the file is missing or malformed."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [row for row in payload if isinstance(row, dict)] if isinstance(payload, list) else []


def _storyboard_by_idx(path: Path) -> dict[int, dict[str, Any]]:
    """The film's briefs, keyed by shot number."""
    return {
        int(row["idx"]): row
        for row in _read_json_list(path)
        if str(row.get("idx", "")).lstrip("-").isdigit()
    }


def _plan_rows_by_idx(root_dir: Path) -> dict[int, dict[str, Any]]:
    """Each shot's plan, which lists the characters it shows by index.

    Reuses ``_shot_dirs`` so both the flat ``shots/<n>/`` and the
    ``scene_<i>/shots/<n>/`` layouts are covered.
    """
    rows: dict[int, dict[str, Any]] = {}
    for key, shot_dir in _shot_dirs(root_dir):
        try:
            payload = json.loads((shot_dir / "shot_description.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        idx = payload.get("idx", key)
        if str(idx).lstrip("-").isdigit():
            rows[int(idx)] = payload
    return rows


def _preflight_duration_overrides(
    root_dir: Path,
    only_shots: list[str] | None,
    policy: dict[str, Any],
) -> None:
    if not isinstance(policy, dict):
        policy = {
            "provider": "active video provider",
            "model": "",
            "configured": False,
            "minSeconds": 1,
            "maxSeconds": 0,
            "allowedSeconds": None,
        }
    selected = {str(slot) for slot in only_shots} if only_shots is not None else None
    for slot, shot_dir in _shot_dirs(root_dir):
        if selected is not None and slot not in selected:
            continue
        try:
            payload = json.loads((shot_dir / "shot_description.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read shot {slot} duration settings: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("duration_seconds") is None:
            continue
        error = validate_video_shot_duration(payload["duration_seconds"], policy)
        if error:
            raise ValueError(f"Shot {slot}: {error}")


def _removed_rows_by_idx(root_dir: Path) -> dict[int, dict[str, Any]]:
    """The briefs of shots that were cut.

    A removed shot's dialogue is no longer spoken by the film, which is exactly
    the gap the review looks for, so these are read separately rather than
    treated as part of the timeline.
    """
    rows: dict[int, dict[str, Any]] = {}
    removed_dir = root_dir / ".removed_shots"
    if not removed_dir.exists():
        return rows
    for slot_dir in sorted(removed_dir.glob("*")):
        brief = slot_dir / "brief.json"
        if not brief.exists():
            continue
        try:
            payload = json.loads(brief.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and str(payload.get("idx", "")).lstrip("-").isdigit():
            rows[int(payload["idx"])] = payload
    return rows


def _clip_seconds(workspace_root: str | Path = ".") -> int:
    """The default video duration after applying the active model's duration policy."""
    return int(video_duration_policy(workspace_root)["defaultSeconds"])


def _timeline_review_prompt(evidence: dict[str, Any]) -> str:
    """The narrative pass: what the script asks for against what the film plays.

    The cut shots are included because a beat is usually missing precisely
    because its shot was removed, and the active timeline alone cannot show that.
    """
    return (
        "You are checking a film's timeline against its script.\n"
        "The timeline is the shots in playing order, each with what it shows and the dialogue it carries.\n\n"
        "Return JSON only, with no Markdown fences and no explanation, in this shape:\n"
        '{"summary": "...", '
        '"beats": [{"text": "...", "status": "covered|partial|missing", "covered_by": [shot numbers], "note": "..."}], '
        '"suggestions": [{"after_shot": shot number, "title": "...", "visual_desc": "...", "audio_desc": "...", '
        '"motion_desc": "...", "characters": ["..."], "rationale": "...", "frames": {"first": "...", "last": "..."}}]}\n\n'
        "Rules:\n"
        "- beats: the story's beats in order. covered_by lists the shots that play each beat; an empty list means missing.\n"
        "- A beat can keep its dialogue while losing the action that set it up. Mark that partial and say so in the note.\n"
        "- suggestions: one shot per missing or partial beat, placed with after_shot — the shot number it should follow, or the last shot number to append.\n"
        "- after_shot must be a shot that is in the timeline above, never the number of a shot that was cut: anchor a suggestion to the shot the gap now follows.\n"
        "- motion_desc describes what moves in the shot, in the same style as the timeline's motion descriptions. Write it for this shot: an added shot takes nothing from the shot it is placed after.\n"
        "- A suggestion that restores a lost dialogue line must carry it verbatim in audio_desc, copied from the script or from the cut shot's audio. Use an empty audio_desc when the shot speaks nothing.\n"
        "- characters must be names from the character list, spelled exactly as given there.\n"
        "- Use canonical character names throughout visual_desc, frames, and motion_desc, retaining angle brackets in visual descriptions; never replace names with long appearance descriptions or repeat profiles.\n"
        "- Follow screenplay/shot-list conventions: concise present-tense, filmable action; clear location and framing; named dialogue cues with verbatim lines; parentheticals only for essential delivery/action. Keep only brief shot-essential costume or prop details.\n"
        "- visual_desc is a shot description in the same style as the timeline's.\n"
        "- Match the framing and shot size the shots around the gap use, and honour every constraint the direction puts on shot size, framing or style: never propose a shot size the direction rules out, and never introduce a framing the surrounding shots do not use.\n"
        "- frames.first and frames.last describe its opening and closing images, at the same level of detail as the timeline's frames, and keep the characters' appearance as the neighbouring shots show it.\n"
        "- Suggest nothing for beats that are already covered, and invent no characters or beats the script does not have.\n\n"
        "Findings already computed from the timeline (exact; do not restate them as suggestions unless they are real gaps):\n"
        f"{json.dumps(evidence['deterministic_findings'], ensure_ascii=False)}\n\n"
        f"Script:\n{evidence['script']}\n\n"
        f"Direction: {evidence['requirement']}\n\n"
        f"Characters:\n{json.dumps(evidence['characters'], ensure_ascii=False)}\n\n"
        f"Timeline, in playing order:\n{json.dumps(evidence['timeline'], ensure_ascii=False)}\n\n"
        f"Shots that were cut and are not in the film:\n{json.dumps(evidence['cut_shots'], ensure_ascii=False)}\n"
    )


def _parse_review_payload(text: str) -> dict[str, Any]:
    """The model's review as JSON, tolerating fences and surrounding prose."""
    cleaned = _strip_markdown_fences(str(text or "").strip())
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("the narrative pass did not return JSON") from None
        payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("the narrative pass did not return a JSON object")
    return payload


def _last_traceback_frames(formatted: str, keep: int = 6) -> str:
    """The tail of a traceback: where it happened, not the machinery around it."""
    lines = [line for line in str(formatted or "").splitlines() if line.strip()]
    return "\n".join(lines[-keep:])


def _stale_keys_for_revision(target: str) -> list[str]:
    if "storyboard.json" in target:
        return ["shot_descriptions", "camera_tree", "frames", "clips", "final_video"]
    if "shot_description.json" in target:
        return ["frames", "clips", "final_video"]
    if "camera_tree.json" in target:
        return ["frames", "clips", "final_video"]
    if target.endswith("script.json") or target.endswith("story.txt"):
        return ["storyboard", "shot_descriptions", "camera_tree", "frames", "clips", "final_video"]
    if target.endswith("characters.json"):
        return ["storyboard", "shot_descriptions", "frames", "clips", "final_video"]
    return ["frames", "clips", "final_video"]


def _ready_for_render(checklist: dict[str, bool]) -> bool:
    return _idea_mode_ready(checklist) or _script_mode_ready(checklist) or _novel_mode_ready(checklist)


ACCEPTANCE_FILENAME = "render_acceptance.json"


def _acceptance_path(working_dir: Path) -> Path:
    return working_dir / ACCEPTANCE_FILENAME


def _read_acceptance(working_dir: Path) -> dict[str, Any]:
    path = _acceptance_path(working_dir)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}

FRAME_CONTINUITY_FILENAME = "frame_continuity.json"


def _continuity_enabled(working_dir: Path, root: str | None = None) -> bool:
    try:
        settings = json.loads((working_dir / FRAME_CONTINUITY_FILENAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        settings = {}
    if isinstance(settings, dict) and settings.get("mode") == "chained_keyframes":
        return True
    if root is None:
        return False
    root_dir = working_dir / root
    order = _continuity_playback_order(root_dir)
    for slot in order[1:]:
        parts = slot.split("/")
        shot_dir = root_dir / parts[0] / "shots" / parts[1] if parts[0].startswith("scene_") else root_dir / "shots" / parts[0]
        try:
            description = json.loads((shot_dir / "shot_description.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(description, dict) and description.get("reuse_previous_last_frame") is True:
            return True
    return False


def _continuity_playback_order(root_dir: Path) -> list[str]:
    """The active film order: numeric camera order, then each camera's stored shot order."""
    scene_dirs = sorted(
        (shots_dir.parent for shots_dir in root_dir.glob("scene_*/shots")),
        key=lambda scene: _shot_sort_key(scene.name.split("_", 1)[1]),
    )
    containers = (
        [(scene.name, scene / "shots", scene / "camera_tree.json") for scene in scene_dirs]
        if scene_dirs else [("", root_dir / "shots", root_dir / "camera_tree.json")]
    )

    order: list[str] = []
    for scene, _shots_dir, tree_path in containers:
        try:
            cameras = json.loads(tree_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(cameras, list):
            continue
        for camera in sorted(
            (camera for camera in cameras if isinstance(camera, dict)),
            key=lambda camera: _shot_sort_key(str(camera.get("idx", ""))),
        ):
            active = camera.get("active_shot_idxs")
            if not isinstance(active, list):
                continue
            for shot in active:
                slot = f"{scene}/{shot}" if scene else str(shot)
                if slot not in order:
                    order.append(slot)
    return order


def _continuity_camera_by_slot(root_dir: Path) -> dict[str, tuple[str, str]]:
    """Map active playback slots to their scene/camera identity."""
    scene_dirs = sorted(
        (shots_dir.parent for shots_dir in root_dir.glob("scene_*/shots")),
        key=lambda scene: _shot_sort_key(scene.name.split("_", 1)[1]),
    )
    containers = (
        [(scene.name, scene / "camera_tree.json") for scene in scene_dirs]
        if scene_dirs else [("", root_dir / "camera_tree.json")]
    )
    result = {}
    for scene, tree_path in containers:
        try:
            cameras = json.loads(tree_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(cameras, list):
            continue
        for camera in cameras:
            if not isinstance(camera, dict) or not isinstance(camera.get("active_shot_idxs"), list):
                continue
            for shot_idx in camera["active_shot_idxs"]:
                slot = f"{scene}/{shot_idx}" if scene else str(shot_idx)
                result[slot] = (scene, str(camera.get("idx", "")))
    return result


def _continuity_target_mode(working_dir: Path, root: str, order: list[str], position: int, cameras) -> str | None:
    if position <= 0:
        return None
    target = order[position]
    parts = target.split("/")
    shot_dir = (
        working_dir / root / parts[0] / "shots" / parts[1]
        if parts[0].startswith("scene_") else working_dir / root / "shots" / parts[0]
    )
    try:
        description = json.loads((shot_dir / "shot_description.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        description = {}
    reuse = description.get("reuse_previous_last_frame") if isinstance(description, dict) else None
    if reuse is False:
        return None
    if reuse is True:
        return "reuse"
    try:
        settings = json.loads((working_dir / FRAME_CONTINUITY_FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        settings = {}
    global_chain = isinstance(settings, dict) and settings.get("mode") == "chained_keyframes"
    if not global_chain:
        return None
    previous = order[position - 1]
    same_camera = cameras.get(previous) == cameras.get(target) and cameras.get(previous, (None,))[0] == cameras.get(target, (None,))[0]
    return "reuse" if same_camera else "reframe"


async def _rendered_ending_preflight_refusal(
    working_dir: Path,
    root: str,
    phase: str,
    image_generator: Any,
    only_shots: list[int | str] | None,
    redo: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Reject a chain before a media provider call unless its accepted clip handoff is current."""
    if phase not in {"stills", "video"} or not _continuity_enabled(working_dir, root):
        return None
    from utils.video_assembly import RENDERED_ENDING_NORMALIZATION, prepare_rendered_ending

    root_dir = working_dir / root
    order = _continuity_playback_order(root_dir)
    if len(order) < 2:
        return None
    cameras = _continuity_camera_by_slot(root_dir)
    scope = {str(slot) for slot in only_shots} if only_shots is not None else None
    redo_slots = (redo or {}).get("slots") or {}
    acceptance = _read_acceptance(working_dir)
    root_acceptance = acceptance.get(root) if isinstance(acceptance.get(root), dict) else {}
    shot_records = root_acceptance.get("shots") if isinstance(root_acceptance.get("shots"), dict) else {}
    image_slug = safe_path_component(str(getattr(image_generator, "model", "") or "unknown-image-model"))
    for position, target_slot in enumerate(order):
        if scope is not None and target_slot not in scope:
            continue
        target_dir_parts = target_slot.split("/")
        target_dir = (
            root_dir / target_dir_parts[0] / "shots" / target_dir_parts[1]
            if target_dir_parts[0].startswith("scene_")
            else root_dir / "shots" / target_dir_parts[0]
        )
        historical_clip = (
            phase == "video"
            and bool(_shot_stage_paths(target_dir, "clips"))
            and target_slot not in redo_slots
        )
        mode = _continuity_target_mode(working_dir, root, order, position, cameras)
        if mode is None:
            continue
        source_slot = order[position - 1]
        source_parts = source_slot.split("/")
        source_dir = (
            root_dir / source_parts[0] / "shots" / source_parts[1]
            if source_parts[0].startswith("scene_")
            else root_dir / "shots" / source_parts[0]
        )
        source_record = shot_records.get(source_slot, {}) if isinstance(shot_records, dict) else {}
        clip_record = source_record.get("clips") if isinstance(source_record, dict) else None
        if (
            not isinstance(clip_record, dict)
            or not clip_record.get("accepted_at")
            or clip_record.get("invalidated_at")
            or clip_record.get("rejected_at")
            or not _lock_matches(working_dir, clip_record)
        ):
            _invalidate_continuity_dependents(working_dir, root, [source_slot])
            if historical_clip:
                continue
            return {
                "error_type": "rendered_ending_source_unavailable",
                "retryable": False,
                "render_mode": root,
                "slot": target_slot,
                "previous_slot": source_slot,
                "error": (
                    f"Cannot prepare the rendered ending for {target_slot}: predecessor clip {source_slot} "
                    "is missing, changed, or not accepted. Render and accept that predecessor clip first; "
                    "no hidden predecessor generation was attempted."
                ),
            }
        source_clips = [
            (working_dir / str(item.get("path") or "")).resolve()
            for item in clip_record.get("artifacts", [])
            if isinstance(item, dict)
            and str(item.get("path") or "").endswith("/video.mp4")
        ]
        clip_path = next(
            (path for path in source_clips if path.is_file() and path.is_relative_to(source_dir.resolve())),
            None,
        )
        if clip_path is None:
            _invalidate_continuity_dependents(working_dir, root, [source_slot])
            if historical_clip:
                continue
            return {
                "error_type": "rendered_ending_source_unavailable",
                "retryable": False,
                "render_mode": root,
                "slot": target_slot,
                "previous_slot": source_slot,
                "error": f"Accepted predecessor clip {source_slot} has no current video.mp4 artifact.",
            }
        source_image_dir = source_dir / image_slug
        first_anchor = source_image_dir / "first_frame.png"
        last_anchor = source_image_dir / "last_frame.png"
        try:
            ending = await asyncio.to_thread(
                prepare_rendered_ending,
                str(clip_path),
                first_frame=str(first_anchor) if first_anchor.is_file() else None,
                last_frame=str(last_anchor) if last_anchor.is_file() else None,
                width=1280,
                height=720,
            )
        except Exception as exc:
            _invalidate_continuity_dependents(working_dir, root, [source_slot])
            if historical_clip:
                continue
            return {
                "error_type": "rendered_ending_source_unavailable",
                "retryable": False,
                "render_mode": root,
                "slot": target_slot,
                "previous_slot": source_slot,
                "error": f"Cannot decode the accepted predecessor clip {source_slot}: {exc}",
            }
        if ending.get("normalization") != RENDERED_ENDING_NORMALIZATION:
            _invalidate_continuity_dependents(working_dir, root, [source_slot])
            if historical_clip:
                continue
            return {
                "error_type": "rendered_ending_source_unavailable",
                "retryable": False,
                "render_mode": root,
                "slot": target_slot,
                "previous_slot": source_slot,
                "error": (
                    f"Predecessor clip {source_slot} produced unsupported rendered-ending normalization "
                    f"{ending.get('normalization')!r}."
                ),
            }
        first_path = target_dir / image_slug / "first_frame.png"
        selector_path = target_dir / "first_frame_selector_output.json"
        try:
            provenance = json.loads(selector_path.read_text(encoding="utf-8")).get("continuity_source")
            first_hash = hashlib.sha256(first_path.read_bytes()).hexdigest()
        except (OSError, ValueError, AttributeError):
            provenance = None
            first_hash = None
        project_relative = lambda path: Path(path).resolve().relative_to(working_dir.resolve()).as_posix()
        expected = {
            "kind": "rendered_clip_end",
            "shot_idx": int(source_parts[-1]),
            "slot": source_slot,
            "path": project_relative(ending["path"]),
            "sha256": ending["sha256"],
            "clip_path": project_relative(ending["clip_path"]),
            "clip_sha256": ending["clip_sha256"],
            "normalization": ending["normalization"],
            "normalization_inputs": ending["normalization_inputs"],
            "frame_type": "rendered_end",
            "mode": mode,
        }
        if not first_path.is_file() or provenance != expected or (mode == "reuse" and first_hash != ending["sha256"]):
            if isinstance(provenance, dict) and provenance.get("kind") == "rendered_clip_end" and provenance != expected:
                _invalidate_continuity_dependents(working_dir, root, [source_slot])
            if historical_clip:
                continue
            message = (
                "The first frame still has legacy planned-still provenance."
                if isinstance(provenance, dict) and provenance.get("frame_type") == "last_frame"
                else "The current rendered ending has not been prepared for this first frame."
            )
            return {
                "error_type": "rendered_ending_preparation_required",
                "retryable": False,
                "render_mode": root,
                "slot": target_slot,
                "previous_slot": source_slot,
                "previous_clip_sha256": ending["clip_sha256"],
                "error": (
                    f"Cannot generate {target_slot}: {message} Explicitly prepare the rendered ending from "
                    f"accepted predecessor clip {source_slot} and review the resulting first frame before "
                    "rendering."
                ),
            }
    return None


def _invalidate_continuity_dependents(
    working_dir: Path,
    root: str,
    changed_slots: list[str] | tuple[str, ...] | set[str] | dict[str, Any] | str | int,
) -> list[str]:
    """Invalidate direct actual-ending dependents after a predecessor source changes."""
    if not _continuity_enabled(working_dir, root):
        return []
    if isinstance(changed_slots, dict):
        changed = {str(slot) for slot in changed_slots}
    elif isinstance(changed_slots, (str, int)):
        changed = {str(changed_slots)}
    else:
        changed = {str(slot) for slot in changed_slots}
    root_dir = working_dir / root
    order = _continuity_playback_order(root_dir)
    cameras = _continuity_camera_by_slot(root_dir)
    dependents = []
    for index, source_slot in enumerate(order[:-1]):
        if source_slot not in changed:
            continue
        target_slot = order[index + 1]
        if target_slot in changed:
            continue
        mode = _continuity_target_mode(working_dir, root, order, index + 1, cameras)
        target_parts = target_slot.split("/")
        target_dir = (
            root_dir / target_parts[0] / "shots" / target_parts[1]
            if target_parts[0].startswith("scene_")
            else root_dir / "shots" / target_parts[0]
        )
        try:
            provenance = json.loads(
                (target_dir / "first_frame_selector_output.json").read_text(encoding="utf-8")
            ).get("continuity_source")
        except (OSError, ValueError, AttributeError):
            provenance = None
        if mode is not None and isinstance(provenance, dict) and provenance.get("kind") == "rendered_clip_end":
            dependents.append(target_slot)

    path = _acceptance_path(working_dir)
    store = _read_acceptance(working_dir)
    root_store = store.get(root) if isinstance(store.get(root), dict) else {}
    shots_store = root_store.get("shots") if isinstance(root_store.get("shots"), dict) else {}
    now = datetime.now().isoformat(timespec="seconds")
    changed_any = False
    for slot in dependents:
        shot_store = shots_store.get(slot) if isinstance(shots_store.get(slot), dict) else {}
        position = order.index(slot)
        source = order[position - 1] if position else ""
        reason = (
            f"Rendered ending from shot {source} changed; re-review this shot's keyframes and clip "
            "to confirm the continuous handoff."
        )
        for stage in ("keyframes", "clips"):
            record = shot_store.get(stage)
            if not isinstance(record, dict):
                continue
            shot_store[stage] = {**record, "invalidated_at": now, "invalidation_reason": reason}
            changed_any = True
        if shot_store:
            shots_store[slot] = shot_store

    final_record = root_store.get("final_video")
    if isinstance(final_record, dict) and changed:
        root_store["final_video"] = {
            **final_record,
            "invalidated_at": now,
            "invalidation_reason": "A predecessor clip or ending changed in the continuous film; rebuild and review the final assembly.",
        }
        changed_any = True

    if changed_any:
        root_store["shots"] = shots_store
        store[root] = root_store
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(store, ensure_ascii=False, indent=4), encoding="utf-8")
        temporary.replace(path)
    return dependents


def _mark_continuity_records(working_dir: Path, root: str, slots: list[str], reason: str) -> None:
    """Preserve reviewed records while requiring a fresh review after a plan revision."""
    if not _continuity_enabled(working_dir, root):
        return
    path = _acceptance_path(working_dir)
    store = _read_acceptance(working_dir)
    root_store = store.get(root) if isinstance(store.get(root), dict) else {}
    shots_store = root_store.get("shots") if isinstance(root_store.get("shots"), dict) else {}
    now = datetime.now().isoformat(timespec="seconds")
    changed = False
    for slot in slots:
        shot_store = shots_store.get(slot) if isinstance(shots_store.get(slot), dict) else {}
        for stage in ("keyframes", "clips"):
            record = shot_store.get(stage)
            if not isinstance(record, dict):
                continue
            shot_store[stage] = {**record, "invalidated_at": now, "invalidation_reason": reason}
            changed = True
        if shot_store:
            shots_store[slot] = shot_store
    final_record = root_store.get("final_video")
    if isinstance(final_record, dict):
        root_store["final_video"] = {
            **final_record,
            "invalidated_at": now,
            "invalidation_reason": "A shot plan changed in the continuous film; rebuild and review the final assembly.",
        }
        changed = True
    if changed:
        root_store["shots"] = shots_store
        store[root] = root_store
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(store, ensure_ascii=False, indent=4), encoding="utf-8")
        temporary.replace(path)


def _revisioned_shot_plan(relative_path: str) -> tuple[str, str] | None:
    parts = Path(relative_path).parts
    if (
        len(parts) == 4 and parts[0] in {"script2video", "idea2video"}
        and parts[1] == "shots" and parts[3] == "shot_description.json"
    ):
        return parts[0], parts[2]
    if (
        len(parts) == 5 and parts[0] == "idea2video" and parts[1].startswith("scene_")
        and parts[2] == "shots" and parts[4] == "shot_description.json"
    ):
        return parts[0], f"{parts[1]}/{parts[3]}"
    return None




def _lock_matches(working_dir: Path, record: Any) -> bool:
    """Whether a lock still describes the artifacts on disk."""
    artifacts = (record or {}).get("artifacts") if isinstance(record, dict) else None
    if not isinstance(artifacts, list) or not artifacts:
        return False
    for item in artifacts:
        if not isinstance(item, dict):
            return False
        path = working_dir / str(item.get("path") or "")
        try:
            data = path.read_bytes()
        except OSError:
            return False
        if item.get("size") != len(data) or item.get("sha256") != hashlib.sha256(data).hexdigest():
            return False
    return True


def _locked_stage_paths(root_dir: Path, stage: str) -> list[Path]:
    """The artifacts a stage covers, from disk, sorted."""
    if stage == "portraits":
        return sorted((root_dir / "character_portraits").rglob("*.png"))
    if stage == "final_video":
        # The concatenated film is the root's own final_video.mp4, never a shot clip.
        path = root_dir / "final_video.mp4"
        return [path] if path.exists() else []
    raise ValueError(f"unknown stage {stage!r}")


def _shot_sort_key(name: str) -> tuple[int, int | str]:
    """Shot directory names are numbers, so 2 comes before 10 and not after it."""
    return (0, int(name)) if name.isdigit() else (1, name)


def _shot_dirs(root_dir: Path) -> list[tuple[str, Path]]:
    """Every active shot slot and directory, in the shared set-helper order."""
    return list(shot_directories(root_dir).items())


def _portrait_acceptance_slots(working_dir: Path, root: str, root_record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Review each visible character's selected trio, never archived alternatives."""
    root_dir = working_dir / root
    scopes = [root_dir]
    if root == "idea2video":
        scopes.extend(sorted(path for path in root_dir.glob("scene_*")
                             if path.is_dir() and path.name[6:].isdigit()))
    try:
        manifest = json.loads((working_dir / RENDER_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        manifest = {}
    model = safe_path_component(str(manifest.get("image_model") or "unknown-image-model"))
    records = root_record.get("portrait_characters")
    records = records if isinstance(records, dict) else {}
    slots = {}
    structured = False
    for scope in scopes:
        try:
            characters = json.loads((scope / "characters.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(characters, list):
            continue
        structured = True
        registry_path = scope / "character_portraits" / model / "registry.json"
        # Planned Idea scenes share the global set until a scene registry exists.
        if scope != root_dir and not registry_path.is_file():
            continue
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            registry = {}
        registry = registry if isinstance(registry, dict) else {}
        for character in characters:
            if not isinstance(character, dict) or character.get("is_visible") is not True:
                continue
            name = str(character.get("identifier_in_scene") or "")
            key = f"{scope.relative_to(working_dir).as_posix()}::{name}"
            entry = registry.get(name)
            entry = entry if isinstance(entry, dict) else {}
            canonical = registry_path.parent / f"{character.get('idx')}_{safe_path_component(name)}"
            paths = []
            complete = True
            for view in ("front", "side", "back"):
                item = entry.get(view)
                stored = item.get("path") if isinstance(item, dict) else None
                candidate = Path(stored) if isinstance(stored, str) and stored else None
                if candidate is not None and not candidate.is_absolute():
                    candidate = (working_dir.parent.parent / candidate if stored.startswith(".working_dir/")
                                 else working_dir / candidate)
                if candidate is None or not candidate.resolve().is_relative_to(working_dir.resolve()) or not candidate.is_file():
                    complete = False
                    candidate = canonical / f"{view}.png"
                paths.append(candidate.resolve())
            slots[key] = {"paths": paths, "record": records.get(key), "missing": not complete}
    if not structured:
        return {"portraits": {"paths": _locked_stage_paths(root_dir, "portraits"), "record": root_record.get("portraits")}}

    def recorded_paths(record):
        if not isinstance(record, dict) or not isinstance(record.get("artifacts"), list):
            return set()
        return {(working_dir / str(item.get("path") or "")).resolve()
                for item in record["artifacts"] if isinstance(item, dict)}

    aggregate = root_record.get("portraits")
    required = {path for slot in slots.values() for path in slot["paths"]}
    aggregate_valid = (
        isinstance(aggregate, dict) and not aggregate.get("invalidated_at") and not aggregate.get("rejected_at")
        and required == recorded_paths(aggregate) and _lock_matches(working_dir, aggregate)
        and all(not slot["missing"] for slot in slots.values())
    )
    for key, slot in slots.items():
        record = slot["record"]
        if slot["missing"]:
            slot["record"] = None
        elif aggregate_valid:
            slot["record"] = aggregate
        elif record is not None and set(slot["paths"]) != recorded_paths(record):
            slot["record"] = {**record, "invalidated_at": "portrait_selection_changed"} if isinstance(record, dict) else None
        elif record is None:
            # Scene registries can reference the same approved global portrait tuple.
            name = key.split("::", 1)[1]
            slot["record"] = next((
                item for other_key, item in records.items()
                if other_key.endswith(f"::{name}") and recorded_paths(item) == set(slot["paths"])
                and isinstance(item, dict) and not item.get("invalidated_at") and not item.get("rejected_at")
                and _lock_matches(working_dir, item)
            ), None)
    return slots


def _acceptance_slots(working_dir: Path, root: str) -> dict[str, dict[str, Any]]:
    """Every reviewable slot of a root: {stage: {slot: {"paths": [...], "record": {...}}}}.

    Mirrors the acceptance contract's path rules: portraits are one session-wide slot,
    keyframes and clips are one slot per planned shot, and the final film is its own slot.
    """
    root_dir = working_dir / root
    root_record = _read_acceptance(working_dir).get(root) or {}
    shots_record = root_record.get("shots") if isinstance(root_record.get("shots"), dict) else {}
    slots: dict[str, dict[str, Any]] = {
        "portraits": _portrait_acceptance_slots(working_dir, root, root_record),
        "keyframes": {},
        "clips": {},
        "final_video": {
            "final_video": {"paths": _locked_stage_paths(root_dir, "final_video"), "record": root_record.get("final_video")}
        },
    }
    for slot, shot_dir in _shot_dirs(root_dir):
        shot_records = shots_record.get(slot) if isinstance(shots_record.get(slot), dict) else {}
        slots["keyframes"][slot] = {"paths": _shot_stage_paths(shot_dir, "keyframes"), "record": shot_records.get("keyframes")}
        slots["clips"][slot] = {"paths": _shot_stage_paths(shot_dir, "clips"), "record": shot_records.get("clips")}
    return slots


def _shot_stage_paths(shot_dir: Path, stage: str) -> list[Path]:
    """The artifacts a stage owns for one shot, which is also what a redo clears."""
    if stage == "keyframes":
        return sorted(path for path in shot_dir.rglob("*.png") if path.name in {"first_frame.png", "last_frame.png"})
    return sorted(shot_dir.rglob("video.mp4"))


def _slot_state(working_dir: Path, slot: dict[str, Any]) -> str:
    """planned | rendered | accepted | rejected | stale — from disk, never from the file alone.

    An explicit continuity invalidation keeps its reason and prior decision as evidence but
    requires a fresh review, even when the accepted bytes still match.
    """
    record = slot.get("record") or {}
    if isinstance(record, dict) and record.get("invalidated_at"):
        return "stale"
    if isinstance(record, dict) and record.get("rejected_at"):
        return "rejected"
    if slot.get("missing"):
        return "planned"
    if not slot["paths"]:
        return "planned"
    if not isinstance(record, dict) or not record.get("accepted_at"):
        return "rendered"
    return "accepted" if _lock_matches(working_dir, record) else "stale"


def _unaccepted_slots(working_dir: Path, root: str, stage: str) -> list[tuple[str, str]]:
    """(slot, state) for every slot of a stage that is not accepted.

    A session-scoped slot with nothing on disk is not a blocker: there is nothing to
    review, and the phase about to run is what draws it. A shot-scoped slot with nothing
    on disk *is* a blocker, because rendering the next phase from it would spend money on
    a shot nobody has seen.
    """
    slots = _acceptance_slots(working_dir, root)
    shot_scoped = stage in {"keyframes", "clips"}
    return [
        (slot, _slot_state(working_dir, entry))
        for slot, entry in slots.get(stage, {}).items()
        if _slot_state(working_dir, entry) != "accepted" and (shot_scoped or entry["paths"])
    ]


def _acceptance_refusal(working_dir: Path, root: str, phase: str, only_shots: list[int | str] | None = None) -> dict[str, Any] | None:
    """The reason a phase may not run yet, or None.

    Rendering a phase spends money on the artifacts the previous phase produced, so the
    review is a gate rather than a suggestion: clips wait for accepted keyframes.

    The gate covers what the run will draw and no more. A run that names shots is spending on
    those shots, and holding it for an unrelated shot's unreviewed frames blocks work that has
    nothing to do with them. Only the shot-scoped stages can be narrowed: the portraits and the
    finished film are one thing per sequence, not one per shot.
    """
    gating_stage = {"stills": "portraits", "video": "keyframes"}.get(phase)
    if gating_stage is None:
        return None
    blocked = _unaccepted_slots(working_dir, root, gating_stage)
    if only_shots is not None and gating_stage in ("keyframes", "clips"):
        scope = {str(slot) for slot in only_shots}
        blocked = [(slot, state) for slot, state in blocked if slot in scope]
    if not blocked:
        return None
    listed = ", ".join(f"{slot} ({state})" for slot, state in sorted(blocked))
    rejected = [slot for slot, state in blocked if state == "rejected"]
    rejected_note = (
        f" {len(rejected)} of them were rejected with a reason: regenerate those shots to answer it before spending on this phase."
        if rejected
        else ""
    )
    return {
        "error_type": "acceptance_required",
        "retryable": False,
        "render_mode": root,
        "stage": gating_stage,
        "unaccepted": [{"slot": slot, "state": state} for slot, state in sorted(blocked)],
        "error": (
            f"Nothing may be generated past {gating_stage} until it is accepted: {listed}. "
            f"Review {gating_stage} in the Timeline and accept what is good, then run this phase.{rejected_note} "
            f"Pass allow_unlocked=true to render anyway."
        ),
    }


def _set_image_acceptance_refusal(
    working_dir: Path,
    root: str,
    phase: str,
    image_model: str,
    style: str,
    only_shots: list[int | str] | None,
    redo: dict[str, Any] | None,
    *,
    requires_last_frame: bool = True,
) -> dict[str, Any] | None:
    """Require approved set references only for shots that will redraw images.

    Cached accepted frames can continue into video rendering without making a project
    generate sets solely for the new contract. A keyframe redo always needs fresh set
    references, and ordinary stills/video phases gate only missing frame slots.
    """
    if phase not in {"stills", "video"}:
        return None
    root_dir = working_dir / root
    directories = shot_directories(root_dir)
    scope = list(only_shots) if only_shots is not None else list(directories)
    redo_slots = (redo or {}).get("slots") or {}
    required: list[str] = []
    model_dir = safe_path_component(image_model or "unknown-image-model")
    for value in scope:
        slot = str(value)
        shot_dir = directories.get(slot)
        if shot_dir is None:
            continue
        if redo_slots.get(slot) == "keyframes":
            required.append(slot)
            continue
        image_dir = shot_dir / model_dir
        frame_names = ["first_frame"]
        if requires_last_frame:
            frame_names.append("last_frame")
        if any(not (image_dir / f"{frame}.png").is_file() for frame in frame_names):
            required.append(slot)
    blocked: list[dict[str, str]] = []
    for slot in required:
        try:
            accepted_set_references(root_dir, slot, style, image_model)
        except Exception as exc:
            blocked.append({"slot": slot, "state": "unaccepted", "reason": str(exc)})
    if not blocked:
        return None
    listed = "; ".join(f"{item['slot']}: {item['reason']}" for item in blocked)
    return {
        "error_type": "set_approval_required",
        "retryable": False,
        "render_mode": root,
        "stage": "sets",
        "unaccepted": blocked,
        "required_slots": required,
        "error": (
            f"Shot images cannot be generated or redrawn until every assigned set reference is accepted: {listed}. "
            "Generate/review the assigned set references first. allow_unlocked=true does not bypass this requirement."
        ),
    }


def _rejection_notes(working_dir: Path, root: str) -> dict[str, dict[str, str]]:
    """The guidance typed against each rejected stage, by slot then stage.

    The note is what a redraw of that stage has to answer, so it is read out of the
    review file and carried into the prompt rather than left behind in it.
    """
    root_record = _read_acceptance(working_dir).get(root) or {}
    shots_record = root_record.get("shots") if isinstance(root_record.get("shots"), dict) else {}
    rejected: dict[str, dict[str, str]] = {}
    for slot, record in shots_record.items():
        if not isinstance(record, dict):
            continue
        notes = {
            stage: str(stage_record.get("reason") or "").strip()
            for stage, stage_record in record.items()
            if isinstance(stage_record, dict) and stage_record.get("rejected_at")
        }
        if notes:
            rejected[str(slot)] = notes
    return rejected


# The phase that redraws a stage, and the stages a redraw of it invalidates downstream.
REDO_PHASE: dict[str, str] = {"keyframes": "stills", "clips": "video"}
REDO_INVALIDATES: dict[str, tuple[str, ...]] = {"keyframes": ("keyframes", "clips"), "clips": ("clips",)}


def _redo_stage(slot_entry: dict[str, Any], record: dict[str, Any]) -> str | None:
    """Which stage of a shot a redo works on: the earliest one that is not accepted.

    A recorded rejection names the stage the human was looking at. Without one, the earliest
    stage holding artifacts that nobody has accepted is what a redraw means.

    A stage with no artifacts at all is the *render* case rather than the redraw one: a shot
    added after the last run has no keyframes, and a shot whose frames are accepted has no
    clip. There is nothing to clear, so nothing a mistyped slot could destroy, and the work
    is the artifact that does not exist yet.
    """
    for stage in ("keyframes", "clips"):
        stage_record = record.get(stage) if isinstance(record.get(stage), dict) else {}
        if stage_record.get("rejected_at"):
            return stage
    for stage in ("keyframes", "clips"):
        stage_record = record.get(stage) if isinstance(record.get(stage), dict) else {}
        if slot_entry.get(stage) and not stage_record.get("accepted_at"):
            return stage
    for stage in ("keyframes", "clips"):
        if not slot_entry.get(stage):
            return stage
    return None


def _resolve_redo(working_dir: Path, root: str, args: dict[str, Any]) -> dict[str, Any]:
    """What a redo request names: {"slots": {slot: stage}, "phase": str} or an error.

    ``redo_shots`` names shots explicitly; ``redo_rejected`` takes every shot carrying a
    rejection note. Both are refused before anything is deleted, so a mistyped slot
    cannot clear artifacts that the render then fails to replace.
    """
    requested = args.get("redo_shots")
    if requested in (None, "", []):
        requested = []
    if isinstance(requested, (str, int)):
        requested = [requested]
    redo_rejected = bool(args.get("redo_rejected", False))
    if not requested and not redo_rejected:
        return {}

    slots = _acceptance_slots(working_dir, root)
    records = _read_acceptance(working_dir).get(root) or {}
    shots_record = records.get("shots") if isinstance(records.get("shots"), dict) else {}
    # Each slot as the artifacts it holds per stage: the redo redraws where there are some.
    known = {
        slot: {stage: (slots[stage].get(slot) or {}).get("paths", []) for stage in ("keyframes", "clips")}
        for slot in slots["keyframes"]
    }
    if not known:
        return {
            "error_type": "redo_unsupported",
            "retryable": False,
            "error": f"{root} has no planned shots to redraw.",
        }
    if redo_rejected:
        named = sorted(_rejection_notes(working_dir, root), key=_shot_sort_key)
        if not named:
            return {
                "error_type": "nothing_to_redo",
                "retryable": False,
                "error": "No shot is currently rejected, so there is nothing to redraw. Reject the shots to redraw in the Timeline first, then redo them.",
            }
    else:
        named = []
        for value in requested:
            slot = str(value)
            if slot not in known:
                # A shot removed from the film is not a typo, and saying "unknown" sends the
                # user looking for a mistyped key. It is recoverable, so say how.
                if root and (working_dir / root / ".removed_shots" / slot).is_dir():
                    return {
                        "error_type": "redo_shot_removed",
                        "retryable": False,
                        "requested_slots": [slot],
                        "error": (
                            f"Slot {slot!r} is not in {root}: it was removed from the film, so there is nothing to redraw. "
                            "Restore it in the Timeline first, or redraw a shot the film still has."
                        ),
                    }
                known_sorted = sorted(known, key=_shot_sort_key)
                # Shot directories are numbered from zero and the Timeline numbers what it
                # shows from one, so "Shot 3" is slot "2". Say which key is meant rather
                # than leaving the caller to guess and redraw the wrong shot.
                shifted = str(int(slot) - 1) if slot.isdigit() else ""
                hint = (
                    f" The Timeline shows the shot whose slot key is {shifted!r} as \"Shot {int(slot)}\","
                    f" so that shot is redo_shots=[\"{shifted}\"]."
                    if shifted in known
                    else ""
                )
                return {
                    "error_type": "unknown_redo_shot",
                    "retryable": False,
                    "known_slots": known_sorted,
                    **({"suggested_slot": shifted} if shifted in known else {}),
                    "error": f"Unknown shot {slot!r} in {root}. Known slots: {', '.join(known_sorted)}.{hint}",
                }
            named.append(slot)

    # The one mistake this cannot let through: a redo of the wrong shot. Shot keys are
    # zero-based and the Timeline is one-based, so "Shot 3" read as slot 3 redraws a shot
    # that is not the one with the note — valid, unreviewed, and silently wrong. When a
    # named slot carries no note while other slots do, say which slots do.
    notes_by_slot = _rejection_notes(working_dir, root)
    if notes_by_slot and not bool(args.get("allow_unreviewed_redo", False)):
        ungrounded = [slot for slot in named if slot not in notes_by_slot]
        if ungrounded:
            rejected = ", ".join(
                f"{slot} (shown as \"{_slot_display(slot)}\")" for slot in sorted(notes_by_slot, key=_shot_sort_key)
            )
            listed = ", ".join(ungrounded)
            return {
                "error_type": "redo_shot_not_rejected",
                "retryable": False,
                "requested_slots": ungrounded,
                "rejected_slots": sorted(notes_by_slot, key=_shot_sort_key),
                "error": (
                    f"{listed} {'carry' if len(ungrounded) > 1 else 'carries'} no note against "
                    f"{'them' if len(ungrounded) > 1 else 'it'}, while the rejected "
                    f"{'slots are' if len(notes_by_slot) > 1 else 'slot is'}: {rejected}. "
                    f"Redo one of those instead — redo_shots takes the slot key, and the Timeline name is the key plus one. "
                    f"To redraw {listed} without a note, pass allow_unreviewed_redo=true."
                ),
            }

    targets: dict[str, str] = {}
    for slot in named:
        record = shots_record.get(slot) if isinstance(shots_record.get(slot), dict) else {}
        stage = _redo_stage(known[slot], record)
        if stage is None:
            return {
                "error_type": "nothing_to_redo",
                "retryable": False,
                "error": f"Shot {slot} has no unaccepted artifacts on disk, so there is nothing to redraw.",
            }
        targets[slot] = stage

    phases = {REDO_PHASE[stage] for stage in targets.values()}
    # Keyframes are redrawn before the clips animated from them.
    phase = "stills" if "stills" in phases else "video"
    return {"slots": targets, "phase": phase}


# A redraw moves what it clears here until it has replaced it. Losing the artifact a redraw
# was meant to improve — because the account ran out of credit, say — is worse than not
# redrawing at all, and the render cannot know beforehand whether it will succeed.
REDO_BACKUP_DIR = ".redo_backup"


def _restore_redo_backup(working_dir: Path, root: str = "") -> list[str]:
    """Put back whatever a failed redraw cleared and never replaced.

    A path the redraw did manage to write is left alone: the redraw's own artifact is the
    newer one, and only the gaps are filled from the backup.

    A shot can be removed from the film while its redraw is still running. Its held
    artifacts then go back to the removed pile instead of into ``shots/``, because
    restoring them where the film no longer lists the shot puts an orphan back on disk.
    """
    backup = working_dir / REDO_BACKUP_DIR
    if not backup.exists():
        return []
    planned = {slot for slot, _ in _shot_dirs(working_dir / root)} if root else set()
    restored: list[str] = []
    for held in sorted(backup.rglob("*")):
        if not held.is_file():
            continue
        relative = held.relative_to(backup)
        target = working_dir / _restored_target(relative, root, planned)
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        held.replace(target)
        restored.append(str(target.relative_to(working_dir)))
    shutil.rmtree(backup, ignore_errors=True)
    return restored


def _restored_target(relative: Path, root: str, planned: set[str]) -> Path:
    """Restore a held shot to its removed pile if it left the active timeline."""
    parts = relative.parts
    if not root or not parts or parts[0] != root:
        return relative
    if len(parts) >= 4 and parts[1] == "shots" and parts[2] not in planned:
        return Path(root) / ".removed_shots" / parts[2] / Path(*parts[3:])
    if (
        len(parts) >= 5
        and parts[1].startswith("scene_")
        and parts[1][6:].isdigit()
        and parts[2] == "shots"
        and f"{parts[1]}/{parts[3]}" not in planned
    ):
        return Path(root) / parts[1] / ".removed_shots" / parts[3] / Path(*parts[4:])
    return relative


def _discard_redo_backup(working_dir: Path) -> None:
    """Drop the held artifacts once the redraw has replaced them."""
    shutil.rmtree(working_dir / REDO_BACKUP_DIR, ignore_errors=True)


def _clear_redo_targets(working_dir: Path, root: str, targets: dict[str, str]) -> list[str]:
    """Move invalidated artifacts and their assembled films aside for rollback."""
    root_dir = working_dir / root
    shot_dirs = dict(_shot_dirs(root_dir))
    removed: list[str] = []
    for slot, stage in sorted(targets.items()):
        shot_dir = shot_dirs.get(slot)
        if shot_dir is None:
            continue
        paths: list[Path] = []
        for invalidated in REDO_INVALIDATES[stage]:
            paths.extend(_shot_stage_paths(shot_dir, invalidated))
        if stage == "keyframes":
            paths.extend(shot_dir / f"{frame_type}_selector_output.json" for frame_type in ("first_frame", "last_frame"))
            # Clearing a first frame must also clear a cached camera still that would
            # otherwise copy the old pixels straight back into the redraw.
            paths.extend(shot_dir.rglob("new_camera_*.png"))
        for path in paths:
            if not path.exists() or not path.is_file():
                continue
            held = working_dir / REDO_BACKUP_DIR / path.relative_to(working_dir)
            held.parent.mkdir(parents=True, exist_ok=True)
            try:
                path.replace(held)
            except OSError:
                continue
            removed.append(str(path.relative_to(working_dir)))

    # A shot clip contributes to its scene film and to the root aggregate. Both
    # ancestors must be rebuilt; unrelated scene films remain reusable.
    if any("clips" in REDO_INVALIDATES[stage] for stage in targets.values()):
        films = {root_dir / "final_video.mp4"}
        if root == "idea2video":
            for slot in targets:
                if "/" in slot:
                    scene, _ = slot.split("/", 1)
                    if scene.startswith("scene_") and scene[6:].isdigit():
                        films.add(root_dir / scene / "final_video.mp4")
        for film in sorted(films):
            if not film.exists() or not film.is_file():
                continue
            held = working_dir / REDO_BACKUP_DIR / film.relative_to(working_dir)
            held.parent.mkdir(parents=True, exist_ok=True)
            try:
                film.replace(held)
            except OSError:
                continue
            removed.append(str(film.relative_to(working_dir)))
    return removed


def _slot_display(slot: str) -> str:
    """How the Timeline names a slot, so a refusal can be read against the cards.

    Shot directories are numbered from zero and the Timeline numbers what it shows from
    one, which is why a redo key and the name on the card differ by one.
    """
    if "/" in slot:
        scene, shot = slot.split("/", 1)
        if scene.startswith("scene_") and shot.isdigit() and scene[6:].isdigit():
            return f"Scene {int(scene[6:]) + 1} · Shot {int(shot) + 1}"
    if slot.isdigit():
        return f"Shot {int(slot) + 1}"
    return slot


def _revision_notes_for(working_dir: Path, root: str, redo: dict[str, Any]) -> dict[str, str]:
    """The guidance a redraw draws with: the note against each redone shot, by stage.

    A shot carries its note either as a rejection waiting to be redrawn, or as the reason
    it was redrawn last time. Redrawing it a second time has to find it in both places:
    the slot whose note moves from one to the other is the slot being worked on, and
    losing the note there is how a redraw quietly stops answering it.
    """
    if not redo:
        return {}
    root_record = _read_acceptance(working_dir).get(root) or {}
    shots_record = root_record.get("shots") if isinstance(root_record.get("shots"), dict) else {}
    notes: dict[str, str] = {}
    for slot, stage in redo["slots"].items():
        stage_record = shots_record.get(slot) if isinstance(shots_record.get(slot), dict) else {}
        stage_record = stage_record.get(stage) if isinstance(stage_record.get(stage), dict) else {}
        note = str(stage_record.get("reason") or "").strip()
        if note:
            notes[slot] = note
    return notes


def _mark_slots_redone(working_dir: Path, root: str, targets: dict[str, str], notes: dict[str, str]) -> None:
    """Replace each redone slot's rejection with the note that prompted the redraw.

    The note is kept as the slot's reason under ``redone_at``: the card still says why
    these pixels were regenerated. Without an acceptance the slot derives as
    ``rendered``, so redrawn artifacts go back to being unreviewed rather than
    inheriting the decision that condemned them.
    """
    path = _acceptance_path(working_dir)
    store = _read_acceptance(working_dir)
    root_store = store.get(root) if isinstance(store.get(root), dict) else {}
    shots_store = root_store.get("shots") if isinstance(root_store.get("shots"), dict) else {}
    now = datetime.now().isoformat(timespec="seconds")
    continuity_enabled = _continuity_enabled(working_dir, root)
    for slot, stage in sorted(targets.items()):
        shot_store = shots_store.get(slot) if isinstance(shots_store.get(slot), dict) else {}
        stage_record = shot_store.get(stage) if isinstance(shot_store.get(stage), dict) else {}
        shot_store[stage] = {
            "redone_at": now,
            "reason": str(stage_record.get("reason") or notes.get(slot) or "").strip(),
        }
        if continuity_enabled:
            invalidation_reason = f"Shot {slot} was redrawn; review its regenerated {', '.join(REDO_INVALIDATES[stage])}."
            for invalidated_stage in REDO_INVALIDATES[stage]:
                if invalidated_stage == stage:
                    continue
                record = shot_store.get(invalidated_stage)
                if isinstance(record, dict):
                    shot_store[invalidated_stage] = {
                        **record,
                        "invalidated_at": now,
                        "invalidation_reason": invalidation_reason,
                    }
        shots_store[slot] = shot_store
    final_record = root_store.get("final_video")
    if continuity_enabled and isinstance(final_record, dict) and targets:
        root_store["final_video"] = {
            **final_record,
            "invalidated_at": now,
            "invalidation_reason": "A shot was redrawn in the continuous film; rebuild and review the final assembly.",
        }
    root_store["shots"] = shots_store
    store[root] = root_store
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(store, ensure_ascii=False, indent=4), encoding="utf-8")
    temporary.replace(path)


def _missing_render_dependencies(checklist: dict[str, bool]) -> list[str]:
    if _ready_for_render(checklist):
        return []
    idea_required = ["idea2video/story.txt", "idea2video/characters.json", "idea2video/script.json", "idea2video/scene_*/storyboard.json", "idea2video/scene_*/shots/*/shot_description.json", "idea2video/scene_*/camera_tree.json"]
    script_required = ["script2video/script.txt", "script2video/characters.json", "script2video/storyboard.json", "script2video/shots/*/shot_description.json", "script2video/camera_tree.json"]
    novel_required = ["novel2video/novel/novel_compressed.txt", "novel2video/events/event_*.json", "novel2video/relevant_chunks/event_*", "novel2video/scenes/event_*/scene_*.json", "novel2video/global_information/characters/event_level/*.json", "novel2video/global_information/characters/novel_level/*.json"]
    return [f"idea mode: {path}" for path in idea_required if not checklist.get(path)] + [f"script mode: {path}" for path in script_required if not checklist.get(path)] + [f"novel mode: {path}" for path in novel_required if not checklist.get(path)]


def _idea_mode_ready(checklist: dict[str, bool]) -> bool:
    return bool(checklist.get("idea2video/story.txt") and checklist.get("idea2video/characters.json") and checklist.get("idea2video/script.json") and checklist.get("idea2video/scene_*/storyboard.json") and checklist.get("idea2video/scene_*/shots/*/shot_description.json") and checklist.get("idea2video/scene_*/camera_tree.json"))


def _novel_text_ready(checklist: dict[str, bool]) -> bool:
    return _novel_mode_ready(checklist)


def _novel_mode_ready(checklist: dict[str, bool]) -> bool:
    return bool(checklist.get("novel2video/novel/novel_compressed.txt") and checklist.get("novel2video/events/event_*.json") and checklist.get("novel2video/relevant_chunks/event_*") and checklist.get("novel2video/scenes/event_*/scene_*.json") and checklist.get("novel2video/global_information/characters/event_level/*.json") and checklist.get("novel2video/global_information/characters/novel_level/*.json"))


def _script_mode_ready(checklist: dict[str, bool]) -> bool:
    return bool(checklist.get("script2video/script.txt") and checklist.get("script2video/characters.json") and checklist.get("script2video/storyboard.json") and checklist.get("script2video/shots/*/shot_description.json") and checklist.get("script2video/camera_tree.json"))
