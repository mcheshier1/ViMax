import os
import re
import shutil
import json
import hashlib
import logging
import asyncio
import time
from typing import Any, Callable, Optional, Dict, List, Tuple, Literal, Type, TypeVar
from moviepy import VideoFileClip, concatenate_videoclips
from PIL import Image
from agents import *
import yaml
from interfaces import *
from langchain.chat_models import init_chat_model
from tools.render_backend import RenderBackend
from utils.provider_presets import resolve_chat_model_config
from pathlib import Path
from utils.text import safe_path_component

from .set_references import (
    accepted_set_references,
    canonical_render_root,
    generate_set_references,
    plan_set_specs,
    validate_set_candidate_count,
)
from .render_contract import DEFAULT_RENDER_PHASE, ModelScopedArtifacts, PortraitGenerationError, RenderOutcome, normalize_phase


TModel = TypeVar("TModel")


def _normalize_model_list(items: Any, model_cls: Type[TModel], field_name: str) -> List[TModel]:
    if items is None:
        return []
    if not isinstance(items, list):
        raise TypeError(f"{field_name} must be a list, got {type(items).__name__}")
    normalized: List[TModel] = []
    for idx, item in enumerate(items):
        if isinstance(item, model_cls):
            normalized.append(item)
        elif isinstance(item, dict):
            normalized.append(model_cls.model_validate(item))
        else:
            raise TypeError(f"{field_name}[{idx}] must be {model_cls.__name__} or dict, got {type(item).__name__}")
    return normalized


def _group_shots_into_cameras(shot_descriptions: List[ShotDescription]) -> List[Camera]:
    cameras_by_idx: Dict[int, Camera] = {}
    for shot_description in shot_descriptions:
        camera = cameras_by_idx.get(shot_description.cam_idx)
        if camera is None:
            camera = Camera(idx=shot_description.cam_idx, active_shot_idxs=[])
            cameras_by_idx[shot_description.cam_idx] = camera
        camera.active_shot_idxs.append(shot_description.idx)
    return list(cameras_by_idx.values())

def _collect_priority_shot_idxs(camera_tree: List[Camera]) -> List[int]:
    """Shot indices that other cameras depend on."""
    return [camera.parent_shot_idx for camera in camera_tree if camera.parent_shot_idx is not None]


def _pipeline_print(quiet: bool, message: str) -> None:
    if not quiet:
        print(message)

def _file_signature(path: str) -> Dict[str, int]:
    stat = os.stat(path)
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}



def _emit_text_plan_progress(progress, stage: str, message: str, metadata: Dict[str, Any] | None = None) -> None:
    if progress is not None:
        progress(stage, message, metadata or {})


def _emit_render_progress(progress, stage: str, message: str, metadata: Dict[str, Any] | None = None) -> None:
    if progress is not None:
        progress(stage, message, metadata or {})


def _scoped_progress(progress, **scope):
    if progress is None:
        return None

    def emit(stage: str, message: str, metadata: Dict[str, Any] | None = None) -> None:
        payload = dict(scope)
        payload.update(metadata or {})
        _emit_render_progress(progress, stage, message, payload)

    return emit


class Script2VideoPipeline(ModelScopedArtifacts):

    def __init__(
        self,
        chat_model: str,
        image_generator,
        video_generator,
        working_dir: str,
    ):

        self.chat_model = chat_model
        self.image_generator = image_generator
        self.video_generator = video_generator

        self.character_extractor = CharacterExtractor(chat_model=self.chat_model)
        self.character_portraits_generator = CharacterPortraitsGenerator(image_generator=self.image_generator)
        self.storyboard_artist = StoryboardArtist(chat_model=self.chat_model)
        self.camera_image_generator = CameraImageGenerator(
            chat_model=self.chat_model,
            image_generator=self.image_generator,
            video_generator=self.video_generator,
            working_dir=working_dir,
        )
        self.reference_image_selector = ReferenceImageSelector(chat_model=self.chat_model)

        self.working_dir = working_dir
        os.makedirs(self.working_dir, exist_ok=True)
        self.character_portrait_events = {}
        self.shot_desc_events = {}
        self.frame_events = {}
        # Set from the video model's catalogue at render start; assumed until then.
        self.frame_bracketing = True
        # One clip/transition video per outer render. Idea mode passes this mutable
        # allowance to each scene; flat renders keep their own mapping.
        self.video_budget: Dict[str, str] = {}
        self._playback_predecessors: Dict[int, Dict[str, Any]] = {}
        self._global_chained_keyframes = False
        self._fresh_frame_types: Dict[int, set[str]] = {}
        self._continuity_scope: set[str] = set()
    def _claim_video(self, what: str) -> bool:
        """Claim the single paid-video allowance shared by an outer render."""
        if self.video_budget.get("claimed_by"):
            return False
        self.video_budget["claimed_by"] = what
        return True


    def reference_image_limit(self) -> Optional[int]:
        """Reference images the image model accepts, or ``None`` when unbounded.

        Read from the generator at selection time: the limit comes from a
        catalogue fetched on the first image request, which happens well after
        this pipeline is constructed, and a provider that refused a request may
        have narrowed it further since.
        """
        return getattr(self.image_generator, "reference_limit", None)

    def clip_seconds(self) -> Optional[int]:
        """Seconds of video each shot renders as.

        The storyboard needs this to meet a requested runtime: the shot count is
        what sets the length of the finished film, and a plan cannot size itself
        against a duration it cannot see.
        """
        return getattr(self.video_generator, "clip_seconds", None)

    def _effective_shot_duration(self, shot_description: ShotDescription) -> int:
        policy = getattr(self.video_generator, "duration_policy", None)
        duration = shot_description.duration_seconds
        if duration is None:
            if isinstance(policy, dict):
                return int(policy.get("defaultSeconds") or self.clip_seconds() or 8)
            return int(self.clip_seconds() or 8)
        if not isinstance(policy, dict) or not policy.get("configured"):
            provider = policy.get("provider") if isinstance(policy, dict) else "active video model"
            model = policy.get("model") if isinstance(policy, dict) else ""
            raise ValueError(
                f"Shot {shot_description.idx} has an explicit duration override, but no maximum is configured "
                f"for {provider}:{model}. Configure a limit in video settings before rendering."
            )
        allowed = policy.get("allowedSeconds")
        if isinstance(allowed, list):
            if duration not in allowed:
                raise ValueError(
                    f"Shot {shot_description.idx} duration {duration}s is unsupported by "
                    f"{policy.get('provider')}:{policy.get('model')}; allowed durations: {allowed}."
                )
        elif duration > int(policy.get("maxSeconds") or 0):
            raise ValueError(
                f"Shot {shot_description.idx} duration {duration}s exceeds the configured maximum "
                f"of {policy.get('maxSeconds')}s."
            )
        if duration < int(policy.get("minSeconds") or 1):
            raise ValueError(
                f"Shot {shot_description.idx} duration {duration}s is below the minimum supported "
                f"duration of {policy.get('minSeconds')}s."
            )
        return duration

    def _preflight_saved_shot_durations(self, only_shots: set[int] | None) -> None:
        shot_root = Path(self.working_dir) / "shots"
        if not shot_root.is_dir():
            return
        for path in shot_root.glob("*/shot_description.json"):
            if not path.parent.name.isdigit():
                continue
            shot_idx = int(path.parent.name)
            if only_shots is not None and shot_idx not in only_shots:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Cannot read shot {shot_idx} duration settings: {exc}") from exc
            if not isinstance(payload, dict) or payload.get("duration_seconds") is None:
                continue
            try:
                shot = ShotDescription.model_validate(payload)
            except Exception as exc:
                raise ValueError(f"Invalid duration override in shot {shot_idx}: {exc}") from exc
            self._effective_shot_duration(shot)

    async def _brackets_clips(self) -> bool:
        """Whether the video model accepts an end keyframe.

        A clip generated between two keyframes cannot wander; given only a start
        frame the model invents the rest of the shot, which is why the room drifts
        across a clip and the cut into the next shot reads as a jump. Generating an
        end keyframe for a model that rejects one would burn image calls for
        nothing, so the answer comes from the model's own catalogue.
        """
        probe = getattr(self.video_generator, "supports_last_frame", None)
        if probe is None:
            # Providers without a capability catalogue (the yunwu/Veo generator)
            # take both frames.
            return True
        return bool(await probe())

    def _frame_stills(self, shot_descriptions: List[ShotDescription]) -> List[str]:
        """Keyframes video generation will consume, in shot order."""
        stills: List[str] = []
        for shot_description in shot_descriptions:
            first_frame = self.frame_path(shot_description.idx, "first_frame")
            if os.path.exists(first_frame):
                stills.append(first_frame)
            last_frame = self.frame_path(shot_description.idx, "last_frame")
            if os.path.exists(last_frame):
                stills.append(last_frame)
        return stills


    async def plan_text_artifacts(
        self,
        script: str,
        user_requirement: str,
        style: str,
        characters: List[CharacterInScene] = None,
        progress: Callable[[str, str, Dict[str, Any] | None], None] | None = None,
        quiet: bool = False,
    ):
        """Generate only structured text artifacts required before rendering.

        This helper intentionally stops before character portraits, frame generation,
        video generation, and final concatenation so an agent loop can pause for
        user review after narrative planning.
        """
        self.character_portrait_events = {}
        self.shot_desc_events = {}
        self.frame_events = {}
        self._deferred_frame_shots = set()

        if characters is None:
            _emit_text_plan_progress(progress, "extract_characters", "Extracting characters from script")
            characters = await self.extract_characters(script=script, quiet=quiet)
        else:
            characters = _normalize_model_list(characters, CharacterInScene, "characters")
            _emit_text_plan_progress(progress, "extract_characters", "Using provided characters", {"provided": True, "count": len(characters)})
            characters_path = os.path.join(self.working_dir, "characters.json")
            if not os.path.exists(characters_path):
                with open(characters_path, "w", encoding="utf-8") as f:
                    json.dump([character.model_dump() for character in characters], f, ensure_ascii=False, indent=4)
            for character in characters:
                self.character_portrait_events[character.idx] = asyncio.Event()
        _emit_text_plan_progress(progress, "design_storyboard", "Designing storyboard")
        storyboard = await self.design_storyboard(
            script=script,
            characters=characters,
            user_requirement=user_requirement,
            quiet=quiet,
            clip_seconds=self.clip_seconds(),
        )
        _emit_text_plan_progress(progress, "decompose_shots", "Decomposing shot visual descriptions", {"shot_count": len(storyboard)})
        shot_descriptions = await self.decompose_visual_descriptions(
            shot_brief_descriptions=storyboard,
            characters=characters,
            quiet=quiet,
        )
        camera_tree = None
        for attempt in range(2):
            try:
                stage = "construct_camera_tree" if attempt == 0 else "construct_camera_tree_retry"
                message = "Constructing camera tree" if attempt == 0 else "Retrying camera tree construction after schema/type failure"
                _emit_text_plan_progress(progress, stage, message, {"shot_count": len(shot_descriptions), "attempt": attempt + 1})
                camera_tree = await self.construct_camera_tree(
                    shot_descriptions=shot_descriptions,
                    quiet=quiet,
                )
                break
            except Exception:
                camera_tree_path = os.path.join(self.working_dir, "camera_tree.json")
                if os.path.exists(camera_tree_path):
                    os.remove(camera_tree_path)
                if attempt == 1:
                    raise
        assert camera_tree is not None
        render_root = canonical_render_root(self.working_dir)
        if Path(self.working_dir).resolve() == render_root:
            script_path = render_root / "script.txt"
            if not script_path.exists():
                script_path.write_text(script, encoding="utf-8")
            _emit_text_plan_progress(progress, "plan_set_references", "Preparing character-free set assignments")
            await plan_set_specs(self.chat_model, render_root)
        return {
            "characters": characters,
            "storyboard": storyboard,
            "shot_descriptions": shot_descriptions,
            "camera_tree": camera_tree,
        }


    @classmethod
    def init_from_config(cls, config_path: str):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        chat_model_args = resolve_chat_model_config(config["chat_model"]["init_args"])
        chat_model = init_chat_model(**chat_model_args)
        backend = RenderBackend.from_config(config)

        return cls(
            chat_model=chat_model,
            image_generator=backend.image_generator,
            video_generator=backend.video_generator,
            working_dir=config["working_dir"],
        )

    async def __call__(
        self,
        script: str,
        user_requirement: str,
        style: str,
        characters: List[CharacterInScene] = None,
        character_portraits_registry: Optional[Dict[str, Dict[str, Dict[str, str]]]] = None,
        quiet: bool = False,
        progress: Callable[[str, str, Dict[str, Any] | None], None] | None = None,
        stop_after: str = DEFAULT_RENDER_PHASE,
        revision_notes: Dict[str, str] | None = None,
        only_shots: Optional[List[int]] = None,
        video_budget: Optional[Dict[str, str]] = None,
        set_ids: Optional[List[str]] = None,
        redo_sets: Optional[List[str]] = None,
        chained_keyframes: bool = False,
        continuity_scope: Optional[List[str]] = None,
        set_candidate_count: int = 1,
    ) -> RenderOutcome:
        stop_after = normalize_phase(stop_after)
        validate_set_candidate_count(set_candidate_count, stop_after)
        self.video_budget = video_budget if video_budget is not None else {}
        self._deferred_frame_shots = set()
        self._fresh_frame_types = {}
        self._continuity_scope = {str(slot) for slot in continuity_scope or []}
        if continuity_scope is None and only_shots is not None:
            self._continuity_scope.update(self._source_slot(int(idx)) for idx in only_shots)
        # A redo names shots, and a run works on those shots: the stills of the others are left
        # for their own turn rather than drawn because a phase run was the easier way to ask.
        only_set = {int(idx) for idx in only_shots} if only_shots is not None else None
        self.revision_notes = {str(shot): note for shot, note in (revision_notes or {}).items() if str(note).strip()}
        self._active_style = style
        self.camera_image_generator.set_reference_context(self.working_dir, style)
        _emit_render_progress(progress, "render_start", "Starting script2video render", {"stop_after": stop_after})
        if stop_after == "sets":
            set_images = await generate_set_references(
                self.working_dir,
                self.image_generator,
                style,
                set_ids=set_ids,
                redo_sets=redo_sets,
                set_candidate_count=set_candidate_count,
                progress=progress,
            )
            return self.render_outcome("sets", style, set_references=set_images, awaiting="stills")
        self._preflight_saved_shot_durations(only_set)
        if characters is None:
            _emit_render_progress(progress, "extract_characters", "Extracting characters before render")
            characters = await self.extract_characters(script=script, quiet=quiet)

        else:
            characters = _normalize_model_list(characters, CharacterInScene, "characters")
            _emit_render_progress(progress, "extract_characters", "Using provided characters for render", {"provided": True, "count": len(characters)})
            for character in characters:
                self.character_portrait_events[character.idx] = asyncio.Event()

        character_portraits_registry = await self.generate_character_portraits(
            characters=characters,
            character_portraits_registry=character_portraits_registry,
            style=style,
            progress=progress,
        )

        if stop_after == "portraits":
            stills = self.portrait_stills(character_portraits_registry)
            _emit_render_progress(
                progress,
                "portraits_ready",
                "Character portraits ready for style review",
                {"character_count": len(character_portraits_registry), "still_count": len(stills), "awaiting_confirmation": "sets", "style": style},
            )
            return self.render_outcome("portraits", style, stills=stills, awaiting="sets")
        self.frame_bracketing = await self._brackets_clips()



        # design shots
        _emit_render_progress(progress, "load_storyboard", "Loading or designing storyboard")
        storyboard = await self.design_storyboard(
            script=script,
            characters=characters,
            user_requirement=user_requirement,
            quiet=quiet,
            clip_seconds=self.clip_seconds(),
        )
        _emit_render_progress(progress, "storyboard_ready", "Storyboard ready", {"shot_count": len(storyboard)})

        # decompose visual descriptions of shots
        _emit_render_progress(progress, "load_shot_descriptions", "Loading or decomposing shot descriptions", {"shot_count": len(storyboard)})
        shot_descriptions = await self.decompose_visual_descriptions(
            shot_brief_descriptions=storyboard,
            characters=characters,
            quiet=quiet,
        )
        _emit_render_progress(progress, "shot_descriptions_ready", "Shot descriptions ready", {"shot_count": len(shot_descriptions)})

        # construct camera tree
        _emit_render_progress(progress, "load_camera_tree", "Loading or constructing camera tree", {"shot_count": len(shot_descriptions)})
        camera_tree = await self.construct_camera_tree(
            shot_descriptions=shot_descriptions,
            quiet=quiet,
        )
        _emit_render_progress(progress, "camera_tree_ready", "Camera tree ready", {"camera_count": len(camera_tree)})
        ordered_playback = self._ordered_active_shots(camera_tree, shot_descriptions)
        self._playback_predecessors = self._build_playback_predecessors(camera_tree, shot_descriptions)
        if only_shots is None:
            self._continuity_scope.update(self._source_slot(shot.idx) for shot in ordered_playback)
        global_chained_keyframes = chained_keyframes or (
            self._frame_continuity_config().get("mode") == "chained_keyframes"
        )
        self._global_chained_keyframes = global_chained_keyframes
        if not global_chained_keyframes:
            for shot in ordered_playback:
                if (
                    (only_set is None or shot.idx in only_set)
                    and self._is_redrawn(shot.idx)
                    and shot.idx in self.frame_events
                ):
                    self.frame_events[shot.idx]["first_frame"].clear()
                    self.frame_events[shot.idx]["last_frame"].clear()
        if not self.frame_bracketing:
            for shot in ordered_playback:
                if shot.reuse_previous_last_frame is not True or shot.idx not in self._playback_predecessors:
                    continue
                if only_set is not None and shot.idx not in only_set:
                    continue
                source = self._playback_predecessors[shot.idx]
                if not os.path.isfile(source["path"]) or (
                    source["slot"] == self._source_slot(source["idx"])
                    and self._is_redrawn(source["idx"])
                    and (only_set is None or source["idx"] in only_set)
                ):
                    raise RuntimeError(
                        f"Cannot reuse shot {shot.idx}: predecessor shot {source['slot']} needs a last frame, "
                        "but the selected video provider does not generate ending keyframes."
                    )

        priority_shot_idxs = [camera.parent_cam_idx for camera in camera_tree if camera.parent_cam_idx is not None]
        if global_chained_keyframes:
            await self.generate_chained_frames(
                camera_tree=camera_tree,
                shot_descriptions=shot_descriptions,
                characters=characters,
                character_portraits_registry=character_portraits_registry,
                only_shots=only_shots,
                progress=progress,
            )
            tasks = []
        else:
            # Read provider limits before the legacy selector assembles references.
            prepare = getattr(self.image_generator, "prepare", None)
            if prepare is not None:
                await prepare()
            self._preflight_pending_frame_sets(camera_tree, shot_descriptions, only_shots)
            tasks = [
                self.generate_frames_for_single_camera(
                    camera=camera,
                    shot_descriptions=shot_descriptions,
                    characters=characters,
                    character_portraits_registry=character_portraits_registry,
                    priority_shot_idxs=priority_shot_idxs,
                    progress=progress,
                    only_shots=only_shots,
                )
                for camera in sorted(camera_tree, key=lambda item: item.idx)
                if only_set is None or only_set & set(camera.active_shot_idxs)
            ]

        if stop_after == "stills":
            await asyncio.gather(*tasks)
            ordered_shots = self._ordered_active_shots(camera_tree, shot_descriptions)
            stills = self._frame_stills(ordered_shots)
            _emit_render_progress(
                progress,
                "stills_ready",
                "Keyframes ready for review",
                {"shot_count": len(ordered_shots), "still_count": len(stills), "awaiting_confirmation": "video"},
            )
            return self.render_outcome("stills", style, stills=stills, awaiting="video")

        # Finish all frame work first: a transition may own the shared video
        # allowance, and a child whose transition was deferred has no frame
        # event for a clip task to await.
        await asyncio.gather(*tasks)

        ordered_shots = self._ordered_active_shots(camera_tree, shot_descriptions)
        final_video_path = os.path.join(self.working_dir, "final_video.mp4")
        assembly_metadata_path = final_video_path + ".inputs.json"
        active_order = [shot.idx for shot in ordered_shots]
        if not active_order:
            for path in (final_video_path, assembly_metadata_path):
                if os.path.isfile(path):
                    os.remove(path)
            _emit_render_progress(progress, "concat_skipped", "There are no active shots to assemble", {"shot_count": 0})
            _emit_render_progress(
                progress, "render_done", "Script2video render complete", {"final_video_path": None, "phase": "video"}
            )
            return self.render_outcome("video", style, final_video_path="")


        eligible_clips = [
            shot
            for shot in ordered_shots
            if (only_set is None or shot.idx in only_set)
            and not os.path.isfile(self.clip_path(shot.idx))
            and shot.idx not in self._deferred_frame_shots
            and os.path.isfile(self.frame_path(shot.idx, "first_frame"))
            and (not self.frame_bracketing or os.path.isfile(self.frame_path(shot.idx, "last_frame")))
        ]
        frame_blocked_clips = [
            shot.idx
            for shot in ordered_shots
            if (only_set is None or shot.idx in only_set)
            and not os.path.isfile(self.clip_path(shot.idx))
            and shot.idx not in self._deferred_frame_shots
            and (
                not os.path.isfile(self.frame_path(shot.idx, "first_frame"))
                or (self.frame_bracketing and not os.path.isfile(self.frame_path(shot.idx, "last_frame")))
            )
        ]
        if frame_blocked_clips:
            _emit_render_progress(
                progress,
                "video_clips_waiting_for_frames",
                "Some clips remain deferred because their required frames are unavailable",
                {"deferred_clips": [str(idx) for idx in frame_blocked_clips]},
            )
        clip_claimed = bool(eligible_clips) and self._claim_video(f"the clip of shot {eligible_clips[0].idx}")
        if clip_claimed:
            if os.path.isfile(final_video_path):
                os.remove(final_video_path)
            if os.path.isfile(assembly_metadata_path):
                os.remove(assembly_metadata_path)
            await self.generate_video_for_single_shot(shot_description=eligible_clips[0], progress=progress)
        deferred_clips = eligible_clips[1:] if clip_claimed else eligible_clips
        if deferred_clips:
            _emit_render_progress(
                progress,
                "video_clips_deferred",
                f"One video is drawn per render: {len(deferred_clips)} more clip(s) are waiting for their own turn",
                {"deferred_clips": [str(shot.idx) for shot in deferred_clips]},
            )

        missing_clips = [
            shot_description.idx
            for shot_description in ordered_shots
            if not os.path.isfile(self.clip_path(shot_description.idx))
        ]
        if missing_clips:
            for path in (final_video_path, assembly_metadata_path):
                if os.path.isfile(path):
                    os.remove(path)
            _emit_render_progress(
                progress,
                "concat_skipped",
                f"{len(missing_clips)} shot(s) still have no clip, so the film was not built yet",
                {"missing_clips": [str(idx) for idx in missing_clips]},
            )
            _emit_render_progress(
                progress, "render_done", "Script2video render complete", {"final_video_path": None, "phase": "video"}
            )
            return self.render_outcome("video", style, final_video_path="")

        assembly_inputs = {
            "version": 1,
            "active_shot_idxs": active_order,
            "clips": [
                {"shot_idx": shot.idx, **_file_signature(self.clip_path(shot.idx))}
                for shot in ordered_shots
            ],
        }
        cache_matches = False
        if os.path.isfile(final_video_path):
            try:
                with open(assembly_metadata_path, "r", encoding="utf-8") as metadata_file:
                    cache_matches = json.load(metadata_file) == assembly_inputs
            except (OSError, ValueError, TypeError):
                pass
        if os.path.isfile(final_video_path) and not cache_matches:
            os.remove(final_video_path)
        if not cache_matches and os.path.isfile(assembly_metadata_path):
            os.remove(assembly_metadata_path)

        if os.path.isfile(final_video_path):
            print(f"🚀 Skipped concatenating videos, cached inputs are unchanged.")
            _emit_render_progress(progress, "final_video_exists", "Final video already exists for the active inputs", {"path": final_video_path})
        else:
            print(f"🎬 Starting concatenating videos...")
            _emit_render_progress(progress, "concat_start", "Concatenating video clips", {"shot_count": len(ordered_shots)})
            video_clips = [
                VideoFileClip(self.clip_path(shot_description.idx))
                for shot_description in ordered_shots
            ]
            final_video = concatenate_videoclips(video_clips)
            final_video.write_videofile(final_video_path, codec="libx264", preset="medium")
            with open(assembly_metadata_path, "w", encoding="utf-8") as metadata_file:
                json.dump(assembly_inputs, metadata_file, ensure_ascii=False, indent=2)
            print(f"☑️ Concatenated videos, saved to {final_video_path}.")
            _emit_render_progress(progress, "concat_done", "Final video concatenated", {"path": final_video_path})

        _emit_render_progress(progress, "render_done", "Script2video render complete", {"final_video_path": final_video_path, "phase": "video"})
        return self.render_outcome("video", style, final_video_path=final_video_path)

    @staticmethod
    def _ordered_active_shots(camera_tree: List[Camera], shot_descriptions: List[ShotDescription]) -> List[ShotDescription]:
        """Resolve playback order from camera order and each camera's stored shot order."""
        by_idx = {shot.idx: shot for shot in shot_descriptions}
        return [
            by_idx[shot_idx]
            for camera in sorted(camera_tree, key=lambda item: item.idx)
            for shot_idx in camera.active_shot_idxs
            if shot_idx in by_idx
        ]
    def _continuity_project_dir(self) -> str:
        render_dir = os.path.abspath(self.working_dir)
        if os.path.basename(render_dir).startswith("scene_"):
            render_dir = os.path.dirname(render_dir)
            if os.path.basename(render_dir).startswith("event_"):
                render_dir = os.path.dirname(os.path.dirname(render_dir))
        return os.path.dirname(render_dir)

    def _frame_continuity_config(self) -> Dict[str, Any]:
        path = os.path.join(self._continuity_project_dir(), "frame_continuity.json")
        try:
            with open(path, "r", encoding="utf-8") as handle:
                config = json.load(handle)
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Cannot read frame continuity configuration {path}: {exc}") from exc
        if not isinstance(config, dict):
            raise RuntimeError(f"Frame continuity configuration {path} must contain a JSON object")
        return config

    @staticmethod
    def _sha256_file(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _mark_frame_fresh(self, shot_idx: int, frame_type: str) -> None:
        self._fresh_frame_types.setdefault(shot_idx, set()).add(frame_type)

    def _source_slot(self, shot_idx: int) -> str:
        return self._shot_slot(shot_idx)

    def _record_continuity_source(
        self,
        shot_idx: int,
        source_idx: int,
        source_path: str,
        mode: str,
        source_slot: Optional[str] = None,
    ) -> None:
        selector_path = self.selector_output_path(shot_idx, "first_frame")
        selector = {
            "reference_image_path_and_text_pairs": [[source_path, "Previous ending frame copied byte-for-byte."]],
            "text_prompt": "",
            "sent_prompt": "",
        }
        source_project_path = Path(os.path.relpath(source_path, self._continuity_project_dir())).as_posix()
        selector["continuity_source"] = {
            "shot_idx": source_idx,
            "slot": source_slot or self._source_slot(source_idx),
            "path": source_project_path,
            "frame_type": "last_frame",
            "sha256": self._sha256_file(source_path),
            "mode": mode,
        }
        os.makedirs(os.path.dirname(selector_path), exist_ok=True)
        with open(selector_path, "w", encoding="utf-8") as handle:
            json.dump(selector, handle, ensure_ascii=False, indent=4)

    def _external_playback_predecessor(self) -> Optional[Dict[str, Any]]:
        """Return the last active shot in the preceding Idea scene, if any."""
        scene_dir = Path(self.working_dir).resolve()
        match = re.fullmatch(r"scene_(\d+)", scene_dir.name)
        if match is None:
            return None
        current_scene_idx = int(match.group(1))
        preceding_scenes = sorted(
            (
                path for path in scene_dir.parent.glob("scene_*")
                if path.is_dir() and (scene_match := re.fullmatch(r"scene_(\d+)", path.name))
                and int(scene_match.group(1)) < current_scene_idx
            ),
            key=lambda path: int(path.name[6:]),
        )
        for previous_scene in reversed(preceding_scenes):
            try:
                cameras = json.loads((previous_scene / "camera_tree.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(cameras, list):
                continue
            active = [
                int(shot_idx)
                for camera in sorted(
                    (item for item in cameras if isinstance(item, dict)),
                    key=lambda item: int(item.get("idx", 0)),
                )
                for shot_idx in camera.get("active_shot_idxs", [])
            ]
            if not active:
                continue
            source_idx = active[-1]
            source_path = previous_scene / "shots" / str(source_idx) / self.image_model_slug / "last_frame.png"
            description = None
            try:
                description = ShotDescription.model_validate(
                    json.loads((previous_scene / "shots" / str(source_idx) / "shot_description.json").read_text(encoding="utf-8"))
                )
            except (OSError, ValueError, TypeError):
                pass
            return {
                "idx": source_idx,
                "slot": f"{previous_scene.name}/{source_idx}",
                "path": str(source_path),
                "description": description,
                "event": None,
            }
        return None

    def _build_playback_predecessors(
        self,
        camera_tree: List[Camera],
        shot_descriptions: List[ShotDescription],
    ) -> Dict[int, Dict[str, Any]]:
        ordered = self._ordered_active_shots(camera_tree, shot_descriptions)
        predecessors: Dict[int, Dict[str, Any]] = {}
        for previous, shot in zip(ordered, ordered[1:]):
            predecessors[shot.idx] = {
                "idx": previous.idx,
                "slot": self._source_slot(previous.idx),
                "path": self.frame_path(previous.idx, "last_frame"),
                "description": previous,
                "event": self.frame_events.get(previous.idx, {}).get("last_frame"),
            }
        if ordered:
            external = self._external_playback_predecessor()
            if external is not None:
                predecessors[ordered[0].idx] = external
        return predecessors

    def _continuity_slot_invalidated(self, slot: str) -> bool:
        render_root = canonical_render_root(self.working_dir)
        approval_path = render_root.parent / "render_acceptance.json"
        try:
            acceptance = json.loads(approval_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return False
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Cannot read frame approvals from {approval_path}: {exc}") from exc
        root_record = acceptance.get(render_root.name, {}) if isinstance(acceptance, dict) else {}
        shots = root_record.get("shots", {}) if isinstance(root_record, dict) else {}
        record = shots.get(slot, {}) if isinstance(shots, dict) else {}
        keyframes = record.get("keyframes", {}) if isinstance(record, dict) else {}
        return isinstance(keyframes, dict) and bool(keyframes.get("invalidated_at"))
    def _continuity_cache_matches(
        self,
        shot_idx: int,
        source: Dict[str, Any],
        mode: str,
        source_hash: Optional[str] = None,
    ) -> bool:
        source_idx = int(source["idx"])
        source_slot = str(source["slot"])
        source_path = str(source["path"])
        expected_hash = source_hash or self._sha256_file(source_path)
        target_path = self.frame_path(shot_idx, "first_frame")
        if not os.path.isfile(target_path):
            return False
        try:
            with open(self.selector_output_path(shot_idx, "first_frame"), "r", encoding="utf-8") as handle:
                provenance = json.load(handle).get("continuity_source")
        except (OSError, ValueError, AttributeError):
            return False
        return (
            isinstance(provenance, dict)
            and provenance.get("shot_idx") == source_idx
            and provenance.get("slot") == source_slot
            and provenance.get("path") == Path(os.path.relpath(source_path, self._continuity_project_dir())).as_posix()
            and provenance.get("frame_type") == "last_frame"
            and provenance.get("sha256") == expected_hash
            and provenance.get("mode") == mode
            and (mode != "reuse" or self._sha256_file(target_path) == expected_hash)
        )

    def _require_cached_continuity(
        self,
        shot_idx: int,
        source: Dict[str, Any],
        mode: str,
        only_shots: Optional[set[int]] = None,
    ) -> None:
        source_idx = int(source["idx"])
        source_slot = str(source["slot"])
        source_path = str(source["path"])
        source_selected = source_slot in self._continuity_scope or (
            only_shots is not None
            and source_slot == self._source_slot(source_idx)
            and source_idx in only_shots
        )
        if not os.path.isfile(source_path):
            raise RuntimeError(
                f"Cannot use excluded shot {shot_idx}: predecessor shot {source_slot} last frame is missing; "
                "include the predecessor explicitly in the render scope."
            )
        source_invalidated = self._continuity_slot_invalidated(source_slot)
        if source_invalidated and (
            not source_selected
            or (
                source_slot == self._source_slot(source_idx)
                and "last_frame" not in self._fresh_frame_types.get(source_idx, set())
            )
        ):
            raise RuntimeError(
                f"Cannot use excluded shot {shot_idx}: predecessor shot {source_slot} has invalidated keyframes; "
                "include the predecessor explicitly in the render scope."
            )
        if not self._continuity_cache_matches(shot_idx, source, mode):
            raise RuntimeError(
                f"Excluded shot {shot_idx} has missing or stale predecessor continuity from {source_slot}; "
                f"explicitly redraw shot {shot_idx} to replace it."
            )
    async def _reuse_previous_last_frame(
        self,
        shot_idx: int,
        source: Dict[str, Any],
        *,
        only_shots: Optional[set[int]] = None,
        progress=None,
    ) -> None:
        source_idx = int(source["idx"])
        source_slot = str(source["slot"])
        source_path = str(source["path"])
        source_selected = source_slot in self._continuity_scope or (
            only_shots is not None
            and source_slot == self._source_slot(source_idx)
            and source_idx in only_shots
        )
        source_invalidated = self._continuity_slot_invalidated(source_slot)
        source_redrawn = (
            source_selected
            and source_slot == self._source_slot(source_idx)
            and self._is_redrawn(source_idx)
        )
        if source_invalidated and not self._global_chained_keyframes and (
            not source_selected
            or (
                source_slot == self._source_slot(source_idx)
                and not source_redrawn
                and "last_frame" not in self._fresh_frame_types.get(source_idx, set())
                and os.path.isfile(source_path)
            )
        ):
            raise RuntimeError(
                f"Cannot reuse shot {shot_idx}: predecessor shot {source_slot} has invalidated keyframes; "
                "include that predecessor explicitly in the render scope."
            )
        event = source.get("event")
        if not self.frame_bracketing and (source_redrawn or not os.path.isfile(source_path)):
            raise RuntimeError(
                f"Cannot reuse shot {shot_idx}: predecessor shot {source_slot} needs a last frame, "
                "but the selected video provider does not generate ending keyframes."
            )
        if source_redrawn:
            if event is None:
                raise RuntimeError(
                    f"Cannot render shot {shot_idx}: selected predecessor shot {source_slot} "
                    "has no available last-frame event."
                )
            await event.wait()
            if "last_frame" not in self._fresh_frame_types.get(source_idx, set()):
                raise RuntimeError(
                    f"Cannot reuse shot {shot_idx}: predecessor shot {source_slot} did not produce "
                    "a fresh last frame in this render."
                )
        elif not os.path.isfile(source_path):
            if not source_selected:
                raise RuntimeError(
                    f"Cannot render shot {shot_idx}: required predecessor shot {source_slot} "
                    "last frame is missing; include that predecessor explicitly in the render scope."
                )
            if event is None:
                raise RuntimeError(
                    f"Cannot render shot {shot_idx}: selected predecessor shot {source_slot} "
                    "has no available last frame."
                )
            await event.wait()
        if not os.path.isfile(source_path):
            raise RuntimeError(f"Cannot render shot {shot_idx}: predecessor shot {source_slot} last frame is missing.")
        if (
            source_invalidated
            and source_slot == self._source_slot(source_idx)
            and "last_frame" not in self._fresh_frame_types.get(source_idx, set())
        ):
            raise RuntimeError(
                f"Cannot reuse shot {shot_idx}: predecessor shot {source_slot} has invalidated keyframes "
                "and was not redrawn in this render."
            )
        if not source_redrawn and event is not None and os.path.isfile(source_path):
            event.set()

        self._assert_frame_cache_usable(shot_idx)
        source_hash = self._sha256_file(source_path)
        target_path = self.frame_path(shot_idx, "first_frame")
        redrawn = self._is_redrawn(shot_idx)
        if os.path.isfile(target_path) and not redrawn:
            if not self._continuity_cache_matches(shot_idx, source, "reuse", source_hash):
                raise RuntimeError(
                    f"Existing first frame for shot {shot_idx} has missing or stale predecessor continuity "
                    f"from {source_slot}; explicitly redraw shot {shot_idx} to replace it."
                )
            self.frame_events[shot_idx]["first_frame"].set()
            return

        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        shutil.copyfile(source_path, target_path)
        self._mark_frame_fresh(shot_idx, "first_frame")
        self._record_continuity_source(shot_idx, source_idx, source_path, "reuse", source_slot)
        self.frame_events[shot_idx]["first_frame"].set()
        _emit_render_progress(
            progress,
            "frame_done",
            f"Copied previous last frame for shot {shot_idx}",
            {"shot_idx": shot_idx, "frame_type": "first_frame", "path": target_path, "source_slot": source_slot},
        )

    async def generate_chained_frames(
        self,
        *,
        camera_tree: List[Camera],
        shot_descriptions: List[ShotDescription],
        characters: List[CharacterInScene],
        character_portraits_registry: Dict[str, Dict[str, Dict[str, str]]],
        only_shots: Optional[List[int]] = None,
        progress: Callable[[str, str, Dict[str, Any] | None], None] | None = None,
    ) -> None:
        """Generate keyframes sequentially in playback order, carrying each cut forward."""
        if not await self._brackets_clips():
            raise RuntimeError(
                "Chained keyframes require a video provider that supports last-frame bracketing; "
                "select a bracketing-capable provider or disable chained_keyframes."
            )
        self.frame_bracketing = True
        self._global_chained_keyframes = True
        ordered = self._ordered_active_shots(camera_tree, shot_descriptions)
        self._playback_predecessors = self._build_playback_predecessors(camera_tree, shot_descriptions)
        if not ordered:
            return
        by_idx = {shot.idx: shot for shot in ordered}
        camera_by_shot = {
            shot_idx: camera
            for camera in sorted(camera_tree, key=lambda item: item.idx)
            for shot_idx in camera.active_shot_idxs
            if shot_idx in by_idx
        }
        wanted = {int(idx) for idx in only_shots} if only_shots is not None else None
        if wanted is not None:
            if not wanted:
                return
            unknown = wanted - by_idx.keys()
            if unknown:
                raise RuntimeError(f"Unknown chained shot(s): {sorted(unknown)}")
            # Saved endings are independent anchors. Only requested shots and their
            # immediate input frames matter; no ancestor chain is replayed.
            for position, shot in enumerate(ordered):
                if shot.idx not in wanted or position == 0:
                    continue
                predecessor = ordered[position - 1]
                if predecessor.idx not in wanted and not os.path.isfile(self.frame_path(predecessor.idx, "last_frame")):
                    raise RuntimeError(
                        f"Cannot render scoped shot {shot.idx}: required predecessor shot {predecessor.idx} "
                        "last frame is missing; include that predecessor in only_shots."
                    )
        for shot in ordered:
            for kind in ("first_frame", "last_frame"):
                self.frame_events.setdefault(shot.idx, {}).setdefault(kind, asyncio.Event())
                if os.path.isfile(self.frame_path(shot.idx, kind)):
                    self.frame_events[shot.idx][kind].set()
        prepare = getattr(self.image_generator, "prepare", None)
        if prepare is not None:
            await prepare()

        first_shot = ordered[0]
        self._preflight_pending_frame_sets(camera_tree, shot_descriptions, only_shots, chained=True)
        if wanted is None or first_shot.idx in wanted:
            camera = camera_by_shot[first_shot.idx]
            root_camera = Camera(idx=camera.idx, active_shot_idxs=[first_shot.idx])
            await self.generate_frames_for_single_camera(
                camera=root_camera,
                shot_descriptions=shot_descriptions,
                characters=characters,
                character_portraits_registry=character_portraits_registry,
                priority_shot_idxs=[],
                progress=progress,
                only_shots=[first_shot.idx],
            )

        for position, shot in enumerate(ordered[1:], start=1):
            if wanted is not None and shot.idx not in wanted:
                continue
            previous = ordered[position - 1]
            previous_last = self.frame_path(previous.idx, "last_frame")
            if not os.path.isfile(previous_last):
                if wanted is not None and previous.idx not in wanted:
                    raise RuntimeError(
                        f"Cannot render scoped shot {shot.idx}: required predecessor shot {previous.idx} "
                        f"last frame is missing; include that predecessor in only_shots."
                    )
                raise RuntimeError(f"Cannot render shot {shot.idx}: predecessor shot {previous.idx} last frame is missing.")
            same_camera = camera_by_shot[shot.idx].idx == camera_by_shot[previous.idx].idx
            expected_reuse = (
                shot.reuse_previous_last_frame
                if shot.reuse_previous_last_frame is not None
                else same_camera
            )
            expected_mode = "reuse" if expected_reuse else "reframe"
            expected_slot = self._source_slot(previous.idx)
            expected_hash = self._sha256_file(previous_last)
            expected_path = Path(os.path.relpath(previous_last, self._continuity_project_dir())).as_posix()
            first_path = self.frame_path(shot.idx, "first_frame")
            redrawn = self._is_redrawn(shot.idx)
            try:
                with open(self.selector_output_path(shot.idx, "first_frame"), "r", encoding="utf-8") as handle:
                    continuity = json.load(handle).get("continuity_source")
            except (OSError, ValueError, AttributeError):
                continuity = None
            valid_existing = (
                os.path.isfile(first_path)
                and isinstance(continuity, dict)
                and continuity.get("shot_idx") == previous.idx
                and (
                    continuity.get("slot") == expected_slot
                    or (continuity.get("slot") is None and not expected_slot.startswith("scene_"))
                )
                and (
                    continuity.get("path") == expected_path
                    or (continuity.get("path") is None and not expected_slot.startswith("scene_"))
                )
                and continuity.get("frame_type") == "last_frame"
                and continuity.get("sha256") == expected_hash
                and continuity.get("mode") == expected_mode
                and (expected_mode != "reuse" or self._sha256_file(first_path) == expected_hash)
            )
            if os.path.isfile(first_path) and not redrawn and not valid_existing:
                raise RuntimeError(
                    f"Existing first frame for shot {shot.idx} has no matching continuity provenance from "
                    f"predecessor shot {previous.idx}; explicitly redraw shot {shot.idx} to replace it."
                )
            if redrawn:
                for path in (first_path, self.frame_path(shot.idx, "last_frame"),
                             self.selector_output_path(shot.idx, "first_frame"),
                             self.selector_output_path(shot.idx, "last_frame")):
                    if os.path.isfile(path):
                        os.remove(path)
                self.frame_events[shot.idx]["first_frame"].clear()
                self.frame_events[shot.idx]["last_frame"].clear()

            if not os.path.isfile(first_path):
                os.makedirs(os.path.dirname(first_path), exist_ok=True)
                if expected_mode == "reuse":
                    shutil.copyfile(previous_last, first_path)
                    self._record_continuity_source(
                        shot.idx, previous.idx, previous_last, "reuse", expected_slot
                    )
                else:
                    await self._generate_chained_image(
                        shot, previous, previous_last, characters, character_portraits_registry,
                        progress, frame_type="first_frame",
                    )
            self.frame_events[shot.idx]["first_frame"].set()
            await self._generate_chained_image(
                shot, shot, first_path, characters, character_portraits_registry,
                progress, frame_type="last_frame",
            )

    async def _generate_chained_image(
        self,
        shot: ShotDescription,
        source_shot: ShotDescription,
        source_path: str,
        characters: List[CharacterInScene],
        character_portraits_registry: Dict[str, Dict[str, Dict[str, str]]],
        progress=None,
        *,
        frame_type: Literal["first_frame", "last_frame"],
        source_slot: Optional[str] = None,
    ) -> None:
        source_type = "last_frame" if frame_type == "first_frame" else "first_frame"
        source = {
            "shot_idx": source_shot.idx,
            "frame_type": source_type,
            "sha256": self._sha256_file(source_path),
        }
        expected_source_slot = source_slot or self._source_slot(source_shot.idx)
        expected_source_path = Path(os.path.relpath(source_path, self._continuity_project_dir())).as_posix()
        frame_path = self.frame_path(shot.idx, frame_type)
        invalidated = self._assert_frame_cache_usable(shot.idx)
        if os.path.isfile(frame_path) and not self._is_redrawn(shot.idx) and not invalidated:
            selector = self.load_selector_output(shot.idx, frame_type) or {}
            source_matches = selector.get("source_frame") == source
            if frame_type == "first_frame":
                provenance = selector.get("continuity_source")
                source_matches = source_matches and (
                    isinstance(provenance, dict)
                    and (
                        provenance.get("slot") == expected_source_slot
                        or (provenance.get("slot") is None and not expected_source_slot.startswith("scene_"))
                    )
                    and (
                        provenance.get("path") == expected_source_path
                        or (provenance.get("path") is None and not expected_source_slot.startswith("scene_"))
                    )
                    and provenance.get("mode") == "reframe"
                )
            if not source_matches:
                raise RuntimeError(
                    f"Existing {frame_type} for shot {shot.idx} has stale source state; "
                    "explicitly redraw its keyframes before rendering."
                )
            self.frame_events[shot.idx][frame_type].set()
            return
        prepare = getattr(self.image_generator, "prepare", None)
        if prepare is not None:
            await prepare()
        mandatory_pairs = self._mandatory_frame_references(
            shot.idx,
            (
                source_path,
                "Mandatory scene state: preserve the existing room geometry, camera-relative positions, "
                "identities, clothing, props and lighting.",
            ),
        )
        limit = self.reference_image_limit()
        budget = None if limit is None else limit - len(mandatory_pairs)
        pairs = list(mandatory_pairs)
        declared = shot.ff_vis_char_idxs if frame_type == "first_frame" else shot.lf_vis_char_idxs
        ranked = [character for character in characters if character.idx in (declared or [])]
        for character in ranked:
            front = (character_portraits_registry.get(character.identifier_in_scene) or {}).get("front")
            if front and front.get("path") and os.path.isfile(front["path"]) and (
                budget is None or len(pairs) - len(mandatory_pairs) < budget
            ):
                pairs.append((front["path"], f"Approved {self._portrait_reference_text(character, 'front')}"))
        description = shot.ff_desc if frame_type == "first_frame" else shot.lf_desc
        direction = (
            "Reframe the immediately preceding ending at the SAME INSTANT for the new camera; do not advance the action."
            if frame_type == "first_frame" else
            "Use this shot's own first frame as the exact starting state. Keep the camera fixed and change only the specified end-state action."
        )
        source_index = next(index for index, (path, _text) in enumerate(pairs) if path == source_path)
        prompt = (
            f"{direction} Image {source_index} is the mandatory scene reference, not a loose inspiration. "
            "Keep all unchanged people, faces, clothes, furniture, held objects and lighting identical to it. "
            "Portrait references define identity only, never their studio background. "
            f"Required frame: {description}"
        )
        bindings = chr(10).join(f"Image {index}: {text}" for index, (_path, text) in enumerate(pairs))
        prompt = f"{bindings}{chr(10)}{chr(10)}{prompt}"
        selector = {
            "mandatory_references": [list(pair) for pair in mandatory_pairs],
            "reference_image_path_and_text_pairs": pairs,
            "text_prompt": prompt,
            "sent_prompt": prompt,
            "source_frame": source,
            "review_note": "",
        }
        if frame_type == "first_frame":
            selector["continuity_source"] = {
                **source,
                "slot": expected_source_slot,
                "path": expected_source_path,
                "mode": "reframe",
            }
        selector_path = self.selector_output_path(shot.idx, frame_type)
        os.makedirs(os.path.dirname(selector_path), exist_ok=True)
        with open(selector_path, "w", encoding="utf-8") as handle:
            json.dump(selector, handle, ensure_ascii=False, indent=4)
        image = await self._generate_frame_image(
            shot_idx=shot.idx,
            frame_type=frame_type,
            prompt=prompt,
            reference_image_paths=[path for path, _ in pairs],
            mandatory_reference_pairs=mandatory_pairs,
        )
        os.makedirs(os.path.dirname(frame_path), exist_ok=True)
        image.save(frame_path)
        self._mark_frame_fresh(shot.idx, frame_type)
        self.frame_events[shot.idx][frame_type].set()
        _emit_render_progress(progress, "frame_done", f"Generated chained {frame_type} for shot {shot.idx}", {"shot_idx": shot.idx, "frame_type": frame_type, "path": frame_path})





    async def generate_frames_for_single_camera(
        self,
        camera: Camera,
        shot_descriptions: List[ShotDescription],
        characters: List[CharacterInScene],
        character_portraits_registry: Dict[str, Dict[str, Dict[str, str]]],
        priority_shot_idxs: List[int],
        progress: Callable[[str, str, Dict[str, Any] | None], None] | None = None,
        only_shots: Optional[List[int]] = None,
    ):
        if not camera.active_shot_idxs:
            # Every shot of this camera was taken out of the film. It has nothing to draw and
            # nothing to play — and the shots that were in it are kept under .removed_shots/,
            # so this is a hole in the plan rather than lost work.
            return
        wanted = {int(idx) for idx in only_shots} if only_shots is not None else None
        first_shot_idx = camera.active_shot_idxs[0]
        by_idx = self._descriptions_by_idx(shot_descriptions)
        first_shot_ff_path = self.frame_path(first_shot_idx, "first_frame")
        first_invalidated = self._assert_frame_cache_usable(first_shot_idx)
        if wanted is not None and first_shot_idx not in wanted and (
            not os.path.exists(first_shot_ff_path) or self._is_redrawn(first_shot_idx) or first_invalidated
        ):
            raise RuntimeError(
                f"Cannot render scoped shot(s) {sorted(wanted)} in camera {camera.idx}: "
                f"required first frame for excluded shot {first_shot_idx} is missing"
            )
        if camera.parent_shot_idx is not None and wanted is not None:
            parent_path = self.frame_path(camera.parent_shot_idx, "first_frame")
            if camera.parent_shot_idx not in wanted and not os.path.isfile(parent_path):
                raise RuntimeError(
                    f"Cannot render scoped shot {first_shot_idx}: required parent frame "
                    f"for excluded shot {camera.parent_shot_idx} is missing"
                )
        prepare = getattr(self.image_generator, "prepare", None)
        if prepare is not None:
            await prepare()
        self._preflight_pending_frame_sets([camera], shot_descriptions, only_shots)
        _emit_render_progress(progress, "camera_frames_start", f"Generating frames for camera {camera.idx}", {"camera_idx": camera.idx, "active_shot_idxs": camera.active_shot_idxs})

        first_predecessor = self._playback_predecessors.get(first_shot_idx)
        if by_idx[first_shot_idx].reuse_previous_last_frame is True and first_predecessor is not None:
            if wanted is not None and first_shot_idx not in wanted:
                self._require_cached_continuity(first_shot_idx, first_predecessor, "reuse", wanted)
                self.frame_events[first_shot_idx]["first_frame"].set()
            else:
                await self._reuse_previous_last_frame(
                    first_shot_idx, first_predecessor, only_shots=wanted, progress=progress
                )
        elif (
            self._global_chained_keyframes
            and by_idx[first_shot_idx].reuse_previous_last_frame is False
            and first_predecessor is not None
        ):
            if wanted is not None and first_shot_idx not in wanted:
                self._require_cached_continuity(first_shot_idx, first_predecessor, "reframe", wanted)
                self.frame_events[first_shot_idx]["first_frame"].set()
            else:
                source_slot = str(first_predecessor["slot"])
                if not os.path.isfile(first_predecessor["path"]):
                    raise RuntimeError(
                        f"Cannot render shot {first_shot_idx}: predecessor shot {source_slot} "
                        "last frame is missing; include that predecessor explicitly in the render scope."
                    )
                source_description = first_predecessor.get("description")
                if source_description is None:
                    raise RuntimeError(f"Cannot reframe shot {first_shot_idx}: predecessor shot {source_slot} description is missing.")
                await self._generate_chained_image(
                    by_idx[first_shot_idx], source_description, first_predecessor["path"],
                    characters, character_portraits_registry, progress, frame_type="first_frame",
                    source_slot=source_slot,
                )
        elif os.path.exists(first_shot_ff_path) and not self._is_redrawn(first_shot_idx) and not first_invalidated:
            print(f"🚀 Skipped generating first_frame for shot {first_shot_idx}, already exists.")
            self.frame_events[first_shot_idx]["first_frame"].set()
            _emit_render_progress(progress, "frame_exists", f"First frame for shot {first_shot_idx} already exists", {"camera_idx": camera.idx, "shot_idx": first_shot_idx, "frame_type": "first_frame", "path": first_shot_ff_path})

        else:
            print(f"🖼️ Starting first_frame generation for shot {first_shot_idx}...")
            _emit_render_progress(progress, "frame_start", f"Generating first frame for shot {first_shot_idx}", {"camera_idx": camera.idx, "shot_idx": first_shot_idx, "frame_type": "first_frame"})
            available_image_path_and_text_pairs = []
            scene_source_pair = None
            visible_first_frame = self._visible_characters(by_idx[first_shot_idx], characters, "first_frame")
            for visible_character in visible_first_frame:
                registry_item = character_portraits_registry.get(visible_character.identifier_in_scene) or {}
                for view, item in registry_item.items():
                    path = item.get("path") if isinstance(item, dict) else None
                    if path:
                        available_image_path_and_text_pairs.append((path, self._portrait_reference_text(visible_character, view)))
            
            # generate the first_frame based on the shot_description.ff_desc
            if camera.parent_shot_idx is not None:
                parent_shot_idx = camera.parent_shot_idx
                parent_event = self.frame_events[parent_shot_idx]["first_frame"]
                parent_frame_path = self.frame_path(parent_shot_idx, "first_frame")
                parent_invalidated = self._assert_frame_cache_usable(parent_shot_idx)
                if os.path.exists(parent_frame_path) and not parent_invalidated and not self._is_redrawn(parent_shot_idx):
                    parent_event.set()
                elif wanted is not None and parent_shot_idx not in wanted:
                    raise RuntimeError(
                        f"Cannot render scoped shot {first_shot_idx}: required parent frame "
                        f"for excluded shot {parent_shot_idx} is missing"
                    )
                await parent_event.wait()
                parent_shot_ff_path = parent_frame_path
                transition_video_path = os.path.join(self.shot_video_dir(first_shot_idx), f"transition_video_from_shot_{parent_shot_idx}.mp4")

                if os.path.exists(transition_video_path):
                    print(f"🚀 Skipped generating transition video for shot {first_shot_idx} from shot {parent_shot_idx}, already exists.")
                    _emit_render_progress(progress, "transition_video_exists", f"Transition video for shot {first_shot_idx} already exists", {"camera_idx": camera.idx, "shot_idx": first_shot_idx, "parent_shot_idx": parent_shot_idx, "path": transition_video_path})
                else:
                    if not self._claim_video(f"the transition into shot {first_shot_idx}"):
                        self._deferred_frame_shots.update(camera.active_shot_idxs)
                        _emit_render_progress(progress, "transition_video_deferred", f"One video is drawn per render: the transition into shot {first_shot_idx} waits for its own turn", {"shot_idx": first_shot_idx, "parent_shot_idx": parent_shot_idx, "camera_idx": camera.idx})
                        return
                    print(f"🖼️ Starting transition video generation for shot {first_shot_idx} from shot {parent_shot_idx}...")
                    _emit_render_progress(progress, "transition_video_start", f"Generating transition video for shot {first_shot_idx}", {"camera_idx": camera.idx, "shot_idx": first_shot_idx, "parent_shot_idx": parent_shot_idx})
                    transition_video_output = await self.camera_image_generator.generate_transition_video(
                        first_shot_visual_desc=by_idx[parent_shot_idx].visual_desc,
                        second_shot_visual_desc=by_idx[first_shot_idx].visual_desc,
                        first_shot_ff_path=parent_shot_ff_path,
                        progress=_scoped_progress(progress, camera_idx=camera.idx, shot_idx=first_shot_idx, parent_shot_idx=parent_shot_idx, artifact="transition_video"),
                    )
                    os.makedirs(os.path.dirname(transition_video_path), exist_ok=True)
                    transition_video_output.save(transition_video_path)
                    print(f"☑️ Generated transition video for shot {first_shot_idx} from shot {parent_shot_idx}, saved to {transition_video_path}.")
                    _emit_render_progress(progress, "transition_video_done", f"Transition video for shot {first_shot_idx} generated", {"camera_idx": camera.idx, "shot_idx": first_shot_idx, "parent_shot_idx": parent_shot_idx, "path": transition_video_path})

                new_camera_image_path = os.path.join(self.shot_video_dir(first_shot_idx), f"new_camera_{camera.idx}.png")
                if os.path.exists(new_camera_image_path):
                    print(f"🚀 Skipped generating new camera image for shot {first_shot_idx}, already exists.")
                    _emit_render_progress(progress, "new_camera_image_exists", f"New camera image for shot {first_shot_idx} already exists", {"camera_idx": camera.idx, "shot_idx": first_shot_idx, "path": new_camera_image_path})
                else:
                    print(f"🖼️ Starting new camera image generation for shot {first_shot_idx}...")
                    _emit_render_progress(progress, "new_camera_image_start", f"Extracting new camera image for shot {first_shot_idx}", {"camera_idx": camera.idx, "shot_idx": first_shot_idx})
                    new_camera_image = self.camera_image_generator.get_new_camera_image(transition_video_path)
                    os.makedirs(os.path.dirname(new_camera_image_path), exist_ok=True)
                    new_camera_image.save(new_camera_image_path)
                    print(f"☑️ Generated new camera image for shot {first_shot_idx} (not completed), saved to {new_camera_image_path}.")
                    _emit_render_progress(progress, "new_camera_image_done", f"New camera image for shot {first_shot_idx} extracted", {"camera_idx": camera.idx, "shot_idx": first_shot_idx, "path": new_camera_image_path})

                scene_source_pair = (
                    new_camera_image_path,
                    f"The composition and background are correct but some elements may be wrong. The wrong elements should be replaced. "
                    f"Wrong elements: {camera.missing_info}. You must replace characters using their portraits without changing the background.",
                )

            # 如果子镜头缺少信息，则需要选择参考图像生成
            # A shot being redrawn goes through the same path as one the plan calls
            # incomplete: its first frame is drawn from the camera still *and the character
            # portraits*, with the review note applied. Copying the still instead — which is
            # what a camera with nothing missing does — carries the video model's own
            # invention of the characters straight into the frame, so a redraw of it can
            # never fix a wrong-looking character.
            if camera.parent_shot_idx is None or camera.missing_info is not None or self._is_redrawn(first_shot_idx):
                mandatory_pairs = self._mandatory_frame_references(first_shot_idx, scene_source_pair)
                reference_pairs, prompt, _selector_output = await self._resolve_frame_references(
                    shot_idx=first_shot_idx,
                    frame_type="first_frame",
                    frame_desc=by_idx[first_shot_idx].ff_desc,
                    available_optional_pairs=available_image_path_and_text_pairs,
                    mandatory_pairs=mandatory_pairs,
                    visible_characters=visible_first_frame,
                    character_portraits_registry=character_portraits_registry,
                )
                reference_image_paths = [path for path, _ in reference_pairs]
                ff_image: ImageOutput = await self._generate_frame_image(
                    shot_idx=first_shot_idx,
                    frame_type="first_frame",
                    prompt=prompt,
                    reference_image_paths=reference_image_paths,
                    mandatory_reference_pairs=mandatory_pairs,
                )
                os.makedirs(os.path.dirname(first_shot_ff_path), exist_ok=True)
                ff_image.save(first_shot_ff_path)
                self._mark_frame_fresh(first_shot_idx, "first_frame")
                self.frame_events[first_shot_idx]["first_frame"].set()
                print(f"☑️ Generated first_frame for shot {first_shot_idx}, saved to {first_shot_ff_path}.")
                _emit_render_progress(progress, "frame_done", f"Generated first frame for shot {first_shot_idx}", {"camera_idx": camera.idx, "shot_idx": first_shot_idx, "frame_type": "first_frame", "path": first_shot_ff_path})
            else:
                os.makedirs(os.path.dirname(first_shot_ff_path), exist_ok=True)
                shutil.copy(new_camera_image_path, first_shot_ff_path)
                self._mark_frame_fresh(first_shot_idx, "first_frame")
                self.frame_events[first_shot_idx]["first_frame"].set()
                print(f"☑️ Generated first_frame for shot {first_shot_idx}, saved to {first_shot_ff_path}.")
                _emit_render_progress(progress, "frame_done", f"Generated first frame for shot {first_shot_idx}", {"camera_idx": camera.idx, "shot_idx": first_shot_idx, "frame_type": "first_frame", "path": first_shot_ff_path})


        # 2. generate the following frames of the camera
        priority_tasks = []
        normal_tasks = []

        # Every shot gets an end keyframe where the model takes one, not only the
        # medium/large variations. The clip is generated between the two frames, and
        # given only a start frame the video model invents the rest of the shot —
        # the room drifts mid-clip. The end frame is referenced from this shot's own
        # start frame so it stays the same scene.
        if self.frame_bracketing and (wanted is None or first_shot_idx in wanted):
            task = self.generate_frame_for_single_shot(
                shot_idx=first_shot_idx,
                frame_type="last_frame",
                source_frame_pair=(first_shot_ff_path, by_idx[first_shot_idx].ff_desc),
                frame_desc=by_idx[first_shot_idx].lf_desc,
                visible_characters=self._visible_characters(by_idx[first_shot_idx], characters, "last_frame"),
                character_portraits_registry=character_portraits_registry,
                progress=progress,
            )
            normal_tasks.append(task)

        for position, shot_idx in enumerate(camera.active_shot_idxs[1:], start=1):
            # A shot the request did not name keeps the frames it has: drawing them would be
            # work nobody asked for, and the card's redraw is about one shot.
            if wanted is not None and shot_idx not in wanted:
                continue
            predecessor_idx = camera.active_shot_idxs[position - 1]
            source_type = "last_frame" if self.frame_bracketing else "first_frame"
            source_path = self.frame_path(predecessor_idx, source_type)
            if wanted is not None and predecessor_idx not in wanted and not os.path.isfile(source_path):
                raise RuntimeError(
                    f"Cannot render scoped shot {shot_idx}: required predecessor shot {predecessor_idx} "
                    f"{source_type} is missing; include that predecessor in only_shots."
                )
            source_event = self.frame_events[predecessor_idx][source_type]
            if os.path.isfile(source_path):
                source_event.set()
            reuse_source = self._playback_predecessors.get(shot_idx)
            if by_idx[shot_idx].reuse_previous_last_frame is True and reuse_source is not None:
                first_frame_task = self._reuse_previous_last_frame(
                    shot_idx, reuse_source, only_shots=wanted, progress=progress
                )
            else:
                first_frame_task = self.generate_frame_for_single_shot(
                    shot_idx=shot_idx,
                    frame_type="first_frame",
                    source_frame_pair=(
                        source_path,
                        by_idx[predecessor_idx].lf_desc if self.frame_bracketing else by_idx[predecessor_idx].ff_desc,
                    ),
                    source_frame_event=source_event,
                    frame_desc=by_idx[shot_idx].ff_desc,
                    visible_characters=self._visible_characters(by_idx[shot_idx], characters, "first_frame"),
                    character_portraits_registry=character_portraits_registry,
                    progress=progress,
                )
            if shot_idx in priority_shot_idxs:
                priority_tasks.append(first_frame_task)
            else:
                normal_tasks.append(first_frame_task)


            if self.frame_bracketing:
                last_frame_task = self.generate_frame_for_single_shot(
                    shot_idx=shot_idx, 
                    frame_type="last_frame", 
                    source_frame_pair=(self.frame_path(shot_idx, "first_frame"), by_idx[shot_idx].ff_desc),
                    frame_desc=by_idx[shot_idx].lf_desc,
                    visible_characters=self._visible_characters(by_idx[shot_idx], characters, "last_frame"),
                    character_portraits_registry=character_portraits_registry,
                    progress=progress,
                )
                normal_tasks.append(last_frame_task)


        # All dependencies must be scheduled together: a priority first frame can
        # depend on an end frame queued among the normal tasks.
        await asyncio.gather(*priority_tasks, *normal_tasks)
        _emit_render_progress(progress, "camera_frames_done", f"Frames for camera {camera.idx} ready", {"camera_idx": camera.idx, "active_shot_idxs": camera.active_shot_idxs})



    async def generate_video_for_single_shot(
        self,
        shot_description: ShotDescription,
        progress: Callable[[str, str, Dict[str, Any] | None], None] | None = None,
    ):
        duration_seconds = self._effective_shot_duration(shot_description)
        video_path = self.clip_path(shot_description.idx)
        if os.path.exists(video_path):
            print(f"🚀 Skipped generating video for shot {shot_description.idx}, already exists.")
            _emit_render_progress(progress, "video_clip_exists", f"Video clip for shot {shot_description.idx} already exists", {"shot_idx": shot_description.idx, "path": video_path})
        else:
            _emit_render_progress(progress, "video_clip_waiting_for_frames", f"Waiting for frames before video clip {shot_description.idx}", {"shot_idx": shot_description.idx})
            # A bracketed clip cannot wander: the model is pinned to both ends.
            # Given only a start frame it is free to invent the rest of the shot,
            # which is what makes the room drift across a clip and the cut into the
            # next shot read as a jump.
            await self.frame_events[shot_description.idx]["first_frame"].wait()
            frame_paths = [self.frame_path(shot_description.idx, "first_frame")]
            if self.frame_bracketing:
                await self.frame_events[shot_description.idx]["last_frame"].wait()
                frame_paths.append(self.frame_path(shot_description.idx, "last_frame"))

            print(f"🎬 Starting video generation for shot {shot_description.idx}...")
            _emit_render_progress(progress, "video_clip_start", f"Generating video clip for shot {shot_description.idx}", {"shot_idx": shot_description.idx, "frame_count": len(frame_paths)})
            video_output = await self.video_generator.generate_single_video(
                prompt=self._review_note(shot_description.idx, shot_description.motion_desc + "\n" + shot_description.audio_desc),
                reference_image_paths=frame_paths,
                duration_seconds=duration_seconds,
                progress=_scoped_progress(progress, shot_idx=shot_description.idx, artifact="video_clip"),
            )
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            video_output.save(video_path)
            print(f"☑️ Generated video for shot {shot_description.idx}, saved to {video_path}.")
            _emit_render_progress(progress, "video_clip_done", f"Generated video clip for shot {shot_description.idx}", {"shot_idx": shot_description.idx, "path": video_path})

    async def generate_frame_for_single_shot(
        self,
        shot_idx: int,
        frame_type: Literal["first_frame", "last_frame"],
        source_frame_pair: Tuple[str, str],
        frame_desc: str,
        visible_characters: List[CharacterInScene],
        character_portraits_registry: Dict[str, Dict[str, Dict[str, str]]],
        progress: Callable[[str, str, Dict[str, Any] | None], None] | None = None,
        source_frame_event: Optional[asyncio.Event] = None,
    ) -> ImageOutput:

        invalidated = self._assert_frame_cache_usable(shot_idx)
        frame_image_path = self.frame_path(shot_idx, frame_type)
        if os.path.exists(frame_image_path) and not self._is_redrawn(shot_idx) and not invalidated:
            _emit_render_progress(progress, "frame_exists", f"{frame_type} for shot {shot_idx} already exists", {"shot_idx": shot_idx, "frame_type": frame_type, "path": frame_image_path})

        else:
            if source_frame_event is not None:
                await source_frame_event.wait()
            if frame_type == "last_frame":
                # The end frame is referenced from this shot's start frame, so it
                # has to exist before the reference list is built.
                await self.frame_events[shot_idx]["first_frame"].wait()
            print(f"🖼️ Starting {frame_type} generation for shot {shot_idx}...")
            _emit_render_progress(progress, "frame_start", f"Generating {frame_type} for shot {shot_idx}", {"shot_idx": shot_idx, "frame_type": frame_type})
            prepare = getattr(self.image_generator, "prepare", None)
            if prepare is not None:
                await prepare()
            available_image_path_and_text_pairs = []
            for visible_character in visible_characters:
                registry_item = character_portraits_registry.get(visible_character.identifier_in_scene) or {}
                for view, item in registry_item.items():
                    path = item.get("path") if isinstance(item, dict) else None
                    if path:
                        available_image_path_and_text_pairs.append(
                            (path, self._portrait_reference_text(visible_character, view))
                        )

            mandatory_pairs = self._mandatory_frame_references(shot_idx, source_frame_pair)
            reference_pairs, prompt, _selector_output = await self._resolve_frame_references(
                shot_idx=shot_idx,
                frame_type=frame_type,
                frame_desc=frame_desc,
                available_optional_pairs=available_image_path_and_text_pairs,
                mandatory_pairs=mandatory_pairs,
                visible_characters=visible_characters,
                character_portraits_registry=character_portraits_registry,
            )
            reference_image_paths = [path for path, _ in reference_pairs]
            frame_image: ImageOutput = await self._generate_frame_image(
                shot_idx=shot_idx,
                frame_type=frame_type,
                prompt=prompt,
                reference_image_paths=reference_image_paths,
                mandatory_reference_pairs=mandatory_pairs,
            )
            os.makedirs(os.path.dirname(frame_image_path), exist_ok=True)
            frame_image.save(frame_image_path)
            self._mark_frame_fresh(shot_idx, frame_type)
            print(f"☑️ Generated {frame_type} frame for shot {shot_idx}, saved to {frame_image_path}.")
            _emit_render_progress(progress, "frame_done", f"Generated {frame_type} for shot {shot_idx}", {"shot_idx": shot_idx, "frame_type": frame_type, "path": frame_image_path})


        self.frame_events[shot_idx][frame_type].set()
        return frame_image_path
    def _shot_slot(self, shot_idx: int) -> str:
        root = canonical_render_root(self.working_dir)
        working_dir = Path(self.working_dir).resolve()
        try:
            prefix = working_dir.relative_to(root).as_posix()
        except ValueError as exc:
            raise RuntimeError(f"Shot directory {working_dir} is outside render root {root}") from exc
        return f"{prefix}/{shot_idx}" if prefix != "." else str(shot_idx)
    def _frame_keyframes_invalidated(self, shot_idx: int) -> bool:
        root = canonical_render_root(self.working_dir)
        acceptance_path = root.parent / "render_acceptance.json"
        try:
            acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return False
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Cannot read frame approvals from {acceptance_path}: {exc}") from exc
        if not isinstance(acceptance, dict):
            raise RuntimeError(f"Frame approvals in {acceptance_path} must be a JSON object")
        root_record = acceptance.get(root.name, {})
        shots = root_record.get("shots", {}) if isinstance(root_record, dict) else {}
        shot_record = shots.get(self._shot_slot(shot_idx), {}) if isinstance(shots, dict) else {}
        keyframes = shot_record.get("keyframes", {}) if isinstance(shot_record, dict) else {}
        return isinstance(keyframes, dict) and bool(keyframes.get("invalidated_at"))

    def _assert_frame_cache_usable(self, shot_idx: int) -> bool:
        invalidated = self._frame_keyframes_invalidated(shot_idx)
        if invalidated and not self._is_redrawn(shot_idx):
            if any(os.path.isfile(self.frame_path(shot_idx, kind)) for kind in ("first_frame", "last_frame")):
                raise RuntimeError(
                    f"Keyframes for shot {self._shot_slot(shot_idx)} were invalidated; explicitly redraw the shot "
                    "after approving its current set references."
                )
        return invalidated

    def _accepted_set_references(self, shot_idx: int) -> List[Tuple[str, str]]:
        return accepted_set_references(
            self.working_dir,
            self._shot_slot(shot_idx),
            getattr(self, "_active_style", ""),
            str(getattr(self.image_generator, "model", "") or "unknown-image-model"),
        )

    def _mandatory_frame_references(
        self,
        shot_idx: int,
        scene_source_pair: Optional[Tuple[str, str]] = None,
    ) -> List[Tuple[str, str]]:
        pairs = self._accepted_set_references(shot_idx)
        if scene_source_pair is not None:
            if scene_source_pair[0] in {path for path, _ in pairs}:
                raise RuntimeError(f"Shot {self._shot_slot(shot_idx)} uses one image as both its set and scene source.")
            pairs = [*pairs, scene_source_pair]
        limit = self.reference_image_limit()
        if limit is not None and len(pairs) > limit:
            raise RuntimeError(
                f"Shot {self._shot_slot(shot_idx)} needs {len(pairs)} mandatory set/scene references, "
                f"but this image model accepts only {limit}; no image was generated."
            )
        return pairs

    def _validate_frame_bindings(
        self,
        shot_idx: int,
        prompt: str,
        pairs: List[Tuple[str, str]],
        mandatory_pairs: List[Tuple[str, str]],
    ) -> None:
        slot = self._shot_slot(shot_idx)
        if pairs[:len(mandatory_pairs)] != mandatory_pairs:
            raise RuntimeError(f"Shot {slot} lost or reordered a mandatory set/scene reference; no image was generated.")
        limit = self.reference_image_limit()
        if limit is not None and len(pairs) > limit:
            raise RuntimeError(f"Shot {slot} exceeds the image model reference limit; no image was generated.")
        missing = [path for path, _ in pairs if not os.path.isfile(path)]
        if missing:
            raise RuntimeError(f"Shot {slot} references missing images: {', '.join(missing)}; no image was generated.")
        invalid = sorted({int(match.group(1)) for match in re.finditer(r"Image[ ]+([0-9]+)", prompt)
                          if int(match.group(1)) >= len(pairs)})
        if invalid:
            raise RuntimeError(
                f"Shot {slot} prompt refers to unavailable image binding(s) {invalid}; no image was generated."
            )

    async def _resolve_frame_references(
        self,
        *,
        shot_idx: int,
        frame_type: str,
        frame_desc: str,
        available_optional_pairs: List[Tuple[str, str]],
        mandatory_pairs: List[Tuple[str, str]],
        visible_characters: List[CharacterInScene],
        character_portraits_registry: Dict[str, Dict[str, Dict[str, str]]],
    ) -> Tuple[List[Tuple[str, str]], str, Dict[str, Any]]:
        limit = self.reference_image_limit()
        optional_limit = None if limit is None else limit - len(mandatory_pairs)
        if optional_limit is not None and optional_limit < 0:
            raise RuntimeError(f"Shot {self._shot_slot(shot_idx)} has no capacity for its mandatory references.")
        mandatory_json = [list(pair) for pair in mandatory_pairs]
        mandatory_fingerprints = {path: self._sha256_file(path) for path, _text in mandatory_pairs}
        current_note = (getattr(self, "revision_notes", None) or {}).get(str(shot_idx), "").strip()
        cached = self.load_selector_output(shot_idx, frame_type)
        if (
            cached is not None
            and cached.get("mandatory_references") == mandatory_json
            and cached.get("mandatory_reference_fingerprints") == mandatory_fingerprints
            and cached.get("review_note", "") == current_note
        ):
            raw_pairs = cached.get("reference_image_path_and_text_pairs")
            try:
                cached_pairs = [(str(pair[0]), str(pair[1])) for pair in raw_pairs]
            except (TypeError, IndexError):
                cached_pairs = []
            cached_prompt = cached.get("sent_prompt")
            if (
                isinstance(cached_prompt, str)
                and cached_pairs[:len(mandatory_pairs)] == mandatory_pairs
                and (optional_limit is None or len(cached_pairs) - len(mandatory_pairs) <= optional_limit)
            ):
                self._validate_frame_bindings(shot_idx, cached_prompt, cached_pairs, mandatory_pairs)
                return cached_pairs, cached_prompt, cached

        optional_available = []
        seen = {path for path, _ in mandatory_pairs}
        for path, text in available_optional_pairs:
            if path in seen or path in {existing for existing, _ in optional_available} or not os.path.isfile(path):
                continue
            optional_available.append((path, text))
        if optional_limit == 0:
            selector_output = {
                "reference_image_path_and_text_pairs": [],
                "selected_indices": [],
                "text_prompt": frame_desc,
            }
        else:
            selector_output = await self.reference_image_selector.select_reference_images_and_generate_prompt(
                available_image_path_and_text_pairs=optional_available,
                frame_description=frame_desc,
                max_reference_images=optional_limit,
            )
        optional_pairs, optional_prompt = self._align_reference_mentions(
            optional_available,
            selector_output["reference_image_path_and_text_pairs"],
            selector_output.get("selected_indices") or [],
            selector_output["text_prompt"],
            reference_limit=optional_limit,
        )
        offered = dict(optional_available)
        if any(path not in offered for path, _text in optional_pairs):
            raise RuntimeError(f"Shot {self._shot_slot(shot_idx)} selector returned an unoffered image reference.")
        optional_pairs = [(path, offered[path]) for path, _text in optional_pairs]
        optional_pairs, optional_prompt = self._ensure_character_portraits(
            visible_characters,
            character_portraits_registry,
            optional_pairs,
            optional_prompt,
            reserved_reference_count=len(mandatory_pairs),
        )
        if optional_limit is not None and len(optional_pairs) > optional_limit:
            raise RuntimeError(
                f"Shot {self._shot_slot(shot_idx)} selected more optional images than its reserved reference budget."
            )
        optional_prompt = re.sub(
            r"Image[ ]+([0-9]+)",
            lambda match: f"Image {int(match.group(1)) + len(mandatory_pairs)}",
            optional_prompt,
        )
        pairs = [*mandatory_pairs, *optional_pairs]
        bindings = chr(10).join(f"Image {index}: {text}" for index, (_path, text) in enumerate(pairs))
        prompt = f"{bindings}{chr(10)}{chr(10)}{optional_prompt}"
        prompt = self._review_note(shot_idx, prompt)
        self._validate_frame_bindings(shot_idx, prompt, pairs, mandatory_pairs)
        result = {
            **selector_output,
            "mandatory_references": mandatory_json,
            "mandatory_reference_fingerprints": mandatory_fingerprints,
            "reference_image_path_and_text_pairs": pairs,
            "sent_prompt": prompt,
            "review_note": current_note,
        }
        selector_path = self.selector_output_path(shot_idx, frame_type)
        os.makedirs(os.path.dirname(selector_path), exist_ok=True)
        with open(selector_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=4)
        return pairs, prompt, result
    def _preflight_pending_frame_sets(
        self,
        camera_tree: List[Camera],
        shot_descriptions: List[ShotDescription],
        only_shots: Optional[List[int]] = None,
        *,
        chained: bool = False,
    ) -> None:
        wanted = {int(idx) for idx in only_shots} if only_shots is not None else None
        ordered = self._ordered_active_shots(camera_tree, shot_descriptions)
        descriptions_by_idx = self._descriptions_by_idx(shot_descriptions)
        camera_by_shot = {
            shot_idx: camera
            for camera in sorted(camera_tree, key=lambda item: item.idx)
            for shot_idx in camera.active_shot_idxs
        }
        pending: Dict[int, int] = {}
        if chained:
            for position, shot in enumerate(ordered):
                if wanted is not None and shot.idx not in wanted:
                    continue
                camera = camera_by_shot[shot.idx]
                invalidated = self._assert_frame_cache_usable(shot.idx)
                previous = ordered[position - 1] if position else None
                same_camera = previous is not None and camera_by_shot[previous.idx].idx == camera.idx
                reuse_previous = (
                    shot.reuse_previous_last_frame
                    if shot.reuse_previous_last_frame is not None
                    else same_camera
                )
                for frame_type in ("first_frame", "last_frame"):
                    if os.path.isfile(self.frame_path(shot.idx, frame_type)) and not self._is_redrawn(shot.idx) and not invalidated:
                        continue
                    needs_source = frame_type == "last_frame" or (
                        previous is not None and not reuse_previous
                    )
                    if position == 0 and frame_type == "first_frame":
                        external_previous = self._external_playback_predecessor()
                        if external_previous is not None and shot.reuse_previous_last_frame is False:
                            needs_source = True
                        elif external_previous is not None and shot.reuse_previous_last_frame is True:
                            needs_source = False
                        else:
                            needs_source = camera.parent_shot_idx is not None and (
                                camera.missing_info is not None or self._is_redrawn(shot.idx)
                            )
                    pending[shot.idx] = max(pending.get(shot.idx, 0), int(needs_source))
        else:
            frame_types = ("first_frame", "last_frame") if self.frame_bracketing else ("first_frame",)
            for camera in camera_tree:
                if not camera.active_shot_idxs:
                    continue
                first_idx = camera.active_shot_idxs[0]
                for shot_idx in camera.active_shot_idxs:
                    if wanted is not None and shot_idx not in wanted:
                        continue
                    invalidated = self._assert_frame_cache_usable(shot_idx)
                    for frame_type in frame_types:
                        if os.path.isfile(self.frame_path(shot_idx, frame_type)) and not self._is_redrawn(shot_idx) and not invalidated:
                            continue
                        shot = descriptions_by_idx[shot_idx]
                        reuse_previous = (
                            shot.reuse_previous_last_frame is True
                            and shot_idx in self._playback_predecessors
                        )
                        needs_source = frame_type == "last_frame" or (
                            shot_idx != first_idx and not reuse_previous
                        )
                        if shot_idx == first_idx:
                            needs_source = not reuse_previous and camera.parent_shot_idx is not None and (
                                camera.missing_info is not None or self._is_redrawn(shot_idx)
                            )
                        pending[shot_idx] = max(pending.get(shot_idx, 0), int(needs_source))

        limit = self.reference_image_limit()
        for shot_idx in sorted(pending):
            references = self._accepted_set_references(shot_idx)
            required = len(references) + pending[shot_idx]
            if limit is not None and required > limit:
                raise RuntimeError(
                    f"Shot {self._shot_slot(shot_idx)} needs {required} mandatory set/scene references, "
                    f"but this image model accepts only {limit}; no shot image was generated."
                )


    def _portrait_reference_text(self, character: CharacterInScene, view: str) -> str:
        """Bind the canonical name and portrait view without repeating a profile."""
        return f"{character.identifier_in_scene}: {view}-view portrait."

    def _descriptions_by_idx(self, shot_descriptions: List[ShotDescription]) -> Dict[int, ShotDescription]:
        """Shot descriptions keyed by the shot's own number.

        A shot's number is its identity, not its place in the list. Removing a shot from the
        film leaves the rest numbered as they were, so reading the list by position draws a
        shot from another shot's description — and past the end it raises instead. Keyed by
        number, both the frames and the clips of a shot find their own plan.
        """
        return {description.idx: description for description in shot_descriptions}

    def _visible_characters(self, shot_description, characters: List[CharacterInScene], frame_type: str) -> List[CharacterInScene]:
        """Who a frame may show: the plan's list for that frame, plus anyone the frame names.

        The plan's chips (`ff_vis_char_idxs` / `lf_vis_char_idxs`) decide whose portraits the
        selector is even offered, and they are often thin: a frame can describe a character
        entering while chipping only the ones who were already there, and then the model draws
        the newcomer from the description alone — a DeepSeek who is not DeepSeek, because his
        portrait was never sent. So whoever the frame's own description names is offered too.

        The *brief* is deliberately not read. It describes the whole shot, so by the end of a
        shot it still names the characters who have left: a frame saying "Only Claude is in the
        room" was being handed the Wife's portrait and a binding for her, which made the prompt
        contradict itself and sent references nobody asked for.
        """
        listed = shot_description.ff_vis_char_idxs if frame_type == "first_frame" else shot_description.lf_vis_char_idxs
        indices = [idx for idx in (listed or []) if 0 <= idx < len(characters)]
        frame_desc = shot_description.ff_desc if frame_type == "first_frame" else shot_description.lf_desc
        spoken = str(frame_desc or "").lower()
        for character in characters:
            if character.idx in indices:
                continue
            name = str(character.identifier_in_scene or "").strip().lower()
            if name and re.search(rf"\b{re.escape(name)}\b", spoken):
                indices.append(character.idx)
        return [characters[idx] for idx in indices]

    def _is_redrawn(self, shot_idx: int) -> bool:
        """Whether this shot is being redrawn from a note rather than restored as it was."""
        return bool((getattr(self, "revision_notes", None) or {}).get(str(shot_idx), "").strip())

    def _review_note(self, shot_idx: int, prompt: str) -> str:
        """Append the guidance typed against a rejected shot to what is redrawn.

        The note is the user's own words about what was wrong, so it goes in verbatim:
        a redraw that does not answer it is a redraw they have to pay for twice.
        """
        note = (getattr(self, "revision_notes", None) or {}).get(str(shot_idx), "").strip()
        if not note:
            return prompt
        return (
            f"{prompt}\n\n<REVIEW_CORRECTION>\n"
            f"A human reviewed an earlier version of this shot and rejected it. What is wrong: {note}\n"
            f"Redraw so that this problem is gone, keeping everything else this description asks for.\n"
            f"</REVIEW_CORRECTION>"
        )

    def _align_reference_mentions(
        self,
        available_image_path_and_text_pairs: List[Tuple[str, str]],
        reference_image_path_and_text_pairs: List[Tuple[str, str]],
        selected_indices: List[int],
        prompt: str,
        reference_limit: Optional[int] = None,
    ) -> Tuple[List[Tuple[str, str]], str]:
        """Make the prompt's ``Image N`` mentions address the images actually sent.

        The selector is told to number its mentions by position in its own
        selection, but it routinely numbers them by position in the list it was
        shown instead: in one real session seven of seventeen cached prompts told
        the image model to use an image that was never sent, leaving the frame
        to be invented. Mentions are renumbered to the sent positions, and an
        image the prompt asks for that the selector did not return is appended
        when the model's reference limit allows it.
        """
        if not selected_indices:
            return reference_image_path_and_text_pairs, prompt
        position_by_available = {index: position for position, index in enumerate(selected_indices)}
        pairs = list(reference_image_path_and_text_pairs)
        limit = self.reference_image_limit() if reference_limit is None else reference_limit
        appended: Dict[int, int] = {}

        def renumber(match):
            index = int(match.group(1))
            if index < len(reference_image_path_and_text_pairs):
                # The selector numbers its mentions by position in its own selection, and
                # that selection is sent first, so "Image 0" is the image it picked first.
                # Reading it as an index into the list it was shown instead sent one real
                # frame's room, couch and environment mentions to a character's portrait:
                # the portraits are shown first and the setting of the shot last.
                return match.group(0)
            if index in position_by_available:
                return f"Image {position_by_available[index]}"
            if index not in appended:
                if index >= len(available_image_path_and_text_pairs) or (limit is not None and len(pairs) >= limit):
                    # Nothing to point the mention at: keep the reference list as it is.
                    return match.group(0)
                appended[index] = len(pairs)
                pairs.append(available_image_path_and_text_pairs[index])
            return f"Image {appended[index]}"

        return pairs, re.sub(r"Image\s+(\d+)", renumber, prompt)

    def _ensure_character_portraits(
        self,
        visible_characters: List[CharacterInScene],
        character_portraits_registry: Dict[str, Dict[str, Dict[str, str]]],
        reference_image_path_and_text_pairs: List[Tuple[str, str]],
        prompt: str,
        reserved_reference_count: int = 0,
    ) -> Tuple[List[Tuple[str, str]], str]:
        """Give the image model a face for every character the frame shows.

        The reference selection comes back with whatever it judged useful, and for a
        character that can be their *back* portrait — which carries the costume and no face —
        or none of their portraits at all. Either way the frame is drawn from the description
        alone and the character stops looking like themselves: a Claude whose face was
        invented, a Wife whose grey dress came from nowhere. A front portrait is added (or a
        back portrait replaced) when the model's reference limit allows.
        """
        limit = self.reference_image_limit()
        if limit is not None:
            limit = max(0, limit - reserved_reference_count)
        pairs = list(reference_image_path_and_text_pairs)
        backed: list[str] = []
        for character in visible_characters:
            registry_item = character_portraits_registry.get(character.identifier_in_scene) or {}
            front, back = registry_item.get("front"), registry_item.get("back")
            if not front or front["path"] in {path for path, _ in pairs}:
                continue
            sent = {path for path, _ in pairs}
            # A face matters more than the view: the description still sets the pose.
            # A side portrait is not much of a face either. It was treated as good
            # enough to leave alone, and given Claude's side portrait for a standing
            # man the image model bound the heavyset profile to a body that was not
            # his -- a second Claude in the frame, next to the real one. Side is
            # swapped for front exactly as back is; only the front portrait settles
            # who a character is.
            side = next((item["path"] for view, item in registry_item.items() if view == "side"), None)
            weak = next((path for path in (side, (back or {}).get("path")) if path in sent), None)
            if weak is not None:
                pairs = [(path, text) if path != weak else (front["path"], self._portrait_reference_text(character, front)) for path, text in pairs]
            elif limit is None or len(pairs) < limit:
                pairs.append((front["path"], self._portrait_reference_text(character, front)))
            else:
                continue
            backed.append(character.identifier_in_scene)
        if backed:
            # Naming the character is not enough on a frame with three of them and four
            # references: the model needs to be told which image is whom, or it binds a
            # heavyset Claude to the standing man and DeepSeek comes out as a second Claude.
            positions = {path: index for index, (path, _) in enumerate(pairs)}
            lines = []
            for character in visible_characters:
                if character.identifier_in_scene not in backed:
                    continue
                front = (character_portraits_registry.get(character.identifier_in_scene) or {}).get("front") or {}
                if front.get("path") in positions:
                    lines.append(
                        f"Image {positions[front['path']]} is {character.identifier_in_scene}: keep that character's face, hair and clothing."
                    )
            if lines:
                prompt = f"{prompt}\n\nWho is who in the references:\n" + "\n".join(lines)
        return pairs, prompt


    async def _generate_frame_image(
        self,
        *,
        shot_idx: int,
        frame_type: str,
        prompt: str,
        reference_image_paths: List[str],
        mandatory_reference_pairs: Optional[List[Tuple[str, str]]] = None,
    ) -> ImageOutput:
        """POST a frame only with current accepted sets and intact scene references."""
        prepare = getattr(self.image_generator, "prepare", None)
        if prepare is not None:
            await prepare()
        slot = self._shot_slot(shot_idx)
        approved_sets = self._accepted_set_references(shot_idx)
        if mandatory_reference_pairs is None:
            mandatory = approved_sets
            existing_sets = [path for path, _ in approved_sets]
            already_prefixed = reference_image_paths[:len(existing_sets)] == existing_sets
            optional_paths = [path for path in reference_image_paths if path not in existing_sets]
            reference_image_paths = existing_sets + optional_paths
            if not already_prefixed:
                prompt = re.sub(
                    r"Image[ ]+([0-9]+)",
                    lambda match: f"Image {int(match.group(1)) + len(mandatory)}",
                    prompt,
                )
                bindings = chr(10).join(
                    f"Image {index}: {text}" for index, (_path, text) in enumerate(mandatory)
                )
                prompt = f"{bindings}{chr(10)}{prompt}"
        else:
            mandatory = list(mandatory_reference_pairs)
            if mandatory[:len(approved_sets)] != approved_sets:
                raise RuntimeError(f"Shot {slot} set references changed before image generation.")
            expected_paths = [path for path, _ in mandatory]
            if reference_image_paths[:len(expected_paths)] != expected_paths:
                raise RuntimeError(f"Shot {slot} lost or reordered mandatory references; no image was generated.")
        limit = self.reference_image_limit()
        if limit is not None and len(reference_image_paths) > limit:
            raise RuntimeError(
                f"Shot {slot} needs {len(reference_image_paths)} references, but this image model accepts only "
                f"{limit}; no image was generated."
            )
        missing = [path for path in reference_image_paths if not os.path.isfile(path)]
        if missing:
            raise RuntimeError(f"Shot {slot} references missing images: {', '.join(missing)}; no image was generated.")
        if not prompt.startswith(chr(10).join(
            f"Image {index}: {text}" for index, (_path, text) in enumerate(mandatory)
        )):
            raise RuntimeError(f"Shot {slot} prompt is missing mandatory image bindings; no image was generated.")
        invalid = sorted({
            int(match.group(1))
            for match in re.finditer(r"Image[ ]+([0-9]+)", prompt)
            if int(match.group(1)) >= len(reference_image_paths)
        })
        if invalid:
            raise RuntimeError(f"Shot {slot} prompt refers to unavailable image binding(s) {invalid}; no image was generated.")
        try:
            return await self.image_generator.generate_single_image(
                prompt=prompt,
                reference_image_paths=reference_image_paths,
                size="1600x900",
            )
        except Exception as exc:
            raise RuntimeError(f"Image generation failed for the {frame_type} of shot {shot_idx}: {exc}") from exc

    async def construct_camera_tree(
        self,
        shot_descriptions: List[ShotDescription],
        quiet: bool = False,
    ):
        camera_tree_path = os.path.join(self.working_dir, "camera_tree.json")

        if os.path.exists(camera_tree_path):
            with open(camera_tree_path, "r", encoding="utf-8") as f:
                camera_tree = json.load(f)
            camera_tree = [Camera.model_validate(camera) for camera in camera_tree]
            _pipeline_print(quiet, f"🚀 Loaded {len(camera_tree)} cameras from existing file.")
            return camera_tree

        shot_descriptions = _normalize_model_list(shot_descriptions, ShotDescription, "shot_descriptions")
        cameras = _group_shots_into_cameras(shot_descriptions)

        camera_tree = await self.camera_image_generator.construct_camera_tree(cameras=cameras, shot_descs=shot_descriptions)
        camera_tree = _normalize_model_list(camera_tree, Camera, "camera_tree")
        with open(camera_tree_path, "w", encoding="utf-8") as f:
            json.dump([camera.model_dump() for camera in camera_tree], f, ensure_ascii=False, indent=4)
        _pipeline_print(quiet, f"✅ Constructed camera tree and saved to {camera_tree_path}.")
        return camera_tree




    async def extract_characters(
        self,
        script: str,
        quiet: bool = False,
    ):
        save_path = os.path.join(self.working_dir, "characters.json")

        if os.path.exists(save_path):
            with open(save_path, "r", encoding="utf-8") as f:
                characters = json.load(f)
            characters = [CharacterInScene.model_validate(character) for character in characters]
            _pipeline_print(quiet, f"🚀 Loaded {len(characters)} characters from existing file.")
        else:
            characters = await self.character_extractor.extract_characters(script)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump([character.model_dump() for character in characters], f, ensure_ascii=False, indent=4)
            _pipeline_print(quiet, f"✅ Extracted {len(characters)} characters from script and saved to {save_path}.")

        for character in characters:
            self.character_portrait_events[character.idx] = asyncio.Event()

        return characters


    async def generate_character_portraits(
        self,
        characters: List[CharacterInScene],
        character_portraits_registry: Optional[Dict[str, Dict[str, Dict[str, str]]]],
        style: str,
        progress: Callable[[str, str, Dict[str, Any] | None], None] | None = None,
    ):
        registry_path = self.portraits_registry_path()
        os.makedirs(self.portraits_dir(), exist_ok=True)
        if character_portraits_registry is None:
            try:
                with open(registry_path, encoding="utf-8") as handle:
                    character_portraits_registry = json.load(handle)
            except (FileNotFoundError, json.JSONDecodeError):
                character_portraits_registry = {}
        registry = character_portraits_registry if isinstance(character_portraits_registry, dict) else {}
        visible = [character for character in characters if character.is_visible]
        registry = {c.identifier_in_scene: registry[c.identifier_in_scene]
                    for c in visible if c.identifier_in_scene in registry}
        failures = {}
        pending = []
        partial_entries = {}

        def complete(entry):
            return isinstance(entry, dict) and all(
                isinstance(entry.get(view), dict)
                and isinstance(entry[view].get("path"), str)
                and os.path.isfile(entry[view]["path"])
                for view in ("front", "side", "back")
            )

        def persist_registry():
            with open(registry_path, "w", encoding="utf-8") as handle:
                json.dump(registry, handle, ensure_ascii=False, indent=4)

        for character in visible:
            name = character.identifier_in_scene
            directory = os.path.join(self.portraits_dir(), f"{character.idx}_{safe_path_component(name)}")
            os.makedirs(directory, exist_ok=True)
            provenance_path = os.path.join(directory, "provenance.json")
            fingerprint = hashlib.sha256(json.dumps(
                {"character": character.model_dump(), "style": style, "model": self.image_model_slug},
                sort_keys=True, ensure_ascii=False,
            ).encode("utf-8")).hexdigest()
            try:
                with open(provenance_path, encoding="utf-8") as handle:
                    provenance = json.load(handle)
            except (FileNotFoundError, json.JSONDecodeError):
                provenance = {}
            if not isinstance(provenance, dict):
                provenance = {}
            entry = registry.get(name)
            changed = provenance.get("fingerprint") not in (None, fingerprint)
            forced = entry is None and provenance.get("complete") is True
            invalidate = changed or forced or (not provenance and not entry)
            if invalidate:
                # Only invalidate this pipeline's files, never supplied shared portraits.
                for filename in ("front.png", "side.png", "back.png", "prompts.json"):
                    path = os.path.join(directory, filename)
                    if os.path.isfile(path):
                        os.unlink(path)
                entry = {}
            if not complete(entry) and isinstance(entry, dict):
                partial_entries[name] = {
                    view: item for view, item in entry.items()
                    if view in ("front", "side", "back") and isinstance(item, dict)
                    and isinstance(item.get("path"), str) and os.path.isfile(item["path"])
                }
            if changed or forced or not complete(entry):
                if partial_entries.get(name):
                    registry[name] = partial_entries[name]
                else:
                    registry.pop(name, None)
                pending.append(character)
                ready = False
            else:
                ready = True
                self.character_portrait_events.setdefault(character.idx, asyncio.Event()).set()
                for view in ("front", "side", "back"):
                    _emit_render_progress(progress, f"character_portrait_{view}_exists",
                                          f"Reusing {view} portrait for {name}", {"path": entry[view]["path"]})
            with open(provenance_path, "w", encoding="utf-8") as handle:
                json.dump({"fingerprint": fingerprint, "complete": ready}, handle)
        persist_registry()

        async def generate(character):
            try:
                result = await self.generate_portraits_for_single_character(
                    character, style, progress=progress,
                    existing_entry=partial_entries.get(character.identifier_in_scene),
                )
                return character, result, None
            except Exception as error:
                return character, None, error

        for future in asyncio.as_completed([generate(character) for character in pending]):
            character, result, error = await future
            name = character.identifier_in_scene
            if error is not None:
                failures[name] = str(error)
                _emit_render_progress(progress, "character_portrait_failed",
                                      f"Portraits failed for {name}: {error}", {"identifier": name})
                continue
            registry.update(result)
            persist_registry()
            directory = os.path.join(self.portraits_dir(), f"{character.idx}_{safe_path_component(name)}")
            provenance_path = os.path.join(directory, "provenance.json")
            with open(provenance_path, encoding="utf-8") as handle:
                provenance = json.load(handle)
            provenance["complete"] = True
            with open(provenance_path, "w", encoding="utf-8") as handle:
                json.dump(provenance, handle)

        missing = []
        for character in visible:
            if not complete(registry.get(character.identifier_in_scene)):
                directory = os.path.join(self.portraits_dir(), f"{character.idx}_{safe_path_component(character.identifier_in_scene)}")
                entry = registry.get(character.identifier_in_scene, {})
                missing_before = len(missing)
                for view in ("front", "side", "back"):
                    item = entry.get(view, {}) if isinstance(entry, dict) else {}
                    registered_path = item.get("path") if isinstance(item, dict) else None
                    canonical_path = os.path.join(directory, f"{view}.png")
                    if not (isinstance(registered_path, str) and os.path.isfile(registered_path)) and not os.path.isfile(canonical_path):
                        missing.append(os.path.relpath(canonical_path, self.working_dir))
                if len(missing) == missing_before:
                    # Files alone are not a completed, registered character batch.
                    missing.extend(os.path.relpath(os.path.join(directory, f"{view}.png"), self.working_dir)
                                   for view in ("front", "side", "back"))
        if missing or failures:
            raise PortraitGenerationError(missing, failures, working_dir=self.working_dir)
        _emit_render_progress(progress, "character_portraits_done", "Character portraits ready",
                              {"character_count": len(registry)})
        return registry


    async def generate_portraits_for_single_character(
        self,
        character: CharacterInScene,
        style: str,
        progress: Callable[[str, str, Dict[str, Any] | None], None] | None = None,
        existing_entry: Optional[Dict[str, Dict[str, str]]] = None,
    ):
        character_dir = os.path.join(self.portraits_dir(), f"{character.idx}_{safe_path_component(character.identifier_in_scene)}")
        os.makedirs(character_dir, exist_ok=True)
        _emit_render_progress(progress, "character_portrait_start", f"Generating portraits for {character.identifier_in_scene}", {"character_idx": character.idx, "identifier": character.identifier_in_scene})

        def portrait_path(view):
            item = (existing_entry or {}).get(view, {})
            path = item.get("path")
            return path if isinstance(path, str) and os.path.isfile(path) else os.path.join(character_dir, f"{view}.png")

        front_portrait_path = portrait_path("front")
        if os.path.isfile(front_portrait_path):
            _emit_render_progress(progress, "character_portrait_front_exists", f"Reusing front portrait for {character.identifier_in_scene}", {"path": front_portrait_path})
        else:
            _emit_render_progress(progress, "character_portrait_front_start", f"Generating front portrait for {character.identifier_in_scene}", {"character_idx": character.idx, "identifier": character.identifier_in_scene})
            front_portrait_output = await self.character_portraits_generator.generate_front_portrait(character, style)
            front_portrait_output.save(front_portrait_path)
            _emit_render_progress(progress, "character_portrait_front_done", f"Generated front portrait for {character.identifier_in_scene}", {"character_idx": character.idx, "identifier": character.identifier_in_scene, "path": front_portrait_path})


        side_portrait_path = portrait_path("side")
        if os.path.isfile(side_portrait_path):
            _emit_render_progress(progress, "character_portrait_side_exists", f"Reusing side portrait for {character.identifier_in_scene}", {"path": side_portrait_path})
        else:
            _emit_render_progress(progress, "character_portrait_side_start", f"Generating side portrait for {character.identifier_in_scene}", {"character_idx": character.idx, "identifier": character.identifier_in_scene})
            side_portrait_output = await self.character_portraits_generator.generate_side_portrait(character, front_portrait_path)
            side_portrait_output.save(side_portrait_path)
            _emit_render_progress(progress, "character_portrait_side_done", f"Generated side portrait for {character.identifier_in_scene}", {"character_idx": character.idx, "identifier": character.identifier_in_scene, "path": side_portrait_path})

        back_portrait_path = portrait_path("back")
        if os.path.isfile(back_portrait_path):
            _emit_render_progress(progress, "character_portrait_back_exists", f"Reusing back portrait for {character.identifier_in_scene}", {"path": back_portrait_path})
        else:
            _emit_render_progress(progress, "character_portrait_back_start", f"Generating back portrait for {character.identifier_in_scene}", {"character_idx": character.idx, "identifier": character.identifier_in_scene})
            back_portrait_output = await self.character_portraits_generator.generate_back_portrait(character, front_portrait_path)
            back_portrait_output.save(back_portrait_path)
            _emit_render_progress(progress, "character_portrait_back_done", f"Generated back portrait for {character.identifier_in_scene}", {"character_idx": character.idx, "identifier": character.identifier_in_scene, "path": back_portrait_path})

        # Record what each portrait was drawn from, beside the portraits themselves, so
        # the Artifacts view can show the prompt rather than guess it.
        recorded_prompts = {
            view: self.character_portraits_generator.prompts.get((character.identifier_in_scene, view), "")
            for view in ("front", "side", "back")
        }
        with open(os.path.join(character_dir, "prompts.json"), "w", encoding="utf-8") as f:
            json.dump({view: prompt for view, prompt in recorded_prompts.items() if prompt}, f, ensure_ascii=False, indent=4)

        self.character_portrait_events.setdefault(character.idx, asyncio.Event()).set()

        print(f"☑️ Completed character portrait generation for {character.identifier_in_scene}.")
        _emit_render_progress(progress, "character_portrait_done", f"Portraits for {character.identifier_in_scene} ready", {"character_idx": character.idx, "identifier": character.identifier_in_scene})

        return {
            character.identifier_in_scene: {
                "front": {
                    "path": front_portrait_path,
                    "description": f"A front view portrait of {character.identifier_in_scene}.",
                },
                "side": {
                    "path": side_portrait_path,
                    "description": f"A side view portrait of {character.identifier_in_scene}.",
                },
                "back": {
                    "path": back_portrait_path,
                    "description": f"A back view portrait of {character.identifier_in_scene}.",
                },
            }
        }



    async def design_storyboard(
        self,
        script: str,
        characters: List[CharacterInScene],
        user_requirement: str,
        quiet: bool = False,
        clip_seconds: Optional[int] = None,
    ):
        storyboard_path = os.path.join(self.working_dir, "storyboard.json")
        if os.path.exists(storyboard_path):
            with open(storyboard_path, 'r', encoding='utf-8') as f:
                storyboard = json.load(f)
            storyboard = [ShotBriefDescription.model_validate(shot) for shot in storyboard]
            _pipeline_print(quiet, f"🚀 Loaded {len(storyboard)} shot brief descriptions from existing file.")
        else:
            _pipeline_print(quiet, f"🔍 Designing storyboard...")
            storyboard = await self.storyboard_artist.design_storyboard(
                script=script,
                characters=characters,
                user_requirement=user_requirement,
                retry_timeout=150,
                clip_seconds=clip_seconds if clip_seconds is not None else self.clip_seconds(),
            )
            storyboard = _normalize_model_list(storyboard, ShotBriefDescription, "storyboard")
            with open(storyboard_path, 'w', encoding='utf-8') as f:
                json.dump([shot.model_dump() for shot in storyboard], f, ensure_ascii=False, indent=4)
            _pipeline_print(quiet, f"✅ Designed storyboard and saved to {storyboard_path}.")

        for shot_brief_description in storyboard:
            self.shot_desc_events[shot_brief_description.idx] = asyncio.Event()

        return storyboard



    async def decompose_visual_descriptions(
        self,
        shot_brief_descriptions: List[ShotBriefDescription],
        characters: List[CharacterInScene],
        quiet: bool = False,
    ):
        tasks = [
            self.decompose_visual_description_for_single_shot_brief_description(shot_brief_description, characters, quiet=quiet)
            for shot_brief_description in shot_brief_descriptions
        ]

        shot_descriptions = await asyncio.gather(*tasks)
        return shot_descriptions


    async def decompose_visual_description_for_single_shot_brief_description(
        self,
        shot_brief_description: ShotBriefDescription,
        characters: List[CharacterInScene],
        quiet: bool = False,
    ):
        shot_description_path = os.path.join(self.working_dir, "shots", f"{shot_brief_description.idx}", "shot_description.json")
        os.makedirs(os.path.dirname(shot_description_path), exist_ok=True)

        if os.path.exists(shot_description_path):
            with open(shot_description_path, 'r', encoding='utf-8') as f:
                shot_description = ShotDescription.model_validate(json.load(f))
            _pipeline_print(quiet, f"🚀 Loaded shot {shot_brief_description.idx} description from existing file.")
        else:
            shot_description = await self.storyboard_artist.decompose_visual_description(
                shot_brief_desc=shot_brief_description,
                characters=characters,
                retry_timeout=120,
            )
            shot_description = _normalize_model_list([shot_description], ShotDescription, "shot_description")[0]
            with open(shot_description_path, 'w', encoding='utf-8') as f:
                json.dump(shot_description.model_dump(), f, ensure_ascii=False, indent=4)
            _pipeline_print(quiet, f"✅ Decomposed visual description for shot {shot_brief_description.idx} and saved to {shot_description_path}.")

        self.shot_desc_events[shot_brief_description.idx].set()

        # Both keyframes are tracked for every shot: the clip is bracketed by its
        # start and end frame, and a clip given only a start frame lets the video
        # model reinvent the room across its seconds (see generate_video_for_single_shot).
        self.frame_events[shot_brief_description.idx] = {
            "first_frame": asyncio.Event(),
            "last_frame": asyncio.Event(),
        }
        for frame_type in ("first_frame", "last_frame"):
            if os.path.exists(self.frame_path(shot_brief_description.idx, frame_type)):
                self.frame_events[shot_brief_description.idx][frame_type].set()

        return shot_description
