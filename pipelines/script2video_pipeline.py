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
from utils.text import safe_path_component

from .render_contract import DEFAULT_RENDER_PHASE, ModelScopedArtifacts, RenderOutcome, normalize_phase


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
        self.camera_image_generator = CameraImageGenerator(chat_model=self.chat_model, image_generator=self.image_generator, video_generator=self.video_generator)
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
        self._deferred_frame_shots: set[int] = set()

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
    ) -> RenderOutcome:
        stop_after = normalize_phase(stop_after)
        self.video_budget = video_budget if video_budget is not None else {}
        self._deferred_frame_shots = set()
        # A redo names shots, and a run works on those shots: the stills of the others are left
        # for their own turn rather than drawn because a phase run was the easier way to ask.
        only_set = {int(idx) for idx in only_shots} if only_shots is not None else None
        self.revision_notes = {str(shot): note for shot, note in (revision_notes or {}).items() if str(note).strip()}
        chained_keyframes = self._frame_continuity_config().get("mode") == "chained_keyframes"
        _emit_render_progress(progress, "render_start", "Starting script2video render", {"stop_after": stop_after})
        self.frame_bracketing = await self._brackets_clips()
        if chained_keyframes and not self.frame_bracketing:
            raise RuntimeError(
                "Chained keyframes require a video provider that supports last-frame bracketing; "
                "select a bracketing-capable provider or disable chained_keyframes."
            )
        if characters is None:
            _emit_render_progress(progress, "extract_characters", "Extracting characters before render")
            characters = await self.extract_characters(script=script, quiet=quiet)

        else:
            characters = _normalize_model_list(characters, CharacterInScene, "characters")
            _emit_render_progress(progress, "extract_characters", "Using provided characters for render", {"provided": True, "count": len(characters)})
            for character in characters:
                self.character_portrait_events[character.idx] = asyncio.Event()

        if character_portraits_registry is None:
            character_portraits_registry_path = self.portraits_registry_path()
            if os.path.exists(character_portraits_registry_path):
                with open(character_portraits_registry_path, "r", encoding="utf-8") as f:
                    character_portraits_registry = json.load(f)
                print(f"🚀 Loaded {len(character_portraits_registry)} character portraits from existing file.")
                _emit_render_progress(progress, "character_portraits_loaded", "Loaded existing character portraits", {"count": len(character_portraits_registry)})
            else:
                print(f"🔍 Generating character portraits...")
                _emit_render_progress(progress, "character_portraits_start", "Generating character portraits", {"character_count": len(characters)})
                character_portraits_registry = await self.generate_character_portraits(
                    characters=characters,
                    character_portraits_registry=None,
                    style=style,
                    progress=progress,
                )

                with open(character_portraits_registry_path, "w", encoding="utf-8") as f:
                    json.dump(character_portraits_registry, f, ensure_ascii=False, indent=4)
                print(f"☑️ Generated {len(character_portraits_registry)} character portraits and saved to {character_portraits_registry_path}.")
                _emit_render_progress(progress, "character_portraits_done", "Character portraits ready", {"count": len(character_portraits_registry)})

        if stop_after == "portraits":
            stills = self.portrait_stills(character_portraits_registry)
            _emit_render_progress(
                progress,
                "portraits_ready",
                "Character portraits ready for style review",
                {"character_count": len(character_portraits_registry), "still_count": len(stills), "awaiting_confirmation": "stills", "style": style},
            )
            return self.render_outcome("portraits", style, stills=stills, awaiting="stills")



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

        priority_shot_idxs = [camera.parent_cam_idx for camera in camera_tree if camera.parent_cam_idx is not None]
        if chained_keyframes:
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

    def _record_continuity_source(self, shot_idx: int, source_idx: int, source_path: str, mode: str) -> None:
        selector_path = self.selector_output_path(shot_idx, "first_frame")
        try:
            with open(selector_path, "r", encoding="utf-8") as handle:
                selector = json.load(handle)
            if not isinstance(selector, dict):
                selector = {}
        except (OSError, ValueError):
            selector = {}
        selector["continuity_source"] = {
            "shot_idx": source_idx,
            "frame_type": "last_frame",
            "sha256": self._sha256_file(source_path),
            "mode": mode,
        }
        os.makedirs(os.path.dirname(selector_path), exist_ok=True)
        with open(selector_path, "w", encoding="utf-8") as handle:
            json.dump(selector, handle, ensure_ascii=False, indent=4)

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
        ordered = self._ordered_active_shots(camera_tree, shot_descriptions)
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
        stop_position = len(ordered) - 1
        if wanted is not None:
            if not wanted:
                return
            unknown = wanted - by_idx.keys()
            if unknown:
                raise RuntimeError(f"Unknown chained shot(s): {sorted(unknown)}")
            positions = [position for position, shot in enumerate(ordered) if shot.idx in wanted]
            stop_position = positions[-1]
            excluded_between = [
                ordered[position].idx for position in range(positions[0], stop_position + 1)
                if ordered[position].idx not in wanted
            ]
            if excluded_between:
                raise RuntimeError(
                    f"Chained scope crosses excluded dependent shots {excluded_between}; "
                    "include those predecessors explicitly in only_shots before rendering."
                )
            project_dir = self._continuity_project_dir()
            acceptance_path = os.path.join(project_dir, "render_acceptance.json")
            try:
                with open(acceptance_path, "r", encoding="utf-8") as handle:
                    acceptance = json.load(handle)
            except FileNotFoundError:
                acceptance = {}
            except (OSError, ValueError) as exc:
                raise RuntimeError(f"Cannot read chained frame approvals: {exc}") from exc
            relative = os.path.relpath(self.working_dir, project_dir).split(os.sep)
            root_store = acceptance.get(relative[0], {})
            shot_store = root_store.get("shots", {})
            prefix = "/".join(relative[1:])
            for position in range(stop_position + 1):
                shot = ordered[position]
                if shot.idx in wanted:
                    continue
                slot = f"{prefix}/{shot.idx}" if prefix else str(shot.idx)
                record = shot_store.get(slot, {}).get("keyframes", {})
                if record.get("invalidated_at"):
                    raise RuntimeError(
                        f"Cannot render scoped shot(s) {sorted(wanted)}: excluded predecessor shot {slot} "
                        "has invalidated keyframes; include it explicitly in only_shots."
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
        if wanted is not None and first_shot.idx not in wanted:
            if not all(os.path.isfile(self.frame_path(first_shot.idx, kind)) for kind in ("first_frame", "last_frame")):
                raise RuntimeError(
                    f"Cannot render scoped shot(s) {sorted(wanted)}: required predecessor shot "
                    f"{first_shot.idx} keyframes are missing; include that predecessor in only_shots."
                )
        else:
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

        previous = first_shot
        for position in range(1, stop_position + 1):
            shot = ordered[position]
            previous_last = self.frame_path(previous.idx, "last_frame")
            if not os.path.isfile(previous_last):
                if wanted is not None and previous.idx not in wanted:
                    raise RuntimeError(
                        f"Cannot render scoped shot {shot.idx}: required predecessor shot {previous.idx} "
                        f"last frame is missing; include that predecessor in only_shots."
                    )
                raise RuntimeError(f"Cannot render shot {shot.idx}: predecessor shot {previous.idx} last frame is missing.")
            same_camera = camera_by_shot[shot.idx].idx == camera_by_shot[previous.idx].idx
            expected_mode = "reuse" if same_camera else "reframe"
            expected_hash = self._sha256_file(previous_last)
            if wanted is not None and shot.idx not in wanted:
                skipped_first = self.frame_path(shot.idx, "first_frame")
                try:
                    with open(self.selector_output_path(shot.idx, "first_frame"), "r", encoding="utf-8") as handle:
                        skipped_source = json.load(handle).get("continuity_source")
                except (OSError, ValueError, AttributeError):
                    skipped_source = None
                if (
                    not os.path.isfile(skipped_first)
                    or not isinstance(skipped_source, dict)
                    or skipped_source.get("shot_idx") != previous.idx
                    or skipped_source.get("frame_type") != "last_frame"
                    or skipped_source.get("sha256") != expected_hash
                    or skipped_source.get("mode") != expected_mode
                    or (same_camera and self._sha256_file(skipped_first) != expected_hash)
                ):
                    raise RuntimeError(
                        f"Cannot render scoped shot(s) {sorted(wanted)}: excluded predecessor shot {shot.idx} "
                        f"has missing or stale chained keyframes; include it in only_shots."
                    )
                previous = shot
                continue
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
                and continuity.get("frame_type") == "last_frame"
                and continuity.get("sha256") == expected_hash
                and continuity.get("mode") == expected_mode
                and (not same_camera or self._sha256_file(first_path) == expected_hash)
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
                if same_camera:
                    shutil.copyfile(previous_last, first_path)
                    self._record_continuity_source(shot.idx, previous.idx, previous_last, "reuse")
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
            previous = shot

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
    ) -> None:
        source_type = "last_frame" if frame_type == "first_frame" else "first_frame"
        source = {
            "shot_idx": source_shot.idx,
            "frame_type": source_type,
            "sha256": self._sha256_file(source_path),
        }
        frame_path = self.frame_path(shot.idx, frame_type)
        if os.path.isfile(frame_path):
            selector = self.load_selector_output(shot.idx, frame_type) or {}
            if selector.get("source_frame") != source:
                raise RuntimeError(
                    f"Existing {frame_type} for shot {shot.idx} has stale source state; "
                    "explicitly redraw its keyframes before rendering."
                )
            self.frame_events[shot.idx][frame_type].set()
            return
        visible_indices = set(shot.ff_vis_char_idxs if frame_type == "first_frame" else shot.lf_vis_char_idxs)
        visible = [character for character in characters if character.is_visible and character.idx in visible_indices]
        source_visible = set(source_shot.lf_vis_char_idxs if source_type == "last_frame" else source_shot.ff_vis_char_idxs)
        newcomers = visible_indices - source_visible
        ranked = sorted(visible, key=lambda character: (character.idx not in newcomers, character.idx))
        limit = self.reference_image_limit()
        if limit is not None and limit < 1:
            raise RuntimeError("Chained keyframes require an image model that accepts a scene reference.")
        budget = (limit - 1) if limit is not None else 2
        pairs = [(source_path, "Mandatory scene state: preserve the existing room geometry, camera-relative positions, identities, clothing, props and lighting.")]
        for character in ranked:
            front = (character_portraits_registry.get(character.identifier_in_scene) or {}).get("front")
            if front and len(pairs) - 1 < budget:
                pairs.append((front["path"], f"{character.identifier_in_scene}: approved front portrait. {self._portrait_reference_text(character, front)}"))
        description = shot.ff_desc if frame_type == "first_frame" else shot.lf_desc
        direction = (
            "Reframe the immediately preceding ending at the SAME INSTANT for the new camera; do not advance the action."
            if frame_type == "first_frame" else
            "Use this shot's own first frame as the exact starting state. Keep the camera fixed and change only the specified end-state action."
        )
        prompt = (
            f"{direction} Image 0 is the mandatory scene reference, not a loose inspiration. "
            "Keep all unchanged people, faces, clothes, furniture, held objects and lighting identical to it. "
            "Portrait references define identity only, never their studio background. "
            f"Required frame: {description}"
        )
        prompt = "\n".join(f"Image {index}: {text}" for index, (_, text) in enumerate(pairs)) + "\n\n" + prompt
        selector = {
            "reference_image_path_and_text_pairs": pairs,
            "text_prompt": prompt,
            "sent_prompt": prompt,
            "source_frame": source,
        }
        if frame_type == "first_frame":
            selector["continuity_source"] = {**source, "mode": "reframe"}
        selector_path = self.selector_output_path(shot.idx, frame_type)
        os.makedirs(os.path.dirname(selector_path), exist_ok=True)
        with open(selector_path, "w", encoding="utf-8") as handle:
            json.dump(selector, handle, ensure_ascii=False, indent=4)
        image = await self._generate_frame_image(
            shot_idx=shot.idx, frame_type=frame_type, prompt=prompt,
            reference_image_paths=[path for path, _ in pairs],
        )
        os.makedirs(os.path.dirname(frame_path), exist_ok=True)
        image.save(frame_path)
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
        if os.path.exists(first_shot_ff_path):
            self.frame_events[first_shot_idx]["first_frame"].set()
        elif wanted is not None and first_shot_idx not in wanted:
            raise RuntimeError(
                f"Cannot render scoped shot(s) {sorted(wanted)} in camera {camera.idx}: "
                f"required first frame for excluded shot {first_shot_idx} is missing"
            )
        _emit_render_progress(progress, "camera_frames_start", f"Generating frames for camera {camera.idx}", {"camera_idx": camera.idx, "active_shot_idxs": camera.active_shot_idxs})

        if os.path.exists(first_shot_ff_path):
            print(f"🚀 Skipped generating first_frame for shot {first_shot_idx}, already exists.")
            self.frame_events[first_shot_idx]["first_frame"].set()
            _emit_render_progress(progress, "frame_exists", f"First frame for shot {first_shot_idx} already exists", {"camera_idx": camera.idx, "shot_idx": first_shot_idx, "frame_type": "first_frame", "path": first_shot_ff_path})

        else:
            print(f"🖼️ Starting first_frame generation for shot {first_shot_idx}...")
            _emit_render_progress(progress, "frame_start", f"Generating first frame for shot {first_shot_idx}", {"camera_idx": camera.idx, "shot_idx": first_shot_idx, "frame_type": "first_frame"})
            available_image_path_and_text_pairs = []

            for visible_character in self._visible_characters(by_idx[first_shot_idx], characters, "first_frame"):
                registry_item = character_portraits_registry[visible_character.identifier_in_scene]
                for view, item in registry_item.items():
                    available_image_path_and_text_pairs.append((item["path"], self._portrait_reference_text(visible_character, item)))
            
            # generate the first_frame based on the shot_description.ff_desc
            if camera.parent_shot_idx is not None:
                parent_shot_idx = camera.parent_shot_idx
                parent_event = self.frame_events[parent_shot_idx]["first_frame"]
                parent_frame_path = self.frame_path(parent_shot_idx, "first_frame")
                if os.path.exists(parent_frame_path):
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

                available_image_path_and_text_pairs.append(
                    (
                        new_camera_image_path,
                        f"The composition and background are correct but some elements may be wrong. The wrong elements should be replaced.\nWrong elements: {camera.missing_info}.\nYou must select this image as the main reference and replace the characters in the image with the provided character portraits. Don't change the background."
                    )
                )

            # 如果子镜头缺少信息，则需要选择参考图像生成
            # A shot being redrawn goes through the same path as one the plan calls
            # incomplete: its first frame is drawn from the camera still *and the character
            # portraits*, with the review note applied. Copying the still instead — which is
            # what a camera with nothing missing does — carries the video model's own
            # invention of the characters straight into the frame, so a redraw of it can
            # never fix a wrong-looking character.
            if camera.parent_shot_idx is None or camera.missing_info is not None or self._is_redrawn(first_shot_idx):
                ff_selector_output_path = self.selector_output_path(first_shot_idx, "first_frame")
                ff_selector_output = self.load_selector_output(first_shot_idx, "first_frame")
                if ff_selector_output is not None:
                    print(f"🚀 Loaded existing reference image selection and prompt for first_frame of shot {first_shot_idx} from {ff_selector_output_path}.")
                    _emit_render_progress(progress, "frame_prompt_exists", f"First frame prompt for shot {first_shot_idx} already exists", {"camera_idx": camera.idx, "shot_idx": first_shot_idx, "frame_type": "first_frame", "path": ff_selector_output_path})
                else:
                    print(f"🔍 Selecting reference images and generating prompt for first_frame of shot {first_shot_idx}...")
                    _emit_render_progress(progress, "frame_prompt_start", f"Selecting references for first frame of shot {first_shot_idx}", {"camera_idx": camera.idx, "shot_idx": first_shot_idx, "frame_type": "first_frame"})
                    ff_selector_output = await self.reference_image_selector.select_reference_images_and_generate_prompt(
                        available_image_path_and_text_pairs=available_image_path_and_text_pairs,
                        frame_description=by_idx[first_shot_idx].ff_desc,
                        max_reference_images=self.reference_image_limit(),
                    )
                    os.makedirs(os.path.dirname(ff_selector_output_path), exist_ok=True)
                    with open(ff_selector_output_path, 'w', encoding='utf-8') as f:
                        json.dump(ff_selector_output, f, ensure_ascii=False, indent=4)

                    print(f"☑️ Selected reference images and generated prompt for first_frame of shot {first_shot_idx}, saved to {ff_selector_output_path}.")
                    _emit_render_progress(progress, "frame_prompt_done", f"Selected references for first frame of shot {first_shot_idx}", {"camera_idx": camera.idx, "shot_idx": first_shot_idx, "frame_type": "first_frame", "path": ff_selector_output_path})

                # The same reference work the other frame path does. A camera's opening shot
                # is a shot like any other, and skipping these here left it with a character's
                # *back* portrait, nobody else's portrait, mentions of images that were not
                # sent, and no record of the prompt — which is how the first frame of a shot
                # could keep coming back wrong however many times it was redrawn.
                reference_image_path_and_text_pairs, prompt = self._align_reference_mentions(
                    available_image_path_and_text_pairs,
                    ff_selector_output["reference_image_path_and_text_pairs"],
                    ff_selector_output.get("selected_indices") or [],
                    ff_selector_output["text_prompt"],
                )
                reference_image_path_and_text_pairs, prompt = self._ensure_character_portraits(
                    self._visible_characters(by_idx[first_shot_idx], characters, "first_frame"),
                    character_portraits_registry,
                    reference_image_path_and_text_pairs,
                    prompt,
                )
                prompt = self._review_note(first_shot_idx, prompt)
                # Record what is sent, so the prompt behind a rendered frame can be read back
                # rather than guessed at, exactly as the other frame path does.
                with open(ff_selector_output_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        **ff_selector_output,
                        "reference_image_path_and_text_pairs": reference_image_path_and_text_pairs,
                        "sent_prompt": prompt,
                    }, f, ensure_ascii=False, indent=4)
                prefix_prompt = ""
                for i, (image_path, text) in enumerate(reference_image_path_and_text_pairs):
                    prefix_prompt += f"Image {i}: {text}\n"
                prompt = f"{prefix_prompt}\n{prompt}"
                reference_image_paths = [item[0] for item in reference_image_path_and_text_pairs]
                ff_image: ImageOutput = await self._generate_frame_image(
                    shot_idx=first_shot_idx,
                    frame_type="first_frame",
                    prompt=prompt,
                    reference_image_paths=reference_image_paths,
                )
                os.makedirs(os.path.dirname(first_shot_ff_path), exist_ok=True)
                ff_image.save(first_shot_ff_path)
                self.frame_events[first_shot_idx]["first_frame"].set()
                print(f"☑️ Generated first_frame for shot {first_shot_idx}, saved to {first_shot_ff_path}.")
                _emit_render_progress(progress, "frame_done", f"Generated first frame for shot {first_shot_idx}", {"camera_idx": camera.idx, "shot_idx": first_shot_idx, "frame_type": "first_frame", "path": first_shot_ff_path})
            else:
                os.makedirs(os.path.dirname(first_shot_ff_path), exist_ok=True)
                shutil.copy(new_camera_image_path, first_shot_ff_path)
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
                first_shot_ff_path_and_text_pair=(first_shot_ff_path, by_idx[first_shot_idx].ff_desc),
                frame_desc=by_idx[first_shot_idx].lf_desc,
                visible_characters=self._visible_characters(by_idx[first_shot_idx], characters, "last_frame"),
                character_portraits_registry=character_portraits_registry,
                progress=progress,
            )
            normal_tasks.append(task)

        for shot_idx in camera.active_shot_idxs[1:]:
            # A shot the request did not name keeps the frames it has: drawing them would be
            # work nobody asked for, and the card's redraw is about one shot.
            if wanted is not None and shot_idx not in wanted:
                continue
            first_frame_task = self.generate_frame_for_single_shot(
                    shot_idx=shot_idx, 
                    frame_type="first_frame", 
                    first_shot_ff_path_and_text_pair=(first_shot_ff_path, by_idx[first_shot_idx].ff_desc),
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
                    first_shot_ff_path_and_text_pair=(self.frame_path(shot_idx, "first_frame"), by_idx[shot_idx].ff_desc),
                    frame_desc=by_idx[shot_idx].lf_desc,
                    visible_characters=self._visible_characters(by_idx[shot_idx], characters, "last_frame"),
                    character_portraits_registry=character_portraits_registry,
                    progress=progress,
                )
                normal_tasks.append(last_frame_task)


        await asyncio.gather(*priority_tasks)
        await asyncio.gather(*normal_tasks)
        _emit_render_progress(progress, "camera_frames_done", f"Frames for camera {camera.idx} ready", {"camera_idx": camera.idx, "active_shot_idxs": camera.active_shot_idxs})



    async def generate_video_for_single_shot(
        self,
        shot_description: ShotDescription,
        progress: Callable[[str, str, Dict[str, Any] | None], None] | None = None,
    ):
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
        first_shot_ff_path_and_text_pair: Tuple[str, str],
        frame_desc: str,
        visible_characters: List[CharacterInScene],
        character_portraits_registry: Dict[str, Dict[str, Dict[str, str]]],
        progress: Callable[[str, str, Dict[str, Any] | None], None] | None = None,
    ) -> ImageOutput:

        frame_image_path = self.frame_path(shot_idx, frame_type)

        if os.path.exists(frame_image_path):
            print(f"🚀 Skipped generating {frame_type} for shot {shot_idx}, already exists.")
            _emit_render_progress(progress, "frame_exists", f"{frame_type} for shot {shot_idx} already exists", {"shot_idx": shot_idx, "frame_type": frame_type, "path": frame_image_path})

        else:
            if frame_type == "last_frame":
                # The end frame is referenced from this shot's start frame, so it
                # has to exist before the reference list is built.
                await self.frame_events[shot_idx]["first_frame"].wait()
            print(f"🖼️ Starting {frame_type} generation for shot {shot_idx}...")
            _emit_render_progress(progress, "frame_start", f"Generating {frame_type} for shot {shot_idx}", {"shot_idx": shot_idx, "frame_type": frame_type})
            available_image_path_and_text_pairs = []
            for visible_character in visible_characters:
                identifier_in_scene = visible_character.identifier_in_scene
                registry_item = character_portraits_registry[identifier_in_scene]
                for view, item in registry_item.items():
                    available_image_path_and_text_pairs.append((item["path"], self._portrait_reference_text(visible_character, item)))

            available_image_path_and_text_pairs.append(first_shot_ff_path_and_text_pair)

            selector_output_path = self.selector_output_path(shot_idx, frame_type)
            selector_output = self.load_selector_output(shot_idx, frame_type)
            if selector_output is not None:
                print(f"🚀 Loaded existing reference image selection and prompt for {frame_type} frame of shot {shot_idx} from {selector_output_path}.")
                _emit_render_progress(progress, "frame_prompt_exists", f"Prompt for {frame_type} of shot {shot_idx} already exists", {"shot_idx": shot_idx, "frame_type": frame_type, "path": selector_output_path})
            else:
                print(f"🔍 Selecting reference images and generating prompt for {frame_type} frame of shot {shot_idx}...")
                _emit_render_progress(progress, "frame_prompt_start", f"Selecting references for {frame_type} of shot {shot_idx}", {"shot_idx": shot_idx, "frame_type": frame_type})
                selector_output = await self.reference_image_selector.select_reference_images_and_generate_prompt(
                    available_image_path_and_text_pairs=available_image_path_and_text_pairs,
                    frame_description=frame_desc,
                    max_reference_images=self.reference_image_limit(),
                )
                os.makedirs(os.path.dirname(selector_output_path), exist_ok=True)
                with open(selector_output_path, 'w', encoding='utf-8') as f:
                    json.dump(selector_output, f, ensure_ascii=False, indent=4)
                print(f"☑️ Selected reference images and generated prompt for {frame_type} frame of shot {shot_idx}, saved to {selector_output_path}.")
                _emit_render_progress(progress, "frame_prompt_done", f"Selected references for {frame_type} of shot {shot_idx}", {"shot_idx": shot_idx, "frame_type": frame_type, "path": selector_output_path})

            if selector_output.get("sent_prompt"):
                # A previous run already resolved this frame's final prompt; reusing it
                # keeps a resumed render from renumbering or re-anchoring it again.
                reference_image_path_and_text_pairs = selector_output["reference_image_path_and_text_pairs"]
                prompt = selector_output["sent_prompt"]
            else:
                reference_image_path_and_text_pairs, prompt = self._align_reference_mentions(
                    available_image_path_and_text_pairs,
                    selector_output["reference_image_path_and_text_pairs"],
                    selector_output.get("selected_indices") or [],
                    selector_output["text_prompt"],
                )
                reference_image_path_and_text_pairs, prompt = self._add_scene_context(shot_idx, reference_image_path_and_text_pairs, prompt)
                reference_image_path_and_text_pairs, prompt = self._ensure_character_portraits(
                    visible_characters, character_portraits_registry, reference_image_path_and_text_pairs, prompt
                )
                prompt = self._review_note(shot_idx, prompt)
                # Record what is sent, so the prompt behind a rendered frame can be
                # read back later (the Artifacts view shows it) rather than guessed.
                selector_output = {
                    **selector_output,
                    "reference_image_path_and_text_pairs": reference_image_path_and_text_pairs,
                    "sent_prompt": prompt,
                }
                with open(selector_output_path, 'w', encoding='utf-8') as f:
                    json.dump(selector_output, f, ensure_ascii=False, indent=4)
            prefix_prompt = ""
            for i, (image_path, text) in enumerate(reference_image_path_and_text_pairs):
                prefix_prompt += f"Image {i}: {text}\n"
            prompt = f"{prefix_prompt}\n{prompt}"
            reference_image_paths = [item[0] for item in reference_image_path_and_text_pairs]

            frame_image: ImageOutput = await self._generate_frame_image(
                shot_idx=shot_idx,
                frame_type=frame_type,
                prompt=prompt,
                reference_image_paths=reference_image_paths,
            )
            os.makedirs(os.path.dirname(frame_image_path), exist_ok=True)
            frame_image.save(frame_image_path)
            print(f"☑️ Generated {frame_type} frame for shot {shot_idx}, saved to {frame_image_path}.")
            _emit_render_progress(progress, "frame_done", f"Generated {frame_type} for shot {shot_idx}", {"shot_idx": shot_idx, "frame_type": frame_type, "path": frame_image_path})


        self.frame_events[shot_idx][frame_type].set()
        return frame_image_path


    def _portrait_reference_text(self, character: CharacterInScene, item: Dict[str, str]) -> str:
        """How a portrait is described to the reference selector.

        Frame descriptions describe characters by their features and never by name — the
        shot descriptions are written that way, and the name is stripped from them — so a
        portrait described only as "A front view portrait of DeepSeek" has nothing in common
        with the frame that says "the younger, fitter man with short black hair and
        rectangular glasses (wearing a t-shirt with a purple whale and carrying a laptop)".
        The selector cannot tell they are the same person, so it leaves the portrait out and
        the image model invents the character from the text. The features are what the two
        descriptions share, so both are given.
        """
        features = "; ".join(part for part in (character.static_features, character.dynamic_features) if part)
        return f"{item['description']} {features}".strip()

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
        limit = self.reference_image_limit()
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

    def _scene_continuity_pair(self, shot_idx: int) -> Optional[Tuple[str, str]]:
        """Keyframe of the most recent earlier shot, to carry the setting forward.

        Frames are rendered camera by camera, so an earlier shot's frame may not
        exist yet; the search walks back until it finds one that does.
        """
        for earlier in range(shot_idx - 1, -1, -1):
            path = self.frame_path(earlier, "first_frame")
            if os.path.exists(path):
                return (path, "The setting of the previous shot: the same room, furniture, lighting and background.")
        return None

    def _add_scene_context(
        self,
        shot_idx: int,
        reference_image_path_and_text_pairs: List[Tuple[str, str]],
        prompt: str,
    ) -> Tuple[List[Tuple[str, str]], str]:
        """Keep a close-up of a character from rendering as a studio portrait.

        The selector picks character portraits when a shot asks for a close-up,
        and the portraits are studio shots on a plain backdrop; with no scene
        reference and a frame description that may not name a setting, the image
        model reproduces the portrait's blank background and the shot lands in the
        film as a white card. An earlier keyframe is therefore always included as
        the setting, appended last so the prompt's existing ``Image N`` references
        keep their positions.
        """
        if any("character_portraits" not in path for path, _ in reference_image_path_and_text_pairs):
            return reference_image_path_and_text_pairs, prompt
        scene_pair = self._scene_continuity_pair(shot_idx)
        if scene_pair is None:
            return reference_image_path_and_text_pairs, prompt + (
                "\nThe portrait references show the character's appearance only: render the setting described above, not their plain studio background."
            )
        reference_image_path_and_text_pairs = [*reference_image_path_and_text_pairs, scene_pair]
        scene_index = len(reference_image_path_and_text_pairs) - 1
        return reference_image_path_and_text_pairs, prompt + (
            f"\nImage {scene_index} shows the setting: render the room, furniture, lighting and background of that image."
            " The other images are character portraits: take the character's appearance from them, never their plain studio background."
        )

    async def _generate_frame_image(
        self,
        *,
        shot_idx: int,
        frame_type: str,
        prompt: str,
        reference_image_paths: List[str],
    ) -> ImageOutput:
        """Generate one frame image, naming the shot if the provider rejects it.

        Provider-side rejections (content filters, unsupported options) do not
        appear in the shot's cached selector output, so without this the failing
        frame is only implied by the last progress event.
        """
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
        character_portraits_registry_path = self.portraits_registry_path()
        if character_portraits_registry is None:
            if os.path.exists(character_portraits_registry_path):
                with open(character_portraits_registry_path, 'r', encoding='utf-8') as f:
                    character_portraits_registry = json.load(f)
            else:
                character_portraits_registry = {}


        tasks = [
            self.generate_portraits_for_single_character(character, style, progress=progress)
            for character in characters
            if character.identifier_in_scene not in character_portraits_registry
        ]
        if tasks:
            for future in asyncio.as_completed(tasks):
                character_portraits_registry.update(await future)
                with open(character_portraits_registry_path, 'w', encoding='utf-8') as f:
                    json.dump(character_portraits_registry, f, ensure_ascii=False, indent=4)

            print(f"✅ Completed character portrait generation for {len(characters)} characters.")
            _emit_render_progress(progress, "character_portraits_done", "Completed character portrait generation", {"character_count": len(characters)})
        else:
            print("🚀 All characters already have portraits, skipping portrait generation.")
            _emit_render_progress(progress, "character_portraits_exist", "All character portraits already exist", {"character_count": len(characters)})
        return character_portraits_registry


    async def generate_portraits_for_single_character(
        self,
        character: CharacterInScene,
        style: str,
        progress: Callable[[str, str, Dict[str, Any] | None], None] | None = None,
    ):
        character_dir = os.path.join(self.portraits_dir(), f"{character.idx}_{safe_path_component(character.identifier_in_scene)}")
        os.makedirs(character_dir, exist_ok=True)
        _emit_render_progress(progress, "character_portrait_start", f"Generating portraits for {character.identifier_in_scene}", {"character_idx": character.idx, "identifier": character.identifier_in_scene})

        front_portrait_path = os.path.join(character_dir, "front.png")
        if os.path.exists(front_portrait_path):
            pass
        else:
            _emit_render_progress(progress, "character_portrait_front_start", f"Generating front portrait for {character.identifier_in_scene}", {"character_idx": character.idx, "identifier": character.identifier_in_scene})
            front_portrait_output = await self.character_portraits_generator.generate_front_portrait(character, style)
            front_portrait_output.save(front_portrait_path)
            _emit_render_progress(progress, "character_portrait_front_done", f"Generated front portrait for {character.identifier_in_scene}", {"character_idx": character.idx, "identifier": character.identifier_in_scene, "path": front_portrait_path})


        side_portrait_path = os.path.join(character_dir, "side.png")
        if os.path.exists(side_portrait_path):
            pass
        else:
            _emit_render_progress(progress, "character_portrait_side_start", f"Generating side portrait for {character.identifier_in_scene}", {"character_idx": character.idx, "identifier": character.identifier_in_scene})
            side_portrait_output = await self.character_portraits_generator.generate_side_portrait(character, front_portrait_path)
            side_portrait_output.save(side_portrait_path)
            _emit_render_progress(progress, "character_portrait_side_done", f"Generated side portrait for {character.identifier_in_scene}", {"character_idx": character.idx, "identifier": character.identifier_in_scene, "path": side_portrait_path})

        back_portrait_path = os.path.join(character_dir, "back.png")
        if os.path.exists(back_portrait_path):
            pass
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

        self.character_portrait_events[character.idx].set()

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
