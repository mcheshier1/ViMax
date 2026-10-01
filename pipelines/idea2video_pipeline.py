import os
import shutil
import logging
from agents import Screenwriter, CharacterExtractor, CharacterPortraitsGenerator
from pipelines.script2video_pipeline import Script2VideoPipeline
from interfaces import CharacterInScene
from typing import Any, Callable, Dict, List, Optional
import asyncio
import json
import yaml
from langchain.chat_models import init_chat_model
from tools.render_backend import RenderBackend
from utils.provider_presets import resolve_chat_model_config
from utils.text import safe_path_component
from utils.video import concatenate_video_files

from .render_contract import DEFAULT_RENDER_PHASE, ModelScopedArtifacts, RenderOutcome, normalize_phase


def _pipeline_print(quiet: bool, message: str) -> None:
    if not quiet:
        print(message)

def _file_signature(path: str) -> Dict[str, int]:
    stat = os.stat(path)
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}



class Idea2VideoPipeline(ModelScopedArtifacts):
    def __init__(
        self,
        chat_model: str,
        image_generator: str,
        video_generator: str,
        working_dir: str,
    ):
        self.chat_model = chat_model
        self.image_generator = image_generator
        self.video_generator = video_generator
        self.working_dir = working_dir
        os.makedirs(self.working_dir, exist_ok=True)

        self.screenwriter = Screenwriter(chat_model=self.chat_model)
        self.character_extractor = CharacterExtractor(
            chat_model=self.chat_model)
        self.character_portraits_generator = CharacterPortraitsGenerator(
            image_generator=self.image_generator)

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

    async def extract_characters(
        self,
        story: str,
        quiet: bool = False,
    ):
        save_path = os.path.join(self.working_dir, "characters.json")

        if os.path.exists(save_path):
            with open(save_path, "r", encoding="utf-8") as f:
                characters = json.load(f)
            characters = [CharacterInScene.model_validate(
                character) for character in characters]
            _pipeline_print(quiet, f"🚀 Loaded {len(characters)} characters from existing file.")
        else:
            characters = await self.character_extractor.extract_characters(story)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump([character.model_dump()
                          for character in characters], f, ensure_ascii=False, indent=4)
            _pipeline_print(quiet, f"✅ Extracted {len(characters)} characters from story and saved to {save_path}.")

        return characters

    async def generate_character_portraits(
        self,
        characters: List[CharacterInScene],
        character_portraits_registry: Optional[Dict[str, Dict[str, Dict[str, str]]]],
        style: str,
    ):
        character_portraits_registry_path = self.portraits_registry_path()
        if character_portraits_registry is None:
            if os.path.exists(character_portraits_registry_path):
                with open(character_portraits_registry_path, 'r', encoding='utf-8') as f:
                    character_portraits_registry = json.load(f)
            else:
                character_portraits_registry = {}

        tasks = [
            self.generate_portraits_for_single_character(character, style)
            for character in characters
            if character.identifier_in_scene not in character_portraits_registry
            # Characters never shown on screen (e.g. a voice or chat-only
            # character) have no physical description, so asking the image
            # model for front/side/back portraits of them is nonsensical and
            # fails repeatedly (finish_reason=IMAGE_OTHER, empty candidates).
            and character.is_visible
        ]
        if tasks:
            for future in asyncio.as_completed(tasks):
                character_portraits_registry.update(await future)
                with open(character_portraits_registry_path, 'w', encoding='utf-8') as f:
                    json.dump(character_portraits_registry,
                              f, ensure_ascii=False, indent=4)

            print(
                f"✅ Completed character portrait generation for {len(characters)} characters.")
        else:
            print(
                "🚀 All characters already have portraits, skipping portrait generation.")

        return character_portraits_registry

    async def develop_story(
        self,
        idea: str,
        user_requirement: str,
        quiet: bool = False,
    ):
        save_path = os.path.join(self.working_dir, "story.txt")
        if os.path.exists(save_path):
            with open(save_path, "r", encoding="utf-8") as f:
                story = f.read()
            _pipeline_print(quiet, f"🚀 Loaded story from existing file.")
        else:
            _pipeline_print(quiet, "🧠 Developing story...")
            story = await self.screenwriter.develop_story(idea=idea, user_requirement=user_requirement)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(story)
            _pipeline_print(quiet, f"✅ Developed story and saved to {save_path}.")

        return story

    async def write_script_based_on_story(
        self,
        story: str,
        user_requirement: str,
        quiet: bool = False,
    ):
        save_path = os.path.join(self.working_dir, "script.json")
        if os.path.exists(save_path):
            with open(save_path, "r", encoding="utf-8") as f:
                script = json.load(f)
            _pipeline_print(quiet, f"🚀 Loaded script from existing file.")
        else:
            _pipeline_print(quiet, "🧠 Writing script based on story...")
            script = await self.screenwriter.write_script_based_on_story(story=story, user_requirement=user_requirement)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(script, f, ensure_ascii=False, indent=4)
            _pipeline_print(quiet, f"✅ Written script based on story and saved to {save_path}.")
        return script

    async def generate_portraits_for_single_character(
        self,
        character: CharacterInScene,
        style: str,
    ):
        character_dir = os.path.join(
            self.portraits_dir(), f"{character.idx}_{safe_path_component(character.identifier_in_scene)}")
        os.makedirs(character_dir, exist_ok=True)

        front_portrait_path = os.path.join(character_dir, "front.png")
        if os.path.exists(front_portrait_path):
            pass
        else:
            front_portrait_output = await self.character_portraits_generator.generate_front_portrait(character, style)
            front_portrait_output.save(front_portrait_path)

        side_portrait_path = os.path.join(character_dir, "side.png")
        if os.path.exists(side_portrait_path):
            pass
        else:
            try:
                side_portrait_output = await self.character_portraits_generator.generate_side_portrait(character, front_portrait_path)
                side_portrait_output.save(side_portrait_path)
            except Exception as e:
                # gemini-2.5-flash-image intermittently (sometimes beyond
                # the tenacity retry budget) fails this front->side
                # re-angling edit with finish_reason=IMAGE_OTHER / empty
                # content. Fall back to the front portrait rather than
                # aborting the whole pipeline.
                print(f"⚠️ Side portrait generation failed for {character.identifier_in_scene} after retries ({e}); reusing front portrait as fallback.")
                shutil.copy(front_portrait_path, side_portrait_path)

        back_portrait_path = os.path.join(character_dir, "back.png")
        if os.path.exists(back_portrait_path):
            pass
        else:
            try:
                back_portrait_output = await self.character_portraits_generator.generate_back_portrait(character, front_portrait_path)
                back_portrait_output.save(back_portrait_path)
            except Exception as e:
                print(f"⚠️ Back portrait generation failed for {character.identifier_in_scene} after retries ({e}); reusing front portrait as fallback.")
                shutil.copy(front_portrait_path, back_portrait_path)

        print(
            f"☑️ Completed character portrait generation for {character.identifier_in_scene}.")

        recorded_prompts = {
            view: self.character_portraits_generator.prompts.get((character.identifier_in_scene, view), "")
            for view in ("front", "side", "back")
        }
        with open(os.path.join(character_dir, "prompts.json"), "w", encoding="utf-8") as f:
            json.dump({view: prompt for view, prompt in recorded_prompts.items() if prompt}, f, ensure_ascii=False, indent=4)

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

    async def __call__(
        self,
        idea: str,
        user_requirement: str,
        style: str,
        quiet: bool = False,
        stop_after: str = DEFAULT_RENDER_PHASE,
        revision_notes: Dict[str, str] | None = None,
        progress: Callable[[str, str, Dict[str, Any] | None], None] | None = None,
        only_shots: Optional[List[str]] = None,
    ) -> RenderOutcome:
        stop_after = normalize_phase(stop_after)
        self.revision_notes = {str(shot): note for shot, note in (revision_notes or {}).items() if str(note).strip()}

        selected_by_scene: Dict[int, List[int]] | None = None
        if only_shots is not None:
            selected_by_scene = {}
            for qualified in only_shots:
                scene, separator, local_id = str(qualified).partition("/")
                if not separator or not scene.startswith("scene_"):
                    raise ValueError(f"Invalid qualified shot identity: {qualified!r}")
                try:
                    scene_idx = int(scene[6:])
                    shot_idx = int(local_id)
                except ValueError as error:
                    raise ValueError(f"Invalid qualified shot identity: {qualified!r}") from error
                selected_by_scene.setdefault(scene_idx, []).append(shot_idx)

        story = await self.develop_story(idea=idea, user_requirement=user_requirement, quiet=quiet)
        characters = await self.extract_characters(story=story, quiet=quiet)
        character_portraits_registry = await self.generate_character_portraits(
            characters=characters,
            character_portraits_registry=None,
            style=style,
        )

        if stop_after == "portraits":
            stills = self.portrait_stills(character_portraits_registry)
            _pipeline_print(quiet, f"🖼️ Character portraits ready for style review ({len(stills)} images).")
            return self.render_outcome("portraits", style, stills=stills, awaiting="stills")

        scene_scripts = await self.write_script_based_on_story(story=story, user_requirement=user_requirement, quiet=quiet)
        all_video_paths: List[str] = []
        stills: List[str] = []
        complete_scenes = 0
        video_budget: Dict[str, str] = {}

        for idx, scene_script in enumerate(scene_scripts):
            scene_working_dir = os.path.join(self.working_dir, f"scene_{idx}")
            scene_film = os.path.join(scene_working_dir, "final_video.mp4")
            scene_scope = None if selected_by_scene is None else selected_by_scene.get(idx)
            selected = selected_by_scene is None or scene_scope is not None

            if stop_after == "video" and not selected:
                if os.path.isfile(scene_film):
                    all_video_paths.append(scene_film)
                    complete_scenes += 1
                    _pipeline_print(quiet, f"🚀 Reusing completed film for unselected scene {idx}.")
                else:
                    _emit_scene_progress(progress, "scene_deferred", f"Scene {idx} was not selected and has no completed film", idx)
                    all_video_paths.append("")
                continue

            if not selected:
                continue

            os.makedirs(scene_working_dir, exist_ok=True)
            script2video_pipeline = Script2VideoPipeline(
                chat_model=self.chat_model,
                image_generator=self.image_generator,
                video_generator=self.video_generator,
                working_dir=scene_working_dir,
            )
            outcome = await script2video_pipeline(
                script=scene_script,
                user_requirement=user_requirement,
                style=style,
                characters=characters,
                character_portraits_registry=character_portraits_registry,
                quiet=quiet,
                stop_after=stop_after,
                progress=progress,
                only_shots=scene_scope,
                video_budget=video_budget,
                revision_notes={
                    slot.split("/", 1)[1]: note
                    for slot, note in (self.revision_notes or {}).items()
                    if slot.startswith(f"scene_{idx}/")
                },
            )
            stills.extend(outcome.stills)
            if stop_after == "stills":
                continue
            if outcome.final_video_path and os.path.isfile(outcome.final_video_path):
                all_video_paths.append(outcome.final_video_path)
                complete_scenes += 1
            else:
                all_video_paths.append("")
                _emit_scene_progress(
                    progress,
                    "scene_partial",
                    f"Scene {idx} has generated progress but no completed film yet",
                    idx,
                )

        if stop_after == "stills":
            _pipeline_print(quiet, f"🖼️ Keyframes ready for review across {len(scene_scripts)} scene(s) ({len(stills)} images).")
            return self.render_outcome("stills", style, stills=stills, awaiting="video")

        final_video_path = os.path.join(self.working_dir, "final_video.mp4")
        assembly_metadata_path = final_video_path + ".inputs.json"
        if (
            not scene_scripts
            or complete_scenes != len(scene_scripts)
            or len(all_video_paths) != len(scene_scripts)
            or not all(all_video_paths)
        ):
            for path in (final_video_path, assembly_metadata_path):
                if os.path.isfile(path):
                    os.remove(path)
            _emit_scene_progress(
                progress,
                "render_partial",
                f"{complete_scenes} of {len(scene_scripts)} scene films are complete; continuing without concatenation",
                None,
            )
            return self.render_outcome("video", style, final_video_path="")

        assembly_inputs = {
            "version": 1,
            "scene_films": [
                {
                    "scene_idx": idx,
                    "path": os.path.relpath(path, self.working_dir),
                    **_file_signature(path),
                }
                for idx, path in enumerate(all_video_paths)
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
            _pipeline_print(quiet, f"🚀 Reusing final film; ordered scene inputs are unchanged.")
            return self.render_outcome("video", style, final_video_path=final_video_path)

        _pipeline_print(quiet, f"🎬 Starting concatenating videos...")
        concatenate_video_files(all_video_paths, final_video_path)
        with open(assembly_metadata_path, "w", encoding="utf-8") as metadata_file:
            json.dump(assembly_inputs, metadata_file, ensure_ascii=False, indent=2)
        _pipeline_print(quiet, f"☑️ Concatenated videos, saved to {final_video_path}.")
        return self.render_outcome("video", style, final_video_path=final_video_path)


def _emit_scene_progress(progress, stage: str, message: str, scene_idx: int | None) -> None:
    if progress is not None:
        metadata = {} if scene_idx is None else {"scene_idx": scene_idx}
        progress(stage, message, metadata)
