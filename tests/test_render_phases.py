"""Render phase gates, single-model pinning, and model-scoped artifact layout."""

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from PIL import Image

from agent_runtime.session_index import SessionIndex
from agent_runtime.vimax_adapters import (
    RENDER_MANIFEST_FILENAME,
    ViMaxAdapters,
    _enforce_render_sequence,
    _read_render_manifest,
    _supersede_clips,
)
from interfaces import Camera, CharacterInScene, ImageOutput, ShotDescription, VideoOutput
from pipelines.idea2video_pipeline import Idea2VideoPipeline
from pipelines.render_contract import RenderOutcome, normalize_phase
from pipelines.script2video_pipeline import Script2VideoPipeline


IMAGE_MODEL = "test/image-model"
VIDEO_MODEL = "test/video-model"


class _Generator:
    """Records calls and returns a usable artifact, standing in for a provider."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.calls: list[dict] = []

    async def generate_single_image(self, **kwargs) -> ImageOutput:
        self.calls.append(kwargs)
        return ImageOutput(fmt="pil", ext="png", data=Image.new("RGB", (16, 9), "blue"))

    async def generate_single_video(self, **kwargs) -> VideoOutput:
        self.calls.append(kwargs)
        return VideoOutput(fmt="bytes", ext="mp4", data=b"video")


def _character() -> CharacterInScene:
    return CharacterInScene(
        idx=0,
        identifier_in_scene="Claude",
        is_visible=True,
        static_features="a heavyset man",
        dynamic_features="an orange shirt",
    )


def _shot(idx: int, variation_type: str = "small") -> ShotDescription:
    return ShotDescription(
        idx=idx,
        is_last=True,
        cam_idx=0,
        visual_desc=f"shot {idx}",
        variation_type=variation_type,
        variation_reason="r",
        ff_desc=f"first frame {idx}",
        ff_vis_char_idxs=[],
        lf_desc=f"last frame {idx}",
        lf_vis_char_idxs=[],
        motion_desc="m",
        audio_desc="a",
    )


def _pipeline(working_dir: str, *, shots=None) -> tuple[Script2VideoPipeline, _Generator, _Generator]:
    """A pipeline with planning stubbed out, so tests exercise only the render phases."""
    shots = shots or [_shot(0)]
    image_generator = _Generator(IMAGE_MODEL)
    video_generator = _Generator(VIDEO_MODEL)
    pipeline = Script2VideoPipeline(
        chat_model=MagicMock(),
        image_generator=image_generator,
        video_generator=video_generator,
        working_dir=working_dir,
    )
    pipeline.design_storyboard = AsyncMock(return_value=[MagicMock(idx=shot.idx) for shot in shots])
    pipeline.decompose_visual_descriptions = AsyncMock(return_value=shots)
    pipeline.construct_camera_tree = AsyncMock(return_value=[Camera(idx=0, active_shot_idxs=[shot.idx for shot in shots])])
    pipeline.reference_image_selector = MagicMock(
        select_reference_images_and_generate_prompt=AsyncMock(
            return_value={"reference_image_path_and_text_pairs": [], "text_prompt": "a prompt"}
        )
    )
    # decompose_visual_descriptions owns this wiring in production; mirror it so the
    # stubbed planning step leaves the same state behind. Both keyframes are tracked
    # for every shot, even though an end frame is only generated for models that
    # accept one.
    for shot in shots:
        pipeline.shot_desc_events[shot.idx] = asyncio.Event()
        pipeline.frame_events[shot.idx] = {"first_frame": asyncio.Event(), "last_frame": asyncio.Event()}
    return pipeline, image_generator, video_generator

def _idea_pipeline(working_dir: str, scene_scripts: list[str]) -> Idea2VideoPipeline:
    """An Idea pipeline with all planning and portraits local to the test."""
    pipeline = Idea2VideoPipeline.__new__(Idea2VideoPipeline)
    pipeline.working_dir = working_dir
    pipeline.chat_model = MagicMock()
    pipeline.image_generator = _Generator(IMAGE_MODEL)
    pipeline.video_generator = _Generator(VIDEO_MODEL)
    pipeline.develop_story = AsyncMock(return_value="story")
    pipeline.extract_characters = AsyncMock(return_value=[])
    pipeline.generate_character_portraits = AsyncMock(return_value={})
    pipeline.write_script_based_on_story = AsyncMock(return_value=scene_scripts)
    return pipeline


class PhaseNormalizationTests(unittest.TestCase):
    def test_unknown_phase_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            normalize_phase("clips")
        self.assertIn("stop_after must be one of", str(caught.exception))

    def test_default_phase_is_portraits(self):
        self.assertEqual(normalize_phase(""), "portraits")


class ModelScopedLayoutTests(unittest.TestCase):
    def test_artifacts_live_under_the_model_that_produced_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline, _, _ = _pipeline(tmp)
            self.assertEqual(pipeline.image_model_slug, "test_image-model")
            self.assertEqual(pipeline.video_model_slug, "test_video-model")
            self.assertEqual(os.path.relpath(pipeline.portraits_dir(), tmp), os.path.join("character_portraits", "test_image-model"))
            self.assertEqual(os.path.relpath(pipeline.portraits_registry_path(), tmp), os.path.join("character_portraits", "test_image-model", "registry.json"))
            self.assertEqual(os.path.relpath(pipeline.frame_path(3, "first_frame"), tmp), os.path.join("shots", "3", "test_image-model", "first_frame.png"))
            self.assertEqual(os.path.relpath(pipeline.clip_path(3), tmp), os.path.join("shots", "3", "test_video-model", "video.mp4"))
            # Prompts are inputs, not model output.
            self.assertEqual(os.path.relpath(pipeline.selector_output_path(3, "first_frame"), tmp), os.path.join("shots", "3", "first_frame_selector_output.json"))


class RenderPhaseGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_portraits_phase_stops_before_any_keyframe_or_clip(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline, image_generator, video_generator = _pipeline(tmp)
            outcome = await pipeline(
                script="script",
                user_requirement="req",
                style="cinematic",
                characters=[_character()],
                stop_after="portraits",
            )

            self.assertEqual(outcome.phase, "portraits")
            self.assertEqual(outcome.awaiting_confirmation, "stills")
            self.assertEqual(outcome.style, "cinematic")
            self.assertEqual(len(outcome.stills), 3)  # front, side, back
            self.assertTrue(all(path.startswith(os.path.join(tmp, "character_portraits", "test_image-model")) for path in outcome.stills))
            self.assertEqual(video_generator.calls, [])
            self.assertFalse(os.path.exists(os.path.join(tmp, "shots", "0", "test_image-model", "first_frame.png")))
            # The portraits phase must not even load the storyboard.
            pipeline.design_storyboard.assert_not_awaited()

    async def test_stills_phase_stops_before_video_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline, _, video_generator = _pipeline(tmp)
            outcome = await pipeline(
                script="script",
                user_requirement="req",
                style="cinematic",
                characters=[_character()],
                stop_after="stills",
            )

            self.assertEqual(outcome.phase, "stills")
            self.assertEqual(outcome.awaiting_confirmation, "video")
            # Both keyframes of every shot are shown, because both bracket the clip.
            expected = [
                os.path.join(tmp, "shots", "0", "test_image-model", "first_frame.png"),
                os.path.join(tmp, "shots", "0", "test_image-model", "last_frame.png"),
            ]
            self.assertEqual(outcome.stills, expected)
            self.assertTrue(all(os.path.exists(path) for path in expected))
            self.assertEqual(video_generator.calls, [], "video must not be generated before the user approves the stills")

    async def test_stills_phase_reports_last_frames_for_interpolated_shots(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline, _, _ = _pipeline(tmp, shots=[_shot(0, variation_type="large")])
            outcome = await pipeline(
                script="script",
                user_requirement="req",
                style="cinematic",
                characters=[_character()],
                stop_after="stills",
            )

            self.assertEqual(
                [os.path.basename(path) for path in outcome.stills],
                ["first_frame.png", "last_frame.png"],
            )

    async def test_video_phase_generates_clips(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Rebuilding the clip invalidates the old film; keep this focused test off MoviePy's encoder.
            Path(tmp, "final_video.mp4").write_bytes(b"stale film")
            pipeline, _, video_generator = _pipeline(tmp)
            with patch("pipelines.script2video_pipeline.VideoFileClip", return_value=MagicMock()), \
                 patch("pipelines.script2video_pipeline.concatenate_videoclips", return_value=MagicMock()):
                outcome = await pipeline(
                    script="script",
                    user_requirement="req",
                    style="cinematic",
                    characters=[_character()],
                    stop_after="video",
                )

            self.assertEqual(outcome.phase, "video")
            self.assertEqual(outcome.awaiting_confirmation, "")
            self.assertEqual(len(video_generator.calls), 1)
            self.assertTrue(outcome.final_video_path.endswith("final_video.mp4"))


    async def test_one_video_is_drawn_per_render_however_many_are_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            # A stale final film must not mask clips still pending this render.
            Path(tmp, "final_video.mp4").write_bytes(b"stale film")
            pipeline, _, video_generator = _pipeline(tmp, shots=[_shot(0), _shot(1), _shot(2)])

            outcome = await pipeline(
                script="script",
                user_requirement="req",
                style="cinematic",
                characters=[_character()],
                stop_after="video",
            )

            # Nine clips were bought in a single run of the live sequence, so the rule is a
            # hard one: the first clip the film is missing is drawn and the rest wait.
            self.assertEqual(len(video_generator.calls), 1)
            self.assertTrue(os.path.exists(pipeline.clip_path(0)))
            self.assertFalse(os.path.exists(pipeline.clip_path(1)))
            self.assertFalse(os.path.exists(pipeline.clip_path(2)))
            self.assertEqual(outcome.final_video_path, "")
            self.assertFalse(Path(tmp, "final_video.mp4").exists())


    async def test_deferred_child_frame_does_not_leave_its_clip_waiting(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline, _, video_generator = _pipeline(tmp, shots=[_shot(0), _shot(1)])
            pipeline.construct_camera_tree = AsyncMock(return_value=[
                Camera(idx=0, active_shot_idxs=[0]),
                Camera(idx=1, active_shot_idxs=[1], parent_shot_idx=0),
            ])
            progress_events = []

            outcome = await asyncio.wait_for(
                pipeline(
                    script="script",
                    user_requirement="req",
                    style="cinematic",
                    characters=[_character()],
                    stop_after="video",
                    video_budget={"claimed_by": "an earlier transition"},
                    progress=lambda stage, message, metadata=None: progress_events.append(stage),
                ),
                timeout=1,
            )

            self.assertEqual(outcome.final_video_path, "")
            self.assertIn("transition_video_deferred", progress_events)
            self.assertEqual(video_generator.calls, [])

    async def test_a_stills_render_draws_at_most_one_transition_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline, _, video_generator = _pipeline(tmp, shots=[_shot(0), _shot(1), _shot(2)])
            pipeline.construct_camera_tree = AsyncMock(return_value=[
                Camera(idx=0, active_shot_idxs=[0]),
                Camera(idx=1, active_shot_idxs=[1], parent_shot_idx=0),
                Camera(idx=2, active_shot_idxs=[2], parent_shot_idx=0),
            ])
            # The new camera image is taken from the transition video; pre-made here so the
            # stub's bytes never reach moviepy's decoder.
            for camera_idx, shot_idx in ((1, 1), (2, 2)):
                os.makedirs(pipeline.shot_video_dir(shot_idx), exist_ok=True)
                Image.new("RGB", (16, 9), "red").save(
                    os.path.join(pipeline.shot_video_dir(shot_idx), f"new_camera_{camera_idx}.png")
                )

            await pipeline(
                script="script",
                user_requirement="req",
                style="cinematic",
                characters=[_character()],
                stop_after="stills",
            )

            # Two cameras parented to shot 0 means two transitions are missing, and a stills
            # render used to bill both of them without a word about it.
            self.assertEqual(len(video_generator.calls), 1)


    def _prime_cached_shots(self, pipeline, shot_ids):
        for shot_idx in shot_ids:
            for path in (
                pipeline.frame_path(shot_idx, "first_frame"),
                pipeline.frame_path(shot_idx, "last_frame"),
                pipeline.clip_path(shot_idx),
            ):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                Path(path).write_bytes(b"cached")

    async def test_assembly_uses_camera_order_for_a_middle_inserted_shot(self):
        with tempfile.TemporaryDirectory() as tmp:
            shots = [_shot(0), _shot(1), _shot(2)]
            pipeline, _, _ = _pipeline(tmp, shots=shots)
            pipeline.construct_camera_tree = AsyncMock(return_value=[
                Camera(idx=0, active_shot_idxs=[0, 2, 1]),
            ])
            self._prime_cached_shots(pipeline, [0, 1, 2])
            clip_order = []
            writes = []

            def open_clip(path):
                clip_order.append(int(Path(path).parent.parent.name))
                return path

            class JoinedFilm:
                def write_videofile(self, path, **kwargs):
                    writes.append(path)
                    Path(path).write_bytes(b"assembled")

            with patch("pipelines.script2video_pipeline.VideoFileClip", side_effect=open_clip), \
                 patch("pipelines.script2video_pipeline.concatenate_videoclips", return_value=JoinedFilm()):
                outcome = await pipeline(
                    script="script",
                    user_requirement="req",
                    style="cinematic",
                    characters=[_character()],
                    character_portraits_registry={},
                    stop_after="video",
                )
                film_mtime = Path(outcome.final_video_path).stat().st_mtime_ns
                repeat = await pipeline(
                    script="script",
                    user_requirement="req",
                    style="cinematic",
                    characters=[_character()],
                    character_portraits_registry={},
                    stop_after="video",
                )

            self.assertEqual(clip_order, [0, 2, 1])
            self.assertEqual(writes, [outcome.final_video_path])
            self.assertEqual(repeat.final_video_path, outcome.final_video_path)
            self.assertEqual(Path(repeat.final_video_path).read_bytes(), b"assembled")
            self.assertEqual(Path(repeat.final_video_path).stat().st_mtime_ns, film_mtime)

    async def test_changed_camera_order_reassembles_cached_clips(self):
        with tempfile.TemporaryDirectory() as tmp:
            shots = [_shot(0), _shot(1), _shot(2)]
            pipeline, _, _ = _pipeline(tmp, shots=shots)
            active_order = [0, 2, 1]

            async def camera_tree(**kwargs):
                return [Camera(idx=0, active_shot_idxs=list(active_order))]

            pipeline.construct_camera_tree = camera_tree
            self._prime_cached_shots(pipeline, [0, 1, 2])
            clip_order = []
            writes = []

            def open_clip(path):
                clip_order.append(int(Path(path).parent.parent.name))
                return path

            class JoinedFilm:
                def write_videofile(self, path, **kwargs):
                    writes.append(path)
                    Path(path).write_bytes(f"assembly {len(writes)}".encode())

            with patch("pipelines.script2video_pipeline.VideoFileClip", side_effect=open_clip), \
                 patch("pipelines.script2video_pipeline.concatenate_videoclips", return_value=JoinedFilm()):
                first = await pipeline(
                    script="script", user_requirement="req", style="cinematic",
                    characters=[_character()], character_portraits_registry={}, stop_after="video",
                )
                active_order[:] = [0, 1, 2]
                second = await pipeline(
                    script="script", user_requirement="req", style="cinematic",
                    characters=[_character()], character_portraits_registry={}, stop_after="video",
                )

            self.assertEqual(clip_order, [0, 2, 1, 0, 1, 2])
            self.assertEqual(writes, [first.final_video_path, second.final_video_path])
            self.assertEqual(Path(second.final_video_path).read_bytes(), b"assembly 2")

    async def test_removed_active_shot_rebuilds_cached_film(self):
        with tempfile.TemporaryDirectory() as tmp:
            shots = [_shot(0), _shot(1), _shot(2)]
            pipeline, _, _ = _pipeline(tmp, shots=shots)
            active_order = [0, 1, 2]

            async def camera_tree(**kwargs):
                return [Camera(idx=0, active_shot_idxs=list(active_order))]

            pipeline.construct_camera_tree = camera_tree
            self._prime_cached_shots(pipeline, [0, 1, 2])
            clip_order = []
            writes = []

            def open_clip(path):
                clip_order.append(int(Path(path).parent.parent.name))
                return path

            class JoinedFilm:
                def write_videofile(self, path, **kwargs):
                    writes.append(path)
                    Path(path).write_bytes(f"assembly {len(writes)}".encode())

            with patch("pipelines.script2video_pipeline.VideoFileClip", side_effect=open_clip), \
                 patch("pipelines.script2video_pipeline.concatenate_videoclips", return_value=JoinedFilm()):
                first = await pipeline(
                    script="script", user_requirement="req", style="cinematic",
                    characters=[_character()], character_portraits_registry={}, stop_after="video",
                )
                active_order[:] = [0, 2]
                second = await pipeline(
                    script="script", user_requirement="req", style="cinematic",
                    characters=[_character()], character_portraits_registry={}, stop_after="video",
                )

            self.assertEqual(clip_order, [0, 1, 2, 0, 2])
            self.assertEqual(writes, [first.final_video_path, second.final_video_path])
            self.assertEqual(Path(second.final_video_path).read_bytes(), b"assembly 2")

    async def test_empty_active_sequence_does_not_assemble_a_film(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline, _, _ = _pipeline(tmp)
            pipeline.construct_camera_tree = AsyncMock(return_value=[Camera(idx=0, active_shot_idxs=[])])
            stale_film = Path(tmp) / "final_video.mp4"
            stale_film.write_bytes(b"old film")

            with patch("pipelines.script2video_pipeline.concatenate_videoclips") as concatenate:
                outcome = await pipeline(
                    script="script",
                    user_requirement="req",
                    style="cinematic",
                    characters=[_character()],
                    stop_after="video",
                )

            self.assertEqual(outcome.final_video_path, "")
            self.assertFalse(stale_film.exists())
            concatenate.assert_not_called()

    async def test_sparse_removal_reassembles_without_the_removed_shot(self):
        with tempfile.TemporaryDirectory() as tmp:
            shots = [_shot(0), _shot(1), _shot(2)]
            pipeline, _, _ = _pipeline(tmp, shots=shots)
            pipeline.construct_camera_tree = AsyncMock(return_value=[
                Camera(idx=0, active_shot_idxs=[0, 2]),
            ])
            self._prime_cached_shots(pipeline, [0, 1, 2])
            film_path = Path(tmp) / "final_video.mp4"
            film_path.write_bytes(b"film containing removed shot 1")
            clip_order = []

            def open_clip(path):
                clip_order.append(int(Path(path).parent.parent.name))
                return path

            class JoinedFilm:
                def write_videofile(self, path, **kwargs):
                    Path(path).write_bytes(b"film without removed shot 1")

            with patch("pipelines.script2video_pipeline.VideoFileClip", side_effect=open_clip), \
                 patch("pipelines.script2video_pipeline.concatenate_videoclips", return_value=JoinedFilm()):
                outcome = await pipeline(
                    script="script",
                    user_requirement="req",
                    style="cinematic",
                    characters=[_character()],
                    character_portraits_registry={},
                    stop_after="video",
                )

            self.assertEqual(clip_order, [0, 2])
            self.assertEqual(Path(outcome.final_video_path).read_bytes(), b"film without removed shot 1")

class ClipLengthTests(unittest.TestCase):
    """A clip of the wrong length is not the film's clip."""

    def _root_with(self, tmp):
        root = Path(tmp) / "script2video"
        clip = root / "shots/2/test_video-model/video.mp4"
        clip.parent.mkdir(parents=True, exist_ok=True)
        clip.write_bytes(b"old clip")
        (root / "final_video.mp4").write_bytes(b"film")
        return root, clip

    def test_wrong_length_clips_and_the_film_move_aside(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, clip = self._root_with(tmp)

            with patch("agent_runtime.vimax_adapters._clip_seconds_of", return_value=5.04):
                moved = _supersede_clips(root, 8)

            self.assertEqual(len(moved), 2)
            self.assertFalse(clip.exists())
            self.assertFalse((root / "final_video.mp4").exists())
            # Kept, under the length they were rendered at, where no slot reads them.
            self.assertTrue((root / ".superseded_clips/5s/shots/2/test_video-model/video.mp4").exists())

    def test_a_clip_at_the_configured_length_stays(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, clip = self._root_with(tmp)

            with patch("agent_runtime.vimax_adapters._clip_seconds_of", return_value=8.04):
                moved = _supersede_clips(root, 8)

            self.assertEqual(moved, ["final_video.mp4"])
            self.assertTrue(clip.exists())

    def test_the_sequence_records_the_length_and_sheds_the_old_clips(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, clip = self._root_with(tmp)
            manifest = Path(tmp) / RENDER_MANIFEST_FILENAME
            manifest.write_text(json.dumps({
                "image_model": IMAGE_MODEL,
                "video_model": VIDEO_MODEL,
                "style": "cinematic",
                "render_mode": "script2video",
            }), encoding="utf-8")
            generator = _Generator(VIDEO_MODEL)
            generator.clip_seconds = 8

            with patch("agent_runtime.vimax_adapters._clip_seconds_of", return_value=5.04):
                refusal = _enforce_render_sequence(
                    Path(tmp),
                    image_generator=_Generator(IMAGE_MODEL),
                    video_generator=generator,
                    style="cinematic",
                    allow_model_change=False,
                    render_mode="script2video",
                )

            self.assertIsNone(refusal)
            self.assertFalse(clip.exists())
            self.assertEqual(json.loads(manifest.read_text(encoding="utf-8"))["clip_seconds"], 8)


class ClipBracketingTests(unittest.IsolatedAsyncioTestCase):
    """A clip is bracketed by both keyframes only where the model accepts an end frame."""

    async def test_no_end_keyframe_is_generated_for_a_model_that_rejects_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "final_video.mp4").write_bytes(b"stale film")
            pipeline, _, video_generator = _pipeline(tmp)

            async def supports_last_frame():
                return False

            video_generator.supports_last_frame = supports_last_frame
            with patch("pipelines.script2video_pipeline.VideoFileClip", return_value=MagicMock()), \
                 patch("pipelines.script2video_pipeline.concatenate_videoclips", return_value=MagicMock()):
                await pipeline(
                    script="script",
                    user_requirement="req",
                    style="cinematic",
                    characters=[_character()],
                    stop_after="video",
                )

            self.assertFalse(os.path.exists(os.path.join(tmp, "shots", "0", IMAGE_MODEL, "last_frame.png")))
            self.assertEqual([os.path.basename(path) for path in video_generator.calls[0]["reference_image_paths"]], ["first_frame.png"])

    async def test_both_keyframes_reach_the_video_model_when_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "final_video.mp4").write_bytes(b"stale film")
            pipeline, _, video_generator = _pipeline(tmp)
            with patch("pipelines.script2video_pipeline.VideoFileClip", return_value=MagicMock()), \
                 patch("pipelines.script2video_pipeline.concatenate_videoclips", return_value=MagicMock()):
                await pipeline(
                    script="script",
                    user_requirement="req",
                    style="cinematic",
                    characters=[_character()],
                    stop_after="video",
                )
            self.assertEqual(
                [os.path.basename(path) for path in video_generator.calls[0]["reference_image_paths"]],
                ["first_frame.png", "last_frame.png"],
            )


class SingleImageModelTests(unittest.TestCase):
    def test_first_render_records_the_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            working_dir = __import__("pathlib").Path(tmp)
            result = _enforce_render_sequence(
                working_dir,
                image_generator=_Generator(IMAGE_MODEL),
                video_generator=_Generator(VIDEO_MODEL),
                style="cinematic",
                allow_model_change=False,
            )
            self.assertIsNone(result)
            self.assertEqual(_read_render_manifest(working_dir)["image_model"], IMAGE_MODEL)
            self.assertEqual(_read_render_manifest(working_dir)["video_model"], VIDEO_MODEL)
            self.assertEqual(_read_render_manifest(working_dir)["style"], "cinematic")

    def test_changing_the_model_mid_sequence_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            working_dir = __import__("pathlib").Path(tmp)
            _enforce_render_sequence(working_dir, image_generator=_Generator(IMAGE_MODEL), video_generator=_Generator(VIDEO_MODEL), style="cinematic", allow_model_change=False)

            result = _enforce_render_sequence(
                working_dir,
                image_generator=_Generator("other/model"),
                video_generator=_Generator(VIDEO_MODEL),
                style="cinematic",
                allow_model_change=False,
            )

            self.assertIsNotNone(result)
            self.assertEqual(result["error_type"], "image_model_changed")
            self.assertFalse(result["retryable"])
            self.assertIn(IMAGE_MODEL, result["error"])
            self.assertIn("other/model", result["error"])
            # The recorded model is untouched, so the sequence stays consistent.
            self.assertEqual(_read_render_manifest(working_dir)["image_model"], IMAGE_MODEL)

    def test_explicit_approval_allows_the_model_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            working_dir = __import__("pathlib").Path(tmp)
            _enforce_render_sequence(working_dir, image_generator=_Generator(IMAGE_MODEL), video_generator=_Generator(VIDEO_MODEL), style="cinematic", allow_model_change=False)

            result = _enforce_render_sequence(
                working_dir,
                image_generator=_Generator("other/model"),
                video_generator=_Generator(VIDEO_MODEL),
                style="cinematic",
                allow_model_change=True,
            )

            self.assertIsNone(result)
            self.assertEqual(_read_render_manifest(working_dir)["image_model"], "other/model")

    def test_changing_the_project_style_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            working_dir = __import__("pathlib").Path(tmp)
            _enforce_render_sequence(working_dir, image_generator=_Generator(IMAGE_MODEL), video_generator=_Generator(VIDEO_MODEL), style="photorealistic cinematic live action", allow_model_change=False)

            # Artifacts rendered under the old style cannot be reused, so the render
            # must not silently proceed with a new one.
            result = _enforce_render_sequence(
                working_dir,
                image_generator=_Generator(IMAGE_MODEL),
                video_generator=_Generator(VIDEO_MODEL),
                style="stylized 3D animated short film",
                allow_model_change=False,
            )

            self.assertIsNotNone(result)
            self.assertEqual(result["error_type"], "style_changed")
            self.assertFalse(result["retryable"])
            self.assertIn("project page", result["error"])
            self.assertIn("photorealistic cinematic live action", result["error"])
            self.assertEqual(_read_render_manifest(working_dir)["style"], "photorealistic cinematic live action")

    def test_repeated_renders_with_the_same_style_are_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            working_dir = __import__("pathlib").Path(tmp)
            for _ in range(2):
                result = _enforce_render_sequence(
                    working_dir,
                    image_generator=_Generator(IMAGE_MODEL),
                    video_generator=_Generator(VIDEO_MODEL),
                    style="cinematic",
                    allow_model_change=False,
                )
                self.assertIsNone(result)


class _PhaseOutcomePipeline:
    """Stands in for a pipeline, returning a gated outcome."""

    def __init__(self, **kwargs) -> None:
        self.working_dir = kwargs.get("working_dir", "")

    async def __call__(self, **kwargs):
        from pipelines.render_contract import RenderOutcome

        return RenderOutcome(
            phase="stills",
            style="cinematic",
            image_model=IMAGE_MODEL,
            video_model=VIDEO_MODEL,
            stills=[os.path.join(str(self.working_dir), "shots", "0", "test_image-model", "first_frame.png")],
            awaiting_confirmation="video",
        )


class RenderToolGateTests(unittest.IsolatedAsyncioTestCase):
    def _adapter(self, tmp: str) -> tuple[ViMaxAdapters, dict]:
        index = SessionIndex(tmp)
        record = index.create(idea="a comedic short")
        root = __import__("pathlib").Path(tmp) / record["working_dir"]
        (root / "script2video" / "shots" / "0").mkdir(parents=True, exist_ok=True)
        (root / "script2video" / "script.txt").write_text("script", encoding="utf-8")
        (root / "script2video" / "characters.json").write_text("[]", encoding="utf-8")
        (root / "script2video" / "storyboard.json").write_text("[]", encoding="utf-8")
        (root / "script2video" / "camera_tree.json").write_text("[]", encoding="utf-8")
        (root / "script2video" / "shots" / "0" / "shot_description.json").write_text("{}", encoding="utf-8")
        return ViMaxAdapters(__import__("pathlib").Path(tmp), index), record

    async def test_gated_phase_asks_for_confirmation_instead_of_completing(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter, _ = self._adapter(tmp)
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_image_generator", return_value=_Generator(IMAGE_MODEL)), \
                 patch("agent_runtime.vimax_adapters._build_video_generator", return_value=_Generator(VIDEO_MODEL)), \
                 patch("agent_runtime.vimax_adapters.Script2VideoPipeline", _PhaseOutcomePipeline):
                result = await adapter.vimax_render_video({"stop_after": "stills"})

            self.assertTrue(result.ok)
            self.assertEqual(result.metadata["phase"], "stills")
            self.assertEqual(result.metadata["awaiting_confirmation"], "video")
            self.assertFalse(result.metadata["render_completed"])
            self.assertIn("stop_after=\"video\"", result.content)
            self.assertIn("cinematic", result.content)

    async def test_model_change_is_refused_by_the_tool(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter, record = self._adapter(tmp)
            working_dir = __import__("pathlib").Path(tmp) / record["working_dir"]
            (working_dir / RENDER_MANIFEST_FILENAME).write_text(json.dumps({"image_model": "meta/muse-image"}), encoding="utf-8")

            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_image_generator", return_value=_Generator(IMAGE_MODEL)), \
                 patch("agent_runtime.vimax_adapters._build_video_generator", return_value=_Generator(VIDEO_MODEL)), \
                 patch("agent_runtime.vimax_adapters.Script2VideoPipeline", _PhaseOutcomePipeline):
                result = await adapter.vimax_render_video({"stop_after": "stills"})

            self.assertFalse(result.ok)
            self.assertEqual(result.metadata["error_type"], "image_model_changed")
            self.assertIn("meta/muse-image", result.content)


class IdeaSceneSchedulingTests(unittest.IsolatedAsyncioTestCase):
    async def test_scoped_partial_scene_does_not_render_others_or_hide_a_stale_film(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _idea_pipeline(tmp, ["scene zero", "scene one"])
            root_film = Path(tmp) / "final_video.mp4"
            root_film.write_bytes(b"old complete film")
            complete_mode = {"enabled": False}
            scene_calls = []
            concat_inputs = []
            progress_events = []

            class ScenePipeline:
                def __init__(self, **kwargs):
                    self.working_dir = kwargs["working_dir"]

                async def __call__(self, **kwargs):
                    scene_idx = int(Path(self.working_dir).name.removeprefix("scene_"))
                    scene_calls.append((scene_idx, kwargs["only_shots"]))
                    if not complete_mode["enabled"]:
                        return RenderOutcome(phase="video", style=kwargs["style"], final_video_path="")
                    film = Path(self.working_dir) / "final_video.mp4"
                    film.parent.mkdir(parents=True, exist_ok=True)
                    film.write_bytes(f"scene {scene_idx}".encode())
                    return RenderOutcome(phase="video", style=kwargs["style"], final_video_path=str(film))

            def concatenate(paths, output):
                concat_inputs.append(list(paths))
                Path(output).write_bytes(b"joined scenes")

            with patch("pipelines.idea2video_pipeline.Script2VideoPipeline", ScenePipeline), \
                 patch("pipelines.idea2video_pipeline.concatenate_video_files", side_effect=concatenate):
                partial = await pipeline(
                    idea="idea",
                    user_requirement="req",
                    style="cinematic",
                    stop_after="video",
                    only_shots=["scene_0/2"],
                    progress=lambda stage, message, metadata=None: progress_events.append(stage),
                )

                self.assertEqual(partial.final_video_path, "")
                self.assertFalse(root_film.exists(), "a pre-existing root film must not survive an incomplete pass")
                self.assertEqual(scene_calls, [(0, [2])])
                self.assertFalse((Path(tmp) / "scene_1" / "final_video.mp4").exists())
                self.assertEqual(concat_inputs, [])
                self.assertIn("scene_partial", progress_events)
                self.assertIn("scene_deferred", progress_events)
                self.assertIn("render_partial", progress_events)

                complete_mode["enabled"] = True
                complete = await pipeline(
                    idea="idea",
                    user_requirement="req",
                    style="cinematic",
                    stop_after="video",
                )

            self.assertEqual(complete.final_video_path, str(root_film))
            self.assertEqual(
                concat_inputs,
                [[str(Path(tmp) / "scene_0" / "final_video.mp4"), str(Path(tmp) / "scene_1" / "final_video.mp4")]],
            )
            self.assertEqual(root_film.read_bytes(), b"joined scenes")

    async def test_idea_rerun_reuses_unchanged_film_and_rebuilds_changed_scene_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _idea_pipeline(tmp, ["scene zero", "scene one"])
            joined_inputs = []

            class ScenePipeline:
                def __init__(self, **kwargs):
                    self.working_dir = kwargs["working_dir"]

                async def __call__(self, **kwargs):
                    scene_idx = int(Path(self.working_dir).name.removeprefix("scene_"))
                    film = Path(self.working_dir) / "final_video.mp4"
                    film.parent.mkdir(parents=True, exist_ok=True)
                    if not film.exists():
                        film.write_bytes(f"scene {scene_idx}".encode())
                    return RenderOutcome(phase="video", style=kwargs["style"], final_video_path=str(film))

            def concatenate(paths, output):
                joined_inputs.append([Path(path).read_bytes() for path in paths])
                Path(output).write_bytes(f"joined {len(joined_inputs)}".encode())

            with patch("pipelines.idea2video_pipeline.Script2VideoPipeline", ScenePipeline), \
                 patch("pipelines.idea2video_pipeline.concatenate_video_files", side_effect=concatenate):
                first = await pipeline(
                    idea="idea", user_requirement="req", style="cinematic", stop_after="video",
                )
                accepted_bytes = Path(first.final_video_path).read_bytes()
                accepted_mtime = Path(first.final_video_path).stat().st_mtime_ns
                repeat = await pipeline(
                    idea="idea", user_requirement="req", style="cinematic", stop_after="video",
                )
                self.assertEqual(Path(repeat.final_video_path).read_bytes(), accepted_bytes)
                self.assertEqual(Path(repeat.final_video_path).stat().st_mtime_ns, accepted_mtime)
                self.assertEqual(len(joined_inputs), 1)

                changed_scene = Path(tmp) / "scene_1" / "final_video.mp4"
                changed_scene.write_bytes(b"updated scene one")
                changed = await pipeline(
                    idea="idea", user_requirement="req", style="cinematic", stop_after="video",
                )

            self.assertEqual(joined_inputs, [[b"scene 0", b"scene 1"], [b"scene 0", b"updated scene one"]])
            self.assertEqual(Path(changed.final_video_path).read_bytes(), b"joined 2")

    async def test_empty_idea_sequence_does_not_assemble_a_film(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _idea_pipeline(tmp, [])
            stale_film = Path(tmp) / "final_video.mp4"
            stale_film.write_bytes(b"old film")

            with patch("pipelines.idea2video_pipeline.concatenate_video_files") as concatenate:
                outcome = await pipeline(
                    idea="idea", user_requirement="req", style="cinematic", stop_after="video",
                )

            self.assertEqual(outcome.final_video_path, "")
            self.assertFalse(stale_film.exists())
            concatenate.assert_not_called()

    async def test_one_video_call_is_shared_across_scene_transitions_and_clips(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _idea_pipeline(tmp, ["scene zero", "scene one"])
            video_calls = []

            class TransitionOutput:
                def save(self, path):
                    Path(path).write_bytes(b"transition")

            class LocalCameraImageGenerator:
                def __init__(self, video_generator):
                    self.video_generator = video_generator

                async def generate_transition_video(self, **kwargs):
                    video_calls.append("transition")
                    await self.video_generator.generate_single_video(
                        prompt="transition",
                        reference_image_paths=[kwargs["first_shot_ff_path"]],
                    )
                    return TransitionOutput()

                def get_new_camera_image(self, path):
                    return Image.new("RGB", (16, 9), "red")

            scene_shots = {
                "scene_0": [_shot(0), _shot(1)],
                "scene_1": [_shot(0)],
            }
            scene_cameras = {
                "scene_0": [
                    Camera(idx=0, active_shot_idxs=[0]),
                    Camera(idx=1, active_shot_idxs=[1], parent_shot_idx=0),
                ],
                "scene_1": [Camera(idx=0, active_shot_idxs=[0])],
            }

            def build_scene_pipeline(**kwargs):
                child = Script2VideoPipeline(**kwargs)
                scene = Path(kwargs["working_dir"]).name
                shots = scene_shots[scene]
                child.design_storyboard = AsyncMock(return_value=[MagicMock(idx=shot.idx) for shot in shots])

                async def decompose(shot_brief_descriptions, characters, quiet=False):
                    for shot in shots:
                        child.shot_desc_events[shot.idx] = asyncio.Event()
                        child.shot_desc_events[shot.idx].set()
                        child.frame_events[shot.idx] = {
                            "first_frame": asyncio.Event(),
                            "last_frame": asyncio.Event(),
                        }
                    return shots

                child.decompose_visual_descriptions = decompose
                child.construct_camera_tree = AsyncMock(return_value=scene_cameras[scene])
                child.reference_image_selector = MagicMock(
                    select_reference_images_and_generate_prompt=AsyncMock(
                        return_value={"reference_image_path_and_text_pairs": [], "text_prompt": "local prompt"}
                    )
                )
                child.camera_image_generator = LocalCameraImageGenerator(pipeline.video_generator)
                return child

            with patch("pipelines.idea2video_pipeline.Script2VideoPipeline", side_effect=build_scene_pipeline), \
                 patch("pipelines.idea2video_pipeline.concatenate_video_files") as concatenate:
                outcome = await pipeline(
                    idea="idea",
                    user_requirement="req",
                    style="cinematic",
                    stop_after="video",
                )

            self.assertEqual(video_calls, ["transition"])
            self.assertEqual(len(pipeline.video_generator.calls), 1)
            self.assertEqual(outcome.final_video_path, "")
            concatenate.assert_not_called()

if __name__ == "__main__":
    unittest.main()