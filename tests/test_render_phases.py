"""Render phase gates, single-model pinning, and model-scoped artifact layout."""

import asyncio
import hashlib
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


def _shot(idx: int, variation_type: str = "small", duration_seconds: int | None = None) -> ShotDescription:
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
        duration_seconds=duration_seconds,
    )



def _approve_test_set(render_root: Path, slots: list[str]) -> str:
    """Create a real, accepted set reference for phase tests without external generation."""
    set_dir = render_root / "set_references" / "saloon"
    set_dir.mkdir(parents=True)
    spec = {
        "id": "saloon",
        "name": "Saloon",
        "description": "A character-free western saloon interior.",
        "shots": [str(slot) for slot in slots],
    }
    spec_path = set_dir / "set.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    image_path = set_dir / "test_image-model" / "reference.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (16, 9), "brown").save(image_path)
    provenance_path = image_path.with_name("provenance.json")
    provenance_path.write_text(json.dumps({
        "image_model": IMAGE_MODEL,
        "style": "cinematic",
        "spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
    }), encoding="utf-8")
    project_dir = render_root.parent
    artifacts = []
    for path in (spec_path, image_path, provenance_path):
        raw = path.read_bytes()
        artifacts.append({
            "path": path.relative_to(project_dir).as_posix(),
            "size": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        })
    (project_dir / "render_acceptance.json").write_text(json.dumps({
        render_root.name: {
            "sets": {"saloon": {"accepted_at": "test", "artifacts": artifacts}},
        },
    }), encoding="utf-8")
    for slot in slots:
        parts = str(slot).split("/")
        shot_description = (render_root / parts[0] / "shots" / parts[1] / "shot_description.json"
                            if render_root.name == "idea2video" else render_root / "shots" / str(slot) / "shot_description.json")
        shot_description.parent.mkdir(parents=True, exist_ok=True)
        shot_description.write_text(json.dumps({"idx": int(parts[-1])}), encoding="utf-8")
    return str(image_path)

def _pipeline(working_dir: str, *, shots=None) -> tuple[Script2VideoPipeline, _Generator, _Generator]:
    """A pipeline with planning stubbed out and one locally accepted set."""
    shots = shots or [_shot(0)]
    render_root = Path(working_dir) / "script2video"
    set_reference = _approve_test_set(render_root, [shot.idx for shot in shots])
    image_generator = _Generator(IMAGE_MODEL)
    video_generator = _Generator(VIDEO_MODEL)
    pipeline = Script2VideoPipeline(
        chat_model=MagicMock(),
        image_generator=image_generator,
        video_generator=video_generator,
        working_dir=str(render_root),
    )
    pipeline.test_set_reference = set_reference
    pipeline._active_style = "cinematic"
    pipeline.camera_image_generator.set_reference_context(str(render_root), "cinematic")
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

def _write_correction_sidecar(output: str) -> None:
    payload = {"outputHash": hashlib.sha256(Path(output).read_bytes()).hexdigest(), "correction": {"version": 3, "clips": []}}
    Path(output + ".correction.json").write_text(json.dumps(payload), encoding="utf-8")


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
            self.assertEqual(os.path.relpath(pipeline.portraits_dir(), tmp), os.path.join("script2video", "character_portraits", "test_image-model"))
            self.assertEqual(os.path.relpath(pipeline.portraits_registry_path(), tmp), os.path.join("script2video", "character_portraits", "test_image-model", "registry.json"))
            self.assertEqual(os.path.relpath(pipeline.frame_path(3, "first_frame"), tmp), os.path.join("script2video", "shots", "3", "test_image-model", "first_frame.png"))
            self.assertEqual(os.path.relpath(pipeline.clip_path(3), tmp), os.path.join("script2video", "shots", "3", "test_video-model", "video.mp4"))
            # Prompts are inputs, not model output.
            self.assertEqual(os.path.relpath(pipeline.selector_output_path(3, "first_frame"), tmp), os.path.join("script2video", "shots", "3", "first_frame_selector_output.json"))


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
            self.assertEqual(outcome.awaiting_confirmation, "sets")
            self.assertEqual(outcome.style, "cinematic")
            self.assertEqual(len(outcome.stills), 3)  # front, side, back
            self.assertTrue(all(path.startswith(os.path.join(pipeline.working_dir, "character_portraits", "test_image-model")) for path in outcome.stills))
            self.assertEqual(video_generator.calls, [])
            self.assertFalse(os.path.exists(pipeline.frame_path(0, "first_frame")))
            # The portraits phase must not even load the storyboard.
            pipeline.design_storyboard.assert_not_awaited()

    async def test_sets_phase_stops_before_portrait_and_shot_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline, image_generator, video_generator = _pipeline(tmp)
            pipeline._brackets_clips = AsyncMock(side_effect=AssertionError("sets phase must not probe video capabilities"))
            outcome = await pipeline(
                script="script",
                user_requirement="req",
                style="cinematic",
                characters=[_character()],
                stop_after="sets",
                set_ids=["saloon"],
            )

            self.assertEqual(outcome.phase, "sets")
            self.assertEqual(outcome.awaiting_confirmation, "stills")
            self.assertEqual(outcome.set_references, [pipeline.test_set_reference])
            self.assertEqual(image_generator.calls, [])
            self.assertEqual(video_generator.calls, [])
            self.assertFalse(os.path.exists(pipeline.frame_path(0, "first_frame")))
            pipeline.design_storyboard.assert_not_awaited()
            pipeline._brackets_clips.assert_not_awaited()

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
                pipeline.frame_path(0, "first_frame"),
                pipeline.frame_path(0, "last_frame"),
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
            pipeline, _, video_generator = _pipeline(tmp)
            # Rebuilding a clip invalidates the previous final; stub local assembly in this focused test.
            def assemble(clips, output, **kwargs):
                Path(output).write_bytes(b"assembled film")
                return {"version": 3, "canvas": {"width": 1280, "height": 720, "fps": 30}, "crop_zoom": 1.0, "clips": [{"slot": clip["slot"], "status": "unchanged", "reason": "test"} for clip in clips]}
            with patch("pipelines.script2video_pipeline.assemble_video_files", side_effect=assemble):
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
            pipeline, _, video_generator = _pipeline(tmp, shots=[_shot(0), _shot(1), _shot(2)])
            Path(pipeline.working_dir, "final_video.mp4").write_bytes(b"stale film")

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
            self.assertFalse(Path(pipeline.working_dir, "final_video.mp4").exists())


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
            # Pre-create transition images so this test only covers frame scheduling.
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



    def _prime_playable_shots(self, pipeline, colors):
        from moviepy import ColorClip

        for shot_idx, color in enumerate(colors):
            for frame_type in ("first_frame", "last_frame"):
                frame_path = pipeline.frame_path(shot_idx, frame_type)
                os.makedirs(os.path.dirname(frame_path), exist_ok=True)
                Image.new("RGB", (1280, 720), color).save(frame_path)
            clip_path = pipeline.clip_path(shot_idx)
            os.makedirs(os.path.dirname(clip_path), exist_ok=True)
            with ColorClip((64, 36), color=color, duration=0.5) as clip:
                clip.write_videofile(
                    clip_path, fps=12, codec="libx264", audio=False, threads=1, logger=None,
                )

    async def test_legacy_assembly_rebuilds_from_cached_shots_without_regeneration(self):
        from moviepy import VideoFileClip

        with tempfile.TemporaryDirectory() as tmp:
            pipeline, images, videos = _pipeline(tmp, shots=[_shot(0), _shot(1)])
            self._prime_playable_shots(pipeline, [(255, 0, 0), (0, 0, 255)])
            source_bytes = {path: Path(path).read_bytes() for path in (
                pipeline.clip_path(0), pipeline.clip_path(1),
                pipeline.frame_path(0, "first_frame"), pipeline.frame_path(0, "last_frame"),
                pipeline.frame_path(1, "first_frame"), pipeline.frame_path(1, "last_frame"),
            )}
            final_path = Path(pipeline.working_dir, "final_video.mp4")
            final_path.write_bytes(b"legacy film")
            Path(str(final_path) + ".inputs.json").write_text(json.dumps({"version": 2}))

            outcome = await pipeline(
                script="script", user_requirement="req", style="cinematic",
                characters=[], character_portraits_registry={}, stop_after="video",
            )

            self.assertEqual(outcome.final_video_path, str(final_path))
            with VideoFileClip(outcome.final_video_path) as film:
                self.assertEqual(tuple(film.size), (1280, 720))
                self.assertAlmostEqual(film.duration, 1, places=1)
                self.assertGreater(film.get_frame(0.25)[360, 640, 0], 220)
                self.assertGreater(film.get_frame(0.75)[360, 640, 2], 220)
            report = json.loads(Path(str(final_path) + ".correction.json").read_text())
            self.assertEqual(report["outputHash"], hashlib.sha256(final_path.read_bytes()).hexdigest())
            self.assertEqual([clip["slot"] for clip in report["correction"]["clips"]], ["0", "1"])
            self.assertEqual(source_bytes, {path: Path(path).read_bytes() for path in source_bytes})
            self.assertEqual(images.calls, [])
            self.assertEqual(videos.calls, [])

    async def test_assembly_failure_preserves_prior_final(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline, _, _ = _pipeline(tmp, shots=[_shot(0)])
            self._prime_playable_shots(pipeline, [(255, 0, 0)])
            final_path = Path(pipeline.working_dir, "final_video.mp4")
            final_path.write_bytes(b"previous complete film")
            missing_ffmpeg = Path(tmp, "missing-ffmpeg")
            with patch.dict(os.environ, {"VIMAX_FFMPEG_CMD": str(missing_ffmpeg)}):
                with self.assertRaises(FileNotFoundError):
                    await pipeline(
                        script="script", user_requirement="req", style="cinematic",
                        characters=[], character_portraits_registry={}, stop_after="video",
                    )
            self.assertEqual(final_path.read_bytes(), b"previous complete film")

    async def test_native_assembly_preserves_playback_order_cache_and_removal(self):
        from moviepy import VideoFileClip

        with tempfile.TemporaryDirectory() as tmp:
            shots = [_shot(0), _shot(1), _shot(2)]
            pipeline, images, videos = _pipeline(tmp, shots=shots)
            source_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            self._prime_playable_shots(pipeline, source_colors)
            source_paths = [
                *(pipeline.clip_path(idx) for idx in range(3)),
                *(pipeline.frame_path(idx, frame_type) for idx in range(3) for frame_type in ("first_frame", "last_frame")),
            ]
            source_bytes = {path: Path(path).read_bytes() for path in source_paths}
            active_order = [0, 2, 1]

            async def camera_tree(**kwargs):
                return [Camera(idx=0, active_shot_idxs=list(active_order))]

            pipeline.construct_camera_tree = camera_tree

            async def render_and_assert(expected_colors):
                outcome = await pipeline(
                    script="script", user_requirement="req", style="cinematic",
                    characters=[], character_portraits_registry={}, stop_after="video",
                )
                with VideoFileClip(outcome.final_video_path) as film:
                    self.assertEqual(tuple(film.size), (1280, 720))
                    self.assertAlmostEqual(film.duration, len(expected_colors) * 0.5, places=1)
                    for index, channel in enumerate(expected_colors):
                        frame = film.get_frame(index * 0.5 + 0.25)[360, 640]
                        self.assertGreater(frame[channel], 220)
                        self.assertLess(max(frame[other] for other in range(3) if other != channel), 30)
                report = json.loads(Path(outcome.final_video_path + ".correction.json").read_text())
                self.assertEqual(report["outputHash"], hashlib.sha256(Path(outcome.final_video_path).read_bytes()).hexdigest())
                self.assertEqual([int(item["slot"]) for item in report["correction"]["clips"]], active_order)
                return outcome

            first = await render_and_assert([0, 2, 1])
            first_mtime = Path(first.final_video_path).stat().st_mtime_ns
            repeated = await render_and_assert([0, 2, 1])
            self.assertEqual(Path(repeated.final_video_path).stat().st_mtime_ns, first_mtime)
            changed_keyframe = pipeline.frame_path(0, "first_frame")
            Image.new("RGB", (1280, 720), (128, 0, 0)).save(changed_keyframe)
            source_bytes[changed_keyframe] = Path(changed_keyframe).read_bytes()
            changed_anchor = await render_and_assert([0, 2, 1])
            self.assertNotEqual(Path(changed_anchor.final_video_path).stat().st_mtime_ns, first_mtime)
            active_order[:] = [0, 1, 2]
            second = await render_and_assert([0, 1, 2])
            active_order[:] = [0, 2]
            third = await render_and_assert([0, 2])

            self.assertEqual(first.final_video_path, second.final_video_path)
            self.assertEqual(second.final_video_path, third.final_video_path)
            self.assertEqual(source_bytes, {path: Path(path).read_bytes() for path in source_paths})
            self.assertEqual(images.calls, [])
            self.assertEqual(videos.calls, [])
    async def test_empty_active_sequence_does_not_assemble_a_film(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline, _, _ = _pipeline(tmp)
            pipeline.construct_camera_tree = AsyncMock(return_value=[Camera(idx=0, active_shot_idxs=[])])
            stale_film = Path(pipeline.working_dir) / "final_video.mp4"
            stale_film.write_bytes(b"old film")

            outcome = await pipeline(
                script="script", user_requirement="req", style="cinematic",
                characters=[_character()], stop_after="video",
            )

            self.assertEqual(outcome.final_video_path, "")
            self.assertFalse(stale_film.exists())


class ClipLengthTests(unittest.TestCase):
    """A default change must not sweep paid clips with unrelated per-shot lengths."""

    def _root_with(self, tmp):
        root = Path(tmp) / "script2video"
        clip = root / "shots/2/test_video-model/video.mp4"
        clip.parent.mkdir(parents=True, exist_ok=True)
        clip.write_bytes(b"old clip")
        (root / "final_video.mp4").write_bytes(b"film")
        return root, clip

    def test_default_duration_change_records_default_without_moving_cached_media(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, clip = self._root_with(tmp)
            manifest = Path(tmp) / RENDER_MANIFEST_FILENAME
            manifest.write_text(json.dumps({
                "image_model": IMAGE_MODEL,
                "video_model": VIDEO_MODEL,
                "style": "cinematic",
                "render_mode": "script2video",
                "clip_seconds": 5,
            }), encoding="utf-8")
            generator = _Generator(VIDEO_MODEL)
            generator.clip_seconds = 8

            refusal = _enforce_render_sequence(
                Path(tmp),
                image_generator=_Generator(IMAGE_MODEL),
                video_generator=generator,
                style="cinematic",
                allow_model_change=False,
                render_mode="script2video",
            )

            self.assertIsNone(refusal)
            self.assertTrue(clip.exists())
            self.assertTrue((root / "final_video.mp4").exists())
            self.assertEqual(json.loads(manifest.read_text(encoding="utf-8"))["clip_seconds"], 8)


class ShotDurationSchemaTests(unittest.TestCase):
    def test_positive_integer_override_survives_saved_schema_roundtrip(self):
        shot = _shot(3, duration_seconds=10)
        reloaded = ShotDescription.model_validate(json.loads(json.dumps(shot.model_dump())))
        self.assertEqual(reloaded.duration_seconds, 10)
        self.assertIsNone(_shot(4).duration_seconds)
        for invalid in (0, -1, 2.5, True):
            with self.subTest(duration_seconds=invalid):
                payload = shot.model_dump()
                payload["duration_seconds"] = invalid
                with self.assertRaises(ValueError):
                    ShotDescription.model_validate(payload)


class ShotDurationPreflightTests(unittest.IsolatedAsyncioTestCase):
    async def test_unconfigured_override_aborts_before_image_or_video_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline, image_generator, video_generator = _pipeline(tmp)
            video_generator.duration_policy = {
                "provider": "openrouter",
                "model": "test/model",
                "configured": False,
                "defaultSeconds": 8,
                "minSeconds": 4,
                "maxSeconds": 8,
                "allowedSeconds": [4, 6, 8],
            }
            description_path = Path(pipeline.working_dir, "shots", "0", "shot_description.json")
            description_path.write_text(json.dumps(_shot(0, duration_seconds=6).model_dump()), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Configure a limit"):
                await pipeline(
                    script="script",
                    user_requirement="req",
                    style="cinematic",
                    characters=[_character()],
                    character_portraits_registry={},
                    stop_after="video",
                )

            self.assertEqual(image_generator.calls, [])
            self.assertEqual(video_generator.calls, [])

    async def test_scoped_preflight_ignores_an_unrelated_invalid_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline, _, video_generator = _pipeline(tmp, shots=[_shot(0), _shot(1)])
            video_generator.duration_policy = {
                "provider": "openrouter",
                "model": "test/model",
                "configured": True,
                "defaultSeconds": 8,
                "minSeconds": 5,
                "maxSeconds": 10,
                "allowedSeconds": [5, 10],
            }
            for shot_idx, duration in ((0, 5), (1, 6)):
                path = Path(pipeline.working_dir, "shots", str(shot_idx), "shot_description.json")
                path.write_text(json.dumps(_shot(shot_idx, duration_seconds=duration).model_dump()), encoding="utf-8")

            pipeline._preflight_saved_shot_durations({0})
            with self.assertRaisesRegex(ValueError, "Shot 1 duration 6s is unsupported"):
                pipeline._preflight_saved_shot_durations(None)


class ClipBracketingTests(unittest.IsolatedAsyncioTestCase):
    """A clip is bracketed by both keyframes only where the model accepts an end frame."""

    async def test_no_end_keyframe_is_generated_for_a_model_that_rejects_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline, _, video_generator = _pipeline(tmp)
            Path(pipeline.working_dir, "final_video.mp4").write_bytes(b"stale film")

            async def supports_last_frame():
                return False

            video_generator.supports_last_frame = supports_last_frame
            def assemble(clips, output, **kwargs):
                Path(output).write_bytes(b"assembled")
                return {"version": 3, "canvas": {"width": 1280, "height": 720, "fps": 30}, "crop_zoom": 1.0, "clips": [{"slot": clip["slot"], "status": "unchanged", "reason": "test"} for clip in clips]}
            with patch("pipelines.script2video_pipeline.assemble_video_files", side_effect=assemble):
                await pipeline(
                    script="script",
                    user_requirement="req",
                    style="cinematic",
                    characters=[_character()],
                    stop_after="video",
                )

            self.assertFalse(os.path.exists(pipeline.frame_path(0, "last_frame")))
            self.assertEqual([os.path.basename(path) for path in video_generator.calls[0]["reference_image_paths"]], ["first_frame.png"])

    async def test_both_keyframes_reach_the_video_model_when_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline, _, video_generator = _pipeline(tmp)
            Path(pipeline.working_dir, "final_video.mp4").write_bytes(b"stale film")
            def assemble(clips, output, **kwargs):
                Path(output).write_bytes(b"assembled")
                return {"version": 3, "canvas": {"width": 1280, "height": 720, "fps": 30}, "crop_zoom": 1.0, "clips": [{"slot": clip["slot"], "status": "unchanged", "reason": "test"} for clip in clips]}
            with patch("pipelines.script2video_pipeline.assemble_video_files", side_effect=assemble):
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


    async def test_model_change_is_refused_by_the_tool(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter, record = self._adapter(tmp)
            working_dir = __import__("pathlib").Path(tmp) / record["working_dir"]
            (working_dir / RENDER_MANIFEST_FILENAME).write_text(json.dumps({"image_model": "meta/muse-image"}), encoding="utf-8")

            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_image_generator", return_value=_Generator(IMAGE_MODEL)), \
                 patch("agent_runtime.vimax_adapters._build_video_generator", return_value=_Generator(VIDEO_MODEL)), \
                 patch("pipelines.script2video_pipeline.Script2VideoPipeline", _PhaseOutcomePipeline):
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
                _write_correction_sidecar(output)

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
                _write_correction_sidecar(output)

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
            render_root = Path(tmp) / "idea2video"
            pipeline = _idea_pipeline(str(render_root), ["scene zero", "scene one"])
            _approve_test_set(render_root, ["scene_0/0", "scene_0/1", "scene_1/0"])
            video_calls = []

            class TransitionOutput:
                def save(self, path):
                    Path(path).write_bytes(b"transition")

            class LocalCameraImageGenerator:
                def __init__(self, video_generator):
                    self.video_generator = video_generator
                def set_reference_context(self, working_dir, style):
                    self.working_dir = working_dir
                    self.style = style

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

class PortraitCacheRegressionTests(unittest.IsolatedAsyncioTestCase):
    def characters(self):
        return [_character()] + [
            _character().model_copy(update={"idx": idx, "identifier_in_scene": name})
            for idx, name in enumerate(("Woman1", "Woman2", "Bartender"), 1)
        ]

    async def test_missing_registry_character_and_changed_spec_regenerate_only_that_character(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline, images, videos = _pipeline(tmp)
            characters = self.characters()
            await pipeline(script="script", user_requirement="", characters=characters, style="cinematic", stop_after="portraits")
            registry_path = Path(pipeline.portraits_registry_path())
            registry = json.loads(registry_path.read_text())
            del registry["Claude"]
            registry_path.write_text(json.dumps(registry))
            images.calls.clear()
            await pipeline(script="script", user_requirement="", characters=characters, style="cinematic", stop_after="portraits")
            self.assertEqual(len(images.calls), 3)
            self.assertTrue(all("character Claude" in call["prompt"] for call in images.calls))
            restored = json.loads(registry_path.read_text())
            self.assertEqual(set(restored), {c.identifier_in_scene for c in characters})
            characters[0] = characters[0].model_copy(update={"dynamic_features": "a green shirt"})
            images.calls.clear()
            await pipeline(script="script", user_requirement="", characters=characters, style="cinematic", stop_after="portraits")
            self.assertEqual(len(images.calls), 3)
            self.assertIn("a green shirt", images.calls[0]["prompt"])
            self.assertEqual(videos.calls, [])

    async def test_dangling_view_regenerates_only_missing_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline, images, _ = _pipeline(tmp)
            await pipeline(script="script", user_requirement="", characters=[_character()], style="cinematic", stop_after="portraits")
            registry = json.loads(Path(pipeline.portraits_registry_path()).read_text())
            Path(registry["Claude"]["side"]["path"]).unlink()
            images.calls.clear()
            await pipeline(script="script", user_requirement="", characters=[_character()], style="cinematic", stop_after="portraits")
            self.assertEqual(len(images.calls), 1)
            self.assertIn("side-view", images.calls[0]["prompt"])

    async def test_failure_drains_other_characters_and_resumes_partial_views(self):
        from pipelines.render_contract import PortraitGenerationError
        with tempfile.TemporaryDirectory() as tmp:
            pipeline, images, _ = _pipeline(tmp)
            generate = images.generate_single_image

            async def fail_side(**kwargs):
                if "side-view portrait of character Claude" in kwargs["prompt"]:
                    raise RuntimeError("controlled side failure")
                await asyncio.sleep(0)
                return await generate(**kwargs)

            images.generate_single_image = fail_side
            with self.assertRaises(PortraitGenerationError) as caught:
                await pipeline(script="script", user_requirement="", characters=self.characters(), style="cinematic", stop_after="portraits")
            self.assertEqual(set(caught.exception.failures), {"Claude"})
            self.assertEqual({Path(path).name for path in caught.exception.missing}, {"side.png", "back.png"})
            registry = json.loads(Path(pipeline.portraits_registry_path()).read_text())
            self.assertEqual(set(registry), {"Woman1", "Woman2", "Bartender"})
            images.generate_single_image = generate
            images.calls.clear()
            await pipeline(script="script", user_requirement="", characters=self.characters(), style="cinematic", stop_after="portraits")
            self.assertEqual(len(images.calls), 2)
            self.assertTrue(all("character Claude" in call["prompt"] for call in images.calls))

    async def test_empty_entry_does_not_reuse_untracked_pngs(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline, images, _ = _pipeline(tmp)
            directory = Path(pipeline.portraits_dir()) / "0_Claude"
            directory.mkdir(parents=True)
            for view in ("front", "side", "back"):
                Image.new("RGB", (16, 9), "red").save(directory / f"{view}.png")
            Path(pipeline.portraits_registry_path()).write_text(json.dumps({"Claude": {}}))
            await pipeline(script="script", user_requirement="", characters=[_character()], style="cinematic", stop_after="portraits")
            self.assertEqual(len(images.calls), 3)
            with Image.open(directory / "front.png") as image:
                self.assertEqual(image.getpixel((0, 0)), (0, 0, 255))

    async def test_partial_shared_registry_preserves_external_views(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline, images, _ = _pipeline(str(Path(tmp) / "scene_0"))
            shared = Path(tmp) / "shared"
            shared.mkdir()
            entry = {}
            for view in ("front", "back"):
                path = shared / f"{view}.png"
                Image.new("RGB", (16, 9), "red").save(path)
                entry[view] = {"path": str(path), "description": view}
            before = {path: path.read_bytes() for path in shared.iterdir()}
            await pipeline(script="script", user_requirement="", characters=[_character()],
                           character_portraits_registry={"Claude": entry}, style="cinematic", stop_after="portraits")
            self.assertEqual(len(images.calls), 1)
            self.assertIn("side-view", images.calls[0]["prompt"])
            self.assertEqual(images.calls[0]["reference_image_paths"], [entry["front"]["path"]])
            self.assertEqual({path: path.read_bytes() for path in shared.iterdir()}, before)



class PredecessorFrameReferenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_boundaries_follow_playback_order_and_use_end_caption(self):
        with tempfile.TemporaryDirectory() as tmp:
            shots = [_shot(idx) for idx in (0, 4, 2)]
            pipeline, images, _ = _pipeline(tmp, shots=shots)
            pipeline.frame_bracketing = True
            generate = images.generate_single_image

            async def require_existing_source(**kwargs):
                for source in kwargs["reference_image_paths"]:
                    self.assertTrue(Path(source).is_file(), source)
                return await generate(**kwargs)

            images.generate_single_image = require_existing_source
            await asyncio.wait_for(pipeline.generate_frames_for_single_camera(
                camera=Camera(idx=0, active_shot_idxs=[0, 4, 2]),
                shot_descriptions=shots, characters=[], character_portraits_registry={},
                priority_shot_idxs=[2],
            ), timeout=2)
            for shot, predecessor in ((4, 0), (2, 4)):
                selector = pipeline.load_selector_output(shot, "first_frame")
                self.assertEqual(selector["reference_image_path_and_text_pairs"], [
                    [pipeline.test_set_reference, "Approved set reference: Saloon. Preserve its architecture, layout, props and lighting. A character-free western saloon interior."],
                    [pipeline.frame_path(predecessor, "last_frame"), f"last frame {predecessor}"],
                ])
                sent = next(call for call in images.calls if f"last frame {predecessor}\n" in call["prompt"])
                self.assertEqual(sent["reference_image_paths"], [pipeline.test_set_reference, pipeline.frame_path(predecessor, "last_frame")])
            for shot in (0, 4, 2):
                self.assertEqual(pipeline.load_selector_output(shot, "last_frame")["reference_image_path_and_text_pairs"], [
                    [pipeline.test_set_reference, "Approved set reference: Saloon. Preserve its architecture, layout, props and lighting. A character-free western saloon interior."],
                    [pipeline.frame_path(shot, "first_frame"), f"first frame {shot}"],
                ])

    async def test_scoped_redraw_replaces_wrong_cached_boundary_without_drawing_predecessor(self):
        await self._check_scoped_redraw("first_frame")

    async def test_cached_end_image_with_opening_caption_is_reselected(self):
        await self._check_scoped_redraw("last_frame")

    async def _check_scoped_redraw(self, cached_source_frame):
        with tempfile.TemporaryDirectory() as tmp:
            shots = [_shot(0), _shot(1)]
            pipeline, images, _ = _pipeline(tmp, shots=shots)
            pipeline.frame_bracketing = True
            for frame in ("first_frame", "last_frame"):
                file = Path(pipeline.frame_path(0, frame))
                file.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (16, 9), "red" if frame == "first_frame" else "green").save(file)
            before = {frame: Path(pipeline.frame_path(0, frame)).read_bytes() for frame in ("first_frame", "last_frame")}
            cached = Path(pipeline.selector_output_path(1, "first_frame"))
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_text(json.dumps({
                "reference_image_path_and_text_pairs": [[pipeline.frame_path(0, cached_source_frame), "first frame 0"]],
                "text_prompt": "old opening-state prompt",
                "sent_prompt": "old opening-state prompt",
            }))
            await asyncio.wait_for(pipeline.generate_frames_for_single_camera(
                camera=Camera(idx=0, active_shot_idxs=[0, 1]),
                shot_descriptions=shots, characters=[], character_portraits_registry={},
                priority_shot_idxs=[], only_shots=[1],
            ), timeout=2)
            self.assertEqual(len(images.calls), 2)
            self.assertEqual(images.calls[0]["reference_image_paths"], [pipeline.test_set_reference, pipeline.frame_path(0, "last_frame")])
            self.assertIn("Image 1: last frame 0", images.calls[0]["prompt"])
            self.assertNotIn("old opening-state prompt", images.calls[0]["prompt"])
            self.assertEqual(before, {frame: Path(pipeline.frame_path(0, frame)).read_bytes() for frame in before})


if __name__ == "__main__":
    unittest.main()