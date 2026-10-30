import asyncio
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from PIL import Image

from agent_runtime.vimax_adapters import _rendered_ending_preflight_refusal
from interfaces import Camera, ShotDescription
from pipelines.script2video_pipeline import Script2VideoPipeline
from utils.video_assembly import prepare_rendered_ending


class _Images:
    model = "test-image"
    reference_limit = 4

    def __init__(self):
        self.calls = []

    async def prepare(self):
        return None

    async def generate_single_image(self, **kwargs):
        self.calls.append(kwargs)
        return Image.new("RGB", (32, 18), "green")


class _Video:
    model = "selected-video"
    clip_seconds = 2
    duration_policy = None

    def __init__(self):
        self.calls = []

    async def supports_last_frame(self):
        return False

    async def generate_single_video(self, **kwargs):
        self.calls.append(kwargs)
        return _VideoOutput()


class _VideoOutput:
    def save(self, path):
        Path(path).write_bytes(b"locally generated target clip")


def _shot(idx, camera, reuse=None):
    return ShotDescription(
        idx=idx,
        is_last=False,
        cam_idx=camera,
        visual_desc=f"shot {idx}",
        variation_type="small",
        variation_reason="",
        ff_desc=f"first {idx}",
        ff_vis_char_idxs=[],
        lf_desc=f"last {idx}",
        lf_vis_char_idxs=[],
        motion_desc="move",
        audio_desc="none",
        reuse_previous_last_frame=reuse,
    )


def _write_clip(path, color):
    from moviepy import ColorClip

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with ColorClip((64, 36), color=color, duration=0.5) as clip:
        clip.write_videofile(
            str(path), fps=12, codec="libx264", audio=False, threads=1, logger=None,
        )
    return path


def _lock(working_dir, path):
    content = path.read_bytes()
    return {
        "path": path.relative_to(working_dir).as_posix(),
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


class RenderedEndingPipelineTests(unittest.IsolatedAsyncioTestCase):
    def _pipeline(self, working_dir):
        images = _Images()
        videos = _Video()
        pipeline = Script2VideoPipeline(MagicMock(), images, videos, str(working_dir))
        pipeline.frame_bracketing = False
        pipeline._accepted_set_references = lambda _shot_idx: []
        pipeline._active_style = "cinematic"
        pipeline.revision_notes = {}
        return pipeline, images, videos

    def _anchor_images(self, pipeline, shot_idx):
        for kind, color in (("first_frame", "blue"), ("last_frame", "yellow")):
            path = Path(pipeline.frame_path(shot_idx, kind))
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 18), color).save(path)

    async def test_cross_scene_reuse_copies_decoded_clip_end_then_sends_that_first_to_first_only_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            previous_scene = project / "idea2video" / "scene_0"
            current_scene = project / "idea2video" / "scene_1"
            previous_scene.mkdir(parents=True)
            current_scene.mkdir(parents=True)
            (previous_scene / "camera_tree.json").write_text(
                json.dumps([{"idx": 0, "active_shot_idxs": [15]}]), encoding="utf-8",
            )
            previous_shot_dir = previous_scene / "shots" / "15"
            previous_shot_dir.mkdir(parents=True)
            (previous_shot_dir / "shot_description.json").write_text(
                json.dumps(_shot(15, 0).model_dump()), encoding="utf-8",
            )
            pipeline, images, videos = self._pipeline(current_scene)
            source_clip = _write_clip(
                previous_shot_dir / "archived-video" / "video.mp4", (240, 20, 20),
            )
            source_images = previous_shot_dir / pipeline.image_model_slug
            source_images.mkdir(parents=True)
            Image.new("RGB", (32, 18), "blue").save(source_images / "first_frame.png")
            Image.new("RGB", (32, 18), "yellow").save(source_images / "last_frame.png")
            target = _shot(2, 0, reuse=True)

            await pipeline.generate_chained_frames(
                camera_tree=[Camera(idx=0, active_shot_idxs=[2])],
                shot_descriptions=[target],
                characters=[],
                character_portraits_registry={},
                only_shots=[2],
            )

            ending = prepare_rendered_ending(
                str(source_clip),
                first_frame=str(previous_scene / "shots" / "15" / "test-image" / "first_frame.png"),
                last_frame=str(previous_scene / "shots" / "15" / "test-image" / "last_frame.png"),
            )
            first = Path(pipeline.frame_path(2, "first_frame"))
            self.assertEqual(first.read_bytes(), Path(ending["path"]).read_bytes())
            with Image.open(first) as image:
                red, green, blue = image.convert("RGB").getpixel((640, 360))
            self.assertGreater(red, 180)
            self.assertLess(green, 70)
            self.assertLess(blue, 70)
            provenance = json.loads(Path(pipeline.selector_output_path(2, "first_frame")).read_text())[
                "continuity_source"
            ]
            self.assertEqual(provenance["kind"], "rendered_clip_end")
            self.assertEqual(provenance["shot_idx"], 15)
            self.assertEqual(provenance["slot"], "scene_0/15")
            self.assertEqual(provenance["clip_sha256"], ending["clip_sha256"])
            self.assertEqual(provenance["normalization"], "rendered-end-v1")
            self.assertEqual(provenance["normalization_inputs"]["last_frame_sha256"], ending["normalization_inputs"]["last_frame_sha256"])
            self.assertEqual(images.calls, [])

            await pipeline.generate_video_for_single_shot(target)
            self.assertEqual(len(videos.calls), 1)
            self.assertEqual(videos.calls[0]["reference_image_paths"], [str(first)])

            changed_anchor = previous_scene / "shots" / "15" / "test-image" / "last_frame.png"
            Image.new("RGB", (32, 18), "purple").save(changed_anchor)
            with self.assertRaisesRegex(RuntimeError, "rendered-ending provenance is missing or stale"):
                await pipeline.generate_chained_frames(
                    camera_tree=[Camera(idx=0, active_shot_idxs=[2])],
                    shot_descriptions=[target],
                    characters=[],
                    character_portraits_registry={},
                    only_shots=[2],
                )
            self.assertEqual(images.calls, [])

    async def test_global_chain_reuses_same_camera_but_reframes_camera_cut_from_actual_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project" / "script2video"
            root.mkdir(parents=True)
            pipeline, images, _videos = self._pipeline(root)
            shots = [_shot(0, 0), _shot(1, 0), _shot(2, 2)]
            source_zero = _write_clip(pipeline.clip_path(0), (220, 15, 15))
            source_one = _write_clip(pipeline.clip_path(1), (15, 15, 220))
            self._anchor_images(pipeline, 0)
            self._anchor_images(pipeline, 1)
            pipeline.revision_notes = {"1": "authorized redraw for the rendered-ending test"}
            camera_tree = [Camera(idx=0, active_shot_idxs=[0, 1]), Camera(idx=2, active_shot_idxs=[2])]

            await pipeline.generate_chained_frames(
                camera_tree=camera_tree,
                shot_descriptions=shots,
                characters=[],
                character_portraits_registry={},
                only_shots=[1, 2],
            )

            first_one = Path(pipeline.frame_path(1, "first_frame"))
            source_zero_end = prepare_rendered_ending(str(source_zero))
            self.assertEqual(first_one.read_bytes(), Path(source_zero_end["path"]).read_bytes())
            first_two = Path(pipeline.frame_path(2, "first_frame"))
            self.assertEqual(len(images.calls), 1)
            source_one_end = prepare_rendered_ending(str(source_one))
            self.assertEqual(images.calls[0]["reference_image_paths"], [source_one_end["path"]])
            reframe = json.loads(Path(pipeline.selector_output_path(2, "first_frame")).read_text())[
                "continuity_source"
            ]
            self.assertEqual(reframe["mode"], "reframe")
            self.assertEqual(reframe["frame_type"], "rendered_end")
            self.assertEqual(reframe["clip_sha256"], source_one_end["clip_sha256"])
            self.assertNotEqual(first_two.read_bytes(), Path(source_one_end["path"]).read_bytes())
            self.assertFalse(Path(pipeline.frame_path(2, "last_frame")).exists())

    async def test_explicit_false_is_independent_when_predecessor_clip_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "script2video"
            root.mkdir()
            pipeline, images, _videos = self._pipeline(root)
            shots = [_shot(0, 0), _shot(1, 0, reuse=False)]
            pipeline.reference_image_selector.select_reference_images_and_generate_prompt = AsyncMock(
                return_value={
                    "reference_image_path_and_text_pairs": [],
                    "selected_indices": [],
                    "text_prompt": "An independently generated view.",
                }
            )

            await pipeline.generate_chained_frames(
                camera_tree=[Camera(idx=0, active_shot_idxs=[0, 1])],
                shot_descriptions=shots,
                characters=[],
                character_portraits_registry={},
                only_shots=[1],
            )

            target_first = Path(pipeline.frame_path(1, "first_frame"))
            with Image.open(target_first) as image:
                self.assertEqual(image.convert("RGB").getpixel((0, 0)), (0, 128, 0))
            self.assertEqual(len(images.calls), 1)
            self.assertEqual(images.calls[0]["reference_image_paths"], [])
            selector = json.loads(
                Path(pipeline.selector_output_path(1, "first_frame")).read_text()
            )
            self.assertNotIn("continuity_source", selector)

    async def test_reviewed_retained_last_frame_is_reused_after_actual_first_preparation(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            root = project / "script2video"
            root.mkdir()
            pipeline, images, videos = self._pipeline(root)
            pipeline.frame_bracketing = True
            shots = [_shot(0, 0), _shot(1, 0, reuse=True)]
            cameras = [Camera(idx=0, active_shot_idxs=[0, 1])]
            source_clip = _write_clip(pipeline.clip_path(0), (220, 15, 15))
            self._anchor_images(pipeline, 0)
            self._anchor_images(pipeline, 1)
            target_first = Path(pipeline.frame_path(1, "first_frame"))
            target_last = Path(pipeline.frame_path(1, "last_frame"))
            prior_first_hash = hashlib.sha256(target_first.read_bytes()).hexdigest()
            last_selector = Path(pipeline.selector_output_path(1, "last_frame"))
            last_selector.write_text(json.dumps({
                "source_frame": {"shot_idx": 1, "frame_type": "first_frame", "sha256": prior_first_hash},
            }))
            retained_last = target_last.read_bytes()
            retained_recipe = last_selector.read_bytes()
            source = pipeline._build_playback_predecessors(cameras, shots)[1]
            ending = prepare_rendered_ending(
                source_clip, first_frame=pipeline.frame_path(0, "first_frame"),
                last_frame=pipeline.frame_path(0, "last_frame"),
            )
            target_first.write_bytes(Path(ending["path"]).read_bytes())
            pipeline._record_continuity_source(1, source, ending, "reuse")
            (project / "render_acceptance.json").write_text(json.dumps({
                "script2video": {"shots": {
                    "0": {"clips": {"accepted_at": "reviewed", "artifacts": [_lock(project, source_clip)]}},
                    "1": {"keyframes": {"accepted_at": "reviewed", "artifacts": [
                        _lock(project, target_first), _lock(project, target_last),
                    ]}},
                }},
            }))
            await pipeline.generate_chained_frames(
                camera_tree=cameras, shot_descriptions=shots, characters=[],
                character_portraits_registry={}, only_shots=[1], reuse_approved_stills=True,
            )
            await pipeline.generate_video_for_single_shot(shots[1])
            self.assertEqual(target_last.read_bytes(), retained_last)
            self.assertEqual(last_selector.read_bytes(), retained_recipe)
            self.assertEqual(images.calls, [])
            self.assertEqual(len(videos.calls), 1)
            self.assertEqual(
                [Path(path).read_bytes() for path in videos.calls[0]["reference_image_paths"]],
                [Path(ending["path"]).read_bytes(), retained_last],
            )

            Image.new("RGB", (32, 18), "white").save(target_last)
            with self.assertRaises(RuntimeError):
                await pipeline.generate_chained_frames(
                    camera_tree=cameras, shot_descriptions=shots, characters=[],
                    character_portraits_registry={}, only_shots=[1], reuse_approved_stills=True,
                )
            self.assertEqual(images.calls, [])
            self.assertEqual(len(videos.calls), 1)

    async def test_missing_concrete_source_clip_blocks_before_image_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "script2video"
            root.mkdir()
            pipeline, images, _videos = self._pipeline(root)
            target = _shot(2, 1, reuse=True)

            with self.assertRaisesRegex(RuntimeError, "rendered video clip is missing"):
                await pipeline.generate_chained_frames(
                    camera_tree=[Camera(idx=0, active_shot_idxs=[0]), Camera(idx=1, active_shot_idxs=[2])],
                    shot_descriptions=[_shot(0, 0), target],
                    characters=[],
                    character_portraits_registry={},
                    only_shots=[2],
                )
            self.assertEqual(images.calls, [])


class RenderedEndingApprovalGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_clip_lock_change_blocks_preflight_and_invalidates_direct_handoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            working = Path(tmp)
            root = working / "script2video"
            root.mkdir()
            (working / "frame_continuity.json").write_text(json.dumps({"mode": "off"}), encoding="utf-8")
            (root / "camera_tree.json").write_text(
                json.dumps([{"idx": 0, "active_shot_idxs": [1, 2]}]), encoding="utf-8",
            )
            source_dir = root / "shots" / "1"
            target_dir = root / "shots" / "2"
            source_dir.mkdir(parents=True)
            target_dir.mkdir(parents=True)
            (source_dir / "shot_description.json").write_text(json.dumps({"idx": 1}), encoding="utf-8")
            (target_dir / "shot_description.json").write_text(
                json.dumps({"idx": 2, "reuse_previous_last_frame": True}), encoding="utf-8",
            )
            clip = _write_clip(source_dir / "archived-video" / "video.mp4", (230, 25, 25))
            target_first = target_dir / "test-image" / "first_frame.png"
            target_first.parent.mkdir(parents=True)
            ending = prepare_rendered_ending(str(clip))
            target_first.write_bytes(Path(ending["path"]).read_bytes())
            target_provenance = {
                "kind": "rendered_clip_end",
                "shot_idx": 1,
                "slot": "1",
                "path": Path(ending["path"]).relative_to(working).as_posix(),
                "sha256": ending["sha256"],
                "clip_path": Path(ending["clip_path"]).relative_to(working).as_posix(),
                "clip_sha256": ending["clip_sha256"],
                "normalization": ending["normalization"],
                "normalization_inputs": ending["normalization_inputs"],
                "frame_type": "rendered_end",
                "mode": "reuse",
            }
            (target_dir / "first_frame_selector_output.json").write_text(
                json.dumps({"continuity_source": target_provenance}), encoding="utf-8",
            )
            target_clip = target_dir / "selected-video" / "video.mp4"
            target_clip.parent.mkdir(parents=True)
            target_clip.write_bytes(b"historical target clip")
            first_lock = _lock(working, target_first)
            source_lock = _lock(working, clip)
            target_clip_lock = _lock(working, target_clip)
            (working / "render_acceptance.json").write_text(json.dumps({
                "script2video": {
                    "shots": {
                        "1": {"clips": {"accepted_at": "reviewed", "artifacts": [source_lock]}},
                        "2": {
                            "keyframes": {"accepted_at": "reviewed", "artifacts": [first_lock]},
                            "clips": {"accepted_at": "reviewed", "artifacts": [target_clip_lock]},
                        },
                    },
                },
            }), encoding="utf-8")
            images = MagicMock(model="test-image")

            refusal = await _rendered_ending_preflight_refusal(
                working, "script2video", "stills", images, ["2"],
            )
            self.assertIsNone(refusal)

            _write_clip(clip, (20, 20, 230))
            refusal = await _rendered_ending_preflight_refusal(
                working, "script2video", "stills", images, ["2"],
            )
            self.assertEqual(refusal["error_type"], "rendered_ending_source_unavailable")
            updated = json.loads((working / "render_acceptance.json").read_text())["script2video"]["shots"]["2"]
            self.assertTrue(updated["keyframes"].get("invalidated_at"))
            self.assertTrue(updated["clips"].get("invalidated_at"))

    async def test_explicit_false_skips_adapter_preflight_for_missing_predecessor_clip(self):
        with tempfile.TemporaryDirectory() as tmp:
            working = Path(tmp)
            root = working / "script2video"
            root.mkdir()
            (working / "frame_continuity.json").write_text(
                json.dumps({"mode": "chained_keyframes"}), encoding="utf-8",
            )
            (root / "camera_tree.json").write_text(
                json.dumps([{"idx": 0, "active_shot_idxs": [0, 1]}]), encoding="utf-8",
            )
            source_dir = root / "shots" / "0"
            target_dir = root / "shots" / "1"
            source_dir.mkdir(parents=True)
            target_dir.mkdir(parents=True)
            (source_dir / "shot_description.json").write_text(
                json.dumps({"idx": 0}), encoding="utf-8",
            )
            (target_dir / "shot_description.json").write_text(
                json.dumps({"idx": 1, "reuse_previous_last_frame": False}), encoding="utf-8",
            )

            refusal = await _rendered_ending_preflight_refusal(
                working, "script2video", "stills", MagicMock(model="test-image"), ["1"],
            )

            self.assertIsNone(refusal)
