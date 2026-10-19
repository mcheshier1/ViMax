import asyncio
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from agents.character_portraits_generator import CharacterPortraitsGenerator
from agents.reference_image_selector import RefImageIndicesAndTextPrompt, ReferenceImageSelector
from agents.storyboard_artist import StoryboardArtist
from interfaces import Camera, CharacterInScene, ShotBriefDescription, ShotDescription
from PIL import Image
from pipelines.script2video_pipeline import Script2VideoPipeline, _group_shots_into_cameras
from pipelines.set_references import canonical_render_root
from utils.text import safe_path_component


def approve_sets(working_dir, specs, *, style="western", image_model="unknown-image-model"):
    root = canonical_render_root(working_dir)
    session_dir = root.parent
    for spec in specs.values():
        for slot in spec["shots"]:
            parts = slot.split("/")
            if root.name == "idea2video" and len(parts) == 2:
                shot_dir = root / parts[0] / "shots" / parts[1]
            elif root.name == "novel2video" and len(parts) == 4:
                shot_dir = root / "videos" / parts[1] / parts[2] / "shots" / parts[3]
            else:
                shot_dir = root / "shots" / parts[-1]
            shot_dir.mkdir(parents=True, exist_ok=True)
            (shot_dir / "shot_description.json").write_text(
                json.dumps({"visual_desc": f"shot {slot}", "ff_desc": "frame"}),
                encoding="utf-8",
            )

    acceptance_path = session_dir / "render_acceptance.json"
    try:
        acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        acceptance = {}
    root_record = acceptance.setdefault(root.name, {})
    set_records = root_record.setdefault("sets", {})
    references = {}
    for set_id, fields in specs.items():
        descriptor = {
            "id": set_id,
            "name": fields["name"],
            "description": fields["description"],
            "shots": fields["shots"],
        }
        descriptor_path = root / "set_references" / set_id / "set.json"
        descriptor_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor_bytes = json.dumps(descriptor, ensure_ascii=False, indent=2).encode("utf-8")
        descriptor_path.write_bytes(descriptor_bytes)
        spec_hash = hashlib.sha256(descriptor_bytes).hexdigest()
        image_path = descriptor_path.parent / safe_path_component(image_model) / "reference.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (2, 2), "tan").save(image_path)
        provenance_path = image_path.with_name("provenance.json")
        provenance_path.write_text(json.dumps({
            "image_model": image_model,
            "style": style,
            "spec_sha256": spec_hash,
        }), encoding="utf-8")
        artifacts = []
        for path in (descriptor_path, image_path, provenance_path):
            raw = path.read_bytes()
            artifacts.append({
                "path": path.relative_to(session_dir).as_posix(),
                "size": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            })
        set_records[set_id] = {"accepted_at": "approved", "artifacts": artifacts}
        references[set_id] = str(image_path)
    session_dir.mkdir(parents=True, exist_ok=True)
    acceptance_path.write_text(json.dumps(acceptance), encoding="utf-8")
    return references


class FailingImageGenerator:
    """Stand-in for a provider rejecting a request (content filter, bad option)."""

    async def generate_single_image(self, **kwargs):
        raise RuntimeError("OpenRouter image generation with model meta/muse-image failed with HTTP 400: filtered")


class FlakyCameraImageGenerator:
    def __init__(self):
        self.calls = 0

    async def construct_camera_tree(self, cameras, shot_descs):
        self.calls += 1
        if self.calls == 1:
            return ["not-a-camera"]
        return cameras


class Script2VideoPipelineGuardTests(unittest.IsolatedAsyncioTestCase):
    def test_group_shots_into_cameras_does_not_use_camera_idx_as_list_index(self):
        shots = [
            ShotDescription(idx=0, is_last=False, cam_idx=2, visual_desc="a", variation_type="small", variation_reason="same", ff_desc="a", ff_vis_char_idxs=[], lf_desc="a", lf_vis_char_idxs=[], motion_desc="a", audio_desc="none"),
            ShotDescription(idx=1, is_last=True, cam_idx=5, visual_desc="b", variation_type="small", variation_reason="same", ff_desc="b", ff_vis_char_idxs=[], lf_desc="b", lf_vis_char_idxs=[], motion_desc="b", audio_desc="none"),
            ShotDescription(idx=2, is_last=True, cam_idx=2, visual_desc="c", variation_type="small", variation_reason="same", ff_desc="c", ff_vis_char_idxs=[], lf_desc="c", lf_vis_char_idxs=[], motion_desc="c", audio_desc="none"),
        ]
        cameras = _group_shots_into_cameras(shots)
        self.assertEqual([camera.idx for camera in cameras], [2, 5])
        self.assertEqual(cameras[0].active_shot_idxs, [0, 2])
        self.assertEqual(cameras[1].active_shot_idxs, [1])

    async def test_plan_text_artifacts_retries_bad_camera_tree_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            class SetPlannerChat:
                def __init__(self):
                    self.prompt = ""

                async def ainvoke(self, prompt):
                    self.prompt = prompt
                    return type("Response", (), {"content": json.dumps({"sets": [{
                        "id": "saloon",
                        "name": "Saloon",
                        "description": "Character-free western saloon interior, bar along the north wall.",
                        "shots": ["0"],
                    }]})})()

            chat_model = SetPlannerChat()
            pipeline = Script2VideoPipeline(chat_model=chat_model, image_generator=object(), video_generator=object(), working_dir=tmp)
            pipeline.camera_image_generator = FlakyCameraImageGenerator()

            async def design_storyboard(script, characters, user_requirement, quiet=False, clip_seconds=None):
                return [{"idx": 0, "is_last": True, "cam_idx": 3, "visual_desc": "wide shot", "audio_desc": "waves"}]

            async def decompose_visual_descriptions(shot_brief_descriptions, characters, quiet=False):
                shot = {"idx": 0, "is_last": True, "cam_idx": 3, "visual_desc": "wide shot", "variation_type": "small", "variation_reason": "simple", "ff_desc": "start", "ff_vis_char_idxs": [], "lf_desc": "end", "lf_vis_char_idxs": [], "motion_desc": "walk", "audio_desc": "waves"}
                shot_dir = Path(tmp) / "shots" / "0"
                shot_dir.mkdir(parents=True, exist_ok=True)
                (shot_dir / "shot_description.json").write_text(json.dumps(shot), encoding="utf-8")
                return [shot]

            pipeline.design_storyboard = design_storyboard
            pipeline.decompose_visual_descriptions = decompose_visual_descriptions
            events = []
            result = await pipeline.plan_text_artifacts(
                "script source",
                "req",
                "style",
                characters=[{"idx": 0, "identifier_in_scene": "Man", "is_visible": True, "static_features": "adult", "dynamic_features": "coat"}],
                progress=lambda stage, message, metadata=None: events.append(stage),
                quiet=True,
            )

            self.assertEqual(pipeline.camera_image_generator.calls, 2)
            self.assertIn("construct_camera_tree_retry", events)
            self.assertEqual(result["camera_tree"][0].idx, 3)
            self.assertTrue((Path(tmp) / "camera_tree.json").exists())
            self.assertTrue((Path(tmp) / "set_references" / "saloon" / "set.json").exists())
            self.assertIn("script source", chat_model.prompt)

    async def test_frame_failure_names_the_shot_and_frame_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            working_dir = Path(tmp) / "project" / "script2video"
            working_dir.mkdir(parents=True)
            approve_sets(working_dir, {"saloon": {
                "name": "Saloon",
                "description": "Character-free western saloon interior.",
                "shots": ["13"],
            }})
            source = working_dir / "reference.png"
            Image.new("RGB", (2, 2), "tan").save(source)
            pipeline = Script2VideoPipeline(chat_model=object(), image_generator=FailingImageGenerator(), video_generator=object(), working_dir=str(working_dir))
            pipeline._active_style = "western"
            pipeline.reference_image_selector = MagicMock(select_reference_images_and_generate_prompt=AsyncMock(
                return_value={"reference_image_path_and_text_pairs": [], "text_prompt": "a prompt"}
            ))

            with self.assertRaises(RuntimeError) as caught:
                await pipeline.generate_frame_for_single_shot(
                    shot_idx=13,
                    frame_type="first_frame",
                    source_frame_pair=(str(source), "a scene source"),
                    frame_desc="the frame",
                    visible_characters=[],
                    character_portraits_registry={},
                )

            self.assertIn("first_frame of shot 13", str(caught.exception))
            self.assertIn("filtered", str(caught.exception))
    async def test_accepted_location_binding_is_first_capped_and_cached_without_other_sets(self):
        class Images:
            model = "test-image-model"
            reference_limit = 3

            def __init__(self):
                self.calls = []

            async def generate_single_image(self, **kwargs):
                self.calls.append(kwargs)
                return Image.new("RGB", (2, 2), "blue")

        with tempfile.TemporaryDirectory() as tmp:
            working_dir = Path(tmp) / "project" / "script2video"
            working_dir.mkdir(parents=True)
            set_paths = approve_sets(working_dir, {
                "saloon": {
                    "name": "Saloon",
                    "description": "Character-free western saloon interior with a bar on the north wall.",
                    "shots": ["0"],
                },
                "street": {
                    "name": "Street",
                    "description": "Character-free dusty street exterior with a boardwalk.",
                    "shots": ["1"],
                },
            }, image_model="test-image-model")
            source = working_dir / "source.png"
            portrait = working_dir / "bartender.png"
            Image.new("RGB", (2, 2), "tan").save(source)
            Image.new("RGB", (2, 2), "white").save(portrait)
            character = CharacterInScene(
                idx=0,
                identifier_in_scene="Bartender",
                is_visible=True,
                static_features="",
                dynamic_features="",
            )
            images = Images()
            pipeline = Script2VideoPipeline(object(), images, object(), str(working_dir))
            pipeline._active_style = "western"
            pipeline.revision_notes = {}
            pipeline.frame_events = {0: {kind: asyncio.Event() for kind in ("first_frame", "last_frame")}}
            selector = MagicMock(select_reference_images_and_generate_prompt=AsyncMock(return_value={
                "reference_image_path_and_text_pairs": [(str(portrait), "Bartender's portrait.")],
                "selected_indices": [0],
                "text_prompt": "Use Image 0 for the bartender while keeping the described saloon layout.",
            }))
            pipeline.reference_image_selector = selector

            await pipeline.generate_frame_for_single_shot(
                shot_idx=0,
                frame_type="first_frame",
                source_frame_pair=(str(source), "Mandatory scene source."),
                frame_desc="The bartender stands behind the saloon bar.",
                visible_characters=[character],
                character_portraits_registry={"Bartender": {"front": {"path": str(portrait)}}},
            )

            call = images.calls[0]
            self.assertEqual(
                call["reference_image_paths"],
                [set_paths["saloon"], str(source), str(portrait)],
            )
            self.assertNotIn(set_paths["street"], call["reference_image_paths"])
            self.assertIn("Image 0: Approved set reference: Saloon.", call["prompt"])
            self.assertIn("Image 1: Mandatory scene source.", call["prompt"])
            self.assertIn("Image 2", call["prompt"])
            cache = json.loads(Path(pipeline.selector_output_path(0, "first_frame")).read_text())
            self.assertEqual(
                [pair[0] for pair in cache["reference_image_path_and_text_pairs"]],
                call["reference_image_paths"],
            )
            self.assertEqual(cache["mandatory_references"][0][0], set_paths["saloon"])

            Path(pipeline.frame_path(0, "first_frame")).unlink()
            descriptor_path = working_dir / "set_references" / "saloon" / "set.json"
            descriptor = json.loads(descriptor_path.read_text())
            descriptor["description"] += " with a changed layout."
            descriptor_path.write_text(json.dumps(descriptor, ensure_ascii=False, indent=2), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "provenance.*stale"):
                await pipeline.generate_frame_for_single_shot(
                    shot_idx=0,
                    frame_type="first_frame",
                    source_frame_pair=(str(source), "Mandatory scene source."),
                    frame_desc="The bartender stands behind the saloon bar.",
                    visible_characters=[character],
                    character_portraits_registry={"Bartender": {"front": {"path": str(portrait)}}},
                )
            self.assertEqual(len(images.calls), 1)
            selector.select_reference_images_and_generate_prompt.assert_awaited_once()

    async def test_mandatory_set_and_scene_references_fail_before_the_image_post_if_they_do_not_fit(self):
        class Images:
            model = "test-image-model"
            reference_limit = 1

            def __init__(self):
                self.calls = []

            async def generate_single_image(self, **kwargs):
                self.calls.append(kwargs)
                return Image.new("RGB", (2, 2), "blue")

        with tempfile.TemporaryDirectory() as tmp:
            working_dir = Path(tmp) / "project" / "script2video"
            working_dir.mkdir(parents=True)
            approve_sets(working_dir, {"saloon": {
                "name": "Saloon",
                "description": "Character-free western saloon interior.",
                "shots": ["0"],
            }}, image_model="test-image-model")
            source = working_dir / "source.png"
            Image.new("RGB", (2, 2), "tan").save(source)
            images = Images()
            pipeline = Script2VideoPipeline(object(), images, object(), str(working_dir))
            pipeline._active_style = "western"
            selector = MagicMock(select_reference_images_and_generate_prompt=AsyncMock())
            pipeline.reference_image_selector = selector

            with self.assertRaisesRegex(RuntimeError, "needs 2 mandatory set/scene references"):
                await pipeline.generate_frame_for_single_shot(
                    shot_idx=0,
                    frame_type="first_frame",
                    source_frame_pair=(str(source), "Mandatory scene source."),
                    frame_desc="A saloon frame.",
                    visible_characters=[],
                    character_portraits_registry={},
                )

            self.assertEqual(images.calls, [])
            selector.select_reference_images_and_generate_prompt.assert_not_awaited()

    async def test_sets_phase_does_not_generate_portraits_or_probe_video_catalog(self):
        class Images:
            model = "new-image-model"

            def __init__(self):
                self.calls = []

            async def generate_single_image(self, **kwargs):
                self.calls.append(kwargs)
                return Image.new("RGB", (2, 2), "tan")

        class UnusedVideo:
            async def supports_last_frame(self):
                raise AssertionError("sets phase must not query the video catalog")

        with tempfile.TemporaryDirectory() as tmp:
            working_dir = Path(tmp) / "project" / "script2video"
            working_dir.mkdir(parents=True)
            original_references = approve_sets(working_dir, {"saloon": {
                "name": "Saloon",
                "description": "Character-free western saloon interior.",
                "shots": ["0"],
            }}, image_model="old-image-model")
            canonical_before = Path(original_references["saloon"]).read_bytes()
            acceptance_path = working_dir.parent / "render_acceptance.json"
            acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
            acceptance["script2video"].pop("sets")
            acceptance_path.write_text(json.dumps(acceptance), encoding="utf-8")
            images = Images()
            pipeline = Script2VideoPipeline(object(), images, UnusedVideo(), str(working_dir))
            pipeline.generate_character_portraits = AsyncMock(
                side_effect=AssertionError("sets phase must not generate portraits")
            )

            outcome = await pipeline(
                script="source script",
                user_requirement="",
                style="western",
                stop_after="sets",
                set_ids=["saloon"],
            )

            self.assertEqual(outcome.phase, "sets")
            self.assertEqual(outcome.stills, [])
            self.assertEqual(len(outcome.set_references), 1)
            candidate = Path(outcome.set_references[0])
            expected_directory = working_dir / "set_references" / "saloon" / safe_path_component("new-image-model")
            self.assertEqual(candidate.parent, expected_directory)
            self.assertRegex(candidate.name, r"candidate_[0-9a-f]{32}\.png")
            self.assertTrue(candidate.with_suffix(".json").exists())
            self.assertEqual(Path(original_references["saloon"]).read_bytes(), canonical_before)
            self.assertEqual(len(images.calls), 1)
            self.assertIn("No people, characters", images.calls[0]["prompt"])
            pipeline.generate_character_portraits.assert_not_awaited()

    async def test_script_render_preflights_every_missing_shot_before_any_image_post(self):
        class Images:
            model = "test-image-model"
            reference_limit = 4

            def __init__(self):
                self.calls = []

            async def prepare(self):
                return None

            async def generate_single_image(self, **kwargs):
                self.calls.append(kwargs)
                return Image.new("RGB", (2, 2), "blue")

        class Video:
            async def supports_last_frame(self):
                return False

        with tempfile.TemporaryDirectory() as tmp:
            working_dir = Path(tmp) / "project" / "script2video"
            working_dir.mkdir(parents=True)
            approve_sets(working_dir, {"saloon": {
                "name": "Saloon",
                "description": "Character-free western saloon interior.",
                "shots": ["0"],
            }}, image_model="test-image-model")
            shot_zero, shot_one = self._shot_description(0), self._shot_description(1)
            shot_one_dir = working_dir / "shots" / "1"
            shot_one_dir.mkdir(parents=True)
            (shot_one_dir / "shot_description.json").write_text(
                json.dumps(shot_one.model_dump()), encoding="utf-8"
            )
            images = Images()
            pipeline = Script2VideoPipeline(object(), images, Video(), str(working_dir))
            pipeline.design_storyboard = AsyncMock(return_value=[])
            pipeline.decompose_visual_descriptions = AsyncMock(return_value=[shot_zero, shot_one])
            pipeline.construct_camera_tree = AsyncMock(return_value=[
                Camera(idx=0, active_shot_idxs=[0]),
                Camera(idx=1, active_shot_idxs=[1]),
            ])

            with self.assertRaisesRegex(RuntimeError, "Assign a set reference"):
                await pipeline(
                    script="source script",
                    user_requirement="",
                    style="western",
                    characters=[],
                    stop_after="stills",
                )

            self.assertEqual(images.calls, [])


    def test_invalid_shot_set_overrides_block_frame_preflight_before_image_requests(self):
        class Images:
            model = "test-image-model"
            reference_limit = 4

        for set_ids, error in (
            (["missing"], "unknown set references"),
            (["saloon", "saloon"], "duplicates"),
        ):
            with self.subTest(set_ids=set_ids), tempfile.TemporaryDirectory() as tmp:
                working_dir = Path(tmp) / "project" / "script2video"
                working_dir.mkdir(parents=True)
                approve_sets(working_dir, {"saloon": {
                    "name": "Saloon",
                    "description": "Character-free western saloon interior.",
                    "shots": ["0"],
                }}, image_model="test-image-model")
                description_path = working_dir / "shots" / "0" / "shot_description.json"
                description = json.loads(description_path.read_text(encoding="utf-8"))
                description["set_ids"] = set_ids
                description_path.write_text(json.dumps(description), encoding="utf-8")
                images = Images()
                images.generate_single_image = AsyncMock()
                pipeline = Script2VideoPipeline(object(), images, object(), str(working_dir))
                pipeline._active_style = "western"
                pipeline.frame_bracketing = False

                with self.assertRaisesRegex(RuntimeError, error):
                    pipeline._preflight_pending_frame_sets(
                        [Camera(idx=0, active_shot_idxs=[0])],
                        [self._shot_description(0)],
                    )

                images.generate_single_image.assert_not_awaited()



    async def test_a_camera_whose_shots_have_all_left_is_skipped(self):
        from interfaces.camera import Camera

        with tempfile.TemporaryDirectory() as tmp:
            pipeline = Script2VideoPipeline(chat_model=object(), image_generator=object(), video_generator=object(), working_dir=tmp)
            camera = Camera(idx=1, active_shot_idxs=[], parent_shot_idx=0)

            # Nothing to draw and nothing to play. Indexing the empty list would raise, which is
            # why a camera could not be emptied; now it is simply skipped.
            result = await pipeline.generate_frames_for_single_camera(
                camera=camera,
                shot_descriptions=[],
                characters=[],
                character_portraits_registry={},
                priority_shot_idxs=[],
            )

            self.assertIsNone(result)


    def _shot_description(self, idx):
        return ShotDescription(
            idx=idx,
            is_last=True,
            cam_idx=idx,
            visual_desc=f"shot {idx}",
            variation_type="small",
            variation_reason="",
            ff_desc=f"first {idx}",
            ff_vis_char_idxs=[],
            lf_desc=f"last {idx}",
            lf_vis_char_idxs=[],
            motion_desc="move",
            audio_desc="none",
        )

    async def test_scoped_child_redraw_uses_a_cached_parent_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            working_dir = Path(tmp) / "project" / "script2video"
            working_dir.mkdir(parents=True)
            approve_sets(working_dir, {"saloon": {
                "name": "Saloon",
                "description": "Character-free western saloon interior.",
                "shots": ["1"],
            }})
            pipeline = Script2VideoPipeline(chat_model=object(), image_generator=object(), video_generator=object(), working_dir=str(working_dir))
            pipeline._active_style = "western"
            pipeline.camera_image_generator.set_reference_context(str(working_dir), "western")
            pipeline.revision_notes = {}
            pipeline.frame_bracketing = True
            pipeline.frame_events = {
                idx: {kind: asyncio.Event() for kind in ("first_frame", "last_frame")}
                for idx in (0, 1)
            }
            parent_frame = pipeline.frame_path(0, "first_frame")
            os.makedirs(os.path.dirname(parent_frame), exist_ok=True)
            Path(parent_frame).write_bytes(b"cached parent frame")

            child_video_dir = Path(pipeline.shot_video_dir(1))
            child_video_dir.mkdir(parents=True, exist_ok=True)
            (child_video_dir / "transition_video_from_shot_0.mp4").write_bytes(b"cached transition")
            Image.new("RGB", (16, 9), "red").save(child_video_dir / "new_camera_1.png")
            child_last_frame = pipeline.frame_path(1, "last_frame")
            os.makedirs(os.path.dirname(child_last_frame), exist_ok=True)
            Path(child_last_frame).write_bytes(b"cached child end frame")

            await asyncio.wait_for(
                pipeline.generate_frames_for_single_camera(
                    camera=Camera(idx=1, active_shot_idxs=[1], parent_shot_idx=0),
                    shot_descriptions=[self._shot_description(0), self._shot_description(1)],
                    characters=[],
                    character_portraits_registry={},
                    priority_shot_idxs=[],
                    only_shots=[1],
                ),
                timeout=1,
            )

            self.assertTrue(os.path.exists(pipeline.frame_path(1, "first_frame")))
            self.assertTrue(pipeline.frame_events[0]["first_frame"].is_set())
            self.assertTrue(pipeline.frame_events[1]["first_frame"].is_set())

    async def test_scoped_child_with_missing_excluded_parent_fails_instead_of_waiting(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = Script2VideoPipeline(chat_model=object(), image_generator=object(), video_generator=object(), working_dir=tmp)
            pipeline.revision_notes = {}
            pipeline.frame_bracketing = True
            pipeline.frame_events = {
                idx: {kind: asyncio.Event() for kind in ("first_frame", "last_frame")}
                for idx in (0, 1)
            }

            with self.assertRaisesRegex(RuntimeError, "required parent frame"):
                await asyncio.wait_for(
                    pipeline.generate_frames_for_single_camera(
                        camera=Camera(idx=1, active_shot_idxs=[1], parent_shot_idx=0),
                        shot_descriptions=[self._shot_description(0), self._shot_description(1)],
                        characters=[],
                        character_portraits_registry={},
                        priority_shot_idxs=[],
                        only_shots=[1],
                    ),
                    timeout=1,
                )


def test_a_character_named_only_in_the_frame_description_is_offered():
    import tempfile
    from types import SimpleNamespace

    from pipelines.script2video_pipeline import Script2VideoPipeline

    with tempfile.TemporaryDirectory() as tmp:
        pipeline = Script2VideoPipeline(chat_model=object(), image_generator=object(), video_generator=object(), working_dir=tmp)
        characters = [
            SimpleNamespace(idx=0, identifier_in_scene='Claude'),
            SimpleNamespace(idx=1, identifier_in_scene='Wife'),
            SimpleNamespace(idx=2, identifier_in_scene='DeepSeek'),
        ]
        # The plan lists who was already in the room; the last frame names a newcomer in prose,
        # with no <Name> tag and no chip. Drawn without his portrait, he is invented.
        shot = SimpleNamespace(
            visual_desc='In the living room, <Claude> sits.',
            ff_desc='Claude on the couch.',
            lf_desc='Claude maintains a shocked expression facing Wife and Deepseek, who are in the shot.',
            ff_vis_char_idxs=[0, 1],
            lf_vis_char_idxs=[0, 1],
        )

        offered = [character.identifier_in_scene for character in pipeline._visible_characters(shot, characters, "last_frame")]

        assert offered == ['Claude', 'Wife', 'DeepSeek']


def test_a_character_only_the_brief_names_is_not_offered():
    """The brief describes the whole shot, so by its end it names who has left the room."""
    import tempfile
    from types import SimpleNamespace

    from pipelines.script2video_pipeline import Script2VideoPipeline

    with tempfile.TemporaryDirectory() as tmp:
        pipeline = Script2VideoPipeline(chat_model=object(), image_generator=object(), video_generator=object(), working_dir=tmp)
        characters = [
            SimpleNamespace(idx=0, identifier_in_scene='Claude'),
            SimpleNamespace(idx=1, identifier_in_scene='Wife'),
        ]
        shot = SimpleNamespace(
            visual_desc='A wide shot shows <Wife> walking away, while <Claude> sits defeated on the couch.',
            ff_desc='Medium shot. Claude sits alone on the couch. Only Claude is in the room.',
            lf_desc='Medium shot. Claude leans forward as the room fades. Only Claude is in the room.',
            ff_vis_char_idxs=[0],
            lf_vis_char_idxs=[0],
        )

        for frame in ('first_frame', 'last_frame'):
            offered = [character.identifier_in_scene for character in pipeline._visible_characters(shot, characters, frame)]
            assert offered == ['Claude'], (frame, offered)


class SelectorCacheTests(unittest.TestCase):
    """A cached per-frame prompt is only reusable while its references still exist."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.working_dir = Path(self.tmp.name)
        self.pipeline = Script2VideoPipeline.__new__(Script2VideoPipeline)
        self.pipeline.working_dir = str(self.working_dir)

    def write_cache(self, payload):
        path = Path(self.pipeline.selector_output_path(0, "first_frame"))
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(payload, str):
            path.write_text(payload, encoding="utf-8")
        else:
            path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_reuses_a_cache_whose_references_exist(self):
        reference = self.working_dir / "character_portraits" / "qwen_qwen-image-3" / "0_Claude" / "front.png"
        reference.parent.mkdir(parents=True, exist_ok=True)
        reference.write_text("png", encoding="utf-8")
        self.write_cache({"reference_image_path_and_text_pairs": [[str(reference), "A front view portrait of Claude."]], "text_prompt": "prompt"})

        cached = self.pipeline.load_selector_output(0, "first_frame")

        self.assertIsNotNone(cached)
        self.assertEqual(cached["text_prompt"], "prompt")

    def test_discards_a_cache_whose_reference_was_deleted_by_a_style_change(self):
        # A style change deletes the portraits a stale prompt points at; reusing
        # it dies inside the image generator with a missing file.
        self.write_cache({"reference_image_path_and_text_pairs": [[str(self.working_dir / "character_portraits" / "0_Claude" / "front.png"), "A front view portrait of Claude."]], "text_prompt": "prompt"})

        self.assertIsNone(self.pipeline.load_selector_output(0, "first_frame"))

    def test_discards_a_missing_or_malformed_cache(self):
        self.assertIsNone(self.pipeline.load_selector_output(0, "first_frame"))

        self.write_cache("{not json")
        self.assertIsNone(self.pipeline.load_selector_output(0, "first_frame"))

        self.write_cache({"text_prompt": "prompt"})
        self.assertIsNone(self.pipeline.load_selector_output(0, "first_frame"))

    def test_reuses_a_cache_with_no_references(self):
        self.write_cache({"reference_image_path_and_text_pairs": [], "text_prompt": "prompt"})

        self.assertIsNotNone(self.pipeline.load_selector_output(0, "first_frame"))


class ReferenceImageSelectorLimitTests(unittest.IsolatedAsyncioTestCase):
    """The selection must fit what the image model accepts, or the frame is refused."""

    def setUp(self):
        # The selector embeds the references themselves, so they have to exist.
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.pairs = []
        for index in range(4):
            path = Path(self.tmp.name) / f"{index}.png"
            path.write_bytes(b"\x89PNG\r\n\x1a\n")
            self.pairs.append((str(path), f"description {index}"))

    def _selector(self, indices):
        response = RefImageIndicesAndTextPrompt(ref_image_indices=indices, text_prompt="Use Image 0 for Claude.")

        class StubChain:
            async def ainvoke(self, messages):
                return response

        class StubChatModel:
            def __or__(self, parser):
                return StubChain()

        return ReferenceImageSelector(chat_model=StubChatModel())

    async def test_selection_is_capped_and_keeps_the_prompt_order(self):
        selector = self._selector([0, 1, 2, 3])

        result = await selector.select_reference_images_and_generate_prompt(self.pairs, "a frame", max_reference_images=3)

        # The tail goes, so the kept references still match the prompt's "Image N".
        self.assertEqual([path for path, _ in result["reference_image_path_and_text_pairs"]], [path for path, _ in self.pairs[:3]])

    async def test_selection_keeps_every_reference_when_no_limit_is_known(self):
        selector = self._selector([0, 1, 2, 3])

        result = await selector.select_reference_images_and_generate_prompt(self.pairs, "a frame")

        self.assertEqual(len(result["reference_image_path_and_text_pairs"]), 4)


class ReferenceMentionAlignmentTests(unittest.TestCase):
    """The prompt's `Image N` mentions must address the images actually sent."""

    def setUp(self):
        self.pipeline = Script2VideoPipeline.__new__(Script2VideoPipeline)
        self.pipeline.image_generator = type("Generator", (), {"model": "qwen/qwen-image-3", "reference_limit": 3})()

    def test_mentions_are_renumbered_to_the_sent_positions(self):
        # The selector returned available images 0 and 3 but numbered the second
        # mention 3 — the index it saw — as one real session's prompts did.
        available = [(f"/ref/{index}.png", f"description {index}") for index in range(6)]
        selected = [available[0], available[3]]

        pairs, prompt = self.pipeline._align_reference_mentions(available, selected, [0, 3], "Use Image 0 for her appearance and Image 3 for the setting.")

        self.assertEqual(pairs, selected)
        self.assertEqual(prompt, "Use Image 0 for her appearance and Image 1 for the setting.")

    def test_a_mention_by_the_selectors_own_numbering_is_not_read_as_the_shown_list(self):
        # Slot 18 of the live session: the list the selector is shown carries the portraits
        # first and the shot's setting last, and it picked the setting as its only reference
        # and wrote "Image 0" for the room, couch and environment. Read against the shown
        # list, Image 0 is Claude's portrait, so the room was handed a portrait to copy.
        available = [
            ("/ref/claude-front.png", "A front view portrait of Claude."),
            ("/ref/claude-side.png", "A side view portrait of Claude."),
            ("/ref/room.png", "Medium shot in the living room."),
        ]
        selected = [available[2]]

        pairs, prompt = self.pipeline._align_reference_mentions(
            available,
            selected,
            [2],
            "The scene, room environment, couch, and Claude's appearance should reference Image 0.",
        )

        self.assertEqual(pairs, selected)
        self.assertEqual(prompt, "The scene, room environment, couch, and Claude's appearance should reference Image 0.")

    def test_an_image_the_prompt_asks_for_is_added_when_there_is_room(self):
        available = [(f"/ref/{index}.png", f"description {index}") for index in range(4)]
        selected = [available[0]]

        pairs, prompt = self.pipeline._align_reference_mentions(available, selected, [0], "Reference Image 2 for the room.")

        self.assertEqual([path for path, _ in pairs], ["/ref/0.png", "/ref/2.png"])
        self.assertEqual(prompt, "Reference Image 1 for the room.")

    def test_the_model_limit_is_respected(self):
        available = [(f"/ref/{index}.png", f"description {index}") for index in range(4)]
        selected = [available[0], available[1], available[2]]

        pairs, prompt = self.pipeline._align_reference_mentions(available, selected, [0, 1, 2], "Reference Image 3 for the room.")

        self.assertEqual(len(pairs), 3)
        self.assertEqual(prompt, "Reference Image 3 for the room.")

    def test_an_out_of_range_mention_is_left_alone(self):
        available = [(f"/ref/{index}.png", f"description {index}") for index in range(2)]
        selected = [available[0]]

        pairs, prompt = self.pipeline._align_reference_mentions(available, selected, [0], "Reference Image 9.")

        self.assertEqual(pairs, selected)
        self.assertEqual(prompt, "Reference Image 9.")

    def test_a_cache_without_selected_indices_is_used_as_is(self):
        available = [(f"/ref/{index}.png", f"description {index}") for index in range(2)]
        selected = [available[0]]

        pairs, prompt = self.pipeline._align_reference_mentions(available, selected, [], "Use Image 3.")

        self.assertEqual(pairs, selected)
        self.assertEqual(prompt, "Use Image 3.")


class StoryboardRuntimeTests(unittest.IsolatedAsyncioTestCase):
    """A plan cannot meet a runtime it cannot see."""

    def _artist(self, captured):
        class StubChain:
            async def ainvoke(self, messages):
                captured["messages"] = messages
                return type("Response", (), {"storyboard": []})()

        class StubChatModel:
            def __or__(self, parser):
                return StubChain()

        return StoryboardArtist(chat_model=StubChatModel())

    async def test_the_storyboard_is_told_the_clip_length(self):
        captured = {}
        artist = self._artist(captured)

        await artist.design_storyboard(script="script", characters=[], user_requirement="45-60 second comedic short", clip_seconds=6)

        prompt = str(captured["messages"][1][1])
        self.assertIn("6 seconds of video", prompt)
        self.assertIn("at most 10 shots per minute", prompt)

    async def test_the_pipeline_forwards_the_clip_length_to_the_artist(self):
        """The planning entry point is the only path a plan can learn the runtime through."""
        captured = {}

        class StubArtist:
            async def design_storyboard(self, *, script, characters, user_requirement, retry_timeout, clip_seconds=None):
                captured["clip_seconds"] = clip_seconds
                return []

        pipeline = Script2VideoPipeline.__new__(Script2VideoPipeline)
        pipeline.working_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(pipeline.working_dir, ignore_errors=True))
        pipeline.storyboard_artist = StubArtist()
        pipeline.video_generator = type("Generator", (), {"clip_seconds": 6})()

        await pipeline.design_storyboard(script="script", characters=[], user_requirement="45-60 second short", quiet=True)
        self.assertEqual(captured["clip_seconds"], 6)

    async def test_no_constraint_is_stated_when_the_clip_length_is_unknown(self):
        captured = {}
        artist = self._artist(captured)

        await artist.design_storyboard(script="script", characters=[], user_requirement="45-60 second comedic short")

        self.assertNotIn("seconds of video", str(captured["messages"][1][1]))


class CharacterPortraitGuaranteeTests(unittest.TestCase):
    """Whatever the selector picks, a character in the frame gets a portrait with a face."""

    def _pipeline(self, limit=3):
        pipeline = Script2VideoPipeline.__new__(Script2VideoPipeline)
        pipeline.reference_image_limit = lambda: limit
        return pipeline

    def _character(self, idx, name):
        return CharacterInScene(idx=idx, identifier_in_scene=name, is_visible=True, static_features="static", dynamic_features="dynamic")

    def _registry(self, name):
        return {name: {
            "front": {"path": f"portraits/{name}/front.png", "description": f"A front view portrait of {name}."},
            "back": {"path": f"portraits/{name}/back.png", "description": f"A back view portrait of {name}."},
        }}

    def test_a_back_portrait_is_replaced_by_the_front_one_it_cannot_replace(self):
        # Shot 8 of the live session: Claude was drawn from his back portrait, so his face was
        # invented, and the Wife's portrait was not selected at all, so her dress was too.
        pipeline = self._pipeline()
        claude = self._character(0, "Claude")
        pairs = [("shots/7/new_camera_3.png", "the camera still"), ("portraits/Claude/back.png", "A back view portrait of Claude.")]

        updated, prompt = pipeline._ensure_character_portraits([claude], self._registry("Claude"), pairs, "the frame prompt")

        paths = [path for path, _ in updated]
        self.assertIn("portraits/Claude/front.png", paths)
        self.assertNotIn("portraits/Claude/back.png", paths)
        self.assertIn("Claude", prompt)

    def test_the_prompt_says_which_image_is_which_character(self):
        pipeline = self._pipeline()
        characters = [self._character(0, "Claude"), self._character(2, "DeepSeek")]
        registry = {**self._registry("Claude"), **self._registry("DeepSeek")}
        pairs = [("shots/7/first_frame.png", "the first frame"), ("portraits/Claude/back.png", "A back view portrait of Claude.")]

        updated, prompt = pipeline._ensure_character_portraits(characters, registry, pairs, "the frame prompt")

        paths = [path for path, _ in updated]
        claude_at = paths.index("portraits/Claude/front.png")
        deepseek_at = paths.index("portraits/DeepSeek/front.png")
        self.assertIn(f"Image {claude_at} is Claude", prompt)
        self.assertIn(f"Image {deepseek_at} is DeepSeek", prompt)

    def test_a_character_with_no_portrait_selected_gets_theirs_when_there_is_room(self):
        pipeline = self._pipeline()
        wife = self._character(1, "Wife")
        pairs = [("shots/7/first_frame.png", "the first frame")]

        updated, prompt = pipeline._ensure_character_portraits([wife], self._registry("Wife"), pairs, "the frame prompt")

        self.assertIn("portraits/Wife/front.png", [path for path, _ in updated])
        self.assertIn("Image 1 is Wife", prompt)

    def test_nothing_is_added_when_the_character_is_already_there_or_the_limit_is_full(self):
        pipeline = self._pipeline(limit=1)
        claude = self._character(0, "Claude")
        already = [("portraits/Claude/front.png", "A front view portrait of Claude.")]
        self.assertEqual(pipeline._ensure_character_portraits([claude], self._registry("Claude"), already, "p")[0], already)

        full = [("a.png", "a"), ("b.png", "b")]
        self.assertEqual(pipeline._ensure_character_portraits([claude], self._registry("Claude"), full, "p")[0], full)


class ShotLookupTests(unittest.TestCase):
    """A shot is found by its number, not by where it sits in a list."""

    def _description(self, idx, desc):
        return ShotDescription(idx=idx, is_last=False, cam_idx=0, visual_desc=desc, ff_desc=f"{desc} first", lf_desc=f"{desc} last", variation_type="small", variation_reason="", ff_vis_char_idxs=[], lf_vis_char_idxs=[], motion_desc="m", audio_desc="none")

    def test_a_shot_whose_number_is_not_its_position_still_finds_its_own_plan(self):
        pipeline = Script2VideoPipeline.__new__(Script2VideoPipeline)
        # What the film looks like after shots 6 and 7 are removed: eleven descriptions,
        # numbered up to 12. Read by position, shot 7's frames would be described as shot 9's,
        # and shot 12 would run off the end of the list.
        descriptions = [self._description(idx, f"shot {idx}") for idx in (0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12)]

        by_idx = pipeline._descriptions_by_idx(descriptions)

        self.assertEqual(by_idx[7].visual_desc, "shot 7")
        self.assertEqual(by_idx[12].visual_desc, "shot 12")
        self.assertNotEqual(by_idx[7].visual_desc, descriptions[7].visual_desc)
        self.assertFalse(hasattr(by_idx.get(6, None), "visual_desc"))


class VisibleCharacterTests(unittest.TestCase):
    """A frame is offered the characters the *frame* names, whatever the plan's list says."""

    def _pipeline(self):
        return Script2VideoPipeline.__new__(Script2VideoPipeline)

    def _character(self, idx, name):
        return CharacterInScene(idx=idx, identifier_in_scene=name, is_visible=True, static_features="s", dynamic_features="d")

    def _description(self, visual_desc, ff, lf):
        # A frame's own description is what the image model is asked for, so the fixture's frames
        # carry it: the brief is the whole shot and is not read for who a frame shows.
        return ShotDescription(idx=0, is_last=False, cam_idx=0, visual_desc=visual_desc, ff_desc=visual_desc, lf_desc=visual_desc, variation_type="small", variation_reason="", ff_vis_char_idxs=ff, lf_vis_char_idxs=lf, motion_desc="m", audio_desc="none")

    def test_a_character_the_shot_names_is_offered_even_when_the_frame_list_omits_them(self):
        # Shot 8 of the live session: the plan lists Claude and Wife for the last frame while
        # the frame's description is about DeepSeek entering, so his portrait was never sent.
        characters = [self._character(0, "Claude"), self._character(1, "Wife"), self._character(2, "DeepSeek")]
        description = self._description("A shot shows <DeepSeek> entering and putting his arm around <Wife>.", [1], [0, 1])

        offered = self._pipeline()._visible_characters(description, characters, "last_frame")

        self.assertEqual([c.identifier_in_scene for c in offered], ["Claude", "Wife", "DeepSeek"])

    def test_nobody_else_is_offered_and_the_frame_list_still_leads(self):
        characters = [self._character(0, "Claude"), self._character(1, "Wife"), self._character(2, "DeepSeek")]
        description = self._description("A shot of <Wife> alone in the room.", [1], [1])

        self.assertEqual([c.identifier_in_scene for c in self._pipeline()._visible_characters(description, characters, "first_frame")], ["Wife"])
        self.assertEqual([c.identifier_in_scene for c in self._pipeline()._visible_characters(description, characters, "last_frame")], ["Wife"])

    def test_an_out_of_range_index_in_a_stored_plan_is_ignored(self):
        characters = [self._character(0, "Claude")]
        description = self._description("A shot of <Claude>.", [0, 9], [])

        self.assertEqual([c.identifier_in_scene for c in self._pipeline()._visible_characters(description, characters, "first_frame")], ["Claude"])


class ReviewNoteTests(unittest.TestCase):
    """A redraw carries the user's note into the prompt it draws from."""

    def _pipeline(self, notes):
        pipeline = Script2VideoPipeline.__new__(Script2VideoPipeline)
        pipeline.revision_notes = notes
        return pipeline

    def test_the_note_is_appended_verbatim_to_the_frame_prompt(self):
        pipeline = self._pipeline({"5": "her dress is blue here but pink in every other shot"})

        prompt = pipeline._review_note(5, "Claude sits at the kitchen table.")

        self.assertIn("Claude sits at the kitchen table.", prompt)
        self.assertIn("her dress is blue here but pink in every other shot", prompt)
        self.assertIn("<REVIEW_CORRECTION>", prompt)

    def test_a_shot_with_no_note_is_drawn_from_its_description_alone(self):
        pipeline = self._pipeline({"5": "something"})

        self.assertEqual(pipeline._review_note(6, "A wide shot of the room."), "A wide shot of the room.")

    def test_a_pipeline_that_was_never_told_about_a_redo_has_no_notes(self):
        pipeline = Script2VideoPipeline.__new__(Script2VideoPipeline)

        self.assertEqual(pipeline._review_note(0, "A shot."), "A shot.")


class PortraitPromptRecordingTests(unittest.IsolatedAsyncioTestCase):
    """A portrait is drawn from a prompt nobody else can reconstruct, so it is recorded."""

    async def test_the_generator_keeps_the_prompt_it_built(self):
        generator = CharacterPortraitsGenerator(FailingImageGenerator())
        character = CharacterInScene(idx=0, identifier_in_scene="Claude", is_visible=True, static_features="heavyset", dynamic_features="orange shirt")

        with self.assertRaises(RuntimeError):
            await generator.generate_front_portrait(character, "photorealistic cinematic live action")

        prompt = generator.prompts[("Claude", "front")]
        self.assertIn("Claude", prompt)
        self.assertIn("heavyset", prompt)
        self.assertIn("photorealistic cinematic live action", prompt)

    async def test_the_prompts_are_written_beside_the_portraits(self):
        from unittest.mock import AsyncMock, MagicMock
        import json as jsonlib

        with tempfile.TemporaryDirectory() as tmp:
            pipeline = Script2VideoPipeline.__new__(Script2VideoPipeline)
            pipeline.working_dir = tmp
            pipeline.image_generator = type("Generator", (), {"model": "qwen/qwen-image-3"})()
            generator = CharacterPortraitsGenerator(FailingImageGenerator())
            generator.prompts = {
                ("Claude", "front"): "front prompt",
                ("Claude", "side"): "side prompt",
                ("Claude", "back"): "back prompt",
            }
            for method in ("generate_front_portrait", "generate_side_portrait", "generate_back_portrait"):
                setattr(generator, method, AsyncMock(return_value=MagicMock()))
            pipeline.character_portraits_generator = generator
            pipeline.character_portrait_events = {0: asyncio.Event()}
            character = CharacterInScene(idx=0, identifier_in_scene="Claude", is_visible=True, static_features="s", dynamic_features="d")

            await pipeline.generate_portraits_for_single_character(character, "style", progress=None)

            written = jsonlib.load(open(os.path.join(pipeline.portraits_dir(), "0_Claude", "prompts.json"), encoding="utf-8"))
            self.assertEqual(written, {"front": "front prompt", "side": "side prompt", "back": "back prompt"})


class CharacterPortraitsGeneratorTests(unittest.IsolatedAsyncioTestCase):
    async def test_portrait_failure_names_the_character_and_view(self):
        generator = CharacterPortraitsGenerator(FailingImageGenerator())
        character = CharacterInScene(
            idx=0,
            identifier_in_scene="Claude",
            is_visible=True,
            static_features="a heavyset man with a round face",
            dynamic_features="wearing an orange shirt",
        )

        with self.assertRaises(RuntimeError) as front:
            await generator.generate_front_portrait(character, "style")
        self.assertIn("the front portrait of Claude", str(front.exception))

        with self.assertRaises(RuntimeError) as side:
            await generator.generate_side_portrait(character, "front.png")
        self.assertIn("the side portrait of Claude", str(side.exception))

        with self.assertRaises(RuntimeError) as back:
            await generator.generate_back_portrait(character, "front.png")
        self.assertIn("the back portrait of Claude", str(back.exception))


class ChainedKeyframeTests(unittest.IsolatedAsyncioTestCase):
    def _shot(self, idx, cam_idx):
        return ShotDescription(
            idx=idx, is_last=False, cam_idx=cam_idx, visual_desc=f"shot {idx}",
            variation_type="small", variation_reason="", ff_desc=f"first {idx}",
            ff_vis_char_idxs=[], lf_desc=f"last {idx}", lf_vis_char_idxs=[],
            motion_desc="move", audio_desc="none",
        )

    async def test_playback_order_copies_same_camera_bytes_and_reframes_from_previous_last(self):
        class BracketVideo:
            async def supports_last_frame(self):
                return True

        class ControlledImages:
            reference_limit = 3

            def __init__(self):
                self.calls = []

            async def generate_single_image(self, **kwargs):
                self.calls.append(kwargs)
                return Image.new("RGB", (2, 2), "blue")

        with tempfile.TemporaryDirectory() as tmp:
            working_dir = Path(tmp) / "project" / "script2video"
            working_dir.mkdir(parents=True)
            set_paths = approve_sets(working_dir, {"saloon": {
                "name": "Saloon",
                "description": "Character-free western saloon interior.",
                "shots": ["0", "15", "2"],
            }})
            images = ControlledImages()
            pipeline = Script2VideoPipeline(object(), images, BracketVideo(), str(working_dir))
            pipeline._active_style = "western"
            shots = [self._shot(0, 0), self._shot(15, 0), self._shot(2, 2)]
            pipeline.frame_events = {
                idx: {kind: asyncio.Event() for kind in ("first_frame", "last_frame")}
                for idx in (15, 0, 2)
            }

            async def root_frames(**kwargs):
                idx = kwargs["camera"].active_shot_idxs[0]
                for kind, content in (("first_frame", b"root-start"), ("last_frame", b"root-end")):
                    path = Path(pipeline.frame_path(idx, kind))
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(content)

            pipeline.generate_frames_for_single_camera = root_frames
            await pipeline.generate_chained_frames(
                camera_tree=[Camera(idx=0, active_shot_idxs=[15, 0]), Camera(idx=2, active_shot_idxs=[2])],
                shot_descriptions=shots,
                characters=[],
                character_portraits_registry={},
            )

            self.assertEqual(Path(pipeline.frame_path(0, "first_frame")).read_bytes(), b"root-end")
            self.assertEqual(
                images.calls[1]["reference_image_paths"],
                [set_paths["saloon"], pipeline.frame_path(0, "last_frame")],
            )
            self.assertEqual(
                images.calls[0]["reference_image_paths"],
                [set_paths["saloon"], pipeline.frame_path(0, "first_frame")],
            )
            self.assertTrue(all(len(call["reference_image_paths"]) <= 3 for call in images.calls))

    async def test_scoped_missing_predecessor_and_first_only_provider_fail_before_generation(self):
        class FirstOnlyVideo:
            async def supports_last_frame(self):
                return False

        with tempfile.TemporaryDirectory() as tmp:
            pipeline = Script2VideoPipeline(object(), object(), FirstOnlyVideo(), tmp)
            with self.assertRaisesRegex(RuntimeError, "bracketing"):
                await pipeline.generate_chained_frames(
                    camera_tree=[],
                    shot_descriptions=[],
                    characters=[],
                    character_portraits_registry={},
                )
            pipeline.video_generator = type(
                "BracketVideo", (), {"supports_last_frame": lambda self: asyncio.sleep(0, result=True)}
            )()
            shots = [self._shot(15, 0), self._shot(2, 2)]
            with self.assertRaisesRegex(RuntimeError, "predecessor shot 15.*only_shots"):
                await pipeline.generate_chained_frames(
                    camera_tree=[Camera(idx=0, active_shot_idxs=[15]), Camera(idx=2, active_shot_idxs=[2])],
                    shot_descriptions=shots,
                    characters=[],
                    character_portraits_registry={},
                    only_shots=[2],
                )

    async def test_scoped_redraw_preserves_other_media_and_uses_invalidated_saved_anchor(self):
        class Images:
            reference_limit = 3

            async def generate_single_image(self, **kwargs):
                return Image.new("RGB", (2, 2), "blue")

        class Video:
            async def supports_last_frame(self):
                return True

        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            root = project / "script2video"
            root.mkdir(parents=True)
            set_paths = approve_sets(root, {"saloon": {
                "name": "Saloon",
                "description": "Character-free western saloon interior.",
                "shots": ["0", "15", "2"],
            }})
            pipeline = Script2VideoPipeline(object(), Images(), Video(), str(root))
            pipeline._active_style = "western"
            pipeline.revision_notes = {}
            shots = [self._shot(0, 0), self._shot(15, 0), self._shot(2, 2)]
            cameras = [Camera(idx=0, active_shot_idxs=[0, 15]), Camera(idx=2, active_shot_idxs=[2])]
            preserved = {}
            for idx in (0, 2):
                for kind in ("first_frame", "last_frame"):
                    path = Path(pipeline.frame_path(idx, kind))
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (2, 2), "red").save(path)
                    preserved[path] = path.read_bytes()
            await pipeline.generate_chained_frames(
                camera_tree=cameras, shot_descriptions=shots, characters=[],
                character_portraits_registry={}, only_shots=[15],
            )
            self.assertEqual(
                Path(pipeline.frame_path(15, "first_frame")).read_bytes(),
                preserved[Path(pipeline.frame_path(0, "last_frame"))],
            )
            self.assertNotEqual(
                Path(pipeline.frame_path(15, "last_frame")).read_bytes(),
                Path(pipeline.frame_path(15, "first_frame")).read_bytes(),
            )
            for path, original in preserved.items():
                self.assertEqual(path.read_bytes(), original)
            approvals_path = project / "render_acceptance.json"
            approvals = json.loads(approvals_path.read_text())
            approvals["script2video"]["shots"] = {"15": {"keyframes": {"invalidated_at": "changed"}}}
            approvals_path.write_text(json.dumps(approvals))
            anchor = Path(pipeline.frame_path(15, "last_frame")).read_bytes()
            pipeline.revision_notes = {"2": "Redraw only this shot."}
            await pipeline.generate_chained_frames(
                camera_tree=cameras, shot_descriptions=shots, characters=[],
                character_portraits_registry={}, only_shots=[2],
            )
            source = json.loads(Path(pipeline.selector_output_path(2, "first_frame")).read_text())["continuity_source"]
            self.assertEqual(source["shot_idx"], 15)
            self.assertEqual(source["sha256"], hashlib.sha256(anchor).hexdigest())
            self.assertEqual(Path(pipeline.frame_path(15, "last_frame")).read_bytes(), anchor)

    async def test_saved_endings_anchor_sparse_scope_without_ancestor_validation(self):
        class Images:
            reference_limit = 3

            def __init__(self):
                self.calls = []

            async def generate_single_image(self, **kwargs):
                self.calls.append(kwargs)
                return Image.new("RGB", (2, 2), (len(self.calls), 0, 0))

        class Video:
            async def supports_last_frame(self):
                return True

        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            root = project / "script2video"
            root.mkdir()
            order = [0, 15, 2, 8, 7]
            approve_sets(root, {"saloon": {
                "name": "Saloon", "description": "Empty timber saloon.",
                "shots": [str(idx) for idx in order],
            }})
            approvals_path = project / "render_acceptance.json"
            approvals = json.loads(approvals_path.read_text())
            approvals["script2video"]["shots"] = {
                "15": {"keyframes": {"invalidated_at": "set changed"}},
                "2": {"keyframes": {"invalidated_at": "earlier chain changed"}},
            }
            approvals_path.write_text(json.dumps(approvals))
            images = Images()
            pipeline = Script2VideoPipeline(object(), images, Video(), str(root))
            pipeline._active_style = "western"
            pipeline.revision_notes = {}
            shots = [self._shot(idx, 0) for idx in reversed(order)]
            cameras = [Camera(idx=0, active_shot_idxs=order)]
            preserved = {}
            for idx in (2, 7):
                for kind in ("first_frame", "last_frame"):
                    if idx == 2 and kind == "first_frame":
                        continue
                    path = Path(pipeline.frame_path(idx, kind))
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (2, 2), "red").save(path)
                    preserved[path] = path.read_bytes()

            async def render(scope):
                await pipeline.generate_chained_frames(
                    camera_tree=cameras, shot_descriptions=shots, characters=[],
                    character_portraits_registry={}, only_shots=scope,
                )

            await render([8])
            source_path = Path(pipeline.frame_path(2, "last_frame"))
            self.assertEqual(Path(pipeline.frame_path(8, "first_frame")).read_bytes(), source_path.read_bytes())
            self.assertEqual(len(images.calls), 1)
            for idx in (0, 15):
                self.assertFalse(Path(pipeline.frame_path(idx, "last_frame")).exists())
            with self.assertRaisesRegex(RuntimeError, "predecessor shot 0.*missing"):
                await render([15, 8])
            self.assertEqual(len(images.calls), 1)

            # A sparse request uses the saved intervening ending, not the last
            # selected shot's newly generated ending.
            root_ending = Path(pipeline.frame_path(0, "last_frame"))
            root_ending.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (2, 2), "green").save(root_ending)
            pipeline.revision_notes = {"15": "Redraw selected shot.", "8": "Redraw selected shot."}
            await render([15, 8])
            for source_idx, target_idx in ((0, 15), (2, 8)):
                source_path = Path(pipeline.frame_path(source_idx, "last_frame"))
                self.assertEqual(Path(pipeline.frame_path(target_idx, "first_frame")).read_bytes(), source_path.read_bytes())
                provenance = json.loads(Path(pipeline.selector_output_path(target_idx, "first_frame")).read_text())["continuity_source"]
                self.assertEqual(provenance["shot_idx"], source_idx)
                self.assertEqual(provenance["sha256"], pipeline._sha256_file(str(source_path)))
            for path, content in preserved.items():
                self.assertEqual(path.read_bytes(), content)

    async def test_explicit_false_overrides_same_camera_automatic_reuse(self):
        class Images:
            reference_limit = 3

            def __init__(self):
                self.calls = []

            async def generate_single_image(self, **kwargs):
                self.calls.append(kwargs)
                return Image.new("RGB", (2, 2), "blue")

        class Video:
            async def supports_last_frame(self):
                return True

        with tempfile.TemporaryDirectory() as tmp:
            working = Path(tmp) / "project" / "script2video"
            working.mkdir(parents=True)
            approve_sets(working, {"saloon": {"name": "Saloon", "description": "Western saloon.", "shots": ["0", "1"]}})
            images = Images()
            pipeline = Script2VideoPipeline(object(), images, Video(), str(working))
            pipeline._active_style = "western"
            pipeline.revision_notes = {}
            shots = [self._shot(0, 0), self._shot(1, 0)]
            shots[1].reuse_previous_last_frame = False
            camera = Camera(idx=0, active_shot_idxs=[0, 1])

            async def root_frames(**kwargs):
                for kind, content in (("first_frame", b"root-start"), ("last_frame", b"root-end")):
                    path = Path(pipeline.frame_path(0, kind))
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(content)

            pipeline.generate_frames_for_single_camera = root_frames
            await pipeline.generate_chained_frames(
                camera_tree=[camera],
                shot_descriptions=shots,
                characters=[],
                character_portraits_registry={},
            )

            source = Path(pipeline.frame_path(0, "last_frame")).read_bytes()
            target = Path(pipeline.frame_path(1, "first_frame"))
            self.assertNotEqual(target.read_bytes(), source)
            self.assertEqual(len(images.calls), 2)
            provenance = json.loads(Path(pipeline.selector_output_path(1, "first_frame")).read_text())["continuity_source"]
            self.assertEqual(provenance["mode"], "reframe")
            self.assertEqual(provenance["slot"], "0")

    def _reuse_test_pipeline(self, working_dir, shots, camera_tree):
        class Images:
            reference_limit = None

            def __init__(self):
                self.calls = []

            async def prepare(self):
                return None

            async def generate_single_image(self, **kwargs):
                self.calls.append(kwargs)
                return Image.new("RGB", (2, 2), "blue")

        class Video:
            async def supports_last_frame(self):
                return True

        images = Images()
        pipeline = Script2VideoPipeline(object(), images, Video(), str(working_dir))
        pipeline._active_style = "western"
        approve_sets(working_dir, {"saloon": {
            "name": "Saloon", "description": "Western saloon.",
            "shots": [pipeline._shot_slot(shot.idx) for shot in shots],
        }})
        for shot in shots:
            description_path = Path(working_dir) / "shots" / str(shot.idx) / "shot_description.json"
            description_path.write_text(json.dumps(shot.model_dump()), encoding="utf-8")
        pipeline.frame_bracketing = True
        pipeline.revision_notes = {}
        pipeline.frame_events = {
            shot.idx: {kind: asyncio.Event() for kind in ("first_frame", "last_frame")}
            for shot in shots
        }
        pipeline._playback_predecessors = pipeline._build_playback_predecessors(camera_tree, shots)
        generated = []

        async def record_frame(**kwargs):
            generated.append((kwargs["shot_idx"], kwargs["frame_type"]))
            pipeline.frame_events[kwargs["shot_idx"]][kwargs["frame_type"]].set()
            return pipeline.frame_path(kwargs["shot_idx"], kwargs["frame_type"])

        pipeline.generate_frame_for_single_shot = record_frame
        return pipeline, images, generated

    async def test_copy_waits_for_selected_predecessor_redraw_instead_of_copying_old_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            working = Path(tmp) / "project" / "script2video"
            working.mkdir(parents=True)
            shots = [self._shot(15, 0), self._shot(2, 2)]
            shots[1].reuse_previous_last_frame = True
            cameras = [Camera(idx=0, active_shot_idxs=[15]), Camera(idx=2, active_shot_idxs=[2])]
            pipeline, _images, _generated = self._reuse_test_pipeline(working, shots, cameras)
            pipeline.revision_notes = {"15": "New ending"}
            source = pipeline._playback_predecessors[2]
            source_path = Path(source["path"])
            source_path.parent.mkdir(parents=True)
            source_path.write_bytes(b"old ending")
            pending = asyncio.create_task(pipeline._reuse_previous_last_frame(2, source, only_shots={15, 2}))
            await asyncio.sleep(0)
            self.assertFalse(Path(pipeline.frame_path(2, "first_frame")).exists())
            self.assertFalse(source["event"].is_set())
            source_path.write_bytes(b"new ending")
            pipeline._mark_frame_fresh(15, "last_frame")
            source["event"].set()
            await pending
            self.assertEqual(Path(pipeline.frame_path(2, "first_frame")).read_bytes(), b"new ending")

    async def test_first_frame_only_provider_can_copy_existing_ending_but_cannot_wait_for_missing_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            working = Path(tmp) / "project" / "script2video"
            working.mkdir(parents=True)
            shots = [self._shot(15, 0), self._shot(2, 2)]
            shots[1].reuse_previous_last_frame = True
            cameras = [Camera(idx=0, active_shot_idxs=[15]), Camera(idx=2, active_shot_idxs=[2])]
            pipeline, images, _generated = self._reuse_test_pipeline(working, shots, cameras)
            pipeline.frame_bracketing = False
            source = pipeline._playback_predecessors[2]
            with self.assertRaisesRegex(RuntimeError, "does not generate ending keyframes"):
                await pipeline._reuse_previous_last_frame(2, source, only_shots={15, 2})
            source_path = Path(source["path"])
            source_path.parent.mkdir(parents=True)
            source_path.write_bytes(b"existing ending")
            await pipeline._reuse_previous_last_frame(2, source, only_shots={2})
            self.assertEqual(Path(pipeline.frame_path(2, "first_frame")).read_bytes(), b"existing ending")
            self.assertEqual(images.calls, [])

    async def test_per_shot_copy_is_byte_identical_cross_camera_and_skips_first_image_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            working = Path(tmp) / "project" / "script2video"
            working.mkdir(parents=True)
            predecessor = self._shot(15, 0)
            target = self._shot(2, 2)
            target.reuse_previous_last_frame = True
            shots = [predecessor, target]
            cameras = [Camera(idx=0, active_shot_idxs=[15]), Camera(idx=2, active_shot_idxs=[2])]
            pipeline, images, generated = self._reuse_test_pipeline(working, shots, cameras)
            source = Path(pipeline.frame_path(15, "last_frame"))
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b"previous ending bytes\x00")
            pipeline.frame_events[15]["last_frame"].set()

            await pipeline.generate_frames_for_single_camera(
                camera=cameras[1], shot_descriptions=shots, characters=[],
                character_portraits_registry={}, priority_shot_idxs=[], only_shots=[2],
            )

            target_path = Path(pipeline.frame_path(2, "first_frame"))
            self.assertEqual(target_path.read_bytes(), source.read_bytes())
            self.assertEqual(images.calls, [])
            continuity = json.loads(Path(pipeline.selector_output_path(2, "first_frame")).read_text())["continuity_source"]
            self.assertEqual(continuity["slot"], "15")
            self.assertEqual(continuity["path"], source.relative_to(working.parent).as_posix())
            self.assertEqual(continuity["sha256"], pipeline._sha256_file(str(source)))

            preserved = target_path.read_bytes()
            source.write_bytes(b"changed predecessor ending")
            with self.assertRaisesRegex(RuntimeError, "stale predecessor continuity"):
                await pipeline.generate_frames_for_single_camera(
                    camera=cameras[1], shot_descriptions=shots, characters=[],
                    character_portraits_registry={}, priority_shot_idxs=[], only_shots=[2],
                )
            self.assertEqual(target_path.read_bytes(), preserved)

    async def test_per_shot_copy_refuses_missing_excluded_predecessor_before_image_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            working = Path(tmp) / "project" / "script2video"
            working.mkdir(parents=True)
            predecessor = self._shot(15, 0)
            target = self._shot(2, 2)
            target.reuse_previous_last_frame = True
            shots = [predecessor, target]
            cameras = [Camera(idx=0, active_shot_idxs=[15]), Camera(idx=2, active_shot_idxs=[2])]
            pipeline, images, generated = self._reuse_test_pipeline(working, shots, cameras)

            with self.assertRaisesRegex(RuntimeError, "predecessor shot 15.*last frame is missing"):
                await pipeline.generate_frames_for_single_camera(
                    camera=cameras[1], shot_descriptions=shots, characters=[],
                    character_portraits_registry={}, priority_shot_idxs=[], only_shots=[2],
                )

            self.assertEqual(images.calls, [])
            self.assertEqual(generated, [])

    async def test_per_shot_copy_uses_previous_scene_active_ending(self):
        with tempfile.TemporaryDirectory() as tmp:
            idea_root = Path(tmp) / "project" / "idea2video"
            previous_scene = idea_root / "scene_0"
            current_scene = idea_root / "scene_1"
            previous_scene.mkdir(parents=True)
            current_scene.mkdir(parents=True)
            previous = self._shot(15, 0)
            (previous_scene / "camera_tree.json").write_text(
                json.dumps([{"idx": 0, "active_shot_idxs": [15]}]), encoding="utf-8",
            )
            previous_description = previous_scene / "shots" / "15" / "shot_description.json"
            previous_description.parent.mkdir(parents=True)
            previous_description.write_text(json.dumps(previous.model_dump()), encoding="utf-8")
            target = self._shot(2, 0)
            target.reuse_previous_last_frame = True
            cameras = [Camera(idx=0, active_shot_idxs=[2])]
            pipeline, images, generated = self._reuse_test_pipeline(current_scene, [target], cameras)
            source = previous_scene / "shots" / "15" / pipeline.image_model_slug / "last_frame.png"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"prior scene ending")

            await pipeline.generate_frames_for_single_camera(
                camera=cameras[0], shot_descriptions=[target], characters=[],
                character_portraits_registry={}, priority_shot_idxs=[], only_shots=[2],
            )

            target_path = Path(pipeline.frame_path(2, "first_frame"))
            self.assertEqual(target_path.read_bytes(), source.read_bytes())
            self.assertEqual(images.calls, [])
            continuity = json.loads(Path(pipeline.selector_output_path(2, "first_frame")).read_text())["continuity_source"]
            self.assertEqual(continuity["slot"], "scene_0/15")
            self.assertEqual(continuity["path"], source.relative_to(idea_root.parent).as_posix())

    async def test_offscreen_name_does_not_override_declared_frame_visibility(self):
        class Images:
            async def generate_single_image(self, **kwargs):
                return Image.new("RGB", (2, 2), "blue")

        with tempfile.TemporaryDirectory() as tmp:
            working_dir = Path(tmp) / "project" / "script2video"
            working_dir.mkdir(parents=True)
            set_paths = approve_sets(working_dir, {"saloon": {
                "name": "Saloon",
                "description": "Character-free western saloon interior.",
                "shots": ["0", "2"],
            }})
            pipeline = Script2VideoPipeline(object(), Images(), object(), str(working_dir))
            pipeline._active_style = "western"
            characters = [
                CharacterInScene(idx=idx, identifier_in_scene=name, is_visible=True,
                                 static_features="approved identity", dynamic_features="same clothes")
                for idx, name in [(0, "Claude"), (2, "DeepSeek")]
            ]
            source = self._shot(0, 0)
            source.lf_vis_char_idxs = [0]
            target = self._shot(2, 2)
            target.ff_desc = "Claude sits alone. DeepSeek has not entered."
            target.ff_vis_char_idxs = [0]
            target.lf_vis_char_idxs = [0, 2]
            registry = {}
            for character in characters:
                portrait = Path(tmp) / f"{character.identifier_in_scene}.png"
                Image.new("RGB", (2, 2), "white").save(portrait)
                registry[character.identifier_in_scene] = {"front": {"path": str(portrait), "description": "portrait"}}
            source_path = Path(pipeline.frame_path(0, "last_frame"))
            source_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (2, 2), "red").save(source_path)
            pipeline.frame_events = {2: {"first_frame": asyncio.Event()}}
            await pipeline._generate_chained_image(
                target, source, str(source_path), characters, registry, frame_type="first_frame",
            )
            selector = json.loads(Path(pipeline.selector_output_path(2, "first_frame")).read_text())
            self.assertEqual(
                [pair[0] for pair in selector["reference_image_path_and_text_pairs"]],
                [set_paths["saloon"], str(source_path), registry["Claude"]["front"]["path"]],
            )

if __name__ == "__main__":
    unittest.main()
