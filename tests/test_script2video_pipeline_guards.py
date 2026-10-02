import asyncio
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
            pipeline = Script2VideoPipeline(chat_model=object(), image_generator=object(), video_generator=object(), working_dir=tmp)
            pipeline.camera_image_generator = FlakyCameraImageGenerator()

            async def design_storyboard(script, characters, user_requirement, quiet=False, clip_seconds=None):
                return [{"idx": 0, "is_last": True, "cam_idx": 3, "visual_desc": "wide shot", "audio_desc": "waves"}]

            async def decompose_visual_descriptions(shot_brief_descriptions, characters, quiet=False):
                return [{"idx": 0, "is_last": True, "cam_idx": 3, "visual_desc": "wide shot", "variation_type": "small", "variation_reason": "simple", "ff_desc": "start", "ff_vis_char_idxs": [], "lf_desc": "end", "lf_vis_char_idxs": [], "motion_desc": "walk", "audio_desc": "waves"}]

            pipeline.design_storyboard = design_storyboard
            pipeline.decompose_visual_descriptions = decompose_visual_descriptions
            events = []
            result = await pipeline.plan_text_artifacts(
                "script",
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

    async def test_frame_failure_names_the_shot_and_frame_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(Path(tmp) / "shots" / "13")
            pipeline = Script2VideoPipeline(chat_model=object(), image_generator=FailingImageGenerator(), video_generator=object(), working_dir=tmp)
            pipeline.reference_image_selector = MagicMock(select_reference_images_and_generate_prompt=AsyncMock(
                return_value={"reference_image_path_and_text_pairs": [], "text_prompt": "a prompt"}
            ))

            with self.assertRaises(RuntimeError) as caught:
                await pipeline.generate_frame_for_single_shot(
                    shot_idx=13,
                    frame_type="first_frame",
                    first_shot_ff_path_and_text_pair=("reference.png", "a reference"),
                    frame_desc="the frame",
                    visible_characters=[],
                    character_portraits_registry={},
                )

            self.assertIn("first_frame of shot 13", str(caught.exception))
            self.assertIn("filtered", str(caught.exception))


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
            pipeline = Script2VideoPipeline(chat_model=object(), image_generator=object(), video_generator=object(), working_dir=tmp)
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


class SceneContextTests(unittest.TestCase):
    """A close-up must not inherit a portrait's plain studio background."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.working_dir = Path(self.tmp.name)
        self.pipeline = Script2VideoPipeline.__new__(Script2VideoPipeline)
        self.pipeline.working_dir = str(self.working_dir)
        self.pipeline.image_generator = type("Generator", (), {"model": "qwen/qwen-image-3"})()

    def write_frame(self, shot_idx):
        path = Path(self.pipeline.frame_path(shot_idx, "first_frame"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x89PNG\r\n\x1a\n")
        return str(path)

    def portrait_pair(self):
        return ("/session/script2video/character_portraits/qwen_qwen-image-3/1_Wife/front.png", "A front view portrait of Wife.")

    def test_an_earlier_keyframe_is_added_when_only_portraits_were_selected(self):
        earlier = self.write_frame(3)
        pairs, prompt = self.pipeline._add_scene_context(4, [self.portrait_pair()], "Use Image 0 for her appearance.")

        self.assertEqual([path for path, _ in pairs], [self.portrait_pair()[0], earlier])
        self.assertIn("Image 1 shows the setting", prompt)
        self.assertIn("never their plain studio background", prompt)

    def test_the_nearest_earlier_keyframe_that_exists_is_used(self):
        self.write_frame(1)
        nearest = self.write_frame(3)
        pairs, _ = self.pipeline._add_scene_context(4, [self.portrait_pair()], "prompt")

        self.assertEqual(pairs[-1][0], nearest)

    def test_a_selected_scene_frame_leaves_the_prompt_alone(self):
        pairs = [self.portrait_pair(), (self.write_frame(1), "The living room.")]
        returned_pairs, prompt = self.pipeline._add_scene_context(4, pairs, "unchanged")

        self.assertEqual(returned_pairs, pairs)
        self.assertEqual(prompt, "unchanged")

    def test_a_first_shot_keeps_the_instruction_without_a_scene_frame(self):
        returned_pairs, prompt = self.pipeline._add_scene_context(0, [self.portrait_pair()], "prompt")

        self.assertEqual(returned_pairs, [self.portrait_pair()])
        self.assertIn("render the setting described above, not their plain studio background", prompt)


class RecordedPromptTests(unittest.IsolatedAsyncioTestCase):
    """What the render sent is recorded, and a resumed run reuses it unchanged."""

    async def test_a_recorded_prompt_is_reused_rather_than_rebuilt(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = Script2VideoPipeline.__new__(Script2VideoPipeline)
            pipeline.working_dir = tmp
            generator = MagicMock(model="qwen/qwen-image-3")
            generator.generate_single_image = AsyncMock(return_value=MagicMock())
            pipeline.image_generator = generator

            reference = os.path.join(tmp, "character_portraits", "qwen_qwen-image-3", "0_Cat", "front.png")
            os.makedirs(os.path.dirname(reference), exist_ok=True)
            open(reference, "wb").close()
            os.makedirs(os.path.join(tmp, "shots", "0"), exist_ok=True)
            cache_path = os.path.join(tmp, "shots", "0", "first_frame_selector_output.json")
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({
                    "reference_image_path_and_text_pairs": [[reference, "A front view portrait of Cat."]],
                    "text_prompt": "the selector prompt",
                    "sent_prompt": "the prompt actually sent, Image 0 for appearance",
                }, f)

            events = {"first_frame": asyncio.Event()}
            events["first_frame"].set()
            pipeline.frame_events = {0: events}

            await pipeline.generate_frame_for_single_shot(
                shot_idx=0,
                frame_type="first_frame",
                first_shot_ff_path_and_text_pair=(reference, "desc"),
                frame_desc="a frame",
                visible_characters=[],
                character_portraits_registry={},
            )

            sent = generator.generate_single_image.await_args.kwargs["prompt"]
            self.assertEqual(sent, "Image 0: A front view portrait of Cat.\n\nthe prompt actually sent, Image 0 for appearance")
            with open(cache_path, encoding="utf-8") as f:
                self.assertEqual(json.load(f)["sent_prompt"], "the prompt actually sent, Image 0 for appearance")


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


class PortraitReferenceTextTests(unittest.TestCase):
    """A portrait is offered to the selector in the words the frame uses for that character."""

    def test_the_characters_features_are_offered_beside_their_name(self):
        pipeline = Script2VideoPipeline.__new__(Script2VideoPipeline)
        character = CharacterInScene(
            idx=2, identifier_in_scene="DeepSeek", is_visible=True,
            static_features="A younger, fitter man with short black hair and rectangular glasses.",
            dynamic_features="Wearing a t-shirt with a purple whale on it; carrying a laptop.",
        )

        text = pipeline._portrait_reference_text(character, {"path": "x", "description": "A front view portrait of DeepSeek."})

        # The frame that introduces him says "the younger, fitter man with short black hair and
        # rectangular glasses (wearing a t-shirt with a purple whale ...)": the features are the
        # only words the two descriptions share, and without them the selector drops the portrait.
        self.assertIn("A front view portrait of DeepSeek.", text)
        self.assertIn("A younger, fitter man with short black hair and rectangular glasses.", text)
        self.assertIn("Wearing a t-shirt with a purple whale on it; carrying a laptop.", text)

    def test_a_character_without_features_keeps_its_description(self):
        pipeline = Script2VideoPipeline.__new__(Script2VideoPipeline)
        character = CharacterInScene(idx=0, identifier_in_scene="Claude", is_visible=True, static_features="", dynamic_features="")

        self.assertEqual(pipeline._portrait_reference_text(character, {"path": "x", "description": "A front view portrait of Claude."}), "A front view portrait of Claude.")


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
            images = ControlledImages()
            pipeline = Script2VideoPipeline(object(), images, BracketVideo(), tmp)
            pipeline.revision_notes = {}
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
                [pipeline.frame_path(0, "last_frame")],
            )
            self.assertEqual(images.calls[0]["reference_image_paths"], [pipeline.frame_path(0, "first_frame")])
            self.assertTrue(all(len(call["reference_image_paths"]) <= 3 for call in images.calls))
            selector = json.loads(Path(pipeline.selector_output_path(2, "first_frame")).read_text())
            self.assertEqual(
                selector["continuity_source"],
                {
                    "shot_idx": 0,
                    "frame_type": "last_frame",
                    "sha256": pipeline._sha256_file(pipeline.frame_path(0, "last_frame")),
                    "mode": "reframe",
                },
            )

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

    async def test_scoped_redraw_preserves_future_media_and_rejects_invalidated_predecessor(self):
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
            pipeline = Script2VideoPipeline(object(), Images(), Video(), str(root))
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
            (project / "render_acceptance.json").write_text(json.dumps({
                "script2video": {"shots": {"15": {"keyframes": {"invalidated_at": "changed"}}}},
            }))
            with self.assertRaisesRegex(RuntimeError, "invalidated keyframes"):
                await pipeline.generate_chained_frames(
                    camera_tree=cameras, shot_descriptions=shots, characters=[],
                    character_portraits_registry={}, only_shots=[2],
                )

    async def test_offscreen_name_does_not_override_declared_frame_visibility(self):
        class Images:
            async def generate_single_image(self, **kwargs):
                return Image.new("RGB", (2, 2), "blue")

        with tempfile.TemporaryDirectory() as tmp:
            pipeline = Script2VideoPipeline(object(), Images(), object(), tmp)
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
                [str(source_path), registry["Claude"]["front"]["path"]],
            )


if __name__ == "__main__":
    unittest.main()
