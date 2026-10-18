import contextlib
import io
import json
import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from PIL import Image

from interfaces import CharacterInEvent, CharacterInNovel, CharacterInScene, Event, Scene
from interfaces.environment import EnvironmentInScene
from agent_runtime.session_index import SessionIndex
from agent_runtime.tools import ToolRuntimeContext
from agent_runtime.vimax_adapters import ViMaxAdapters, _run_planning_step
from agents.global_information_planner import GlobalInformationPlanner, MergeCharactersAcrossScenesInEventResponse
from pipelines.novel2movie_pipeline import Novel2MoviePipeline


class FakeCompressor:
    def split(self, novel_text):
        return [novel_text]

    async def compress_single_novel_chunk(self, semaphore, index, novel_chunk):
        return index, f"compressed {novel_chunk}"

    def aggregate(self, chunks):
        return "\n".join(chunks)


class FakeEventExtractor:
    def extract_next_event(self, novel_text, extracted_events):
        return Event(index=len(extracted_events), is_last=True, description="Hero leaves home", process_chain=["Hero opens the door"])


class FakeKnowledgeBase:
    def similarity_search(self, process, k=10):
        return [SimpleNamespace(page_content="Hero opens the old wooden door.")]


class FakeReranker:
    async def __call__(self, documents, query, top_n):
        return [(documents[0], 0.95)]


class FakeSceneExtractor:
    async def get_next_scene(self, relevant_chunks, event, previous_scenes):
        return Scene(
            idx=len(previous_scenes),
            is_last=True,
            environment=EnvironmentInScene(slugline="INT. HOUSE - DAY", description="A quiet room."),
            characters=[CharacterInScene(idx=0, identifier_in_scene="Hero", is_visible=True, static_features="adult", dynamic_features="coat")],
            script="<Hero> opens the door.",
        )


class FakeGlobalPlanner:
    async def merge_characters_across_scenes_in_event(self, event_idx, scenes):
        return [CharacterInEvent(index=0, identifier_in_event="Hero", active_scenes={0: "Hero"}, static_features="adult")]

    def merge_characters_to_existing_characters_in_novel(self, event_idx, existing_characters_in_novel, characters_in_event):
        return [CharacterInNovel(index=0, identifier_in_novel="Hero", active_events={event_idx: "Hero"}, static_features="adult")]

class FakeSetPlanModel:
    async def ainvoke(self, prompt):
        return SimpleNamespace(content=json.dumps({"sets": [{
            "id": "house",
            "name": "House room",
            "description": "A quiet room with an old wooden door, pale plaster walls, and a low table.",
            "shots": ["videos/event_0/scene_0/0"],
        }]}))



class FakeNovelPipeline:
    def __init__(self, working_dir: Path):
        self.working_dir = working_dir

    async def plan_text_artifacts(self, novel_text, user_requirement="", style="", progress=None, quiet=False):
        if progress:
            for stage in ["save_novel", "compress_novel", "extract_events", "retrieve_chunks", "extract_scenes", "merge_characters", "completed"]:
                progress(stage, stage, {})
        novel = self.working_dir / "novel"
        novel.mkdir(parents=True, exist_ok=True)
        (novel / "novel.txt").write_text(novel_text, encoding="utf-8")
        (novel / "novel_compressed.txt").write_text("compressed", encoding="utf-8")
        events = self.working_dir / "events"
        events.mkdir(parents=True, exist_ok=True)
        (events / "event_0.json").write_text(json.dumps(Event(index=0, is_last=True, description="d", process_chain=["p"]).model_dump()), encoding="utf-8")
        chunks = self.working_dir / "relevant_chunks" / "event_0"
        chunks.mkdir(parents=True, exist_ok=True)
        (chunks / "chunk_0-score_0.95.txt").write_text("chunk", encoding="utf-8")
        scenes = self.working_dir / "scenes" / "event_0"
        scenes.mkdir(parents=True, exist_ok=True)
        scene = Scene(idx=0, is_last=True, environment=EnvironmentInScene(slugline="INT. ROOM - DAY", description="room"), characters=[CharacterInScene(idx=0, identifier_in_scene="Hero", is_visible=True, static_features="adult", dynamic_features="coat")], script="<Hero> walks.")
        (scenes / "scene_0.json").write_text(json.dumps(scene.model_dump()), encoding="utf-8")
        event_level = self.working_dir / "global_information" / "characters" / "event_level"
        event_level.mkdir(parents=True, exist_ok=True)
        event_char = CharacterInEvent(index=0, identifier_in_event="Hero", active_scenes={0: "Hero"}, static_features="adult")
        (event_level / "event_0_characters.json").write_text(json.dumps([event_char.model_dump()]), encoding="utf-8")
        novel_level = self.working_dir / "global_information" / "characters" / "novel_level"
        novel_level.mkdir(parents=True, exist_ok=True)
        novel_char = CharacterInNovel(index=0, identifier_in_novel="Hero", active_events={0: "Hero"}, static_features="adult")
        (novel_level / "novel_characters_after_event_0.json").write_text(json.dumps([novel_char.model_dump()]), encoding="utf-8")
        return {}

    async def plan_scene_artifacts(self, chat_model, **kwargs):
        shot = self.working_dir / "videos" / "event_0" / "scene_0" / "shots" / "0"
        shot.mkdir(parents=True, exist_ok=True)
        (shot / "shot_description.json").write_text(json.dumps({
            "idx": 0,
            "visual_desc": "An old wooden door opens into a quiet room.",
            "ff_desc": "A quiet room with an old wooden door.",
            "lf_desc": "The same quiet room and door.",
        }), encoding="utf-8")
        return ["videos/event_0/scene_0/0"]




class FakeMergeChain:
    async def ainvoke(self, messages):
        return MergeCharactersAcrossScenesInEventResponse(
            characters=[CharacterInEvent(index=0, identifier_in_event="Hero", active_scenes={0: "Hero"}, static_features="adult")]
        )


class FakeMergeChatModel:
    def __or__(self, parser):
        return FakeMergeChain()


class GlobalInformationPlannerCompatibilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_merge_event_characters_uses_scene_character_idx(self):
        planner = GlobalInformationPlanner.__new__(GlobalInformationPlanner)
        planner.chat_model = FakeMergeChatModel()
        scene = Scene(
            idx=0,
            is_last=True,
            environment=EnvironmentInScene(slugline="INT. ROOM - DAY", description="room"),
            characters=[CharacterInScene(idx=0, identifier_in_scene="Hero", is_visible=True, static_features="adult", dynamic_features="coat")],
            script="<Hero> walks.",
        )
        characters = await planner.merge_characters_across_scenes_in_event(event_idx=0, scenes=[scene])
        self.assertEqual(characters[0].identifier_in_event, "Hero")


class PlanningStepOutputSuppressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_planning_step_suppresses_stdout_stderr_and_warnings(self):
        async def noisy_step():
            print("NOISE_STDOUT")
            logging.warning("NOISE_WARNING")
            return "ok"

        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = await _run_planning_step("message", "stage", noisy_step(), runtime=None)
        self.assertEqual(result, "ok")
        self.assertNotIn("NOISE_STDOUT", stdout.getvalue())
        self.assertNotIn("NOISE_WARNING", stderr.getvalue())


class Novel2MoviePlanningTests(unittest.IsolatedAsyncioTestCase):
    async def test_plan_text_artifacts_writes_structured_text_and_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = Novel2MoviePipeline(
                novel_compressor=FakeCompressor(),
                event_extractor=FakeEventExtractor(),
                embeddings=SimpleNamespace(model="fake-embedding"),
                rerank_model=FakeReranker(),
                scene_extractor=FakeSceneExtractor(),
                global_information_planner=FakeGlobalPlanner(),
                image_generator=object(),
                rewriter=object(),
                script2video_pipeline=object(),
                working_dir=tmp,
            )
            events = []
            with patch("pipelines.novel2movie_pipeline.CacheBackedEmbeddings.from_bytes_store", return_value=object()), \
                 patch("pipelines.novel2movie_pipeline.FAISS.from_texts", return_value=FakeKnowledgeBase()):
                result = await pipeline.plan_text_artifacts("Hero opens a door.", progress=lambda stage, message, metadata=None: events.append(stage), quiet=True)
            root = Path(tmp)
            self.assertTrue((root / "novel" / "novel_compressed.txt").exists())
            self.assertTrue((root / "events" / "event_0.json").exists())
            self.assertTrue((root / "relevant_chunks" / "event_0" / "chunk_0-score_0.95.txt").exists())
            self.assertTrue((root / "scenes" / "event_0" / "scene_0.json").exists())
            self.assertTrue((root / "global_information" / "characters" / "novel_level" / "novel_characters_after_event_0.json").exists())
            self.assertFalse((root / "character_portraits").exists())
            self.assertFalse((root / "videos").exists())

class FakeSetImageGenerator:
    model = "test-image-model"

    def __init__(self):
        self.calls = []

    async def generate_single_image(self, **kwargs):
        self.calls.append(kwargs)
        return Image.new("RGB", (1, 1), "white")


class NovelSetPhaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_media_phases_refuse_missing_scene_plans_without_paid_planning(self):
        with tempfile.TemporaryDirectory() as tmp:
            working = Path(tmp)
            write_minimal_novel_artifacts(working)
            model = SimpleNamespace(ainvoke=AsyncMock())
            images = FakeSetImageGenerator()
            pipeline = Novel2MoviePipeline.__new__(Novel2MoviePipeline)
            pipeline.working_dir = working / "novel2video"
            pipeline.image_generator = images
            pipeline.script2video_pipeline = SimpleNamespace(chat_model=model)
            for phase in ("sets", "stills", "video"):
                with self.subTest(phase=phase), self.assertRaises(RuntimeError):
                    await pipeline.render_video_artifacts(style="noir", stop_after=phase, set_ids=["house"] if phase == "sets" else None)
            model.ainvoke.assert_not_awaited()
            self.assertEqual(images.calls, [])

    async def test_sets_phase_generates_only_the_selected_location_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            working = Path(tmp)
            write_minimal_novel_artifacts(working)
            render_root = working / "novel2video"
            shot = render_root / "videos/event_0/scene_0/shots/0"
            shot.mkdir(parents=True)
            (shot / "shot_description.json").write_text(json.dumps({
                "idx": 0, "visual_desc": "An old wooden door opens into a quiet room.",
            }), encoding="utf-8")
            (shot.parent.parent / "storyboard.json").write_text("[]", encoding="utf-8")
            (shot.parent.parent / "camera_tree.json").write_text("[]", encoding="utf-8")
            set_dir = render_root / "set_references/house"
            set_dir.mkdir(parents=True)
            (set_dir / "set.json").write_text(json.dumps({
                "id": "house",
                "name": "House room",
                "description": "A quiet room with an old wooden door and pale plaster walls.",
                "shots": ["videos/event_0/scene_0/0"],
            }), encoding="utf-8")
            image_generator = FakeSetImageGenerator()
            pipeline = Novel2MoviePipeline.__new__(Novel2MoviePipeline)
            pipeline.working_dir = render_root
            pipeline.image_generator = image_generator
            pipeline.script2video_pipeline = SimpleNamespace(
                chat_model=object(),
                video_generator=SimpleNamespace(model="test-video-model"),
            )
            progress = []

            outcome = await pipeline.render_video_artifacts(
                style="noir",
                stop_after="sets",
                set_ids=["house"],
                progress=lambda stage, message, metadata=None: progress.append(stage),
            )

            self.assertEqual(outcome.phase, "sets")
            self.assertEqual(outcome.awaiting_confirmation, "stills")
            self.assertEqual(len(outcome.set_references), 1)
            self.assertEqual(len(image_generator.calls), 1)
            self.assertEqual(image_generator.calls[0]["reference_image_paths"], [])
            self.assertFalse((render_root / "character_portraits").exists())
            self.assertFalse(any(render_root.glob("videos/**/qwen_*/first_frame.png")))
            self.assertFalse((render_root / "videos/event_0/scene_0/final_video.mp4").exists())



class FakeNovelRenderPipeline:

    def __init__(self, working_dir: Path):
        self.working_dir = Path(working_dir)
        self.render_calls = 0

    async def plan_scene_artifacts(self, chat_model, **kwargs):
        shot = self.working_dir / "videos" / "event_0" / "scene_0" / "shots" / "0"
        shot.mkdir(parents=True, exist_ok=True)
        (shot / "shot_description.json").write_text('{"idx":0,"visual_desc":"room"}', encoding="utf-8")
        return ["videos/event_0/scene_0/0"]

    async def render_video_artifacts(self, style, user_requirement="", progress=None, quiet=False, stop_after="portraits", **kwargs):
        self.render_calls += 1
        return {"phase": stop_after, "scene_video_dirs": [], "scene_count": 0}

def write_minimal_novel_artifacts(root: Path):
    novel = root / "novel2video"
    (novel / "novel").mkdir(parents=True, exist_ok=True)
    (novel / "novel" / "novel.txt").write_text("novel", encoding="utf-8")
    (novel / "novel" / "novel_compressed.txt").write_text("compressed", encoding="utf-8")
    events = novel / "events"
    events.mkdir(parents=True, exist_ok=True)
    (events / "event_0.json").write_text(json.dumps(Event(index=0, is_last=True, description="d", process_chain=["p"]).model_dump()), encoding="utf-8")
    chunks = novel / "relevant_chunks" / "event_0"
    chunks.mkdir(parents=True, exist_ok=True)
    (chunks / "chunk_0-score_0.95.txt").write_text("chunk", encoding="utf-8")
    scenes = novel / "scenes" / "event_0"
    scenes.mkdir(parents=True, exist_ok=True)
    scene = Scene(idx=0, is_last=True, environment=EnvironmentInScene(slugline="INT. ROOM - DAY", description="room"), characters=[CharacterInScene(idx=0, identifier_in_scene="Hero", is_visible=True, static_features="adult", dynamic_features="coat")], script="<Hero> walks.")
    (scenes / "scene_0.json").write_text(json.dumps(scene.model_dump()), encoding="utf-8")
    event_level = novel / "global_information" / "characters" / "event_level"
    event_level.mkdir(parents=True, exist_ok=True)
    event_char = CharacterInEvent(index=0, identifier_in_event="Hero", active_scenes={0: "Hero"}, static_features="adult")
    (event_level / "event_0_characters.json").write_text(json.dumps([event_char.model_dump()]), encoding="utf-8")
    novel_level = novel / "global_information" / "characters" / "novel_level"
    novel_level.mkdir(parents=True, exist_ok=True)
    novel_char = CharacterInNovel(index=0, identifier_in_novel="Hero", active_events={0: "Hero"}, static_features="adult")
    (novel_level / "novel_characters_after_event_0.json").write_text(json.dumps([novel_char.model_dump()]), encoding="utf-8")


class NovelAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_novel_initializes_named_empty_active_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            empty = index.create(project_name="Novel draft")
            adapter = ViMaxAdapters(Path(tmp), index)
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=FakeSetPlanModel()), \
                 patch("agent_runtime.vimax_adapters._build_novel_pipeline", side_effect=lambda working_dir: FakeNovelPipeline(Path(working_dir))):
                result = await adapter.vimax_novel_planning({"novel_text": "Hero opens a door."})
            self.assertTrue(result.ok)
            payload = json.loads(result.content)
            self.assertEqual(payload["session_id"], empty["session_id"])
            self.assertEqual(index.active()["project_name"], "Novel draft")
            self.assertEqual(len(index.load()["sessions"]), 1)

    async def test_missing_rag_config_returns_tool_error_and_marks_session_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            adapter = ViMaxAdapters(Path(tmp), index)
            with patch.dict("os.environ", {"VIMAX_LLM_API_KEY": "llm-key", "VIMAX_LLM_BASE_URL": "https://llm.test/v1"}, clear=True), \
                 patch("agent_runtime.vimax_adapters._build_embedding_model", side_effect=RuntimeError("embedding config missing")):
                result = await adapter.vimax_novel_planning({"novel_text": "Hero opens a door."})
            self.assertFalse(result.ok)
            self.assertIn("embedding", result.content.lower())
            self.assertEqual(index.active()["stage"], "error")

    async def test_success_writes_novel2video_artifacts_and_marks_novel_planned(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            adapter = ViMaxAdapters(Path(tmp), index)
            progress_events = []
            runtime = ToolRuntimeContext("vimax_novel_planning", "vimax_novel_planning", turn_id="turn-test", progress_callback=progress_events.append)
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=FakeSetPlanModel()), \
                 patch("agent_runtime.vimax_adapters._build_novel_pipeline", side_effect=lambda working_dir: FakeNovelPipeline(Path(working_dir))):
                result = await adapter.vimax_novel_planning({"novel_text": "Hero opens a door.", "style": "noir"}, runtime)
            self.assertTrue(result.ok)
            payload = json.loads(result.content)
            root = Path(tmp) / payload["working_dir"]
            self.assertTrue((root / "novel2video" / "novel" / "novel_compressed.txt").exists())
            self.assertTrue((root / "novel2video" / "events" / "event_0.json").exists())
            self.assertIn("novel2video/scenes/event_*/scene_*.json", payload["generated"])
            self.assertEqual(payload["sets"][0]["shots"], ["videos/event_0/scene_0/0"])
            self.assertFalse(payload["ready_for_scene_render"])
            self.assertEqual(index.active()["stage"], "novel_planned")
            stages = [event["progress"]["stage"] for event in progress_events if event.get("type") == "tool_progress"]
            self.assertIn("novel_plan_text_artifacts", stages)
            self.assertIn("merge_characters", stages)
            self.assertIn("novel_plan_set_specs", stages)

    async def test_novel_shot_render_waits_for_approved_sets_even_when_unlocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            record = index.create(idea="novel", user_requirement="scene render", style="noir")
            root = Path(tmp) / record["working_dir"]
            write_minimal_novel_artifacts(root)
            adapter = ViMaxAdapters(Path(tmp), index)
            runtime = ToolRuntimeContext("vimax_render_video", "vimax_render_video", turn_id="turn-test")
            render_pipeline = FakeNovelRenderPipeline(root / "novel2video")
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=FakeSetPlanModel()), \
                 patch("agent_runtime.vimax_adapters._build_image_generator", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_video_generator", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_novel_render_pipeline", return_value=render_pipeline):
                result = await adapter.vimax_render_video({
                    "stop_after": "video", "render_mode": "novel2video", "allow_unlocked": True,
                }, runtime)

            self.assertFalse(result.ok)
            self.assertEqual(result.metadata["error_type"], "set_approval_required")
            self.assertEqual(render_pipeline.render_calls, 0)
            self.assertTrue((root / "novel2video/videos/event_0/scene_0/shots/0/shot_description.json").exists())
            self.assertFalse((root / "novel2video/character_portraits").exists())
            self.assertFalse((root / "novel2video/videos/event_0/scene_0/final_video.mp4").exists())


if __name__ == "__main__":
    unittest.main()
