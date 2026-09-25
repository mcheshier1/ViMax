import asyncio
import contextlib
import io
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from interfaces import Camera, CharacterInScene, ShotBriefDescription, ShotDescription
from agent_runtime.session_index import SessionIndex
from agent_runtime.vimax_adapters import ViMaxAdapters
from pipelines.render_contract import RenderOutcome
from agent_runtime.tools import ToolRuntimeContext
from pipelines.idea2video_pipeline import Idea2VideoPipeline
from pipelines.script2video_pipeline import Script2VideoPipeline


class FakeIdeaPipeline:
    def __init__(self, chat_model, image_generator, video_generator, working_dir):
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)

    async def develop_story(self, idea, user_requirement, quiet=False):
        path = self.working_dir / "story.txt"
        path.write_text("story", encoding="utf-8")
        return "story"

    async def extract_characters(self, story, quiet=False):
        chars = [CharacterInScene(idx=0, identifier_in_scene="Cat", is_visible=True, static_features="black cat", dynamic_features="helmet")]
        (self.working_dir / "characters.json").write_text(json.dumps([c.model_dump() for c in chars]), encoding="utf-8")
        return chars

    async def write_script_based_on_story(self, story, user_requirement, quiet=False):
        script = [{"scene": "cat jumps"}]
        (self.working_dir / "script.json").write_text(json.dumps(script), encoding="utf-8")
        return script




class HangingIdeaPipeline(FakeIdeaPipeline):
    async def develop_story(self, idea, user_requirement, quiet=False):
        await asyncio.sleep(10)
        return "story"



class FakeRevisionModel:
    async def ainvoke(self, prompt):
        return SimpleNamespace(content='[{"idx": 0, "description": "more oppressive"}]')


class FailRenderIdeaPipeline(FakeIdeaPipeline):
    async def __call__(self, idea, user_requirement, style, quiet=False, stop_after="portraits", revision_notes=None, progress=None):
        raise RuntimeError("render failed")


class FailRender403IdeaPipeline(FakeIdeaPipeline):
    async def __call__(self, idea, user_requirement, style, quiet=False, stop_after="portraits", revision_notes=None, progress=None):
        raise RuntimeError("OpenRouter video create failed with HTTP 403: {'error': {'message': 'Key limit exceeded (total limit). Manage it using token sk-short', 'code': 403}}")


class FailRenderContentFilterIdeaPipeline(FakeIdeaPipeline):
    async def __call__(self, idea, user_requirement, style, quiet=False, stop_after="portraits", revision_notes=None, progress=None):
        raise RuntimeError("Image generation failed for the first_frame of shot 13: OpenRouter image generation with model meta/muse-image failed with HTTP 400: {'error': {'message': 'The response was filtered due to the prompt triggering our content management policy.', 'code': 400, 'metadata': {'provider_name': 'Meta'}}}")


class FailRenderMissingReferenceIdeaPipeline(FakeIdeaPipeline):
    async def __call__(self, idea, user_requirement, style, quiet=False, stop_after="portraits", revision_notes=None, progress=None):
        raise RuntimeError("Image generation failed for the first_frame of shot 0: [Errno 2] No such file or directory: '/tmp/session/script2video/character_portraits/0_Claude/front.png'")


class NoisyRenderIdeaPipeline(FakeIdeaPipeline):
    async def __call__(self, idea, user_requirement, style, quiet=False, stop_after="portraits", revision_notes=None, progress=None):
        print("NOISE_FROM_RENDER_PIPELINE")
        final = self.working_dir / "final_video.mp4"
        final.write_text("video", encoding="utf-8")
        return RenderOutcome(phase="video", style=style, final_video_path=str(final))


class FakeScriptPipeline:
    def __init__(self, chat_model, image_generator, video_generator, working_dir):
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)

    async def plan_text_artifacts(self, script, user_requirement, style, characters=None, progress=None, quiet=False):
        if progress:
            progress("design_storyboard", "Designing storyboard", {})
            progress("decompose_shots", "Decomposing shot visual descriptions", {"shot_count": 1})
            progress("construct_camera_tree", "Constructing camera tree", {"shot_count": 1})
        (self.working_dir / "storyboard.json").write_text("[]", encoding="utf-8")
        (self.working_dir / "camera_tree.json").write_text("[]", encoding="utf-8")
        shot_dir = self.working_dir / "shots" / "0"
        shot_dir.mkdir(parents=True, exist_ok=True)
        (shot_dir / "shot_description.json").write_text("{}", encoding="utf-8")
        if characters:
            (self.working_dir / "characters.json").write_text(json.dumps([c.model_dump() for c in characters]), encoding="utf-8")
        return {}




class FailingScriptPipeline(FakeScriptPipeline):
    async def plan_text_artifacts(self, script, user_requirement, style, characters=None, progress=None, quiet=False):
        if progress:
            progress("design_storyboard", "Designing storyboard", {})
        raise RuntimeError("storyboard failed")


class FakeInitChatModel:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return object()


class Script2VideoPlanningProgressTests(unittest.IsolatedAsyncioTestCase):
    async def test_plan_text_artifacts_emits_progress_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = Script2VideoPipeline(chat_model=object(), image_generator=object(), video_generator=object(), working_dir=tmp)
            chars = [CharacterInScene(idx=0, identifier_in_scene="Cat", is_visible=True, static_features="black cat", dynamic_features="helmet")]
            storyboard = [ShotBriefDescription(idx=0, is_last=True, cam_idx=0, visual_desc="cat jumps", audio_desc="wind")]
            shot = ShotDescription(idx=0, is_last=True, cam_idx=0, visual_desc="cat jumps", variation_type="small", variation_reason="simple motion", ff_desc="cat starts", ff_vis_char_idxs=[0], lf_desc="cat lands", lf_vis_char_idxs=[0], motion_desc="cat jumps", audio_desc="wind")
            camera = [Camera(idx=0, active_shot_idxs=[0])]

            async def design_storyboard(script, characters, user_requirement, quiet=False, clip_seconds=None):
                return storyboard

            async def decompose_visual_descriptions(shot_brief_descriptions, characters, quiet=False):
                return [shot]

            async def construct_camera_tree(shot_descriptions, quiet=False):
                return camera

            pipeline.design_storyboard = design_storyboard
            pipeline.decompose_visual_descriptions = decompose_visual_descriptions
            pipeline.construct_camera_tree = construct_camera_tree
            events = []
            await pipeline.plan_text_artifacts("script", "req", "style", characters=chars, progress=lambda stage, message, metadata=None: events.append(stage))
            self.assertEqual(events, ["extract_characters", "design_storyboard", "decompose_shots", "construct_camera_tree"])


    async def test_idea_pipeline_quiet_suppresses_text_planning_prints(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = Idea2VideoPipeline(chat_model=object(), image_generator=object(), video_generator=object(), working_dir=tmp)

            async def develop_story(idea, user_requirement):
                return "story"

            pipeline.screenwriter = SimpleNamespace(develop_story=develop_story)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                result = await pipeline.develop_story("idea", "req", quiet=True)
            self.assertEqual(result, "story")
            self.assertEqual(stdout.getvalue(), "")


_IDEA_READY_KEYS = ("idea2video/story.txt", "idea2video/characters.json", "idea2video/script.json", "idea2video/scene_*/storyboard.json", "idea2video/scene_*/shots/*/shot_description.json", "idea2video/scene_*/camera_tree.json")
_SCRIPT_READY_KEYS = ("script2video/script.txt", "script2video/characters.json", "script2video/storyboard.json", "script2video/shots/*/shot_description.json", "script2video/camera_tree.json")
_NOVEL_READY_KEYS = ("novel2video/novel/novel_compressed.txt", "novel2video/events/event_*.json", "novel2video/relevant_chunks/event_*", "novel2video/scenes/event_*/scene_*.json", "novel2video/global_information/characters/event_level/*.json", "novel2video/global_information/characters/novel_level/*.json")


class RenderModeTests(unittest.TestCase):
    """A session with two complete plans must render the one the caller means."""

    def _checklist(self, idea=True, script=True, novel=False):
        keys = {}
        if idea:
            keys.update({key: True for key in _IDEA_READY_KEYS})
        if script:
            keys.update({key: True for key in _SCRIPT_READY_KEYS})
        if novel:
            keys.update({key: True for key in _NOVEL_READY_KEYS})
        return keys

    def test_an_explicit_mode_wins_over_the_first_ready_root(self):
        from agent_runtime.vimax_adapters import _resolve_render_mode

        self.assertEqual(_resolve_render_mode(self._checklist(), {}, "script2video"), "script2video")

    def test_the_pinned_root_wins_when_no_mode_is_given(self):
        from agent_runtime.vimax_adapters import _resolve_render_mode

        self.assertEqual(_resolve_render_mode(self._checklist(), {"render_mode": "script2video"}, ""), "script2video")

    def test_the_first_ready_root_is_used_when_nothing_is_pinned(self):
        from agent_runtime.vimax_adapters import _resolve_render_mode

        self.assertEqual(_resolve_render_mode(self._checklist(), {}, ""), "idea2video")

    def test_a_mode_is_rejected_when_its_root_has_no_plan(self):
        from agent_runtime.vimax_adapters import _resolve_render_mode

        with self.assertRaises(ValueError) as caught:
            _resolve_render_mode(self._checklist(novel=False), {}, "novel2video")
        self.assertIn("no plan yet", str(caught.exception))

    def test_an_unknown_mode_is_rejected(self):
        from agent_runtime.vimax_adapters import _resolve_render_mode

        with self.assertRaises(ValueError) as caught:
            _resolve_render_mode(self._checklist(), {}, "slide2video")
        self.assertIn("render_mode must be one of", str(caught.exception))


class FailingRedrawPipeline:
    """Stands in for a redraw that dies mid-phase, as an out-of-credit render does."""

    def __init__(self, chat_model, image_generator, video_generator, working_dir):
        self.working_dir = Path(working_dir)

    async def __call__(self, script, user_requirement, style, characters=None, quiet=False, progress=None, stop_after="portraits", revision_notes=None):
        raise RuntimeError("HTTP 402: Insufficient credits")


class RecordingScriptPipeline:
    """Stands in for the script-mode render so the test can see which root ran."""

    def __init__(self, chat_model, image_generator, video_generator, working_dir):
        self.working_dir = Path(working_dir)

    last_revision_notes = None
    last_only_shots = None

    async def __call__(self, script, user_requirement, style, characters=None, quiet=False, progress=None, stop_after="portraits", revision_notes=None, only_shots=None):
        type(self).last_revision_notes = revision_notes
        type(self).last_only_shots = only_shots
        return RenderOutcome(phase=stop_after, style=style)


def va_write_render_status_recorder(seen: list) -> object:
    """Records each render-status write, so a test can see what a reader would have seen."""

    def record(working_dir, *, status, payload):
        seen.append({"status": status, "payload": dict(payload)})

    return record


def _lock(working, relative_path):
    """An acceptance entry for a file on disk, the shape the review API writes."""
    data = (working / relative_path).read_bytes()
    return {"path": relative_path, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _acceptance_session(tmp):
    """A session with planned shots, keyframes on disk and no acceptance yet."""
    index = SessionIndex(tmp)
    record = index.create(idea="x", style="cinematic")
    working = Path(tmp) / record["working_dir"]
    shots = working / "script2video" / "shots"
    for shot, files in ((0, ["first_frame.png", "last_frame.png"]), (1, ["first_frame.png"])):
        shot_dir = shots / str(shot) / "qwen_qwen-image-3"
        shot_dir.mkdir(parents=True, exist_ok=True)
        (shots / str(shot) / "shot_description.json").write_text("{}", encoding="utf-8")
        for name in files:
            (shot_dir / name).write_bytes(b"frame")
    (working / "script2video" / "final_video.mp4").write_bytes(b"film")
    return working, index


class AcceptanceGateTests(unittest.TestCase):
    """A phase may not spend money on artifacts nobody has reviewed."""

    def _session(self, tmp):
        return _acceptance_session(tmp)

    def _accept(self, working, shot, sha):
        (working / "render_acceptance.json").write_text(json.dumps({
            "script2video": {"shots": {str(shot): {"keyframes": {"accepted_at": "now", "artifacts": [{"path": f"script2video/shots/{shot}/qwen_qwen-image-3/first_frame.png", "size": 5, "sha256": sha}]}}}}
        }), encoding="utf-8")

    def test_the_gate_covers_only_the_shots_a_run_names(self):
        from agent_runtime.vimax_adapters import _acceptance_refusal

        with tempfile.TemporaryDirectory() as tmp:
            working, _ = self._session(tmp)
            self._accept(working, 0, hashlib.sha256(b"frame").hexdigest())

            # Shot 0 is accepted, so a run that names it may go. A run that names nothing is the
            # whole phase and still waits for shot 1, and so does one that names shot 1.
            self.assertIsNone(_acceptance_refusal(working, "script2video", "video", [0]))
            self.assertIsNotNone(_acceptance_refusal(working, "script2video", "video"))
            self.assertIsNotNone(_acceptance_refusal(working, "script2video", "video", [1]))

    def test_a_named_run_does_not_narrow_the_session_wide_gate(self):
        from agent_runtime.vimax_adapters import _acceptance_refusal

        with tempfile.TemporaryDirectory() as tmp:
            working, _ = self._session(tmp)
            portrait = working / "script2video" / "character_portraits" / "qwen_qwen-image-3" / "0_Claude" / "front.png"
            portrait.parent.mkdir(parents=True, exist_ok=True)
            portrait.write_bytes(b"portrait")

            # The portraits are one set per sequence, not one per shot: naming a shot for a
            # stills run cannot excuse portraits nobody has accepted.
            self.assertIsNotNone(_acceptance_refusal(working, "script2video", "stills", [0]))

    def test_keyframes_start_out_rendered_and_block_the_video_phase(self):
        from agent_runtime.vimax_adapters import _acceptance_refusal

        with tempfile.TemporaryDirectory() as tmp:
            working, _ = self._session(tmp)
            refusal = _acceptance_refusal(working, "script2video", "video")
            self.assertIsNotNone(refusal)
            self.assertEqual(refusal["error_type"], "acceptance_required")
            self.assertEqual([item["state"] for item in refusal["unaccepted"]], ["rendered", "rendered"])
            self.assertIn("Review keyframes in the Timeline", refusal["error"])

    def test_an_accepted_keyframe_slot_stops_blocking(self):
        from agent_runtime.vimax_adapters import _acceptance_refusal, _lock_matches

        with tempfile.TemporaryDirectory() as tmp:
            working, _ = self._session(tmp)
            digest = hashlib.sha256(b"frame").hexdigest()
            self._accept(working, 0, digest)
            refusal = _acceptance_refusal(working, "script2video", "video")
            self.assertEqual([item["slot"] for item in refusal["unaccepted"]], ["1"])
            self.assertTrue(_lock_matches(working, json.loads((working / "render_acceptance.json").read_text())["script2video"]["shots"]["0"]["keyframes"]))

    def test_a_regenerated_artifact_turns_an_acceptance_stale(self):
        from agent_runtime.vimax_adapters import _acceptance_refusal

        with tempfile.TemporaryDirectory() as tmp:
            working, _ = self._session(tmp)
            self._accept(working, 0, hashlib.sha256(b"frame").hexdigest())
            (working / "script2video" / "shots" / "0" / "qwen_qwen-image-3" / "first_frame.png").write_bytes(b"redrawn")
            refusal = _acceptance_refusal(working, "script2video", "video")
            self.assertIn("stale", [item["state"] for item in refusal["unaccepted"]])

    def test_a_rejected_slot_blocks_and_carries_its_reason(self):
        from agent_runtime.vimax_adapters import _acceptance_refusal, _slot_state, _acceptance_slots

        with tempfile.TemporaryDirectory() as tmp:
            working, _ = self._session(tmp)
            (working / "render_acceptance.json").write_text(json.dumps({
                "script2video": {"shots": {"1": {"keyframes": {"rejected_at": "now", "reason": "her dress is the wrong colour here"}}}}
            }), encoding="utf-8")

            refusal = _acceptance_refusal(working, "script2video", "video")

            states = {item["slot"]: item["state"] for item in refusal["unaccepted"]}
            self.assertEqual(states["1"], "rejected")
            self.assertEqual(_slot_state(working, _acceptance_slots(working, "script2video")["keyframes"]["1"]), "rejected")
            self.assertIn("rejected with a reason", refusal["error"])

    def test_a_rejection_outranks_an_acceptance_on_the_same_slot(self):
        from agent_runtime.vimax_adapters import _slot_state, _acceptance_slots

        with tempfile.TemporaryDirectory() as tmp:
            working, _ = self._session(tmp)
            (working / "render_acceptance.json").write_text(json.dumps({
                "script2video": {"shots": {"0": {"keyframes": {
                    "accepted_at": "earlier",
                    "artifacts": [{"path": "script2video/shots/0/qwen_qwen-image-3/first_frame.png", "size": 5, "sha256": hashlib.sha256(b"frame").hexdigest()}],
                    "rejected_at": "later",
                    "reason": "the backdrop is a studio white",
                }}}}
            }), encoding="utf-8")

            self.assertEqual(_slot_state(working, _acceptance_slots(working, "script2video")["keyframes"]["0"]), "rejected")

    def test_the_portraits_phase_needs_no_review(self):
        from agent_runtime.vimax_adapters import _acceptance_refusal

        with tempfile.TemporaryDirectory() as tmp:
            working, _ = self._session(tmp)
            self.assertIsNone(_acceptance_refusal(working, "script2video", "portraits"))

    def test_acceptance_is_per_root(self):
        from agent_runtime.vimax_adapters import _acceptance_refusal

        with tempfile.TemporaryDirectory() as tmp:
            working, _ = self._session(tmp)
            self.assertEqual([item["slot"] for item in _acceptance_refusal(working, "script2video", "video")["unaccepted"]], ["0", "1"])

class RedoResolutionTests(unittest.TestCase):
    """A redraw names exactly the shots it replaces, and is refused before it deletes."""

    def _session(self, tmp):
        return _acceptance_session(tmp)

    def test_no_redo_arguments_means_an_ordinary_render(self):
        from agent_runtime.vimax_adapters import _resolve_redo

        with tempfile.TemporaryDirectory() as tmp:
            working, _ = self._session(tmp)
            self.assertEqual(_resolve_redo(working, "script2video", {"stop_after": "stills"}), {})

    def test_a_named_shot_is_redrawn_from_the_stage_holding_unaccepted_artifacts(self):
        from agent_runtime.vimax_adapters import _resolve_redo

        with tempfile.TemporaryDirectory() as tmp:
            working, _ = self._session(tmp)
            self.assertEqual(_resolve_redo(working, "script2video", {"redo_shots": [1]}), {"slots": {"1": "keyframes"}, "phase": "stills"})

    def test_a_rejected_clip_is_redrawn_at_the_clip_stage(self):
        from agent_runtime.vimax_adapters import _resolve_redo

        with tempfile.TemporaryDirectory() as tmp:
            working, _ = self._session(tmp)
            clip = working / "script2video" / "shots" / "0" / "kwaivgi_kling-video-o1" / "video.mp4"
            clip.parent.mkdir(parents=True, exist_ok=True)
            clip.write_bytes(b"clip")
            (working / "render_acceptance.json").write_text(json.dumps({
                "script2video": {"shots": {"0": {
                    "keyframes": {"accepted_at": "now", "artifacts": []},
                    "clips": {"rejected_at": "later", "reason": "the ceiling sags"},
                }}}
            }), encoding="utf-8")

            resolved = _resolve_redo(working, "script2video", {"redo_rejected": True})
            self.assertEqual(resolved, {"slots": {"0": "clips"}, "phase": "video"})

    def test_keyframes_win_the_phase_when_both_stages_are_redrawn(self):
        from agent_runtime.vimax_adapters import _resolve_redo

        with tempfile.TemporaryDirectory() as tmp:
            working, _ = self._session(tmp)
            (working / "render_acceptance.json").write_text(json.dumps({
                "script2video": {"shots": {
                    "0": {"keyframes": {"rejected_at": "later", "reason": "studio white"}},
                    "1": {"keyframes": {"accepted_at": "now", "artifacts": []}, "clips": {"rejected_at": "later", "reason": "sags"}},
                }}
            }), encoding="utf-8")

            self.assertEqual(_resolve_redo(working, "script2video", {"redo_rejected": True}), {"slots": {"0": "keyframes", "1": "clips"}, "phase": "stills"})

    def test_an_unknown_shot_is_refused_with_the_ones_that_exist(self):
        from agent_runtime.vimax_adapters import _resolve_redo

        with tempfile.TemporaryDirectory() as tmp:
            working, _ = self._session(tmp)
            refused = _resolve_redo(working, "script2video", {"redo_shots": ["9"]})
            self.assertEqual(refused["error_type"], "unknown_redo_shot")
            self.assertEqual(refused["known_slots"], ["0", "1"])

    def test_redoing_without_a_rejection_says_so(self):
        from agent_runtime.vimax_adapters import _resolve_redo

        with tempfile.TemporaryDirectory() as tmp:
            working, _ = self._session(tmp)
            self.assertEqual(_resolve_redo(working, "script2video", {"redo_rejected": True})["error_type"], "nothing_to_redo")

    def test_a_shot_with_nothing_on_disk_is_a_render_rather_than_a_redraw(self):
        from agent_runtime.vimax_adapters import _resolve_redo

        with tempfile.TemporaryDirectory() as tmp:
            working, _ = self._session(tmp)
            empty = working / "script2video" / "shots" / "2"
            empty.mkdir(parents=True)
            (empty / "shot_description.json").write_text("{}", encoding="utf-8")

            # A shot nobody has drawn has nothing to clear, so asking for it asks for the work
            # that does not exist yet. It is not the mistake "nothing to redraw" guards against.
            self.assertEqual(
                _resolve_redo(working, "script2video", {"redo_shots": ["2"]}),
                {"slots": {"2": "keyframes"}, "phase": "stills"},
            )

    def test_a_one_based_shot_number_is_translated_to_the_slot_key(self):
        from agent_runtime.vimax_adapters import _resolve_redo

        with tempfile.TemporaryDirectory() as tmp:
            working, _ = self._session(tmp)
            # There is no slot "2" here, and the shot the Timeline calls "Shot 2" is slot "1".
            refused = _resolve_redo(working, "script2video", {"redo_shots": [2]})
            self.assertEqual(refused["error_type"], "unknown_redo_shot")
            self.assertEqual(refused["suggested_slot"], "1")
            self.assertIn('redo_shots=["1"]', refused["error"])

    def test_a_redo_of_the_wrong_shot_is_refused_rather_than_silently_redrawn(self):
        from agent_runtime.vimax_adapters import _resolve_redo

        with tempfile.TemporaryDirectory() as tmp:
            working, _ = self._session(tmp)
            # Slot "2" is the shot the Timeline calls "Shot 3"; asking for slot 3 instead
            # is a valid key, so only the recorded note can tell the two apart.
            empty = working / "script2video" / "shots" / "2"
            (empty / "qwen_qwen-image-3").mkdir(parents=True)
            (empty / "shot_description.json").write_text("{}", encoding="utf-8")
            (empty / "qwen_qwen-image-3" / "first_frame.png").write_bytes(b"frame")
            (working / "render_acceptance.json").write_text(json.dumps({
                "script2video": {"shots": {"1": {"keyframes": {"rejected_at": "later", "reason": "a white studio card"}}}}
            }), encoding="utf-8")

            refused = _resolve_redo(working, "script2video", {"redo_shots": ["2"]})
            self.assertEqual(refused["error_type"], "redo_shot_not_rejected")
            self.assertEqual(refused["rejected_slots"], ["1"])
            self.assertIn('shown as "Shot 2"', refused["error"])
            self.assertIn("allow_unreviewed_redo=true", refused["error"])

            # Redrawing an unreviewed shot stays possible, but has to say so.
            self.assertEqual(
                _resolve_redo(working, "script2video", {"redo_shots": ["2"], "allow_unreviewed_redo": True}),
                {"slots": {"2": "keyframes"}, "phase": "stills"},
            )
            # And the shot the note is against stays redrawable without any flag.
            self.assertEqual(
                _resolve_redo(working, "script2video", {"redo_shots": ["1"]}),
                {"slots": {"1": "keyframes"}, "phase": "stills"},
            )

    def test_a_key_that_is_simply_wrong_gets_no_made_up_suggestion(self):
        from agent_runtime.vimax_adapters import _resolve_redo

        with tempfile.TemporaryDirectory() as tmp:
            working, _ = self._session(tmp)
            refused = _resolve_redo(working, "script2video", {"redo_shots": ["scene_0/0"]})
            self.assertEqual(refused["error_type"], "unknown_redo_shot")
            self.assertNotIn("suggested_slot", refused)
            self.assertIn("Known slots: 0, 1", refused["error"])

    def test_idea_shots_are_addressed_by_scene(self):
        from agent_runtime.vimax_adapters import _resolve_redo

        with tempfile.TemporaryDirectory() as tmp:
            working, _ = self._session(tmp)
            for scene in (0, 1):
                shot_dir = working / "idea2video" / f"scene_{scene}" / "shots" / "0"
                (shot_dir / "qwen_qwen-image-3").mkdir(parents=True)
                (shot_dir / "shot_description.json").write_text("{}", encoding="utf-8")
                (shot_dir / "qwen_qwen-image-3" / "first_frame.png").write_bytes(b"frame")

            self.assertEqual(_resolve_redo(working, "idea2video", {"redo_shots": ["scene_1/0"]}), {"slots": {"scene_1/0": "keyframes"}, "phase": "stills"})

    def test_idea_slots_are_numbered_in_scene_order(self):
        from agent_runtime.vimax_adapters import _acceptance_slots

        with tempfile.TemporaryDirectory() as tmp:
            working, _ = self._session(tmp)
            for scene in (2, 10, 1):
                shot_dir = working / "idea2video" / f"scene_{scene}" / "shots" / "0"
                shot_dir.mkdir(parents=True)
                (shot_dir / "shot_description.json").write_text("{}", encoding="utf-8")

            self.assertEqual(list(_acceptance_slots(working, "idea2video")["keyframes"]), ["scene_1/0", "scene_2/0", "scene_10/0"])


class RedoArtifactTests(unittest.TestCase):
    """A redraw replaces one shot's artifacts, and the note survives as its reason."""

    def _rejected_session(self, tmp):
        working, _ = _acceptance_session(tmp)
        (working / "render_acceptance.json").write_text(json.dumps({
            "script2video": {"shots": {"0": {"keyframes": {"rejected_at": "later", "reason": "the backdrop is a studio white"}}}}
        }), encoding="utf-8")
        return working

    def test_only_the_redrawn_shot_is_cleared(self):
        from agent_runtime.vimax_adapters import _clear_redo_targets

        working = self._rejected_session(tempfile.mkdtemp())
        shot = working / "script2video" / "shots"
        (shot / "0" / "last_frame_selector_output.json").write_text("{}", encoding="utf-8")
        (shot / "0" / "kwaivgi_kling-video-o1").mkdir(parents=True)
        (shot / "0" / "kwaivgi_kling-video-o1" / "video.mp4").write_bytes(b"clip")

        removed = _clear_redo_targets(working, "script2video", {"0": "keyframes"})

        self.assertEqual(sorted(removed), [
            "script2video/final_video.mp4",
            "script2video/shots/0/kwaivgi_kling-video-o1/video.mp4",
            "script2video/shots/0/last_frame_selector_output.json",
            "script2video/shots/0/qwen_qwen-image-3/first_frame.png",
            "script2video/shots/0/qwen_qwen-image-3/last_frame.png",
        ])
        # The plan is not an artifact: the shot description and the other shot stay put.
        self.assertTrue((shot / "0" / "shot_description.json").exists())
        self.assertTrue((shot / "1" / "qwen_qwen-image-3" / "first_frame.png").exists())

    def test_a_first_frame_taken_from_a_transition_video_is_redrawn_too(self):
        from agent_runtime.vimax_adapters import _clear_redo_targets

        working = self._rejected_session(tempfile.mkdtemp())
        shot = working / "script2video" / "shots" / "0"
        # What the pipeline does for a camera with a parent shot: the first frame is a
        # camera still copied out of the transition video leading into the shot.
        (shot / "kwaivgi_kling-video-o1").mkdir(parents=True, exist_ok=True)
        (shot / "kwaivgi_kling-video-o1" / "new_camera_2.png").write_bytes(b"camera still")
        (shot / "kwaivgi_kling-video-o1" / "transition_video_from_shot_1.mp4").write_bytes(b"transition")

        removed = _clear_redo_targets(working, "script2video", {"0": "keyframes"})

        self.assertIn("script2video/shots/0/kwaivgi_kling-video-o1/new_camera_2.png", removed)
        # Left in place, the redraw would copy the same still back in as the new first frame.
        self.assertFalse((shot / "kwaivgi_kling-video-o1" / "new_camera_2.png").exists())
        # The transition video itself is kept: re-rendering it is a video-model call, and a
        # redrawn first frame is now drawn from the still and the portraits instead.
        self.assertTrue((shot / "kwaivgi_kling-video-o1" / "transition_video_from_shot_1.mp4").exists())
        self.assertTrue((shot / "shot_description.json").exists())

    def test_a_clip_redraw_leaves_the_frames_it_animates(self):
        from agent_runtime.vimax_adapters import _clear_redo_targets

        working = self._rejected_session(tempfile.mkdtemp())
        clip = working / "script2video" / "shots" / "0" / "kwaivgi_kling-video-o1" / "video.mp4"
        clip.parent.mkdir(parents=True)
        clip.write_bytes(b"clip")

        # The clip goes, and so does the film built from it: concatenation is skipped while a
        # film exists, so leaving it would strand the old cut on disk.
        self.assertEqual(
            _clear_redo_targets(working, "script2video", {"0": "clips"}),
            ["script2video/shots/0/kwaivgi_kling-video-o1/video.mp4", "script2video/final_video.mp4"],
        )
        self.assertTrue((working / "script2video" / "shots" / "0" / "qwen_qwen-image-3" / "first_frame.png").exists())

    def test_the_note_survives_as_the_reason_the_shot_was_redrawn(self):
        from agent_runtime.vimax_adapters import _acceptance_slots, _mark_slots_redone, _slot_state

        working = self._rejected_session(tempfile.mkdtemp())
        _mark_slots_redone(working, "script2video", {"0": "keyframes"}, {"0": "the backdrop is a studio white"})

        record = json.loads((working / "render_acceptance.json").read_text(encoding="utf-8"))["script2video"]["shots"]["0"]["keyframes"]
        self.assertEqual(record["reason"], "the backdrop is a studio white")
        self.assertIn("redone_at", record)
        self.assertNotIn("rejected_at", record)
        # Redrawn is not reviewed: the shot blocks the next phase again until it is looked at.
        self.assertEqual(_slot_state(working, _acceptance_slots(working, "script2video")["keyframes"]["0"]), "rendered")

    def test_the_notes_to_draw_with_are_the_ones_recorded_against_the_redrawn_shots(self):
        from agent_runtime.vimax_adapters import _revision_notes_for

        working = self._rejected_session(tempfile.mkdtemp())
        self.assertEqual(
            _revision_notes_for(working, "script2video", {"slots": {"0": "keyframes", "1": "keyframes"}}),
            {"0": "the backdrop is a studio white"},
        )
        self.assertEqual(_revision_notes_for(working, "script2video", {}), {})


class VideoModelChangeTests(unittest.TestCase):
    """Switching the video model has to take the old film with it."""

    def _sequence(self, tmp, video_model):
        working = Path(tmp)
        (working / "render_manifest.json").write_text(json.dumps({
            "render_mode": "script2video",
            "image_model": "qwen/qwen-image-3",
            "video_model": "kwaivgi/kling-video-o1",
        }), encoding="utf-8")
        film = working / "script2video" / "final_video.mp4"
        film.parent.mkdir(parents=True, exist_ok=True)
        film.write_bytes(b"the old model's film")
        from agent_runtime.vimax_adapters import _enforce_render_sequence

        result = _enforce_render_sequence(
            working,
            image_generator=type("G", (), {"model": "qwen/qwen-image-3"})(),
            video_generator=type("G", (), {"model": video_model})(),
            style="",
            allow_model_change=False,
            render_mode="script2video",
        )
        return result, film

    def test_a_new_video_model_drops_the_film_made_from_the_old_clips(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, film = self._sequence(tmp, "kwaivgi/kling-v3.0-std")

            self.assertIsNone(result)
            # Concatenation is skipped while a film exists, so leaving this one would keep the
            # old model's cut on disk under the new model's clips.
            self.assertFalse(film.exists())

    def test_the_same_video_model_leaves_the_film_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, film = self._sequence(tmp, "kwaivgi/kling-video-o1")

            self.assertTrue(film.exists())


class RedoRollbackTests(unittest.TestCase):
    """A redraw that fails must not leave the shot it was fixing worse off."""

    def _cleared_session(self, tmp):
        working, _ = _acceptance_session(tmp)
        (working / "render_acceptance.json").write_text(json.dumps({
            "script2video": {"shots": {"1": {"keyframes": {"rejected_at": "later", "reason": "the wrong red"}}}}
        }), encoding="utf-8")
        return working

    def test_clearing_holds_the_artifacts_rather_than_destroying_them(self):
        from agent_runtime.vimax_adapters import _clear_redo_targets

        working = self._cleared_session(tempfile.mkdtemp())
        shot = working / "script2video" / "shots" / "0"
        before = (shot / "qwen_qwen-image-3" / "first_frame.png").read_bytes()

        _clear_redo_targets(working, "script2video", {"0": "keyframes"})

        self.assertFalse((shot / "qwen_qwen-image-3" / "first_frame.png").exists())
        held = working / ".redo_backup" / "script2video" / "shots" / "0" / "qwen_qwen-image-3" / "first_frame.png"
        self.assertEqual(held.read_bytes(), before)

    def test_a_failed_redraw_puts_back_what_it_never_replaced(self):
        from agent_runtime.vimax_adapters import _clear_redo_targets, _restore_redo_backup

        working = self._cleared_session(tempfile.mkdtemp())
        shot = working / "script2video" / "shots" / "0"
        _clear_redo_targets(working, "script2video", {"0": "keyframes"})
        # The redraw replaced one frame and then died on the second, as a 402 does.
        (shot / "qwen_qwen-image-3" / "first_frame.png").write_bytes(b"the new frame")

        restored = _restore_redo_backup(working)

        # The frame the redraw did produce is kept; the one it never replaced comes back.
        self.assertEqual((shot / "qwen_qwen-image-3" / "first_frame.png").read_bytes(), b"the new frame")
        self.assertEqual((shot / "qwen_qwen-image-3" / "last_frame.png").read_bytes(), b"frame")
        self.assertIn("script2video/shots/0/qwen_qwen-image-3/last_frame.png", restored)
        self.assertFalse((working / ".redo_backup").exists())

    def test_a_successful_redraw_drops_what_it_held(self):
        from agent_runtime.vimax_adapters import _clear_redo_targets, _discard_redo_backup

        working = self._cleared_session(tempfile.mkdtemp())
        _clear_redo_targets(working, "script2video", {"0": "keyframes"})

        _discard_redo_backup(working)

        self.assertFalse((working / ".redo_backup").exists())

    def test_redrawing_a_clip_clears_the_film_that_holds_it(self):
        from agent_runtime.vimax_adapters import _clear_redo_targets

        working = self._cleared_session(tempfile.mkdtemp())
        film = working / "script2video" / "final_video.mp4"

        _clear_redo_targets(working, "script2video", {"1": "clips"})

        # The film is stale the moment one of its clips is redrawn, and concatenation is
        # skipped while a film exists: leaving it strands the old edit on disk.
        self.assertFalse(film.exists())
        self.assertTrue((working / ".redo_backup" / "script2video" / "final_video.mp4").exists())

    def test_redrawing_keyframes_clears_the_film_too(self):
        from agent_runtime.vimax_adapters import _clear_redo_targets

        working = self._cleared_session(tempfile.mkdtemp())
        film = working / "script2video" / "final_video.mp4"

        # A redrawn frame invalidates the clip animated from it, and so the film as well.
        _clear_redo_targets(working, "script2video", {"1": "keyframes"})

        self.assertFalse(film.exists())

    def test_a_redraw_of_a_shot_that_left_the_film_is_refused_as_removed(self):
        from agent_runtime.vimax_adapters import _resolve_redo

        working = self._cleared_session(tempfile.mkdtemp())
        shots = working / "script2video" / "shots"
        removed = working / "script2video" / ".removed_shots" / "1"
        removed.mkdir(parents=True)
        (shots / "1").rename(removed)

        refused = _resolve_redo(working, "script2video", {"redo_shots": ["1"]})

        # Not "unknown": the shot is out of the film, and saying it is a typo sends the
        # caller looking for a mistyped key instead of the way back.
        self.assertEqual(refused["error_type"], "redo_shot_removed")
        self.assertIn("Restore it in the Timeline", refused["error"])

    def test_a_failed_redraw_of_a_removed_shot_returns_it_to_the_removed_pile(self):
        from agent_runtime.vimax_adapters import _clear_redo_targets, _restore_redo_backup

        working = self._cleared_session(tempfile.mkdtemp())
        _clear_redo_targets(working, "script2video", {"0": "keyframes"})
        # The shot is removed from the film while its redraw is still running, which is
        # exactly when a rollback would otherwise put it back into the film's own directory.
        removed = working / "script2video" / ".removed_shots" / "0"
        removed.mkdir(parents=True)
        (working / "script2video" / "shots" / "0").rename(removed)

        _restore_redo_backup(working, "script2video")

        self.assertFalse((working / "script2video" / "shots" / "0").exists())
        self.assertEqual((removed / "qwen_qwen-image-3" / "first_frame.png").read_bytes(), b"frame")
        self.assertEqual((removed / "qwen_qwen-image-3" / "last_frame.png").read_bytes(), b"frame")
        self.assertFalse((working / ".redo_backup").exists())

    def test_a_failed_redraw_still_restores_a_shot_the_film_has(self):
        from agent_runtime.vimax_adapters import _clear_redo_targets, _restore_redo_backup

        working = self._cleared_session(tempfile.mkdtemp())
        _clear_redo_targets(working, "script2video", {"0": "keyframes"})

        restored = _restore_redo_backup(working, "script2video")

        frame = working / "script2video" / "shots" / "0" / "qwen_qwen-image-3" / "last_frame.png"
        self.assertEqual(frame.read_bytes(), b"frame")
        self.assertIn("script2video/shots/0/qwen_qwen-image-3/last_frame.png", restored)


class RenderTallyTests(unittest.TestCase):
    """A phase that regenerated nothing must not read as success."""

    def test_only_drawing_stages_count_as_generated(self):
        from agent_runtime.vimax_adapters import _pipeline_progress

        emitted = []
        runtime = type("Runtime", (), {"emit_progress": lambda self, message, **kw: emitted.append(kw)})()
        tally: dict[str, int] = {}
        progress = _pipeline_progress(runtime, "session", tally=tally)

        progress("frame_done", "drew one")
        progress("frame_exists", "reused one")
        progress("frame_done", "drew another")
        progress("video_clip_exists", "reused a clip")
        progress("frame_prompt_done", "selected references")

        self.assertEqual(tally, {"generated": 2, "reused": 2})
        self.assertEqual(len(emitted), 5)

    def test_no_tally_is_asked_for_when_the_caller_does_not_need_one(self):
        from agent_runtime.vimax_adapters import _pipeline_progress

        runtime = type("Runtime", (), {"emit_progress": lambda self, message, **kw: None})()
        self.assertIsNotNone(_pipeline_progress(runtime, "session"))


    def test_references_dropped_is_kept_for_the_result_to_report(self):
        from agent_runtime.vimax_adapters import _pipeline_progress

        tally: dict[str, object] = {}
        progress = _pipeline_progress(None, "session", tally=tally)

        progress("frame_done", "drew one")
        progress("references_dropped", "ming-image takes no reference images, so this frame was drawn without them", {"model": "ming"})

        # The frame is still counted as drawn, and the reason it is not the planned frame is
        # carried into the result rather than only into the progress stream.
        self.assertEqual(tally["generated"], 1)
        self.assertEqual(len(tally["warnings"]), 1)
        self.assertIn("no reference images", tally["warnings"][0])


class RedoRenderTests(unittest.IsolatedAsyncioTestCase):
    """What the render tool does with a redo request, end to end."""

    def _session(self, tmp):
        working, index = _acceptance_session(tmp)
        (working / "script2video" / "script.txt").write_text("script", encoding="utf-8")
        (working / "script2video" / "characters.json").write_text("[]", encoding="utf-8")
        (working / "script2video" / "storyboard.json").write_text("[]", encoding="utf-8")
        (working / "script2video" / "camera_tree.json").write_text("[]", encoding="utf-8")
        for shot in (0, 1):
            (working / "script2video" / "shots" / str(shot) / "first_frame_selector_output.json").write_text("{}", encoding="utf-8")
        return working, index

    async def test_the_trail_names_the_redraw_while_it_runs_not_only_when_it_ends(self):
        with tempfile.TemporaryDirectory() as tmp:
            working, index = self._session(tmp)
            (working / "render_acceptance.json").write_text(json.dumps({
                "script2video": {
                    "portraits": {"accepted_at": "now", "artifacts": [_lock(working, "script2video/shots/0/qwen_qwen-image-3/first_frame.png")]},
                    "shots": {"1": {"keyframes": {"rejected_at": "later", "reason": "a white studio card"}}},
                }
            }), encoding="utf-8")
            adapter = ViMaxAdapters(Path(tmp), index)
            seen_at_start: list[dict] = []
            written = va_write_render_status_recorder(seen_at_start)

            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_image_generator", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_video_generator", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._write_render_status", side_effect=written), \
                 patch("agent_runtime.vimax_adapters.Script2VideoPipeline", RecordingScriptPipeline):
                await adapter.vimax_render_video({"stop_after": "stills", "render_mode": "script2video", "redo_rejected": True})

            # The first row is written as the render starts, which is when a reader needs it.
            self.assertEqual(seen_at_start[0]["payload"]["redone_shots"], ["1"])
            self.assertEqual(seen_at_start[0]["status"], "rendering")

    async def test_the_render_clears_the_redrawn_shot_and_draws_it_with_the_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            working, index = self._session(tmp)
            (working / "render_acceptance.json").write_text(json.dumps({
                "script2video": {
                    "portraits": {"accepted_at": "now", "artifacts": [_lock(working, "script2video/shots/0/qwen_qwen-image-3/first_frame.png")]},
                    "shots": {"1": {"keyframes": {"rejected_at": "later", "reason": "a white studio card"}}},
                }
            }), encoding="utf-8")
            adapter = ViMaxAdapters(Path(tmp), index)
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_image_generator", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_video_generator", return_value=object()), \
                 patch("agent_runtime.vimax_adapters.Script2VideoPipeline", RecordingScriptPipeline):
                result = await adapter.vimax_render_video({"stop_after": "stills", "render_mode": "script2video", "redo_rejected": True})

            self.assertTrue(result.ok, result.content)
            self.assertEqual(result.metadata["redone_shots"], ["1"])
            self.assertEqual(result.metadata["cleared"], 3)  # two frames, and the film they fed
            self.assertEqual(RecordingScriptPipeline.last_revision_notes, {"1": "a white studio card"})
            # and the run works on those shots alone: the rest are left for their own turn
            self.assertEqual(RecordingScriptPipeline.last_only_shots, [1])
            self.assertFalse((working / "script2video" / "shots" / "1" / "qwen_qwen-image-3" / "first_frame.png").exists())
            self.assertFalse((working / "script2video" / "shots" / "1" / "first_frame_selector_output.json").exists())
            self.assertTrue((working / "script2video" / "shots" / "0" / "qwen_qwen-image-3" / "first_frame.png").exists())

    async def test_a_refused_phase_deletes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            working, index = self._session(tmp)
            (working / "render_acceptance.json").write_text(json.dumps({
                "script2video": {"shots": {"1": {"keyframes": {"rejected_at": "later", "reason": "a white studio card"}}}}
            }), encoding="utf-8")
            adapter = ViMaxAdapters(Path(tmp), index)
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_image_generator", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_video_generator", return_value=object()), \
                 patch("agent_runtime.vimax_adapters.Script2VideoPipeline", RecordingScriptPipeline):
                # Rendering video needs accepted keyframes, which this session does not have.
                result = await adapter.vimax_render_video({"stop_after": "video", "render_mode": "script2video", "redo_shots": ["1"]})

            self.assertFalse(result.ok)
            self.assertEqual(result.metadata["error_type"], "acceptance_required")
            # The refusal came before the redraw, so the rejected shot was not cleared.
            self.assertTrue((working / "script2video" / "shots" / "1" / "qwen_qwen-image-3" / "first_frame.png").exists())

    async def test_redrawing_clips_runs_the_clip_phase_even_when_the_call_asked_for_less(self):
        with tempfile.TemporaryDirectory() as tmp:
            working, index = self._session(tmp)
            clip = working / "script2video" / "shots" / "1" / "kwaivgi_kling-video-o1" / "video.mp4"
            clip.parent.mkdir(parents=True)
            clip.write_bytes(b"clip")
            (working / "render_acceptance.json").write_text(json.dumps({
                "script2video": {
                    "portraits": {"accepted_at": "now", "artifacts": [_lock(working, "script2video/shots/0/qwen_qwen-image-3/first_frame.png")]},
                    "shots": {
                        "0": {"keyframes": {"accepted_at": "now", "artifacts": [_lock(working, "script2video/shots/0/qwen_qwen-image-3/first_frame.png")]}},
                        "1": {"keyframes": {"accepted_at": "now", "artifacts": [_lock(working, "script2video/shots/1/qwen_qwen-image-3/first_frame.png")]},
                              "clips": {"rejected_at": "later", "reason": "the ceiling sags"}},
                    },
                }
            }), encoding="utf-8")
            adapter = ViMaxAdapters(Path(tmp), index)
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_image_generator", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_video_generator", return_value=object()), \
                 patch("agent_runtime.vimax_adapters.Script2VideoPipeline", RecordingScriptPipeline):
                result = await adapter.vimax_render_video({"stop_after": "portraits", "render_mode": "script2video", "redo_rejected": True})

            self.assertTrue(result.ok, result.content)
            # The deleted clip has to be replaced in the same call, so the phase is the floor.
            self.assertEqual(result.metadata["phase"], "video")
            self.assertFalse(clip.exists())

    async def test_a_redraw_that_fails_gives_back_what_it_cleared(self):
        with tempfile.TemporaryDirectory() as tmp:
            working, index = self._session(tmp)
            (working / "render_acceptance.json").write_text(json.dumps({
                "script2video": {
                    "portraits": {"accepted_at": "now", "artifacts": [_lock(working, "script2video/shots/0/qwen_qwen-image-3/first_frame.png")]},
                    "shots": {"1": {"keyframes": {"rejected_at": "later", "reason": "a white studio card"}}},
                }
            }), encoding="utf-8")
            kept = (working / "script2video" / "shots" / "1" / "qwen_qwen-image-3" / "first_frame.png").read_bytes()
            adapter = ViMaxAdapters(Path(tmp), index)
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_image_generator", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_video_generator", return_value=object()), \
                 patch("agent_runtime.vimax_adapters.Script2VideoPipeline", FailingRedrawPipeline):
                result = await adapter.vimax_render_video({"stop_after": "stills", "render_mode": "script2video", "redo_rejected": True})

            self.assertFalse(result.ok)
            self.assertEqual(result.metadata["error_type"], "render_failed")
            self.assertIn("rolled back", result.content)
            self.assertTrue(result.metadata["restored"])
            # The shot still holds what it held: the redraw replaced nothing.
            self.assertEqual((working / "script2video" / "shots" / "1" / "qwen_qwen-image-3" / "first_frame.png").read_bytes(), kept)
            self.assertFalse((working / ".redo_backup").exists())

    async def test_a_redraw_may_not_move_the_sequence_to_another_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            working, index = self._session(tmp)
            (working / "render_manifest.json").write_text(json.dumps({"render_mode": "script2video"}), encoding="utf-8")
            idea_shot = working / "idea2video" / "scene_0" / "shots" / "0"
            (idea_shot / "qwen_qwen-image-3").mkdir(parents=True)
            (idea_shot / "shot_description.json").write_text("{}", encoding="utf-8")
            (idea_shot / "qwen_qwen-image-3" / "first_frame.png").write_bytes(b"frame")
            # The abandoned root has to be a complete plan, or the render never gets as
            # far as the root check.
            idea = working / "idea2video"
            (idea / "story.txt").write_text("story", encoding="utf-8")
            (idea / "characters.json").write_text("[]", encoding="utf-8")
            (idea / "script.json").write_text("[]", encoding="utf-8")
            (idea / "scene_0" / "storyboard.json").write_text("[]", encoding="utf-8")
            (idea / "scene_0" / "camera_tree.json").write_text("[]", encoding="utf-8")
            adapter = ViMaxAdapters(Path(tmp), index)
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_image_generator", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_video_generator", return_value=object()), \
                 patch("agent_runtime.vimax_adapters.Script2VideoPipeline", RecordingScriptPipeline), \
                 patch("agent_runtime.vimax_adapters.Idea2VideoPipeline", RecordingScriptPipeline):
                # Even the escape hatch that moves a sequence may not be used for a redraw.
                result = await adapter.vimax_render_video({"stop_after": "stills", "render_mode": "idea2video", "allow_model_change": True, "redo_shots": ["scene_0/0"]})

            self.assertFalse(result.ok)
            self.assertEqual(result.metadata["error_type"], "redo_wrong_root")
            self.assertTrue((idea_shot / "qwen_qwen-image-3" / "first_frame.png").exists())

    async def test_an_unknown_shot_is_refused_before_anything_is_deleted(self):
        with tempfile.TemporaryDirectory() as tmp:
            working, index = self._session(tmp)
            adapter = ViMaxAdapters(Path(tmp), index)
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_image_generator", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_video_generator", return_value=object()), \
                 patch("agent_runtime.vimax_adapters.Script2VideoPipeline", RecordingScriptPipeline):
                result = await adapter.vimax_render_video({"stop_after": "stills", "render_mode": "script2video", "redo_shots": [7]})

            self.assertFalse(result.ok)
            self.assertEqual(result.metadata["error_type"], "unknown_redo_shot")
            self.assertTrue((working / "script2video" / "shots" / "1" / "qwen_qwen-image-3" / "first_frame.png").exists())


class AcceptanceGateToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_render_tool_refuses_an_unreviewed_phase_and_can_be_overridden(self):
        with tempfile.TemporaryDirectory() as tmp:
            working, index = _acceptance_session(tmp)
            (working / "script2video" / "script.txt").write_text("script", encoding="utf-8")
            (working / "script2video" / "characters.json").write_text("[]", encoding="utf-8")
            (working / "script2video" / "storyboard.json").write_text("[]", encoding="utf-8")
            (working / "script2video" / "camera_tree.json").write_text("[]", encoding="utf-8")
            adapter = ViMaxAdapters(Path(tmp), index)
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_image_generator", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_video_generator", return_value=object()), \
                 patch("agent_runtime.vimax_adapters.Script2VideoPipeline", RecordingScriptPipeline):
                refused = await adapter.vimax_render_video({"stop_after": "video", "render_mode": "script2video"})
                overridden = await adapter.vimax_render_video({"stop_after": "video", "render_mode": "script2video", "allow_unlocked": True})

            self.assertFalse(refused.ok)
            self.assertEqual(refused.metadata["error_type"], "acceptance_required")
            self.assertTrue(overridden.ok, overridden.content)


class PlanningModeTests(unittest.TestCase):
    """Planning must not re-enter an abandoned root the session no longer works in."""

    def test_a_script_in_the_script_root_beats_the_stored_idea(self):
        from agent_runtime.vimax_adapters import _resolve_planning_mode

        self.assertEqual(_resolve_planning_mode("", {}, existing_script="a script"), "script2video")

    def test_an_idea_only_session_plans_the_idea_root(self):
        from agent_runtime.vimax_adapters import _resolve_planning_mode

        self.assertEqual(_resolve_planning_mode("", {}, existing_script=""), "idea2video")

    def test_the_pinned_root_wins(self):
        from agent_runtime.vimax_adapters import _resolve_planning_mode

        self.assertEqual(_resolve_planning_mode("", {"render_mode": "script2video"}, existing_script=""), "script2video")

    def test_an_explicit_root_wins_over_the_pin(self):
        from agent_runtime.vimax_adapters import _resolve_planning_mode

        self.assertEqual(_resolve_planning_mode("idea2video", {"render_mode": "script2video"}, existing_script="a script"), "idea2video")

    def test_an_unknown_root_is_rejected(self):
        from agent_runtime.vimax_adapters import _resolve_planning_mode

        with self.assertRaises(ValueError) as caught:
            _resolve_planning_mode("movie2video", {}, existing_script="")
        self.assertIn("render_mode must be one of", str(caught.exception))


class RenderModeDispatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_render_mode_picks_the_root_and_pins_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            record = index.create(idea="x", style="cinematic")
            root = Path(tmp) / record["working_dir"]
            # Both roots are complete: an abandoned idea plan beside the script plan.
            idea = root / "idea2video"
            (idea / "scene_0" / "shots" / "0").mkdir(parents=True, exist_ok=True)
            for name, body in (("story.txt", "story"), ("characters.json", "[]"), ("script.json", "[]")):
                (idea / name).write_text(body, encoding="utf-8")
            (idea / "scene_0" / "storyboard.json").write_text("[]", encoding="utf-8")
            (idea / "scene_0" / "camera_tree.json").write_text("[]", encoding="utf-8")
            (idea / "scene_0" / "shots" / "0" / "shot_description.json").write_text("{}", encoding="utf-8")
            script = root / "script2video" / "shots" / "0"
            script.mkdir(parents=True, exist_ok=True)
            (root / "script2video" / "script.txt").write_text("script", encoding="utf-8")
            (root / "script2video" / "characters.json").write_text("[]", encoding="utf-8")
            (root / "script2video" / "storyboard.json").write_text("[]", encoding="utf-8")
            (root / "script2video" / "camera_tree.json").write_text("[]", encoding="utf-8")
            (script / "shot_description.json").write_text("{}", encoding="utf-8")

            adapter = ViMaxAdapters(Path(tmp), index)
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_image_generator", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_video_generator", return_value=object()), \
                 patch("agent_runtime.vimax_adapters.Idea2VideoPipeline", FakeIdeaPipeline), \
                 patch("agent_runtime.vimax_adapters.Script2VideoPipeline", RecordingScriptPipeline):
                result = await adapter.vimax_render_video({"stop_after": "portraits", "render_mode": "script2video"})

            self.assertTrue(result.ok, result.content)
            self.assertEqual(result.metadata["render_mode"], "script2video")
            manifest = json.loads((root / "render_manifest.json").read_text())
            self.assertEqual(manifest["render_mode"], "script2video")

    async def test_a_pinned_sequence_refuses_to_switch_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            record = index.create(idea="x", style="cinematic")
            root = Path(tmp) / record["working_dir"]
            (root / "render_manifest.json").write_text(json.dumps({
                "image_model": "", "video_model": "", "style": "cinematic", "render_mode": "script2video",
            }), encoding="utf-8")
            for name, body in (("story.txt", "s"), ("characters.json", "[]"), ("script.json", "[]")):
                (root / "idea2video" / name).parent.mkdir(parents=True, exist_ok=True)
                (root / "idea2video" / name).write_text(body, encoding="utf-8")
            (root / "idea2video" / "scene_0" / "shots" / "0").mkdir(parents=True, exist_ok=True)
            (root / "idea2video" / "scene_0" / "storyboard.json").write_text("[]", encoding="utf-8")
            (root / "idea2video" / "scene_0" / "camera_tree.json").write_text("[]", encoding="utf-8")
            (root / "idea2video" / "scene_0" / "shots" / "0" / "shot_description.json").write_text("{}", encoding="utf-8")

            adapter = ViMaxAdapters(Path(tmp), index)
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_image_generator", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_video_generator", return_value=object()):
                result = await adapter.vimax_render_video({"stop_after": "portraits", "render_mode": "idea2video"})

            self.assertFalse(result.ok)
            self.assertEqual(result.metadata["error_type"], "render_mode_changed")
            self.assertIn("script2video", result.metadata["error"])


class ViMaxAdapterTests(unittest.IsolatedAsyncioTestCase):
    def test_build_chat_model_uses_bounded_init_chat_model_kwargs(self):
        fake = FakeInitChatModel()
        with patch.dict("os.environ", {
            "VIMAX_LLM_API_KEY": "test-key",
            "VIMAX_LLM_MODEL": "test-model",
            "VIMAX_LLM_BASE_URL": "https://example.invalid/v1",
            "VIMAX_LLM_REQUEST_TIMEOUT_SECONDS": "12",
            "VIMAX_NARRATIVE_MAX_TOKENS": "1234",
        }), patch("agent_runtime.vimax_adapters.init_chat_model", fake):
            from agent_runtime.vimax_adapters import _build_chat_model

            _build_chat_model()

        self.assertEqual(fake.calls[0]["model"], "test-model")
        self.assertEqual(fake.calls[0]["base_url"], "https://example.invalid/v1")
        self.assertEqual(fake.calls[0]["timeout"], 12.0)
        self.assertEqual(fake.calls[0]["max_retries"], 0)
        self.assertEqual(fake.calls[0]["max_completion_tokens"], 1234)


    async def test_narrative_planning_uses_text_only_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            adapter = ViMaxAdapters(Path(tmp), index)
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()), \
                 patch("agent_runtime.vimax_adapters.Idea2VideoPipeline", FakeIdeaPipeline), \
                 patch("agent_runtime.vimax_adapters.Script2VideoPipeline", FakeScriptPipeline):
                result = await adapter.vimax_narrative_planning({"idea": "moon cat", "user_requirement": "short", "style": "anime"})
            self.assertTrue(result.ok)
            payload = json.loads(result.content)
            self.assertTrue(payload["ready_for_render"])
            root = Path(tmp) / payload["working_dir"]
            self.assertTrue((root / "idea2video" / "scene_0" / "storyboard.json").exists())
            self.assertTrue((root / "idea2video" / "scene_0" / "camera_tree.json").exists())
            self.assertTrue((root / "idea2video" / "scene_0" / "shots" / "0" / "shot_description.json").exists())
            self.assertFalse((root / "script2video" / "storyboard.json").exists())
            self.assertFalse((root / "script2video" / "final_video.mp4").exists())


    async def test_script_mode_persists_source_script_for_render(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            adapter = ViMaxAdapters(Path(tmp), index)
            script = "A red ball rolls across a white table."
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()), \
                 patch("agent_runtime.vimax_adapters.Script2VideoPipeline", FakeScriptPipeline):
                result = await adapter.vimax_narrative_planning({"script": script, "user_requirement": "one shot"})
            self.assertTrue(result.ok)
            payload = json.loads(result.content)
            root = Path(tmp) / payload["working_dir"]
            self.assertEqual((root / "script2video" / "script.txt").read_text(encoding="utf-8"), script)
            self.assertEqual(index.artifact_checklist(payload["session_id"])["script2video/script.txt"], True)
            from agent_runtime.vimax_adapters import _load_script_text
            self.assertEqual(_load_script_text(root), script)


    async def test_narrative_planning_forwards_pipeline_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            adapter = ViMaxAdapters(Path(tmp), index)
            events = []
            runtime = ToolRuntimeContext("vimax_narrative_planning", "vimax_narrative_planning", turn_id="turn-test", progress_callback=events.append)
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()), \
                 patch("agent_runtime.vimax_adapters.Idea2VideoPipeline", FakeIdeaPipeline), \
                 patch("agent_runtime.vimax_adapters.Script2VideoPipeline", FakeScriptPipeline):
                result = await adapter.vimax_narrative_planning({"idea": "moon cat"}, runtime)
            self.assertTrue(result.ok)
            stages = [event["progress"]["stage"] for event in events if event.get("type") == "tool_progress"]
            self.assertIn("initializing_llm", stages)
            self.assertIn("develop_story", stages)
            self.assertIn("design_storyboard", stages)
            self.assertIn("decompose_shots", stages)
            self.assertIn("construct_camera_tree", stages)


    async def test_plan_scene_failure_marks_session_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            adapter = ViMaxAdapters(Path(tmp), index)
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()), \
                 patch("agent_runtime.vimax_adapters.Idea2VideoPipeline", FakeIdeaPipeline), \
                 patch("agent_runtime.vimax_adapters.Script2VideoPipeline", FailingScriptPipeline):
                result = await adapter.vimax_narrative_planning({"idea": "moon cat"})
            self.assertFalse(result.ok)
            self.assertEqual(result.metadata["error_type"], "recoverable_planning_step_failed")
            self.assertTrue(result.metadata["retryable"])
            session = index.active()
            self.assertEqual(session["stage"], "error")
            self.assertIn("storyboard failed", session["summary"])


    async def test_narrative_planning_timeout_marks_session_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            adapter = ViMaxAdapters(Path(tmp), index)
            with patch.dict("os.environ", {"VIMAX_NARRATIVE_STEP_TIMEOUT_SECONDS": "0.01"}), \
                 patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()), \
                 patch("agent_runtime.vimax_adapters.Idea2VideoPipeline", HangingIdeaPipeline):
                result = await adapter.vimax_narrative_planning({"idea": "moon cat"})
            self.assertFalse(result.ok)
            self.assertEqual(result.metadata["error_type"], "recoverable_planning_step_failed")
            session = index.active()
            self.assertIsNotNone(session)
            self.assertEqual(session["stage"], "error")
            self.assertIn("timed out", session["summary"])



    async def test_active_session_without_new_input_continues_existing_idea(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            record = index.create(idea="moon cat", user_requirement="short", style="anime")
            adapter = ViMaxAdapters(Path(tmp), index)
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()),                  patch("agent_runtime.vimax_adapters.Idea2VideoPipeline", FakeIdeaPipeline),                  patch("agent_runtime.vimax_adapters.Script2VideoPipeline", FakeScriptPipeline):
                result = await adapter.vimax_narrative_planning({})
            self.assertTrue(result.ok)
            payload = json.loads(result.content)
            self.assertEqual(payload["session_id"], record["session_id"])
            self.assertEqual(index.active()["session_id"], record["session_id"])


    async def test_active_session_continuation_preserves_existing_style(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            record = index.create(idea="moon cat", user_requirement="short", style="anime")
            adapter = ViMaxAdapters(Path(tmp), index)
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()),                  patch("agent_runtime.vimax_adapters.Idea2VideoPipeline", FakeIdeaPipeline),                  patch("agent_runtime.vimax_adapters.Script2VideoPipeline", FakeScriptPipeline):
                result = await adapter.vimax_narrative_planning({"session_id": record["session_id"]})
            self.assertTrue(result.ok)
            self.assertEqual(index.get(record["session_id"])["style"], "anime")

    async def test_new_idea_creates_new_session_instead_of_reusing_active(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            adapter = ViMaxAdapters(Path(tmp), index)
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()), \
                 patch("agent_runtime.vimax_adapters.Idea2VideoPipeline", FakeIdeaPipeline), \
                 patch("agent_runtime.vimax_adapters.Script2VideoPipeline", FakeScriptPipeline):
                first = await adapter.vimax_narrative_planning({"idea": "moon cat"})
                second = await adapter.vimax_narrative_planning({"idea": "ocean robot"})
            self.assertNotEqual(json.loads(first.content)["session_id"], json.loads(second.content)["session_id"])

    async def test_new_idea_initializes_named_empty_active_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            empty = index.create(project_name="00")
            adapter = ViMaxAdapters(Path(tmp), index)
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()), \
                 patch("agent_runtime.vimax_adapters.Idea2VideoPipeline", FakeIdeaPipeline), \
                 patch("agent_runtime.vimax_adapters.Script2VideoPipeline", FakeScriptPipeline):
                result = await adapter.vimax_narrative_planning({"idea": "moon cat"})
            self.assertTrue(result.ok)
            payload = json.loads(result.content)
            self.assertEqual(payload["session_id"], empty["session_id"])
            self.assertEqual(index.active()["project_name"], "00")
            self.assertEqual(index.active()["idea"], "moon cat")
            self.assertEqual(len(index.load()["sessions"]), 1)


    async def test_explicit_session_with_different_idea_creates_new_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            old = index.create(idea="old cat")
            adapter = ViMaxAdapters(Path(tmp), index)
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()), \
                 patch("agent_runtime.vimax_adapters.Idea2VideoPipeline", FakeIdeaPipeline), \
                 patch("agent_runtime.vimax_adapters.Script2VideoPipeline", FakeScriptPipeline):
                result = await adapter.vimax_narrative_planning({"session_id": old["session_id"], "idea": "new robot"})
            self.assertTrue(result.ok)
            payload = json.loads(result.content)
            self.assertNotEqual(payload["session_id"], old["session_id"])
            self.assertEqual(index.get(payload["session_id"])["idea"], "new robot")

    async def test_revision_mode_rewrites_existing_artifact_and_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            record = index.create(idea="x")
            target = Path(tmp) / record["working_dir"] / "idea2video" / "scene_0" / "storyboard.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text('[{"idx": 0, "description": "calm"}]', encoding="utf-8")
            adapter = ViMaxAdapters(Path(tmp), index)
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=FakeRevisionModel()):
                result = await adapter.vimax_narrative_planning({"revision_target": "idea2video/scene_0/storyboard.json", "revision_instruction": "make it oppressive"})
            self.assertTrue(result.ok)
            self.assertIn("more oppressive", target.read_text(encoding="utf-8"))
            self.assertTrue((Path(tmp) / ".vimax" / "logs" / "revisions.jsonl").exists())
            self.assertTrue(index.get(record["session_id"])["stale"]["final_video"])


    async def test_revision_missing_instruction_marks_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            record = index.create(idea="x")
            target = Path(tmp) / record["working_dir"] / "idea2video" / "scene_0" / "storyboard.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text('[]', encoding="utf-8")
            adapter = ViMaxAdapters(Path(tmp), index)
            result = await adapter.vimax_narrative_planning({"revision_target": "idea2video/scene_0/storyboard.json"})
            self.assertFalse(result.ok)
            self.assertEqual(result.metadata["error_type"], "missing_revision_instruction")
            self.assertEqual(index.get(record["session_id"])["stage"], "error")


    async def test_revision_missing_target_marks_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            record = index.create(idea="x")
            adapter = ViMaxAdapters(Path(tmp), index)
            result = await adapter.vimax_narrative_planning({"revision_target": "idea2video/scene_0/missing.json", "revision_instruction": "change it"})
            self.assertFalse(result.ok)
            self.assertEqual(result.metadata["error_type"], "dependency_missing")
            self.assertEqual(index.get(record["session_id"])["stage"], "error")

    async def test_render_setup_failure_marks_session_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            record = index.create(idea="x")
            root = Path(tmp) / record["working_dir"] / "idea2video"
            (root / "scene_0" / "shots" / "0").mkdir(parents=True, exist_ok=True)
            (root / "story.txt").write_text("story", encoding="utf-8")
            (root / "characters.json").write_text("[]", encoding="utf-8")
            (root / "script.json").write_text("[]", encoding="utf-8")
            (root / "scene_0" / "storyboard.json").write_text("[]", encoding="utf-8")
            (root / "scene_0" / "camera_tree.json").write_text("[]", encoding="utf-8")
            (root / "scene_0" / "shots" / "0" / "shot_description.json").write_text("{}", encoding="utf-8")
            adapter = ViMaxAdapters(Path(tmp), index)
            with patch("agent_runtime.vimax_adapters._build_chat_model", side_effect=RuntimeError("missing key")):
                result = await adapter.vimax_render_video({"stop_after": "video", "allow_unlocked": True})
            self.assertFalse(result.ok)
            self.assertEqual(result.metadata["error_type"], "render_failed")
            self.assertIn("missing key", result.content)
            self.assertEqual(index.get(record["session_id"])["stage"], "error")

    async def test_render_failure_marks_session_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            record = index.create(idea="x")
            root = Path(tmp) / record["working_dir"] / "idea2video"
            (root / "scene_0" / "shots" / "0").mkdir(parents=True, exist_ok=True)
            (root / "story.txt").write_text("story", encoding="utf-8")
            (root / "characters.json").write_text("[]", encoding="utf-8")
            (root / "script.json").write_text("[]", encoding="utf-8")
            (root / "scene_0" / "storyboard.json").write_text("[]", encoding="utf-8")
            (root / "scene_0" / "camera_tree.json").write_text("[]", encoding="utf-8")
            (root / "scene_0" / "shots" / "0" / "shot_description.json").write_text("{}", encoding="utf-8")
            adapter = ViMaxAdapters(Path(tmp), index)
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_image_generator", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_video_generator", return_value=object()), \
                 patch("agent_runtime.vimax_adapters.Idea2VideoPipeline", FailRenderIdeaPipeline):
                result = await adapter.vimax_render_video({"stop_after": "video", "allow_unlocked": True})
            self.assertFalse(result.ok)
            self.assertEqual(result.metadata["error_type"], "render_failed")
            self.assertIn("render failed", result.content)
            self.assertEqual(index.get(record["session_id"])["stage"], "error")
            status_path = Path(tmp) / record["working_dir"] / "render_status.json"
            events_path = Path(tmp) / record["working_dir"] / "render_events.jsonl"
            self.assertTrue(status_path.exists())
            self.assertTrue(events_path.exists())
            status = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "error")
            self.assertEqual(status["error_type"], "render_failed")

    async def test_render_403_key_limit_is_non_retryable_and_sanitized(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            record = index.create(idea="x")
            root = Path(tmp) / record["working_dir"] / "idea2video"
            (root / "scene_0" / "shots" / "0").mkdir(parents=True, exist_ok=True)
            (root / "story.txt").write_text("story", encoding="utf-8")
            (root / "characters.json").write_text("[]", encoding="utf-8")
            (root / "script.json").write_text("[]", encoding="utf-8")
            (root / "scene_0" / "storyboard.json").write_text("[]", encoding="utf-8")
            (root / "scene_0" / "camera_tree.json").write_text("[]", encoding="utf-8")
            (root / "scene_0" / "shots" / "0" / "shot_description.json").write_text("{}", encoding="utf-8")
            adapter = ViMaxAdapters(Path(tmp), index)
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_image_generator", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_video_generator", return_value=object()), \
                 patch("agent_runtime.vimax_adapters.Idea2VideoPipeline", FailRender403IdeaPipeline):
                result = await adapter.vimax_render_video({"stop_after": "video", "allow_unlocked": True})
            self.assertFalse(result.ok)
            self.assertFalse(result.metadata["retryable"])
            self.assertIn("<redacted>", result.metadata["error"])
            self.assertNotIn("sk-short", result.metadata["error"])
            status = json.loads((Path(tmp) / record["working_dir"] / "render_status.json").read_text(encoding="utf-8"))
            self.assertFalse(status["retryable"])
            self.assertNotIn("sk-short", status["error"])


    async def test_render_content_filter_is_non_retryable_and_names_the_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            record = index.create(idea="x")
            root = Path(tmp) / record["working_dir"] / "idea2video"
            (root / "scene_0" / "shots" / "0").mkdir(parents=True, exist_ok=True)
            (root / "story.txt").write_text("story", encoding="utf-8")
            (root / "characters.json").write_text("[]", encoding="utf-8")
            (root / "script.json").write_text("[]", encoding="utf-8")
            (root / "scene_0" / "storyboard.json").write_text("[]", encoding="utf-8")
            (root / "scene_0" / "camera_tree.json").write_text("[]", encoding="utf-8")
            (root / "scene_0" / "shots" / "0" / "shot_description.json").write_text("{}", encoding="utf-8")
            adapter = ViMaxAdapters(Path(tmp), index)
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_image_generator", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_video_generator", return_value=object()), \
                 patch("agent_runtime.vimax_adapters.Idea2VideoPipeline", FailRenderContentFilterIdeaPipeline):
                result = await adapter.vimax_render_video({"stop_after": "video", "allow_unlocked": True})

            self.assertFalse(result.ok)
            # A provider content filter rejects the same prompt every time, so an
            # identical re-render cannot succeed.
            self.assertFalse(result.metadata["retryable"])
            self.assertIn("first_frame of shot 13", result.metadata["error"])

    async def test_render_missing_reference_is_non_retryable(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            record = index.create(idea="x")
            root = Path(tmp) / record["working_dir"] / "idea2video"
            (root / "scene_0" / "shots" / "0").mkdir(parents=True, exist_ok=True)
            (root / "story.txt").write_text("story", encoding="utf-8")
            (root / "characters.json").write_text("[]", encoding="utf-8")
            (root / "script.json").write_text("[]", encoding="utf-8")
            (root / "scene_0" / "storyboard.json").write_text("[]", encoding="utf-8")
            (root / "scene_0" / "camera_tree.json").write_text("[]", encoding="utf-8")
            (root / "scene_0" / "shots" / "0" / "shot_description.json").write_text("{}", encoding="utf-8")
            adapter = ViMaxAdapters(Path(tmp), index)
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_image_generator", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_video_generator", return_value=object()), \
                 patch("agent_runtime.vimax_adapters.Idea2VideoPipeline", FailRenderMissingReferenceIdeaPipeline):
                result = await adapter.vimax_render_video({"stop_after": "video", "allow_unlocked": True})

            self.assertFalse(result.ok)
            # The file is still missing on the next attempt, so a retry re-fails.
            self.assertFalse(result.metadata["retryable"])
            self.assertIn("first_frame of shot 0", result.metadata["error"])

    def test_an_out_of_credit_account_is_not_retried(self):
        from agent_runtime.vimax_adapters import _is_retryable_render_error

        for text in ("HTTP 402", "Insufficient credits. Add more using https://openrouter.ai/settings/credits"):
            self.assertFalse(_is_retryable_render_error(RuntimeError(text)))
        self.assertTrue(_is_retryable_render_error(RuntimeError("connection reset by peer")))

    async def test_render_pipeline_stdout_is_suppressed(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            record = index.create(idea="x", style="anime")
            root = Path(tmp) / record["working_dir"] / "idea2video"
            (root / "scene_0" / "shots" / "0").mkdir(parents=True, exist_ok=True)
            (root / "story.txt").write_text("story", encoding="utf-8")
            (root / "characters.json").write_text("[]", encoding="utf-8")
            (root / "script.json").write_text("[]", encoding="utf-8")
            (root / "scene_0" / "storyboard.json").write_text("[]", encoding="utf-8")
            (root / "scene_0" / "camera_tree.json").write_text("[]", encoding="utf-8")
            (root / "scene_0" / "shots" / "0" / "shot_description.json").write_text("{}", encoding="utf-8")
            adapter = ViMaxAdapters(Path(tmp), index)
            stdout = io.StringIO()
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()),                  patch("agent_runtime.vimax_adapters._build_image_generator", return_value=object()),                  patch("agent_runtime.vimax_adapters._build_video_generator", return_value=object()),                  patch("agent_runtime.vimax_adapters.Idea2VideoPipeline", NoisyRenderIdeaPipeline),                  contextlib.redirect_stdout(stdout):
                result = await adapter.vimax_render_video({"stop_after": "video", "allow_unlocked": True})
            self.assertTrue(result.ok)
            self.assertNotIn("NOISE_FROM_RENDER_PIPELINE", stdout.getvalue())

    async def test_render_dependency_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            index.create(idea="x")
            adapter = ViMaxAdapters(Path(tmp), index)
            result = await adapter.vimax_render_video({"stop_after": "video", "allow_unlocked": True})
            self.assertFalse(result.ok)
            self.assertEqual(result.metadata["error_type"], "dependency_missing")
