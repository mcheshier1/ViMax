import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

from PIL import Image

from interfaces import ImageOutput
from pipelines.set_references import (
    _invalidate_bound_shots,
    accepted_set_references,
    active_shot_slots,
    effective_set_ids,
    generate_set_references,
    plan_set_specs,
    read_set_specs,
    validate_set_candidate_count,
    validate_set_coverage,
)
from utils.text import safe_path_component


class SetReferenceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.session = Path(self.temporary.name)
        self.root = self.session / "script2video"
        self.root.mkdir()
        self.model = "bytedance-seed/seedream-5-0-flash"
        self.style = "cinematic western"
        for slot, location in (("0", "SAL OON"), ("1", "SHERIFF OFFICE")):
            directory = self.root / "shots" / slot
            directory.mkdir(parents=True)
            (directory / "shot_description.json").write_text(json.dumps({"idx": int(slot), "visual_desc": location, "ff_desc": location, "lf_desc": location}))
        (self.root / "script.txt").write_text("INT. SALOON - DAY\nALICE stands at the bar.\nINT. SHERIFF OFFICE - DAY\nALICE studies a poster.")
        self.write_spec("saloon", ["0"])
        self.write_spec("office", ["1"])

    def write_spec(self, set_id, shots, description="An empty room with a door on the left"):
        path = self.root / "set_references" / set_id / "set.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"id": set_id, "name": set_id.title(), "description": description, "shots": shots}))
        return path

    def generated(self, set_id):
        specs, errors = read_set_specs(self.root)
        self.assertEqual(errors, [])
        directory = self.root / "set_references" / set_id / safe_path_component(self.model)
        directory.mkdir(parents=True, exist_ok=True)
        image = directory / "reference.png"
        Image.new("RGB", (32, 18), "blue").save(image)
        provenance = directory / "provenance.json"
        provenance.write_text(json.dumps({"image_model": self.model, "style": self.style, "spec_sha256": specs[set_id]["spec_sha256"]}))
        return image, provenance

    def approve(self, set_id):
        image, provenance = self.generated(set_id)
        paths = [self.root / "set_references" / set_id / "set.json", image, provenance]
        store_path = self.session / "render_acceptance.json"
        store = json.loads(store_path.read_text()) if store_path.exists() else {}
        store.setdefault("script2video", {}).setdefault("sets", {})[set_id] = {
            "accepted_at": "2026-01-01T00:00:00Z",
            "artifacts": [{"path": path.relative_to(self.session).as_posix(), "size": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths],
        }
        store_path.write_text(json.dumps(store))
        return image

    def generator(self):
        class Generator:
            model = "bytedance-seed/seedream-5-0-flash"
            generate_single_image = AsyncMock(return_value=ImageOutput(fmt="pil", ext="png", data=Image.new("RGB", (32, 18), "green")))
        return Generator()

    def test_scoped_approval_uses_matching_location_only(self):
        image = self.approve("saloon")
        references = accepted_set_references(self.root, "0", self.style, self.model)
        self.assertEqual([path for path, _ in references], [str(image)])
        with self.assertRaisesRegex(RuntimeError, "office"):
            accepted_set_references(self.root, "1", self.style, self.model)

    def test_explicit_empty_binding_opts_out_but_missing_or_null_still_need_defaults(self):
        shot = self.root / "shots" / "0" / "shot_description.json"
        shot.write_text(json.dumps({"idx": 0, "set_ids": []}))
        self.assertEqual(effective_set_ids(self.root, "0"), [])
        self.write_spec("saloon", [])
        self.assertEqual(validate_set_coverage(self.root, ["0"])[1], [])
        self.assertEqual(accepted_set_references(self.root, "0", self.style, self.model), [])

        for description in ({"idx": 0}, {"idx": 0, "set_ids": None}):
            with self.subTest(description=description):
                shot.write_text(json.dumps(description))
                self.assertTrue(validate_set_coverage(self.root, ["0"])[1])
                with self.assertRaisesRegex(RuntimeError, "Assign a set reference"):
                    accepted_set_references(self.root, "0", self.style, self.model)

        shot.write_text(json.dumps({"idx": 0, "set_ids": "saloon"}))
        with self.assertRaisesRegex(RuntimeError, "must be a list of safe set IDs"):
            accepted_set_references(self.root, "0", self.style, self.model)

    def test_bound_rejected_set_still_blocks_still_reference_admission(self):
        self.approve("saloon")
        store_path = self.session / "render_acceptance.json"
        store = json.loads(store_path.read_text())
        store["script2video"]["sets"]["saloon"]["rejected_at"] = "2026-01-02T00:00:00Z"
        store_path.write_text(json.dumps(store))
        with self.assertRaisesRegex(RuntimeError, "not accepted"):
            accepted_set_references(self.root, "0", self.style, self.model)

    def test_scoped_coverage_does_not_require_unrelated_unassigned_shots(self):
        self.write_spec("office", [])
        self.approve("saloon")
        self.assertEqual(validate_set_coverage(self.root, ["0"])[1], [])
        self.assertTrue(validate_set_coverage(self.root)[1])
        self.assertEqual(len(accepted_set_references(self.root, "0", self.style, self.model)), 1)
    def test_description_image_and_approval_mutations_fail_closed(self):
        image = self.approve("saloon")
        image.write_bytes(b"not a PNG")
        with self.assertRaises(RuntimeError):
            accepted_set_references(self.root, "0", self.style, self.model)
        self.approve("saloon")
        self.write_spec("saloon", ["0"], "A new bar along the right wall")
        with self.assertRaisesRegex(RuntimeError, "stale"):
            accepted_set_references(self.root, "0", self.style, self.model)
        self.approve("saloon")
        store_path = self.session / "render_acceptance.json"
        store = json.loads(store_path.read_text())
        store["script2video"]["sets"]["saloon"]["invalidated_at"] = "later"
        store_path.write_text(json.dumps(store))
        with self.assertRaisesRegex(RuntimeError, "not accepted"):
            accepted_set_references(self.root, "0", self.style, self.model)

    def test_approval_cannot_read_unexpected_paths(self):
        self.approve("saloon")
        store_path = self.session / "render_acceptance.json"
        store = json.loads(store_path.read_text())
        store["script2video"]["sets"]["saloon"]["artifacts"][0]["path"] = "../must-not-be-read.json"
        store_path.write_text(json.dumps(store))
        with self.assertRaisesRegex(RuntimeError, "unexpected artifact"):
            accepted_set_references(self.root, "0", self.style, self.model)

    async def test_accepted_redraw_and_invalid_scopes_make_no_generation_request(self):
        self.approve("saloon")
        generator = self.generator()
        for selected, redo in ((["saloon"], ["saloon"]), (["unknown"], []), (["saloon"], ["office"]), ([], [])):
            with self.subTest(selected=selected, redo=redo):
                with self.assertRaises((RuntimeError, ValueError)):
                    await generate_set_references(self.root, generator, self.style, set_ids=selected, redo_sets=redo)
        generator.generate_single_image.assert_not_awaited()


    async def test_current_canonical_reference_is_reused_with_or_without_approval(self):
        canonical, _ = self.generated("saloon")
        generator = self.generator()

        reused = await generate_set_references(self.root, generator, self.style, set_ids=["saloon"])
        self.assertEqual(reused, [str(canonical)])

        self.approve("saloon")
        accepted_reused = await generate_set_references(self.root, generator, self.style, set_ids=["saloon"])
        self.assertEqual(accepted_reused, [str(canonical)])
        generator.generate_single_image.assert_not_awaited()

    async def test_symlinked_output_directory_refuses_generation_without_writing_outside_root(self):
        outside = self.session / "outside"
        outside.mkdir()
        image_directory = self.root / "set_references" / "saloon" / safe_path_component(self.model)
        image_directory.symlink_to(outside, target_is_directory=True)
        generator = self.generator()
        with self.assertRaisesRegex(RuntimeError, "without symlinks"):
            await generate_set_references(self.root, generator, self.style, set_ids=["saloon"], redo_sets=["saloon"])
        generator.generate_single_image.assert_not_awaited()
        self.assertEqual(list(outside.iterdir()), [])

    async def test_failed_explicit_redraw_preserves_old_reference_and_provenance(self):
        image, provenance = self.generated("saloon")
        previous = (image.read_bytes(), provenance.read_bytes())
        generator = self.generator()
        generator.generate_single_image.side_effect = RuntimeError("provider unavailable")
        with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
            await generate_set_references(self.root, generator, self.style, set_ids=["saloon"], redo_sets=["saloon"])
        self.assertEqual((image.read_bytes(), provenance.read_bytes()), previous)

    async def test_spec_changed_during_generation_preserves_previous_image(self):
        image, provenance = self.generated("saloon")
        previous = (image.read_bytes(), provenance.read_bytes())
        generator = self.generator()
        async def change_spec(**kwargs):
            self.write_spec("saloon", ["0"], "The user moved the door to the right")
            return ImageOutput(fmt="pil", ext="png", data=Image.new("RGB", (32, 18), "green"))
        generator.generate_single_image.side_effect = change_spec
        with self.assertRaisesRegex(RuntimeError, "changed during generation"):
            await generate_set_references(self.root, generator, self.style, set_ids=["saloon"], redo_sets=["saloon"])
        self.assertEqual((image.read_bytes(), provenance.read_bytes()), previous)

    async def test_candidate_count_generates_distinct_files_without_mutating_canonical_state(self):
        canonical, provenance = self.generated("saloon")
        previous_reference = (canonical.read_bytes(), provenance.read_bytes())
        store_path = self.session / "render_acceptance.json"
        store_path.write_text(json.dumps({
            "script2video": {
                "shots": {"0": {"keyframes": {"accepted_at": "before"}}},
                "final_video": {"accepted_at": "before"},
            }
        }))
        previous_acceptance = store_path.read_bytes()
        generator = self.generator()

        references = await generate_set_references(
            self.root,
            generator,
            self.style,
            set_ids=["saloon"],
            redo_sets=["saloon"],
            set_candidate_count=2,
        )

        self.assertEqual(len(references), 2)
        self.assertEqual(generator.generate_single_image.await_count, 2)
        self.assertEqual((canonical.read_bytes(), provenance.read_bytes()), previous_reference)
        self.assertEqual(store_path.read_bytes(), previous_acceptance)
        specs, errors = read_set_specs(self.root)
        self.assertEqual(errors, [])
        candidate_paths = [Path(path) for path in references]
        self.assertEqual(len({path.name for path in candidate_paths}), 2)
        for image in candidate_paths:
            self.assertRegex(image.name, r"candidate_[0-9a-f]{32}\.png")
            metadata = json.loads(image.with_suffix(".json").read_text())
            candidate_id = image.stem.removeprefix("candidate_")
            self.assertEqual(metadata["candidate_id"], candidate_id)
            self.assertEqual(metadata["image_model"], self.model)
            self.assertEqual(metadata["style"], self.style)
            self.assertEqual(metadata["spec_sha256"], specs["saloon"]["spec_sha256"])
            self.assertTrue(metadata["prompt"])

    async def test_existing_candidates_are_reused_without_paid_generation(self):
        first_generator = self.generator()
        first = await generate_set_references(
            self.root,
            first_generator,
            self.style,
            set_ids=["saloon"],
            set_candidate_count=2,
        )
        reuse_generator = self.generator()

        reused = await generate_set_references(
            self.root,
            reuse_generator,
            self.style,
            set_ids=["saloon"],
            set_candidate_count=2,
        )

        self.assertEqual(set(reused), set(first))
        reuse_generator.generate_single_image.assert_not_awaited()

    async def test_redo_appends_candidates_instead_of_overwriting_old_alternatives(self):
        first = await generate_set_references(
            self.root, self.generator(), self.style, set_ids=["saloon"]
        )
        previous = {Path(path): Path(path).read_bytes() for path in first}

        await generate_set_references(
            self.root,
            self.generator(),
            self.style,
            set_ids=["saloon"],
            redo_sets=["saloon"],
        )

        for path, contents in previous.items():
            self.assertEqual(path.read_bytes(), contents)
        all_candidates = list((self.root / "set_references" / "saloon" / safe_path_component(self.model)).glob("candidate_*.png"))
        self.assertEqual(len(all_candidates), 2)

    async def test_candidate_count_is_bounded_and_overrides_follow_shot_descriptions(self):
        self.assertEqual(validate_set_candidate_count(1), 1)
        self.assertEqual(validate_set_candidate_count(4, "sets"), 4)
        for invalid in (True, 0, 5, 1.5, "2"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_set_candidate_count(invalid)
        with self.assertRaises(ValueError):
            validate_set_candidate_count(2, "stills")

        shot_zero = self.root / "shots" / "0" / "shot_description.json"
        shot_one = self.root / "shots" / "1" / "shot_description.json"
        self.assertEqual(effective_set_ids(self.root, "0"), ["saloon"])
        shot_zero.write_text(json.dumps({"idx": 0, "set_ids": ["office"]}))
        shot_one.write_text(json.dumps({"idx": 1, "set_ids": []}))
        specs, errors = read_set_specs(self.root)
        self.assertEqual(errors, [])
        self.assertEqual(effective_set_ids(self.root, "0", specs), ["office"])
        self.assertEqual(effective_set_ids(self.root, "1", specs), [])

        shot_zero.write_text(json.dumps({"idx": 0, "set_ids": ["office", "office"]}))
        with self.assertRaisesRegex(RuntimeError, "duplicates"):
            accepted_set_references(self.root, "0", self.style, self.model)
        shot_zero.write_text(json.dumps({"idx": 0, "set_ids": ["unknown"]}))
        with self.assertRaisesRegex(RuntimeError, "unknown set references"):
            accepted_set_references(self.root, "0", self.style, self.model)
        generator = self.generator()
        with self.assertRaisesRegex(RuntimeError, "unknown set references"):
            await generate_set_references(self.root, generator, self.style, set_ids=["saloon"])
        generator.generate_single_image.assert_not_awaited()
        shot_zero.write_text(json.dumps({"idx": 0, "set_ids": ["../escape"]}))
        generator = self.generator()
        with self.assertRaisesRegex(RuntimeError, "safe set IDs"):
            await generate_set_references(self.root, generator, self.style, set_ids=["saloon"])
        generator.generate_single_image.assert_not_awaited()

    def test_invalidation_uses_effective_set_overrides(self):
        shot_zero = self.root / "shots" / "0" / "shot_description.json"
        shot_one = self.root / "shots" / "1" / "shot_description.json"
        shot_zero.write_text(json.dumps({"idx": 0, "set_ids": ["office"]}))
        shot_one.write_text(json.dumps({"idx": 1, "set_ids": ["saloon"]}))
        specs, errors = read_set_specs(self.root)
        self.assertEqual(errors, [])
        before = {"accepted_at": "before"}
        store = {
            "script2video": {
                "shots": {
                    "0": {"keyframes": dict(before), "clips": dict(before)},
                    "1": {"keyframes": dict(before), "clips": dict(before)},
                }
            }
        }

        _invalidate_bound_shots(store, self.root, specs["saloon"], "set changed")

        shots = store["script2video"]["shots"]
        self.assertEqual(shots["0"]["keyframes"], before)
        self.assertEqual(shots["0"]["clips"], before)
        self.assertIn("invalidated_at", shots["1"]["keyframes"])
        self.assertIn("invalidated_at", shots["1"]["clips"])

    async def test_invalid_later_planned_set_does_not_persist_a_partial_plan(self):
        for path in (self.root / "set_references").glob("*/set.json"):
            path.unlink()
            path.parent.rmdir()
        class Model:
            async def ainvoke(self, prompt):
                return json.dumps({"sets": [
                    {"id": "saloon", "name": "Saloon", "description": "An empty bar", "shots": ["0"]},
                    {"id": "../office", "name": "Office", "description": "A desk", "shots": ["1"]},
                ]})
        with self.assertRaisesRegex(RuntimeError, "invalid plan"):
            await plan_set_specs(Model(), self.root)
        self.assertEqual(list((self.root / "set_references").glob("*/set.json")), [])

    def test_idea_and_novel_slots_preserve_scene_identity_without_archive_leakage(self):
        idea = self.session / "idea2video"
        novel = self.session / "novel2video"
        for directory in (idea / "scene_0" / "shots" / "0", idea / "scene_1" / "shots" / "0", novel / "videos" / "event_2" / "scene_3" / "shots" / "0"):
            directory.mkdir(parents=True)
            (directory / "shot_description.json").write_text("{}")
        removed = idea / "scene_0" / ".removed_shots" / "7"
        removed.mkdir(parents=True)
        (removed / "shot_description.json").write_text("{}")
        (idea / "scene_0" / "shots" / "0" / "shot_description.json").write_text(json.dumps({"set_ids": ["barn"]}))

        for root, set_id, shots in (
            (idea, "barn", []),
            (idea, "room", ["scene_1/0"]),
            (novel, "cellar", ["videos/event_2/scene_3/0"]),
        ):
            descriptor = root / "set_references" / set_id / "set.json"
            descriptor.parent.mkdir(parents=True, exist_ok=True)
            descriptor.write_text(json.dumps({
                "id": set_id,
                "name": set_id.title(),
                "description": "A quiet, character-free location.",
                "shots": shots,
            }))

        self.assertEqual(active_shot_slots(idea), ["scene_0/0", "scene_1/0"])
        self.assertEqual(active_shot_slots(novel), ["videos/event_2/scene_3/0"])
        self.assertEqual(effective_set_ids(idea, "scene_0/0"), ["barn"])
        self.assertEqual(effective_set_ids(idea, "scene_1/0"), ["room"])
        self.assertEqual(effective_set_ids(novel, "videos/event_2/scene_3/0"), ["cellar"])


if __name__ == "__main__":
    unittest.main()
