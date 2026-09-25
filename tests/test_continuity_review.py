"""Script coverage checks.

The live project these were written against has a film whose script has 11
dialogue lines, two of which were cut with their shots; the checks must name
exactly those two and must not invent gaps for the characters and runtime that
are still covered.
"""

from __future__ import annotations

import pytest

from pipelines.continuity_review import (
    Check,
    DialogueLine,
    absent_characters,
    build_evidence,
    characters_in_shots,
    dialogue_coverage,
    empty_cameras,
    film_order,
    load_review,
    normalize_dialogue,
    normalize_review,
    parse_runtime_range,
    run_checks,
    script_dialogue,
    staleness,
    write_review,
)

SCRIPT = """A heavyset man named Claude is on a couch eating chips.

WIFE (disappointed and sad): "Claude, we need to talk."

CLAUDE (distractedly, with mouth full): "What? I'm eating tokens and reasoning here!"

WIFE visibly steels herself. "Claude, I've met someone new."

CLAUDE (dismissively): "Whatever. You think you can do better than me, you go right ahead. Good luck with that!" <snorts and returns to eating chips>

DEEPSEEK enters the room and puts his arm around the WIFE.

<WIFE and DEEPSEEK walk off, hand in hand.>
"""


def _rows(*, visual="In the living room, <Claude> sits.", audio="[Speaker] Claude (Arrogantly): \"What? I'm eating tokens and reasoning here!\"", cam_idx=0):
    return {"visual_desc": visual, "audio_desc": audio, "cam_idx": cam_idx}


def _tree():
    return [
        {"idx": 0, "active_shot_idxs": [0, 3]},
        {"idx": 1, "active_shot_idxs": [1]},
        {"idx": 2, "active_shot_idxs": []},
    ]


def _characters():
    return [
        {"idx": 0, "identifier_in_scene": "Claude", "is_visible": True},
        {"idx": 1, "identifier_in_scene": "Wife", "is_visible": True},
        {"idx": 2, "identifier_in_scene": "DeepSeek", "is_visible": True},
        {"idx": 3, "identifier_in_scene": "Landlord", "is_visible": False},
        # never mentioned by the script, so its absence from the film is not a gap
        {"idx": 4, "identifier_in_scene": "Postman", "is_visible": True},
    ]


def test_script_dialogue_reads_every_line_form():
    lines = script_dialogue(SCRIPT)
    assert [line.speaker for line in lines] == ["WIFE", "CLAUDE", "WIFE", "CLAUDE"]
    # the trailing <stage direction> must not hide the line
    assert lines[-1].text.endswith("Good luck with that!")
    # the inline narrative form has no colon after the speaker
    assert lines[2].text == "Claude, I've met someone new."
    # stage directions are not dialogue
    assert all("walk off" not in line.text for line in lines)


def test_dialogue_coverage_matches_through_speaker_attribution():
    spoken = "[Speaker] Claude (Distractedly, with mouth full): \"What? I'm eating tokens and reasoning here!\""
    gaps = dialogue_coverage(script_dialogue(SCRIPT), {0: _rows(audio=spoken)}, [0])
    assert [gap.text for gap in gaps] == [
        "Claude, we need to talk.",
        "Claude, I've met someone new.",
        "Whatever. You think you can do better than me, you go right ahead. Good luck with that!",
    ]


def test_dialogue_coverage_names_the_removed_slot_that_carried_a_line():
    rows = {0: _rows(audio="[Speaker] Claude: \"What? I'm eating tokens and reasoning here!\"")}
    removed = {5: {"audio_desc": '[Speaker] Wife: "Claude, I\'ve met someone new."', "visual_desc": "", "cam_idx": 2}}
    gaps = dialogue_coverage(script_dialogue(SCRIPT), rows, [0], removed)
    cut = next(gap for gap in gaps if gap.text == "Claude, I've met someone new.")
    assert cut.removed_slots == [5]


def test_short_line_matches_exactly_not_by_containment():
    lines = [DialogueLine(speaker="WIFE", text="Goodbye.")]
    # "Goodbye." appears inside a longer line, which is a different line being spoken
    rows = {0: _rows(audio='[Speaker] Wife: "Sorry Claude, Goodbye. It is over."')}
    assert len(dialogue_coverage(lines, rows, [0])) == 1
    exact = {0: _rows(audio='[Speaker] Wife: "Goodbye."')}
    assert dialogue_coverage(lines, exact, [0]) == []


def test_film_order_follows_the_cameras_not_the_shot_numbers():
    assert film_order(_tree()) == [0, 3, 1]


def test_absent_characters_reports_a_character_the_script_uses_but_no_shot_shows():
    rows = {0: _rows(visual="In the living room, <Claude> sits.", audio='[Speaker] Claude: "What? I\'m eating tokens and reasoning here!"')}
    absent = absent_characters(_characters(), rows, [0], SCRIPT)
    # DeepSeek never speaks, but the script has him enter the room — still a gap
    assert absent == ["Wife", "DeepSeek"]
    # Claude is in the shot; the Landlord is off-screen by design; the Postman is
    # never used by the script, so the film not showing him is not a gap
    assert "Claude" not in absent and "Landlord" not in absent and "Postman" not in absent


def test_character_shown_only_by_plan_visible_index_still_counts():
    rows = {0: _rows(visual="In the living room, two people sit.")}
    plans = {0: {"ff_vis_char_idxs": [0, 2], "lf_vis_char_idxs": [0]}}
    assert characters_in_shots(rows, [0], plans, _characters()) == ["claude", "deepseek"]
    assert "DeepSeek" not in absent_characters(_characters(), rows, [0], SCRIPT, plans)


def test_empty_camera_is_reported():
    assert empty_cameras(_tree()) == [2]


def test_runtime_range_comes_from_the_requirement():
    assert parse_runtime_range("45-60 second comedic short.") == (45, 60)
    assert parse_runtime_range("no runtime stated") is None
    assert parse_runtime_range("60 to 45 second") == (45, 60)


def test_runtime_check_warns_outside_the_requested_range():
    checks = run_checks(
        script=SCRIPT, requirement="45-60 second comedic short.", characters=_characters(),
        camera_tree=_tree(), rows_by_idx={0: _rows()}, clip_seconds=5,
    )
    runtime = next(check for check in checks if check.id == "runtime")
    assert runtime.status == "warn" and "15s" in runtime.message
    inside = run_checks(
        script=SCRIPT, requirement="10-20 second comedic short.", characters=_characters(),
        camera_tree=_tree(), rows_by_idx={0: _rows()}, clip_seconds=5,
    )
    assert next(check for check in inside if check.id == "runtime").status == "ok"


def test_dialogue_check_reports_ok_when_nothing_is_missing():
    rows = {0: _rows(audio='[Speaker] Claude: "What? I\'m eating tokens and reasoning here!"')}
    checks = run_checks(
        script='CLAUDE: "What? I\'m eating tokens and reasoning here!"', requirement="", characters=[],
        camera_tree={"idx": 0, "active_shot_idxs": [0]} and [{"idx": 0, "active_shot_idxs": [0]}],
        rows_by_idx=rows,
    )
    assert [check.status for check in checks if check.id == "dialogue_uncovered"] == ["ok"]


def test_staleness_follows_the_shot_set_and_plan_edits():
    review = {"reviewed_at": "2026-09-22T10:00:00", "shots_reviewed": [0, 1, 2]}
    assert staleness(review, active_shots=[2, 1, 0], plan_mtimes={}) == (False, "")

    stale, reason = staleness(review, active_shots=[0, 1], plan_mtimes={})
    assert stale and "shots [2] removed" in reason

    stale, reason = staleness(
        review, active_shots=[0, 1, 2],
        plan_mtimes={"shots/1/shot_description.json": 1_800_000_000.0},
    )
    assert stale and "shots/1/shot_description.json" in reason
    # an edit older than the review does not invalidate it
    assert staleness(review, active_shots=[0, 1, 2], plan_mtimes={"camera_tree.json": 1.0})[0] is False


def test_staleness_without_a_review():
    stale, reason = staleness({}, active_shots=[0], plan_mtimes={})
    assert stale and reason


def test_normalize_review_keeps_identity_and_rejects_junk():
    review = normalize_review(
        {
            "beats": [
                {"text": "DeepSeek enters", "status": "missing", "covered_by": []},
                {"text": "   "},
                {"text": "Wife reveals someone new", "status": "nonsense", "covered_by": [5]},
                {"text": "Walk-off", "status": "covered", "covered_by": ["7"]},
            ],
            "suggestions": [
                {"visual_desc": "DeepSeek enters the room", "after_shot": "3", "characters": ["Wife", "DeepSeek"]},
                {"title": "no description"},
            ],
        },
        root="script2video", active_shots=[0, 3, 1], reviewed_at="2026-09-22T10:30:00",
    )
    assert review["root"] == "script2video" and review["shots_reviewed"] == [0, 3, 1]
    assert review["reviewed_at"] == "2026-09-22T10:30:00"
    assert [beat["text"] for beat in review["beats"]] == ["DeepSeek enters", "Wife reveals someone new", "Walk-off"]
    assert review["beats"][1]["status"] == "covered"  # a listed shot outranks a bad status
    assert review["beats"][0]["status"] == "missing"
    assert review["beats"][2]["covered_by"] == [7]
    assert len(review["suggestions"]) == 1
    assert review["suggestions"][0]["after_shot"] == 3
    assert review["suggestions"][0]["characters"] == ["Wife", "DeepSeek"]


def test_normalize_review_defaults_a_suggestion_position_to_the_last_shot():
    review = normalize_review(
        {"suggestions": [{"visual_desc": "An ending"}]},
        root="script2video", active_shots=[4, 9], reviewed_at="2026-09-22T10:30:00",
    )
    assert review["suggestions"][0]["after_shot"] == 9
    assert review["suggestions"][0]["id"] == "s1"


def test_a_suggestion_anchored_to_a_cut_shot_snaps_to_the_shot_it_now_follows():
    # A review anchors a restored beat to the shot it was cut from, which the film no longer
    # has; left alone, the suggestion could never be added.
    review = normalize_review(
        {"suggestions": [{"visual_desc": "restores the cut beat", "after_shot": 5}]},
        root="script2video", active_shots=[0, 3, 4, 9], reviewed_at="2026-09-22T10:30:00",
    )
    assert review["suggestions"][0]["after_shot"] == 4

    before_everything = normalize_review(
        {"suggestions": [{"visual_desc": "opens the film", "after_shot": 1}]},
        root="script2video", active_shots=[3, 4], reviewed_at="2026-09-22T10:30:00",
    )
    assert before_everything["suggestions"][0]["after_shot"] == 3

    no_shots = normalize_review(
        {"suggestions": [{"visual_desc": "x"}]}, root="script2video", active_shots=[], reviewed_at="2026-09-22T10:30:00"
    )
    assert no_shots["suggestions"][0]["after_shot"] == 0


def test_evidence_lists_shots_in_play_order_and_separates_cut_shots():
    evidence = build_evidence(
        script=SCRIPT, requirement="45-60 second comedic short.", characters=_characters(),
        camera_tree=_tree(), rows_by_idx={0: _rows(), 1: _rows(), 3: _rows()},
        removed_rows_by_idx={5: {"visual_desc": "Wife steels herself", "audio_desc": "", "cam_idx": 2}},
        plan_rows_by_idx={0: {"ff_desc": "Medium shot of the living room.", "lf_desc": "The same frame."}},
        checks=[Check(id="runtime", status="ok", message="15s")], clip_seconds=5,
    )
    assert [row["shot"] for row in evidence["timeline"]] == [0, 3, 1]
    assert [row["shot"] for row in evidence["cut_shots"]] == [5]
    # a shot carries the frames it will be drawn from, so suggestions can match the film's style
    assert evidence["timeline"][0]["frames"] == {"first": "Medium shot of the living room.", "last": "The same frame."}
    assert evidence["timeline"][1]["frames"] == {"first": None, "last": None}
    assert evidence["deterministic_findings"][0]["id"] == "runtime"
    assert evidence["clip_seconds"] == 5


def test_review_round_trips_through_disk(tmp_path):
    review = {"reviewed_at": "2026-09-22T10:30:00", "root": "script2video", "beats": []}
    write_review(str(tmp_path), review)
    assert load_review(str(tmp_path)) == review
    # a missing or unreadable file reads as "no review" rather than raising
    assert load_review(str(tmp_path / "elsewhere")) == {}
    (tmp_path / "continuity_review.json").write_text("{not json")
    assert load_review(str(tmp_path)) == {}


def test_normalize_dialogue_strips_attribution_and_punctuation():
    assert normalize_dialogue('[Speaker] Claude (Arrogantly): "Hey, someone\'s got to make the money!"') == (
        "hey someones got to make the money"
    )
    assert normalize_dialogue("[Sound Effect] Footsteps walking away") == "footsteps walking away"
