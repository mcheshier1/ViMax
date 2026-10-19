from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from PIL import Image

from utils.text import safe_path_component

_SET_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_RENDER_ROOTS = {"script2video", "idea2video", "novel2video"}


def validate_set_candidate_count(candidate_count: Any, stop_after: str | None = None) -> int:
    if isinstance(candidate_count, bool) or not isinstance(candidate_count, int) or not 1 <= candidate_count <= 4:
        raise ValueError("set_candidate_count must be an integer from 1 to 4")
    if stop_after is not None and candidate_count != 1 and stop_after != "sets":
        raise ValueError("set_candidate_count other than 1 is only valid with stop_after='sets'")
    return candidate_count



def canonical_render_root(working_dir: str | Path) -> Path:
    path = Path(working_dir).resolve()
    return next((parent for parent in (path, *path.parents) if parent.name in _RENDER_ROOTS), path)


def project_context(working_dir: str | Path) -> tuple[Path, str]:
    root = canonical_render_root(working_dir)
    return root.parent, root.name


def _slot_sort(slot: str) -> tuple:
    return tuple((0, int(part)) if part.isdigit() else (1, part) for part in re.split(r"(\d+)", slot))


def shot_directories(render_root: str | Path) -> dict[str, Path]:
    root = Path(render_root)
    if (root / "shots").is_dir():
        found = {path.parent.name: path.parent for path in (root / "shots").glob("*/shot_description.json")
                 if not path.parent.name.startswith(".") and not path.parent.is_symlink()}
    elif root.name == "idea2video":
        found = {f"{path.parents[2].name}/{path.parent.name}": path.parent
                 for path in root.glob("scene_*/shots/*/shot_description.json")
                 if path.parents[2].name[6:].isdigit() and not path.parent.name.startswith(".")}
    elif root.name == "novel2video":
        found = {f"videos/{path.parents[3].name}/{path.parents[2].name}/{path.parent.name}": path.parent
                 for path in root.glob("videos/event_*/scene_*/shots/*/shot_description.json")
                 if path.parents[3].name[6:].isdigit() and path.parents[2].name[6:].isdigit()
                 and not path.parent.name.startswith(".")}
    else:
        found = {}
    return dict(sorted(found.items(), key=lambda pair: _slot_sort(pair[0])))


def active_shot_slots(render_root: str | Path) -> list[str]:
    return list(shot_directories(render_root))


def _validate_spec(data: Any, set_id: str, active: set[str] | None = None) -> dict[str, Any]:
    if not _SET_ID.fullmatch(set_id) or not isinstance(data, dict) or set(data) != {"id", "name", "description", "shots"}:
        raise ValueError("set.json requires a safe ID and exactly id, name, description, shots")
    if data["id"] != set_id or any(not isinstance(data[field], str) or not data[field].strip() for field in ("name", "description")):
        raise ValueError("set ID must match its directory; name and description must be nonblank strings")
    shots = data["shots"]
    if not isinstance(shots, list) or any(not isinstance(slot, str) for slot in shots) or len(shots) != len(set(shots)):
        raise ValueError("shots must be a list of unique slot strings")
    if active is not None and any(slot not in active for slot in shots):
        raise ValueError("set assignments must name active shot slots")
    return data


def read_set_specs(render_root: str | Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    directory = Path(render_root) / "set_references"
    specs, errors = {}, []
    if not directory.exists():
        return specs, errors
    if directory.is_symlink():
        return specs, ["Set reference directory must not be a symlink"]
    for item in sorted(directory.iterdir(), key=lambda path: path.name):
        if not item.is_dir():
            continue
        try:
            if item.is_symlink() or (item / "set.json").is_symlink():
                raise ValueError("set descriptor must not be a symlink")
            raw = (item / "set.json").read_bytes()
            data = _validate_spec(json.loads(raw), item.name)
            specs[item.name] = {**data, "spec_sha256": hashlib.sha256(raw).hexdigest()}
        except (OSError, ValueError) as exc:
            errors.append(f"Set {item.name}: {exc}")
    return specs, errors

def effective_set_ids(
    render_root: str | Path,
    slot: str,
    specs: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    root = Path(render_root)
    if specs is None:
        specs, errors = read_set_specs(root)
        if errors:
            raise ValueError("Correct set descriptions before resolving shot bindings: " + "; ".join(errors))
    directory = shot_directories(root).get(slot)
    if directory is None:
        raise ValueError(f"Unknown shot slot: {slot}")
    description = directory / "shot_description.json"
    if description.is_symlink():
        raise ValueError(f"Shot {slot} description must not be a symlink")
    try:
        data = json.loads(description.read_bytes())
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read shot {slot} set binding: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Shot {slot} description must be a JSON object")
    selected = data.get("set_ids")
    if selected is None:
        return [set_id for set_id, spec in specs.items() if slot in spec["shots"]]
    if not isinstance(selected, list) or any(not isinstance(set_id, str) or not _SET_ID.fullmatch(set_id) for set_id in selected):
        raise ValueError(f"Shot {slot} set_ids must be a list of safe set IDs")
    if len(set(selected)) != len(selected):
        raise ValueError(f"Shot {slot} set_ids must not contain duplicates")
    unknown = [set_id for set_id in selected if set_id not in specs]
    if unknown:
        raise ValueError(f"Shot {slot} set_ids name unknown set references: {', '.join(unknown)}")
    return selected


def _has_explicit_empty_set_binding(render_root: str | Path, slot: str) -> bool:
    directory = shot_directories(render_root).get(slot)
    if directory is None:
        return False
    try:
        data = json.loads((directory / "shot_description.json").read_bytes())
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and data.get("set_ids") == []




def validate_set_coverage(render_root: str | Path, slots: list[str] | None = None) -> tuple[dict[str, dict[str, Any]], list[str]]:
    specs, errors = read_set_specs(render_root)
    active = set(active_shot_slots(render_root))
    requested = active if slots is None else set(slots)
    unknown = requested - active
    if unknown:
        errors.append("Unknown shot slots: " + ", ".join(sorted(unknown, key=_slot_sort)))
    for set_id, spec in specs.items():
        invalid = set(spec["shots"]) - active
        if invalid:
            errors.append(f"Set {set_id} assigns inactive shots: " + ", ".join(sorted(invalid, key=_slot_sort)))
    uncovered = []
    for slot in sorted(requested & active, key=_slot_sort):
        try:
            if not effective_set_ids(render_root, slot, specs) and not _has_explicit_empty_set_binding(render_root, slot):
                uncovered.append(slot)
        except ValueError as exc:
            errors.append(str(exc))
    if uncovered:
        errors.append("Assign a set reference to these shots: " + ", ".join(uncovered))
    return specs, errors






def _approval_store(session_dir: Path) -> dict[str, Any]:
    try:
        store = json.loads((session_dir / "render_acceptance.json").read_bytes())
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot read set approvals: {exc}") from exc
    if not isinstance(store, dict):
        raise RuntimeError("Set approval store must be a JSON object")
    return store


def _verify_png(path: Path) -> None:
    with Image.open(path) as image:
        if image.format != "PNG" or image.width <= 0 or image.height <= 0:
            raise ValueError("set reference must be a valid PNG image")
        image.verify()


def _reference_paths(root: Path, set_id: str, image_model: str) -> tuple[Path, Path, Path]:
    base = root / "set_references" / set_id
    image = base / safe_path_component(image_model) / "reference.png"
    paths = (base / "set.json", image, image.with_name("provenance.json"))
    for path in paths:
        if not path.resolve().is_relative_to(root.resolve()) or any(
            parent.is_symlink() for parent in (path, *path.parents) if parent != root and parent.is_relative_to(root)
        ):
            raise RuntimeError(f"Set {set_id} reference paths must remain inside their authoritative directory without symlinks")
    return paths

def _candidate_paths(root: Path, set_id: str, image_model: str, candidate_id: str) -> tuple[Path, Path]:
    if not re.fullmatch(r"[0-9a-f]{32}", candidate_id):
        raise ValueError("candidate ID must be 32 lowercase hexadecimal characters")
    _, canonical, _ = _reference_paths(root, set_id, image_model)
    image = canonical.parent / f"candidate_{candidate_id}.png"
    provenance = image.with_suffix(".json")
    for path in (image, provenance):
        if not path.resolve().is_relative_to(root.resolve()) or any(
            parent.is_symlink() for parent in (path, *path.parents) if parent != root and parent.is_relative_to(root)
        ):
            raise RuntimeError(f"Set {set_id} candidate paths must remain inside their authoritative directory without symlinks")
    return image, provenance


def _candidate_inventory(root: Path, spec: dict[str, Any], style: str, image_model: str) -> tuple[list[Path], bool]:
    _, canonical, _ = _reference_paths(root, spec["id"], image_model)
    candidates: list[Path] = []
    stale = False
    for entry in sorted(canonical.parent.glob("candidate_*")):
        if entry.is_symlink():
            raise RuntimeError(f"Set {spec['id']} candidate paths must not be symlinks")
        match = re.fullmatch(r"candidate_([0-9a-f]{32})\.png", entry.name)
        if not match:
            if re.fullmatch(r"candidate_[0-9a-f]{32}\.json", entry.name):
                stale = True
            continue
        image, provenance_path = _candidate_paths(root, spec["id"], image_model, match.group(1))
        try:
            provenance = json.loads(provenance_path.read_bytes())
            expected = {
                "image_model": image_model,
                "style": style,
                "spec_sha256": spec["spec_sha256"],
                "candidate_id": match.group(1),
            }
            if not isinstance(provenance, dict) or any(provenance.get(key) != value for key, value in expected.items()):
                raise ValueError("candidate provenance is stale for the current model, style, or description")
            if not isinstance(provenance.get("prompt"), str) or not provenance["prompt"].strip():
                raise ValueError("candidate provenance has no prompt")
            _verify_png(image)
        except (OSError, ValueError):
            stale = True
            continue
        candidates.append(image)
    return candidates, stale


def _active_set_acceptance(store: dict[str, Any], root: Path, set_id: str) -> dict[str, Any] | None:
    root_record = store.get(root.name, {})
    records = root_record.get("sets", {}) if isinstance(root_record, dict) else {}
    record = records.get(set_id) if isinstance(records, dict) else None
    if isinstance(record, dict) and record.get("accepted_at") and not record.get("rejected_at") and not record.get("invalidated_at"):
        return record
    return None


def _generated_reference(root: Path, spec: dict[str, Any], style: str, image_model: str) -> Path:
    _, image, provenance_path = _reference_paths(root, spec["id"], image_model)
    try:
        provenance = json.loads(provenance_path.read_bytes())
        expected = {"image_model": image_model, "style": style, "spec_sha256": spec["spec_sha256"]}
        if not isinstance(provenance, dict) or any(provenance.get(key) != value for key, value in expected.items()):
            raise ValueError("reference provenance is stale for the current model, style, or description")
        _verify_png(image)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Set {spec['id']}: {exc}") from exc
    return image


def _accepted_reference(root: Path, spec: dict[str, Any], style: str, image_model: str, store: dict[str, Any]) -> Path:
    image = _generated_reference(root, spec, style, image_model)
    root_record = store.get(root.name, {})
    records = root_record.get("sets", {}) if isinstance(root_record, dict) else {}
    record = records.get(spec["id"]) if isinstance(records, dict) else None
    if not isinstance(record, dict) or not record.get("accepted_at") or record.get("rejected_at") or record.get("invalidated_at"):
        raise RuntimeError(f"Set {spec['id']} is not accepted; review it before generating shot images")
    expected = {path.resolve() for path in _reference_paths(root, spec["id"], image_model)}
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(expected):
        raise RuntimeError(f"Set {spec['id']} approval must fingerprint its description, image, and provenance")
    observed = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
            raise RuntimeError(f"Set {spec['id']} has an invalid approval artifact")
        relative = Path(artifact["path"])
        path = (root.parent / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or path not in expected or path in observed:
            raise RuntimeError(f"Set {spec['id']} approval references an unexpected artifact")
        raw = path.read_bytes()
        if artifact.get("size") != len(raw) or artifact.get("sha256") != hashlib.sha256(raw).hexdigest():
            raise RuntimeError(f"Set {spec['id']} approval fingerprint is stale")
        observed.add(path)
    if observed != expected:
        raise RuntimeError(f"Set {spec['id']} approval artifacts are incomplete")
    return image


def accepted_set_references(working_dir: str | Path, slot: str, style: str, image_model: str) -> list[tuple[str, str]]:
    root = canonical_render_root(working_dir)
    specs, errors = validate_set_coverage(root, [slot])
    if errors:
        raise RuntimeError("Set references are not ready: " + "; ".join(errors))
    set_ids = effective_set_ids(root, slot, specs)
    if not set_ids:
        return []
    store = _approval_store(root.parent)
    references = []
    for set_id in set_ids:
        spec = specs[set_id]
        image = _accepted_reference(root, spec, style, image_model, store)
        references.append((str(image), f"Approved set reference: {spec['name']}. Preserve its architecture, layout, props and lighting. {spec['description']}"))
    return references



def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


async def plan_set_specs(chat_model: Any, render_root: str | Path) -> dict[str, dict[str, Any]]:
    root = canonical_render_root(render_root)
    existing, errors = read_set_specs(root)
    if errors:
        raise RuntimeError("Correct existing set descriptors before planning: " + "; ".join(errors))
    if existing:
        return existing
    directories = shot_directories(root)
    if not directories:
        raise RuntimeError("Plan shots before preparing set references")
    shots = []
    for slot, directory in directories.items():
        data = json.loads((directory / "shot_description.json").read_bytes())
        shots.append({"slot": slot, "visual": data.get("visual_desc", ""), "first_frame": data.get("ff_desc", ""), "last_frame": data.get("lf_desc", "")})
    source_paths = [root / "script.txt", root / "script.json"]
    if root.name == "novel2video":
        source_paths.extend(sorted(root.glob("scenes/event_*/scene_*.json")))
    source = "\n\n".join(path.read_text(encoding="utf-8") for path in source_paths if path.is_file())
    response = await chat_model.ainvoke(
        "Plan distinct filming sets from the actual script and planned shots. Return JSON only, no Markdown: "
        '{"sets":[{"id":"safe_ascii_id","name":"location label","description":"empty set architecture/layout/props/lighting","shots":["exact slot"]}]}. '
        "Use safe lowercase ASCII IDs, distinct interiors/exteriors, and only locations actually present in the source. "
        "Describe a character-free set, not character action or portraits. Preserve spatial anchors such as doors, the bar, tables and windows. "
        "Assign every listed shot to at least one matching set; use multiple references only when the shot actually spans locations. "
        "Do not invent story details or copy character profiles.\nSCRIPT:\n" + source + "\nSHOT PLAN:\n" + json.dumps(shots, ensure_ascii=False)
    )
    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    try:
        payload = json.loads(str(content))
        if not isinstance(payload, dict) or not isinstance(payload.get("sets"), list) or not payload["sets"]:
            raise ValueError("expected a nonempty sets array")
        prepared = {}
        active = set(directories)
        for item in payload["sets"]:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                raise ValueError("each set requires an ID")
            spec = _validate_spec(item, item["id"], active)
            if spec["id"] in prepared:
                raise ValueError("duplicate set IDs")
            prepared[spec["id"]] = spec
        if active - {slot for spec in prepared.values() for slot in spec["shots"]}:
            raise ValueError("every planned shot requires a matching set")
    except (ValueError, TypeError) as exc:
        raise RuntimeError(f"Set planning returned an invalid plan: {exc}") from exc
    # A user may have saved a manual set while the model was running.
    existing, errors = read_set_specs(root)
    if errors:
        raise RuntimeError("Set descriptions changed during planning: " + "; ".join(errors))
    if existing:
        return existing
    for set_id, spec in prepared.items():
        _atomic_json(root / "set_references" / set_id / "set.json", spec)
    return read_set_specs(root)[0]


def _invalidate_bound_shots(store: dict[str, Any], root: Path, spec: dict[str, Any], note: str) -> None:
    root_record = store.setdefault(root.name, {})
    shots = root_record.get("shots", {})
    timestamp = datetime.now(timezone.utc).isoformat()
    specs, errors = read_set_specs(root)
    if errors:
        raise RuntimeError("Cannot invalidate set-bound shots: " + "; ".join(errors))
    bound_slots = []
    for slot in shot_directories(root):
        try:
            if spec["id"] in effective_set_ids(root, slot, specs):
                bound_slots.append(slot)
        except ValueError as exc:
            raise RuntimeError(f"Cannot resolve shot {slot} set binding during invalidation: {exc}") from exc
    for slot in bound_slots:
        row = shots.get(slot, {}) if isinstance(shots, dict) else {}
        for stage in ("keyframes", "clips"):
            record = row.get(stage)
            if isinstance(record, dict):
                record.update(invalidated_at=timestamp, invalidated_reason=note)
    final = root_record.get("final_video")
    if isinstance(final, dict):
        final.update(invalidated_at=timestamp, invalidated_reason=note)
    set_record = root_record.setdefault("sets", {}).get(spec["id"], {})
    old_reason = set_record.get("reason", "") if isinstance(set_record, dict) else ""
    root_record["sets"][spec["id"]] = {"redone_at": timestamp, **({"note": old_reason} if old_reason else {})}


def _prepare_set_generation(
    working_dir: str | Path,
    image_generator: Any,
    style: str,
    *,
    set_ids: list[str] | None = None,
    redo_sets: list[str] | None = None,
    set_candidate_count: int = 1,
):
    validate_set_candidate_count(set_candidate_count)
    root = canonical_render_root(working_dir)
    specs, errors = read_set_specs(root)
    if errors:
        raise RuntimeError("Correct set descriptions before generation: " + "; ".join(errors))
    if not specs:
        raise RuntimeError("Prepare and review the set reference plan before generating set images")
    for slot in shot_directories(root):
        try:
            effective_set_ids(root, slot, specs)
        except ValueError as exc:
            raise RuntimeError(f"Correct shot set bindings before generating set images: {exc}") from exc
    if not isinstance(set_ids, list) or not set_ids or any(
        not isinstance(value, str) or not _SET_ID.fullmatch(value) or value not in specs for value in set_ids
    ) or len(set(set_ids)) != len(set_ids):
        raise ValueError("set_ids must explicitly name unique existing set references")
    if redo_sets is None:
        redo_sets = []
    if not isinstance(redo_sets, list) or any(
        not isinstance(value, str) or not _SET_ID.fullmatch(value) for value in redo_sets
    ) or len(set(redo_sets)) != len(redo_sets):
        raise ValueError("redo_sets must be a list of unique safe set IDs")
    redo = set(redo_sets)
    if not redo.issubset(set_ids):
        raise ValueError("redo_sets must be a subset of selected set_ids")
    if not isinstance(style, str) or not style.strip():
        raise ValueError("Confirm the project style before generating set references")
    image_model = str(getattr(image_generator, "model", "") or "unknown-image-model")
    store = _approval_store(root.parent)
    root_record = store.get(root.name, {})
    if not isinstance(root_record, dict) or not isinstance(root_record.get("sets", {}), dict) or not isinstance(root_record.get("shots", {}), dict):
        raise RuntimeError("Set approval records must be JSON objects")
    jobs = []
    ready = []
    # Validate the whole decision before any image request, not half a batch later.
    for set_id in set_ids:
        spec = specs[set_id]
        _, canonical_image, _ = _reference_paths(root, set_id, image_model)
        active_approval = _active_set_acceptance(store, root, set_id)
        if active_approval is not None:
            if set_id in redo:
                raise RuntimeError(f"Set {set_id} is accepted; unlock it before generating candidates")
            try:
                accepted = _accepted_reference(root, spec, style, image_model, store)
            except RuntimeError as exc:
                raise RuntimeError(f"Set {set_id} is accepted; unlock it before requesting new candidates: {exc}") from exc
            ready.append(str(accepted))
            continue
        candidates, stale_candidates = _candidate_inventory(root, spec, style, image_model)
        if set_id in redo:
            jobs.append(spec)
            continue
        if candidates:
            ready.extend(str(path) for path in candidates)
            continue
        try:
            generated = _generated_reference(root, spec, style, image_model)
        except RuntimeError:
            generated = None
        if generated is not None:
            ready.append(str(generated))
            continue
        if canonical_image.exists() or stale_candidates:
            raise RuntimeError(f"Set {set_id} has stale images; explicitly include it in redo_sets before generating candidates")
        jobs.append(spec)
    return root, image_model, store, jobs, ready


def validate_set_reference_generation(
    working_dir: str | Path,
    image_generator: Any,
    style: str,
    *,
    set_ids: list[str] | None = None,
    redo_sets: list[str] | None = None,
    set_candidate_count: int = 1,
) -> None:
    """Validate a complete paid decision before any render cutover or image request."""
    _prepare_set_generation(
        working_dir,
        image_generator,
        style,
        set_ids=set_ids,
        redo_sets=redo_sets,
        set_candidate_count=set_candidate_count,
    )


async def generate_set_references(
    working_dir: str | Path,
    image_generator: Any,
    style: str,
    *,
    set_ids: list[str] | None = None,
    redo_sets: list[str] | None = None,
    set_candidate_count: int = 1,
    progress: Callable | None = None,
) -> list[str]:
    root, image_model, store, jobs, ready = _prepare_set_generation(
        working_dir,
        image_generator,
        style,
        set_ids=set_ids,
        redo_sets=redo_sets,
        set_candidate_count=set_candidate_count,
    )
    for spec in jobs:
        set_id = spec["id"]
        descriptor, _, _ = _reference_paths(root, set_id, image_model)
        records = store.get(root.name, {}).get("sets", {})
        record = records.get(set_id, {}) if isinstance(records, dict) else {}
        reason = str(record.get("reason", "")) if isinstance(record, dict) else ""
        if progress:
            progress(
                "set_reference_start",
                f"Generating {set_candidate_count} set reference candidate(s): {spec['name']}",
                {"set_id": set_id, "phase": "sets", "candidate_count": set_candidate_count},
            )
        for candidate_index in range(1, set_candidate_count + 1):
            candidate_id = uuid4().hex
            if hashlib.sha256(descriptor.read_bytes()).hexdigest() != spec["spec_sha256"]:
                raise RuntimeError(f"Set {set_id} changed before generation; existing candidates were preserved")
            latest = _approval_store(root.parent)
            if _active_set_acceptance(latest, root, set_id) is not None:
                raise RuntimeError(f"Set {set_id} was accepted before generation; new candidates were not requested")
            prompt = (
                f"Create a character-free set/location reference image of {spec['name']}.\n"
                f"Set direction: {spec['description']}\nStyle: {style}\n"
                f"Alternative design {candidate_index} of {set_candidate_count}: make this a distinct, coherent visual treatment "
                "of the same location while preserving its described spatial layout and architectural anchors. "
                "Use a single wide establishing view, not a collage or multiple camera angles. "
                "Preserve the stated doors, windows, furniture and prop placement. "
                "No people, characters, silhouettes, dialogue, labels, or text overlays."
                + (f"\nReview direction: {reason}" if reason else "")
            )
            output = await image_generator.generate_single_image(
                prompt=prompt,
                reference_image_paths=[],
                aspect_ratio="16:9",
            )
            if hashlib.sha256(descriptor.read_bytes()).hexdigest() != spec["spec_sha256"]:
                raise RuntimeError(f"Set {set_id} changed during generation; previous candidates were preserved")
            latest = _approval_store(root.parent)
            if _active_set_acceptance(latest, root, set_id) is not None:
                raise RuntimeError(f"Set {set_id} was accepted during generation; new candidates were not stored")

            image_path, provenance_path = _candidate_paths(root, set_id, image_model, candidate_id)
            image_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_image = image_path.with_name(f".candidate.{uuid4().hex}.tmp.png")
            temporary_provenance = provenance_path.with_name(f".candidate.{uuid4().hex}.tmp.json")
            provenance = {
                "image_model": image_model,
                "style": style,
                "spec_sha256": spec["spec_sha256"],
                "candidate_id": candidate_id,
                "prompt": prompt,
            }
            try:
                output.save(str(temporary_image))
                _verify_png(temporary_image)
                temporary_provenance.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
                try:
                    os.link(temporary_image, image_path)
                    os.link(temporary_provenance, provenance_path)
                except FileExistsError as exc:
                    raise RuntimeError(f"Set {set_id} candidate ID collision; existing candidate files were preserved") from exc
            finally:
                temporary_image.unlink(missing_ok=True)
                temporary_provenance.unlink(missing_ok=True)
            ready.append(str(image_path))
            if progress:
                progress(
                    "set_reference_done",
                    f"Set reference candidate ready for review: {spec['name']}",
                    {
                        "set_id": set_id,
                        "candidate_id": candidate_id,
                        "candidate_index": candidate_index,
                        "phase": "sets",
                        "path": str(image_path),
                    },
                )
            store = latest
    return ready
