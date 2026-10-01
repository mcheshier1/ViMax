from __future__ import annotations

import json
import os
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class ProjectBusyError(RuntimeError):
    """Another process currently owns this project's writer lock."""


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OverflowError, OSError):
        # EPERM means a process exists but belongs to someone else. Unknown OS errors
        # are treated the same way: only ESRCH is evidence that recovery is safe.
        return True
    return True


def _read_owner(path: Path) -> tuple[int, str] | None:
    try:
        owner = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(owner, dict):
        return None
    pid, token = owner.get("pid"), owner.get("token")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0 or not isinstance(token, str) or not token:
        return None
    return pid, token


def _write_owner(path: Path, token: str) -> None:
    path.write_text(json.dumps({"pid": os.getpid(), "token": token}), encoding="utf-8")


def _remove_owned_dir(directory: Path, token: str) -> None:
    owner = _read_owner(directory / "owner.json")
    if owner == (os.getpid(), token):
        shutil.rmtree(directory, ignore_errors=True)


@contextmanager
def project_write_lock(working_dir: str | Path) -> Iterator[None]:
    """Exclusively hold the cross-runtime project lock for a mutation lifetime."""
    working_dir = Path(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)
    lock_dir = working_dir / ".timeline-write-lock"
    reclaim_dir = working_dir / ".timeline-write-lock-reclaim"
    token = uuid.uuid4().hex
    try:
        lock_dir.mkdir()
    except FileExistsError:
        owner = _read_owner(lock_dir / "owner.json")
        if owner is None or _pid_is_alive(owner[0]):
            raise ProjectBusyError("Project is busy: another writer holds .timeline-write-lock") from None

        # Reapers in both runtimes serialize before removing a dead lock. A contender
        # that sees this guard fails closed; the guard is never reclaimed if malformed
        # or orphaned, because deleting it could overlap a live reaper.
        try:
            reclaim_dir.mkdir()
        except FileExistsError:
            raise ProjectBusyError("Project is busy: stale writer lock recovery is already in progress") from None
        reclaim_token = uuid.uuid4().hex
        try:
            _write_owner(reclaim_dir / "owner.json", reclaim_token)
            current = _read_owner(lock_dir / "owner.json")
            if current != owner or _pid_is_alive(owner[0]):
                raise ProjectBusyError("Project is busy: writer lock owner changed during stale-lock recovery")
            shutil.rmtree(lock_dir)
            try:
                lock_dir.mkdir()
            except FileExistsError:
                # Another writer may win the gap after stale removal. Never touch the
                # newly-created lock; report busy and release only our reclaim guard.
                raise ProjectBusyError("Project is busy: another writer acquired the project") from None
        except ProjectBusyError:
            raise
        except OSError:
            raise ProjectBusyError("Project is busy: stale writer lock could not be safely recovered") from None
        finally:
            _remove_owned_dir(reclaim_dir, reclaim_token)

    try:
        _write_owner(lock_dir / "owner.json", token)
    except BaseException:
        # This directory was just created by this caller. If metadata cannot be
        # written it is safer to clean up our incomplete acquisition than leave a
        # permanent malformed lock behind.
        shutil.rmtree(lock_dir, ignore_errors=True)
        raise
    try:
        yield
    finally:
        _remove_owned_dir(lock_dir, token)
