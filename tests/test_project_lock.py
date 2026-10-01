import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from utils.project_lock import ProjectBusyError, project_write_lock


class ProjectWriteLockTests(unittest.TestCase):
    def test_competing_cross_runtime_owner_blocks_and_preserves_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            working = Path(tmp)
            lock = working / ".timeline-write-lock"
            with project_write_lock(working):
                owner = json.loads((lock / "owner.json").read_text(encoding="utf-8"))
                self.assertEqual(owner["pid"], os.getpid())
                self.assertTrue(owner["token"])
                with self.assertRaises(ProjectBusyError):
                    with project_write_lock(working):
                        self.fail("a second writer entered an occupied project")
                self.assertEqual(json.loads((lock / "owner.json").read_text(encoding="utf-8")), owner)

            self.assertFalse(lock.exists())

    def test_lock_is_released_when_the_protected_operation_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / ".timeline-write-lock"
            with self.assertRaisesRegex(RuntimeError, "operation failed"):
                with project_write_lock(tmp):
                    self.assertTrue(lock.is_dir())
                    raise RuntimeError("operation failed")
            self.assertFalse(lock.exists())

    def test_release_does_not_remove_a_lock_with_a_different_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / ".timeline-write-lock"
            with project_write_lock(tmp):
                replacement = {"pid": os.getpid(), "token": "replacement-owner"}
                (lock / "owner.json").write_text(json.dumps(replacement), encoding="utf-8")

            self.assertTrue(lock.is_dir())
            self.assertEqual(json.loads((lock / "owner.json").read_text(encoding="utf-8")), replacement)

    def test_dead_owner_is_reclaimed_and_new_owner_released(self):
        with tempfile.TemporaryDirectory() as tmp:
            child = subprocess.Popen([sys.executable, "-c", "pass"])
            dead_pid = child.pid
            child.wait()
            lock = Path(tmp) / ".timeline-write-lock"
            lock.mkdir()
            (lock / "owner.json").write_text(json.dumps({"pid": dead_pid, "token": "dead-owner"}), encoding="utf-8")
            with project_write_lock(tmp):
                owner = json.loads((lock / "owner.json").read_text(encoding="utf-8"))
                self.assertEqual(owner["pid"], os.getpid())
                self.assertNotEqual(owner["token"], "dead-owner")
                self.assertFalse((Path(tmp) / ".timeline-write-lock-reclaim").exists())

            self.assertFalse(lock.exists())

    def test_missing_or_malformed_owner_fails_closed(self):
        for metadata in (None, "{bad json", json.dumps({"pid": True, "token": "x"}), json.dumps({"pid": 0, "token": "x"}), json.dumps({"pid": 1})):
            with self.subTest(metadata=metadata), tempfile.TemporaryDirectory() as tmp:
                lock = Path(tmp) / ".timeline-write-lock"
                lock.mkdir()
                if metadata is not None:
                    (lock / "owner.json").write_text(metadata, encoding="utf-8")

                with self.assertRaises(ProjectBusyError):
                    with project_write_lock(tmp):
                        self.fail("an unowned lock must not be reclaimed")

                self.assertTrue(lock.is_dir())
                self.assertFalse((Path(tmp) / ".timeline-write-lock-reclaim").exists())

    def test_orphaned_reclaim_guard_is_never_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            child = subprocess.Popen([sys.executable, "-c", "pass"])
            dead_pid = child.pid
            child.wait()
            lock = Path(tmp) / ".timeline-write-lock"
            lock.mkdir()
            (lock / "owner.json").write_text(json.dumps({"pid": dead_pid, "token": "dead"}), encoding="utf-8")
            reclaim = Path(tmp) / ".timeline-write-lock-reclaim"
            reclaim.mkdir()
            (reclaim / "owner.json").write_text("{}", encoding="utf-8")

            with self.assertRaises(ProjectBusyError):
                with project_write_lock(tmp):
                    self.fail("a second reaper must not run while a reclaim guard exists")

            self.assertTrue(lock.is_dir())
            self.assertTrue(reclaim.is_dir())


if __name__ == "__main__":
    unittest.main()
