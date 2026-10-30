"""Regression tests for rate-limiter lock behavior, media assembly, packaging, and suite isolation."""

import asyncio
import subprocess
import sys
import tempfile
import tomllib
import unittest
from contextlib import suppress
from pathlib import Path

import yaml

import numpy as np
from moviepy import ImageClip, VideoFileClip
from PIL import Image, ImageDraw
from utils.rate_limiter import RateLimiter
from utils.video import concatenate_video_files

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestRateLimiterLocking(unittest.IsolatedAsyncioTestCase):
    async def test_waiting_acquirer_does_not_hold_the_lock(self):
        limiter = RateLimiter(max_requests_per_minute=1)
        await limiter.acquire()  # consume the only slot in the window
        waiter = asyncio.create_task(limiter.acquire())
        await asyncio.sleep(0.05)  # waiter is now waiting for the window to free
        try:
            try:
                await asyncio.wait_for(limiter.lock.acquire(), timeout=0.25)
                limiter.lock.release()
            except asyncio.TimeoutError:
                self.fail("rate limiter sleeps while holding its lock, blocking every other caller")
        finally:
            waiter.cancel()
            with suppress(asyncio.CancelledError):
                await waiter

    async def test_min_delay_smoothing_still_applies(self):
        limiter = RateLimiter(max_requests_per_minute=600)  # min delay 0.1s
        loop = asyncio.get_running_loop()
        start = loop.time()
        await limiter.acquire()
        await limiter.acquire()
        self.assertGreaterEqual(loop.time() - start, 0.08)


class TestVideoAssemblyContract(unittest.TestCase):
    def test_mixed_dimensions_keep_pixels_centered_on_a_fixed_canvas(self):
        with tempfile.TemporaryDirectory() as tmp:
            sources = []
            for index, (size, color) in enumerate([((64, 36), "red"), ((60, 40), "blue")]):
                image = Image.new("RGB", size, color)
                ImageDraw.Draw(image).rectangle((12, 10, 19, 17), fill="white")
                source = str(Path(tmp, f"{index}.mp4"))
                with ImageClip(np.asarray(image)).with_duration(0.5) as clip:
                    clip.write_videofile(source, fps=12, codec="libx264", audio=False, threads=1, logger=None)
                sources.append(source)
            output = str(Path(tmp, "joined.mp4"))
            concatenate_video_files(sources, output)
            with VideoFileClip(output) as film:
                self.assertEqual(tuple(film.size), (64, 40))
                self.assertAlmostEqual(film.duration, 1, places=1)
                red, blue = film.get_frame(0.25), film.get_frame(0.75)
                np.testing.assert_allclose(red[20, 32], (255, 0, 0), atol=18)
                np.testing.assert_allclose(blue[20, 32], (0, 0, 255), atol=18)



class TestPackagingMetadata(unittest.TestCase):
    def test_pyproject_is_consistent(self):
        with open(REPO_ROOT / "pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        readme = data["project"]["readme"]
        self.assertTrue((REPO_ROOT / readme).exists(), f"readme points at missing file: {readme}")
        self.assertNotIn("index", data, "top-level [[index]] is not valid pyproject TOML and is silently ignored")
        dev_group = data.get("dependency-groups", {}).get("dev", [])
        self.assertTrue(any("pytest" in str(item) for item in dev_group), "pytest must be a declared dev dependency so the suite runs from the venv")


class TestProviderConfigTemplates(unittest.TestCase):
    def test_minimax_templates_do_not_ship_truthy_placeholders(self):
        for name in ("idea2video_minimax.yaml", "script2video_minimax.yaml"):
            with open(REPO_ROOT / "configs" / name, encoding="utf-8") as f:
                config = yaml.safe_load(f)
            chat_key = config["chat_model"]["init_args"].get("api_key")
            self.assertFalse(chat_key, f"{name}: a truthy api_key placeholder defeats the MINIMAX_API_KEY env fallback")
            for section in ("image_generator", "video_generator"):
                key = config[section]["init_args"].get("api_key")
                self.assertNotIn("<", str(key or ""), f"{name}: {section} ships an angle-bracket placeholder that would be sent as a bearer token")


class TestSuiteIsolation(unittest.TestCase):
    def test_importing_minimax_tests_does_not_stub_global_modules(self):
        code = (
            "import sys; import tests.test_minimax_integration; "
            "mod = sys.modules.get('cv2'); "
            "from unittest.mock import MagicMock; "
            "exit(1 if isinstance(mod, MagicMock) else 0)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode, 0,
            "importing tests.test_minimax_integration replaces sys.modules entries at import time, "
            "making every later-collected test module see MagicMock stubs instead of real libraries",
        )


if __name__ == "__main__":
    unittest.main()
