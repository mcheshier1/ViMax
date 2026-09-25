import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from PIL import Image

from tools.video_generator_openrouter_api import VideoGeneratorOpenRouterAPI
from interfaces.video_output import VideoOutput


GROK_CAPABILITIES = {
    "id": "x-ai/grok-imagine-video",
    "supported_frame_images": ["first_frame"],
    "supported_durations": [1, 8],
    "supported_resolutions": ["480p", "720p"],
    "supported_aspect_ratios": ["16:9"],
    "generate_audio": None,
}
VEO_CAPABILITIES = {
    "id": "google/veo-3.1",
    "supported_frame_images": ["first_frame", "last_frame"],
    "supported_durations": [4, 8],
    "supported_resolutions": ["720p", "1080p"],
    "supported_aspect_ratios": ["16:9"],
    "generate_audio": True,
}


def _frame_paths(directory: str) -> tuple[str, str]:
    first = Path(directory) / "first.png"
    last = Path(directory) / "last.png"
    Image.new("RGB", (16, 9), "blue").save(first)
    Image.new("RGB", (16, 9), "red").save(last)
    return str(first), str(last)


async def _generate(generator, catalogue, *, reference_image_paths=(), progress=None, env=None):
    """Run one video job against a stubbed catalogue and return (payload, output)."""
    captured = {}

    async def fake_post_json(url, *, headers, payload, timeout, hard_timeout_seconds):
        captured["payload"] = payload
        return 200, {"id": "job-1", "polling_url": "/videos/job-1", "status": "queued"}

    async def fake_get_json(url, *, headers, timeout, hard_timeout_seconds):
        if url.endswith("/videos/models"):
            return 200, {"data": list(catalogue)}
        return 200, {"status": "completed", "unsigned_urls": ["https://cdn.example/out.mp4"]}

    async def fake_get_bytes(url, *, headers, timeout, hard_timeout_seconds):
        return 200, b"video"

    async def fake_sleep(seconds):
        return None

    with patch.dict(os.environ, env or {}, clear=True), \
         patch("tools.video_generator_openrouter_api._MODEL_CAPABILITIES", {}), \
         patch("tools.video_generator_openrouter_api._post_json", fake_post_json), \
         patch("tools.video_generator_openrouter_api._get_json", fake_get_json), \
         patch("tools.video_generator_openrouter_api._get_bytes", fake_get_bytes), \
         patch("tools.video_generator_openrouter_api.asyncio.sleep", fake_sleep):
        output = await generator.generate_single_video(
            prompt="a cinematic walk",
            reference_image_paths=list(reference_image_paths),
            progress=progress,
        )
    return captured["payload"], output


class PollToleranceTests(unittest.IsolatedAsyncioTestCase):
    """A blip while asking about a running job must not abandon the job."""

    async def test_a_transient_poll_failure_is_retried_in_place(self):
        from tools import video_generator_openrouter_api as module

        calls = {"count": 0}

        async def flaky_get(url, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                return 503, {"error": "upstream busy"}
            return 200, {"status": "completed"}

        with patch.object(module, "_get_json", side_effect=flaky_get), \
             patch.object(module.asyncio, "sleep", new=AsyncMock()):
            status, payload = await module._poll_with_retries("http://x", headers={}, timeout=None, hard_timeout_seconds=1.0)

        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(calls["count"], 2)

    async def test_a_persistent_poll_failure_still_surfaces(self):
        from tools import video_generator_openrouter_api as module

        async def always_down(url, **kwargs):
            return 500, {"error": "down"}

        with patch.object(module, "_get_json", side_effect=always_down), \
             patch.object(module.asyncio, "sleep", new=AsyncMock()):
            status, _ = await module._poll_with_retries("http://x", headers={}, timeout=None, hard_timeout_seconds=1.0)

        self.assertEqual(status, 500)


class OpenRouterVideoGeneratorTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_duration_is_eight_seconds(self):
        captured = {}

        async def fake_post_json(url, *, headers, payload, timeout, hard_timeout_seconds):
            captured["payload"] = payload
            return 200, {"id": "job-1", "polling_url": "/videos/job-1", "status": "queued"}

        async def fake_get_json(url, *, headers, timeout, hard_timeout_seconds):
            return 200, {"status": "completed", "unsigned_urls": ["https://cdn.example/out.mp4"]}

        async def fake_get_bytes(url, *, headers, timeout, hard_timeout_seconds):
            return 200, b"video"

        async def fake_sleep(seconds):
            return None

        generator = VideoGeneratorOpenRouterAPI(api_key="test-key", model="google/veo-3.1-lite")
        with patch.dict(os.environ, {}, clear=True), \
             patch("tools.video_generator_openrouter_api._post_json", fake_post_json), \
             patch("tools.video_generator_openrouter_api._get_json", fake_get_json), \
             patch("tools.video_generator_openrouter_api._get_bytes", fake_get_bytes), \
             patch("tools.video_generator_openrouter_api.asyncio.sleep", fake_sleep):
            output = await generator.generate_single_video(prompt="hello")

        self.assertIsInstance(output, VideoOutput)
        self.assertEqual(captured["payload"]["duration"], 8)
        self.assertEqual(captured["payload"]["model"], "google/veo-3.1-lite")

    async def test_seedance_fast_uses_supported_openrouter_payload(self):
        captured = {}

        async def fake_post_json(url, *, headers, payload, timeout, hard_timeout_seconds):
            captured["payload"] = payload
            return 200, {"id": "job-2", "polling_url": "/videos/job-2", "status": "queued"}

        async def fake_get_json(url, *, headers, timeout, hard_timeout_seconds):
            if url.endswith("/videos/models"):
                return 200, {"data": [{
                    "id": "bytedance/seedance-2.0-fast",
                    "supported_frame_images": ["first_frame", "last_frame"],
                    "supported_durations": [8],
                    "supported_resolutions": ["720p"],
                    "supported_aspect_ratios": ["16:9"],
                    "generate_audio": True,
                }]}
            return 200, {"status": "completed", "unsigned_urls": ["https://cdn.example/seedance.mp4"]}

        async def fake_get_bytes(url, *, headers, timeout, hard_timeout_seconds):
            return 200, b"seedance-video"

        async def fake_sleep(seconds):
            return None

        with tempfile.TemporaryDirectory() as tmp:
            first_frame = Path(tmp) / "first.png"
            last_frame = Path(tmp) / "last.png"
            Image.new("RGB", (16, 9), "blue").save(first_frame)
            Image.new("RGB", (16, 9), "red").save(last_frame)
            generator = VideoGeneratorOpenRouterAPI(api_key="test-key", model="bytedance/seedance-2.0-fast")
            with patch.dict(os.environ, {}, clear=True), \
                 patch("tools.video_generator_openrouter_api._MODEL_CAPABILITIES", {}), \
                 patch("tools.video_generator_openrouter_api._post_json", fake_post_json), \
                 patch("tools.video_generator_openrouter_api._get_json", fake_get_json), \
                 patch("tools.video_generator_openrouter_api._get_bytes", fake_get_bytes), \
                 patch("tools.video_generator_openrouter_api.asyncio.sleep", fake_sleep):
                output = await generator.generate_single_video(
                    prompt="a cinematic walk",
                    reference_image_paths=[str(first_frame), str(last_frame)],
                )

        payload = captured["payload"]
        self.assertIsInstance(output, VideoOutput)
        self.assertEqual(payload["model"], "bytedance/seedance-2.0-fast")
        self.assertEqual(payload["duration"], 8)
        self.assertEqual(payload["resolution"], "720p")
        self.assertEqual(payload["aspect_ratio"], "16:9")
        self.assertTrue(payload["generate_audio"])
        self.assertEqual(
            [frame["frame_type"] for frame in payload["frame_images"]],
            ["first_frame", "last_frame"],
        )

    async def test_last_frame_is_dropped_for_integrations_that_reject_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            first_frame, last_frame = _frame_paths(tmp)
            events = []
            payload, output = await _generate(
                VideoGeneratorOpenRouterAPI(api_key="test-key", model="x-ai/grok-imagine-video"),
                [GROK_CAPABILITIES],
                reference_image_paths=[first_frame, last_frame],
                progress=lambda stage, message, metadata: events.append((stage, message, metadata)),
            )

        self.assertIsInstance(output, VideoOutput)
        self.assertEqual([frame["frame_type"] for frame in payload["frame_images"]], ["first_frame"])
        self.assertNotIn("generate_audio", payload)  # grok exposes no audio option
        trimmed = [message for stage, message, _ in events if stage == "video_options_trimmed"]
        self.assertEqual(len(trimmed), 1)
        self.assertIn("last_frame", trimmed[0])

    async def test_both_frames_survive_for_integrations_that_support_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            first_frame, last_frame = _frame_paths(tmp)
            payload, output = await _generate(
                VideoGeneratorOpenRouterAPI(api_key="test-key", model="google/veo-3.1"),
                [VEO_CAPABILITIES],
                reference_image_paths=[first_frame, last_frame],
            )

        self.assertIsInstance(output, VideoOutput)
        self.assertEqual(
            [frame["frame_type"] for frame in payload["frame_images"]],
            ["first_frame", "last_frame"],
        )
        self.assertTrue(payload["generate_audio"])

    async def test_unsupported_duration_fails_before_a_job_is_created(self):
        generator = VideoGeneratorOpenRouterAPI(api_key="test-key", model="x-ai/grok-imagine-video")
        with self.assertRaises(ValueError) as caught:
            await _generate(generator, [GROK_CAPABILITIES], env={"VIMAX_OPENROUTER_VIDEO_DURATION": "9"})

        self.assertIn("does not support duration=9", str(caught.exception))
        self.assertIn("supported values: [1, 8]", str(caught.exception))

    async def test_request_is_left_unvalidated_when_the_catalogue_cannot_be_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            first_frame, last_frame = _frame_paths(tmp)
            captured = {}

            async def fake_post_json(url, *, headers, payload, timeout, hard_timeout_seconds):
                captured["payload"] = payload
                return 200, {"id": "job-3", "polling_url": "/videos/job-3", "status": "queued"}

            async def failing_get_json(url, *, headers, timeout, hard_timeout_seconds):
                if url.endswith("/videos/models"):
                    raise RuntimeError("catalogue unavailable")
                return 200, {"status": "completed", "unsigned_urls": ["https://cdn.example/out.mp4"]}

            async def fake_get_bytes(url, *, headers, timeout, hard_timeout_seconds):
                return 200, b"video"

            async def fake_sleep(seconds):
                return None

            generator = VideoGeneratorOpenRouterAPI(api_key="test-key", model="x-ai/grok-imagine-video")
            with patch.dict(os.environ, {}, clear=True), \
                 patch("tools.video_generator_openrouter_api._MODEL_CAPABILITIES", {}), \
                 patch("tools.video_generator_openrouter_api._post_json", fake_post_json), \
                 patch("tools.video_generator_openrouter_api._get_json", failing_get_json), \
                 patch("tools.video_generator_openrouter_api._get_bytes", fake_get_bytes), \
                 patch("tools.video_generator_openrouter_api.asyncio.sleep", fake_sleep):
                output = await generator.generate_single_video(
                    prompt="a cinematic walk",
                    reference_image_paths=[first_frame, last_frame],
                )

        self.assertIsInstance(output, VideoOutput)
        self.assertEqual(
            [frame["frame_type"] for frame in captured["payload"]["frame_images"]],
            ["first_frame", "last_frame"],
        )


if __name__ == "__main__":
    unittest.main()
