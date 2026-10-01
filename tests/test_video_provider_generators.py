import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestServer

from tools.video_generator_agnes_api import VideoGeneratorAgnesAPI, _retry_after_seconds
from tools.video_generator_ltx_api import VideoGeneratorLTXAPI


VIDEO_BYTES = b"\x00\x00\x00\x18ftypmp42video"


class RetryAfterTests(unittest.TestCase):
    def test_http_date_and_invalid_retry_after(self):
        with patch("tools.video_generator_agnes_api.datetime") as clock:
            clock.now.return_value = datetime(2026, 1, 1, tzinfo=timezone.utc)
            self.assertEqual(_retry_after_seconds("Thu, 01 Jan 2026 00:00:30 GMT"), 30)
            self.assertEqual(_retry_after_seconds("Wed, 31 Dec 2025 23:59:59 GMT"), 0)
        for value in ("invalid", "NaN", "Infinity", None):
            self.assertIsNone(_retry_after_seconds(value))


class ProviderGenerationTests(unittest.IsolatedAsyncioTestCase):
    async def test_ltx_sends_model_duration_resolution_and_audio_and_returns_video(self):
        requests = []

        async def generate(request):
            requests.append((request.path, await request.json(), request.headers.get("Authorization")))
            return web.Response(body=VIDEO_BYTES, content_type="video/mp4")

        app = web.Application()
        app.router.add_post("/v1/text-to-video", generate)
        server = TestServer(app)
        await server.start_server()
        self.addAsyncCleanup(server.close)
        generator = VideoGeneratorLTXAPI(
            api_key="ltx-secret", model="ltx-2-5-fast", base_url=str(server.make_url("/"))[:-1],
            clip_seconds=10, resolution="720p", generate_audio=False,
        )

        output = await generator.generate_single_video(prompt="A lighthouse in a storm", aspect_ratio="9:16")

        self.assertEqual(requests, [("/v1/text-to-video", {
            "prompt": "A lighthouse in a storm", "model": "ltx-2-5-fast", "duration": 10,
            "resolution": "720x1280", "generate_audio": False,
        }, "Bearer ltx-secret")])
        self.assertEqual(output.data, VIDEO_BYTES)

    async def test_agnes_creates_polls_and_downloads_completed_video(self):
        requests = []

        async def create(request):
            requests.append((request.path, await request.json()))
            return web.json_response({"video_id": "video-1", "status": "queued"})

        async def status(request):
            requests.append((request.path, dict(request.query)))
            return web.json_response({"video_id": "video-1", "status": "completed", "url": str(server.make_url("/result.mp4"))})

        async def result(_request):
            return web.Response(body=VIDEO_BYTES, content_type="video/mp4")

        app = web.Application()
        app.router.add_post("/v1/videos", create)
        app.router.add_get("/agnesapi", status)
        app.router.add_get("/result.mp4", result)
        server = TestServer(app)
        await server.start_server()
        self.addAsyncCleanup(server.close)
        generator = VideoGeneratorAgnesAPI(
            api_key="agnes-secret", model="agnes-video-2.5", base_url=str(server.make_url("/v1")),
            clip_seconds=8, resolution="1080p",
        )
        with patch("tools.video_generator_agnes_api.asyncio.sleep"):
            output = await generator.generate_single_video(prompt="A quiet sunrise")

        self.assertEqual(requests[0], ("/v1/videos", {
            "model": "agnes-video-2.5", "prompt": "A quiet sunrise", "mode": "text",
            "seconds": "8", "size": "1080P", "aspect_ratio": "16:9",
        }))
        self.assertEqual(requests[1], ("/agnesapi", {"video_id": "video-1", "model_name": "agnes-video-2.5"}))
        self.assertEqual(output.data, VIDEO_BYTES)


    async def test_agnes_retries_capacity_response_then_completes(self):
        create_attempts = 0
        status_calls = 0

        async def create(_request):
            nonlocal create_attempts
            create_attempts += 1
            if create_attempts < 3:
                return web.json_response({"code": "insufficient_capacity", "message": "Insufficient capacity"}, status=503)
            return web.json_response({"video_id": "video-2", "status": "queued"})

        async def status(_request):
            nonlocal status_calls
            status_calls += 1
            return web.json_response({"video_id": "video-2", "status": "completed", "url": str(server.make_url("/result.mp4"))})

        async def result(_request):
            return web.Response(body=VIDEO_BYTES, content_type="video/mp4")

        app = web.Application()
        app.router.add_post("/v1/videos", create)
        app.router.add_get("/agnesapi", status)
        app.router.add_get("/result.mp4", result)
        server = TestServer(app)
        await server.start_server()
        self.addAsyncCleanup(server.close)
        generator = VideoGeneratorAgnesAPI(api_key="agnes-secret", base_url=str(server.make_url("/v1")))
        progress = []
        with patch("tools.video_generator_agnes_api.asyncio.sleep") as retry_sleep:
            output = await generator.generate_single_video(
                prompt="A quiet sunrise",
                progress=lambda stage, message, metadata: progress.append((stage, message, metadata)),
            )

        self.assertEqual(create_attempts, 3)
        self.assertEqual([call.args[0] for call in retry_sleep.await_args_list], [2, 4, 1.5])
        self.assertEqual(status_calls, 1)
        self.assertEqual([event[0] for event in progress].count("video_create_retry"), 2)
        self.assertEqual(output.data, VIDEO_BYTES)

    async def test_agnes_stops_after_three_capacity_responses(self):
        create_attempts = 0

        async def create(_request):
            nonlocal create_attempts
            create_attempts += 1
            return web.json_response({"code": "video_queue_full", "message": "Video queue is full"}, status=503)

        app = web.Application()
        app.router.add_post("/v1/videos", create)
        server = TestServer(app)
        await server.start_server()
        self.addAsyncCleanup(server.close)
        generator = VideoGeneratorAgnesAPI(api_key="agnes-secret", base_url=str(server.make_url("/v1")))
        with patch("tools.video_generator_agnes_api.asyncio.sleep") as retry_sleep:
            with self.assertRaisesRegex(RuntimeError, "video_queue_full"):
                await generator.generate_single_video(prompt="A quiet sunrise")

        self.assertEqual(create_attempts, 3)
        self.assertEqual([call.args[0] for call in retry_sleep.await_args_list], [2, 4])

    async def test_agnes_does_not_retry_unrelated_503_response(self):
        create_attempts = 0

        async def create(_request):
            nonlocal create_attempts
            create_attempts += 1
            return web.json_response({"code": "model_unavailable", "message": "Model is unavailable"}, status=503)

        app = web.Application()
        app.router.add_post("/v1/videos", create)
        server = TestServer(app)
        await server.start_server()
        self.addAsyncCleanup(server.close)
        generator = VideoGeneratorAgnesAPI(api_key="agnes-secret", base_url=str(server.make_url("/v1")))
        with patch("tools.video_generator_agnes_api.asyncio.sleep") as retry_sleep:
            with self.assertRaisesRegex(RuntimeError, "HTTP 503"):
                await generator.generate_single_video(prompt="A quiet sunrise")

        self.assertEqual(create_attempts, 1)
        retry_sleep.assert_not_awaited()

    async def test_agnes_rate_limits_retry_same_job_with_backoff_and_reset(self):
        creates = []
        queries = []
        responses = [
            ("Too many queries", {}),
            ("Too many queries", {"Retry-After": "7"}),
            *[("Too many queries", {"Retry-After": "invalid"}) for _ in range(5)],
            ({"status": "queued"}, {}),
            ("Too many queries", {}),
        ]

        async def create(request):
            creates.append(await request.json())
            return web.json_response({"video_id": "existing-job"})

        async def status(request):
            queries.append(dict(request.query))
            if responses:
                payload, headers = responses.pop(0)
                if isinstance(payload, str):
                    return web.Response(status=429, text=payload, headers=headers)
                return web.json_response(payload)
            return web.json_response({"status": "completed", "url": str(server.make_url("/result.mp4"))})

        async def result(_request):
            return web.Response(body=VIDEO_BYTES)

        app = web.Application()
        app.router.add_post("/v1/videos", create)
        app.router.add_get("/agnesapi", status)
        app.router.add_get("/result.mp4", result)
        server = TestServer(app)
        await server.start_server()
        self.addAsyncCleanup(server.close)
        generator = VideoGeneratorAgnesAPI(api_key="test", base_url=str(server.make_url("/v1")))
        progress = []
        with patch("tools.video_generator_agnes_api.asyncio.sleep") as sleep, patch("tools.video_generator_agnes_api.random.uniform", side_effect=lambda low, high: high):
            output = await generator.generate_single_video(progress=lambda *event: progress.append(event))
        self.assertEqual(len(creates), 1)
        self.assertEqual(queries, [{"video_id": "existing-job", "model_name": "agnes-video-2.5"}] * 10)
        self.assertEqual([call.args[0] for call in sleep.await_args_list], [1.5, 3, 7, 12, 24, 48, 60, 60, 1.5, 3])
        self.assertEqual(output.data, VIDEO_BYTES)
        retries = [event for event in progress if event[0] == "video_status_retry"]
        self.assertTrue(all(event[2]["video_id"] == "existing-job" for event in retries))
        self.assertIn("rate-limited", retries[0][1])

    async def test_agnes_retry_after_cannot_extend_query_deadline(self):
        creates = 0
        queries = 0

        async def create(_request):
            nonlocal creates
            creates += 1
            return web.json_response({"video_id": "still-running"})

        async def status(_request):
            nonlocal queries
            queries += 1
            return web.Response(status=429, text="Too many queries", headers={"Retry-After": "600"})

        app = web.Application()
        app.router.add_post("/v1/videos", create)
        app.router.add_get("/agnesapi", status)
        server = TestServer(app)
        await server.start_server()
        self.addAsyncCleanup(server.close)
        generator = VideoGeneratorAgnesAPI(api_key="test", base_url=str(server.make_url("/v1")))
        with patch.dict("os.environ", {"VIMAX_VIDEO_QUERY_TIMEOUT_SECONDS": "0.1"}), patch("tools.video_generator_agnes_api.VIDEO_STATUS_INTERVAL", 0.001):
            with self.assertRaisesRegex(RuntimeError, "timed out.*still-running"):
                await generator.generate_single_video()
        self.assertEqual((creates, queries), (1, 1))


if __name__ == "__main__":
    unittest.main()
