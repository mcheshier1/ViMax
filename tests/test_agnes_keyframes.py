import base64
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestServer
from PIL import Image

from tools.video_generator_agnes_api import VideoGeneratorAgnesAPI


VIDEO_BYTES = b"\x00\x00\x00\x18ftypmp42video"


def _decode_image(data_url):
    return Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[-1])))


class AgnesKeyframeRequestTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.requests = []

        async def create(request):
            self.requests.append(await request.json())
            return web.json_response({"video_id": "video-1"})

        async def status(_request):
            return web.json_response({"status": "completed", "url": str(self.server.make_url("/result.mp4"))})

        async def result(_request):
            return web.Response(body=VIDEO_BYTES, content_type="video/mp4")

        app = web.Application()
        app.router.add_post("/v1/videos", create)
        app.router.add_get("/agnesapi", status)
        app.router.add_get("/result.mp4", result)
        self.server = TestServer(app)
        await self.server.start_server()
        self.addAsyncCleanup(self.server.close)
        self.generator = VideoGeneratorAgnesAPI("offline-test-key", base_url=str(self.server.make_url("/v1")))

    async def _submit(self, paths, aspect_ratio="16:9"):
        with patch("tools.video_generator_agnes_api.asyncio.sleep"):
            return await self.generator.generate_single_video(
                prompt="offline geometry test", reference_image_paths=paths, aspect_ratio=aspect_ratio,
            )

    async def test_both_frames_are_letterboxed_to_same_canvas_without_mutating_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / "first.png", Path(directory) / "last.png"]
            colors = ((230, 20, 30), (10, 210, 40))
            for path, color, dimensions in zip(paths, colors, ((40, 20), (20, 40))):
                Image.new("RGB", dimensions, color).save(path)
            original = [path.read_bytes() for path in paths]

            await self._submit([str(path) for path in paths])

            payload = self.requests[0]
            decoded = [_decode_image(payload[name]) for name in ("first_frame", "last_frame")]
            self.assertEqual([image.size for image in decoded], [(1280, 720), (1280, 720)])
            self.assertEqual(decoded[0].getpixel((640, 250)), colors[0])
            self.assertEqual(decoded[1].getpixel((640, 250)), colors[1])
            self.assertEqual(decoded[0].getpixel((640, 10)), (0, 0, 0))
            self.assertEqual(decoded[1].getpixel((10, 360)), (0, 0, 0))
            self.assertEqual([path.read_bytes() for path in paths], original)

    async def test_correctly_sized_reference_passes_through_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.png"
            Image.new("RGB", (1280, 720), (40, 80, 120)).save(path)
            original = path.read_bytes()

            await self._submit([str(path)])

            self.assertEqual(base64.b64decode(self.requests[0]["first_frame"].split(",", 1)[-1]), original)
            self.assertEqual(path.read_bytes(), original)

    async def test_smaller_matching_aspect_reference_is_upscaled_to_canvas_edges(self):
        color = (70, 130, 190)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "small.png"
            Image.new("RGB", (640, 360), color).save(path)

            await self._submit([str(path)])

            frame = _decode_image(self.requests[0]["first_frame"])
            self.assertEqual(frame.size, (1280, 720))
            self.assertEqual(frame.getpixel((0, 0)), color)
            self.assertEqual(frame.getpixel((1279, 719)), color)

    async def test_exif_orientation_is_applied_before_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oriented.jpg"
            image = Image.new("RGB", (40, 20), (240, 30, 20))
            exif = Image.Exif()
            exif[274] = 6
            image.save(path, exif=exif)
            original = path.read_bytes()
            with Image.open(path) as source:
                decoded_color = source.getpixel((20, 10))

            await self._submit([str(path)])

            frame = _decode_image(self.requests[0]["first_frame"])
            self.assertEqual(frame.size, (1280, 720))
            self.assertEqual(frame.getpixel((640, 360)), decoded_color)
            self.assertEqual(frame.getpixel((640, 10)), decoded_color)
            self.assertEqual(frame.getpixel((100, 360)), (0, 0, 0))
            self.assertEqual(path.read_bytes(), original)

    async def test_unsupported_ratio_and_unusable_image_fail_before_provider_request(self):
        with self.assertRaisesRegex(ValueError, "Unsupported Agnes reference aspect ratio"):
            await self._submit(["not-opened.png"], "5:4")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.png"
            path.write_bytes(b"not an image")
            with self.assertRaisesRegex(ValueError, "Unable to use Agnes keyframe image"):
                await self._submit([str(path)])
        self.assertEqual(self.requests, [])

    async def test_supported_aspect_canvas_is_used_for_portrait(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.png"
            Image.new("RGB", (20, 40), (50, 60, 70)).save(path)
            await self._submit([str(path)], "9:16")

            frame = _decode_image(self.requests[0]["first_frame"])
            self.assertEqual(frame.size, (720, 1280))
    async def test_flash_resolution_and_non_square_one_k_references_fail_pre_request(self):
        flash = VideoGeneratorAgnesAPI(
            "offline-test-key", model="agnes-video-2.5-flash",
            base_url=str(self.server.make_url("/v1")), resolution="1080p",
        )
        with patch("tools.video_generator_agnes_api.asyncio.sleep"):
            with self.assertRaisesRegex(ValueError, "Flash supports 720p"):
                await flash.generate_single_video()

        one_k = VideoGeneratorAgnesAPI(
            "offline-test-key", base_url=str(self.server.make_url("/v1")), resolution="1K",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.png"
            Image.new("RGB", (20, 30), (1, 2, 3)).save(path)
            with patch("tools.video_generator_agnes_api.asyncio.sleep"):
                with self.assertRaisesRegex(ValueError, "fixed at 1024x1024"):
                    await one_k.generate_single_video(reference_image_paths=[str(path)], aspect_ratio="9:16")
        self.assertEqual(self.requests, [])


if __name__ == "__main__":
    unittest.main()
