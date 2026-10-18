import base64
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image
from aiohttp import web

from tools.image_generator_agnes_api import AgnesImageAPIError, ImageGeneratorAgnesAPI


def _png(size=(32, 18)):
    buffer = BytesIO()
    Image.new("RGB", size, "green").save(buffer, format="PNG")
    return buffer.getvalue()


class AgnesImageGeneratorTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.requests = []
        self.png = _png()
        self.portrait = False
        self.failure = False
        self.response_payload = None
        self.return_url = False
        self.download_authorization = []
        app = web.Application()

        async def generate(request):
            body = await request.json()
            self.requests.append((request.headers, body))
            if self.failure:
                return web.json_response({"error": "bad key"}, status=401)
            if self.response_payload is not None:
                return web.json_response(self.response_payload)
            if self.return_url:
                return web.json_response({"data": [{"url": self.base_url + "/image.png"}]})
            data = _png((9, 16)) if self.portrait else self.png
            return web.json_response({"data": [{"b64_json": base64.b64encode(data).decode(), "mime_type": "image/png"}]})

        async def download(request):
            self.download_authorization.append(request.headers.get("Authorization"))
            return web.Response(body=self.png, content_type="image/png")

        app.router.add_post("/v1/images/generations", generate)
        app.router.add_get("/v1/image.png", download)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        self.runner = runner
        self.base_url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/v1"

    async def asyncTearDown(self):
        await self.runner.cleanup()

    async def test_request_protocol_and_reference_order(self):
        generator = ImageGeneratorAgnesAPI("secret", base_url=self.base_url)
        result = await generator.generate_single_image("scene")
        self.assertEqual(result.data.size, (32, 18))
        headers, body = self.requests[-1]
        self.assertEqual(headers["Authorization"], "Bearer secret")
        self.assertEqual(body["ratio"], "16:9")
        self.assertEqual(body["size"], "1K")
        self.assertTrue(body["return_base64"])
        self.assertNotIn("aspect_ratio", body)
        self.assertNotIn("image", body)
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for name, color in (("first.png", "red"), ("second.png", "blue")):
                path = Path(directory) / name
                Image.new("RGB", (2, 2), color).save(path)
                paths.append(str(path))
            await generator.generate_single_image("ref scene", paths)
            references = self.requests[-1][1]["extra_body"]["image"]
            self.assertEqual(self.requests[-1][1]["extra_body"]["response_format"], "b64_json")
            self.assertNotIn("response_format", self.requests[-1][1])
            self.assertEqual([base64.b64decode(ref.split(",", 1)[1]) for ref in references], [Path(p).read_bytes() for p in paths])

    async def test_20_flash_default_size_and_portrait_override(self):
        generator = ImageGeneratorAgnesAPI("secret", model="agnes-image-2.0-flash", base_url=self.base_url)
        await generator.generate_single_image("scene", aspect_ratio="3:2")
        body = self.requests[-1][1]
        self.assertEqual(body["size"], "1024x768")
        self.assertNotIn("ratio", body)
        self.assertNotIn("aspect_ratio", body)

    async def test_configured_resolution_size_override_and_unsupported_model(self):
        generator = ImageGeneratorAgnesAPI("secret", base_url=self.base_url, resolution="2K")
        await generator.generate_single_image("scene")
        self.assertEqual(self.requests[-1][1]["size"], "2K")
        await generator.generate_single_image("scene", size="3K")
        self.assertEqual(self.requests[-1][1]["size"], "3K")
        generator = ImageGeneratorAgnesAPI("secret", model="agnes-image-2.0-flash", base_url=self.base_url, resolution="1K")
        count = len(self.requests)
        with self.assertRaisesRegex(ValueError, "does not support explicit image resolution"):
            await generator.generate_single_image("scene")
        self.assertEqual(len(self.requests), count)

    async def test_portrait_guard_override_and_http_error(self):
        generator = ImageGeneratorAgnesAPI("secret", base_url=self.base_url)
        self.portrait = True
        with self.assertRaisesRegex(ValueError, "portrait-oriented"):
            await generator.generate_single_image("scene")
        result = await generator.generate_single_image("scene", allow_portrait=True)
        self.assertEqual(result.data.size, (9, 16))
        self.failure = True
        with self.assertRaises(AgnesImageAPIError):
            await generator.generate_single_image("scene")

    async def test_bad_response_fails(self):
        generator = ImageGeneratorAgnesAPI("secret", base_url=self.base_url)
        self.response_payload = {"data": []}
        with self.assertRaisesRegex(ValueError, "missing data"):
            await generator.generate_single_image("scene")
        self.response_payload = {"data": [{"b64_json": "not-base64"}]}
        with self.assertRaisesRegex(ValueError, "invalid image data"):
            await generator.generate_single_image("scene")

    async def test_downloaded_image_does_not_receive_api_credentials(self):
        generator = ImageGeneratorAgnesAPI("secret", base_url=self.base_url)
        self.return_url = True
        result = await generator.generate_single_image("scene")
        self.assertEqual(result.data.getpixel((0, 0)), (0, 128, 0))
        self.assertEqual(self.download_authorization, [None])


if __name__ == "__main__":
    unittest.main()
