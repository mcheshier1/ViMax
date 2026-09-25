import base64
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, patch

from PIL import Image

import tools.image_generator_openrouter_api as image_generator_openrouter_api
from agent_runtime.vimax_adapters import _build_image_generator
from tools.image_generator_openrouter_api import (
    ImageGeneratorOpenRouterAPI,
    OpenRouterImageAPIError,
    _is_retryable_image_error,
)


def _encoded_png(size: tuple[int, int] = (16, 9)) -> str:
    buffer = BytesIO()
    Image.new("RGB", size, "blue").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _catalogue_get_json(entries: dict[str, dict]) -> "callable":
    async def fake_get_json(url, *, headers, timeout, hard_timeout_seconds):
        return 200, {"data": [{"id": model, **parameters} for model, parameters in entries.items()]}

    return fake_get_json


class OpenRouterImageGeneratorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Every test runs against an explicit catalogue: the real endpoint is
        # never contacted, and the module cache never leaks between tests.
        self.catalogue: dict[str, dict] = {}
        for patcher in (
            patch("tools.image_generator_openrouter_api._MODEL_CAPABILITIES", {}),
            patch("tools.image_generator_openrouter_api._LEARNED_REFERENCE_LIMITS", {}),
            patch("tools.image_generator_openrouter_api._get_json", _catalogue_get_json(self.catalogue)),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    async def test_generates_image_with_dedicated_images_api(self):
        captured = {}

        async def fake_post(url, *, headers, payload, timeout):
            captured.update(url=url, headers=headers, payload=payload, timeout=timeout)
            return 200, {"data": [{"b64_json": _encoded_png(), "media_type": "image/png"}]}

        progress = []
        generator = ImageGeneratorOpenRouterAPI(api_key="secret", model="openai/gpt-image-2")
        with patch("tools.image_generator_openrouter_api._post_json", fake_post):
            result = await generator.generate_single_image(
                "a cinematic beach",
                aspect_ratio="16:9",
                progress=lambda stage, message, metadata: progress.append((stage, message, metadata)),
            )

        self.assertEqual(captured["url"], "https://openrouter.ai/api/v1/images")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer secret")
        self.assertEqual(captured["payload"]["model"], "openai/gpt-image-2")
        self.assertNotIn("aspect_ratio", captured["payload"])
        self.assertIn("landscape image", captured["payload"]["prompt"])
        self.assertEqual(result.data.size, (16, 9))
        self.assertEqual(result.ext, "png")
        self.assertEqual([item[0] for item in progress], ["image_generation", "image_completed"])

    async def test_prepare_warms_the_reference_limit_the_pipeline_reads(self):
        """Without this the first frame of a run is built with no reference limit at all."""
        self.catalogue["qwen/qwen-image-3"] = {
            "supported_parameters": {"input_references": {"type": "range", "min": 0, "max": 4}},
        }
        generator = ImageGeneratorOpenRouterAPI(api_key="secret", model="qwen/qwen-image-3")

        # Unknown until the catalogue is read, so the caller is unbounded rather than capped.
        self.assertIsNone(generator.reference_limit)

        await generator.prepare()

        self.assertEqual(generator.reference_limit, 4)

    async def test_it_says_when_a_model_takes_no_references_at_all(self):
        """A model that accepts no references invents the cast, and silence would hide it."""
        self.catalogue["inclusionai/ming-image-0.1-design"] = {
            "supported_parameters": {"input_references": {"type": "range", "min": 0, "max": 0}},
        }
        captured = {}
        progress = []

        async def fake_post(url, *, headers, payload, timeout):
            captured.update(payload)
            return 200, {"data": [{"b64_json": _encoded_png(), "media_type": "image/png"}]}

        generator = ImageGeneratorOpenRouterAPI(api_key="secret", model="inclusionai/ming-image-0.1-design")
        with patch("tools.image_generator_openrouter_api._post_json", fake_post):
            await generator.generate_single_image(
                "a cinematic beach",
                reference_image_paths=["/tmp/a-portrait-that-is-never-sent.png"],
                progress=lambda stage, message, metadata: progress.append((stage, message, metadata)),
            )

        # The references are not sent, because the model would refuse them …
        self.assertNotIn("input_references", captured)
        # … and dropping every one of them is reported rather than swallowed.
        dropped = [item for item in progress if item[0] == "references_dropped"]
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0][2]["offered_references"], 1)
        self.assertIn("no reference images", dropped[0][1])

    async def test_a_model_that_takes_references_is_not_warned_about(self):
        self.catalogue["qwen/qwen-image-3"] = {
            "supported_parameters": {"input_references": {"type": "range", "min": 0, "max": 3}},
        }
        captured = {}
        progress = []

        async def fake_post(url, *, headers, payload, timeout):
            captured.update(payload)
            return 200, {"data": [{"b64_json": _encoded_png(), "media_type": "image/png"}]}

        generator = ImageGeneratorOpenRouterAPI(api_key="secret", model="qwen/qwen-image-3")
        with tempfile.TemporaryDirectory() as tmp:
            portrait = Path(tmp) / "portrait.png"
            portrait.write_bytes(base64.b64decode(_encoded_png((4, 4))))
            with patch("tools.image_generator_openrouter_api._post_json", fake_post):
                await generator.generate_single_image(
                    "a cinematic beach",
                    reference_image_paths=[str(portrait)],
                    progress=lambda stage, message, metadata: progress.append((stage, message, metadata)),
                )

        self.assertIn("input_references", captured)
        self.assertEqual([item[0] for item in progress], ["image_generation", "image_completed"])

    async def test_models_without_a_parameter_never_receive_it(self):
        """meta/muse-image advertises no optional parameters and rejects any extra."""
        self.catalogue["meta/muse-image"] = {"supported_parameters": {}}
        captured = {}

        async def fake_post(url, *, headers, payload, timeout):
            captured.update(payload)
            return 200, {"data": [{"b64_json": _encoded_png(), "media_type": "image/webp"}]}

        generator = ImageGeneratorOpenRouterAPI(api_key="secret", model="meta/muse-image")
        with patch("tools.image_generator_openrouter_api._post_json", fake_post):
            result = await generator.generate_single_image("a cinematic beach")

        self.assertEqual(captured["model"], "meta/muse-image")
        self.assertNotIn("quality", captured)
        self.assertNotIn("background", captured)
        self.assertEqual(result.ext, "webp")

    async def test_advertised_parameters_are_sent(self):
        self.catalogue["openai/gpt-image-2"] = {"supported_parameters": {
            "quality": {"type": "enum", "values": ["auto", "high"]},
            "background": {"type": "enum", "values": ["auto", "opaque"]},
        }}
        captured = {}

        async def fake_post(url, *, headers, payload, timeout):
            captured.update(payload)
            return 200, {"data": [{"b64_json": _encoded_png()}]}

        generator = ImageGeneratorOpenRouterAPI(api_key="secret", model="openai/gpt-image-2", quality="high", background="opaque")
        with patch("tools.image_generator_openrouter_api._post_json", fake_post):
            await generator.generate_single_image("a cinematic beach")

        self.assertEqual(captured["quality"], "high")
        self.assertEqual(captured["background"], "opaque")

    async def test_values_the_model_does_not_list_are_omitted(self):
        """xAI advertises `quality` but rejects `auto`; sending it makes the request unroutable."""
        self.catalogue["x-ai/grok-imagine-image-2.0"] = {"supported_parameters": {
            "quality": {"type": "enum", "values": ["low", "medium"]},
            "aspect_ratio": {"type": "enum", "values": ["16:9"]},
            "input_references": {"type": "range", "min": 0, "max": 3},
        }}
        captured = {}

        async def fake_post(url, *, headers, payload, timeout):
            captured.update(payload)
            return 200, {"data": [{"b64_json": _encoded_png()}]}

        generator = ImageGeneratorOpenRouterAPI(api_key="secret", model="x-ai/grok-imagine-image-2.0")
        with patch("tools.image_generator_openrouter_api._post_json", fake_post):
            await generator.generate_single_image("a cinematic beach", aspect_ratio="16:9")

        self.assertNotIn("quality", captured)  # default "auto" is not offered here
        self.assertEqual(captured["aspect_ratio"], "16:9")

    async def test_aspect_ratio_is_pinned_when_the_model_supports_it(self):
        """Models that ignore the landscape wording in the prompt return square frames."""
        self.catalogue["google/gemini-3.1-flash-image"] = {"supported_parameters": {
            "aspect_ratio": {"type": "enum", "values": ["1:1", "16:9"]},
        }}
        captured = {}

        async def fake_post(url, *, headers, payload, timeout):
            captured.update(payload)
            return 200, {"data": [{"b64_json": _encoded_png()}]}

        generator = ImageGeneratorOpenRouterAPI(api_key="secret", model="google/gemini-3.1-flash-image")
        with patch("tools.image_generator_openrouter_api._post_json", fake_post):
            await generator.generate_single_image("a cinematic beach", aspect_ratio="16:9")

        self.assertEqual(captured["aspect_ratio"], "16:9")

    async def test_aspect_ratio_is_left_out_when_the_model_does_not_offer_it(self):
        self.catalogue["meta/muse-image"] = {"supported_parameters": {}}
        captured = {}

        async def fake_post(url, *, headers, payload, timeout):
            captured.update(payload)
            return 200, {"data": [{"b64_json": _encoded_png()}]}

        generator = ImageGeneratorOpenRouterAPI(api_key="secret", model="meta/muse-image")
        with patch("tools.image_generator_openrouter_api._post_json", fake_post):
            await generator.generate_single_image("a cinematic beach", aspect_ratio="16:9")

        self.assertNotIn("aspect_ratio", captured)

    async def test_reference_images_beyond_the_model_limit_are_trimmed_from_the_end(self):
        """A model that accepts fewer references must not fail the frame.

        Reference positions address the prompt's "Image N" mentions, so the excess
        goes from the end: the kept images keep the indices the prompt used.
        """
        self.catalogue["qwen/qwen-image-3"] = {"supported_parameters": {
            "input_references": {"type": "range", "min": 0, "max": 3},
        }}
        post = AsyncMock(return_value=(200, {"data": [{"b64_json": _encoded_png()}]}))
        generator = ImageGeneratorOpenRouterAPI(api_key="secret", model="qwen/qwen-image-3")
        with tempfile.TemporaryDirectory() as tmp:
            references = []
            for index in range(5):
                path = Path(tmp) / f"reference-{index}.png"
                Image.new("RGB", (16, 9), "red").save(path)
                references.append(str(path))

            with patch("tools.image_generator_openrouter_api._post_json", post):
                await generator.generate_single_image("edit this", references)

        sent = post.await_args.kwargs["payload"]["input_references"]
        self.assertEqual(len(sent), 3)
        self.assertEqual(post.await_count, 1)

    async def test_a_provider_that_names_a_lower_limit_is_retried_within_it(self):
        """The catalogue over-promises: Qwen advertises 4 but the model takes 3."""
        self.catalogue["qwen/qwen-image-3"] = {"supported_parameters": {
            "input_references": {"type": "range", "min": 0, "max": 4},
        }}
        refusal = {
            "error": {"message": "Model 'qwen-image-3.0-distill' supports 0~3 image content items. Got 4 image items. (0 images = T2I mode, 1~3 images = I2I mode)", "code": 400}
        }
        post = AsyncMock(side_effect=[(400, refusal), (200, {"data": [{"b64_json": _encoded_png()}]}), (200, {"data": [{"b64_json": _encoded_png()}]})])
        generator = ImageGeneratorOpenRouterAPI(api_key="secret", model="qwen/qwen-image-3")
        with tempfile.TemporaryDirectory() as tmp:
            references = []
            for index in range(4):
                path = Path(tmp) / f"reference-{index}.png"
                Image.new("RGB", (16, 9), "red").save(path)
                references.append(str(path))

            with patch("tools.image_generator_openrouter_api._post_json", post):
                await generator.generate_single_image("edit this", references)
                # The revealed limit is remembered, so the next frame is never refused for this.
                await generator.generate_single_image("edit this", references)

        self.assertEqual(len(post.await_args.kwargs["payload"]["input_references"]), 3)

    def test_reference_limit_reports_the_narrowest_known_limit(self):
        self.catalogue["qwen/qwen-image-3"] = {"supported_parameters": {
            "input_references": {"type": "range", "min": 0, "max": 4},
        }}
        # A provider refusal reveals the real limit, which wins over the catalogue.
        image_generator_openrouter_api._learn_reference_limit("https://openrouter.ai/api/v1", "qwen/qwen-image-3", {"error": {"message": "supports 0~3 image content items. Got 4 image items."}})
        self.assertEqual(image_generator_openrouter_api.reference_limit_for("https://openrouter.ai/api/v1", "qwen/qwen-image-3", self.catalogue["qwen/qwen-image-3"]), 3)

    async def test_parameters_are_omitted_when_the_model_is_not_in_the_catalogue(self):
        captured = {}

        async def fake_post(url, *, headers, payload, timeout):
            captured.update(payload)
            return 200, {"data": [{"b64_json": _encoded_png()}]}

        generator = ImageGeneratorOpenRouterAPI(api_key="secret", model="some/future-image-model")
        with patch("tools.image_generator_openrouter_api._post_json", fake_post):
            await generator.generate_single_image("a cinematic beach")

        self.assertNotIn("quality", captured)
        self.assertNotIn("background", captured)

    async def test_reference_images_use_data_urls(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference_path = Path(tmp) / "reference.png"
            Image.new("RGB", (16, 9), "red").save(reference_path)
            post = AsyncMock(return_value=(200, {"data": [{"b64_json": _encoded_png()}]}))
            generator = ImageGeneratorOpenRouterAPI(api_key="secret")
            with patch("tools.image_generator_openrouter_api._post_json", post):
                await generator.generate_single_image("edit this", [str(reference_path)])

        payload = post.await_args.kwargs["payload"]
        reference_url = payload["input_references"][0]["image_url"]["url"]
        self.assertTrue(reference_url.startswith("data:image/png;base64,"))

    async def test_non_retryable_client_error_is_not_repeated(self):
        post = AsyncMock(return_value=(400, {"error": {"message": "bad request"}}))
        generator = ImageGeneratorOpenRouterAPI(api_key="secret")
        with patch("tools.image_generator_openrouter_api._post_json", post):
            with self.assertRaises(OpenRouterImageAPIError):
                await generator.generate_single_image("bad request")
        self.assertEqual(post.await_count, 1)

    def test_retry_policy_is_bounded_to_transient_errors_and_portrait_outputs(self):
        self.assertTrue(_is_retryable_image_error(OpenRouterImageAPIError(429, {})))
        self.assertTrue(_is_retryable_image_error(OpenRouterImageAPIError(500, {})))
        self.assertTrue(_is_retryable_image_error(ValueError("Generated image is portrait-oriented (9x16); retrying for a landscape frame")))
        self.assertFalse(_is_retryable_image_error(OpenRouterImageAPIError(401, {})))
        self.assertFalse(_is_retryable_image_error(ValueError("invalid reference image")))

    def test_a_400_the_provider_says_is_its_own_fault_is_repeated(self):
        # Alibaba answers qwen-image-3 with this for a prompt and references, drawn with a
        # little trouble, that succeed on the next attempt: the request is OpenRouter's to
        # forward, so the failure is the provider's and not the request's.
        alibaba = OpenRouterImageAPIError(400, {"error": {"message": "<400> InternalError.Algo: Invalid request.", "code": 400, "metadata": {"provider_name": "Alibaba"}}})
        self.assertTrue(_is_retryable_image_error(alibaba))
        # A 400 with no provider behind it is the request itself, and repeating it only
        # asks the same wrong question.
        self.assertFalse(_is_retryable_image_error(OpenRouterImageAPIError(400, {"error": {"message": "No endpoints found that support the requested output modalities."}})))
        self.assertFalse(_is_retryable_image_error(OpenRouterImageAPIError(400, {"error": {"message": "bad request", "metadata": {}}})))
        self.assertFalse(_is_retryable_image_error(OpenRouterImageAPIError(400, None)))

    def test_agent_factory_selects_openrouter_from_image_base_url(self):
        with patch("agent_runtime.vimax_adapters.image_api_key", return_value="secret"), \
             patch("agent_runtime.vimax_adapters.image_model", return_value="openai/gpt-image-2"), \
             patch("agent_runtime.vimax_adapters.image_base_url", return_value="https://openrouter.ai/api/v1"):
            generator = _build_image_generator()
        self.assertIsInstance(generator, ImageGeneratorOpenRouterAPI)
        self.assertEqual(generator.model, "openai/gpt-image-2")


if __name__ == "__main__":
    unittest.main()
