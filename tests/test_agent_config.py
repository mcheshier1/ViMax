import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from agent_runtime.config import (
    api_provider_from_base_url,
    embedding_api_key,
    embedding_base_url,
    embedding_model,
    embedding_model_provider,
    image_api_key,
    image_base_url,
    image_model,
    image_resolution,
    llm_api_key,
    llm_base_url,
    llm_model,
    llm_model_provider,
    load_agent_config,
    reranker_api_key,
    reranker_base_url,
    reranker_model,
    video_api_key,
    video_base_url,
    video_model,
    video_provider,
)


class AgentConfigTests(unittest.TestCase):
    def setUp(self):
        load_agent_config.cache_clear()

    def tearDown(self):
        load_agent_config.cache_clear()

    def test_reads_agent_local_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / "configs"
            config_dir.mkdir()
            (config_dir / "agent.local.yaml").write_text(yaml.safe_dump({
                "llm": {"model_provider": "openai", "model": "config-llm", "base_url": "https://config.test/v1", "api_key": "config-key"},
                "image": {"model": "config-image", "base_url": "https://image.test", "api_key": "image-key"},
                "video": {"model": "config-video", "base_url": "https://openrouter.ai/api/v1", "api_key": "video-key"},
                "embedding": {"model_provider": "openai", "model": "config-embedding", "base_url": "https://embedding.test/v1", "api_key": "embedding-key"},
                "reranker": {"model": "config-reranker", "base_url": "https://reranker.test", "api_key": "reranker-key"},
            }), encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(llm_model(tmp), "config-llm")
                self.assertEqual(llm_model_provider(tmp), "openai")
                self.assertEqual(llm_base_url(tmp), "https://config.test/v1")
                self.assertEqual(llm_api_key(tmp), "config-key")
                self.assertEqual(image_model(tmp), "config-image")
                self.assertEqual(image_base_url(tmp), "https://image.test")
                self.assertEqual(image_api_key(tmp), "image-key")
                self.assertEqual(video_model(tmp), "config-video")
                self.assertEqual(video_provider(tmp), "openrouter")
                self.assertEqual(video_base_url(tmp), "https://openrouter.ai/api/v1")
                self.assertEqual(video_api_key(tmp), "video-key")
                self.assertEqual(embedding_model_provider(tmp), "openai")
                self.assertEqual(embedding_model(tmp), "config-embedding")
                self.assertEqual(embedding_base_url(tmp), "https://embedding.test/v1")
                self.assertEqual(embedding_api_key(tmp), "embedding-key")
                self.assertEqual(reranker_model(tmp), "config-reranker")
                self.assertEqual(reranker_base_url(tmp), "https://reranker.test")
                self.assertEqual(reranker_api_key(tmp), "reranker-key")

    def test_environment_overrides_agent_local_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / "configs"
            config_dir.mkdir()
            (config_dir / "agent.local.yaml").write_text(yaml.safe_dump({"llm": {"model": "config-llm", "api_key": "config-key"}}), encoding="utf-8")
            with patch.dict(os.environ, {"VIMAX_LLM_MODEL": "env-llm", "VIMAX_LLM_MODEL_PROVIDER": "openai", "VIMAX_LLM_API_KEY": "env-key", "VIMAX_VIDEO_BASE_URL": "https://openrouter.ai/api/v1", "VIMAX_EMBEDDING_MODEL": "env-embedding", "VIMAX_EMBEDDING_BASE_URL": "https://env-embedding.test/v1", "VIMAX_EMBEDDING_API_KEY": "env-embedding-key", "VIMAX_RERANKER_MODEL": "env-reranker", "VIMAX_RERANKER_BASE_URL": "https://env-reranker.test", "VIMAX_RERANKER_API_KEY": "env-reranker-key"}, clear=True):
                self.assertEqual(llm_model(tmp), "env-llm")
                self.assertEqual(llm_model_provider(tmp), "openai")
                self.assertEqual(llm_api_key(tmp), "env-key")
                self.assertEqual(video_provider(tmp), "openrouter")
                self.assertEqual(video_base_url(tmp), "https://openrouter.ai/api/v1")
                self.assertEqual(embedding_model(tmp), "env-embedding")
                self.assertEqual(embedding_base_url(tmp), "https://env-embedding.test/v1")
                self.assertEqual(embedding_api_key(tmp), "env-embedding-key")
                self.assertEqual(reranker_model(tmp), "env-reranker")
                self.assertEqual(reranker_base_url(tmp), "https://env-reranker.test")
                self.assertEqual(reranker_api_key(tmp), "env-reranker-key")

    def test_image_and_video_keys_fall_back_to_llm_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / "configs"
            config_dir.mkdir()
            (config_dir / "agent.local.yaml").write_text(yaml.safe_dump({"llm": {"api_key": "shared-key"}}), encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(image_api_key(tmp), "shared-key")
                self.assertEqual(video_api_key(tmp), "shared-key")

    def test_provider_banks_select_each_category_key_without_using_stale_active_mirror(self):
        getters = {
            "llm": llm_api_key, "image": image_api_key, "video": video_api_key,
            "embedding": embedding_api_key, "reranker": reranker_api_key,
        }
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            config_dir = Path(tmp) / "configs"
            config_dir.mkdir()
            config_path = config_dir / "agent.local.yaml"
            for provider, base_url in [
                ("openrouter", "https://openrouter.ai/api/v1"),
                ("agnes", "https://apihub.agnes-ai.com/v1"),
                ("endpoint:https://custom.test/v1", "https://custom.test/v1/"),
            ]:
                with self.subTest(provider=provider):
                    config_path.write_text(yaml.safe_dump({
                        category: {
                            "base_url": base_url, "api_key": "stale-other-provider",
                            "api_keys": {
                                "openrouter": f"{category}-router",
                                "agnes": f"{category}-agnes",
                                "endpoint:https://custom.test/v1": f"{category}-custom",
                            },
                        }
                        for category in getters
                    }), encoding="utf-8")
                    load_agent_config.cache_clear()
                    suffix = "router" if provider == "openrouter" else "agnes" if provider == "agnes" else "custom"
                    for category, getter in getters.items():
                        self.assertEqual(getter(tmp), f"{category}-{suffix}")

    def test_provider_bank_does_not_forward_llm_key_to_an_unconfigured_other_provider(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"VIMAX_LLM_API_KEY": "router-env"}, clear=True):
            config_dir = Path(tmp) / "configs"
            config_dir.mkdir()
            (config_dir / "agent.local.yaml").write_text(yaml.safe_dump({
                "llm": {"base_url": "https://openrouter.ai/api/v1", "api_keys": {"openrouter": "router"}},
                "image": {"base_url": "https://apihub.agnes-ai.com/v1", "api_keys": {"openrouter": "image-router"}},
                "video": {"base_url": "https://api.ltx.io", "api_keys": {"openrouter": "video-router"}},
            }), encoding="utf-8")
            self.assertEqual(image_api_key(tmp), "")
            self.assertEqual(video_api_key(tmp), "")
            with patch.dict(os.environ, {"VIMAX_IMAGE_API_KEY": "explicit-image", "VIMAX_VIDEO_API_KEY": "explicit-video"}):
                self.assertEqual(image_api_key(tmp), "explicit-image")
                self.assertEqual(video_api_key(tmp), "explicit-video")

    def test_provider_bank_keeps_same_provider_llm_fallback(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            config_dir = Path(tmp) / "configs"
            config_dir.mkdir()
            (config_dir / "agent.local.yaml").write_text(yaml.safe_dump({
                "llm": {"base_url": "https://openrouter.ai/api/v1", "api_keys": {"openrouter": "shared-router"}},
                "image": {"base_url": "https://openrouter.ai/api/v1", "api_keys": {}},
                "video": {"base_url": "https://openrouter.ai/api/v1", "api_keys": {}},
            }), encoding="utf-8")
            self.assertEqual(image_api_key(tmp), "shared-router")
            self.assertEqual(video_api_key(tmp), "shared-router")

    def test_endpoint_override_selects_matching_saved_key_not_old_endpoint_key(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"VIMAX_EMBEDDING_BASE_URL": "https://new.test/v1"}, clear=True):
            config_dir = Path(tmp) / "configs"
            config_dir.mkdir()
            (config_dir / "agent.local.yaml").write_text(yaml.safe_dump({
                "embedding": {
                    "base_url": "https://old.test/v1", "api_key": "old",
                    "api_keys": {"endpoint:https://old.test/v1": "old", "endpoint:https://new.test/v1": "new"},
                },
            }), encoding="utf-8")
            self.assertEqual(embedding_api_key(tmp), "new")
            with patch.dict(os.environ, {"VIMAX_EMBEDDING_BASE_URL": "https://openrouter.ai.attacker.test/v1"}):
                self.assertEqual(embedding_api_key(tmp), "")

    def test_seedream_resolution_defaults_to_1k_without_changing_video_resolution(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            config_dir = Path(tmp) / "configs"
            config_dir.mkdir()
            (config_dir / "agent.local.yaml").write_text(yaml.safe_dump({
                "image": {"model": "bytedance-seed/seedream-5-0-flash"},
                "video": {"resolution": "480p"},
            }), encoding="utf-8")
            self.assertEqual(image_resolution(tmp), "1K")
            self.assertEqual(load_agent_config(tmp)["video"]["resolution"], "480p")

    def test_image_resolution_supports_saved_tiers_provider_default_and_environment_override(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            config_dir = Path(tmp) / "configs"
            config_dir.mkdir()
            config_path = config_dir / "agent.local.yaml"
            for configured in ("1K", "2K", "auto"):
                with self.subTest(resolution=configured):
                    config_path.write_text(yaml.safe_dump({
                        "image": {"model": "bytedance-seed/seedream-5-0-flash", "resolution": configured},
                    }), encoding="utf-8")
                    load_agent_config.cache_clear()
                    self.assertEqual(image_resolution(tmp), configured)
                    with patch.dict(os.environ, {"VIMAX_IMAGE_RESOLUTION": "1K"}):
                        self.assertEqual(image_resolution(tmp), "1K")
            config_path.write_text(yaml.safe_dump({"image": {"model": "qwen/qwen-image-3"}}), encoding="utf-8")
            load_agent_config.cache_clear()
            self.assertEqual(image_resolution(tmp), "auto")

    def test_video_provider_is_inferred_from_base_url(self):
        self.assertEqual(api_provider_from_base_url("https://openrouter.ai/api/v1"), "openrouter")
        self.assertEqual(api_provider_from_base_url("https://yunwu.ai/v1"), "yunwu")
        self.assertEqual(api_provider_from_base_url("https://example.com/v1"), "")


if __name__ == "__main__":
    unittest.main()
