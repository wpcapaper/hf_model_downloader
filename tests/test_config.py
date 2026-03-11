"""Unit tests for configuration management."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from hf_model_downloader.config import (
    ModelConfig,
    RetrySettings,
    Settings,
    load_settings,
)


class TestConfigLoadCreate:
    """Tests for config loading and auto-creation."""

    def test_load_creates_default_config_when_missing(self, tmp_path: Path) -> None:
        """Test that Settings.load() creates a default config if it doesn't exist."""
        config_path = tmp_path / "config.toml"
        assert not config_path.exists()

        # Load config - should auto-create
        settings = Settings.load(config_path)

        # Verify file was created
        assert config_path.exists()

        # Verify default values
        assert settings.endpoint == "https://hf-mirror.com"
        assert settings.hf_hub_download_timeout == 60.0
        assert settings.hf_hub_etag_timeout == 30.0
        assert settings.hf_xet_high_performance is True
        assert settings.hf_xet_num_concurrent_range_gets == 16
        assert settings.max_workers == 8
        assert settings.cache_dir is None
        assert settings.retry.forever is True
        assert settings.retry.max_attempts is None
        assert settings.retry.max_total_seconds is None
        assert settings.retry.base_wait == 1.0
        assert settings.retry.max_wait == 60.0
        assert settings.retry.jitter == 0.2
        assert settings.retry.log_every_attempt is True
        assert settings.models == []

    def test_load_existing_config(self, tmp_path: Path) -> None:
        """Test loading from an existing config file."""
        config_path = tmp_path / "config.toml"

        # Create a config file with custom values
        config_content = """
endpoint = "https://custom.huggingface.co"
hf_hub_download_timeout = 120.0
max_workers = 16

[retry]
forever = false
max_attempts = 10
base_wait = 2.0
max_wait = 120.0

[[models]]
name = "bert"
repo_id = "bert-base-uncased"
revision = "main"
"""
        config_path.write_text(config_content)

        # Load config
        settings = Settings.load(config_path)

        # Verify custom values
        assert settings.endpoint == "https://custom.huggingface.co"
        assert settings.hf_hub_download_timeout == 120.0
        assert settings.max_workers == 16
        assert settings.retry.forever is False
        assert settings.retry.max_attempts == 10
        assert settings.retry.base_wait == 2.0
        assert settings.retry.max_wait == 120.0
        assert len(settings.models) == 1
        assert settings.models[0].name == "bert"
        assert settings.models[0].repo_id == "bert-base-uncased"

    def test_load_partial_config_uses_defaults(self, tmp_path: Path) -> None:
        """Test that missing config values use defaults."""
        config_path = tmp_path / "config.toml"

        # Create a minimal config
        config_content = """
endpoint = "https://test.endpoint"
"""
        config_path.write_text(config_content)

        # Load config
        settings = Settings.load(config_path)

        # Verify specified value
        assert settings.endpoint == "https://test.endpoint"
        # Verify defaults for missing values
        assert settings.hf_hub_download_timeout == 60.0
        assert settings.max_workers == 8


class TestCLIOverrides:
    """Tests for CLI override precedence."""

    def test_merge_cli_overrides_endpoint(self, tmp_path: Path) -> None:
        """Test that CLI endpoint overrides config value."""
        config_path = tmp_path / "config.toml"
        config_content = """
endpoint = "https://config.endpoint"
max_workers = 8
"""
        config_path.write_text(config_content)

        settings = Settings.load(config_path)
        assert settings.endpoint == "https://config.endpoint"

        # Apply CLI override
        overridden = settings.merge_cli_overrides(endpoint="https://cli.endpoint")
        assert overridden.endpoint == "https://cli.endpoint"
        # Other values unchanged
        assert overridden.max_workers == 8

    def test_merge_cli_overrides_cache_dir(self, tmp_path: Path) -> None:
        """Test that CLI cache_dir overrides config value."""
        config_path = tmp_path / "config.toml"
        config_content = """
cache_dir = "/config/cache"
"""
        config_path.write_text(config_content)

        settings = Settings.load(config_path)
        assert settings.cache_dir == "/config/cache"

        # Apply CLI override
        overridden = settings.merge_cli_overrides(cache_dir="/cli/cache")
        assert overridden.cache_dir == "/cli/cache"

    def test_merge_cli_overrides_max_workers(self, tmp_path: Path) -> None:
        """Test that CLI max_workers overrides config value."""
        config_path = tmp_path / "config.toml"
        config_content = """
max_workers = 4
"""
        config_path.write_text(config_content)

        settings = Settings.load(config_path)
        assert settings.max_workers == 4

        # Apply CLI override
        overridden = settings.merge_cli_overrides(max_workers=32)
        assert overridden.max_workers == 32

    def test_merge_cli_overrides_multiple(self, tmp_path: Path) -> None:
        """Test that multiple CLI overrides work together."""
        config_path = tmp_path / "config.toml"
        config_content = """
endpoint = "https://config.endpoint"
cache_dir = "/config/cache"
max_workers = 8
"""
        config_path.write_text(config_content)

        settings = Settings.load(config_path)

        # Apply multiple overrides
        overridden = settings.merge_cli_overrides(
            endpoint="https://cli.endpoint",
            cache_dir="/cli/cache",
            max_workers=16,
        )

        assert overridden.endpoint == "https://cli.endpoint"
        assert overridden.cache_dir == "/cli/cache"
        assert overridden.max_workers == 16

    def test_merge_cli_overrides_none_unchanged(self, tmp_path: Path) -> None:
        """Test that None values don't override config."""
        config_path = tmp_path / "config.toml"
        config_content = """
endpoint = "https://config.endpoint"
max_workers = 8
"""
        config_path.write_text(config_content)

        settings = Settings.load(config_path)

        # Apply override with None values
        overridden = settings.merge_cli_overrides(
            endpoint=None,
            cache_dir=None,
            max_workers=None,
        )

        # Values unchanged
        assert overridden.endpoint == "https://config.endpoint"
        assert overridden.cache_dir is None
        assert overridden.max_workers == 8

    def test_merge_cli_overrides_creates_new_instance(self, tmp_path: Path) -> None:
        """Test that merge_cli_overrides creates a new Settings instance."""
        settings = Settings()

        # Apply override
        overridden = settings.merge_cli_overrides(max_workers=16)

        # Original unchanged
        assert settings.max_workers == 8
        # New instance has override
        assert overridden.max_workers == 16
        assert settings is not overridden


class TestEffectiveEndpoint:
    """Tests for effective endpoint resolution."""

    def test_get_effective_endpoint_respects_env_var(self) -> None:
        """Test that HF_ENDPOINT env var is respected."""
        settings = Settings(endpoint="https://config.endpoint")

        with patch.dict(os.environ, {"HF_ENDPOINT": "https://env.endpoint"}):
            effective = settings.get_effective_endpoint(force_endpoint=False)
            assert effective == "https://env.endpoint"

    def test_get_effective_endpoint_force_overrides_env_var(self) -> None:
        """Test that force_endpoint=True overrides HF_ENDPOINT env var."""
        settings = Settings(endpoint="https://config.endpoint")

        with patch.dict(os.environ, {"HF_ENDPOINT": "https://env.endpoint"}):
            effective = settings.get_effective_endpoint(force_endpoint=True)
            assert effective == "https://config.endpoint"

    def test_get_effective_endpoint_no_env_var(self) -> None:
        """Test that config value is used when no env var is set."""
        settings = Settings(endpoint="https://config.endpoint")

        with patch.dict(os.environ, {}, clear=True):
            effective = settings.get_effective_endpoint(force_endpoint=False)
            assert effective == "https://config.endpoint"


class TestHFToken:
    """Tests for HuggingFace token retrieval."""

    def test_get_hf_token_from_env(self) -> None:
        """Test that HF_TOKEN is read from environment."""
        settings = Settings()

        with patch.dict(os.environ, {"HF_TOKEN": "test_token_123"}):
            token = settings.get_hf_token()
            assert token == "test_token_123"

    def test_get_hf_token_missing(self) -> None:
        """Test that None is returned when HF_TOKEN is not set."""
        settings = Settings()

        with patch.dict(os.environ, {}, clear=True):
            token = settings.get_hf_token()
            assert token is None


class TestLoadSettings:
    """Tests for the load_settings convenience function."""

    def test_load_settings_with_overrides(self, tmp_path: Path) -> None:
        """Test that load_settings applies CLI overrides."""
        config_path = tmp_path / "config.toml"
        config_content = """
endpoint = "https://config.endpoint"
max_workers = 8
"""
        config_path.write_text(config_content)

        # Load with overrides
        settings = load_settings(
            config_path,
            endpoint="https://cli.endpoint",
            max_workers=16,
        )

        assert settings.endpoint == "https://cli.endpoint"
        assert settings.max_workers == 16

    def test_load_settings_auto_creates(self, tmp_path: Path) -> None:
        """Test that load_settings auto-creates config if missing."""
        config_path = tmp_path / "config.toml"
        assert not config_path.exists()

        # Load should auto-create
        settings = load_settings(config_path)

        assert config_path.exists()
        assert settings.endpoint == "https://hf-mirror.com"


class TestConfigModels:
    """Tests for Pydantic config models."""

    def test_retry_settings_defaults(self) -> None:
        """Test RetrySettings default values."""
        retry = RetrySettings()
        assert retry.forever is True
        assert retry.max_attempts is None
        assert retry.max_total_seconds is None
        assert retry.base_wait == 1.0
        assert retry.max_wait == 60.0
        assert retry.jitter == 0.2
        assert retry.log_every_attempt is True

    def test_model_config_defaults(self) -> None:
        """Test ModelConfig default values."""
        model = ModelConfig(name="test", repo_id="test/model")
        assert model.name == "test"
        assert model.repo_id == "test/model"
        assert model.revision == "main"
        assert model.repo_type == "model"
        assert model.allow_patterns is None
        assert model.ignore_patterns is None
        assert model.output_dir is None

    def test_settings_defaults(self) -> None:
        """Test Settings default values."""
        settings = Settings()
        assert settings.endpoint == "https://hf-mirror.com"
        assert settings.hf_hub_download_timeout == 60.0
        assert settings.hf_hub_etag_timeout == 30.0
        assert settings.hf_xet_high_performance is True
        assert settings.hf_xet_num_concurrent_range_gets == 16
        assert settings.max_workers == 8
        assert settings.cache_dir is None
        assert isinstance(settings.retry, RetrySettings)
        assert settings.models == []
