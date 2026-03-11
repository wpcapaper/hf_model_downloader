"""Configuration models using Pydantic with TOML file support."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from platformdirs import PlatformDirs
from pydantic import BaseModel, Field

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import tomli_w

# Platform-specific config directory
_DIRS = PlatformDirs(appname="hfmdl", appauthor=False)
CONFIG_FILE_NAME = "config.toml"


class RetrySettings(BaseModel):
    """Retry configuration for downloads."""

    forever: bool = Field(
        default=True, description="Retry indefinitely on failures"
    )
    max_attempts: int | None = Field(
        default=None, description="Maximum retry attempts (ignored if forever=True)"
    )
    max_total_seconds: float | None = Field(
        default=None, description="Maximum total seconds to retry (None = no limit)"
    )
    base_wait: float = Field(
        default=1.0, description="Base wait time between retries in seconds"
    )
    max_wait: float = Field(
        default=60.0, description="Maximum wait time between retries in seconds"
    )
    jitter: float = Field(
        default=0.2, description="Jitter factor for wait time (0.0-1.0)"
    )
    log_every_attempt: bool = Field(
        default=True, description="Log every retry attempt"
    )


class ModelConfig(BaseModel):
    """Configuration for a single model to download."""

    name: str = Field(..., description="Friendly name for this model")
    repo_id: str = Field(
        ..., description="HuggingFace repository ID (e.g., 'bert-base-uncased')"
    )
    revision: str = Field(default="main", description="Model revision/branch/tag")
    repo_type: str = Field(
        default="model", description="Repository type: 'model', 'dataset', or 'space'"
    )
    allow_patterns: list[str] | None = Field(
        default=None, description="File patterns to include"
    )
    ignore_patterns: list[str] | None = Field(
        default=None, description="File patterns to exclude"
    )
    output_dir: str | None = Field(
        default=None, description="Output directory (overrides global setting)"
    )


class Settings(BaseModel):
    """Application settings loaded from config file."""

    # Endpoint configuration
    endpoint: str = Field(
        default="https://hf-mirror.com",
        description="HuggingFace endpoint URL",
    )

    # Timeout settings
    hf_hub_download_timeout: float = Field(
        default=60.0, description="Timeout for downloads in seconds"
    )
    hf_hub_etag_timeout: float = Field(
        default=30.0, description="Timeout for etag requests in seconds"
    )

    # Performance settings
    hf_xet_high_performance: bool = Field(
        default=True, description="Enable high performance mode for hf_xet"
    )
    hf_xet_num_concurrent_range_gets: int = Field(
        default=16, description="Number of concurrent range requests"
    )
    max_workers: int = Field(
        default=8, description="Maximum number of parallel workers"
    )

    # Storage settings
    cache_dir: str | None = Field(
        default=None, description="Cache directory for downloads (None = platform default)"
    )

    # Retry configuration
    retry: RetrySettings = Field(
        default_factory=RetrySettings, description="Retry configuration"
    )

    # Model list
    models: list[ModelConfig] = Field(
        default_factory=list, description="List of models to download"
    )

    @classmethod
    def get_config_path(cls) -> Path:
        """Get the platform-specific config file path."""
        config_dir = Path(_DIRS.user_config_dir)
        return config_dir / CONFIG_FILE_NAME

    @classmethod
    def load(cls, config_path: Path | None = None) -> Settings:
        """Load settings from TOML file, creating default if missing."""
        if config_path is None:
            config_path = cls.get_config_path()

        if not config_path.exists():
            # Auto-create minimal default config
            cls._create_default_config(config_path)
            return cls()

        with open(config_path, "rb") as f:
            data = tomllib.load(f)

        return cls.model_validate(data)

    @classmethod
    def _create_default_config(cls, config_path: Path) -> None:
        """Create a minimal default config file."""
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Create minimal default settings
        default = cls()
        data = default.model_dump(mode="json", exclude_none=True)

        # Write to TOML file
        with open(config_path, "wb") as f:
            tomli_w.dump(data, f)

    def merge_cli_overrides(
        self,
        *,
        endpoint: str | None = None,
        force_endpoint: bool = False,
        cache_dir: str | None = None,
        max_workers: int | None = None,
    ) -> Settings:
        """Create a new Settings with CLI overrides applied.

        CLI flags override config file values.
        """
        data = self.model_dump()

        if endpoint is not None:
            data["endpoint"] = endpoint

        if cache_dir is not None:
            data["cache_dir"] = cache_dir

        if max_workers is not None:
            data["max_workers"] = max_workers

        return Settings.model_validate(data)

    def get_effective_endpoint(self, force_endpoint: bool = False) -> str:
        """Get the effective endpoint, respecting environment variables.

        Environment variable HF_ENDPOINT is respected unless force_endpoint=True.
        """
        if not force_endpoint:
            env_endpoint = os.environ.get("HF_ENDPOINT")
            if env_endpoint:
                return env_endpoint
        return self.endpoint

    def get_hf_token(self) -> str | None:
        """Get HuggingFace token from environment.

        Token is NEVER persisted to disk, only read from environment.
        """
        return os.environ.get("HF_TOKEN")


def load_settings(
    config_path: Path | None = None,
    *,
    endpoint: str | None = None,
    force_endpoint: bool = False,
    cache_dir: str | None = None,
    max_workers: int | None = None,
) -> Settings:
    """Load settings with optional CLI overrides.

    This is the main entry point for loading configuration.
    """
    settings = Settings.load(config_path)
    return settings.merge_cli_overrides(
        endpoint=endpoint,
        force_endpoint=force_endpoint,
        cache_dir=cache_dir,
        max_workers=max_workers,
    )
