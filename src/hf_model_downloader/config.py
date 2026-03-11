"""Configuration models using Pydantic."""

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class DownloadConfig(BaseModel):
    """Configuration for a single download."""

    model_id: str = Field(..., description="HuggingFace model ID")
    revision: str = Field(default="main", description="Model revision/branch")
    output_dir: str | None = Field(default=None, description="Output directory")
    include_patterns: list[str] | None = Field(default=None, description="File patterns to include")
    exclude_patterns: list[str] | None = Field(default=None, description="File patterns to exclude")


class ProfileConfig(BaseModel):
    """Configuration profile for downloads."""

    name: str = Field(..., description="Profile name")
    default_output_dir: str | None = Field(default=None, description="Default output directory")
    downloads: list[DownloadConfig] = Field(
        default_factory=list, description="Download configurations"
    ),


class AppConfig(BaseSettings):
    """Application-wide configuration."""

    cache_dir: str | None = Field(default=None, description="Cache directory for downloads")
    default_profile: str | None = Field(default=None, description="Default profile to use")

    class Config:
        env_prefix = "HFMDL_"
