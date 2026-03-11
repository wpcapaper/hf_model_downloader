"""Environment configuration for HuggingFace Hub.

This module handles setting environment variables BEFORE any huggingface_hub import.
Critical: HF_ENDPOINT must be set before importing huggingface_hub for it to take effect.
"""

from __future__ import annotations

import os

from rich.console import Console

from .config import Settings

console = Console()


def apply_hf_env(settings: Settings, force_endpoint: bool = False) -> dict[str, str]:
    """Apply HuggingFace environment variables before any HF hub import.

    Sets environment variables for:
    - HF_ENDPOINT (mirror URL, respects existing env var unless force_endpoint=True)
    - HF_HUB_DOWNLOAD_TIMEOUT
    - HF_HUB_ETAG_TIMEOUT
    - HF_XET_HIGH_PERFORMANCE (if enabled)
    - HF_XET_NUM_CONCURRENT_RANGE_GETS
    - HF_HOME (if cache_dir is set)

    Args:
        settings: Application settings containing endpoint, timeouts, etc.
        force_endpoint: If True, override existing HF_ENDPOINT env var with settings.endpoint

    Returns:
        Dict of effective environment variable values (for display/debugging)

    IMPORTANT: Never log HF_TOKEN value - it's read from env but never displayed.
    """
    effective_env: dict[str, str] = {}

    # Handle HF_ENDPOINT
    existing_endpoint = os.environ.get("HF_ENDPOINT")
    if existing_endpoint and not force_endpoint:
        # Respect user's existing HF_ENDPOINT unless forced
        effective_env["HF_ENDPOINT"] = existing_endpoint
        console.print(
            f"[dim]Using existing HF_ENDPOINT from environment: {existing_endpoint}[/]"
        )
    else:
        # Set endpoint from settings
        os.environ["HF_ENDPOINT"] = settings.endpoint
        effective_env["HF_ENDPOINT"] = settings.endpoint
        if force_endpoint and existing_endpoint:
            console.print(
                f"[yellow]Overriding HF_ENDPOINT: {existing_endpoint} → {settings.endpoint}[/]"
            )
        else:
            console.print(f"[dim]Setting HF_ENDPOINT: {settings.endpoint}[/]")

    # Set download timeout
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(settings.hf_hub_download_timeout)
    effective_env["HF_HUB_DOWNLOAD_TIMEOUT"] = str(settings.hf_hub_download_timeout)
    console.print(
        f"[dim]Setting HF_HUB_DOWNLOAD_TIMEOUT: {settings.hf_hub_download_timeout}s[/]"
    )

    # Set etag timeout
    os.environ["HF_HUB_ETAG_TIMEOUT"] = str(settings.hf_hub_etag_timeout)
    effective_env["HF_HUB_ETAG_TIMEOUT"] = str(settings.hf_hub_etag_timeout)
    console.print(
        f"[dim]Setting HF_HUB_ETAG_TIMEOUT: {settings.hf_hub_etag_timeout}s[/]"
    )

    # Set high performance mode for hf_xet
    if settings.hf_xet_high_performance:
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
        effective_env["HF_XET_HIGH_PERFORMANCE"] = "1"
        console.print("[dim]Setting HF_XET_HIGH_PERFORMANCE: 1[/]")
    else:
        # Explicitly disable if configured
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "0"
        effective_env["HF_XET_HIGH_PERFORMANCE"] = "0"
        console.print("[dim]Setting HF_XET_HIGH_PERFORMANCE: 0[/]")

    # Set concurrent range gets
    os.environ["HF_XET_NUM_CONCURRENT_RANGE_GETS"] = str(
        settings.hf_xet_num_concurrent_range_gets
    )
    effective_env["HF_XET_NUM_CONCURRENT_RANGE_GETS"] = str(
        settings.hf_xet_num_concurrent_range_gets
    )
    console.print(
        f"[dim]Setting HF_XET_NUM_CONCURRENT_RANGE_GETS: "
        f"{settings.hf_xet_num_concurrent_range_gets}[/]"
    )

    # Set HF_HOME if cache_dir is configured
    if settings.cache_dir:
        os.environ["HF_HOME"] = settings.cache_dir
        effective_env["HF_HOME"] = settings.cache_dir
        console.print(f"[dim]Setting HF_HOME: {settings.cache_dir}[/]")

    # Note: HF_TOKEN is read from environment but NEVER logged or persisted
    if os.environ.get("HF_TOKEN"):
        effective_env["HF_TOKEN"] = "***REDACTED***"
        console.print("[dim]HF_TOKEN found in environment (value not logged)[/]")

    return effective_env
