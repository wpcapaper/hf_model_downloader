#NJ|"""Downloader module for HuggingFace models."""

from __future__ import annotations

import random
import signal
import time
from pathlib import Path
from typing import Any

from rich.console import Console

from .config import Settings
from .env import apply_hf_env
from .errors import DownloadError, classify_error

console = Console()

# Global flag for Ctrl+C handling
_interrupted = False


def _handle_interrupt(signum: int, frame: Any) -> None:
    """Handle Ctrl+C gracefully."""
    global _interrupted
    _interrupted = True
    console.print("\n[yellow]Interrupted by user. Exiting gracefully...[/]")
    raise KeyboardInterrupt("Download interrupted by user")


def _calculate_backoff(attempt: int, base_wait: float, max_wait: float, jitter: float) -> float:
    """Calculate exponential backoff with jitter.

    Args:
        attempt: Current attempt number (0-indexed)
        base_wait: Base wait time in seconds
        max_wait: Maximum wait time in seconds
        jitter: Jitter factor (0.0-1.0)

    Returns:
        Wait time in seconds
    """
    # Exponential backoff: base_wait * 2^attempt
    wait = base_wait * (2 ** attempt)
    # Cap at max_wait
    wait = min(wait, max_wait)
    # Apply jitter: random value between (1-jitter)*wait and (1+jitter)*wait
    if jitter > 0:
        jitter_factor = 1 + random.uniform(-jitter, jitter)
        wait = wait * jitter_factor
    return wait


def download_snapshot(
    repo_id: str,
    *,
    revision: str = "main",
    repo_type: str = "model",
    cache_dir: str | Path | None = None,
    force_download: bool = False,
    token: str | None = None,
    settings: Settings | None = None,
    force_endpoint: bool = False,
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
) -> Path:
    """Download a snapshot from HuggingFace Hub with retry logic.

    This is the main download function that wraps huggingface_hub.snapshot_download
    with retry logic, progress reporting, and error handling.

    Args:
        repo_id: HuggingFace repository ID (e.g., 'bert-base-uncased')
        revision: Model revision/branch/tag
        repo_type: Repository type ('model', 'dataset', or 'space')
        cache_dir: Cache directory (None = platform default)
        force_download: Force re-download even if cached
        token: HuggingFace API token (None = use HF_TOKEN env var)
        settings: Application settings (loaded if None)
        force_endpoint: Override HF_ENDPOINT env var with settings.endpoint
        allow_patterns: File patterns to include
        ignore_patterns: File patterns to exclude

    Returns:
        Path to the downloaded snapshot

    Raises:
        DownloadError: If download fails after all retries
        KeyboardInterrupt: If user interrupts with Ctrl+C
    """
    global _interrupted
    _interrupted = False

    # Set up Ctrl+C handler
    old_handler = signal.signal(signal.SIGINT, _handle_interrupt)

    try:
        # Load settings if not provided
        if settings is None:
            settings = Settings.load()

        retry_config = settings.retry

        # Apply HF environment variables before importing huggingface_hub
        apply_hf_env(settings, force_endpoint=force_endpoint)

        # LAZY IMPORT of huggingface_hub - must be after apply_hf_env
        from huggingface_hub import snapshot_download as hf_snapshot_download

        # Prepare download kwargs
        download_kwargs: dict[str, Any] = {
            "repo_id": repo_id,
            "revision": revision,
            "repo_type": repo_type,
            "force_download": force_download,
            "etag_timeout": settings.hf_hub_etag_timeout,
            "token": token,
        }

        if cache_dir is not None:
            download_kwargs["cache_dir"] = str(cache_dir)
        if allow_patterns is not None:
            download_kwargs["allow_patterns"] = allow_patterns
        if ignore_patterns is not None:
            download_kwargs["ignore_patterns"] = ignore_patterns

        # Retry loop
        attempt = 0
        start_time = time.monotonic()
        last_error: Exception | None = None

        while True:
            # Check for interruption
            if _interrupted:
                raise KeyboardInterrupt("Download interrupted")

            # Check retry limits
            if retry_config.max_total_seconds is not None:
                elapsed = time.monotonic() - start_time
                if elapsed >= retry_config.max_total_seconds:
                    console.print(
                        f"[red]Max total time ({retry_config.max_total_seconds}s) exceeded[/]"
                    )
                    if last_error:
                        raise DownloadError(
                            f"Download failed after {elapsed:.1f}s: {last_error}"
                        ) from last_error
                    raise DownloadError(
                        f"Download failed after {elapsed:.1f}s: max time exceeded"
                    )

            if retry_config.max_attempts is not None and attempt >= retry_config.max_attempts:
                console.print(
                    f"[red]Max attempts ({retry_config.max_attempts}) exceeded[/]"
                )
                if last_error:
                    raise DownloadError(
                        f"Download failed after {attempt} attempts: {last_error}"
                    ) from last_error
                raise DownloadError(
                    f"Download failed after {attempt} attempts"
                )

            # Log attempt
            if attempt > 0 and retry_config.log_every_attempt:
                console.print(f"[dim]Attempt {attempt + 1}...[/]")

            try:
                # Attempt download
                result = hf_snapshot_download(**download_kwargs)
                path = Path(result)

                # Success!
                if attempt > 0:
                    console.print(
                        f"[green]✓ Download succeeded on attempt {attempt + 1}[/]"
                    )

                console.print(f"[green]Downloaded to:[/] {path}")
                return path

            except KeyboardInterrupt:
                # Re-raise Ctrl+C
                raise
            except Exception as exc:
                last_error = exc
                is_retriable, reason = classify_error(exc)

                if not is_retriable:
                    console.print(f"[red]✗ Non-retriable error: {reason}[/]")
                    raise DownloadError(f"Download failed: {reason}") from exc

                # Retriable error
                if not retry_config.forever:
                    if retry_config.max_attempts is not None:
                        remaining = retry_config.max_attempts - attempt - 1
                        if remaining <= 0:
                            console.print("[red]Max attempts exceeded[/]")
                            raise DownloadError(
                                f"Download failed after {attempt + 1} attempts: {reason}"
                            ) from exc

                # Calculate backoff
                backoff = _calculate_backoff(
                    attempt=attempt,
                    base_wait=retry_config.base_wait,
                    max_wait=retry_config.max_wait,
                    jitter=retry_config.jitter,
                )

                console.print(
                    f"[yellow]⚠ Attempt {attempt + 1} failed: {reason}[/]\n"
                    f"[dim]Retrying in {backoff:.1f}s...[/]"
                )

                # Wait with interrupt check
                wait_end = time.monotonic() + backoff
                while time.monotonic() < wait_end:
                    if _interrupted:
                        raise KeyboardInterrupt("Download interrupted during retry wait")
                    time.sleep(0.1)

                attempt += 1

    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, old_handler)


class ModelDownloader:
    """Handles downloading models from HuggingFace."""

    def __init__(self, model_id: str, **kwargs: Any) -> None:
        """Initialize the downloader.

        Args:
            model_id: HuggingFace model ID
            **kwargs: Additional configuration options
        """
        self.model_id = model_id
        self.config = kwargs

    def download(self) -> Path:
        """Download the model.

        Returns:
            Path to the downloaded model

        Raises:
            DownloadError: If download fails
        """
        return download_snapshot(self.model_id, **self.config)

    def validate(self) -> bool:
        """Validate the model configuration.

        Returns:
            True if valid, False otherwise
        """
        # TODO: Implement validation logic
        return True
