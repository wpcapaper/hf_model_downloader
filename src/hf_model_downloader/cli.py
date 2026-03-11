"""CLI interface for hf_model_downloader using Typer."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import ModelConfig, Settings, load_settings
from .downloader import download_snapshot
from .env import apply_hf_env
from .errors import (
    ConfigurationError,
    DownloadError,
    ProfileNotFoundError,
    ValidationError,
)

app = typer.Typer(
    name="hfmdl",
    help="HuggingFace Model Downloader - Download and manage HuggingFace models",
    no_args_is_help=True,
)
console = Console()

# Exit codes
EXIT_SUCCESS = 0
EXIT_CONFIG_ERROR = 2
EXIT_NON_RETRIABLE_ERROR = 3
EXIT_ABORTED = 4

# Version info
__version__ = "0.1.0"


def _version_callback(value: bool) -> None:
    """Callback to show version and exit."""
    if value:
        console.print(f"[blue]hfmdl[/] version [green]{__version__}[/]")
        raise typer.Exit(code=EXIT_SUCCESS)


def _get_model_by_profile(settings: Settings, profile: str) -> ModelConfig:
    """Get model configuration by profile name.

    Args:
        settings: Application settings
        profile: Profile name to look up

    Returns:
        ModelConfig for the profile

    Raises:
        ProfileNotFoundError: If profile doesn't exist
    """
    for model in settings.models:
        if model.name == profile:
            return model
    raise ProfileNotFoundError(f"Unknown profile: '{profile}'")


def _print_effective_settings(
    settings: Settings,
    repo_id: str,
    revision: str,
    repo_type: str,
    force_endpoint: bool,
    endpoint: str | None,
    output_dir: str | None,
    allow_patterns: list[str] | None,
    ignore_patterns: list[str] | None,
    max_workers: int | None,
    token: str | None,
    retry_forever: bool | None,
    max_attempts: int | None,
    max_total_seconds: float | None,
) -> None:
    """Print effective settings block at start of download."""
    # Apply environment to get effective endpoint
    effective_env = apply_hf_env(settings, force_endpoint=force_endpoint)
    effective_endpoint = effective_env.get("HF_ENDPOINT", settings.endpoint)
    if endpoint:
        effective_endpoint = endpoint

    # Build settings display
    settings_lines = [
        f"[cyan]Repository:[/] [green]{repo_id}[/]",
        f"[cyan]Revision:[/] [yellow]{revision}[/]",
        f"[cyan]Type:[/] [blue]{repo_type}[/]",
        "",
        f"[cyan]Endpoint:[/] [green]{effective_endpoint}[/]",
    ]

    if output_dir:
        settings_lines.append(f"[cyan]Output Dir:[/] [green]{output_dir}[/]")
    elif settings.cache_dir:
        settings_lines.append(f"[cyan]Cache Dir:[/] [green]{settings.cache_dir}[/]")
    else:
        settings_lines.append("[cyan]Cache Dir:[/] [dim](platform default)[/")

    if max_workers:
        settings_lines.append(f"[cyan]Max Workers:[/] [green]{max_workers}[/]")
    else:
        settings_lines.append(f"[cyan]Max Workers:[/] [green]{settings.max_workers}[/]")

    if allow_patterns:
        settings_lines.append(f"[cyan]Allow Patterns:[/] [green]{', '.join(allow_patterns)}[/]")
    if ignore_patterns:
        settings_lines.append(f"[cyan]Ignore Patterns:[/] [red]{', '.join(ignore_patterns)}[/]")

    # Retry settings
    retry_settings = settings.retry
    effective_forever = (
        retry_forever if retry_forever is not None else retry_settings.forever
    )
    effective_max_attempts = (
        max_attempts if max_attempts is not None else retry_settings.max_attempts
    )
    effective_max_seconds = (
        max_total_seconds if max_total_seconds is not None
        else retry_settings.max_total_seconds
    )

    settings_lines.append("")
    settings_lines.append(f"[cyan]Retry Forever:[/] [green]{effective_forever}[/]")
    if effective_max_attempts:
        settings_lines.append(f"[cyan]Max Attempts:[/] [green]{effective_max_attempts}[/]")
    if effective_max_seconds:
        settings_lines.append(f"[cyan]Max Total Time:[/] [green]{effective_max_seconds}s[/]")

    # Token status (never show value)
    if token or settings.get_hf_token():
        settings_lines.append("[cyan]Token:[/] [green]***REDACTED***[/")
    else:
        settings_lines.append("[cyan]Token:[/] [dim](not set)[/")

    panel = Panel(
        "\n".join(settings_lines),
        title="[bold blue]Effective Settings[/]",
        border_style="blue",
    )
    console.print(panel)
    console.print()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
) -> None:
    """HuggingFace Model Downloader - Download and manage HuggingFace models."""
    pass


@app.command()
def download(
    repo_id: str | None = typer.Argument(
        None, help="HuggingFace repository ID (e.g., 'bert-base-uncased')"
    ),
    profile: str | None = typer.Option(
        None, "--profile", "-p", help="Configuration profile to use"
    ),
    revision: str = typer.Option(
        "main", "--revision", help="Model revision/branch/tag"
    ),
    repo_type: str = typer.Option(
        "model", "--repo-type", help="Repository type: model, dataset, or space"
    ),
    endpoint: str | None = typer.Option(
        None, "--endpoint", help="Override endpoint URL"
    ),
    force_endpoint: bool = typer.Option(
        False,
        "--force-endpoint",
        "-f",
        help="Force config endpoint, ignore HF_ENDPOINT env var",
    ),
    output: str | None = typer.Option(
        None, "--output", "-o", help="Output/cache directory"
    ),
    allow_pattern: list[str] = typer.Option(
        [], "--allow-pattern", help="File patterns to include (can use multiple times)"
    ),
    ignore_pattern: list[str] = typer.Option(
        [], "--ignore-pattern", help="File patterns to exclude (can use multiple times)"
    ),
    max_workers: int | None = typer.Option(
        None, "--max-workers", help="Maximum parallel workers"
    ),
    token: str | None = typer.Option(
        None, "--token", help="HuggingFace API token (default: HF_TOKEN env var)"
    ),
    force_download: bool = typer.Option(
        False, "--force-download", help="Force re-download even if cached"
    ),
    retry_forever: bool | None = typer.Option(
        None, "--retry-forever", help="Retry indefinitely on failures"
    ),
    no_retry_forever: bool = typer.Option(
        False, "--no-retry-forever", help="Disable retry forever mode"
    ),
    max_attempts: int | None = typer.Option(
        None, "--max-attempts", help="Maximum retry attempts"
    ),
    max_total_seconds: float | None = typer.Option(
        None, "--max-total-seconds", help="Maximum total seconds to retry"
    ),
) -> None:
    """Download a model from HuggingFace.

    Either REPO_ID or --profile must be provided.

    Exit codes:
        0: Success
        2: Configuration error (invalid profile, etc.)
        3: Non-retriable remote error
        4: Aborted by user
    """
    # Load settings
    settings = load_settings(
        endpoint=endpoint,
        force_endpoint=force_endpoint,
        cache_dir=output,
        max_workers=max_workers,
    )

    # Handle profile
    if profile:
        try:
            model_config = _get_model_by_profile(settings, profile)
        except ProfileNotFoundError as e:
            console.print(f"[red]✗ Configuration error:[/] {e}")
            raise typer.Exit(code=EXIT_CONFIG_ERROR)

        # Use profile settings (CLI args override profile)
        if repo_id is None:
            repo_id = model_config.repo_id
        if revision == "main" and model_config.revision != "main":
            revision = model_config.revision
        if repo_type == "model" and model_config.repo_type != "model":
            repo_type = model_config.repo_type
        if not allow_pattern and model_config.allow_patterns:
            allow_pattern = model_config.allow_patterns
        if not ignore_pattern and model_config.ignore_patterns:
            ignore_pattern = model_config.ignore_patterns
        if output is None and model_config.output_dir:
            output = model_config.output_dir

    # Validate repo_id
    if not repo_id:
        console.print("[red]✗ Error:[/] Either REPO_ID or --profile must be provided")
        raise typer.Exit(code=EXIT_CONFIG_ERROR)

    # Handle retry flags
    effective_retry_forever = None
    if retry_forever is True:
        effective_retry_forever = True
    elif no_retry_forever:
        effective_retry_forever = False

    # Print effective settings
    _print_effective_settings(
        settings=settings,
        repo_id=repo_id,
        revision=revision,
        repo_type=repo_type,
        force_endpoint=force_endpoint,
        endpoint=endpoint,
        output_dir=output,
        allow_patterns=allow_pattern if allow_pattern else None,
        ignore_patterns=ignore_pattern if ignore_pattern else None,
        max_workers=max_workers,
        token=token,
        retry_forever=effective_retry_forever,
        max_attempts=max_attempts,
        max_total_seconds=max_total_seconds,
    )

    # Override retry settings if specified
    if effective_retry_forever is not None:
        new_retry = settings.retry.model_copy(
            update={"forever": effective_retry_forever}
        )
        settings = settings.model_copy(update={"retry": new_retry})
    if max_attempts is not None:
        new_retry = settings.retry.model_copy(update={"max_attempts": max_attempts})
        settings = settings.model_copy(update={"retry": new_retry})
    if max_total_seconds is not None:
        new_retry = settings.retry.model_copy(
            update={"max_total_seconds": max_total_seconds}
        )
        settings = settings.model_copy(update={"retry": new_retry})

    try:
        path = download_snapshot(
            repo_id=repo_id,
            revision=revision,
            repo_type=repo_type,
            cache_dir=output,
            force_download=force_download,
            force_endpoint=force_endpoint,
            token=token,
            settings=settings,
            allow_patterns=allow_pattern if allow_pattern else None,
            ignore_patterns=ignore_pattern if ignore_pattern else None,
        )
        console.print(f"[green]✓ Successfully downloaded to:[/] {path}")
        raise typer.Exit(code=EXIT_SUCCESS)
    except ProfileNotFoundError as e:
        console.print(f"[red]✗ Configuration error:[/] {e}")
        raise typer.Exit(code=EXIT_CONFIG_ERROR)
    except ConfigurationError as e:
        console.print(f"[red]✗ Configuration error:[/] {e}")
        raise typer.Exit(code=EXIT_CONFIG_ERROR)
    except ValidationError as e:
        console.print(f"[red]✗ Validation error:[/] {e}")
        raise typer.Exit(code=EXIT_CONFIG_ERROR)
    except DownloadError as e:
        console.print(f"[red]✗ Download failed:[/] {e}")
        raise typer.Exit(code=EXIT_NON_RETRIABLE_ERROR)
    except KeyboardInterrupt:
        console.print("\n[yellow]✗ Download aborted by user[/]")
        raise typer.Exit(code=EXIT_ABORTED)


@app.command("list-profiles")
def list_profiles() -> None:
    """List all available configuration profiles.

    Profiles are model configurations defined in the config file.
    """
    try:
        settings = Settings.load()
    except ConfigurationError as e:
        console.print(f"[red]✗ Error loading config:[/] {e}")
        raise typer.Exit(code=EXIT_CONFIG_ERROR)

    if not settings.models:
        console.print("[dim]No profiles configured[/]")
        console.print()
        console.print("[dim]Add profiles to your config file:[/]")
        config_path = Settings.get_config_path()
        console.print(f"[dim]  {config_path}[/]")
        console.print()
        console.print("[dim]Example profile configuration:[/]")
        console.print("""[dim]  [[models]]
  name = "bert-tiny"
  repo_id = "prajjwal1/bert-tiny"
  revision = "main"
  repo_type = "model"[/]""")
        raise typer.Exit(code=EXIT_SUCCESS)

    # Create profiles table
    table = Table(
        title=f"Configured Profiles ({len(settings.models)})",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Name", style="green", no_wrap=True)
    table.add_column("Repository ID", style="blue")
    table.add_column("Revision", style="yellow")
    table.add_column("Type", style="magenta")
    table.add_column("Output Dir", style="dim")

    for model in settings.models:
        table.add_row(
            model.name,
            model.repo_id,
            model.revision,
            model.repo_type,
            model.output_dir or "(default)",
        )

    console.print(table)
    console.print()
    console.print("[dim]Use with: hfmdl download --profile <name>[/")
    raise typer.Exit(code=EXIT_SUCCESS)


@app.command("show-config")
def show_config(
    endpoint: str | None = typer.Option(
        None, "--endpoint", "-e", help="Override endpoint URL"
    ),
    force_endpoint: bool = typer.Option(
        False, "--force-endpoint", "-f", help="Force config endpoint, ignore HF_ENDPOINT env var"
    ),
) -> None:
    """Show current configuration."""
    try:
        settings = Settings.load()
    except ConfigurationError as e:
        console.print(f"[red]✗ Error loading config:[/] {e}")
        raise typer.Exit(code=EXIT_CONFIG_ERROR)

    if endpoint:
        settings = settings.merge_cli_overrides(endpoint=endpoint)

    config_path = Settings.get_config_path()

    # Apply environment variables before any HF operations
    effective_env = apply_hf_env(settings, force_endpoint=force_endpoint)

    console.print(f"[blue]Config file:[/] {config_path}")
    console.print()

    # Main settings table
    table = Table(title="Effective Configuration", show_header=True, header_style="bold cyan")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    # Show effective endpoint from environment (respects HF_ENDPOINT)
    table.add_row("Endpoint", effective_env.get("HF_ENDPOINT", settings.endpoint))
    table.add_row("Download Timeout", str(settings.hf_hub_download_timeout))
    table.add_row("ETag Timeout", str(settings.hf_hub_etag_timeout))
    table.add_row("High Performance Mode", str(settings.hf_xet_high_performance))
    table.add_row(
        "Concurrent Range Gets", str(settings.hf_xet_num_concurrent_range_gets)
    )
    table.add_row("Max Workers", str(settings.max_workers))
    table.add_row("Cache Dir", settings.cache_dir or "(platform default)")

    console.print(table)
    console.print()

    # Retry settings
    retry_table = Table(title="Retry Configuration", show_header=True, header_style="bold cyan")
    retry_table.add_column("Setting", style="cyan")
    retry_table.add_column("Value", style="green")

    retry_table.add_row("Forever", str(settings.retry.forever))
    retry_table.add_row(
        "Max Attempts", str(settings.retry.max_attempts or "(unlimited)")
    )
    retry_table.add_row(
        "Max Total Seconds", str(settings.retry.max_total_seconds or "(unlimited)")
    )
    retry_table.add_row("Base Wait", str(settings.retry.base_wait))
    retry_table.add_row("Max Wait", str(settings.retry.max_wait))
    retry_table.add_row("Jitter", str(settings.retry.jitter))
    retry_table.add_row("Log Every Attempt", str(settings.retry.log_every_attempt))

    console.print(retry_table)
    console.print()

    # Models
    if settings.models:
        models_table = Table(
            title=f"Configured Models ({len(settings.models)})",
            show_header=True,
            header_style="bold cyan",
        )
        models_table.add_column("Name", style="cyan")
        models_table.add_column("Repo ID", style="green")
        models_table.add_column("Revision", style="yellow")
        models_table.add_column("Type", style="blue")

        for model in settings.models:
            models_table.add_row(model.name, model.repo_id, model.revision, model.repo_type)

        console.print(models_table)
    else:
        console.print("[dim]No models configured[/]")

    raise typer.Exit(code=EXIT_SUCCESS)


@app.command()
def validate(
    repo_id: str = typer.Argument(..., help="HuggingFace repository ID to validate"),
    revision: str = typer.Option(
        "main", "--revision", help="Model revision/branch/tag"
    ),
    repo_type: str = typer.Option(
        "model", "--repo-type", help="Repository type: model, dataset, or space"
    ),
    endpoint: str | None = typer.Option(
        None, "--endpoint", help="Override endpoint URL"
    ),
    force_endpoint: bool = typer.Option(
        False,
        "--force-endpoint",
        "-f",
        help="Force config endpoint, ignore HF_ENDPOINT env var",
    ),
    token: str | None = typer.Option(
        None, "--token", help="HuggingFace API token (default: HF_TOKEN env var)"
    ),
) -> None:
    """Validate a repository without downloading.

    Checks if the repository exists and is accessible.

    Exit codes:
        0: Repository is valid and accessible
        2: Configuration error
        3: Repository not found or not accessible
    """
    try:
        settings = load_settings(
            endpoint=endpoint,
            force_endpoint=force_endpoint,
        )
    except ConfigurationError as e:
        console.print(f"[red]✗ Configuration error:[/] {e}")
        raise typer.Exit(code=EXIT_CONFIG_ERROR)

    # Apply environment
    apply_hf_env(settings, force_endpoint=force_endpoint)

    console.print(f"[blue]Validating repository:[/] {repo_id}")
    console.print(f"[dim]Revision: {revision}[/]")
    console.print(f"[dim]Type: {repo_type}[/]")
    console.print()

    try:
        # LAZY IMPORT after apply_hf_env
        from huggingface_hub import HfApi
        from huggingface_hub.utils import (
            GatedRepoError,
            RepositoryNotFoundError,
            RevisionNotFoundError,
        )

        api = HfApi(token=token)

        # Try to get repo info
        console.print("[dim]Checking repository info...[/]")
        repo_info = api.repo_info(
            repo_id=repo_id,
            revision=revision,
            repo_type=repo_type,
        )

        # Success!
        console.print("[green]✓ Repository is valid and accessible[/]")
        console.print()
        console.print(f"[cyan]Repository ID:[/] {repo_info.id}")
        console.print(f"[cyan]Author:[/] {repo_info.author}")
        console.print(f"[cyan]SHA:[/] {repo_info.sha[:12]}...")
        if hasattr(repo_info, "tags") and repo_info.tags:
            tags_str = ', '.join(repo_info.tags[:5])
            tags_str += "..." if len(repo_info.tags) > 5 else ""
            console.print(f"[cyan]Tags:[/] {tags_str}")
        console.print(f"[cyan]Last Modified:[/] {repo_info.lastModified}")
        console.print(f"[cyan]Private:[/] {repo_info.private}")

        raise typer.Exit(code=EXIT_SUCCESS)

    except RepositoryNotFoundError as e:
        console.print(f"[red]✗ Repository not found:[/] {e}")
        console.print("[dim]Check that the repository ID is correct and you have access.[/]")
        raise typer.Exit(code=EXIT_NON_RETRIABLE_ERROR)
    except RevisionNotFoundError as e:
        console.print(f"[red]✗ Revision not found:[/] {e}")
        console.print(f"[dim]Revision '{revision}' does not exist in this repository.[/]")
        raise typer.Exit(code=EXIT_NON_RETRIABLE_ERROR)
    except GatedRepoError as e:
        console.print(f"[red]✗ Gated repository:[/] {e}")
        console.print(
            "[dim]This repository requires access approval. "
            "Visit the repository page to request access.[/]"
        )
        raise typer.Exit(code=EXIT_NON_RETRIABLE_ERROR)
    except Exception as e:
        console.print(f"[red]✗ Validation failed:[/] {e}")
        raise typer.Exit(code=EXIT_NON_RETRIABLE_ERROR)


if __name__ == "__main__":
    app()
