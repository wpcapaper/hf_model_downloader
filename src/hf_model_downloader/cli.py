#VS|"""CLI interface for hf_model_downloader using Typer."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from .config import Settings, load_settings
from .downloader import download_snapshot
from .env import apply_hf_env
from .errors import DownloadError

app = typer.Typer(
    name="hfmdl",
    help="HuggingFace Model Downloader - Download and manage HuggingFace models",
)
console = Console()


# Version info
__version__ = "0.1.0"


def _version_callback(value: bool) -> None:
    """Callback to show version and exit."""
    if value:
        console.print(f"[blue]hfmdl[/] version [green]{__version__}[/]")
        raise typer.Exit()


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
    model_id: str = typer.Argument(
        ..., help="HuggingFace model ID (e.g., 'bert-base-uncased')"
    ),
    profile: str | None = typer.Option(
        None, "--profile", "-p", help="Configuration profile to use (future feature)"
    ),
    output_dir: str | None = typer.Option(
        None, "--output", "-o", help="Output/cache directory"
    ),
    revision: str = typer.Option(
        "main", "--revision", "-r", help="Model revision/branch/tag"
    ),
    repo_type: str = typer.Option(
        "model", "--repo-type", "-t", help="Repository type: model, dataset, or space"
    ),
    force_download: bool = typer.Option(
        False, "--force-download", "-f", help="Force re-download even if cached"
    ),
    force_endpoint: bool = typer.Option(
        False,
        "--force-endpoint",
        "-e",
        help="Force config endpoint, ignore HF_ENDPOINT env var",
    ),
    endpoint: str | None = typer.Option(
        None, "--endpoint", help="Override endpoint URL"
    ),
    token: str | None = typer.Option(
        None, "--token", help="HuggingFace API token (default: HF_TOKEN env var)"
    ),
) -> None:
    """Download a model from HuggingFace."""
    console.print(f"[blue]Downloading model:[/] {model_id}")
    console.print(f"[dim]Revision: {revision}[/]")
    console.print(f"[dim]Repo type: {repo_type}[/]")

    if profile:
        console.print(f"[dim]Profile: {profile} (not yet implemented)[/]")

    # Load settings with CLI overrides
    settings = load_settings(
        endpoint=endpoint,
        force_endpoint=force_endpoint,
        cache_dir=output_dir,
    )

    try:
        path = download_snapshot(
            repo_id=model_id,
            revision=revision,
            repo_type=repo_type,
            cache_dir=output_dir,
            force_download=force_download,
            force_endpoint=force_endpoint,
            token=token,
            settings=settings,
        )
        console.print(f"[green]✓ Successfully downloaded to:[/] {path}")
    except DownloadError as e:
        console.print(f"[red]✗ Download failed:[/] {e}")
        raise typer.Exit(code=1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Download cancelled by user[/]")
        raise typer.Exit(code=130)

@app.command()
def list_profiles() -> None:
    """List all available configuration profiles."""
    console.print("[blue]Available profiles:[/]")
    # TODO: Implement profile listing
    console.print("[dim]No profiles configured yet[/]")


@app.command()
def show_config(
    endpoint: str | None = typer.Option(
        None, "--endpoint", "-e", help="Override endpoint URL"
    ),
    force_endpoint: bool = typer.Option(
        False, "--force-endpoint", "-f", help="Force config endpoint, ignore HF_ENDPOINT env var"
    ),
) -> None:
    """Show current configuration."""
    settings = Settings.load()
    if endpoint:
        settings = settings.merge_cli_overrides(endpoint=endpoint)
    config_path = Settings.get_config_path()

    # Apply environment variables before any HF operations
    effective_env = apply_hf_env(settings, force_endpoint=force_endpoint)

    console.print(f"[blue]Config file:[/] {config_path}")
    console.print()

    # Main settings table
    table = Table(title="Effective Configuration", show_header=True)
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
    retry_table = Table(title="Retry Configuration", show_header=True)
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
            title=f"Configured Models ({len(settings.models)})", show_header=True
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


@app.command()
def validate(
    model_id: str = typer.Argument(..., help="HuggingFace model ID to validate"),
) -> None:
    """Validate a model configuration or download."""
    console.print(f"[blue]Validating model:[/] {model_id}")
    # TODO: Implement validation logic
    console.print("[yellow]Validation functionality not yet implemented[/]")


if __name__ == "__main__":
    app()
