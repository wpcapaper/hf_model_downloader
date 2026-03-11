"""CLI interface for hf_model_downloader using Typer."""

import typer
from rich.console import Console
from rich.table import Table

from .config import Settings

app = typer.Typer(
    name="hfmdl",
    help="HuggingFace Model Downloader - Download and manage HuggingFace models",
)
console = Console()


@app.command()
def download(
    model_id: str = typer.Argument(
        ..., help="HuggingFace model ID (e.g., 'bert-base-uncased')"
    ),
    profile: str | None = typer.Option(
        None, "--profile", "-p", help="Configuration profile to use"
    ),
    output_dir: str | None = typer.Option(None, "--output", "-o", help="Output directory"),
) -> None:
    """Download a model from HuggingFace."""
    console.print(f"[blue]Downloading model:[/] {model_id}")
    if profile:
        console.print(f"[dim]Using profile: {profile}[/]")
    if output_dir:
        console.print(f"[dim]Output directory: {output_dir}[/]")
    # TODO: Implement actual download logic
    console.print("[yellow]Download functionality not yet implemented[/]")


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
) -> None:
    """Show current configuration."""
    settings = Settings.load()
    if endpoint:
        settings = settings.merge_cli_overrides(endpoint=endpoint)
    config_path = Settings.get_config_path()

    console.print(f"[blue]Config file:[/] {config_path}")
    console.print()

    # Main settings table
    table = Table(title="Effective Configuration", show_header=True)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Endpoint", settings.endpoint)
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
