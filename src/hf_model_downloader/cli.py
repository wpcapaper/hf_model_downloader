"""CLI interface for hf_model_downloader using Typer."""

import typer
from rich.console import Console

app = typer.Typer(
    name="hfmdl",
    help="HuggingFace Model Downloader - Download and manage HuggingFace models",
)
console = Console()


@app.command()
def download(
    model_id: str = typer.Argument(..., help="HuggingFace model ID (e.g., 'bert-base-uncased')"),
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
def show_config() -> None:
    """Show current configuration."""
    console.print("[blue]Current configuration:[/]")
    # TODO: Implement config display
    console.print("[dim]Configuration not yet implemented[/]")


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
