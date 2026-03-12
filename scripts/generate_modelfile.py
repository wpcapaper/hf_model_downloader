#!/usr/bin/env python3
"""Generate Ollama Modelfile from GGUF and HuggingFace repository metadata."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import requests

# Reuse hfmdl configuration
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from hf_model_downloader.config import load_settings


def fetch_repo_file(
    repo_id: str, filename: str, endpoint: str = "https://huggingface.co"
) -> str | None:
    """Fetch a file from HuggingFace repository."""
    url = f"{endpoint}/{repo_id}/raw/main/{filename}"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Warning: Could not fetch {filename}: {e}", file=sys.stderr)
        return None


def parse_chat_template(template_content: str) -> dict[str, Any]:
    """Parse Jinja chat template to extract key information."""
    result = {
        "has_system": "system" in template_content.lower(),
        "message_start": "<|im_start|>",
        "message_end": "<|im_end|>",
        "assistant_start": "<|im_start|>assistant",
        "stop_tokens": [],
    }

    # Extract stop tokens from template
    stop_tokens = set()
    for match in re.finditer(r'\{\{-?\s*\(?["\'](<\|[^"\']+\|?\u003e)["\']\)?', template_content):
        token = match.group(1)
        if token not in ["{{", "}}", "{%", "%}"]:
            stop_tokens.add(token)

    # Common Qwen tokens
    if "qwen" in template_content.lower() or "im_start" in template_content:
        stop_tokens.update(["<|im_end|>", "<|endoftext|>"])

    # Common Llama tokens
    if "llama" in template_content.lower() or "bos_token" in template_content:
        stop_tokens.update(["<|eot_id|>", "<|end_of_text|>", "[/INST]"])

    result["stop_tokens"] = list(stop_tokens)
    return result


def parse_config(config_content: str) -> dict[str, Any]:
    """Parse config.json to extract model parameters."""
    try:
        config = json.loads(config_content)
    except json.JSONDecodeError:
        return {}

    result = {}

    # Extract common parameters
    text_config = config.get("text_config", config)

    # Context length
    if "max_position_embeddings" in text_config:
        result["num_ctx"] = text_config["max_position_embeddings"]

    # EOS token ID
    if "eos_token_id" in text_config:
        result["eos_token_id"] = text_config["eos_token_id"]

    # Model type for template selection
    result["model_type"] = config.get("model_type", text_config.get("model_type", "unknown"))

    # Architectures
    if "architectures" in config:
        result["architectures"] = config["architectures"]

    return result


def generate_template(chat_info: dict[str, Any], model_type: str) -> str:
    """Generate Ollama TEMPLATE string."""

    # Qwen style (IM format)
    if model_type in ["qwen3", "qwen3_5", "qwen2", "qwen"] or "im_start" in str(
        chat_info.get("message_start", "")
    ):
        return """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ range .Messages }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end }}<|im_start|>assistant
"""

    # Llama 3 style
    if model_type in ["llama3", "llama"] or "eot_id" in str(chat_info.get("stop_tokens", [])):
        return """{{- if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ range .Messages }}<|start_header_id|>{{ .Role }}<|end_header_id|>

{{ .Content }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

"""

    # Generic fallback
    return """{{- if .System }}System: {{ .System }}

{{ end }}{{ range .Messages }}{{ .Role }}: {{ .Content }}
{{ end }}Assistant:"""


def generate_modelfile(
    gguf_path: Path,
    chat_template: str | None,
    config: dict[str, Any],
    system_prompt: str | None = None,
    temperature: float | None = None,
) -> str:
    """Generate complete Modelfile content."""

    lines = [f"FROM {gguf_path.name}", ""]

    # Parse template and config
    chat_info = parse_chat_template(chat_template or "")

    # Generate TEMPLATE
    template = generate_template(chat_info, config.get("model_type", "unknown"))
    lines.append(f'TEMPLATE """{template}"""')
    lines.append("")

    # Add parameters
    if temperature is not None:
        lines.append(f"PARAMETER temperature {temperature}")
    else:
        lines.append("PARAMETER temperature 0.7")

    lines.append("PARAMETER top_p 0.9")
    lines.append("PARAMETER top_k 40")

    # Context length from config
    if "num_ctx" in config:
        lines.append(f"PARAMETER num_ctx {config['num_ctx']}")

    lines.append("")

    # Stop tokens
    stop_tokens = chat_info.get("stop_tokens", [])

    # Default stop tokens based on model type
    if not stop_tokens:
        model_type = config.get("model_type", "")
        if "qwen" in model_type.lower():
            stop_tokens = ["<|im_end|>", "<|endoftext|>"]
        elif "llama" in model_type.lower():
            stop_tokens = ["<|eot_id|>", "<|end_of_text|>"]

    for token in stop_tokens:
        lines.append(f'PARAMETER stop "{token}"')

    lines.append("")

    # System prompt
    if system_prompt:
        lines.append(f'SYSTEM """{system_prompt}"""')
    else:
        lines.append('SYSTEM """You are a helpful AI assistant."""')

    return "\n".join(lines)


def get_endpoint_from_config() -> str:
    """Get endpoint from hfmdl configuration."""
    try:
        settings = load_settings()
        # Use effective endpoint (respects HF_ENDPOINT env var)
        return settings.get_effective_endpoint()
    except Exception:
        # Fallback to default if config not found or error
        return "https://huggingface.co"


def main():
    # Get default endpoint from config
    default_endpoint = get_endpoint_from_config()

    parser = argparse.ArgumentParser(
        description="Generate Ollama Modelfile from GGUF and HuggingFace repository",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  %(prog)s ./model.gguf Qwen/Qwen3.5-0.8B
  %(prog)s ./model.gguf Qwen/Qwen3.5-0.8B --system "You are a coding expert"
  %(prog)s ./model.gguf unsloth/Qwen3.5-35B-A3B-GGUF --temperature 0.6

Configuration:
  Uses hfmdl config from ~/.config/hfmdl/config.toml
  Current default endpoint: {default_endpoint}
        """,
    )

    parser.add_argument("gguf_path", type=Path, help="Path to GGUF file")
    parser.add_argument("repo_id", help="HuggingFace repository ID (e.g., 'Qwen/Qwen3.5-0.8B')")
    parser.add_argument(
        "--endpoint",
        default=default_endpoint,
        help=f"HuggingFace endpoint (default: {default_endpoint})",
    )
    parser.add_argument("--system", help="System prompt")
    parser.add_argument("--temperature", type=float, help="Temperature parameter")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output Modelfile path (default: Modelfile in GGUF directory)",
    )
    parser.add_argument(
        "--create",
        "-c",
        action="store_true",
        help="Automatically run 'ollama create' after generation",
    )
    parser.add_argument(
        "--name", "-n", help="Ollama model name (default: auto-generated from GGUF filename)"
    )

    args = parser.parse_args()

    # Validate GGUF path
    if not args.gguf_path.exists():
        print(f"Error: GGUF file not found: {args.gguf_path}", file=sys.stderr)
        sys.exit(1)

    # Fetch metadata from repository
    print(f"Fetching metadata from {args.endpoint}/{args.repo_id}...")

    chat_template = fetch_repo_file(args.repo_id, "chat_template.jinja", args.endpoint)
    config_content = fetch_repo_file(args.repo_id, "config.json", args.endpoint)

    if not chat_template and not config_content:
        print("Warning: Could not fetch any metadata from repository", file=sys.stderr)
        print("Will generate Modelfile with default settings", file=sys.stderr)

    # Parse config
    config = parse_config(config_content) if config_content else {}

    # Generate Modelfile
    modelfile_content = generate_modelfile(
        args.gguf_path,
        chat_template,
        config,
        args.system,
        args.temperature,
    )

    # Determine output path
    output_path = args.output or (args.gguf_path.parent / "Modelfile")

    # Write Modelfile
    output_path.write_text(modelfile_content)
    print(f"Generated Modelfile: {output_path}")

    # Optionally create Ollama model
    if args.create:
        import subprocess

        model_name = args.name or args.gguf_path.stem.replace(".", "-").lower()

        print(f"Creating Ollama model: {model_name}")
        try:
            subprocess.run(
                ["ollama", "create", model_name, "-f", str(output_path)],
                check=True,
                cwd=str(args.gguf_path.parent),
            )
            print(f"Successfully created model: {model_name}")
            print(f"Run with: ollama run {model_name}")
        except subprocess.CalledProcessError as e:
            print(f"Error creating Ollama model: {e}", file=sys.stderr)
            sys.exit(1)
        except FileNotFoundError:
            print("Error: 'ollama' command not found. Is Ollama installed?", file=sys.stderr)
            sys.exit(1)
if __name__ == "__main__":
    main()
