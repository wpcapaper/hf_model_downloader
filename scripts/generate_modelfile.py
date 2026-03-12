#!/usr/bin/env python3
"""
Generate Ollama Modelfile from GGUF and HuggingFace model configuration.

This script fetches chat_template.jinja and config.json from the source repository
and generates an accurate Ollama Modelfile that preserves the original configuration.
"""

# ruff: noqa: E501

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

# We need to add src to path for the package imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hf_model_downloader.config import load_settings  # noqa: I001


def get_qwen3_template() -> str:
    """Qwen3 template - ChatML format with tool support."""
    return """{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 }}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ if $last }}<|im_start|>assistant
{{ end }}
{{- else if eq .Role "assistant" }}<|im_start|>assistant
{{ .Content }}{{ if not $last }}<|im_end|>
{{ end }}
{{- else if eq .Role "system" }}<|im_start|>system
{{ .Content }}<|im_end|>
{{ if $last }}<|im_start|>assistant
{{ end }}
{{- else if eq .Role "tool" }}<|im_start|>user
<tool_response>
{{ .Content }}
</tool_response><|im_end|>
{{ if $last }}<|im_start|>assistant
{{ end }}
{{- end }}
{{- end }}"""


def get_llama3_template() -> str:
    """Llama3 template - Header format with tool support."""
    return """{{- if .Tools }}<|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: {{ now | date "02 Jan 2006" }}

When you receive a tool call response, use the output to format an answer to the original user question.

You are a helpful assistant.

{{- range .Tools }}
{{- .Function }}
{{- end }}<|eot_id|>
{{- end }}
{{- range .Messages }}
{{- if eq .Role "user" }}<|start_header_id|>user<|end_header_id|>

{{ .Content }}<|eot_id|>
{{- else if eq .Role "assistant" }}<|start_header_id|>assistant<|end_header_id|>

{{ .Content }}
{{- if .ToolCalls }}
{{- range .ToolCalls }}
<tool_call>
{{ .Function }}
</tool_call>
{{- end }}
{{- end }}<|eot_id|>
{{- else if eq .Role "tool" }}<|start_header_id|>ipython<|end_header_id|>

{{ .Content }}<|eot_id|>
{{- else if eq .Role "system" }}<|start_header_id|>system<|end_header_id|>

{{ .Content }}<|eot_id|>
{{- end }}
{{- end }}<|start_header_id|>assistant<|end_header_id|>

"""


def get_mistral_template() -> str:
    """Mistral template - INST format with tool support."""
    return """{{- if .Tools }}[AVAILABLE_TOOLS] {{ json .Tools }}[/AVAILABLE_TOOLS]
{{- end }}
{{- range .Messages }}
{{- if eq .Role "user" }}[INST] {{ .Content }} [/INST]
{{- else if eq .Role "assistant" }}
{{- if .ToolCalls }}[TOOL_CALLS] {{ json .ToolCalls }}[/TOOL_CALLS]
{{- else }}{{ .Content }}</s>
{{- end }}
{{- else if eq .Role "tool" }}[TOOL_RESULTS] {{ .Content }} [/TOOL_RESULTS]
{{- end }}
{{- end }}"""


def get_phi3_template() -> str:
    """Phi3 template - Simple token format."""
    return """{{- range .Messages }}
{{- if eq .Role "system" }}<|system|>
{{ .Content }}<|end|>
{{- else if eq .Role "user" }}<|user|>
{{ .Content }}<|end|>
{{- else if eq .Role "assistant" }}<|assistant|>
{{ .Content }}<|end|>
{{- else if eq .Role "tool" }}<|tool|>
{{ .Content }}<|end|>
{{- end }}
{{- end }}<|assistant|>
"""


def get_generic_template() -> str:
    """Generic fallback template - ChatML-like format."""
    return """{{- range .Messages }}
{{- if eq .Role "system" }}<|system|>
{{ .Content }}
{{- else if eq .Role "user" }}<|user|>
{{ .Content }}
{{- else if eq .Role "assistant" }}<|assistant|>
{{ .Content }}
{{- end }}
{{- end }}<|assistant|>
"""


# Model family detection and template mapping
TEMPLATE_MAP: dict[str, dict[str, Any]] = {
    "qwen": {
        "template_func": get_qwen3_template,
        "stop_tokens": ["<|im_start|>", "<|im_end|>"],
    },
    "llama": {
        "template_func": get_llama3_template,
        "stop_tokens": ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"],
    },
    "mistral": {
        "template_func": get_mistral_template,
        "stop_tokens": ["[INST]", "[/INST]", "<|s|>"],
    },
    "phi": {
        "template_func": get_phi3_template,
        "stop_tokens": ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"],
    },
}


def detect_model_family(config: dict) -> str:
    """Detect model family from config.json content."""
    model_type = config.get("model_type", "").lower()
    arch_list = config.get("architectures", [])
    arch = arch_list[0].lower() if arch_list else ""

    # Check model type first
    if "qwen" in model_type or "qwen" in arch:
        return "qwen"
    elif "llama" in model_type or "llama" in arch:
        return "llama"
    elif "mistral" in model_type or "mistral" in arch:
        return "mistral"
    elif "phi" in model_type or "phi" in arch:
        return "phi"

    # Check chat template if available
    chat_template = config.get("chat_template", "")
    if "<|im_start|>" in chat_template:
        return "qwen"
    elif "<|start_header_id|>" in chat_template:
        return "llama"
    elif "[INST]" in chat_template:
        return "mistral"
    elif "<|system|>" in chat_template and "<|user|>" in chat_template:
        return "phi"

    return "generic"


def fetch_config_files(repo_id: str, endpoint: str) -> tuple[Optional[dict], Optional[str]]:
    """Fetch config.json and chat_template.jinja from HF repo."""
    # Set environment for endpoint
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint

    config_content: Optional[dict] = None
    chat_template_content: Optional[str] = None

    # Try to fetch config.json
    try:
        config_path = hf_hub_download(
            repo_id=repo_id, filename="config.json", local_files_only=False
        )
        with open(config_path, encoding="utf-8") as f:
            config_content = json.load(f)
    except EntryNotFoundError:
        print(f"Warning: config.json not found in {repo_id}", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Failed to fetch config.json: {e}", file=sys.stderr)

    # Try to fetch chat_template.jinja
    try:
        template_path = hf_hub_download(
            repo_id=repo_id, filename="chat_template.jinja", local_files_only=False
        )
        with open(template_path, encoding="utf-8") as f:
            chat_template_content = f.read()
    except EntryNotFoundError:
        # chat_template.jinja might not exist, extract from config.json instead
        pass
    except Exception as e:
        print(f"Warning: Failed to fetch chat_template.jinja: {e}", file=sys.stderr)

    return config_content, chat_template_content


def generate_modelfile(
    gguf_path: str,
    repo_id: str,
    endpoint: Optional[str] = None,
    parameters: Optional[dict] = None,
) -> str:
    """Generate Modelfile content."""
    if endpoint is None:
        settings = load_settings()
        endpoint = settings.get_effective_endpoint()

    # Fetch configuration files
    config, chat_template_jinja = fetch_config_files(repo_id, endpoint)

    # Detect model family
    model_family = detect_model_family(config) if config else "generic"

    # Get template info
    if model_family in TEMPLATE_MAP:
        template_info = TEMPLATE_MAP[model_family]
        template = template_info["template_func"]()
        stop_tokens = template_info["stop_tokens"]
    else:
        template = get_generic_template()
        stop_tokens = ["<|endoftext|>"]

    # Build Modelfile content
    lines = [f"FROM {gguf_path}", ""]

    # Add TEMPLATE
    lines.append('TEMPLATE """')
    lines.append(template)
    lines.append('"""')
    lines.append("")

    # Add stop parameters
    for stop in stop_tokens:
        lines.append(f'PARAMETER stop "{stop}"')
    lines.append("")

    # Add optional parameters
    if parameters:
        for key, value in parameters.items():
            if isinstance(value, bool):
                lines.append(f"PARAMETER {key} {'true' if value else 'false'}")
            elif isinstance(value, (int, float)):
                lines.append(f"PARAMETER {key} {value}")
            else:
                lines.append(f'PARAMETER {key} "{value}"')

    # Add source information as comments
    lines.append("")
    lines.append(f"# Generated from: {repo_id}")
    lines.append(f"# Model family: {model_family}")
    if chat_template_jinja:
        lines.append("# Note: chat_template.jinja was fetched and mapped to Ollama template")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Ollama Modelfile from GGUF and HuggingFace model configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ./model.gguf Qwen/Qwen3.5-0.8B
  %(prog)s ./model.gguf meta-llama/Llama-3.1-8B --output Modelfile
  %(prog)s ./model.gguf mistralai/Mistral-7B-Instruct-v0.3 --create my-model
        """,
    )

    parser.add_argument("gguf_path", help="Path to the GGUF file")
    parser.add_argument("repo_id", help="HuggingFace model repository ID (e.g., Qwen/Qwen3.5-0.8B)")
    parser.add_argument("-o", "--output", help="Output file path (default: print to stdout)")
    parser.add_argument(
        "--create",
        metavar="MODEL_NAME",
        help="Automatically create the model in Ollama with the given name",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Set temperature parameter (default: from config or 0.7)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        help="Set top_p parameter",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="Set top_k parameter",
    )

    args = parser.parse_args()

    # Collect optional parameters
    parameters: dict[str, Any] = {}
    if args.temperature is not None:
        parameters["temperature"] = args.temperature
    if args.top_p is not None:
        parameters["top_p"] = args.top_p
    if args.top_k is not None:
        parameters["top_k"] = args.top_k

    try:
        # Generate Modelfile content
        modelfile_content = generate_modelfile(
            gguf_path=args.gguf_path,
            repo_id=args.repo_id,
            parameters=parameters if parameters else None,
        )

        # Output handling
        if args.output:
            output_path = Path(args.output)
            output_path.write_text(modelfile_content, encoding="utf-8")
            print(f"Modelfile written to: {output_path}")
        else:
            print(modelfile_content)

        # Auto-create if requested
        if args.create:
            import subprocess

            cmd = ["ollama", "create", args.create, "-f", args.output or "-"]
            if not args.output:
                # Pipe content directly
                result = subprocess.run(
                    cmd,
                    input=modelfile_content,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                )
            else:
                result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"Error creating model: {result.stderr}", file=sys.stderr)
                sys.exit(1)
            else:
                print(f"Successfully created Ollama model: {args.create}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
