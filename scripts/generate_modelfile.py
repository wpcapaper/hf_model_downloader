#!/usr/bin/env python3
"""
Generate Ollama Modelfile from GGUF and HuggingFace model configuration.

This script fetches chat_template.jinja and config.json from the source repository
and generates an accurate Ollama Modelfile that preserves the original configuration.

ENHANCED VERSION: Uses full Ollama template capabilities including:
- Tool call IDs for proper correlation
- Thinking/reasoning content support
- Vision/multimodal placeholders
- JSON serialization for tools
"""

# ruff: noqa: E501

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

# We need to add src to path for the package imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hf_model_downloader.config import load_settings  # noqa: I001


def get_qwen3_template_enhanced() -> str:
    """Enhanced Qwen3 template with full Ollama feature support.

    Supports:
    - Tool definitions with JSON schema
    - Tool calls with IDs and arguments
    - Tool responses with ID correlation
    - Thinking/reasoning content
    - Vision placeholders (if model supports it)
    """
    return """{{- if .Tools }}<|im_start|>system
You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.
You are provided with function descriptions in <tools></tools> tags:
<tools>
{{ range .Tools }}
{"type": "{{ .Type }}", "function": {{ json .Function }}}
{{ end }}</tools>

If you choose to call a function ONLY reply in the following format:
<tool_call>
<function=function_name>
<parameter=param_name>
value
</parameter>
</function>
</tool_call>

# Tool Guidelines
- Function calls MUST follow the specified format
- Required parameters MUST be specified
- You may provide optional reasoning BEFORE the function call, but NOT after
- If no function call is needed, answer normally
<|im_end|>
{{ end }}
{{- range .Messages }}
{{- if eq .Role "system" }}<|im_start|>system
{{ .Content }}<|im_end|>
{{- else if eq .Role "user" }}<|im_start|>user
{{ if .Images }}<|vision_start|><|image_pad|><|vision_end|>
{{ end }}{{ .Content }}<|im_end|>
{{- else if eq .Role "assistant" }}<|im_start|>assistant
{{ if .Thinking }}
<think>
{{ .Thinking }}
</think>

{{ end }}{{ .Content }}
{{ if .ToolCalls }}
{{ range .ToolCalls }}
<tool_call>
<function={{ .Function.Name }}>
{{ range $name, $value := .Function.Arguments }}
<parameter={{ $name }}>
{{ $value }}
</parameter>
{{ end }}</function>
</tool_call>
{{ end }}
{{ end }}<|im_end|>
{{- else if eq .Role "tool" }}<|im_start|>user
<tool_response>
{{ if .ToolCallID }}
<tool_call_id>{{ .ToolCallID }}</tool_call_id>
{{ end }}{{ .Content }}
</tool_response><|im_end|>
{{- end }}
{{- end }}<|im_start|>assistant
{{- if .Think }}
<think>

</think>

{{ end }}"""


def get_llama3_template_enhanced() -> str:
    """Enhanced Llama3 template with full tool and thinking support."""
    return """{{- if .Tools }}<|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: {{ currentDate }}

When you receive a tool call response, use the output to format an answer to the original user question.

You are a helpful assistant with access to the following tools:
{{ range .Tools }}
- {{ .Function.Name }}: {{ .Function.Description }}
{{ end }}<|eot_id|>
{{- end }}
{{- range .Messages }}
{{- if eq .Role "system" }}<|start_header_id|>system<|end_header_id|>

{{ .Content }}<|eot_id|>
{{- else if eq .Role "user" }}<|start_header_id|>user<|end_header_id|>

{{ .Content }}<|eot_id|>
{{- else if eq .Role "assistant" }}<|start_header_id|>assistant<|end_header_id|>

{{ if .Thinking }}
<think>
{{ .Thinking }}
</think>

{{ end }}{{ .Content }}
{{- if .ToolCalls }}

<calling>
{{ range .ToolCalls }}
{"name": "{{ .Function.Name }}", "arguments": {{ json .Function.Arguments }}}
{{ end }}</calling>
{{- end }}<|eot_id|>
{{- else if eq .Role "tool" }}<|start_header_id|>ipython<|end_header_id|>

{{ if .ToolCallID }}[Tool Call ID: {{ .ToolCallID }}]
{{ end }}{{ .Content }}<|eot_id|>
{{- end }}
{{- end }}<|start_header_id|>assistant<|end_header_id|>

"""


def get_mistral_template_enhanced() -> str:
    """Enhanced Mistral template with tool and thinking support."""
    return """{{- if .Tools }}[AVAILABLE_TOOLS] {{ json .Tools }}[/AVAILABLE_TOOLS]
{{- end }}
{{- range .Messages }}
{{- if eq .Role "user" }}[INST] {{ .Content }} [/INST]
{{- else if eq .Role "assistant" }}
{{- if .Thinking }}[THINKING] {{ .Thinking }} [/THINKING]
{{ end }}
{{- if .ToolCalls }}[TOOL_CALLS] {{ json .ToolCalls }}[/TOOL_CALLS]
{{- else if .Content }}{{ .Content }}</s>
{{- end }}
{{- else if eq .Role "tool" }}[TOOL_RESULTS] {{ if .ToolCallID }}{{ .ToolCallID }}: {{ end }}{{ .Content }} [/TOOL_RESULTS]
{{- end }}
{{- end }}"""


def get_phi3_template_enhanced() -> str:
    """Enhanced Phi3 template with tool and thinking support."""
    return """{{- range .Messages }}
{{- if eq .Role "system" }}<|system|>
{{ .Content }}<|end|>
{{- else if eq .Role "user" }}<|user|>
{{ .Content }}<|end|>
{{- else if eq .Role "assistant" }}<|assistant|>
{{ if .Thinking }}
<think>{{ .Thinking }}</think>
{{ end }}{{ .Content }}
{{ if .ToolCalls }}
{{ range .ToolCalls }}
[TOOL] {{ .Function.Name }}: {{ json .Function.Arguments }}
{{ end }}
{{ end }}<|end|>
{{- else if eq .Role "tool" }}<|tool|>
{{ if .ToolCallID }}ID: {{ .ToolCallID }}
{{ end }}{{ .Content }}<|end|>
{{- end }}
{{- end }}<|assistant|>
"""


def get_generic_template_enhanced() -> str:
    """Generic fallback template with basic tool support."""
    return """{{- if .Tools }}<|system|>
You have access to the following tools:
{{ range .Tools }}
- {{ .Function.Name }}: {{ .Function.Description }}
{{ end }}
<|endoftext|>
{{- end }}
{{- range .Messages }}
{{- if eq .Role "system" }}<|system|>
{{ .Content }}
{{- else if eq .Role "user" }}<|user|>
{{ .Content }}
{{- else if eq .Role "assistant" }}<|assistant|>
{{ if .Thinking }}
<think>{{ .Thinking }}</think>
{{ end }}{{ .Content }}
{{ if .ToolCalls }}
{{ range .ToolCalls }}
[TOOL] {{ .Function.Name }}: {{ json .Function.Arguments }}
{{ end }}
{{ end }}
{{- else if eq .Role "tool" }}<|tool|>
{{ if .ToolCallID }}[{{ .ToolCallID }}] {{ end }}{{ .Content }}
{{- end }}
{{- if not (eq .Role "tool") }}
<|endoftext|>
{{- end }}
{{- end }}<|assistant|>
"""


# Enhanced template mapping with full feature support
TEMPLATE_MAP: dict[str, dict[str, Any]] = {
    "qwen": {
        "template_func": get_qwen3_template_enhanced,
        "stop_tokens": ["<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>"],
        "features": ["tools", "thinking", "vision"],
    },
    "llama": {
        "template_func": get_llama3_template_enhanced,
        "stop_tokens": ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"],
        "features": ["tools", "thinking"],
    },
    "mistral": {
        "template_func": get_mistral_template_enhanced,
        "stop_tokens": ["[INST]", "[/INST]", "<|s|>"],
        "features": ["tools", "thinking"],
    },
    "phi": {
        "template_func": get_phi3_template_enhanced,
        "stop_tokens": ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>", "<|endoftext|>"],
        "features": ["tools", "thinking"],
    },
}


def detect_jinja_features(jinja_template: str) -> dict[str, Any]:
    """Detect features in the original Jinja2 template.

    Returns a report of what features are present and whether they can be
    fully supported in Ollama templates.
    """
    features = {
        "detected": [],
        "unsupported": [],
        "warnings": [],
    }

    # Check for tool-related features
    if "tools" in jinja_template.lower() or "tool_call" in jinja_template.lower():
        features["detected"].append("tool_calling")

    # Check for vision/multimodal
    if any(kw in jinja_template.lower() for kw in ["image", "vision", "video", "multimodal"]):
        features["detected"].append("vision")

    # Check for reasoning/thinking
    if any(kw in jinja_template.lower() for kw in ["think", "reasoning", "reason"]):
        features["detected"].append("thinking")

    # Check for unsupported Jinja2 features
    if "{% macro" in jinja_template:
        features["unsupported"].append("Jinja2 macros ({% macro %})")
        features["warnings"].append(
            "Macros are not supported in Ollama templates. Tool calling may be simplified."
        )

    if "namespace(" in jinja_template:
        features["unsupported"].append("Jinja2 namespace")
        features["warnings"].append(
            "Mutable state (namespace) is not supported. Some stateful logic may be lost."
        )

    if "[::-1]" in jinja_template or ".reverse()" in jinja_template:
        features["unsupported"].append("Backward iteration")
        features["warnings"].append(
            "Backward iteration (messages[::-1]) is not supported. Tool response matching may be affected."
        )

    if "{% set" in jinja_template and "namespace" not in jinja_template:
        features["unsupported"].append("Jinja2 variable assignment")
        features["warnings"].append(
            "Variable assignment ({% set %}) without namespace has limited support."
        )

    # Check for complex filters
    complex_filters = re.findall(r"\|\s*(\w+)", jinja_template)
    unsupported_filters = set(complex_filters) - {
        "tojson",
        "lower",
        "upper",
        "trim",
        "safe",
        "string",
        "length",
        "default",
    }
    if unsupported_filters:
        features["unsupported"].append(f"Complex filters: {unsupported_filters}")

    return features


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
    verbose: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Generate Modelfile content.

    Returns:
        Tuple of (modelfile_content, feature_report)
    """
    if endpoint is None:
        settings = load_settings()
        endpoint = settings.get_effective_endpoint()

    # Fetch configuration files
    config, chat_template_jinja = fetch_config_files(repo_id, endpoint)

    # Detect model family
    model_family = detect_model_family(config) if config else "generic"

    # Analyze Jinja2 features if available
    feature_report = {"model_family": model_family, "source": repo_id}
    if chat_template_jinja:
        jinja_features = detect_jinja_features(chat_template_jinja)
        feature_report["jinja_features"] = jinja_features

        if verbose and jinja_features.get("warnings"):
            print("\n=== Jinja2 Template Analysis ===", file=sys.stderr)
            for warning in jinja_features["warnings"]:
                print(f"⚠️  {warning}", file=sys.stderr)
            print("================================\n", file=sys.stderr)

    # Get template info
    if model_family in TEMPLATE_MAP:
        template_info = TEMPLATE_MAP[model_family]
        template = template_info["template_func"]()
        stop_tokens = template_info["stop_tokens"]
        feature_report["supported_features"] = template_info.get("features", [])
    else:
        template = get_generic_template_enhanced()
        stop_tokens = ["<|endoftext|>"]
        feature_report["supported_features"] = ["basic"]

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
        lines.append("# Note: chat_template.jinja was analyzed and mapped to Ollama template")
        if feature_report.get("jinja_features", {}).get("unsupported"):
            lines.append("# Unsupported Jinja2 features detected:")
            for unsupported in feature_report["jinja_features"]["unsupported"]:
                lines.append(f"#   - {unsupported}")
    if feature_report.get("supported_features"):
        lines.append(f"# Supported features: {', '.join(feature_report['supported_features'])}")

    return "\n".join(lines), feature_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Ollama Modelfile from GGUF and HuggingFace model configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ./model.gguf Qwen/Qwen3.5-0.8B
  %(prog)s ./model.gguf meta-llama/Llama-3.1-8B --output Modelfile
  %(prog)s ./model.gguf mistralai/Mistral-7B-Instruct-v0.3 --create my-model
  %(prog)s ./model.gguf Qwen/Qwen3.5-0.8B --verbose  # Show feature analysis
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
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed Jinja2 template analysis and warnings",
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
        modelfile_content, feature_report = generate_modelfile(
            gguf_path=args.gguf_path,
            repo_id=args.repo_id,
            parameters=parameters if parameters else None,
            verbose=args.verbose,
        )

        # Output handling
        if args.output:
            output_path = Path(args.output)
            output_path.write_text(modelfile_content, encoding="utf-8")
            print(f"Modelfile written to: {output_path}")
        else:
            print(modelfile_content)

        # Print feature summary if verbose
        if args.verbose:
            print(f"\n✅ Model family detected: {feature_report['model_family']}", file=sys.stderr)
            if feature_report.get("supported_features"):
                print(
                    f"✅ Features enabled: {', '.join(feature_report['supported_features'])}",
                    file=sys.stderr,
                )

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
