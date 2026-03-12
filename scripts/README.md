# Modelfile Generator Script

Generate Ollama Modelfile from GGUF files and HuggingFace model configuration.

This script fetches `chat_template.jinja` and `config.json` from the source repository and generates an accurate Ollama Modelfile that preserves the original configuration as much as possible.

## Features

- ✅ **Enhanced Templates** - Full Ollama feature support including tool IDs, thinking/reasoning, and vision
- ✅ **Jinja2 Analysis** - Detects and reports unsupported Jinja2 features with `--verbose`
- ✅ **Configuration Reuse** - Automatically reads mirror settings from `~/.config/hfmdl/config.toml`
- ✅ **Model Family Detection** - Automatically detects model type from config.json
- ✅ **One-click Model Creation** - Supports `--create` flag to directly create Ollama models
- ✅ **No External Dependencies** - Uses existing `huggingface_hub` package

## What's New in v2.0

### Enhanced Template Features

All templates now support:

1. **Tool Call IDs** - Proper correlation between tool calls and responses
2. **Thinking/Reasoning Content** - Support for models like DeepSeek-R1 and Qwen3 with thinking
3. **Vision/Multimodal** - Placeholders for image inputs (Qwen3 only)
4. **JSON Serialization** - Proper `{{ json .Tools }}` for tool definitions

### Jinja2 Feature Detection

Use `--verbose` to see what features are detected and what's unsupported:

```bash
python scripts/generate_modelfile.py ./model.gguf Qwen/Qwen3.5-0.8B --verbose
```

Output example:
```
=== Jinja2 Template Analysis ===
⚠️  Mutable state (namespace) is not supported. Some stateful logic may be lost.
⚠️  Backward iteration (messages[::-1]) is not supported. Tool response matching may be affected.
================================

✅ Model family detected: qwen
✅ Features enabled: tools, thinking, vision
```

## Prerequisites

The script uses packages already installed with `hfmdl`:
- `huggingface_hub` - For fetching files from HF Hub
- `hf_model_downloader` - For configuration reuse

## Usage

### Basic Usage

```bash
# Generate Modelfile for Qwen3.5 model
python scripts/generate_modelfile.py ./Qwen3.5-35B-A3B-UD-Q6_K_XL.gguf Qwen/Qwen3.5-0.8B

# Save to file instead of stdout
python scripts/generate_modelfile.py ./model.gguf meta-llama/Llama-3.1-8B --output Modelfile

# Generate and automatically create Ollama model
python scripts/generate_modelfile.py ./model.gguf mistralai/Mistral-7B-Instruct-v0.3 --create my-mistral

# Show detailed Jinja2 analysis (recommended)
python scripts/generate_modelfile.py ./model.gguf Qwen/Qwen3.5-0.8B --verbose
```

### With Parameters

```bash
# Set temperature and other parameters
python scripts/generate_modelfile.py ./model.gguf Qwen/Qwen3.5-0.8B \
  --temperature 0.6 \
  --top-p 0.9 \
  --top-k 40
```

### Full Workflow Example

```bash
# 1. Download GGUF model
hfmdl download unsloth/Qwen3.5-35B-A3B-GGUF \
  --allow-pattern "*UD-Q6_K_XL*" \
  --output ./my-models

# 2. Generate and create Ollama model
python scripts/generate_modelfile.py \
  ./my-models/Qwen3.5-35B-A3B-UD-Q6_K_XL.gguf \
  Qwen/Qwen3.5-0.8B \
  --temperature 0.6 \
  --create qwen35-35b-q6

# 3. Run the model
ollama run qwen35-35b-q6
```

## How It Works

### 1. Configuration Fetching

The script fetches these files from the HuggingFace repository:
- `config.json` - Model architecture and configuration
- `chat_template.jinja` - Chat template (if available separately)

### 2. Jinja2 Feature Analysis (New in v2.0)

With `--verbose`, the script analyzes the original Jinja2 template:

**Detected Features:**
- Tool calling patterns
- Vision/multimodal content
- Thinking/reasoning blocks

**Unsupported Features (reported as warnings):**
- Jinja2 macros (`{% macro %}`)
- Mutable namespace state
- Backward iteration (`messages[::-1]`)
- Complex filters

### 3. Template Mapping

Maps detected model family to enhanced Ollama templates:

| Model Family | Features | Stop Tokens |
|--------------|----------|-------------|
| **Qwen3** | Tools, Thinking, Vision | `<|im_start|>`, `<|im_end|>`, `<|vision_start|>`, `<|vision_end|>` |
| **Llama3** | Tools, Thinking | `<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>` |
| **Mistral** | Tools, Thinking | `[INST]`, `[/INST]`, `<|s|>` |
| **Phi3** | Tools, Thinking | `<|system|>`, `<|user|>`, `<|assistant|>`, `<|end|>` |
| **Generic** | Basic | `<|endoftext|>` |

### 4. Enhanced Template Examples

#### Qwen3 Template (Enhanced with Full Feature Support)

```go
{{- if .Tools }}<|im_start|>system
You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.
<tools>
{{ range .Tools }}
{"type": "{{ .Type }}", "function": {{ json .Function }}}
{{ end }}</tools>

# Tool Guidelines
- Function calls MUST follow the specified format
- Required parameters MUST be specified
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
```

**Key Improvements:**
1. **Tool Definitions**: Full JSON schema with `{{ json .Function }}`
2. **Tool Calls**: Proper argument iteration with names and values
3. **Tool Call IDs**: `{{ .ToolCallID }}` for response correlation
4. **Thinking Content**: `{{ .Thinking }}` for reasoning models
5. **Vision**: `{{ .Images }}` with placeholder tokens

## Command Line Options

```
usage: generate_modelfile.py [-h] [-o OUTPUT] [--create MODEL_NAME]
                             [--temperature TEMPERATURE] [--top-p TOP_P]
                             [--top-k TOP_K] [--verbose]
                             gguf_path repo_id

positional arguments:
  gguf_path             Path to the GGUF file
  repo_id               HuggingFace model repository ID (e.g.,
                        Qwen/Qwen3.5-0.8B)

options:
  -h, --help            Show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Output file path (default: print to stdout)
  --create MODEL_NAME   Automatically create the model in Ollama with the
                        given name
  --temperature TEMPERATURE
                        Set temperature parameter
  --top-p TOP_P         Set top_p parameter
  --top-k TOP_K         Set top_k parameter
  --verbose, -v         Show detailed Jinja2 template analysis and warnings
```

## Configuration

### Mirror Endpoint

The script automatically uses your hfmdl configuration:

```bash
# Check current endpoint
python -c "from hf_model_downloader.config import load_settings; print(load_settings().get_effective_endpoint())"

# Edit configuration
nano ~/.config/hfmdl/config.toml
```

### Environment Variable

You can also override the endpoint via environment variable:

```bash
export HF_ENDPOINT=https://huggingface.co
python scripts/generate_modelfile.py ./model.gguf Qwen/Qwen3.5-0.8B
```

## Supported Model Types

### Fully Supported (with all enhanced features)

| Model | Tool ID | Thinking | Vision |
|-------|---------|----------|--------|
| **Qwen3 / Qwen3.5** | ✅ | ✅ | ✅ |
| **Llama 3 / 3.1** | ✅ | ✅ | ❌ |
| **Mistral / Mixtral** | ✅ | ✅ | ❌ |
| **Phi3 / Phi4** | ✅ | ✅ | ❌ |

### Notes on Vision Support

Vision support is currently only available in the Qwen3 template because:
1. It uses `<|vision_start|><|image_pad|><|vision_end|>` tokens
2. The Qwen3 Jinja template shows the expected format

Other models (Llama3, Mistral) don't have vision-specific tokens in their original Jinja templates.

## Troubleshooting

### Repository Not Found

```bash
# Verify the repository exists
python -c "from huggingface_hub import model_info; print(model_info('Qwen/Qwen3.5-0.8B'))"
```

### Permission Denied

For gated models, ensure you have access:

```bash
export HF_TOKEN=your_token_here
python scripts/generate_modelfile.py ./model.gguf gated-model/repo
```

### Unsupported Jinja2 Features

If you see warnings about unsupported features, the generated template may not perfectly match the original behavior. Common workarounds:

1. **Macros**: Tool calling may be simplified but still functional
2. **Backward iteration**: Tool response matching may not be perfect
3. **Namespace state**: Some complex stateful logic may be lost

### Template Not Working as Expected

1. Check the source repository's `chat_template.jinja` manually
2. Use `--verbose` to see what features were detected and unsupported
3. Edit the generated Modelfile's TEMPLATE section manually if needed
4. Report an issue with the model type, expected behavior, and Jinja2 template

## Implementation Notes

### Design Decisions

1. **Hybrid Approach**: Rather than attempting full Jinja2→Go conversion (which is theoretically impossible), we:
   - Use pre-researched templates for known model families
   - Analyze Jinja2 to detect features and warn about limitations
   - Enhance templates to use all available Ollama features

2. **Information Preservation**:
   - **Preserved**: Role formats, stop tokens, basic message flow
   - **Enhanced**: Tool calling with IDs, thinking content, vision placeholders
   - **Lost**: Complex Jinja2 macros, stateful namespace logic, backward iteration

3. **Ollama Features Used**:
   - `{{ .Tools }}` - Tool definitions
   - `{{ .Messages[].ToolCalls }}` - Tool call IDs and arguments
   - `{{ .Messages[].ToolCallID }}` - Response correlation
   - `{{ .Messages[].Thinking }}` - Reasoning content
   - `{{ .Messages[].Images }}` - Vision inputs
   - `{{ json }}` - JSON serialization
   - `{{ currentDate }}` - Date injection

### Limitations

**Cannot be fully converted from Jinja2:**
- Template macros (`{% macro %}`)
- Mutable state (`namespace()`)
- Backward iteration (`[::-1]`)
- Complex filters beyond basic string manipulation
- Multi-pass template logic

These are fundamental limitations of the Go template system, not bugs in this script.

## Future Enhancements

Potential improvements:
- Interactive template selection
- Custom template override files
- Batch processing multiple GGUFs
- Integration into hfmdl CLI
- Template validation against Ollama runtime

## References

- [Ollama Modelfile Documentation](https://github.com/ollama/ollama/blob/main/docs/modelfile.md)
- [Ollama Template Documentation](https://github.com/ollama/ollama/blob/main/docs/TEMPLATE.md)
- [Ollama Thinking Capability](https://docs.ollama.com/capabilities/thinking)
- [HuggingFace Chat Templates](https://huggingface.co/docs/transformers/main/chat_templating)
