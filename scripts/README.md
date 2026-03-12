# Modelfile Generator Script

Generate Ollama Modelfile from GGUF files and HuggingFace model configuration.

This script fetches `chat_template.jinja` and `config.json` from the source repository and generates an accurate Ollama Modelfile that preserves the original configuration as much as possible.

## Features

- ✅ **Accurate Templates** - Uses Oracle-researched templates from Ollama official library (Qwen3, Llama3, Mistral, Phi3)
- ✅ **Configuration Reuse** - Automatically reads mirror settings from `~/.config/hfmdl/config.toml`
- ✅ **Model Family Detection** - Automatically detects model type from config.json
- ✅ **One-click Model Creation** - Supports `--create` flag to directly create Ollama models
- ✅ **No External Dependencies** - Uses existing `huggingface_hub` package

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

### 2. Model Family Detection

Detects model type from:
- `model_type` field in config.json
- `architectures` field in config.json
- Chat template content patterns

### 3. Template Mapping

Maps detected model family to accurate Ollama templates:

| Model Family | Template Features | Stop Tokens |
|--------------|-------------------|-------------|
| **Qwen3** | ChatML format with tool support | `<|im_start|>`, `<|im_end|>` |
| **Llama3** | Header format with tool support | `<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>` |
| **Mistral** | INST format with tool support | `[INST]`, `[/INST]`, `<|s|>` |
| **Phi3** | Simple token format | `<|system|>`, `<|user|>`, `<|assistant|>`, `<|end|>` |
| **Generic** | Basic fallback template | `<|endoftext|>` |

### 4. Template Details

#### Qwen3 Template (ChatML)
```go
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 }}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ if $last }}<|im_start|>assistant
{{ end }}
{{- else if eq .Role "assistant" }}<|im_start|>assistant
{{ .Content }}{{ if not $last }}<|im_end|>
{{ end }}
{{- end }}
{{- end }}
```

#### Llama3 Template (Header)
```go
{{- range .Messages }}
{{- if eq .Role "user" }}<|start_header_id|>user<|end_header_id|>

{{ .Content }}<|eot_id|>
{{- else if eq .Role "assistant" }}<|start_header_id|>assistant<|end_header_id|>

{{ .Content }}<|eot_id|>
{{- end }}
{{- end }}<|start_header_id|>assistant<|end_header_id|>
```

## Command Line Options

```
usage: generate_modelfile.py [-h] [-o OUTPUT] [--create MODEL_NAME]
                             [--temperature TEMPERATURE] [--top-p TOP_P]
                             [--top-k TOP_K]
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

### Fully Supported
- **Qwen3 / Qwen3.5** - Complete ChatML template with tool support
- **Llama 3 / 3.1** - Full header template with tool support
- **Mistral / Mixtral** - INST format with tool support
- **Phi3 / Phi4** - Simple token format

### Partially Supported
Any other model type will use a generic fallback template. You can manually edit the generated Modelfile to adjust the TEMPLATE section.

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

### Template Not Matching

If the generated template doesn't work correctly:

1. Check the source repository's `chat_template.jinja` manually
2. Edit the generated Modelfile's TEMPLATE section
3. Report an issue with the model type and expected behavior

## Implementation Notes

### Template Research

Templates were researched from Ollama's official library:
- Qwen3 template from `ollama pull qwen3`
- Llama3 template from `ollama pull llama3`
- Mistral template from `ollama pull mistral`
- Phi3 template from `ollama pull phi3`

### Design Decisions

1. **Hardcoded Templates**: Rather than attempting to convert Jinja2 to Go templates (which is complex and error-prone), we use accurate pre-researched templates for known model families.

2. **Configuration Reuse**: The script reuses hfmdl's configuration system to ensure consistent mirror usage across tools.

3. **Graceful Degradation**: If files can't be fetched, the script still generates a working Modelfile with sensible defaults.

## Future Enhancements

Potential improvements:
- Interactive template selection
- Custom template override files
- Batch processing multiple GGUFs
- Integration into hfmdl CLI

## References

- [Ollama Modelfile Documentation](https://github.com/ollama/ollama/blob/main/docs/modelfile.md)
- [Ollama Template Documentation](https://github.com/ollama/ollama/blob/main/docs/TEMPLATE.md)
- [HuggingFace Chat Templates](https://huggingface.co/docs/transformers/main/chat_templating)
