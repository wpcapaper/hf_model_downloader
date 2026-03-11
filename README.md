# HuggingFace Model Downloader (hfmdl)

A command line tool for downloading models, datasets, and spaces from HuggingFace Hub with automatic retry logic and mirror support.

## Installation

Install with uv:

```bash
uv tool install hf-model-downloader
```

Or clone and install locally:

```bash
git clone <repo-url>
cd hf-model-downloader
uv sync
```

## Quick Start

```bash
# Download a model
hfmdl download bert-base-uncased

# Download a specific revision
hfmdl download bert-base-uncased --revision v1.0

# Download to a specific directory
hfmdl download bert-base-uncased --output ./models

# Use a mirror (default: hf-mirror.com)
HF_ENDPOINT=https://hf-mirror.com hfmdl download bert-base-uncased
```

## Configuration

### Config File Location

The config file is stored in a platform-specific location:

- **macOS**: `~/Library/Application Support/hfmdl/config.toml`
- **Linux**: `~/.config/hfmdl/config.toml`
- **Windows**: `C:\Users\<user>\AppData\Roaming\hfmdl\config.toml`

A default config is auto created on first run.

### Config File Example

```toml
# HuggingFace endpoint (default: mirror)
endpoint = "https://hf-mirror.com"

# Download timeout in seconds
hf_hub_download_timeout = 60.0

# ETag request timeout
hf_hub_etag_timeout = 30.0

# High performance mode for parallel downloads
hf_xet_high_performance = true

# Number of concurrent range requests
hf_xet_num_concurrent_range_gets = 16

# Maximum parallel workers
max_workers = 8

# Cache directory (None = platform default)
cache_dir = null

[retry]
# Retry indefinitely on transient failures
forever = true

# Max retry attempts (ignored if forever=true)
max_attempts = null

# Max total seconds to retry (null = no limit)
max_total_seconds = null

# Base wait time between retries (seconds)
base_wait = 1.0

# Maximum wait time between retries (seconds)
max_wait = 60.0

# Jitter factor (0.0-1.0) for randomized backoff
jitter = 0.2

# Log every retry attempt
log_every_attempt = true

# Model profiles
[[models]]
name = "bert-tiny"
repo_id = "prajjwal1/bert-tiny"
revision = "main"
repo_type = "model"

[[models]]
name = "gpt2-small"
repo_id = "gpt2"
revision = "main"
repo_type = "model"
allow_patterns = ["*.json", "*.bin"]
output_dir = "./models/gpt2"
```

## Commands

### download

Download a model, dataset, or space from HuggingFace.

```bash
# Basic download
hfmdl download bert-base-uncased

# Download using a profile
hfmdl download --profile bert-tiny

# Download a specific revision
hfmdl download bert-base-uncased --revision v1.0

# Download a dataset
hfmdl download squad --repo-type dataset

# Download to a specific directory
hfmdl download bert-base-uncased --output ./models

# Filter files with patterns
hfmdl download bert-base-uncased --allow-pattern "*.bin" --allow-pattern "*.json"

# Ignore specific files
hfmdl download bert-base-uncased --ignore-pattern "*.safetensors"

# Override endpoint
hfmdl download bert-base-uncased --endpoint https://huggingface.co

# Force re download
hfmdl download bert-base-uncased --force-download

# Set max workers
hfmdl download bert-base-uncased --max-workers 4
```

#### Download Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--profile` | `-p` | Use a configured profile | None |
| `--revision` | | Model revision/branch/tag | `main` |
| `--repo-type` | | Repository type: model, dataset, space | `model` |
| `--endpoint` | | Override endpoint URL | Config value |
| `--force-endpoint` | `-f` | Ignore HF_ENDPOINT env var | False |
| `--output` | `-o` | Output/cache directory | Config value |
| `--allow-pattern` | | File patterns to include (repeatable) | None |
| `--ignore-pattern` | | File patterns to exclude (repeatable) | None |
| `--max-workers` | | Maximum parallel workers | Config value |
| `--token` | | HuggingFace API token | `HF_TOKEN` env |
| `--force-download` | | Force re download even if cached | False |
| `--retry-forever` | | Retry indefinitely on failures | Config value |
| `--no-retry-forever` | | Disable retry forever mode | False |
| `--max-attempts` | | Maximum retry attempts | Config value |
| `--max-total-seconds` | | Maximum total seconds to retry | Config value |

### validate

Check if a repository exists and is accessible without downloading.

```bash
# Validate a model
hfmdl validate bert-base-uncased

# Validate a specific revision
hfmdl validate bert-base-uncased --revision v1.0

# Validate a dataset
hfmdl validate squad --repo-type dataset
```

### show-config

Display current configuration settings.

```bash
# Show config
hfmdl show-config

# Override endpoint for display
hfmdl show-config --endpoint https://huggingface.co
```

### list-profiles

List all configured model profiles.

```bash
hfmdl list-profiles
```

## Authentication

### Using HF_TOKEN

Set the `HF_TOKEN` environment variable:

```bash
export HF_TOKEN=your_token_here
hfmdl download gated-model
```

Or pass it directly:

```bash
hfmdl download gated-model --token your_token_here
```

### Gated Models

For gated models that require access approval:

1. Visit the model page on HuggingFace
2. Request and receive access
3. Set your `HF_TOKEN` as shown above
4. Download normally

The tool will fail fast with a clear error message if you try to access a gated model without proper authentication.

## Mirror Support

### Default Mirror

The tool uses `https://hf-mirror.com` as the default endpoint. This is a community maintained mirror that can provide faster downloads in some regions.

### Overriding the Mirror

Use the `HF_ENDPOINT` environment variable:

```bash
export HF_ENDPOINT=https://huggingface.co
hfmdl download bert-base-uncased
```

Or use the `--endpoint` flag:

```bash
hfmdl download bert-base-uncased --endpoint https://huggingface.co
```

### Force Config Endpoint

To ignore the `HF_ENDPOINT` environment variable and use the config file value:

```bash
hfmdl download bert-base-uncased --force-endpoint
```

## Retry Behavior

### Transient Failures Only

The tool only retries on transient errors:

- **429** (rate limiting)
- **5xx** server errors (500-599)
- Timeouts
- Connection errors (DNS failures, connection reset)

### Non-Retriable Failures

These errors fail immediately without retry:

- **401** (authentication failed)
- **403** (forbidden/gated repo)
- **404** (not found)
- Repository not found
- Revision not found
- Gated repo errors
- Disk errors (no space, permission denied)
- Configuration errors

### Retry Configuration

By default, the tool retries indefinitely on transient errors with exponential backoff.

Control retry behavior:

```bash
# Disable infinite retry
hfmdl download bert-base-uncased --no-retry-forever

# Limit to 5 attempts
hfmdl download bert-base-uncased --max-attempts 5

# Limit total retry time to 300 seconds
hfmdl download bert-base-uncased --max-total-seconds 300

# Combine limits
hfmdl download bert-base-uncased --no-retry-forever --max-attempts 10 --max-total-seconds 600
```

### Stopping a Download

Press `Ctrl+C` to abort a download. The tool exits with code 4 (aborted by user).

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 2 | Configuration error (invalid profile, missing repo_id) |
| 3 | Non-retriable remote error (404, auth failure, etc.) |
| 4 | Aborted by user (Ctrl+C) |

## Examples

### Download Specific Files

```bash
# Download only model weights
hfmdl download bert-base-uncased --allow-pattern "*.bin"

# Download config and tokenizer
hfmdl download bert-base-uncased --allow-pattern "*.json" --allow-pattern "*.txt"

# Download everything except safetensors
hfmdl download bert-base-uncased --ignore-pattern "*.safetensors"
```

### Using Profiles

First, add a profile to your config:

```toml
[[models]]
name = "my-model"
repo_id = "organization/my-model"
revision = "v2.0"
repo_type = "model"
allow_patterns = ["*.bin", "config.json"]
output_dir = "./models/my-model"
```

Then download:

```bash
hfmdl download --profile my-model
```

### Download Datasets and Spaces

```bash
# Download a dataset
hfmdl download squad --repo-type dataset

# Download a space
hfmdl download gradio/hello-world --repo-type space
```

### Version Check

```bash
hfmdl --version
```

## Troubleshooting

### Connection Timeout

If downloads timeout frequently, increase the timeout in your config:

```toml
hf_hub_download_timeout = 120.0
hf_hub_etag_timeout = 60.0
```

### Slow Downloads

1. Try the mirror (default)
2. Increase concurrent workers:

```bash
hfmdl download bert-base-uncased --max-workers 16
```

3. Enable high performance mode in config:

```toml
hf_xet_high_performance = true
hf_xet_num_concurrent_range_gets = 32
```

### Gated Model Access Denied

1. Verify you have access on the HuggingFace website
2. Check your token: `hfmdl validate gated-model --token your_token`
3. Ensure `HF_TOKEN` is set correctly

### Repository Not Found

1. Check the repository ID spelling
2. Verify it's a public repo or you have access
3. Try validating first: `hfmdl validate org/model-name`

### Config File Not Found

The config file is auto created on first run. If you need to reset it:

```bash
# Remove config (backup first if needed)
rm ~/.config/hfmdl/config.toml  # Linux
rm ~/Library/Application\ Support/hfmdl/config.toml  # macOS

# Run any command to recreate
hfmdl show-config
```

### Retry Loop Won't Stop

The default retry behavior is infinite. To stop:

1. Press `Ctrl+C`
2. For future runs, use limits:

```bash
hfmdl download model --no-retry-forever --max-attempts 5
```

### Wrong Files Downloaded

Use allow/ignore patterns:

```bash
# Only specific extensions
hfmdl download model --allow-pattern "*.bin" --allow-pattern "*.json"

# Exclude large files
hfmdl download model --ignore-pattern "*.safetensors"
```

## License

MIT License
