[English](./README.md) | 简体中文

# HuggingFace 模型下载器 (hfmdl)

一个命令行工具，用于从 HuggingFace Hub 下载模型、数据集和空间，支持自动重试和镜像加速。

## 安装

使用 uv 安装：

```bash
uv tool install hf-model-downloader
```

或克隆到本地安装：

```bash
git clone <repo-url>
cd hf-model-downloader
uv sync
```

## 快速开始

```bash
# 下载 GGUF 模型的特定量化版本
hfmdl download unsloth/Qwen3.5-35B-A3B-GGUF --allow-pattern "*UD-Q6_K_XL*"

# 下载到指定目录
hfmdl download unsloth/Qwen3.5-35B-A3B-GGUF --allow-pattern "*UD-Q6_K_XL*" --output ./models

# 使用镜像（默认：hf-mirror.com）
HF_ENDPOINT=https://hf-mirror.com hfmdl download unsloth/Qwen3.5-35B-A3B-GGUF --allow-pattern "*UD-Q6_K_XL*"
```

## 配置

### 配置文件位置

配置文件存储在平台特定的位置：

- **macOS**: `~/Library/Application Support/hfmdl/config.toml`
- **Linux**: `~/.config/hfmdl/config.toml`
- **Windows**: `C:\Users\<user>\AppData\Roaming\hfmdl\config.toml`

首次运行时会自动创建默认配置。

### 配置文件示例

```toml
# HuggingFace 端点（默认：镜像）
endpoint = "https://hf-mirror.com"

# 下载超时时间（秒）
hf_hub_download_timeout = 60.0

# ETag 请求超时时间
hf_hub_etag_timeout = 30.0

# 高性能并行下载模式
hf_xet_high_performance = true

# 并发范围请求数量
hf_xet_num_concurrent_range_gets = 16

# 最大并行工作线程数
max_workers = 8

# 缓存目录（null = 平台默认）
cache_dir = null

[retry]
# 对瞬态错误无限重试
forever = true

# 最大重试次数（forever=true 时忽略）
max_attempts = null

# 最大重试总时间（null = 无限制）
max_total_seconds = null

# 重试基础等待时间（秒）
base_wait = 1.0

# 重试最大等待时间（秒）
max_wait = 60.0

# 抖动因子（0.0-1.0）用于随机退避
jitter = 0.2

# 记录每次重试
log_every_attempt = true

# 模型配置文件
[[models]]
name = "qwen35-35b-q6"
repo_id = "unsloth/Qwen3.5-35B-A3B-GGUF"
revision = "main"
repo_type = "model"
allow_patterns = ["*UD-Q6_K_XL*"]
output_dir = "./models/qwen35"
```

## 命令

### download

从 HuggingFace 下载模型、数据集或空间。

```bash
# 基本下载
hfmdl download bert-base-uncased

# 使用配置文件下载
hfmdl download --profile qwen35-35b-q6

# 下载指定版本
hfmdl download bert-base-uncased --revision v1.0

# 下载数据集
hfmdl download squad --repo-type dataset

# 下载到指定目录
hfmdl download bert-base-uncased --output ./models

# 使用模式过滤文件
hfmdl download bert-base-uncased --allow-pattern "*.bin" --allow-pattern "*.json"

# 忽略特定文件
hfmdl download bert-base-uncased --ignore-pattern "*.safetensors"

# 覆盖端点
hfmdl download bert-base-uncased --endpoint https://huggingface.co

# 强制重新下载
hfmdl download bert-base-uncased --force-download

# 设置最大工作线程数
hfmdl download bert-base-uncased --max-workers 4
```

#### 下载选项

| 选项 | 简写 | 描述 | 默认值 |
|------|------|------|--------|
| `--profile` | `-p` | 使用配置文件中的配置 | None |
| `--revision` | | 模型版本/分支/标签 | `main` |
| `--repo-type` | | 仓库类型：model, dataset, space | `model` |
| `--endpoint` | | 覆盖端点 URL | 配置值 |
| `--force-endpoint` | `-f` | 忽略 HF_ENDPOINT 环境变量 | False |
| `--output` | `-o` | 输出/缓存目录 | 配置值 |
| `--allow-pattern` | | 包含的文件模式（可重复） | None |
| `--ignore-pattern` | | 排除的文件模式（可重复） | None |
| `--max-workers` | | 最大并行工作线程数 | 配置值 |
| `--token` | | HuggingFace API 令牌 | `HF_TOKEN` 环境变量 |
| `--force-download` | | 强制重新下载（即使已缓存） | False |
| `--retry-forever` | | 对失败无限重试 | 配置值 |
| `--no-retry-forever` | | 禁用无限重试模式 | False |
| `--max-attempts` | | 最大重试次数 | 配置值 |
| `--max-total-seconds` | | 最大重试总时间（秒） | 配置值 |

### validate

检查仓库是否存在且可访问，无需下载。

```bash
# 验证模型
hfmdl validate bert-base-uncased

# 验证指定版本
hfmdl validate bert-base-uncased --revision v1.0

# 验证数据集
hfmdl validate squad --repo-type dataset
```

### show-config

显示当前配置设置。

```bash
# 显示配置
hfmdl show-config

# 覆盖端点显示
hfmdl show-config --endpoint https://huggingface.co
```

### list-profiles

列出所有配置的模型配置文件。

```bash
hfmdl list-profiles
```

## 身份认证

### 使用 HF_TOKEN

设置 `HF_TOKEN` 环境变量：

```bash
export HF_TOKEN=your_token_here
hfmdl download gated-model
```

或直接传入：

```bash
hfmdl download gated-model --token your_token_here
```

### 受限模型

对于需要访问审批的受限模型：

1. 访问 HuggingFace 上的模型页面
2. 申请并获得访问权限
3. 按上述方法设置 `HF_TOKEN`
4. 正常下载

如果你在没有正确认证的情况下尝试访问受限模型，工具会立即失败并显示清晰的错误信息。

## 镜像支持

### 默认镜像

工具默认使用 `https://hf-mirror.com` 作为端点。这是一个社区维护的镜像，在某些地区可以提供更快的下载速度。

### 覆盖镜像

使用 `HF_ENDPOINT` 环境变量：

```bash
export HF_ENDPOINT=https://huggingface.co
hfmdl download bert-base-uncased
```

或使用 `--endpoint` 标志：

```bash
hfmdl download bert-base-uncased --endpoint https://huggingface.co
```

### 强制使用配置端点

要忽略 `HF_ENDPOINT` 环境变量并使用配置文件中的值：

```bash
hfmdl download bert-base-uncased --force-endpoint
```

## 重试行为

### 仅对瞬态错误重试

工具仅对以下瞬态错误进行重试：

- **429**（请求频率限制）
- **5xx** 服务器错误（500-599）
- 超时
- 连接错误（DNS 失败、连接重置）

### 不可重试的错误

以下错误会立即失败，不进行重试：

- **401**（认证失败）
- **403**（禁止访问/受限仓库）
- **404**（未找到）
- 仓库不存在
- 版本不存在
- 受限仓库错误
- 磁盘错误（空间不足、权限被拒绝）
- 配置错误

### 重试配置

默认情况下，工具对瞬态错误使用指数退避策略无限重试。

控制重试行为：

```bash
# 禁用无限重试
hfmdl download bert-base-uncased --no-retry-forever

# 限制最多 5 次尝试
hfmdl download bert-base-uncased --max-attempts 5

# 限制重试总时间为 300 秒
hfmdl download bert-base-uncased --max-total-seconds 300

# 组合限制
hfmdl download bert-base-uncased --no-retry-forever --max-attempts 10 --max-total-seconds 600
```

### 停止下载

按 `Ctrl+C` 中止下载。工具会以退出码 4 退出（用户中止）。

## 退出码

| 代码 | 含义 |
|------|------|
| 0 | 成功 |
| 2 | 配置错误（无效的配置文件、缺少 repo_id） |
| 3 | 不可重试的远程错误（404、认证失败等） |
| 4 | 用户中止（Ctrl+C） |

## 示例

### 下载 GGUF 模型的特定量化版本

以 [unsloth/Qwen3.5-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF) 为例：

```bash
# 下载 UD-Q6_K_XL 量化版本（约 29GB）
hfmdl download unsloth/Qwen3.5-35B-A3B-GGUF --allow-pattern "*UD-Q6_K_XL*"

# 下载多个量化版本
hfmdl download unsloth/Qwen3.5-35B-A3B-GGUF \
  --allow-pattern "*UD-Q4_K_XL*" \
  --allow-pattern "*UD-Q6_K_XL*"

# 下载模型和配置文件
hfmdl download unsloth/Qwen3.5-35B-A3B-GGUF \
  --allow-pattern "*UD-Q6_K_XL*" \
  --allow-pattern "config.json"
```

该仓库可用的量化版本：

| 量化版本 | 模式 | 大小 |
|----------|------|------|
| UD-Q2_K_XL | `*UD-Q2_K_XL*` | ~11 GB |
| UD-Q3_K_XL | `*UD-Q3_K_XL*` | ~15 GB |
| UD-Q4_K_XL | `*UD-Q4_K_XL*` | ~20 GB |
| UD-Q5_K_XL | `*UD-Q5_K_XL*` | ~24 GB |
| UD-Q6_K_XL | `*UD-Q6_K_XL*` | ~29 GB |
| UD-Q8_K_XL | `*UD-Q8_K_XL*` | ~45 GB |

### 下载特定文件

```bash
# 仅下载模型权重
hfmdl download bert-base-uncased --allow-pattern "*.bin"

# 下载配置和分词器
hfmdl download bert-base-uncased --allow-pattern "*.json" --allow-pattern "*.txt"

# 下载除 safetensors 外的所有内容
hfmdl download bert-base-uncased --ignore-pattern "*.safetensors"
```

### 使用配置文件

首先，在配置中添加一个配置文件：

```toml
[[models]]
name = "qwen35-35b-q6"
repo_id = "unsloth/Qwen3.5-35B-A3B-GGUF"
revision = "main"
repo_type = "model"
allow_patterns = ["*UD-Q6_K_XL*"]
output_dir = "./models/qwen35"
```

然后下载：

```bash
hfmdl download --profile qwen35-35b-q6
```

### 下载数据集和空间

```bash
# 下载数据集
hfmdl download squad --repo-type dataset

# 下载空间
hfmdl download gradio/hello-world --repo-type space
```

### 查看版本

```bash
hfmdl --version
```

## 故障排除

### 连接超时

如果下载经常超时，在配置中增加超时时间：

```toml
hf_hub_download_timeout = 120.0
hf_hub_etag_timeout = 60.0
```

### 下载速度慢

1. 尝试使用镜像（默认）
2. 增加并发工作线程数：

```bash
hfmdl download unsloth/Qwen3.5-35B-A3B-GGUF --allow-pattern "*UD-Q6_K_XL*" --max-workers 16
```

3. 在配置中启用高性能模式：

```toml
hf_xet_high_performance = true
hf_xet_num_concurrent_range_gets = 32
```

### 受限模型访问被拒绝

1. 确认你已在 HuggingFace 网站上获得访问权限
2. 检查你的令牌：`hfmdl validate gated-model --token your_token`
3. 确保 `HF_TOKEN` 设置正确

### 仓库未找到

1. 检查仓库 ID 拼写
2. 确认是公开仓库或你有访问权限
3. 先尝试验证：`hfmdl validate org/model-name`

### 配置文件未找到

首次运行时会自动创建配置文件。如果需要重置：

```bash
# 删除配置（如需要请先备份）
rm ~/.config/hfmdl/config.toml  # Linux
rm ~/Library/Application\ Support/hfmdl/config.toml  # macOS

# 运行任意命令重新创建
hfmdl show-config
```

### 重试循环无法停止

默认重试行为是无限的。要停止：

1. 按 `Ctrl+C`
2. 以后运行时使用限制：

```bash
hfmdl download model --no-retry-forever --max-attempts 5
```

### 下载了错误的文件

使用 allow/ignore 模式：

```bash
# 仅特定扩展名
hfmdl download model --allow-pattern "*.bin" --allow-pattern "*.json"

# 排除大文件
hfmdl download model --ignore-pattern "*.safetensors"
```

## 许可证

MIT License
