# Modelfile 生成脚本

基于 GGUF 文件和 HuggingFace 仓库元数据自动生成 Ollama Modelfile。

## 安装依赖

```bash
pip install requests
```

## 基本用法

```bash
# 基础用法
python scripts/generate_modelfile.py ./Qwen3.5-35B-A3B-UD-Q6_K_XL.gguf Qwen/Qwen3.5-0.8B

# 指定系统提示词
python scripts/generate_modelfile.py ./model.gguf Qwen/Qwen3.5-0.8B \
  --system "You are a coding expert"

# 指定温度参数
python scripts/generate_modelfile.py ./model.gguf Qwen/Qwen3.5-0.8B \
  --temperature 0.6

# 生成并自动创建 Ollama 模型
python scripts/generate_modelfile.py ./model.gguf Qwen/Qwen3.5-0.8B \
  --create --name my-model
```

## 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `gguf_path` | GGUF 文件路径 | `./model.gguf` |
| `repo_id` | HuggingFace 仓库 ID | `Qwen/Qwen3.5-0.8B` |
| `--system` | 系统提示词 | `"You are a helpful AI"` |
| `--temperature` | 温度参数 | `0.6` |
| `--output` | Modelfile 输出路径 | `./Modelfile` |
| `--create` | 自动生成 Ollama 模型 | - |
| `--name` | Ollama 模型名称 | `my-model` |

## 工作原理

脚本会从源仓库获取以下文件：
1. `chat_template.jinja` - 聊天模板
2. `config.json` - 模型配置

然后自动解析：
- 消息格式（IM 格式、Llama 格式等）
- 停止词（stop tokens）
- 上下文长度
- 模型类型

生成包含正确 TEMPLATE 和 PARAMETER 的 Modelfile。

## 完整示例

### 1. 下载 GGUF

```bash
hfmdl download unsloth/Qwen3.5-35B-A3B-GGUF \
  --allow-pattern "*UD-Q6_K_XL*" \
  --output ./my-models
```

### 2. 生成 Modelfile

```bash
python scripts/generate_modelfile.py \
  ./my-models/Qwen3.5-35B-A3B-UD-Q6_K_XL.gguf \
  Qwen/Qwen3.5-0.8B \
  --system "You are Qwen, a large language model created by Alibaba Cloud." \
  --temperature 0.6 \
  --create \
  --name qwen35-35b-q6
```

### 3. 运行模型

```bash
ollama run qwen35-35b-q6
```

## 支持的模型类型

- **Qwen3/3.5**: 自动识别 `<|im_start|>` 格式
- **Llama 3**: 自动识别 `<|eot_id|>` 格式
- **其他**: 使用通用模板

## 故障排除

### 无法获取仓库元数据

如果脚本无法访问 HuggingFace，可以使用镜像：

```bash
python scripts/generate_modelfile.py ./model.gguf Qwen/Qwen3.5-0.8B \
  --endpoint https://hf-mirror.com
```

### 模板不正确

可以手动编辑生成的 Modelfile，调整 TEMPLATE 部分。

## 与下载器集成（未来）

可以考虑将此功能集成到 `hfmdl` 中：

```bash
# 未来可能的用法
hfmdl download unsloth/Qwen3.5-35B-A3B-GGUF \
  --allow-pattern "*UD-Q6_K_XL*" \
  --output ./my-models \
  --ollama-create \
  --ollama-name qwen35-35b-q6 \
  --ollama-template-source Qwen/Qwen3.5-0.8B
```
