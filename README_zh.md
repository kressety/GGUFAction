[English](README.md) | 中文

# GGUFAction

**GGUFAction** 是一个自动化工具，通过 GitHub Actions 从 Hugging Face 下载模型，将其转换为 GGUF 格式并进行 Q8_0 量化，然后上传到 ModelScope。它支持缓存不支持的模型以避免重复失败，并生成包含原始模型卡片和量化说明的 `README.md`。

## 功能特点

- **自动化转换**：从 Hugging Face 下载模型，转换为 GGUF 格式，使用 `llama.cpp` 进行 Q8_0 量化。
- **智能缓存**：记录不支持 GGUF 转换的模型，下次输入时自动终止。
- **模型上传**：将量化后的模型和生成的 `README.md` 上传到 ModelScope。
- **文档生成**：从 Hugging Face 获取原始模型卡片（`---` 包裹的 YAML 部分），并添加量化说明。

## 先决条件

- **GitHub 账户**：用于运行 Actions 和存储 Secrets。
- **Hugging Face 账户**：获取 API 令牌（`HF_API_KEY`）。
- **ModelScope 账户**：获取 API 令牌（`MS_API_KEY`）和用户名（`MS_USERNAME`）。

## 安装

1. **克隆仓库**：
   ```bash
   git clone https://github.com/your-username/GGUFAction.git
   cd GGUFAction
   ```

2. **设置 Secrets**：
   在 GitHub 仓库的 `Settings > Secrets and variables > Actions > Secrets` 中添加：
   - `HF_API_KEY`：Hugging Face API 令牌。
   - `MS_API_KEY`：ModelScope API 令牌。
   - `MS_USERNAME`：ModelScope 用户名。

3. **依赖项**：
   项目使用 Python 3.13 和以下库（已包含在 `requirements.txt`）：
   ```
   huggingface_hub
   modelscope
   torch
   sentencepiece
   numpy
   transformers
   requests
   --extra-index-url https://download.pytorch.org/whl/cpu
   ```
   安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

## 使用方法

### 通过 GitHub Actions
1. **触发工作流**：
   - 访问仓库的 `Actions` 标签。
   - 选择 `Model Quantization` 工作流。
   - 点击 `Run workflow`，输入 Hugging Face 模型的 `repo_id`（如 `Classical/Yinka`）。

2. **检查结果**：
   - 查看 Actions 日志（`quantization.log`）确认转换和上传状态。
   - 在 ModelScope 上查看 `{MS_USERNAME}/{model_name}-Q8_0-GGUF` 仓库。

### 本地运行
1. **设置环境变量**：
   ```bash
   export HF_API_KEY="your_hf_token"
   export MS_API_KEY="your_ms_token"
   export MS_USERNAME="your_ms_username"
   export REPO_ID="Classical/Yinka"
   ```

2. **编译 llama.cpp**：
   ```bash
   git clone https://github.com/ggerganov/llama.cpp.git
   cd llama.cpp
   cmake -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build --config Release -j$(nproc)
   cd ..
   ```

3. **运行脚本**：
   ```bash
   python script.py
   ```

## 输出
- **量化模型**：`model_q8.gguf`，上传至 ModelScope。
- **README.md**：包含原始模型卡片（`---` 包裹的 YAML）和量化说明，示例：
  ```markdown
  ---
  tags:
  - mteb
  model-index:
  ...
  ---

  ## Classical/Yinka-Q8_0-GGUF
  This model has been quantized to Q8_0 format using `llama.cpp`...
  ```

## 已知限制
- **模型兼容性**：并非所有 Hugging Face 模型支持 GGUF 转换（如某些 Mistral 变体）。不支持的模型会被缓存并记录在 `unsupported_models.txt`。
- **构建时间**：每次运行重新编译 `llama.cpp`，可能需要 1-2 分钟。

## 贡献
欢迎提交 Issues 或 Pull Requests！改进方向包括：
- 支持更多量化格式（如 Q4_0）。
- 优化 `llama.cpp` 构建时间。
- 增强模型卡片解析逻辑。

## 许可证
采用 [MIT License](LICENSE)，详见仓库根目录。

## 致谢
- [llama.cpp](https://github.com/ggerganov/llama.cpp)：提供 GGUF 转换和量化工具。
- [Hugging Face](https://huggingface.co) 和 [ModelScope](https://modelscope.cn)：模型托管平台。
