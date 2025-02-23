English | [中文](README_zh.md)

# GGUFAction

**GGUFAction** is an automated tool that uses GitHub Actions to download models from Hugging Face, convert them to GGUF format, perform Q8_0 quantization, and upload them to ModelScope. It caches unsupported models to avoid repeated failures and generates a `README.md` with the original model card and quantization details.

## Features

- **Automated Conversion**: Downloads models from Hugging Face, converts them to GGUF format, and quantizes to Q8_0 using `llama.cpp`.
- **Smart Caching**: Records models unsupported for GGUF conversion and terminates early on subsequent attempts.
- **Model Upload**: Uploads the quantized model and a generated `README.md` to ModelScope.
- **Documentation Generation**: Fetches the original model card (YAML section wrapped in `---`) from Hugging Face and adds quantization instructions.

## Prerequisites

- **GitHub Account**: Required to run Actions and store Secrets.
- **Hugging Face Account**: Obtain an API token (`HF_API_KEY`).
- **ModelScope Account**: Obtain an API token (`MS_API_KEY`) and username (`MS_USERNAME`).

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/GGUFAction.git
   cd GGUFAction
   ```

2. **Set Up Secrets**:
   In your GitHub repository's `Settings > Secrets and variables > Actions > Secrets`, add:
   - `HF_API_KEY`: Hugging Face API token.
   - `MS_API_KEY`: ModelScope API token.
   - `MS_USERNAME`: ModelScope username.

3. **Dependencies**:
   The project uses Python 3.13 and the following libraries (listed in `requirements.txt`):
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
   Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Via GitHub Actions
1. **Trigger the Workflow**:
   - Go to the `Actions` tab in your repository.
   - Select the `Model Quantization` workflow.
   - Click `Run workflow` and enter the Hugging Face model `repo_id` (e.g., `Classical/Yinka`).

2. **Check Results**:
   - Review the Actions log (`quantization.log`) to confirm conversion and upload status.
   - Check the resulting `{MS_USERNAME}/{model_name}-Q8_0-GGUF` repository on ModelScope.

### Local Execution
1. **Set Environment Variables**:
   ```bash
   export HF_API_KEY="your_hf_token"
   export MS_API_KEY="your_ms_token"
   export MS_USERNAME="your_ms_username"
   export REPO_ID="Classical/Yinka"
   ```

2. **Build llama.cpp**:
   ```bash
   git clone https://github.com/ggerganov/llama.cpp.git
   cd llama.cpp
   cmake -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build --config Release -j$(nproc)
   cd ..
   ```

3. **Run the Script**:
   ```bash
   python script.py
   ```

## Output
- **Quantized Model**: `model_q8.gguf`, uploaded to ModelScope.
- **README.md**: Contains the original model card (wrapped in `---`) and quantization details, e.g.:
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

## Known Limitations
- **Model Compatibility**: Not all Hugging Face models support GGUF conversion (e.g., some Mistral variants). Unsupported models are cached in `unsupported_models.txt`.
- **Build Time**: Recompiling `llama.cpp` each run takes approximately 1-2 minutes.

## Contributing
We welcome Issues and Pull Requests! Potential improvements include:
- Supporting additional quantization formats (e.g., Q4_0).
- Optimizing `llama.cpp` build time.
- Enhancing model card parsing logic.

## License
This project is licensed under the [MIT License](LICENSE). See the root directory for details.

## Acknowledgments
- [llama.cpp](https://github.com/ggerganov/llama.cpp): Provides GGUF conversion and quantization tools.
- [Hugging Face](https://huggingface.co) and [ModelScope](https://modelscope.cn): Model hosting platforms.
