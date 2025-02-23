import os
import subprocess
import logging
import requests
from huggingface_hub import snapshot_download, hf_hub_download
from modelscope.hub.api import HubApi
from modelscope.hub.constants import Licenses, ModelVisibility

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("quantization.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def check_model_files(directory):
    """Check if model directory contains required files."""
    weight_files = [f for f in os.listdir(directory) if f.endswith(('.bin', '.safetensors'))]
    config_file = os.path.join(directory, "config.json")
    if not weight_files:
        raise FileNotFoundError(f"No weight files (.bin or .safetensors) found in {directory}")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"config.json not found in {directory}")
    logger.info(f"Found weight files: {weight_files}")
    logger.info(f"Found config.json")

def load_unsupported_models():
    """Load list of unsupported models from file."""
    unsupported_file = "unsupported_models.txt"
    unsupported_models = set()
    if os.path.exists(unsupported_file):
        with open(unsupported_file, "r") as f:
            unsupported_models = set(line.strip() for line in f if line.strip())
    return unsupported_models

def save_unsupported_model(repo_id):
    """Append unsupported model to file."""
    unsupported_file = "unsupported_models.txt"
    with open(unsupported_file, "a") as f:
        f.write(f"{repo_id}\n")

def get_hf_model_card(repo_id, token):
    """Fetch raw README.md from Hugging Face and extract content between ---."""
    try:
        # Download README.md from Hugging Face
        readme_path = hf_hub_download(repo_id=repo_id, filename="README.md", token=token)
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract content between first --- and second ---
        lines = content.splitlines()
        start_idx = None
        end_idx = None
        for i, line in enumerate(lines):
            if line.strip() == "---":
                if start_idx is None:
                    start_idx = i
                elif end_idx is None:
                    end_idx = i
                    break
        if start_idx is not None and end_idx is not None and start_idx < end_idx:
            extracted_content = "\n".join(lines[start_idx:end_idx + 1])
            return extracted_content
        else:
            logger.warning(f"No valid --- section found in README.md for {repo_id}")
            return "---\nNo YAML metadata found in README.md\n---"

    except Exception as e:
        logger.warning(f"Failed to fetch README.md for {repo_id}: {str(e)}")
        return "---\nNo model card available on Hugging Face.\n---"

def create_readme(repo_id, hf_token):
    """Generate README.md with model card and quantization info."""
    model_card = get_hf_model_card(repo_id, hf_token)
    readme_content = f"""{model_card}

## {repo_id}-Q8_0-GGUF
This model has been quantized to Q8_0 format using `llama.cpp`. The quantization process reduces the model size and accelerates inference while maintaining reasonable accuracy.

### Usage
To use this model with `llama.cpp`:
```bash
./llama.cpp/build/bin/llama-cli -m model_q8.gguf [other options]
```

### Source
- Original Model: [{repo_id}](https://huggingface.co/{repo_id})
- Quantization Tool: [llama.cpp](https://github.com/ggerganov/llama.cpp)
"""
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    return "README.md"

def main():
    try:
        repo_id = os.environ["REPO_ID"]
        hf_token = os.environ["HF_API_KEY"]

        # Check unsupported models
        unsupported_models = load_unsupported_models()
        if repo_id in unsupported_models:
            logger.error(f"Model {repo_id} is known to be unsupported for GGUF conversion. Terminating.")
            exit(1)

        # Step 1: Download model from Hugging Face
        logger.info(f"Starting download from Hugging Face for repo: {repo_id}")
        local_dir = snapshot_download(repo_id=repo_id, token=hf_token)
        logger.info(f"Model downloaded successfully to: {local_dir}")

        # Check model files
        check_model_files(local_dir)

        # Step 2: Convert to GGUF format
        logger.info("Converting model to GGUF format")
        convert_cmd = [
            "python", "llama.cpp/convert_hf_to_gguf.py", local_dir,
            "--outfile", "model.gguf"
        ]
        logger.debug(f"Running convert command: {' '.join(convert_cmd)}")
        try:
            result = subprocess.run(convert_cmd, capture_output=True, text=True, check=True, shell=False)
            logger.debug(f"Convert stdout: {result.stdout}")
            logger.debug(f"Convert stderr: {result.stderr}")
            logger.info("Model converted to GGUF: model.gguf")
        except subprocess.CalledProcessError as e:
            logger.error(f"Conversion failed for {repo_id}. Adding to unsupported list.")
            save_unsupported_model(repo_id)
            raise

        # Step 3: Quantize to Q8_0
        logger.info("Starting Q8_0 quantization with llama.cpp")
        quantize_cmd = ["llama.cpp/build/bin/llama-quantize", "model.gguf", "model_q8.gguf", "q8_0"]
        logger.debug(f"Running quantize command: {' '.join(quantize_cmd)}")
        result = subprocess.run(quantize_cmd, capture_output=True, text=True, check=True)
        logger.debug(f"Quantize stdout: {result.stdout}")
        logger.info("Model quantized to Q8_0: model_q8.gguf")

        # Step 4: Upload to ModelScope with README
        logger.info("Preparing to upload quantized model to ModelScope")
        ms_token = os.environ["MS_API_KEY"]
        ms_username = os.environ.get("MS_USERNAME")
        if not ms_username:
            raise ValueError("MS_USERNAME environment variable is not set")
        
        api = HubApi()
        logger.info("Logging into ModelScope Hub")
        api.login(ms_token)

        model_name = repo_id.split("/")[1]
        model_id = f"{ms_username}/{model_name}-Q8_0-GGUF"
        logger.info(f"Creating ModelScope model with ID: {model_id}")
        api.create_model(
            model_id=model_id,
            visibility=ModelVisibility.PUBLIC,
            license=Licenses.APACHE_V2,
            chinese_name=f"{model_name} Q8_0 GGUF 量化模型"
        )

        # Upload quantized model file
        local_file = "model_q8.gguf"
        repo_path = "model_q8.gguf"
        logger.info(f"Uploading file {local_file} to {model_id}")
        api.upload_file(
            path_or_fileobj=local_file,
            path_in_repo=repo_path,
            repo_id=model_id,
            commit_message="Upload Q8_0 quantized GGUF model from Hugging Face"
        )

        # Generate and upload README.md
        readme_file = create_readme(repo_id, hf_token)
        logger.info(f"Uploading README.md to {model_id}")
        api.upload_file(
            path_or_fileobj=readme_file,
            path_in_repo="README.md",
            repo_id=model_id,
            commit_message="Add README with model card and quantization info"
        )

        logger.info(f"Quantized model and README uploaded successfully to: {model_id}")

    except subprocess.CalledProcessError as e:
        logger.error(f"Subprocess failed with return code {e.returncode}")
        logger.error(f"Command: {' '.join(e.cmd)}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")
        raise
    except FileNotFoundError as e:
        logger.error(f"Model validation error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
