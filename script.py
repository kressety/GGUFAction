import os
import subprocess
import logging
from huggingface_hub import snapshot_download
from modelscope.hub.api import HubApi

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("quantization.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def check_model_files(directory):
    """Check if model directory contains weight files."""
    weight_files = [f for f in os.listdir(directory) if f.endswith(('.bin', '.safetensors'))]
    if not weight_files:
        raise FileNotFoundError(f"No weight files (.bin or .safetensors) found in {directory}")
    logger.info(f"Found weight files: {weight_files}")

def main():
    try:
        # Step 1: Download model from Hugging Face
        logger.info(f"Starting download from Hugging Face for repo: {os.environ['REPO_ID']}")
        repo_id = os.environ["REPO_ID"]
        hf_token = os.environ["HF_API_KEY"]
        local_dir = snapshot_download(repo_id=repo_id, token=hf_token)
        logger.info(f"Model downloaded successfully to: {local_dir}")

        # Check model files
        check_model_files(local_dir)

        # Step 2: Convert to GGUF format
        logger.info("Converting model to GGUF format")
        convert_cmd = [
            "python", "llama.cpp/convert-hf-to-gguf.py", local_dir,
            "--outfile", "model.gguf"
        ]
        logger.debug(f"Running convert command: {' '.join(convert_cmd)}")
        result = subprocess.run(convert_cmd, capture_output=True, text=True, check=True)
        logger.debug(f"Convert stdout: {result.stdout}")
        logger.info("Model converted to GGUF: model.gguf")

        # Step 3: Quantize to Q8_0
        logger.info("Starting Q8_0 quantization with llama.cpp")
        quantize_cmd = ["llama.cpp/build/bin/quantize", "model.gguf", "model_q8.gguf", "q8_0"]
        logger.debug(f"Running quantize command: {' '.join(quantize_cmd)}")
        result = subprocess.run(quantize_cmd, capture_output=True, text=True, check=True)
        logger.debug(f"Quantize stdout: {result.stdout}")
        logger.info("Model quantized to Q8_0: model_q8.gguf")

        # Step 4: Upload to ModelScope using HubApi
        logger.info("Preparing to upload quantized model to ModelScope")
        ms_token = os.environ["MS_API_KEY"]
        api = HubApi()
        logger.info("Logging into ModelScope Hub")
        api.login(ms_token)
        model_name = repo_id.split("/")[1]
        model_id = f"{os.getenv('GITHUB_ACTOR', 'user')}/quantized_{model_name}"
        logger.info(f"Pushing model to ModelScope with ID: {model_id}")
        api.push_model(
            model_id=model_id,
            model_dir=os.path.dirname("model_q8.gguf"),
            visibility=5,
            license="Apache License 2.0",
            commit_message="Upload Q8_0 quantized model from Hugging Face",
            revision="master"
        )
        logger.info(f"Quantized model uploaded successfully: {model_id}")

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
