import os
import json
import sys
import subprocess
import logging
import torch

# --- Configuration ---

# Set up logging for this tool
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("HF_In_Process_Tool")

# --- Path Configuration ---
# Get the absolute path of the directory this script is in
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
# The project root *is* the script directory in this case.
PROJECT_ROOT = SCRIPT_DIR
# Define paths relative to the project root
MODEL_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "../../Model"))
# --- End Path Configuration ---

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Use the currently running Python's executable for pip
PYTHON_EXEC = sys.executable
PIP_EXEC = sys.executable.replace("python3", "pip3").replace("python", "pip") # Simple replacement for pip

# Store loaded models in memory to avoid reloading
_model_cache = {}


def install_requirements(requirements: list):
    """
    Installs packages into the *currently running* Python environment.
    """
    if not requirements:
        logger.info("No new requirements to install.")
        return True

    logger.info(f"--- Starting pip install in current environment ---")
    try:
        # Run pip install with Popen to get live logs
        process = subprocess.Popen(
            [PIP_EXEC, "install"] + requirements,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # Combine stdout and stderr
            text=True,
            bufsize=1, # Line-buffered
            encoding='utf-8',
            errors='replace'
        )
        
        # Real-time logging for pip
        for line in iter(process.stdout.readline, ''):
            logger.info(f"[pip] {line.strip()}")
        
        process.stdout.close()
        return_code = process.wait()
        
        if return_code != 0:
            logger.error(f"Pip install failed with code {return_code}")
            return False
        
        logger.info(f"--- Pip install finished ---")
        return True

    except Exception as e:
        logger.error(f"Pip install failed: {e}")
        return False


def run_hf_inference_in_process(
    model_id: str,
    task: str,
    prompt: str,
    requirements: list = None,
    model_args: dict = None
) -> dict:
    """
    A single-function tool to run HF inference *in the main process*.
    
    WARNING: This is a BLOCKING function. It will freeze your agent
    until the model is loaded and inference is complete.
    
    WARNING: A crash in this function (e.g., Out-of-Memory)
    WILL crash your entire agent.
    """
    
    global pipeline, _model_cache
    
    if model_args is None:
        model_args = {}

    try:
        # --- 1. Handle Dependencies ---
        try:
            from transformers import pipeline
        except ImportError:
            logger.warning("Module 'transformers' not found. Attempting installation...")
            if not requirements:
                requirements = ["torch", "transformers", "huggingface-hub"]
            
            if not install_requirements(requirements):
                return {"status": "error", "message": "Failed to install required packages."}
            
            # After install, retry import
            try:
                from transformers import pipeline
                logger.info("Successfully imported 'transformers' after installation.")
            except ImportError as e:
                logger.error(f"Failed to import 'transformers' even after install: {e}")
                return {"status": "error", "message": str(e)}

        # --- 2. Load Model (from memory or disk) ---
        
        # Use a unique key for the cache
        cache_key = f"{model_id}_{task}"
        
        if cache_key in _model_cache:
            logger.info(f"Loading model '{model_id}' from memory cache...")
            model_pipeline = _model_cache[cache_key]
        else:
            logger.info(f"Loading model '{model_id}' from disk cache... (This may take time)")
            device = 0 if torch.cuda.is_available() else -1
            logger.info(f"Device set to use {'cuda:0' if device == 0 else 'cpu'}")

            # Arguments for the model loader (from_pretrained)
            loader_model_kwargs = {
                "cache_dir": MODEL_DIR # Use our ./Model directory
            }

            model_pipeline = pipeline(
                task=task,
                model=model_id,
                device=device,
                model_kwargs=loader_model_kwargs
            )
            
            # Store in memory for next time
            _model_cache[cache_key] = model_pipeline
            logger.info(f"Model '{model_id}' loaded and cached in memory.")

        # --- 3. Run Inference ---
        logger.info(f"Running inference for model {model_id}...")
        
        result = model_pipeline(prompt, **model_args)
        
        logger.info("Inference complete.")
        return {"status": "success", "output": result}

    except Exception as e:
        logger.error(f"In-process inference failed: {e}", exc_info=True)
        return {"status": "error", "message": f"Inference Error: {str(e)}"}


# --- Example of how to use this tool ---
if __name__ == "__main__":
    
    print("--- Running Sentiment Task (In-Process) ---")
    
    sentiment_result = run_hf_inference_in_process(
        model_id="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        task="sentiment-analysis",
        prompt="This is a great tool, it's fast and reliable!",
        requirements=["torch", "transformers", "huggingface-hub","pyautogui"] # Example requirements
    )
    
    print("\n--- RESULT 1 ---")
    print(json.dumps(sentiment_result, indent=2))
    
    
    # print("\n--- Running Second Sentiment Task (Should be fast) ---")
    
    # # This time, it should load from the *in-memory* cache, making it instant.
    # sentiment_result_2 = run_hf_inference_in_process(
    #     model_id="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    #     task="sentiment-analysis",
    #     prompt="I hate this, it's slow and buggy.",
    #     requirements=None # No need to check requirements again
    # )
    
    # print("\n--- RESULT 2 ---")
    # print(json.dumps(sentiment_result_2, indent=2))
