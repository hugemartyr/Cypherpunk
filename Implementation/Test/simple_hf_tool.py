import os
import json
import sys
import subprocess
import logging
import torch
import numpy as np

# --- New Imports for Image/Audio Handling ---
try:
    from PIL import Image
    import scipy.io.wavfile as wavfile
except ImportError:
    print("Missing PIL or Scipy. The script will try to install them.")

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("HF_In_Process_Tool")

# --- Path Configuration ---
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = SCRIPT_DIR
MODEL_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "Model"))
OUTPUT_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "Output"))

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

PYTHON_EXEC = sys.executable
PIP_EXEC = sys.executable.replace("python3", "pip3").replace("python", "pip")

_model_cache = {}


def install_requirements(requirements: list):
    """
    Installs packages into the *currently running* Python environment.
    """
    if not requirements:
        logger.info("No new requirements to install.")
        return True

    # Add base requirements for all tasks
    base_reqs = ["torch", "transformers", "huggingface-hub"]
    # Add requirements for new tasks
    media_reqs = ["Pillow", "scipy", "diffusers", "accelerate", "audiocraft", "einops"]
    
    full_requirements = list(set(base_reqs + media_reqs + requirements))
    
    logger.info(f"--- Starting pip install in current environment ---")
    try:
        process = subprocess.Popen(
            [PIP_EXEC, "install"] + full_requirements,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding='utf-8',
            errors='replace'
        )
        
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
    Handles text, image, and audio outputs.
    """
    
    global pipeline, _model_cache, Image, wavfile
    
    if model_args is None:
        model_args = {}
    if requirements is None:
        requirements = []

    try:
        # --- 1. Handle Dependencies ---
        try:
            from transformers import pipeline
            if 'Image' not in globals():
                from PIL import Image
            if 'wavfile' not in globals():
                import scipy.io.wavfile as wavfile

        except ImportError:
            logger.warning("Core modules not found. Attempting installation...")
            if not install_requirements(requirements):
                return {"status": "error", "message": "Failed to install required packages."}
            
            try:
                from transformers import pipeline
                from PIL import Image
                import scipy.io.wavfile as wavfile
                logger.info("Successfully imported modules after installation.")
            except ImportError as e:
                logger.error(f"Failed to import modules even after install: {e}")
                return {"status": "error", "message": str(e)}

        # --- 2. Load Model (from memory or disk) ---
        cache_key = f"{model_id}_{task}"
        
        if cache_key in _model_cache:
            logger.info(f"Loading model '{model_id}' from memory cache...")
            model_pipeline = _model_cache[cache_key]
        else:
            logger.info(f"Loading model '{model_id}' from disk cache... (This may take time)")
            device = 0 if torch.cuda.is_available() else -1
            logger.info(f"Device set to use {'cuda:0' if device == 0 else 'cpu'}")

            loader_model_kwargs = {
                "cache_dir": MODEL_DIR,
                **model_args  # Pass model_args like trust_remote_code
            }

            model_pipeline = pipeline(
                task=task,
                model=model_id,
                device=device,
                torch_dtype=torch.float16 if task in ["text-to-image", "text-generation"] and device == 0 else torch.float32,
                **loader_model_kwargs
            )
            
            _model_cache[cache_key] = model_pipeline
            logger.info(f"Model '{model_id}' loaded and cached in memory.")

        # --- 3. Run Inference ---
        logger.info(f"Running inference for model {model_id}...")
        
        # Special handling for pipeline arguments
        inference_args = {}
        if task == "text-to-image":
            # SDXL-Turbo runs best with fewer steps
            if "turbo" in model_id:
                inference_args = {"num_inference_steps": 2, "guidance_scale": 0.0}
            else:
                inference_args = {"num_inference_steps": 10}
        elif task == "text-generation":
            inference_args = {"max_new_tokens": 50, "do_sample": True, "temperature": 0.7}
        elif task == "summarization":
            inference_args = {"min_length": 10, "max_length": 50}

        result = model_pipeline(prompt, **inference_args)
        
        logger.info("Inference complete. Processing output...")

        # --- 4. Process Output (Handle non-JSON types) ---
        output_data = None
        
        if task == "text-to-image":
            # Result is a list of PIL Images
            image = result.images[0]
            output_filename = f"{model_id.replace('/', '_')}_output.png"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            image.save(output_path)
            output_data = f"Image saved to: {output_path}"
        
        elif task == "text-to-audio":
            # Result is a dict {'audio': np.array, 'sampling_rate': int}
            audio_array = result["audio"][0]
            sampling_rate = result["sampling_rate"]
            output_filename = f"{model_id.replace('/', '_')}_output.wav"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            # Normalize to 16-bit PCM
            audio_int16 = (audio_array * 32767).astype(np.int16)
            wavfile.write(output_path, rate=sampling_rate, data=audio_int16)
            output_data = f"Audio saved to: {output_path}"
            
        else:
            # Default: Assume JSON serializable (text, sentiment, etc.)
            output_data = result

        return {"status": "success", "output": output_data}

    except Exception as e:
        logger.error(f"In-process inference failed: {e}", exc_info=True)
        return {"status": "error", "message": f"Inference Error: {str(e)}"}


# --- Main execution block to test all models ---
if __name__ == "__main__":
    
    # Define all the test jobs
    test_jobs = [
        {
            "name": "Sentiment Analysis (Lightweight)",
            "model_id": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
            "task": "sentiment-analysis",
            "prompt": "This is a great tool, it's fast and reliable!",
            "requirements": [], # Base reqs are handled
            "model_args": {}
        },
        {
            "name": "Text Summarization (Lightweight)",
            "model_id": "sshleifer/distilbart-cnn-12-6",
            "task": "summarization",
            "prompt": (
                "The James Webb Space Telescope (JWST) is a space telescope "
                "designed primarily to conduct infrared astronomy. As the largest optical "
                "telescope in space, its high resolution and sensitivity allow it to "
                "view objects too old, distant, or faint for the Hubble Space Telescope."
            ),
            "requirements": [],
            "model_args": {}
        },
        {
            "name": "Text Generation (Lightweight)",
            "model_id": "microsoft/Phi-3-mini-4k-instruct",
            "task": "text-generation",
            "prompt": "<|user|>\nWhat is the capital of France?<|end|>\n<|assistant|>",
            "requirements": ["einops"], # Phi-3 needs 'einops'
            "model_args": {"trust_remote_code": True} # Phi-3 requires this
        },
        {
            "name": "Image Generation (Fast/Low-Quality)",
            "model_id": "segmind/tiny-sd",
            "task": "text-to-image",
            "prompt": "A cute cat wearing a tiny wizard hat",
            "requirements": ["diffusers", "accelerate"],
            "model_args": {}
        },
        {
            "name": "Image Generation (High-Quality/Fast)",
            "model_id": "stabilityai/sdxl-turbo",
            "task": "text-to-image",
            "prompt": "A photorealistic astronaut riding a horse on Mars, cinematic lighting",
            "requirements": ["diffusers", "accelerate"],
            "model_args": {}
        },
        {
            "name": "Sound Generation (Music)",
            "model_id": "facebook/musicgen-small",
            "task": "text-to-audio",
            "prompt": "A fast-paced electronic beat with a catchy synth melody",
            "requirements": ["audiocraft"], # musicgen needs this
            "model_args": {}
        }
    ]
    
    # Run all jobs
    for i, job in enumerate(test_jobs):
        print(f"\n--- Running Test {i+1}: {job['name']} ---")
        
        result = run_hf_inference_in_process(
            model_id=job["model_id"],
            task=job["task"],
            prompt=job["prompt"],
            requirements=job["requirements"],
            model_args=job["model_args"]
        )
        
        print(f"\n--- RESULT {i+1} ---")
        print(json.dumps(result, indent=2))
        
        if result['status'] == 'success' and 'saved to' in str(result.get('output')):
            print(f"-> Check your '{OUTPUT_DIR}' folder for the generated file!")

    print("\n--- All tests complete. ---")