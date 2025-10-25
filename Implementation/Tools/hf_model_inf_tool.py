import os
import json
import sys
import subprocess
import logging
import tempfile
import threading

# --- Configuration ---

# Set up logging for this tool
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("HF_Inference_Tool")

# --- START: PATH CORRECTION ---
# Get the absolute path of the directory this script is in
# e.g., /Users/ajitesh/Desktop/Cypherpunk/
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

# The project root *is* the script directory in this case.
PROJECT_ROOT = SCRIPT_DIR

# Define paths relative to the project root
VENV_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, ".venv"))
MODEL_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "./Model"))
# --- END: PATH CORRECTION ---


# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Find the Python and Pip executables *inside* the target venv
if sys.platform == "win32":
    PYTHON_EXEC = os.path.join(VENV_DIR, "Scripts", "python.exe")
    PIP_EXEC = os.path.join(VENV_DIR, "Scripts", "pip.exe")
else:
    # Matching your log files
    PYTHON_EXEC = os.path.join(VENV_DIR, "bin", "python3")
    PIP_EXEC = os.path.join(VENV_DIR, "bin", "pip3")



def _install_requirements(requirements: list):
    """
    Installs packages into the specific .venv environment.
    """
    if not requirements:
        logger.info("No new requirements to install.")
        return

    if not os.path.exists(PIP_EXEC):
        logger.error(f"Target pip executable not found at: {PIP_EXEC}")
        logger.error(f"Please ensure the virtual environment exists at: {VENV_DIR}")
        raise FileNotFoundError(f"Pip not found at {PIP_EXEC}")

    logger.info(f"--- Starting pip install in {VENV_DIR} ---")
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
            raise subprocess.CalledProcessError(return_code, "pip install")
        
        logger.info(f"--- Pip install finished ---")

    except subprocess.CalledProcessError as e:
        logger.error(f"Pip install failed: {e}")
        raise


def run_hf_inference(
    model_id: str,
    task: str,
    prompt: str,
    requirements: list = None,
    model_args: dict = None
) -> dict:
    """
    A single-function tool to run HF inference in the shared venv.

    :param model_id: The model ID from Hugging Face Hub
    :param task: The HF pipeline task (e.g., "sentiment-analysis")
    :param prompt: The input text for the model
    :param requirements: (Optional) List of pip packages (e.g., ["torch", "transformers"])
    :param model_args: (Optional) A dict of extra args for the pipeline call
    :return: A dictionary with the inference result or an error
    """
    
    # This payload code will be run *inside* the .venv environment
    payload_code = """
import os, json, sys, torch
from transformers import pipeline

print("Payload starting...", file=sys.stderr, flush=True)

try:
    task = os.environ.get("TASK")
    model_id = os.environ.get("MODEL_ID")
    prompt = os.environ.get("PROMPT")
    cache_dir = os.environ.get("HF_HOME")

    device = 0 if torch.cuda.is_available() else -1
    print(f"Device set to use {'cuda:0' if device == 0 else 'cpu'}", file=sys.stderr, flush=True)

    # 1. Arguments for the inference call (e.g., max_length)
    inference_args = {}
    if os.environ.get("MODEL_ARGS"):
        inference_args = json.loads(os.environ.get("MODEL_ARGS"))

    # 2. Arguments for the model loader (from_pretrained)
    loader_model_kwargs = {
        "cache_dir": cache_dir
    }

    print(f"Loading pipeline for task: {task}, model: {model_id}", file=sys.stderr, flush=True)

    model_pipeline = pipeline(
        task=task,
        model=model_id,
        device=device,
        model_kwargs=loader_model_kwargs # Pass cache_dir here
    )
    
    print("Pipeline loaded. Running inference...", file=sys.stderr, flush=True)
    
    # 3. Pass inference arguments to the call itself
    result = model_pipeline(prompt, **inference_args)
    
    print("Inference complete.", file=sys.stderr, flush=True)

    # Final success output to stdout
    print(json.dumps({"status": "success", "output": result}), flush=True)

except Exception as e:
    # Final error output to stderr
    print(json.dumps({"status": "error", "message": f"Payload Error: {e}"}), file=sys.stderr, flush=True)
    sys.exit(1)
"""

    # Define process and payload_script_path as None initially
    # so they exist in the 'finally' and 'except' blocks
    payload_script_path = None 
    process = None

    try:
        # 1. Check if the target Python exists before doing anything
        if not os.path.exists(PYTHON_EXEC):
            logger.error(f"Target Python executable not found at: {PYTHON_EXEC}")
            logger.error(f"Please create the virtual environment at: {VENV_DIR}")
            raise FileNotFoundError(f"Python not found at {PYTHON_EXEC}")

        # 2. Install dependencies *into* the target venv
        if requirements:
            _install_requirements(requirements)
        
        # 3. Set up environment for the payload
        payload_env = os.environ.copy()
        payload_env["HF_HOME"] = MODEL_DIR # Cache models here
        payload_env["TASK"] = task
        payload_env["MODEL_ID"] = model_id
        payload_env["PROMPT"] = prompt
        if model_args:
            payload_env["MODEL_ARGS"] = json.dumps(model_args)

        logger.info(f"Running inference for model {model_id} using {PYTHON_EXEC}")
        
        # 4. Create a temporary file for the payload script
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
            f.write(payload_code)
            payload_script_path = f.name
        
        # 5. Execute the payload script *inside the venv* using Popen
        process = subprocess.Popen(
            [PYTHON_EXEC, payload_script_path],
            env=payload_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            encoding='utf-8',
            errors='replace'
        )
        
        # 6. Use threads to read stdout and stderr in real-time
        stdout_lines = []
        stderr_lines = []

        def read_stream(stream, line_list, log_prefix):
            """Helper function to read a stream and log lines."""
            try:
                for line in iter(stream.readline, ''):
                    line = line.strip()
                    if line:
                        logger.info(f"[{log_prefix}] {line}")
                        line_list.append(line)
            except Exception as e:
                logger.warning(f"Error reading stream {log_prefix}: {e}")
            finally:
                stream.close()

        stdout_thread = threading.Thread(target=read_stream, args=(process.stdout, stdout_lines, "worker_out"))
        stderr_thread = threading.Thread(target=read_stream, args=(process.stderr, stderr_lines, "worker_err"))

        stdout_thread.start()
        stderr_thread.start()

        # Wait for process and threads to finish
        try:
            return_code = process.wait(timeout=300) # 5-min timeout
        except subprocess.TimeoutExpired:
            process.kill()
            logger.error("Worker process timed out and was killed.")
            return {"status": "error", "message": "Inference process timed out."}
            
        stdout_thread.join()
        stderr_thread.join()
        
        logger.info(f"--- Worker finished with code {return_code} ---")

        # 7. Parse and return the result
        if return_code == 0:
            logger.info("Inference successful.")
            if not stdout_lines:
                return {"status": "error", "message": "Worker produced no output."}
            full_stdout = "".join(stdout_lines) 
            return json.loads(full_stdout)
        else:
            logger.error(f"Inference failed. Full stderr: {' '.join(stderr_lines)}")
            if not stderr_lines:
                return {"status": "error", "message": f"Worker failed with code {return_code} and no error output."}
            try:
                # The last line of stderr should be our JSON error
                return json.loads(stderr_lines[-1])
            except (json.JSONDecodeError, IndexError):
                # Fallback if parsing fails
                return {"status": "error", "message": " ".join(stderr_lines)}

    except KeyboardInterrupt:
        logger.warning("--- Keyboard interrupt detected! ---")
        if process:
            logger.info("Terminating worker subprocess...")
            process.kill()
            logger.info("Worker terminated.")
        return {"status": "error", "message": "Orchestrator interrupted by user."}

    except Exception as e:
        logger.error(f"Orchestrator failed: {e}", exc_info=True)
        return {"status": "error", "message": f"Orchestrator Error: {str(e)}"}
    
    finally:
        # 8. Clean up the temporary script
        if payload_script_path and os.path.exists(payload_script_path):
            os.remove(payload_script_path)
            logger.debug(f"Cleaned up temp script: {payload_script_path}")


def cleanup_hf_cache():
    """
    Runs the huggingface-hub CLI to scan and delete corrupted files.
    """
    logger.info("--- Running Hugging Face Cache Cleanup ---")
    
    # 1. Ensure huggingface_hub is installed in the venv
    _install_requirements(["huggingface-hub"])
    
    # 2. Define the command to run
    # This will run: .venv/bin/python3 -m huggingface_hub.commands.scan_cache -D --cache-dir Model
    command = [
        PYTHON_EXEC,
        "-m", "huggingface_hub.commands.scan_cache",
        "-D",  # This flag means "auto-delete" corrupted files
        "--cache-dir", MODEL_DIR
    ]
    
    logger.info(f"Scanning cache at: {MODEL_DIR}")
    
    try:
        # Run scan and log output in real-time
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding='utf-8',
            errors='replace',
            env=os.environ.copy() # Pass environment
        )
        
        for line in iter(process.stdout.readline, ''):
            logger.info(f"[hf-cleanup] {line.strip()}")
        
        process.stdout.close()
        return_code = process.wait()
        
        if return_code == 0:
            logger.info("--- Cache cleanup finished successfully ---")
        else:
            logger.error(f"Cache cleanup failed with code {return_code}")
            
    except Exception as e:
        logger.error(f"Cache cleanup tool failed to run: {e}", exc_info=True)


# --- Example of how to use this tool ---
if __name__ == "__main__":
    
    # Allow running: python3 hf_tool.py --cleanup
    if len(sys.argv) > 1 and sys.argv[1] == "--cleanup":
        cleanup_hf_cache()
    
    else:
        # NOTE: Before running this, you must create the virtual environment:
        # python3 -m venv .venv
        
        print("--- Running Sentiment Task (Small Model) ---")
        
        # We will use a very small, fast-downloading model for this test
        sentiment_result = run_hf_inference(
            model_id="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
            task="sentiment-analysis",
            prompt="This is a great tool, it's fast and reliable!",
            requirements=["torch", "transformers", "huggingface-hub"] # This will install into .venv
        )
        
        print("\n--- RESULT 1 ---")
        print(json.dumps(sentiment_result, indent=2))
        
        
        print("\n--- Running Second Sentiment Task (Should be fast) ---")
        
        # This time, we pass 'None' for requirements because they are already installed.
        # The model is already cached in Model, so it will load instantly.
        sentiment_result_2 = run_hf_inference(
            model_id="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
            task="sentiment-analysis",
            prompt="I hate this, it's slow and buggy.",
            requirements=None # Skip pip install
        )
        
        print("\n--- RESULT 2 ---")
        print(json.dumps(sentiment_result_2, indent=2))

