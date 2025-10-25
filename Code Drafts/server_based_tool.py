
summarize_text="""
        A new era of Agent Interoperability AI agents offer a unique opportunity to help people be more productive by autonomously handling many daily recurring or complex tasks. Today, enterprises are increasingly building and deploying autonomous agents to help scale, automate and enhance processes throughout the workplacefrom ordering new laptops, to aiding customer service representatives, to assisting in supply chain planning. To maximize the benefits from agentic AI, it is critical for these agents to be able to collaborate in a dynamic, multi-agent ecosystem across siloed data systems and applications. Enabling agents to interoperate with each other, even if they were built by different vendors or in a different framework, will increase autonomy and multiply productivity gains, while lowering long-term costs for enterprises.
            
        """

import os
import json
import sys
import subprocess
import logging
import tempfile
import threading

# --- Configuration ---

# Get the absolute path of the directory this script is in
# e.g., /home/user/my_agent_project/
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

# Define persistent directories relative to this script
VENVS_DIR = os.path.join(SCRIPT_DIR, "venvs")
MODELS_DIR = os.path.join(SCRIPT_DIR, "Models")

# Create them if they don't exist
os.makedirs(VENVS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("HF_Inference_Tool")

# --- Private Helper ---

def _setup_venv(env_name: str, requirements: list) -> str:
    """
    Creates or updates a virtual environment and installs packages.
    Returns the path to the venv's python executable.
    """
    # if not env_name.isalnum():
    #     raise ValueError("Environment name must be alphanumeric.")

    venv_path = os.path.join(VENVS_DIR, env_name)
    
    # Check for OS-specific python executable path
    if sys.platform == "win32":
        python_executable = os.path.join(venv_path, "Scripts", "python.exe")
        pip_executable = os.path.join(venv_path, "Scripts", "pip.exe")
    else:
        python_executable = os.path.join(venv_path, "bin", "python")
        pip_executable = os.path.join(venv_path, "bin", "pip")

    if not os.path.exists(python_executable):
        logger.info(f"Creating new venv: {env_name}")
        subprocess.run([sys.executable, "-m", "venv", venv_path], check=True, capture_output=True)
    
    if requirements:
        logger.info(f"--- Starting pip install in {env_name} ---")
        try:
            # Run pip install with Popen to get live logs
            process = subprocess.Popen(
                [pip_executable, "install"] + requirements,
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
            logger.error(f"Pip install failed: {e.stderr}")
            raise
    
    return python_executable

# --- The "Tool" Function ---

def run_hf_inference_tool(
    env_name: str,
    requirements: list,
    task: str,
    model_id: str,
    prompt: str,
    model_args: dict = None
) -> dict:
    """
    A single-function tool to run HF inference in a managed venv.

    :param env_name: A name for the venv (e.g., "transformers_env")
    :param requirements: List of pip packages (e.g., '["torch", "transformers"]')
    :param task: The HF pipeline task (e.g., "summarization")
    :param model_id: The model ID from Hugging Face Hub
    :param prompt: The input text for the model
    :param model_args: (Optional) A dict of extra args for the pipeline
    :return: A dictionary with the inference result or an error
    """
    
    # This is the code for the temporary payload script.
    # It will be run *inside* the venv.
    # Added flush=True to all print statements for real-time logging.






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
    
    # --- START OF FIX ---

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
        # Do NOT pass **inference_args here
    )
    
    print("Pipeline loaded. Running inference...", file=sys.stderr, flush=True)
    
    # 3. Pass inference arguments to the call itself
    result = model_pipeline(prompt, **inference_args)
    
    # --- END OF FIX ---
    
    print("Inference complete.", file=sys.stderr, flush=True)

    # Final success output to stdout
    
except Exception as e:
    print(json.dumps({"status": "error", "message": str(e)}), file=sys.stderr, flush=True)
    sys.exit(1)    
    """




    try:
        # 1. Set up the venv and get the path to its Python
        python_executable = _setup_venv(env_name, requirements)
        
        # 2. Set up environment for the payload
        payload_env = os.environ.copy()
        payload_env["HF_HOME"] = MODELS_DIR  # Use our custom ./Models directory
        payload_env["TASK"] = task
        payload_env["MODEL_ID"] = model_id
        payload_env["PROMPT"] = prompt
        if model_args:
            payload_env["MODEL_ARGS"] = json.dumps(model_args)

        logger.info(f"Running inference for model {model_id} in venv {env_name}")
        
        # 3. Create a temporary file for the payload script
        # Using 'delete=False' so we control when it's deleted
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
            f.write(payload_code)
            payload_script_path = f.name
        
        # 4. Execute the payload script *inside the venv* using Popen
        process = subprocess.Popen(
            [python_executable, payload_script_path],
            env=payload_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            encoding='utf-8',
            errors='replace'
        )
        
        # 5. Use threads to read stdout and stderr in real-time
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
            return_code = process.wait(timeout=300) # 5-min timeout for inference
        except subprocess.TimeoutExpired:
            process.kill()
            logger.error("Worker process timed out and was killed.")
            return {"status": "error", "message": "Inference process timed out."}
            
        stdout_thread.join()
        stderr_thread.join()
        
        logger.info(f"--- Worker finished with code {return_code} ---")

        # 6. Clean up the temporary script
        os.remove(payload_script_path)

        # 7. Return the result
        if return_code == 0:
            logger.info("Inference successful.")
            if not stdout_lines:
                return {"status": "error", "message": "Worker produced no output."}
            # Re-join in case JSON was split across lines
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

    except Exception as e:
        logger.error(f"Orchestrator failed: {e}")
        # Clean up temp file on error, if it still exists
        if 'payload_script_path' in locals() and os.path.exists(payload_script_path):
            os.remove(payload_script_path)
        return {"status": "error", "message": f"Orchestrator Error: {str(e)}"}


# --- Example of how to use this tool ---
if __name__ == "__main__":
    
    print("--- Running Summarization Task ---")
    
    summarization_result = run_hf_inference_tool(
        env_name=".venv",
        requirements=["torch", "transformers[sentencepiece]"], # sentencepiece is needed for T5
        task="text2text-generation", # T5 uses this task for summarization
        model_id="t5-small", # This is much smaller than BART
        prompt=f"summarize: {summarize_text}" # Add the "summarize:" prefix
    )
    
    print(json.dumps(summarization_result, indent=2))
    
    # print("\n--- Running Sentiment Task (in same env) ---")
    
    # sentiment_result = run_hf_inference_tool(
    #     env_name="transformers_env",  # Will use the *existing* venv, no re-install
    #     requirements=[], # No new requirements needed
    #     task="sentiment-analysis",
    #     model_id="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    #     prompt="I love this hackathon, this is a fantastic idea!"
    # )
    
    # print(json.dumps(sentiment_result, indent=2))

