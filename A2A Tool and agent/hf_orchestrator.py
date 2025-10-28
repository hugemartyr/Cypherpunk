import os
import sys
import subprocess
import json
from typing import Dict, Any
import time

# Define the directory for persistent models
MODELS_DIR = "Models"

# --- Runner Script Generation Functions ---

def generate_transient_runner_content() -> str:
    """
    Generates the runner script content for Path A (single-use, cleanup).
    """
    return """
import sys
import json
from transformers import pipeline
import torch
import warnings
warnings.filterwarnings('ignore')

def execute_hf_model(model_id, prompt, task_type):
    try:
        # Determine device
        device = 0 if torch.cuda.is_available() else -1
        
        # Load pipeline based on task type
        if task_type == "auto":
            pipe = pipeline(model=model_id, device=device, trust_remote_code=True)
        else:
            pipe = pipeline(task_type, model=model_id, device=device, trust_remote_code=True)
        
        # Execute model
        if task_type == "text-generation" or (task_type == "auto" and any(keyword in model_id.lower() for keyword in ["gpt", "llama"])):
            result = pipe(
                prompt, 
                max_new_tokens=100,
                return_full_text=False,
                do_sample=True,
                temperature=0.7
            )
        else:
            result = pipe(prompt)
        
        # Parse result
        output = ""
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict):
                output_key = next((k for k in ['generated_text', 'summary_text', 'translation_text', 'answer'] if k in result[0]), 'label')
                if output_key == 'label':
                    output = f"Label: {result[0]['label']}, Score: {result[0].get('score', 'N/A')}"
                else:
                    output = result[0][output_key]
            else:
                output = str(result[0])
        else:
            output = str(result)
        
        # Return as JSON
        response = {
            "success": True,
            "output": output,
            "model_id": model_id
        }
        print(json.dumps(response))
        
    except Exception as e:
        error_response = {
            "success": False,
            "error": str(e),
            "model_id": model_id
        }
        print(json.dumps(error_response))
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(json.dumps({"success": False, "error": "Missing arguments"}))
        sys.exit(1)
    
    model_id = sys.argv[1]
    prompt = sys.argv[2]
    task_type = sys.argv[3]
    
    execute_hf_model(model_id, prompt, task_type)
"""

def generate_persistent_runner_content(model_name_safe: str) -> str:
    """
    Generates the runner script content for Path B (loads once, runs perpetually, and saves output).
    This script performs the initial inference and then stays alive.
    """
    return f"""
import sys
import json
import os
import time
from transformers import pipeline
import torch
import warnings
warnings.filterwarnings('ignore')

# Configuration from the main orchestrator
MODEL_ID = sys.argv[1]
INITIAL_PROMPT = sys.argv[2]
TASK_TYPE = sys.argv[3]
LOG_FILE = sys.argv[4]

def log(message):
    with open(LOG_FILE, "a") as f:
        f.write(f"[{{time.strftime('%Y-%m-%d %H:%M:%S')}}] {{message}}\\n")
    # Also print to stdout/stderr for real-time monitoring
    if "ERROR" in message or "FATAL" in message:
        print(message, file=sys.stderr)
    else:
        print(message, file=sys.stdout)

def initialize_and_run():
    log(f"--- Persistent Runner Started for {{MODEL_ID}} ---")
    log("Status: Initializing model...")
    
    pipe = None
    try:
        # Determine device
        device = 0 if torch.cuda.is_available() or torch.backends.mps.is_available() else -1
        device_name = "GPU" if torch.cuda.is_available() else ("MPS" if torch.backends.mps.is_available() else "CPU")
        log(f"Loading model on device: {{device_name}}")

        # Load pipeline based on task type
        if TASK_TYPE == "auto":
            pipe = pipeline(model=MODEL_ID, device=device, trust_remote_code=True)
        else:
            pipe = pipeline(TASK_TYPE, model=MODEL_ID, device=device, trust_remote_code=True)

        log("Status: Model loaded successfully.")
        
        # --- 1. Execute Initial Prompt ---
        log(f"Executing initial prompt: '{{INITIAL_PROMPT[:50]}}...'")
        
        if TASK_TYPE == "text-generation" or (TASK_TYPE == "auto" and any(keyword in MODEL_ID.lower() for keyword in ["gpt", "llama"])):
            result = pipe(
                INITIAL_PROMPT, 
                max_new_tokens=100,
                return_full_text=False,
                do_sample=True,
                temperature=0.7
            )
        else:
            result = pipe(INITIAL_PROMPT)

        # Parse initial result
        output = ""
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict):
                output_key = next((k for k in ['generated_text', 'summary_text', 'translation_text', 'answer'] if k in result[0]), 'label')
                if output_key == 'label':
                    output = f"Label: {{result[0]['label']}}, Score: {{result[0].get('score', 'N/A')}}"
                else:
                    output = result[0][output_key]
            else:
                output = str(result[0])
        else:
            output = str(result)
        
        log("Initial Inference Complete. Output saved to log file.")
        
        # --- 2. Enter Interactive/Persistent Loop (Simulated) ---
        log("Status: READY. Awaiting new prompts on STDIN.")
        log("To send a new prompt, pipe text to this process or run a new script.")
        
        # Keep process alive until manually stopped (Ctrl+C or kill)
        while True:
            # Simulate waiting for a new prompt
            new_prompt = sys.stdin.readline().strip() 
            if new_prompt:
                log(f"Processing new prompt: '{{new_prompt[:50]}}...'")
                # Perform inference on new prompt (re-using the pipe)
                if TASK_TYPE == "text-generation" or (TASK_TYPE == "auto" and any(keyword in MODEL_ID.lower() for keyword in ["gpt", "llama"])):
                    new_result = pipe(
                        new_prompt, 
                        max_new_tokens=50,
                        return_full_text=False,
                        do_sample=True,
                        temperature=0.7
                    )
                else:
                    new_result = pipe(new_prompt)
                
                # Simple logging of new result for persistent mode
                if isinstance(new_result, list) and isinstance(new_result[0], dict) and 'generated_text' in new_result[0]:
                    new_output = new_result[0]['generated_text']
                else:
                    new_output = str(new_result)
                log(f"New Response: '{{new_output[:100]}}...'")
            
            time.sleep(1) # Sleep briefly to prevent high CPU usage in the loop

    except Exception as e:
        log(f"[FATAL ERROR] {{str(e)}}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("[FATAL ERROR] Persistent runner missing required arguments.")
        sys.exit(1)
    
    initialize_and_run()
"""


# --- Path A: Transient Model Execution ---

def run_transient_model(model_id: str, prompt: str, task_type: str = "auto") -> Dict[str, Any]:
    """
    Executes Path A: Generates, runs, and cleans up the temporary runner script.
    """
    filename = "hf_transient_runner.py"
    
    try:
        # 1. Generate and Write the script
        script_content = generate_transient_runner_content()
        with open(filename, "w", encoding="utf-8") as f:
            f.write(script_content)
        
        print(f"\n[AGENT A] Running transient model: {model_id}. This is a single-use inference.")
        print(f"[AGENT A] Prompt: {prompt[:50]}{'...' if len(prompt) > 50 else ''}\n")
        
        # 2. Execute the script
        command = [sys.executable, filename, model_id, prompt, task_type]
        
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout for download/inference
        )
        
        # 3. Parse the output
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        
        if result.returncode != 0 and not stdout:
             # Handle execution errors (e.g., Python syntax errors, missing packages)
            error_msg = f"Runner failed (Code {result.returncode}). See debug info."
            return {
                "success": False,
                "error": error_msg,
                "stderr": stderr
            }

        # Parse JSON response from runner's stdout
        try:
            response_data = json.loads(stdout)
            
            return {
                "success": response_data.get("success", False),
                "model_id": model_id,
                "prompt": prompt,
                "model_response": response_data.get("output", response_data.get("error", "No output or specific error found.")),
                "stderr": stderr
            }
                
        except json.JSONDecodeError:
            error_msg = f"Failed to parse model output JSON. Raw output: '{stdout[:100]}...'"
            return {
                "success": False,
                "error": error_msg,
                "stderr": stderr
            }

    except subprocess.TimeoutExpired:
        error_msg = "Execution timed out (10 minutes). Model may be too large."
        return {"success": False, "error": error_msg}

    except Exception as e:
        error_msg = f"Fatal error during orchestration: {str(e)}"
        return {"success": False, "error": error_msg}

    finally:
        # 4. Clean up the file
        if os.path.exists(filename):
            os.remove(filename)
            print(f"[AGENT A] Cleaned up: {filename}")


# --- Path B: Persistent Model Execution ---

def run_persistent_model(model_id: str, prompt: str, task_type: str = "auto") -> Dict[str, Any]:
    """
    Executes Path B: Creates directory, generates persistent script, and runs it in the background.
    """
    # Create the Models directory if it doesn't exist
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        print(f"[AGENT B] Created directory: '{MODELS_DIR}'")

    # Create a safe name for the runner and log files
    model_name_safe = model_id.replace('/', '__').replace('-', '_')
    runner_filename = os.path.join(MODELS_DIR, f"{model_name_safe}_runner.py")
    log_filename = os.path.join(MODELS_DIR, f"{model_name_safe}_log.txt")

    try:
        # 1. Generate and Write the persistent script
        script_content = generate_persistent_runner_content(model_name_safe)
        with open(runner_filename, "w", encoding="utf-8") as f:
            f.write(script_content)
        
        # 2. Clear previous log file
        if os.path.exists(log_filename):
            os.remove(log_filename)
        
        print(f"\n[AGENT B] Starting persistent model: {model_id}...")
        print(f"[AGENT B] Runner saved to: {runner_filename}")
        print(f"[AGENT B] Initial Prompt: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")

        # 3. Execute the script in the background using Popen
        command = [sys.executable, runner_filename, model_id, prompt, task_type, log_filename]

        # Use Popen to launch and detach the process (or simulate detaching)
        # Note on cross-platform backgrounding: 
        # Using stdout/stderr=subprocess.DEVNULL helps detach the child's I/O 
        # but the process remains a child of the current script's shell.
        # For a true background process that survives the parent, OS-specific tricks 
        # (like nohup or screen/tmux) or specific shell calls are needed.
        process = subprocess.Popen(
            command,
            cwd=MODELS_DIR, # Run from inside the Models directory
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        # Give the model a moment to start loading and log the initial status
        print("\n[AGENT B] Waiting 5 seconds for model initialization logs...")
        time.sleep(5) 
        
        # Read initial log output to check for success/failure
        # In a real setup, you'd monitor the process status or log file.
        initial_log_check = ""
        if os.path.exists(log_filename):
            with open(log_filename, 'r') as f:
                initial_log_check = f.read()

        # Check if the process started successfully
        if process.poll() is not None and process.poll() != 0:
            return {
                "success": False,
                "error": f"Process exited immediately with code {process.poll()}. Check {log_filename} for errors.",
                "log_file": log_filename
            }

        # Success message with instructions
        return {
            "success": True,
            "model_id": model_id,
            "runner_path": runner_filename,
            "log_path": log_filename,
            "pid": process.pid,
            "status_check": initial_log_check
        }

    except Exception as e:
        error_msg = f"Fatal error during persistent orchestration: {str(e)}"
        return {"success": False, "error": error_msg}


# --- User Interface and Main Entry Point ---

def main():
    print("=" * 80)
    print("        HuggingFace Model Orchestrator: Select Execution Path")
    print("=" * 80)
    print("1. [Path A: Transient Run] - Single inference, fast, cleans up after running.")
    print("2. [Path B: Persistent Run] - Loads model once, keeps it running in the background.")
    print("   (Model script and output logs are saved in the './Models' folder.)")
    print("-" * 80)
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice not in ['1', '2']:
        print("Invalid choice. Exiting.")
        sys.exit(1)

    print("\n" + "#" * 30)
    print("  INPUT CONFIGURATION  ")
    print("#" * 30)

    # Get user input for model, prompt, and task type
    model_id = input("Enter HuggingFace Model ID: ").strip()
    if not model_id:
        print("Error: Model ID cannot be empty")
        sys.exit(1)
    
    prompt = input("Enter your initial prompt/text: ").strip()
    if not prompt:
        print("Error: Prompt cannot be empty")
        sys.exit(1)
    
    print("\nTask type (press Enter for auto-detection):")
    print("  Options: text-generation, summarization, translation, sentiment-analysis, question-answering")
    task_type = input("Task type: ").strip() or "auto"

    print("\n" + "=" * 80)
    
    if choice == '1':
        print("EXECUTING PATH A: TRANSIENT RUN")
        print("=" * 80)
        result = run_transient_model(model_id, prompt, task_type)
        
        # Display results for Path A
        print("\n" + "=" * 80)
        print("TRANSIENT RUN RESPONSE")
        print("=" * 80)
        if result["success"]:
            print(f"✓ Status: SUCCESS (Model: {result['model_id']})")
            print(f"✓ Prompt: {result['prompt'][:60]}{'...' if len(result['prompt']) > 60 else ''}")
            print("-" * 80)
            print("MODEL OUTPUT:")
            print("-" * 80)
            print(result["model_response"])
        else:
            print(f"✗ Status: FAILURE")
            print(f"✗ Error: {result['error']}")
            if result.get("stderr"):
                print(f"\nDebug Info (first 10 lines of stderr):")
                print('\n'.join(result["stderr"].split('\n')[:10]))
        print("=" * 80)
        
    elif choice == '2':
        print("EXECUTING PATH B: PERSISTENT RUN")
        print("=" * 80)
        result = run_persistent_model(model_id, prompt, task_type)
        
        # Display results for Path B
        print("\n" + "=" * 80)
        print("PERSISTENT RUN RESPONSE")
        print("=" * 80)
        if result["success"]:
            print(f"✓ Status: PERSISTENT PROCESS STARTED")
            print(f"✓ Model ID: {result['model_id']}")
            print(f"✓ Process ID (PID): {result['pid']}")
            print(f"✓ Runner Script: {result['runner_path']}")
            print(f"✓ Log File: {result['log_path']}")
            print("-" * 80)
            print("NOTE: The model is loading in the background. Check the log file.")
            print("Initial log check:")
            print('\n'.join(result['status_check'].split('\n')[-3:])) # Show last 3 lines
            print("\n--- macOS/Unix COMMANDS ---")
            print("To view the live status/output in a new terminal:")
            print(f"  tail -f {result['log_path']}")
            print("\nTo stop the model and free memory, run the command:")
            print(f"  kill {result['pid']}")
            print("-" * 80)
        else:
            print(f"✗ Status: FAILED TO START PERSISTENT MODEL")
            print(f"✗ Error: {result['error']}")
            if result.get('log_file'):
                 print(f"✗ Check log file for details: {result['log_file']}")
        print("=" * 80)

if __name__ == "__main__":
    main()
