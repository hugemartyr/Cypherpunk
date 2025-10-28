import os
import sys
import subprocess
import json
from typing import Dict, Any
import time
import signal

# Define the directory for persistent models
MODELS_DIR = "Models"

# -------------------------------
# --- Runner Script Generators ---
# -------------------------------

def generate_transient_runner_content() -> str:
    """
    Path A Runner: Single-use, executes one prompt then exits.
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
        device = 0 if torch.cuda.is_available() else -1
        if task_type == "auto":
            pipe = pipeline(model=model_id, device=device, trust_remote_code=True)
        else:
            pipe = pipeline(task_type, model=model_id, device=device, trust_remote_code=True)

        if task_type == "text-generation" or (task_type == "auto" and any(x in model_id.lower() for x in ["gpt", "llama"])):
            result = pipe(prompt, max_new_tokens=100, return_full_text=False, do_sample=True, temperature=0.7)
        else:
            result = pipe(prompt)

        output = ""
        if isinstance(result, list) and result and isinstance(result[0], dict):
            key = next((k for k in ["generated_text", "summary_text", "translation_text", "answer"] if k in result[0]), "label")
            if key == "label":
                output = f"Label: {result[0]['label']}, Score: {result[0].get('score', 'N/A')}"
            else:
                output = result[0][key]
        else:
            output = str(result)

        print(json.dumps({"success": True, "output": output, "model_id": model_id}))
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e), "model_id": model_id}))
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(json.dumps({"success": False, "error": "Missing arguments"}))
        sys.exit(1)
    execute_hf_model(sys.argv[1], sys.argv[2], sys.argv[3])
"""

# ---------------------------------------------------------
# Path B: Persistent Model Runner Generator (Rewritten)
# ---------------------------------------------------------

def generate_persistent_runner_content(model_id: str, prompt: str, task_type: str, log_file: str) -> str:
    """
    Generates a persistent runner script with variables embedded directly
    (so it doesn’t rely on sys.argv inputs at runtime).
    """
    return f"""
import sys, json, os, time, torch, signal, warnings, select
from transformers import pipeline
warnings.filterwarnings('ignore')

# === Embedded Configuration (Auto-generated) ===
MODEL_ID = "{model_id}"
INITIAL_PROMPT = \"\"\"{prompt}\"\"\"
TASK_TYPE = "{task_type}"
LOG_FILE = "{log_file}"
RUNNING = True

def log(msg):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{{time.strftime('%Y-%m-%d %H:%M:%S')}}] {{msg}}\\n")
    print(msg, flush=True)

def handle_exit(signum, frame):
    global RUNNING
    RUNNING = False
    log(" Signal received. Shutting down gracefully...")

signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

def infer(pipe, text):
    if TASK_TYPE == "text-generation" or (TASK_TYPE == "auto" and any(x in MODEL_ID.lower() for x in ["gpt", "llama"])):
        out = pipe(text, max_new_tokens=100, return_full_text=False, do_sample=True, temperature=0.7)
    else:
        out = pipe(text)
    if isinstance(out, list) and out and isinstance(out[0], dict):
        key = next((k for k in ['generated_text', 'summary_text', 'translation_text', 'answer'] if k in out[0]), 'label')
        return out[0].get(key, str(out[0]))
    return str(out)

def main():
    global RUNNING
    log(f"Persistent Runner started for {{MODEL_ID}}")
    device = 0 if torch.cuda.is_available() else -1
    log(f"Loading model on {{'GPU' if device == 0 else 'CPU'}}...")

    try:
        pipe = pipeline(TASK_TYPE if TASK_TYPE != "auto" else None, model=MODEL_ID, device=device, trust_remote_code=True)
        log("Model loaded successfully.")
    except Exception as e:
        log(f"[FATAL] Model loading failed: {{e}}")
        sys.exit(1)

    # Run initial inference
    try:
        log(f"⚡ Running initial prompt: {{INITIAL_PROMPT[:60]}}...")
        output = infer(pipe, INITIAL_PROMPT)
        log(f"Initial Output: {{output[:120]}}...")
    except Exception as e:
        log(f"[ERROR] During initial inference: {{e}}")

    log("Waiting for new prompts (via stdin)...")

    while RUNNING:
        try:
            if sys.stdin in select.select([sys.stdin], [], [], 1)[0]:
                new_prompt = sys.stdin.readline().strip()
                if new_prompt:
                    log(f"New prompt received: {{new_prompt[:60]}}...")
                    resp = infer(pipe, new_prompt)
                    log(f"Response: {{resp[:150]}}...")
            time.sleep(0.5)
        except Exception as e:
            log(f"[ERROR] Runtime: {{e}}")
            time.sleep(1)
    log("Persistent runner terminated.")

if __name__ == "__main__":
    main()
"""


# -----------------------------------------
# --- Execution Functions ---
# -----------------------------------------

def run_transient_model(model_id: str, prompt: str, task_type: str = "auto") -> Dict[str, Any]:
    """Executes Path A: Transient Run (unchanged)."""
    filename = "hf_transient_runner.py"
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(generate_transient_runner_content())

        command = [sys.executable, filename, model_id, prompt, task_type]
        result = subprocess.run(command, capture_output=True, text=True, timeout=600)
        stdout, stderr = result.stdout.strip(), result.stderr.strip()

        if result.returncode != 0 or not stdout:
            return {"success": False, "error": stderr or "Unknown failure"}
        return json.loads(stdout)
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if os.path.exists(filename):
            os.remove(filename)

def run_persistent_model(model_id: str, prompt: str, task_type: str = "auto") -> Dict[str, Any]:
    """Executes Path B: Persistent Run (using improved runner with embedded parameters)."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_safe = model_id.replace("/", "__").replace("-", "_")
    model_dir = os.path.join(MODELS_DIR, model_safe)
    os.makedirs(model_dir, exist_ok=True)

    runner_file = os.path.join(model_dir, "runner.py")
    log_file = os.path.join(model_dir, "log.txt")

    # ✅ Generate runner content with parameters embedded
    script_content = generate_persistent_runner_content(model_id, prompt, task_type, log_file)

    # Write the runner script
    with open(runner_file, "w", encoding="utf-8") as f:
        f.write(script_content)

    # Clear previous logs if any
    if os.path.exists(log_file):
        os.remove(log_file)

    # ✅ Launch the persistent runner (no need to pass CLI args now)
    command = [sys.executable, runner_file]
    process = subprocess.Popen(command, cwd=model_dir, text=True)
    time.sleep(5)

    return {
        "success": True,
        "pid": process.pid,
        "runner": runner_file,
        "log": log_file,
        "status": f"Model {model_id} started in persistent mode with embedded parameters."
    }


# -----------------------------------------
# --- Main Interface ---
# -----------------------------------------

def main():
    print("=" * 80)
    print("HuggingFace Model Orchestrator")
    print("=" * 80)
    print("1. Path A: Transient Run")
    print("2. Path B: Persistent Run")
    choice = input("Enter choice (1 or 2): ").strip()

    model_id = input("Model ID: ").strip()
    prompt = input("Prompt: ").strip()
    task_type = input("Task Type (press Enter for auto): ").strip() or "auto"

    if choice == "1":
        res = run_transient_model(model_id, prompt, task_type)
        print(json.dumps(res, indent=2))
    elif choice == "2":
        res = run_persistent_model(model_id, prompt, task_type)
        print(json.dumps(res, indent=2))
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()

