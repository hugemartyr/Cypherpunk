import sys
import json
from transformers import pipeline
import torch
import warnings
# We need repr for safe string formatting
from reprlib import repr as safe_repr

def generate_transient_runner_content(model_id: str, prompt: str, task_type: str) -> str:
    """
    Path A Runner: Single-use, executes one prompt then exits.
    
    - Safely injects variables using repr().
    - Prints all debug logs to sys.stderr to keep stdout clean for JSON.
    """
    
    # --- FIX 1: Safe Variable Injection ---
    # Use repr() to create a valid Python string representation
    # (e.g., "gpt2" becomes "'gpt2'")
    # We use safe_repr to truncate very long prompts in the debug log.
    
    model_id_repr = repr(model_id)
    prompt_repr = repr(prompt)
    task_type_repr = repr(task_type)
    
    # This is for debug logging only, to avoid printing a massive prompt
    truncated_prompt_repr = safe_repr(prompt)

    # --- FIX 2: Use an f-string for the whole template ---
    # This is much cleaner and avoids the .replace() and {SET} issues.
    
    return f"""
import sys
import json
from transformers import pipeline
import torch
import warnings
import traceback

# --- DEBUG: Redirect warnings to stderr ---
warnings.filterwarnings('ignore')
def showwarning(message, category, filename, lineno, file=None, line=None):
    print(f"Warning: {{message}}", file=sys.stderr)
warnings.showwarning = showwarning

# --- Safely injected variables ---
MODEL_ID = {model_id_repr}
PROMPT = {prompt_repr}
TASK_TYPE = {task_type_repr}

def execute_hf_model(model_id, prompt, task_type):
    try:
        # --- DEBUG: Print to stderr ---
        print(f"[DEBUG] Transient runner started.", file=sys.stderr)
        print(f"[DEBUG] Task: {{task_type}}, Model: {{model_id}}", file=sys.stderr)
        print(f"[DEBUG] Prompt (truncated): {truncated_prompt_repr}", file=sys.stderr)
        
        device = 0 if torch.cuda.is_available() else -1
        print(f"[DEBUG] Using device: {{device}} (0=CUDA, -1=CPU)", file=sys.stderr)

        if task_type == "auto":
            print("[DEBUG] Auto-detecting pipeline...", file=sys.stderr)
            pipe = pipeline(model=model_id, device=device, trust_remote_code=True)
        else:
            print(f"[DEBUG] Loading pipeline for task: {{task_type}}...", file=sys.stderr)
            pipe = pipeline(task_type, model=model_id, device=device, trust_remote_code=True)
        
        print(f"[DEBUG] Pipeline loaded. Inferred task: {{pipe.task}}", file=sys.stderr)

        print("[DEBUG] Running inference...", file=sys.stderr)
        if pipe.task == "text-generation":
            print("[DEBUG] Using text-generation parameters (max_new_tokens=100).", file=sys.stderr)
            result = pipe(prompt, max_new_tokens=100, return_full_text=False, do_sample=True, temperature=0.7)
        else:
            print("[DEBUG] Using default pipeline parameters.", file=sys.stderr)
            result = pipe(prompt)
        
        print("[DEBUG] Inference complete. Parsing result...", file=sys.stderr)
        # print(f"[DEBUG] Raw result: {{result}}", file=sys.stderr) # Uncomment for very verbose logging

        output = ""
        if isinstance(result, list) and result and isinstance(result[0], dict):
            # Try to find a standard output key
            key = next((k for k in ["generated_text", "summary_text", "translation_text", "answer"] if k in result[0]), None)
            
            if key:
                output = result[0][key]
            elif "label" in result[0]: # Handle classification
                output = f"Label: {{result[0]['label']}}, Score: {{result[0].get('score', 'N/A')}}"
            else: # Fallback: just serialize the first item
                output = json.dumps(result[0])
        else:
            output = str(result)

        print("[DEBUG] Result parsed. Sending JSON to stdout.", file=sys.stderr)
        
        # --- FINAL OUTPUT: This goes to stdout ---
        print(json.dumps({{"success": True, "output": output, "model_id": model_id}}))

    except Exception as e:
        # --- ERROR HANDLING: Print all errors to stderr ---
        print(f"[DEBUG] !!! An error occurred: {{str(e)}}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        
        # --- FINAL OUTPUT: Send JSON error to stdout ---
        print(json.dumps({{"success": False, "error": str(e), "model_id": model_id}}))
        sys.exit(1)

if __name__ == "__main__":
    # --- FIX 3: Call with the variables directly ---
    execute_hf_model(MODEL_ID, PROMPT, TASK_TYPE)
"""