import sys
from transformers import pipeline
import torch

def execute_hf_model(model_id: str, prompt: str, task_type: str = "auto") -> str:
    """Loads and executes the specified Hugging Face Model."""
    
    # 1. Setup Environment
    print(f"\n[HF_RUNNER_SCRIPT] Model ID: {model_id} | Task: {task_type}")
    try:
        # Determine device (GPU if available, else CPU)
        device = 0 if torch.cuda.is_available() else -1
        device_name = "GPU" if device == 0 else "CPU"
        print(f"[HF_RUNNER_SCRIPT] Using device: {device_name}")
        
        # 2. Load Pipeline
        pipe = pipeline(task_type, model=model_id, device=device)
        print("[HF_RUNNER_SCRIPT] Model loaded. Processing...")
        
        # 3. Execute Model
        # Use appropriate parameters for text generation if applicable
        if task_type == "text-generation":
            result = pipe(prompt, max_length=100, num_return_sequences=1, truncation=True)
        else:
            result = pipe(prompt)
            
        # 4. Format Output
        if isinstance(result, list) and result:
            response_text = result[0].get('generated_text', result[0].get('summary_text', str(result[0])))
        else:
            response_text = str(result)
        
        return response_text
        
    except Exception as e:
        # Print error to stderr for Agent A to capture and report
        print(f"[HF_RUNNER_SCRIPT ERROR] Failed to execute: {str(e)}", file=sys.stderr)
        # Return a clear error message as stdout for easy parsing if stderr is not captured
        return f"ERROR: Execution failed. Check stderr for details."

if __name__ == "__main__":
    # Expects model_id, prompt, and optional task_type
    if len(sys.argv) < 3:
        print("Usage: python hf_runner_script.py <model_id> '<prompt>' [task_type]", file=sys.stderr)
        sys.exit(1)
    
    runner_model_id = sys.argv[1]
    runner_prompt = sys.argv[2]
    runner_task_type = sys.argv[3] if len(sys.argv) > 3 else "auto"
    
    final_result = execute_hf_model(runner_model_id, runner_prompt, runner_task_type)
    
    # Critical: Print the final result to stdout for the parent process (Orchestrator) to capture
    print(">>>AGENT_RESPONSE_START<<<")
    print(final_result)
    print(">>>AGENT_RESPONSE_END<<<")
