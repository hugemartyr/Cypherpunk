# import os
# import sys
# import subprocess # NEW: For running the generated script as a separate process
# from typing import Dict, Any

# # --- Agent A's Tool: File Generation ---

# def generate_runner_script_content(agent_id: str, prompt: str) -> str:
#     """
#     Agent A uses this tool to generate the code for the task-specific agent runner.
    
#     This script is designed to load the specified Hugging Face agent/pipeline 
#     and execute the given prompt against it.
#     """
    
#     # We use a text-generation-like structure as a placeholder for the inner agent
 
# import os
# import sys
# from transformers import pipeline

# def execute_hf_agent(agent_id: str, prompt: str) -> str:
#     print(f"[HF RUNNER] Loading pipeline for model: {agent_id} ...")
#     try:
#         generator = pipeline("text-generation", model=agent_id)
#         result = generator(prompt, max_length=100, num_return_sequences=1)[0]["generated_text"]
#         return result
#     except Exception as e:
#         return f"[ERROR] Failed to load pipeline for {agent_id}: {str(e)}"


# if __name__ == "__main__":
#     # The runner script expects agent_id and prompt as command-line arguments
#     if len(sys.argv) != 3:
#         # Print error to stderr so Agent A only captures the result from stdout
#         print("Usage: python hf_agent_runner.py <agent_id> \"<prompt>\"", file=sys.stderr)
#         sys.exit(1)
        
#     runner_agent_id = sys.argv[1]
#     # The prompt is passed as the second argument
#     runner_prompt = sys.argv[2]
    
#     result = execute_hf_agent(runner_agent_id, runner_prompt)
    
#     # Print the result to stdout so the calling script (Agent A) can capture it
#     print(result)


    

# # --- Agent A's Core Execution Logic ---

# def run_agent_A(hf_agent_id: str, prompt: str) -> Dict[str, Any]:
#     """
#     The main logic of Agent A: 
#     1. Generates the runner script.
#     2. Writes it to a file.
#     3. Executes the file using subprocess and captures output.
#     4. Cleans up the file.
#     """
    
#     filename = "hf_agent_runner.py"
    
#     # 1. Generate the script content (Tool Use)
#     script_content = generate_runner_script_content(hf_agent_id, prompt)
    
#     try:
#         # 2. Write the script to a file (Tool Execution Preparation)
#         with open(filename, "w") as f:
#             f.write(script_content)
#         print(f"[AGENT A] Successfully created tool file: {filename}")
        
#         # 3. Execute the script using subprocess (REAL EXECUTION)
#         print(f"[AGENT A] Executing {filename}...")
        
#         # Command: [python interpreter path, script name, arg1: agent_id, arg2: prompt]
#         command = [sys.executable, filename, hf_agent_id, prompt]
        
#         # Use subprocess.run to execute the command, capturing stdout/stderr
#         result = subprocess.run(
#             command, 
#             capture_output=True, 
#             text=True, 
#             check=True # Raise CalledProcessError if the script exits with non-zero code
#         )
        
#         # The agent's final response is the standard output of the runner script
#         captured_output = result.stdout.strip()
#         print(f"[AGENT A] Execution finished. Stdout captured.")

#         # 4. Clean up the file
#         os.remove(filename)
#         print(f"[AGENT A] Cleaned up file: {filename}")
        
#         return {
#             "success": True,
#             "agent_id": hf_agent_id,
#             "prompt": prompt,
#             "agent_response": captured_output
#         }

#     except subprocess.CalledProcessError as e:
#         # Handles errors raised by the tool script (e.g., if the script itself fails)
#         error_output = e.stderr.strip() or "No detailed error output."
#         print(f"[AGENT A] Tool Execution Error (Exit Code {e.returncode}): The runner script failed.", file=sys.stderr)
        
#         # Clean up on error
#         if os.path.exists(filename):
#             os.remove(filename)
            
#         return {
#             "success": False,
#             "error": f"Execution failed. Stderr from runner script: {error_output}"
#         }

#     except Exception as e:
#         # Handles general errors (e.g., file writing permission issues)
#         print(f"[AGENT A] Fatal error during file processing: {e}", file=sys.stderr)
        
#         # Attempt to clean up the file even on general errors
#         if os.path.exists(filename):
#             os.remove(filename)
#             print(f"[AGENT A] Cleaned up file after error: {filename}")
        
#         return {
#             "success": False,
#             "error": f"General execution error: {str(e)}"
#         }


# # --- User Interface ---

# if __name__ == "__main__":
#     print("--- Agent A: Hugging Face Agent Deployment Interface ---")
#     print("This agent creates, executes, and deletes a temporary runner script.")
    
#     # 1. Get Agent ID from User
#     hf_agent_id_input = input("\nEnter the Hugging Face Agent ID (e.g., 'tasks/flight-search-agent'): ").strip()
#     if not hf_agent_id_input:
#         print("Agent ID cannot be empty. Exiting.")
#         sys.exit(1)
        
#     # 2. Get Prompt from User
#     prompt_input = input("Enter the prompt for the agent (e.g., 'Book me the cheapest flight from London to Berlin next Friday'): ").strip()
#     if not prompt_input:
#         print("Prompt cannot be empty. Exiting.")
#         sys.exit(1)

#     print("\n---------------------------------------------------------")
#     print(f"Agent A running task for ID: {hf_agent_id_input}")
#     print("---------------------------------------------------------")
    
#     # 3. Execute Agent A's logic
#     result = run_agent_A(hf_agent_id_input, prompt_input)
    
#     print("\n=========================================================")
#     print("Final Agent A Response to User:")
#     print("=========================================================")
    
#     if result["success"]:
#         print(f"Task Status: SUCCESS")
#         print(f"Target Agent: {result['agent_id']}")
#         print(f"-----------------------------------")
#         print(result["agent_response"])
#     else:
#         print(f"Task Status: FAILURE")
#         print(f"Error: {result['error']}")
#     print("=========================================================")




import os
import sys
import subprocess
from typing import Dict, Any

# --- Agent A's Tool: File Generation ---

def generate_runner_script_content(model_id: str, prompt: str, task_type: str = "auto") -> str:
    """
    Agent A uses this tool to generate the code for the task-specific model runner.
    
    This script loads a real Hugging Face model/pipeline and executes the given prompt.
    
    Args:
        model_id: HuggingFace model ID (e.g., 'gpt2', 'facebook/bart-large-cnn')
        prompt: The input prompt/text to process
        task_type: Task type for pipeline ('auto', 'text-generation', 'summarization', etc.)
    """
    
    content = f"""
import sys
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch

def execute_hf_model(model_id: str, prompt: str, task_type: str = "auto") -> str:
    \"\"\"Loads and executes the specified Hugging Face Model.\"\"\"
    print(f"\\n[HF RUNNER] Loading model: {{model_id}}...")
    print(f"[HF RUNNER] Task type: {{task_type}}")
    
    try:
        # Determine device (GPU if available, else CPU)
        device = 0 if torch.cuda.is_available() else -1
        device_name = "GPU" if device == 0 else "CPU"
        print(f"[HF RUNNER] Using device: {{device_name}}")
        
        # Load the pipeline based on task type
        if task_type == "auto":
            # Let transformers auto-detect the task
            print("[HF RUNNER] Auto-detecting task from model...")
            pipe = pipeline(model=model_id, device=device)
        else:
            # Use specified task type
            pipe = pipeline(task_type, model=model_id, device=device)
        
        print("[HF RUNNER] Model loaded successfully. Processing prompt...")
        
        # Execute the model
        result = pipe(prompt)
        
        # Format output based on result type
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict):
                # Handle various output formats
                if 'generated_text' in result[0]:
                    response_text = result[0]['generated_text']
                elif 'summary_text' in result[0]:
                    response_text = result[0]['summary_text']
                elif 'translation_text' in result[0]:
                    response_text = result[0]['translation_text']
                elif 'label' in result[0]:
                    # Classification output
                    response_text = f"Classification: {{result[0]['label']}} (confidence: {{result[0].get('score', 'N/A')}})"
                else:
                    response_text = str(result[0])
            else:
                response_text = str(result[0])
        else:
            response_text = str(result)
        
        print("[HF RUNNER] Task completed successfully.")
        return response_text
        
    except Exception as e:
        error_msg = f"[HF RUNNER ERROR] Failed to execute model: {{str(e)}}"
        print(error_msg, file=sys.stderr)
        return error_msg

if __name__ == "__main__":
    # The runner script expects model_id, prompt, and optional task_type as command-line arguments
    if len(sys.argv) < 3:
        print("Usage: python hf_model_runner.py <model_id> '<prompt>' [task_type]", file=sys.stderr)
        sys.exit(1)
    
    runner_model_id = sys.argv[1]
    runner_prompt = sys.argv[2]
    runner_task_type = sys.argv[3] if len(sys.argv) > 3 else "auto"
    
    result = execute_hf_model(runner_model_id, runner_prompt, runner_task_type)
    
    # Print the result to stdout so the calling script (Agent A) can capture it
    print("\\n" + "="*60)
    print("MODEL OUTPUT:")
    print("="*60)
    print(result)
    print("="*60)
"""
    return content.strip()

# --- Agent A's Core Execution Logic ---

def run_agent_A(model_id: str, prompt: str, task_type: str = "auto") -> Dict[str, Any]:
    """
    The main logic of Agent A: 
    1. Generates the runner script with real HuggingFace model loading.
    2. Writes it to a file.
    3. Executes the file using subprocess and captures output.
    4. Cleans up the file.
    
    Args:
        model_id: HuggingFace model ID (e.g., 'gpt2', 'facebook/bart-large-cnn')
        prompt: The input text/prompt to process
        task_type: Optional task type ('auto', 'text-generation', 'summarization', etc.)
    """
    
    filename = "hf_model_runner.py"
    
    # 1. Generate the script content (Tool Use)
    script_content = generate_runner_script_content(model_id, prompt, task_type)
    
    try:
        # 2. Write the script to a file (Tool Execution Preparation)
        with open(filename, "w") as f:
            f.write(script_content)
        print(f"[AGENT A] Successfully created tool file: {filename}")
        
        # 3. Execute the script using subprocess (REAL EXECUTION)
        print(f"[AGENT A] Executing {filename}...")
        print(f"[AGENT A] This may take a while depending on model size and your hardware...")
        
        # Command: [python interpreter path, script name, arg1: model_id, arg2: prompt, arg3: task_type]
        command = [sys.executable, filename, model_id, prompt, task_type]
        
        # Use subprocess.run to execute the command, capturing stdout/stderr
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            timeout=300  # 5 minute timeout for model loading and execution
        )
        
        # The model's response is in stdout
        captured_output = result.stdout.strip()
        captured_errors = result.stderr.strip()
        
        print(f"[AGENT A] Execution finished.")
        
        # 4. Clean up the file
        os.remove(filename)
        print(f"[AGENT A] Cleaned up file: {filename}")
        
        # Check if execution was successful
        if result.returncode != 0 or "[HF RUNNER ERROR]" in captured_output:
            return {
                "success": False,
                "model_id": model_id,
                "prompt": prompt,
                "error": captured_errors or captured_output,
                "stderr": captured_errors
            }
        
        return {
            "success": True,
            "model_id": model_id,
            "prompt": prompt,
            "model_response": captured_output,
            "stderr": captured_errors  # May contain loading messages
        }

    except subprocess.TimeoutExpired:
        error_msg = "Execution timed out (5 minutes). Model may be too large or system too slow."
        print(f"[AGENT A] {error_msg}", file=sys.stderr)
        
        if os.path.exists(filename):
            os.remove(filename)
            
        return {
            "success": False,
            "error": error_msg
        }

    except subprocess.CalledProcessError as e:
        error_output = e.stderr.strip() or "No detailed error output."
        print(f"[AGENT A] Tool Execution Error (Exit Code {e.returncode})", file=sys.stderr)
        
        if os.path.exists(filename):
            os.remove(filename)
            
        return {
            "success": False,
            "error": f"Execution failed. Stderr: {error_output}"
        }

    except Exception as e:
        print(f"[AGENT A] Fatal error during file processing: {e}", file=sys.stderr)
        
        if os.path.exists(filename):
            os.remove(filename)
            print(f"[AGENT A] Cleaned up file after error: {filename}")
        
        return {
            "success": False,
            "error": f"General execution error: {str(e)}"
        }


# --- User Interface ---

if __name__ == "__main__":
    print("=" * 70)
    print("    Agent A: Real HuggingFace Model Execution Interface")
    print("=" * 70)
    print("This agent creates, executes, and deletes a temporary runner script")
    print("that loads and runs REAL HuggingFace models.\n")
    
    print("Example model IDs:")
    print("  - gpt2 (text generation)")
    print("  - facebook/bart-large-cnn (summarization)")
    print("  - distilbert-base-uncased-finetuned-sst-2-english (sentiment)")
    print("  - t5-small (translation, summarization, Q&A)")
    print()
    
    # 1. Get Model ID from User
    model_id_input = input("Enter the HuggingFace Model ID: ").strip()
    if not model_id_input:
        print("Model ID cannot be empty. Exiting.")
        sys.exit(1)
    
    # 2. Get Prompt from User
    prompt_input = input("Enter the prompt/text to process: ").strip()
    if not prompt_input:
        print("Prompt cannot be empty. Exiting.")
        sys.exit(1)
    
    # 3. Get Task Type (optional)
    print("\nTask type (press Enter for auto-detection):")
    print("  Options: text-generation, summarization, translation, sentiment-analysis, etc.")
    task_type_input = input("Task type: ").strip() or "auto"

    print("\n" + "-" * 70)
    print(f"Agent A executing task for model: {model_id_input}")
    print(f"Task type: {task_type_input}")
    print("-" * 70)
    
    # 4. Execute Agent A's logic
    result = run_agent_A(model_id_input, prompt_input, task_type_input)
    
    print("\n" + "=" * 70)
    print("FINAL AGENT A RESPONSE TO USER:")
    print("=" * 70)
    
    if result["success"]:
        print(f"✓ Task Status: SUCCESS")
        print(f"✓ Model: {result['model_id']}")
        print("-" * 70)
        print(result["model_response"])
        if result.get("stderr"):
            print("\n[Debug Info from Model Loading:]")
            print(result["stderr"])
    else:
        print(f"✗ Task Status: FAILURE")
        print(f"✗ Error: {result['error']}")
        if result.get("stderr"):
            print(f"✗ Stderr: {result['stderr']}")
    
    print("=" * 70)