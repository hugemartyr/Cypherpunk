import os
import sys
import subprocess
from typing import Dict, Any

# Assuming the tools directory is correctly set up
# We import the FileGeneratorTool for Agent A to use
sys.path.append(os.path.join(os.path.dirname(__file__), 'tools'))
from tool_generators import FileGeneratorTool

class HFAgentOrchestrator:
    """
    The Core Agent (Agent A). Manages the execution flow, state, and tool delegation.
    It orchestrates subprocess calls to run external tasks (like the HF model).
    """
    def __init__(self, runner_filename: str = "temp_hf_runner.py", timeout: int = 300):
        self.runner_filename = runner_filename
        self.timeout = timeout
        print("[ORCHESTRATOR] Initialized. Ready to delegate tasks.")

    def _execute_script(self, command: list) -> Dict[str, Any]:
        """Handles the generic subprocess execution and result extraction."""
        print(f"[ORCHESTRATOR] Executing command: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                command, 
                capture_output=True, 
                text=True, 
                timeout=self.timeout
            )
            
            # Use marker logic to cleanly extract response from stdout
            stdout_lines = result.stdout.split('\n')
            
            response_start_index = stdout_lines.index(">>>AGENT_RESPONSE_START<<<") + 1 if ">>>AGENT_RESPONSE_START<<<" in stdout_lines else -1
            response_end_index = stdout_lines.index(">>>AGENT_RESPONSE_END<<<") if ">>>AGENT_RESPONSE_END<<<" in stdout_lines else -1

            if response_start_index != -1 and response_end_index != -1:
                model_response = "\n".join(stdout_lines[response_start_index:response_end_index]).strip()
            else:
                model_response = result.stdout.strip()
                
            return {
                "success": result.returncode == 0,
                "response": model_response,
                "stderr": result.stderr.strip(),
                "full_stdout": result.stdout.strip(),
            }
        
        except subprocess.TimeoutExpired:
            return {"success": False, "response": "Execution timed out.", "stderr": ""}
        except Exception as e:
            return {"success": False, "response": f"General execution error: {str(e)}", "stderr": ""}

    # --- Core Functionality 1: Run HuggingFace Model ---
    def run_hf_model(self, model_id: str, prompt: str, task_type: str = "auto") -> Dict[str, Any]:
        """Agent A uses its internal file generation logic and subprocess tool to run a model."""
        
        print(f"\n[ORCHESTRATOR] Task: Run HF Model '{model_id}'")
        
        # 1. GENERATE TOOL SCRIPT CONTENT (Tool Use: File Generation)
        # This uses the specific content generator logic (e.g., from tools/tool_generators.py)
        script_content = FileGeneratorTool.generate_task_runner_script_content(
            model_id, prompt, task_type
        )
        
        try:
            # 2. WRITE SCRIPT TO DISK
            with open(self.runner_filename, "w") as f:
                f.write(script_content)
            print(f"[ORCHESTRATOR] Temporary runner script created: {self.runner_filename}")
            
            # 3. EXECUTE SCRIPT (Tool Execution: Subprocess)
            command = [sys.executable, self.runner_filename]
            
            execution_result = self._execute_script(command)
            
            # 4. CLEAN UP
            os.remove(self.runner_filename)
            print(f"[ORCHESTRATOR] Cleaned up temporary file.")
            
            # 5. POST-EXECUTION ANALYSIS (Agent A's logic)
            if execution_result["success"]:
                return {
                    "status": "SUCCESS",
                    "model_response": execution_result["response"],
                    "debug_logs": execution_result["full_stdout"]
                }
            else:
                return {
                    "status": "FAILED",
                    "error": execution_result["response"],
                    "stderr": execution_result["stderr"]
                }
        
        except Exception as e:
            if os.path.exists(self.runner_filename):
                os.remove(self.runner_filename)
            return {"status": "CRITICAL FAILURE", "error": str(e)}

    # --- Core Functionality 2: Another Feature (Example of future functionality) ---
    def generate_project_config(self, project_name: str, models: list, config_path: str = "project_config.json") -> Dict[str, str]:
        """Agent A uses its file generation tool to create a project config file."""
        print(f"\n[ORCHESTRATOR] Task: Generate Config File '{config_path}'")
        
        # 1. GENERATE FILE CONTENT (Tool Use: File Generation)
        config_content = FileGeneratorTool.generate_config_file(project_name, models)
        
        # 2. WRITE FILE TO DISK
        try:
            with open(config_path, "w") as f:
                f.write(config_content)
            
            return {
                "status": "SUCCESS",
                "message": f"Successfully generated config file at {config_path}",
                "content_preview": config_content[:150] + "..."
            }
        except Exception as e:
            return {"status": "FAILED", "message": f"Could not write config file: {str(e)}"}


if __name__ == "__main__":
    print("="*60)
    print("   Hugging Face Agent Orchestrator Demo")
    print("="*60)

    # Instantiate Agent A
    agent_a = HFAgentOrchestrator()
    
    # --- DEMO 1: Running the HF Model (using generated script) ---
    model = "facebook/bart-large-cnn"
    user_prompt = "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog."
    
    print("\n--- DEMO 1: Running HF Model Task (Simulated) ---")
    hf_result = agent_a.run_hf_model(model, user_prompt, task_type="summarization")
    
    print("\n[FINAL ORCHESTRATOR REPORT - DEMO 1]")
    print(f"Status: {hf_result.get('status')}")
    if hf_result['status'] == 'SUCCESS':
        print("Response:")
        print(hf_result['model_response'])
    else:
        print(f"Error: {hf_result.get('error')}")

    # --- DEMO 2: Using the File Generation Tool directly (for config) ---
    print("\n--- DEMO 2: Generating a Config File (Tool Use) ---")
    config_result = agent_a.generate_project_config(
        "Financial_Llama_Project",
        ["llama-3-8b", "facebook/bart-large-cnn"]
    )

    print("\n[FINAL ORCHESTRATOR REPORT - DEMO 2]")
    print(f"Status: {config_result.get('status')}")
    print(f"Message: {config_result.get('message')}")
    print("Content Preview:")
    print(config_result.get('content_preview'))
    
    # Clean up the generated config file for a clean run next time
    if os.path.exists("project_config.json"):
        os.remove("project_config.json")
