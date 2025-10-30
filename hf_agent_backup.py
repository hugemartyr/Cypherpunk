import os
import sys
import subprocess
import json
from typing import Dict, Any, List
import time
import signal
import requests
import dotenv
import logging
from uuid import uuid4
from datetime import datetime, timezone

from uagents import Agent, Protocol, Context, Model
from uagents.setup import fund_agent_if_low

# Import the necessary components from the chat protocol
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    TextContent,
    EndSessionContent,
    StartSessionContent,
    chat_protocol_spec,
)

# --- Configuration ---
dotenv.load_dotenv()

# Set up logging
logger = logging.getLogger("HFManagerAgent")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Agent Configuration
AGENT_SEED = os.getenv("HF_MANAGER_SEED", "hf_manager_agent_secret_seed_phrase_abc")
AGENT_PORT = int(os.getenv("HF_MANAGER_PORT", "8000")) # Use a different port than specialists

# ASI:One API Configuration
API_KEY = os.getenv("ASI_ONE_API_KEY")
BASE_URL = "https://api.asi1.ai/v1/chat/completions"
MODEL_NAME = "asi1-mini" # Use the mini model

if not API_KEY:
    logger.error("FATAL ERROR: ASI_ONE_API_KEY environment variable is not set.")
    sys.exit(1)

# Define the directory for persistent models relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "Agent_Models")) # Persistent models go here
os.makedirs(MODELS_DIR, exist_ok=True)

# Port counter for persistent agents
curr_port = AGENT_PORT + 1 # Start ports for specialists after the manager's port

# --- Tool Implementations ---

# Assume these functions are defined in separate files

from specialist_agent_runner_script import generate_persistent_runner_content
from transient_runner_script import generate_transient_runner_content


def run_transient_model(model_id: str, prompt: str, task_type: str = "auto") -> Dict[str, Any]:
    """
    Executes Path A: Transient Run (single-use inference).
    Logs all subprocess output (stdout and stderr) for debugging.
    """
    
    # --- MODIFICATION ---
    # Removed the incorrect repr() and manual quoting lines.
    # The `generate_transient_runner_content` function is solely responsible
    # for safely embedding these variables into the script.
    model_id = model_id.replace('"', '') # Keep sanitization
    # --- END MODIFICATION ---

    # We use the original, unquoted variables here
    print(f"[DEBUG] Running transient model: {model_id} with prompt: {prompt} and task_type: {task_type}")
    filename = f"hf_transient_runner_{uuid4()}.py" # Unique filename
    runner_path = os.path.join(SCRIPT_DIR, filename) # Place in script dir
    logger.info(f"Starting transient run for {model_id} with script {filename}")
    
    try:
        script_content = generate_transient_runner_content(model_id, prompt, task_type)
        if not script_content:
             raise ValueError("Failed to generate transient runner script content.")

        with open(runner_path, "w", encoding="utf-8") as f:
            f.write(script_content)

        python_executable = sys.executable
        command = [python_executable, runner_path]

        # Increased timeout for model download/run
        logger.info(f"Executing command: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', timeout=900) # 15 min timeout
        
        stdout, stderr = result.stdout.strip(), result.stderr.strip()

        if stderr:
            logger.info(f"--- Subprocess STDERR for {filename} ---")
            logger.info(stderr)
            logger.info(f"--- End Subprocess STDERR for {filename} ---")
        
        if stdout:
            logger.info(f"--- Subprocess STDOUT for {filename} ---")
            logger.info(stdout)
            logger.info(f"--- End Subprocess STDOUT for {filename} ---")
        # --- END MODIFICATION ---

        if result.returncode != 0 or not stdout:
            # Use stderr as the primary error message if it exists
            error_message = stderr or f"Unknown failure (return code {result.returncode})"
            if not stdout:
                error_message += " (No STDOUT received)"
            logger.error(f"Transient run failed for {model_id}: {error_message}")
            return {"status": "error", "message": error_message}

        logger.info(f"Transient run subprocess finished for {model_id}.")
        
        # Attempt to parse JSON output
        try:
            output_json = json.loads(stdout)
            if isinstance(output_json, dict) and "status" not in output_json:
                 output_json["status"] = "success"
            logger.info(f"Transient run successful for {model_id}.")
            return output_json
        except json.JSONDecodeError:
            # This is why we log stdout above
            logger.error(f"Transient runner output for {model_id} was NOT valid JSON.")
            # Return the raw output so the agent can see what went wrong
            return {"status": "error", "message": "Failed to decode JSON response from script.", "raw_output": stdout}

    except subprocess.TimeoutExpired:
        logger.error(f"Transient run timed out for {model_id}")
        return {"status": "error", "message": "Process timed out after 15 minutes."}
    except Exception as e:
        logger.error(f"Error during transient run for {model_id}: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}
    finally:
        # Cleanup the temporary script
        if os.path.exists(runner_path):
            try:
                os.remove(runner_path)
                logger.debug(f"Cleaned up transient script: {filename}")
            except OSError as e:
                logger.warning(f"Could not remove transient script {filename}: {e}")
                
                
def run_persistent_model(model_id: str, task_type: str = "auto") -> Dict[str, Any]:
    """Executes Path B: Persistent Run (starts a long-running specialist agent)."""
    model_id.replace('"', '') # Sanitize input
    repr(model_id)
    repr(task_type)

    print(f"[DEBUG] Running persistent model: {model_id} with task_type: {task_type}")
    global curr_port
    port_to_use = curr_port
    curr_port += 1 # Increment for the next one

    logger.info(f"Starting persistent agent for {model_id} on port {port_to_use}")

    # Create a safe directory name
    model_safe = model_id.replace("/", "__").replace("-", "_")
    model_safe += "_model"
    model_dir = os.path.join(MODELS_DIR, model_safe)
    os.makedirs(model_dir, exist_ok=True)

    runner_file = os.path.join(model_dir, "runner.py")
    log_file = os.path.join(model_dir, "log.txt")

    # Generate the specialist agent script content
    script_content = generate_persistent_runner_content(model_id,  port_to_use, task_type, log_file=log_file)
    if not script_content:
        error_msg = f"Failed to generate persistent runner script content for {model_id}"
        logger.error(error_msg)
        return {"status": "error", "message": error_msg}

    # Write the runner script
    with open(runner_file, "w", encoding="utf-8") as f:
        f.write(script_content)

    # Ensure the script uses the correct Python executable (from the current venv)
    python_executable = sys.executable
    command = [python_executable, runner_file]
    
    
    
    # TODO: Actually
    # need to run the agent and give the agent address back
    
    
    

    # try:
    #     # Start the specialist agent as a background process
    #     # Use Popen for non-blocking start. We don't wait for it here.
    #     process = subprocess.Popen(
    #         command,
    #         cwd=model_dir, # Run from the model's directory
    #         stdout=subprocess.PIPE, # Capture output if needed, but primarily check logs
    #         stderr=subprocess.PIPE,
    #         text=True,
    #         # Use start_new_session on non-Windows to detach from parent
    #         start_new_session=True if sys.platform != "win32" else False
    #     )
    #     logger.info(f"Started persistent agent process for {model_id} with PID {process.pid}")

    #     # Give it a moment to start up
    #     time.sleep(5)

    #     # Check if the process is still running (basic check)
    #     if process.poll() is not None: # Process terminated early
    #         stdout, stderr = process.communicate()
    #         error_msg = f"Persistent agent for {model_id} failed to start. Return code: {process.returncode}. Stderr: {stderr}"
    #         logger.error(error_msg)
    #         return {"status": "error", "message": error_msg}

    #     # Return success with info on how to access the agent
    #     return {
    #         "status": "success",
    #         "message": f"Model {model_id} started in persistent mode.",
    #         "agent_name": f"{model_id.replace('/', '_')}_specialist",
    #         "port": port_to_use,
    #         "endpoint": f"http://localhost:{port_to_use}/submit", # Assuming local deployment
    #         "log_file": log_file,
    #         "pid": process.pid
    #     }
    # except Exception as e:
    #     logger.error(f"Error starting persistent agent for {model_id}: {e}", exc_info=True)
    #     return {"status": "error", "message": f"Failed to start persistent process: {str(e)}"}
    
    

# --- Tool Schemas for ASI:One ---
# Keep schemas consistent with function definitions

run_transient_model_schema={
    "type": "function",
    "function": {
        "name": "run_transient_model",
        "description": "Run a HuggingFace model in transient mode (single-use). Use this for one-off requests where persistence is not needed.",
        "parameters": {
            "type": "object",
            "properties": {
                "model_id": {
                    "type": "string",
                    "description": "The HuggingFace model ID (e.g., 'gpt2', 'facebook/bart-large-cnn')."
                },
                "prompt": {
                    "type": "string",
                    "description": "The input prompt or text for the model."
                },
                "task_type": {
                    "type": "string",
                    "description": "Optional: Task type ('text-generation', 'summarization', etc.). Default is 'auto'.",
                    "default": "auto"
                }
            },
            "required": ["model_id", "prompt"],
            "additionalProperties": False
        },
        "strict": True # Ensure only defined parameters are passed
    }
}

run_persistent_model_schema={
    "type": "function", 
    "function": {
        "name": "run_persistent_model",
        "description": "Run a HuggingFace model in persistent mode (long-running). Use this when expecting multiple requests for the same model.",
        "parameters": {
            "type": "object",
            "properties": {
                "model_id": {
                    "type": "string",
                    "description": "The HuggingFace model ID (e.g., 'gpt2', 'facebook/bart-large-cnn')."
                },
                "task_type": {
                    "type": "string",
                    "description": "Optional: Task type ('text-generation', 'summarization', etc.). Default is 'auto'.",
                    "default": "auto"
                }
            },
            "required": ["model_id"],
            "additionalProperties": False
        },
        "strict": True # Ensure only defined parameters are passed
    }
}

tools_list = [run_transient_model_schema, run_persistent_model_schema]

# Available functions map for execution
available_tools = {
    "run_transient_model": run_transient_model,
    "run_persistent_model": run_persistent_model,
}

# --- Agent Definition ---
agent = Agent(
    name="HF_MANAGEMENT_AGENT",
    seed=AGENT_SEED,
    port=AGENT_PORT,
    endpoint=[f"http://localhost:{AGENT_PORT}/submit"], # Manager agent endpoint
)
chat_proto = Protocol(spec=chat_protocol_spec)
conversations: Dict[str, List[Dict[str, Any]]] = {}

# --- Agent Event Handlers ---

@agent.on_event('startup')
async def startup_handler(ctx: Context):
    ctx.logger.info(f'My name is {ctx.agent.name} and my address is {ctx.agent.address}')
    ctx.logger.info(f'Agent running on http://localhost:{AGENT_PORT}')

@chat_proto.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    session_id = sender # Use sender address as session identifier

    id = -1

    # Initialize conversation history if it's a new session
    if session_id not in conversations:
        logger.info(f"New conversation started with {sender}")
        conversations[session_id] = [
            {"role": "system", "content": "You are the HuggingFace Model Management Agent. Decide whether to run models transiently (single-use) or persistently (long-running) based on the user request. Use the available tools: `run_transient_model` for one-off tasks, and `run_persistent_model` to start a model agent if persistence seems needed. Always report the outcome of the tool call back to the user."}
        ]

    # Process incoming text content
    user_input = None
    for item in msg.content:
        if isinstance(item, TextContent):
            user_input = item.text.strip()
            logger.info(f"Received message from {sender} (Session: {session_id}): {user_input}")
            # Append user message to history
            conversations[session_id].append({"role": "user", "content": user_input})
            break # Process only the first text content for now

    if not user_input:
        logger.warning(f"Received message from {sender} without text content.")
        await ctx.send(sender, ChatAcknowledgement(timestamp=datetime.now(timezone.utc), acknowledged_msg_id=msg.msg_id))
        return # Ignore messages without text

    # Send initial acknowledgment
    await ctx.send(sender, ChatAcknowledgement(timestamp=datetime.now(timezone.utc), acknowledged_msg_id=msg.msg_id))

    # --- LLM Interaction Loop ---
    max_turns = 5 # Limit agentic turns to prevent infinite loops
    for turn in range(max_turns):
        logger.info(f"Session {session_id} - Turn {turn + 1}")
        try:
            # Call ASI:One API
            response = requests.post(
                BASE_URL,
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={"model": MODEL_NAME, "messages": conversations[session_id], "tools": tools_list},
                timeout=120 # 2 minute timeout
            )
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            resp_data = response.json()

            # Process response message
            response_message = resp_data["choices"][0]["message"]
            conversations[session_id].append(response_message) # Add assistant's raw response to history

            # Check for tool calls
            if tool_calls := response_message.get("tool_calls"):
                logger.info(f"LLM requested tool calls: {[call['function']['name'] for call in tool_calls]}")
                tool_outputs = []
                for call in tool_calls:
                    func_name = call["function"]["name"]
                    tool_call_id = call["id"]
                    args_str = call["function"].get("arguments", "{}")

                    try:
                        args = json.loads(args_str)
                        logger.info(f"Executing tool '{func_name}' with args: {args}")
                        
                        # Execute the corresponding function
                        tool_function = available_tools.get(func_name)
                        if tool_function:
                            # Run the tool function (blocking for simplicity in this example)
                            # In a production agent, consider running tools in separate threads/processes
                            result = tool_function(**args)
                        else:
                            logger.error(f"Unknown tool requested: {func_name}")
                            result = {"status": "error", "message": f"Unknown tool: {func_name}"}

                        logger.info(f"Tool '{func_name}' result: {result}")
                        tool_outputs.append({
                            "tool_call_id": tool_call_id,
                            "role": "tool",
                            "name": func_name,
                            "content": json.dumps(result) # Ensure content is a JSON string
                        })

                    except json.JSONDecodeError:
                        logger.error(f"Failed to decode arguments for tool {func_name}: {args_str}")
                        tool_outputs.append({
                            "tool_call_id": tool_call_id, "role": "tool", "name": func_name,
                            "content": json.dumps({"status": "error", "message": "Invalid arguments format."})
                        })
                    except Exception as e:
                        logger.error(f"Error executing tool {func_name}: {e}", exc_info=True)
                        tool_outputs.append({
                            "tool_call_id": tool_call_id, "role": "tool", "name": func_name,
                            "content": json.dumps({"status": "error", "message": f"Tool execution failed: {str(e)}"})
                        })

                # Add tool results to history for the next LLM call
                conversations[session_id].extend(tool_outputs)
                # Continue the loop to let the LLM process the tool results

            # No tool calls, LLM gave a final answer
            else:
                final_answer = response_message.get("content")
                logger.info(f"LLM provided final answer: {final_answer}")
                # Send the final answer back to the user
                await ctx.send(sender, ChatMessage(content=[TextContent(type="text", text=final_answer)]))
                # Optional: Clear conversation history after final answer?
                # conversations.pop(session_id, None)
                return # Exit the loop and handler

        except requests.exceptions.Timeout:
            logger.error("ASI:One API request timed out.")
            await ctx.send(sender, ChatMessage(content=[TextContent(type="text", text="Sorry, the request to the AI model timed out.")]))
            return
        except requests.exceptions.RequestException as e:
            logger.error(f"ASI:One API request failed: {e}")
            await ctx.send(sender, ChatMessage(content=[TextContent(type="text", text=f"Sorry, there was an error communicating with the AI model: {e}")]))
            return
        except Exception as e:
            logger.error(f"An unexpected error occurred in LLM loop: {e}", exc_info=True)
            await ctx.send(sender, ChatMessage(content=[TextContent(type="text", text=f"Sorry, an unexpected error occurred: {e}")]))
            # Clean up potentially broken history state
            conversations.pop(session_id, None)
            return

    # If loop finishes without returning (e.g., max_turns reached)
    logger.warning(f"Max turns ({max_turns}) reached for session {session_id}. Sending partial response.")
    await ctx.send(sender, ChatMessage(content=[TextContent(type="text", text="Sorry, I couldn't complete the request within the allowed steps.")]))
    # Clean up history for the session
    conversations.pop(session_id, None)


class HFManagerChat(Model):
    ChatMessage: ChatMessage
    caller_Agent_address: str


# handle messages from ORC agent
@chat_proto.on_message(HFManagerChat)
async def handle_message_from_orc_agent(ctx: Context, sender: str, msg: HFManagerChat):
    session_id = sender 

    # Initialize conversation history if it's a new session
    if session_id not in conversations:
        logger.info(f"New conversation started with {sender}")
        conversations[session_id] = [
            {"role": "system", "content": "You are the HuggingFace Model Management Agent. Decide whether to run models transiently (single-use) or persistently (long-running) based on the user request. Use the available tools: `run_transient_model` for one-off tasks, and `run_persistent_model` to start a model agent if persistence seems needed. Always report the outcome of the tool call back to the user."}
        ]

    # Process incoming text content
    # just call tool based on prompt like 
    #<generate>

    
    
    # Clean up history for the session
    conversations.pop(session_id, None)



@chat_proto.on_message(ChatAcknowledgement)
async def handle_acknowledgement(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.debug(f"Received acknowledgement from {sender} for message: {msg.acknowledged_msg_id}")


agent.include(chat_proto, publish_manifest=True)

if __name__ == "__main__":
    logger.info(f"Starting HF Manager Agent on port {AGENT_PORT}...")
    agent.run()