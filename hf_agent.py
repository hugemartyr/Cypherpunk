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

import subprocess
import sys
import os
import re  
import time 
import atexit 
import signal 
import psutil 
from typing import Dict, Any, List

from specialist_agent_runner_script import generate_persistent_runner_content
from transient_runner_script import generate_transient_runner_content

# --- Configuration ---
dotenv.load_dotenv()
logger = logging.getLogger("HFManagerAgent")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Agent Configuration
AGENT_SEED = os.getenv("HF_MANAGER_SEED", "hf_manager_agent_secret_seed_phrase_abc")
AGENT_PORT = int(os.getenv("HF_MANAGER_PORT", "8000"))

# ASI:One API Configuration
API_KEY = os.getenv("ASI_ONE_API_KEY")
BASE_URL = "https://api.asi1.ai/v1/chat/completions"
MODEL_NAME = "asi1-mini" # Use the mini model

if not API_KEY:
    logger.error("FATAL ERROR: ASI_ONE_API_KEY environment variable is not set.")
    sys.exit(1)

# Define the directory for persistent models relative to this script

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "Agent_Models")) 
os.makedirs(MODELS_DIR, exist_ok=True)
curr_port = 6000 + 1 # Start ports for specialists after the manager's port



# --- Child Process Management ---
child_processes: List[psutil.Process] = []
def cleanup_child_processes():
    """
    Terminates all child processes started by this script.
    """
    logger.info(f"Cleaning up {len(child_processes)} child process(es)...")
    for proc in child_processes:
        try:
            # Find all children of the process (the agent might spawn its own)
            children = proc.children(recursive=True)
            for child in children:
                logger.debug(f"Terminating grandchild process {child.pid}")
                child.terminate()
            
            # Terminate the main child process
            logger.debug(f"Terminating child process {proc.pid}")
            proc.terminate()
            
            proc.wait(timeout=3) # Wait max 3 seconds
        except psutil.NoSuchProcess:
            logger.debug(f"Process {proc.pid} already terminated.")
        except psutil.TimeoutExpired:
            logger.warning(f"Process {proc.pid} did not terminate, killing.")
            proc.kill()
        except Exception as e:
            logger.error(f"Error during cleanup of process {proc.pid}: {e}")

# Register the cleanup function to run on normal script exit
atexit.register(cleanup_child_processes)

def signal_handler(sig, frame):
    """Handle signals like SIGINT (Ctrl+C)."""
    logger.info("Signal received, initiating cleanup and exit...")
    # The atexit handler will automatically run
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# --- Data Models for ORC Agent Communication ---
class HFManagerChat(Model):
    ChatMessage: ChatMessage
    caller_Agent_address: str

class HFManagerChatAcknowledgement(Model):
    ChatAcknowledgement: ChatAcknowledgement
    caller_Agent_address: str

def run_transient_model(model_id: str, prompt: str, task_type: str = "auto") -> Dict[str, Any]:
    """
    Executes Path A: Transient Run (single-use inference).
    Logs all subprocess output (stdout and stderr) for debugging.
    """

    model_id = model_id.replace('"', '') 
    print(f"[DEBUG] Running transient model: {model_id} with prompt: {prompt} and task_type: {task_type}")
    filename = f"hf_transient_runner_{uuid4()}.py" 
    runner_path = os.path.join(SCRIPT_DIR, filename)
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
        
        try:
            output_json = json.loads(stdout)
            if isinstance(output_json, dict) and "status" not in output_json:
                 output_json["status"] = "success"
            logger.info(f"Transient run successful for {model_id}.")
            return output_json
        except json.JSONDecodeError:
            logger.error(f"Transient runner output for {model_id} was NOT valid JSON.")
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
    """
    Executes Path B: Persistent Run.
    Starts a long-running specialist agent as a subprocess,
    captures its address, and ensures it's cleaned up on exit.
    """

    model_id = model_id.replace('"', '')
    print(f"[DEBUG] Running persistent model: {model_id} with task_type: {task_type}")
    global curr_port
    port_to_use = curr_port
    curr_port += 1 # Increment for the next one

    logger.info(f"Attempting to start persistent agent for {model_id} on port {port_to_use}")

    # --- 1. Create the Runner Script ---
    model_safe = model_id.replace("/", "__").replace("-", "_")
    model_safe += "_model"
    model_dir = os.path.join(MODELS_DIR, model_safe)
    os.makedirs(model_dir, exist_ok=True)

    runner_file = os.path.join(model_dir, "runner.py")
    log_file = os.path.join(model_dir, "log.txt")

    script_content = generate_persistent_runner_content(model_id,  port_to_use, task_type, log_file=log_file)
    if not script_content:
        error_msg = f"Failed to generate persistent runner script content for {model_id}"
        logger.error(error_msg)
        return {"status": "error", "message": error_msg}

    with open(runner_file, "w", encoding="utf-8") as f:
        f.write(script_content)

    # --- 2. Launch the Subprocess ---
    python_executable = sys.executable
    command = [python_executable, runner_file]
    
    # Regex to capture the agent address
    address_regex = re.compile(r"Starting agent with address: (agent1[a-zA-Z0-9]+)")
    agent_address = None
    STARTUP_TIMEOUT = 20 # 20 seconds
    
    try:
        process = subprocess.Popen(
            command,
            cwd=model_dir,
            stdout=subprocess.PIPE,    # Capture stdout
            stderr=subprocess.PIPE,    # Capture stderr
            text=True,                 # Decode as text 
            bufsize=1,                 # Line-buffered
        )
        
        # Add to global list for cleanup
        child_processes.append(psutil.Process(process.pid))
        logger.info(f"Started persistent agent process for {model_id} with PID {process.pid}")

        start_time = time.time()
        
        # --- 3. Read STDOUT to find the address ---
        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            if not line:
                continue
                
            logger.info(f"[Runner STDOUT - {model_id}]: {line}") # Log runner output
            
            # Check for address
            match = address_regex.search(line)
            if match:
                agent_address = match.group(1)
                logger.info(f"Successfully captured agent address for {model_id}: {agent_address}")
                break # Success!
            
            # Check for timeout
            if time.time() - start_time > STARTUP_TIMEOUT:
                logger.error(f"Timeout: Agent {model_id} (PID {process.pid}) failed to report address in {STARTUP_TIMEOUT}s.")
                # We kill it here; atexit will handle the rest
                process.kill()
                stderr_output = process.stderr.read()
                return {"status": "error", "message": f"Timeout waiting for agent to start. Stderr: {stderr_output}"}
            
            # Check if process died prematurely
            if process.poll() is not None:
                logger.error(f"Agent process {model_id} (PID {process.pid}) terminated prematurely.")
                break # Exit loop
        
        # --- 4. Handle Results ---
        if agent_address is None:
            # Process terminated before address was found
            stderr_output = process.stderr.read()
            logger.error(f"Agent {model_id} failed to start. Return code: {process.returncode}. Stderr: {stderr_output}")
            return {"status": "error", "message": f"Agent failed to start. Stderr: {stderr_output}"}
        
        

        # Return success with the agent address
        return {
            "status": "success",
            "message": f"Model {model_id} started in persistent mode.",
            "agent_name": f"{model_id.replace('/', '_')}_specialist",
            "agent_address": agent_address, # The captured address
            "port": port_to_use,
            "endpoint": f"http://localhost:{port_to_use}/submit",
            "log_file": log_file,
            "pid": process.pid
        }
    except Exception as e:
        logger.error(f"Error starting persistent agent for {model_id}: {e}", exc_info=True)
        return {"status": "error", "message": f"Failed to start persistent process: {str(e)}"}    

# --- Tool Schemas for ASI:One ---
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
new_chat_protocol = Protocol("AgentChatProtocol", "0.3.0")
chat_proto = Protocol(spec=chat_protocol_spec)
conversations: Dict[str, List[Dict[str, Any]]] = {}

# --- Agent Event Handlers ---
@agent.on_event('startup')
async def startup_handler(ctx: Context):
    ctx.logger.info(f'My name is {ctx.agent.name} and my address is {ctx.agent.address}')
    ctx.logger.info(f'Agent running on http://localhost:{AGENT_PORT}')

# handle chat from normal User
@chat_proto.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    """
    Handle incoming chat messages from users.
    Maintains conversation history per session.
    Interacts with ASI:One API to process messages and execute tools as needed.
    """
    session_id = sender 
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
                        
                        # send the response back to user
                        await ctx.send(sender, ChatMessage(content=[TextContent(type="text", text=json.dumps(result['output'] if 'output' in result else result))]))
                        conversations.pop(session_id, None)
                        return # Exit after tool execution and response

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
                conversations[session_id].extend(tool_outputs)
            else:
                final_answer = response_message.get("content")
                logger.info(f"LLM provided final answer: {final_answer}")
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
    
@chat_proto.on_message(ChatAcknowledgement)
async def handle_acknowledgement(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.debug(f"Received acknowledgement from {sender} for message: {msg.acknowledged_msg_id}")


# handle messages from ORC agent
@new_chat_protocol.on_message(HFManagerChat)
async def handle_message_from_orc_agent(ctx: Context, sender: str, msg: HFManagerChat):
    """
    Handle messages from the Orchestrator (ORC) agent.
    Expects commands in the format:
    <generate> <type> <model_id> <prompt/task>
    where <type> is either 'transient' or 'persistent'."""
    print(f"[DEBUG] Handling message from ORC agent: {msg}")

    # 1. Get the command text
    try:
        # Assuming the command is in the first content block
        command_text = msg.ChatMessage.content[0].text.strip()
        logger.info(f"Received command from {sender}: {command_text}")
    except (IndexError, AttributeError):
        error_msg = "Error: Received message with invalid or empty content."
        logger.error(error_msg)
        await ctx.send(msg.caller_Agent_address, ChatMessage(content=[TextContent(text=error_msg)]))
        await ctx.send(sender, HFManagerChat(ChatMessage(content=[TextContent(text=error_msg)]), caller_Agent_address=msg.caller_Agent_address))
        return

    # 2. Parse the command
    if not command_text.startswith("<generate>"):
        error_msg = "Error: Command must start with <generate>."
        logger.warning(f"Invalid command from {sender}: {command_text}")
        await ctx.send(msg.caller_Agent_address, ChatMessage(content=[TextContent(text=error_msg)]))
        await ctx.send(sender, HFManagerChat(ChatMessage(content=[TextContent(text=error_msg)]), caller_Agent_address=msg.caller_Agent_address))
        return

    # Split: <generate> <type> <model_id> <prompt/task>
    parts = command_text.split(maxsplit=3)
    
    tool_result = None
    error_msg = None
    
    print(f"[DEBUG] Command parts: {parts}")

    # 3. Execute the correct tool
    try:
        if len(parts) < 3:
            error_msg = "Error: Invalid command. Expected: <generate> <type> <model_id> [prompt]"
        else:
            tool_type = parts[1].lower()
            model_id = parts[2]

            if tool_type == "transient":
                if len(parts) == 4:
                    prompt = parts[3]
                    tool_result =  run_transient_model(model_id=model_id, prompt=prompt)
                else:
                    error_msg = "Error: 'transient' tool requires a prompt. Expected: <generate> transient <model_id> <prompt>"
                    
                    
            
            elif tool_type == "persistent":
                # Note: This ignores parts[3] (the prompt) if provided,
                # as the persistent tool doesn't take one.
                repr(model_id)
                tool_result =  run_persistent_model(model_id=model_id)
                
                new_deployed_agent_address = tool_result.get("agent_address", "unknown_address")
                if(new_deployed_agent_address != "unknown_address"):
                    response_text = f"Hugging Face Persistent Model Agent started at address: {new_deployed_agent_address} for model {model_id}"
                    logger.info(f"Sending response to original caller {msg.caller_Agent_address}: {response_text}")
                    await ctx.send(sender, # ORC agent : Send Agent address back
                                   HFManagerChat(ChatMessage= ChatMessage(content=[TextContent(text=response_text)]), caller_Agent_address=msg.caller_Agent_address))
                
            else:
                error_msg = f"Error: Unknown tool type '{tool_type}'. Must be 'transient' or 'persistent'."

    except Exception as e:
        logger.error(f"Error during tool execution for {sender}: {e}")
        error_msg = f"Error: Tool execution failed. {e}"

    # 4. Formulate and send the response
    if tool_result:
        response_text = f"Hugging Face Model Response: {tool_result.get('output', 'No result string.')}"
    else:
        response_text = error_msg or "Error: An unknown error occurred."

    if tool_type == "transient":
        # Send response directly to the original caller
        logger.info(f"Sending response to original caller {msg.caller_Agent_address}: {response_text}")
        await ctx.send(msg.caller_Agent_address, ChatMessage(content=[TextContent(text=response_text)]))
        
    return    
    

@new_chat_protocol.on_message(HFManagerChatAcknowledgement)
async def handle_acknowledgement_of_Manager(ctx: Context, sender: str, msg: HFManagerChatAcknowledgement):
    ctx.logger.debug(f"Received acknowledgement from {sender} for message: {msg.ChatAcknowledgement.acknowledged_msg_id}")


agent.include(chat_proto, publish_manifest=True)
agent.include(new_chat_protocol, publish_manifest=True)

if __name__ == "__main__":
    logger.info(f"Starting HF Manager Agent on port {AGENT_PORT}...")
    agent.run()