import logging
import os
import re
from datetime import datetime, timezone
from uuid import uuid4, UUID
import json
from typing import Dict, Any, List

import requests
from dotenv import load_dotenv

from uagents import Agent, Context, Model, Protocol
from uagents.setup import fund_agent_if_low

# Import the chat protocol specification
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    EndSessionContent,
    StartSessionContent,
    TextContent,
    chat_protocol_spec,
)

# Import our MeTTa Knowledge Graph
# Assuming knowledge_graph.py is in Implementation/meTTa/
THRESHOLD_TO_DEPLOY_NEW_AGENT = 3

from Implementation.meTTa.knowledge_graph import OrchestratorKnowledgeGraph, 
    


# --- Configuration ---
load_dotenv()

# Agent Configuration
AGENT_SEED = os.getenv("ORCHESTRATOR_SEED", "hf_orchestrator_agent_secret_seed_phrase_1")
AGENT_PORT = int(os.getenv("ORCHESTRATOR_PORT", "8002")) # Use a different port

# ASI:One API Configuration (Optional, for LLM parsing)
API_KEY = os.getenv("ASI_ONE_API_KEY")
BASE_URL = "https://api.asi1.ai/v1/chat/completions"
MODEL_NAME = os.getenv("ASI_ONE_MODEL", "asi1-mini") # Use MODEL_NAME consistently

if not API_KEY:
    logging.warning("WARNING: ASI_ONE_API_KEY environment variable is not set. LLM parsing will be disabled.")
    # Agent can still function with regex fallback

# HF Manager Agent Address (Get this from running hf_manager_agent.py)
HF_MANAGER_ADDRESS = os.getenv("HF_MANAGER_ADDRESS", "agent1q04wcekamg3rzekxhnmh776jmkhlkd0s2p5dqpum7nz8ff6jd5yhvwprta3") # Default for testing

# Set up logging
logger = logging.getLogger("OrchestratorAgent")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Initialize Agent and Knowledge Graph
agent = Agent(
    name="hf_orchestrator_agent",
    seed=AGENT_SEED,
    port=AGENT_PORT,
    endpoint=[f"http://localhost:{AGENT_PORT}/submit"] # Assuming local for now
)
kg = OrchestratorKnowledgeGraph()


# --- Chat Protocol Setup ---

chat_proto = Protocol(spec=chat_protocol_spec)

def create_text_chat(text: str, end_session: bool = False) -> ChatMessage:
    """Helper function to create a text chat message for the user."""
    content = [TextContent(type="text", text=text)]
    if end_session:
        content.append(EndSessionContent(type="end-session"))
    return ChatMessage(
        timestamp=datetime.now(timezone.utc),
        msg_id=uuid4(),
        content=content,
    )

# --- Agent's Core Logic ---

def parse_user_query(user_query: str) -> (str | None, str | None):
    """
    Parses the user's free-text query using regex (LLM parsing commented out).
    """
    logger.info(f"Attempting to parse query: '{user_query}'")
    task = None
    prompt = None

    # --- Fallback Regex ---
    # generate using regex for now
    match = re.match(r"generate\s+([a-zA-Z0-9/_-]+)\s+(.+)", user_query, re.IGNORECASE) # Allow '/' in model ID task
    if match:
        task = match.group(1).lower().strip()
        prompt = match.group(2).strip()
        logger.info(f"Regex parsed query: task='{task}', prompt='{prompt[:50]}...'")
        return task, prompt

    logger.error(f"Could not parse task or prompt from query: {user_query}")
    return None, None


async def main_orchestrator_logic(ctx: Context, sender: str, user_query: str, kg: OrchestratorKnowledgeGraph):
    # 1. Parse the user's intent
    task, prompt = parse_user_query(user_query)

    if not task or not prompt: # Both are needed now
        logger.warning(f"Could not parse valid task and prompt from query: {user_query}")
        await ctx.send(sender, create_text_chat(f"Sorry, I couldn't understand the task or prompt. Please use the format: 'generate <task> <prompt>'."))
        return

    ctx.logger.info(f"Processing Request: task='{task}', prompt='{prompt[:50]}...'")
    request_id = str(uuid4()) # Unique ID for this specific interaction

    # --- Start Flowchart ---
    ctx.logger.info(f"Checking MeTTa for task: '{task}'")
    model_id = kg.find_model_for_task(task)

    if model_id:
        # --- PATH 1: Yes, I have knowledge ---
        ctx.logger.info(f"Knowledge found. Preferred Model ID: {model_id}")
        # Store sender and model_id for this request
        ctx.storage.set(request_id, json.dumps({"sender": sender, "model_id": model_id}))

        specialist_agent = kg.find_specialist_agent(model_id)

        if specialist_agent:
            # --- PATH 1a: Yes, agent exists ---
            ctx.logger.info(f"Specialist agent found: {specialist_agent}")
            ctx.logger.info(f"[ACTION] Forwarding prompt to specialist agent: {specialist_agent} (Request ID: {request_id})")

            # Send prompt + request_id to specialist via ChatMessage
            # Specialist needs to know how to handle this and include request_id in reply
            message_to_specialist = f"Request ID: {request_id}\nPrompt: {prompt}"
            await ctx.send(specialist_agent, ChatMessage(
                content=[TextContent(type="text", text=message_to_specialist)]
            ))
            await ctx.send(sender, create_text_chat(f"Found specialist agent for '{model_id}'. Forwarding your request..."))
            # We now wait for the specialist's reply in the handle_message function

        else:
            # --- PATH 1b: No, agent does not exist ---
            ctx.logger.info(f"No specialist agent found for '{model_id}'.")
            new_count = kg.increment_usage_count(model_id)
            ctx.logger.info(f"Incremented usage count for '{model_id}' to: {new_count}")

            # Decide whether to deploy or just run transiently
            if new_count >= THRESHOLD_TO_DEPLOY_NEW_AGENT:
                ctx.logger.info(f"Usage threshold ({THRESHOLD_TO_DEPLOY_NEW_AGENT}) reached!")
                ctx.logger.info(f"[ACTION] Requesting PERSISTENT run from HF Manager for '{model_id}' (Request ID: {request_id})")
                await ctx.send(sender, create_text_chat(f"Usage high for '{model_id}'. Requesting dedicated agent deployment..."))
                
                # Send instruction via ChatMessage
                instruction = (
                    f"Request ID: {request_id}\n"
                    f"Mode: persistent\n"
                    f"Model ID: {model_id}\n"
                    f"Task Type: {task}"
                )
                await ctx.send(HF_MANAGER_ADDRESS, create_text_chat(instruction))

            else:
                ctx.logger.info(f"[ACTION] Requesting TRANSIENT run from HF Manager for '{model_id}' (Request ID: {request_id})")
                await ctx.send(sender, create_text_chat(f"No specialist found. Requesting temporary run for '{model_id}'..."))

                # Send instruction via ChatMessage
                instruction = (
                    f"Request ID: {request_id}\n"
                    f"Mode: transient\n"
                    f"Model ID: {model_id}\n"
                    f"Task Type: {task}\n"
                    f"Prompt: {prompt}"
                )
                await ctx.send(HF_MANAGER_ADDRESS, create_text_chat(instruction))

    else:
        # --- PATH 2: No, I don't have knowledge ---
        ctx.logger.info(f"No knowledge found for new task: '{task}'")
        ctx.logger.info(f"[ACTION] Searching Hugging Face Hub for '{task}'...") # Simulation
        # TODO: Implement actual HF Hub search if desired
        simulated_new_model = f"hf-hub/simulated-model-for-{task}" # Simulation placeholder
        ctx.logger.info(f"Found new model (simulated): {simulated_new_model}")

        # Add this new knowledge to MeTTa and set its count to 1
        kg.add_new_task_model(task, simulated_new_model)
        ctx.logger.info(f"New task and model added to MeTTa. Initial count set to 1.")

        # Store sender and model_id for this request
        ctx.storage.set(request_id, json.dumps({"sender": sender, "model_id": simulated_new_model}))

        ctx.logger.info(f"[ACTION] Requesting TRANSIENT run from HF Manager for NEW model '{simulated_new_model}' (Request ID: {request_id})")
        await ctx.send(sender, create_text_chat(f"Found new model '{simulated_new_model}' (simulated). Requesting temporary run..."))
        
        # Send instruction via ChatMessage
        instruction = (
            f"Request ID: {request_id}\n"
            f"Mode: transient\n"
            f"Model ID: {simulated_new_model}\n"
            f"Task Type: {task}\n" # Use the parsed task
            f"Prompt: {prompt}"
        )
        await ctx.send(HF_MANAGER_ADDRESS, create_text_chat(instruction))

# --- Agent Message Handlers ---

def parse_manager_response(text: str) -> Dict[str, Any]:
    """Parses the text response from the HF Manager."""
    print(f"Parsing manager response...\n{text}")
    response_data = {"request_id": None, "status": "error", "message": "Could not parse manager response."}
    lines = text.strip().split('\n')
    try:
        # Extract Request ID first
        if lines[0].startswith("Request ID:"):
            response_data["request_id"] = lines[0].split(":", 1)[1].strip()
        else: return response_data # Cannot proceed without request ID

        # Parse based on expected keywords
        if "Transient run successful:" in text:
            response_data["status"] = "success"
            # Try to find JSON output block
            match_json = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
            if match_json:
                response_data["output"] = json.loads(match_json.group(1))
            else:
                 # Assume rest of the message after the status line is raw output
                response_data["output"] = text.split("Transient run successful:", 1)[1].strip()

        elif "Persistent agent deployed:" in text:
            response_data["status"] = "success"
            details = {}
            for line in lines[1:]: # Skip Request ID line
                if ":" in line:
                    key, value = line.split(":", 1)
                    details[key.strip().lower().replace(" ", "_")] = value.strip()
            response_data.update(details) # Add parsed details like agent_address, etc.

        elif "Error:" in text:
            response_data["status"] = "error"
            response_data["message"] = text.split("Error:", 1)[1].strip()

        else:
             response_data["message"] = f"Unrecognized response format from manager: {text[:100]}..."

    except json.JSONDecodeError as e:
        response_data["status"] = "error"
        response_data["message"] = f"Failed to parse JSON output from manager: {e}"
    except Exception as e:
        logger.error(f"Error parsing manager response text: {e}", exc_info=True)
        response_data["status"] = "error"
        response_data["message"] = f"Internal error parsing manager response: {e}"

    return response_data


@chat_proto.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    """
    Handle incoming chat messages from Users OR the HF Manager Agent.
    """
    # Try to acknowledge immediately
    try:
        await ctx.send(
            sender,
            ChatAcknowledgement(timestamp=datetime.now(timezone.utc), acknowledged_msg_id=msg.msg_id),
        )
    except Exception as e:
        ctx.logger.warning(f"Failed to send ACK to {sender}: {e}")

    # --- Differentiate Sender ---
    is_start_session = any(isinstance(item, StartSessionContent) for item in msg.content)
    text_content = next((item.text.strip() for item in msg.content if isinstance(item, TextContent)), None)

    # --- Case 1: Message from HF Manager Agent ---
    if sender == HF_MANAGER_ADDRESS:
        ctx.logger.info(f"Received reply from HF Manager: {text_content[:100]}...")
        if not text_content:
            ctx.logger.warning("Received empty message from HF Manager.")
            return

        # Parse the manager's text response
        parsed_response = parse_manager_response(text_content)
        request_id = parsed_response.get("request_id")

        if not request_id:
            ctx.logger.error(f"Could not extract request_id from manager response: {text_content[:100]}...")
            # Maybe send an error back to manager? For now, just log.
            return

        # Retrieve original sender and model_id
        stored_data_json = ctx.storage.get(request_id)
        if not stored_data_json:
            ctx.logger.warning(f"Could not find original sender/model_id for request ID: {request_id}. Ignoring manager response.")
            return

        try:
            stored_data = json.loads(stored_data_json)
            original_sender = stored_data.get("sender")
            model_id = stored_data.get("model_id") # Needed for KG update
            if not original_sender or not model_id:
                raise ValueError("Missing sender or model_id in stored data")
        except (json.JSONDecodeError, ValueError) as e:
             ctx.logger.error(f"Failed to parse stored data for request {request_id}: {e}. Data: {stored_data_json}")
             return

        # Process parsed response and reply to original user
        reply_message = "An unexpected response was received from the manager."
        if parsed_response["status"] == "success":
            if "output" in parsed_response: # Transient success
                ctx.logger.info(f"Transient run successful for {model_id} (Req ID: {request_id}).")
                # Format output potentially based on type
                if isinstance(parsed_response['output'], dict):
                     formatted_output = json.dumps(parsed_response['output'], indent=2)
                else:
                     formatted_output = str(parsed_response['output'])
                reply_message = f"Result for '{model_id}':\n```\n{formatted_output}\n```"

            elif "agent_address" in parsed_response: # Persistent success
                agent_addr = parsed_response.get("agent_address")
                agent_name = parsed_response.get("agent_name", "Unknown")
                endpoint = parsed_response.get("endpoint", "N/A")
                ctx.logger.info(f"Persistent agent deployed for {model_id} (Req ID: {request_id}). Address: {agent_addr}")

                # Register the new agent in our knowledge graph
                kg.register_specialist_agent(model_id, agent_addr)
                ctx.logger.info(f"Updated KG: Specialist for {model_id} is at {agent_addr}")

                reply_message = (
                    f"Successfully deployed a dedicated agent for '{model_id}'!\n"
                    f"Name: {agent_name}\n"
                    f"Address: {agent_addr}\n"
                    f"Endpoint: {endpoint}\n"
                    f"You can interact with it directly for future requests."
                )
            else:
                 ctx.logger.warning(f"Manager success response for {request_id} lacked output/agent details.")
                 reply_message = "Manager reported success, but no result details were provided."

        elif parsed_response["status"] == "error":
            ctx.logger.error(f"HF Manager error for request {request_id} ({model_id}): {parsed_response.get('message')}")
            reply_message = f"Sorry, there was an error processing your request for '{model_id}':\n{parsed_response.get('message')}"

        else:
            ctx.logger.warning(f"Received unknown status '{parsed_response.get('status')}' from Manager for request {request_id}.")
            reply_message = f"Received an unclear response from the processing agent."

        # Send the final reply to the original user
        await ctx.send(original_sender, create_text_chat(reply_message))

        # Clean up storage for this request ID
        ctx.storage.delete(request_id)

    # --- Case 2: Message is a Start Session from User ---
    elif is_start_session:
        ctx.logger.info(f"Got a start session message from USER {sender}")
        await ctx.send(sender, create_text_chat("Hello! I am the HF Orchestrator Agent. How can I help you generate content? (e.g., 'generate <task> <prompt>')"))

    # --- Case 3: Message is a Text Query from User ---
    elif text_content:
        ctx.logger.info(f"Processing query from USER {sender}: {text_content}")
        try:
            # Call the main logic function to handle the user's request
            await main_orchestrator_logic(ctx, sender, text_content, kg)
        except Exception as e:
            ctx.logger.error(f"Error processing user query: {e}", exc_info=True)
            await ctx.send(sender, create_text_chat(f"I'm sorry, an internal error occurred while processing your request."))

    # --- Case 4: Other (Ignore) ---
    else:
         ctx.logger.info(f"Got unexpected or empty content from {sender}")


@chat_proto.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    """Handle chat acknowledgements."""
    ctx.logger.debug(f"Got an acknowledgement from {sender} for {msg.acknowledged_msg_id}")

# Register the protocol
agent.include(chat_proto, publish_manifest=True)

# Fund agent if low
fund_agent_if_low(agent.wallet.address())

if __name__ == "__main__":
    logger.info(f"Starting agent '{agent.name}' on address: {agent.address} | Port: {AGENT_PORT}")
    agent.run()

