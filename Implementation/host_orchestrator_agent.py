import logging
import re
import os
import sys
import uvicorn
import asyncio
from dotenv import load_dotenv
from datetime import datetime, timezone
from uuid import uuid4
from uagents import Agent, Context, Model, Protocol, Bureau
from uagents.mailbox import MailboxClient
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    EndSessionContent,
    StartSessionContent,
    TextContent,
    chat_protocol_spec,
)
from fastapi import FastAPI
from uagents.asgi import ASGIAgent

# --- Import Project Files ---

# Add the project root to the path to find our modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from Implementation.meTTa.knowledge_graph import OrchestratorKnowledgeGraph, 
THRESHOLD_TO_DEPLOY_NEW_AGENT=5
# from Implementation.Tools.llm_parser import parse_intent_with_llm

# --- Agent Configuration ---

# Load environment variables from .env file
load_dotenv()

AGENT_SEED_PHRASE = os.environ.get("AGENT_SEED_PHRASE")
AGENT_MAILBOX_KEY = os.environ.get("AGENT_MAILBOX_KEY")
AGENT_NAME = "hf_orchestrator_agent"

if not AGENT_SEED_PHRASE:
    raise Exception("AGENT_SEED_PHRASE not found in .env file. Please set it.")
if not AGENT_MAILBOX_KEY:
    raise Exception("AGENT_MAILBOX_KEY not found in .env file. Please set it.")

# Set up logging
logger = logging.getLogger("OrchestratorAgent")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Initialize Knowledge Graph
kg = OrchestratorKnowledgeGraph()

# --- Chat Protocol Setup ---

chat_proto = Protocol(spec=chat_protocol_spec)

def create_text_chat(text: str, end_session: bool = False) -> ChatMessage:
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
    Parses the user's free-text query to extract the task and prompt.
    This is a simple mock of the "intent parsing" step.
    
    Expected format: "generate <task> <prompt>"
    """
    match = re.match(r"generate\s+([^\s]+)\s+(.+)", user_query, re.IGNORECASE)
    
    ## yaha pe query sahi se parse karna hai.....pass it to llm for better parsing and give all the available tasks and get prompt + task from user query
    
    
    if match:
        task = match.group(1).lower()  # e.g., "image-generation"
        prompt = match.group(2)       # e.g., "a red car"
        return task, prompt
    return None, None



async def main_orchestrator_logic(ctx: Context, sender: str, user_query: str, session_id: str, kg: OrchestratorKnowledgeGraph):
    
    ctx.logger.info(f"[{session_id}] Processing query: {user_query}")
    
    # 1. Parse the user's intent using the LLM parser
    try:
        task, prompt = parse_user_query(user_query)

        
    except Exception as e:
        ctx.logger.error(f"[{session_id}] Failed to parse intent: {e}")
        await ctx.send(sender, create_text_chat(f"Sorry, I had trouble understanding that. Could you rephrase? Error: {e}"))
        return

    ctx.logger.info(f"[{session_id}] Parsed intent: task='{task}', prompt='{prompt[:20]}...'")
    
    # --- Start Flowchart ---
    
    # 2. Check MeTTa for knowledge of this task
    ctx.logger.info(f"[{session_id}] Checking MeTTa for task: '{task}'")
    model_id = kg.find_model_for_task(task)
    
    if model_id:
        # --- PATH 1: Yes, I have knowledge ---
        ctx.logger.info(f"[{session_id}] Knowledge found. Preferred Model ID: {model_id}")
        
        specialist_agent = kg.find_specialist_agent(model_id)
        
        if specialist_agent:
            # --- PATH 1a: Yes, agent exists ---
            ctx.logger.info(f"[{session_id}] Specialist agent found: {specialist_agent}")
            ctx.logger.info(f"[ACTION] Forwarding prompt to specialist agent: {specialist_agent}")
            response_text = f"Specialist agent found for '{model_id}'. [Simulating call: Forwarded prompt...]"
            await ctx.send(sender, create_text_chat(response_text))
            
        else:
            # --- PATH 1b: No, agent does not exist ---
            ctx.logger.info(f"[{session_id}] No specialist agent found for '{model_id}'.")
            
            new_count = kg.increment_usage_count(model_id)
            ctx.logger.info(f"[{session_id}] Incremented usage count for '{model_id}' to: {new_count}")

            ctx.logger.info(f"[ACTION] Calling local 'hf_tool' to run '{model_id}'...")
            response_text = f"No specialist found. [Simulating call: Running '{model_id}' locally. Usage count: {new_count}]"
            await ctx.send(sender, create_text_chat(response_text))

            if new_count >= THRESHOLD_TO_DEPLOY_NEW_AGENT:
                ctx.logger.info(f"[{session_id}] Usage threshold ({THRESHOLD_TO_DEPLOY_NEW_AGENT}) reached!")
                ctx.logger.info(f"[ACTION] Calling 'provisioner.py' to deploy new agent for '{model_id}'...")
                
                new_agent_address = f"agent1q...simulated-addr-for-{model_id.split('/')[0]}"
                kg.register_specialist_agent(model_id, new_agent_address)
                
                ctx.logger.info(f"[{session_id}] New agent registered in MeTTa: {new_agent_address}")
                response_text_2 = f"Usage threshold reached! [Simulating call: Deployed new specialist agent for {model_id} at {new_agent_address}]"
                await ctx.send(sender, create_text_chat(response_text_2))
    
    else:
        # --- PATH 2: No, I don't have knowledge ---
        ctx.logger.info(f"[{session_id}] No knowledge found for new task: '{task}'")
        
        ctx.logger.info(f"[ACTION] Searching Hugging Face Hub for '{task}'...")
        
        simulated_new_model = f"hf-hub/new-model-for-{task}"
        ctx.logger.info(f"[{session_id}] Found new model: {simulated_new_model}")
        
        kg.add_new_task_model(task, simulated_new_model)
        ctx.logger.info(f"[{session_id}] New task and model added to MeTTa. Initial count set to 1.")
        
        ctx.logger.info(f"[ACTION] Calling local 'hf_tool' to run '{simulated_new_model}'...")
        response_text = f"Found new model '{simulated_new_model}' on HF Hub. [Simulating call: Running model locally...]"
        await ctx.send(sender, create_text_chat(response_text))

# --- Agent Message Handlers ---

@chat_proto.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    """Handle incoming chat messages."""
    
    # Generate a unique session ID for this conversation
    session_id = str(ctx.session)
    
    await ctx.send(
        sender,
        ChatAcknowledgement(timestamp=datetime.now(timezone.utc), acknowledged_msg_id=msg.msg_id),
    )

    for item in msg.content:
        if isinstance(item, StartSessionContent):
            ctx.logger.info(f"[{session_id}] Got a start session message from {sender}")
            await ctx.send(sender, create_text_chat("Hello! I am the HF Orchestrator Agent. How can I help?"))
            continue
        
        elif isinstance(item, TextContent):
            user_query = item.text.strip()
            ctx.logger.info(f"[{session_id}] Received query from {sender}: {user_query}")
            
            try:
                # Call the main logic function, passing the session_id
                await main_orchestrator_logic(ctx, sender, user_query, session_id, kg)
            
            except Exception as e:
                ctx.logger.error(f"[{session_id}] Error processing query: {e}", exc_info=True)
                await ctx.send(sender, create_text_chat(f"I'm sorry, an internal error occurred: {e}"))
        
        else:
            ctx.logger.info(f"[{session_id}] Got unexpected content from {sender}")

@chat_proto.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    """Handle chat acknowledgements."""
    ctx.logger.debug(f"Got an acknowledgement from {sender} for {msg.acknowledged_msg_id}")

# --- FastAPI Server Setup (as per docs) ---

# Create the agent with mailbox details
agent = Agent(
    name=AGENT_NAME,
    seed=AGENT_SEED_PHRASE,
    mailbox=AGENT_MAILBOX_KEY,
)

# Include the chat protocol
agent.include(chat_proto, publish_manifest=True)

# Create the FastAPI app
app = FastAPI()

# Create the ASGIAgent wrapper
agent_asgi = ASGIAgent(agent, http_rate_limit=10, process_rate_limit=10)

# Add the agent's endpoints to the FastAPI app
app.include_router(agent_asgi.router)

# --- Main block to run the server ---
if __name__ == "__main__":
    
    # This is the main entry point for the deployed agent
    
    PORT = int(os.environ.get("PORT", 8000))
    
    logger.info(f"Starting agent '{agent.name}' on port {PORT}")
    logger.info(f"Agent address: {agent.address}")
    
    # Run the FastAPI server with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)

