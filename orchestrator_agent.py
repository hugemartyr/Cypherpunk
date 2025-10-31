import logging
import os
import re
from datetime import datetime, timezone
import threading
from uuid import uuid4
from uagents import Agent, Context, Model, Protocol
from uagents.setup import fund_agent_if_low
from dotenv import load_dotenv
import time

load_dotenv()

from fastapi import FastAPI
import requests

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
from knowledge_graph import OrchestratorKnowledgeGraph, print_all_atoms

THRESHOLD_TO_DEPLOY_NEW_AGENT = 2

# --- Agent Configuration ---

AGENT_SEED = "hf_orchestrator_agent_secret_seed_phrase_2"
API_KEY = os.getenv("ASI_ONE_API_KEY")
BASE_URL = "https://api.asi1.ai/v1/chat/completions"
MODEL = "asi1-mini"

AGENT_MAILBOX_BEARER_TOKEN = os.getenv("AGENTVERSE_MAILBOX_KEY")

# Set up logging
logger = logging.getLogger("OrchestratorAgent")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Initialize Agent and Knowledge Graph
# agent = Agent(name="hf_orchestrator_agent", seed=AGENT_SEED,port=8001)
AGENT_PORT = 5080
agent = Agent(
    name="hf_orchestrator_agent",
    seed=AGENT_SEED,
    port=AGENT_PORT,
    endpoint=[f"http://localhost:{AGENT_PORT}/submit"], # Assuming local for now,
    mailbox=True
    
)
kg = OrchestratorKnowledgeGraph()


# --- Chat Protocol Setup (from your example) ---
new_chat_protocol = Protocol("AgentChatProtocol", "0.3.0")
chat_proto = Protocol(spec=chat_protocol_spec)

hf_manager_address = "agent1q04wcekamg3rzekxhnmh776jmkhlkd0s2p5dqpum7nz8ff6jd5yhvwprta3"

class HFManagerChat(Model):
    ChatMessage: ChatMessage
    caller_Agent_address: str
class HFManagerChatAcknowledgement(Model):
    ChatAcknowledgement: ChatAcknowledgement
    caller_Agent_address: str
    


def create_text_chat(text: str, end_session: bool = False) -> ChatMessage:
    """Helper function to create a text chat message."""
    content = [TextContent(type="text", text=text)]
    if end_session:
        content.append(EndSessionContent(type="end-session"))
    return ChatMessage(
        timestamp=datetime.now(timezone.utc),
        msg_id=uuid4(),
        content=content,
    )
def create_text_hf_agent_chat(text: str, caller_agent_address: str, end_session: bool = False) -> HFManagerChat:
    """
    Helper function to create a text chat message for the HFManagerAgent.

    """
    content = [TextContent(type="text", text=text)]
    
    return HFManagerChat(
        ChatMessage=ChatMessage(
            timestamp=datetime.utcnow(),
            msg_id=uuid4(), 
            content=content,
        ),  
        caller_Agent_address=caller_agent_address
    )

# --- Agent's Core Logic ---
def parse_user_query(user_query: str) -> (str | None, str | None):
    # """
    # Parses the user's free-text query to extract the task and prompt.
    
    # pass query to llm to parse it better
    # """
    
    # match = re.match(r"generate\s+([^\s]+)\s+(.+)", user_query, re.IGNORECASE)
    
    # messages=[
    #     {"role": "system", "content": "You are an expert at parsing user queries into structured tasks and prompts."},
    #     {"role": "user", "content": f"Parse the following user query into a task and prompt: '{user_query}'.\n"
    #                                 "Return only the task and prompt in the format: task:<task_name>; prompt:<prompt_text>."}
    # ]
    # task_prompt_response = requests.post(
    #     BASE_URL,
    #     headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
    #     json={
    #         "model": MODEL,
    #         "messages": messages
    #     }
    # )

    # if task_prompt_response.status_code == 200:
    #     response_data = task_prompt_response.json()
    #     # Extract task and prompt from the response
    #     task = response_data.get("task")
    #     prompt = response_data.get("prompt")
    #     return task, prompt
    # else:
    #     logger.error(f"Failed to parse user query: {task_prompt_response.text}")
    #     return None, None
    
    # generate using regex for now
    match = re.match(r"generate\s+([^\s]+)\s+(.+)", user_query, re.IGNORECASE)
    if match:
        task = match.group(1).strip()
        prompt = match.group(2).strip()
        return task, prompt
    return None, None

async def main_orchestrator_logic(ctx: Context, sender: str, user_query: str, kg: OrchestratorKnowledgeGraph):    
    # 1. Parse the user's intent
    task, prompt = parse_user_query(user_query)
    
    if not task:
        logger.warning(f"Could not parse task from query: {user_query}")
        await ctx.send(sender, create_text_chat(f"Sorry, I don't understand. Please use the format: 'generate <task> <prompt>'"))
        return
    
    ctx.logger.info(f"Parsed query: task='{task}', prompt='{prompt[:20]}...'")
    
    # --- Start Flowchart ---
    
    # 2. Check MeTTa for knowledge of this task
    ctx.logger.info(f"Checking MeTTa for task: '{task}'")
    model_id = kg.find_model_for_task(task)
    
    if model_id:
        # --- PATH 1: Yes, I have knowledge ---
        ctx.logger.info(f"Knowledge found. Preferred Model ID: {model_id}")
        
        # 3. Check if a specialist agent exists for this model
        repr(model_id)
        specialist_agent = kg.find_specialist_agent(model_id)
        
        if specialist_agent:
            # --- PATH 1a: Yes, agent exists ---
            ctx.logger.info(f"Specialist agent found: {specialist_agent}")
            specialist_agent = specialist_agent.strip()
            repr(specialist_agent)
            specialist_agent = specialist_agent.replace('"','')
            
            # [ACTION] Pass prompt to that agent
            ctx.logger.info(f"[ACTION] Forwarding prompt to specialist agent: {specialist_agent}")
            response_text = f"{prompt}"
            ctx.logger.info(f"[ACTION] Sending {prompt} to specialist agent: {specialist_agent}")
            # await ctx.send(specialist_agent, create_text_hf_agent_chat(response_text, caller_agent_address=sender))
            initial_message = HFManagerChat(
                ChatMessage=ChatMessage(
                    timestamp=datetime.utcnow(),
                    msg_id=uuid4(),
                    content=[TextContent(type="text", text=response_text)],
                ),
                caller_Agent_address=sender
            )

            await ctx.send(specialist_agent, initial_message)
            ctx.logger.info(f"Prompt sent to specialist agent: {specialist_agent}")

        else:
            # --- PATH 1b: No, agent does not exist ---
            ctx.logger.info(f"No specialist agent found for '{model_id}'.")
            
            # 4. Increment usage count for this model
            new_count = kg.increment_usage_count(model_id)
            ctx.logger.info(f"Incremented usage count for '{model_id}' to: {new_count}")

            # [ACTION] Tell HF_agent (our local tool) to build and run the model
            ctx.logger.info(f"[ACTION] Calling local 'hf_tool' to run '{model_id}'...")
            response_text = f"generate transient {model_id} '{prompt}'"
            # await ctx.send(hf_manager_address, create_text_chat(response_text))
            transient_command = f"<generate> transient {model_id} {prompt}"
    
            ctx.logger.info(f"Sending command: {transient_command}")
            await ctx.send(
                hf_manager_address, 
                create_text_hf_agent_chat(
                    text=transient_command, 
                    caller_agent_address=sender
                )
            )
            
            # 5. Check if usage count meets the threshold to deploy
            if new_count == THRESHOLD_TO_DEPLOY_NEW_AGENT: 
                ctx.logger.info(f"Usage threshold ({THRESHOLD_TO_DEPLOY_NEW_AGENT}) reached!")
                
                # [ACTION] Tell HF_agent (our deploy tool) to deploy this model
                ctx.logger.info(f"[ACTION] Calling 'provisioner.py' to deploy new agent for '{model_id}'...")
                persistent_command = f"<generate> persistent {model_id}"
    
                ctx.logger.info(f"Sending command: {persistent_command}")
                await ctx.send(
                    hf_manager_address, 
                    create_text_hf_agent_chat(
                        text=persistent_command, 
                        caller_agent_address=sender
                    )
                )
                
                
                # here we will recieve message from HF_agent about new deployed agent address
                
                
                # # Simulate a new address and update the knowledge graph
                # new_agent_address = f"agent1q...simulated-addr-for-new-agent"
                # kg.register_specialist_agent(model_id, new_agent_address)
                
                # ctx.logger.info(f"New agent registered in MeTTa: {new_agent_address}")
                # response_text_2 = f"generate a persistant chat model with HF model_id = '{model_id}' and task_type = 'auto' and give me the address of new deployed agent."
                # await ctx.send(hf_manager_address, create_text_chat(response_text_2))
    
    else:
        # --- PATH 2: No, I don't have knowledge ---
        ctx.logger.info(f"No knowledge found for new task: '{task}'")
        
        # [ACTION] Search Hugging Face Hub for the best model for this task
        ctx.logger.info(f"[ACTION] Searching Hugging Face Hub for '{task}'...")
        
        
        
        
        # Simulate finding a new model
        simulated_new_model = f"hf-hub/new-model-for-{task}"
        ctx.logger.info(f"Found new model: {simulated_new_model}")
        
        # Add this new knowledge to MeTTa and set its count to 1
        kg.add_new_task_model(task, simulated_new_model)
        ctx.logger.info(f"New task and model added to MeTTa. Initial count set to 1.")
        
        # [ACTION] Call the local HF_agent to run this new model
        ctx.logger.info(f"[ACTION] Calling local 'hf_tool' to run '{simulated_new_model}'...")
        response_text = f"Found new model '{simulated_new_model}' on HF Hub. [Simulating call: Running model locally...]"
        await ctx.send(sender, create_text_chat(response_text))


def register_agent_with_agentverse(
    agent_address: str,
    bearer_token: str,
    port: int,
    agent_name: str,
    description: str,
    readme_content: str = None
):
    """
    Connects a local agent to its Agentverse mailbox and updates its
    public profile on agentverse.ai.
    """
    if not bearer_token:
        print("AGENTVERSE_API_KEY not set, skipping registration.")
        return

    print(f"Agent '{agent_name}' starting registration...")

    # Give the agent's local server time to start up
    # This is crucial for the /connect call
    time.sleep(8)

    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "Content-Type": "application/json",
    }

    # --- Step 1: Connect agent's local server to the mailbox ---
    connect_url = f"http://127.0.0.1:{port}/connect"
    connect_payload = {
        "agent_type": "mailbox",
        "user_token": bearer_token
    }

    try:
        connect_response = requests.post(
            connect_url, json=connect_payload, headers=headers, timeout=10
        )
        if connect_response.status_code in [200, 201]:
            print(f"Successfully connected '{agent_name}' to Agentverse mailbox.")
        else:
            print(
                f"Failed to connect '{agent_name}' to mailbox: "
                f"{connect_response.status_code} - {connect_response.text}"
            )
            # Don't stop, agent might already be connected.
            # Try to register/update anyway.
    except Exception as e:
        print(f"Error connecting '{agent_name}' to Agentverse: {str(e)}")
        print("Will still attempt to register/update profile...")
    
    # --- Step 2: Register agent (creates profile if it doesn't exist) ---
    print(f"Registering '{agent_name}' with Agentverse API...")
    register_url = "https://agentverse.ai/v1/agents"
    register_payload = {
        "address": agent_address,
        "agent_type": "mailbox"
    }

    try:
        requests.post(
            register_url, json=register_payload, headers=headers, timeout=10
        )
        # We ignore the response. If it's 200/201 (created) or 409 (conflict),
        # it's fine. We just want to ensure it exists before updating.
    except Exception as e:
        print(f"Error during initial registration (continuing to update): {str(e)}")

    # --- Step 3: Update agent's public profile (README, etc.) ---
    print(f"Updating '{agent_name}' README on Agentverse...")
    update_url = f"https://agentverse.ai/v1/agents/{agent_address}"

    # Use default README if none provided
    if not readme_content:
        readme_content = f"""
# {agent_name}
{description}

This is a Orchestrator Agent that manages Hugging Face models.
- **Address:** `{agent_address}`
- **Protocols:** ChatProtocol
"""

    update_payload = {
        "name": agent_name,
        "readme": readme_content,
        "short_description": description,
    }

    try:
        update_response = requests.put(
            update_url, json=update_payload, headers=headers, timeout=10
        )
        if update_response.status_code == 200:
            print(f"Successfully updated '{agent_name}' profile on Agentverse.")
        else:
            print(
                f"Failed to update '{agent_name}' profile: "
                f"{update_response.status_code} - {update_response.text}"
            )
    except Exception as e:
        print(f"Error updating '{agent_name}' profile: {str(e)}")

    print(f"Agent '{agent_name}' registration complete!")




@agent.on_event("startup")
async def startup_handler(ctx: Context):
    """Handles agent startup and triggers registration."""
    ctx.logger.info(f"My name is {ctx.agent.name} and my address is {ctx.agent.address}")
    ctx.logger.info(f"Local server running on port: {AGENT_PORT}")

   
    agent_info = {
        "agent_address": ctx.agent.address,
        "bearer_token": AGENT_MAILBOX_BEARER_TOKEN,
        "port": AGENT_PORT,
        "agent_name": ctx.agent.name,
        "description": "An Orchestrator agent To Manage Hugging Face models.",
        "readme_content": None,
    }

    # Run registration in a separate thread to avoid blocking the agent
    registration_thread = threading.Thread(
        target=register_agent_with_agentverse,
        kwargs=agent_info
    )
    registration_thread.start()

# --- Agent Message Handlers ---
@chat_proto.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    """Handle incoming chat messages."""
    ctx.storage.set(str(ctx.session), sender)
    await ctx.send(
        sender,
        ChatAcknowledgement(timestamp=datetime.now(timezone.utc), acknowledged_msg_id=msg.msg_id),
    )
    for item in msg.content:
        if isinstance(item, StartSessionContent):
            ctx.logger.info(f"Got a start session message from {sender}")
            await ctx.send(sender, create_text_chat("Hello! I am the HF Orchestrator Agent. How can I help?"))
            continue
        
        elif isinstance(item, TextContent):
            user_query = item.text.strip()
            ctx.logger.info(f"Received query from {sender}: {user_query}")
            
            try:
                # Call the main logic function
                await main_orchestrator_logic(ctx, sender, user_query, kg)
            
            except Exception as e:
                ctx.logger.error(f"Error processing query: {e}", exc_info=True)
                await ctx.send(sender, create_text_chat(f"I'm sorry, an internal error occurred: {e}"))
        
        else:
            ctx.logger.info(f"Got unexpected content from {sender}")

@chat_proto.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    """Handle chat acknowledgements."""
    ctx.logger.info(f"Got an acknowledgement from {sender} for {msg.acknowledged_msg_id}")

@new_chat_protocol.on_message(HFManagerChat)
async def handle_message_hf_manager(ctx: Context, sender: str, msg: HFManagerChat):
    """Handle chat messages from HF Manager agent."""
    
    pass
    # just check if 
    for item in msg.ChatMessage.content:
        if isinstance(item, TextContent):
            ctx.logger.info(f"Got message from HF Manager: {item.text}")
            if(item.text=="Tool executed successfully."):
                ctx.logger.info(f"Tool executed successfully.")
                # # send ack to caller agent
                # await ctx.send(msg.caller_Agent_address, ChatMessage(content=[TextContent(type="text", text="Tool executed successfully.")]))
                
            if("Hugging Face Persistent Model Agent started at address:" in item.text):
                
                ctx.logger.info(f"Message from HF Manager: {item.text}")
                # Extract the address from the message
                match = re.search(r"Hugging Face Persistent Model Agent started at address: (\S+) for model (\S+)", item.text)
                if match:
                    new_agent_address = match.group(1)
                    model_id = match.group(2)
                    repr(model_id)
                    ctx.logger.info(f"New agent address: {new_agent_address} for model: {model_id}")
                    # add address to metta kg
                    kg.register_specialist_agent(model_id, new_agent_address)
                    print("\n\n\n\n")
                    print_all_atoms(kg.metta)
                    print("\n\n\n\n")


@new_chat_protocol.on_message(HFManagerChatAcknowledgement)
async def handle_ack_of_manager(ctx: Context, sender: str, msg: HFManagerChatAcknowledgement):
    """Handle chat acknowledgements."""
    ctx.logger.info(f"Got an acknowledgement from {sender} for {msg.ChatAcknowledgement.acknowledged_msg_id}")


# Register the protocol
agent.include(chat_proto, publish_manifest=True)
agent.include(new_chat_protocol, publish_manifest=True)


fund_agent_if_low(agent.wallet.address())

if __name__ == "__main__":
    pass
    # To run the actual agent (uncomment this):
    logger.info(f"Starting agent '{agent.name}' on address: {agent.address}")
    agent.run()

