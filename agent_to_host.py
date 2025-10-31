# from datetime import datetime
# from uuid import uuid4
# from uagents import Agent, Protocol, Context, Model
# from time import sleep

# #import the necessary components from the chat protocol
# from uagents_core.contrib.protocols.chat import (
#     ChatAcknowledgement,
#     ChatMessage,
#     TextContent,
#     chat_protocol_spec,
# )

# # Intialise agent1
# agent1 = Agent(
#     name="Agent1",
#     seed="NewAgentSeedValueForDeterministicAddress12345",
#     port=5060,
#     endpoint=["http://localhost:5060/submit"],
# )


# agent2_address = "agent1qgl8etxfyrhrdqasrukt6xm23gs9lde2dsdy5zrft59tphf4592m5p082ut"

# # Initialize the chat protocol
# chat_proto = Protocol(spec=chat_protocol_spec)



# #Startup Handler - Print agent details and send initial message
# @agent1.on_event("startup")
# async def startup_handler(ctx: Context):
#     # Print agent details
#     ctx.logger.info(f"My name is {ctx.agent.name} and my address is {ctx.agent.address}")
    
#     # Send initial message to agent2
#     # initial_message = ChatMessage(
#     #     timestamp=datetime.utcnow(),
#     #     msg_id=uuid4(),
#     #     content=[TextContent(type="text", text="generate text-summarization history of artificial intelligence in brief using the best hugging face models, use necessary tools and follow main_orchestrator logic")],
#     # )
    
#     # await ctx.send(agent2_address, initial_message)

# # Message Handler - Process received messages and send acknowledgements
# @chat_proto.on_message(ChatMessage)
# async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
#     for item in msg.content:
#         if isinstance(item, TextContent):
#             # Log received message
#             ctx.logger.info(f"Received message from {sender}: {item.text}")
            

            
        
# # Acknowledgement Handler - Process received acknowledgements
# @chat_proto.on_message(ChatAcknowledgement)
# async def handle_acknowledgement(ctx: Context, sender: str, msg: ChatAcknowledgement):
#     ctx.logger.info(f"Received acknowledgement from {sender} for message: {msg.acknowledged_msg_id}")



# # Include the protocol in the agent to enable the chat functionality
# # This allows the agent to send/receive messages and handle acknowledgements using the chat protocol
# agent1.include(chat_proto, publish_manifest=True)

# if __name__ == '__main__':
#     agent1.run()

import os
import time
import requests
import threading
from datetime import datetime
from uuid import uuid4
from uagents import Agent, Protocol, Context, Model
from uagents.setup import fund_agent_if_low
import dotenv
dotenv.load_dotenv()

# --- Import Chat Protocol ---
# (You can add any other protocols your agent uses here)
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    TextContent,
    chat_protocol_spec,
)

# --- 2. Your Agent Definition (Modified for Hosting) ---

# Get your Agentverse API Key from environment variables
# How to set: export AGENTVERSE_API_KEY='your_long_api_key_here'
# AGENTVERSE_API_KEY = os.environ.get("AGENTVERSE_API_KEY")





# --- 1. Refined Generic Registration Function ---

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

This is a `uagents` agent.
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




# Intialise agent1
agent1 = Agent(
    name="Agent1",
    seed="NewAgentSeedValueForDetermin2345678scc",
    endpoint=["http://localhost:8000/submit"],
    mailbox=True,
)

# Agent address you want to talk to
agent2_address = "agent1qgl8etxfyrhrdqasrukt6xm23gs9lde2dsdy5zrft59tphf4592m5p082ut"

# Initialize the chat protocol
chat_proto = Protocol(spec=chat_protocol_spec)

# --- 3. Agent Event Handlers ---

@agent1.on_event("startup")
async def startup_handler(ctx: Context):
    """Handles agent startup and triggers registration."""
    ctx.logger.info(f"My name is {ctx.agent.name} and my address is {ctx.agent.address}")
    # ctx.logger.info(f"Local server running on port: {ctx.agent.port}")
    ctx.logger.info(f"Local server running on port: {8000}")

    # Agent details to pass to the registration function
    agent_info = {
        "agent_address": ctx.agent.address,
        "bearer_token": AGENTVERSE_MAILBOX_KEY,
        # "port": ctx.agent.port,
        "port": 8000,
        "agent_name": ctx.agent.name,
        "description": "A simple chat agent for demonstration. Ajitesh's test agent.",
    }

    # Run registration in a separate thread to avoid blocking the agent
    registration_thread = threading.Thread(
        target=register_agent_with_agentverse,
        kwargs=agent_info
    )
    registration_thread.start()
    
    # You can still send an initial message if you want
    # initial_message = ChatMessage(...)
    # await ctx.send(agent2_address, initial_message)

@chat_proto.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    ctx.logger.info(f"[DEBUG] handle_message called with sender: {sender}, msg: {msg}")
    print(f"[DEBUG] handle_message called with sender: {sender}, msg: {msg}")
    for item in msg.content:
        if isinstance(item, TextContent):
            ctx.logger.info(f"Received message from {sender}: {item.text}")
            # You could add logic here to reply
            # reply = f"I received your message: '{item.text}'"
            # await ctx.send(sender, ChatMessage(...))
            
@chat_proto.on_message(ChatAcknowledgement)
async def handle_acknowledgement(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.info(f"Received acknowledgement from {sender} for message: {msg.acknowledged_msg_id}")

# Include the protocol in the agent
# publish_manifest=True is important for others to know how to talk to your agent
agent1.include(chat_proto, publish_manifest=True)

fund_agent_if_low(agent1.wallet.address())

if __name__ == '__main__':
    agent1.run()