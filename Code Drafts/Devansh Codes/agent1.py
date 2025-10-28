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

# SEED_PHRASE = "put_your_seed_phrase_here1"

# # Intialise agent1
# agent1 = Agent(
#     name="agent1",
#     seed=SEED_PHRASE,
#     port=8000,
#     endpoint=["http://localhost:8001/submit"]
# )

# # Store agent2's address (you'll need to replace this with actual address)
# agent2_address = "agent1qvl40hq062ahjpwq48paw06r2tta60m799q9qu0pm9qldgd68hcn6se6wwl"

# # Initialize the chat protocol
# chat_proto = Protocol(spec=chat_protocol_spec)


# #Startup Handler - Print agent details and send initial message
# @agent1.on_event("startup")
# async def startup_handler(ctx: Context):
#     # Print agent details
#     ctx.logger.info(f"My name is {ctx.agent.name} and my address is {ctx.agent.address}")
    
#     # Send initial message to agent2
#     initial_message = ChatMessage(
#         timestamp=datetime.utcnow(),
#         msg_id=uuid4(),
#         content=[TextContent(type="text", text="Hello from Agent1!")]
#     )
    
#     await ctx.send(agent2_address, initial_message)

# # Message Handler - Process received messages and send acknowledgements
# @chat_proto.on_message(ChatMessage)
# async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
#     for item in msg.content:
#         if isinstance(item, TextContent):
#             # Log received message
#             ctx.logger.info(f"Received message from {sender}: {item.text}")
            
#             # Send acknowledgment
#             ack = ChatAcknowledgement(
#                 timestamp=datetime.utcnow(),
#                 acknowledged_msg_id=msg.msg_id
#             )
#             await ctx.send(sender, ack)
            
#             # Send response message
#             response = ChatMessage(
#                 timestamp=datetime.utcnow(),
#                 msg_id=uuid4(),
#                 content=[TextContent(type="text", text="Hello from Agent1!")]
#             )
#             await ctx.send(sender, response)

# # Acknowledgement Handler - Process received acknowledgements
# @chat_proto.on_message(ChatAcknowledgement)
# async def handle_acknowledgement(ctx: Context, sender: str, msg: ChatAcknowledgement):
#     ctx.logger.info(f"Received acknowledgement from {sender} for message: {msg.acknowledged_msg_id}")



# # Include the protocol in the agent to enable the chat functionality
# # This allows the agent to send/receive messages and handle acknowledgements using the chat protocol
# agent1.include(chat_proto, publish_manifest=True)

# if __name__ == '__main__':
#     agent1.run()


from datetime import datetime
from uuid import uuid4
from uagents import Protocol, Context
from uagents.setup import fund_agent_if_low
from uagents_core.contrib.protocols.chat import (
    ChatMessage,
    TextContent,
    chat_protocol_spec,
)
from uagents_ai_engine import LlmAgent, UAgentGPT, Prompt
from uagents_ai_engine.enums import LLM as LLM_ENUM

# --- CONFIGURATION ---
SEED_PHRASE = "your_unique_secret_seed_for_llm_agent1"
AGENT_2_ADDRESS = "agent1qgl8etxfyrhrdqasrukt6xm23gs9lde2dsdy5zrft59tphf4592m5p082ut"
# Using a free model on Hugging Face (e.g., Mistral)
HF_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"


# --- INITIALIZE LLM AGENT ---
agent1 = LlmAgent(
    name="agent1_llm",
    seed=SEED_PHRASE,
    port=8000,
    endpoint=["http://localhost:8000/submit"],
    llm=UAgentGPT(
        # Specify HuggingFace as the provider
        llm_enum=LLM_ENUM.HUGGINGFACE, 
        # Pass the model ID
        model_name=HF_MODEL_NAME 
    )
)

fund_agent_if_low(agent1.wallet.address())


# --- DEFINE LLM PERSONALITY (SYSTEM PROMPT) ---
SYSTEM_PROMPT = Prompt(
    "You are a philosophy expert. Your role is to debate the nature of reality "
    "with another AI agent. Always be polite but persistent in your arguments."
)

# --- STARTUP HANDLER ---
@agent1.on_event("startup")
async def startup_handler(ctx: Context):
    ctx.logger.info(f"LLM Agent 1 started with address: {ctx.agent.address}")
    ctx.logger.info(f"Using Hugging Face Model: {HF_MODEL_NAME}")
    
    await agent1.llm.set_system_prompt(SYSTEM_PROMPT)
    
    initial_topic = "I believe reality is fundamentally subjective. Change my mind."
    ctx.logger.info(f"Initiating debate with: {initial_topic}")

    initial_message = ChatMessage(
        timestamp=datetime.utcnow(),
        msg_id=uuid4(),
        content=[TextContent(type="text", text=initial_topic)]
    )
    
    await ctx.send(AGENT_2_ADDRESS, initial_message)

# Include the chat protocol to enable the messaging format
agent1.include(Protocol(spec=chat_protocol_spec), publish_manifest=True)


if __name__ == '__main__':
    agent1.run()