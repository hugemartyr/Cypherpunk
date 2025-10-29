from datetime import datetime
from uuid import uuid4
from uagents import Agent, Protocol, Context, Model
from time import sleep

#import the necessary components from the chat protocol
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    TextContent,
    chat_protocol_spec,
)

# Intialise agent1
agent1 = Agent(
    name="Agent1",
    port=5060,
    endpoint=["http://localhost:5060/submit"],
)

# Store agent2's address (you'll need to replace this with actual address)
# agent2_address = "agent1qt5n8ma2g8hc5mcj0ne6hlnx8e8k9xnh9mmxa3jfa9kkgcwqpwlwxd2tddu"
# agent2_address = "agent1q0g5qx5cx0jzqvjjn3lpsdzcyvp8m2smkekh8stqeke5uax770k8cdtngc9"
# agent2_address = "agent1q04wcekamg3rzekxhnmh776jmkhlkd0s2p5dqpum7nz8ff6jd5yhvwprta3"
agent2_address = "agent1qgl8etxfyrhrdqasrukt6xm23gs9lde2dsdy5zrft59tphf4592m5p082ut"



# Initialize the chat protocol
chat_proto = Protocol(spec=chat_protocol_spec)


#Startup Handler - Print agent details and send initial message
@agent1.on_event("startup")
async def startup_handler(ctx: Context):
    # Print agent details
    ctx.logger.info(f"My name is {ctx.agent.name} and my address is {ctx.agent.address}")
    
    # Send initial message to agent2
    initial_message = ChatMessage(
        timestamp=datetime.utcnow(),
        msg_id=uuid4(),
        content=[TextContent(type="text", text="sumarize the history of artificial intelligence in brief using the best hugging face models, use necessary tools and follow main_orchestrator logic")],
    )
    
    await ctx.send(agent2_address, initial_message)

# Message Handler - Process received messages and send acknowledgements
@chat_proto.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    for item in msg.content:
        if isinstance(item, TextContent):
            # Log received message
            ctx.logger.info(f"Received message from {sender}: {item.text}")
            
            # Send acknowledgment
            ack = ChatAcknowledgement(
                timestamp=datetime.utcnow(),
                acknowledged_msg_id=msg.msg_id
            )
            await ctx.send(sender, ack)
            
            
            
            # # Send response message
            # response = ChatMessage(
            #     timestamp=datetime.utcnow(),
            #     msg_id=uuid4(),
            #     content=[TextContent(type="text", text="Hello from Agent1!")]
            # )
            # await ctx.send(sender, response)

# Acknowledgement Handler - Process received acknowledgements
@chat_proto.on_message(ChatAcknowledgement)
async def handle_acknowledgement(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.info(f"Received acknowledgement from {sender} for message: {msg.acknowledged_msg_id}")



# Include the protocol in the agent to enable the chat functionality
# This allows the agent to send/receive messages and handle acknowledgements using the chat protocol
agent1.include(chat_proto, publish_manifest=True)

if __name__ == '__main__':
    agent1.run()