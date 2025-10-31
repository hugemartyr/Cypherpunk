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
    name="Simple Agent to Test HF Agent",
    port=5060,
    endpoint=["http://localhost:5060/submit"],
)


agent_address_to_send_message = "agent1q04wcekamg3rzekxhnmh776jmkhlkd0s2p5dqpum7nz8ff6jd5yhvwprta3"

prompt= "generate persistent ie deploy agent text-summarization history of agentic AI in brief using gpt2 model from hugging face, use necessary tools, just use tool and don't suggest any other logic"



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
        content=[TextContent(type="text", text=prompt)],
    )
    
    await ctx.send(agent_address_to_send_message, initial_message)

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
            

# Acknowledgement Handler - Process received acknowledgements
@chat_proto.on_message(ChatAcknowledgement)
async def handle_acknowledgement(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.info(f"Received acknowledgement from {sender} for message: {msg.acknowledged_msg_id}")

agent1.include(chat_proto, publish_manifest=True)

if __name__ == '__main__':
    agent1.run()