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
    name="Simple Agent to Test ORC and HF Agents",
    port=5060,
    endpoint=["http://localhost:5060/submit"],
)


agent_address_to_send_message = "agent1qt0wzle7f9gmduadkxat8j6cd49zn3f5pd6qnut52zpfkuw4um2cczt2tvz"


prompt= "generate text-summarization history of artificial intelligence in brief using the best hugging face models, use necessary tools and follow main_orchestrator logic"



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