import asyncio
from uagents import Agent, Context, Model, Protocol
from uagents.setup import fund_agent_if_low
import sys

# Import the chat protocol specification
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    EndSessionContent,
    StartSessionContent,
    TextContent,
    chat_protocol_spec,
)

# --- Configuration ---

# This is the address of your main orchestrator agent
ORCHESTRATOR_ADDRESS = "agent1qgl8etxfyrhrdqasrukt6xm23gs9lde2dsdy5zrft59tphf4592m5p082ut"

# This is the agent for our terminal client (it needs its own identity)
CHAT_CLIENT_SEED = "chat_client_secret_seed_phrase_qwert"
agent = Agent(name="chat_client", seed=CHAT_CLIENT_SEED,
              port=8001,
              endpoint=["http://localhost:8001/submit"])

# Global to signal when the conversation is starting
is_starting_session = True

# --- Protocol Setup ---

chat_proto = Protocol(spec=chat_protocol_spec)

@chat_proto.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    """Handle incoming chat messages from the orchestrator."""
    for item in msg.content:
        if isinstance(item, TextContent):
            print(f"\nAgent: {item.text}")
        elif isinstance(item, EndSessionContent):
            print("\n[Agent ended the session.]")
            # We can add logic here to exit the client if needed
    
    # Prompt for next input
    print("You: ", end="", flush=True)

@chat_proto.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    """Handle chat acknowledgements."""
    ctx.logger.debug(f"Got an acknowledgement from {sender}")

agent.include(chat_proto)

# --- Main Terminal Loop ---

async def main_loop():
    """The main terminal chat loop."""
    global is_starting_session
    
    print("--- Terminal Chat Client ---")
    print(f"Connecting to orchestrator at: {ORCHESTRATOR_ADDRESS}")
    print("Type 'exit' to quit.")
    
    # Fund the agent if it's low on FET
    fund_agent_if_low(agent.wallet.address())
    
    # Wait for agent to be ready
    await asyncio.sleep(1) 

    while True:
        try:
            # Get user input from the terminal asynchronously
            user_query = await asyncio.to_thread(sys.stdin.readline)
            user_query = user_query.strip()

            if user_query.lower() == 'exit':
                print("Exiting...")
                break
            
            if not user_query:
                continue

            # Create the chat message content
            content = [TextContent(type="text", text=user_query)]
            
            # Add StartSessionContent if this is the first message
            if is_starting_session:
                content.insert(0, StartSessionContent(type="start-session"))
                is_starting_session = False

            # Create and send the message
            msg = ChatMessage(content=content)
            await agent.send(ORCHESTRATOR_ADDRESS, msg)

        except KeyboardInterrupt:
            print("\nExiting...")
            break

# --- Run the Client ---

if __name__ == "__main__":
    
    # Define an async main function to run the agent and the loop
    async def run_client():
        # Start the agent in the background
        # --- FIX: Use agent.run_async() instead of agent.run() ---
        agent_task = asyncio.create_task(agent.run_async())
        
        # Run the main terminal loop
        await main_loop()
        
        # Stop the agent when the loop is done
        agent_task.cancel()
        try:
            await agent_task
        except asyncio.CancelledError:
            print("Agent stopped.")

    # Run the async main function
    try:
        asyncio.run(run_client())
    except KeyboardInterrupt:
        print("Client shutting down.")

