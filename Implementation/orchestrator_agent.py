import logging
import re
from datetime import datetime, timezone
from uuid import uuid4
from uagents import Agent, Context, Model, Protocol
from uagents.setup import fund_agent_if_low
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI

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
from meTTa.knowledge_graph import OrchestratorKnowledgeGraph, print_all_atoms

THRESHOLD_TO_DEPLOY_NEW_AGENT = 5

# --- Agent Configuration ---

AGENT_SEED = "hf_orchestrator_agent_secret_seed_phrase_1"

# Set up logging
logger = logging.getLogger("OrchestratorAgent")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Initialize Agent and Knowledge Graph
agent = Agent(name="hf_orchestrator_agent", seed=AGENT_SEED)
kg = OrchestratorKnowledgeGraph()

# --- Chat Protocol Setup (from your example) ---

chat_proto = Protocol(spec=chat_protocol_spec)

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
        specialist_agent = kg.find_specialist_agent(model_id)
        
        if specialist_agent:
            # --- PATH 1a: Yes, agent exists ---
            ctx.logger.info(f"Specialist agent found: {specialist_agent}")
            
            # [ACTION] Pass prompt to that agent
            ctx.logger.info(f"[ACTION] Forwarding prompt to specialist agent: {specialist_agent}")
            response_text = f"Specialist agent found for '{model_id}'. [Simulating call: Forwarded prompt...]"
            await ctx.send(sender, create_text_chat(response_text))
            
        else:
            # --- PATH 1b: No, agent does not exist ---
            ctx.logger.info(f"No specialist agent found for '{model_id}'.")
            
            # 4. Increment usage count for this model
            new_count = kg.increment_usage_count(model_id)
            ctx.logger.info(f"Incremented usage count for '{model_id}' to: {new_count}")

            # [ACTION] Tell HF_agent (our local tool) to build and run the model
            ctx.logger.info(f"[ACTION] Calling local 'hf_tool' to run '{model_id}'...")
            response_text = f"No specialist found. [Simulating call: Running '{model_id}' locally. Usage count: {new_count}]"
            await ctx.send(sender, create_text_chat(response_text))

            # 5. Check if usage count meets the threshold to deploy
            if new_count >= THRESHOLD_TO_DEPLOY_NEW_AGENT:
                ctx.logger.info(f"Usage threshold ({THRESHOLD_TO_DEPLOY_NEW_AGENT}) reached!")
                
                # [ACTION] Tell HF_agent (our deploy tool) to deploy this model
                ctx.logger.info(f"[ACTION] Calling 'provisioner.py' to deploy new agent for '{model_id}'...")
                
                
                # here we will recieve message from HF_agent about new deployed agent address
                
                
                # Simulate a new address and update the knowledge graph
                new_agent_address = f"agent1q...simulated-addr-for-new-agent"
                kg.register_specialist_agent(model_id, new_agent_address)
                
                ctx.logger.info(f"New agent registered in MeTTa: {new_agent_address}")
                response_text_2 = f"Usage threshold reached! [Simulating call: Deployed new specialist agent for {model_id} at {new_agent_address}]"
                await ctx.send(sender, create_text_chat(response_text_2))
    
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

# Register the protocol
agent.include(chat_proto, publish_manifest=True)


# --- Main block for Testing (as requested) ---
# We create a "Mock" Context to test the logic without running the agent
class MockContext:
    def __init__(self, name="mock_agent"):
        self.logger = logging.getLogger(name)
        self.storage = {}
        self.session = uuid4()
    
    async def send(self, sender, msg):
        print(f"\n[MOCK SEND to {sender}]:")
        if isinstance(msg, ChatMessage):
            for item in msg.content:
                if isinstance(item, TextContent):
                    print(f"  - {item.text}")
        else:
            print(f"  - {msg}")
    
    async def query(self, dest, msg, timeout):
        pass # Not needed for this test

async def run_tests():
    """Runs a test suite against the main logic function."""
    print("--- STARTING ORCHESTATOR LOGIC TEST SUITE ---")
    
    # Initialize a fresh KG and Mock Context
    test_kg = OrchestratorKnowledgeGraph()
    mock_ctx = MockContext()
    mock_sender = "local_tester"

    # --- Test 1: Path 1a (Knowledge exists, Specialist agent exists) ---
    print("\n--- TEST 1: Path 1a (Find existing specialist) ---")
    query1 = "generate crypto-news get latest on $FETCH"
    await main_orchestrator_logic(mock_ctx, mock_sender, query1, test_kg)

    # --- Test 2: Path 1b (Knowledge exists, No specialist, No threshold) ---
    print("\n--- TEST 2: Path 1b (Run locally, no threshold) ---")
    query2 = "generate text-generation a poem about a robot"
    await main_orchestrator_logic(mock_ctx, mock_sender, query2, test_kg)

    # --- Test 3: Path 2 (No knowledge, discover new) ---
    print("\n--- TEST 3: Path 2 (Discover new task 'sound-generation') ---")
    query3 = "generate sound-generation a cat purring"
    await main_orchestrator_logic(mock_ctx, mock_sender, query3, test_kg)

    # --- Test 4: Path 1b (Trigger deployment threshold) ---
    print(f"\n--- TEST 4: Triggering deployment for 'text-generation' (Threshold={THRESHOLD_TO_DEPLOY_NEW_AGENT}) ---")
    # We already ran it once. Let's run it (THRESHOLD - 1) more times.
    for i in range(THRESHOLD_TO_DEPLOY_NEW_AGENT - 1):
        print(f"\n  ... usage loop {i+2} ...")
        # In the last loop, it should trigger the deployment
        await main_orchestrator_logic(mock_ctx, mock_sender, query2, test_kg)
        test_kg.find_specialist_agent("microsoft/Phi-3-mini-4k-instruct")
        

    print("\n--- TEST 5: Check if specialist was registered ---")
    model_id = test_kg.find_model_for_task("text-generation")
    agent_addr = test_kg.find_specialist_agent(model_id)
    print(f"MeTTa now lists specialist for 'text-generation': {agent_addr}")
    if agent_addr:
        print("TEST 5 PASSED")
    else:
        print("TEST 5 FAILED")

    print_all_atoms (test_kg.metta)

    print("\n--- TEST SUITE COMPLETE ---")

fund_agent_if_low(agent.wallet.address())



# name = "Chat Protocol Adapter"
# identity = Identity.from_seed(os.environ["AGENT_SEED_PHRASE"], 0)
# readme = "# Chat Protocol Adapter \nExample of how to integrate chat protocol."
# endpoint = "AGENT_EXTERNAL_ENDPOINT"
# app = FastAPI()
# @app.get("/status")
# async def healthcheck():
#     return {"status": "OK - Agent is running"}
# @app.post("/chat")
# async def handle_message(env: Envelope):
#     msg = cast(ChatMessage, parse_envelope(env, ChatMessage))
#     print(f"Received message from {env.sender}: {msg.text()}")
#     handle_message(agent.context, env.sender, msg)

if __name__ == "__main__":
    
    pass
    
    # To run the test suite (as requested):
    # import asyncio
    # asyncio.run(run_tests())
    
    # To run the actual agent (uncomment this):
    logger.info(f"Starting agent '{agent.name}' on address: {agent.address}")
    agent.run()

