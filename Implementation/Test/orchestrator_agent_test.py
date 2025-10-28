import logging
import asyncio
from uuid import uuid4

# --- START OF PATH FIX ---
# Add the project's root directory (Cypherpunk) to the Python path
# This allows us to import modules from other folders
import sys
import os

# Get the directory of this test script (e.g., /.../Cypherpunk/Implementation/Test)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up two levels to get to the project root (e.g., /.../Cypherpunk)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

# Add the project root to sys.path
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
# --- END OF PATH FIX ---


# --- Imports from our project files ---
# These imports will now work because the root is in sys.path
from Implementation.meTTa.knowledge_graph import OrchestratorKnowledgeGraph
THRESHOLD_TO_DEPLOY_NEW_AGENT=5
from orchestrator_agent import main_orchestrator_logic # Import from orchestrator_agent.py in the root
from orchestrator_agent import create_text_chat      # Import from orchestrator_agent.py in the root

# --- Imports for Mocking ---
# We need these models for the MockContext to work correctly
from uagents_core.contrib.protocols.chat import (
    ChatMessage,
    TextContent,
)

# Set up logging
logger = logging.getLogger("OrchestratorTest")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Main block for Testing (as requested) ---

# We create a "Mock" Context to test the logic without running the agent
class MockContext:
    def __init__(self, name="mock_agent"):
        self.logger = logging.getLogger(name)
        self.storage = {}
        self.session = uuid4()
    
    async def send(self, sender, msg):
        print(f"\n[MOCK SEND to {sender}]:")
        # This check requires ChatMessage and TextContent to be imported
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

    print("\n--- TEST 5: Check if specialist was registered ---")
    model_id = test_kg.find_model_for_task("text-generation")
    agent_addr = test_kg.find_specialist_agent(model_id)
    print(f"MeTTa now lists specialist for 'text-generation': {agent_addr}")
    if agent_addr:
        print("TEST 5 PASSED")
    else:
        print("TEST 5 FAILED")

    print("\n--- TEST SUITE COMPLETE ---")

if __name__ == "__main__":
    
    # To run the test suite (as requested):
    asyncio.run(run_tests())

