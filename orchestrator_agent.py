import logging
import os
import re
from datetime import datetime, timezone
from uuid import uuid4
from uagents import Agent, Context, Model, Protocol
from uagents.setup import fund_agent_if_low
from dotenv import load_dotenv

load_dotenv()

# We need requests for the parse_user_query, but let's stub it for now
# import requests
# Since requests is used in a non-async function, it's fine.
# Let's assume the user has 'requests' installed.
try:
    import requests
except ImportError:
    print("Please install 'requests' library: pip install requests")
    # Mocking for environments where requests isn't installed
    class MockResponse:
        def __init__(self, json_data, status_code):
            self.json_data = json_data
            self.status_code = status_code
        def json(self):
            return self.json_data
        @property
        def text(self):
            return str(self.json_data)

    class MockRequests:
        def post(self, *args, **kwargs):
            print("--- MOCKING 'requests.post' ---")
            # Mock a successful parse
            user_query = kwargs.get("json", {}).get("messages", [{}])[-1].get("content", "")
            if "about robots" in user_query:
                return MockResponse({"task": "text-generation", "prompt": "about robots"}, 200)
            return MockResponse({"task": None, "prompt": None}, 400)
    requests = MockRequests()


# Import the chat protocol specification
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    EndSessionContent,
    StartSessionContent,
    TextContent,
    chat_protocol_spec,
)
from knowledge_graph import OrchestratorKnowledgeGraph, print_all_atoms

THRESHOLD_TO_DEPLOY_NEW_AGENT = 5

# --- Agent Configuration ---

AGENT_SEED = "hf_orchestrator_agent_secret_seed_phrase_1"
API_KEY = os.getenv("ASI_ONE_API_KEY")
BASE_URL = "https://api.asi1.ai/v1/chat/completions"
MODEL = "asi1-mini"


# Set up logging
logger = logging.getLogger("OrchestratorAgent")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Initialize Agent and Knowledge Graph
agent = Agent(name="hf_orchestrator_agent", 
              seed=AGENT_SEED,
              port=8001,
              endpoint=[f"http://localhost:8001/submit"])
kg = OrchestratorKnowledgeGraph()

# --- Chat Protocol Setup ---

chat_proto = Protocol(spec=chat_protocol_spec)

# --- MODIFIED ---: This is your primary "worker" agent
hf_manager_address = "agent1q04wcekamg3rzekxhnmh776jmkhlkd0s2p5dqpum7nz8ff6jd5yhvwprta3"

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
    """
    
    # --- MODIFIED ---: This LLM call is good, but let's make the prompt more robust
    # The original regex was too simple.
    
    messages=[
        {"role": "system", "content": (
            "You are an expert at parsing user queries. Extract the 'task' and 'prompt'."
            "Tasks are usually things like 'text-generation', 'summarization', 'image-generation', etc."
            "The prompt is the specific instruction."
            "Example: 'generate texts about robots' -> task:text-generation; prompt:texts about robots"
            "Example: 'summarize this document...' -> task:summarization; prompt:this document..."
            "Return *only* the task and prompt in the format: task:<task_name>; prompt:<prompt_text>"
        )},
        {"role": "user", "content": f"Parse: '{user_query}'"}
    ]
    
    try:
        task_prompt_response = requests.post(
            BASE_URL,
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={
                "model": MODEL,
                "messages": messages
            }
        )

        if task_prompt_response.status_code == 200:
            response_text = task_prompt_response.json().get('choices', [{}])[0].get('message', {}).get('content', '')
            
            # --- MODIFIED ---: Parse the LLM's structured output
            task_match = re.search(r"task:([^\s;]+)", response_text, re.IGNORECASE)
            prompt_match = re.search(r"prompt:(.+)", response_text, re.IGNORECASE)
            
            task = task_match.group(1).strip() if task_match else None
            prompt = prompt_match.group(1).strip() if prompt_match else None
            
            if task and prompt:
                return task, prompt
            else:
                logger.warning(f"Could not parse LLM response: {response_text}")
                return None, None
        else:
            logger.error(f"Failed to parse user query (API Error): {task_prompt_response.text}")
            return None, None
    except Exception as e:
        logger.error(f"Exception in parse_user_query: {e}")
        return None, None


async def main_orchestrator_logic(ctx: Context, sender: str, user_query: str, kg: OrchestratorKnowledgeGraph):    
    # 1. Parse the user's intent
    task, prompt = parse_user_query(user_query)
    
    if not task or not prompt:
        logger.warning(f"Could not parse task/prompt from query: {user_query}")
        # --- MODIFIED ---: Send error back to the original caller
        await ctx.send(sender, create_text_chat(f"Sorry, I couldn't understand the task. Please be more specific, e.g., 'generate text about robots'."))
        return
    
    ctx.logger.info(f"Parsed query: task='{task}', prompt='{prompt[:30]}...'")
    
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
            
            # --- MODIFIED ---: [ACTION] Pass prompt to that agent
            ctx.logger.info(f"[ACTION] Forwarding query to specialist agent: {specialist_agent}")
            # We can just forward the original user query
            await ctx.send(specialist_agent, create_text_chat(user_query))
            
            # --- MODIFIED ---: Notify the original caller
            await ctx.send(sender, create_text_chat(f"Found a specialist agent for this task. Forwarding your request..."))

        else:
            # --- PATH 1b: No, agent does not exist ---
            ctx.logger.info(f"No specialist agent found for '{model_id}'.")
            
            # 4. Increment usage count for this model
            new_count = kg.increment_usage_count(model_id)
            ctx.logger.info(f"Incremented usage count for '{model_id}' to: {new_count}")

            # --- MODIFIED ---: [ACTION] Tell HF_agent to run the model
            ctx.logger.info(f"[ACTION] Sending request to HF Manager to run '{model_id}'...")
            # We create a structured request for our HF Manager
            request_text = f"generate from {model_id} {prompt}"
            await ctx.send(hf_manager_address, create_text_chat(request_text))
            
            # --- MODIFIED ---: Notify the original caller
            await ctx.send(sender, create_text_chat(f"No specialist found. Requesting generation from a general worker using '{model_id}'..."))
            

            # 5. Check if usage count meets the threshold to deploy
            if new_count >= THRESHOLD_TO_DEPLOY_NEW_AGENT:
                ctx.logger.info(f"Usage threshold ({THRESHOLD_TO_DEPLOY_NEW_AGENT}) reached!")
                
                # --- MODIFIED ---: [ACTION] Tell HF_agent to deploy this model
                ctx.logger.info(f"[ACTION] Sending deploy request to HF Manager for '{model_id}'...")
                deploy_request_text = f"deploy persistent model_id={model_id} task={task}"
                await ctx.send(hf_manager_address, create_text_chat(deploy_request_text))
                
                # --- MODIFIED ---: Notify user about the *deployment*, not the result
                await ctx.send(sender, create_text_chat(f"Note: Usage threshold for '{model_id}' reached. Initiating background deployment of a new specialist agent."))
                
                # --- MODIFIED ---: Removed the synchronous simulation block.
                # The response to this deploy request will be handled asynchronously
                # by the handle_message function when the HF Manager replies.
    
    else:
        # --- PATH 2: No, I don't have knowledge ---
        ctx.logger.info(f"No knowledge found for new task: '{task}'")
        
        # [ACTION] Search Hugging Face Hub for the best model for this task
        # --- MODIFIED ---: This should also be a request to the HF Manager
        ctx.logger.info(f"[ACTION] Asking HF Manager to find a model for '{task}'...")
        
        # This message asks the manager to *both* find and run the model
        request_text = f"find and generate for task={task} {prompt}"
        await ctx.send(hf_manager_address, create_text_chat(request_text))
        
        # --- MODIFIED ---: Notify the original caller
        await ctx.send(sender, create_text_chat(f"I don't know that task yet. Asking the HF Manager to find a suitable model on the Hub..."))

        # --- MODIFIED ---: Removed simulation block.
        # The HF Manager will find the model, run it, and return the result.
        # It could also optionally send a system message like:
        # "FOUND model=hf-hub/new-model-for-task task=text-generation"
        # which our handle_message could listen for to update the KG.


# --- Agent Message Handlers ---

# --- MODIFIED ---: This is now the main router
@chat_proto.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    """
    Handle incoming chat messages.
    Differentiates between new requests from callers and results from workers.
    """
    # Always acknowledge the message
    await ctx.send(
        sender,
        ChatAcknowledgement(timestamp=datetime.now(timezone.utc), acknowledged_msg_id=msg.msg_id),
    )

    # --- MODIFIED ---: Check if this sender is a known worker
    known_specialists = kg.get_all_specialist_agents() # Get current list
    is_worker_response = (sender == hf_manager_address or sender in known_specialists)
    
    caller_storage_key = f"caller_for_session_{ctx.session}"

    if is_worker_response:
        # --- PATH 1: This is a RESPONSE from a worker ---
        ctx.logger.info(f"Received a RESPONSE from worker: {sender}")
        
        # 1. Get the text content from the worker
        result_text_parts = []
        is_end_session = False
        for item in msg.content:
            if isinstance(item, TextContent):
                result_text_parts.append(item.text)
            elif isinstance(item, EndSessionContent):
                is_end_session = True
        
        result_text = "\n".join(result_text_parts).strip()
        
        # 2. Check if this is a system notification (e.g., deployment complete)
        #    (This is a convention you must define with your hf_manager_agent)
        deploy_match = re.match(r"DEPLOYED\s+model_id=([^\s]+)\s+address=([^\s]+)", result_text, re.IGNORECASE)
        found_match = re.match(r"FOUND\s+model_id=([^\s]+)\s+task=([^\s]+)", result_text, re.IGNORECASE)

        if deploy_match:
            # It's a system message: a new agent was deployed
            model_id = deploy_match.group(1)
            new_agent_address = deploy_match.group(2)
            ctx.logger.info(f"System Notification: Registering new agent for '{model_id}' at {new_agent_address}")
            kg.register_specialist_agent(model_id, new_agent_address)
            # We don't forward this to the user, but we could if we wanted to
            return # Stop processing
            
        elif found_match:
            # It's a system message: a new model was found
            model_id = found_match.group(1)
            task = found_match.group(2)
            ctx.logger.info(f"System Notification: Registering new model '{model_id}' for task '{task}'")
            kg.add_new_task_model(task, model_id)
            return # Stop processing

        # 3. If not a system message, it's a RESULT. Find the original caller.
        original_caller = ctx.storage.get(caller_storage_key)
        
        if not original_caller:
            ctx.logger.error(f"Received result from {sender} but could not find original caller for session {ctx.session}. Ignoring.")
            return

        # 4. Forward the result to the original caller
        if not result_text:
            result_text = "[Received empty response from worker]"
            
        ctx.logger.info(f"Forwarding result from {sender} to original caller {original_caller}")
        result_msg = create_text_chat(result_text, end_session=is_end_session)
        await ctx.send(original_caller, result_msg)
        
        # 5. If session ended, clean up storage
        if is_end_session:
            ctx.storage.delete(caller_storage_key)
            
    else:
        # --- PATH 2: This is a NEW REQUEST from a caller ---
        ctx.logger.info(f"Received a new REQUEST from caller: {sender}")
        
        # --- MODIFIED ---: Store the caller's address for this session!
        # This is the most important part.
        ctx.storage.set(caller_storage_key, sender)

        for item in msg.content:
            if isinstance(item, StartSessionContent):
                ctx.logger.info(f"Got a start session message from {sender}")
                await ctx.send(sender, create_text_chat("Hello! I am the HF Orchestrator Agent. How can I help? (e.g., 'generate text about robots')"))
                continue
            
            elif isinstance(item, TextContent):
                user_query = item.text.strip()
                ctx.logger.info(f"Processing query from {sender}: {user_query}")
                
                try:
                    # Call the main logic function to dispatch the task
                    await main_orchestrator_logic(ctx, sender, user_query, kg)
                
                except Exception as e:
                    ctx.logger.error(f"Error processing query: {e}", exc_info=True)
                    await ctx.send(sender, create_text_chat(f"I'm sorry, an internal error occurred: {e}"))
            
            elif isinstance(item, EndSessionContent):
                ctx.logger.info(f"Session ended by caller {sender}")
                # Clean up session data
                ctx.storage.delete(caller_storage_key)

            else:
                ctx.logger.info(f"Got unexpected content type from {sender}: {type(item)}")


@chat_proto.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    """Handle chat acknowledgements."""
    ctx.logger.info(f"Got an acknowledgement from {sender} for {msg.acknowledged_msg_id}")

# Register the protocol
agent.include(chat_proto, publish_manifest=True)


fund_agent_if_low(agent.wallet.address())

if __name__ == "__main__":
    logger.info(f"Starting agent '{agent.name}' on address: {agent.address}")
    agent.run()