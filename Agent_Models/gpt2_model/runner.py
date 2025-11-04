

import threading
import sys, json, os, time, torch, signal, warnings, select
from transformers import pipeline
import requests

warnings.filterwarnings('ignore')
from uagents.setup import fund_agent_if_low

from datetime import datetime, timezone
from uuid import uuid4
from uagents import Agent, Protocol, Context, Model

#import the necessary components from the chat protocol
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    TextContent,
    chat_protocol_spec,
)

# === Configuration ===
MODEL_ID = "gpt2"
TASK_TYPE = "auto"
PORT = 7603
SEED_PHRASE = f"{MODEL_ID}_seed_phrase_123457890112"

class HFManagerChat(Model):
    ChatMessage: ChatMessage
    caller_Agent_address: str
class HFManagerChatAcknowledgement(Model):
    ChatAcknowledgement: ChatAcknowledgement
    caller_Agent_address: str
        
AGENT_MAILBOX_BEARER_TOKEN="eyJhbGciOiJSUzI1NiJ9.eyJleHAiOjE3NjQ4NzM3OTcsImlhdCI6MTc2MjI4MTc5NywiaXNzIjoiZmV0Y2guYWkiLCJqdGkiOiJiNjlkZWNhNTVhODk4NmZhMTQxNDAyODUiLCJzY29wZSI6ImF2Iiwic3ViIjoiZGY5NDllZWM0ODk1MTg5NjU5MmI3NDFkNjA2MmU1MjU0MWVlNGY2ZWU2NTU1MmI3In0.U3pvvM19LBxpy9RniuILWO8OY_QZLsl2vrQfvGJYu3QtvuZctsQQnAnjMoY34kAF7AycLL-gJXHmtnZdlSIvdpu5Jbx1QdJDpI4zwU34HYXNMxvC_WnZ46tD08Rl7EGDE_J2EgiF4RwAn1wrN70n5N1IrHVhDHBYBJsTytBFm9YrUPXwcWcRCHt0UDyGqNrpBGOaec2nb8LFnevLtwvwAXl4F__6CBuqwMfAx1MEz3mmOVu0ZwFfxAfbw8ruxUlhYcwDuUPK3LfNDmFRiFlc9qi9u0xiIeOqyeh6DrWCcuD7aVf5CIbaxTmonTts8pmT4o6mRHmDCuMYf0Bd493Q2g"
        
# --- Path Configuration ---
# Get the absolute path of the directory this script is in
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Log file will be in this directory
LOG_FILE = os.path.join(SCRIPT_DIR, "log.txt") 

# Model cache will be two levels up in the ../../Model directory
MODEL_CACHE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "Model"))
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
# --- End Path Configuration ---

def register_agent_with_agentverse(
    agent_name: str,
    agent_address: str,
    bearer_token: str,
    port: int,
    description: str = "",
    readme_content: str = None,
):
    port=PORT
    

    if not bearer_token:
        print("AGENTVERSE_API_KEY not set, skipping registration.")
        return

    print(f"Agent '{agent_name}' starting registration...")
    time.sleep(8)

    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "Content-Type": "application/json",
    }

    # --- Step 1: Connect local mailbox ---
    connect_url = f"http://127.0.0.1:{port}/connect"
    connect_payload = {"agent_type": "mailbox", "user_token": bearer_token}
    connect_response = requests.post(connect_url, json=connect_payload, headers=headers, timeout=10)
    print(f"[DEBUG] /connect status: {connect_response.status_code} - {connect_response.text}")

    # --- Step 2: Request challenge ---
    challenge_url = f"https://agentverse.ai/v1/agents/challenge/{agent_address}"
    challenge_response = requests.get(challenge_url, headers=headers, timeout=10)

    if challenge_response.status_code != 200:
        print(f"Failed to get challenge: {challenge_response.status_code} - {challenge_response.text}")
        return

    challenge = challenge_response.json().get("challenge")
    print(f"[DEBUG] Challenge received: {challenge}")

    # --- Step 3: Sign challenge locally ---
    sign_url = f"http://127.0.0.1:{port}/sign"
    sign_payload = {"message": challenge}
    sign_response = requests.post(sign_url, json=sign_payload, headers=headers, timeout=10)

    if sign_response.status_code != 200:
        print(f"Failed to sign challenge: {sign_response.status_code} - {sign_response.text}")
        return

    challenge_sig = sign_response.json().get("signature")
    print(f"[DEBUG] Challenge signature: {challenge_sig}")

    # --- Step 4: Register agent ---
    register_url = "https://agentverse.ai/v1/agents"
    register_payload = {
        "address": agent_address,
        "agent_type": "mailbox",
        "challenge": challenge,
        "challenge_response": challenge_sig,
    }

    register_response = requests.post(register_url, json=register_payload, headers=headers, timeout=10)
    print(f"[DEBUG] /agents status: {register_response.status_code} - {register_response.text}")

    # --- Step 5: Update profile ---
    update_url = f"https://agentverse.ai/v1/agents/{agent_address}"

    if not readme_content:
        readme_content = "This is an Specialist Agent that manages Hugging Face models."


    update_payload = {
        "name": agent_name,
        "readme": readme_content,
        "short_description": description,
    }

    update_response = requests.put(update_url, json=update_payload, headers=headers, timeout=10)
    print(f"[DEBUG] /update status: {update_response.status_code} - {update_response.text}")

    print(f"Agent '{agent_name}' registration complete!")



# Global variable to hold the model pipeline
pipe = None

def log(msg):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")
    print(msg, flush=True)

def infer(pipe, text):
    #Runs inference using the loaded model pipeline.
    if TASK_TYPE == "text-generation" or (TASK_TYPE == "auto" and any(x in MODEL_ID.lower() for x in ["gpt", "llama"])):
        out = pipe(text, max_new_tokens=100, return_full_text=False, do_sample=True, temperature=0.7)
    else:
        out = pipe(text)
    
    # Process the output
    if isinstance(out, list) and out and isinstance(out[0], dict):
        key = next((k for k in ['generated_text', 'summary_text', 'translation_text', 'answer'] if k in out[0]), 'label')
        return out[0].get(key, str(out[0]))
    return str(out)

# --- Agent Setup ---

log(f"Specialist Agent starting for {MODEL_ID}")

# Initialise agent
agent = Agent(
    name=f"{MODEL_ID}_specialist_agent",
    seed=SEED_PHRASE,
    port=PORT,
    # endpoint=[f"http://localhost:{PORT}/submit"] # Example endpoint, change if using cloudflared
    mailbox=True
)

# Initialize the chat protocol
new_chat_protocol = Protocol("AgentChatProtocol", "0.3.0")
chat_proto = Protocol(spec=chat_protocol_spec)

@agent.on_event("startup")
async def load_model(ctx: Context):
    # #Load the Hugging Face model when the agent starts up.
    print("Loading model...")
    global pipe
    ctx.logger.info(f"Persistent Runner started for {MODEL_ID}")
    device = 0 if torch.cuda.is_available() else -1
    ctx.logger.info(f"Loading model on {'GPU' if device == 0 else 'CPU'}...")

    try:
        # Use the cache_dir to prevent path confusion and re-downloading
        pipe = pipeline(
            TASK_TYPE if TASK_TYPE != "auto" else None, 
            model=MODEL_ID, 
            device=device, 
            trust_remote_code=True
        )
        ctx.logger.info("Model loaded successfully.")
    except Exception as e:
        ctx.logger.error(f"[FATAL] Model loading failed: {e}")
        sys.exit(1) # Quit if the model can't load
        
    ctx.logger.info(f"My name is {ctx.agent.name} and my address is {ctx.agent.address}")
    ctx.logger.info("Waiting for prompts...")


# Message Handler - Process received messages
@chat_proto.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    global pipe
    
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
            
            response_text = "An error occurred."
            try:
                if pipe is None:
                    ctx.logger.error("Model pipe is not initialized!")
                    response_text = "ERROR: Model is not loaded. Please check agent logs."
                else:
                    new_prompt = item.text.strip() 
                    if new_prompt:
                        log(f"New prompt received: {new_prompt[:60]}...")
                        resp = infer(pipe, new_prompt)
                        log(f"Response: {resp[:150]}...")
                        response_text = resp # This is the generated text
                
            except Exception as e:
                log(f"[ERROR] Runtime: {e}")
                response_text = f"An error occurred during inference: {e}"
            
            # Send the actual response message
            response = ChatMessage(
                timestamp=datetime.utcnow(),
                msg_id=uuid4(),
                content=[TextContent(type="text", text=response_text)]
            )
            await ctx.send(sender, response)

# Acknowledgement Handler - Process received acknowledgements
@chat_proto.on_message(ChatAcknowledgement)
async def handle_acknowledgement(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.info(f"Received acknowledgement from {sender} for message: {msg.acknowledged_msg_id}")

# Message Handler - Process received messages
@new_chat_protocol.on_message(HFManagerChat) 
async def handle_message_hf_agent(ctx: Context, sender: str, msg: HFManagerChat):
    global pipe
    
    for item in msg.ChatMessage.content:
        if isinstance(item, TextContent):
            # Log received message
            ctx.logger.info(f"Received message from {sender}: {item.text}")
            
            # Send acknowledgment
            ack = ChatAcknowledgement(
                timestamp=datetime.utcnow(),
                acknowledged_msg_id=msg.ChatMessage.msg_id
            )
            await ctx.send(sender, ack)
            
            response_text = "An error occurred."
            try:
                if pipe is None:
                    ctx.logger.error("Model pipe is not initialized!")
                    response_text = "ERROR: Model is not loaded. Please check agent logs."
                else:
                    new_prompt = item.text.strip() 
                    if new_prompt:
                        log(f"New prompt received: {new_prompt[:60]}...")
                        resp = infer(pipe, new_prompt)
                        log(f"Response: {resp[:150]}...")
                        response_text = resp # This is the generated text
                
            except Exception as e:
                log(f"[ERROR] Runtime: {e}")
                response_text = f"An error occurred during inference: {e}"
            
            # Send the message response to the original caller
            response = ChatMessage(
                timestamp=datetime.utcnow(),
                msg_id=uuid4(),
                content=[TextContent(type="text", text=response_text)]
            )
            await ctx.send(msg.caller_Agent_address, response)
            
            # skip sending back to HF Manager to avoid confusion


@new_chat_protocol.on_message(HFManagerChatAcknowledgement)
async def handle_ack_of_manager(ctx: Context, sender: str, msg: HFManagerChatAcknowledgement):
    
    ctx.logger.info(f"Got an acknowledgement from {sender} for {msg.ChatAcknowledgement.acknowledged_msg_id}")
    
@agent.on_event("startup")
async def startup_handler(ctx: Context):
    # Handles agent startup and triggers registration.
    ctx.logger.info(f"My name is {ctx.agent.name} and my address is {ctx.agent.address}")
    ctx.logger.info(f"Local server running on port: {PORT}")
   
    agent_info = {
        "agent_address": ctx.agent.address,
        "bearer_token": AGENT_MAILBOX_BEARER_TOKEN,
        "port": PORT,
        "agent_name": ctx.agent.name,
        "description": "An Specialist agent To Manage Hugging Face models.",
        "readme_content": None,
    }

    # Run registration in a separate thread to avoid blocking the agent
    registration_thread = threading.Thread(
        target=register_agent_with_agentverse,
        kwargs=agent_info
    )
    registration_thread.start()    


fund_agent_if_low(agent.wallet.address())
# Include the protocol in the agent
agent.include(chat_proto, publish_manifest=True)
agent.include(new_chat_protocol, publish_manifest=True)

if __name__ == '__main__':
    agent.run()

