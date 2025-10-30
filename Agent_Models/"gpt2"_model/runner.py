
import sys, json, os, time, torch, signal, warnings, select
from transformers import pipeline
warnings.filterwarnings('ignore')
from uagents.setup import fund_agent_if_low

from datetime import datetime, timezone
from uuid import uuid4
from uagents import Agent, Protocol, Context

#import the necessary components from the chat protocol
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    TextContent,
    chat_protocol_spec,
)

# === Configuration ===
MODEL_ID = ""gpt2""
TASK_TYPE = "auto"
PORT = 8001
SEED_PHRASE = f"{MODEL_ID}_seed_phrase_12345"

# --- Path Configuration ---
# Get the absolute path of the directory this script is in
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Log file will be in this directory
LOG_FILE = os.path.join(SCRIPT_DIR, "log.txt") 

# Model cache will be two levels up in the ../../Model directory
MODEL_CACHE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "Model"))
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
# --- End Path Configuration ---

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
    endpoint=[f"http://localhost:{PORT}/submit"] # Example endpoint, change if using cloudflared
)

# Initialize the chat protocol
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
                timestamp=datetime.now(timezone.utcnow()),
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
                timestamp=datetime.now(timezone.utc),
                msg_id=uuid4(),
                content=[TextContent(type="text", text=response_text)]
            )
            await ctx.send(sender, response)

# Acknowledgement Handler - Process received acknowledgements
@chat_proto.on_message(ChatAcknowledgement)
async def handle_acknowledgement(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.info(f"Received acknowledgement from {sender} for message: {msg.acknowledged_msg_id}")



fund_agent_if_low(agent.wallet.address())
# Include the protocol in the agent
agent.include(chat_proto, publish_manifest=True)

if __name__ == '__main__':
    agent.run()


