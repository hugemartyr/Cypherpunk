import logging
from uagents import Agent, Context, Model

# Import the tool we created
from Tools.simple_hf_inf_tool import run_hf_inference_in_process, install_requirements

# Set up logging
logger = logging.getLogger("OrchestratorAgent")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

AGENT_SEED = "hf_orchestrator_agent_seed_phrase_1234"
agent = Agent(
    name="hf_orchestrator",
    seed=AGENT_SEED,
    port=8000,
    endpoint=["http://localhost:8000/submit"]
)

# --- 1. Define the Message Models ---
# These define the "API" for our agent

class HFRequest(Model):
    model_id: str
    task: str
    prompt: str
    requirements: list = None # e.g., ["torch", "transformers"]
    model_args: dict = None   # e.g., {"max_length": 150}

class HFResponse(Model):
    status: str      # "success" or "error"
    result: dict     # The full JSON output from the tool

# --- 2. The Main Agent Logic ---
@agent.on_message(model=HFRequest)
async def handle_hf_request(ctx: Context, sender: str, msg: HFRequest):
    """
    This function listens for HFRequest messages, runs the tool,
    and sends back an HFResponse.
    """
    ctx.logger.info(f"Received HFRequest from {sender} for model '{msg.model_id}'")
    
    # --- This is where we use the in-process tool ---
    # WARNING: As we discussed, this is a BLOCKING call and will freeze
    # this agent while it's working.
    result = run_hf_inference_in_process(
        model_id=msg.model_id,
        task=msg.task,
        prompt=msg.prompt,
        requirements=msg.requirements,
        model_args=msg.model_args
    )
    # --- Tool execution finished ---
    
    ctx.logger.info(f"Inference complete. Status: {result.get('status')}")

    # Send the result back to the original sender
    await ctx.send(sender, HFResponse(
        status=result.get("status", "error"),
        result=result
    ))

# --- 3. Run the Agent (if this file is run directly) ---
if __name__ == "__main__":
    logger.info("Starting HF Orchestrator Agent...")
    logger.info(f"Agent Address: {agent.address}")
    agent.run()
