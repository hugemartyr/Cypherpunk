import asyncio
import json
from uagents import Agent, Context, Model
from uagents.query import query

# --- Define the same Message Models as the agent ---
# This tells our client how to structure the request and parse the response

class HFRequest(Model):
    model_id: str
    task: str
    prompt: str
    requirements: list = None # e.g., ["torch", "transformers"]
    model_args: dict = None   # e.g., {"max_length": 150}

class HFResponse(Model):
    status: str
    result: dict # This will contain the full JSON output from the tool

# --- Configuration ---
# 1. Run 'orchestrator_agent.py' in a terminal
# 2. Copy the "Agent Address" it logs
# 3. Paste that address here
ORCHESTRATOR_AGENT_ADDRESS = "agent1qv65rtun6r7n5jc93asez0yq0vmpzp95l0u6kgvstu4kuhunzs0jv75umgg" # <-- PASTE YOUR AGENT'S ADDRESS HERE

client = Agent(name="hf_client",
               port=8001,
               endpoint=["http://localhost:8001/submit"])

# @client.on_startup
@client.on_event("startup")
async def run_test(ctx: Context):
    ctx.logger.info("Client agent running...")
    
    if ORCHESTRATOR_AGENT_ADDRESS == "agent1q...":
        ctx.logger.error("\n\n!!! ERROR: Please update 'ORCHESTRATOR_AGENT_ADDRESS' with the address from your agent's logs.\n")
        return

    # 1. Create the request message
    ctx.logger.info(f"Sending request to agent: {ORCHESTRATOR_AGENT_ADDRESS}")
    request_msg = HFRequest(
        model_id="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        task="sentiment-analysis",
        prompt="This is a great tool, it's fast and reliable!",
        requirements=None # Requirements should already be installed by the agent's startup
    )

    # 2. Send the query and wait for the response
    try:
        response = await ctx.query(
            destination=ORCHESTRATOR_AGENT_ADDRESS,
            message=request_msg,
            timeout=300.0 # 5 minute timeout for model download/inference
        )
        
        if response is None:
            ctx.logger.error("\n--- ERROR ---")
            ctx.logger.error("Received no response from agent (is it running?)")
            return

        # 3. Decode and print the response
        response_data = HFResponse.parse_raw(response.decode_payload())
        ctx.logger.info("\n--- AGENT RESPONSE ---")
        ctx.logger.info(f"Status: {response_data.status}")
        ctx.logger.info("Result:")
        ctx.logger.info(json.dumps(response_data.result, indent=2))

    except Exception as e:
        ctx.logger.error(f"\n--- AN ERROR OCCURRED ---")
        ctx.logger.error(e)
    
    finally:
        ctx.logger.info("Client shutting down.")
        ctx.stop()


if __name__ == "__main__":
    client.run()
