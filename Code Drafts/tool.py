import os
import json
import sys
import torch
from transformers import pipeline, AutoConfig

def run_inference(task, model_id, prompt, model_args=None):
    """
    Runs Hugging Face inference based on environment variables.

    Expects:
    - TASK: The HF pipeline task (e.g., "summarization", "sentiment-analysis")
    - MODEL_ID: The model ID from Hugging Face Hub (e.g., "facebook/bart-large-cnn")
    - PROMPT: The input text to be processed.
    - (Optional) MODEL_ARGS: A JSON string of extra args for the pipeline.
    """
    try:
        # # 1. Get inputs from environment variables
        # task = os.environ.get("TASK")
        # model_id = os.environ.get("MODEL_ID")
        # prompt = os.environ.get("PROMPT")

        if not task or not model_id or not prompt:
            raise ValueError("Missing required env vars: TASK, MODEL_ID, PROMPT")
        
        # 2. Set up device (use GPU if CUDOS provides one)
        device = 0 if torch.cuda.is_available() else -1

        # 3. Load extra arguments (if any)
        # This lets you pass things like max_length, etc.
        model_args = {}
        if os.environ.get("MODEL_ARGS"):
            model_args = json.loads(os.environ.get("MODEL_ARGS"))

        # 4. Load the model and create the pipeline
        # This will download the model *inside the container* the first time
        # and cache it for future runs (if the container is re-used, 
        # but we assume it's new).
        model_pipeline = pipeline(
            task=task,
            model=model_id,
            device=device,
            **model_args
        )

        # 5. Run inference
        result = model_pipeline(prompt)

        # 6. Return the result as a JSON string to stdout
        # The main agent will capture this.
        print(json.dumps({"status": "success", "output": result}))

    except Exception as e:
        # 7. Return any errors as JSON to stderr
        # This is good for debugging.
        print(json.dumps({"status": "error", "message": str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    run_inference("summarization", "facebook/bart-large-cnn", "The quick brown fox jumps over the lazy dog. The dog, being lazy, did not mind at all. However, the fox was quite energetic and wanted to play more.")
