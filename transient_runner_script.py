def generate_transient_runner_content(model_id: str, prompt: str, task_type: str) -> str:
    """
    Path A Runner: Single-use, executes one prompt then exits.
    """
    return """
import sys
import json
from transformers import pipeline
import torch
import warnings
warnings.filterwarnings('ignore')

MODEL_ID = 1
PROMPT = 2
TASK_TYPE = 3

def execute_hf_model(model_id, prompt, task_type):
    try:
        device = 0 if torch.cuda.is_available() else -1
        if task_type == "auto":
            pipe = pipeline(model=model_id, device=device, trust_remote_code=True)
        else:
            pipe = pipeline(task_type, model=model_id, device=device, trust_remote_code=True)

        if task_type == "text-generation" or (task_type == "auto" and any(x in model_id.lower() for x in ["gpt", "llama"])):
            result = pipe(prompt, max_new_tokens=100, return_full_text=False, do_sample=True, temperature=0.7)
        else:
            result = pipe(prompt)

        output = ""
        if isinstance(result, list) and result and isinstance(result[0], dict):
            key = next((k for k in ["generated_text", "summary_text", "translation_text", "answer"] if k in result[0]), "label")
            if key == "label":
                output = f"Label: {result[0]['label']}, Score: {result[0].get('score', 'N/A')}"
            else:
                output = result[0][key]
        else:
            output = str(result)

        print(json.dumps({"success": True, "output": output, "model_id": model_id}))
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e), "model_id": model_id}))
        sys.exit(1)

if __name__ == "__main__":
    execute_hf_model({MODEL_ID}, {PROMPT}, {TASK_TYPE})
""".replace("MODEL_ID=1", f'"MODEL_ID = { model_id}"') \
  .replace("PROMPT=2", f'"PROMPT = {prompt}"') \
  .replace("TASK_TYPE=3", f'"TASK_TYPE = {task_type}"')
