import sys
import json
from transformers import pipeline
import torch
import warnings
warnings.filterwarnings('ignore')



#
pipe=None
def execute_hf_model(model_id, prompt, task_type):
    try:
        # Determine device
        device = 0 if torch.cuda.is_available() else -1
        
        print(f"Loading model: {model_id}", file=sys.stderr)
        print(f"Task type: {task_type}", file=sys.stderr)
        print(f"Device: {'GPU' if device == 0 else 'CPU'}", file=sys.stderr)
        
        # Load pipeline based on task type
        if task_type == "auto":
            global pipe
            pipe = pipeline(model=model_id, device=device, trust_remote_code=True)
        else:
            global pipe
            pipe = pipeline(task_type, model=model_id, device=device, trust_remote_code=True)
        
        print("Model loaded successfully", file=sys.stderr)
        print("Processing prompt...", file=sys.stderr)
        
        # Execute model with appropriate parameters
        if task_type == "text-generation" or (task_type == "auto" and "gpt" in model_id.lower()):
            result = pipe(
                prompt, 
                max_new_tokens=100,
                return_full_text=False,
                do_sample=True,
                temperature=0.7
            )
        else:
            result = pipe(prompt)
        
        # Parse result
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict):
                if 'generated_text' in result[0]:
                    output = result[0]['generated_text']
                elif 'summary_text' in result[0]:
                    output = result[0]['summary_text']
                elif 'translation_text' in result[0]:
                    output = result[0]['translation_text']
                elif 'label' in result[0]:
                    output = f"Label: {result[0]['label']}, Score: {result[0].get('score', 'N/A')}"
                else:
                    output = str(result[0])
            else:
                output = str(result[0])
        else:
            output = str(result)
        
        # Return as JSON for reliable parsing
        response = {
            "success": True,
            "output": output,
            "model_id": model_id
        }
        print(json.dumps(response))
        
    except Exception as e:
        error_response = {
            "success": False,
            "error": str(e),
            "model_id": model_id
        }
        print(json.dumps(error_response))
        sys.exit(1)



def generate_outptut(model_id, prompt, task_type):
    pass


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(json.dumps({"success": False, "error": "Missing arguments"}))
        sys.exit(1)
    
    model_id = sys.argv[1]
    prompt = sys.argv[2]
    task_type = sys.argv[3]
    
    execute_hf_model(model_id, prompt, task_type)