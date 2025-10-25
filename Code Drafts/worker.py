import os, json, sys, torch
from transformers import pipeline


summarize_text="""
        A new era of Agent Interoperability AI agents offer a unique opportunity to help people be more productive by autonomously handling many daily recurring or complex tasks. Today, enterprises are increasingly building and deploying autonomous agents to help scale, automate and enhance processes throughout the workplacefrom ordering new laptops, to aiding customer service representatives, to assisting in supply chain planning. To maximize the benefits from agentic AI, it is critical for these agents to be able to collaborate in a dynamic, multi-agent ecosystem across siloed data systems and applications. Enabling agents to interoperate with each other, even if they were built by different vendors or in a different framework, will increase autonomy and multiply productivity gains, while lowering long-term costs for enterprises.
"""
print("Payload starting...", file=sys.stderr, flush=True)

# summarize_text = "The quick brown fox jumps over the lazy dog. The dog, being lazy, did not mind at all. However, the fox was quite energetic and wanted to play more."
        
try:
    
    task = "text2text-generation"
    model_id = "t5-small"
    prompt = f"summarize: {summarize_text}"
    cache_dir = os.environ.get("HF_HOME")

    device = 0 if torch.cuda.is_available() else -1
    
    # --- START OF FIX ---

    # 1. Arguments for the inference call (e.g., max_length)
    inference_args = {}
    # if os.environ.get("MODEL_ARGS"):
    #     inference_args = json.loads(os.environ.get("MODEL_ARGS"))

    # 2. Arguments for the model loader (from_pretrained)
    loader_model_kwargs = {
        "cache_dir": cache_dir
    }

    print(f"Loading pipeline for task: {task}, model: {model_id}", file=sys.stderr, flush=True)

    model_pipeline = pipeline(
        task=task,
        model=model_id,
        device=device,
        model_kwargs=loader_model_kwargs # Pass cache_dir here
        # Do NOT pass **inference_args here
    )
    
    print("Pipeline loaded. Running inference...", file=sys.stderr, flush=True)
    
    # 3. Pass inference arguments to the call itself
    result = model_pipeline(prompt, **inference_args)
    
    # --- END OF FIX ---
    
    print("Inference complete.", file=sys.stderr, flush=True)
    
    print(json.dumps({"status": "success", "output": result}), flush=True)

    # Final success output to stdout
    
except Exception as e:
    print(json.dumps({"status": "error", "message": str(e)}), file=sys.stderr, flush=True)
    sys.exit(1)