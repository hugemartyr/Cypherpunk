import os
import sys
import json
import uuid
import dotenv
from openai import OpenAI

# Load environment variables from .env file
dotenv.load_dotenv()

# --- ASI1.ai API Configuration ---
API_KEY = os.environ.get('ASI_ONE_API_KEY') 
BASE_URL = "https://api.asi1.ai/v1"
# We will use the agentic model which is optimized for tool use and multi-step tasks
MODEL_NAME = "asi1-agentic" 

if not API_KEY:
    print("FATAL ERROR: ASI_ONE_API_KEY environment variable is not set.")
    sys.exit(1)

# 1. Initialize the OpenAI client
try:
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )
except Exception as e:
    print(f"FATAL ERROR: Failed to initialize OpenAI client: {e}")
    sys.exit(1)


# 2. Initialize conversation history with a system prompt
conversation_history = [
    {"role": "system", "content": "You are a helpful and concise agent specializing in autonomous economic agents and task execution. Keep your answers brief and focused."}
]

# 3. Generate a single session ID for the entire conversation session
# This is crucial for agentic models to maintain state across related calls.
session_id = str(uuid.uuid4())

# --- Start Chat Loop ---
print("\n--- ASI1.ai Streaming Agent Chat Terminal ---")
print(f"Model: {MODEL_NAME}. Session ID: {session_id}")
print("Type 'quit' or 'exit' to end the chat.")
print("-------------------------------------------\n")

while True:
    # Get user input
    user_input = input("You: ")

    # Check for exit command
    if user_input.lower() in ["quit", "exit"]:
        print("\nChat ended. Goodbye!")
        break

    # Append the new user message to the conversation history
    conversation_history.append({"role": "user", "content": user_input})
    
    # Placeholder for the assistant's full response content from the stream
    full_assistant_response = ""
    
    # Display waiting message
    print("ASI1: ", end="", flush=True)

    try:
        # --- API Call using OpenAI Client ---
        # The client automatically uses the base_url and api_key set during initialization
        response_stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=conversation_history,
            # Pass the session ID via extra_headers
            extra_headers={
                "x-session-id": session_id
            },
            temperature=0.7,
            stream=True # Crucial for streaming
        )

        # Process the streaming response
        for chunk in response_stream:
            # Check if there is content in the delta
            content = chunk.choices[0].delta.content
            if content:
                # Print the chunk immediately
                print(content, end="", flush=True)
                # Accumulate the content for history
                full_assistant_response += content

        # Print a newline after the stream is complete for clean terminal formatting
        print() 

        # 4. Append the full assistant's reply to the history for context in the next turn
        if full_assistant_response:
            conversation_history.append({"role": "assistant", "content": full_assistant_response})
        else:
            # Handle cases where the response was empty but the call didn't error out
            print("\n[Warning: Received an empty response from the model.]")
            conversation_history.pop() # Remove the last user message

    except Exception as e:
        # Handle all exceptions (network, API errors, etc.)
        print(f"\n[ASI1 ERROR: An unexpected error occurred: {e}]")
        # Remove the user message that caused the failure to prevent context pollution
        conversation_history.pop() 
        
