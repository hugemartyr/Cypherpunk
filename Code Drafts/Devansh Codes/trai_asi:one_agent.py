import requests
import os
import json
import dotenv
import sys # Import sys for flushing output

dotenv.load_dotenv()

# --- ASI1.ai API Configuration ---
url = "https://api.asi1.ai/v1"
api_key = os.environ.get('ASI_ONE_API_KEY') 
model_name = "asi1-mini" # The model you are using

if not api_key:
    print("FATAL ERROR: ASI_ONE_API_KEY environment variable is not set.")
    sys.exit(1) # Use sys.exit(1) for cleaner exit on error

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# 1. Initialize conversation history with an optional system prompt
# The API requires a list of message objects for context
conversation_history = [
    {"role": "system", "content": "You are a helpful and concise assistant specializing in Fetch.ai and autonomous economic agents. Keep your answers brief and focused."}
]

# --- Start Chat Loop ---
print("\n--- ASI1.ai Chat Terminal ---")
print(f"Model: {model_name}. Type 'quit' or 'exit' to end the chat.")
print("---------------------------\n")

while True:
    # 2. Get user input
    user_input = input("You: ")

    # Check for exit command
    if user_input.lower() in ["quit", "exit"]:
        print("\nChat ended. Goodbye!")
        break

    # 3. Append the new user message to the conversation history
    conversation_history.append({"role": "user", "content": user_input})
    
    # 4. Construct the body with the full conversation history
    body = {
        "model": model_name,
        "messages": conversation_history
    }

    # Display waiting message (optional)
    print("ASI1: Thinking...", end="", flush=True)

    try:
        # --- API Call ---
        response = requests.post(url, headers=headers, json=body)
        response_data = response.json()

        # Check for HTTP errors (e.g., 401, 404, 500)
        response.raise_for_status() 

        # --- Process Successful Response ---
        if 'choices' in response_data and response_data['choices']:
            # Extract the assistant's reply
            llm_response_content = response_data["choices"][0]["message"]["content"]
            
            # Print the LLM's response
            print("\rASI1: " + llm_response_content) # \r returns cursor to start of line to overwrite "Thinking..."
            
            # 5. Append the assistant's reply to the history for context in the next turn
            conversation_history.append({"role": "assistant", "content": llm_response_content})
            
        else:
            # Handle cases where status is 200, but 'choices' is missing/empty
            print("\rASI1: ERROR: Successful request but missing 'choices' key in response.")
            print("Full Response (for debugging):", json.dumps(response_data, indent=2))
            # Remove the last user message to avoid using it in the next loop
            conversation_history.pop()

    except requests.exceptions.HTTPError as e:
        # Handle 4xx/5xx errors
        error_message = response_data.get('error', {}).get('message', 'No error message provided.')
        print(f"\rASI1: HTTP ERROR {response.status_code}: {error_message}")
        conversation_history.pop() # Remove failed user message

    except Exception as e:
        print(f"\rASI1: An unexpected error occurred: {e}")
        conversation_history.pop() # Remove failed user message