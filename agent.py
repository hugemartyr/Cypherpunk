# A simple Flask server to act as your AI agent backend.
#
# 1. Install Flask:
#    pip install Flask flask-cors
#
# 2. Run this server:
#    python python_server.py
#
# It will run on http://localhost:8080

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
# Enable CORS to allow your VS Code extension (which runs on a different origin)
# to make requests to this server.
CORS(app)

@app.route('/chat', methods=['POST'])
def chat():
    # Get the prompt from the VS Code extension
    data = request.json
    prompt = data.get('prompt')

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    print(f"Received prompt: {prompt}")

    # --- THIS IS WHERE YOU WOULD CALL GEMINI ---
    # For now, we'll just echo the prompt.
    # Replace this logic with your actual AI call.
    ai_response = f"###. My Python server received your prompt: '{prompt}'"
    
    # Send the response back to the VS Code extension
    return jsonify({
        "response": ai_response
    })

if __name__ == '__main__':
    # Run the server on port 8080, accessible from any IP
    app.run(host='0.0.0.0', port=8080, debug=True)
