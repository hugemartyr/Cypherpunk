const vscode = require("vscode");
const { TextEncoder } = require("util"); // Node.js util for TextEncoder

/**
 * @param {vscode.ExtensionContext} context
 */
function activate(context) {
  console.log('Congratulations, your extension "my-ai-agent" is now active!');

  const provider = new ChatViewProvider(context.extensionUri);

  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(
      ChatViewProvider.viewType,
      provider
    )
  );
}

/**
 * Manages the chat view in the sidebar.
 */
class ChatViewProvider {
  static viewType = "my-ai-agent.chatView";

  _view;
  _extensionUri;

  /**
   * @param {vscode.Uri} extensionUri
   */
  constructor(extensionUri) {
    this._extensionUri = extensionUri;
  }

  /**
   * @param {vscode.WebviewView} webviewView
   * @param {vscode.WebviewViewResolveContext} context
   * @param {vscode.CancellationToken} _token
   */
  resolveWebviewView(webviewView, context, _token) {
    this._view = webviewView;

    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: [this._extensionUri],
    };

    webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);

    // Listen for messages from the webview
    webviewView.webview.onDidReceiveMessage(async (message) => {
      switch (message.type) {
        // Case: The webview's JS told us the user sent a prompt
        case "userPrompt":
          const userPrompt = message.text;

          // --- MODIFICATION START ---
          // Instead of echoing, let's call our backend server
          try {
            const agentResponse = await this.callAgentServer(userPrompt);

            // Send the server's response back to the webview
            this._view?.webview.postMessage({
              type: "agentResponse",
              text: agentResponse,
            });
          } catch (error) {
            console.error("Error calling agent server:", error);
            const errorMessage = `Error connecting to agent: ${error.message}. Is your Python server running?`;
            this._view?.webview.postMessage({
              type: "agentResponse",
              text: errorMessage,
            });
          }
          // --- MODIFICATION END ---
          break;

        // Case: The webview told us it has finished loading
        case "agentReady":
          this._view?.webview.postMessage({
            type: "agentResponse",
            text: "Hello! I am your AI Agent, connected to a local server. How can I help?",
          });
          break;
      }
    });
  }

  /**
   * New function to call the local Python server.
   * @param {string} prompt
   */
  async callAgentServer(prompt) {
    // URL of your local server
    const url = "http://localhost:8080/chat";

    try {
      // Use fetch (available in Node.js 18+, which VS Code uses)
      const response = await fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          prompt: prompt,
        }),
      });

      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }

      const data = await response.json();

      // We expect the server to return a JSON like {"response": "..."}
      return data.response || "Server returned an unexpected format.";
    } catch (error) {
      console.error("Fetch error:", error);
      // Re-throw to be caught by the message handler
      throw new Error(error.message);
    }
  }

  /**
   * Generates the self-contained HTML for the webview.
   * @param {vscode.Webview} webview
   */
  _getHtmlForWebview(webview) {
    // --- THIS HTML IS UNCHANGED FROM THE PREVIOUS VERSION ---
    // (It already knows how to send 'userPrompt' and receive 'agentResponse')
    return `
	<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Agent Chat - Crypto Theme</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body, html {
            height: 100%;
            width: 100%;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1428 100%);
            color: #e0e0ff;
            overflow: hidden;
        }

        #chat-container {
            display: flex;
            flex-direction: column;
            height: 100vh;
            background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1428 100%);
            position: relative;
        }

        /* Animated background gradient */
        #chat-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: radial-gradient(circle at 20% 50%, rgba(0, 255, 200, 0.05) 0%, transparent 50%),
                        radial-gradient(circle at 80% 80%, rgba(138, 43, 226, 0.05) 0%, transparent 50%);
            pointer-events: none;
            animation: gradientShift 15s ease-in-out infinite;
        }

        @keyframes gradientShift {
            0%, 100% {
                opacity: 0.5;
            }
            50% {
                opacity: 1;
            }
        }

        /* Header */
        #header {
            padding: 20px;
            background: linear-gradient(90deg, rgba(0, 255, 200, 0.1) 0%, rgba(138, 43, 226, 0.1) 100%);
            border-bottom: 2px solid rgba(0, 255, 200, 0.3);
            text-align: center;
            position: relative;
            z-index: 1;
            box-shadow: 0 4px 20px rgba(0, 255, 200, 0.1);
        }

        #header h1 {
            font-size: 24px;
            background: linear-gradient(90deg, #00ffc8 0%, #8a2be2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: 700;
            letter-spacing: 1px;
            text-shadow: 0 0 20px rgba(0, 255, 200, 0.3);
        }

        #messages {
            flex-grow: 1;
            overflow-y: auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 12px;
            position: relative;
            z-index: 1;
        }

        /* Scrollbar styling */
        #messages::-webkit-scrollbar {
            width: 8px;
        }

        #messages::-webkit-scrollbar-track {
            background: rgba(0, 255, 200, 0.05);
            border-radius: 10px;
        }

        #messages::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, #00ffc8 0%, #8a2be2 100%);
            border-radius: 10px;
        }

        #messages::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(180deg, #00ffdd 0%, #a855f7 100%);
        }

        .message {
            display: flex;
            animation: slideIn 0.5s ease-out;
            max-width: 70%;
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        /* Agent message - left aligned */
        .agent-message {
            align-self: flex-start;
            animation: slideInLeft 0.5s ease-out;
        }

        @keyframes slideInLeft {
            from {
                opacity: 0;
                transform: translateX(-30px) translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateX(0) translateY(0);
            }
        }

        .agent-message .message-content {
            background: linear-gradient(135deg, rgba(0, 255, 200, 0.15) 0%, rgba(0, 200, 150, 0.1) 100%);
            border: 1px solid rgba(0, 255, 200, 0.4);
            border-left: 3px solid #00ffc8;
            box-shadow: 0 8px 32px rgba(0, 255, 200, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        }

        .agent-message .message-sender {
            color: #00ffc8;
            text-shadow: 0 0 10px rgba(0, 255, 200, 0.5);
        }

        /* User message - right aligned */
        .user-message {
            align-self: flex-end;
            animation: slideInRight 0.5s ease-out;
        }

        @keyframes slideInRight {
            from {
                opacity: 0;
                transform: translateX(30px) translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateX(0) translateY(0);
            }
        }

        .user-message .message-content {
            background: linear-gradient(135deg, rgba(138, 43, 226, 0.2) 0%, rgba(100, 50, 200, 0.15) 100%);
            border: 1px solid rgba(138, 43, 226, 0.5);
            border-right: 3px solid #a855f7;
            box-shadow: 0 8px 32px rgba(138, 43, 226, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        }

        .user-message .message-sender {
            color: #a855f7;
            text-shadow: 0 0 10px rgba(168, 85, 247, 0.5);
        }

        .message-content {
            padding: 12px 16px;
            border-radius: 12px;
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
        }

        .message-content:hover {
            box-shadow: 0 12px 40px rgba(0, 255, 200, 0.2);
            transform: translateY(-2px);
        }

        .message-sender {
            font-weight: 700;
            font-size: 12px;
            margin-bottom: 6px;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }

        .message-text {
            font-size: 14px;
            line-height: 1.5;
            white-space: pre-wrap;
            word-wrap: break-word;
            color: #e0e0ff;
        }

        /* Input area */
        #input-area {
            display: flex;
            gap: 12px;
            padding: 20px;
            background: linear-gradient(90deg, rgba(0, 255, 200, 0.05) 0%, rgba(138, 43, 226, 0.05) 100%);
            border-top: 2px solid rgba(0, 255, 200, 0.2);
            position: relative;
            z-index: 1;
            box-shadow: 0 -4px 20px rgba(0, 0, 0, 0.3);
        }

        #prompt-input {
            flex-grow: 1;
            padding: 12px 16px;
            border: 2px solid rgba(0, 255, 200, 0.3);
            background: rgba(15, 20, 40, 0.8);
            color: #e0e0ff;
            border-radius: 8px;
            font-family: inherit;
            font-size: 14px;
            resize: none;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
            max-height: 100px;
        }

        #prompt-input:focus {
            outline: none;
            border-color: #00ffc8;
            box-shadow: 0 0 20px rgba(0, 255, 200, 0.4), inset 0 0 10px rgba(0, 255, 200, 0.1);
            background: rgba(15, 20, 40, 0.95);
        }

        #prompt-input::placeholder {
            color: rgba(224, 224, 255, 0.4);
        }

        #send-button {
            padding: 12px 24px;
            background: linear-gradient(135deg, #00ffc8 0%, #00d9b3 100%);
            color: #0a0e27;
            border: none;
            border-radius: 8px;
            font-weight: 700;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            box-shadow: 0 4px 15px rgba(0, 255, 200, 0.3);
            position: relative;
            overflow: hidden;
        }

        #send-button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
            transition: left 0.5s ease;
        }

        #send-button:hover {
            background: linear-gradient(135deg, #00ffdd 0%, #00e6c3 100%);
            box-shadow: 0 6px 25px rgba(0, 255, 200, 0.5);
            transform: translateY(-2px);
        }

        #send-button:hover::before {
            left: 100%;
        }

        #send-button:active {
            transform: translateY(0);
            box-shadow: 0 2px 10px rgba(0, 255, 200, 0.3);
        }

        /* Responsive design */
        @media (max-width: 768px) {
            .message {
                max-width: 85%;
            }

            #header h1 {
                font-size: 18px;
            }

            #messages {
                padding: 12px;
                gap: 8px;
            }

            #input-area {
                padding: 12px;
                gap: 8px;
            }

            #prompt-input {
                font-size: 13px;
                padding: 10px 12px;
            }

            #send-button {
                padding: 10px 16px;
                font-size: 12px;
            }
        }

        @media (max-width: 480px) {
            .message {
                max-width: 95%;
            }

            #header h1 {
                font-size: 16px;
            }

            #messages {
                padding: 8px;
                gap: 6px;
            }

            #input-area {
                padding: 8px;
                gap: 6px;
                flex-direction: column;
            }

            #send-button {
                width: 100%;
            }
        }
    </style>
</head>
<body>
    <div id="chat-container">
        <div id="header">
            <h1>⚡ AI Agent Chat</h1>
        </div>
        <div id="messages">
            <!-- Messages will be added here by JS -->
        </div>
        <div id="input-area">
            <textarea id="prompt-input" rows="2" placeholder="Type your message..."></textarea>
            <button id="send-button">Send</button>
        </div>
    </div>

    <script>
        // Standard VS Code webview script boilerplate
        const vscode = acquireVsCodeApi();

        const messagesDiv = document.getElementById('messages');
        const input = document.getElementById('prompt-input');
        const sendButton = document.getElementById('send-button');

        /**
         * Appends a message to the chat display with animation.
         */
        function addMessage(sender, text, type) {
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message ' + type;
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            
            const senderDiv = document.createElement('div');
            senderDiv.className = 'message-sender';
            senderDiv.textContent = sender;
            
            const textDiv = document.createElement('div');
            textDiv.className = 'message-text';
            textDiv.textContent = text;
            
            contentDiv.appendChild(senderDiv);
            contentDiv.appendChild(textDiv);
            messageDiv.appendChild(contentDiv);
            messagesDiv.appendChild(messageDiv);
            
            // Scroll to bottom with smooth behavior
            setTimeout(() => {
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }, 50);
        }

        /**
         * Handles sending the prompt.
         */
        function sendPrompt() {
            const text = input.value.trim();
            if (text) {
                // Display the user's message immediately
                addMessage('You', text, 'user-message');
                
                // Send the message to the extension
                vscode.postMessage({
                    type: 'userPrompt',
                    text: text
                });
                
                // Clear the input
                input.value = '';
                input.style.height = 'auto';
            }
        }

        // Auto-resize textarea
        input.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 100) + 'px';
        });

        // --- Event Listeners ---

        // Send on button click
        sendButton.addEventListener('click', sendPrompt);

        // Send on 'Enter' (but not 'Shift+Enter')
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendPrompt();
            }
        });

        // Listen for messages from the extension
        window.addEventListener('message', event => {
            const message = event.data;
            if (message.type === 'agentResponse') {
                addMessage('AI Agent', message.text, 'agent-message');
            }
        });

        // Tell the extension that the webview is ready
        window.onload = () => {
            vscode.postMessage({ type: 'agentReady' });
        };
    </script>
</body>
</html>

	`;
  }
}

// This method is called when your extension is deactivated
function deactivate() {}

module.exports = {
  activate,
  deactivate,
};
