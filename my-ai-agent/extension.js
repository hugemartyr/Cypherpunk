const vscode = require("vscode");
const { TextEncoder } = require("util"); // Node.js util for TextEncoder
const fs = require("fs");
const path = require("path");
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
    // Read the HTML file from the file system
    const htmlPath = path.join(__dirname, "chatbox.html");

    try {
      const htmlContent = fs.readFileSync(htmlPath, "utf8");
      return htmlContent;
    } catch (error) {
      console.error("Error reading chatbox.html:", error);
      return `
        <!DOCTYPE html>
        <html>
        <head>
          <title>AI Agent Chat</title>
        </head>
        <body>
          <h1>Error loading chat interface</h1>
          <p>Could not load chatbox.html: ${error.message}</p>
        </body>
        </html>
      `;
    }
  }
}

// This method is called when your extension is deactivated
function deactivate() {}

module.exports = {
  activate,
  deactivate,
};
