// // The module 'vscode' contains the VS Code extensibility API
// // Import the module and reference it with the alias vscode in your code below
// const vscode = require('vscode');

// // This method is called when your extension is activated
// // Your extension is activated the very first time the command is executed

// /**
//  * @param {vscode.ExtensionContext} context
//  */
// function activate(context) {

// 	// Use the console to output diagnostic information (console.log) and errors (console.error)
// 	// This line of code will only be executed once when your extension is activated
// 	console.log('Congratulations, your extension "my-ai-agent" is now active!');

// 	// The command has been defined in the package.json file
// 	// Now provide the implementation of the command with  registerCommand
// 	// The commandId parameter must match the command field in package.json
// 	const disposable = vscode.commands.registerCommand('my-ai-agent.helloWorld', function () {
// 		// The code you place here will be executed every time your command is executed
// 		console.log('### hello bhaii Executing Hello World command from my-ai-agent extension.');
// 		// Display a message box to the user
// 		vscode.window.showInformationMessage('Hello World from my-ai-agent!');
// 	});
// 	let disposable2 = vscode.commands.registerCommand('my-ai-agent.createFile', async () => {

// 		console.log('### Executing Create File command from my-ai-agent extension.');
// 		vscode.window.showInformationMessage('Creating file using AI Agent...');
		
// 		// 1. Get the user's open folder (workspace)
// 		const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
// 		if (!workspaceFolder) {
// 		  vscode.window.showErrorMessage('You must have a folder open to use this extension.');
// 		  return;
// 		}
// 		const workspaceUri = workspaceFolder.uri;

// 		// 2. Ask the user for instructions
// 		const userPrompt = await vscode.window.showInputBox({
// 		  prompt: 'What file should I create and what code should it contain?',
// 		  placeHolder: 'e.g., "Create a python file app.py that prints hello world"',
// 		});

// 		if (!userPrompt) {
// 		  vscode.window.showInformationMessage('Command cancelled.');
// 		  return;
// 		}

// 		createFileInWorkspace(workspaceUri, "hello.txt", "This is a test file created by AI Agent.");

// 		// Show a loading message
// 		vscode.window.withProgress({
// 		  location: vscode.ProgressLocation.Notification,
// 		  title: 'AI Agent is working...',
// 		  cancellable: false
// 		}, async (progress) => {

// 		  try {
// 		    // 3. Call the AI agent
// 		    progress.report({ message: 'Contacting AI...' });
// 		    // const aiResponse = await callAIAgent(userPrompt);

// 			console.log("User Prompt: ", userPrompt);

// 			sleep(2000);

// 		    // if (aiResponse && aiResponse.fileName && typeof aiResponse.content === 'string') {

// 		    //   // 4. Create and write the file
// 		    //   progress.report({ message: `Creating file: ${aiResponse.fileName}` });
// 		    //   await createFileInWorkspace(workspaceUri, aiResponse.fileName, aiResponse.content);
			
// 		    //   vscode.window.showInformationMessage(`Successfully created ${aiResponse.fileName}!`);
// 		    // } else {
// 		    //   throw new Error('Received invalid response from AI.');
// 		    // }

			
// 		  } catch (error) {
// 		    console.error(error);
// 		    vscode.window.showErrorMessage(`Error communicating with AI: ${error instanceof Error ? error.message : 'Unknown error'}`);
// 		  }
// 		});
// 	});

// 	context.subscriptions.push(disposable, disposable2);
// }


// // This method is called when your extension is deactivated
// function deactivate() {}

// module.exports = {
// 	activate,
// 	deactivate
// }
// // 

// // import * as vscode from 'vscode';
// // import { TextEncoder } from 'util'; // Use Node.js util for TextEncoder

// // // --- Gemini API Configuration ---
// // const GEMINI_API_KEY = ""; // Leave this as-is
// // const GEMINI_API_URL = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key=${GEMINI_API_KEY}`;


// // // Define the JSON schema for the AI's response
// // const RESPONSE_SCHEMA = {
// //   type: "OBJECT",
// //   properties: {
// //     "fileName": {
// //       "type": "STRING",
// //       "description": "The relative path for the file to be created, e.g., 'src/app.js' or 'index.html'."
// //     },
// //     "content": {
// //       "type": "STRING",
// //       "description": "The code or text content to be written into the file. All code should be complete."
// //     }
// //   },
// //   required: ["fileName", "content"]
// // };

// // /**
// //  * This is the main entry point for your extension.
// //  * It's called when your extension is activated (e.g., when the command is run).
// //  */
// // export function activate(context) {

// //   // Register the command from package.json
// //   let disposable = vscode.commands.registerCommand('my-ai-agent.createFile', async () => {

// // 	console.log('### Executing Create File command from my-ai-agent extension.');
// // 	vscode.window.showInformationMessage('Creating file using AI Agent...');
    
// //     // // 1. Get the user's open folder (workspace)
// //     // const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
// //     // if (!workspaceFolder) {
// //     //   vscode.window.showErrorMessage('You must have a folder open to use this extension.');
// //     //   return;
// //     // }
// //     // const workspaceUri = workspaceFolder.uri;

// //     // // 2. Ask the user for instructions
// //     // const userPrompt = await vscode.window.showInputBox({
// //     //   prompt: 'What file should I create and what code should it contain?',
// //     //   placeHolder: 'e.g., "Create a python file app.py that prints hello world"',
// //     // });

// //     // if (!userPrompt) {
// //     //   vscode.window.showInformationMessage('Command cancelled.');
// //     //   return;
// //     // }

// //     // // Show a loading message
// //     // vscode.window.withProgress({
// //     //   location: vscode.ProgressLocation.Notification,
// //     //   title: 'AI Agent is working...',
// //     //   cancellable: false
// //     // }, async (progress) => {

// //     //   try {
// //     //     // 3. Call the AI agent
// //     //     progress.report({ message: 'Contacting AI...' });
// //     //     // const aiResponse = await callAIAgent(userPrompt);

// // 	// 	console.log("User Prompt: ", userPrompt);

// // 	// 	sleep(2000);

// //     //     // if (aiResponse && aiResponse.fileName && typeof aiResponse.content === 'string') {

// //     //     //   // 4. Create and write the file
// //     //     //   progress.report({ message: `Creating file: ${aiResponse.fileName}` });
// //     //     //   await createFileInWorkspace(workspaceUri, aiResponse.fileName, aiResponse.content);
          
// //     //     //   vscode.window.showInformationMessage(`Successfully created ${aiResponse.fileName}!`);
// //     //     // } else {
// //     //     //   throw new Error('Received invalid response from AI.');
// //     //     // }

		
// //     //   } catch (error) {
// //     //     console.error(error);
// //     //     vscode.window.showErrorMessage(`Error communicating with AI: ${error instanceof Error ? error.message : 'Unknown error'}`);
// //     //   }
// //     // });
// //   });

// //   context.subscriptions.push(disposable);
// // }

// // /**
// //  * Calls the Gemini API to get file instructions.
// //  * @param userPrompt The user's text prompt.
// //  * @returns A structured AIFileResponse.
// //  */
// async function callAIAgent(userPrompt) {
//   const systemInstruction = `You are an expert programmer and file system agent. 
//   The user will give you a prompt to create a file. 
//   You MUST respond ONLY with a single, valid JSON object that matches the following schema.
//   Do not include markdown, backticks, or any text outside of the JSON object.
//   The 'content' field must contain the full code for the file, correctly formatted.`;

//   const payload = {
//     contents: [{
//       parts: [{ text: userPrompt }]
//     }],
//     systemInstruction: {
//       parts: [{ text: systemInstruction }]
//     },
//     generationConfig: {
//       responseMimeType: "application/json",
//       responseSchema: RESPONSE_SCHEMA,
//     }
//   };

//   // Call API with exponential backoff
//   let response;
//   let retries = 0;
//   const maxRetries = 3;

//   while (retries < maxRetries) {
//     try {
//       response = await fetch(GEMINI_API_URL, {
//         method: 'POST',
//         headers: { 'Content-Type': 'application/json' },
//         body: JSON.stringify(payload)
//       });

//       if (response.ok) {
//         break; // Success
//       }

//       if (response.status === 429 || response.status >= 500) {
//         // Throttling or server error, wait and retry
//         const delay = Math.pow(2, retries) * 1000 + Math.random() * 1000;
//         await new Promise(resolve => setTimeout(resolve, delay));
//         retries++;
//       } else {
//         // Other client error
//         throw new Error(`API request failed with status ${response.status}: ${await response.text()}`);
//       }

//     } catch (error) {
//       if (retries + 1 >= maxRetries) {
//         throw error; // Max retries reached
//       }
//       const delay = Math.pow(2, retries) * 1000 + Math.random() * 1000;
//       await new Promise(resolve => setTimeout(resolve, delay));
//       retries++;
//     }
//   }

// //   // @ts-ignore
// //   if (!response.ok) {
// //     throw new Error(`API request failed after retries with status ${response.status}`);
// //   }

// //   const result = await response.json();
// //   const text = result.candidates?.[0]?.content?.parts?.[0]?.text;

// //   if (!text) {
// //     throw new Error('No text response from AI.');
// //   }

// //   try {
// //     // The AI's response text *is* the JSON object
// //     return JSON.parse(text) ;
// //   } catch (e) {
// //     console.error("Failed to parse AI JSON response:", text);
// //     throw new Error('AI returned malformed JSON.');
// //   }
// }

// /**
//  * Creates a file in the workspace and opens it.
//  * @param workspaceUri The URI of the root workspace folder.
//  * @param fileName The relative path of the file to create.
//  * @param content The string content to write to the file.
//  */
// async function createFileInWorkspace(workspaceUri, fileName, content) {
//   try {
//     // 1. Get the full URI for the new file
//     const fileUri = vscode.Uri.joinPath(workspaceUri, fileName);

//     // 2. Convert string content to Uint8Array (required by fs.writeFile)
//     const encodedContent = new TextEncoder().encode(content);

//     // 3. Write the file
//     // This will create directories if they don't exist and overwrite the file if it does.
//     await vscode.workspace.fs.writeFile(fileUri, encodedContent);

//     // 4. Open the new file in the editor
//     const document = await vscode.workspace.openTextDocument(fileUri);
//     await vscode.window.showTextDocument(document);

//   } catch (error) {
//     console.error(error);
//     vscode.window.showErrorMessage(`Failed to create file: ${error instanceof Error ? error.message : 'Unknown error'}`);
//   }
// }

// // This function is called when your extension is deactivated
// // export function deactivate() {}
// function sleep(ms) {
// 	// Returns a promise that resolves after `ms` milliseconds.
// 	// Use: await sleep(2000);
// 	return new Promise(resolve => setTimeout(resolve, ms));
// }

// The module 'vscode' contains the VS Code extensibility API
// Import the module and reference it with the alias vscode in your code below
const vscode = require('vscode');
const { TextEncoder } = require('util'); // Node.js util for TextEncoder

// This method is called when your extension is activated
// Your extension is activated the very first time the command is executed

/**
 * @param {vscode.ExtensionContext} context
 */
function activate(context) {
  // Use the console to output diagnostic information (console.log) and errors (console.error)
  // This line of code will only be executed once when your extension is activated
  console.log('Congratulations, your extension "my-ai-agent" is now active!');

  // Create a new instance of our ChatViewProvider
  const provider = new ChatViewProvider(context.extensionUri);

  // Register the provider for the sidebar view defined in package.json
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(ChatViewProvider.viewType, provider)
  );
}

/**
 * Manages the chat view in the sidebar.
 */
class ChatViewProvider {
  static viewType = 'my-ai-agent.chatView';

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

    // Configure the webview
    webviewView.webview.options = {
      // Enable scripts in the webview
      enableScripts: true,
      // Restrict the webview to only loading content from our extension's directory
      localResourceRoots: [this._extensionUri]
    };

    // Set the HTML content for the webview
    webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);

    // Handle messages from the webview (our HTML/JS)
    webviewView.webview.onDidReceiveMessage(message => {
      switch (message.type) {
        // Case: The webview's JS told us the user sent a prompt
        case 'userPrompt':
          const userPrompt = message.text;

          // --- This is where you would call the Gemini API ---
          // For now, we'll just echo the response as requested.
          const agentResponse = `Your prompt is "${userPrompt}"`;

          // Send the response back to the webview
          this._view?.webview.postMessage({ type: 'agentResponse', text: agentResponse });
          break;

        // Case: The webview told us it has finished loading
        case 'agentReady':
          this._view?.webview.postMessage({ type: 'agentResponse', text: 'Hello! I am your AI Agent. How can I help you today?' });
          break;
      }
    });
  }

  /**
   * Generates the self-contained HTML for the webview.
   * @param {vscode.Webview} webview
   */
  _getHtmlForWebview(webview) {
    // Note: We're inlining all HTML, CSS, and JS for simplicity.
    // In a larger extension, you would load these from separate files.

    return `<!DOCTYPE html>
      <html lang="en">
      <head>
          <meta charset="UTF-8">
          <meta name="viewport" content="width=device-width, initial-scale=1.0">
          <title>AI Agent Chat</title>
          <style>
            /* Use VS Code CSS variables for a native look */
            body, html {
              margin: 0;
              padding: 0;
              height: 100%;
              color: var(--vscode-foreground);
              background-color: var(--vscode-sideBar-background, var(--vscode-editor-background));
              font-family: var(--vscode-font-family);
            }
            #chat-container {
              display: flex;
              flex-direction: column;
              height: 100vh;
              padding: 8px;
              box-sizing: border-box; /* Include padding in height */
            }
            #messages {
              flex-grow: 1;
              overflow-y: auto;
              padding-bottom: 8px; /* Space above input */
            }
            .message {
              padding: 8px;
              margin-bottom: 8px;
              border-radius: 4px;
              white-space: pre-wrap; /* Respect newlines */
            }
            .user-message {
              background-color: var(--vscode-input-background);
            }
            .agent-message {
              background-color: var(--vscode-list-hoverBackground);
            }
            .message-sender {
              font-weight: bold;
              font-size: 0.9em;
              margin-bottom: 4px;
            }
            #input-area {
              display: flex;
              border-top: 1px solid var(--vscode-input-border, var(--vscode-sideBar-border));
              padding-top: 8px;
            }
            #prompt-input {
              flex-grow: 1;
              width: 100%;
              border: 1px solid var(--vscode-input-border);
              background-color: var(--vscode-input-background);
              color: var(--vscode-input-foreground);
              border-radius: 4px;
              padding: 6px;
              resize: none; /* Don't allow manual resize */
            }
            #send-button {
              margin-left: 8px;
              background-color: var(--vscode-button-background);
              color: var(--vscode-button-foreground);
              border: none;
              border-radius: 4px;
              padding: 6px 12px;
              cursor: pointer;
            }
            #send-button:hover {
              background-color: var(--vscode-button-hoverBackground);
            }
          </style>
      </head>
      <body>
          <div id="chat-container">
            <div id="messages">
              <!-- Messages will be added here by JS -->
            </div>
            <div id="input-area">
              <textarea id="prompt-input" rows="3" placeholder="Type your message..."></textarea>
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
             * Appends a message to the chat display.
             */
            function addMessage(sender, text, type) {
              const messageDiv = document.createElement('div');
              messageDiv.className = 'message ' + type;
              
              const senderDiv = document.createElement('div');
              senderDiv.className = 'message-sender';
              senderDiv.textContent = sender;
              
              const textDiv = document.createElement('div');
              textDiv.textContent = text; // Use textContent to prevent XSS
              
              messageDiv.appendChild(senderDiv);
              messageDiv.appendChild(textDiv);
              messagesDiv.appendChild(messageDiv);
              
              // Scroll to bottom
              messagesDiv.scrollTop = messagesDiv.scrollHeight;
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
              }
            }

            // --- Event Listeners ---

            // Send on button click
            sendButton.addEventListener('click', sendPrompt);

            // Send on 'Enter' (but not 'Shift+Enter')
            input.addEventListener('keydown', (e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault(); // Prevent new line
                sendPrompt();
              }
            });

            // Listen for messages from the extension
            window.addEventListener('message', event => {
              const message = event.data; // The JSON data sent from extension
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
      </html>`;
  }
}

// This method is called when your extension is deactivated
function deactivate() {}

module.exports = {
  activate,
  deactivate
}

