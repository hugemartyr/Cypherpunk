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
// 	console.log('Congratulations, your extension "fetchaidev" is now active!');

// 	// The command has been defined in the package.json file
// 	// Now provide the implementation of the command with  registerCommand
// 	// The commandId parameter must match the command field in package.json
// 	const disposable = vscode.commands.registerCommand('fetchaidev.helloWorld', function () {
// 		// The code you place here will be executed every time your command is executed

// 		// Display a message box to the user
// 		vscode.window.showInformationMessage('Hello World from FetchAiDev!');
// 	});

// 	context.subscriptions.push(disposable);
// }

// // This method is called when your extension is deactivated
// function deactivate() {}

// module.exports = {
// 	activate,
// 	deactivate
// }

import * as vscode from 'vscode';
import { TextEncoder } from 'util'; // Use Node.js util for TextEncoder

// --- Gemini API Configuration ---
const GEMINI_API_KEY = ""; // Leave this as-is
const GEMINI_API_URL = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key=${GEMINI_API_KEY}`;

// Define the JSON structure we expect from the AI
// interface AIFileResponse {
//   fileName: string;
//   content: string;
// }

// Define the JSON schema for the AI's response
const RESPONSE_SCHEMA = {
  type: "OBJECT",
  properties: {
    "fileName": {
      "type": "STRING",
      "description": "The relative path for the file to be created, e.g., 'src/app.js' or 'index.html'."
    },
    "content": {
      "type": "STRING",
      "description": "The code or text content to be written into the file. All code should be complete."
    }
  },
  required: ["fileName", "content"]
};

/**
 * This is the main entry point for your extension.
 * It's called when your extension is activated (e.g., when the command is run).
 */
export function activate(context) {

  // Register the command from package.json
  let disposable = vscode.commands.registerCommand('my-ai-agent.createFile', async () => {
    
    // 1. Get the user's open folder (workspace)
    const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
    if (!workspaceFolder) {
      vscode.window.showErrorMessage('You must have a folder open to use this extension.');
      return;
    }
    const workspaceUri = workspaceFolder.uri;

    // 2. Ask the user for instructions
    const userPrompt = await vscode.window.showInputBox({
      prompt: 'What file should I create and what code should it contain?',
      placeHolder: 'e.g., "Create a python file app.py that prints hello world"',
    });

    if (!userPrompt) {
      vscode.window.showInformationMessage('Command cancelled.');
      return;
    }

    // Show a loading message
    vscode.window.withProgress({
      location: vscode.ProgressLocation.Notification,
      title: 'AI Agent is working...',
      cancellable: false
    }, async (progress) => {

      try {
        // 3. Call the AI agent
        progress.report({ message: 'Contacting AI...' });
        const aiResponse = await callAIAgent(userPrompt);

        if (aiResponse && aiResponse.fileName && typeof aiResponse.content === 'string') {
          // 4. Create and write the file
          progress.report({ message: `Creating file: ${aiResponse.fileName}` });
          await createFileInWorkspace(workspaceUri, aiResponse.fileName, aiResponse.content);
          
          vscode.window.showInformationMessage(`Successfully created ${aiResponse.fileName}!`);
        } else {
          throw new Error('Received invalid response from AI.');
        }
      } catch (error) {
        console.error(error);
        vscode.window.showErrorMessage(`Error communicating with AI: ${error instanceof Error ? error.message : 'Unknown error'}`);
      }
    });
  });

  context.subscriptions.push(disposable);
}

/**
 * Calls the Gemini API to get file instructions.
 * @param userPrompt The user's text prompt.
 * @returns A structured AIFileResponse.
 */
async function callAIAgent(userPrompt){
  const systemInstruction = `You are an expert programmer and file system agent. 
  The user will give you a prompt to create a file. 
  You MUST respond ONLY with a single, valid JSON object that matches the following schema.
  Do not include markdown, backticks, or any text outside of the JSON object.
  The 'content' field must contain the full code for the file, correctly formatted.`;

  const payload = {
    contents: [{
      parts: [{ text: userPrompt }]
    }],
    systemInstruction: {
      parts: [{ text: systemInstruction }]
    },
    generationConfig: {
      responseMimeType: "application/json",
      responseSchema: RESPONSE_SCHEMA,
    }
  };

  // Call API with exponential backoff
  let response;
  let retries = 0;
  const maxRetries = 3;

  while (retries < maxRetries) {
    try {
      response = await fetch(GEMINI_API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (response.ok) {
        break; // Success
      }

      if (response.status === 429 || response.status >= 500) {
        // Throttling or server error, wait and retry
        const delay = Math.pow(2, retries) * 1000 + Math.random() * 1000;
        await new Promise(resolve => setTimeout(resolve, delay));
        retries++;
      } else {
        // Other client error
        throw new Error(`API request failed with status ${response.status}: ${await response.text()}`);
      }

    } catch (error) {
      if (retries + 1 >= maxRetries) {
        throw error; // Max retries reached
      }
      const delay = Math.pow(2, retries) * 1000 + Math.random() * 1000;
      await new Promise(resolve => setTimeout(resolve, delay));
      retries++;
    }
  }

  // @ts-ignore
  if (!response.ok) {
    throw new Error(`API request failed after retries with status ${response.status}`);
  }

  const result = await response.json();
  const text = result.candidates?.[0]?.content?.parts?.[0]?.text;

  if (!text) {
    throw new Error('No text response from AI.');
  }

  try {
    // The AI's response text *is* the JSON object
    return JSON.parse(text);
  } catch (e) {
    console.error("Failed to parse AI JSON response:", text);
    throw new Error('AI returned malformed JSON.');
  }
}

/**
 * Creates a file in the workspace and opens it.
 * @param workspaceUri The URI of the root workspace folder.
 * @param fileName The relative path of the file to create.
 * @param content The string content to write to the file.
 */
async function createFileInWorkspace(workspaceUri, fileName, content) {
  try {
    // 1. Get the full URI for the new file
    const fileUri = vscode.Uri.joinPath(workspaceUri, fileName);

    // 2. Convert string content to Uint8Array (required by fs.writeFile)
    const encodedContent = new TextEncoder().encode(content);

    // 3. Write the file
    // This will create directories if they don't exist and overwrite the file if it does.
    await vscode.workspace.fs.writeFile(fileUri, encodedContent);

    // 4. Open the new file in the editor
    const document = await vscode.workspace.openTextDocument(fileUri);
    await vscode.window.showTextDocument(document);

  } catch (error) {
    console.error(error);
    vscode.window.showErrorMessage(`Failed to create file: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

// This function is called when your extension is deactivated
export function deactivate() {}
