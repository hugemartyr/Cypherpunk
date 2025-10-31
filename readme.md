# Cerebrum

### _The Superintelligent Agentverse_

---

## Overview

**Cerebrum** is a unified **superintelligent orchestration layer** that bridges all AI models under one system.
It uses the **ASI:One** agent to analyze prompts, select the optimal model, and decide between two workflows:

- **Transient Execution** → One-time prompt runs that self-clean after execution.
- **Persistent Execution** → Keeps frequently used models running for faster responses.

Cerebrum integrates **Hugging Face models** with a **knowledge graph (MeTTa)** and dynamic orchestration to remove the need to manually switch between multiple AIs.

---

## Explaining the Idea with an Example

Suppose we don't have any agent capable of generating sound but we have a sound generation model on Hugging Face (HF).

Now suppose a user on Agentverse says: "Generate me sound of old class rocking music"
There is no model in Agentverse good enough to do this, but there is a model on HF to do exactly this. We bridge the gap between Agentverse and HF.

By querying the orchestrator agent, it sees there is a new type of task: sound generation. It keeps the task in mind and finds models from HF. It gets the model ID `facebook-medium` from HF. It will then tell the `HF_agent` to temporarily make a `facebook-medium` model, generate the inference, and return it to the original caller user.

Now, suppose many users are generating sounds from `facebook-medium`. It would be better if an agent exists in Agentverse that has the capability of generating sound. The orchestrator agent, on seeing high demand, will command the `HF_agent` to deploy an agent with sound generation capability.

Currently, we haven't done this for image/sound generation; we have made the workflow for textual messages.

Also, if you know a `model_id`, you can directly query the `HF_agent` to make your model rather than going through the orchestrator path (That is why we have 2 agents).

Agents are created on demand basis

### Now Users Can Query in Agentverse

> Hey i wanna _test_ MiniMaxAI/MiniMax-M2 hugging face model for my work, can you generate text about Ai Agents and blockchains from MiniMaxAI/MiniMax-M2 hugging face models

> Hey can you make me an agent with text generation capability from hugging face model id : MiniMaxAI/MiniMax-M2

> Hey can generate me text about Fetch Ai from _best_ hugging face models

## Execution Flow

You can run `./run_instructions` to install required libraries or follow the instructions below.

### Print Instructions

```
=========================================
Project Run Instructions
=========================================
You need 3 terminals to run this project:

---
Terminal 1: HF Manager Agent
---

1. Run the HF Manager Agent:
   python3 hf_agent.py
2. Copy the agent address that is printed (e.g., agent1q...).

---
Terminal 2: Orchestrator Agent
---
3. Open orchestrator_agent.py in your editor.
4. Paste the HF Manager's address into the 'hf_manager_address' variable.
5. Run the Orchestrator Agent:
   python3.12 orchestrator_agent.py

---
Terminal 3: Client Agent (Choose one path)
---
You can now test any of the following paths:

Path 1: Test Transient Model (Directly)
- Edit 'simple_agent_to_send_message_hf_transient.py' to set the HF agent address and prompt.
- Run: python3 simple_agent_to_send_message_hf_transient.py

Path 2: Test Persistent Model (Directly)
- Edit 'simple_agent_to_send_message_hf_persistent.py' to set the HF agent address and prompt.
- Run: python3 simple_agent_to_send_message_hf_persistent.py

Path 3: Test Orchestrator (Knowledge Graph)
- Edit 'simple_agent_to_send_message_orc_agent.py' to set the ORC agent address.
- Run: python3 simple_agent_to_send_message_orc_agent.py

Path 4: Test Orchestrator (Auto-Deployment)
- Set THRESHOLD_TO_DEPLOY_NEW_AGENT = 2 in the orchestrator_agent.py config.
- Run the command from Path 3 twice.
- This will trigger the deployment of a new agent.
```

## Behind the Scenes

- **Knowledge Graph (knowledge_graph.py)** – Stores model-task relationships and helps the orchestrator pick the right model.
- **Orchestrator Agent** – Acts as the ASI:One superintelligence, deciding execution type and managing frequency thresholds.
- **HF Agent** – Executes Hugging Face models as subprocesses in persistent or transient modes.
- **Message Agent** – Simulates the user sending prompts to the system.

---

## Output

The **HF Agent** executes the selected model, retrieves the output, and sends it back through the **orchestrator** to the **message sender**, completing the agentverse loop.

---

## Run Summary

| Step | File                    | Description                       |
| :--- | :---------------------- | :-------------------------------- |
| 1    | `hf_agent.py`           | Initializes model execution agent |
| 2    | `orchestrator_agent.py` | Runs ASI:One orchestrator         |
| -    | `knowledge_graph.py`    | Supports orchestration logic      |

---

**Cerebrum – Where Superintelligence Meets Execution.**
