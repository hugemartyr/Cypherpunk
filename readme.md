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

## Execution Flow

### Step 1: Start the Hugging Face Agent

Run the **hf_agent.py** file first.  
This initializes the agent responsible for model execution and provides its agent address.

```bash
python3 hf_agent.py
```

Copy the displayed **agent address** — you’ll need this for the next step.

---

### Step 2: Start the Orchestrator Agent

Next, run the **orchestrator_agent.py** and pass it the address of the HF Agent.  
This is the central brain that decides which model to use and how (persistent or transient).

```bash
python3 orchestrator_agent.py
```

Copy the orchestrator’s **agent address** for use in the next step.

---

### Step 3: Run the Message Sender Agent

Open **simple_agent_to_send_message.py**, paste the **orchestrator address** in the designated place, and modify the prompt you want to send.

Then run:

```bash
python3 simple_agent_to_send_message.py
```

This script sends your request to the orchestrator, which interprets the prompt, retrieves required model parameters (model ID, task type, etc.) from the **knowledge graph**, and communicates with the **hf_agent** to execute the model.

---

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

| Step | File                              | Description                        |
| ---- | --------------------------------- | ---------------------------------- |
| 1    | `hf_agent.py`                     | Initializes model execution agent  |
| 2    | `orchestrator_agent.py`           | Runs ASI:One orchestrator          |
| 3    | `simple_agent_to_send_message.py` | Sends user prompts to orchestrator |
| -    | `knowledge_graph.py`              | Supports orchestration logic       |

---

**Cerebrum – Where Superintelligence Meets Execution.**
