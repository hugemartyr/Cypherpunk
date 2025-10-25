import os, uuid, json, requests, sys, time
import dotenv
dotenv.load_dotenv()

API_KEY = os.getenv("ASI_ONE_API_KEY") or "sk-REPLACE_ME"
ENDPOINT = "https://api.asi1.ai/v1/chat/completions"
MODEL = "asi1-fast-agentic"
TIMEOUT = 90

SESSION_MAP: dict[str, str] = {}

def get_session_id(conv_id: str) -> str:
    sid = SESSION_MAP.get(conv_id)
    if sid is None:
        sid = str(uuid.uuid4())
        SESSION_MAP[conv_id] = sid
    return sid

def _post(payload: dict, session_id: str, stream: bool = False):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "x-session-id": session_id,
        "Content-Type": "application/json",
    }
    return requests.post(ENDPOINT, headers=headers, json=payload, timeout=TIMEOUT, stream=stream)

def ask(conv_id: str, messages: list[dict], *, stream: bool = False) -> str:
    session_id = get_session_id(conv_id)
    print(f"[session] Using session-id: {session_id}")

    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": stream,
    }

    if not stream:
        resp = _post(payload, session_id)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    # Streaming
    with _post(payload, session_id, stream=True) as resp:
        resp.raise_for_status()
        full_text = ""
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                line = line[len("data: ") :]
            if line == "[DONE]":
                break
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue

            choices = chunk.get("choices")
            if not choices:
                continue

            delta = choices[0].get("delta", {})
            if "content" in delta:
                token = delta["content"]
                sys.stdout.write(token)
                sys.stdout.flush()
                full_text += token
        print()
        return full_text

def poll_for_async_reply(conv_id: str, history: list[dict], *, wait_sec: int = 10, max_attempts: int = 30):
    for attempt in range(max_attempts):
        time.sleep(wait_sec)
        print(f"\n🔄 polling (attempt {attempt + 1}) …", flush=True)
        reply = ask(conv_id, history, stream=False)
        if reply and "no new message" not in reply.lower():
            return reply
    return None

# Interactive CLI
if __name__ == "__main__":
    conv_id = str(uuid.uuid4())
    history: list[dict] = []

    print("Agentic LLM demo. Type Ctrl+C to exit.\n")
    try:
        while True:
            user_input = input("you > ").strip()
            if not user_input:
                continue
                
            history.append({"role": "user", "content": user_input})
            reply = ask(conv_id, history, stream=True)
            history.append({"role": "assistant", "content": reply})

            if "I've sent the message" in reply:
                follow = poll_for_async_reply(conv_id, history)
                if follow:
                    print(f"\n[Agentverse agent reply]\n{follow}")
                    history.append({"role": "assistant", "content": follow})
    except KeyboardInterrupt:
        print("\nBye!")