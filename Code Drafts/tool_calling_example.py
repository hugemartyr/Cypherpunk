import os, json, requests
import dotenv
dotenv.load_dotenv()

API_KEY = os.getenv("ASI_ONE_API_KEY")
BASE_URL = "https://api.asi1.ai/v1/chat/completions"
MODEL = "asi1-mini"

def get_weather(location: str) -> float:
    """Return current temperature in °C for 'City, Country'."""
    # Geocode the city name to coordinates
    geo = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": location, "format": "json", "limit": 1},
        timeout=10,
        headers={"User-Agent": "asi-demo"}
    ).json()
    if not geo:
        raise ValueError(f"Could not geocode {location!r}")

    lat, lon = geo[0]["lat"], geo[0]["lon"]

    # Get weather data
    wx_resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "current_weather": "true",
            "temperature_unit": "celsius"
        },
        timeout=10,
        headers={"User-Agent": "asi-demo"}
    )
    if wx_resp.status_code != 200:
        raise RuntimeError(f"Open-Meteo {wx_resp.status_code}: {wx_resp.text[:120]}")

    return wx_resp.json()["current_weather"]["temperature"]

weather_tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current temperature (°C) for a given city name.",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
            "additionalProperties": False
        },
        "strict": True
    }
}


messages = [
    {"role": "user", "content": "Get the current temperature in Paris, France using the get_weather tool."}
]

resp1 = requests.post(
    BASE_URL,
    headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
    json={"model": MODEL, "messages": messages, "tools": [weather_tool]},
).json()

choice = resp1["choices"][0]["message"]
if "tool_calls" not in choice:
    print("Model replied normally:", choice["content"])
    exit()

tool_call = choice["tool_calls"][0]
print("Tool-call from model:", json.dumps(tool_call, indent=2))


# Extract arguments
arg_str = tool_call.get("arguments") or tool_call.get("function", {}).get("arguments")
args = json.loads(arg_str)
print("Parsed arguments:", args)

# Execute the tool
temp_c = get_weather(**args)
print(f"Backend weather result: {temp_c:.1f} °C")

# Send the result back to ASI:One
assistant_msg = {
    "role": "assistant",
    "content": "",
    "tool_calls": [tool_call]
}

tool_result_msg = {
    "role": "tool",
    "tool_call_id": tool_call["id"],
    "content": json.dumps({"temperature_celsius": temp_c})
}

messages += [assistant_msg, tool_result_msg]

resp2 = requests.post(
    BASE_URL,
    headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
    json={
        "model": MODEL,
        "messages": messages,
        "tools": [weather_tool]  # repeat schema for safety
    },
).json()

if "choices" not in resp2:
    print("Error response:", json.dumps(resp2, indent=2))
    exit()

final_answer = resp2["choices"][0]["message"]["content"]
print("Assistant's final reply:")
print(final_answer)