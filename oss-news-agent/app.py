import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from tools import FUNCTION_MAP, TOOLS
import time
import json

# Load environment variables from a .env file
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set")

AVAILABLE_MODELS = [
    "openai/gpt-oss-120b:fireworks-ai",
    "openai/gpt-oss-20b:fireworks-ai"
]

# Default model
DEFAULT_MODEL = "openai/gpt-oss-120b:fireworks-ai"

client = OpenAI(
    base_url="https://router.huggingface.co/v1", #base_url
    api_key=HF_TOKEN
)

def call_model(messages: List[Dict[str, str]], tools=TOOLS, model: str = DEFAULT_MODEL,
            temperature: float = 0.3):
    """Call the OpenAI-compatible model with messages and tools."""
    try:
        return client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=1024,
        )
    except Exception as e:
        print(f"Error calling model {e}")
        raise e
    
def run_agent(user_prompt: str, site_limit: Optional[str] = None, model: str = DEFAULT_MODEL,
              temperature: float = 0.3) -> str:
    """Run the agent with a user prompt and optional site limit."""
    system = {
        "role": "system",
        "content": ()
    }

    messages: List[Dict[str, str]] = [system, {"role": "user", "content": user_prompt}]
    if site_limit:
        messages.append({"role": "system", "content": f"Limit search to site: {site_limit} when appropriate."})

    for step in range(6):
        try:
            resp = call_model(messages, model=model, temperature=temperature)
            msg = resp.choices[0].message

            if getattr(msg, "tool_calls", None) and msg.tool_calls:
                assistant_message = {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": tool_call.type,
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments
                            }
                        } for tool_call in msg.tool_calls
                    ]
                }
                messages.append(assistant_message)

                for tool_call in msg.tool_calls:
                    name = tool_call.function.name
                    args = {}
                    try:
                        args = json.loads(tool_call.function.arguments or "{}")
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON for tool call {name}: {tool_call.function.arguments}")
                        args = {}
                    
                    fn = FUNCTION_MAP.get(name)
                    
                    if not fn:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": name,
                            "content": json.dumps({"ok": False, "error": "unknown_tool"})
                        })
                        continue


                
        except Exception as e:
            if step == 5:
                print(f"Error after 6 attempts: {e}")
                return f"Error: {e}"
            time.sleep(2 ** step)  # Exponential backoff
            continue


