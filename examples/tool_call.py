"""Passthrough tool calling with the OpenAI SDK. Client executes the tool."""
import json
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="YOUR_WRAPPER_API_KEY")

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]

messages = [{"role": "user", "content": "What's the weather in Paris?"}]

# 1) model asks for a tool call
r = client.chat.completions.create(
    model="claude-haiku-4-5", messages=messages, tools=tools, tool_choice="required"
)
call = r.choices[0].message.tool_calls[0]
print("model wants:", call.function.name, call.function.arguments)

# 2) YOU execute it
args = json.loads(call.function.arguments)
result = f"18C, sunny in {args['city']}"

# 3) send the result back (stateless: resend full history)
messages.append(r.choices[0].message.model_dump())
messages.append({"role": "tool", "tool_call_id": call.id,
                 "name": call.function.name, "content": result})

final = client.chat.completions.create(
    model="claude-haiku-4-5", messages=messages, tools=tools
)
print("final:", final.choices[0].message.content)
