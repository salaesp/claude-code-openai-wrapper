"""Structured output via response_format json_schema."""
import json
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="YOUR_WRAPPER_API_KEY")

r = client.chat.completions.create(
    model="claude-sonnet-5",
    messages=[{"role": "user", "content": "Extract: John is 30 years old."}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "person",
            "schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name", "age"],
            },
        },
    },
)
print(json.loads(r.choices[0].message.content))  # {'name': 'John', 'age': 30}
