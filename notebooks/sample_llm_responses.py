# %%
import os

import requests

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

api_url = "https://openrouter.ai/api/v1/completions"
headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}

payload = {
    "model": "qwen/qwen3-235b-a22b-thinking-2507",
    "prompt": "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\nI'll",
    "temperature": 0.6,
    "top_p": 1,
    "max_tokens": 10000,
    "stream": False,
    "provider": {
        "only": ["DeepInfra"],
    },
}

response = requests.post(api_url, json=payload, headers=headers)
print(response.json())
