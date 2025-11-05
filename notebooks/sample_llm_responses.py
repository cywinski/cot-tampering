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
    "model": "deepseek/deepseek-r1",
    "prompt": "<｜begin▁of▁sentence｜><｜User｜>What is 2+2?<｜Assistant｜><think>\nI'll",
    "temperature": 0.6,
    "top_p": 1,
    "max_tokens": 10000,
    "stream": False,
    "provider": {
        "only": ["DeepInfra"],
    },
}

response = requests.post(api_url, json=payload, headers=headers)
# %%
print(response.json())
