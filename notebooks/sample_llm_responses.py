# %%
import os

import requests

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

api_url = "https://openrouter.ai/api/v1/completions"
headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}
# %%
reasoning = """
analysisHmm, I'
"""
# %%
payload = {
    "model": "openai/gpt-oss-120b",
    "prompt": "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\n\nReasoning: medium\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>\n<|start|>user<|message|>Please solve the following problem:\n When the analyst reads the data from a critical perspective to determine whether or not this fits with their expectations, this constitutes what is termed a:\n\nA. Face validity.\nB. Reliability.\nC. Control group.\nD. Focus group.\n\nChoose the correct answer and put only the letter (A, B, C, or D) between <answer></answer> tags.<|end|>\n<|start|>assistant<|channel|>analysis<|message|>I'",
    "temperature": 0.6,
    "top_p": 1,
    "max_tokens": 10000,
    "stream": False,
    "provider": {
        "only": ["DeepInfra"],
    },
}
# %%
response = requests.post(api_url, json=payload, headers=headers)
print(response.json())

# %%
