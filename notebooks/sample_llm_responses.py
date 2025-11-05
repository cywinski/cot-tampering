# ABOUTME: Example script demonstrating how to sample LLM responses using Nebius client
# ABOUTME: Shows dataset loading, prompt formatting, and parallel sampling with configuration

# %%
# Parameters
dataset_name = "math500"
n_problems = 10
n_responses_per_problem = 3
model_name = "deepseek-ai/DeepSeek-R1-0528"
temperature = 0.7
max_retries = 3

# %%
import os

import dotenv
from openai import OpenAI

dotenv.load_dotenv()

client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.getenv("NEBIUS_API_KEY"),
)

completion = client.chat.completions.create(
    model=model_name,
    messages=[
        {
            "role": "system",
            "content": "You are a chemistry expert. Add jokes about cats to your responses from time to time.",
        },
        {"role": "user", "content": "Hello!"},
    ],
    max_tokens=10,
    temperature=1,
    top_p=1,
    n=1,
)
# %%
print(completion.to_json())
