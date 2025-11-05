# %%
import requests
import json
import os

from dotenv import load_dotenv

load_dotenv()

# %%
# Sample from DeepSeek R1 with reasoning enabled
response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json",
    },
    data=json.dumps(
        {
            "model": "deepseek/deepseek-r1",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
        }
    ),
)

# %%
data = response.json()
print("Full response:")
print(json.dumps(data, indent=2))

# %%
# Extract reasoning and content (like our client does)
choice = data["choices"][0]
message = choice["message"]
content = message.get("content", "")
reasoning = message.get("reasoning", "")

print("\n" + "=" * 80)
print("REASONING TRACE:")
print("=" * 80)
print(reasoning[:500] if reasoning else "(no reasoning)")

print("\n" + "=" * 80)
print("FINAL ANSWER:")
print("=" * 80)
print(content)

print("\n" + "=" * 80)
print("COMBINED - without tags (wrap_thinking=False):")
print("=" * 80)
if reasoning:
    combined = f"{reasoning}\n\n{content}"
else:
    combined = content
print(combined[:500] + "..." if len(combined) > 500 else combined)

print("\n" + "=" * 80)
print("COMBINED - with tags (wrap_thinking=True):")
print("=" * 80)
if reasoning:
    # This is what the client does when wrap_thinking=True
    thinking_tag = "think"
    wrapped = f"<{thinking_tag}>\n{reasoning}\n</{thinking_tag}>\n\n{content}"
else:
    wrapped = content
print(wrapped[:500] + "..." if len(wrapped) > 500 else wrapped)
