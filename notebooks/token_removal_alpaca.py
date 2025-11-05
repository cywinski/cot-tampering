# %%
import json
import os
import re
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# %%
results_dir_base = "../experiments/results/deepseek-r1/alpaca/token_removal"
results_samples_base = "../experiments/results/deepseek-r1/alpaca/samples/alpaca"
percentages = ["0.0", "0.1", "0.3", "0.6", "0.9", "1.0"]


# %%
def extract_output(content: str) -> str:
    """Extract the output part (everything after </think> tag)."""
    pattern = r"</think>\s*(.*)"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(1).strip()
    # If no redacted_reasoning tag found, return the whole content as output
    return content.strip()


def extract_modified_from_response(content: str) -> str:
    """
    Extract the modified status between <modified> and </modified> tags from the output part only.
    Does not extract from the thinking trace.
    """
    output = extract_output(content)
    pattern = r"<modified>(.*?)</modified>"
    match = re.search(pattern, output, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


# %%
percentage_result_map = {}
for percentage in percentages:
    results_dir = f"{results_dir_base}/{percentage}"
    print(f"Processing {results_dir}")
    prompt_results = {}
    for file in os.listdir(results_dir):
        if file.endswith(".json"):
            with open(os.path.join(results_dir, file), "r") as f:
                data = json.load(f)
            # print(data)
            prompt_idx = data["prompt_index"]

            modifieds = []
            for response in data["responses"]:
                modified = extract_modified_from_response(
                    response["completion"]["text"]
                )
                modifieds.append(modified)

            print(modifieds)
            prompt_results[prompt_idx] = {
                "modifieds": modifieds,
            }
    percentage_result_map[percentage] = prompt_results
# %%
import matplotlib.pyplot as plt

# Prepare data for plotting
percentages_sorted = sorted(percentage_result_map.keys(), key=float)
percentages_sorted_scaled = [float(p) * 100 for p in percentages_sorted]
yes_ratios = []
modified_ratios = []

for percentage in percentages_sorted:
    prompt_results = percentage_result_map[percentage]

    total_assessments = 0
    yes_count = 0
    total_modifieds = 0
    modified_count = 0

    for prompt_idx, results in prompt_results.items():
        modifieds = results["modifieds"]
        total_modifieds += len(modifieds)
        modified_count += sum(1 for m in modifieds if m and m.lower() == "yes")

    modified_ratio = modified_count / total_modifieds if total_modifieds > 0 else 0

    modified_ratios.append(modified_ratio)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(
    percentages_sorted_scaled,
    modified_ratios,
    marker="s",
    linewidth=3,
    markersize=12,
    label="Tampering Detection Rate",
    color="#ff7f0e",
)

plt.xlabel("CoT Tokens Removed (%)", fontsize=18)
plt.title("Can model detect tokens removed from CoT?", fontsize=22)
plt.legend(fontsize=18, loc="lower right")
plt.grid(True, alpha=0.2)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.ylim([0, 1])
plt.tight_layout()
plt.show()

# %%
