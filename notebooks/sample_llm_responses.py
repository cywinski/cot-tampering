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
# Imports
import json
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from dataset_loaders import get_dataset, list_available_datasets
from nebius_client import NebiusClient, SamplingConfig
from prompt_formatter import MathFormatter, RawFormatter, SimpleQAFormatter

# Create prompt formatter
formatter = RawFormatter(field_name="problem")

# Initialize client
config = SamplingConfig(
    model=model_name,
    temperature=temperature,
    n_responses=n_responses_per_problem,
    max_retries=max_retries,
)

client = NebiusClient(config)
print(f"\nClient initialized with model: {config.model}")
print(f"Will sample {config.n_responses} responses per problem")

# %%
# Load dataset
dataset = get_dataset(dataset_name)
print(f"Loaded {len(dataset)} problems from {dataset_name}")

# %%
# Example: Sample responses for a single prompt
print("\n=== SINGLE PROMPT EXAMPLE ===")
single_problem = dataset[0]
single_prompt = formatter.format(single_problem)

print(f"Problem: {single_problem['problem'][:100]}...")
print(f"\nSampling {n_responses_per_problem} responses...")

# Use sample_prompt_sync for a single prompt
single_responses = client.sample_prompt_sync(single_prompt)

print(f"\nGot {len(single_responses)} responses")
for i, resp in enumerate(single_responses):
    if resp["success"]:
        print(f"\nResponse {i + 1} (first 200 chars):")
        print(resp["content"][:200] + "...")
    else:
        print(f"\nResponse {i + 1} failed: {resp.get('error', 'Unknown error')}")

# %%
# Example: Sample responses for multiple prompts (batch)
print("\n=== BATCH SAMPLING EXAMPLE ===")
prompts = [formatter.format(dataset[i]) for i in range(n_problems)]
print(f"\nSampling responses for {n_problems} problems...")
print("This will run in parallel...")

results = client.sample_batch_sync(prompts)

# %%
# Check results
successful_samples = sum(
    1 for problem_results in results for r in problem_results if r["success"]
)
total_expected = len(prompts) * n_responses_per_problem
print(f"\nSuccessfully sampled: {successful_samples}/{total_expected} responses")
print(f"Success rate: {successful_samples / total_expected * 100:.1f}%")

# %%
# Display sample result
if results and results[0]:
    print("\n" + "=" * 80)
    print("SAMPLE RESULT")
    print("=" * 80)
    print(f"\nProblem: {dataset[0].get('problem', '')[:200]}...")
    if results[0][0]["success"]:
        print(f"\nResponse 1: {results[0][0]['content'][:500]}...")
    print("=" * 80)

# %%
# Save results
output_dir = Path("experiments/results")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / f"{dataset_name}_responses.jsonl"
with open(output_file, "w") as f:
    for problem_idx, responses in enumerate(results):
        f.write(
            json.dumps(
                {
                    "problem_index": problem_idx,
                    "problem": dataset[problem_idx],
                    "responses": responses,
                }
            )
            + "\n"
        )

print(f"\nResults saved to {output_file}")
