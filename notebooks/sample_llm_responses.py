# ABOUTME: Example script demonstrating how to sample LLM responses using Nebius client
# ABOUTME: Shows dataset loading, prompt formatting, and parallel sampling with configuration

# %%
# Parameters
dataset_name = "math500"
n_problems = 10
n_responses_per_problem = 3
model_name = "meta-llama/Meta-Llama-3.1-70B-Instruct"
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
from prompt_formatter import MathFormatter, SimpleQAFormatter

# %%
# List available datasets
print("Available datasets:", list_available_datasets())

# %%
# Load dataset
print(f"\nLoading {dataset_name}...")
problems = get_dataset(dataset_name)
print(f"Loaded {len(problems)} problems")

# Take subset
problems = problems[:n_problems]
print(f"Using first {n_problems} problems")

# %%
# Create prompt formatter
formatter = MathFormatter(
    system_prompt="You are a math expert. Solve the following problem step by step.",
    include_cot_prompt=True,
)

# Format prompts
prompts = [formatter.format(problem) for problem in problems]
print(f"\nExample prompt for first problem:")
print(json.dumps(prompts[0], indent=2))

# %%
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
# Sample responses
print(f"\nSampling responses for {len(prompts)} problems...")
print("This will run in parallel...")

results = client.sample_batch_sync(prompts)

# %%
# Check results
successful_samples = sum(
    1 for problem_results in results for r in problem_results if r["success"]
)
total_expected = len(prompts) * n_responses_per_problem
print(f"\nSuccessfully sampled: {successful_samples}/{total_expected} responses")

# %%
# Display sample result
if results and results[0]:
    print("\nExample response for first problem:")
    print("-" * 80)
    print(f"Problem: {problems[0].get('problem', problems[0].get('question', ''))}")
    print("-" * 80)
    print(f"Response 1: {results[0][0]['content'][:500]}...")
    print("-" * 80)

# %%
# Save results
output_dir = Path("experiments/results")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / f"{dataset_name}_responses.jsonl"
with open(output_file, "w") as f:
    for problem, responses in zip(problems, results):
        f.write(
            json.dumps(
                {
                    "problem": problem,
                    "responses": responses,
                }
            )
            + "\n"
        )

print(f"\nResults saved to {output_file}")
