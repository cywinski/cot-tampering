# ABOUTME: Production script for running LLM sampling experiments with YAML configuration
# ABOUTME: Handles dataset loading, prompt formatting, parallel sampling, and result saving

import asyncio
import json
import random
import sys
from pathlib import Path

import fire
import yaml

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dataset_loaders import get_dataset
from prompt_formatter import PromptFormatter
from sampling_config import SamplingConfig, create_client


def load_config(config_path: str) -> dict:
    """Load YAML configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Config dict
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_formatter(config: dict):
    """Create prompt formatter from config.

    Args:
        config: Config dict with prompt settings

    Returns:
        PromptFormatter instance
    """
    return PromptFormatter(
        template_name=config.get("template", "raw"),
        field_name=config.get("field_name", "problem"),
        custom_template=config.get("custom_template"),
    )


async def sample_and_save_prompt(
    client,
    prompt_idx: int,
    prompt: list[dict[str, str]],
    problem: dict,
    output_dir: Path,
    total_prompts: int,
) -> dict:
    """Sample responses for a single prompt and save immediately (async).

    Args:
        client: LLM client instance (NebiusClient or OpenRouterClient)
        prompt_idx: Index of the prompt (for filename)
        prompt: Formatted prompt messages
        problem: Original problem dict
        output_dir: Directory to save results
        total_prompts: Total number of prompts (for progress display)

    Returns:
        Dict with success statistics
    """
    output_file = output_dir / f"prompt_{prompt_idx:04d}.json"

    # Check if file already exists
    if output_file.exists():
        print(f"Prompt {prompt_idx:04d}/{total_prompts:04d}: already exists, skipping")

        # Load existing file to get statistics
        with open(output_file, "r") as f:
            existing_data = json.load(f)

        responses = existing_data.get("responses", [])
        successful = sum(1 for r in responses if r.get("success", False))
        total = len(responses)

        return {"successful": successful, "total": total, "responses": responses, "skipped": True}

    # Sample responses for this prompt (use async method if available)
    if hasattr(client, 'sample_prompt_async'):
        responses = await client.sample_prompt_async(prompt)
    else:
        # Fallback to sync method in thread pool
        loop = asyncio.get_event_loop()
        responses = await loop.run_in_executor(None, client.sample_prompt, prompt)

    # Save immediately to individual file
    result_data = {
        "prompt_index": prompt_idx,
        "problem": problem,
        "prompt": prompt,
        "responses": responses,
    }

    with open(output_file, "w") as f:
        json.dump(result_data, f, indent=2)

    # Calculate success stats
    successful = sum(1 for r in responses if r.get("success", False))
    total = len(responses)

    print(
        f"Prompt {prompt_idx:04d}/{total_prompts:04d}: {successful}/{total} responses saved"
    )

    return {"successful": successful, "total": total, "responses": responses, "skipped": False}


async def run_sampling_async(
    client,
    prompts: list[list[dict[str, str]]],
    problems: list[dict],
    output_dir: Path,
    config: SamplingConfig,
) -> list[dict]:
    """Run sampling asynchronously with optional batching.

    Args:
        client: LLM client instance
        prompts: List of formatted prompts
        problems: List of problem dicts
        output_dir: Directory to save results
        config: Sampling configuration

    Returns:
        List of result dicts
    """
    # Determine if we should use parallel batch processing
    use_batch_processing = config.batch_size > 1 and config.n_responses == 1

    if use_batch_processing:
        print(f"\nStarting parallel batch sampling (batch_size={config.batch_size})...")

        # Use semaphore to limit concurrent API calls
        semaphore = asyncio.Semaphore(config.batch_size)

        async def sample_with_semaphore(idx, prompt, problem):
            async with semaphore:
                return await sample_and_save_prompt(
                    client, idx, prompt, problem, output_dir, len(prompts)
                )

        # Create all tasks
        tasks = [
            sample_with_semaphore(idx, prompt, problem)
            for idx, (prompt, problem) in enumerate(zip(prompts, problems))
        ]

        # Execute all tasks concurrently (limited by semaphore)
        results = await asyncio.gather(*tasks)
    else:
        print("\nStarting sequential sampling...")
        results = []
        for idx, (prompt, problem) in enumerate(zip(prompts, problems)):
            result = await sample_and_save_prompt(
                client, idx, prompt, problem, output_dir, len(prompts)
            )
            results.append(result)

    return results


def run_sampling(config_path: str = "experiments/configs/sampling_config.yaml"):
    """Run LLM sampling experiment from config file.

    Args:
        config_path: Path to YAML configuration file
    """
    # Load config
    print(f"Loading config from {config_path}")
    config = load_config(config_path)

    # Load dataset
    dataset_config = config["dataset"]
    dataset_name = dataset_config["name"]
    dataset_params = dataset_config.get("params", {})
    dataset_limit = dataset_config.get("limit")

    print(f"\nLoading dataset: {dataset_name}")
    problems = get_dataset(dataset_name, **dataset_params)

    # Apply limit if specified
    if dataset_limit is not None:
        problems = problems[:dataset_limit]
        print(f"Loaded {len(problems)} problems (limited from full dataset)")
    else:
        print(f"Loaded {len(problems)} problems")

    # Apply random sampling if specified
    random_sample = dataset_config.get("random_sample")
    if random_sample is not None:
        if random_sample > len(problems):
            print(
                f"Warning: random_sample ({random_sample}) > dataset size ({len(problems)}), using all problems"
            )
        else:
            # Set seed for reproducibility if specified
            random_seed = dataset_config.get("random_seed", 42)
            random.seed(random_seed)
            problems = random.sample(problems, random_sample)
            print(
                f"Randomly sampled {len(problems)} problems (seed={random_seed})"
            )

    # Create prompt formatter
    prompt_config = config["prompt"]
    formatter = create_formatter(prompt_config)
    template_name = prompt_config.get("template", "raw")
    print(f"Using template: {template_name}")

    # Format prompts
    prompts = [formatter.format(problem) for problem in problems]

    # Create client
    model_config = config["model"]
    sampling_config_dict = config["sampling"]

    # Get provider (defaults to nebius for backward compatibility)
    provider = model_config.get("provider", "nebius")

    sampling_config = SamplingConfig(
        model=model_config["name"],
        temperature=model_config["temperature"],
        max_tokens=model_config["max_tokens"],
        top_p=model_config["top_p"],
        top_k=model_config.get("top_k"),
        n_responses=sampling_config_dict["n_responses"],
        max_retries=sampling_config_dict["max_retries"],
        timeout=sampling_config_dict["timeout"],
        batch_size=sampling_config_dict.get("batch_size", 1),
        # OpenRouter-specific options (ignored by Nebius)
        logprobs=model_config.get("logprobs", False),
        top_logprobs=model_config.get("top_logprobs", 5),
        reasoning=model_config.get("reasoning", False),
        site_url=model_config.get("site_url"),
        site_name=model_config.get("site_name"),
        # Thinking tags for wrapping reasoning traces
        wrap_thinking=model_config.get("wrap_thinking", False),
        thinking_tag=model_config.get("thinking_tag", "think"),
    )

    client = create_client(provider, sampling_config)
    print(f"\nInitialized {provider} client with model: {sampling_config.model}")
    print(f"Sampling {sampling_config.n_responses} responses per problem")
    print(f"Total API calls: {len(prompts) * sampling_config.n_responses}")

    # Setup output directory
    output_config = config["output"]
    output_dir = Path(output_config["save_dir"]) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving results to: {output_dir}/")

    # Run async sampling
    results = asyncio.run(
        run_sampling_async(client, prompts, problems, output_dir, sampling_config)
    )

    # Calculate overall success rate
    total_successful = sum(r["successful"] for r in results)
    total_expected = len(prompts) * sampling_config.n_responses
    total_skipped = sum(1 for r in results if r.get("skipped", False))
    total_sampled = len(prompts) - total_skipped

    print(f"\n{'=' * 80}")
    print("SAMPLING COMPLETE")
    print("=" * 80)
    print(f"Total prompts processed: {len(prompts)}")
    print(f"  - Newly sampled: {total_sampled}")
    print(f"  - Skipped (already existed): {total_skipped}")
    print(f"Successfully sampled: {total_successful}/{total_expected} responses")
    print(f"Success rate: {total_successful / total_expected * 100:.1f}%")
    print(f"Results saved to: {output_dir}/")

    # Print sample result
    if results and results[0]["responses"] and results[0]["responses"][0]["success"]:
        print(f"\n{'=' * 80}")
        print("SAMPLE RESULT (Prompt 0)")
        print(f"{'=' * 80}")
        print("\nProblem:")
        print(problems[0].get("problem") or problems[0].get("question", ""))
        print("\nResponse 1:")
        print(results[0]["responses"][0]["content"][:500] + "...")
        print(f"{'=' * 80}")


if __name__ == "__main__":
    fire.Fire(run_sampling)
