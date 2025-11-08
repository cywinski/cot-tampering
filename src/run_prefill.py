# ABOUTME: Main script for prefill experiment
# ABOUTME: Loads prompts from dataset, prefills CoT reasoning, and completes with OpenRouter completions API

import asyncio
import random
import re
import sys
from pathlib import Path

import fire

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from sampling import (
    OpenRouterCompletionsClient,
    SampleItem,
    build_prefill_prompt,
    get_dataset,
    group_results_by_prompt,
    sample_batch_async,
    save_prompt_results,
)
from utils import create_formatter, load_config


def run_prefill(config_path: str = "experiments/configs/prefill_experiment.yaml"):
    """Run prefill experiment from config file.

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
            print(f"Randomly sampled {len(problems)} problems (seed={random_seed})")

    # Create prompt formatter
    prompt_config = config["prompt"]
    formatter = create_formatter(prompt_config)
    template_name = prompt_config.get("template", "raw")
    print(f"Using template: {template_name}")

    prompt_messages = [formatter.format(problem) for problem in problems]
    user_messages = [msg[0]["content"] if msg else "" for msg in prompt_messages]

    # Setup client
    model_config = config["model"]
    openrouter_config = config.get("openrouter", {})

    client = OpenRouterCompletionsClient(
        model=model_config["name"],
        temperature=model_config.get("temperature", 0.6),
        max_tokens=model_config.get("max_tokens", 16384),
        top_p=model_config.get("top_p", 0.95),
        providers=openrouter_config.get("providers"),
        timeout=model_config.get("timeout"),
    )

    print(f"\nInitialized OpenRouter client with model: {model_config['name']}")
    if openrouter_config.get("providers"):
        print(f"Providers: {openrouter_config['providers']}")

    # Get experiment config
    exp_config = config["experiment"]
    prefill = exp_config.get("prefill", "")
    if not prefill:
        raise ValueError("Must specify 'prefill' in experiment config")
    n_responses = exp_config.get("n_responses", 1)
    batch_size = exp_config.get("batch_size", 10)
    print(
        f"\nPrefill text: {prefill[:100]}..."
        if len(prefill) > 100
        else f"\nPrefill text: {prefill}"
    )
    print(f"Sampling {n_responses} responses per prompt")
    print(f"Batch size: {batch_size} (processing in parallel)")

    fmt_config = config["prompt_formatting"]
    prefix_user_tokens = fmt_config.get("prefix_user_tokens", "")
    postfix_user_tokens = fmt_config.get("postfix_user_tokens", "")
    prefix_assistant_tokens = fmt_config.get("prefix_assistant_tokens", "")
    thinking_tag = fmt_config.get("thinking_tag", "think")

    prefill_prompts = [
        build_prefill_prompt(
            prefix_user_tokens=prefix_user_tokens,
            user_message=user_msg,
            postfix_user_tokens=postfix_user_tokens,
            prefix_assistant_tokens=prefix_assistant_tokens,
            thinking_tag=thinking_tag,
            prefill=prefill,
        )
        for user_msg in user_messages
    ]

    # Create flat list of sample items (repeat each prompt n_responses times)
    sample_items: list[SampleItem] = []
    for prompt_idx, prompt_str in enumerate(prefill_prompts):
        for response_idx in range(n_responses):
            sample_items.append(
                SampleItem(
                    prompt_str=prompt_str,
                    prompt_index=prompt_idx,
                    response_index=response_idx,
                    metadata={},
                )
            )

    print(f"Created {len(sample_items)} sample items")

    # Setup output directory
    output_dir = Path(exp_config["output_dir"])
    print(f"\nSaving results to: {output_dir}/")

    # Check for existing files and filter out already-processed prompts
    existing_indices = set()
    if output_dir.exists():
        for file_path in output_dir.glob("prompt_*.json"):
            match = re.search(r"prompt_(\d+)\.json", file_path.name)
            if match:
                existing_indices.add(int(match.group(1)))

    if existing_indices:
        print(f"Found {len(existing_indices)} existing prompt files")
        sample_items = [
            item for item in sample_items if item.prompt_index not in existing_indices
        ]
        print(f"Will generate {len(sample_items)} new samples")

    if not sample_items:
        print("All prompts already exist! Nothing to generate.")
        return

    print(f"\nStarting parallel batch sampling (batch_size={batch_size})...")
    results = asyncio.run(sample_batch_async(client, sample_items, batch_size))
    grouped_results = group_results_by_prompt(results)
    problems_dict = {i: problem for i, problem in enumerate(problems)}
    prompts_dict = {i: msg for i, msg in enumerate(user_messages)}
    save_prompt_results(grouped_results, output_dir, problems_dict, prompts_dict)
    total_successful = sum(1 for r in results if r["success"])
    total_expected = len(sample_items)
    total_prompts_processed = len(grouped_results)
    total_prompts_target = len(problems)

    print(f"\n{'=' * 80}")
    print("PREFILL EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"Target prompts: {total_prompts_target}")
    print(f"  - Already existed: {len(existing_indices)}")
    print(f"  - Newly processed: {total_prompts_processed}")
    print(f"Successfully completed: {total_successful}/{total_expected}")
    print(f"Success rate: {total_successful / total_expected * 100:.1f}%")
    print(
        f"Total prompts now available: {len(existing_indices) + total_prompts_processed}/{total_prompts_target}"
    )
    print(f"Results saved to: {output_dir}/")

    if results and results[0]["success"]:
        print(f"\n{'=' * 80}")
        print("SAMPLE RESULT (First sample)")
        print(f"{'=' * 80}")
        first_result = results[0]
        prompt_idx = first_result["prompt_index"]
        if prompt_idx < len(problems):
            print("\nProblem:")
            print(
                problems[prompt_idx].get("problem")
                or problems[prompt_idx].get("question", "")
            )
        print("\nPrefill:")
        print(prefill[:200] + "...")
        print("\nGenerated completion:")
        completion = first_result["completion"]
        text = completion.get("text", "")[:500]
        if completion.get("reasoning"):
            text = completion["reasoning"][:500]
        print(text + "...")
        print(f"{'=' * 80}")


if __name__ == "__main__":
    fire.Fire(run_prefill)
