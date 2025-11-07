# ABOUTME: Main script for prefill experiment
# ABOUTME: Loads prompts from dataset, prefills CoT reasoning, and completes with OpenRouter completions API

import asyncio
import json
import sys
from pathlib import Path

import fire
import yaml

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dataset_loaders import get_dataset
from openrouter_completions_client import OpenRouterCompletionsClient
from prompt_formatter import PromptFormatter
from thinking_trace_utils import build_prefill_prompt


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


async def process_single_prompt_async(
    client: OpenRouterCompletionsClient,
    prompt_idx: int,
    prompt: list[dict[str, str]],
    problem: dict,
    output_dir: Path,
    config: dict,
    total_prompts: int,
) -> dict:
    """Process a single prompt with prefill and generate completion (async).

    Args:
        client: OpenRouter completions client
        prompt_idx: Index of the prompt (for filename)
        prompt: Formatted prompt messages
        problem: Original problem dict
        output_dir: Directory to save results
        config: Full experiment config
        total_prompts: Total number of prompts (for progress display)

    Returns:
        Dict with result data
    """
    output_file = output_dir / f"prompt_{prompt_idx:04d}.json"

    # Check if file already exists
    if output_file.exists():
        print(f"Prompt {prompt_idx:04d}/{total_prompts:04d}: already exists, skipping")
        with open(output_file, "r") as f:
            existing_data = json.load(f)
        return {"skipped": True, **existing_data}

    # Extract configuration
    exp_config = config["experiment"]
    fmt_config = config["prompt_formatting"]

    # Get user message from prompt
    user_message = prompt[0]["content"] if prompt else ""

    # Get prefill text
    prefill = exp_config.get("prefill", "")
    if not prefill:
        return {
            "prompt_index": prompt_idx,
            "success": False,
            "error": "No prefill text specified in config",
        }

    # Build prefill prompt
    completion_prompt = build_prefill_prompt(
        prefix_tokens=fmt_config.get("prefix_tokens", ""),
        user_message=user_message,
        postfix_tokens=fmt_config.get("postfix_tokens", ""),
        assistant_prefix=fmt_config.get("assistant_prefix", ""),
        thinking_tag=fmt_config.get("thinking_tag", "think"),
        prefill=prefill,
    )

    # Generate completion
    print(
        f"Prompt {prompt_idx:04d}/{total_prompts:04d}: Generating completion with prefill..."
    )
    # Use async method directly since we're already in an async context
    completion_result = await client._complete_async(completion_prompt)

    # Analyze the completion
    generated_text = completion_result.get("text", "")
    reasoning_content = completion_result.get("reasoning", "")
    completion_tokens = completion_result.get("usage", {}).get("completion_tokens", 0)

    # For reasoning models, reasoning might be in separate field
    # Combine reasoning + text to get full generated content
    if reasoning_content:
        full_generated = reasoning_content + generated_text
    else:
        full_generated = generated_text

    # Reconstruct full response (prompt + completion)
    full_reconstructed_response = completion_prompt + full_generated

    # Build result
    result_data = {
        "prompt_index": prompt_idx,
        "problem": problem,
        "prompt": prompt,
        "prefill": prefill,
        "prompt_sent_to_api": completion_prompt,
        "completion": completion_result,
        "full_reconstructed_response": full_reconstructed_response,
        "success": completion_result.get("success", False),
    }

    # Save immediately to individual file
    with open(output_file, "w") as f:
        json.dump(result_data, f, indent=2)

    if completion_result.get("success", False):
        print(
            f"Prompt {prompt_idx:04d}/{total_prompts:04d}: Success! Generated {completion_tokens} tokens"
        )
    else:
        error_msg = completion_result.get("error", "Unknown error")
        print(
            f"Prompt {prompt_idx:04d}/{total_prompts:04d}: Failed - {error_msg}"
        )

    return result_data


async def run_prefill_async(
    client: OpenRouterCompletionsClient,
    prompts: list[list[dict[str, str]]],
    problems: list[dict],
    output_dir: Path,
    config: dict,
) -> list[dict]:
    """Run prefill experiment asynchronously.

    Args:
        client: OpenRouter completions client
        prompts: List of formatted prompts
        problems: List of problem dicts
        output_dir: Directory to save results
        config: Full experiment config

    Returns:
        List of result dicts
    """
    print("\nStarting prefill experiment...")

    # Process prompts sequentially (can be parallelized if needed)
    results = []
    for idx, (prompt, problem) in enumerate(zip(prompts, problems)):
        result = await process_single_prompt_async(
            client, idx, prompt, problem, output_dir, config, len(prompts)
        )
        results.append(result)

    return results


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

    # Create prompt formatter
    prompt_config = config["prompt"]
    formatter = create_formatter(prompt_config)
    template_name = prompt_config.get("template", "raw")
    print(f"Using template: {template_name}")

    # Format prompts
    prompts = [formatter.format(problem) for problem in problems]

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

    # Get prefill text
    exp_config = config["experiment"]
    prefill = exp_config.get("prefill", "")
    if not prefill:
        raise ValueError("Must specify 'prefill' in experiment config")
    print(f"\nPrefill text: {prefill[:100]}..." if len(prefill) > 100 else f"\nPrefill text: {prefill}")

    # Setup output directory
    output_dir = Path(exp_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving results to: {output_dir}/")

    # Run async experiment
    results = asyncio.run(
        run_prefill_async(client, prompts, problems, output_dir, config)
    )

    # Calculate overall success rate
    total_successful = sum(1 for r in results if r.get("success", False))
    total_skipped = sum(1 for r in results if r.get("skipped", False))
    total_processed = len(results) - total_skipped

    print(f"\n{'=' * 80}")
    print("PREFILL EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"Total prompts processed: {len(prompts)}")
    print(f"  - Newly processed: {total_processed}")
    print(f"  - Skipped (already existed): {total_skipped}")
    print(f"Successfully completed: {total_successful}/{total_processed}")
    if total_processed > 0:
        print(f"Success rate: {total_successful / total_processed * 100:.1f}%")
    print(f"Results saved to: {output_dir}/")

    # Print sample result
    if results and results[0].get("success") and not results[0].get("skipped"):
        print(f"\n{'=' * 80}")
        print("SAMPLE RESULT (Prompt 0)")
        print(f"{'=' * 80}")
        print("\nProblem:")
        print(problems[0].get("problem") or problems[0].get("question", ""))
        print("\nPrefill:")
        print(exp_config.get("prefill", "")[:200] + "...")
        print("\nGenerated completion:")
        completion = results[0].get("completion", {})
        generated_text = completion.get("text", "")[:500]
        print(generated_text + "...")
        print(f"{'=' * 80}")


if __name__ == "__main__":
    fire.Fire(run_prefill)
