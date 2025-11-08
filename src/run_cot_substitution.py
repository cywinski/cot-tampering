# ABOUTME: Main script for CoT substitution experiment
# ABOUTME: Loads prompts from dataset, extracts CoT from existing generated responses,
# ABOUTME: and uses them as prefill for inference on the same prompts

import asyncio
import json
import random
import re
import sys
from pathlib import Path

import fire

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from sampling import (
    OpenRouterCompletionsClient,
    get_dataset,
    build_completion_prompt,
    build_prefill_prompt,
    extract_thinking_trace,
)
from utils import create_formatter, load_config


def load_source_response(source_dir: Path, prompt_idx: int) -> dict | None:
    """Load source response file for a given prompt index.

    Args:
        source_dir: Directory containing source response files
        prompt_idx: Index of the prompt

    Returns:
        Response data dict, or None if file not found
    """
    source_file = source_dir / f"prompt_{prompt_idx:04d}.json"
    if not source_file.exists():
        return None

    with open(source_file, "r") as f:
        return json.load(f)


def extract_cot_from_source_response(
    source_data: dict, thinking_tag: str = "think"
) -> str | None:
    """Extract CoT (thinking trace) from source response data.

    Args:
        source_data: Source response data dict
        thinking_tag: Tag name used to wrap thinking (e.g., "think", "redacted_reasoning")

    Returns:
        Thinking trace content without tags, or None if not found
    """
    # Try to extract from responses array (sampling format)
    responses = source_data.get("responses", [])
    if responses:
        for response in responses:
            if response.get("success", False):
                content = response.get("content", "")
                if content:
                    cot = extract_thinking_trace(content, thinking_tag)
                    if cot is not None:
                        return cot

    # Try to extract from full_reconstructed_response (prefill format)
    full_response = source_data.get("full_reconstructed_response", "")
    if full_response:
        cot = extract_thinking_trace(full_response, thinking_tag)
        if cot is not None:
            return cot

    # Try to extract from completion.reasoning (if present)
    completion = source_data.get("completion", {})
    reasoning = completion.get("reasoning", "")
    if reasoning:
        # Reasoning might already be extracted, or might have tags
        cot = extract_thinking_trace(reasoning, thinking_tag)
        if cot is not None:
            return cot
        # If no tags, assume it's already the thinking trace
        if reasoning.strip():
            return reasoning.strip()

    return None


async def process_single_prompt_async(
    client: OpenRouterCompletionsClient,
    prompt_idx: int,
    prompt: list[dict[str, str]],
    problem: dict,
    output_dir: Path,
    config: dict,
    total_prompts: int,
    source_dir: Path,
) -> dict:
    """Process a single prompt with CoT substitution and generate completion (async).

    Args:
        client: OpenRouter completions client
        prompt_idx: Index of the prompt (for filename)
        prompt: Formatted prompt messages
        problem: Original problem dict
        output_dir: Directory to save results
        config: Full experiment config
        total_prompts: Total number of prompts (for progress display)
        source_dir: Directory containing source response files

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

    exp_config = config["experiment"]
    fmt_config = config["prompt_formatting"]
    thinking_tag = fmt_config.get("thinking_tag", "think")
    prefix_user_tokens = fmt_config.get("prefix_user_tokens", "")
    postfix_user_tokens = fmt_config.get("postfix_user_tokens", "")
    prefix_assistant_tokens = fmt_config.get("prefix_assistant_tokens", "")
    close_thinking_tag = exp_config.get("close_thinking_tag", False)

    user_message = prompt[0]["content"] if prompt else ""

    source_data = load_source_response(source_dir, prompt_idx)
    if source_data is None:
        return {
            "prompt_index": prompt_idx,
            "success": False,
            "error": f"Source response file not found: prompt_{prompt_idx:04d}.json",
        }

    cot_prefill = extract_cot_from_source_response(source_data, thinking_tag)
    if cot_prefill is None:
        return {
            "prompt_index": prompt_idx,
            "success": False,
            "error": f"Could not extract CoT from source response (looking for <{thinking_tag}> tags)",
        }

    if close_thinking_tag:
        completion_prompt = build_completion_prompt(
            prefix_user_tokens=prefix_user_tokens,
            user_message=user_message,
            postfix_user_tokens=postfix_user_tokens,
            prefix_assistant_tokens=prefix_assistant_tokens,
            thinking_tag=thinking_tag,
            full_thinking=cot_prefill,
            close_thinking_tag=True,
        )
    else:
        completion_prompt = build_prefill_prompt(
            prefix_user_tokens=prefix_user_tokens,
            user_message=user_message,
            postfix_user_tokens=postfix_user_tokens,
            prefix_assistant_tokens=prefix_assistant_tokens,
            thinking_tag=thinking_tag,
            prefill=cot_prefill,
        )

    # Generate completion
    mode_str = "closed" if close_thinking_tag else "open"
    print(
        f"Prompt {prompt_idx:04d}/{total_prompts:04d}: Generating completion with substituted CoT ({mode_str} mode)..."
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

    # Extract only essential completion fields (avoid redundancy)
    completion_summary = {
        "text": generated_text,
        "reasoning": reasoning_content if reasoning_content else None,
        "finish_reason": completion_result.get("finish_reason"),
        "usage": completion_result.get("usage", {}),
        "model": completion_result.get("model"),
        "success": completion_result.get("success", False),
    }

    # Add error if present
    if "error" in completion_result:
        completion_summary["error"] = completion_result["error"]

    # Build result
    result_data = {
        "prompt_index": prompt_idx,
        "problem": problem,
        "prompt": prompt,
        "source_response_file": f"prompt_{prompt_idx:04d}.json",
        "substituted_cot": cot_prefill,
        "close_thinking_tag_mode": close_thinking_tag,
        "prompt_sent_to_api": completion_prompt,
        "completion": completion_summary,
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
        print(f"Prompt {prompt_idx:04d}/{total_prompts:04d}: Failed - {error_msg}")

    return result_data


async def run_cot_substitution_async_with_indices(
    client: OpenRouterCompletionsClient,
    prompts: list[list[dict[str, str]]],
    problems: list[dict],
    original_indices: list[int],
    output_dir: Path,
    config: dict,
    total_prompts: int,
    source_dir: Path,
) -> list[dict]:
    """Run CoT substitution experiment asynchronously with parallel batch processing.

    Args:
        client: OpenRouter completions client
        prompts: List of formatted prompts to process
        problems: List of problem dicts to process
        original_indices: Original indices for file naming (may not be sequential)
        output_dir: Directory to save results
        config: Full experiment config
        total_prompts: Total number of prompts (including existing ones, for progress display)
        source_dir: Directory containing source response files

    Returns:
        List of result dicts
    """
    exp_config = config["experiment"]
    batch_size = exp_config.get("batch_size", 10)

    if batch_size > 1:
        print(f"\nStarting parallel batch processing (batch_size={batch_size})...")

        # Use semaphore to limit concurrent API calls
        semaphore = asyncio.Semaphore(batch_size)

        async def process_with_semaphore(original_idx, prompt, problem):
            async with semaphore:
                return await process_single_prompt_async(
                    client,
                    original_idx,
                    prompt,
                    problem,
                    output_dir,
                    config,
                    total_prompts,
                    source_dir,
                )

        # Create all tasks with original indices
        tasks = [
            process_with_semaphore(original_idx, prompt, problem)
            for original_idx, prompt, problem in zip(
                original_indices, prompts, problems
            )
        ]

        # Execute all tasks concurrently (limited by semaphore)
        results = await asyncio.gather(*tasks)
    else:
        print("\nStarting sequential processing...")
        results = []
        for original_idx, prompt, problem in zip(original_indices, prompts, problems):
            result = await process_single_prompt_async(
                client,
                original_idx,
                prompt,
                problem,
                output_dir,
                config,
                total_prompts,
                source_dir,
            )
            results.append(result)

    return results


def run_cot_substitution(
    config_path: str = "experiments/configs/cot_substitution.yaml",
):
    """Run CoT substitution experiment from config file.

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

    # Get source directory and batch size
    exp_config = config["experiment"]
    source_dir = Path(exp_config["source_responses_dir"])
    if not source_dir.exists():
        raise ValueError(f"Source responses directory does not exist: {source_dir}")
    batch_size = exp_config.get("batch_size", 10)
    print(f"\nSource responses directory: {source_dir}")
    print(f"Batch size: {batch_size} (processing prompts in parallel)")

    # Setup output directory
    output_dir = Path(exp_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving results to: {output_dir}/")

    # Check for existing prompt files and filter them out
    existing_indices = set()
    if output_dir.exists():
        for file_path in output_dir.glob("prompt_*.json"):
            # Extract index from filename (e.g., "prompt_0001.json" -> 1)
            match = re.search(r"prompt_(\d+)\.json", file_path.name)
            if match:
                existing_indices.add(int(match.group(1)))

    if existing_indices:
        print(f"Found {len(existing_indices)} existing prompt files")
        # Filter out problems/prompts that already exist
        filtered_data = [
            (idx, prompt, problem)
            for idx, (prompt, problem) in enumerate(zip(prompts, problems))
            if idx not in existing_indices
        ]
        if filtered_data:
            remaining_indices, remaining_prompts, remaining_problems = zip(
                *filtered_data
            )
            # Remap indices to be sequential starting from 0 for processing
            # But we need to preserve original indices for file naming
            prompts_to_process = list(remaining_prompts)
            problems_to_process = list(remaining_problems)
            original_indices = list(remaining_indices)
            print(
                f"Will generate {len(prompts_to_process)} new prompts (target: {len(prompts)} total)"
            )
        else:
            print("All prompts already exist! Nothing to generate.")
            prompts_to_process = []
            problems_to_process = []
            original_indices = []
    else:
        print(f"Will generate {len(prompts)} prompts")
        prompts_to_process = prompts
        problems_to_process = problems
        original_indices = list(range(len(prompts)))

    # Store existing_indices count for summary
    total_existing = len(existing_indices)

    # Run async experiment only if there are prompts to process
    if prompts_to_process:
        results = asyncio.run(
            run_cot_substitution_async_with_indices(
                client,
                prompts_to_process,
                problems_to_process,
                original_indices,
                output_dir,
                config,
                len(prompts),
                source_dir,
            )
        )
    else:
        results = []

    # Calculate overall success rate
    total_successful = sum(1 for r in results if r.get("success", False))
    total_skipped = sum(1 for r in results if r.get("skipped", False))
    total_processed = len(results) - total_skipped
    total_target = len(prompts)

    print(f"\n{'=' * 80}")
    print("COT SUBSTITUTION EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"Target prompts: {total_target}")
    print(f"  - Already existed: {total_existing}")
    print(f"  - Newly processed: {total_processed}")
    print(f"  - Skipped (errors): {total_skipped}")
    print(f"Successfully completed: {total_successful}/{total_processed}")
    if total_processed > 0:
        print(f"Success rate: {total_successful / total_processed * 100:.1f}%")
    print(
        f"Total prompts now available: {total_existing + total_successful}/{total_target}"
    )
    print(f"Results saved to: {output_dir}/")

    # Print sample result
    if results and results[0].get("success") and not results[0].get("skipped"):
        print(f"\n{'=' * 80}")
        print("SAMPLE RESULT (Prompt 0)")
        print(f"{'=' * 80}")
        print("\nProblem:")
        print(problems[0].get("problem") or problems[0].get("question", ""))
        print("\nSubstituted CoT (first 200 chars):")
        substituted_cot = results[0].get("substituted_cot", "")
        print(
            substituted_cot[:200] + "..."
            if len(substituted_cot) > 200
            else substituted_cot
        )
        print("\nGenerated completion:")
        completion = results[0].get("completion", {})
        generated_text = completion.get("text", "")[:500]
        print(generated_text + "...")
        print(f"{'=' * 80}")


if __name__ == "__main__":
    fire.Fire(run_cot_substitution)
