# ABOUTME: Main script for sentence removal experiment
# ABOUTME: Loads sampled responses, removes random sentences from thinking traces, and completes with OpenRouter API

import asyncio
import re
import sys
from pathlib import Path

import fire

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from sampling import (
    OpenRouterCompletionsClient,
    SampleItem,
    build_completion_prompt,
    extract_thinking_trace,
    group_results_by_prompt,
    remove_random_sentences,
    sample_batch_async,
    save_prompt_results,
)
from utils import load_config, load_response_file


def run_sentence_removal(
    config_path: str = "experiments/configs/sentence_removal_experiment.yaml",
):
    """Run sentence removal experiment.

    Args:
        config_path: Path to YAML configuration file
    """
    # Load config
    print(f"Loading config from {config_path}")
    config = load_config(config_path)

    # Setup client
    model_config = config["model"]
    openrouter_config = config["openrouter"]

    client = OpenRouterCompletionsClient(
        model=model_config["name"],
        temperature=model_config["temperature"],
        max_tokens=model_config["max_tokens"],
        top_p=model_config["top_p"],
        providers=openrouter_config.get("providers"),
        timeout=model_config.get("timeout"),
    )

    print(f"\nInitialized OpenRouter client with model: {model_config['name']}")
    if openrouter_config.get("providers"):
        print(f"Providers: {openrouter_config['providers']}")

    # Setup input/output
    exp_config = config["experiment"]
    input_dir = Path(exp_config["input_responses_dir"])
    output_dir = Path(exp_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nInput directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    # Find response files
    response_files = sorted(input_dir.glob("prompt_*.json"))
    print(f"\nFound {len(response_files)} response files")

    # Filter by indices
    start_idx = exp_config.get("start_index", 0)
    end_idx = exp_config.get("end_index")
    if end_idx is not None:
        response_files = [
            f
            for f in response_files
            if start_idx <= int(f.stem.split("_")[1]) < end_idx
        ]
    else:
        response_files = [
            f for f in response_files if int(f.stem.split("_")[1]) >= start_idx
        ]

    end_display = end_idx if end_idx is not None else "end"
    print(
        f"Processing {len(response_files)} files (indices {start_idx} to {end_display})"
    )
    print("Processing ALL responses per prompt")

    if exp_config.get("n_sentences_to_remove") is not None:
        print(f"Removing {exp_config['n_sentences_to_remove']} sentences per trace")
    elif exp_config.get("percentage_to_remove") is not None:
        print(
            f"Removing {exp_config['percentage_to_remove'] * 100:.1f}% of sentences per trace"
        )
    else:
        print(
            "ERROR: Must specify either n_sentences_to_remove or percentage_to_remove"
        )
        return

    fmt_config = config["prompt_formatting"]
    prefix_user_tokens = fmt_config["prefix_user_tokens"]
    postfix_user_tokens = fmt_config["postfix_user_tokens"]
    prefix_assistant_tokens = fmt_config["prefix_assistant_tokens"]
    thinking_tag = fmt_config["thinking_tag"]
    close_thinking_tag = exp_config.get("close_thinking_tag", True)

    # Prepare sample items
    sample_items: list[SampleItem] = []
    problems_dict: dict[int, dict] = {}
    prompts_dict: dict[int, str] = {}
    seed = exp_config.get("seed")

    for file_path in response_files:
        prompt_idx = int(file_path.stem.split("_")[1])

        # Load response file
        response_data = load_response_file(file_path)

        # Store problem and prompt
        if "problem" in response_data:
            problems_dict[prompt_idx] = response_data["problem"]
        if "prompt" in response_data:
            # Extract user message if it's a list format
            prompt = response_data["prompt"]
            if isinstance(prompt, list) and prompt:
                prompts_dict[prompt_idx] = prompt[0].get("content", "")
            elif isinstance(prompt, str):
                prompts_dict[prompt_idx] = prompt
        # Get all responses
        responses = response_data.get("responses", [])
        print(f"\nProcessing prompt {prompt_idx:04d}: {len(responses)} responses")

        # Get user message
        user_message = prompts_dict.get(prompt_idx, "")
        if not user_message:
            # Try to extract from prompt field
            prompt = response_data.get("prompt", [])
            if isinstance(prompt, list) and prompt:
                user_message = prompt[0].get("content", "")

        for response_idx, response in enumerate(responses):
            # Check if original response was successful
            if not response.get("success", False):
                continue

            content = response["content"]

            # Extract thinking trace
            thinking_trace = extract_thinking_trace(content, thinking_tag)
            if thinking_trace is None:
                continue

            # Remove sentences from thinking trace
            n_sentences = exp_config.get("n_sentences_to_remove")
            percentage = exp_config.get("percentage_to_remove")

            if seed is not None:
                deterministic_seed = seed + prompt_idx * 1000 + response_idx
            else:
                deterministic_seed = seed

            try:
                modified_thinking, removed_positions, n_sentences_removed = (
                    remove_random_sentences(
                        thinking_trace,
                        n_sentences=n_sentences,
                        percentage=percentage,
                        seed=deterministic_seed,
                    )
                )
            except ValueError as e:
                print(f"  Response {response_idx}: Error - {str(e)}")
                continue

            # Build completion prompt
            completion_prompt = build_completion_prompt(
                prefix_user_tokens=prefix_user_tokens,
                user_message=user_message,
                postfix_user_tokens=postfix_user_tokens,
                prefix_assistant_tokens=prefix_assistant_tokens,
                thinking_tag=thinking_tag,
                full_thinking=modified_thinking,
                close_thinking_tag=close_thinking_tag,
            )

            sentence_pattern = r"(?<=[.!?])\s+|\n+|$"
            parts = [
                p.strip()
                for p in re.split(sentence_pattern, thinking_trace)
                if p.strip()
            ]
            original_sentence_count = len(parts)
            percentage_removed = (
                (n_sentences_removed / original_sentence_count) * 100
                if original_sentence_count > 0
                else 0
            )

            # Create sample item
            sample_items.append(
                SampleItem(
                    prompt_str=completion_prompt,
                    prompt_index=prompt_idx,
                    response_index=response_idx,
                    metadata={
                        "n_sentences_removed": n_sentences_removed,
                        "percentage_removed": percentage_removed,
                        "original_sentence_count": original_sentence_count,
                        "close_thinking_tag_mode": close_thinking_tag,
                    },
                )
            )

    if not sample_items:
        print("No valid samples to process!")
        return

    print(f"\nCreated {len(sample_items)} sample items")
    batch_size = exp_config.get("batch_size", 10)
    print(f"Batch size: {batch_size} (processing in parallel)")

    print(f"\nStarting parallel batch sampling (batch_size={batch_size})...")
    results = asyncio.run(sample_batch_async(client, sample_items, batch_size))
    grouped_results = group_results_by_prompt(results)
    save_prompt_results(grouped_results, output_dir, problems_dict, prompts_dict)
    successful = sum(1 for r in results if r["success"])
    total = len(results)

    print(f"\n{'=' * 80}")
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"Total responses processed: {total}")
    print(f"Successful: {successful}/{total} ({successful / total * 100:.1f}%)")
    print(f"Failed: {total - successful}/{total}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    fire.Fire(run_sentence_removal)
