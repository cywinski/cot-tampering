# ABOUTME: Main script for token removal experiment
# ABOUTME: Loads sampled responses, removes random tokens from thinking traces, and completes with OpenRouter API

import json
import sys
from pathlib import Path

import fire
import yaml

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from openrouter_completions_client import OpenRouterCompletionsClient
from thinking_trace_utils import (
    build_completion_prompt,
    extract_thinking_trace,
    remove_random_tokens,
)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Config dict
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_response_file(response_path: Path) -> dict:
    """Load a response JSON file.

    Args:
        response_path: Path to response JSON file

    Returns:
        Response data dict
    """
    with open(response_path) as f:
        return json.load(f)


def process_single_response(
    response: dict,
    response_idx: int,
    user_message: str,
    client: OpenRouterCompletionsClient,
    config: dict,
    prompt_idx: int,
) -> dict:
    """Process a single response: remove tokens and generate completion.

    Args:
        response: Single response dict
        response_idx: Index of response
        user_message: Original user message/prompt
        client: OpenRouter completions client
        config: Full experiment config
        prompt_idx: Prompt index for logging

    Returns:
        Result dict with experiment data
    """
    # Extract configuration
    exp_config = config["experiment"]
    fmt_config = config["prompt_formatting"]

    # Check if original response was successful
    if not response.get("success", False):
        return {
            "response_index": response_idx,
            "success": False,
            "error": "Original response was not successful",
        }

    content = response["content"]

    # Extract thinking trace
    thinking_tag = fmt_config["thinking_tag"]
    thinking_trace = extract_thinking_trace(content, thinking_tag)

    if thinking_trace is None:
        return {
            "response_index": response_idx,
            "success": False,
            "error": f"No thinking trace found (looking for <{thinking_tag}> tags)",
        }

    # Remove tokens from thinking trace
    n_tokens = exp_config.get("n_tokens_to_remove")
    percentage = exp_config.get("percentage_to_remove")
    seed = exp_config.get("seed")

    # Use a deterministic seed based on prompt and response indices for reproducibility
    if seed is not None:
        deterministic_seed = seed + prompt_idx * 1000 + response_idx
    else:
        deterministic_seed = seed

    try:
        modified_thinking, removed_positions, n_tokens_removed = remove_random_tokens(
            thinking_trace,
            n_tokens=n_tokens,
            percentage=percentage,
            seed=deterministic_seed,
        )
    except ValueError as e:
        return {
            "response_index": response_idx,
            "success": False,
            "error": str(e),
        }

    # Check if we should close the thinking tag or leave it open
    close_thinking_tag = exp_config.get("close_thinking_tag", True)

    completion_prompt = build_completion_prompt(
        prefix_tokens=fmt_config["prefix_tokens"],
        user_message=user_message,
        assistant_prefix=fmt_config["assistant_prefix"],
        thinking_tag=thinking_tag,
        full_thinking=modified_thinking,
        close_thinking_tag=close_thinking_tag,
    )

    # Generate completion
    original_token_count = len(thinking_trace.split())
    percentage_removed = (n_tokens_removed / original_token_count) * 100 if original_token_count > 0 else 0

    print(
        f"  Response {response_idx}: Generating completion ({n_tokens_removed} tokens removed = {percentage_removed:.1f}% of {original_token_count} tokens)..."
    )
    completion_result = client.complete(completion_prompt)

    # Build result
    return {
        "response_index": response_idx,
        "success": completion_result.get("success", False),
        "modified_thinking_full": modified_thinking,
        "n_tokens_removed": n_tokens_removed,
        "percentage_removed": percentage_removed,
        "original_token_count": original_token_count,
        "modified_token_count": len(modified_thinking.split()),
        "prompt_sent_to_api": completion_prompt,
        "completion": completion_result,
    }


def run_token_removal(
    config_path: str = "experiments/configs/token_removal_experiment.yaml",
):
    """Run token removal experiment.

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
            f for f in response_files if start_idx <= int(f.stem.split("_")[1]) < end_idx
        ]
    else:
        response_files = [
            f for f in response_files if int(f.stem.split("_")[1]) >= start_idx
        ]

    print(f"Processing {len(response_files)} files (indices {start_idx} to {end_idx})")
    print("Processing ALL responses per prompt")

    if exp_config.get("n_tokens_to_remove") is not None:
        print(f"Removing {exp_config['n_tokens_to_remove']} tokens per trace")
    elif exp_config.get("percentage_to_remove") is not None:
        print(f"Removing {exp_config['percentage_to_remove']*100:.1f}% of tokens per trace")
    else:
        print("ERROR: Must specify either n_tokens_to_remove or percentage_to_remove")

    # Process each file
    all_results = []
    for file_path in response_files:
        prompt_idx = int(file_path.stem.split("_")[1])
        print(f"\nProcessing prompt {prompt_idx:04d}...")

        # Load response file
        response_data = load_response_file(file_path)

        # Get original prompt
        prompt = response_data["prompt"]
        user_message = prompt[0]["content"] if prompt else ""

        # Get all responses
        responses = response_data.get("responses", [])
        print(f"  Found {len(responses)} responses")

        # Process all responses for this prompt
        prompt_results = []
        for response_idx, response in enumerate(responses):
            try:
                result = process_single_response(
                    response,
                    response_idx,
                    user_message,
                    client,
                    config,
                    prompt_idx,
                )
                prompt_results.append(result)

                if result["success"]:
                    print(f"  Response {response_idx}: Success!")
                else:
                    print(
                        f"  Response {response_idx}: Failed - {result.get('error', 'Unknown error')}"
                    )

            except Exception as e:
                print(f"  Response {response_idx}: Error - {str(e)}")
                prompt_results.append(
                    {
                        "response_index": response_idx,
                        "success": False,
                        "error": str(e),
                    }
                )

        # Save all responses for this prompt in one file
        output_file = output_dir / f"prompt_{prompt_idx:04d}.json"
        output_data = {
            "prompt_index": prompt_idx,
            "responses": prompt_results,
        }
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        successful = sum(1 for r in prompt_results if r.get("success", False))
        print(f"  Saved {successful}/{len(prompt_results)} successful responses to {output_file.name}")

        all_results.extend(prompt_results)

    # Summary
    successful = sum(1 for r in all_results if r.get("success", False))
    total = len(all_results)

    print(f"\n{'=' * 80}")
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"Total responses processed: {total}")
    if total > 0:
        print(f"Successful: {successful}/{total} ({successful/total*100:.1f}%)")
        print(f"Failed: {total - successful}/{total}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    fire.Fire(run_token_removal)
