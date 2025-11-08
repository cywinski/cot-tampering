# ABOUTME: Main script for thinking trace insertion experiment
# ABOUTME: Loads sampled responses, inserts text in thinking traces, and completes with OpenRouter API

import json
import sys
from pathlib import Path

import fire

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from sampling import (
    OpenRouterCompletionsClient,
    build_completion_prompt,
    extract_thinking_trace,
    insert_text_in_thinking,
    _clean_completion_result,
)
from utils import load_config, load_response_file


def process_single_response(
    response: dict,
    response_idx: int,
    user_message: str,
    client: OpenRouterCompletionsClient,
    config: dict,
    prompt_idx: int,
) -> dict:
    """Process a single response: insert text and generate completion.

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
    exp_config = config["experiment"]
    fmt_config = config["prompt_formatting"]

    if not response.get("success", False):
        return {
            "response_index": response_idx,
            "success": False,
            "error": "Original response was not successful",
        }

    content = response["content"]
    thinking_tag = fmt_config["thinking_tag"]
    thinking_trace = extract_thinking_trace(content, thinking_tag)

    if thinking_trace is None:
        return {
            "response_index": response_idx,
            "success": False,
            "error": f"No thinking trace found (looking for <{thinking_tag}> tags)",
        }

    insertion_text = exp_config["insertion_text"]
    insertion_position = exp_config["insertion_position"]
    seed = exp_config.get("seed")

    if seed is not None and insertion_position == "random":
        deterministic_seed = seed + prompt_idx * 1000 + response_idx
    else:
        deterministic_seed = seed

    thinking_before, thinking_after, prompt_prefix, insert_pos = (
        insert_text_in_thinking(
            thinking_trace, insertion_text, insertion_position, deterministic_seed
        )
    )

    modified_thinking_full = thinking_before + insertion_text + thinking_after
    full_thinking = modified_thinking_full
    close_thinking_tag = exp_config.get("close_thinking_tag", True)

    prefix_user_tokens = fmt_config["prefix_user_tokens"]
    postfix_user_tokens = fmt_config["postfix_user_tokens"]
    prefix_assistant_tokens = fmt_config["prefix_assistant_tokens"]

    completion_prompt = build_completion_prompt(
        prefix_user_tokens=prefix_user_tokens,
        user_message=user_message,
        postfix_user_tokens=postfix_user_tokens,
        prefix_assistant_tokens=prefix_assistant_tokens,
        thinking_tag=thinking_tag,
        full_thinking=full_thinking,
        close_thinking_tag=close_thinking_tag,
    )

    # Generate completion
    print(
        f"  Response {response_idx}: Generating completion (insertion at pos {insert_pos}/{len(thinking_trace)})..."
    )
    completion_result = client.complete(completion_prompt)

    # Analyze the completion
    generated_text = completion_result.get("text", "")
    reasoning_content = completion_result.get("reasoning", "")
    completion_tokens = completion_result.get("usage", {}).get("completion_tokens", 0)

    # For open mode with reasoning models, reasoning might be in separate field
    # Combine reasoning + text to get full generated content
    if reasoning_content:
        full_generated = reasoning_content + f"</{thinking_tag}>\n" + generated_text
    else:
        full_generated = generated_text

    # Reconstruct full response (prompt + completion)
    full_reconstructed_response = completion_prompt + full_generated

    print(f"  Response {response_idx}: Generated {completion_tokens} tokens, ")

    # Build result
    return {
        "response_index": response_idx,
        "success": completion_result.get("success", False),
        "modified_thinking_full": modified_thinking_full,
        "insertion_position": insert_pos,
        "insertion_text": insertion_text,
        "close_thinking_tag_mode": close_thinking_tag,
        "completion_token_count": completion_tokens,
        "prompt_sent_to_api": completion_prompt,
        "completion": _clean_completion_result(completion_result),
        "full_reconstructed_response": full_reconstructed_response,
    }


def run_thinking_insertion(
    config_path: str = "experiments/configs/thinking_insertion_experiment.yaml",
):
    """Run thinking trace insertion experiment.

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

    print(f"Processing {len(response_files)} files (indices {start_idx} to {end_idx})")
    print("Processing ALL responses per prompt")

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
        print(
            f"  Saved {successful}/{len(prompt_results)} successful responses to {output_file.name}"
        )

        all_results.extend(prompt_results)

    # Summary
    successful = sum(1 for r in all_results if r.get("success", False))
    total = len(all_results)

    print(f"\n{'=' * 80}")
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"Total responses processed: {total}")
    print(f"Successful: {successful}/{total} ({successful / total * 100:.1f}%)")
    print(f"Failed: {total - successful}/{total}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    fire.Fire(run_thinking_insertion)
