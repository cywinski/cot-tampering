# ABOUTME: Main script for sentence removal experiment
# ABOUTME: Loads sampled responses, removes random sentences from thinking traces, and completes with OpenRouter API

import asyncio
import re
import sys
from pathlib import Path

import fire
from transformers import AutoTokenizer

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from sampling import (
    OpenRouterCompletionsClient,
    SampleItem,
    build_completion_prompt,
    cut_thinking_trace_at_percentage,
    extract_thinking_trace,
    group_results_by_prompt,
    remove_random_sentences,
    sample_batch_async,
    save_prompt_results,
)
from utils import create_formatter, load_config, load_response_file


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

    # Load tokenizer if specified
    tokenizer_name = config.get("experiment", {}).get("tokenizer_name")
    tokenizer = None
    if tokenizer_name:
        print(f"\nLoading tokenizer: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        print("Tokenizer loaded successfully")
    else:
        print("\nNo tokenizer specified, using whitespace tokenization")

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

    # Determine which operations to perform
    n_sentences_to_remove = exp_config.get("n_sentences_to_remove")
    percentage_to_remove = exp_config.get("percentage_to_remove")
    cut_at_percentage = exp_config.get("cut_at_percentage")
    cut_start_percentage = exp_config.get("cut_start_percentage")
    cut_end_percentage = exp_config.get("cut_end_percentage")

    # Validate operations
    has_cut = cut_at_percentage is not None or (
        cut_start_percentage is not None and cut_end_percentage is not None
    )
    has_removal = n_sentences_to_remove is not None or percentage_to_remove is not None

    if not has_cut and not has_removal:
        print(
            "ERROR: Must specify at least one of: cut_at_percentage, (cut_start_percentage + cut_end_percentage), n_sentences_to_remove, or percentage_to_remove"
        )
        return

    # Ensure cutting happens first: if removal is specified, cutting must also be specified
    if has_removal and not has_cut:
        print(
            "ERROR: When removing sentences, cutting must be specified first. Please specify cut_at_percentage or (cut_start_percentage + cut_end_percentage)"
        )
        return

    if n_sentences_to_remove is not None and percentage_to_remove is not None:
        print(
            "ERROR: Cannot specify both n_sentences_to_remove and percentage_to_remove"
        )
        return

    if cut_at_percentage is not None and (
        cut_start_percentage is not None or cut_end_percentage is not None
    ):
        print(
            "ERROR: Cannot specify both cut_at_percentage and (cut_start_percentage/cut_end_percentage)"
        )
        return

    if (cut_start_percentage is not None) != (cut_end_percentage is not None):
        print(
            "ERROR: Must specify both cut_start_percentage and cut_end_percentage together"
        )
        return

    # Print operation summary
    operations = []
    if has_cut:
        if cut_start_percentage is not None and cut_end_percentage is not None:
            operations.append(
                f"Cutting trace from {cut_start_percentage * 100:.1f}% to {cut_end_percentage * 100:.1f}% "
                f"(keeping {cut_end_percentage * 100 - cut_start_percentage * 100:.1f}% from middle)"
            )
        elif cut_at_percentage is not None:
            operations.append(
                f"Cutting trace at {cut_at_percentage * 100:.1f}% (keeping first {cut_at_percentage * 100:.1f}%)"
            )
    if has_removal:
        if n_sentences_to_remove is not None:
            operations.append(f"Removing {n_sentences_to_remove} random sentences")
        elif percentage_to_remove is not None:
            operations.append(
                f"Removing {percentage_to_remove * 100:.1f}% of random sentences"
            )

    if len(operations) > 1:
        print(f"Operations: {' -> '.join(operations)}")
    else:
        print(f"Operation: {operations[0]}")

    # Use completion_prompt_formatting if specified, otherwise fall back to prompt_formatting
    fmt_config = (
        config.get("completion_prompt_formatting") or config["prompt_formatting"]
    )
    prefix_user_tokens = fmt_config["prefix_user_tokens"]
    postfix_user_tokens = fmt_config["postfix_user_tokens"]
    prefix_assistant_tokens = fmt_config["prefix_assistant_tokens"]

    # Get thinking tokens from config
    open_thinking_token = fmt_config.get("open_thinking_token") or fmt_config.get(
        "thinking_token"
    )
    close_thinking_token = fmt_config.get("close_thinking_token")

    # Fallback: construct from thinking_tag if tokens not provided (backward compatibility)
    if open_thinking_token is None:
        thinking_tag = fmt_config.get("thinking_tag") or config.get("model", {}).get(
            "thinking_tag", "think"
        )
        open_thinking_token = f"<{thinking_tag}>"
        if close_thinking_token is None:
            close_thinking_token = f"</{thinking_tag}>"
    elif close_thinking_token is None:
        # Derive close token from open token
        if open_thinking_token.startswith("<") and open_thinking_token.endswith(">"):
            tag_name = open_thinking_token[1:-1]
            close_thinking_token = f"</{tag_name}>"
        else:
            raise ValueError(
                f"Cannot derive close_thinking_token from open_thinking_token: {open_thinking_token}"
            )

    # Derive thinking_tag for extraction purposes (strip angle brackets)
    if open_thinking_token.startswith("<") and open_thinking_token.endswith(">"):
        thinking_tag = open_thinking_token[1:-1]
    else:
        thinking_tag = open_thinking_token

    close_thinking_tag = exp_config.get("close_thinking_tag", True)

    # Print which formatting is being used
    if config.get("completion_prompt_formatting"):
        print("\nUsing completion_prompt_formatting for output prompts")
    else:
        print("\nUsing prompt_formatting for output prompts")

    # Setup prompt formatter if prompt template is specified
    prompt_formatter = None
    if "prompt" in config:
        prompt_config = config["prompt"]
        prompt_formatter = create_formatter(prompt_config)
        template_name = prompt_config.get("template", "raw")
        print(f"Using prompt template: {template_name}")

    # Prepare sample items
    sample_items: list[SampleItem] = []
    problems_dict: dict[int, dict] = {}
    prompts_dict: dict[int, str] = {}
    original_responses_dict: dict[
        tuple[int, int], dict
    ] = {}  # (prompt_idx, response_idx) -> original response
    seed = exp_config.get("seed")
    save_original_response = exp_config.get("save_original_response", False)

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

        # Get user message - apply template if formatter is specified
        if prompt_formatter is not None and prompt_idx in problems_dict:
            # Reformat using template
            problem = problems_dict[prompt_idx]
            formatted_prompt = prompt_formatter.format(problem)
            user_message = (
                formatted_prompt[0].get("content", "") if formatted_prompt else ""
            )
        else:
            # Use original prompt from response file
            user_message = prompts_dict.get(prompt_idx, "")
            if not user_message:
                # Try to extract from prompt field
                prompt = response_data.get("prompt", [])
                if isinstance(prompt, list) and prompt:
                    user_message = prompt[0].get("content", "")

        for response_idx, response in enumerate(responses):
            # Check if original response was successful
            if not response.get("success", False):
                print(f"  Response {response_idx}: Not successful, skipping")
                continue

            # Store original response if flag is set
            if save_original_response:
                original_responses_dict[(prompt_idx, response_idx)] = response.copy()

            # Extract thinking trace - first try "reasoning" field, then extract from content
            thinking_trace = response.get("reasoning")

            if thinking_trace is None:
                # Try to get from completion.reasoning if not directly available
                completion = response.get("completion", {})
                if isinstance(completion, dict):
                    thinking_trace = completion.get("reasoning")

            if thinking_trace is None:
                # Fall back to extracting from content field using tags
                content = response.get("content")
                if content is None:
                    # Try to get from completion.text if content is not directly available
                    completion = response.get("completion", {})
                    if isinstance(completion, dict):
                        content = completion.get("text") or completion.get("content")

                if content:
                    thinking_trace = extract_thinking_trace(content, thinking_tag)

            if thinking_trace is None:
                print(
                    f"  Response {response_idx}: No thinking trace found (checked 'reasoning' field and <{thinking_tag}> tags), skipping"
                )
                print(f"    Response keys: {list(response.keys())}")
                continue

            # Process thinking trace based on operation type
            n_sentences = exp_config.get("n_sentences_to_remove")
            percentage = exp_config.get("percentage_to_remove")
            cut_at_percentage = exp_config.get("cut_at_percentage")
            cut_start_percentage = exp_config.get("cut_start_percentage")
            cut_end_percentage = exp_config.get("cut_end_percentage")

            try:
                # Calculate original token count using tokenizer if available
                if tokenizer is not None:
                    original_token_count = len(
                        tokenizer.encode(thinking_trace, add_special_tokens=False)
                    )
                else:
                    original_token_count = len(thinking_trace.split())

                # Start with original thinking trace
                current_thinking = thinking_trace
                total_sentences_removed = 0

                # Step 1: ALWAYS cut the original CoT first (if specified)
                # This ensures we work with a subset of the original trace
                if cut_start_percentage is not None and cut_end_percentage is not None:
                    # Cut from start_percentage to end_percentage
                    current_thinking, n_tokens_kept = cut_thinking_trace_at_percentage(
                        current_thinking,
                        cut_end_percentage,
                        tokenizer=tokenizer,
                        start_percentage=cut_start_percentage,
                    )
                elif cut_at_percentage is not None:
                    # Cut at percentage (backward compatibility - keeps first X%)
                    current_thinking, n_tokens_kept = cut_thinking_trace_at_percentage(
                        current_thinking, cut_at_percentage, tokenizer=tokenizer
                    )

                # Step 2: Remove random sentences from the already cut CoT
                # This operates on current_thinking, which is the result of Step 1
                if n_sentences is not None or percentage is not None:
                    if seed is not None:
                        deterministic_seed = seed + prompt_idx * 1000 + response_idx
                    else:
                        deterministic_seed = seed

                    modified_thinking, removed_positions, n_sentences_removed = (
                        remove_random_sentences(
                            current_thinking,  # Operates on the cut trace from Step 1
                            n_sentences=n_sentences,
                            percentage=percentage,
                            seed=deterministic_seed,
                        )
                    )
                    total_sentences_removed = n_sentences_removed
                else:
                    # Only cutting, no random removal
                    modified_thinking = current_thinking

                # Calculate final statistics
                if tokenizer is not None:
                    final_token_count = len(
                        tokenizer.encode(modified_thinking, add_special_tokens=False)
                    )
                else:
                    final_token_count = len(modified_thinking.split())

                # Calculate sentence statistics
                sentence_pattern = r"(?<=[.!?])\s+|\n+|$"
                original_parts = [
                    p.strip()
                    for p in re.split(sentence_pattern, thinking_trace)
                    if p.strip()
                ]
                original_sentence_count = len(original_parts)

                # Calculate percentage_removed as percentage of tokens removed relative to original whole CoT
                percentage_removed = (
                    ((original_token_count - final_token_count) / original_token_count)
                    * 100
                    if original_token_count > 0
                    else 0
                )
                n_sentences_removed = total_sentences_removed

            except ValueError as e:
                print(f"  Response {response_idx}: Error - {str(e)}")
                continue

            # Build completion prompt
            completion_prompt = build_completion_prompt(
                prefix_user_tokens=prefix_user_tokens,
                user_message=user_message,
                postfix_user_tokens=postfix_user_tokens,
                prefix_assistant_tokens=prefix_assistant_tokens,
                open_thinking_token=open_thinking_token,
                close_thinking_token=close_thinking_token,
                full_thinking=modified_thinking,
                close_thinking_tag=close_thinking_tag,
            )

            sample_items.append(
                SampleItem(
                    prompt_str=completion_prompt,
                    prompt_index=prompt_idx,
                    response_index=response_idx,
                    metadata={
                        "n_sentences_removed": n_sentences_removed,
                        "percentage_removed": percentage_removed,
                        "original_sentence_count": original_sentence_count,
                        "original_token_count": original_token_count,
                        "modified_token_count": final_token_count,
                        "close_thinking_tag_mode": close_thinking_tag,
                        "tokenizer_used": tokenizer_name if tokenizer else None,
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

    # Add original responses to results if flag is set
    if save_original_response:
        for prompt_idx, responses in grouped_results.items():
            for result in responses:
                response_idx = result["response_index"]
                key = (prompt_idx, response_idx)
                if key in original_responses_dict:
                    result["original_response"] = original_responses_dict[key]

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
