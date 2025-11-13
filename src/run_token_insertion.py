# ABOUTME: Main script for token insertion experiment
# ABOUTME: Loads sampled responses, inserts random tokens into thinking traces, and completes with OpenRouter API

import asyncio
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
    extract_thinking_trace,
    group_results_by_prompt,
    insert_random_tokens,
    sample_batch_async,
    save_prompt_results,
)
from utils import create_formatter, load_config, load_response_file


def run_token_insertion(
    config_path: str = "experiments/configs/token_insertion_experiment.yaml",
):
    """Run token insertion experiment.

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
    n_tokens_to_insert = exp_config.get("n_tokens_to_insert")
    percentage_to_insert = exp_config.get("percentage_to_insert")

    # Validate operations
    if n_tokens_to_insert is None and percentage_to_insert is None:
        print(
            "ERROR: Must specify at least one of: n_tokens_to_insert or percentage_to_insert"
        )
        return

    if n_tokens_to_insert is not None and percentage_to_insert is not None:
        print("ERROR: Cannot specify both n_tokens_to_insert and percentage_to_insert")
        return

    # Print operation summary
    if n_tokens_to_insert is not None:
        operations = f"Inserting {n_tokens_to_insert} random tokens"
    elif percentage_to_insert is not None:
        operations = (
            f"Inserting {percentage_to_insert * 100:.1f}% of original trace length "
            f"({percentage_to_insert * 100:.1f}% means inserting tokens equal to {percentage_to_insert * 100:.1f}% of original length)"
        )

    print(f"Operation: {operations}")

    # Use completion_prompt_formatting if specified, otherwise fall back to prompt_formatting
    fmt_config = (
        config.get("completion_prompt_formatting") or config["prompt_formatting"]
    )
    prefix_user_tokens = fmt_config["prefix_user_tokens"]
    postfix_user_tokens = fmt_config.get("postfix_user_tokens", "")
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

            # Process thinking trace - insert random tokens
            n_tokens = exp_config.get("n_tokens_to_insert")
            percentage = exp_config.get("percentage_to_insert")

            try:
                # Calculate original token count using tokenizer if available
                if tokenizer is not None:
                    original_token_count = len(
                        tokenizer.encode(thinking_trace, add_special_tokens=False)
                    )
                else:
                    original_token_count = len(thinking_trace.split())

                if seed is not None:
                    deterministic_seed = seed + prompt_idx * 1000 + response_idx
                else:
                    deterministic_seed = seed

                modified_thinking, insertion_positions, tokens_inserted = (
                    insert_random_tokens(
                        thinking_trace,
                        n_tokens=n_tokens,
                        percentage=percentage,
                        seed=deterministic_seed,
                        tokenizer=tokenizer,
                    )
                )

                # Calculate final statistics
                if tokenizer is not None:
                    final_token_count = len(
                        tokenizer.encode(modified_thinking, add_special_tokens=False)
                    )
                else:
                    final_token_count = len(modified_thinking.split())

                percentage_inserted = (
                    (tokens_inserted / original_token_count) * 100
                    if original_token_count > 0
                    else 0
                )
                n_tokens_inserted = tokens_inserted

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
                        "n_tokens_inserted": n_tokens_inserted,
                        "percentage_inserted": percentage_inserted,
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
    fire.Fire(run_token_insertion)
