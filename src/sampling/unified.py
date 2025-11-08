# ABOUTME: Unified sampling pipeline for all experiment scripts
# ABOUTME: Provides common functions for preparing prompts, sampling in batches, and saving results

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .completions_client import OpenRouterCompletionsClient


@dataclass
class SampleItem:
    """Represents a single sample to be generated.

    Attributes:
        prompt_str: The formatted completion prompt string to send to API
        prompt_index: Index of the original prompt (for grouping results)
        response_index: Index of the response for this prompt (0, 1, 2, ...)
        metadata: Additional experiment-specific metadata dict
    """

    prompt_str: str
    prompt_index: int
    response_index: int
    metadata: dict[str, Any]


def format_completion_prompt(
    user_message: str,
    prefix_user_tokens: str = "",
    postfix_user_tokens: str = "",
    prefix_assistant_tokens: str = "",
    thinking_token: str = "",
) -> str:
    """Format a user message into a completion prompt string.

    Args:
        user_message: The user message content (string)
        prefix_user_tokens: Tokens before user message (e.g., "<｜begin▁of▁sentence｜><｜User｜>")
        postfix_user_tokens: Tokens after user message (e.g., "<|im_end|>\n")
        prefix_assistant_tokens: Tokens before assistant response (e.g., "<｜Assistant｜>")
        thinking_token: Opening thinking tag (e.g., "<think>")

    Returns:
        Formatted completion prompt string
    """
    return f"{prefix_user_tokens}{user_message}{postfix_user_tokens}{prefix_assistant_tokens}{thinking_token}"


async def sample_batch_async(
    client: OpenRouterCompletionsClient,
    sample_items: list[SampleItem],
    batch_size: int = 10,
) -> list[dict[str, Any]]:
    """Sample responses for a batch of prompts asynchronously with controlled parallelism.

    Args:
        client: OpenRouter completions client
        sample_items: List of SampleItem objects to sample
        batch_size: Maximum number of concurrent API calls

    Returns:
        List of result dicts in same order as sample_items, each containing:
        - completion: API response dict
        - success: bool
        - prompt_sent_to_api: str (the exact prompt sent)
        - prompt_index: int
        - response_index: int
        - metadata: dict (from SampleItem)
    """
    semaphore = asyncio.Semaphore(batch_size)

    async def sample_with_semaphore(item: SampleItem) -> dict[str, Any]:
        async with semaphore:
            completion_result = await client._complete_async(item.prompt_str)
            return {
                "completion": completion_result,
                "success": completion_result.get("success", False),
                "prompt_sent_to_api": item.prompt_str,
                "prompt_index": item.prompt_index,
                "response_index": item.response_index,
                "metadata": item.metadata,
            }

    tasks = [sample_with_semaphore(item) for item in sample_items]
    results = await asyncio.gather(*tasks)
    return list(results)


def group_results_by_prompt(
    results: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    """Group sampling results by prompt_index.

    Args:
        results: List of result dicts from sample_batch_async

    Returns:
        Dict mapping prompt_index to list of results for that prompt
    """
    grouped: dict[int, list[dict[str, Any]]] = {}
    for result in results:
        prompt_idx = result["prompt_index"]
        if prompt_idx not in grouped:
            grouped[prompt_idx] = []
        grouped[prompt_idx].append(result)
    return grouped


def _clean_completion_result(completion: dict[str, Any]) -> dict[str, Any]:
    """Remove redundant reasoning fields from completion result.

    Args:
        completion: Raw completion result dict

    Returns:
        Cleaned completion dict without redundant reasoning
    """
    cleaned = {
        "text": completion.get("text"),
        "finish_reason": completion.get("finish_reason"),
        "usage": completion.get("usage", {}),
        "model": completion.get("model"),
        "success": completion.get("success", False),
    }

    if "reasoning" in completion:
        cleaned["reasoning"] = completion["reasoning"]
    if "reasoning_content" in completion:
        cleaned["reasoning_content"] = completion["reasoning_content"]
    if "error" in completion:
        cleaned["error"] = completion["error"]

    if "raw_choice" in completion:
        raw_choice = completion["raw_choice"].copy()
        raw_choice.pop("reasoning", None)
        raw_choice.pop("reasoning_content", None)
        cleaned["raw_choice"] = raw_choice

    if "raw_response" in completion:
        raw_response = completion["raw_response"].copy()
        if "choices" in raw_response and len(raw_response["choices"]) > 0:
            choice = raw_response["choices"][0].copy()
            choice.pop("reasoning", None)
            choice.pop("reasoning_content", None)
            raw_response["choices"] = [choice]
        cleaned["raw_response"] = raw_response

    return cleaned


def save_prompt_results(
    grouped_results: dict[int, list[dict[str, Any]]],
    output_dir: Path,
    problems: dict[int, dict[str, Any]] | None = None,
    original_prompts: dict[int, str] | None = None,
) -> None:
    """Save grouped results to JSON files (one per prompt).

    Args:
        grouped_results: Dict mapping prompt_index to list of results
        output_dir: Directory to save JSON files
        problems: Optional dict mapping prompt_index to problem dict
        original_prompts: Optional dict mapping prompt_index to original prompt string
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for prompt_idx, responses in grouped_results.items():
        responses_sorted = sorted(responses, key=lambda r: r["response_index"])
        result_data: dict[str, Any] = {
            "prompt_index": prompt_idx,
            "responses": [
                {
                    "response_index": r["response_index"],
                    "prompt_sent_to_api": r["prompt_sent_to_api"],
                    "completion": _clean_completion_result(r["completion"]),
                    "success": r["success"],
                    **r["metadata"],
                }
                for r in responses_sorted
            ],
        }

        if problems and prompt_idx in problems:
            result_data["problem"] = problems[prompt_idx]

        if original_prompts and prompt_idx in original_prompts:
            result_data["prompt"] = original_prompts[prompt_idx]

        output_file = output_dir / f"prompt_{prompt_idx:04d}.json"
        with open(output_file, "w") as f:
            json.dump(result_data, f, indent=2)
