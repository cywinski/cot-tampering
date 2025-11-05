# ABOUTME: Utilities for extracting and manipulating thinking traces from LLM responses
# ABOUTME: Supports insertion of text at specific positions and prompt reconstruction

import random
import re
from typing import Literal


def extract_thinking_trace(content: str, thinking_tag: str = "think") -> str | None:
    """Extract thinking trace from response content.

    Args:
        content: Full response content with thinking tags
        thinking_tag: Tag name used to wrap thinking (e.g., "think")

    Returns:
        Thinking trace content without tags, or None if not found
    """
    pattern = f"<{thinking_tag}>(.*?)</{thinking_tag}>"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def insert_text_in_thinking(
    thinking_trace: str,
    insertion_text: str,
    position: Literal["random", "start", "middle", "end"] | int = "random",
    seed: int | None = None,
) -> tuple[str, str, str, int]:
    """Insert text at a specified position in thinking trace.

    Args:
        thinking_trace: Original thinking trace content
        insertion_text: Text to insert
        position: Where to insert - "random", "start", "middle", "end", or specific index
        seed: Random seed for reproducibility (only used if position="random")

    Returns:
        Tuple of (thinking_before, thinking_after, prompt_prefix, insertion_position)
        - thinking_before: Thinking trace before insertion point
        - thinking_after: Thinking trace after insertion point
        - prompt_prefix: thinking_before + insertion_text (for completion prompt)
        - insertion_position: Character position where insertion occurred
    """
    if seed is not None:
        random.seed(seed)

    # Determine insertion position
    if position == "start":
        insert_pos = 0
    elif position == "end":
        insert_pos = len(thinking_trace)
    elif position == "middle":
        insert_pos = len(thinking_trace) // 2
    elif position == "random":
        insert_pos = random.randint(0, len(thinking_trace))
    elif isinstance(position, int):
        insert_pos = max(0, min(position, len(thinking_trace)))
    else:
        raise ValueError(f"Invalid position: {position}")

    # Split thinking trace at insertion point
    thinking_before = thinking_trace[:insert_pos]
    thinking_after = thinking_trace[insert_pos:]

    # Build prompt prefix (what goes into the completion prompt)
    prompt_prefix = thinking_before + insertion_text

    return thinking_before, thinking_after, prompt_prefix, insert_pos


def build_completion_prompt(
    prefix_tokens: str,
    user_message: str,
    assistant_prefix: str,
    thinking_tag: str,
    full_thinking: str,
    close_thinking_tag: bool = True,
) -> str:
    """Build a completion prompt with thinking trace.

    Args:
        prefix_tokens: Tokens at the start (e.g., "<｜begin▁of▁sentence｜>")
        user_message: User's message content
        assistant_prefix: Assistant start tokens (e.g., "<｜Assistant｜>")
        thinking_tag: Tag name for thinking (e.g., "think")
        full_thinking: Complete thinking trace content (can be empty string for 100% removal)
        close_thinking_tag: If True, close </think> tag (model generates only answer).
                           If False, leave tag open (model can continue thinking).

    Returns:
        Formatted prompt string for completions API
    """
    if close_thinking_tag:
        # Closed tag: model generates only the final answer after </think>
        # Handle empty thinking (100% removal case)
        if full_thinking.strip():
            prompt = f"{prefix_tokens}{user_message}{assistant_prefix}<{thinking_tag}>\n{full_thinking}\n</{thinking_tag}>\n"
        else:
            # Empty thinking: just tags with no content
            prompt = f"{prefix_tokens}{user_message}{assistant_prefix}<{thinking_tag}>\n</{thinking_tag}>\n"
    else:
        # Open tag: model can continue extending the thinking trace
        prompt = f"{prefix_tokens}{user_message}{assistant_prefix}<{thinking_tag}>\n{full_thinking}"

    return prompt


def remove_random_tokens(
    thinking_trace: str,
    n_tokens: int | None = None,
    percentage: float | None = None,
    seed: int | None = None,
) -> tuple[str, list[int], int]:
    """Remove random tokens from thinking trace (excluding thinking tags).

    Args:
        thinking_trace: Original thinking trace content (without <think> tags)
        n_tokens: Number of tokens to remove (mutually exclusive with percentage)
        percentage: Percentage of CoT tokens to remove as float 0-1 (1.0 = 100% = remove all tokens)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (modified_thinking_trace, removed_positions, n_tokens_removed)
        - modified_thinking_trace: Thinking trace with tokens removed (can be empty string if 100%)
        - removed_positions: List of character positions where tokens were removed
        - n_tokens_removed: Actual number of tokens removed
    """
    if n_tokens is None and percentage is None:
        raise ValueError("Must specify either n_tokens or percentage")
    if n_tokens is not None and percentage is not None:
        raise ValueError("Cannot specify both n_tokens and percentage")

    if seed is not None:
        random.seed(seed)

    # Simple tokenization by whitespace
    tokens = thinking_trace.split()
    original_token_count = len(tokens)

    # Calculate number of tokens to remove
    if percentage is not None:
        if not 0 <= percentage <= 1:
            raise ValueError(f"Percentage must be between 0 and 1, got {percentage}")
        # Allow removing up to 100% of tokens (all tokens)
        n_tokens_to_remove = int(original_token_count * percentage)
        # Special case: if percentage is very small but not 0, remove at least 1 token
        if percentage > 0 and n_tokens_to_remove == 0:
            n_tokens_to_remove = 1
    else:
        n_tokens_to_remove = n_tokens

    # Allow removing all tokens (100%)
    if n_tokens_to_remove > original_token_count:
        raise ValueError(
            f"Cannot remove {n_tokens_to_remove} tokens from trace with only {original_token_count} tokens"
        )

    # Special case: removing 100% means empty trace
    if n_tokens_to_remove == original_token_count:
        return "", [], n_tokens_to_remove

    # Select random token indices to remove
    indices_to_remove = sorted(
        random.sample(range(len(tokens)), n_tokens_to_remove), reverse=True
    )

    # Track character positions of removed tokens
    removed_positions = []
    current_pos = 0
    for i, token in enumerate(tokens):
        if i in indices_to_remove:
            removed_positions.append(current_pos)
        current_pos += len(token) + 1  # +1 for space

    # Remove selected tokens
    for idx in indices_to_remove:
        tokens.pop(idx)

    # Reconstruct thinking trace
    modified_thinking = " ".join(tokens)

    return modified_thinking, sorted(removed_positions), n_tokens_to_remove


def extract_completion_after_insertion(
    completed_text: str,
    original_thinking_before: str,
    insertion_text: str,
) -> str:
    """Extract only the newly generated text after insertion point.

    Args:
        completed_text: Full completed text from API
        original_thinking_before: Original thinking trace before insertion point
        insertion_text: Text that was inserted

    Returns:
        Only the newly generated continuation
    """
    # Remove the prefix that was in the prompt
    prefix = original_thinking_before + insertion_text
    if completed_text.startswith(prefix):
        return completed_text[len(prefix):]
    return completed_text
