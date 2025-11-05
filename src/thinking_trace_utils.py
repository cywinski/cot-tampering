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
) -> str:
    """Build a completion prompt with complete thinking trace and closed tag.

    Args:
        prefix_tokens: Tokens at the start (e.g., "<｜begin▁of▁sentence｜>")
        user_message: User's message content
        assistant_prefix: Assistant start tokens (e.g., "<｜Assistant｜>")
        thinking_tag: Tag name for thinking (e.g., "think")
        full_thinking: Complete thinking trace (possibly modified with insertion)

    Returns:
        Formatted prompt string for completions API, ending with closed thinking tag
    """
    # Build the prompt with complete thinking trace and closed tag
    # Model should generate only the final answer after </think>
    prompt = f"{prefix_tokens}{user_message}{assistant_prefix}<{thinking_tag}>\n{full_thinking}\n</{thinking_tag}>\n"
    return prompt


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
