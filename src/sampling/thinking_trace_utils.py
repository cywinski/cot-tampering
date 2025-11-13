# ABOUTME: Utilities for extracting and manipulating thinking traces from LLM responses
# ABOUTME: Supports insertion of text at specific positions and prompt reconstruction

from __future__ import annotations

import random
import re
from typing import Callable, Literal

from transformers import AutoTokenizer


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
    prefix_user_tokens: str,
    user_message: str,
    postfix_user_tokens: str,
    prefix_assistant_tokens: str,
    open_thinking_token: str,
    close_thinking_token: str,
    full_thinking: str,
    close_thinking_tag: bool = True,
) -> str:
    """Build a completion prompt with thinking trace.

    Args:
        prefix_user_tokens: Tokens before user message
        user_message: User's message content
        postfix_user_tokens: Tokens after user message
        prefix_assistant_tokens: Tokens before assistant response
        open_thinking_token: Opening thinking token (e.g., "<think>")
        close_thinking_token: Closing thinking token (e.g., "</think>")
        full_thinking: Complete thinking trace content (can be empty string for 100% removal)
        close_thinking_tag: If True, include close_thinking_token (model generates only answer).
                           If False, leave tag open (model can continue thinking).

    Returns:
        Formatted prompt string for completions API
    """
    if close_thinking_tag:
        if full_thinking.strip():
            prompt = f"{prefix_user_tokens}{user_message}{postfix_user_tokens}{prefix_assistant_tokens}{open_thinking_token}{full_thinking}{close_thinking_token}\n"
        else:
            prompt = f"{prefix_user_tokens}{user_message}{postfix_user_tokens}{prefix_assistant_tokens}{open_thinking_token}{close_thinking_token}"
    else:
        prompt = f"{prefix_user_tokens}{user_message}{postfix_user_tokens}{prefix_assistant_tokens}{open_thinking_token}{full_thinking}"

    return prompt


def build_prefill_prompt(
    prefix_user_tokens: str,
    user_message: str,
    postfix_user_tokens: str,
    prefix_assistant_tokens: str,
    thinking_tag: str,
    prefill: str,
) -> str:
    """Build a completion prompt with prefill text in the thinking trace.

    The prompt format is: {prefix_user_tokens}{user_message}{postfix_user_tokens}{prefix_assistant_tokens}<{thinking_tag}>\n{prefill}
    The model will complete the rest of the thinking trace.

    Args:
        prefix_user_tokens: Tokens before user message
        user_message: User's message content
        postfix_user_tokens: Tokens after user message
        prefix_assistant_tokens: Tokens before assistant response
        thinking_tag: Tag name for thinking (e.g., "think")
        prefill: Text to prefill in the thinking trace (model will complete after this)

    Returns:
        Formatted prompt string for completions API with prefill
    """
    thinking_token = f"<{thinking_tag}>"
    prompt = f"{prefix_user_tokens}{user_message}{postfix_user_tokens}{prefix_assistant_tokens}{thinking_token}\n{prefill}"
    return prompt


def remove_random_tokens(
    thinking_trace: str,
    n_tokens: int | None = None,
    percentage: float | None = None,
    seed: int | None = None,
    tokenizer: AutoTokenizer | None = None,
) -> tuple[str, list[int], int]:
    """Remove random tokens from thinking trace (excluding thinking tags).

    Args:
        thinking_trace: Original thinking trace content (without <think> tags)
        n_tokens: Number of tokens to remove (mutually exclusive with percentage)
        percentage: Percentage of CoT tokens to remove as float 0-1 (1.0 = 100% = remove all tokens)
        seed: Random seed for reproducibility
        tokenizer: Optional tokenizer from transformers. If None, uses whitespace tokenization.

    Returns:
        Tuple of (modified_thinking_trace, removed_positions, n_tokens_removed)
        - modified_thinking_trace: Thinking trace with tokens removed (can be empty string if 100%)
        - removed_positions: List of character positions where tokens were removed (empty if using tokenizer)
        - n_tokens_removed: Actual number of tokens removed
    """
    if n_tokens is None and percentage is None:
        raise ValueError("Must specify either n_tokens or percentage")
    if n_tokens is not None and percentage is not None:
        raise ValueError("Cannot specify both n_tokens and percentage")

    if seed is not None:
        random.seed(seed)

    if tokenizer is not None:
        # Use tokenizer for proper tokenization
        token_ids = tokenizer.encode(thinking_trace, add_special_tokens=False)
        original_token_count = len(token_ids)

        if original_token_count == 0:
            return "", [], 0

        # Calculate number of tokens to remove
        if percentage is not None:
            if not 0 <= percentage <= 1:
                raise ValueError(
                    f"Percentage must be between 0 and 1, got {percentage}"
                )
            n_tokens_to_remove = int(original_token_count * percentage)
            if percentage > 0 and n_tokens_to_remove == 0:
                n_tokens_to_remove = 1
        else:
            assert n_tokens is not None
            n_tokens_to_remove = n_tokens

        if n_tokens_to_remove > original_token_count:
            raise ValueError(
                f"Cannot remove {n_tokens_to_remove} tokens from trace with only {original_token_count} tokens"
            )

        if n_tokens_to_remove == original_token_count:
            return "", [], n_tokens_to_remove

        # Select random token indices to remove
        indices_to_remove = set(
            random.sample(range(len(token_ids)), n_tokens_to_remove)
        )

        # Remove selected tokens
        kept_token_ids = [
            token_id
            for i, token_id in enumerate(token_ids)
            if i not in indices_to_remove
        ]

        # Decode back to text
        modified_thinking = tokenizer.decode(kept_token_ids, skip_special_tokens=True)

        # For removed_positions, we'll return empty list since character positions don't map well with tokenizer tokens
        return modified_thinking, [], n_tokens_to_remove
    else:
        # Simple tokenization by whitespace (backward compatibility)
        tokens = thinking_trace.split()
        original_token_count = len(tokens)

        # Calculate number of tokens to remove
        if percentage is not None:
            if not 0 <= percentage <= 1:
                raise ValueError(
                    f"Percentage must be between 0 and 1, got {percentage}"
                )
            # Allow removing up to 100% of tokens (all tokens)
            n_tokens_to_remove = int(original_token_count * percentage)
            # Special case: if percentage is very small but not 0, remove at least 1 token
            if percentage > 0 and n_tokens_to_remove == 0:
                n_tokens_to_remove = 1
        else:
            assert n_tokens is not None
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
        indices_to_remove = set(random.sample(range(len(tokens)), n_tokens_to_remove))
        indices_to_remove = set(sorted(indices_to_remove, reverse=True))

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


def insert_random_tokens(
    thinking_trace: str,
    n_tokens: int | None = None,
    percentage: float | None = None,
    seed: int | None = None,
    tokenizer: AutoTokenizer | None = None,
) -> tuple[str, list[int], int]:
    """Insert random tokens into thinking trace at random positions.

    Args:
        thinking_trace: Original thinking trace content (without <think> tags)
        n_tokens: Number of tokens to insert (mutually exclusive with percentage)
        percentage: Percentage of original trace length to insert as float 0-1
                    (0.5 = 50% = insert tokens equal to 50% of original length)
        seed: Random seed for reproducibility
        tokenizer: Optional tokenizer from transformers. If None, uses whitespace tokenization.

    Returns:
        Tuple of (modified_thinking_trace, insertion_positions, n_tokens_inserted)
        - modified_thinking_trace: Thinking trace with random tokens inserted
        - insertion_positions: List of character positions where tokens were inserted
        - n_tokens_inserted: Actual number of tokens inserted
    """
    if n_tokens is None and percentage is None:
        raise ValueError("Must specify either n_tokens or percentage")
    if n_tokens is not None and percentage is not None:
        raise ValueError("Cannot specify both n_tokens and percentage")

    if seed is not None:
        random.seed(seed)

    if tokenizer is not None:
        # Use tokenizer for proper tokenization
        token_ids = tokenizer.encode(thinking_trace, add_special_tokens=False)
        original_token_count = len(token_ids)

        # Calculate number of tokens to insert based on original length
        if percentage is not None:
            if not 0 <= percentage <= 1:
                raise ValueError(
                    f"Percentage must be between 0 and 1, got {percentage}"
                )
            n_tokens_to_insert = int(original_token_count * percentage)
            if percentage > 0 and n_tokens_to_insert == 0:
                n_tokens_to_insert = 1
        else:
            assert n_tokens is not None
            n_tokens_to_insert = n_tokens

        if n_tokens_to_insert == 0:
            return thinking_trace, [], 0

        # Generate random tokens from vocabulary and insert them one by one
        # Get valid token IDs (some tokenizers have gaps in the vocab)
        vocab = tokenizer.get_vocab()
        valid_token_ids = list(vocab.values())

        modified_thinking = thinking_trace
        insertion_char_positions = []

        # Insert tokens one by one at random positions
        for _ in range(n_tokens_to_insert):
            # Sample a random token ID from valid vocabulary
            random_token_id = random.choice(valid_token_ids)
            # Decode to get text representation
            random_token_text = tokenizer.decode(
                [random_token_id], skip_special_tokens=True
            )

            # Choose a random position in the current modified thinking trace
            if len(modified_thinking) == 0:
                insert_pos = 0
            else:
                insert_pos = random.randint(0, len(modified_thinking))

            # Insert the token with a space
            if insert_pos == 0:
                modified_thinking = random_token_text + " " + modified_thinking
            elif insert_pos == len(modified_thinking):
                modified_thinking = modified_thinking + " " + random_token_text
            else:
                modified_thinking = (
                    modified_thinking[:insert_pos]
                    + " "
                    + random_token_text
                    + " "
                    + modified_thinking[insert_pos:]
                )
            insertion_char_positions.append(insert_pos)

        # Count actual tokens inserted using tokenizer
        final_token_ids = tokenizer.encode(modified_thinking, add_special_tokens=False)
        actual_n_tokens_inserted = len(final_token_ids) - original_token_count

        return (
            modified_thinking,
            sorted(insertion_char_positions),
            actual_n_tokens_inserted,
        )
    else:
        # Simple tokenization by whitespace (backward compatibility)
        tokens = thinking_trace.split()
        original_token_count = len(tokens)

        # Calculate number of tokens to insert based on original length
        if percentage is not None:
            if not 0 <= percentage <= 1:
                raise ValueError(
                    f"Percentage must be between 0 and 1, got {percentage}"
                )
            n_tokens_to_insert = int(original_token_count * percentage)
            if percentage > 0 and n_tokens_to_insert == 0:
                n_tokens_to_insert = 1
        else:
            assert n_tokens is not None
            n_tokens_to_insert = n_tokens

        if n_tokens_to_insert == 0:
            return thinking_trace, [], 0

        # Generate random words (simple approach)
        # Use a mix of common words and random characters
        random_words = [
            random.choice(
                [
                    "the",
                    "a",
                    "an",
                    "and",
                    "or",
                    "but",
                    "in",
                    "on",
                    "at",
                    "to",
                    "for",
                    "of",
                    "with",
                    "by",
                    "from",
                    "about",
                    "into",
                    "through",
                    "during",
                    "including",
                    "considering",
                    "analyzing",
                    "evaluating",
                    "reasoning",
                    "thinking",
                    "concluding",
                    "determining",
                    "calculating",
                    "computing",
                ]
            )
            + str(random.randint(0, 9999))
            for _ in range(n_tokens_to_insert)
        ]

        # Insert at random positions throughout the trace
        insertion_positions = sorted(
            random.sample(
                range(len(tokens) + 1), min(n_tokens_to_insert, len(tokens) + 1)
            )
        )

        # Build modified token sequence with insertions
        modified_tokens = []
        insertion_char_positions = []
        current_char_pos = 0
        insert_idx = 0

        for i, token in enumerate(tokens):
            # Insert random tokens before this position if needed
            while (
                insert_idx < len(insertion_positions)
                and insertion_positions[insert_idx] == i
            ):
                modified_tokens.append(random_words[insert_idx % len(random_words)])
                insertion_char_positions.append(current_char_pos)
                insert_idx += 1

            modified_tokens.append(token)
            current_char_pos += len(token) + 1  # +1 for space

        # Insert any remaining tokens at the end
        while insert_idx < len(insertion_positions):
            modified_tokens.append(random_words[insert_idx % len(random_words)])
            insertion_char_positions.append(current_char_pos)
            insert_idx += 1

        # Reconstruct thinking trace
        modified_thinking = " ".join(modified_tokens)

        return modified_thinking, sorted(insertion_char_positions), n_tokens_to_insert


def cut_thinking_trace_at_percentage(
    thinking_trace: str,
    percentage: float,
    tokenizer: AutoTokenizer | None = None,
    start_percentage: float | None = None,
) -> tuple[str, int]:
    """Cut thinking trace at specific percentages, keeping tokens from start_percentage to percentage.

    Args:
        thinking_trace: Original thinking trace content (without <think> tags)
        percentage: End percentage (percentage of tokens to keep from the beginning as float 0-1)
                    If start_percentage is None, keeps first X% (0.5 = 50% = keep first half)
                    If start_percentage is provided, this is the end percentage (0.8 = keep up to 80%)
        tokenizer: Optional tokenizer from transformers. If None, uses whitespace tokenization.
        start_percentage: Optional start percentage (float 0-1). If provided, keeps tokens from
                         start_percentage to percentage (e.g., 0.2 to 0.8 keeps middle 60%)

    Returns:
        Tuple of (modified_thinking_trace, n_tokens_kept)
        - modified_thinking_trace: Thinking trace with tokens from [start_percentage, percentage] range
        - n_tokens_kept: Actual number of tokens kept
    """
    if not 0 <= percentage <= 1:
        raise ValueError(f"Percentage must be between 0 and 1, got {percentage}")

    if start_percentage is not None:
        if not 0 <= start_percentage <= 1:
            raise ValueError(
                f"Start percentage must be between 0 and 1, got {start_percentage}"
            )
        if start_percentage >= percentage:
            raise ValueError(
                f"Start percentage ({start_percentage}) must be less than end percentage ({percentage})"
            )

    if tokenizer is not None:
        # Use tokenizer for proper tokenization
        token_ids = tokenizer.encode(thinking_trace, add_special_tokens=False)
        original_token_count = len(token_ids)

        if original_token_count == 0:
            return "", 0

        if start_percentage is not None:
            # Keep tokens from start_percentage to percentage
            start_idx = int(original_token_count * start_percentage)
            end_idx = int(original_token_count * percentage)
            kept_token_ids = token_ids[start_idx:end_idx]
        else:
            # Keep only the first percentage of tokens (backward compatibility)
            n_tokens_to_keep = int(original_token_count * percentage)
            if n_tokens_to_keep == 0:
                return "", 0
            kept_token_ids = token_ids[:n_tokens_to_keep]

        if len(kept_token_ids) == 0:
            return "", 0

        # Decode back to text
        modified_thinking = tokenizer.decode(kept_token_ids, skip_special_tokens=True)

        return modified_thinking, len(kept_token_ids)
    else:
        # Simple tokenization by whitespace (backward compatibility)
        tokens = thinking_trace.split()
        original_token_count = len(tokens)

        if original_token_count == 0:
            return "", 0

        if start_percentage is not None:
            # Keep tokens from start_percentage to percentage
            start_idx = int(original_token_count * start_percentage)
            end_idx = int(original_token_count * percentage)
            kept_tokens = tokens[start_idx:end_idx]
        else:
            # Keep only the first percentage of tokens (backward compatibility)
            n_tokens_to_keep = int(original_token_count * percentage)
            if n_tokens_to_keep == 0:
                return "", 0
            kept_tokens = tokens[:n_tokens_to_keep]

        if len(kept_tokens) == 0:
            return "", 0

        # Reconstruct thinking trace
        modified_thinking = " ".join(kept_tokens)

        return modified_thinking, len(kept_tokens)


def remove_random_sentences(
    thinking_trace: str,
    n_sentences: int | None = None,
    percentage: float | None = None,
    seed: int | None = None,
    sentence_filter: Callable[[str], bool] | None = None,
) -> tuple[str, list[int], int]:
    """Remove random sentences from thinking trace (excluding thinking tags).

    Args:
        thinking_trace: Original thinking trace content (without <think> tags)
        n_sentences: Number of sentences to remove (mutually exclusive with percentage)
        percentage: Percentage of sentences to remove as float 0-1 (1.0 = 100% = remove all sentences)
        seed: Random seed for reproducibility
        sentence_filter: Optional function that takes a sentence string and returns True if it's
                        a candidate for removal. If None, all sentences are candidates.
                        This enables future classification/categorization of sentences.

    Returns:
        Tuple of (modified_thinking_trace, removed_positions, n_sentences_removed)
        - modified_thinking_trace: Thinking trace with sentences removed (can be empty string if 100%)
        - removed_positions: List of character positions where sentences started (before removal)
        - n_sentences_removed: Actual number of sentences removed
    """
    if n_sentences is None and percentage is None:
        raise ValueError("Must specify either n_sentences or percentage")
    if n_sentences is not None and percentage is not None:
        raise ValueError("Cannot specify both n_sentences and percentage")

    if seed is not None:
        random.seed(seed)

    # Split into sentences using regex (matches . ! ? followed by space/newline or end of string)
    # This preserves sentence boundaries
    sentence_pattern = r"(?<=[.!?])\s+|\n+|$"
    parts = re.split(sentence_pattern, thinking_trace)

    # Filter out empty strings and reconstruct sentences with their terminators
    sentences = []
    current_pos = 0
    sentence_positions = []

    for part in parts:
        if part.strip():
            sentences.append(part)
            sentence_positions.append(current_pos)
        current_pos += len(part)
        # Account for the separator that was consumed by split
        if current_pos < len(thinking_trace):
            separator_match = re.match(sentence_pattern, thinking_trace[current_pos:])
            if separator_match:
                current_pos += len(separator_match.group(0))

    original_sentence_count = len(sentences)

    if original_sentence_count == 0:
        return thinking_trace, [], 0

    # Apply sentence filter if provided to get candidate sentences
    if sentence_filter is not None:
        candidate_indices = [
            i for i, sent in enumerate(sentences) if sentence_filter(sent)
        ]
    else:
        candidate_indices = list(range(original_sentence_count))

    candidate_count = len(candidate_indices)

    if candidate_count == 0:
        return thinking_trace, [], 0

    # Calculate number of sentences to remove from candidates
    if percentage is not None:
        if not 0 <= percentage <= 1:
            raise ValueError(f"Percentage must be between 0 and 1, got {percentage}")
        n_sentences_to_remove = int(candidate_count * percentage)
        # Special case: if percentage is very small but not 0, remove at least 1 sentence
        if percentage > 0 and n_sentences_to_remove == 0:
            n_sentences_to_remove = 1
    else:
        assert n_sentences is not None
        n_sentences_to_remove = n_sentences

    # Validate removal count
    if n_sentences_to_remove > candidate_count:
        raise ValueError(
            f"Cannot remove {n_sentences_to_remove} sentences from {candidate_count} candidate sentences"
        )

    # Special case: removing 100% of all sentences means empty trace
    if n_sentences_to_remove == original_sentence_count and sentence_filter is None:
        return "", [], n_sentences_to_remove

    # Select random sentence indices to remove from candidates
    indices_to_remove = set(random.sample(candidate_indices, n_sentences_to_remove))

    # Track character positions of removed sentences
    removed_positions = [sentence_positions[i] for i in sorted(indices_to_remove)]

    # Reconstruct thinking trace without removed sentences
    kept_sentences = [
        sent for i, sent in enumerate(sentences) if i not in indices_to_remove
    ]

    # Join sentences with appropriate spacing
    if kept_sentences:
        modified_thinking = " ".join(kept_sentences)
    else:
        modified_thinking = ""

    return modified_thinking, removed_positions, n_sentences_to_remove


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
        return completed_text[len(prefix) :]
    return completed_text
