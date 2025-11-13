# ABOUTME: Sampling utilities module
# ABOUTME: Exports clients, unified sampling functions, prompt formatting, dataset loaders, and thinking trace utilities

from .completions_client import OpenRouterCompletionsClient
from .dataset_loaders import get_dataset, list_available_datasets, load_mmlu
from .prompt_formatter import PROMPT_TEMPLATES, PromptFormatter
from .thinking_trace_utils import (
    build_completion_prompt,
    build_prefill_prompt,
    cut_thinking_trace_at_percentage,
    extract_completion_after_insertion,
    extract_thinking_trace,
    insert_random_tokens,
    insert_text_in_thinking,
    remove_random_sentences,
    remove_random_tokens,
)
from .unified import (
    SampleItem,
    _clean_completion_result,
    format_completion_prompt,
    group_results_by_prompt,
    sample_batch_async,
    save_prompt_results,
)

__all__ = [
    "OpenRouterCompletionsClient",
    "SampleItem",
    "format_completion_prompt",
    "sample_batch_async",
    "group_results_by_prompt",
    "save_prompt_results",
    "_clean_completion_result",
    "get_dataset",
    "list_available_datasets",
    "load_mmlu",
    "PromptFormatter",
    "PROMPT_TEMPLATES",
    "build_completion_prompt",
    "build_prefill_prompt",
    "cut_thinking_trace_at_percentage",
    "extract_completion_after_insertion",
    "extract_thinking_trace",
    "insert_random_tokens",
    "insert_text_in_thinking",
    "remove_random_sentences",
    "remove_random_tokens",
]
