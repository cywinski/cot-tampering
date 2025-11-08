# ABOUTME: Shared configuration and client factory for LLM sampling
# ABOUTME: Supports OpenRouter API with unified interface for sampling configuration

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class SamplingConfig:
    """Configuration for LLM sampling across different providers."""

    model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.95
    top_k: int | None = None
    n_responses: int = 1
    max_retries: int = 3
    timeout: float | None = (
        60.0  # Set to None for no timeout (useful for reasoning models)
    )
    batch_size: int = 1  # Number of prompts to process in parallel (for n_responses=1)

    # OpenRouter-specific options
    logprobs: bool = False
    top_logprobs: int = 5
    reasoning: bool = False
    site_url: str | None = None
    site_name: str | None = None

    # Thinking tags for wrapping reasoning traces (OpenRouter only)
    wrap_thinking: bool = False  # Whether to wrap reasoning in tags
    thinking_tag: str = "think"  # Tag name to use (will be <tag> and </tag>)


def create_client(provider: Literal["openrouter"], config: SamplingConfig):
    """Create an LLM client based on provider type.

    Args:
        provider: Provider name ("openrouter")
        config: Sampling configuration

    Returns:
        Client instance (OpenRouterClient)

    Raises:
        ValueError: If provider is not supported
    """
    if provider == "openrouter":
        try:
            from .openrouter_client import OpenRouterClient
        except ImportError:
            from openrouter_client import OpenRouterClient  # type: ignore[no-redef]
        return OpenRouterClient(config)
    else:
        raise ValueError(
            f"Unknown provider: {provider}. Supported providers: openrouter"
        )
