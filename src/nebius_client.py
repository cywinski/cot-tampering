# ABOUTME: Client for sampling responses from LLMs via Nebius API with parallel sampling and retry logic
# ABOUTME: Supports multiple responses per prompt with automatic failure handling and retries

import asyncio
import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()


@dataclass
class SamplingConfig:
    """Configuration for LLM sampling."""

    model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.95
    top_k: int | None = None
    n_responses: int = 1
    max_retries: int = 3
    timeout: float = 60.0


class NebiusClient:
    """Client for sampling from LLMs through Nebius API with parallel sampling."""

    def __init__(self, config: SamplingConfig):
        """Initialize the Nebius client.

        Args:
            config: Configuration for sampling behavior
        """
        self.config = config
        api_key = os.getenv("NEBIUS_API_KEY")
        if not api_key:
            raise ValueError("NEBIUS_API_KEY environment variable not set")

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.studio.nebius.com/v1",
        )

    async def _sample_single_response(
        self, messages: list[dict[str, str]], retry_count: int = 0
    ) -> dict[str, Any] | None:
        """Sample a single response with retry logic.

        Args:
            messages: List of message dicts with 'role' and 'content'
            retry_count: Current retry attempt number

        Returns:
            Response dict with content and metadata, or None if all retries failed
        """
        try:
            # Build API call parameters
            api_params = {
                "model": self.config.model,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "top_p": self.config.top_p,
            }

            # Add top_k if specified
            if self.config.top_k is not None:
                api_params["top_k"] = self.config.top_k

            response = await asyncio.wait_for(
                self.client.chat.completions.create(**api_params),
                timeout=self.config.timeout,
            )

            return {
                "content": response.choices[0].message.content,
                "finish_reason": response.choices[0].finish_reason,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                "model": response.model,
                "success": True,
            }

        except Exception as e:
            if retry_count < self.config.max_retries:
                # Exponential backoff
                await asyncio.sleep(2**retry_count)
                return await self._sample_single_response(messages, retry_count + 1)
            else:
                return {
                    "content": None,
                    "error": str(e),
                    "success": False,
                }

    async def sample_prompt(
        self, messages: list[dict[str, str]]
    ) -> list[dict[str, Any]]:
        """Sample multiple responses for a single prompt in parallel.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            List of response dicts, one per requested response
        """
        tasks = [
            self._sample_single_response(messages)
            for _ in range(self.config.n_responses)
        ]
        responses = await asyncio.gather(*tasks)

        # Filter out None responses and check if we got all responses
        valid_responses = [r for r in responses if r is not None]

        if len(valid_responses) < self.config.n_responses:
            print(
                f"Warning: Only got {len(valid_responses)}/{self.config.n_responses} responses"
            )

        return valid_responses

    async def sample_batch(
        self, prompts: list[list[dict[str, str]]]
    ) -> list[list[dict[str, Any]]]:
        """Sample responses for a batch of prompts.

        Each prompt gets n_responses samples in parallel.

        Args:
            prompts: List of prompts, where each prompt is a list of message dicts

        Returns:
            List of response lists, one per prompt
        """
        tasks = [self.sample_prompt(messages) for messages in prompts]
        results = await asyncio.gather(*tasks)
        return results

    def sample_batch_sync(
        self, prompts: list[list[dict[str, str]]]
    ) -> list[list[dict[str, Any]]]:
        """Synchronous wrapper for sample_batch.

        Args:
            prompts: List of prompts, where each prompt is a list of message dicts

        Returns:
            List of response lists, one per prompt
        """
        return asyncio.run(self.sample_batch(prompts))

    def sample_prompt_sync(
        self, messages: list[dict[str, str]]
    ) -> list[dict[str, Any]]:
        """Synchronous wrapper for sample_prompt.

        Sample multiple responses for a single prompt. Useful for interactive
        notebook usage where async/await is inconvenient.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            List of response dicts, one per requested response
        """
        return asyncio.run(self.sample_prompt(messages))
