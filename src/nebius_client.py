# ABOUTME: Client for sampling responses from LLMs via Nebius API with parallel sampling per prompt and retry logic
# ABOUTME: Supports multiple responses per prompt in parallel using ThreadPoolExecutor with automatic failure handling

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

try:
    from .sampling_config import SamplingConfig
except ImportError:
    from sampling_config import SamplingConfig

load_dotenv()


class NebiusClient:
    """Client for sampling from LLMs through Nebius API with parallel sampling per prompt."""

    def __init__(self, config: SamplingConfig):
        """Initialize the Nebius client.

        Args:
            config: Configuration for sampling behavior
        """
        self.config = config
        api_key = os.getenv("NEBIUS_API_KEY")
        if not api_key:
            raise ValueError("NEBIUS_API_KEY environment variable not set")

        # Initialize client with optional timeout
        client_kwargs = {
            "api_key": api_key,
            "base_url": "https://api.studio.nebius.com/v1",
        }
        if self.config.timeout is not None:
            client_kwargs["timeout"] = self.config.timeout

        self.client = OpenAI(**client_kwargs)

        # Thread pool for parallel sampling
        self.executor = ThreadPoolExecutor(max_workers=self.config.n_responses)

    def _sample_single_response(
        self, messages: list[dict[str, str]], retry_count: int = 0
    ) -> dict[str, Any]:
        """Sample a single response with retry logic.

        Args:
            messages: List of message dicts with 'role' and 'content'
            retry_count: Current retry attempt number

        Returns:
            Response dict with content and metadata
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

            # Note: Nebius API does not support top_k parameter
            # It is kept in config for compatibility but not passed to API

            response = self.client.chat.completions.create(**api_params)

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
                time.sleep(2**retry_count)
                return self._sample_single_response(messages, retry_count + 1)
            else:
                return {
                    "content": None,
                    "error": str(e),
                    "success": False,
                }

    def sample_prompt(
        self, messages: list[dict[str, str]]
    ) -> list[dict[str, Any]]:
        """Sample multiple responses for a single prompt in parallel.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            List of response dicts, one per requested response
        """
        # Submit all sampling tasks in parallel
        futures = [
            self.executor.submit(self._sample_single_response, messages)
            for _ in range(self.config.n_responses)
        ]

        # Gather results as they complete
        responses = []
        for future in as_completed(futures):
            response = future.result()
            responses.append(response)

        # Check if we got all successful responses
        successful = sum(1 for r in responses if r.get("success", False))
        if successful < self.config.n_responses:
            print(
                f"Warning: Only got {successful}/{self.config.n_responses} successful responses"
            )

        return responses

    async def sample_prompt_async(
        self, messages: list[dict[str, str]]
    ) -> list[dict[str, Any]]:
        """Sample multiple responses for a single prompt in parallel (async).

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            List of response dicts, one per requested response
        """
        # Run the sync method in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.sample_prompt, messages)

    def sample_batch(
        self, prompts: list[list[dict[str, str]]]
    ) -> list[list[dict[str, Any]]]:
        """Sample responses for a batch of prompts.

        Each prompt gets n_responses samples in parallel, prompts processed sequentially.

        Args:
            prompts: List of prompts, where each prompt is a list of message dicts

        Returns:
            List of response lists, one per prompt
        """
        results = []
        for messages in prompts:
            responses = self.sample_prompt(messages)
            results.append(responses)
        return results

    def __del__(self):
        """Cleanup thread pool on deletion."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
