# ABOUTME: Client for sampling responses from LLMs via OpenRouter API with parallel sampling per prompt and retry logic
# ABOUTME: Supports multiple responses per prompt in parallel using asyncio for efficient concurrent API calls

import asyncio
import os
from typing import Any

import httpx
from dotenv import load_dotenv

try:
    from .sampling_config import SamplingConfig
except ImportError:
    from sampling_config import SamplingConfig

load_dotenv()


class OpenRouterClient:
    """Client for sampling from LLMs through OpenRouter API with parallel sampling per prompt."""

    def __init__(self, config: SamplingConfig):
        """Initialize the OpenRouter client.

        Args:
            config: Configuration for sampling behavior
        """
        self.config = config
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")

        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"

        # Build headers for OpenRouter
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.config.site_url:
            self.headers["HTTP-Referer"] = self.config.site_url
        if self.config.site_name:
            self.headers["X-Title"] = self.config.site_name

    async def _sample_single_response_async(
        self, messages: list[dict[str, str]], retry_count: int = 0
    ) -> dict[str, Any]:
        """Sample a single response with retry logic (async).

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

            # Add optional parameters
            if self.config.top_k is not None:
                api_params["top_k"] = self.config.top_k

            if self.config.logprobs:
                api_params["logprobs"] = True
                api_params["top_logprobs"] = self.config.top_logprobs

            if self.config.reasoning:
                api_params["reasoning"] = {"enabled": True}

            # Make async request with timeout
            timeout = self.config.timeout if self.config.timeout is not None else 60.0
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=api_params,
                )
                response.raise_for_status()
                data = response.json()

            # Extract response in Nebius-compatible format
            choice = data["choices"][0]
            message = choice["message"]

            # OpenRouter returns reasoning in separate fields, but Nebius combines them
            # To maintain compatibility, combine reasoning + content into single field
            content = message.get("content", "")

            # OpenRouter puts reasoning in message['reasoning'] field
            reasoning_content = message.get("reasoning")

            # Combine reasoning and final answer (if reasoning exists)
            if reasoning_content:
                # Optionally wrap reasoning in thinking tags
                if self.config.wrap_thinking:
                    tag = self.config.thinking_tag
                    wrapped_reasoning = f"<{tag}>\n{reasoning_content}\n</{tag}>"
                else:
                    wrapped_reasoning = reasoning_content

                # Format similar to how Nebius returns it: reasoning followed by answer
                combined_content = f"{wrapped_reasoning}\n\n{content}"
            else:
                combined_content = content

            result = {
                "content": combined_content,
                "finish_reason": choice["finish_reason"],
                "usage": {
                    "prompt_tokens": data["usage"]["prompt_tokens"],
                    "completion_tokens": data["usage"]["completion_tokens"],
                    "total_tokens": data["usage"]["total_tokens"],
                },
                "model": data["model"],
                "success": True,
            }

            # Include logprobs if requested
            if self.config.logprobs and "logprobs" in choice:
                result["logprobs"] = choice["logprobs"]

            return result

        except Exception as e:
            if retry_count < self.config.max_retries:
                # Exponential backoff
                await asyncio.sleep(2**retry_count)
                return await self._sample_single_response_async(messages, retry_count + 1)
            else:
                return {
                    "content": None,
                    "error": str(e),
                    "success": False,
                }

    async def _sample_prompt_async(
        self, messages: list[dict[str, str]]
    ) -> list[dict[str, Any]]:
        """Sample multiple responses for a single prompt in parallel (async).

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            List of response dicts, one per requested response
        """
        # Create all sampling tasks
        tasks = [
            self._sample_single_response_async(messages)
            for _ in range(self.config.n_responses)
        ]

        # Gather results concurrently
        responses = await asyncio.gather(*tasks)

        # Check if we got all successful responses
        successful = sum(1 for r in responses if r.get("success", False))
        if successful < self.config.n_responses:
            print(
                f"Warning: Only got {successful}/{self.config.n_responses} successful responses"
            )

        return list(responses)

    async def sample_prompt_async(
        self, messages: list[dict[str, str]]
    ) -> list[dict[str, Any]]:
        """Sample multiple responses for a single prompt in parallel (async).

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            List of response dicts, one per requested response
        """
        return await self._sample_prompt_async(messages)

    def sample_prompt(
        self, messages: list[dict[str, str]]
    ) -> list[dict[str, Any]]:
        """Sample multiple responses for a single prompt in parallel.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            List of response dicts, one per requested response
        """
        return asyncio.run(self._sample_prompt_async(messages))

    async def _sample_batch_async(
        self, prompts: list[list[dict[str, str]]]
    ) -> list[list[dict[str, Any]]]:
        """Sample responses for a batch of prompts (async).

        Each prompt gets n_responses samples in parallel, prompts processed sequentially.

        Args:
            prompts: List of prompts, where each prompt is a list of message dicts

        Returns:
            List of response lists, one per prompt
        """
        results = []
        for messages in prompts:
            responses = await self._sample_prompt_async(messages)
            results.append(responses)
        return results

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
        return asyncio.run(self._sample_batch_async(prompts))
