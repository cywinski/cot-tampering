# ABOUTME: Client for OpenRouter completions API (raw prompt completions, not chat)
# ABOUTME: Supports custom prompt formatting tokens and provider filtering for experiments

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()


class OpenRouterCompletionsClient:
    """Client for OpenRouter completions API with raw prompt support."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.6,
        max_tokens: int = 10000,
        top_p: float = 1.0,
        providers: list[str] | None = None,
        timeout: float | None = None,
    ):
        """Initialize the OpenRouter completions client.

        Args:
            model: Model identifier (e.g., "deepseek/deepseek-r1")
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            providers: List of allowed providers (e.g., ["DeepInfra"])
            timeout: Request timeout in seconds
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.providers = providers
        self.timeout = timeout if timeout is not None else 120.0

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")

        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def _complete_async(self, prompt: str, retry_count: int = 0) -> dict[str, Any]:
        """Generate completion for a prompt (async).

        Args:
            prompt: Raw string prompt (including any special tokens)
            retry_count: Current retry attempt number

        Returns:
            Response dict with completion and metadata
        """
        try:
            # Build API call parameters
            api_params = {
                "model": self.model,
                "prompt": prompt,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
                "stream": False,
            }

            # Add provider filtering if specified
            if self.providers:
                api_params["provider"] = {"only": self.providers}

            # Make async request with timeout
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/completions",
                    headers=self.headers,
                    json=api_params,
                )
                response.raise_for_status()
                data = response.json()

            # Extract response
            choice = data["choices"][0]

            result = {
                "text": choice.get("text", ""),
                "finish_reason": choice.get("finish_reason"),
                "usage": {
                    "prompt_tokens": data["usage"]["prompt_tokens"],
                    "completion_tokens": data["usage"]["completion_tokens"],
                    "total_tokens": data["usage"]["total_tokens"],
                },
                "model": data["model"],
                "success": True,
            }

            # Check for reasoning/thinking content in response
            # OpenRouter may return reasoning in different fields depending on the model
            if "reasoning_content" in choice:
                result["reasoning_content"] = choice["reasoning_content"]

            if "reasoning" in choice:
                result["reasoning"] = choice["reasoning"]

            # Store the full raw response for debugging
            result["raw_choice"] = choice
            result["raw_response"] = data

            return result

        except Exception as e:
            if retry_count < 3:
                # Exponential backoff
                await asyncio.sleep(2**retry_count)
                return await self._complete_async(prompt, retry_count + 1)
            else:
                return {
                    "text": None,
                    "error": str(e),
                    "success": False,
                }

    def complete(self, prompt: str) -> dict[str, Any]:
        """Generate completion for a prompt.

        Args:
            prompt: Raw string prompt (including any special tokens)

        Returns:
            Response dict with completion and metadata
        """
        return asyncio.run(self._complete_async(prompt))

    async def complete_batch_async(self, prompts: list[str]) -> list[dict[str, Any]]:
        """Generate completions for multiple prompts in parallel.

        Args:
            prompts: List of raw string prompts

        Returns:
            List of response dicts
        """
        tasks = [self._complete_async(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)

    def complete_batch(self, prompts: list[str]) -> list[dict[str, Any]]:
        """Generate completions for multiple prompts in parallel.

        Args:
            prompts: List of raw string prompts

        Returns:
            List of response dicts
        """
        return asyncio.run(self.complete_batch_async(prompts))
