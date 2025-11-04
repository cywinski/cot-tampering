# ABOUTME: Flexible prompt formatting classes for different prompt templates and structures
# ABOUTME: Easily extensible to support various prompt formats for different tasks

from abc import ABC, abstractmethod
from typing import Any


class PromptFormatter(ABC):
    """Base class for prompt formatters."""

    @abstractmethod
    def format(self, problem: dict[str, Any]) -> list[dict[str, str]]:
        """Format a problem into OpenAI message format.

        Args:
            problem: Dict containing problem data (structure depends on dataset)

        Returns:
            List of message dicts with 'role' and 'content' keys
        """
        pass


class SimpleQAFormatter(PromptFormatter):
    """Simple Q&A formatter for datasets with question/answer structure."""

    def __init__(self, system_prompt: str = "You are a helpful assistant."):
        """Initialize formatter.

        Args:
            system_prompt: System message to prepend to conversations
        """
        self.system_prompt = system_prompt

    def format(self, problem: dict[str, Any]) -> list[dict[str, str]]:
        """Format problem into messages.

        Expected problem keys: 'question' or 'problem'

        Args:
            problem: Dict with 'question' or 'problem' key

        Returns:
            List of message dicts
        """
        question = problem.get("question") or problem.get("problem", "")
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]


class MathFormatter(PromptFormatter):
    """Formatter for math problems with chain-of-thought prompting."""

    def __init__(
        self,
        system_prompt: str = "You are a math expert. Solve the following problem step by step.",
        include_cot_prompt: bool = True,
    ):
        """Initialize formatter.

        Args:
            system_prompt: System message for math problems
            include_cot_prompt: Whether to include chain-of-thought instruction
        """
        self.system_prompt = system_prompt
        self.include_cot_prompt = include_cot_prompt

    def format(self, problem: dict[str, Any]) -> list[dict[str, str]]:
        """Format math problem into messages.

        Expected problem keys: 'problem' or 'question'

        Args:
            problem: Dict with problem text

        Returns:
            List of message dicts
        """
        problem_text = problem.get("problem") or problem.get("question", "")

        if self.include_cot_prompt:
            problem_text += (
                "\n\nPlease solve this step by step, showing your reasoning."
            )

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": problem_text},
        ]


class CustomTemplateFormatter(PromptFormatter):
    """Flexible formatter using string templates."""

    def __init__(
        self,
        system_prompt: str = "You are a helpful assistant.",
        user_template: str = "{question}",
        problem_key: str = "question",
    ):
        """Initialize formatter with custom template.

        Args:
            system_prompt: System message
            user_template: Template for user message, can use any keys from problem dict
            problem_key: Primary key to extract from problem dict
        """
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.problem_key = problem_key

    def format(self, problem: dict[str, Any]) -> list[dict[str, str]]:
        """Format problem using template.

        Args:
            problem: Dict with problem data

        Returns:
            List of message dicts
        """
        # Try to format using all available keys from problem dict
        try:
            user_content = self.user_template.format(**problem)
        except KeyError:
            # Fallback to primary key
            user_content = problem.get(self.problem_key, str(problem))

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]


class FewShotFormatter(PromptFormatter):
    """Formatter that includes few-shot examples."""

    def __init__(
        self,
        system_prompt: str = "You are a helpful assistant.",
        examples: list[dict[str, str]] | None = None,
    ):
        """Initialize formatter with examples.

        Args:
            system_prompt: System message
            examples: List of example dicts with 'question' and 'answer' keys
        """
        self.system_prompt = system_prompt
        self.examples = examples or []

    def format(self, problem: dict[str, Any]) -> list[dict[str, str]]:
        """Format problem with few-shot examples.

        Args:
            problem: Dict with problem data

        Returns:
            List of message dicts including examples
        """
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add examples as user/assistant pairs
        for example in self.examples:
            messages.append({"role": "user", "content": example["question"]})
            messages.append({"role": "assistant", "content": example["answer"]})

        # Add actual problem
        question = problem.get("question") or problem.get("problem", str(problem))
        messages.append({"role": "user", "content": question})

        return messages
