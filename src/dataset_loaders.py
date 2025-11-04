# ABOUTME: HuggingFace dataset loaders for various prompt datasets
# ABOUTME: Registry-based system for easy extension with new datasets

from typing import Any, Callable

from datasets import load_dataset


def load_math500() -> list[dict[str, Any]]:
    """Load MATH-500 dataset.

    Returns:
        List of problem dicts with 'problem' and 'solution' keys
    """
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return [
        {
            "problem": item["problem"],
            "solution": item["solution"],
            "answer": item["answer"],
        }
        for item in dataset
    ]


def load_gsm8k() -> list[dict[str, Any]]:
    """Load GSM8K math reasoning dataset.

    Returns:
        List of problem dicts with 'question' and 'answer' keys
    """
    dataset = load_dataset("gsm8k", "main", split="test")
    return [
        {
            "question": item["question"],
            "answer": item["answer"],
        }
        for item in dataset
    ]


def load_mmlu(subject: str = "abstract_algebra") -> list[dict[str, Any]]:
    """Load MMLU dataset for a specific subject.

    Args:
        subject: MMLU subject name (e.g., 'abstract_algebra', 'anatomy')

    Returns:
        List of problem dicts with question and choices
    """
    dataset = load_dataset("cais/mmlu", subject, split="test")
    return [
        {
            "question": item["question"],
            "choices": item["choices"],
            "answer": item["answer"],
        }
        for item in dataset
    ]


def load_truthfulqa() -> list[dict[str, Any]]:
    """Load TruthfulQA dataset.

    Returns:
        List of problem dicts with questions and correct/incorrect answers
    """
    dataset = load_dataset("truthful_qa", "generation", split="validation")
    return [
        {
            "question": item["question"],
            "best_answer": item["best_answer"],
            "correct_answers": item.get("correct_answers", []),
            "incorrect_answers": item.get("incorrect_answers", []),
        }
        for item in dataset
    ]


def load_arc_challenge() -> list[dict[str, Any]]:
    """Load ARC Challenge dataset.

    Returns:
        List of problem dicts with questions and multiple choice answers
    """
    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    return [
        {
            "question": item["question"],
            "choices": {
                "text": item["choices"]["text"],
                "label": item["choices"]["label"],
            },
            "answerKey": item.get("answerKey", ""),
        }
        for item in dataset
    ]


def load_hellaswag() -> list[dict[str, Any]]:
    """Load HellaSwag commonsense reasoning dataset.

    Returns:
        List of problem dicts with context and continuations
    """
    dataset = load_dataset("Rowan/hellaswag", split="validation")
    return [
        {
            "context": item["ctx"],
            "endings": item["endings"],
            "label": item.get("label", ""),
        }
        for item in dataset
    ]


# Dataset registry: maps dataset names to loader functions
DATASET_REGISTRY: dict[str, Callable[[], list[dict[str, Any]]]] = {
    "math500": load_math500,
    "gsm8k": load_gsm8k,
    "mmlu": load_mmlu,
    "truthfulqa": load_truthfulqa,
    "arc_challenge": load_arc_challenge,
    "hellaswag": load_hellaswag,
}


def get_dataset(name: str, **kwargs) -> list[dict[str, Any]]:
    """Load a dataset by name.

    Args:
        name: Dataset name from DATASET_REGISTRY
        **kwargs: Additional arguments to pass to loader function

    Returns:
        List of problem dicts

    Raises:
        ValueError: If dataset name not in registry
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Dataset '{name}' not found. Available datasets: {list(DATASET_REGISTRY.keys())}"
        )

    loader_func = DATASET_REGISTRY[name]

    # Check if loader accepts kwargs
    import inspect

    sig = inspect.signature(loader_func)
    if len(sig.parameters) > 0:
        return loader_func(**kwargs)
    else:
        return loader_func()


def list_available_datasets() -> list[str]:
    """List all available dataset names.

    Returns:
        List of dataset names
    """
    return list(DATASET_REGISTRY.keys())
