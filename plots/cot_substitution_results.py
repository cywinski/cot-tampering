#!/usr/bin/env python3
"""
Extract answer and substituted values from COT substitution experiment results.
Creates a bar plot showing results per model.
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def letter_to_index(letter: str) -> Optional[int]:
    """Convert answer letter (A/B/C/D) to index (0/1/2/3)."""
    letter = letter.strip().upper()
    if letter == "A":
        return 0
    elif letter == "B":
        return 1
    elif letter == "C":
        return 2
    elif letter == "D":
        return 3
    return None


def extract_answer_and_substituted(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract answer and substituted values from text.

    Args:
        text: Text containing <answer> and <substituted> tags

    Returns:
        Tuple of (answer, substituted) where answer is A/B/C/D and substituted is yes/no
    """
    # Extract answer - find all matches and use the last valid one
    # (sometimes there are multiple answer tags, with the first being empty/malformed)
    # Look for answer tags that contain only a single letter (A, B, C, or D)
    answer_pattern = r"<answer>([A-Da-d])</answer>"
    answer_matches = re.findall(answer_pattern, text)
    answer = None
    if answer_matches:
        # Use the last match (most likely to be the correct one)
        answer = answer_matches[-1].upper()
    else:
        # Fallback: look for standalone letter at the end of text (after </think>)
        # This handles cases where the answer is just "D" without tags
        # Only extract if it appears immediately after </think> with minimal text
        after_reasoning = re.search(r"</think>\s*(.*?)$", text, re.DOTALL)
        if after_reasoning:
            remaining_text = after_reasoning.group(1).strip()
            # Look for a standalone letter (A, B, C, or D) with optional whitespace/newlines
            # But only if the remaining text is very short (just the letter, maybe with some trailing text)
            # This prevents matching letters in the middle of sentences
            if len(remaining_text) <= 10:  # Only if very short remaining text
                standalone_match = re.search(r"^([A-Da-d])\s*$", remaining_text)
                if standalone_match:
                    answer = standalone_match.group(1).upper()
                else:
                    # Check if it ends with just a letter (possibly after some text)
                    end_letter_match = re.search(
                        r"([A-Da-d])\s*$", remaining_text.strip()
                    )
                    if end_letter_match:
                        # Make sure there's a clear separator before it (newline or space)
                        letter_pos = remaining_text.rfind(end_letter_match.group(1))
                        if letter_pos > 0:
                            before_letter = remaining_text[:letter_pos].strip()
                            # Only extract if there's a clear break (newline or end of sentence)
                            if not before_letter or before_letter[-1] in [
                                "\n",
                                ".",
                                "!",
                                "?",
                            ]:
                                answer = end_letter_match.group(1).upper()

    # Extract substituted
    substituted_pattern = r"<substituted>(.*?)</substituted>"
    substituted_match = re.search(substituted_pattern, text, re.DOTALL)
    substituted = (
        substituted_match.group(1).strip().lower() if substituted_match else None
    )

    return answer, substituted


def process_cot_substitution_directory(model_dir: Path) -> Dict[str, float]:
    """
    Process all JSON files in a COT substitution model directory and calculate accuracy.

    Args:
        model_dir: Path to directory containing JSON result files from COT substitution

    Returns:
        Dictionary with accuracy metrics: {
            'total': total files processed,
            'correct': count of correct answers,
            'accuracy': accuracy rate (0.0 to 1.0)
        }
    """
    results = {"total": 0, "correct": 0, "accuracy": 0.0}

    json_files = sorted(model_dir.glob("prompt_*.json"))

    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            # Get correct answer from problem
            problem = data.get("problem", {})
            correct_answer_idx = problem.get("answer")
            if correct_answer_idx is None:
                continue

            # Extract answer from response
            text = None
            if "completion" in data:
                completion = data.get("completion", {})
                text = completion.get("text", "")
            elif "responses" in data:
                responses = data.get("responses", [])
                if responses:
                    text = responses[0].get("content", "")

            if not text:
                continue

            answer, _ = extract_answer_and_substituted(text)

            if answer:
                results["total"] += 1
                answer_idx = letter_to_index(answer)
                if answer_idx == correct_answer_idx:
                    results["correct"] += 1

        except Exception as e:
            print(f"Warning: Error processing {json_file.name}: {e}")
            continue

    # Calculate accuracy
    if results["total"] > 0:
        results["accuracy"] = results["correct"] / results["total"]

    return results


def calculate_substituted_accuracy(
    results_dir: Path, control_model_name: str
) -> Dict[str, float]:
    """
    Calculate substituted detection accuracy for the model being tested.

    Args:
        results_dir: Path to directory containing model subdirectories
        control_model_name: Name of the model being tested (control trials are in this directory)

    Returns:
        Dictionary with accuracy metrics: {
            'total': total files processed,
            'correct': count of correct substitution detections,
            'accuracy': accuracy rate (0.0 to 1.0)
        }
    """
    results = {"total": 0, "correct": 0, "accuracy": 0.0}

    # Get all model directories
    model_dirs = [d for d in results_dir.iterdir() if d.is_dir()]

    for model_dir in model_dirs:
        model_name = model_dir.name
        # Determine ground truth: control (no substitution) or substituted (yes)
        is_substituted = model_name != control_model_name

        json_files = sorted(model_dir.glob("prompt_*.json"))

        for json_file in json_files:
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)

                # Extract substituted value from response
                text = None
                if "completion" in data:
                    completion = data.get("completion", {})
                    text = completion.get("text", "")
                elif "responses" in data:
                    responses = data.get("responses", [])
                    if responses:
                        text = responses[0].get("content", "")

                if not text:
                    continue

                _, substituted = extract_answer_and_substituted(text)

                if substituted:
                    results["total"] += 1
                    # Check if prediction matches ground truth
                    predicted_substituted = substituted == "yes"
                    if predicted_substituted == is_substituted:
                        results["correct"] += 1

            except Exception as e:
                print(f"Warning: Error processing {json_file.name}: {e}")
                continue

    # Calculate accuracy
    if results["total"] > 0:
        results["accuracy"] = results["correct"] / results["total"]

    return results


def process_samples_base_directory(samples_dir: Path) -> Dict[str, float]:
    """
    Process all JSON files in a samples/base directory and calculate accuracy.

    Args:
        samples_dir: Path to samples/base directory containing JSON result files

    Returns:
        Dictionary with accuracy metrics: {
            'total': total files processed,
            'correct': count of correct answers,
            'accuracy': accuracy rate (0.0 to 1.0)
        }
    """
    results = {"total": 0, "correct": 0, "accuracy": 0.0}

    json_files = sorted(samples_dir.glob("prompt_*.json"))

    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            # Get correct answer from problem
            problem = data.get("problem", {})
            correct_answer_idx = problem.get("answer")
            if correct_answer_idx is None:
                continue

            # Extract answer from response
            text = None
            if "completion" in data:
                completion = data.get("completion", {})
                text = completion.get("text", "")
            elif "responses" in data:
                responses = data.get("responses", [])
                if responses:
                    text = responses[0].get("content", "")

            if not text:
                continue

            answer, _ = extract_answer_and_substituted(text)

            if answer:
                results["total"] += 1
                answer_idx = letter_to_index(answer)
                if answer_idx == correct_answer_idx:
                    results["correct"] += 1

        except Exception as e:
            print(f"Warning: Error processing {json_file.name}: {e}")
            continue

    # Calculate accuracy
    if results["total"] > 0:
        results["accuracy"] = results["correct"] / results["total"]

    return results


def process_model_directory(model_dir: Path) -> Dict[str, int]:
    """
    Process all JSON files in a model directory.

    Args:
        model_dir: Path to directory containing JSON result files

    Returns:
        Dictionary with counts: {
            'total': total files processed,
            'yes': count of 'yes' for substituted,
            'no': count of 'no' for substituted,
            'answers': dict of answer counts (A, B, C, D)
        }
    """
    results = {"total": 0, "yes": 0, "no": 0, "answers": defaultdict(int)}

    json_files = sorted(model_dir.glob("prompt_*.json"))

    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            # Handle different JSON structures
            text = None
            if "completion" in data:
                # Format with completion.text
                completion = data.get("completion", {})
                text = completion.get("text", "")
            elif "responses" in data:
                # Format with responses array
                responses = data.get("responses", [])
                if responses:
                    text = responses[0].get("content", "")

            if not text:
                continue

            answer, substituted = extract_answer_and_substituted(text)

            if answer:
                results["answers"][answer.upper()] += 1

            if substituted:
                results["total"] += 1
                if substituted == "yes":
                    results["yes"] += 1
                elif substituted == "no":
                    results["no"] += 1

        except Exception as e:
            print(f"Warning: Error processing {json_file.name}: {e}")
            continue

    return results


def create_bar_plot(results_dir: str, output_path: str = None):
    """
    Create a bar plot showing substituted detection results per model.

    Args:
        results_dir: Path to directory containing model subdirectories
        output_path: Optional path to save the plot
    """
    results_dir_path = Path(results_dir)
    if not results_dir_path.exists():
        raise ValueError(f"Directory does not exist: {results_dir}")

    # Get all model directories
    model_dirs = [d for d in results_dir_path.iterdir() if d.is_dir()]
    model_dirs.sort()

    if not model_dirs:
        raise ValueError(f"No model directories found in {results_dir}")

    # Process each model directory
    model_results = {}
    accuracy_results = {}

    # Find the base results directory (e.g., experiments/results/)
    # Assuming results_dir is something like experiments/results/deepseek-r1/mmlu_marketing/other_model_cot_substitution
    # We need to go up to experiments/results/ to find model directories
    base_results_path = (
        results_dir_path.parent.parent.parent
    )  # Go up to experiments/results/

    for model_dir in model_dirs:
        model_name = model_dir.name
        print(f"\nProcessing {model_name}...")
        results = process_model_directory(model_dir)
        model_results[model_name] = results
        print(f"  Total: {results['total']}")
        print(f"  Yes: {results['yes']}")
        print(f"  No: {results['no']}")
        print(f"  Answers: {dict(results['answers'])}")

        # Try to find corresponding samples/base directory
        # The model name in the COT substitution dir should match a model dir in results/
        samples_base_path = (
            base_results_path / model_name / "mmlu_marketing" / "samples" / "base"
        )
        if samples_base_path.exists():
            print(f"  Processing accuracy from {samples_base_path}...")
            acc_results = process_samples_base_directory(samples_base_path)
            accuracy_results[model_name] = acc_results
            print(
                f"  Accuracy: {acc_results['accuracy']:.3f} ({acc_results['correct']}/{acc_results['total']})"
            )
        else:
            # Try alternative: maybe the model is in a different location
            # For "base", we might need special handling
            if model_name == "base":
                # Try to find if there's a parent model directory
                parent_model = results_dir_path.parent.name
                samples_base_path = (
                    base_results_path
                    / parent_model
                    / "mmlu_marketing"
                    / "samples"
                    / "base"
                )
                if samples_base_path.exists():
                    print(
                        f"  Processing accuracy from {samples_base_path} (mapped from 'base')..."
                    )
                    acc_results = process_samples_base_directory(samples_base_path)
                    accuracy_results[model_name] = acc_results
                    print(
                        f"  Accuracy: {acc_results['accuracy']:.3f} ({acc_results['correct']}/{acc_results['total']})"
                    )
                else:
                    print("  Warning: Samples directory not found for 'base' model")
                    accuracy_results[model_name] = {
                        "total": 0,
                        "correct": 0,
                        "accuracy": 0.0,
                    }
            else:
                print(f"  Warning: Samples directory not found at {samples_base_path}")
                accuracy_results[model_name] = {
                    "total": 0,
                    "correct": 0,
                    "accuracy": 0.0,
                }

    # Process each model directory for COT substitution accuracy
    cot_substitution_accuracy_results = {}
    for model_dir in model_dirs:
        model_name = model_dir.name
        print(f"  Calculating COT substitution accuracy for {model_name}...")
        acc_results = process_cot_substitution_directory(model_dir)
        cot_substitution_accuracy_results[model_name] = acc_results
        print(
            f"    Accuracy: {acc_results['accuracy']:.3f} ({acc_results['correct']}/{acc_results['total']})"
        )

    # Determine the model being tested (control model)
    # This is typically the parent directory name (e.g., "deepseek-r1")
    # Path structure: experiments/results/deepseek-r1/mmlu_marketing/other_model_cot_substitution
    control_model_name = results_dir_path.parent.parent.name

    # Calculate substituted accuracy for the model being tested
    print(f"\nCalculating substituted accuracy for {control_model_name}...")
    substituted_acc = calculate_substituted_accuracy(
        results_dir_path, control_model_name
    )
    print(
        f"  Substituted accuracy: {substituted_acc['accuracy']:.3f} ({substituted_acc['correct']}/{substituted_acc['total']})"
    )

    # Create the plot with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # Prepare data
    model_names = list(model_results.keys())
    yes_counts = [model_results[m]["yes"] for m in model_names]
    no_counts = [model_results[m]["no"] for m in model_names]

    # Plot 1: Substituted detection (yes vs no)
    x = np.arange(len(model_names))
    width = 0.35

    bars1 = ax1.bar(
        x - width / 2,
        yes_counts,
        width,
        label="Substituted (yes)",
        color="#ff7f0e",
        edgecolor="black",
        linewidth=1.5,
        alpha=0.8,
    )
    bars2 = ax1.bar(
        x + width / 2,
        no_counts,
        width,
        label="Not Substituted (no)",
        color="#2ca02c",
        edgecolor="black",
        linewidth=1.5,
        alpha=0.8,
    )

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{int(height)}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

    ax1.set_xlabel("Model", fontsize=14)
    ax1.set_ylabel("Count", fontsize=14)
    ax1.set_title("Substitution Detection by Model", fontsize=16, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha="right", fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    # Plot 2: Answer accuracy
    accuracies = []
    for model in model_names:
        if model in accuracy_results:
            accuracies.append(accuracy_results[model]["accuracy"] * 100)
        else:
            accuracies.append(0.0)

    bars3 = ax2.bar(
        x,
        accuracies,
        width=0.6,
        color="#d62728",
        edgecolor="black",
        linewidth=1.5,
        alpha=0.8,
    )

    # Add value labels on bars
    for bar in bars3:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax2.set_xlabel("Model", fontsize=14)
    ax2.set_ylabel("Accuracy (%)", fontsize=14)
    ax2.set_title("Answer Accuracy by Model", fontsize=16, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=45, ha="right", fontsize=12)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    if accuracies and max(accuracies) > 0:
        ax2.set_ylim([0, max(accuracies) * 1.15])
    else:
        ax2.set_ylim([0, 100])

    # Plot 3: COT substitution accuracy (from other_model_cot_substitution directories)
    cot_accuracies = []
    for model in model_names:
        if model in cot_substitution_accuracy_results:
            cot_accuracies.append(
                cot_substitution_accuracy_results[model]["accuracy"] * 100
            )
        else:
            cot_accuracies.append(0.0)

    bars4 = ax3.bar(
        x,
        cot_accuracies,
        width=0.6,
        color="#9467bd",
        edgecolor="black",
        linewidth=1.5,
        alpha=0.8,
    )

    # Add value labels on bars
    for bar in bars4:
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax3.set_xlabel("Model", fontsize=14)
    ax3.set_ylabel("Accuracy (%)", fontsize=14)
    ax3.set_title("Answer Accuracy (COT Substitution)", fontsize=16, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(model_names, rotation=45, ha="right", fontsize=12)
    ax3.grid(axis="y", alpha=0.3, linestyle="--")
    if cot_accuracies and max(cot_accuracies) > 0:
        ax3.set_ylim([0, max(cot_accuracies) * 1.15])
    else:
        ax3.set_ylim([0, 100])

    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"\nPlot saved to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python cot_substitution_results.py <results_dir> [output_path]")
        print("\nExample:")
        print(
            "  python cot_substitution_results.py experiments/results/deepseek-r1/mmlu_marketing/other_model_cot_substitution plots/cot_substitution_results.png"
        )
        sys.exit(1)

    results_dir = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    create_bar_plot(results_dir, output_path)
