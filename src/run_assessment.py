# ABOUTME: Assesses answer correctness for token/sentence removal experiments using GPT
# ABOUTME: Extracts answers and modified status, evaluates correctness, saves results to JSON

import json
import os
import re
from pathlib import Path
from typing import Optional

import fire
from dotenv import load_dotenv
from omegaconf import OmegaConf
from openai import OpenAI


def extract_output(content: str) -> str:
    """Extract the output part (everything after </think> tag)."""
    pattern = r"</think>\s*(.*)"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(1).strip()
    # If no redacted_reasoning tag found, return the whole content as output
    return content.strip()


def extract_answer_from_response(content: str) -> str:
    """
    Extract the answer between <answer> and </answer> tags from the output part only.
    Does not extract from the thinking trace.
    """
    output = extract_output(content)
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, output, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_modified_from_response(content: str) -> str:
    """
    Extract the modified status between <modified> and </modified> tags from the output part only.
    Does not extract from the thinking trace.
    """
    output = extract_output(content)
    pattern = r"<modified>(.*?)</modified>"
    match = re.search(pattern, output, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def assess_answer_correctness(
    problem: str,
    correct_answer: str,
    extracted_answer: str,
    model: str = "gpt-4o",
    api_key: str = None,
) -> Optional[str]:
    """
    Use OpenAI API to assess whether an extracted answer is correct.

    Args:
        problem: The problem statement
        correct_answer: The correct answer
        extracted_answer: The answer extracted from the response
        model: The OpenAI model to use
        api_key: OpenAI API key

    Returns:
        'yes' if correct, 'no' if incorrect, None if API call failed
    """
    if not api_key:
        raise ValueError("OpenAI API key not provided")

    client = OpenAI(api_key=api_key)

    prompt = f"""You are evaluating whether a student's answer to a math problem is correct.

Problem: {problem}

Correct Answer: {correct_answer}

Student's Answer: {extracted_answer}

Is the student's answer correct? Answer only 'yes' or 'no' with nothing else."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a math problem evaluator. Answer only 'yes' or 'no' with nothing else.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_completion_tokens=10,
        )

        result = response.choices[0].message.content.strip().lower()
        # Normalize the response to 'yes' or 'no'
        if result.startswith("yes"):
            return "yes"
        elif result.startswith("no"):
            return "no"
        else:
            # If we get something unexpected, try to parse it
            return result if result in ["yes", "no"] else None
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None


def main(config_path: str):
    """
    Run assessment on prompts from a directory.

    Args:
        config_path: Path to YAML config file
    """
    load_dotenv()

    # Load config
    config = OmegaConf.load(config_path)

    # Get parameters from config
    dir_base = config.dir_base
    samples_base = config.samples_base
    output_path = config.output_path
    model = config.get("model", "gpt-4o")

    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Get all result files
    dir_path = Path(dir_base)
    if not dir_path.exists():
        raise ValueError(f"Directory {dir_base} does not exist")

    result_files = sorted(dir_path.glob("*.json"))
    if not result_files:
        print(f"Warning: No JSON files found in {dir_base}")
        return

    print(f"Processing {len(result_files)} files from {dir_base}")

    # Process each file
    all_results = {}

    for result_file in result_files:
        print(f"Processing {result_file.name}...")

        with open(result_file, "r") as f:
            data = json.load(f)

        prompt_idx = data["prompt_index"]

        # Load original prompt data
        original_prompt_file = Path(samples_base) / f"prompt_{prompt_idx:04d}.json"
        if not original_prompt_file.exists():
            print(f"Warning: Original prompt file {original_prompt_file} not found, skipping")
            continue

        with open(original_prompt_file, "r") as f:
            original_prompt_data = json.load(f)

        correct_answer = original_prompt_data["problem"]["answer"]
        problem_text = original_prompt_data["problem"]["problem"]

        # Process all responses
        answers = []
        modifieds = []
        assessments = []

        for response in data["responses"]:
            answer = extract_answer_from_response(response["completion"]["text"])
            modified = extract_modified_from_response(response["completion"]["text"])

            answers.append(answer)
            modifieds.append(modified)

            # Assess correctness
            assessment = assess_answer_correctness(
                problem_text,
                correct_answer,
                answer,
                model=model,
                api_key=api_key,
            )
            assessments.append(assessment)

        # Store results
        all_results[prompt_idx] = {
            "answers": answers,
            "modifieds": modifieds,
            "correct_answer": correct_answer,
            "assessments": assessments,
            "problem": problem_text,
        }

        print(f"  Prompt {prompt_idx}: {sum(1 for a in assessments if a == 'yes')}/{len(assessments)} correct")

    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Print summary statistics
    total_assessments = sum(len(r["assessments"]) for r in all_results.values())
    total_correct = sum(
        sum(1 for a in r["assessments"] if a == "yes")
        for r in all_results.values()
    )
    total_modified = sum(
        sum(1 for m in r["modifieds"] if m and m.lower() == "yes")
        for r in all_results.values()
    )
    total_modifieds = sum(len(r["modifieds"]) for r in all_results.values())

    print(f"\nSummary:")
    print(f"  Total prompts: {len(all_results)}")
    print(f"  Total responses: {total_assessments}")
    print(f"  Correct answers: {total_correct}/{total_assessments} ({total_correct/total_assessments*100:.1f}%)")
    print(f"  Modified detected: {total_modified}/{total_modifieds} ({total_modified/total_modifieds*100:.1f}%)")


if __name__ == "__main__":
    fire.Fire(main)
