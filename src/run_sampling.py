# ABOUTME: Production script for running LLM sampling experiments with YAML configuration
# ABOUTME: Handles dataset loading, prompt formatting, parallel sampling, and result saving

import json
from pathlib import Path

import fire
import yaml

from dataset_loaders import get_dataset
from nebius_client import NebiusClient, SamplingConfig
from prompt_formatter import (
    CustomTemplateFormatter,
    FewShotFormatter,
    MathFormatter,
    SimpleQAFormatter,
)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Config dict
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_formatter(config: dict):
    """Create prompt formatter from config.

    Args:
        config: Config dict with prompt settings

    Returns:
        PromptFormatter instance
    """
    formatter_type = config.get("formatter", "simple")
    system_prompt = config.get("system_prompt", "You are a helpful assistant.")

    if formatter_type == "simple":
        return SimpleQAFormatter(system_prompt=system_prompt)
    elif formatter_type == "math":
        return MathFormatter(
            system_prompt=system_prompt,
            include_cot_prompt=config.get("include_cot_prompt", True),
        )
    elif formatter_type == "custom":
        return CustomTemplateFormatter(
            system_prompt=system_prompt,
            user_template=config.get("user_template", "{question}"),
            problem_key=config.get("problem_key", "question"),
        )
    elif formatter_type == "few_shot":
        return FewShotFormatter(
            system_prompt=system_prompt,
            examples=config.get("examples", []),
        )
    else:
        raise ValueError(f"Unknown formatter type: {formatter_type}")


def run_sampling(config_path: str = "experiments/configs/sampling_config.yaml"):
    """Run LLM sampling experiment from config file.

    Args:
        config_path: Path to YAML configuration file
    """
    # Load config
    print(f"Loading config from {config_path}")
    config = load_config(config_path)

    # Load dataset
    dataset_config = config["dataset"]
    dataset_name = dataset_config["name"]
    dataset_params = dataset_config.get("params", {})

    print(f"\nLoading dataset: {dataset_name}")
    problems = get_dataset(dataset_name, **dataset_params)
    print(f"Loaded {len(problems)} problems")

    # Create prompt formatter
    prompt_config = config["prompt"]
    formatter = create_formatter(prompt_config)
    print(f"Using formatter: {prompt_config['formatter']}")

    # Format prompts
    prompts = [formatter.format(problem) for problem in problems]

    # Create client
    model_config = config["model"]
    sampling_config_dict = config["sampling"]

    sampling_config = SamplingConfig(
        model=model_config["name"],
        temperature=model_config["temperature"],
        max_tokens=model_config["max_tokens"],
        top_p=model_config["top_p"],
        top_k=model_config.get("top_k", 40),
        n_responses=sampling_config_dict["n_responses"],
        max_retries=sampling_config_dict["max_retries"],
        timeout=sampling_config_dict["timeout"],
    )

    client = NebiusClient(sampling_config)
    print(f"\nInitialized client with model: {sampling_config.model}")
    print(f"Sampling {sampling_config.n_responses} responses per problem")
    print(f"Total API calls: {len(prompts) * sampling_config.n_responses}")

    # Sample responses
    print("\nStarting parallel sampling...")
    results = client.sample_batch_sync(prompts)

    # Check success rate
    successful_samples = sum(
        1 for problem_results in results for r in problem_results if r["success"]
    )
    total_expected = len(prompts) * sampling_config.n_responses
    print(f"\nSuccessfully sampled: {successful_samples}/{total_expected} responses")
    print(f"Success rate: {successful_samples / total_expected * 100:.1f}%")

    # Save results
    output_config = config["output"]
    output_dir = Path(output_config["save_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    output_format = output_config.get("save_format", "jsonl")
    output_file = output_dir / f"{dataset_name}_samples.{output_format}"

    if output_format == "jsonl":
        with open(output_file, "w") as f:
            for problem, responses in zip(problems, results):
                f.write(
                    json.dumps(
                        {
                            "problem": problem,
                            "responses": responses,
                        }
                    )
                    + "\n"
                )
    else:  # json
        with open(output_file, "w") as f:
            json.dump(
                [
                    {"problem": problem, "responses": responses}
                    for problem, responses in zip(problems, results)
                ],
                f,
                indent=2,
            )

    print(f"\nResults saved to {output_file}")

    # Print sample result
    if results and results[0] and results[0][0]["success"]:
        print("\n" + "=" * 80)
        print("SAMPLE RESULT")
        print("=" * 80)
        print(f"\nProblem:")
        print(problems[0].get("problem") or problems[0].get("question", ""))
        print(f"\nResponse 1:")
        print(results[0][0]["content"][:500] + "...")
        print("=" * 80)


if __name__ == "__main__":
    fire.Fire(run_sampling)
