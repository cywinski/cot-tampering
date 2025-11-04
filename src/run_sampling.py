# ABOUTME: Production script for running LLM sampling experiments with YAML configuration
# ABOUTME: Handles dataset loading, prompt formatting, parallel sampling, and result saving

import asyncio
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
    RawFormatter,
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

    if formatter_type == "raw":
        return RawFormatter(field_name=config.get("field_name", "problem"))
    elif formatter_type == "simple":
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


async def sample_and_save_prompt(
    client: NebiusClient,
    prompt_idx: int,
    prompt: list[dict[str, str]],
    problem: dict,
    output_dir: Path,
) -> dict:
    """Sample responses for a single prompt and save immediately.

    Args:
        client: NebiusClient instance
        prompt_idx: Index of the prompt (for filename)
        prompt: Formatted prompt messages
        problem: Original problem dict
        output_dir: Directory to save results

    Returns:
        Dict with success statistics
    """
    # Sample responses for this prompt
    responses = await client.sample_prompt(prompt)

    # Save immediately to individual file
    output_file = output_dir / f"prompt_{prompt_idx:04d}.json"
    result_data = {
        "prompt_index": prompt_idx,
        "problem": problem,
        "prompt": prompt,
        "responses": responses,
    }

    with open(output_file, "w") as f:
        json.dump(result_data, f, indent=2)

    # Calculate success stats
    successful = sum(1 for r in responses if r.get("success", False))
    total = len(responses)

    print(
        f"Prompt {prompt_idx:04d}: {successful}/{total} responses saved to {output_file.name}"
    )

    return {"successful": successful, "total": total, "responses": responses}


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
        top_k=model_config.get("top_k"),
        n_responses=sampling_config_dict["n_responses"],
        max_retries=sampling_config_dict["max_retries"],
        timeout=sampling_config_dict["timeout"],
    )

    client = NebiusClient(sampling_config)
    print(f"\nInitialized client with model: {sampling_config.model}")
    print(f"Sampling {sampling_config.n_responses} responses per problem")
    print(f"Total API calls: {len(prompts) * sampling_config.n_responses}")

    # Setup output directory
    output_config = config["output"]
    output_dir = Path(output_config["save_dir"]) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving results to: {output_dir}/")

    # Sample responses with parallel execution and immediate saving
    print("\nStarting parallel sampling (saving each prompt as completed)...")

    async def sample_all():
        tasks = [
            sample_and_save_prompt(client, idx, prompt, problem, output_dir)
            for idx, (prompt, problem) in enumerate(zip(prompts, problems))
        ]
        return await asyncio.gather(*tasks)

    results = asyncio.run(sample_all())

    # Calculate overall success rate
    total_successful = sum(r["successful"] for r in results)
    total_expected = len(prompts) * sampling_config.n_responses
    print(f"\n{'=' * 80}")
    print("SAMPLING COMPLETE")
    print("=" * 80)
    print(f"Total prompts processed: {len(prompts)}")
    print(f"Successfully sampled: {total_successful}/{total_expected} responses")
    print(f"Success rate: {total_successful / total_expected * 100:.1f}%")
    print(f"Results saved to: {output_dir}/")

    # Print sample result
    if results and results[0]["responses"] and results[0]["responses"][0]["success"]:
        print(f"\n{'=' * 80}")
        print("SAMPLE RESULT (Prompt 0)")
        print(f"{'=' * 80}")
        print("\nProblem:")
        print(problems[0].get("problem") or problems[0].get("question", ""))
        print("\nResponse 1:")
        print(results[0]["responses"][0]["content"][:500] + "...")
        print(f"{'=' * 80}")


if __name__ == "__main__":
    fire.Fire(run_sampling)
