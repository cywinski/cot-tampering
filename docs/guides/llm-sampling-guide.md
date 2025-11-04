# LLM Sampling with Nebius API - User Guide

This guide explains how to use the Nebius LLM sampling client for parallel response generation.

## Overview

The sampling system provides:
- **Parallel sampling**: Generate multiple responses per prompt efficiently
- **Automatic retries**: Handle API failures with configurable retry logic
- **Flexible prompts**: Easy-to-customize prompt formatting for different tasks
- **Dataset integration**: Built-in loaders for common HuggingFace datasets
- **Configuration-based**: YAML configs for reproducible experiments

## Quick Start

### 1. Setup Environment

Add your Nebius API key to `.env`:

```bash
NEBIUS_API_KEY=your-api-key-here
```

### 2. Install Dependencies

```bash
uv sync
```

### 3. Run Sampling

```bash
python src/run_sampling.py --config_path experiments/configs/sampling_config.yaml
```

## Configuration

Configuration is done via YAML files in `experiments/configs/`. Here's the structure:

```yaml
# Model settings
model:
  name: "meta-llama/Meta-Llama-3.1-70B-Instruct"
  temperature: 0.7
  max_tokens: 2048
  top_p: 0.95
  top_k: 40

# Sampling behavior
sampling:
  n_responses: 5        # Responses per prompt
  max_retries: 3        # Retry failed requests
  timeout: 60.0         # Timeout per request

# Dataset to use
dataset:
  name: "math500"       # Dataset from registry
  # Optional dataset parameters:
  # params:
  #   subject: "abstract_algebra"

# Prompt formatting
prompt:
  formatter: "math"     # simple | math | custom | few_shot
  system_prompt: "You are a math expert..."
  include_cot_prompt: true

# Output
output:
  save_dir: "experiments/results"
  # Results saved as individual JSON files per prompt
```

## Available Datasets

Built-in dataset loaders:

- `math500`: MATH-500 dataset (math problems)
- `gsm8k`: GSM8K math reasoning
- `mmlu`: MMLU benchmark (requires `subject` parameter)
- `truthfulqa`: TruthfulQA dataset
- `arc_challenge`: ARC Challenge dataset
- `hellaswag`: HellaSwag commonsense reasoning

### Adding Custom Datasets

Edit `src/dataset_loaders.py`:

```python
def load_my_dataset() -> list[dict[str, Any]]:
    """Load my custom dataset."""
    dataset = load_dataset("org/dataset-name", split="test")
    return [{"question": item["text"], ...} for item in dataset]

# Add to registry
DATASET_REGISTRY["my_dataset"] = load_my_dataset
```

## Prompt Formatters

### SimpleQAFormatter

Basic question-answer format:

```python
formatter = SimpleQAFormatter(
    system_prompt="You are a helpful assistant."
)
```

### MathFormatter

Optimized for math problems with chain-of-thought:

```python
formatter = MathFormatter(
    system_prompt="You are a math expert...",
    include_cot_prompt=True  # Adds step-by-step instruction
)
```

### CustomTemplateFormatter

Flexible template-based formatting:

```python
formatter = CustomTemplateFormatter(
    system_prompt="You are an expert.",
    user_template="Question: {question}\nContext: {context}",
    problem_key="question"
)
```

### FewShotFormatter

Include examples in prompts:

```python
formatter = FewShotFormatter(
    system_prompt="You are a helpful assistant.",
    examples=[
        {"question": "What is 2+2?", "answer": "4"},
        {"question": "What is 3+3?", "answer": "6"},
    ]
)
```

## Programmatic Usage

For Jupyter notebooks or custom scripts:

```python
from nebius_client import NebiusClient, SamplingConfig
from prompt_formatter import MathFormatter
from dataset_loaders import get_dataset

# Load data
problems = get_dataset("math500")[:10]

# Create formatter
formatter = MathFormatter()
prompts = [formatter.format(p) for p in problems]

# Configure client
config = SamplingConfig(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    temperature=0.7,
    n_responses=3,
    max_retries=3,
)

client = NebiusClient(config)

# Option 1: Sample a single prompt (useful for interactive notebooks)
single_prompt = prompts[0]
responses = client.sample_prompt_sync(single_prompt)
for i, resp in enumerate(responses):
    if resp['success']:
        print(f"Response {i+1}: {resp['content']}")

# Option 2: Sample batch of prompts (parallel)
results = client.sample_batch_sync(prompts)

# results[i] contains n_responses responses for prompts[i]
for problem, responses in zip(problems, results):
    print(f"Problem: {problem['problem']}")
    for i, resp in enumerate(responses):
        if resp['success']:
            print(f"Response {i+1}: {resp['content']}")
```

## Output Format

Results are saved as **individual JSON files per prompt** in a dataset-specific subdirectory:

```
experiments/results/
└── math500/
    ├── prompt_0000.json
    ├── prompt_0001.json
    ├── prompt_0002.json
    └── ...
```

Each JSON file contains:

```json
{
  "prompt_index": 0,
  "problem": {
    "problem": "Solve for x: ...",
    "solution": "...",
    "level": "Level 5",
    "type": "Algebra"
  },
  "prompt": [
    {"role": "user", "content": "Solve for x: ..."}
  ],
  "responses": [
    {
      "content": "Let's solve this step by step...",
      "success": true,
      "finish_reason": "stop",
      "usage": {
        "prompt_tokens": 45,
        "completion_tokens": 250,
        "total_tokens": 295
      },
      "model": "Qwen/QwQ-32B"
    },
    ...
  ]
}
```

Each response contains:
- `content`: Generated text
- `success`: Whether generation succeeded
- `finish_reason`: Why generation stopped
- `usage`: Token usage statistics
- `model`: Model used for generation
- `error`: Error message (if failed)

Files are saved **immediately** after all responses for a prompt are sampled, allowing you to monitor progress in real-time.

## Parallel Sampling

The client automatically parallelizes:
1. **Per-prompt parallelism**: All n_responses for a prompt run in parallel
2. **Batch parallelism**: All prompts in a batch are processed concurrently

This significantly speeds up sampling when requesting multiple responses.

## Retry Logic

Failed requests are automatically retried with exponential backoff:
- Retry up to `max_retries` times
- Wait time: 2^retry_count seconds
- Returns error info if all retries fail

## Tips

1. **Start small**: Test with a few prompts first
2. **Monitor costs**: Each response counts toward API usage
3. **Adjust temperature**: Lower for consistency, higher for diversity
4. **Use timeouts**: Prevent hanging on slow requests
5. **Check success rate**: Review logs for failed samples
