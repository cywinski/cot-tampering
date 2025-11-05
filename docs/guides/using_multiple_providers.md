# Using Multiple API Providers

The sampling framework supports both Nebius and OpenRouter APIs through a unified configuration interface. You can switch between providers by simply changing the `provider` field in your config file.

## Implementation Details

- **Nebius Client**: Uses ThreadPoolExecutor for parallel sampling
- **OpenRouter Client**: Uses asyncio for efficient concurrent API calls
- **Interface**: Both clients expose identical synchronous public methods (`sample_prompt`, `sample_batch`)

This means you can use either client without changing any code - they're fully interchangeable.

## Configuration

All configs use the same YAML structure. Add a `provider` field under the `model` section:

```yaml
model:
  provider: "nebius"  # or "openrouter"
  name: "model-name"
  temperature: 0.7
  # ... other params
```

## Provider-Specific Settings

### Nebius
- **API Key**: Set `NEBIUS_API_KEY` in `.env`
- **Supported params**: `temperature`, `max_tokens`, `top_p`
- **Note**: Does not support `top_k`

### OpenRouter
- **API Key**: Set `OPENROUTER_API_KEY` in `.env`
- **Supported params**: All Nebius params plus:
  - `top_k`: Top-k sampling parameter
  - `reasoning`: Enable reasoning mode for reasoning models
  - `logprobs`: Return log probabilities
  - `top_logprobs`: Number of top logprobs per token
  - `site_url`: Optional site URL for rankings
  - `site_name`: Optional site name for rankings

## Example Configs

### Nebius Config
```yaml
model:
  provider: "nebius"
  name: "Qwen/QwQ-32B"
  temperature: 0.6
  max_tokens: 16384
  top_p: 0.95
  top_k: null
```

### OpenRouter Config
```yaml
model:
  provider: "openrouter"
  name: "deepseek/deepseek-r1-distill-qwen-32b"
  temperature: 0.7
  max_tokens: 4096
  top_p: 0.95
  top_k: 50  # Supported on OpenRouter
  reasoning: true  # Enable for reasoning models
  logprobs: false  # Optional: get log probabilities
```

## Running Experiments

Use the same command regardless of provider:

```bash
python src/run_sampling.py experiments/configs/your_config.yaml
```

The script will automatically:
1. Detect the provider from the config
2. Initialize the appropriate client
3. Save responses in the same format

## Output Format

Both providers save responses in identical JSON format:

```json
{
  "prompt_index": 0,
  "problem": {...},
  "prompt": [...],
  "responses": [
    {
      "content": "...",
      "finish_reason": "stop",
      "usage": {
        "prompt_tokens": 100,
        "completion_tokens": 200,
        "total_tokens": 300
      },
      "model": "model-name",
      "success": true
    }
  ]
}
```

This ensures downstream analysis code works the same regardless of which provider was used.

### Reasoning Traces

**Important difference in how providers return reasoning:**

- **Nebius**: Returns reasoning traces and final answer combined in the `content` field
- **OpenRouter**: Returns reasoning in a separate `reasoning` field within the message

**OpenRouter response structure:**
```json
{
  "choices": [{
    "message": {
      "reasoning": "reasoning steps here...",
      "content": "Final answer: 42"
    }
  }]
}
```

**The OpenRouter client automatically handles this:** It detects the `message['reasoning']` field and combines it with `message['content']` to match Nebius format.

#### Wrapping Reasoning in Tags

You can optionally wrap reasoning traces in XML-style tags when saving to files:

```yaml
model:
  reasoning: true  # Enable reasoning mode
  wrap_thinking: true  # Wrap reasoning in tags
  thinking_tag: "think"  # Tag name to use
```

**Output with wrapping enabled:**
```
<think>
reasoning steps here...
</think>

Final answer: 42
```

**Output with wrapping disabled (default):**
```
reasoning steps here...

Final answer: 42
```

**Tag customization:** You can use any tag name (e.g., `"reasoning"`, `"thought"`, `"cot"`) by changing the `thinking_tag` parameter. The tag will be wrapped as `<tag>` and `</tag>`.

This feature is useful for:
- Easier parsing of reasoning vs. answer sections
- Compatibility with systems expecting tagged reasoning
- Clearer visual separation in JSON files
