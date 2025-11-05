# %%
# Test MMLU formatting
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / "src"))

from dataset_loaders import load_mmlu
from prompt_formatter import PromptFormatter

# %%
# Load a sample from MMLU
problems = load_mmlu(subject="marketing")
print(f"Loaded {len(problems)} problems")
print("\nFirst problem:")
print(problems[0])

# %%
# Test formatting
formatter = PromptFormatter(template_name="mmlu_multiple_choice", field_name="question")
formatted = formatter.format(problems[0])
print("\nFormatted prompt:")
print(formatted[0]["content"])

# %%
