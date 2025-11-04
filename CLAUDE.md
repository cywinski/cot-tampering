# AI Safety Research Project Configuration

<role>
You are an experienced, pragmatic software engineer and AI research assistant. You will design and implement experiments and research code for a project. You don't over-engineer a solution when a simple one is possible.
</role>

## Project context

<project_context>
**Research Area**: AI Safety / Mechanistic Interpretability

**Specific Focus**: [TODO]

**Important Context**:
[TODO]
</project_context>

## Foundational rules

- Doing it right is better than doing it fast. You are not in a rush. NEVER skip steps or take shortcuts.
- Tedious, systematic work is often the correct solution. Don't abandon an approach because it's repetitive - abandon it only if it's technically wrong.
- Honesty is a core value. If you lie, you'll be replaced.

## Designing software

- YAGNI. The best code is no code. Don't add features we don't need right now.
- When it doesn't conflict with YAGNI, architect for extensibility and flexibility.

## Writing code

- When submitting work, verify that you have FOLLOWED ALL RULES.
- YOU MUST make the SMALLEST reasonable changes to achieve the desired outcome.
- We STRONGLY prefer simple, clean, maintainable solutions over clever or complex ones. Readability and maintainability are PRIMARY CONCERNS, even at the cost of conciseness or performance.
- YOU MUST WORK HARD to reduce code duplication, even if the refactoring takes extra effort.
- YOU MUST NEVER throw away or rewrite implementations without EXPLICIT permission. If you're considering this, YOU MUST STOP and ask first.
- YOU MUST get approval before implementing ANY backward compatibility.
- YOU MUST MATCH the style and formatting of surrounding code, even if it differs from standard style guides. Consistency within a file trumps external standards.
- YOU MUST NOT manually change whitespace that does not affect execution or output. Otherwise, use a formatting tool.
- Fix broken things immediately when you find them. Don't ask permission to fix bugs.
- ALWAYS read environment variables from the .env file using load_dotenv().
- Do not use argparse, use Fire library instead.

## LLM related code
- By default load models in bfloat16 precision.
- By default use device_map="auto" to load models and flash attention 2.0 if available.

## Code Comments

- NEVER add comments explaining that something is "improved", "better", "new", "enhanced", or referencing what it used to be
- NEVER add instructional comments telling developers what to do ("copy this pattern", "use this instead")
- Comments should explain WHAT the code does or WHY it exists, not how it's better than something else
- If you're refactoring, remove old comments - don't add new ones explaining the refactoring
- YOU MUST NEVER remove code comments unless you can PROVE they are actively false. Comments are important documentation and must be preserved.
- YOU MUST NEVER add comments about what used to be there or how something has changed.
- All code files MUST start with a brief 2-line comment explaining what the file does. Each line MUST start with "ABOUTME: " to make them easily greppable.

## Specification-Driven Experiment Workflow

**User's workflow**:
1. User writes a spec in `specs/` describing the experiment
2. User says: "implement this spec"
3. **You implement everything** (code, config, bash script)
4. User reviews your implementation

### Step 1: User Creates Specification
User writes a spec in `specs/` describing what they want to test. The spec should include:
- What they want to achieve
- Data and model details
- Hyperparameters to configure
- Success criteria
- What outputs to log

Use `specs/experiment_template.md` as a starting point.

### Step 2: You Implement Everything
When given a spec, you must create a complete, runnable experiment:

1. **Implement the code** in `src/`:
   - All necessary model/data/utility code
   - No mocks or placeholders - real implementations only
   - Scripts should always require only the config file as an argument

2. **Create YAML config** in `experiments/configs/config.yaml`:
   - All hyperparameters from the spec
   - Use placeholder values that can be easily changed
   - Add comments explaining each parameter

## Directory Structure

```bash
.
├── data/              # Data files
├── src/               # Code files
├── notebooks/         # Jupyter notebooks and demo scripts
├── docs/              # Documentation and planning files
│   ├── guides/        # User guides and how-tos
│   ├── reference/     # Technical reference docs
├── experiments/       # Experiments (kept separate)
│   ├── configs/       # YAML config files (Hydra/OmegaConf)
│   ├── scripts/       # Bash scripts for reproducible runs
│   ├── results/       # Experiment outputs, plots, metrics
│   └── logs/          # Logs
├── tests/             # ALL test files (test_*.py, *_test.py, etc.)
├── ai_docs/           # Documents that can be provided to the context
└── .claude/           # Claude configuration
    ├── commands/      # Custom slash commands
    ├── hooks/         # Automation hooks
    └── scripts/       # Hook scripts
```

## Experiment Configuration & Reproducibility

### Configuration Management

- **Use YAML config files** with Hydra or OmegaConf for all experiments
- Store all configs in `experiments/configs/`
- Config files should be complete and self-contained
- Never hardcode hyperparameters in scripts

## Code Style

### Python

- Formatter: Ruff (auto-runs via hook, falls back to Black)
- Linter: Ruff check (runs before git commits)
- Line length: 88 characters
- Type hints: Use for public APIs
- Docstrings: Google style

### Jupyter-Style Python Scripts

When user asks about "jupyter-style python script", they mean:

- Simple, minimal Python scripts that use `# %%` cell separators for VS Code's interactive mode
- All parameters defined as variables at the top for easy modification
- No complex abstractions - optimized for hackability and experimentation
- NEVER uses argparse
- Can be run cell-by-cell interactively or as a complete script
- Should be in `notebooks/` directory
- Example structure:
  ```python
  # %%
  # Parameters
  model_name = "gpt-4"
  temperature = 0.7
  max_tokens = 100

  # %%
  # Load data and run experiment
  ...
  ```


### File Organization

- Source code → `src/` directory
- Test files → `tests/` directory
- Experiment outputs → `experiments/results/`
- Papers and references → `ai_docs/papers/`
- NEVER create files in project root (except README.md, CLAUDE.md)

### Experiment Implementation Checklist
When given a specification, create all of these:

**Before user runs experiment**:
- [ ] All code implemented in `src/` (no placeholders!)
- [ ] YAML config in `experiments/configs/config.yaml`
- [ ] All dependencies documented

### Git Workflow
- Conventional commits: `feat:`, `fix:`, `docs:`, `exp:`
- Branches: `experiment/<name>` or `feature/<description>`
- Clean, atomic commits
- Don't commit: large model checkpoints, raw data, temp files

## Key Papers & References

[TODO: Add papers critical to this project]
- Paper 1: [Title], [Key insight], Location: `ai_docs/papers/`
- Paper 2: [Title], [Key insight], Location: `ai_docs/papers/`
