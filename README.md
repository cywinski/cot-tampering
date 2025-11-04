# AI Safety Research Template

## The Workflow

**Specification-driven experiments in 2 steps**:

1. **Write a spec** describing what you want to test (`specs/experiment_template.md`)
2. **Give it to Claude** → Get complete, runnable code + config

That's it. Simple and reproducible.

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
