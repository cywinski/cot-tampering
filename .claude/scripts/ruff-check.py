#!/usr/bin/env python3
# ABOUTME: Runs ruff linter on Python files before git commits
# ABOUTME: Provides feedback on code quality issues that should be fixed

import json
import sys
import subprocess
import os
from pathlib import Path


def main():
    try:
        # Read input from Claude Code
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError:
        # Not JSON input, exit quietly
        sys.exit(0)

    # Extract tool name
    tool_name = input_data.get("tool_name", "")

    # Only run for Bash tool when running git commit
    if tool_name != "Bash":
        sys.exit(0)

    # Get the command being run
    tool_input = input_data.get("tool_input", {})
    command = tool_input.get("command", "")

    # Check if this is a git commit command
    if "git commit" not in command:
        sys.exit(0)

    # Get list of staged Python files
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            # Not a git repository or git not available
            sys.exit(0)

        staged_files = [
            f for f in result.stdout.strip().split("\n") if f.endswith(".py")
        ]

        if not staged_files:
            # No Python files staged
            sys.exit(0)

    except FileNotFoundError:
        # git not installed
        sys.exit(0)

    # Run ruff check on staged files
    try:
        result = subprocess.run(
            ["ruff", "check"] + staged_files, capture_output=True, text=True
        )

        if result.returncode != 0:
            # Ruff found issues
            print("⚠️  Ruff linting issues found:", file=sys.stderr)
            print(result.stdout, file=sys.stderr)
            print("\nPlease fix these issues before committing.", file=sys.stderr)
            print(
                "Run: ruff check --fix <file> to auto-fix some issues", file=sys.stderr
            )
            # Don't block the commit, just warn
            sys.exit(0)
        else:
            # All good
            print(f"✓ Ruff check passed for {len(staged_files)} Python file(s)")
            sys.exit(0)

    except FileNotFoundError:
        # Ruff not installed, exit quietly
        sys.exit(0)


if __name__ == "__main__":
    main()
