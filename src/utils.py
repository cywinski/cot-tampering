# ABOUTME: Common utility functions used across experiment scripts
# ABOUTME: Configuration loading, prompt formatting, and file I/O utilities

import json
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from sampling import PromptFormatter


def load_config(config_path: str) -> dict:
    """Load YAML configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Config dict
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_formatter(config: dict) -> PromptFormatter:
    """Create prompt formatter from config.

    Args:
        config: Config dict with prompt settings

    Returns:
        PromptFormatter instance
    """
    return PromptFormatter(
        template_name=config.get("template", "raw"),
        field_name=config.get("field_name", "problem"),
        custom_template=config.get("custom_template"),
    )


def load_response_file(response_path: Path) -> dict:
    """Load a response JSON file.

    Args:
        response_path: Path to response JSON file

    Returns:
        Response data dict
    """
    with open(response_path) as f:
        return json.load(f)
