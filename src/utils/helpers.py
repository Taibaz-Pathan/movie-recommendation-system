"""Shared utility functions for the movie recommendation system."""

import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load and return the project configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file, relative to the
            project root or an absolute path.

    Returns:
        Dictionary containing all configuration values.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file not found at '{config_path}'. "
            "Ensure you are running from the project root directory."
        )
    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    return config


def set_random_seed(seed: int = 42) -> None:
    """Set the random seed for reproducibility across numpy and the stdlib random module.

    Args:
        seed: Integer seed value. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)


def get_project_root() -> str:
    """Return the absolute path to the project root directory.

    Assumes this file lives at <project_root>/src/utils/helpers.py.

    Returns:
        Absolute path string of the project root directory.
    """
    this_file = Path(__file__).resolve()
    # Navigate up: helpers.py -> utils -> src -> project root
    project_root = this_file.parent.parent.parent
    return str(project_root)
