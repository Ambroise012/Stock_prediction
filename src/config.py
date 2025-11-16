""" Script to load config from config.json """
import json
import os
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.json"

class ConfigNamespace:
    """Convert dictionary to object-like namespace (dot access)."""
    def __init__(self, data: dict):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigNamespace(value))
            else:
                setattr(self, key, value)

def load_config():
    """Load configuration from JSON + environment variables with dot-access."""
    # Load JSON config
    with open(CONFIG_PATH, "r") as f:
        config_data = json.load(f)

    return ConfigNamespace(config_data)

config = load_config()
