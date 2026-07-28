import sys
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        print(
            f"❌ Error: Config file not found in path: '{config_path}'", file=sys.stderr
        )
        sys.exit(1)

    with open(path, "r") as f:
        return yaml.safe_load(f)
