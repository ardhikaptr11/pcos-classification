import sys
from pathlib import Path

import optuna
import optunahub
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


def load_sampler(sampler_cfg: dict) -> optuna.samplers.BaseSampler:
    if not sampler_cfg:
        return optuna.samplers.TPESampler(seed=42)

    use_optunahub = sampler_cfg.get("use_optunahub", True)
    kwargs = sampler_cfg.get("kwargs", {})

    if use_optunahub:
        package = sampler_cfg["package"]
        class_name = sampler_cfg["class_name"]

        module = optunahub.load_module(package=package)
        sampler_class = getattr(module, class_name)
        return sampler_class(**kwargs)
    else:
        class_name = sampler_cfg.get("class_name", "TPESampler")
        sampler_class = getattr(optuna.samplers, class_name)
        return sampler_class(**kwargs)
