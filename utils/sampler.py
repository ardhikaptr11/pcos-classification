import optuna
import optunahub


def get_sampler(sampler_cfg: dict) -> optuna.samplers.BaseSampler:
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
