import importlib

_module = importlib.import_module("preprocessing.automate_i-putu-crisna-putra-ardhika")

load_dataset = _module.load_dataset
run_preprocessing = _module.run_preprocessing

__all__ = ["load_dataset", "run_preprocessing"]
