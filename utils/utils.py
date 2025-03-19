import importlib


def load_model(model_name):
    try:
        model_class = getattr(importlib.import_module("models"), model_name)
        return model_class()
    except AttributeError:
        raise ValueError(f"'{model_name}' not defined.")
    

def load_criterion(loss_name):
    try:
        model_class = getattr(importlib.import_module("models"), loss_name)
        return model_class
    except AttributeError:
        raise ValueError(f"'{loss_name}' not defined.")