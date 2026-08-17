import sys
from pathlib import Path
from typing import Optional

import numpy as np
from rich.box import DOUBLE_EDGE
from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text

_root: Optional[Path] = None
console = Console()


def root():
    if _root is None:
        sys.exit(1)
    return _root


def report_error(message: str, error_type: str = "System Failure", desc: str = ""):
    error_panel = Panel(
        Group(
            Text(f" {error_type} ", style="bold black on white"),
            Text(f"\n{message}", style="bold white"),
            Text(f"\n{desc}", style="dim white")
        ),
        box=DOUBLE_EDGE,
        border_style="grey50",
        title_align="left",
        expand=False
    )
    console.print(error_panel)


def report_success(message: str, title: str = "Success", desc: str = ""):
    """
    Displays a high-visibility success message.
    """
    success_panel = Panel(
        Group(
            Text(f" {title.upper()} ", style="bold black on white"),
            Text(f"\n{message}", style="bold white"),
            Text(f"\n{desc}", style="dim white") if desc else Text("")
        ),
        box=DOUBLE_EDGE,
        border_style="grey50",
        title_align="left",
        expand=False,
        padding=(1, 2)
    )
    console.print(success_panel)


# ============================================================
# Model loader
# Flexible loader:
# - Torchvision: supports weights + quantized models
# - HuggingFace: tries causal LM → classifier → base model
# ============================================================

def load_model(name, source="hf", weights=None, quantized=False):
    if source == "torchvision":
        return load_torchvision_model(name, weights, quantized)

    return load_hf_model(name)


from utils import console

from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoModel, \
    AutoModelForSeq2SeqLM



def load_hf_model(name: str):
    try:
        return AutoModelForCausalLM.from_pretrained(name)
    except:
        pass

    try:
        return AutoModelForSequenceClassification.from_pretrained(name)
    except:
        pass

    try:
        return AutoModel.from_pretrained(name)
    except:
        pass

    raise RuntimeError(f"Could not load model: {name}")



def load_torchvision_model(model_name, weights_name=None, quantized=False):
    import torchvision.models as models

    print(f"Loading torchvision model: {model_name}")

    # Dynamically get constructor
    if not hasattr(models, model_name):
        raise ValueError(f"Unknown torchvision model: {model_name}")

    model_fn = getattr(models, model_name)

    weights = None

    if weights_name is not None:
        # Example: ResNet50_Weights.DEFAULT
        weights = eval(f"models.{weights_name}")

    if quantized:
        model = model_fn(weights=weights, quantize=True)
    else:
        model = model_fn(weights=weights)

    return model


def get_np_type(bitwidth: int):
    if bitwidth <= 8:
        np_type = np.int8
    elif bitwidth <= 16:
        np_type = np.int16
    else:
        np_type = np.int32
    return np_type
