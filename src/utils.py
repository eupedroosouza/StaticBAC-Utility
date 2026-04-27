import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import transformers
from rich.box import DOUBLE_EDGE
from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoModel, AutoConfig, \
    BitsAndBytesConfig

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


class LogCaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.logs = []

    def emit(self, record):
        self.logs.append(self.format(record))


log_capturer = LogCaptureHandler()
log_capturer.setFormatter(logging.Formatter("%(name)s: %(message)s"))

# Interceptar warnings nativos
logging.captureWarnings(True)
logging.getLogger("py.warnings").addHandler(log_capturer)

# Interceptar logs do HuggingFace Hub
hf_logger = logging.getLogger("huggingface_hub")
hf_logger.setLevel(logging.WARNING)
hf_logger.addHandler(log_capturer)
hf_logger.propagate = False

# Interceptar logs verbosos do transformers e enviá-los ao nosso capturador
transformers.logging.set_verbosity_warning()
transformers.logging.disable_default_handler()
transformers.logging.add_handler(log_capturer)

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


def load_hf_model(name: str):

    # Model Class Priority List
    model_classes = [
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoModel
    ]

    for model_class in model_classes:
        try:
            return model_class.from_pretrained(name)
        except Exception as e:
            console.print(
                f"[bold black on white] MODEL [/bold black on white] [bold white on red] MODEL [/bold white on red] {e}.")
            continue

    raise RuntimeError(f"Could not load model: {name}.")


def load_torchvision_model(model_name, weights_name=None, quantized=False):
    import torch

    # Check and define engine for quantized models
    if quantized:
        console.print(
            f"[bold black on white] MODEL [/bold black on white] You trying load a quantized torchvision model: {model_name} ({weights_name}).")
        engines = torch.backends.quantized.supported_engines
        engines_str = ", ".join(engines)
        console.print(
            f"[bold black on white] MODEL [/bold black on white] Supported quantization engines: [magenta]{engines_str}[/magenta].")
        required_engine = None
        if weights_name:
            if "QNNPACK" in weights_name.upper():
                required_engine = "qnnpack"
            elif "FBGEMM" in weights_name.upper():
                required_engine = "fbgemm"
        console.print(
            f"[bold black on white] MODEL [/bold black on white] Model required engine: [green]{required_engine}[/green].")

        if required_engine not in engines:
            raise RuntimeError(f"Model needs engine: {required_engine}, but your CPU supports only: {engines_str}.")

        torch.backends.quantized.engine = required_engine

    import torchvision.models as models
    import torchvision.models.quantization as q_models

    target_module = q_models if quantized else models

    if not hasattr(target_module, model_name):
        raise ValueError(f"Unknown torchvision model: {model_name} (quantized={quantized})")

    model_fn = getattr(target_module, model_name)
    weights = None

    if weights_name is not None:
        try:
            weights = eval(f"target_module.{weights_name}")
        except AttributeError:
            raise ValueError(f"Weights {weights_name} not found in target module.")

    if quantized:
        model = model_fn(weights=weights, quantize=True)
    else:
        model = model_fn(weights=weights)

    return model


# Sha256 Checksum System
def compute_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def verify_checksums(out_dir, checksum_file):
    if not os.path.exists(checksum_file):
        return False
    try:
        with open(checksum_file, "r") as f:
            checksums = json.load(f)
        for filepath, expected_hash in checksums.items():
            full_path = os.path.join(out_dir, filepath)
            if not os.path.exists(full_path):
                return False
            if compute_sha256(full_path) != expected_hash:
                return False
        return True
    except Exception:
        return False


def get_np_type(bitwidth: int):
    if bitwidth <= 8:
        np_type = np.int8
    elif bitwidth <= 16:
        np_type = np.int16
    else:
        np_type = np.int32
    return np_type

def get_model_from_gguf(model_name: str) -> str:
    return model_name.split('/')[-1].split('-GGUF')[0]