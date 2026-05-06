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


from utils import console  # assumindo que seu console vem daqui

from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoModel, \
    AutoModelForSeq2SeqLM


# (Assumindo que você tem o utils importado)

def load_hf_model(name: str):
    try:
        config = AutoConfig.from_pretrained(name)

        if hasattr(config, "architectures") and config.architectures:
            arch = config.architectures[0].lower()

            # Classification models (BERT, RoBERTa no SST-2)
            if "sequenceclassification" in arch:
                console.print(
                    f"[bold black on white] MODEL [/bold black on white] [white]Loading {name} as Sequence Classification[/white]")
                return AutoModelForSequenceClassification.from_pretrained(name)

            # Text generations models (GPT-2, LLaMA, OPT, Bloom)
            elif "causallm" in arch or "lmhead" in arch:
                console.print(
                    f"[bold black on white] MODEL [/bold black on white] [white]Loading {name} as Causal LM[/white]")
                return AutoModelForCausalLM.from_pretrained(name)

            # Sequence-to-Sequence / Encoder-Decoder (T5, BART)
            elif "conditionalgeneration" in arch or "seq2seqlm" in arch:
                console.print(
                    f"[bold black on white] MODEL [/bold black on white] [white]Loading {name} as Seq2Seq LM[/white]")
                return AutoModelForSeq2SeqLM.from_pretrained(name)

            # Mask models (bert-base)
            elif "maskedlm" in arch:
                from transformers import AutoModelForMaskedLM
                console.print(
                    f"[bold black on white] MODEL [/bold black on white] [white]Loading {name} as Masked LM[/white]")
                return AutoModelForMaskedLM.from_pretrained(name)

        # Fallback to Auto Model
        console.print(
            f"[bold black on white] MODEL [/bold black on white] [white on yellow] WARNING [/white on yellow] Architecture unclear. Loading raw AutoModel.")
        return AutoModel.from_pretrained(name)

    except Exception as e:
        console.print(
            f"[bold black on white] MODEL [/bold black on white] [bold white on red] ERROR [/bold white on red] {e}"
        )
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
        model = model_fn(weights=weights, quantize=True, progress=False)
    else:
        model = model_fn(weights=weights, progress=False)

    return model


def get_np_type(bitwidth: int):
    if bitwidth <= 8:
        np_type = np.int8
    elif bitwidth <= 16:
        np_type = np.int16
    else:
        np_type = np.int32
    return np_type
