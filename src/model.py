from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import questionary
from rich.panel import Panel
from rich.box import DOUBLE_EDGE

import config
import utils
from config import ModelConfig


@dataclass
class SelectModel:
    model: ModelConfig
    dir: Path


def select_model_name() -> Optional[str]:
    model_name = (questionary.select("Select the model?",
                                     choices=config.get_config().models)
                  .ask())
    return model_name


def select_model() -> tuple[str, ModelConfig] | None:
    model_name = select_model_name()
    if not model_name:
        return None

    model = config.get_config().models[model_name]
    source_name = "HuggingFace" if model.type == "hf" else "Torchvision" if model.type == "torchvision" else model.type

    summary_text = (
        f"[bold white]Name:[/bold white] {model.name}\n"
        f"[bold white]Source:[/bold white] {source_name}\n"
        f"[bold white]Weights:[/bold white] {model.weights}\n"
        f"[bold white]Inference Dataset:[/bold white] {model.inference}\n"
        f"[bold white]Quantized:[/bold white] {'Yes' if model.quantized else 'No'}"
    )

    utils.console.print()
    utils.console.print(Panel(summary_text, title="Selected Model", expand=False, box=DOUBLE_EDGE, border_style="grey50"))
    utils.console.print()

    return model_name, model