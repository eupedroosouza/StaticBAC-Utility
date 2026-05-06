import os
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np
import torch
from rich.progress import track

import app
import inference.imagenet
import meta
import reconstruction
import sources
import utils
from inference import mediawiki, sts2, sts2seq2seq


class InferenceType(Enum):
    IMAGENET = 1
    MEDIAWIKI = 2
    STS2 = 3
    STS2_SEQ2SEQ = 4


@dataclass
class InferenceResult:
    metric: str
    result: dict[str, float]


@dataclass
class InferenceTypeInfo:
    name: str
    supported_sources: list[sources.Source]
    run_inference: Callable[[app.Context], InferenceResult | None]


info: dict[InferenceType, InferenceTypeInfo] = {
    InferenceType.IMAGENET: InferenceTypeInfo("ImageNet", [sources.Source.TORCHVISION],
                                              run_inference=imagenet.inference_with_imagenet),
    InferenceType.MEDIAWIKI: InferenceTypeInfo("MediaWiki", [sources.Source.HUGGING_FACE],
                                               run_inference=mediawiki.inference_with_mediawiki),
    InferenceType.STS2: InferenceTypeInfo("STS-2", [sources.Source.HUGGING_FACE],
                                          run_inference=sts2.inference_with_sts2),
    InferenceType.STS2_SEQ2SEQ: InferenceTypeInfo("STS-2 to Seq2Seq", [sources.Source.HUGGING_FACE],
                                                  run_inference=sts2seq2seq.inference_with_sts2)
}


def load_model_from_npz(ctx: app.Context, model):
    npz_path = os.path.join(ctx.resultsDir, f"{ctx.modelName}_reconstructed.npz")
    if not os.path.exists(npz_path):
        utils.console.print(
            f"[bold black on magenta] INFERENCE [/bold black on magenta] [bold white on red] ERROR [/bold white on red] File not found at [dim white]{npz_path}[/dim white]")
        return None

    npz = np.load(npz_path)

    # Restore reconstructed model from NPZ
    decoded_meta = reconstruction.read_decoded_meta(ctx.decodedDir / "decoded_tensors.meta")
    _, _, name_to_id = meta.read_encoder_meta(ctx.modelDir / "tensor.meta", ctx.model.quantized)
    loaded = {}
    with utils.console.status(
            "[bold black on magenta] INFERENCE [/bold black on magenta] [white]Loading reconstructed model...[/white]"):
        for k in track(npz.files, description="", transient=True):
            if ctx.model.quantized:  # Quantized models needs bitwidth to load model correctly (check another possibilities to do same this)
                tensor_id = name_to_id[k]

                for t in decoded_meta:
                    d_tensor_id = t["idx"]
                    if tensor_id == d_tensor_id:
                        bitwidth = t["bitwidth"]
                        arr = npz[k].astype(utils.get_np_type(bitwidth))
                        loaded[k] = torch.from_numpy(arr)
                        break
                else:
                    raise ValueError(f"Missing decoded value for tensor ID: {tensor_id}")
            else:
                arr = npz[k].astype(np.float32)
                loaded[k] = torch.from_numpy(arr)

    try:
        missing, unexpected = model.load_state_dict(loaded, strict=False)
        if missing:
            utils.console.print(
                f"[bold black on magenta] INFERENCE [/bold black on magenta] [white on yellow] WARNING [/white on yellow] Missing keys: {missing}")
        utils.console.print(
            f"[bold black on magenta] INFERENCE [/bold black on magenta] [white]Successfully mapped and loaded {len(loaded)} tensors to model![/white]")
    except Exception as e:
        utils.console.print(
            f"[bold black on magenta] INFERENCE [/bold black on magenta] [white on red] ERROR [/white on red] Error loading state dict: {e}")
        return None

    return True
