"""
===============================================================================
Neural Network Tensor Export & Quantization Tool
===============================================================================

Description:
------------
This script extracts tensors from deep learning models (PyTorch / Torchvision /
HuggingFace), optionally quantizes them, and exports:

1. Binary tensor files (.bin, int32)
2. A metadata file describing all tensors

The output is designed to be consumed by the StaticBAC encoder/decoder pipeline.

-------------------------------------------------------------------------------
Supported Model Sources:
-------------------------------------------------------------------------------

1. HuggingFace (default)
   - NLP / transformer models
   - Examples:
        bert-base-uncased
        gpt2
        meta-llama/Llama-2-7b-hf

2. Torchvision
   - Image models
   - Examples:
        resnet50
        mobilenet_v2
        vgg19
        efficientnet_b0

-------------------------------------------------------------------------------
Usage:
-------------------------------------------------------------------------------

    python export_model.py \
        --model <model_name_or_path> \
        --out_dir <output_directory> \
        [--source hf|torchvision] \
        [--weights <weights_enum>] \
        [--quantized] \
        [--no_quant]

-------------------------------------------------------------------------------
Arguments:
-------------------------------------------------------------------------------

--model <string>   (required)
    Model name or path

--out_dir <path>   (required)
    Output directory where binaries and metadata will be stored

--source <string>  (default: hf)
    Model source:
        hf           → HuggingFace models
        torchvision  → Torchvision models

--weights <string> (optional)
    Torchvision weights enum
    Example:
        ResNet50_Weights.DEFAULT

--quantized (flag)
    Load a pre-quantized torchvision model
    (uses int_repr() internally)

--no_quant (flag)
    Skip quantization (assumes tensors are already quantized)

-------------------------------------------------------------------------------
Outputs:
-------------------------------------------------------------------------------

<out_dir>/
├── binaries/
│   ├── layer1.weight.bin
│   ├── layer1.bias.bin
│   └── ...
├── tensor.meta
└── checksum.json

-------------------------------------------------------------------------------
Metadata Format:
-------------------------------------------------------------------------------

    numTensors <N>

    <id> <name> <type> <bitwidth> <numDims> <shape...> <qstep>

Example:
    0 conv1.weight weight 8 4 64 3 7 7 0.02

-------------------------------------------------------------------------------
Tensor Handling:
-------------------------------------------------------------------------------

1. Parameters (named_parameters):
    - Weights → quantized to 8-bit
    - Bias / norm → quantized to 12-bit
    - Small tensors (<32 elements) → 12-bit

2. Buffers (named_buffers):
    - NEVER quantized
    - Always stored as int32
    - Bitwidth = 32

3. Pre-quantized models:
    - Uses int_repr()
    - No additional quantization applied

-------------------------------------------------------------------------------
Quantization:
-------------------------------------------------------------------------------

- Uniform symmetric quantization
- Step size (qstep) optimized via golden-section search (MSE minimization)
- Output stored as int32 regardless of bitwidth
- Bitwidth used later for entropy coding (StaticBAC)

-------------------------------------------------------------------------------
Notes:
-------------------------------------------------------------------------------

- All outputs are stored as int32 for compatibility with C++ pipeline
- Bitwidth does NOT change storage format, only coding behavior
- Tensor names are used as filenames
- Buffers (e.g., BatchNorm stats) are preserved exactly
- A checksum.json is saved to avoid re-extracting already cached parameters

-------------------------------------------------------------------------------
Example:
-------------------------------------------------------------------------------

# HuggingFace model
python meta.py \
    --model bert-base-uncased \
    --out_dir ./bert_export

# Torchvision model with weights
python meta.py \
    --model resnet50 \
    --source torchvision \
    --weights ResNet50_Weights.DEFAULT \
    --out_dir ./resnet_export

# Quantized torchvision model
python meta.py \
    --model resnet50 \
    --source torchvision \
    --weights ResNet50_QuantizedWeights.DEFAULT \
    --quantized \
    --out_dir ./resnet_quant_export


===============================================================================
"""

import json
import logging
import os

# Desativar barras de progresso de download nativas do HuggingFace

import numpy as np
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.text import Text
from rich.box import DOUBLE_EDGE

import transformers

import utils


# may need to add more libs here for other models!


# ============================================================
# Utility functions
# ============================================================

def classify_tensor(name):
    lname = name.lower()

    if "bias" in lname:
        return "bias"
    elif "norm" in lname or "layernorm" in lname or "ln" in lname:
        return "norm"
    elif "weight" in lname:
        return "weight"
    else:
        return "other"


def convert_bitdepth(q, bitwidth):
    qmin = -(1 << (bitwidth - 1))
    qmax = (1 << (bitwidth - 1)) - 1
    return np.clip(q, qmin, qmax).astype(np.int32)


# ============================================================
# Quantization methods
# ============================================================

def optimal_uniform_quant(x, bitwidth, search_steps=40):
    x = x.astype(np.float32)
    Qmax = (1 << (bitwidth - 1)) - 1

    if x.size == 0 or np.all(x == 0):
        return np.zeros_like(x, dtype=np.int32), 1.0

    std = float(np.std(x))
    if std == 0:
        return np.zeros_like(x, dtype=np.int32), 1.0

    qstep_min = max(std / (1 << (bitwidth + 2)), 1e-12)
    qstep_max = max(std * 4.0, qstep_min * 2.0)

    phi = (1 + np.sqrt(5)) / 2.0
    invphi = 1.0 / phi

    a, b = qstep_min, qstep_max
    c = b - (b - a) * invphi
    d = a + (b - a) * invphi

    def mse(qstep):
        q = np.clip(np.round(x / qstep), -Qmax, Qmax)
        return np.mean((x - q * qstep) ** 2)

    fc, fd = mse(c), mse(d)

    for _ in range(search_steps):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - (b - a) * invphi
            fc = mse(c)
        else:
            a, c, fc = c, d, fd
            d = a + (b - a) * invphi
            fd = mse(d)

    qstep = (a + b) / 2.0
    q = np.clip(np.round(x / qstep), -Qmax, Qmax)

    return q.astype(np.int32), float(qstep)


# Core quantization entry point
# - Handles weights, biases, and buffers differently
# - Always outputs int32 (for C++ compatibility)
# - Bitwidth is used later for entropy coding, not storage
def quantize_tensor(arr, use_quant=True, tensor_kind="weight"):
    numel = arr.size

    if tensor_kind == "buffer":
        return arr, 1.0, 32

    if not use_quant:
        # assume already quantized
        return arr, 1.0, 8

    if numel < 32:
        bitwidth = 12
        qstep = np.max(np.abs(arr)) / (2 ** (bitwidth - 1) - 1 + 1e-8)
        q = np.round(arr / qstep)

    elif tensor_kind == "weight":
        bitwidth = 8
        q, qstep = optimal_uniform_quant(arr, bitwidth)
    else:
        bitwidth = 12
        q, qstep = optimal_uniform_quant(arr, bitwidth)

    q = convert_bitdepth(q, bitwidth)

    return q.astype(np.int32), qstep, bitwidth


# ============================================================
# Metadata
# ============================================================

def write_metadata(path, tensors):
    with open(path, "w") as f:
        f.write(f"numTensors {len(tensors)}\n\n")

        for t in tensors:
            shape_str = " ".join(map(str, t["shape"]))

            f.write(
                f'{t["id"]} {t["name"]} {t["type"]} '
                f'{t["bitwidth"]} {len(t["shape"])} '
                f'{shape_str} {t["qstep"]}\n'
            )


# ============================================================
# MAIN
# ============================================================

def create_meta(model_name: str, out_dir: str, source: str, weights: str = None, quantized: bool = False, gguf_file: str = None):
    bin_dir = os.path.join(out_dir, "binaries")
    meta_file = os.path.join(out_dir, "tensor.meta")
    checksum_file = os.path.join(out_dir, "checksum.json")

    tensors: int
    # Checksum Verification
    if utils.verify_checksums(out_dir, str(checksum_file)):
        utils.console.print("[bold black on white] MODEL [/bold black on white] Cached model found successfully.")
        with open(meta_file, "r") as f:
            firstLine = f.readline()
            tensors = int(firstLine.split(" ")[1]) # numTensors {num_tensors, i.e: 128}

    else:
        with utils.console.status(f"[bold black on white] MODEL [/bold black on white] Loading model [bold white]{model_name}[/bold white] from [dim white]{source}[/dim white]...", spinner="dots", spinner_style="white"):
            model = utils.load_model(
                model_name,
                source=source,
                weights=weights,
                quantized=quantized
            )
            model.eval()

        utils.console.print("[bold black on white] MODEL [/bold black on white] Model loaded successfully.")

        os.makedirs(bin_dir, exist_ok=True)

        tensor_meta_list = []
        tensor_id = 0
        checksums = {}

        parameters = list(model.named_parameters())
        buffers = list(model.named_buffers())

        with Progress(
                SpinnerColumn(spinner_name="dots", style="white"),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(style="dim white", complete_style="bold white"),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=utils.console
        ) as progress:
            param_task = progress.add_task("[bold white]Extracting parameters...", total=len(parameters))

            # --- PARAMETERS ---
            for name, param in parameters:
                tensor_kind = "weight" if "weight" in name.lower() else "bias" if "bias" in name.lower() else "other"

                # If model is already quantized (e.g., torchvision quantized models),
                # use int_repr() to extract integer values directly
                if quantized and hasattr(param, "int_repr"):
                    arr = param.int_repr().cpu().numpy().astype(np.int32)
                    q, qstep, bitwidth = quantize_tensor(
                        arr,
                        use_quant=False,  # no quantization
                        tensor_kind=tensor_kind
                    )
                else:
                    arr = param.detach().cpu().numpy().astype(np.float32)
                    q, qstep, bitwidth = quantize_tensor(
                        arr,
                        use_quant=True,
                        tensor_kind=tensor_kind
                    )

                tensor_meta_list.append({
                    "id": tensor_id,
                    "name": name,
                    "type": tensor_kind,
                    "bitwidth": bitwidth,
                    "shape": list(q.shape),
                    "qstep": qstep
                })

                # Save binary
                tensor_file = f"{name}.bin"
                filepath = os.path.join(bin_dir, tensor_file)
                np.ascontiguousarray(q.astype(np.int32)).tofile(filepath)

                # Compute checksum
                checksums[os.path.join("binaries", tensor_file).replace("\\", "/")] = utils.compute_sha256(filepath)

                tensor_id += 1
                progress.update(param_task, advance=1)

            buf_task = progress.add_task("[bold white]Extracting buffers...", total=len(buffers))

            # --- BUFFERS ---
            # Buffers are NOT part of named_parameters (e.g., BatchNorm stats)
            # They must NOT be quantized to preserve correctness
            # We store them as raw int32 with bitwidth=32
            for name, buf in buffers:
                arr = buf.detach().cpu().numpy().astype(np.float32)
                tensor_kind = "buffer"

                # Always cast to int32, never quantize
                q, qstep, bitwidth = quantize_tensor(
                    arr,
                    use_quant=False,
                    tensor_kind=tensor_kind
                )

                tensor_meta_list.append({
                    "id": tensor_id,
                    "name": name,
                    "type": tensor_kind,
                    "bitwidth": bitwidth,
                    "shape": list(q.shape),
                    "qstep": qstep
                })

                # Save binary
                tensor_file = f"{name}.bin"
                filepath = os.path.join(bin_dir, tensor_file)
                np.ascontiguousarray(q.astype(np.int32)).tofile(filepath)

                # Compute checksum
                checksums[os.path.join("binaries", tensor_file).replace("\\", "/")] = utils.compute_sha256(filepath)

                tensor_id += 1
                progress.update(buf_task, advance=1)

        write_metadata(meta_file, tensor_meta_list)
        checksums["tensor.meta"] = utils.compute_sha256(meta_file)

        with open(checksum_file, "w") as f:
            json.dump(checksums, f, indent=4)

        tensors = len(tensor_meta_list)

    summary = (
        f"[bold white]Binaries:[/]   {bin_dir}\n"
        f"[bold white]Metadata:[/]   {meta_file}\n"
        f"[bold white]Tensors:[/]    {tensors}"
    )
    utils.console.print(Panel(summary, title="Loaded Model", expand=False, box=DOUBLE_EDGE, border_style="grey50"))
