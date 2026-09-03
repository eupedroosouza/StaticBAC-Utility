import os
from pathlib import Path

import numpy as np
from rich.box import DOUBLE_EDGE
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

import app
import quantization
import sources
import utils


# ============================================================
# Metadata
# ============================================================

def read_encoder_meta(path, quantized):
    qsteps = {}
    id_to_name = {}
    name_to_id = {}

    with open(path) as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith("numTensors"):
            continue

        parts = line.split()
        if len(parts) < 6:
            continue

        tensor_id = int(parts[0])
        name = parts[1]

        dims = int(parts[4])

        # qstep is last field
        qstep = float(parts[5 + dims])

        qsteps[tensor_id] = qstep
        id_to_name[tensor_id] = name
        name_to_id[name] = tensor_id

    return qsteps, id_to_name, name_to_id


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

def get_submodule_safe(m, path):
    for p in path.split('.'):
        m = getattr(m, p)
    return m

def create_meta(ctx: app.Context):
    out_dir = ctx.modelDir
    bin_dir = Path(os.path.join(out_dir, "binaries"))
    meta_file = Path(os.path.join(out_dir, "tensor.meta"))

    tensors: int

    # Check if model is already cached and check if all tensors files exists
    if out_dir.exists() and meta_file.exists():
        _, id_to_name, _ = read_encoder_meta(meta_file, ctx.model.quantized)
        tensors = len(id_to_name.values())
        for name in id_to_name.values():
            if not (bin_dir / f"{name}.bin").exists():
                raise RuntimeError(f"Missing tensor file: {name}.")
    else:
        quantized = ctx.model.quantized

        with utils.console.status(
                f"[bold black on white] MODEL [/bold black on white] Loading model [bold white]{ctx.model.name}[/bold white] from [dim white]{sources.info[ctx.model.source].display_name}[/dim white]...",
                spinner="dots", spinner_style="white"):
            model = utils.load_model(
                ctx.model.name,
                source=sources.info[ctx.model.source].id,
                weights=ctx.model.weights,
                quantized=quantized
            )
            model.eval()

        utils.console.print("[bold black on white] MODEL [/bold black on white] Model loaded successfully.")

        os.makedirs(bin_dir, exist_ok=True)

        tensor_meta_list = []
        tensor_id = 0

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
                    q, qstep, bitwidth = quantization.quantize_tensor(
                        arr,
                        use_quant=False,  # no quantization
                        tensor_kind=tensor_kind
                    )
                else:
                    arr = param.detach().cpu().numpy().astype(np.float32)
                    q, qstep, bitwidth = quantization.quantize_tensor(
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
                q, qstep, bitwidth = quantization.quantize_tensor(
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

                tensor_id += 1
                progress.update(buf_task, advance=1)

        write_metadata(meta_file, tensor_meta_list)

        tensors = len(tensor_meta_list)

    summary = (
        f"[bold white]Binaries:[/]   {bin_dir}\n"
        f"[bold white]Metadata:[/]   {meta_file}\n"
        f"[bold white]Tensors:[/]    {tensors}"
    )
    utils.console.print(Panel(summary, title="Loaded Model", expand=False, box=DOUBLE_EDGE, border_style="grey50"))
