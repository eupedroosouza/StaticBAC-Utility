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

        state_dict = model.state_dict()
        export_tensors = {}

        for name, tensor in state_dict.items():
            # Separate Weights/Bias when model is quantized by Torch
            # Torch saves weights in a int8 structure but bias in a float32; therefore, they need to be separated
            # The biases will undergo to quantization normally
            # The weights will be save as int8 normally, as expected
            if name.endswith("._packed_params._packed_params"):
                prefix = name.replace("._packed_params._packed_params", "")
                mod = get_submodule_safe(model, prefix)
                export_tensors[f"{name}_weight"] = mod.weight()
                if hasattr(mod, "bias") and mod.bias() is not None:
                    export_tensors[f"{name}_bias"] = mod.bias()
            else:
                export_tensors[name] = tensor

        with Progress(
                SpinnerColumn(spinner_name="dots", style="white"),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(style="dim white", complete_style="bold white"),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=utils.console
        ) as progress:
            task = progress.add_task("[bold white]Extracting model...", total=len(export_tensors))

            for name, param in export_tensors.items():
                # Dtype is only metadata from a Torch quantized tensor, skip
                if ".dtype" in name:
                    continue

                tensor_kind = "weight" if "weight" in name.lower() else "bias" if "bias" in name.lower() else "buffer"

                # If is already quantized
                if quantized and getattr(param, "is_quantized", False):
                    arr = param.int_repr().cpu().numpy().astype(np.int32)
                    q, qstep, bitwidth = quantization.quantize_tensor(
                        arr,
                        use_quant=False,  # no quantization
                        tensor_kind=tensor_kind
                    )
                else:
                    arr = param.detach().cpu().numpy().astype(np.float32)

                    # Skip if is buffer (buffer not should quantize)
                    use_quant = (tensor_kind != "buffer") and not quantized

                    q, qstep, bitwidth = quantization.quantize_tensor(
                        arr,
                        use_quant=use_quant,
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

                tensor_file = f"{name}.bin"
                filepath = os.path.join(bin_dir, tensor_file)
                np.ascontiguousarray(q.astype(np.int32)).tofile(filepath)

                tensor_id += 1
                progress.update(task, advance=1)

        write_metadata(meta_file, tensor_meta_list)

        tensors = len(tensor_meta_list)

    summary = (
        f"[bold white]Binaries:[/]   {bin_dir}\n"
        f"[bold white]Metadata:[/]   {meta_file}\n"
        f"[bold white]Tensors:[/]    {tensors}"
    )
    utils.console.print(Panel(summary, title="Loaded Model", expand=False, box=DOUBLE_EDGE, border_style="grey50"))
