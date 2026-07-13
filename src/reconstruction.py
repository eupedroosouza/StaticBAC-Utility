import csv
import os.path
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy import floating
from rich.box import DOUBLE_EDGE
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.table import Table

import app
import meta
import sources
import utils

BITWIDTH_MAP = {
    0: 4, 1: 8, 2: 12, 3: 16, 4: 20, 5: 24, 6: 32
}


def check_reconstruction_errors(model, save_dir: Path, loaded_tensors: dict):
    """
    Compare reconstructed tensors (loaded_tensors) against the original model.
    """
    errors = {}
    sd = model.state_dict()
    unmatched_shapes = []

    for name, recon in loaded_tensors.items():
        if name not in sd:
            continue

        orig = sd[name].cpu().float()

        if isinstance(recon, np.ndarray):
            recon = torch.from_numpy(recon)

        recon = recon.cpu().float()

        if orig.shape != recon.shape:
            unmatched_shapes.append((name, orig.shape, recon.shape))
            continue

        abs_diff = torch.abs(orig - recon)
        max_err = torch.max(abs_diff).item()
        mean_err = torch.mean(abs_diff).item()

        errors[name] = {"max_abs": max_err, "mean_abs": mean_err}

    max_errors = []
    mean_errors = []
    for name, e in errors.items():
        max_errors.append(e['max_abs'])
        mean_errors.append(e['mean_abs'])

    overall_max_error = np.max(max_errors)
    overall_mean_error = np.mean(mean_errors)

    fields = ['tensor', 'max_abs', 'mean_abs', '', 'name', 'orig_shape', 'recon_shape', '', 'overall_max_error',
              'overall_mean_error']
    rows = []

    for i in range(max(len(errors), len(unmatched_shapes))):

        if i < len(errors):
            tensor = list(errors.keys())[i]
            e = errors[tensor]
            max_abs = f"{e['max_abs']:.20f}"
            mean_abs = f"{e['mean_abs']:.20f}"
        else:
            tensor = ''
            max_abs = ''
            mean_abs = ''

        if i < len(unmatched_shapes):
            name, o_shape, r_shape = unmatched_shapes[i]
            s_tensor = name
        else:
            s_tensor = ''
            o_shape = ''
            r_shape = ''

        if i == 0:
            overall_max_error_f = str(overall_max_error)
            overall_mean_error_f = str(overall_mean_error)
        else:
            overall_max_error_f = ''
            overall_mean_error_f = ''

        rows.append(
            [tensor, max_abs, mean_abs, '', s_tensor, o_shape, r_shape, '', overall_max_error_f, overall_mean_error_f])

    csv_file = save_dir / "reconstruction_errors.csv"
    with open(csv_file, mode='w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(fields)
        writer.writerows(rows)

    if unmatched_shapes:
        shape_table = Table(title="[bold white]Unmatched Tensor Shapes[/bold white]", box=DOUBLE_EDGE,
                            border_style="white")
        shape_table.add_column("Tensor Name", style="white")
        shape_table.add_column("Original Shape", style="dim white")
        shape_table.add_column("Reconstructed Shape", style="dim white")
        for name, o_shape, r_shape in unmatched_shapes:
            shape_table.add_row(name, str(list(o_shape)), str(list(r_shape)))
        utils.console.print(shape_table)

    if errors:
        err_table = Table(title="[bold white]Tensors Reconstruction Errors[/bold white]", box=DOUBLE_EDGE,
                          border_style="grey50")
        err_table.add_column("Tensor", style="white")
        err_table.add_column("Max Abs Error", justify="right", style="white")
        err_table.add_column("Mean Abs Error", justify="right", style="grey50")

        for name, e in errors.items():
            err_table.add_row(name, f"{e['max_abs']:.20f}", f"{e['mean_abs']:.20f}")
        utils.console.print(err_table)

    summary_text = (
        f"[bold white]Overall Max Error:[/bold white] {overall_max_error:.20f}\n"
        f"[bold white]Overall Mean Error:[/bold white] {overall_mean_error:.20f}"
    )
    utils.console.print(
        Panel(summary_text, title="Reconstruction Error", expand=False, box=DOUBLE_EDGE, border_style="grey50"))

    utils.console.print(
        f"[bold white on yellow] RECONSTRUCTION [/bold white on yellow] [bold white]Checked reconstructions errors.[/bold white] Saved to [dim white]{csv_file}[/dim white]")

    return overall_max_error, overall_mean_error, unmatched_shapes, errors


def read_decoded_meta(path):
    tensors = []

    with open(path) as f:
        lines = f.readlines()

    for line in lines:

        line = line.strip()

        if not line or line.startswith("numTensors"):
            continue

        parts = line.split()

        if len(parts) < 5:
            continue

        idx = int(parts[0])
        filename = parts[1]
        bw_enum = int(parts[3])
        dims = int(parts[4])

        # Consider dims = 0 (e.g resnet50, efficient_b0), for solve that, add empty tuple (but is it correct?)
        if dims > 0:
            shape = tuple(map(int, parts[5:5 + dims]))
        else:
            shape = ()

        tensors.append({
            "idx": idx,
            "filename": filename,
            "bitwidth": BITWIDTH_MAP[bw_enum],
            "shape": shape
        })

    return tensors


def load_tensor(path, shape):
    arr = np.fromfile(path, dtype=np.int32)

    expected = np.prod(shape)

    if arr.size != expected:
        raise RuntimeError(
            f"{path}: expected {expected} values but found {arr.size}"
        )

    return arr.reshape(shape)


def build_npz(progress,
              decoded_meta,
              folder,
              qsteps,
              id_to_name,
              quantized,
              apply_qstep=True):
    data = {}

    if len(qsteps) != len(decoded_meta):
        utils.console.print(
            f"[bold white on yellow] RECONSTRUCTION [/bold white on yellow] [bold white on red] ERROR [/bold white on red] Mismatch in tensor count! qsteps({len(qsteps)}) x meta({len(decoded_meta)})")
        raise ValueError("Mismatch in tensor count!")

    task = progress.add_task("[bold white]Reconstructing tensors", total=len(decoded_meta))

    for t in decoded_meta:
        tensor_id = t["idx"]

        if tensor_id not in id_to_name:
            utils.console.print(
                f"[bold white on yellow] RECONSTRUCTION [/bold white on yellow] [bold white on red] ERROR [/bold white on red] Missing name for tensor ID: {tensor_id}")
            raise ValueError(f"Missing name for tensor ID: {tensor_id}")

        name = id_to_name[tensor_id]

        bin_path = os.path.join(folder, t["filename"])
        bitwidth = t["bitwidth"]
        if quantized:
            # Load tensor based on bitwidth (to support 8, 16 or 32 bits quantized models)
            tensor = load_tensor(bin_path, t["shape"]).astype(utils.get_np_type(bitwidth))
        else:
            tensor = load_tensor(bin_path, t["shape"]).astype(np.float32)

        # Skip apply qstep if model is quantized
        if not quantized and apply_qstep and bitwidth != 32:
            if tensor_id not in qsteps:
                utils.console.print(
                    f"[bold white on yellow] RECONSTRUCTION [/bold white on yellow] [bold white on red] Error [/bold white on red] Missing qstep for tensor ID: {tensor_id}")
                raise ValueError(f"Missing qstep for tensor ID: {tensor_id}")

            tensor *= qsteps[tensor_id]

        data[name] = tensor

        progress.update(task, advance=1)

    return data


def reconstruct(ctx: app.Context) -> tuple[np._ScalarT, floating[Any]] | None:
    model = utils.load_model(
        ctx.model.name,
        source=sources.info[ctx.model.source].id,
        weights=ctx.model.weights,
        quantized=ctx.model.quantized
    )

    model_meta_path = ctx.modelDir / "tensor.meta"
    decoded_dir = ctx.decodedDir
    if not model_meta_path.exists() or not model_meta_path.is_file():
        utils.console.print(
            f"[bold white on yellow] RECONSTRUCTION [/bold white on yellow] [bold white on red] Error [/bold white on red] Model meta file not found at [dim white]{model_meta_path}[/dim white]")
        return None
    if not decoded_dir.exists() or not decoded_dir.is_dir():
        utils.console.print(
            f"[bold white on yellow] RECONSTRUCTION [/bold white on yellow] [bold white on red] Error [/bold white on red] Decoded directory not found at [dim white]{decoded_dir}[/dim white]")
        return None
    decoded_meta_path = decoded_dir / "decoded_tensors.meta"
    if not decoded_meta_path.exists() or not decoded_meta_path.is_file():
        utils.console.print(
            f"[bold white on yellow] RECONSTRUCTION [/bold white on yellow] [bold white on red] Error [/bold white on red] Decoded meta file not found at [dim white]{decoded_meta_path}[/dim white]")
        return None

    qstep_param, id_to_name, _ = meta.read_encoder_meta(model_meta_path, ctx.model.quantized)
    decoded_meta = read_decoded_meta(decoded_meta_path)

    utils.console.print(
        "[bold white on yellow] RECONSTRUCTION [/bold white on yellow] [white]Reconstructing model...[/white]")
    with Progress(
            SpinnerColumn(spinner_name="dots", style="white"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(style="grey50", complete_style="white"),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=utils.console
    ) as progress:

        # Reconstruct (re-quantization) params
        param_data = build_npz(
            progress,
            decoded_meta,
            decoded_dir,
            qstep_param,
            id_to_name,
            ctx.model.quantized,
            apply_qstep=True
        )

        # ----------------------------------------------------------------
        # Use ORIGINAL buffers directly from the pretrained model
        # This bypasses reconstruction entirely for diagnostic purposes
        # ----------------------------------------------------------------
        buffer_data = {}
        buffers = list(model.named_buffers())

        task = progress.add_task("[bold white]Extracting buffers", total=len(buffers))
        for i, (name, buf) in enumerate(buffers):
            buffer_data[name] = buf.numpy()
            progress.update(task, advance=1)

    all_data = {}
    all_data.update(param_data)
    all_data.update(buffer_data)

    overall_max_error, overall_mean_error, _, _ = check_reconstruction_errors(model, ctx.resultsDir, all_data)

    npzPath = os.path.join(ctx.resultsDir, f"{ctx.modelName}_reconstructed.npz")
    np.savez(npzPath, **all_data)
    utils.console.print(
        f"[bold white on yellow] RECONSTRUCTION [/bold white on yellow] [bold white]Reconstruction complete![/bold white] Saved to [dim white]{npzPath}[/dim white]")

    return overall_max_error, overall_mean_error
