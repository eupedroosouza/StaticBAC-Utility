import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from os.path import abspath
from pathlib import Path
from subprocess import CompletedProcess
from typing import Optional

import app
import config
import results
import utils


@dataclass
class StaticBacExecResult:
    tensors: int
    encode: StaticBacEncodeResult
    decode: StaticBacDecodeResult
    result: CompletedProcess[str]


@dataclass
class StaticBacMemResult:
    baseline: float
    peak: float
    delta: float


@dataclass
class StaticBacEncodeResult:
    originalSize: float
    compressedSize: float
    compressionRatio: float
    time: float
    speed: float
    mem: StaticBacMemResult


@dataclass
class StaticBacDecodeResult:
    decodedSize: float
    time: float
    speed: float
    mem: StaticBacMemResult


@dataclass
class StaticBacEnvironment:
    folder: Path
    executable: Path


_environment: Optional[StaticBacEnvironment] = None


def environment():
    if _environment is None:
        utils.console.print("[bold black on white]Error:[/bold black on white] [bold white] Configuration requested before initialization. [/bold white]")
        sys.exit(1)
    return _environment


def setup_environment():
    if config.get_config().staticbac is None:
        utils.console.print(
            f"[bold black on white] ENVIRONMENT [/bold black on white] [bold white]StaticBAC environment not found. See config.yml.[/bold white]")
        return 0

    if config.get_config().staticbac.folder is None:
        utils.console.print(
            f"[bold black on white] ENVIRONMENT [/bold black on white] [bold white]You need define folder of StaticBAC on config.yml. Please edit.[/bold white]")
        return 0

    folder = Path(config.get_config().staticbac.folder)
    abs_folder = abspath(folder)
    if not folder.exists():
        utils.console.print(
            f"[bold black on white] ENVIRONMENT [/bold black on white] [bold white]StaticBAC environment not found in {abs_folder}. See config.yml.[/bold white]")
        return 0
    else:
        utils.console.print(
            f"[bold black on white] ENVIRONMENT [/bold black on white] [bold white]StaticBAB environment found in {abs_folder}.[/bold white]")
    executable: Path
    if not config.get_config().staticbac.executable is None:
        path = Path(config.get_config().staticbac.executable)
        if not path.exists():
            utils.console.print(
                f"[bold black on white] ENVIRONMENT [/bold black on white] [bold white]StaticBAC executable defined by config.yml in: {abspath(path)} not found. See config.yml.[/bold white]")
            return 0
        if not path.is_file():
            utils.console.print(
                f"[bold black on white] ENVIRONMENT [/bold black on white] [bold white]StaticBAC executable defined by config.yml in: {abspath(path)} not is a file. See config.yml.[/bold white]")
            return 0
        executable = path
    else:
        staticbac_windows_executable = Path(folder) / "StaticBac.exe"
        staticbac_linux_executable = Path(folder) / "StaticBac"
        if staticbac_windows_executable.exists() & staticbac_windows_executable.is_file():
            executable = staticbac_windows_executable
        elif staticbac_linux_executable.exists() & staticbac_linux_executable.is_file():
            executable = staticbac_linux_executable
        else:
            utils.console.print(
                f"[bold black on white] ENVIRONMENT [/bold black on white] [bold white]StaticBAC executable not found in folder: {abs_folder}. Define custom StaticBAC executable path in config.yml or check if StaticBAC was compiled in {abspath(folder)}.[/bold white]")
            return 0
        utils.console.print(
            f"[bold black on white] ENVIRONMENT [/bold black on white] [bold white]StaticBAB executable found in: {abspath(executable)}.[/bold white]")
    global _environment
    _environment = StaticBacEnvironment(folder, executable)

    return 1


def run(ctx: app.Context, save_dir: Path) -> StaticBacExecResult:
    model_binaries_dir = ctx.modelDir / "binaries"
    model_tensors_meta_path = ctx.modelDir / "tensor.meta"

    command = f'"{environment().executable}" --encode --decode --binaries "{abspath(model_binaries_dir)}" --meta "{abspath(model_tensors_meta_path)}" --name "{ctx.model.name}"'

    sbac_exec_result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        check=True
    )

    decoded_dir = save_dir / "decoded/"
    decoded_bitstream_path = os.path.join(save_dir, f"{ctx.modelName}.bin")

    # Move the folders and decoded
    decoded_sbac_folder = os.path.join(utils.root(), f"{ctx.model.name}_decoded")
    decoded_sbac_bitstream = os.path.join(utils.root(), f"{ctx.model.name}.bin")
    shutil.move(decoded_sbac_folder, abspath(decoded_dir))
    shutil.move(decoded_sbac_bitstream, abspath(decoded_bitstream_path))

    stdout = sbac_exec_result.stdout.strip()
    enc_mem_block = results.extract(r"Encode memory:(.*?)(?:=====|$)", text=stdout, flags=re.DOTALL)
    if enc_mem_block == "N/A": enc_mem_block = ""
    dec_mem_block = results.extract(r"Decode memory:(.*?)(?:=====|$)", text=stdout, flags=re.DOTALL)
    if dec_mem_block == "N/A": dec_mem_block = ""

    # Parse Results
    tensors = results.to_num(results.extract(r"Tensors processed\s+:\s+([\d.]+)", text=stdout))
    encode_result = StaticBacEncodeResult(
        results.to_num(results.extract(r"Original size\s+:\s+([\d.]+)", text=stdout)),
        results.to_num(results.extract(r"Compressed size\s+:\s+([\d.]+)\s+MB", text=stdout)),
        results.to_num(results.extract(r"Compression ratio\s+:\s+([\d.]+)", text=stdout)),
        results.to_num(results.extract(r"Encoding time\s+:\s+([\d.]+)", text=stdout)),
        results.to_num(results.extract(r"Encode speed\s+:\s+([\d.]+)", text=stdout)),
        StaticBacMemResult(
            results.to_num(results.extract(r"baseline\s+:\s+([\d.]+)", text=enc_mem_block)),
            results.to_num(results.extract(r"peak\s+:\s+([\d.]+)", text=enc_mem_block)),
            results.to_num(results.extract(r"delta\s+:\s+([\d.]+)", text=enc_mem_block))
        )
    )
    decode_result = StaticBacDecodeResult(
        results.to_num(results.extract(r"Decoded size\s+:\s+([\d.]+)", text=stdout)),
        results.to_num(results.extract(r"Decoding time\s+:\s+([\d.]+)", text=stdout)),
        results.to_num(results.extract(r"Decode speed\s+:\s+([\d.]+)", text=stdout)),
        StaticBacMemResult(
            results.to_num(results.extract(r"baseline\s+:\s+([\d.]+)", text=dec_mem_block)),
            results.to_num(results.extract(r"peak\s+:\s+([\d.]+)", text=dec_mem_block)),
            results.to_num(results.extract(r"delta\s+:\s+([\d.]+)", text=dec_mem_block))
        )
    )

    result = StaticBacExecResult(tensors, encode_result, decode_result, sbac_exec_result)

    return result