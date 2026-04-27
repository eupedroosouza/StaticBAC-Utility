import os.path
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import questionary
from rich.box import DOUBLE_EDGE
from rich.console import Group
from rich.panel import Panel
from rich.text import Text

import config
import sbac
import utils
from inference.imagenet import inference_with_imagenet
from inference.mediawiki import inference_with_mediawiki
from inference.sts2 import inference_with_sts2
from meta import create_meta
from model import select_model
from reconstruction import reconstruct
from results import save_result, avg_sbac_results
from sbac import StaticBacExecResult


@dataclass
class Context:
    modelName: str
    model: config.ModelConfig
    modelDir: Path
    resultsDir: Path
    decodedDir: Path


@dataclass
class IterationResult:
    staticbac: StaticBacExecResult
    overall_max_error: float
    overall_mean_error: float
    accuracy_metric: str
    accuracy_result: dict[str, float]


def run():
    # Header
    header = Panel(
        Group(
            Text(f"StaticBAC Utility", style="bold white", justify="center"),
            Text("A utility tool to execute StaticBAC tests", style="dim white", justify="center")
        ),
        box=DOUBLE_EDGE,
        border_style="grey50",
        title_align="center",
        expand=False,
        padding=(1, 2)
    )
    utils.console.print(header)
    utils.console.print()

    # Configuration
    result = config.load_configuration()
    if result == 0:
        return

    # StaticBAC Environment
    staticbac_environment_result = sbac.setup_environment()
    if staticbac_environment_result == 0:
        return

    # Select model
    res = select_model()
    if not res:
        return
    model_name, model = res

    # Context
    output_dir = Path(config.get_config().utility.output_dir)
    model_dir = output_dir / "models" / model_name
    now_time = int(time.time())
    results_dir = output_dir / "results" / model_name / (str(now_time))
    decoded_dir = results_dir / "decoded"
    results_path = results_dir / "results.xlsx"

    ctx = Context(model_name, model, model_dir, results_dir, decoded_dir)

    iterations_s: str = questionary.text("How many iterations?", default="5").ask()
    if iterations_s is None:
        return
    iterations: int
    try:
        iterations = int(iterations_s)
    except TypeError:
        utils.console.print(f"Invalid number of iterations: {iterations_s}.")
        return

    utils.console.print()

    # 1. Load model
    create_meta(model.name, str(model_dir), model.type, model.weights, model.quantized)


    # 2. Running StaticBAC iterations
    results: list[StaticBacExecResult] = []
    for i in range(iterations):
        itr = i + 1

        save_dir: Path
        if itr == iterations:
            save_dir = ctx.resultsDir
        else:
            save_dir = Path(os.path.join(results_dir, f"iteration_{itr}/"))
        utils.console.print(f"\n[bold black on white] ITERATION {itr}/{iterations} [/bold black on white]")

        # 1. Running StaticBAC
        sbac_result: StaticBacExecResult
        # Use status here to show anyone view to user
        with utils.console.status("[bold white on cyan] STATICBAC [/bold white on cyan] Running Encode/Decode...",
                                  spinner="dots", spinner_style="cyan"):
            try:
                sbac_result = sbac.run(ctx, save_dir)
                if sbac_result is None:
                    return
                sbac_exec_result = sbac_result.result

                output_text = Text(sbac_exec_result.stdout.strip()) if sbac_exec_result.stdout.strip() else Text(
                    "StaticBAC Encode/Decode successful (No output).", style="dim white")
                utils.console.print(
                    Panel(output_text, title=f"[bold cyan]StaticBAC Run ({itr})[/bold cyan]",
                          border_style="cyan",
                          expand=False, box=DOUBLE_EDGE))

                results.append(sbac_result)
            except subprocess.CalledProcessError as e:
                error_text = Text()
                error_text.append(f"Exit Code: {e.returncode}\n\n", style="bold red")
                error_text.append("Standard Output (stdout):\n", style="bold red")
                error_text.append(f"{e.stdout.strip() if e.stdout else 'None'}\n\n")
                error_text.append("Standard Error (stderr):\n", style="bold red")
                error_text.append(f"{e.stderr.strip() if e.stderr else 'None'}")

                utils.console.print(
                    Panel(error_text, title="[bold red]Execution Failed[/bold red]", border_style="red",
                          expand=False, box=DOUBLE_EDGE))
                return

        # Remove iterate dir (iteration_{itr})
        if itr != iterations:
            shutil.rmtree(save_dir)

    avg_sbac_result = avg_sbac_results(results)
    if avg_sbac_result is None:
        return

    # 2. Run reconstruction model
    rec_res = reconstruct(ctx)
    if rec_res is None:
        return
    overall_max_error, overall_mean_error = rec_res

    accuracy_metric: str = ""
    accuracy_result: dict[str, float] = {}
    # 3. Run inference
    if model.inference.lower() == "imagenet" or model.inference.lower() == "inet":
        res = inference_with_imagenet(ctx)
        if res is None:
            return
        top1, top5 = res
        accuracy_metric = "top1"
        accuracy_result = {"top1": top1, "top5": top5}
    elif model.inference.lower() == "mediawiki" or model.inference.lower() == "mw":
        ppl, avg_nll, token_count = inference_with_mediawiki(ctx)
        accuracy_metric = "perplexity"
        accuracy_result = {"perplexity": ppl, "avg_nll": avg_nll, "token_count": token_count}
    elif model.inference.lower() == "sts2" or model.inference.lower() == "sts-2":
        accuracy = inference_with_sts2(ctx)
        if accuracy is None:
            return
        accuracy_metric = "accuracy"
        accuracy_result = {"accuracy": accuracy}
    else:
        utils.console.print(
            f"[bold black on magenta] INFERENCE [/bold black on magenta] [bold black on red] ERROR [/bold black on red] Invalid inference type: {model.inference}.")
        return

    # 4. Results
    results_file = ctx.resultsDir / "results.csv"
    result = IterationResult(avg_sbac_result, overall_max_error, overall_mean_error, accuracy_metric,
                             accuracy_result)
    save_result(results_file, ctx, result)
    utils.console.print(
        f"[bold white on green] RESULTS [/bold white on green] [white]Saved results to:[/white] [dim white]{results_file}[/dim white]..")

    return
