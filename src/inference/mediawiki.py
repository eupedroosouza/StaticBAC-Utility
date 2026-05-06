import math
from typing import Optional

import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer

import inference
from utils import load_hf_model

transformers.logging.set_verbosity_error()

from rich.progress import track
from rich.box import DOUBLE_EDGE
from rich.panel import Panel

import app
import utils


def perplexity_sliding_window(model, tokenizer, text, max_length=512, stride=512, console=None):
    # Tokenize once (no truncation)
    enc = tokenizer(text, return_tensors="pt", truncation=False, add_special_tokens=True)
    input_ids = enc["input_ids"][0]  # shape: (n_tokens,)
    n_tokens = input_ids.size(0)

    model.eval()
    total_loss = 0.0
    total_count = 0

    device = model.device

    chunks = list(range(0, n_tokens, stride))

    iterable = track(chunks, description=" [bold white]Calculating Perplexity[/bold white]",
                     console=console) if console else chunks

    for begin_idx in iterable:
        end_idx = min(begin_idx + max_length, n_tokens)
        input_ids_chunk = input_ids[begin_idx:end_idx]

        labels = input_ids_chunk.clone()

        input_batch = input_ids_chunk.unsqueeze(0).to(device)
        labels = labels.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_batch, labels=labels)
            valid = (labels != -100).sum().item()
            if valid == 0:
                continue
            nll = outputs.loss.item() * valid
            total_loss += nll
            total_count += valid

        if end_idx == n_tokens:
            break

    avg_nll = total_loss / total_count if total_count > 0 else 0
    ppl = math.exp(avg_nll) if total_count > 0 else float('inf')
    return ppl, avg_nll, total_count


def inference_with_mediawiki(ctx: app.Context) -> Optional[inference.InferenceResult]:
    torch.set_num_threads(8)

    model_name = ctx.model.name

    try:
        model = load_hf_model(ctx.model.name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        utils.console.print(
            f"[bold black on magenta] INFERENCE [/bold black on magenta] [white]Loaded model:[/white] [dim white]{model_name}[/dim white]")
    except Exception as e:
        utils.console.print(
            f"[bold black on magenta] INFERENCE [/bold black on magenta] [white on red] ERROR [/white on red] Error loading model from HuggingFace: {e}")
        return None

    # Load tensors to model
    res = inference.load_model_from_npz(ctx, model)
    if res is None:
        return None

    # Dataset loading
    with utils.console.status(
            "[bold black on magenta] INFERENCE [/bold black on magenta] [white]Loading dataset (wikitext-2-raw-v1)...[/white]"):
        try:
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
            text = "\n".join([t for t in dataset["text"] if t.strip()])
        except Exception as e:
            utils.console.print(
                f"[bold black on magenta] INFERENCE [/bold black on magenta] [white on red] ERROR [/white on red] Error loading dataset: {e}")
            return None

    utils.console.print("[bold black on magenta] INFERENCE [/bold black on magenta] [white]Loaded dataset.[/white]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    utils.console.print(
        f"[bold black on magenta] INFERENCE [/bold black on magenta] [white]Running on device:[/white] [magenta]{device}[/magenta]")

    # Inference
    ppl, avg_nll, token_count = perplexity_sliding_window(model, tokenizer, text, max_length=512, stride=256,
                                                          console=utils.console)

    inference_summary = (
        f"[bold white]Model:[/bold white] {ctx.model.name}\n"
        f"[bold white]Perplexity:[/bold white] {ppl:.4f}\n"
        f"[bold white]Average NLL:[/bold white] {avg_nll:.6f}\n"
        f"[bold white]Token Count:[/bold white] {token_count}"
    )
    utils.console.print(Panel(inference_summary, title="[bold white]Inference Results[/bold white]", box=DOUBLE_EDGE,
                              border_style="white", expand=False))

    accuracy_metric = "perplexity"
    accuracy_result = {"perplexity": ppl, "avg_nll": avg_nll, "token_count": token_count}

    return inference.InferenceResult(accuracy_metric, accuracy_result)
