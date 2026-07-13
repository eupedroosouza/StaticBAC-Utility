from typing import Optional, TYPE_CHECKING

import torch
from datasets import load_dataset
from rich.box import DOUBLE_EDGE
from rich.panel import Panel
from rich.progress import track
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

if TYPE_CHECKING:
    import app
import inference
import utils


def inference_with_sts2(ctx: "app.Context") -> Optional[inference.InferenceResult]:
    """
    Evaluates a Seq2Seq model (like T5) on the SST-2 validation dataset.
    """
    torch.set_num_threads(8)

    model_name = ctx.model.name

    try:
        # Load to CPU first to ensure compatibility with custom .npz weight loading logic
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_config(config).to("cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        utils.console.print(
            f"[bold black on magenta] INFERENCE [/bold black on magenta] [white]Loaded base model:[/white] [dim white]{model_name}[/dim white]"
        )
    except Exception as e:
        utils.console.print(
            f"[bold black on magenta] INFERENCE [/bold black on magenta] [white on red] ERROR [/white on red] Error loading model from HuggingFace: {e}"
        )
        return None

    res = inference.load_model_from_npz(ctx, model)
    if res is None:
        return None

    model.eval()

    # Load SST-2 Dataset
    with utils.console.status(
            "[bold black on magenta] INFERENCE [/bold black on magenta] [white]Loading dataset (stanfordnlp/sst2)...[/white]"):
        try:
            dataset = load_dataset("stanfordnlp/sst2", split="validation")
        except Exception as e:
            utils.console.print(
                f"[bold black on magenta] INFERENCE [/bold black on magenta] [white on red] ERROR [/white on red] Error loading dataset: {e}"
            )
            return None

    utils.console.print("[bold black on magenta] INFERENCE [/bold black on magenta] [white]Loaded dataset.[/white]")

    # 4. Hardware Accelerator Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    utils.console.print(
        f"[bold black on magenta] INFERENCE [/bold black on magenta] [white]Running on device:[/white] [magenta]{device}[/magenta]"
    )

    # 5. Data Processing (Tokenization and Batching)
    def tokenize_function(examples):
        """Tokenizes text batches using consistent padding, truncation, and T5 task prefix."""
        # T5 requires a specific prefix to know which task to perform
        prefixed_sentences = ["sst2 sentence: " + sentence for sentence in examples["sentence"]]
        return tokenizer(prefixed_sentences, padding="max_length", truncation=True, max_length=128)

    # Apply tokenization to the entire dataset efficiently
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Remove text/string columns to prevent PyTorch DataLoader errors
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
    tokenized_datasets.set_format("torch")

    # Configure DataLoader for batched processing
    batch_size = 32
    dataloader = DataLoader(tokenized_datasets, batch_size=batch_size)

    # 6. Evaluation Loop
    correct = 0
    total = len(dataset)

    utils.console.print(
        "[bold black on magenta] INFERENCE [/bold black on magenta] [white]Running inference on validation set...[/white]"
    )

    # Setup progress bar if rich console is available
    iterable = track(dataloader, description=f" [bold white]Inference {ctx.model.name}[/bold white]",
                     console=utils.console) if utils.console else dataloader

    for batch in iterable:
        # Move all batch tensors to the active device (CPU or GPU)
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("label")

        # Disable gradient calculations to save memory and speed up computation
        with torch.no_grad():
            generated_tokens = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=3  # We only need enough tokens to write "positive" or "negative"
            )

        # Extract predicted classes
        pred_strings = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        # Map the strings back to integers and count correctly classified samples
        for pred_str, label in zip(pred_strings, labels):
            # Clean the string (remove spaces and make lowercase)
            pred_str = pred_str.strip().lower()

            # Map text to SST-2 classes (1 for positive, 0 for negative)
            predicted_class = 1 if pred_str == "positive" else 0

            if predicted_class == label.item():
                correct += 1

    # Results Calculation and Display
    accuracy = correct / total
    accuracy_percent = accuracy * 100
    inference_summary = (
        f"[bold white]Model:[/bold white] {ctx.model.name}\n"
        f"[bold white]Accuracy:[/bold white] {accuracy_percent:.2f}% ({accuracy:.4f})"
    )
    utils.console.print(Panel(inference_summary, title="[bold white]Inference Results[/bold white]", box=DOUBLE_EDGE,
                              border_style="white", expand=False))

    accuracy_metric = "accuracy"
    accuracy_result = {"accuracy": accuracy}

    return inference.InferenceResult(accuracy_metric, accuracy_result)
