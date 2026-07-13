from typing import Optional, TYPE_CHECKING

import torch
from datasets import load_dataset
from rich.box import DOUBLE_EDGE
from rich.panel import Panel
from rich.progress import track
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

if TYPE_CHECKING:
    import app
import inference
import utils


def inference_with_sts2(ctx: "app.Context") -> Optional[inference.InferenceResult]:
    """
    Evaluates a sequence classification model on the SST-2 validation dataset.

    This function loads a pre-trained base model from Hugging Face, injects custom
    weights from an .npz file, and performs batched inference to calculate accuracy.
    The model is initially loaded to the CPU to ensure compatibility with the custom
    weight loading mechanism before being moved to an available hardware accelerator (e.g., GPU).

    Args:
        ctx (app.Context): Application context containing model configuration and metadata.

    Returns:
        float | None: The classification accuracy on the validation set as a float
        (between 0.0 and 1.0), or None if an error occurs during initialization.
    """
    # Optimize CPU operations
    torch.set_num_threads(8)

    model_name = ctx.model.name

    # 1. Initialize Base Model and Tokenizer
    try:
        # Load to CPU first to ensure compatibility with custom .npz weight loading logic
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name)

        model = AutoModelForSequenceClassification.from_config(config).to("cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        utils.console.print(
            f"[bold black on magenta] INFERENCE [/bold black on magenta] [white]Loaded base model:[/white] [dim white]{model_name}[/dim white]"
        )
    except Exception as e:
        utils.console.print(
            f"[bold black on magenta] INFERENCE [/bold black on magenta] [white on red] ERROR [/white on red] Error loading model from HuggingFace: {e}"
        )
        return None

    # 2. Inject Custom Weights
    res = inference.load_model_from_npz(ctx, model)
    if res is None:
        return None

    # Set evaluation mode to disable dropout and batch normalization updates
    model.eval()

    # 3. Load SST-2 Dataset
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
        """Tokenizes text batches using consistent padding and truncation."""
        return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=128)

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
            outputs = model(**batch)

        # Extract predicted classes
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        # Count correctly classified samples in the current batch
        correct += (predictions == labels).sum().item()

    # 7. Results Calculation and Display
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
