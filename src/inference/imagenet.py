from typing import Optional, TYPE_CHECKING

import torch
from rich.box import DOUBLE_EDGE
from rich.panel import Panel
from rich.progress import track
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import Weights

if TYPE_CHECKING:
    import app
import config
import inference
import utils
from utils import load_torchvision_model


def inference_with_imagenet(ctx: "app.Context") -> Optional[inference.InferenceResult]:
    import torchvision.models as models # dont remove
    torch.set_num_threads(8)

    model = load_torchvision_model(ctx.model.name, ctx.model.weights, ctx.model.quantized)
    utils.console.print(
        f"[bold black on magenta] INFERENCE [/bold black on magenta] [white]Loaded model:[/white] [dim white]{ctx.model.name}[/dim white]")

    # Load tensors to model
    res = inference.load_model_from_npz(ctx, model)
    if res is None:
        return None

    model.eval()

    # Inference
    with utils.console.status(
            "[bold black on magenta] INFERENCE [/bold black on magenta] [white]Loading dataset...[/white]"):
        weights: Weights = eval(f"models.{ctx.model.weights}")
        transform = weights.transforms()

        dataset = datasets.ImageFolder(
            root=config.get_config().dataset.image_net,
            transform=transform
        )
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=8)

    utils.console.print("[bold black on magenta] INFERENCE [/bold black on magenta] [white]Loaded dataset.[/white]")

    if ctx.model.quantized:  # To torchvision quantized models use CPU to inference
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use CPU or GPU?
    model.to(device)
    utils.console.print(
        f"[bold black on magenta] INFERENCE [/bold black on magenta] [white]Running on device:[/white] [magenta]{device}[/magenta]")

    top1_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in track(dataloader, description=f" [bold white]Inference {ctx.model.name}[/bold white]",
                                    console=utils.console):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            # Top-1
            _, pred1 = outputs.topk(1, dim=1)
            top1_correct += (pred1.squeeze() == labels).sum().item()

            # Top-5
            _, pred5 = outputs.topk(5, dim=1)
            top5_correct += (pred5 == labels.view(-1, 1)).sum().item()

            total += labels.size(0)

    top1_acc = top1_correct / total * 100
    top5_acc = top5_correct / total * 100

    inference_summary = (
        f"[bold white]Model:[/bold white] {ctx.model.name}\n"
        f"[bold white]Top-1 Accuracy:[/bold white] {top1_acc:.2f}%\n"
        f"[bold white]Top-5 Accuracy:[/bold white] {top5_acc:.2f}%"
    )
    utils.console.print(Panel(inference_summary, title="[bold white]Inference Results[/bold white]", box=DOUBLE_EDGE,
                              border_style="white", expand=False))

    accuracy_metric = "top1"
    accuracy_result = {"top1": top1_acc, "top5": top5_acc}

    return inference.InferenceResult(accuracy_metric, accuracy_result)
