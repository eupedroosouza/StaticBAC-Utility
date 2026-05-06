from dataclasses import dataclass
from enum import Enum


class Source(Enum):
    HUGGING_FACE = 1
    TORCHVISION = 2


@dataclass
class SourceInfo:
    id: str
    display_name: str


info: dict[Source, SourceInfo] = {
    Source.HUGGING_FACE: SourceInfo("hf", "Hugging Face"),
    Source.TORCHVISION: SourceInfo("torchvision", "Torchvision")
}
