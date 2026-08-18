from enum import Enum


class InferenceType(Enum):
    IMAGENET = 1
    MEDIAWIKI = 2
    STS2 = 3
    STS2_SEQ2SEQ = 4
