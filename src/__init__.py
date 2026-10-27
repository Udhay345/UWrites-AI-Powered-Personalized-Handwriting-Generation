"""
UWrites - AI-Powered Personalized Handwriting Generation
Core reusable pipeline components: preprocessing, dataset loading, and generative architectures.
"""

from .preprocessing import (
    binarize,
    deskew,
    remove_ruled_lines,
    segment_lines,
    segment_words,
    process_page,
    normalize_crop_with_padding,
)
from .dataset import (
    CHARS,
    CHAR2IDX,
    IDX2CHAR,
    VOCAB_SIZE,
    HandwritingDataset,
    collate_handwriting_batch,
)
from .models import (
    TextEncoder,
    Generator,
    Discriminator,
    ConditionalDiscriminator,
)

__all__ = [
    "binarize",
    "deskew",
    "remove_ruled_lines",
    "segment_lines",
    "segment_words",
    "process_page",
    "normalize_crop_with_padding",
    "CHARS",
    "CHAR2IDX",
    "IDX2CHAR",
    "VOCAB_SIZE",
    "HandwritingDataset",
    "collate_handwriting_batch",
    "TextEncoder",
    "Generator",
    "Discriminator",
    "ConditionalDiscriminator",
]
