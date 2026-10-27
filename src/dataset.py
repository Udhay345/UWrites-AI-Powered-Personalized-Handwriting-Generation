"""
UWrites - Dataset and Character Encoding Module
Handles character vocabulary mapping, tensor conversions, and PyTorch Dataset / DataLoader integration.
"""

import os
import csv
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch.nn as nn

# Standard character set covering letters, numbers, punctuation, and whitespace
CHARS = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?'\"-"
CHAR2IDX = {c: i for i, c in enumerate(CHARS)}
IDX2CHAR = {i: c for i, c in enumerate(CHARS)}
VOCAB_SIZE = len(CHARS)

DEFAULT_IMG_HEIGHT = 64
DEFAULT_IMG_WIDTH = 256


class HandwritingDataset(Dataset):
    """
    PyTorch Dataset loading segmented handwriting image crops paired with transcription labels.
    """
    def __init__(
        self,
        csv_path: str,
        img_root: str,
        img_size: tuple[int, int] = (DEFAULT_IMG_WIDTH, DEFAULT_IMG_HEIGHT),
        max_seq_len: int = 20,
        preserve_aspect_ratio: bool = True,
    ):
        """
        Args:
            csv_path: Path to labeled CSV file (must have 'relpath' and 'transcription' headers).
            img_root: Directory containing segmented category subfolders.
            img_size: Tuple (width, height) for image tensor dimensions.
            max_seq_len: Maximum character sequence length to encode.
            preserve_aspect_ratio: If True, centers image on canvas with padding instead of stretch.
        """
        self.samples = []
        self.img_root = img_root
        self.img_width, self.img_height = img_size
        self.max_seq_len = max_seq_len
        self.preserve_aspect_ratio = preserve_aspect_ratio

        if os.path.exists(csv_path):
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    relpath = row.get("relpath", "").strip()
                    transcription = row.get("transcription", "").strip()
                    if relpath and transcription:
                        self.samples.append((relpath, transcription))
        else:
            print(f"WARNING: CSV annotation file not found at {csv_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        relpath, text = self.samples[idx]
        img_path = os.path.join(self.img_root, relpath)
        
        # Load grayscale image
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("L")
        else:
            # Fallback blank white image if file path is missing
            img = Image.new("L", (self.img_width, self.img_height), color=255)

        if self.preserve_aspect_ratio:
            # Aspect-ratio preserving letterbox scaling
            w, h = img.size
            scale = min(self.img_width / w, self.img_height / h)
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            img_resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
            
            canvas = Image.new("L", (self.img_width, self.img_height), color=255)
            x_offset = (self.img_width - new_w) // 2
            y_offset = (self.img_height - new_h) // 2
            canvas.paste(img_resized, (x_offset, y_offset))
            img_arr = np.array(canvas, dtype=np.float32)
        else:
            # Direct stretch resizing (retains original notebook behavior)
            img_resized = img.resize((self.img_width, self.img_height), Image.Resampling.BILINEAR)
            img_arr = np.array(img_resized, dtype=np.float32)

        # Normalize pixel values from [0, 255] to [-1.0, 1.0] for GAN training
        img_norm = img_arr / 127.5 - 1.0
        img_tensor = torch.tensor(img_norm, dtype=torch.float32).unsqueeze(0)  # Shape: (1, H, W)

        # Tokenize character string into integer IDs
        text_sub = text[: self.max_seq_len]
        text_ids = [CHAR2IDX.get(c, 0) for c in text_sub]
        if not text_ids:
            text_ids = [0]
        text_tensor = torch.tensor(text_ids, dtype=torch.long)

        return img_tensor, text_tensor


def collate_handwriting_batch(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function to stack image tensors and pad variable-length character sequences with 0.
    """
    images = torch.stack([item[0] for item in batch], dim=0)
    sequences = [item[1] for item in batch]
    padded_texts = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    return images, padded_texts
