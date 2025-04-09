from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


import torch


@dataclass
class DatasetSchema:
    file_path: Path | None = None
    dataset: str = "sarulab"
    mos: int | None = None
    audio_tensor: torch.Tensor | None = None
