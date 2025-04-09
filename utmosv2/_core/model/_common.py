from __future__ import annotations

import abc
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

from utmosv2._settings._config import Config
from utmosv2.dataset._schema import DatasetSchema
from utmosv2.utils import get_dataset

if TYPE_CHECKING:
    import torch.nn as nn


class UTMOSv2ModelMixin(abc.ABC):
    """
    Abstract mixin for UTMOSv2 models, providing a template for prediction.
    """

    @property
    @abc.abstractmethod
    def _cfg(self) -> Config:
        pass

    @abc.abstractmethod
    def eval(self) -> "nn.Module":
        pass

    @abc.abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        pass

    def predict(
        self,
        *,
        input_path: Path | str | None = None,
        input_dir: Path | str | None = None,
        input_tensor: torch.Tensor | None = None,
        input_tensors: list[torch.Tensor] | None = None,
        val_list: list[str] | None = None,
        val_list_path: Path | str | None = None,
        predict_dataset: str = "sarulab",
        device: str | torch.device = "cuda:0",
        num_workers: int = 4,
        batch_size: int = 16,
        num_repetitions: int = 1,
        remove_silent_section: bool = True,
        verbose: bool = True,
        sample_rate: int | None = None,
    ) -> float | list[dict[str, str | float]]:
        """
        Predict the MOS (Mean Opinion Score) of audio files.

        Args:
            input_path (Path | str | None):
                Path to a single audio file (`.wav`) to predict MOS.
                Either `input_path` or `input_dir` must be provided, but not both.
            input_dir (Path | str | None):
                Path to a directory of `.wav` files to predict MOS.
                Either `input_path` or `input_dir` must be provided, but not both.
            val_list (list[str] | None):
                List of filenames to include for prediction. Defaults to None.
            val_list_path (Path | str | None):
                Path to a text file containing a list of filenames to include for prediction. Defaults to None.
            predict_dataset (str):
                Name of the dataset to associate with the prediction. Defaults to "sarulab".
            device (str | torch.device):
                Device to use for prediction (e.g., "cuda:0" or "cpu"). Defaults to "cuda:0".
            num_workers (int):
                Number of workers for data loading. Defaults to 4.
            batch_size (int):
                Batch size for the data loader. Defaults to 16.
            num_repetitions (int):
                Number of prediction repetitions to average results. Defaults to 1.
            remove_silent_section (bool):
                Whether to remove silent sections from the audio before prediction. Defaults to True.
            verbose (bool):
                Whether to display progress during prediction. Defaults to True.

        Returns:
            float: If the `input_path` is specified, returns the predicted MOS.
            list[dict[str, str | float]]: If the `input_dir` is specified, returns a list of dicts containing file paths and predicted MOS scores.

        Raises:
            ValueError: If both `input_path` and `input_dir` are provided, or if neither is provided.
        """
        input_modes = sum(
            x is not None
            for x in [input_path, input_dir, input_tensor, input_tensors]
        )
        if input_modes != 1:
            raise ValueError(
                "Exactly one of `input_path`, `input_dir`, `input_tensor`, or `input_tensors` must be provided."
            )
            
        original_sr = None
        if (input_tensor is not None or input_tensors is not None):
            if sample_rate is None:
                raise ValueError("sample_rate must be provided when using input_tensor or input_tensors")
            original_sr = getattr(self._cfg, "sr", None)
            self._cfg.sr = sample_rate

        data = self._prepare_data(
            input_path,
            input_dir,
            input_tensor,
            input_tensors,
            val_list,
            val_list_path,
            predict_dataset,
        )

        if remove_silent_section:
            initial_state = (
                hasattr(self._cfg.dataset, "remove_silent_section")
                and self._cfg.dataset.remove_silent_section
            )
            self._cfg.dataset.remove_silent_section = True
            
        dataset = get_dataset(self._cfg, data, self._cfg.phase)
        if remove_silent_section and (initial_state is not None) and not initial_state:
            self._cfg.dataset.remove_silent_section = False

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        pred = self._predict_impl(dataloader, num_repetitions, device, verbose)

        if original_sr is not None:
            self._cfg.sr = original_sr

        if input_path is not None:
            return float(pred[0])
        elif input_tensor is not None:
            return float(pred[0])
        elif input_tensors is not None:
            return [float(p) for p in pred]
        else:
            return [
                {"file_path": d.file_path.as_posix() if d.file_path else "tensor_input", "predicted_mos": float(p)}
                for d, p in zip(data, pred)
            ]

    def _prepare_data(
        self,
        input_path: Path | str | None,
        input_dir: Path | str | None,
        input_tensor: torch.Tensor | None,
        input_tensors: list[torch.Tensor] | None,
        val_list: list[str] | None,
        val_list_path: Path | str | None,
        predict_dataset: str,
    ) -> list[DatasetSchema]:
        """
        Prepare the data for prediction from either file paths or tensors.
        """
        if val_list_path is not None:
            if val_list:
                warnings.warn(
                    "Both `val_list` and `val_list_path` are provided. "
                    "The union of the two will be used."
                )
            if val_list is None:
                val_list = []
            if isinstance(val_list_path, str):
                val_list_path = Path(val_list_path)
            if not val_list_path.exists():
                raise FileNotFoundError(f"File not found: {val_list_path}")
            with open(val_list_path, "r") as f:
                val_list.extend(f.read().splitlines())

        if input_path is not None or input_dir is not None:
            if isinstance(input_path, str) and input_path is not None:
                input_path = Path(input_path)
            if isinstance(input_dir, str) and input_dir is not None:
                input_dir = Path(input_dir)
                
            if input_path is not None and not input_path.exists():
                raise FileNotFoundError(f"File not found: {input_path}")
            if input_dir is not None and not input_dir.exists():
                raise FileNotFoundError(f"Directory not found: {input_dir}")
                
            res: list[DatasetSchema]
            if input_path is not None:
                res = [
                    DatasetSchema(
                        file_path=input_path,
                        dataset=predict_dataset,
                    )
                ]
            else:
                res = [
                    DatasetSchema(
                        file_path=p,
                        dataset=predict_dataset,
                    )
                    for p in sorted(input_dir.glob("*.wav"))
                ]
                if not res:
                    raise ValueError(f"No wav files found in {input_dir}")
                    
            if val_list is not None:
                val_list = [d.replace(".wav", "") for d in val_list]
                res = [
                    d
                    for d in res
                    if d.file_path.as_posix().split("/")[-1].replace(".wav", "")
                    in set(val_list)
                ]
                
        elif input_tensor is not None:
            res = [
                DatasetSchema(
                    dataset=predict_dataset,
                    audio_tensor=input_tensor,
                )
            ]
        elif input_tensors is not None:
            res = [
                DatasetSchema(
                    dataset=predict_dataset,
                    audio_tensor=tensor,
                )
                for tensor in input_tensors
            ]
        else:
            raise ValueError("No valid input provided")

        if not res:
            raise ValueError(
                f"None of the data were found in the validation list: {val_list_path}"
            )
            
        return res

    def _predict_impl(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_repetitions: int,
        device,
        verbose: bool,
    ) -> np.ndarray:
        """
        Internal implementation of the prediction logic.
        """
        self.eval().to(device)
        res = 0.0
        for i in range(num_repetitions):
            pred = []
            pbar = tqdm(
                dataloader,
                disable=not verbose,
                total=len(dataloader),
                desc=(
                    f"Predicting [{i + 1}/{num_repetitions}]: "
                    if num_repetitions > 1
                    else "Predicting: "
                ),
            )
            with torch.no_grad():
                for t in pbar:
                    x = t[:-1]
                    x = [t.to(device, non_blocking=True) for t in x]
                    with torch.amp.autocast('cuda'):
                        output = self.__call__(*x).squeeze(1)
                    pred.append(output.cpu().numpy())
            res += np.concatenate(pred) / num_repetitions
        assert isinstance(res, np.ndarray)
        return res
