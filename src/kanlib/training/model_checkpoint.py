from copy import deepcopy
from typing import Any

import torch


class ModelCheckpoint:
    def __init__(self, model: torch.nn.Module) -> None:
        self._model = model
        self._best_loss = float("inf")
        self._best_checkpoint: dict[str, Any] | None = None

    def update(self, loss: float) -> None:
        if loss < self._best_loss:
            self._best_loss = loss
            self._best_checkpoint = self._current_model_checkpoint()

    def load_best(self) -> None:
        if self._best_checkpoint is None:
            raise RuntimeError("Call `update` before calling `load_best`.")

        self._model.load_state_dict(self._best_checkpoint)

    def _current_model_checkpoint(self) -> dict[str, Any]:
        return deepcopy(self._model.state_dict())
