from collections.abc import Callable
from functools import partial
from typing import Any, Protocol, runtime_checkable

import torch
from torch.utils.data import Dataset

from .history import History
from .training_loop import train as train_without_refinement


def train(
    model: torch.nn.Module,
    ds_train: Dataset,
    ds_val: Dataset,
    epochs_per_grid_size: int,
    new_grid_sizes: list[int],
    batch_size: int,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer_cls: type[torch.optim.Optimizer],
    optimizer_kwargs: dict[str, Any],
    load_best: bool,
    device: torch.device,
) -> History:
    train_on_fixed_grid = partial(
        train_without_refinement,
        model=model,
        ds_train=ds_train,
        ds_val=ds_val,
        epochs=epochs_per_grid_size,
        batch_size=batch_size,
        loss_fn=loss_fn,
        optimizer_cls=optimizer_cls,
        optimizer_kwargs=optimizer_kwargs,
        device=device,
    )
    grid_sizes = [0] + new_grid_sizes
    history = History()

    for step_idx, grid_size in enumerate(grid_sizes):
        if step_idx > 0:
            _refine_grid(model, grid_size)

        new_history = train_on_fixed_grid(
            load_best=load_best and step_idx == len(grid_sizes) - 1
        )

        history.merge(new_history)

    return history


def _refine_grid(model: torch.nn.Module, grid_size: int) -> None:
    @runtime_checkable
    class RefineableLayer(Protocol):
        def refine_grid(self, new_grid_size: int) -> None: ...

    def refine(module: torch.nn.Module) -> None:
        if isinstance(module, RefineableLayer):
            module.refine_grid(grid_size)

    model.apply(refine)
