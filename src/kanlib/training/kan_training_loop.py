from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, Optional, overload

import torch
from torch.utils.data import Dataset

from kanlib.kan_utils import refine_grid, update_grid

from .history import History
from .training_loop import train as raw_train_loop

type ModelState = dict[str, Any]


@dataclass
class Checkpoint:
    untrained_state: dict[str, Any]
    trained_state: dict[str, Any]


@overload
def train(
    model: torch.nn.Module,
    ds_train: Dataset,
    ds_val: Dataset,
    epochs: int,
    batch_size: int,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer_cls: type[torch.optim.Optimizer],
    optimizer_kwargs: dict[str, Any],
    load_best: bool,
    update_grid_every_nth_epochs: Optional[int],
    refine_grid_sizes: Optional[list[int]],
    device: torch.device,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
) -> History: ...


@overload
def train(
    model: torch.nn.Module,
    ds_train: Dataset,
    ds_val: Dataset,
    epochs: int,
    batch_size: int,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer_cls: type[torch.optim.Optimizer],
    optimizer_kwargs: dict[str, Any],
    load_best: bool,
    update_grid_every_nth_epochs: Optional[int],
    refine_grid_sizes: Optional[list[int]],
    device: torch.device,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    return_checkpoints: Literal[False],
) -> History: ...


@overload
def train(
    model: torch.nn.Module,
    ds_train: Dataset,
    ds_val: Dataset,
    epochs: int,
    batch_size: int,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer_cls: type[torch.optim.Optimizer],
    optimizer_kwargs: dict[str, Any],
    load_best: bool,
    update_grid_every_nth_epochs: Optional[int],
    refine_grid_sizes: Optional[list[int]],
    device: torch.device,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    return_checkpoints: Literal[True],
) -> tuple[History, list[Checkpoint]]: ...


def train(
    model: torch.nn.Module,
    ds_train: Dataset,
    ds_val: Dataset,
    epochs: int,
    batch_size: int,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer_cls: type[torch.optim.Optimizer],
    optimizer_kwargs: dict[str, Any],
    load_best: bool,
    update_grid_every_nth_epochs: Optional[int] = None,
    refine_grid_sizes: Optional[list[int]] = None,
    device: torch.device = torch.device("cpu"),
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    return_checkpoints: bool = False,
) -> History | tuple[History, list[Checkpoint]]:
    if refine_grid_sizes is not None and len(refine_grid_sizes) > 0:
        epochs = epochs // len(refine_grid_sizes)

    def update_grid_hook(epoch: int, m: torch.nn.Module) -> None:
        assert update_grid_every_nth_epochs is not None

        if epoch != 0 and epoch % update_grid_every_nth_epochs == 0:
            indices = torch.randint(len(ds_train), size=(batch_size,))  # type: ignore
            inputs = ds_train[indices][0].to(device)

            update_grid(m, inputs)

    _train = partial(
        raw_train_loop,
        model=model,
        ds_train=ds_train,
        ds_val=ds_val,
        epochs=epochs,
        batch_size=batch_size,
        loss_fn=loss_fn,
        optimizer_cls=optimizer_cls,
        optimizer_kwargs=optimizer_kwargs,
        device=device,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        on_epoch_starts=(
            None if update_grid_every_nth_epochs is None else update_grid_hook
        ),
    )

    grid_sizes = [0]
    if refine_grid_sizes is not None and len(refine_grid_sizes) > 0:
        grid_sizes += refine_grid_sizes

    history = History()
    checkpoints: list[Checkpoint] = []

    for refinement_step_idx, grid_size in enumerate(grid_sizes):
        if refinement_step_idx > 0:
            refine_grid(model, grid_size)

        untrained_state = _model_state(model)

        new_history = _train(
            load_best=load_best and refinement_step_idx == len(grid_sizes) - 1,
        )
        history.merge(new_history)

        trained_state = _model_state(model)

        if return_checkpoints:
            checkpoints.append(Checkpoint(untrained_state, trained_state))

    return (history, checkpoints) if return_checkpoints else history


def _model_state(model: torch.nn.Module) -> ModelState:
    return deepcopy(model.state_dict())
