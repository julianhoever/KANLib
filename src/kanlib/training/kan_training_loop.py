from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from math import floor
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


def train(
    model: torch.nn.Module,
    ds_train: Dataset,
    ds_val: Dataset,
    epochs: int,
    batch_size: int,
    loss_fn: torch.nn.Module,
    optimizer_cls: type[torch.optim.Optimizer],
    optimizer_kwargs: dict[str, Any],
    num_grid_updates: int = 0,
    start_grid_updates: int = 0,
    stop_grid_updates: int = -1,
    grid_size_refinements: Optional[list[int]] = None,
    load_best: bool = False,
    device: torch.device = torch.device("cpu"),
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
) -> History:
    return _train(
        model=model,
        ds_train=ds_train,
        ds_val=ds_val,
        epochs=epochs,
        batch_size=batch_size,
        loss_fn=loss_fn,
        optimizer_cls=optimizer_cls,
        optimizer_kwargs=optimizer_kwargs,
        load_best=load_best,
        num_grid_updates=num_grid_updates,
        start_grid_updates=start_grid_updates,
        stop_grid_updates=stop_grid_updates,
        grid_size_refinements=grid_size_refinements,
        device=device,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        return_checkpoints=False,
    )


def train_with_checkpoints(
    model: torch.nn.Module,
    ds_train: Dataset,
    ds_val: Dataset,
    epochs: int,
    batch_size: int,
    loss_fn: torch.nn.Module,
    optimizer_cls: type[torch.optim.Optimizer],
    optimizer_kwargs: dict[str, Any],
    num_grid_updates: int = 0,
    start_grid_updates: int = 0,
    stop_grid_updates: int = -1,
    grid_size_refinements: Optional[list[int]] = None,
    load_best: bool = False,
    device: torch.device = torch.device("cpu"),
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
) -> tuple[History, list[Checkpoint]]:
    return _train(
        model=model,
        ds_train=ds_train,
        ds_val=ds_val,
        epochs=epochs,
        batch_size=batch_size,
        loss_fn=loss_fn,
        optimizer_cls=optimizer_cls,
        optimizer_kwargs=optimizer_kwargs,
        load_best=load_best,
        num_grid_updates=num_grid_updates,
        start_grid_updates=start_grid_updates,
        stop_grid_updates=stop_grid_updates,
        grid_size_refinements=grid_size_refinements,
        device=device,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        return_checkpoints=True,
    )


@overload
def _train(
    model: torch.nn.Module,
    ds_train: Dataset,
    ds_val: Dataset,
    epochs: int,
    batch_size: int,
    loss_fn: torch.nn.Module,
    optimizer_cls: type[torch.optim.Optimizer],
    optimizer_kwargs: dict[str, Any],
    num_grid_updates: int,
    start_grid_updates: int,
    stop_grid_updates: int,
    grid_size_refinements: Optional[list[int]],
    load_best: bool,
    device: torch.device,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    return_checkpoints: Literal[False],
) -> History: ...


@overload
def _train(
    model: torch.nn.Module,
    ds_train: Dataset,
    ds_val: Dataset,
    epochs: int,
    batch_size: int,
    loss_fn: torch.nn.Module,
    optimizer_cls: type[torch.optim.Optimizer],
    optimizer_kwargs: dict[str, Any],
    num_grid_updates: int,
    start_grid_updates: int,
    stop_grid_updates: int,
    grid_size_refinements: Optional[list[int]],
    load_best: bool,
    device: torch.device,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    return_checkpoints: Literal[True],
) -> tuple[History, list[Checkpoint]]: ...


def _train(
    model: torch.nn.Module,
    ds_train: Dataset,
    ds_val: Dataset,
    epochs: int,
    batch_size: int,
    loss_fn: torch.nn.Module,
    optimizer_cls: type[torch.optim.Optimizer],
    optimizer_kwargs: dict[str, Any],
    num_grid_updates: int,
    start_grid_updates: int,
    stop_grid_updates: int,
    grid_size_refinements: Optional[list[int]],
    load_best: bool,
    device: torch.device,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    return_checkpoints: bool,
) -> History | tuple[History, list[Checkpoint]]:
    epochs_to_update_grid = _compute_grid_update_epochs(
        start_epoch=start_grid_updates,
        stop_epoch=epochs if stop_grid_updates == -1 else stop_grid_updates,
        num_updates=num_grid_updates,
    )

    def update_grid_hook(epoch: int, m: torch.nn.Module) -> None:
        if epoch in epochs_to_update_grid:
            indices = torch.randint(len(ds_train), size=(batch_size,))  # type: ignore
            inputs = torch.stack([ds_train[idx][0] for idx in indices]).to(device)
            update_grid(m, inputs)

    train_loop = partial(
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
        on_epoch_starts=update_grid_hook,
    )

    grid_sizes = [0]
    if grid_size_refinements is not None and len(grid_size_refinements) > 0:
        grid_sizes += grid_size_refinements

    history = History()
    checkpoints: list[Checkpoint] = []

    for refinement_step_idx, grid_size in enumerate(grid_sizes):
        if refinement_step_idx > 0:
            refine_grid(model, grid_size)

        untrained_state = _model_state(model)

        new_history = train_loop(
            load_best=load_best and refinement_step_idx == len(grid_sizes) - 1,
        )
        history.merge(new_history)

        trained_state = _model_state(model)

        if return_checkpoints:
            checkpoints.append(Checkpoint(untrained_state, trained_state))

    return (history, checkpoints) if return_checkpoints else history


def _model_state(model: torch.nn.Module) -> ModelState:
    return deepcopy(model.state_dict())


def _compute_grid_update_epochs(
    start_epoch: int, stop_epoch: int, num_updates: int
) -> set[int]:
    if num_updates == 0:
        return set()
    step_size = (stop_epoch - start_epoch) / num_updates
    return {start_epoch + floor(step_size * i) for i in range(num_updates)}
