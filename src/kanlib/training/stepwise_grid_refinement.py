from collections.abc import Callable
from copy import deepcopy
from functools import partial
from typing import Any, Literal, Protocol, overload, runtime_checkable

import torch
from torch.utils.data import Dataset

from .history import History
from .training_loop import train as train_without_refinement

type ModelState = dict[str, Any]


@overload
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
) -> History: ...


@overload
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
    return_model_states: Literal[False],
) -> History: ...


@overload
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
    return_model_states: Literal[True],
) -> tuple[History, list[ModelState]]: ...


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
    return_model_states: bool = False,
) -> History | tuple[History, list[ModelState]]:
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
    model_states: list[ModelState] = []

    for step_idx, grid_size in enumerate(grid_sizes):
        if step_idx > 0:
            _refine_grid(model, grid_size)

        new_history = train_on_fixed_grid(
            load_best=load_best and step_idx == len(grid_sizes) - 1
        )

        history.merge(new_history)

        if return_model_states:
            model_states.append(_model_state(model))

    return (history, model_states) if return_model_states else history


def _refine_grid(model: torch.nn.Module, grid_size: int) -> None:
    @runtime_checkable
    class RefineableLayer(Protocol):
        def refine_grid(self, new_grid_size: int) -> None: ...

    def refine(module: torch.nn.Module) -> None:
        if isinstance(module, RefineableLayer):
            module.refine_grid(grid_size)

    model.apply(refine)


def _model_state(model: torch.nn.Module) -> ModelState:
    return deepcopy(model.state_dict())
