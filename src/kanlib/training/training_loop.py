from collections.abc import Callable
from functools import partial

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .history import History
from .model_checkpoint import ModelCheckpoint

type _OnEpochStartsHook = Callable[[int, torch.nn.Module], None]
type OptimizerFactory = Callable[[torch.nn.Module], torch.optim.Optimizer]
type LRSchedulerFactory = Callable[
    [torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler
]


def train(
    *,
    model: torch.nn.Module,
    ds_train: Dataset,
    ds_val: Dataset,
    epochs: int,
    batch_size: int,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer_factory: OptimizerFactory,
    lr_scheduler_factory: LRSchedulerFactory | None = None,
    load_best: bool = False,
    device: torch.device | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    on_epoch_starts: _OnEpochStartsHook | None = None,
) -> History:
    dataloader = partial(
        DataLoader,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    dl_train = dataloader(ds_train, shuffle=True)
    dl_val = dataloader(ds_val, shuffle=False)

    if device is None:
        device = torch.device("cpu")

    model.to(device)
    optimizer = optimizer_factory(model)
    lr_scheduler = (
        lr_scheduler_factory(optimizer) if lr_scheduler_factory is not None else None
    )
    model_ckpt = ModelCheckpoint(model)
    history = History()

    with tqdm(total=epochs) as pbar:
        for epoch in range(1, epochs + 1):
            model.train()

            if on_epoch_starts is not None:
                on_epoch_starts(epoch, model)

            running_loss = 0.0

            for samples, targets in dl_train:
                samples = samples.to(device)
                targets = targets.to(device)

                def closure() -> float:
                    optimizer.zero_grad()

                    outputs = model(samples)
                    loss = loss_fn(outputs, targets)
                    loss.backward()

                    return loss.item()

                running_loss += optimizer.step(closure)

            train_loss = running_loss / len(dl_train)

            model.eval()
            running_loss = 0.0

            with torch.no_grad():
                for samples, targets in dl_val:
                    samples = samples.to(device)
                    targets = targets.to(device)

                    outputs = model(samples)
                    running_loss += loss_fn(outputs, targets).item()

            val_loss = running_loss / len(dl_val)

            if lr_scheduler is not None:
                if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(val_loss)
                else:
                    lr_scheduler.step()

            history.update(epoch, train_loss, val_loss)
            model_ckpt.update(val_loss)

            pbar.update(1)
            pbar.set_postfix_str(_epoch_info(history=history), refresh=True)

    if load_best:
        model_ckpt.load_best()

    return history


def _epoch_info(history: History) -> str:
    def get(key: str) -> float:
        return history[key][-1]

    return f"train_loss: {get('train_loss'):.4f}, val_loss: {get('val_loss'):.4f}"
