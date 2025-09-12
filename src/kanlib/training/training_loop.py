from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .history import History
from .model_checkpoint import ModelCheckpoint


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
    device: torch.device,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
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

    model.to(device)
    optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)
    model_ckpt = ModelCheckpoint(model)
    history = History()

    with tqdm(total=epochs) as pbar:
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0

            for samples, labels in dl_train:
                samples = samples.to(device)
                labels = labels.to(device)

                model.zero_grad()

                predictions = model(samples)
                loss = loss_fn(predictions, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            train_loss = running_loss / len(dl_train)

            model.eval()
            running_loss = 0.0

            with torch.no_grad():
                for samples, labels in dl_val:
                    samples = samples.to(device)
                    labels = labels.to(device)

                    predictions = model(samples)
                    loss = loss_fn(predictions, labels)

                    running_loss += loss.item()

            val_loss = running_loss / len(dl_val)

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
