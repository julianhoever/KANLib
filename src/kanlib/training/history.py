import csv
from pathlib import Path


class History:
    def __init__(self) -> None:
        self._history: dict[str, list[float]] = dict(
            epoch=[], train_loss=[], val_loss=[]
        )

    def __len__(self) -> int:
        return len(self["epoch"])

    def __getitem__(self, key: str) -> list[float]:
        return self._history[key]

    def update(self, epoch: int, train_loss: float, val_loss: float) -> None:
        self["epoch"].append(epoch)
        self["train_loss"].append(train_loss)
        self["val_loss"].append(val_loss)

    def save_csv(self, destination: Path) -> None:
        num_epochs = len(self["epoch"])
        history = dict(
            epoch=self["epoch"] * 2,
            split=["training"] * num_epochs + ["validation"] * num_epochs,
            loss=self["train_loss"] + self["val_loss"],
        )

        with open(destination, "w", newline="") as out_file:
            writer = csv.DictWriter(out_file, fieldnames=["epoch", "split", "loss"])
            writer.writeheader()
            for idx in range(num_epochs * 2):
                row = {name: values[idx] for name, values in history.items()}
                writer.writerow(row)

    def merge(self, new_history: "History") -> None:
        next_epoch = 1 if len(self) == 0 else int(self["epoch"][-1]) + 1
        rewritten_epochs = range(next_epoch, next_epoch + len(new_history))
        self["epoch"].extend(rewritten_epochs)
        self["train_loss"].extend(new_history["train_loss"])
        self["val_loss"].extend(new_history["val_loss"])
