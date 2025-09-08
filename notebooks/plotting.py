import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from functools import partial
    from collections.abc import Callable

    import torch
    from torch.utils.data import TensorDataset
    import matplotlib.pyplot as plt

    from kanlib.nn.bspline import Linear
    from kanlib.training.training_loop import train
    from kanlib.spline_plotting import plot_spline, linear_spline_index
    return (
        Callable,
        Linear,
        TensorDataset,
        linear_spline_index,
        partial,
        plot_spline,
        plt,
        torch,
        train,
    )


@app.cell
def _(Callable, TensorDataset, partial, torch):
    def function_dataset(
        function: Callable[[torch.Tensor], torch.Tensor],
        num_samples: int,
        var_ranges: list[tuple[int, int]]
    ) -> TensorDataset:
        def apply_function(inputs: torch.Tensor) -> torch.Tensor:
            single_inputs = inputs.unbind(dim=0)
            return torch.stack([function(x) for x in single_inputs]).unsqueeze(dim=-1)

        samples = torch.rand((num_samples, len(var_ranges)))
        for i, (var_min, var_max) in enumerate(var_ranges):
            samples[:, i] = samples[:, i] * (var_max - var_min) + var_min

        targets = apply_function(samples)

        return TensorDataset(samples, targets)

    def func(inputs: torch.Tensor) -> torch.Tensor:
        x, y = torch.unbind(inputs)
        return torch.exp(torch.sin(x * torch.pi) + y**2)

    create_ds = partial(function_dataset, function=func, var_ranges=[(-1, 1), (-1, 1)])
    ds_train = create_ds(num_samples=2000)
    ds_val = create_ds(num_samples=400)
    return ds_train, ds_val


@app.cell
def _(Linear, ds_train, ds_val, partial, torch, train):
    linear = partial(
        Linear,
        spline_order=3,
        grid_size=3,
        grid_range=(-1, 1),
        use_base_branch=True,
        use_spline_weight=True,
    )

    model = torch.nn.Sequential(
        linear(in_features=2, out_features=1),
        linear(in_features=1, out_features=1),
    )
    print("Params:", sum(p.numel() for p in model.parameters()))

    def run_training() -> None:
        train(
            model=model,
            ds_train=ds_train,
            ds_val=ds_val,
            epochs=100,
            batch_size=32,
            loss_fn=torch.nn.MSELoss(),
            optimizer_cls=torch.optim.Adam,
            optimizer_kwargs=dict(lr=1e-3),
            load_best=False,
            device=torch.device("cpu")
        )
        model.eval()

    run_training()
    return model, run_training


@app.cell
def _(linear_spline_index, model, plot_spline, plt):
    def plot() -> None:
        fig, axs = plt.subplots(nrows=2, ncols=2, tight_layout=True)
        plot_spline(
            layer=model[0],
            spline_index=linear_spline_index(model[0], input_idx=0, output_idx=0),
            show_grid=True,
            ax=axs[0, 0]
        )
        plot_spline(
            layer=model[0],
            spline_index=linear_spline_index(model[0], input_idx=1, output_idx=0),
            show_grid=True,
            ax=axs[0, 1]
        )
        plot_spline(
            layer=model[1],
            spline_index=linear_spline_index(model[1], input_idx=0, output_idx=0),
            show_grid=True,
            ax=axs[1, 0]
        )
        return fig

    plot()
    return (plot,)


@app.cell
def _(model, plot, run_training):
    model[0].refine_grid(5)
    run_training()
    plot()
    return


@app.cell
def _(model, plot, run_training):
    model[0].refine_grid(10)
    run_training()
    plot()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
