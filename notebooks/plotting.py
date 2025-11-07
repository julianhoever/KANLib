import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from collections.abc import Callable
    from functools import partial

    import matplotlib.pyplot as plt
    import torch
    from torch.utils.data import TensorDataset

    from kanlib.nn.bspline import Linear
    from kanlib.spline_plotting import linear_spline_index, plot_spline
    from kanlib.training.training_loop import train

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
        var_ranges: list[tuple[int, int]],
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
        x, y, z = torch.unbind(inputs)
        return x * torch.exp(x**2 * torch.exp(z**2))

    create_ds = partial(
        function_dataset, function=func, var_ranges=[(-1, 1), (-1, 1), (-1, 1)]
    )
    ds_train = create_ds(num_samples=2000)
    ds_val = create_ds(num_samples=400)
    return ds_train, ds_val


@app.cell
def _(Linear, ds_train, ds_val, partial, torch, train):
    linear = partial(
        Linear,
        spline_order=3,
        grid_size=5,
        grid_range=(-1, 1),
        use_residual_branch=True,
        use_spline_weight=True,
    )

    model = torch.nn.Sequential(
        linear(in_features=3, out_features=3),
        linear(in_features=3, out_features=1),
    )
    print("Params:", sum(p.numel() for p in model.parameters()))

    train(
        model=model,
        ds_train=ds_train,
        ds_val=ds_val,
        epochs=400,
        batch_size=32,
        loss_fn=torch.nn.MSELoss(),
        optimizer_cls=torch.optim.Adam,
        optimizer_kwargs=dict(lr=1e-3),
        load_best=False,
        device=torch.device("cpu"),
    )
    _ = model.eval()
    return (model,)


@app.cell
def _(linear_spline_index, model, plot_spline, plt):
    fig, axs = plt.subplots(
        nrows=len(model),
        ncols=max(l.in_features * l.out_features for l in model),
        figsize=(10, 5),
        sharey=True,
        tight_layout=True,
    )

    for layer_idx in range(len(model)):
        for out_idx in range(model[layer_idx].out_features):
            for in_idx in range(model[layer_idx].in_features):
                spline_idx = linear_spline_index(model[layer_idx], in_idx, out_idx)
                plot_spline(
                    layer=model[layer_idx],
                    spline_index=spline_idx,
                    show_grid=True,
                    ax=axs[layer_idx, spline_idx],
                )

    for ax in axs.flatten():
        ax.set_box_aspect(1)

    fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
