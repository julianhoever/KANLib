import marimo

__generated_with = "0.18.3"
app = marimo.App(width="medium")

with app.setup:
    from functools import partial

    import marimo as mo
    import matplotlib.pyplot as plt
    import torch

    from kanlib.nn.bspline import BSplineBasis
    from kanlib.nn.gaussian_rbf import GaussianRbfBasis


@app.cell
def _():
    basis_fn = mo.ui.radio(
        options={
            "BSpline": partial(BSplineBasis, spline_order=3),
            "GRBF": GaussianRbfBasis,
        },
        value="GRBF",
        label="Basis",
        inline=True,
    )
    grid_size = mo.ui.number(value=5, label="Grid Size")
    start = mo.ui.number(value=-1, label="Start")
    stop = mo.ui.number(value=1, label="Stop")
    mo.vstack([basis_fn, grid_size, start, stop])
    return basis_fn, grid_size, start, stop


@app.cell
def _(basis_fn, grid_size, start, stop):
    inputs = torch.linspace(start.value, stop.value, 1000).unsqueeze(dim=-1)
    basis = basis_fn.value(
        grid_size=grid_size.value, spline_range=torch.tensor([[-1, 1]])
    )

    grid_update = basis.grid_update_from_samples(inputs)
    basis.update_grid(grid_update)
    return basis, inputs


@app.cell
def _(basis, inputs):
    basis_values = basis(inputs).squeeze(dim=-2)
    plt.plot(inputs.squeeze(dim=-1), basis_values)
    plt.plot(inputs.squeeze(dim=-1), basis_values.sum(dim=-1), "k--")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
