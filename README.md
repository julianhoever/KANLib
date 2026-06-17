# KANLib

KANLib is a PyTorch library of modular, extensible neural-network modules for
**Kolmogorov-Arnold Networks (KANs)**. It is meant to simplify development and
experimentation with KANs across different domains. The implementation is designed to be modular and extensible, allowing for the easy integration of new features.

## Features

- **Basis functions**
  - B-splines (`BSplineBasis`)
  - Gaussian radial basis functions (`GaussianRbfBasis`)
- **Neural network modules** (available for every basis function)
  - `Linear`
  - `Conv1d`, `Conv2d`
- **Adaptive grids** — update the basis grid from data statistics during training.
- **Grid refinement** — increase the grid size of each layer independently while training.
- **Spline visualization** — plot the learned activation of any KAN layer.
- **Predefined training loops** — `train` / `train_with_checkpoints` for common regression and classification tasks, with optional grid updates and refinement.

## Installation

KANLib requires **Python >= 3.13**. It is not published on PyPI; install it
directly from the repository:

```bash
pip install git+https://github.com/julianhoever/KANLib.git
```

## Quick start

KAN layers are regular `torch.nn.Module`s, so they compose with anything in
PyTorch:

```python
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset

from kanlib.nn.bspline import Linear
from kanlib.training import train
from kanlib.visualization import spline_curve

model = torch.nn.Sequential(
    Linear(in_features=2, out_features=8, grid_size=5, spline_order=3),
    Linear(in_features=8, out_features=1, grid_size=5, spline_order=3),
)

# Toy regression data within the default spline range [-1, 1]
x = torch.rand(512, 2) * 2 - 1
y = (x**2).sum(dim=1, keepdim=True)
ds_train = ds_val = TensorDataset(x, y)

history = train(
    model=model,
    ds_train=ds_train,
    ds_val=ds_val,
    epochs=50,
    batch_size=64,
    loss_fn=torch.nn.MSELoss(),
    optimizer_cls=torch.optim.Adam,
    optimizer_kwargs={"lr": 1e-3},
)

# Inspect the learned activation connecting input feature 0 to output feature 0
xs, ys = spline_curve(model[0], spline_index=(0, 0))
plt.plot(xs, ys)
plt.show()
```

Swap `from kanlib.nn.bspline import Linear` for
`from kanlib.nn.gaussian_rbf import Linear` to use Gaussian RBF layers instead.

## Project structure

An incomplete overview of the most important project files and packages:

```
src/kanlib/
├── nn/                     # Neural-network modules used to build KANs
│   ├── kan_base_layer.py   # KANBaseLayer ABC + layer/param/basis specs
│   ├── base_modules/       # Basis-agnostic layer topologies (LinearBase, ConvBase)
│   ├── spline_basis/       # SplineBasis ABC + AdaptiveGrid mixin
│   ├── bspline/            # B-spline basis + Layers
│   └── gaussian_rbf/       # Gaussian RBF basis + Layers
├── training/               # Predefined training loops for KANs
└── visualization/          # Utilities for plotting learned splines
tests/                      # pytest suite mirroring the package layout
```

## Extending KANLib

KANLib separates **basis functions** (how an input is expanded into basis
values) from **layer topologies** (how those values are combined — linear,
convolutional, ...). Each concrete layer is a thin wrapper that injects a basis
into a basis-agnostic base layer, so the two can be extended independently.

### Creating a new basis function

Subclass `SplineBasis` from `kanlib.nn.spline_basis`. In `__init__`, call
`super().__init__(grid_size, spline_range, initialize_grid)`, where
`initialize_grid` is a zero-argument callable returning the grid buffer of shape
`(num_features, num_grid_points)`. Then implement:

- the `num_basis_functions` property, and
- `forward(x)`, mapping `x` of shape `(*, num_features)` to
  `(*, num_features, num_basis_functions)`.

To support adaptive grids and grid refinement, also subclass `AdaptiveGrid`
(`kanlib.nn.spline_basis`) and implement `grid_update_from_samples(x) ->
GridUpdate`. The built-in `BSplineBasis` and `GaussianRbfBasis` are complete
reference implementations.

### Creating a new layer

To pair a basis with an existing topology, subclass one of the base layers —
`LinearBase` (`kanlib.nn.base_modules.linear`), `Conv1dBase` or `Conv2dBase`
(`kanlib.nn.base_modules.convolution`) — and pass a `BasisSpec`
(`kanlib.nn.kan_base_layer`) whose `basis_factory` is your basis class (use
`functools.partial` to bind extra arguments). The built-in `bspline.Linear` and
`bspline.Conv1d` layers follow exactly this pattern.

For a genuinely new layer type, subclass `KANBaseLayer`
(`kanlib.nn.kan_base_layer`): forward a `LayerSpec`, `param_specs` (built with
`default_param_specs`), and `basis_spec` to `super().__init__`, then implement
`spline_forward` and `residual_forward` (the base `forward` combines them with
the optional output bias). `base_modules/linear.py` is the canonical example.

## Credits

This project is influenced by essential works in the field of Kolmogorov-Arnold networks:

- **KANs: Kolmogorov-Arnold Networks ([Paper](https://arxiv.org/abs/2404.19756) | [Implementation](https://github.com/KindXiaoming/pykan))**
    - Original paper and implementation introducing KANs.
    - ***Contribution to this work**: Basic understanding of KANs and their implementation.*
- **Kolmogorov-Arnold Networks are Radial Basis Function Networks ([Paper](https://arxiv.org/abs/2405.06721) | [Implementation](https://github.com/ZiyaoLi/fast-kan))**
    - Implementation of KANs using gaussian radial basis functions.
    - ***Contribution to this work**: Inspiration for KANs with radial basis functions and how to implement them efficiently (without copying code from the repository).*
- **Efficient-KAN: An Efficient Implementation of Kolmogorov-Arnold Network ([Implementation](https://github.com/Blealtan/efficient-kan))**
    - An implementation of KANs with B-splines that focuses on performance improvements.
    - ***Contribution to this work**: Use of the efficient B-spline basis function implementation of Efficient-KAN ([here](https://github.com/julianhoever/KANLib/blob/b76c6a47ec91acdfe4f208ad0498e4a9a04dbb21/src/kanlib/nn/bspline/bspline_basis.py#L23-L38)).*

## Citation

```bibtex
@article{hoever2026kanlib,
      title={KANLib -- An Modular, Extensible and Fast Kolmogorov-Arnold Network Implementation}, 
      author={Julian Hoever and Gregor Schiele},
      year={2026},
      eprint={2606.17927},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.17927}, 
}
```

## License

KANLib is released under the [MIT License](LICENSE).

