from typing import Protocol, runtime_checkable

import torch

from kanlib.nn.spline_basis import AdaptiveGrid


def refine_grid(module: torch.nn.Module, grid_size: int) -> None:
    @runtime_checkable
    class Refineable(Protocol):
        def refine_grid(self, grid_size: int) -> None: ...

    for child in module.children():
        if isinstance(child, Refineable):
            child.refine_grid(grid_size)


def update_grid(module: torch.nn.Module, x: torch.Tensor) -> None:
    @runtime_checkable
    class Updateable(Protocol):
        def update_grid(self, x: torch.Tensor) -> None: ...

    def hook(m: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        if (
            isinstance(m, Updateable)
            and hasattr(m, "basis")
            and isinstance(m.basis, AdaptiveGrid)
        ):
            m.update_grid(inputs[0])

    hook_handles = [
        child.register_forward_pre_hook(hook) for child in module.children()
    ]

    _ = module(x)

    for handle in hook_handles:
        handle.remove()
