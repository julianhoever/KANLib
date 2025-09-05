import torch

from kanlib.nn.spline_basis import SplineBasis


def compute_spline(
    basis: SplineBasis, coefficient: torch.Tensor, inputs: torch.Tensor
) -> torch.Tensor:
    assert inputs.dim() == 1
    assert coefficient.shape[-1] == basis.num_basis_functions

    inputs = inputs.unsqueeze(dim=-1)
    flat_coeff = coefficient.view(-1, basis.num_basis_functions)
    flat_spline = torch.sum(basis(inputs) * flat_coeff, dim=-1).transpose(0, 1)
    return flat_spline.view(*coefficient.shape[:-1], -1)


def compute_refined_coefficients(
    basis_coarse: SplineBasis, basis_fine: SplineBasis, coeff_coarse: torch.Tensor
) -> torch.Tensor:
    if basis_fine.grid_size <= basis_coarse.grid_size:
        raise ValueError("Finer grid size must be larger than the coarse grid size.")

    reshaped_coeff_coarse = coeff_coarse.view(-1, 1, basis_coarse.num_basis_functions)
    reshaped_basis_values = basis_coarse(basis_fine.grid).unsqueeze(dim=0)

    y_fine = torch.sum(reshaped_basis_values * reshaped_coeff_coarse, dim=-1).t()
    basis_values_fine = basis_fine(basis_fine.grid)

    coeff_fine = torch.linalg.lstsq(basis_values_fine, y_fine).solution
    coeff_fine = coeff_fine.t().view(
        *coeff_coarse.shape[:-1], basis_fine.num_basis_functions
    )

    return coeff_fine
