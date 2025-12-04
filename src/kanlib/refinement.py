import torch

from kanlib.nn.spline_basis import SplineBasis


def compute_refined_coefficients(
    basis_coarse: SplineBasis, basis_fine: SplineBasis, coeff_coarse: torch.Tensor
) -> torch.Tensor:
    if basis_fine.grid_size < basis_coarse.grid_size:
        raise ValueError("Finer grid size must be larger than the coarse grid size.")

    if basis_coarse.num_features != basis_fine.num_features:
        raise ValueError("Number of features must be equal for coarse and fine basis.")

    if coeff_coarse.shape[-2:] != (
        basis_coarse.num_features,
        basis_coarse.num_basis_functions,
    ):
        raise ValueError(
            "`coeff_coarse` must be of shape `(*, num_features, num_basis_functions_coarse)`."
        )

    reshaped_coeff_coarse = coeff_coarse.view(-1, *coeff_coarse.shape[-2:])
    reshaped_coeff_coarse = reshaped_coeff_coarse.permute(1, 2, 0)
    reshaped_coeff_coarse = reshaped_coeff_coarse.unsqueeze(dim=1)

    # reshaped_coeff_coarse: (num_features, 1, num_basis_fn_coarse, -1)

    basis_values_coarse = basis_coarse(basis_fine.grid.t())
    basis_values_coarse = basis_values_coarse.transpose(0, 1)
    basis_values_coarse = basis_values_coarse.unsqueeze(dim=-1)

    # basis_values_coarse: (num_features, gird_points_fine, num_basis_fn_coarse, 1)

    y_fine = torch.sum(basis_values_coarse * reshaped_coeff_coarse, dim=-2)

    # y_fine: (num_features, gird_points_fine, -1)

    basis_values_fine = basis_fine(basis_fine.grid.t())
    basis_values_fine = basis_values_fine.transpose(0, 1)

    # basis_values_fine: (num_features, gird_points_fine, num_basis_fn_fine)

    coeff_fine = torch.linalg.lstsq(basis_values_fine, y_fine).solution
    coeff_fine = coeff_fine.permute(2, 0, 1)
    coeff_fine = coeff_fine.view(
        *coeff_coarse.shape[:-1], basis_fine.num_basis_functions
    )

    # coeff_fine: (*, num_features, num_basis_fn_fine)

    return coeff_fine
