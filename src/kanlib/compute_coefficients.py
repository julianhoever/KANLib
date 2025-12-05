import torch


def compute_coefficients(
    original_coefficients: torch.Tensor,
    original_basis_values: torch.Tensor,
    target_basis_values: torch.Tensor,
) -> torch.Tensor:
    if original_basis_values.ndim != target_basis_values.ndim:
        raise ValueError(
            "`[original/target]_basis_values` must have the same number of dimensions."
        )
    if original_basis_values.ndim < 2:
        raise ValueError(
            "`[original/target]_basis_values` must have at least 2 dimensions. "
            "Expected shape: `(*, num_features, [original/target]_num_basis_functions)`"
        )
    if original_basis_values.shape[:-1] != target_basis_values.shape[:-1]:
        raise ValueError(
            "`[original/target]_basis_values` must have the same shape, except the last dimension. "
            "Expected shape: `(*, num_features, [original/target]_num_basis_functions)`"
        )
    if original_coefficients.ndim < 2:
        raise ValueError(
            "`original_coefficients` must have at least 2 dimensions. "
            "Expected shape: `(*, num_features, original_num_basis_functions)`"
        )
    if original_coefficients.shape[-2:] != original_basis_values.shape[-2:]:
        raise ValueError(
            "`original_coefficients` must be of shape `(*, num_features, original_num_basis_functions)`."
        )

    *_, num_features, num_basis_fn_orig = original_basis_values.shape
    *_, num_basis_fn_targ = target_basis_values.shape

    orig_coeff = original_coefficients.view(-1, num_features, num_basis_fn_orig)
    orig_coeff = orig_coeff.permute(1, 2, 0)
    orig_coeff = orig_coeff.unsqueeze(dim=1)

    # reshaped_coeff: (num_features, 1, num_basis_fn_orig, -1)

    orig_values = original_basis_values.view(-1, num_features, num_basis_fn_orig)
    orig_values = orig_values.transpose(0, 1)
    orig_values = orig_values.unsqueeze(dim=-1)

    # orig_values: (num_features, num_values, num_basis_fn_orig, 1)

    y_spline = torch.sum(orig_values * orig_coeff, dim=-2)

    # y_spline: (num_features, num_values, -1)

    targ_values = target_basis_values.view(-1, num_features, num_basis_fn_targ)
    targ_values = targ_values.transpose(0, 1)

    # targ_values: (num_features, num_values, num_basis_fn_targ)

    targ_coeff = torch.linalg.lstsq(targ_values, y_spline).solution
    targ_coeff = targ_coeff.permute(2, 0, 1)
    targ_coeff = targ_coeff.view(*original_coefficients.shape[:-1], num_basis_fn_targ)

    # targ_coeff: (*, num_features, num_basis_fn_targ)

    return targ_coeff
