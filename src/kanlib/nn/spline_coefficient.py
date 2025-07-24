import torch
from torch.nn import Parameter


class SplineCoefficient(torch.nn.Module):
    def __init__(
        self,
        shape: tuple[int, ...],
        init_coefficient_std: float,
        use_coefficient_weight: bool,
    ) -> None:
        super().__init__()
        self.unweighted: torch.nn.Parameter
        self.weight: torch.nn.Parameter | None

        self.register_parameter(
            "unweighted", Parameter(_init_coeff(shape, init_coefficient_std))
        )

        if use_coefficient_weight:
            self.register_parameter(
                "weight", Parameter(_init_coeff_weight(self.unweighted))
            )
        else:
            self.weight = None

    @property
    def coefficient(self) -> torch.Tensor:
        return (
            self.unweighted * self.weight
            if self.weight is not None
            else self.unweighted
        )

    @coefficient.setter
    def coefficient(self, value: torch.Tensor) -> None:
        self.unweighted.data = value
        if self.weight is not None:
            self.weight.data = _init_coeff_weight(self.unweighted).to(
                self.weight.device
            )


def _init_coeff(shape: tuple[int, ...], init_coefficient_std: float) -> torch.Tensor:
    coefficient = torch.nn.Parameter(torch.zeros(shape))
    torch.nn.init.normal_(coefficient, mean=0, std=init_coefficient_std)
    return coefficient


def _init_coeff_weight(coefficient: torch.Tensor) -> torch.Tensor:
    return torch.ones((*coefficient.shape[:-1], 1))
