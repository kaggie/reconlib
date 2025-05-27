import abc
import torch

class Regularizer(abc.ABC, torch.nn.Module):
    """
    Abstract base class for regularizers.

    Regularizers should implement methods to compute their value (cost)
    and their proximal operator.
    """
    def __init__(self):
        super().__init__() # Call __init__ for torch.nn.Module

    @abc.abstractmethod
    def value(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the value of the regularization term R(x).

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: A scalar tensor representing the value of R(x).
        """
        pass

    @abc.abstractmethod
    def proximal_operator(self, x: torch.Tensor, steplength: float) -> torch.Tensor:
        """
        Computes the proximal operator of the regularization term R.
        The proximal operator is defined as:
        prox_R(x, steplength) = argmin_u { R(u) + (1/(2*steplength)) * ||u - x||_2^2 }

        Args:
            x (torch.Tensor): The input tensor.
            steplength (float): The steplength parameter, often denoted as alpha or lambda*step_size
                                in optimization algorithms (where lambda is the regularization strength).
                                Here, it's the 'gamma' or 'tau' in `argmin_u { R(u) + (1/(2*gamma)) * ||u - x||_2^2 }`.
                                If R(u) is defined as `lambda_reg * R_base(u)`, then the `steplength` argument
                                here often corresponds to `lambda_reg * actual_optimizer_step_size`.

        Returns:
            torch.Tensor: The result of the proximal operation.
        """
        pass

    def forward(self, x: torch.Tensor, steplength: float) -> torch.Tensor:
        """
        Default forward behavior is to apply the proximal operator.
        This allows the regularizer to be used as a torch.nn.Module in some contexts,
        though typically the proximal_operator method will be called directly by solvers.
        """
        return self.proximal_operator(x, steplength)
