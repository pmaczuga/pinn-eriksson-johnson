# -*- coding: utf-8 -*-

import math
from typing import Any
import torch

from .pinn_core import PINN, dfdx, dfdy, f

class Loss:
    def __init__(self, 
                 x: torch.Tensor, 
                 y: torch.Tensor,
                 x_around: torch.Tensor,
                 y_around: torch.Tensor,
                 x_left: torch.Tensor,
                 y_left: torch.Tensor,
                 epsilon: float):
        self.x = x
        self.y = y
        self.x_around = x_around
        self.y_around = y_around
        self.x_left = x_left
        self.y_left = y_left
        self.epsilon = epsilon

    def __call__(self, pinn: PINN) -> torch.Tensor:
        return self.compute_loss(pinn, self.x, self.y, self.epsilon)

    def f_inter_loss(self, x, y, pinn, epsilon) -> torch.Tensor:
        return (
            dfdx(pinn, x, y)
            - epsilon * dfdx(pinn, x, y, order=2)
            - epsilon * dfdy(pinn, x, y, order=2)
        )

    def compute_loss(
        self, pinn: PINN, x: torch.Tensor, y: torch.Tensor, epsilon: float
    ) -> torch.Tensor:
        """Compute the full loss function as interior loss + boundary loss
        This custom loss function is fully defined with differentiable tensors therefore
        the .backward() method can be applied to it
        """

        # PDE residual
        interior_loss = self.f_inter_loss(x, y, pinn, epsilon)

        # Zero dirichlet
        boundary_loss_around = f(pinn, self.x_around, self.y_around)

        # sin(pi * y) on the left (x = 0)
        boundary_loss_left = f(pinn, self.x_left, self.y_left) - torch.sin(math.pi * self.y_left)

        final_loss = (
            interior_loss.pow(2).mean()
            + boundary_loss_left.pow(2).mean()
            + boundary_loss_around.pow(2).mean()
        )

        return final_loss
