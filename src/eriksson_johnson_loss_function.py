# -*- coding: utf-8 -*-

import math
from typing import Any
import torch

from .pinn_core import PINN, dfdx, dfdy, f

class Loss:
    def __init__(self, x: torch.Tensor, y: torch.Tensor, epsilon: float):
        self.x = x
        self.y = y
        self.epsilon = epsilon
        self.around_x, self.around_y, self.left_x, self.left_y = self._prepare_boundary(x, y)

    def __call__(self, pinn: PINN) -> torch.Tensor:
        return self.compute_loss(pinn, self.x, self.y, self.epsilon)

    def _prepare_boundary(self, x: torch.Tensor, y: torch.Tensor):
        min_x = torch.min(x)
        min_y = torch.min(y)
        max_x = torch.max(x)
        max_y = torch.max(y)

        min_x_i = x == min_x
        min_y_i = y == min_y
        max_x_i = x == max_x
        max_y_i = y == max_y

        left_x = x[min_x_i].reshape(-1, 1)
        left_y = y[min_x_i].reshape(-1, 1)

        right_x = x[max_x_i]
        right_y = y[max_x_i]
        up_x = x[max_y_i]
        up_y = y[max_y_i]
        down_x = x[min_y_i]
        down_y = y[min_y_i]

        around_x = torch.cat((right_x, up_x, down_x)).reshape(-1, 1)
        around_y = torch.cat((right_y, up_y, down_y)).reshape(-1, 1)

        return around_x, around_y, left_x, left_y


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
        boundary_loss_around = f(pinn, self.around_x, self.around_y)

        # sin(pi * y) on the left (x = 0)
        boundary_loss_left = f(pinn, self.left_x, self.left_y) - torch.sin(math.pi * self.left_y)

        final_loss = (
            interior_loss.pow(2).mean()
            + boundary_loss_left.pow(2).mean()
            + boundary_loss_around.pow(2).mean()
        )

        return final_loss
