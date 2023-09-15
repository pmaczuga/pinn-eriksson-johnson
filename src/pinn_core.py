# -*- coding: utf-8 -*-
from typing import Callable

import torch
from torch import nn


class PINN(nn.Module):
    def __init__(self, num_hidden: int, dim_hidden: int, act=nn.Tanh()):
        super().__init__()

        self.layer_in = nn.Linear(2, dim_hidden)
        self.layer_out = nn.Linear(dim_hidden, 1)

        num_middle = num_hidden - 1
        self.middle_layers = nn.ModuleList(
            [nn.Linear(dim_hidden, dim_hidden) for _ in range(num_middle)]
        )
        self.act = act

    def forward(self, x, y):
        x_stack = torch.cat([x, y], dim=1)
        out = self.act(self.layer_in(x_stack))
        for layer in self.middle_layers:
            out = self.act(layer(out))
        logits = self.layer_out(out)

        return logits


def f(pinn: PINN, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return pinn(x, y)


def df(output: torch.Tensor, input: torch.Tensor, order: int = 1) -> torch.Tensor:
    df_value = output
    for _ in range(order):
        df_value = torch.autograd.grad(
            df_value,
            input,
            grad_outputs=torch.ones_like(input),
            create_graph=True,
            retain_graph=True,
        )[0]

    return df_value


def dfdx(pinn: PINN, x: torch.Tensor, y: torch.Tensor, order: int = 1):
    """Derivative with respect to the spatial variable of arbitrary order"""
    f_value = f(pinn, x, y)
    return df(f_value, x, order=order)

def dfdy(pinn: PINN, x: torch.Tensor, y: torch.Tensor, order: int = 1):
    """Derivative with respect to the spatial variable of arbitrary order"""
    f_value = f(pinn, x, y)
    return df(f_value, y, order=order)

def train_model(
    pinn: PINN,
    loss_fn: Callable,
    learning_rate: float = 0.01,
    max_epochs: int = 1_000,
) -> torch.Tensor:
    optimizer = torch.optim.Adam(pinn.parameters(), lr=learning_rate)

    convergence_data = torch.empty((max_epochs))

    for epoch in range(max_epochs):
        loss = loss_fn(pinn)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        convergence_data[epoch] = loss.detach().cpu()

        if epoch % 1000 == 0:
            print(f"Epoch: {epoch} - Loss: {float(loss):>7f}")

    print(f"Final Epoch: - Loss: {float(loss):>7f}")

    return convergence_data
