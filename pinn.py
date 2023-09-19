# -*- coding: utf-8 -*-

import os
import torch

from params import *
from src.boundary_points import get_boundary_points
from src.eriksson_johnson_loss_function import Loss
from src.pinn_core import PINN, f, train_model

x_raw = torch.linspace(X_INI, X_FIN, steps=NUM_POINTS_X+2, requires_grad=True, device=DEVICE)
y_raw = torch.linspace(Y_INI, Y_FIN, steps=NUM_POINTS_X+2, requires_grad=True, device=DEVICE)
x, y = torch.meshgrid(x_raw, y_raw)
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

x_around, y_around, x_left, y_left = get_boundary_points(X_INI, X_FIN, Y_INI, Y_FIN, NUM_POINTS_X, NUM_POINTS_Y, DEVICE)

pinn = PINN(LAYERS, NEURONS).to(DEVICE)

loss_fn = Loss(x, y, x_around, y_around, x_left, y_left, EPSILON)

convergence_data = train_model(
    pinn, loss_fn, learning_rate=LEARNING_RATE, max_epochs=NUMBER_EPOCHS
)

# Create result directory if it doesn't exist
try:
    os.makedirs("results")
except OSError as error:
    pass

pinn = pinn.cpu()

torch.save(pinn, "results/pinn.pt")
torch.save(convergence_data.detach(), "results/convergence_data.pt")
