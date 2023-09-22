# -*- coding: utf-8 -*-

import os
import torch
import time
from src.error import H1_error, L2_error

from params import *
from src.boundary_points import get_boundary_points
from src.eriksson_johnson_loss_function import Loss
from src.exact import exact_solution
from src.pinn_core import PINN, f, train_model

x_raw = torch.linspace(X_INI, X_FIN, steps=NUM_POINTS_X+2, requires_grad=True, device=DEVICE)[1:-1]
y_raw = torch.linspace(Y_INI, Y_FIN, steps=NUM_POINTS_X+2, requires_grad=True, device=DEVICE)[1:-1]
x, y = torch.meshgrid(x_raw, y_raw)
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

x_around, y_around, x_left, y_left = get_boundary_points(X_INI, X_FIN, Y_INI, Y_FIN, NUM_POINTS_X, NUM_POINTS_Y, DEVICE)

pinn = PINN(LAYERS, NEURONS).to(DEVICE)

loss_fn = Loss(x, y, x_around, y_around, x_left, y_left, EPSILON)

start_time = time.time()
convergence_data = train_model(
    pinn, loss_fn, learning_rate=LEARNING_RATE, max_epochs=NUMBER_EPOCHS
)
end_time = time.time()
exec_time = end_time - start_time

# Create result directory if it doesn't exist
try:
    os.makedirs("results/data")
except OSError as error:
    pass

pinn = pinn.cpu()

l2 = L2_error(pinn, exact_solution, EPSILON)
h1 = H1_error(pinn, exact_solution, EPSILON)

torch.save((l2, h1), "results/data/l2_h1.pt")
torch.save(pinn, "results/data/pinn.pt")
torch.save(convergence_data.detach(), "results/data/convergence_data.pt")
torch.save(exec_time, "results/data/exec_time.pt")
