# -*- coding: utf-8 -*-

import math
import os
import sys
from functools import partial

import matplotlib.pyplot as plt
import torch
from matplotlib import rc
from src.eriksson_johnson_loss_function import Loss

from src.pinn_core import PINN, f, train_model
from params import *

plt.rcParams["figure.dpi"] = 150
rc("animation", html="html5")


x_raw = torch.linspace(X_INI, X_FIN, steps=NUM_POINTS_X, requires_grad=True, device=DEVICE)
y_raw = torch.linspace(Y_INI, Y_FIN, steps=NUM_POINTS_X, requires_grad=True, device=DEVICE)
x, y = torch.meshgrid(x_raw, y_raw)
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

pinn = PINN(3, 15).to(DEVICE)

loss_fn = Loss(x, y, EPSILON)

convergence_data = train_model(
    pinn, loss_fn, learning_rate=0.005, max_epochs=NUMBER_EPOCHS
)


# Create result directory if it doesn't exist
try:
    os.makedirs("results")
except OSError as error:
    pass

pinn = pinn.cpu()

torch.save(pinn, "results/pinn.pt")
torch.save(convergence_data.detach(), "results/convergence_data.pt")
