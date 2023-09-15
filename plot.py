# -*- coding: utf-8 -*-

import math
import os
import sys
from functools import partial

import matplotlib.pyplot as plt
import torch
from matplotlib import rc
from src.exact import exact_solution

from src.pinn_core import PINN, f
from params import *

plt.rcParams["figure.dpi"] = 150
rc("animation", html="html5")


pinn = torch.load("results/pinn.pt")
convergence_data = torch.load("results/convergence_data.pt")

# Plot the solution in a "dense" mesh
n_x = torch.linspace(X_INI, X_FIN, steps=PLOT_POINTS)
n_y = torch.linspace(Y_INI, Y_FIN, steps=PLOT_POINTS)
n_x, n_y = torch.meshgrid(n_x, n_y)
n_x_reshaped = n_x.reshape(-1, 1)
n_y_reshaped = n_y.reshape(-1, 1)

z = f(pinn, n_x_reshaped, n_y_reshaped).detach().reshape(PLOT_POINTS, PLOT_POINTS)
fig, ax = plt.subplots()
c = ax.pcolor(n_x, n_y, z)
ax.set_title("PINN solution")
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.colorbar(c, ax=ax)
fig.savefig(f"results/solution")

# Exact solution
z_exact = exact_solution(n_x, n_y, EPSILON)
fig, ax = plt.subplots()
c = ax.pcolor(n_x, n_y, z_exact)
ax.set_title("Exact solution")
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.colorbar(c, ax=ax)
fig.savefig(f"results/exact")

# Difference
z_exact = exact_solution(n_x, n_y, EPSILON)
fig, ax = plt.subplots()
c = ax.pcolor(n_x, n_y, z_exact - z)
ax.set_title("Exact - PINN")
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.colorbar(c, ax=ax)
fig.savefig(f"results/difference")

# Initial solution
n_y = torch.linspace(Y_INI, Y_FIN, steps=PLOT_POINTS)
n_x = torch.zeros_like(n_y)
z_pinn = f(pinn, n_x.reshape(-1, 1), n_y.reshape(-1, 1)).detach().reshape(-1)
z_exact = torch.sin(n_y * math.pi)
fig, ax = plt.subplots()
ax.set_title("Initial condition")
ax.plot(n_y, z_pinn, label="PINN")
ax.plot(n_y, z_exact, label="Exact")
ax.legend()
fig.savefig(f"results/initial")

# Slice along x axis
n_x = torch.linspace(X_INI, X_FIN, steps=PLOT_POINTS)
n_y = torch.full_like(n_x, (X_INI + X_FIN)/2.0)
z_pinn = f(pinn, n_x.reshape(-1, 1), n_y.reshape(-1, 1)).detach().reshape(-1)
z_exact = exact_solution(n_x, n_y, EPSILON)
fig, ax = plt.subplots()
ax.set_title("Slice along x axis at y=0.5")
ax.plot(n_x, z_pinn, label="PINN")
ax.plot(n_x, z_exact, label="Exact")
ax.legend()
fig.savefig(f"results/x_slice")

# Draw the convergence plot
fig, ax = plt.subplots()
ax.semilogy(convergence_data.cpu().detach().numpy())
ax.set_title("Convergence")
fig.savefig(f"results/convergence")

plt.show()
