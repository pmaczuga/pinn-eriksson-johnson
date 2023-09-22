# -*- coding: utf-8 -*-

import math
import os
import sys
from functools import partial

import matplotlib.pyplot as plt
from src.error import H1_error, L2_error
from src.boundary_points import get_boundary_points
from src.eriksson_johnson_loss_function import Loss
import torch
from matplotlib import rc
from src.exact import exact_solution

from src.pinn_core import PINN, f
from params import *

plt.rcParams["figure.dpi"] = 150
rc("animation", html="html5")


pinn = torch.load("results/data/pinn.pt")
convergence_data = torch.load("results/data/convergence_data.pt")
exec_time = torch.load("results/data/exec_time.pt")
l2, h1 = torch.load("results/data/l2_h1.pt")
x_around, y_around, x_left, y_left = get_boundary_points(X_INI, X_FIN, Y_INI, Y_FIN, NUM_POINTS_X, NUM_POINTS_Y, DEVICE)

# Write params and results to text file
with open("results/result.txt", "w") as file:
    file.write(f"DEVICE = {DEVICE}\n")
    file.write(f"NUM_POINTS_X = {NUM_POINTS_X}\n")
    file.write(f"NUM_POINTS_Y = {NUM_POINTS_Y}\n")
    file.write(f"EPSILON = {EPSILON}\n")
    file.write(f"NUMBER_EPOCHS = {NUMBER_EPOCHS}\n")
    file.write(f"LEARNING_RATE = {LEARNING_RATE}\n")
    file.write(f"LAYERS = {LAYERS}\n")
    file.write(f"NEURONS = {NEURONS}\n")
    file.write(f"\n")
    file.write(f"Time = {exec_time}\n")
    file.write(f"L2 error = {l2}\n")
    file.write(f"H1 error = {h1}\n")

# Plot the solution in a "dense" mesh
n_x = torch.linspace(X_INI, X_FIN, steps=PLOT_POINTS)
n_y = torch.linspace(Y_INI, Y_FIN, steps=PLOT_POINTS)
n_x, n_y = torch.meshgrid(n_x, n_y)
n_x_reshaped = n_x.reshape(-1, 1).requires_grad_(True)
n_y_reshaped = n_y.reshape(-1, 1).requires_grad_(True)

loss_fn = Loss(n_x_reshaped, n_x_reshaped, x_around, y_around, x_left, y_left, EPSILON)
interior_loss = loss_fn.f_inter_loss(n_x_reshaped, n_y_reshaped, pinn, EPSILON).reshape(PLOT_POINTS, PLOT_POINTS)

z = f(pinn, n_x_reshaped, n_y_reshaped).detach().reshape(PLOT_POINTS, PLOT_POINTS)
fig, ax = plt.subplots()
c = ax.pcolor(n_x, n_y, z)
ax.set_title("PINN solution, eps={}, L2 = {:.2e}, H1={:.2e}".format(EPSILON, l2, h1))
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.colorbar(c, ax=ax)
fig.savefig(f"results/solution")

# Exact solution
z_exact = exact_solution(n_x, n_y, EPSILON)
fig, ax = plt.subplots()
c = ax.pcolor(n_x, n_y, z_exact)
ax.set_title("Exact solution, eps={}, L2 = {:.2e}, H1={:.2e}".format(EPSILON, l2, h1))
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.colorbar(c, ax=ax)
fig.savefig(f"results/exact")

# Difference
z_exact = exact_solution(n_x, n_y, EPSILON)
fig, ax = plt.subplots()
c = ax.pcolor(n_x, n_y, z_exact - z)
ax.set_title("Exact - PINN, eps={}, L2 = {:.2e}, H1={:.2e}".format(EPSILON, l2, h1))
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.colorbar(c, ax=ax)
fig.savefig(f"results/difference")

# Interior loss
fig, ax = plt.subplots()
c = ax.pcolor(n_x, n_y, interior_loss.detach())
ax.set_title("Interior loss")
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.colorbar(c, ax=ax)
fig.savefig(f"results/interior_loss")

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
ax.set_title("Slice along x axis at y=0.5, eps={}".format(EPSILON))
ax.plot(n_x, z_pinn, "--", label="PINN")
ax.plot(n_x, z_exact, label="Exact")
ax.legend()
fig.savefig(f"results/x_slice")

# Draw the convergence plot
fig, ax = plt.subplots()
ax.semilogy(convergence_data.cpu().detach().numpy())
ax.set_title(f"Convergence, time = {exec_time} s")
fig.savefig(f"results/convergence")

plt.show()
