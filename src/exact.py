import torch
import math

import matplotlib.pyplot as plt

def exact_solution(x, y, epsilon) -> torch.Tensor:
    r1 = (1.0 + math.sqrt(1.0 + 4.0*epsilon*math.pi*math.pi))/(2.0*epsilon)
    r2 = (1.0 - math.sqrt(1.0 + 4.0*epsilon*math.pi*math.pi))/(2.0*epsilon)
    res_y = (torch.exp(r1*(x-1.0))-torch.exp(r2*(x-1.0)))/(math.exp(-r1)-math.exp(-r2))
    res_x = torch.sin(math.pi*y)
    res = res_y.mul(res_x)
    return res

# x_raw = torch.linspace(0, 1, 1000)
# y_raw = torch.linspace(0, 1, 1000)
# x, y = torch.meshgrid(x_raw, y_raw)
# z = exact_solution(x, y)

# fig, ax = plt.subplots()
# c = ax.pcolor(x, y, z)
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# fig.colorbar(c, ax=ax)

# fig, ax = plt.subplots()
# ax.plot(x_raw, torch.sin(math.pi*x_raw))

# y05 = torch.full_like(x_raw, 0.5)
# z_slice = exact_solution(x_raw, y05)
# fig, ax = plt.subplots()
# ax.set_title("Slice along x axis ay y=0.5")
# ax.plot(x_raw, z_slice)

# plt.show()