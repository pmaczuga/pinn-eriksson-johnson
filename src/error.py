import math
import torch
import numpy as np
from scipy.integrate import dblquad
from src.exact import exact_solution

from params import *
from src.pinn_core import dfdx, dfdy

def np_to_torch(v: torch.tensor):
    return torch.tensor(v, dtype=torch.float32, requires_grad=True).cpu().reshape(-1, 1)

def L2_error(pinn, exact, epsilon):
    def pinn_np(y, x):
        return pinn(np_to_torch(x), np_to_torch(y)).detach().reshape(-1)

    def exact_np(y, x):
        return exact(np_to_torch(x), np_to_torch(y), epsilon).detach().reshape(-1)
    
    up   = lambda y, x: (pinn_np(y, x) - exact_np(y, x))**2
    dwon = lambda y, x: exact_np(y, x)**2

    up_int   = dblquad(up, 0, 1, 0, 1)[0]
    down_int = dblquad(dwon, 0, 1, 0, 1)[0]

    return math.sqrt(up_int / down_int)

def H1_error(pinn, exact, epsilon):
    def pinn_np(y, x):
        return pinn(np_to_torch(x), np_to_torch(y)).detach().reshape(-1)

    def pinn_grad_np(y, x):
        dx = dfdx(pinn, np_to_torch(x), np_to_torch(y))
        dy = dfdy(pinn, np_to_torch(x), np_to_torch(y))
        return (dx + dy).detach().reshape(-1)

    def exact_np(y, x):
        return exact(np_to_torch(x), np_to_torch(y), epsilon).detach().reshape(-1)
    
    def exact_grad_np(y, x):
        dx = dfdx(exact, np_to_torch(x), np_to_torch(y))
        dy = dfdy(exact, np_to_torch(x), np_to_torch(y))
        return (dx + dy).detach().reshape(-1)

    up1  = lambda y, x: (pinn_np(y, x) - exact_np(y, x))**2
    up2  = lambda y, x: (pinn_grad_np(y, x) - exact_np(y, x))**2
    dwon = lambda y, x: exact_np(y, x)**2

    up1_int  = dblquad(up1, 0, 1, 0, 1)[0]
    up2_int  = dblquad(up1, 0, 1, 0, 1)[0]
    down_int = dblquad(dwon, 0, 1, 0, 1)[0]

    return math.sqrt(up1_int + up2_int / down_int)