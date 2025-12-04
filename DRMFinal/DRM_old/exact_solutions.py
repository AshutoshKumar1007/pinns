import torch
import numpy as np


def get_example_data(x, y, example_id):
    """
    Inputs: x, y (torch tensors requiring grad)
    Returns: u (exact), f (source), u_xx, u_yy (for g2 calculation)
    """

    if example_id == "3.1":
        # Example 3.1: u = 1/(2pi^2) * sin(pi*x1) * sin(pi*x2)
        factor = 1.0 / (2 * (np.pi ** 2))
        u = factor * torch.sin(np.pi * x) * torch.sin(np.pi * y)

    elif example_id == "3.2":
        # Example 3.2: u = x^2 * y^2 * (1-x)^2 * (1-y)^2
        u = (x ** 2) * (y ** 2) * ((1 - x) ** 2) * ((1 - y) ** 2)

    # Automatic differentiation to get derivatives and source term f = Delta^2 u
    grads = torch.autograd.grad(u.sum(), [x, y], create_graph=True)
    u_x, u_y = grads[0], grads[1]

    grads_x = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]  # u_xx
    grads_y = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]  # u_yy

    u_xx = grads_x
    u_yy = grads_y

    # Laplacian u
    lap_u = u_xx + u_yy

    # Bi-Laplacian (Delta^2 u) -> f
    grads_lap = torch.autograd.grad(lap_u.sum(), [x, y], create_graph=True)
    lap_x, lap_y = grads_lap[0], grads_lap[1]

    f_val = torch.autograd.grad(lap_x.sum(), x, create_graph=True)[0] + \
            torch.autograd.grad(lap_y.sum(), y, create_graph=True)[0]

    return u, f_val, u_xx, u_yy


def get_boundary_g2(u_xx, u_yy, nx, ny):
    """
    Calculate g2 = d^2u / dn^2
    """
    # g2 = u_xx * nx^2 + u_yy * ny^2
    g2 = u_xx * (nx ** 2) + u_yy * (ny ** 2)
    return g2