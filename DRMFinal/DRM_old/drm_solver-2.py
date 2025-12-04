import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# --- CRITICAL CHANGE: DOUBLE PRECISION ---
torch.set_default_dtype(torch.float64)

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if not os.path.exists('saved_models'): os.makedirs('saved_models')
if not os.path.exists('saved_data'): os.makedirs('saved_data')

# --- Hyperparameters ---
adam_iterations = 20000
lbfgs_max_iter = 10000
learning_rate_adam = 1e-3

# Dynamic Penalty Parameters
lambda_start = 100.0
lambda_end = 5000.0

# Batch sizes
batch_size_int = 5000
batch_size_bdry = 2000


# --- 1. DEEPER RESNET ARCHITECTURE ---
class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.activation = nn.Tanh()

    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = self.fc2(out)
        return self.activation(x + out)


class DRM_Model(nn.Module):
    def __init__(self):
        super(DRM_Model, self).__init__()
        self.input_layer = nn.Linear(2, 50)
        # 6 Residual Blocks
        self.resblocks = nn.Sequential(
            ResBlock(50), ResBlock(50), ResBlock(50),
            ResBlock(50), ResBlock(50), ResBlock(50)
        )
        self.output_layer = nn.Linear(50, 1)

    def forward(self, x):
        out = torch.tanh(self.input_layer(x))
        out = self.resblocks(out)
        out = self.output_layer(out)
        return out


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)


# --- 2. EXACT SOLUTIONS ---
def get_example_data(x, y, example_id):
    if example_id == "3.1":
        # u = 1/(2pi^2) * sin(pi*x) * sin(pi*y)
        factor = 1.0 / (2 * (np.pi ** 2))
        u = factor * torch.sin(np.pi * x) * torch.sin(np.pi * y)
    elif example_id == "3.2":
        # u = x^2 y^2 (1-x)^2 (1-y)^2
        u = (x ** 2) * (y ** 2) * ((1 - x) ** 2) * ((1 - y) ** 2)

    grads = torch.autograd.grad(u.sum(), [x, y], create_graph=True)
    u_x, u_y = grads[0], grads[1]

    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
    lap_u = u_xx + u_yy

    grads_lap = torch.autograd.grad(lap_u.sum(), [x, y], create_graph=True)
    lap_x, lap_y = grads_lap[0], grads_lap[1]

    f_val = torch.autograd.grad(lap_x.sum(), x, create_graph=True)[0] + \
            torch.autograd.grad(lap_y.sum(), y, create_graph=True)[0]

    return u, f_val, u_xx, u_yy


def get_boundary_g2(u_xx, u_yy, nx, ny):
    return u_xx * (nx ** 2) + u_yy * (ny ** 2)


# --- 3. POINTS GENERATOR ---
def generate_interior_points(n):
    x = torch.rand(n, 1, requires_grad=True, device=device)
    y = torch.rand(n, 1, requires_grad=True, device=device)
    return x, y


def generate_boundary_points(n):
    n_side = n // 4
    x1 = torch.rand(n_side, 1, device=device);
    y1 = torch.zeros(n_side, 1, device=device)
    nx1 = torch.zeros(n_side, 1, device=device);
    ny1 = -torch.ones(n_side, 1, device=device)

    x2 = torch.rand(n_side, 1, device=device);
    y2 = torch.ones(n_side, 1, device=device)
    nx2 = torch.zeros(n_side, 1, device=device);
    ny2 = torch.ones(n_side, 1, device=device)

    x3 = torch.zeros(n_side, 1, device=device);
    y3 = torch.rand(n_side, 1, device=device)
    nx3 = -torch.ones(n_side, 1, device=device);
    ny3 = torch.zeros(n_side, 1, device=device)

    x4 = torch.ones(n_side, 1, device=device);
    y4 = torch.rand(n_side, 1, device=device)
    nx4 = torch.ones(n_side, 1, device=device);
    ny4 = torch.zeros(n_side, 1, device=device)

    x = torch.cat([x1, x2, x3, x4], dim=0).requires_grad_(True)
    y = torch.cat([y1, y2, y3, y4], dim=0).requires_grad_(True)
    nx = torch.cat([nx1, nx2, nx3, nx4], dim=0)
    ny = torch.cat([ny1, ny2, ny3, ny4], dim=0)
    return x, y, nx, ny


# --- 4. LOSS FUNCTION ---
def compute_loss_P2(model, x_int, y_int, x_bdry, y_bdry, nx, ny, example_id, current_lambda):
    u_int = model(torch.cat([x_int, y_int], dim=1))
    _, f_exact, _, _ = get_example_data(x_int, y_int, example_id)

    grads = torch.autograd.grad(u_int.sum(), [x_int, y_int], create_graph=True)
    u_x, u_y = grads[0], grads[1]

    u_xx = torch.autograd.grad(u_x.sum(), x_int, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y_int, create_graph=True)[0]
    u_xy = torch.autograd.grad(u_x.sum(), y_int, create_graph=True)[0]

    D2v_sq = u_xx ** 2 + 2 * (u_xy ** 2) + u_yy ** 2
    loss_int = torch.mean(0.5 * D2v_sq - f_exact * u_int)

    u_bdry = model(torch.cat([x_bdry, y_bdry], dim=1))
    u_exact_bdry, _, u_xx_bdry, u_yy_bdry = get_example_data(x_bdry, y_bdry, example_id)

    g1 = u_exact_bdry
    g2 = get_boundary_g2(u_xx_bdry, u_yy_bdry, nx, ny)

    grads_b = torch.autograd.grad(u_bdry.sum(), [x_bdry, y_bdry], create_graph=True)
    u_bx, u_by = grads_b[0], grads_b[1]
    dv_dn = u_bx * nx + u_by * ny

    loss_bdry = torch.mean(- g2 * dv_dn + (current_lambda / 2.0) * (u_bdry - g1) ** 2)

    return loss_int + 4.0 * loss_bdry


# --- 5. METRICS ---
def calculate_errors(model, example_id):
    n_val = 60
    x = torch.linspace(0, 1, n_val, device=device)
    y = torch.linspace(0, 1, n_val, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    X = X.reshape(-1, 1).requires_grad_(True)
    Y = Y.reshape(-1, 1).requires_grad_(True)

    u_pred = model(torch.cat([X, Y], dim=1))
    u_exact, _, u_xx_ex, u_yy_ex = get_example_data(X, Y, example_id)

    grads = torch.autograd.grad(u_pred.sum(), [X, Y], create_graph=True)
    u_x, u_y = grads[0], grads[1]
    u_xx = torch.autograd.grad(u_x.sum(), X, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), Y, create_graph=True)[0]
    u_xy = torch.autograd.grad(u_x.sum(), Y, create_graph=True)[0]

    grads_ex = torch.autograd.grad(u_exact.sum(), [X, Y], create_graph=True)
    u_x_ex, u_y_ex = grads_ex[0], grads_ex[1]
    u_xy_ex = torch.autograd.grad(u_x_ex.sum(), Y, create_graph=True)[0]

    u_p, u_e = u_pred.detach().cpu().numpy(), u_exact.detach().cpu().numpy()
    dx_p, dx_e = u_x.detach().cpu().numpy(), u_x_ex.detach().cpu().numpy()
    dy_p, dy_e = u_y.detach().cpu().numpy(), u_y_ex.detach().cpu().numpy()
    dxx_p, dxx_e = u_xx.detach().cpu().numpy(), u_xx_ex.detach().cpu().numpy()
    dyy_p, dyy_e = u_yy.detach().cpu().numpy(), u_yy_ex.detach().cpu().numpy()
    dxy_p, dxy_e = u_xy.detach().cpu().numpy(), u_xy_ex.detach().cpu().numpy()

    # Relative Errors
    def rel_err(n, d): return np.sqrt(np.mean(n) / np.mean(d + 1e-20))

    l2 = rel_err((u_p - u_e) ** 2, u_e ** 2)

    num_h1 = (u_p - u_e) ** 2 + (dx_p - dx_e) ** 2 + (dy_p - dy_e) ** 2
    den_h1 = u_e ** 2 + dx_e ** 2 + dy_e ** 2
    h1 = rel_err(num_h1, den_h1)

    num_h2 = num_h1 + (dxx_p - dxx_e) ** 2 + (dyy_p - dyy_e) ** 2 + 2 * (dxy_p - dxy_e) ** 2
    den_h2 = den_h1 + dxx_e ** 2 + dyy_e ** 2 + 2 * (dxy_e ** 2)
    h2 = rel_err(num_h2, den_h2)

    return l2, h1, h2, u_p, u_e, X.detach().cpu().numpy(), Y.detach().cpu().numpy()


# --- MAIN EXECUTION ---
def run_experiment(example_id):
    print(f"\n{'=' * 20} RUNNING Ex {example_id} (DOUBLE PRECISION) {'=' * 20}")

    model = DRM_Model().to(device)
    model.apply(init_weights)

    loss_history = []

    # --- STAGE 1: ADAM WITH PENALTY ANNEALING ---
    print(f"--- Stage 1: Adam ({adam_iterations} iters) ---")
    optimizer_adam = optim.Adam(model.parameters(), lr=learning_rate_adam)

    # REMOVED verbose=True
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_adam, mode='min', factor=0.5, patience=1000)

    x_fixed_int, y_fixed_int = generate_interior_points(batch_size_int)
    x_fixed_bdry, y_fixed_bdry, nx_fixed, ny_fixed = generate_boundary_points(batch_size_bdry)

    for it in range(adam_iterations):
        optimizer_adam.zero_grad()

        # Anneal Lambda
        progress = min(1.0, it / (0.8 * adam_iterations))
        current_lambda = lambda_start + (lambda_end - lambda_start) * progress

        # Stochastic Sampling
        x_int, y_int = generate_interior_points(batch_size_int)
        x_bdry, y_bdry, nx, ny = generate_boundary_points(batch_size_bdry)

        loss = compute_loss_P2(model, x_int, y_int, x_bdry, y_bdry, nx, ny, example_id, current_lambda)
        loss.backward()
        optimizer_adam.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if it % 1000 == 0:
            l2, h1, h2, _, _, _, _ = calculate_errors(model, example_id)
            scheduler.step(loss_val)
            # Manual print for LR
            current_lr = optimizer_adam.param_groups[0]['lr']
            print(
                f"Adam {it:5d} | Loss: {loss_val:.2e} | Lam: {current_lambda:.0f} | LR: {current_lr:.1e} | H2: {h2:.4f}")

    # --- STAGE 2: LBFGS ---
    print(f"--- Stage 2: LBFGS ({lbfgs_max_iter} iters) ---")

    optimizer_lbfgs = optim.LBFGS(model.parameters(),
                                  max_iter=lbfgs_max_iter,
                                  history_size=100,
                                  line_search_fn="strong_wolfe",
                                  tolerance_grad=1e-15,
                                  tolerance_change=1e-15)

    def closure():
        optimizer_lbfgs.zero_grad()
        loss = compute_loss_P2(model, x_fixed_int, y_fixed_int, x_fixed_bdry, y_fixed_bdry, nx_fixed, ny_fixed,
                               example_id, lambda_end)
        loss.backward()
        return loss

    optimizer_lbfgs.step(closure)

    l2, h1, h2, u_pred, u_exact, X_grid, Y_grid = calculate_errors(model, example_id)
    print(f"\nFinal Results Ex {example_id}:")
    print(f"L2 Error: {l2:.6f}")
    print(f"H1 Error: {h1:.6f}")
    print(f"H2 Error: {h2:.6f}")

    np.savez(f'saved_data/results_ex{example_id}_final.npz',
             loss=loss_history, u_pred=u_pred, u_exact=u_exact)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(loss_history)
    plt.yscale('log')
    plt.title('Loss')
    plt.subplot(1, 3, 2)
    plt.imshow(u_exact.reshape(60, 60), extent=[0, 1, 0, 1], cmap='jet')
    plt.title('Exact')
    plt.subplot(1, 3, 3)
    plt.imshow(u_pred.reshape(60, 60), extent=[0, 1, 0, 1], cmap='jet')
    plt.title(f'Pred (H2={h2:.4f})')
    plt.savefig(f'Result_Ex{example_id}_Final.png')
    plt.show()


if __name__ == "__main__":
    run_experiment("3.1")
    run_experiment("3.2")