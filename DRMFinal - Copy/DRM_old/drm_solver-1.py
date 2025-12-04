import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
from model import DRM_Model, init_weights
from exact_solutions import get_example_data, get_boundary_g2

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parameters
iterations = 20000
learning_rate = 1e-3  # Adam LR
lambda_penalty = 500.0  # Penalty weight for essential BC (u = g1)
batch_size_int = 1000  # Interior points
batch_size_bdry = 400  # Boundary points


def generate_interior_points(n):
    x = torch.rand(n, 1, requires_grad=True, device=device)
    y = torch.rand(n, 1, requires_grad=True, device=device)
    return x, y


def generate_boundary_points(n):
    # Generate points on 4 sides
    n_side = n // 4
    pts = []
    normals = []

    # Bottom (y=0), Normal=(0,-1)
    x1 = torch.rand(n_side, 1, device=device)
    y1 = torch.zeros(n_side, 1, device=device)
    nx1 = torch.zeros(n_side, 1, device=device)
    ny1 = -torch.ones(n_side, 1, device=device)

    # Top (y=1), Normal=(0,1)
    x2 = torch.rand(n_side, 1, device=device)
    y2 = torch.ones(n_side, 1, device=device)
    nx2 = torch.zeros(n_side, 1, device=device)
    ny2 = torch.ones(n_side, 1, device=device)

    # Left (x=0), Normal=(-1,0)
    x3 = torch.zeros(n_side, 1, device=device)
    y3 = torch.rand(n_side, 1, device=device)
    nx3 = -torch.ones(n_side, 1, device=device)
    ny3 = torch.zeros(n_side, 1, device=device)

    # Right (x=1), Normal=(1,0)
    x4 = torch.ones(n_side, 1, device=device)   
    y4 = torch.rand(n_side, 1, device=device)
    nx4 = torch.ones(n_side, 1, device=device)
    ny4 = torch.zeros(n_side, 1, device=device)

    x = torch.cat([x1, x2, x3, x4], dim=0).requires_grad_(True)
    y = torch.cat([y1, y2, y3, y4], dim=0).requires_grad_(True)
    nx = torch.cat([nx1, nx2, nx3, nx4], dim=0)
    ny = torch.cat([ny1, ny2, ny3, ny4], dim=0)

    return x, y, nx, ny


def compute_loss_P2(model, x_int, y_int, x_bdry, y_bdry, nx, ny, example_id):
    # --- Interior Energy ---
    # Equation (2.7): 1/2 |D^2 v|^2 - f*v

    # Forward pass interior
    u_int = model(torch.cat([x_int, y_int], dim=1))

    # Exact f
    _, f_exact, _, _ = get_example_data(x_int, y_int, example_id)

    # Derivatives
    grads = torch.autograd.grad(u_int.sum(), [x_int, y_int], create_graph=True)
    u_x, u_y = grads[0], grads[1]

    u_xx = torch.autograd.grad(u_x.sum(), x_int, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y_int, create_graph=True)[0]
    u_xy = torch.autograd.grad(u_x.sum(), y_int, create_graph=True)[0]

    # |D^2 v|^2 = u_xx^2 + 2*u_xy^2 + u_yy^2
    D2v_sq = u_xx ** 2 + 2 * (u_xy ** 2) + u_yy ** 2

    loss_int = torch.mean(0.5 * D2v_sq - f_exact * u_int)

    # --- Boundary Energy (P2) ---
    # Equation (2.7) for P2: -g2 * (dv/dn) + lambda/2 * (v - g1)^2

    u_bdry = model(torch.cat([x_bdry, y_bdry], dim=1))

    # Exact g1 (u) and g2 (d^2u/dn^2)
    u_exact_bdry, _, u_xx_bdry, u_yy_bdry = get_example_data(x_bdry, y_bdry, example_id)
    g1 = u_exact_bdry
    g2 = get_boundary_g2(u_xx_bdry, u_yy_bdry, nx, ny)

    # Normal Derivative dv/dn
    grads_b = torch.autograd.grad(u_bdry.sum(), [x_bdry, y_bdry], create_graph=True)
    u_bx, u_by = grads_b[0], grads_b[1]
    dv_dn = u_bx * nx + u_by * ny

    # Penalty terms
    term1 = - g2 * dv_dn
    term2 = (lambda_penalty / 2.0) * (u_bdry - g1) ** 2

    loss_bdry = torch.mean(term1 + term2)

    # Total Empirical Loss (Assuming area |Omega|=1 and |dOmega|=4 approx scaling)
    # The paper (2.7) sums expectations weighted by measure
    total_loss = loss_int + 4.0 * loss_bdry

    return total_loss


def calculate_errors(model, example_id):
    # Grid for error calculation
    n_val = 50
    x = torch.linspace(0, 1, n_val, device=device)
    y = torch.linspace(0, 1, n_val, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    X = X.reshape(-1, 1).requires_grad_(True)
    Y = Y.reshape(-1, 1).requires_grad_(True)

    u_pred = model(torch.cat([X, Y], dim=1))
    u_exact, _, u_xx_ex, u_yy_ex = get_example_data(X, Y, example_id)

    # Derivatives for H1/H2
    grads = torch.autograd.grad(u_pred.sum(), [X, Y], create_graph=True)
    u_x, u_y = grads[0], grads[1]
    u_xx = torch.autograd.grad(u_x.sum(), X, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), Y, create_graph=True)[0]
    u_xy = torch.autograd.grad(u_x.sum(), Y, create_graph=True)[0]

    grads_ex = torch.autograd.grad(u_exact.sum(), [X, Y], create_graph=True)
    u_x_ex, u_y_ex = grads_ex[0], grads_ex[1]
    u_xy_ex = torch.autograd.grad(u_x_ex.sum(), Y, create_graph=True)[0]

    # Detach for numpy calc
    u_p, u_e = u_pred.detach().cpu().numpy(), u_exact.detach().cpu().numpy()
    dx_p, dx_e = u_x.detach().cpu().numpy(), u_x_ex.detach().cpu().numpy()
    dy_p, dy_e = u_y.detach().cpu().numpy(), u_y_ex.detach().cpu().numpy()
    dxx_p, dxx_e = u_xx.detach().cpu().numpy(), u_xx_ex.detach().cpu().numpy()
    dyy_p, dyy_e = u_yy.detach().cpu().numpy(), u_yy_ex.detach().cpu().numpy()
    dxy_p, dxy_e = u_xy.detach().cpu().numpy(), u_xy_ex.detach().cpu().numpy()

    # L2 Error
    diff_sq = (u_p - u_e) ** 2
    norm_sq = u_e ** 2
    l2_err = np.sqrt(np.mean(diff_sq) / np.mean(norm_sq))

    # H1 Error (L2 + gradients)
    diff_h1 = diff_sq + (dx_p - dx_e) ** 2 + (dy_p - dy_e) ** 2
    norm_h1 = norm_sq + dx_e ** 2 + dy_e ** 2
    h1_err = np.sqrt(np.mean(diff_h1) / np.mean(norm_h1))

    # H2 Error (H1 + 2nd derivs)
    diff_h2 = diff_h1 + (dxx_p - dxx_e) ** 2 + (dyy_p - dyy_e) ** 2 + 2 * (dxy_p - dxy_e) ** 2
    norm_h2 = norm_h1 + dxx_e ** 2 + dyy_e ** 2 + 2 * (dxy_e ** 2)
    h2_err = np.sqrt(np.mean(diff_h2) / np.mean(norm_h2))

    return l2_err, h1_err, h2_err, u_p, u_e


def run_experiment(example_id):
    print(f"\n{'=' * 20} Running DRM for Example {example_id} (Problem P2) {'=' * 20}")

    model = DRM_Model().to(device)
    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)

    loss_history = []

    start_time = time.time()

    for it in range(iterations):
        optimizer.zero_grad()

        # Resample points every iteration (Stochastic Gradient Descent / DRM approach)
        x_int, y_int = generate_interior_points(batch_size_int)
        x_bdry, y_bdry, nx, ny = generate_boundary_points(batch_size_bdry)

        loss = compute_loss_P2(model, x_int, y_int, x_bdry, y_bdry, nx, ny, example_id)

        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if it % 1000 == 0 or it == iterations - 1:
            l2, h1, h2, _, _ = calculate_errors(model, example_id)
            print(f"Iter {it:5d} | Loss: {loss_val:.2e} | L2 Rel: {l2:.2e} | H1 Rel: {h1:.2e} | H2 Rel: {h2:.2e}")

    elapsed = time.time() - start_time

    # Final Evaluation
    l2, h1, h2, u_pred, u_exact = calculate_errors(model, example_id)
    print(f"\nFinal Results for Ex {example_id}:")
    print(f"Time: {elapsed:.2f} seconds")
    print(f"Loss: {loss_history[-1]:.2e}")
    print(f"L2 Error: {l2:.4e}")
    print(f"H1 Error: {h1:.4e}")
    print(f"H2 Error: {h2:.4e}")

    # Plotting
    plt.figure(figsize=(15, 5))

    # Loss History
    plt.subplot(1, 3, 1)
    plt.plot(loss_history)
    plt.yscale('log')
    plt.title(f'Training Dynamics (Ex {example_id})')
    plt.xlabel('Iteration')
    plt.ylabel('Loss (Energy)')

    # Exact Solution
    plt.subplot(1, 3, 2)
    plt.title('Exact Solution')
    plt.imshow(u_exact.reshape(50, 50), extent=[0, 1, 0, 1], origin='lower', cmap='jet')
    plt.colorbar()

    # Computed Solution
    plt.subplot(1, 3, 3)
    plt.title('DRM Computed Solution')
    plt.imshow(u_pred.reshape(50, 50), extent=[0, 1, 0, 1], origin='lower', cmap='jet')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(f'Result_Ex{example_id}.png')
    plt.show()


if __name__ == "__main__":
    # Run Example 3.1
    run_experiment("3.1")

    # Run Example 3.2
    run_experiment("3.2")