import time
import os
import pickle as pkl
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Added for 3D plotting

# Import the alternative model_og architecture
import model_og,pde, tools


# --- Configuration Class ---
class Config:
    def __init__(self, example_name):
        self.example_name = example_name
        if self.example_name == "Example3.1":
            self.dataname = 'ex3.1_5k_4k'
        elif self.example_name == "Example3.2":
            self.dataname = 'ex3.2_5k_4k'
        else:
            raise ValueError("Unknown example name")

        self.adam_epochs = 30000
        self.lbfgs_max_iter = 10000

        self.lr = 1e-4
        self.bw_dir = 5000.0
        self.bw_neu = 5000.0
        self.results_dir = f"results_{self.example_name.replace('.', '_')}"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.plot_resolution = 100  # Reduced resolution for faster 3D plotting


# --- Helper Function for Derivatives ---
def get_pinn_derivatives(net, x1, x2):
    u = net(x1, x2)
    u_x = torch.autograd.grad(u.sum(), x1, create_graph=True, allow_unused=True)[0]
    u_y = torch.autograd.grad(u.sum(), x2, create_graph=True, allow_unused=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x1, create_graph=True, allow_unused=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), x2, create_graph=True, allow_unused=True)[0]
    u_xy = torch.autograd.grad(u_x.sum(), x2, create_graph=True, allow_unused=True)[0]
    return {'u': u, 'u_x': u_x, 'u_y': u_y, 'u_xx': u_xx, 'u_yy': u_yy, 'u_xy': u_xy}


# --- Main Solver Function ---
def solve_pde(config):
    os.makedirs(config.results_dir, exist_ok=True)
    print(f"--- Solving for {config.example_name} ---")
    print(f"Using device: {config.device}")

    gt_module_name = f"g_tr_{config.example_name.replace('Example', 'ex').replace('.', '_')}"
    gt = __import__(gt_module_name)

    net = model_og.NN().to(config.device)
    net.apply(model_og.init_weights)
    print("\n--- Neural Network Architecture ---")
    print(net)
    print(f"Total trainable parameters: {sum(p.numel() for p in net.parameters())}")
    print("-----------------------------------")

    with open(f"dataset/{config.dataname}", 'rb') as f:
        int_col, bdry_col, normal_vec = pkl.load(f), pkl.load(f), pkl.load(f)
    intx1, intx2 = np.split(int_col, 2, axis=1)
    bdx1, bdx2 = np.split(bdry_col, 2, axis=1)
    nx1, nx2 = np.split(normal_vec, 2, axis=1)
    tintx1, tintx2, tbdx1, tbdx2, tnx1, tnx2 = tools.from_numpy_to_tensor(
        [intx1, intx2, bdx1, bdx2, nx1, nx2], [True, True, False, False, False, False]
    )
    with open(f"dataset/gt_on_{config.dataname}", 'rb') as f:
        _, f_np, dir_np, neu_np = pkl.load(f), pkl.load(f), pkl.load(f), pkl.load(f)
    f, bdry_dir, bdry_neu, _ = tools.from_numpy_to_tensor([f_np, dir_np, neu_np, f_np], [False] * 4)

    tensors_to_device = [tintx1, tintx2, tbdx1, tbdx2, tnx1, tnx2, f, bdry_dir, bdry_neu]
    tintx1, tintx2, tbdx1, tbdx2, tnx1, tnx2, f, bdry_dir, bdry_neu = [t.to(config.device) for t in tensors_to_device]
    tintx1.requires_grad_(True);
    tintx2.requires_grad_(True)

    start_time = time.time()

    # Phase 1: Adam Training
    print("\n--- Phase 1: Adam Training (for broad exploration) ---")
    optimizer_adam = optim.Adam(net.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_adam, 'min', patience=1000, factor=0.75)
    adam_loss_history = []

    for epoch in range(1, config.adam_epochs + 1):
        net.train()
        optimizer_adam.zero_grad()
        loss, pres, _, _, _ = pde.pdeloss(net, tintx1, tintx2, f, tbdx1, tbdx2, tnx1, tnx2, bdry_dir, bdry_neu,
                                          config.bw_dir, config.bw_neu)
        loss.backward()
        optimizer_adam.step()
        scheduler.step(loss)
        adam_loss_history.append(loss.item())
        if epoch % 1000 == 0:
            current_lr = optimizer_adam.param_groups[0]['lr']
            print(
                f"Adam Epoch {epoch:5d}/{config.adam_epochs} | Loss: {loss.item():.4e} | PDE Res: {pres.item():.4e} | LR: {current_lr:.2e}")

    # Phase 2: L-BFGS Fine-Tuning
    print("\n--- Phase 2: L-BFGS Training (for precision) ---")
    optimizer_lbfgs = optim.LBFGS(net.parameters(), max_iter=config.lbfgs_max_iter, line_search_fn='strong_wolfe')
    lbfgs_loss_history = []
    lbfgs_iter_count = 0

    def closure():
        nonlocal lbfgs_iter_count
        optimizer_lbfgs.zero_grad()
        loss, _, _, _, _ = pde.pdeloss(net, tintx1, tintx2, f, tbdx1, tbdx2, tnx1, tnx2, bdry_dir, bdry_neu,
                                       config.bw_dir, config.bw_neu)
        loss.backward()

        lbfgs_loss_history.append(loss.item())
        if lbfgs_iter_count % 100 == 0:
            print(f"L-BFGS Iter {lbfgs_iter_count:5d} | Loss: {loss.item():.4e}")
        lbfgs_iter_count += 1
        return loss

    optimizer_lbfgs.step(closure)

    computation_time = time.time() - start_time
    print(f"Training finished in {computation_time:.2f} seconds.")

    loss_history = adam_loss_history + lbfgs_loss_history

    # --- Evaluation ---
    print("\n--- Evaluation ---")
    net.eval()
    res = config.plot_resolution
    x_pts = np.linspace(0, 1, res)
    y_pts = np.linspace(0, 1, res)
    ms_x, ms_y = np.meshgrid(x_pts, y_pts)
    grid_x, grid_y = np.ravel(ms_x).reshape(-1, 1), np.ravel(ms_y).reshape(-1, 1)

    gt_vals = gt.get_all_derivatives(np.hstack([grid_x, grid_y]))
    pt_x = Variable(torch.from_numpy(grid_x).float(), requires_grad=True).to(config.device)
    pt_y = Variable(torch.from_numpy(grid_y).float(), requires_grad=True).to(config.device)
    pinn_derivs_torch = get_pinn_derivatives(net, pt_x, pt_y)
    pinn_derivs = {k: v.detach().cpu().numpy() for k, v in pinn_derivs_torch.items()}

    l2_err = np.linalg.norm(gt_vals['u'] - pinn_derivs['u']) / np.linalg.norm(gt_vals['u'])
    h1_err = np.sqrt((np.linalg.norm(gt_vals['u'] - pinn_derivs['u']) ** 2 + np.linalg.norm(
        gt_vals['u_x'] - pinn_derivs['u_x']) ** 2 + np.linalg.norm(gt_vals['u_y'] - pinn_derivs['u_y']) ** 2) / (
                                 np.linalg.norm(gt_vals['u']) ** 2 + np.linalg.norm(
                             gt_vals['u_x']) ** 2 + np.linalg.norm(gt_vals['u_y']) ** 2))
    h2_err = np.sqrt((np.linalg.norm(gt_vals['u'] - pinn_derivs['u']) ** 2 + np.linalg.norm(
        gt_vals['u_x'] - pinn_derivs['u_x']) ** 2 + np.linalg.norm(
        gt_vals['u_y'] - pinn_derivs['u_y']) ** 2 + np.linalg.norm(
        gt_vals['u_xx'] - pinn_derivs['u_xx']) ** 2 + np.linalg.norm(
        gt_vals['u_yy'] - pinn_derivs['u_yy']) ** 2 + np.linalg.norm(gt_vals['u_xy'] - pinn_derivs['u_xy']) ** 2) / (
                                 np.linalg.norm(gt_vals['u']) ** 2 + np.linalg.norm(
                             gt_vals['u_x']) ** 2 + np.linalg.norm(gt_vals['u_y']) ** 2 + np.linalg.norm(
                             gt_vals['u_xx']) ** 2 + np.linalg.norm(gt_vals['u_yy']) ** 2 + np.linalg.norm(
                             gt_vals['u_xy']) ** 2))

    # --- Reporting and Plotting ---
    print("\n--- Summary Report ---")
    print(f"Example: {config.example_name}")
    print(f"Computation Time: {computation_time:.2f} s")
    print(f"Final Loss: {loss_history[-1]:.4e}")
    print(f"L2 Relative Error: {l2_err:.4e}")
    print(f"H1 Relative Error: {h1_err:.4e}")
    print(f"H2 Relative Error: {h2_err:.4e}")
    print("----------------------")

    # Plotting: Loss History
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history)
    plt.axvline(x=config.adam_epochs, color='r', linestyle='--', label='Switched to L-BFGS')
    plt.yscale('log');
    plt.title(f'Loss History ({config.example_name})');
    plt.xlabel('Iteration');
    plt.ylabel('Loss');
    plt.grid(True);
    plt.legend()
    plt.savefig(os.path.join(config.results_dir, "loss_history.png"))
    plt.close()

    # Plotting: 2D Heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle(f'PINN Solution for {config.example_name}', fontsize=16)
    v_min, v_max = gt_vals['u'].min(), gt_vals['u'].max()
    plots = [("Exact Solution", gt_vals['u'].reshape(res, res)), ("PINN Solution", pinn_derivs['u'].reshape(res, res)),
             ("Absolute Error", np.abs(gt_vals['u'] - pinn_derivs['u']).reshape(res, res))]
    for i, (title, data) in enumerate(plots):
        ax = axes[i]
        im = ax.pcolormesh(ms_x, ms_y, data, cmap='jet', shading='auto', vmin=v_min if i < 2 else None,
                           vmax=v_max if i < 2 else None)
        ax.set_title(title);
        ax.set_xlabel('$x_1$');
        ax.set_aspect('equal', 'box')
        if i == 0: ax.set_ylabel('$x_2$')
        fig.colorbar(im, ax=ax)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(config.results_dir, "solution_comparison.png"))
    plt.close()

    # Plotting: 3D Surface Plots
    fig = plt.figure(figsize=(16, 7))
    fig.suptitle(f'3D Surface Plot for {config.example_name}', fontsize=16)

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    pinn_sol_grid = pinn_derivs['u'].reshape(res, res)
    surf1 = ax1.plot_surface(ms_x, ms_y, pinn_sol_grid, cmap='jet', edgecolor='none')
    ax1.set_title("PINN Computed Solution")
    ax1.set_xlabel('$x_1$');
    ax1.set_ylabel('$x_2$');
    ax1.set_zlabel('u(x,y)')
    fig.colorbar(surf1, ax=ax1, shrink=0.7, aspect=10)

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    exact_sol_grid = gt_vals['u'].reshape(res, res)
    surf2 = ax2.plot_surface(ms_x, ms_y, exact_sol_grid, cmap='jet', edgecolor='none')
    ax2.set_title("Exact Solution")
    ax2.set_xlabel('$x_1$');
    ax2.set_ylabel('$x_2$');
    ax2.set_zlabel('u(x,y)')
    fig.colorbar(surf2, ax=ax2, shrink=0.7, aspect=10)

    plt.savefig(os.path.join(config.results_dir, "solution_3d_surface.png"))
    plt.close()

    print(f"Plots saved to '{config.results_dir}' directory.")
9

if __name__ == "__main__":
    if not (os.path.exists('dataset/ex3.1_5k_4k') and os.path.exists('dataset/ex3.2_5k_4k')):
        print("Dataset files not found. Please run 'generate_data.py' first.")
    else:
        solve_pde(Config(example_name="Example3.2"))
        print("\n\n" + "=" * 80 + "\n\n")
        solve_pde(Config(example_name="Example3.1"))