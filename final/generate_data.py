import pickle as pkl
from scipy.stats import uniform
import numpy as np
import os
import argparse


def generate_random_bdry(Nb):
    bdry_col = uniform.rvs(size=Nb * 2).reshape([Nb, 2])
    for i in range(Nb):
        randind = np.random.randint(0, 2)
        bdry_col[i, randind] = 0.0 if bdry_col[i, randind] <= 0.5 else 1.0
    return bdry_col


def compute_normals(bdry_col, eps=1e-8):
    x, y = bdry_col[:, 0], bdry_col[:, 1]
    n1, n2 = np.zeros_like(x), np.zeros_like(y)
    n1[np.isclose(x, 0.0, atol=eps)] = -1.0
    n1[np.isclose(x, 1.0, atol=eps)] = 1.0
    n2[np.isclose(y, 0.0, atol=eps)] = -1.0
    n2[np.isclose(y, 1.0, atol=eps)] = 1.0
    return n1.reshape(-1, 1), n2.reshape(-1, 1)


def create_dataset(example_name):
    if example_name == "Example3.1":
        import g_tr_ex3_1 as gt
        dataname = 'ex3.1_5k_4k'
    elif example_name == "Example3.2":
        import g_tr_ex3_2 as gt
        dataname = 'ex3.2_5k_4k'
    else:
        raise ValueError("Unknown example name")

    N_interior, N_boundary = 5000, 4000

    domain_data = uniform.rvs(size=N_interior * 2).reshape(N_interior, 2)
    bdry_col = generate_random_bdry(N_boundary)
    normal_vec = np.hstack(compute_normals(bdry_col))

    os.makedirs('dataset/', exist_ok=True)
    with open(f'dataset/{dataname}', 'wb') as pfile:
        pkl.dump(domain_data, pfile)
        pkl.dump(bdry_col, pfile)
        pkl.dump(normal_vec, pfile)

    ygt, fgt = gt.data_gen_interior(domain_data)
    dirichlet_data, neumann_data = gt.data_gen_bdry(bdry_col, normal_vec)

    with open(f"dataset/gt_on_{dataname}", 'wb') as pfile:
        pkl.dump(ygt, pfile)
        pkl.dump(fgt, pfile)
        pkl.dump(dirichlet_data, pfile)
        pkl.dump(neumann_data, pfile)

    print(f"Generated data for {example_name} and saved to 'dataset/{dataname}'")


if __name__ == "__main__":
    create_dataset("Example3.1")
    create_dataset("Example3.2")