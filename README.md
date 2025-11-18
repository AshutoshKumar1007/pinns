# PINNs (Physics-Informed Neural Networks) — Biharmonic Examples

Minimal project implementing PINNs for biharmonic PDE examples (Example 3.1 and Example 3.2).

Quick summary
- Purpose: train neural networks to satisfy a biharmonic PDE and boundary conditions using automatic differentiation.
- Primary code lives inside the `attempt-1` and `attempt-2` folders (two training attempts / variations).

Requirements
- Create a Python virtual environment and install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Install torch separately if you need a CUDA-enabled wheel; see https://pytorch.org/
```

Quick start
- Generate datasets (saves `dataset/*` files):

```powershell
python attempt-1/generate_data.py
# or
python attempt-2/generate_data.py
```

- Run the solver (example):

```powershell
python attempt-1/run_solver.py
# or to run attempt-2
python attempt-2/run_solver.py
```

Key files
- `attempt-1/run_solver.py` — main training/evaluation script for Example 3.1 / 3.2
- `attempt-1/model.py` — network architecture and weight init
- `attempt-1/pde.py` — PDE residuals and loss assembly
- `attempt-1/generate_data.py` — generates interior and boundary samples and ground truth
- `attempt-1/tools.py` — small helper to convert numpy arrays to torch tensors

Notes
- The repository uses PyTorch for automatic differentiation. Choose the correct `torch` wheel for CUDA if you want GPU acceleration.
- Datasets are stored under `attempt-1/dataset/` and `attempt-2/dataset/` after running `generate_data.py`.

License
- This project is provided under the MIT License — see the `LICENSE` file.

Notebooks
- Location: `Notebooks/` contains Jupyter notebooks and a couple of pre-trained checkpoint folders.
- What you'll find:
	- `P2example1.ipynb`, `P2example2.ipynb` — example notebooks demonstrating training, evaluation and plotting for the biharmonic PINNs.
	- `biharmonic_pinns_version2/` and `Q2biharmonic_pinns/` — checkpoints (PyTorch `.pt` files) saved from earlier runs. These are large model files kept for convenience.
- How to use:
	- Start Jupyter Lab/Notebook and open one of the notebooks:

```powershell
jupyter lab
# or
jupyter notebook Notebooks/P2example2.ipynb
```

	- The notebooks are runnable end-to-end if you have the same Python environment and the required checkpoints available in the `Notebooks/` subfolders. If a checkpoint is missing, the notebooks usually include code cells to train from scratch (long-running).

Notes about checkpoints
- Model checkpoint files are large and may not be included in some clones of this repository (or may be managed separately). If a notebook expects a `.pt` file and it is missing, either run the training cell in the notebook or copy the checkpoint into the appropriate `Notebooks/*` folder.

