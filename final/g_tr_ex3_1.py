import numpy as np
from sympy import symbols, diff, sin, pi, lambdify

# Define symbols
x1, x2 = symbols('x1 x2')

# Exact solution for Example 3.1
y_sym = (1/(2*pi**2))*sin(pi*x1)*sin(pi*x2)

# Source term f = Δ²y
laplacian_y = diff(y_sym, x1, 2) + diff(y_sym, x2, 2)
bilaplacian_y = diff(laplacian_y, x1, 2) + diff(laplacian_y, x2, 2)
f_sym = bilaplacian_y

# Derivatives for BCs and error calculations
y_x_sym = diff(y_sym, x1, 1)
y_y_sym = diff(y_sym, x2, 1)
y_xx_sym = diff(y_sym, x1, 2)
y_xy_sym = diff(y_sym, x1, x2)
y_yy_sym = diff(y_sym, x2, 2)

# Lambdify all symbolic expressions into fast numpy functions
ldy     = lambdify((x1, x2), y_sym,     'numpy')
ldf     = lambdify((x1, x2), f_sym,     'numpy')
ldy_x   = lambdify((x1, x2), y_x_sym,   'numpy')
ldy_y   = lambdify((x1, x2), y_y_sym,   'numpy')
ldy_xx  = lambdify((x1, x2), y_xx_sym,  'numpy')
ldy_xy  = lambdify((x1, x2), y_xy_sym,  'numpy')
ldy_yy  = lambdify((x1, x2), y_yy_sym,  'numpy')


def from_seq_to_array(items):
    out = []
    for item in items:
        out.append(np.array(item).reshape(-1, 1))
    if len(out) == 1:
        return out[0]
    return out


def data_gen_interior(collocations):
    y_gt = [ldy(x, y) for x, y in collocations]
    f_gt = [ldf(x, y) for x, y in collocations]
    return from_seq_to_array([y_gt, f_gt])


def data_gen_bdry(collocations, normal_vec):
    ybdry_vals = []
    neumann_vals = [] # This is g2 = ∂²u/ ∂n²
    for (x, y), (n1, n2) in zip(collocations, normal_vec):
        ybdry_vals.append(ldy(x, y))
        u_xx = ldy_xx(x, y)
        u_xy = ldy_xy(x, y)
        u_yy = ldy_yy(x, y)
        neumann_vals.append(u_xx*(n1**2) + 2*u_xy*(n1*n2) + u_yy*(n2**2))
    return from_seq_to_array([ybdry_vals, neumann_vals])


def get_all_derivatives(collocations):
    """Returns u and its derivatives up to 2nd order for error calculation."""
    vals = {
        'u':    [ldy(x, y)    for x, y in collocations],
        'u_x':  [ldy_x(x, y)  for x, y in collocations],
        'u_y':  [ldy_y(x, y)  for x, y in collocations],
        'u_xx': [ldy_xx(x, y) for x, y in collocations],
        'u_xy': [ldy_xy(x, y) for x, y in collocations],
        'u_yy': [ldy_yy(x, y) for x, y in collocations],
    }
    return {k: np.array(v).reshape(-1, 1) for k, v in vals.items()}