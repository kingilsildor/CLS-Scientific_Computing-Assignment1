import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from numba import njit, prange

from master_function import init_grid, compute_diffusion, update_grid, successive_over_relaxation
from master_function import jacobi, gauss_seidel

N = 50
D = 5
T = 0.5
N_TIME_STEPS = 1000
dx = 1 / N
dt = T / N_TIME_STEPS

grid_copy, _ = init_grid(N)

omega_values = [1.7, 1.8, 1.9]
# Empirically found optimal ω: 1.8941
for omega in omega_values:
        grid_copy, _ = init_grid(N)  # Reset grid for each ω
        _, residuals_sor, _ = successive_over_relaxation(grid_copy, _, omega)
        plt.plot(residuals_sor, label=f'ω = {omega}', linestyle=':', linewidth=2)
        

# Reinitialize grid before running Jacobi
grid_copy, _ = init_grid(N)
_, residuals_jacobi, _ = jacobi(grid_copy)
plt.plot(residuals_jacobi, label='Jacobi', linestyle='--', linewidth=2)

# Reinitialize grid before running Gauss-Seidel
grid_copy, _ = init_grid(N)
_, residuals_gauss, _ = gauss_seidel(grid_copy)
plt.plot(residuals_gauss, label='Gauss-Seidel', linestyle='-.', linewidth=2)

plt.xlabel('Iteration')
plt.ylabel('Residual (Infinity Norm)')
plt.yscale('log')  # Log scale to better visualize convergence rates
plt.legend()
plt.title('Convergence Comparison of Iterative Methods')

plt.show()