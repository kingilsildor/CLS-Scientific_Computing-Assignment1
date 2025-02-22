import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from numba import njit, prange

from master_function import init_grid, compute_diffusion, update_grid, successive_over_relaxation


# Run the simulation
N = 50
D = 5
T = 0.5
N_TIME_STEPS = 1000
dx = 1 / N
dt = T / N_TIME_STEPS

N_values = np.arange(10, 100, 10)  # Test grid sizes from 10 to 90
omega_range = np.linspace(1.7, 1.95, 18)  # Test relaxation factors from 1.7 to 2.0

optimal_omegas = []

for N in N_values:
    print(f"Computing optimal omega for N = {N}...")
    min_iterations = float('inf')
    best_omega = None

    for omega in omega_range:
        grid_copy, _ = init_grid(N)  # Reset grid for each ω
        _, _, iterations = successive_over_relaxation(grid_copy, _, omega)

        if iterations < min_iterations:
            min_iterations = iterations
            best_omega = omega

    optimal_omegas.append(best_omega)

#**Plot N vs Optimal ω**
plt.figure(figsize=(8, 6))
plt.plot(optimal_omegas, N_values, marker='o', linestyle='-')
plt.xlabel("Optimal Relaxation Factor (ω)")
plt.ylabel("Grid Size (N)")
plt.title("Effect of Grid Size (N) on Optimal ω")
plt.grid()
plt.show()