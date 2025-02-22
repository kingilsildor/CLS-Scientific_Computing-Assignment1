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

grid = init_grid(N)

plt.figure(figsize=(8, 6))

omega_range = np.linspace(1.7, 2.0, 18)  # Test values from 1.0 to 2.0
iterations_needed = []

for omega in omega_range:
    grid_copy, _ = init_grid(N)  # Reset grid for each ω
    _, _, iterations = successive_over_relaxation(grid_copy, _, omega)
    iterations_needed.append(iterations)

optimal_omega = omega_range[np.argmin(iterations_needed)]
print(f"Empirically found optimal ω: {optimal_omega:.4f}")

# Empirically found optimal ω: 1.8941

plt.plot(omega_range, iterations_needed, marker='o')
plt.xlabel("Relaxation Factor (ω)")
plt.ylabel("Iterations to Convergence")
plt.title("Empirical Search for Optimal ω")
plt.grid()
plt.show()