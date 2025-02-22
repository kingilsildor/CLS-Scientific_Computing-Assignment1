import matplotlib.pyplot as plt
import numpy as np

# Import functions from your master module
from master_function import init_grid, add_rectangle, successive_over_relaxation

# Fixed grid size
N = 50

# Define a range of ω values to test
omega_range = np.linspace(1.7, 1.95, 18)

# We'll determine the optimal ω based on the minimum iterations required.
min_iters_no_shape = float('inf')
best_omega_no_shape = None
min_iters_with_shape = float('inf')
best_omega_with_shape = None

# Lists to record iterations vs ω for plotting
iterations_no_shape = []
iterations_with_shape = []

# Define rectangle parameters (for the "with shape" case)
x_start = int(N * 0.3)
y_start = int(N * 0.3)
width   = int(N * 0.2)
height  = int(N * 0.2)

for omega in omega_range:
    # --- Case 1: Without any shape ---
    grid_ns, mask_ns = init_grid(N)  # init_grid returns (grid, object_mask)
    # Run SOR and extract the iteration count (assumed to be the third return value)
    _, _, iters_ns = successive_over_relaxation(grid_ns, mask_ns, omega)
    iterations_no_shape.append(iters_ns)
    if iters_ns < min_iters_no_shape:
        min_iters_no_shape = iters_ns
        best_omega_no_shape = omega

    # --- Case 2: With a rectangle added ---
    grid_shape, mask_shape = init_grid(N)
    add_rectangle(mask_shape, x_start, y_start, width, height)
    _, _, iters_shape = successive_over_relaxation(grid_shape, mask_shape, omega)
    iterations_with_shape.append(iters_shape)
    if iters_shape < min_iters_with_shape:
        min_iters_with_shape = iters_shape
        best_omega_with_shape = omega

print("Best ω without shape:", best_omega_no_shape, "in", min_iters_no_shape, "iterations")
print("Best ω with shape:   ", best_omega_with_shape, "in", min_iters_with_shape, "iterations")

# Best ω without shape: 1.8911764705882352 in 203 iterations
# Best ω with shape:    1.861764705882353 in 136 iterations

# Plot iterations versus ω for both cases
plt.figure(figsize=(8, 6))
plt.plot(omega_range, iterations_no_shape, marker='o', linestyle='-', label='No Shape')
plt.plot(omega_range, iterations_with_shape, marker='s', linestyle='-', label='With Shape')
plt.xlabel("Relaxation Factor (ω)")
plt.ylabel("Iterations to Convergence")
plt.title("Effect of Shape on SOR Convergence (N = 50)")
plt.legend()
plt.grid()
plt.show()
