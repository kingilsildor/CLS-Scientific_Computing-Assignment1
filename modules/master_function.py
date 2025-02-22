import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from numba import njit, prange

def init_grid(N):
    """Initialize the computational grid and object mask."""
    grid = np.zeros((N+1, N+1))  # Main concentration grid
    grid[0, :] = 1  # Boundary condition

    object_mask = np.zeros((N+1, N+1), dtype=np.int8)  # Mask for objects (1 = object, 0 = free)

    return grid, object_mask

def add_rectangle(mask, x_start, y_start, width, height):
    """Adds a rectangular object by marking it in the object mask."""
    mask[x_start:x_start+width, y_start:y_start+height] = 1

@njit(parallel=True, fastmath=True)

def compute_diffusion(grid, new_grid, object_mask, D, dt, dx, N):
    for i in prange(1, N):
        for j in prange(1, N):
            if object_mask[i, j] == 1:
                new_grid[i, j] = 0  # If it's part of an object, keep concentration at zero
            else:
                new_grid[i, j] = grid[i, j] + ((D * dt / dx) ** 2) * (
                    grid[i + 1, j]
                    + grid[i - 1, j]
                    + grid[i, j + 1]
                    + grid[i, j - 1]
                    - 4 * grid[i, j]
                )
    return new_grid

def update_grid(grid, D, dt, dx, N_TIME_STEPS, smooth=False):
    N = grid.shape[0] - 1
    new_grid = grid.copy()

    fig, ax = plt.subplots(figsize=(6, 6))
    img = ax.imshow(grid, cmap="inferno", interpolation="bilinear")
    ax.set_title("Diffusion Simulation")

    def _update(frame, new_grid):
        new_grid = compute_diffusion(grid, new_grid, D, dt, dx, N)

        grid[0, :], grid[-1, :] = 1, 0
        grid[:] = new_grid
        if smooth:
            grid[:] = scipy.ndimage.gaussian_filter(grid, sigma=0.5)

        img.set_data(grid)
        ax.set_title(f"Time step {frame + 1}")
        return [img]

    ani = FuncAnimation(
        fig, _update, frames=N_TIME_STEPS, blit=True, interval=50, fargs=(new_grid,)
    )
    ani.save("results/diffusion_simulation.mp4", writer="ffmpeg", fps=60)

def jacobi(grid, epsilon=1e-5, max_iterations=5000):
    """
    Run the Jacobi method to solve the time independent diffusion equation

    Params:
    -------
    - grid (np.ndarray): The initial spatial grid
    - epsilon (float, optional): The convergence criterion. Defaults to 1e-5.
    - max_iterations (int, optional): The maximum number of iterations. Defaults to 100000.

    Returns:
    --------
    - results (List[np.ndarray]): A list of spatial grids at each iteration
    - residuals (List[float]): The residuals at each iteration
    - k (int): The number of iterations
    """
    results = []
    residuals = []
    results.append(grid)
    N = grid.shape[0] - 1

    delta = float('inf')
    k = 0

    while delta > epsilon and k < max_iterations:
        new_grid = np.zeros((N+1, N+1))
        new_grid[0, :] = 1

        for i in range(1, N):
            for j in range(1, N):
                new_grid[i][j] = 1/4 * (grid[i+1][j] + grid[i-1][j] + grid[i][j+1] + grid[i][j-1])
        
            # Boundary conditions
            new_grid[i][0] = 1/4 * (grid[i+1][0] + grid[i-1][0] + grid[i][1] + grid[i][N-1])
            new_grid[i][N] = new_grid[i][0]

        delta = np.max(np.abs(new_grid - grid))

        results.append(new_grid)
        residuals.append(delta)
        grid = new_grid

        k +=1
    
    return results, residuals, k

def gauss_seidel(grid, epsilon=1e-5, max_iterations=5000):
    """
    Run the Gauss Seidel method to solve the time independent diffusion equation

    Params:
    -------
    - grid (np.ndarray): The initial spatial grid
    - epsilon (float, optional): The convergence criterion. Defaults to 1e-5.
    - max_iterations (int, optional): The maximum number of iterations. Defaults to 100000.

    Returns:
    --------
    - results (List[np.ndarray]): A list of spatial grids at each iteration
    - residuals (List[float]): The residuals at each iteration
    - k (int): The number of iterations
    """
    results = []
    residuals = []
    results.append(grid.copy())
    N = grid.shape[0] - 1

    delta = float('inf')
    k = 0

    while delta > epsilon and k < max_iterations:
        delta = 0

        # First column
        for i in range(1, N):
            old_cell = grid[i][0].copy()
            grid[i][0] = 1/4 * (grid[i+1][0] + grid[i-1][0] + grid[i][1] + grid[i][N-1])

            if np.abs(grid[i][0] - old_cell) > delta:
                delta = np.abs(grid[i][0] - old_cell)

        for j in range(1, N):
            for i in range(1, N):
                old_cell = grid[i][j].copy()
                grid[i][j] = 1/4 * (grid[i+1][j] + grid[i-1][j] + grid[i][j+1] + grid[i][j-1])

                if np.abs(grid[i][j] - old_cell) > delta:
                    delta = np.abs(grid[i][j] - old_cell)
        
        # Last column
        for i in range(1, N):
            old_cell = grid[i][N].copy()
            grid[i][N] = 1/4 * (grid[i+1][0] + grid[i-1][0] + grid[i][1] + grid[i][N-1])

            if np.abs(grid[i][N] - old_cell) > delta:
                delta = np.abs(grid[i][N] - old_cell)
            
        results.append(grid.copy())
        residuals.append(delta)

        k += 1
    
    return results, residuals, k

def successive_over_relaxation(grid, object_mask, omega=1.8, epsilon=1e-5, max_iterations=5000):
    """
    Run the Successive Under Relaxation method to solve the time independent diffusion equation

    Params:
    -------
    - grid (np.ndarray): The initial spatial grid
    - epsilon (float, optional): The convergence criterion. Defaults to 1e-5.
    - max_iterations (int, optional): The maximum number of iterations. Defaults to 100000.
    - omega (float, optional): The relaxation factor. Defaults to 1.8

    Returns:
    --------
    - results (List[np.ndarray]): A list of spatial grids at each iteration
    - residuals (List[float]): The residuals at each iteration
    - k (int): The number of iterations
    """
    residuals = []
    results = []
    results.append(grid.copy())
    N = grid.shape[0] - 1

    delta = float('inf')
    k = 0

    while delta > epsilon and k < max_iterations:
        delta = 0

        #First column
        for i in range(1, N):
            if object_mask[i, 0] == 1:
                continue
            
            old_cell = grid[i][0].copy()
            grid[i][0] = omega/4 * (grid[i+1][0] + grid[i-1][0] + grid[i][1] + grid[i][N-1]) + (1-omega) * grid[i][0]
            
            if np.abs(grid[i][0] - old_cell) > delta:
                delta = np.abs(grid[i][0] - old_cell)
        
        for j in range(1, N):
            for i in range(1, N):
                if object_mask[i, j] == 1:
                    continue

                old_cell = grid[i][j].copy()
                grid[i][j] = omega/4 * (grid[i+1][j] + grid[i-1][j] + grid[i][j+1] + grid[i][j-1]) + (1-omega) * grid[i][j]
                
                delta = max(delta, np.abs(grid[i][j] - old_cell))


        # Last column
        for i in range(1, N):
            if object_mask[i, N] == 1:
                continue

            old_cell = grid[i][N].copy()
            grid[i][N] = omega/4 * (grid[i+1][N] + grid[i-1][N] + grid[i][0] + grid[i][N-1]) + (1-omega) * grid[i][N]
            
            if np.abs(grid[i][N] - old_cell) > delta:
                delta = np.abs(grid[i][N] - old_cell)

        results.append(grid.copy())
        residuals.append(delta)

        k +=1
    
    return results, residuals, k
