from time import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from numba import njit

start_time = time()


def init_grid(N):
    grid = np.zeros((N+1, N+1))
    grid[0, :] = 1
    return grid


@njit()
def compute_diffusion(grid, new_grid, D, dt, dx, N):
    for i in range(1, N):
        for j in range(1, N):
            new_grid[i, j] = grid[i, j] + ((D * dt / dx) ** 2) * (
                grid[(i + 1), j]
                + grid[(i - 1), j]
                + grid[i, (j + 1)]
                + grid[i, (j - 1)]
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


def jacobi(grid, epsilon=1e-5, max_iterations=100000):
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
    """
    results = []
    results.append(grid)
    N = grid.shape[0] - 1

    delta = 10000 # Just a big number
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
        grid = new_grid

        k +=1
    
    return results


def gauss_seidel(grid, epsilon=1e-5, max_iterations=100000):
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
    """
    results = []
    results.append(grid.copy())
    N = grid.shape[0] - 1

    delta = 10000 # Just a big number
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

        k += 1
    
    return results


def diffusion_animation(results, save_animation=False, animation_name="diffusion.mp4"):
    """
    Create an animation of the diffusion process

    Params:
    -------
    - results (List[np.ndarray]): A list of spatial grids at each iteration
    - save_animation (bool, optional): Whether to save the animation. Defaults to False.
    - animation_name (str, optional): The name of the animation file. Defaults to "diffusion.mp4".

    Returns:
    --------
    - HTML: The animation
    """
    # Set up the figure
    init_grid = results[0]
    fig = plt.figure()
    im = plt.imshow(init_grid, cmap='inferno', interpolation="none", animated=True)
    plt.colorbar()

    # Animation function, called sequentally
    def animate(i):
        im.set_array(results[i])
        return (im,)

    # Call the animator
    anim = FuncAnimation(
        fig, animate, frames=len(results), interval=200, blit=True
    )

    # Save the animation as an mp4
    if save_animation:
        anim.save(animation_name, fps=30, extra_args=["-vcodec", "libx264"])

    return HTML(anim.to_html5_video())

# N = 50
# D = 5
# T = 0.5
# N_TIME_STEPS = 1000
# dx = 1 / N
# dt = T / N_TIME_STEPS

# grid = init_grid(N)
# update_grid(grid, D, dt, dx, N_TIME_STEPS, smooth=True)

# end_time = time()
# print(f"Execution time: {end_time - start_time:.2f} seconds.")