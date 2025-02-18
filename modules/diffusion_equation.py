import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.ndimage as ndimage
from numba import njit
from tqdm import tqdm


def init_grid(N: int, N_time_steps: int, init_condition: str) -> np.ndarray:
    """
    Initialize the grid with heat diffusion

    Params
    ------
    - N (int): Grid size
    - N_time_steps (int): Number of time steps
    - init_condition (str): Initial condition. Options: "center", "random", "top", "bottom". Default: "top"

    Returns
    -------
    - np.ndarray: Grid with heat diffusion
    """
    grid = np.zeros((N_time_steps, N, N))
    assert grid.shape == (N_time_steps, N, N), "Invalid grid shape"

    assert init_condition in ["center", "random", "top", "bottom"], (
        "Invalid initial condition"
    )
    # Enables different initial conditions
    if init_condition == "center":
        grid[0, N // 2, N // 2] = 1
    elif init_condition == "random":
        grid[0] = np.random.rand(N, N)
    elif init_condition == "top":
        grid[0, 0, :], grid[0, -1, :] = 1, 0
    elif init_condition == "bottom":
        grid[0, 0, :], grid[0, -1, :] = 0, 1

    assert isinstance(grid, np.ndarray), "Invalid grid type"
    assert grid.shape == (N_time_steps, N, N), "Invalid grid shape"
    return grid


@njit
def diffuse(grid: np.ndarray, diffusion_coefficient: float) -> np.ndarray:
    """
    Update the grid with the diffusion equation

    Params
    ------
    - grid (np.ndarray): Grid with heat diffusion
    - diffusion_coefficient (float): Diffusion coefficient

    Returns
    -------
    - np.ndarray: Updated grid with heat diffusion
    """
    new_grid = np.copy(grid)
    N = grid.shape[1]
    assert isinstance(N, int), "Invalid grid size"
    assert N > 0, "Invalid grid size"

    for i in range(N):
        for j in range(1, N - 1):
            new_grid[i, j] += diffusion_coefficient * (
                grid[(i + 1) % N, j]
                + grid[(i - 1) % N, j]
                + grid[i, j + 1]
                + grid[i, j - 1]
                - 4 * grid[i, j]
            )
    new_grid[0, :], new_grid[-1, :] = 1, 0

    assert new_grid.shape == grid.shape, "Invalid grid shape"
    return new_grid


def calc_diffusion(
    N: int,
    N_time_steps: int,
    D: float,
    dx: float,
    dt: float,
    init_condition: str = "top",
) -> np.ndarray:
    """
    Calculate the diffusion of heat in a grid for each time step

    Params
    ------
    - N (int): Grid size
    - N_time_steps (int): Number of time steps
    - D (float): Diffusion coefficient
    - dx (float): Grid spacing
    - dt (float): Time spacing
    - init_condition (str): Initial condition. Options: "center", "random", "top", "bottom". Default: "top"

    Returns
    -------
    - np.ndarray: Grid with heat diffusion over time
    """
    diffusion_coefficient = (D * dt / dx) ** 2
    assert isinstance(diffusion_coefficient, float), "Invalid diffusion coefficient"

    grid = init_grid(N, N_time_steps, init_condition)

    for i in tqdm(range(1, N_time_steps), desc="Calculating diffusion"):
        grid[i] = diffuse(grid[i - 1], diffusion_coefficient)
        grid[i] = ndimage.gaussian_filter(grid[i], sigma=0.4)

    # Option to write away the data
    np.save("data/diffusion_grid.npy", grid)

    assert isinstance(grid, np.ndarray), "Invalid grid type"
    assert grid.shape == (N_time_steps, N, N), "Invalid grid shape"
    return grid


def animate_diffusion(grid: np.ndarray, skips: int = 2) -> None:
    """
    Animate the diffusion of heat in a grid

    Params
    ------
    - grid (np.ndarray): Grid with heat diffusion
    - skips (int): Number of frames to skip

    Returns
    -------
    - Animation of heat diffusion over time
    """
    steps = grid.shape[0]

    fig, ax = plt.subplots()
    cmap = plt.get_cmap("inferno")
    im = ax.imshow(grid[0], cmap=cmap, interpolation="nearest")
    plt.colorbar(im)

    total_steps = steps // skips
    progress_bar = tqdm(total=total_steps, desc="Animating diffusion")

    def animate(frame):
        im.set_array(grid[frame])
        ax.set_title(f"Time step {frame}")
        progress_bar.update(1)

    ani = FuncAnimation(fig, animate, frames=total_steps, interval=50)
    ani.save("results/heat_diffusion.mp4", writer="ffmpeg", fps=60)
    progress_bar.close()
    plt.cla()


def calculate_distance(grid):
    timesteps = grid.shape[0]
    distances = np.zeros(timesteps)

    for t in range(timesteps):
        non_zero_indices = np.argwhere(grid[t, :, :] > 0.01)
        distances[t] = np.max(non_zero_indices[:, 0]) / grid.shape[1]

    plt.plot(distances)
    plt.xlabel("Time")
    plt.ylabel("Distance")
    plt.title("Distance of heat diffusion")
    plt.show()


def plot_grid(grid: np.ndarray, frame: int = -1):
    """
    Plot the grid with heat diffusion at a specific time step

    Params
    ------
    - grid (np.ndarray): Grid with heat diffusion
    - frame (int): Time step to plot. Default: -1 (last time step)

    Returns
    -------
    - Plot of the grid with heat diffusion at a specific time step
    """
    fig, ax = plt.subplots()

    cmap = plt.get_cmap("inferno")
    im = ax.imshow(grid[frame], cmap=cmap, interpolation="nearest")
    plt.colorbar(im)
    plt.show()


def plot_analytical_solution():
    pass


def main():
    N = 50
    D = 1
    dx = 1 / N
    dt = 0.001
    steps = 10000
    grid = calc_diffusion(N, steps, D, dx, dt)
    # animate_diffusion(grid)
    # calculate_distance(grid)
    plot_grid(grid)


if __name__ == "__main__":
    main()
