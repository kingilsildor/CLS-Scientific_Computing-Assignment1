import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.ndimage as ndimage
from numba import njit
from tqdm import tqdm


def init_grid(N: int, N_time_steps: int, init_condition: str) -> np.ndarray:
    grid = np.zeros((N_time_steps, N, N))
    assert grid.shape == (N_time_steps, N, N), "Invalid grid shape"

    assert init_condition in ["center", "random", "top", "bottom"], (
        "Invalid initial condition"
    )
    if init_condition == "center":
        grid[0, N // 2, N // 2] = 1
    elif init_condition == "random":
        grid[0] = np.random.rand(N, N)
    elif init_condition == "top":
        grid[0, 0, :], grid[0, -1, :] = 1, 0
    elif init_condition == "bottom":
        grid[0, 0, :], grid[0, -1, :] = 0, 1

    return grid

@njit
def diffuse(grid: np.ndarray, diffusion_coefficient: float) -> np.ndarray:
    new_grid = np.copy(grid)
    N = grid.shape[1]
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
    return new_grid


def calc_diffusion(
    N: int,
    N_time_steps: int,
    D: float,
    dx: float,
    dt: float,
    init_condition: str = "top",
) -> np.ndarray:
    diffusion_coefficient = (D * dt / dx) ** 2
    grid = init_grid(N, N_time_steps, init_condition)

    for i in tqdm(range(1, N_time_steps), desc="Calculating diffusion"):
        grid[i] = diffuse(grid[i - 1], diffusion_coefficient)
        grid[i] = ndimage.gaussian_filter(grid[i], sigma=0.4)
    np.save("data/diffusion.npy", grid)
    return grid


def animate_diffusion(grid: np.ndarray, skips: int = 2) -> None:
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

def plot_grid(grid, frame=-1):
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
