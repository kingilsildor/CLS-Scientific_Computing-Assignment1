import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.ndimage as ndimage
from numba import njit

def init_grid(N: int, N_time_steps: int, init_condition: str) -> np.ndarray:
    grid = np.zeros((N_time_steps, N, N)) 
    assert grid.shape == (N_time_steps, N, N), "Invalid grid shape"
    
    assert init_condition in ["center", "random", "top", "bottom"], "Invalid initial condition"
    if init_condition == "center":
        grid[0, N//2, N//2] = 1
    elif init_condition == "random":
        grid[0] = np.random.rand(N, N)
    elif init_condition == "top":
        grid[0, 0, :], grid[0, -1, :] = 1, 0
    elif init_condition == "bottom":
        grid[0, 0, :], grid[0, -1, :] = 0, 1

    return grid

# @njit
def diffuse(grid: np.ndarray, diffusion_coefficient: float) -> np.ndarray:
    new_grid = np.copy(grid)
    N = grid.shape[1]
    for i in range(N):
        for j in range(1, N-1):
            new_grid[i, j] += diffusion_coefficient * (
                grid[(i + 1) % N, j] +
                grid[(i - 1) % N, j] +
                grid[i, j + 1] +
                grid[i, j - 1] - 4 * grid[i, j]
            )
    new_grid[0, :], new_grid[-1, :] = 1, 0
    return new_grid

def calc_diffusion(N: int, N_time_steps: int, D: float, dx: float, dt: float, init_condition: str = "top", smooth: bool = True) -> np.ndarray:
    diffusion_coefficient = ((D * dt / dx) ** 2)
    grid = init_grid(N, N_time_steps, init_condition)
    for i in range(1, N_time_steps):
        grid[i] = diffuse(grid[i-1], diffusion_coefficient)
        if smooth:
            grid[:] = ndimage.gaussian_filter(grid, sigma=0.5)
    return grid

def animate_diffusion(grid : np.ndarray) -> None:
    steps = grid.shape[0]

    fig, ax = plt.subplots()
    cmap = plt.get_cmap('inferno')
    im = ax.imshow(grid[0], cmap=cmap, interpolation='nearest')
    plt.colorbar(im)

    def animate(frame):
        im.set_array(grid[frame])
        ax.set_title(f"Time step {frame}")

    ani = FuncAnimation(fig, animate, frames=steps, interval=50)
    ani.save("heat_diffusion.mp4", writer="ffmpeg", fps=30)

def calc_coverage(grid: np.ndarray) -> float:
    return np.mean(grid)

def main():
    N = 50
    D = 1
    dx = 1/N
    dt = 0.01
    steps = 2000
    grid = calc_diffusion(N, steps, D, dx, dt)
    animate_diffusion(grid)

if __name__ == "__main__":
    main()