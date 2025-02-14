from time import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
from matplotlib.animation import FuncAnimation
from numba import njit




def init_grid(N):
    grid = np.zeros((N, N))
    grid[0, :] = 1
    return grid

def create_empty_dict(length):
    keys = [f"key_{i}" for i in np.arange(length)]
    values = np.full(length, None)
    empty_dict = dict(zip(keys, values))
    assert len(empty_dict) == length
    return empty_dict


@njit()
def compute_diffusion(grid, D, dt, dx, N):
    diffusion_coefficient = ((D * dt / dx) ** 2)
    new_grid = np.empty_like(grid)

    for i in range(1, N - 1):
        for j in range(1, N - 1):
            new_grid[i, j] = grid[i, j] + diffusion_coefficient * (
                grid[(i + 1) % N, j]
                + grid[(i - 1) % N, j]
                + grid[i, (j + 1)]
                + grid[i, (j - 1)]
                - 4 * grid[i, j]
            )
    return new_grid


def update_grid(grid, D, dt, dx, N_TIME_STEPS, smooth=True):
    N = grid.shape[0]
    grid_dict = create_empty_dict(N_TIME_STEPS)

    for i in grid_dict.keys():
        grid = compute_diffusion(grid, D, dt, dx, N)

        grid[0, :], grid[-1, :] = 1, 0
        if smooth:
            grid[:] = scipy.ndimage.gaussian_filter(grid, sigma=0.5)

        grid_dict[i] = grid.copy()

    return grid_dict

def calculate_diffusion(N, D, dt, dx, N_TIME_STEPS):
    grid = init_grid(N)
    data = update_grid(grid, D, dt, dx, N_TIME_STEPS, smooth=True)

    np.savez("data/diffusion_grid.npz", **data)

def animate_diffusion(data=None):
    data = np.load("data/diffusion_grid.npz")
    fig, ax = plt.subplots()

    cax = ax.imshow(data['key_0'], cmap='inferno', interpolation='nearest')
    
    def update(frame):
        cax.set_array(data[f'key_{frame}'])
        ax.set_title(f"Time step {frame + 1}")
        return cax,

    N_frames = len(data)
    ani = FuncAnimation(fig, update, frames=N_frames, interval=50, blit=True)
    ani.save("results/diffusion_simulation.mp4", writer="ffmpeg", fps=60)


N = 200
D = 1
T = 1
N_TIME_STEPS = 2000
dx = 1 / N
dt = T / N_TIME_STEPS

calculate_diffusion(N, D, dt, dx, N_TIME_STEPS)
animate_diffusion()
