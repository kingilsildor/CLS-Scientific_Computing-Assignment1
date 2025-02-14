import numpy as np
from numba import njit
import scipy.ndimage
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def init_grid(N):
    grid = np.zeros((N, N))
    grid[0, :] = 1
    assert grid.shape == (N, N)
    return grid

def create_empty_dict(length):
    keys = [f"key_{i}" for i in np.arange(length)]
    values = np.full(length, None)
    empty_dict = dict(zip(keys, values))
    assert len(empty_dict) == length
    return empty_dict

@njit()
def compute_diffusion(grid, new_grid, D, dt, dx, N):
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            new_grid[i, j] = grid[i, j] + ((D * dt / dx) ** 2) * (
                grid[(i + 1) % N, j]
                + grid[(i - 1) % N, j]
                + grid[i, (j + 1)]
                + grid[i, (j - 1)]
                - 4 * grid[i, j]
            )
    return new_grid

def update_grid(grid, D, dt, dx, N, smooth):
    new_grid = np.empty_like(grid)
    new_grid = compute_diffusion(grid, new_grid, D, dt, dx, N)
    new_grid[0, :], new_grid[-1, :] = 1, 0
    if smooth:
        new_grid[:] = scipy.ndimage.gaussian_filter(new_grid, sigma=0.5)
    return new_grid

def run_diffusion(N, D, dt, dx, N_TIME_STEPS, smooth=True):
    grid = init_grid(N)
    grid_dict = create_empty_dict(N_TIME_STEPS)

    for key in grid_dict.keys():
        grid = update_grid(grid, D, dt, dx, N, smooth)
        grid_dict[key] = grid.copy()

    np.savez("data/diffusion_grid.npz", **grid_dict)

def read_diffusion_grid():
    data = np.load("data/diffusion_grid.npz")
    return data

def animate_diffusion(data):
    fig, ax = plt.subplots()
    cax = ax.imshow(data['key_0'], cmap='hot', interpolation='nearest')
    
    def update(frame):
        cax.set_array(data[f'key_{frame}'])
        return cax,

    ani = FuncAnimation(fig, update, frames=len(data.files), interval=50, blit=True)
    ani.save("results/diffusion_simulation.mp4", writer="ffmpeg", fps=60)


# Example


N = 200
D = 1
T = 1
N_TIME_STEPS = 1000
dx = 1 / N
dt = T / N_TIME_STEPS

run_diffusion(N, D, dt, dx, N_TIME_STEPS, smooth=True)
data = read_diffusion_grid()
animate_diffusion(data)