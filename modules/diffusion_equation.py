import glob
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
from numba import njit
from scipy.special import erfc
from tqdm import tqdm


def init_grid(N: int, init_condition: str = "top") -> np.ndarray:
    """
    Initialize the grid with heat diffusion

    Params
    ------
    - N (int): Grid size
    - init_condition (str): Initial condition. Options: "center", "random", "top", "bottom". Default: "top"

    Returns
    -------
    - np.ndarray: Grid with heat diffusion
    """
    assert N > 0, "Invalid grid size"
    grid = np.zeros((N, N))
    assert grid.shape == (N, N), "Invalid grid shape"

    assert init_condition in ["center", "random", "top", "bottom"], (
        "Invalid initial condition"
    )
    # Enables different initial conditions
    if init_condition == "center":
        grid[N // 2, N // 2] = 1
    elif init_condition == "random":
        grid = np.random.rand(N, N)
    elif init_condition == "top":
        grid[0, :], grid[-1, :] = 1, 0
    elif init_condition == "bottom":
        grid[0, :], grid[-1, :] = 0, 1

    assert isinstance(grid, np.ndarray), "Invalid grid type"
    assert grid.shape == (N, N), "Invalid grid shape"
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

    for i in range(1, N - 1):
        for j in range(N):
            new_grid[i, j] += diffusion_coefficient * (
                grid[i + 1, j]
                + grid[i - 1, j]
                + grid[i, (j + 1) % N]
                + grid[i, (j - 1) % N]
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
):
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
    - Write the results to a file
    """
    if ((4 * D * dt) / (dx**2)) > 1:
        warnings.warn("Stability condition not met")
    diffusion_coefficient = (D * dt / dx) ** 2
    assert isinstance(diffusion_coefficient, float), "Invalid diffusion coefficient"

    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    grid = init_grid(N, init_condition)
    np.save(os.path.join(output_dir, "diffusion_step_0.npy"), grid)

    for i in tqdm(range(1, N_time_steps), desc="Calculating diffusion"):
        grid = diffuse(grid, diffusion_coefficient)
        np.save(os.path.join(output_dir, f"diffusion_step_{i}.npy"), grid)


@njit
def jacobi(
    grid: np.ndarray, epsilon: float = 1e-5, max_iterations: int = 100000
) -> np.ndarray:
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

    delta = float("inf")
    k = 0

    while delta > epsilon and k < max_iterations:
        new_grid = np.zeros((N + 1, N + 1))
        new_grid[0, :] = 1

        for i in range(1, N):
            for j in range(1, N):
                new_grid[i][j] = (
                    1
                    / 4
                    * (
                        grid[i + 1][j]
                        + grid[i - 1][j]
                        + grid[i][j + 1]
                        + grid[i][j - 1]
                    )
                )

            # Boundary conditions
            new_grid[i][0] = (
                1 / 4 * (grid[i + 1][0] + grid[i - 1][0] + grid[i][1] + grid[i][N - 1])
            )
            new_grid[i][N] = new_grid[i][0]

        delta = np.max(np.abs(new_grid - grid))

        results.append(new_grid)
        grid = new_grid

        k += 1

    return results


def gauss_seidel(
    grid: np.ndarray, epsilon: float = 1e-5, max_iterations: int = 100000
) -> np.ndarray:
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

    delta = float("inf")
    k = 0

    while delta > epsilon and k < max_iterations:
        delta = 0

        # First column
        for i in range(1, N):
            old_cell = grid[i][0].copy()
            grid[i][0] = (
                1 / 4 * (grid[i + 1][0] + grid[i - 1][0] + grid[i][1] + grid[i][N - 1])
            )

            if np.abs(grid[i][0] - old_cell) > delta:
                delta = np.abs(grid[i][0] - old_cell)

        for j in range(1, N):
            for i in range(1, N):
                old_cell = grid[i][j].copy()
                grid[i][j] = (
                    1
                    / 4
                    * (
                        grid[i + 1][j]
                        + grid[i - 1][j]
                        + grid[i][j + 1]
                        + grid[i][j - 1]
                    )
                )

                if np.abs(grid[i][j] - old_cell) > delta:
                    delta = np.abs(grid[i][j] - old_cell)

        # Last column
        for i in range(1, N):
            old_cell = grid[i][N].copy()
            grid[i][N] = (
                1 / 4 * (grid[i + 1][0] + grid[i - 1][0] + grid[i][1] + grid[i][N - 1])
            )

            if np.abs(grid[i][N] - old_cell) > delta:
                delta = np.abs(grid[i][N] - old_cell)

        results.append(grid.copy())

        k += 1

    return results


def animate_diffusion(
    results: np.ndarray | None = None,
    steps: int | None = None,
    skips: int = 2,
    save_animation: bool = False,
    animation_name: str = "diffusion.mp4",
) -> HTML:
    """
    Animate the diffusion of heat in a grid

    Params
    ------
    - results (np.ndarray): Grid with heat diffusion over time. Default: None
    - steps (int): Number of time steps. Default: None
    - skips (int): Number of frames to skip. Default: 2
    - save_animation (bool): Save the animation as a video file. Default: False
    - animation_name (str): Name of the animation file. Default: "diffusion.mp4"

    Returns
    -------
    - HTML: Animation of the diffusion of heat in a grid
    - Video file with the animation of the diffusion of heat in a grid
    """
    if results is None:
        assert steps is not None, "Invalid number of steps"
        init_grid = np.load("data/diffusion_step_0.npy")
        indices = np.linspace(0, steps - 1, num=skips, dtype=int)

        results = np.zeros((len(indices), *init_grid.shape))

        def _load_data(results: np.ndarray, indices: np.ndarray) -> np.ndarray:
            """
            Load the diffusion data

            Params
            ------
            - results (np.ndarray): Grid with heat diffusion over time
            - indices (np.ndarray): Indices to load the data

            Returns
            -------
            - np.ndarray: Updated grid with heat diffusion over time
            """
            for i, frame in enumerate(indices):
                results[i] = np.load(f"data/diffusion_step_{frame}.npy")
            return results

        results = _load_data(results, indices)
    else:
        assert isinstance(results, np.ndarray), (
            "Invalid results type, should be np.ndarray"
        )

        indices = np.arange(0, results.shape[0], skips)
        results = results[indices]

    steps = results.shape[0]
    progress_bar = tqdm(total=steps, desc="Animating diffusion")

    fig, ax = plt.subplots()
    init_grid = results[0]
    im = ax.imshow(init_grid, cmap="inferno", interpolation="nearest")
    plt.colorbar(im)

    def _animate(frame):
        """
        Update the animation frame

        Params
        ------
        - frame (int): Frame number

        Returns
        -------
        - im: Updated image

        """
        im.set_array(results[frame])
        ax.set_title(f"Time step {frame}")
        progress_bar.update(1)
        return (im,)

    ani = FuncAnimation(fig, _animate, frames=steps, interval=50, blit=True)
    if save_animation:
        ani.save(
            "results/heat_diffusion.mp4",
            writer="ffmpeg",
            fps=60,
        )
    progress_bar.close()
    plt.cla()

    return HTML(ani.to_html5_video())


def plot_grid(frame: int = 999999, times: list | None = None):
    """
    Plot the grid with heat diffusion at a specific time step.
    If multiple time steps are provided, it arranges them in a 2x2 layout.

    Params
    ------
    - frame (int): Time step to plot. Default: 999999.
    - times (list of float, optional): Specific time steps to plot.

    Returns
    -------
    - Displays and saves the heat diffusion plots.
    """
    if times is None:
        frames = [frame]
        rows, cols = 1, 1
        figsize = (5, 5)
    else:
        assert isinstance(times, list), "Invalid times type"
        assert all(isinstance(t, float) for t in times), "Invalid times values"

        frames = [int(t * 1_000_000) - 1 for t in times]
        rows, cols = 2, 2
        figsize = (8, 8)

        while len(frames) < rows * cols:
            frames.append(frames[-1])

    grids = [np.load(f"data/diffusion_step_{frame}.npy") for frame in frames]

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=figsize,
        dpi=300,
        gridspec_kw={"wspace": 0.1, "right": 0.85},
    )
    if times is not None:
        fig.suptitle("Heat Diffusion Over Time", fontsize=14)

    axes = np.array(axes).reshape(-1)

    # Make it so that all plots share the same colorbar
    vmin, vmax = min(grid.min() for grid in grids), max(grid.max() for grid in grids)
    cmap = "inferno"

    for ax, grid, frame in zip(axes, grids, frames):
        im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(f"Time step {frame}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[len(frames) :]:
        ax.axis("off")

    # Anchor the colorbar to the right of the plots
    cbar_ax = fig.add_axes([0.87, 0.2, 0.03, 0.6])
    fig.colorbar(im, cax=cbar_ax)

    save_path = (
        "results/diffusion_grid.png"
        if times is None
        else "results/diffusion_grid_comparison.png"
    )
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def analytical_solution(y, t, D, terms=50):
    """
    Calculate the analytical solution of the diffusion equation

    Params:
    -------
    - y (np.ndarray): Spatial grid
    - t (float): Time
    - D (float): Diffusion coefficient
    - terms (int): Number of terms in the series expansion. Default: 50

    Returns:
    --------
    - C (np.ndarray): Analytical solution of the diffusion equation
    """
    C = np.zeros_like(y)
    for i in range(terms):
        term1 = erfc((1 - y + 2 * i) / (2 * np.sqrt(D * t)))
        term2 = erfc((1 + y + 2 * i) / (2 * np.sqrt(D * t)))
        C += term1 - term2

    assert isinstance(C, np.ndarray), "Invalid solution type"
    assert C.shape == y.shape, "Invalid solution shape"
    return C


def plot_concentration(steps: int, times: np.ndarray):
    """
    Plot the concentration of a diffusing substance over time.

    Params:
    -------
    - steps (int): The number of time steps.
    - times (np.ndarray): The times at which to plot the concentration.

    Returns:
    --------
    - Plot of the concentration of a diffusing substance over time.
    """
    init_grid = np.load("data/diffusion_step_0.npy")
    N = init_grid.shape[1] - 1
    y = np.linspace(0, 1, N + 1)
    y = np.flip(y)

    plt.figure(dpi=300)

    for t in times:
        frame = int(t * steps) - 1
        grid = np.load(f"data/diffusion_step_{frame}.npy")
        C_sim = grid[:, N // 2]
        C_ana = analytical_solution(y, t, D=1)
        assert C_sim.shape == C_ana.shape, "Invalid concentration shape"

        (line,) = plt.plot(y, C_sim, label=f"t={t}")
        plt.plot(y, C_ana, marker="o", linestyle="None", color=line.get_color())

    plt.plot([], [], label="Simulated solutions (solid)", color="black")
    plt.plot([], [], "o", label="Analytical solutions (dots)", color="black")

    plt.xlabel("y")
    plt.ylabel("concentration")
    plt.title("Concentration of Diffusing over time")
    plt.legend(
        loc="upper left",
        frameon=False,
    )
    plt.tight_layout()

    plt.savefig("results/diffusion_concentration.png")
    plt.show()


def delete_files():
    """ "
    Delete the files in the data directory
    """
    output_dir = "data"
    files = glob.glob(f"{output_dir}/*")
    for f in files:
        os.remove(f)


def main():
    # N = 30
    # D = 1
    # dx = 1 / N
    # dt = 0.001
    # steps = 1_000_000
    # calc_diffusion(N, steps, D, dx, dt)

    # animate_diffusion(steps=steps, skips=1000, save_animation=True)
    plot_grid(times=[0.001, 0.01, 0.1, 1.0])
    # plot_concentration(steps, [0.001, 0.01, 0.1, 1])


if __name__ == "__main__":
    main()
