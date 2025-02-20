import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import scipy.ndimage as ndimage
from numba import njit
from tqdm import tqdm
from scipy.special import erfc
import warnings


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
    if ((4 * D * dt) / (dx ** 2)) > 1:
        warnings.warn("Stability condition not met")
    diffusion_coefficient = (D * dt / dx) ** 2
    assert isinstance(diffusion_coefficient, float), "Invalid diffusion coefficient"

    results = np.zeros((N_time_steps, N, N))
    results[0] = init_grid(N, init_condition)

    for i in tqdm(range(1, N_time_steps), desc="Calculating diffusion"):
        results[i] = diffuse(results[i - 1], diffusion_coefficient)
        # results[i] = ndimage.gaussian_filter(results[i - 1], sigma=0.4)

    # Option to write away the data
    np.save("data/diffusion_grid.npy", results)

    assert isinstance(results, np.ndarray), "Invalid grid type"
    assert results.shape == (N_time_steps, N, N), "Invalid grid shape"
    return results


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


def diffusion_animation(
    results: np.ndarray,
    save_animation: bool = False,
    animation_name: str = "diffusion.mp4",
) -> HTML:
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
    im = plt.imshow(init_grid, cmap="inferno", interpolation="none", animated=True)
    plt.colorbar()

    # Animation function, called sequentally
    def animate(i):
        im.set_array(results[i])
        return (im,)

    # Call the animator
    anim = FuncAnimation(fig, animate, frames=len(results), interval=200, blit=True)

    # Save the animation as an mp4
    if save_animation:
        anim.save(animation_name, fps=30, extra_args=["-vcodec", "libx264"])

    return HTML(anim.to_html5_video())


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
    D = 1
    times = [0.001, 0.01, 0.1, 1]
    y = np.linspace(0, 1, 100)


    def analytical_solution(y, t, D, terms=50):
        C = np.zeros_like(y)
        for i in range(terms):
            term1 = erfc((1 - y + 2 * i) / (2 * np.sqrt(D * t)))
            term2 = erfc((1 + y + 2 * i) / (2 * np.sqrt(D * t)))
            C += term1 - term2
        return C


    plt.figure(figsize=(6, 5))

    for t in times:
        C = analytical_solution(y, t, D)
        plt.plot(y, C, label=f"t={t}")

    plt.xlabel("y")
    plt.ylabel("C")
    plt.legend(title="Time")
    plt.title("Analytical Solution of 2D Diffusion Equation")

    plt.show()

def plot_concentration(results: np.ndarray, times: np.ndarray):
    """
    Plot the concentration of a diffusing substance over time

    Params:
    -------
    - results (np.ndarray): The spatial grids at each time step
    - times (np.ndarray): The times at which to plot the concentration
    """
    N = results[0].shape[0] - 1
    y = np.linspace(0, 1, N + 1)

    plt.figure(figsize=(6, 5))

    for t in times:
        frame = int(t / (0.001))
        C = results[frame][N // 2]
        plt.plot(y, C, label=f"t={t}")

    plt.xlabel("y")
    plt.ylabel("C")
    plt.legend(title="Time")
    plt.title("Concentration of Diffusing Substance Over Time")

    plt.show()


def main():
    N = 50
    D = 1
    dx = 1 / N
    dt = 0.001
    steps = 1_000_000
    result = calc_diffusion(N, steps, D, dx, dt)
    # animate_diffusion(grid)
    # calculate_distance(grid)
    # plot_grid(result)
    plot_concentration(result, [0.001, 0.01, 0.1, 1])


if __name__ == "__main__":
    main()
