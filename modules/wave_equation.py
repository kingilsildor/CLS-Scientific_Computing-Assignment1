import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from numba import jit


def initialize_spatial_grid(L: float, N_spatial_steps: int) -> np.ndarray:
    """
    Initialize the spatial grid

    Params:
    -------
    - L (float): Length of the spatial domain
    - N_spatial_steps (int): Number of spatial steps

    Returns:
    --------
    - x_vec (np.ndarray): Spatial grid
    """
    grid = np.linspace(0, L, N_spatial_steps)
    assert grid.shape == (N_spatial_steps,)
    return grid


def initialize_wave_field(
    N_spatial_steps: int, N_time_steps: int, init_func: callable, x_vec: np.ndarray
) -> np.ndarray:
    """
    Initialize the wave field

    Params:
    -------
    - N_spatial_steps (int): Number of spatial steps
    - N_time_steps (int): Number of time steps
    - init_func (callable): Function to initialize the wave field
    - x_vec (np.ndarray): Spatial grid

    Returns:
    --------
    - phi (np.ndarray): Wave field
    """
    assert callable(init_func)
    assert x_vec.shape == (N_spatial_steps,)

    phi = np.zeros((N_spatial_steps, N_time_steps))
    phi[:, 0] = init_func(x_vec)
    phi[0, :], phi[-1, :] = 0.0, 0.0
    phi[:, 1] = phi[:, 0]

    assert phi.shape == (N_spatial_steps, N_time_steps)
    assert np.all(np.isfinite(phi))
    return phi


@jit
def update_wave_field(
    phi: np.ndarray,
    c: float,
    N_time_steps: int,
    N_spatial_steps: int,
    dt: float,
    dx: float,
) -> np.ndarray:
    """
    Update the wave field using the wave equation

    Params:
    -------
    - phi (np.ndarray): Wave field
    - c (float): Wave speed
    - N_time_steps (int): Number of time steps
    - N_spatial_steps (int): Number of spatial steps
    - dt (float): Time step
    - dx (float): Spatial step

    Returns:
    --------
    - phi (np.ndarray): Updated wave field
    """
    assert phi.shape == (N_spatial_steps, N_time_steps)

    for t in range(1, N_time_steps - 1):
        for x in range(1, N_spatial_steps - 1):
            phi[x, t + 1] = (
                2 * phi[x, t]
                - phi[x, t - 1]
                + (c * (dt / dx)) ** 2 * (phi[x + 1, t] - 2 * phi[x, t] + phi[x - 1, t])
            )

    assert phi.shape == (N_spatial_steps, N_time_steps)
    assert np.all(np.isfinite(phi))
    return phi


def animate_wave(
    x_vec: np.ndarray,
    phi: np.ndarray,
    dt: float,
    N_time_steps: int,
    save_location: str,
):
    """
    Animate the wave equation and export it as a video file

    Params:
    -------
    - x_vec (np.ndarray): Spatial grid
    - phi (np.ndarray): Wave field
    - dt (float): Time step
    - N_time_steps (int): Number of time steps
    - save_location (str): Location to save the video file

    Returns:
    --------
    - Video file with the wave equation animation
    """
    assert x_vec.shape == (phi.shape[0],)
    assert phi.shape[1] == N_time_steps

    fig, ax = plt.subplots()
    (line,) = ax.plot(x_vec, phi[:, 0], "black")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\Psi(x, t)$")
    ax.set_ylim(-1.5, 1.5)

    def _update(frame: int) -> plt.Line2D:
        """
        Helper function to update the plot for each frame.

        Params:
        -------
        - frame (int): Frame number.

        Returns:
        --------
        - line (matplotlib.lines.Line2D): Updated line object
        """
        line.set_ydata(phi[:, frame])
        ax.set_title(f"1D Wave Equation Animation (t = {frame * dt:.2f})")
        return (line,)

    ani = animation.FuncAnimation(
        fig, _update, frames=N_time_steps, interval=50, blit=True
    )
    
    assert save_location.endswith(".mp4")
    ani.save(save_location, writer="ffmpeg", fps=60)
    plt.close()


def plot_wave(
    x_vec: np.ndarray,
    phi: np.ndarray,
    N_lines: int,
    N_time_steps: int,
    save_location: str,
):
    """
    Animate the wave equation and export it as a video file

    Params:
    -------
    - x_vec (np.ndarray): Spatial grid
    - phi (np.ndarray): Wave field
    - N_lines (int): Number of lines to plot
    - N_time_steps (int): Number of time steps
    - save_location (str): Location to save the plot

    Returns:
    --------
    - Plot of the wave equation
    """
    t_indices = np.linspace(0, N_time_steps - 1, N_lines, dtype=int)
    assert len(t_indices) == N_lines

    plt.figure(dpi=300)
    for t in t_indices:
        plt.plot(x_vec, phi[:, t], label=f"$t = {(t + 1) / 1000}$")

    plt.xlabel("x", fontsize=14)
    plt.ylabel(r"$\Psi(x, t)$", fontsize=14)
    plt.title("Numerical solution for the 1D Wave Equation")
    plt.legend(
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        frameon=False,
    )
    plt.tight_layout()

    assert save_location.endswith(".png")
    plt.savefig(save_location)
    plt.show()


def solve_wave_equation(
    L: float,
    T: float,
    N_spatial_steps: int,
    N_time_steps: int,
    c: float,
    init_func: callable,
    N_lines: int = 4,
    animate: bool = False,
    save_location: str = "results/wave_equation",
):
    """
    Solve the 1D wave equation and animate the results.

    Params:
    -------
    - L (float): Length of the spatial domain
    - T (float): Length of the temporal domain
    - N_spatial_steps (int): Number of spatial steps
    - N_time_steps (int): Number of time steps
    - c (float): Wave speed
    - init_func (callable): Function to initialize the wave field
    - N_lines (int): Number of lines to plot. Default is 4
    - animate (bool): Whether to animate the wave equation. Default is False
    - save_location (str): Location to save the results. Default is "results/wave_equation". File extension is added automatically
    """
    dx, dt = L / N_spatial_steps, T / N_time_steps

    x_vec = initialize_spatial_grid(L, N_spatial_steps)
    phi = initialize_wave_field(N_spatial_steps, N_time_steps, init_func, x_vec)
    phi = update_wave_field(phi, c, N_time_steps, N_spatial_steps, dt, dx)
    
    if animate:
        save_location += ".mp4"
        animate_wave(x_vec, phi, dt, N_time_steps, save_location)
    else:
        save_location += ".png"
        plot_wave(x_vec, phi, N_lines, N_time_steps, save_location)


def main():
    def init_one(x_vec):
        return np.sin(2 * np.pi * x_vec)


    def init_two(x_vec):
        return np.sin(5 * np.pi * x_vec)


    def init_three(x_vec):
        return np.sin(5 * np.pi * x_vec) * ((1 / 5 < x_vec) & (x_vec < 2 / 5))
    
    L, T = 1.0, 0.5
    N_SPATIAL_STEPS, N_TIME_STEPS = 100, 500
    c = 1.0

    solve_wave_equation(L, T, N_SPATIAL_STEPS, N_TIME_STEPS, c, init_two, animate=False)


if "__name__" == main():
    main()
