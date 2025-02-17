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

    # Run the first time step
    for x in range(1, N_spatial_steps - 1):
        phi[x, 1] = phi[x, 0] + 0.5 * (c * dt / dx) ** 2 * (
            phi[x + 1, 0] - 2 * phi[x, 0] + phi[x - 1, 0]
        )

    # Run the remaining time steps
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
    N_lines: int,
    T: int,
    N_time_steps: int,
    output_file: str | None = None,
):
    """
    Animate the wave equation and export it as a video file or plot every 10th frame.

    Params:
    -------
    - x_vec (np.ndarray): Spatial grid
    - phi (np.ndarray): Wave field
    - dt (float): Time step
    - N_LINES (int): Number of lines to plot
    - T (int): Length of the temporal domain
    - N_TIME_STEPS (int): Number of time steps
    - output_file (str | None): Name of the output video file. If None, plot every 10th frame.

    Returns:
    --------
    - If output_file is None, the plot is displayed.
    - If output_file is not None, the animation is saved to the output_file.
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

    if output_file is not None:
        ani = animation.FuncAnimation(
            fig, _update, frames=N_time_steps, interval=50, blit=True
        )
        ani.save(output_file, writer="ffmpeg", fps=30)
        print(f"Animation saved to {output_file}")
        plt.close()
    else:
        n_steps = N_time_steps // N_lines
        for t in range(0, N_time_steps, n_steps):
            plt.plot(x_vec, phi[:, t])

        plt.xlabel("x")
        plt.ylabel(r"$\Psi(x, t)$")
        plt.title(f"1D Wave Equation at t = {T:.2f}")
        plt.legend(
            [f"t = {t * dt:.2f}" for t in range(0, N_time_steps, n_steps)],
            loc="center left",
            bbox_to_anchor=(1, 0.5),
        )
        plt.ylim(-1.5, 1.5)
        plt.tight_layout()
        plt.show()


def solve_wave_equation(
    L: float,
    T: float,
    N_INTERVALS: int,
    N_TIME_STEPS: int,
    c: float,
    N_LINES: int,
    init_func: callable,
    output_file: str | None = None,
):
    """
    Solve the 1D wave equation and animate the results.

    Params:
    -------
    - L (float): Length of the spatial domain
    - T (float): Length of the temporal domain
    - N_INTERVALS (int): Number of intervals to break the spatial domain into
    - N_TIME_STEPS (int): Number of time steps
    - c (float): Wave speed
    - N_LINES (int): Number of lines to plot
    - init_func (callable): Function to initialize the wave field
    - output_file (str): Name of the output video file. If None, plot every 10th frame.
    """
    dx, dt = L / N_INTERVALS, T / N_TIME_STEPS

    N_SPATIAL_STEPS = N_INTERVALS + 1

    x_vec = initialize_spatial_grid(L, N_SPATIAL_STEPS)
    phi = initialize_wave_field(N_SPATIAL_STEPS, N_TIME_STEPS, init_func, x_vec)
    phi = update_wave_field(phi, c, N_TIME_STEPS, N_SPATIAL_STEPS, dt, dx)

    animate_wave(x_vec, phi, dt, N_LINES, T, N_TIME_STEPS, output_file)
