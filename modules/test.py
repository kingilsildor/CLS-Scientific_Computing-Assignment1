import numpy as np
import matplotlib.pyplot as plt

# Define parameters
L = 1.0  # Length of the domain
Nx = 100  # Number of spatial points
dx = L / Nx  # Spatial step
c = 1.0  # Wave speed
dt = 0.001  # Time step
Nt = 500  # Number of time steps

# Stability condition (CFL condition)
cfl = c * dt / dx
if cfl > 1:
    raise ValueError("CFL condition violated! Reduce dt or increase dx.")

# Initialize spatial domain
x_vec = np.linspace(0, L, Nx)

# Initialize field variables
phi = np.zeros((Nx, Nt))
phi[:, 0] = np.sin(2 * np.pi * x_vec)  # Initial condition phi(x, t=0)
phi[:, 1] = phi[:, 0]  # Assume initial velocity is zero

# Finite difference time evolution
for t in range(1, Nt - 1):
    for i in range(1, Nx - 1):
        phi[i, t + 1] = (2 * phi[i, t] - phi[i, t - 1] +
                         cfl ** 2 * (phi[i + 1, t] - 2 * phi[i, t] + phi[i - 1, t]))

# Animation
plt.figure()
for t in range(Nt):
    plt.cla()
    plt.plot(x_vec, phi[:, t], 'black')
    plt.xlabel("x")
    plt.ylabel("u(x, t)")
    plt.title(f"1D Wave Equation Animation (t = {t * dt:.2f})")
    plt.ylim(-1.5, 1.5)
    plt.pause(0.001)

plt.show()
