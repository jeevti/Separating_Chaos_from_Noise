# import numpy as np
# import matplotlib.pyplot as plt

# # Parameters
# x_min, x_max = -5, 5  # Spatial domain
# nx = 100              # Number of spatial points
# dx = (x_max - x_min) / (nx - 1)
# dt = 0.01             # Time step
# nt = 500              # Number of time steps
# D = 0.1               # Diffusion coefficient
# A = lambda x: -x      # Drift term (e.g., harmonic potential: -x)

# # Grid and initial condition
# x = np.linspace(x_min, x_max, nx)
# P = np.exp(-x**2)     # Initial probability distribution (Gaussian)
# P /= np.sum(P) * dx   # Normalize

# # Discretized Fokker-Planck Equation
# for _ in range(nt):
#     # Compute derivatives
#     dPdx = np.gradient(P, dx)         # First derivative
#     d2Pdx2 = np.gradient(dPdx, dx)   # Second derivative

#     # Update P using the Fokker-Planck equation
#     dPdt = -np.gradient(A(x) * P, dx) + D * d2Pdx2
#     P += dt * dPdt

#     # Boundary conditions (e.g., zero flux at boundaries)
#     P[0] = P[-1] = 0

#     # Normalize to conserve probability
#     P /= np.sum(P) * dx

# # Plot results
# plt.plot(x, P, label="Final")
# plt.xlabel("x")
# plt.ylabel("P(x, t)")
# plt.title("Fokker-Planck Equation Solution")
# plt.legend()
# plt.show()
import numpy as np
import matplotlib.pyplot as plt

# Parameters
x_min, x_max = -5, 5  # Spatial domain
nx = 100              # Number of spatial points
dx = (x_max - x_min) / (nx - 1)
dt = 0.01             # Time step
nt = 500              # Number of time steps
D = 0.1               # Diffusion coefficient
A = lambda x: -x      # Drift term (e.g., harmonic potential: -x)

# Grid for Fokker-Planck solution
x = np.linspace(x_min, x_max, nx)
P = np.exp(-x**2)     # Initial probability distribution (Gaussian)
P /= np.sum(P) * dx   # Normalize

# Langevin simulation parameters
N_particles = 10000                     # Number of particles
particle_positions = np.random.normal(0, 1, N_particles)  # Initial positions

# To store particle density
histograms = []

# Simulate both Langevin and Fokker-Planck
for t in range(nt):
    # Fokker-Planck Equation (Finite Differences)
    dPdx = np.gradient(P, dx)         # First derivative
    d2Pdx2 = np.gradient(dPdx, dx)   # Second derivative
    dPdt = -np.gradient(A(x) * P, dx) + D * d2Pdx2
    P += dt * dPdt
    P[0] = P[-1] = 0  # Zero-flux boundary conditions
    P /= np.sum(P) * dx  # Normalize

    # Langevin Equation (Euler-Maruyama)
    noise = np.sqrt(2 * D * dt) * np.random.randn(N_particles)
    particle_positions += A(particle_positions) * dt + noise

    # Record histogram of particle positions
    hist, _ = np.histogram(particle_positions, bins=nx, range=(x_min, x_max), density=True)
    histograms.append(hist)

# Plot Final Results
plt.plot(x, P, label="Fokker-Planck (P(x, t))", lw=2)
plt.plot(x, histograms[-1], label="Langevin (Particle Histogram)", linestyle="--")
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.title("Comparison: Fokker-Planck vs. Langevin Simulation")
plt.legend()
plt.show()
