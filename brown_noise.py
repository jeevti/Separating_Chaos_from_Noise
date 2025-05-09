import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# Parameters
fs = 1000  # Sampling frequency (Hz)
n = 10000000  # Number of samples

# Generate white noise
white_noise = np.random.normal(0, 1, n)

# Generate Brownian noise by cumulative summation (integrating white noise)
brown_noise = np.cumsum(white_noise)
brown_noise -= np.mean(brown_noise)  # Remove drift to center around zero

# Compute PSD using Welch’s method
frequencies, psd = welch(brown_noise, fs=fs, nperseg=1024)

# Plot PSD
plt.figure(figsize=(8, 5))
plt.loglog(frequencies, psd, label="Brown Noise PSD", color="brown")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density (dB/Hz)")
plt.title("Power Spectral Density of Brown Noise")
#plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.show()
