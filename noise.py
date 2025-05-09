import numpy as np
import matplotlib.pyplot as plt

# Generate x values
x = np.linspace(0, 10, 200)

# Straight line y = x
y_clean = x

# Noisy line: y = x + noise
noise = np.random.normal(0, 0.5, size=x.shape)  # Gaussian noise with mean 0, std 0.5
y_noisy = y_clean + noise

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Plot clean line
axes[0].plot(x, y_clean, label='y = x', color='green')
axes[0].set_title('Straight Line')
axes[0].legend()

# Plot noisy line
axes[1].plot(x, y_noisy, label='y = x + noise', color='green')
axes[1].set_title('Noisy Line')
axes[1].legend()

# Show plot
plt.tight_layout()
plt.show()
