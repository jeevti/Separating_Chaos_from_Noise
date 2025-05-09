from sympy import sympify, symbols
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.stats import gaussian_kde , norm
from numpy.polynomial.polynomial import Polynomial  # For detrending

sigma  = float(10)
rho = float(28)
beta = sympify(8/3)
dt = float(0.001) 
xco = float(1)
yco = float(1)
zco = float(1)
noise_std = 20  # Standard deviation of noise

xval = float(xco)
yval = float(yco)
zval = float(zco)
coord = [(xval,yval,zval)]

for i in range(100000):
    dx = sigma*(yval - xval)
    dy = xval*(rho - zval) - yval
    dz = xval*yval - beta*zval

    xval = xval + dt*dx + noise_std * np.sqrt(dt) * np.random.normal()
    yval = yval + dt*dy + noise_std * np.sqrt(dt) * np.random.normal()
    zval = zval + dt*dz + noise_std * np.sqrt(dt) * np.random.normal()
    coord.append((xval,yval,zval))

xlist = np.array([float(xvals) for xvals, yvals, zvals in coord])
ylist = np.array([float(yvals) for xvals, yvals, zvals in coord])
zlist = np.array([float(zvals) for xvals, yvals, zvals in coord])

# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(xlist, ylist, zlist)

n = len(coord)

min_s = 10           
max_s = n // 4       
step = 10            
scales = np.arange(min_s, max_s, step)  

log_s = np.log(scales)

#DFA for xlist
xprofile = np.cumsum(xlist -np.mean(xlist))
plt.plot(xprofile)

xfluctuations = []  
for s in scales:
    xsegments = n // s  # Number of segments
    xrms_values = []    # root-mean-square fluctuation
    for i in range(xsegments):
        start, end = i * s, (i + 1) * s
        a = np.arange(s)  # Time indices
        b = xprofile[start:end]  # Segment of the profile
        # Fit a polynomial (linear detrending)
        p = Polynomial.fit(a, b, 1)  
        trend = p(a)  
        # Compute RMS fluctuation
        xrms = np.sqrt(np.mean((b - trend) ** 2))
        xrms_values.append(xrms)
    # Compute the mean RMS fluctuation for this segment size
    xfluctuations.append(np.sqrt(np.mean(np.array(xrms_values) ** 2)))

xlog_F = np.log(xfluctuations)

# Fit a line to find the slope (scaling exponent α)
xalpha, xintercept = np.polyfit(log_s, xlog_F, 1)

# Plot the DFA results
plt.figure(figsize=(8, 6))
plt.loglog(scales, xfluctuations, '.', label=f"α = {xalpha:.2f}")
plt.xlabel("Segment Size (s)")
plt.ylabel("Fluctuation Function F(s)")
plt.title("Detrended Fluctuation Analysis (DFA)")
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.show()

print(f"Estimated scaling exponent α for x: {xalpha:.2f}")

#DFA for ylist
yprofile = np.cumsum(ylist -np.mean(ylist))

yfluctuations = []  
for s in scales:
    ysegments = n // s  # Number of segments
    yrms_values = []    # root-mean-square fluctuation
    for i in range(ysegments):
        start, end = i * s, (i + 1) * s
        a = np.arange(s)  # Time indices
        b = yprofile[start:end]  # Segment of the profile
        # Fit a polynomial (linear detrending)
        p = Polynomial.fit(a, b, 1)  
        trend = p(a)  
        # Compute RMS fluctuation
        yrms = np.sqrt(np.mean((b - trend) ** 2))
        yrms_values.append(yrms)
    # Compute the mean RMS fluctuation for this segment size
    yfluctuations.append(np.sqrt(np.mean(np.array(yrms_values) ** 2)))

ylog_F = np.log(yfluctuations)

# Fit a line to find the slope (scaling exponent α)
yalpha, yintercept = np.polyfit(log_s, ylog_F, 1)

# Plot the DFA results
plt.figure(figsize=(8, 6))
plt.loglog(scales, yfluctuations, '.', label=f"α = {yalpha:.2f}")
plt.xlabel("Segment Size (s)")
plt.ylabel("Fluctuation Function F(s)")
plt.title("Detrended Fluctuation Analysis (DFA)")
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.show()

print(f"Estimated scaling exponent α for y: {yalpha:.2f}")

