#%%
from sympy import sympify, symbols
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.stats import gaussian_kde , norm

sigma  = float(10)
rho = float(28)
beta = sympify(8/3)
dt = float(0.005) 
xco = float(1)
yco = float(1)
zco = float(1)
noise_std = 1  # Standard deviation of noise

xval = float(xco)
yval = float(yco)
zval = float(zco)
coord = [(xval,yval,zval)]

for i in range(5000):
    dx = sigma*(yval - xval)
    dy = xval*(rho - zval) - yval
    dz = xval*yval - beta*zval

    xval = xval + dt*dx #+ noise_std * np.sqrt(dt) * np.random.normal()
    yval = yval + dt*dy + noise_std * np.sqrt(dt) * np.random.normal()
    zval = zval + dt*dz + noise_std * np.sqrt(dt) * np.random.normal()
    coord.append((xval,yval,zval))

xlist = np.array([float(xvals) for xvals, yvals, zvals in coord])
ylist = np.array([float(yvals) for xvals, yvals, zvals in coord])
zlist = np.array([float(zvals) for xvals, yvals, zvals in coord])
#print(xlist)
# plt.plot(xlist,ylist)
# plt.grid(True)
# plt.show() 

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(xlist, ylist, zlist)
#plt.show()

#PDF
mean_x = np.mean(xlist)
std_x = np.sqrt(np.var(xlist))
x_range = np.linspace(np.min(xlist), np.max(xlist), 1000)
pdf = norm.pdf(x_range, mean_x, std_x)

#KDE PDF
kde = gaussian_kde(xlist)
x_range = np.linspace(np.min(xlist), np.max(xlist), 1000)  # Range for plotting
xpdf_values = kde(x_range)  # KDE values


# Finalize plot
plt.figure()
plt.plot(x_range, pdf, color='red', lw=2, label="PDF")
plt.plot(x_range, xpdf_values, color='blue', lw=2, label=" KDE PDF")
plt.xlabel("Lorenz x vals")
plt.ylabel("Probability Density")
plt.title("x vals PDF")
plt.legend()
plt.grid()
plt.show()


#PDF
mean_y = np.mean(ylist)
std_y = np.sqrt(np.var(ylist))
y_range = np.linspace(np.min(ylist), np.max(ylist), 1000)
pdf = norm.pdf(y_range, mean_y, std_y)

#KDE PDF
kde = gaussian_kde(ylist)
y_range = np.linspace(np.min(ylist), np.max(ylist), 1000)  # Range for plotting
ypdf_values = kde(y_range)  # KDE values

# Finalize plot
plt.figure()
plt.plot(y_range, pdf, color='red', lw=2, label="PDF")
plt.plot(y_range, ypdf_values, color='blue', lw=2, label="KDE PDF")
plt.xlabel("Lorenz y vals")
plt.ylabel("Probability Density")
plt.title("y vals PDF")
plt.legend()
plt.grid()
plt.show()

#PDF
mean_z = np.mean(zlist)
std_z = np.sqrt(np.var(zlist))
z_range = np.linspace(np.min(zlist), np.max(zlist), 1000)
pdf = norm.pdf(z_range, mean_z, std_z)

#KDE PDF
kde = gaussian_kde(zlist)
z_range = np.linspace(np.min(zlist), np.max(zlist), 1000)  # Range for plotting
zpdf_values = kde(z_range)  # KDE values

# Finalize plot
plt.figure()
plt.plot(z_range, pdf, color='red', lw=2, label="PDF")
plt.plot(z_range, zpdf_values, color='blue', lw=2, label="KDE PDF")
plt.xlabel("Lorenz z vals")
plt.ylabel("Probability Density")
plt.title("z vals PDF")
plt.legend()
plt.grid()
plt.show()

# %%
