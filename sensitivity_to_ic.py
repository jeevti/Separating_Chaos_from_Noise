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
# noise_std = (0,10,20)  # Standard deviation of noise

xval = float(1)
yval = float(1)
zval = float(1)
coord = [(xval,yval,zval)]
for i in range(6000):
    dx = sigma*(yval - xval)
    dy = xval*(rho - zval) - yval
    dz = xval*yval - beta*zval
    xval = xval + dt*dx #+ i * np.sqrt(dt) * np.random.normal()
    yval = yval + dt*dy #+ i * np.sqrt(dt) * np.random.normal()
    zval = zval + dt*dz #+ i * np.sqrt(dt) * np.random.normal()
    coord.append((xval,yval,zval))

xlist = np.array([float(xvals) for xvals, yvals, zvals in coord])
ylist = np.array([float(yvals) for xvals, yvals, zvals in coord])
zlist = np.array([float(zvals) for xvals, yvals, zvals in coord])



xval1 = float(1)
yval1 = float(1)
zval1 = float(1.0000001)
coord1 = [(xval1,yval1,zval1)]
for i in range(6000):
    dx = sigma*(yval1 - xval1)
    dy = xval1*(rho - zval1) - yval1
    dz = xval1*yval1 - beta*zval1
    xval1 = xval1 + dt*dx #+ i * np.sqrt(dt) * np.random.normal()
    yval1 = yval1 + dt*dy #+ i * np.sqrt(dt) * np.random.normal()
    zval1 = zval1 + dt*dz #+ i * np.sqrt(dt) * np.random.normal()
    coord1.append((xval1,yval1,zval1))

xlist1 = np.array([float(xvals) for xvals, yvals, zvals in coord1])
ylist1 = np.array([float(yvals) for xvals, yvals, zvals in coord1])
zlist1 = np.array([float(zvals) for xvals, yvals, zvals in coord1])
# plt.figure(figsize=(10, 6))

# # Create the first subplot for xlist
# plt.subplot(211)
# plt.plot(xlist)
# plt.title('Subplot 1')
# plt.xlabel('Index')
# plt.ylabel('Value')

# # Create the second subplot for xlist - xlist1
# plt.subplot(212)
# plt.plot(xlist - xlist1)
# plt.title('Subplot 2')
# plt.xlabel('Index')
# plt.ylabel('Difference')

# # Display the plots
# plt.tight_layout()
# plt.show()
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(xlist, ylist, zlist,linewidth = 1, color = "darkorange", label = "Lorenz at (1,1,1)")
ax.plot3D(xlist1, ylist1, zlist1,linewidth = 1, color = "dodgerblue", label = "Lorenz at (1,1,1.0000001)")
plt.show()



# %%
