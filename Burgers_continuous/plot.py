import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

u_sol = np.loadtxt("tmp.txt")
idx = np.linspace(0, 1, num=u_sol.shape[0])
idy = np.linspace(0, 1, num=u_sol.shape[1])
idx, idy = np.meshgrid(idx, idy, indexing="ij")
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(idx, idy, u_sol)
plt.show()

