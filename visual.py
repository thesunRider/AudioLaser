import numpy as np
from mpl_toolkits.mplot3d import Axes3D  
# Axes3D import has side effects, it enables using projection='3d' in add_subplot
import matplotlib.pyplot as plt
import random
from snnpy import *

RADIUS = 1
SIZE_STEP = 100

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


r = np.linspace(0, RADIUS, SIZE_STEP)
theta = np.linspace(0, 2 * np.pi, SIZE_STEP)

r, theta = np.meshgrid(r, theta)
x = np.cos(theta) * r
y = np.sin(theta) * r
z = np.zeros_like(x)

xyz = np.zeros((np.size(x), 3))
xyz[:, 0] = np.reshape(x, -1)
xyz[:, 1] = np.reshape(y, -1)
xyz[:, 2] = np.reshape(z, -1)


def get_nearbypoints(indx,radius):
    snn_model = build_snn_model(xyz)  
    ind = snn_model.query_radius(xyz[indx], radius)
    return xyz[ind]

val = get_nearbypoints(15,0.05)
ax.scatter(val[:,0], val[:,1], val[:,2], c= 'red',zorder=13)
#ax.plot_surface(x, y, z,zorder=1)


ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()