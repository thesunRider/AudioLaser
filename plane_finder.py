from functools import partial
import numpy as np
from sklearn.neighbors import KDTree

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Grab some test data.
X, Y, Z = axes3d.get_test_data(0.25)

X, Y, Z = map(lambda x: x.flatten(), [X, Y, Z])

plt.plot(X, Y, Z, '.')
plt.show(block=False)

print(np.shape(X),np.shape(Y),np.shape(Z))

data = np.array([X, Y, Z]).T

tree = KDTree(data, metric='minkowski') # minkowki is p2 (euclidean)


# Get indices and distances:
dist, ind = tree.query(data, k=3) #k=3 points including itself

def calc_cross(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1
    v3 =  np.cross(v1, v2)
    return v3 / np.linalg.norm(v3)

def PCA_unit_vector(array, pca=PCA(n_components=3)):
    pca.fit(array)
    eigenvalues = pca.explained_variance_
    return pca.components_[ np.argmin(eigenvalues) ]

combinations = data[ind]

normals = list(map(lambda x: calc_cross(*x), combinations))

# lazy with map
normals2 = list(map(PCA_unit_vector, combinations))


## NEW ##

def calc_angle_with_xy(vectors):
    '''
    Assuming unit vectors!
    '''
    l = np.sum(vectors[:,:2]**2, axis=1) ** 0.5
    return np.arctan2(vectors[:, 2], l)

    

dist, ind = tree.query(data, k=5) #k=3 points including itself
combinations = data[ind]
# map with functools
pca = PCA(n_components=3)
normals3 = list(map(partial(PCA_unit_vector, pca=pca), combinations))

print( combinations[10] )
print(normals3[10])


n = np.array(normals3)
n[calc_angle_with_xy(n) < 0] *= -1

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    
    FROM: https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


u, v, w = n.T

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# ax.set_aspect('equal')

# Make the grid
ax.quiver(X, Y, Z, u, v, w, length=10, normalize=True)
set_axes_equal(ax)
plt.show()