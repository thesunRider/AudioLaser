import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy.special import jv, jn_zeros
from tqdm.auto import tqdm
from functools import lru_cache
import pandas as pd
import scipy.optimize
import scipy as sc
from functools import partial
from sklearn.neighbors import KDTree
from mpl_toolkits.mplot3d import axes3d
from sklearn.decomposition import PCA


RADIUS = 1
SPEED_OF_SOUND = 0.75
BESSEL_ROOTS = [jn_zeros(m, 10) for m in range(10)]
FPS = 25
TIME_PER_MODE = 10

MODES = (
    (0, 1),
    (1, 1)
)
all_cords = np.array([])
tree = None
data = None
FRAMES = len(MODES) * TIME_PER_MODE * FPS

plotted_points = [None] * 5
#np.seterr(divide='ignore', invalid='ignore')

p0 = np.array([1, 1, 1]) # starting point for the line
direction = np.array( [1,1,1]) # direction vector

def line_func(t):
    """Function of the straight line.
    :param t:     curve-parameter of the line

    :returns      xyz-value as array"""
    return p0 + t*direction

def opt_line(t, time, m, n, RADIUS, SPEED_OF_SOUND):
    """Function that will be minimized by fmin
    :param t:      curve parameter of the straight line

    :returns:      (z_line(t) - z_surface(t))**2 – this is zero
                   at intersection points"""
    p_line = line_func(t)
    z_surface = circular_membrane(*p_line[:2], time, m, n, RADIUS, SPEED_OF_SOUND) 
    return np.sum((p_line[2] - z_surface)**2)



def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
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

@lru_cache()
def lambda_mn(m, n, radius):
    return BESSEL_ROOTS[m][n - 1] / radius


@lru_cache()
def get_vmin_vmax(m, n):
    vmax = np.max(jv(m, np.linspace(0, BESSEL_ROOTS[m][n], 100)))
    return -vmax, vmax

def calc_cross(p1, p2, p3):
    global tree
    v1 = p2 - p1
    v2 = p3 - p1
    v3 =  np.cross(v1, v2)
    mod_v3 = np.linalg.norm(v3)
    if mod_v3 == 0:
        print("zero cross:",p1,p2,p3,"antiparallel vectors")
        mod_v3 = 1

    return v3 / mod_v3

def PCA_unit_vector(array, pca=PCA(n_components=3)):
    pca.fit(array)
    eigenvalues = pca.explained_variance_
    return pca.components_[ np.argmin(eigenvalues) ]

def circular_membrane(r, theta, t, m, n, radius, speed_of_sound):
    l = lambda_mn(m, n, radius)

    T = np.sin(speed_of_sound * l * t)
    R = jv(m, l * r)
    Theta = np.cos(m * theta )

    return R * T * Theta

def calc_angle_with_xy(vectors):
    l = np.sum(vectors[:,:2]**2, axis=1) ** 0.5
    return np.arctan2(vectors[:, 2], l)


r = np.linspace(0, RADIUS, 100)
theta = np.linspace(0, 2 * np.pi, 100)

m, n = MODES[0]
r, theta = np.meshgrid(r, theta)
x = np.cos(theta) * r
y = np.sin(theta) * r
z = circular_membrane(r, theta, 0, m, n, RADIUS, SPEED_OF_SOUND)
vmin, vmax = get_vmin_vmax(m, n)

fig = plt.figure( dpi=100)
ax = fig.add_axes([0, 0, 1, 1], projection='3d')
ax.set_xlabel('$X$')
ax.set_ylabel('$Y$')
ax.set_zlabel('$Z$')

#(x1,x2),(y1,y2),(z1,z2)
laser_axis = ((0,1),(0,1),(0,1))
laser_out = ax.plot3D(laser_axis[0],laser_axis[1],laser_axis[2] ,color='k')

plot = ax.plot_surface(
    x,
    y,
    z,
    linewidth=0,
    cmap='Spectral',
    vmin=vmin,
    vmax=vmax,
    rcount=10,
    ccount=10,
    zorder=1,
    alpha=1
)

omega = SPEED_OF_SOUND * lambda_mn(m, n, RADIUS)
text = ax.text2D(
    0.5, 0.95,
    f'Circular membrane, m = {m}, n = {n}, ω={omega:.2f}',
    size=12, weight='bold',
    va='top', ha='center',
    transform=ax.transAxes,
)


def init():
    pass


def update(i, bar=None):
    global plot
    global ax
    global laser_out
    global plotted_points,tree

    if plotted_points[0] is not None:
        plotted_points[0][0].remove()
        plotted_points[2][0].remove()

    if bar is not None:
        bar.update()

    if plotted_points[1] is not None:
        plotted_points[1].remove()


    t = i / FPS
    m, n = MODES[int(t // TIME_PER_MODE)]

    z = circular_membrane(r, theta, t, m, n, RADIUS, SPEED_OF_SOUND)


    #find normals of surface
    X = x.flatten()
    Y = y.flatten()
    Z = z.flatten()

    #decrease points for processing
    iterations_remove_procs = 3
    for i in range(0,iterations_remove_procs):
        X = np.delete(X, np.arange(0, X.size, 2)) 
        Y = np.delete(Y, np.arange(0, Y.size, 2))
        Z = np.delete(Z, np.arange(0, Z.size, 2))

    data = np.array([X, Y, Z]).T

    tree = KDTree(data, metric='minkowski') # minkowki is p2 (euclidean)
    
    dist, ind = tree.query(data, k=5) #k=3 points including itself
    combinations = np.array(data[ind])

    # map with functools
    pca = PCA(n_components=3)
    normals = list(map(partial(PCA_unit_vector, pca=pca), combinations))

    #normals = list(map(lambda kx: calc_cross(*kx), combinations))

    kn = np.array(normals)
    kn[calc_angle_with_xy(kn) < 0] *= -1
    u, v, w = kn.T

    #find intersection point
    t_opt = sc.optimize.fmin(opt_line, x0=-1,args=( t, m, n, RADIUS, SPEED_OF_SOUND),disp=False,retall=False,full_output=False)
    intersection_point = line_func(t_opt)


    #find nearest normal
    min_val = 10000000000
    min_i = 0
    for i in range(0,len(X)):
        mod_val = np.linalg.norm(intersection_point - [X[i],Y[i],Z[i]])
        if (mod_val < min_val):
            min_val = mod_val
            min_i = i


    normal_direc = [u[min_i],v[min_i],w[min_i]]
    normal_intersect = [X[min_i],Y[min_i],Z[min_i]]

    #decrease points for plotting
    iterations_remove_plot = 2
    for i in range(0,iterations_remove_plot):
        X = np.delete(X, np.arange(0, X.size, 2)) 
        Y = np.delete(Y, np.arange(0, Y.size, 2))
        Z = np.delete(Z, np.arange(0, Z.size, 2))
        u = np.delete(u, np.arange(0, u.size, 2)) 
        v = np.delete(v, np.arange(0, v.size, 2))
        w = np.delete(w, np.arange(0, w.size, 2))


    print("line  at:",intersection_point)
    print("Normal at:",normal_intersect)
    print("Normal cords:",normal_direc)

    
    vmin, vmax = get_vmin_vmax(m, n)
    plot.remove()

    #plot all points
    plotted_points[0] = ax.plot([intersection_point[0]],[intersection_point[1]],[intersection_point[2]], markerfacecolor='k', markeredgecolor='k', marker='o', markersize=5, alpha=0.9)
    plotted_points[1] = ax.quiver(X, Y, Z, u, v, w, length=0.1, normalize=True,zorder=105)
    plotted_points[2] = ax.plot([normal_intersect[0]],[normal_intersect[1]],[normal_intersect[2]], markerfacecolor='r', markeredgecolor='r', marker='o', markersize=5, alpha=0.9,zorder=103)

    plot = ax.plot_surface(
        x,
        y,
        z,
        linewidth=10,
        cmap='Spectral',
        vmin=vmin,
        vmax=vmax,
        rcount=10,
        ccount=10,
        zorder=1,
        alpha=0.9
    )


    set_axes_equal(ax)

    omega = SPEED_OF_SOUND * lambda_mn(m, n, RADIUS)
    text.set_text(f'Circular membrane, m = {m}, n = {n}, ω={omega:.2f}')


bar = tqdm(total=FRAMES)
ani = FuncAnimation(fig, update, init_func=init, frames=FRAMES, interval=1000/FPS, repeat=False, fargs=(bar, ))
set_axes_equal(ax)
plt.show()