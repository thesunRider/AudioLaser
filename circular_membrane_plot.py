import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy.special import jv, jn_zeros
from tqdm.auto import tqdm
from functools import lru_cache
import pandas as pd
import scipy.optimize
import scipy as sc


RADIUS = 1
SPEED_OF_SOUND = 0.75
BESSEL_ROOTS = [jn_zeros(m, 10) for m in range(10)]
FPS = 25
TIME_PER_MODE = 2

MODES = (
    (0, 1),
    (1, 1)
)
all_cords = np.array([])
FRAMES = len(MODES) * TIME_PER_MODE * FPS

plotted_points = [None] * 5


p0 = np.array([1, 1, 1]) # starting point for the line
direction = np.array( np.sqrt([1/3, 1/3, 1/3])) # direction vector

def line_func(t):
    """Function of the straight line.
    :param t:     curve-parameter of the line

    :returns      xyz-value as array"""
    return p0 + t*direction

def target_func(t, time, m, n, RADIUS, SPEED_OF_SOUND):
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


def circular_membrane(r, theta, t, m, n, radius, speed_of_sound):
    l = lambda_mn(m, n, radius)

    T = np.sin(speed_of_sound * l * t)
    R = jv(m, l * r)
    Theta = np.cos(m * theta )

    return R * T * Theta


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
laser_axis = ((-1,1),(-1,1),(-1,1))
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
    global plotted_points

    if plotted_points[0] is not None:
        plotted_points[0][0].remove()

    if bar is not None:
        bar.update()

    t = i / FPS
    m, n = MODES[int(t // TIME_PER_MODE)]

    z = circular_membrane(r, theta, t, m, n, RADIUS, SPEED_OF_SOUND)

    t_opt = sc.optimize.fmin(target_func, x0=-1,args=( t, m, n, RADIUS, SPEED_OF_SOUND))
    intersection_point = line_func(t_opt)

    print("line intersects at:",intersection_point)
    plotted_points[0] = ax.plot([intersection_point[0]],[intersection_point[1]],[intersection_point[2]], markerfacecolor='k', markeredgecolor='k', marker='o', markersize=5, alpha=0.9)

    
    vmin, vmax = get_vmin_vmax(m, n)
    plot.remove()

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