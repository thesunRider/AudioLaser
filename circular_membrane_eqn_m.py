import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy.special import jv, jn_zeros
from tqdm.auto import tqdm
from functools import lru_cache
import pandas as pd
import scipy.optimize
import scipy as sc
from sklearn.neighbors import KDTree
import pdb

RADIUS = 1
SPEED_OF_SOUND = 0.75
BESSEL_ROOTS = [jn_zeros(m, 10) for m in range(10)]
FPS = 25
TIME_PER_MODE = 20
error_cross = False

MODES = (
    (0, 1),
    (1, 1)
)
FRAMES = len(MODES) * TIME_PER_MODE * FPS

def calc_angle_with_xy(vectors):
    l = np.sum(vectors[:,:2]**2, axis=1) ** 0.5
    return np.arctan2(vectors[:, 2], l)

def calc_cross(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1
    v3 =  np.cross(v1, v2)
    mod_v3 = np.linalg.norm(v3)

    if mod_v3 == 0:
        print("zero cross:",p1,p2,p3,"antiparallel vectors")
        mod_v3 = 1

    return v3 / mod_v3

@lru_cache()
def lambda_mn(m, n, radius):
    return BESSEL_ROOTS[m][n - 1] / radius


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


#x,y,z plot here
omega = SPEED_OF_SOUND * lambda_mn(m, n, RADIUS)
#Circular membrane, m = {m}, n = {n}, ω={omega:.2f}',


def calculate_vibration(frame):
    global error_cross
    t = frame / FPS
    m, n = MODES[int(t // TIME_PER_MODE)]

    z = circular_membrane(r, theta, t, m, n, RADIUS, SPEED_OF_SOUND)    
    #plot x,y,z here


    #find normals of surface
    X = x.flatten()
    Y = y.flatten()
    Z = z.flatten()

    #decrease points for plotting
    iterations_remove = 4
    X = np.delete(X, np.arange(0, X.size, iterations_remove)) 
    Y = np.delete(Y, np.arange(0, Y.size, iterations_remove))
    Z = np.delete(Z, np.arange(0, Z.size, iterations_remove))

    data = np.array([X, Y, Z]).T

    #remove vars that are duplicate ,ie at origin

    
    tree = KDTree(data, metric='minkowski') # minkowki is p2 (euclidean)
    
    dist, ind = tree.query(data, k=3) #k=3 points including itself
    combinations = np.array(data[ind])
    # map with functools

    normals = list(map(lambda kx: calc_cross(*kx), combinations))
    

    kn = np.array(normals)
    kn[calc_angle_with_xy(kn) < 0] *= -1
    u, v, w = kn.T



    t_opt = sc.optimize.fmin(target_func, x0=-10,args=( t, m, n, RADIUS, SPEED_OF_SOUND))
    intersection_point = line_func(t_opt)

    print(intersection_point)

    omega = SPEED_OF_SOUND * lambda_mn(m, n, RADIUS)
    #Circular membrane, m = {m}, n = {n}, ω={omega:.2f}


for i in range(0,FRAMES):
    calculate_vibration(i)