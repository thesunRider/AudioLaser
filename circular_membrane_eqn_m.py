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
TIME_PER_MODE = 20

MODES = (
    (0, 1),
    (1, 1)
)
FRAMES = len(MODES) * TIME_PER_MODE * FPS


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
    t = frame / FPS
    m, n = MODES[int(t // TIME_PER_MODE)]

    #z = circular_membrane(r, theta, t, m, n, RADIUS, SPEED_OF_SOUND)    
    #plot x,y,z here

    t_opt = sc.optimize.fmin(target_func, x0=-10,args=( t, m, n, RADIUS, SPEED_OF_SOUND))
    intersection_point = line_func(t_opt)

    print(intersection_point)

    omega = SPEED_OF_SOUND * lambda_mn(m, n, RADIUS)
    #Circular membrane, m = {m}, n = {n}, ω={omega:.2f}


for i in range(0,FRAMES):
    calculate_vibration(i)