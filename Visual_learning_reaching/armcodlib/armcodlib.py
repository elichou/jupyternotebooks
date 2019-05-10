#!/usr/bin/env python

from numpy import *
from matplotlib.pylab import *
from mpl_toolkits.mplot3d import Axes3D

import time
import random
import scipy
import math
import matplotlib.animation as animation

matplotlib.rcParams.update({'font.size': 16})
L1=0.3
L2=0.3

def randrange(n, vmin, vmax):
    return (vmax-vmin)*rand(n) + vmin

def generate_posture(state=randrange(4, 0, 2*pi),L1=0.3,L2=0.3):
	state[3] = 0 # NEW 170111
	x = cumsum([0,L1 * cos(state[0]) * cos(state[2]), L2 * cos(state[0]+state[1]) * cos(state[2]+state[3])])
	y = cumsum([0,L1 * cos(state[0]) * sin(state[2]), L2 * cos(state[0]+state[1]) * sin(state[2]+state[3])])
	z = cumsum([0,L1 * sin(state[0]), L2 * sin(state[0]+state[1])])
	return x,y,z

def generate_random_robot_data(n,L1=0.3,L2=0.3 ):
	phi1 = randrange(n, 0, 2*pi)
	theta1 = randrange(n, 0, 2*pi)
	phi2 = randrange(n, 0, 2*pi)
	theta2 =  zeros(n)
	x = L1 * cos(phi1) * cos(theta1) + L2 * cos(phi1 + phi2) * cos(theta1 + theta2) # ELBOW + HAND
	y = L1 * cos(phi1) * sin(theta1) + L2 * cos(phi1 + phi2) * sin(theta1 + theta2) # ELBOW + HAND
	z = L1 * sin(phi1)               + L2 * sin(phi1 + phi2) # ELBOW + HAND
	vx=L2 * cos(phi1 + phi2) * cos(theta1 + theta2)
	vy=L2 * cos(phi1 + phi2) * sin(theta1 + theta2)
	vz=L2 * sin(phi1 + phi2)
	return (phi1,phi2,theta1,x,y,z,vx,vy,vz)


def control_robot(phi1,phi2,theta1, theta2):
	x = L1 * cos(phi1) * cos(theta1) + L2 * cos(phi1 + phi2) * cos(theta1 + theta2) # ELBOW + HAND
	y = L1 * cos(phi1) * sin(theta1) + L2 * cos(phi1 + phi2) * sin(theta1 + theta2) # ELBOW + HAND
	z = L1 * sin(phi1)               + L2 * sin(phi1 + phi2) # ELBOW + HAND
	vx=L2 * cos(phi1 + phi2) * cos(theta1 + theta2)
	vy=L2 * cos(phi1 + phi2) * sin(theta1 + theta2)
	vz=L2 * sin(phi1 + phi2)
	return (x,y,z,vx,vy,vz)

def plot_arm(phi1, phi2, theta1, time):
    fig = figure()
    ax = fig.gca(projection='3d')
    x = [0, 0, L1 * cos(phi1) * cos(theta1), L2 * cos(phi1 + phi2) * cos(theta1)]
    y = [0, 0, L1 * cos(phi1) * sin(theta1), L2 * cos(phi1 + phi2) * sin(theta1)] # ELBOW + HAND
    z = [0, 0, L1 * sin(phi1) , L2 * sin(phi1 + phi2)] # ELBOW + HAND
    # ax.plot(x[0:1], y[0:1], z[0:1], label='shoulder', lw=2, color= 'k')
    # ax.plot(x[2:3], y[2:3], z[2:3], label='elbow', lw=2, color= 'c')
    # Hide grid lines
    ax.grid(False)
    # ax.set_autoscale_on(True)

    ax.set_xlim(left=-0.2, right=0.2)
    ax.set_ylim(bottom=-0.2, top=0.2)
    ax.set_zlim(bottom=-0.2, top=0.2)
    ax.axis('off')
    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.plot(x, y, z, label='shoulder', lw=2, color= 'k')
    filename = 'fig_%s.png' %time
    savefig(filename)
    return ax
