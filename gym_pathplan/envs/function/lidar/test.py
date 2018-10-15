#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import math
from raycast import *

print("test")

# set pose
pose = np.array([10.0, 10.0, math.radians(90)]) # x, y, yaw

# set resulution
xyreso = 0.50 # x-y grid resolution
yawreso = math.radians(10) # yaw angle resolution [rad]

# set lidar param
min_range = 0.05 # [m]
max_range = 25.0 # [m]
angle_limit = math.radians(270) # [rad]

# grid_map
grid_size = 100

# grid map
grid_map = np.zeros((grid_size, grid_size))
# add obstacle
ox = np.random.rand(20) * grid_size * 0.75
oy = np.random.rand(20) * grid_size * 0.75

for ix, iy in zip(ox, oy):
    print("ix:%d iy:%d" % (ix, iy))

for ix, iy in zip(ox, oy):
    grid_map[int(ix)][int(iy)] = 1.0

Raycast = raycast(pose, grid_map, grid_size, xyreso, yawreso, min_range, max_range, angle_limit)

Raycast.raycasting()
