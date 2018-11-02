#coding:utf-8

import numpy as np
import math
import sys
import os

from raycast import *

# world param
map_size = 200
xyreso = 0.25
yawreso = math.radians(10) # [rad]
min_range = 0.30 # [m]
max_range = 25.0 # [m]
lidar_num = int(round(math.radians(360)/yawreso)+1)

grid_map = np.zeros((map_size, map_size), dtype=np.int32)
for i in range(0, map_size):
    grid_map[i][0] = 1
    grid_map[i][map_size-1] = 1
    grid_map[0][i] = 1
    grid_map[map_size-1][i] = 1

for i in range(0, map_size/2):
    grid_map[i][map_size/2] = 1

state = np.array([20, 10, math.radians(90)])

Raycast = raycast(state, grid_map, map_size,
                  xyreso, yawreso,
                  min_range, max_range, math.radians(360))
lidar = Raycast.raycasting()


