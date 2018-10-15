#coding:utf-8

import numpy as np
import math
from function import calc_angleid, atan_zero_to_twopi

class lidarinfo:
    def __init__(self):
        self.px = 0.0
        self.py = 0.0
        self.d = 0.0
        self.angle = 0.0
        self.angleid = 0
        self.init = False
    def __str__(self):
        return str(self.px) + "," + str(self.py) + "," + str(self.d) + "," + str(self.angle) + "," + str(self.angleid) + "," + str(self.init)


class raycast(self, pose, 
              grid_map, grid_size,
              xyreso, yawreso,
              min_range, max_range, angle_limit):
    def __init__(self):
        self.pose = pose # x, y, yaw
        self.grid_map = grid_map # obstacle_map
        self.grid_size = grid_size # grid_size
        self.xyreso = xyreso # x-y grid resolution [m]
        self.yawreso = yawreso # yaw angle resolution [rad]
        self.min_range = min_range # lidar min range
        self.max_range = max_range # lidar max range
        self.angle_limit = angle_limit

        self.min_x = -grid_size/2
        self.max_x = grid_size/2
        self.min_y = -grid_size/2
        self.max_y = grid_size/2

    def calc_obstacle_id(self):
        obstacle_grid = np.where(0<self.grid_map)
        obstacle_position = np.zeros(len(obstacle_grid[0]), 2)
        for i, (ix, iy) in enumerate(zip(obstacle_grid[0], obstacle_grid[1])):
            x = (ix+min_x)*self.xyreso
            y = (iy+min_y)*self.xyreso
            obstacle_position[i][0] = x
            obstacle_position[i][1] = y
        return obstacle_position

    def transform(self, pose, obstacle_position):
        transform_position = np.zeros((len(obstacle_position[0]), 2))
        for i in range(len(obstacle_position[0])):
            transform_position[i][0] = obstacle_position[i][0]-pose[0]
            transform_position[i][1] = obstacle_position[i][1]-pose[1]
        return transform_position

    def rotation(self, pose, obstacle_position):
        rotation = np.array([[math.cos(-pose[2]), -math.sin(-pose[2])],
                         [math.sin(-pose[2]),  math.cos(-pose[2])]]);
        rotation_position = np.zeros((len(obstacle_position[0]), 2))
        for i in range(len(obstacle_position[0])):
            rotation_obstacle[i] = np.dot(rotation, obstacle_position)
        return rotation_position

    def raycasting(self):
        
        obstacle_position = self.calc_obstacle_id()
        transform_position = self.transform(self.pose, obstacle_position)
        rotation_position = self.rotation(pose, transform_position)

        lidar_num = int(round((math.pi * 2.0) / yawreso) + 1)
        lidar_array = [[] for i in range(lidar_num)]
        for i in range(lidar_num):
            lidar = lidarinfo()
            lidar.angle = i*yawreso
            lidar.angleid = i
            lidar_array[i] = lidar










