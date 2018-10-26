#coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import math

def atan_zero_to_twopi(y, x):
    angle = math.atan2(y, x)
    if angle < 0.0:
        angle += math.pi * 2.0
    return angle

def calc_angleid(angle, yaw_reso):
    if angle < -yaw_reso/2:
        angle += math.pi * 2.0
    trans_angle = angle + yaw_reso/2
    angleid = math.floor(trans_angle/yaw_reso)
    return angleid

class lidarinfo:
    def __init__(self):
        self.px = 0.0
        self.py = 0.0
        self.d = 0.0
        self.angle = 0.0
        self.angleid = 0
        self.init = True
    def __str__(self):
        return str(self.px) + "," + str(self.py) + "," + str(self.d) + "," + str(self.angle) + "," + str(self.angleid) + "," + str(self.init)

class raycast(object):
    def __init__(self, pose, grid_map, grid_size, xyreso, yawreso, 
                 min_range, max_range, angle_limit):
        self.pose      = pose
        self.grid_map  = grid_map
        self.grid_size = grid_size
        self.xyreso    = xyreso
        self.yawreso   = yawreso
        self.min_range = min_range
        self.max_range = max_range
        self.angle_limit = angle_limit

    def calc_obstacle_position(self):
        obstacle_grid = np.where(0<self.grid_map)
        obstacle_position = np.zeros((len(obstacle_grid[0]), 2))
        for i, (ix, iy) in enumerate(zip(obstacle_grid[0], obstacle_grid[1])):
            x = ix*self.xyreso
            y = iy*self.xyreso
            obstacle_position[i][0] = x
            obstacle_position[i][1] = y
        return obstacle_position

    def transform(self, x, y, obstacle_position):
        transform_position = np.zeros((obstacle_position.shape))
        for i in range(obstacle_position.shape[0]):
            transform_position[i][0] = obstacle_position[i][0]-x
            transform_position[i][1] = obstacle_position[i][1]-y
        return transform_position

    def rotation(self, radian, obstacle_position):
        rotation = np.array([[math.cos(radian),   math.sin(radian)],
                             [-math.sin(radian),  math.cos(radian)]]);
        rotation_position = np.zeros(obstacle_position.shape)
        for i in range(obstacle_position.shape[0]):
            rotation_position[i] = np.dot(rotation, obstacle_position[i])
        return rotation_position

    def raycasting(self):
        obstacle_position  = self.calc_obstacle_position()
        transform_position = self.transform(self.pose[0], self.pose[1], obstacle_position)
        rotation_position = self.rotation(self.pose[2], transform_position)

        lidar_num = int(round((math.pi * 2.0) / self.yawreso) + 1)

        lidar_array = [[] for i in range(lidar_num)]
        for i in range(lidar_num):
            lidar = lidarinfo()
            lidar.angle = i*self.yawreso
            lidar.angleid = i
            lidar_array[i] = lidar

        for pose in rotation_position:
            x = pose[0]
            y = pose[1]
            d = np.sqrt(x**2+y**2)
            angle = math.atan2(y, x)

            if(self.max_range<d):
                d = self.max_range

            angleid = calc_angleid(angle, self.yawreso)

            if lidar_array[int(angleid)].init:
                lidar_array[int(angleid)].px = x
                lidar_array[int(angleid)].py = y
                lidar_array[int(angleid)].d  = d
                lidar_array[int(angleid)].init = False
            
            elif d < lidar_array[int(angleid)].d:
                lidar_array[int(angleid)].px = x
                lidar_array[int(angleid)].py = y
                lidar_array[int(angleid)].d  = d

        raycast_map = []

        for lidar in lidar_array:
            angle = lidar.angle
            if not lidar.init:
                x = lidar.d*math.cos(angle)
                y = lidar.d*math.sin(angle)
                raycast_map.append([x, y, angle])
        raycast_array = np.array(raycast_map)
        
        return raycast_array
        # plt.cla()
        # plt.plot(rotation_position.T[0], rotation_position.T[1], "ob")
        # plt.plot(0, 0, "or")
        # for x, y in zip(raycast_array.T[0], raycast_array.T[1]):
        #    plt.plot([0.0, x], [0.0, y], 'c-')
        # plt.show()
