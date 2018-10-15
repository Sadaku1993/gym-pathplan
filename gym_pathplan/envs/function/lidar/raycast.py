#coding:utf-8

import numpy as np
import math
import matplotlib.pyplot as plt
from function import calc_angleid, atan_zero_to_twopi

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
    def __init__(self, 
            pose, 
            grid_map, grid_size,
            xyreso, yawreso,
            min_range, max_range, angle_limit):
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

    # grid_mapからobstacleのindexを取得し、world座標系の位置情報に変換する
    def calc_obstacle_position(self):
        obstacle_grid = np.where(0<self.grid_map)
        obstacle_position = np.zeros((len(obstacle_grid[0]), 2))
        for i, (ix, iy) in enumerate(zip(obstacle_grid[0], obstacle_grid[1])):
            x = ix*self.xyreso
            y = iy*self.xyreso
            # print("x:%6.2f y:%6.2f  ix:%d iy:%d" % (x, y, ix, iy))
            obstacle_position[i][0] = x
            obstacle_position[i][1] = y
        return obstacle_position

    def transform(self, pose, obstacle_position):
        transform_position = np.zeros((obstacle_position.shape))
        for i in range(obstacle_position.shape[0]):
            transform_position[i][0] = obstacle_position[i][0]-pose[0]
            transform_position[i][1] = obstacle_position[i][1]-pose[1]
        return transform_position

    def rotation(self, pose, obstacle_position):
        rotation = np.array([[math.cos(-pose[2]), -math.sin(-pose[2])],
                             [math.sin(-pose[2]),  math.cos(-pose[2])]]);
        rotation_position = np.zeros(obstacle_position.shape)
        for i in range(obstacle_position.shape[0]):
            rotation_position[i] = np.dot(rotation, obstacle_position[i])
        return rotation_position

    def raycasting(self):
        # World座標系での障害物位置情報を取得
        obstacle_position = self.calc_obstacle_position()
        print("obstacle_position", obstacle_position)
        
        # LiDAR座標系に変換
        transform_position = self.transform(self.pose, obstacle_position)
        print("transform_position", transform_position)
        rotation_position = self.rotation(self.pose, transform_position)
        print("rotation_position", rotation_position)

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

            # 測定可能距離外の場合、minrangeとmaxrangeに距離情報を変更
            if(d<self.min_range):
                d = self.min_range
            elif(self.max_range<d):
                d = self.max_range
            
            # outof angle
            if(angle<-self.angle_limit/2 or self.angle_limit/2<angle):
                continue

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
            if math.pi < angle:
                angle -= math.pi*2
            if not lidar.init:
                x = lidar.d*math.cos(angle)
                y = lidar.d*math.sin(angle)
                raycast_map.append([x, y])
            elif(-self.angle_limit/2<=angle and angle<=self.angle_limit/2):
                x = self.max_range*math.cos(angle)
                y = self.max_range*math.sin(angle)
                raycast_map.append([x, y])

        raycast_array = np.array(raycast_map)

        for lidar in lidar_array:
            print("x:%6.2f y:%6.2f dis:%6.2f yaw:%6.2f id:%3d shot:%s" % 
                  (lidar.px, lidar.py, lidar.d, lidar.angle, lidar.angleid, lidar.init))
        
        self.show(rotation_position, np.array([0.0, 0.0, 0.0]), raycast_array, "rotation_position")

    def show(self, obstacle, pose, raycast, title):
        plt.title(title)
        plt.xlim(-self.grid_size*self.xyreso*0.75, self.grid_size*self.xyreso*0.75)
        plt.ylim(-self.grid_size*self.xyreso*0.75, self.grid_size*self.xyreso*0.75)
        for x, y in zip(raycast.T[0], raycast.T[1]):
           plt.plot([0.0, x], [0.0, y], 'c-')
        plt.plot(obstacle.T[0], obstacle.T[1], "ob")
        plt.quiver(pose[0], pose[1], 5*math.cos(pose[2]), 5*math.sin(pose[2]),
                   angles='xy',scale_units='xy',scale=1)

  
        plt.show()
