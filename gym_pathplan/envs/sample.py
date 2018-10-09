#coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import math

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import sys
import os

class Sample(gym.Env):
    metadata = {'render.modes' : ['human', 'rgb_array']}

    def __init__(self):
        # world param
        self.map_size = 200   # [cell]
        self.xyreso   = 0.25  # [m] 
        self.dt = 0.1 # [s]

        # robot param
        self.robot_radius = 0.3 # [m]

        # action param
        self.max_velocity = 1.0   # [m/s]
        self.min_velocity = -0.5  # [m/s]
        self.max_velocity_acceleration = 0.2  # [m/ss]
        self.min_velocity_acceleration = -0.2 # [m/ss]
        self.min_angular_velocity = math.radians(-40)  # [rad/s]
        self.max_angular_velocity = math.radians(40) # [rad/s]
        self.min_angular_acceleration = math.radians(-40)  # [rad/ss]
        self.max_angular_acceleration = math.radians(40) # [rad/ss]

        # lidar param
        self.yawreso = math.radians(10) # [rad]
        self.min_range = 0.30 # [m]
        self.max_range = 30.0 # [m]
        self.angle_limit = math.radians(270) # [rad]
        self.lidar_num = int(round(self.angle_limit/self.yawreso)+1)

        # set action_space (velocity[m/s], omega[rad/s])
        self.action_low  = np.array([self.min_velocity, self.min_angular_velocity]) 
        self.action_high = np.array([self.max_velocity, self.max_angular_velocity]) 
        self.action_space = spaces.Box(self.action_low, self.action_high, dtype=np.float32)

        # set observation_space
        self.min_yawrate  = math.radians(0)  # [rad]
        self.max_yawrate  = math.radians(360) # [rad]
        # state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
        self.state_low  = np.array([0.0, 0.0, self.min_yawrate, self.min_velocity, self.min_angular_velocity])
        self.state_high = np.array([0.0, 0.0, self.max_yawrate, self.max_velocity, self.max_angular_velocity])
        self.observation_space = spaces.Dict({
            "state": spaces.Box(self.state_low, self.state_high, dtype=np.float32),
            "lidar": spaces.Box(low=self.min_range, high=self.max_range, shape=(self.lidar_num, )),
            "goal" : spaces.Box(low=0.0, high=self.map_size*self.xyreso, shape=(2, )),
            })

        self.viewer = None

    def reset(self):
        self.map = self.reset_map() # [cell]
        
        while True:
            # static start and goal position
            # start = np.array([25, 25], dtype=np.int32)
            # goal  = np.array([75, 25], dtype=np.int32)

            # random start and goal position
            start = np.array((np.random.rand(2) * self.map_size), dtype=np.int32) # [m] 
            goal  = np.array((np.random.rand(2) * self.map_size), dtype=np.int32) # [y]
            # yaw = np.random.rand(1) * math.pi * 2

            if self.map[start[0]][start[1]] or self.map[goal[0]][goal[1]]:
                print("obstacle is put")
            elif np.all(start==goal):
                print("start and goal position is same")
            else:
                self.start = start * self.xyreso
                self.goal = goal * self.xyreso
                break
        
        # state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)
        self.state = np.array([self.start[0], self.start[1], 0.0, 0.0, 0.0])  
        self.observation = self.observe()
        self.done = False

        return self.observation

    def step(self, action):
        self.state[0] += action[0] * math.cos(self.state[2]) * self.dt
        self.state[1] += action[0] * math.sin(self.state[2]) * self.dt
        self.state[2] += action[1] * self.dt

        if self.state[2]<0.0:
            self.state[2] += math.pi * 2.0
        elif math.pi * 2.0 < self.state[2]:
            self.state[2] -= math.pi * 2.0
        self.state[3] = action[0]
        self.state[4] = action[1]

        self.observation = self.observe()
        reward = self.reward()
        self.done = self.is_done(True)
        return self.observation, reward, self.done, {}

    def observe(self):
        observation = {'state': self.state,
                       'goal' : self.goal}
        return observation
    
    def reward(self):
        reward = 0
        if self.is_goal():
            reward = 100
        elif not self.is_movable():
            reward = -5
        elif self.is_collision():
            reward = -5
        else:
            reward = -1

        return reward
    
    def reset_map(self):
        grid_map = np.zeros((self.map_size,self.map_size), dtype=np.int32)
        # wall
        for i in range(0,self.map_size):
            grid_map[i][0] = 1
            grid_map[i][self.map_size-1] = 1
        for j in range(0,self.map_size):
            grid_map[0][j] = 1
            grid_map[self.map_size-1][j] = 1
        # wall
        for i in range(0, self.map_size/4):
            grid_map[i][self.map_size/2] = 1
            grid_map[i+self.map_size/4*3][self.map_size/2] = 1
        
        # obstacle
        ox = np.array((np.random.rand(10) * self.map_size), dtype=np.int32)
        oy = np.array((np.random.rand(10) * self.map_size), dtype=np.int32)
        for ix, iy in zip(ox, oy):
            if not grid_map[ix][iy]:
                grid_map[ix][iy] = 2

        return grid_map

    def is_done(self, show=False):
        return (not self.is_movable(show)) or self.is_collision(show) or self.is_goal(show)

    def is_goal(self, show=False):
        if math.sqrt( (self.state[0]-self.goal[0])**2 + (self.state[1]-self.goal[1])**2 ) <= self.robot_radius:
            if show:
                print("Goal")
            return True
        else:
            return False

    def is_movable(self, show=False):
        x = int(self.state[0]/self.xyreso)
        y = int(self.state[1]/self.xyreso)

        if(0<=x<self.map_size and 0<=y<self.map_size and self.map[x,y] == 0):
            return True
        else:
            if show:
                print("%f %f is not movable area" % (x*self.xyreso, y*self.xyreso))
            return False

    def is_collision(self, show=False):
        flag = False
        x = int(self.state[0]/self.xyreso) #[cell]
        y = int(self.state[1]/self.xyreso) #[cell]

        obstacle = np.where(0<self.map)

        for ox, oy in zip(obstacle[0], obstacle[1]):
            distance = np.sqrt((x-ox)**2 + (y-oy)**2) * self.xyreso
            if distance < self.robot_radius:
                if show:
                    print("collision")
                flag = True
                break
        
        return flag

    def render(self, mode='human', close=False):
        screen_width  = 1000
        screen_height = 1000
        scale_width = screen_width / float(self.map_size)
        scale_height = screen_height / float(self.map_size)

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            # wall
            wall = rendering.make_capsule(screen_width, 
                                          self.robot_radius/self.xyreso*scale_width*2)
            self.walltrans = rendering.Transform()
            wall.add_attr(self.walltrans)
            wall.set_color(0.2, 0.4, 1.0)
            self.walltrans.set_translation(0, 0)
            self.walltrans.set_rotation(0)
            self.viewer.add_geom(wall)

            wall = rendering.make_capsule(screen_width, 
                                          self.robot_radius/self.xyreso*scale_width*2)
            self.walltrans = rendering.Transform()
            wall.add_attr(self.walltrans)
            wall.set_color(0.2, 0.4, 1.0)
            self.walltrans.set_translation(0, 0)
            self.walltrans.set_rotation(math.pi/2)
            self.viewer.add_geom(wall)

            wall = rendering.make_capsule(screen_width, 
                                          self.robot_radius/self.xyreso*scale_width*2)
            self.walltrans = rendering.Transform()
            wall.add_attr(self.walltrans)
            wall.set_color(0.2, 0.4, 1.0)
            self.walltrans.set_translation(0, screen_height)
            self.walltrans.set_rotation(0)
            self.viewer.add_geom(wall)

            wall = rendering.make_capsule(screen_width, 
                                          self.robot_radius/self.xyreso*scale_width*2)
            self.walltrans = rendering.Transform()
            wall.add_attr(self.walltrans)
            wall.set_color(0.2, 0.4, 1.0)
            self.walltrans.set_translation(screen_width, 0)
            self.walltrans.set_rotation(math.pi/2)
            self.viewer.add_geom(wall)

            wall = rendering.make_capsule(self.map_size*0.25*scale_width,
                                          self.robot_radius/self.xyreso*scale_width)
            self.walltrans = rendering.Transform()
            wall.add_attr(self.walltrans)
            wall.set_color(0, 0, 0)
            self.walltrans.set_translation(0, self.map_size*0.5*scale_height)
            self.walltrans.set_rotation(0)
            self.viewer.add_geom(wall)

            wall = rendering.make_capsule(self.map_size*0.25*scale_width, 
                                          self.robot_radius/self.xyreso*scale_width)
            self.walltrans = rendering.Transform()
            wall.add_attr(self.walltrans)
            wall.set_color(0, 0, 0)
            self.walltrans.set_translation(self.map_size*0.75*scale_width, self.map_size/2*scale_height)
            self.walltrans.set_rotation(0)
            self.viewer.add_geom(wall)

            # goal
            goal = rendering.make_circle(self.robot_radius/self.xyreso*scale_width*1.5)
            self.goaltrans = rendering.Transform()
            goal.add_attr(self.goaltrans)
            goal.set_color(1.0, 0.0, 0.0)
            self.goaltrans.set_translation(self.goal[0]/self.xyreso*scale_width, 
                                           self.goal[1]/self.xyreso*scale_height)
            self.viewer.add_geom(goal)

            # robot pose
            robot = rendering.make_circle(self.robot_radius/self.xyreso*scale_width)
            self.robottrans = rendering.Transform()
            robot.add_attr(self.robottrans)
            robot.set_color(0.0, 0.0, 1.0)
            self.viewer.add_geom(robot)

            # robot yawrate
            orientation = rendering.make_capsule(self.robot_radius/self.xyreso*scale_width, 2.0)
            self.orientationtrans = rendering.Transform()
            orientation.add_attr(self.orientationtrans)
            orientation.set_color(0.0, 1.0, 0.0)
            self.viewer.add_geom(orientation)

            # obstacle
            index = np.where(self.map==2)
            for ix, iy in zip(index[0], index[1]):
                obstacle = rendering.make_circle(self.robot_radius/self.xyreso*scale_width)
                self.obstacletrans = rendering.Transform()
                obstacle.add_attr(self.obstacletrans)
                obstacle.set_color(0.0, 1.0, 0.0)
                self.obstacletrans.set_translation(ix*scale_width, iy*scale_height)
                self.viewer.add_geom(obstacle)

        robot_x = self.state[0]/self.xyreso * scale_width
        robot_y = self.state[1]/self.xyreso * scale_height
        self.robottrans.set_translation(robot_x, robot_y)
        self.orientationtrans.set_translation(robot_x, robot_y)
        self.orientationtrans.set_rotation(self.state[2])
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def show(self):
        wall = np.where(self.map==1)
        obstacle = np.where(self.map==2)
        plt.cla()
        plt.plot(wall[0], wall[1], "ob")
        plt.plot(obstacle[0], obstacle[1], "og")
        plt.plot(int(self.state[0]/self.xyreso), int(self.state[1]/self.xyreso), "or")
        plt.pause(0.1)
