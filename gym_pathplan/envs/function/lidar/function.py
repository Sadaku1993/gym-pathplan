import numpy as np
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
