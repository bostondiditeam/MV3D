#!/usr/bin/env python

from __future__ import print_function
import rospy, tf
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
import cv2, cv_bridge
from image_geometry import PinholeCameraModel
import numpy as np
import csv, sys, os
from camera_info import *


def drawBbox(img, corners, color=(255,255,0)) :
    image = np.copy(img)
    #color=(255,255,0)
    thickness = 3
    for i in range(4) :
        pt1, pt2 = corners[2*i], corners[2*i+1]
        cv2.line(image, tuple(pt1.astype(int)), tuple(pt2.astype(int)), 
                 color=color, thickness=thickness)

        pt1, pt2 = corners[i+4*(i/2)], corners[2+i]
        cv2.line(image, tuple(pt1.astype(int)), tuple(pt2.astype(int)), 
                 color=color, thickness=thickness)

        pt1, pt2 = corners[i], corners[i+4]
        cv2.line(image, tuple(pt1.astype(int)), tuple(pt2.astype(int)), 
                 color=color, thickness=thickness)
    return image
    

def load_metadata(md_path):
    data = []
    with open(md_path, 'r') as f:
        reader = csv.DictReader(f) 
        for row in reader:
            # convert str to float
            row['l'] = float(row['l'])
            row['w'] = float(row['w'])
            row['h'] = float(row['h'])
            if 'gps_l' in row.keys() :
                row['gps_l'] = float(row['gps_l'])
                row['gps_w'] = float(row['gps_w'])
                row['gps_h'] = float(row['gps_h'])
            else :
                row['front_gps_l'] = float(row['front_gps_l'])
                row['front_gps_w'] = float(row['front_gps_w'])
                row['front_gps_h'] = float(row['front_gps_h'])
                row['rear_gps_l'] = float(row['rear_gps_l'])
                row['rear_gps_w'] = float(row['rear_gps_w'])
                row['rear_gps_h'] = float(row['rear_gps_h'])
            data.append(row)
    return data


def rtk_position_to_numpy(msg):
    assert isinstance(msg, Odometry)
    p = msg.pose.pose.position
    return np.array([p.x, p.y, p.z])


def get_yaw(p1, p2):
    return np.arctan2(p1[1] - p2[1], p1[0] - p2[0])


def rotMatZ(a):
    cos = np.cos(a)
    sin = np.sin(a)
    return np.array([
        [cos, -sin, 0.],
        [sin, cos,  0.],
        [0,    0,   1.]
    ])
