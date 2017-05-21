#!/usr/bin/env python

from __future__ import print_function
import rospy
from sensor_msgs.msg import CameraInfo
import sys, os
import yaml

def load_cam_info(calib_file):
    cam_info = CameraInfo()
    with open(calib_file,'r') as cam_calib_file :
        cam_calib = yaml.load(cam_calib_file)
        cam_info.height = cam_calib['image_height']
        cam_info.width = cam_calib['image_width']
        cam_info.K = cam_calib['camera_matrix']['data']
        cam_info.D = cam_calib['distortion_coefficients']['data']
        cam_info.R = cam_calib['rectification_matrix']['data']
        cam_info.P = cam_calib['projection_matrix']['data']
        cam_info.distortion_model = cam_calib['distortion_model']
    return cam_info

if __name__ == "__main__":
    calib_file=None
    argv = rospy.myargv()
    if len(argv) == 2:
        calib_file = argv[1]
    else :
        print('Invalid argument, use as follows')
        print(argv[0]+' <calibration file>')
        sys.exit(1)

    if not os.path.exists(calib_file) :
        print(calib_file+' does not exist.')
        sys.exit(1)
 
    try:
        #load_cam_info(calib_file)
        pub = rospy.Publisher('camera', CameraInfo,  queue_size=1)
        rospy.init_node('CameraInfo')   
        cam_info = load_cam_info(calib_file)
        rate = rospy.Rate(30) # 30 Hz
        while not rospy.is_shutdown():
            pub.publish(cam_info)
            rate.sleep()
    except rospy.ROSInterruptException:
        pass 
