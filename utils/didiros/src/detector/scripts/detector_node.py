#!/usr/bin/env python

from __future__ import print_function, division 
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import NavSatFix
from sensor_msgs.msg import PointCloud2, PointField
from cv_bridge import CvBridge, CvBridgeError
import message_filters
from matplotlib import cm

import numpy as np
import math
import sys
import cv2


#---------------------------------------------------------------------------------------------------------

# PointCloud2 to array
#       https://gist.github.com/dlaz/11435820
#       https://github.com/pirobot/ros-by-example/blob/master/rbx_vol_1/rbx1_apps/src/point_cloud2.py
#       http://answers.ros.org/question/202787/using-pointcloud2-data-getting-xy-points-in-python/
#       https://github.com/eric-wieser/ros_numpy/blob/master/src/ros_numpy/point_cloud2.py

def point_cloud_2_top(points,
                      res=0.1,
                      zres=0.3,
                      side_range=(-10., 10.),  # left-most to right-most
                      fwd_range=(-10., 10.),  # back-most to forward-most
                      height_range=(-2., 2.),  # bottom-most to upper-most
                      ):
    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]
    reflectance = points[:,3]

    # INITIALIZE EMPTY ARRAY - of the dimensions we want
    x_max = int((side_range[1] - side_range[0]) / res)
    y_max = int((fwd_range[1] - fwd_range[0]) / res)
    z_max = int((height_range[1] - height_range[0]) / zres)
    # z_max =
    top = np.zeros([y_max+1, x_max+1, z_max+2], dtype=np.float32)

    # FILTER - To return only indices of points within desired cube
    # Three filters for: Front-to-back, side-to-side, and height ranges
    # Note left side is positive y axis in LIDAR coordinates
    f_filt = np.logical_and(
        (x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and(
        (y_points > -side_range[1]), (y_points < -side_range[0]))
    filter = np.logical_and(f_filt, s_filt)

    # filter capture vehicle
    c_filt = np.logical_or(np.abs(x_points)>4.7/2, np.abs(y_points)>2.1/2)
    filter = np.logical_and(filter, c_filt)

    for i, height in enumerate(np.arange(height_range[0], height_range[1], zres)):

        z_filt = np.logical_and((z_points >= height),
                                (z_points < height + zres))
        zfilter = np.logical_and(filter, z_filt)
        indices = np.argwhere(zfilter).flatten()

        # KEEPERS
        xi_points = x_points[indices]
        yi_points = y_points[indices]
        zi_points = z_points[indices]
        ref_i = reflectance[indices]

        # CONVERT TO PIXEL POSITION VALUES - Based on resolution
        #print("[{},{},{},{}] {}".format(xi_points, yi_points, zi_points, ref_i, res))
        x_img = (-yi_points / res).astype(np.int32)  # x axis is -y in LIDAR
        y_img = (-xi_points / res).astype(np.int32)  # y axis is -x in LIDAR

        # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
        # floor & ceil used to prevent anything being rounded to below 0 after
        # shift
        x_img -= int(np.floor(side_range[0] / res))
        y_img += int(np.floor(fwd_range[1] / res))

        # CLIP HEIGHT VALUES - to between min and max heights
        pixel_values = (zi_points - height)/zres
        # pixel_values = zi_points

        # FILL PIXEL VALUES IN IMAGE ARRAY
        top[y_img, x_img, i] = pixel_values

        # intensity at max height
        top[y_img, x_img, z_max] = ref_i

        # density
        for y,x in zip(y_img, x_img) :
            top[y, x, z_max+1] += 1.
    top[:,:,z_max] /= 255. 
    density = np.log(1.+top[:,:,z_max+1])/math.log(32.)
    density[density>1] = 1.
    top[:,:,z_max+1] = density
    return top

def draw_top_image(top):
    top_image = np.moveaxis(top, -1, 0)
    top_image = cm.hot(np.concatenate(top_image, axis=1))
    return cv2.cvtColor(top_image.astype(np.float32), cv2.COLOR_RGB2BGR)

# https://github.com/eric-wieser/ros_numpy #############################################################################################

DUMMY_FIELD_PREFIX = '__'

# mappings between PointField types and numpy types
type_mappings = [(PointField.INT8, np.dtype('int8')), (PointField.UINT8, np.dtype('uint8')), (PointField.INT16, np.dtype('int16')),
                 (PointField.UINT16, np.dtype('uint16')), (PointField.INT32, np.dtype('int32')), (PointField.UINT32, np.dtype('uint32')),
                 (PointField.FLOAT32, np.dtype('float32')), (PointField.FLOAT64, np.dtype('float64'))]

pftype_to_nptype = dict(type_mappings)
nptype_to_pftype = dict((nptype, pftype) for pftype, nptype in type_mappings)

# sizes (in bytes) of PointField types
pftype_sizes = {PointField.INT8: 1, PointField.UINT8: 1, PointField.INT16: 2, PointField.UINT16: 2,
                PointField.INT32: 4, PointField.UINT32: 4, PointField.FLOAT32: 4, PointField.FLOAT64: 8}



def fields_to_dtype(fields, point_step):
    '''Convert a list of PointFields to a numpy record datatype.
    '''
    offset = 0
    np_dtype_list = []
    for f in fields:
        while offset < f.offset:
            # might be extra padding between fields
            np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
            offset += 1

        dtype = pftype_to_nptype[f.datatype]
        if f.count != 1:
            dtype = np.dtype((dtype, f.count))

        np_dtype_list.append((f.name, dtype))
        offset += pftype_sizes[f.datatype] * f.count

    # might be extra padding between points
    while offset < point_step:
        np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
        offset += 1

    return np_dtype_list

def msg_to_arr(msg):

    dtype_list = fields_to_dtype(msg.fields, msg.point_step)
    arr = np.fromstring(msg.data, dtype_list)

    # remove the dummy fields that were added
    arr = arr[[fname for fname, _type in dtype_list if not (fname[:len(DUMMY_FIELD_PREFIX)] == DUMMY_FIELD_PREFIX)]]

    if msg.height == 1:
        return np.reshape(arr, (msg.width,))
    else:
        return np.reshape(arr, (msg.height, msg.width))

def gpsfix_front_callback(msg):
    print("Received front gps fix message")
    pass

def gpsfix_rear_callback(msg):
    print("Received rear gps fix message")
    pass

def radar_points_callback(msg):
    print("Received radar_points message")
    pass

class Frame():
    def __init__(self) :
        # TODO : 
        # create config file and load parameters here

        # camera image
        self.camera_image = None
        # lidar top view
        self.lidar_top = None
        # Instantiate CvBridge
        self.bridge = CvBridge()
        # current time (assume sync with rosbag time)
        self.current_time = None

    def image_callback(self, msg):
        print("Receive image_callback message seq=%d, timestamp=%19d" % (msg.header.seq, msg.header.stamp.to_nsec()))
        self.camera_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.current_time =  msg.header.stamp

    def velodyne_callback(self, msg) :
        if msg.header.stamp < self.current_time - rospy.Duration(0.04):
            return
        print("Receive velodyne_points message seq=%d, timestamp=%19d" % (msg.header.seq, msg.header.stamp.to_nsec()))
        arr = msg_to_arr(msg)
        lidar = np.array([[item[0], item[1], item[2], item[3]] for item in arr])
        side_range = (-10., 10.)
        fwd_range = (-45., 45.)
        height_range = (-3., 0.5)
        res = 0.2
    
        camera_image = self.camera_image.astype(np.float32)/255.
        print("camera_image is {}".format(camera_image.shape))
    
        top_view = point_cloud_2_top(lidar, res=res, zres=0.5, side_range=side_range, fwd_range=fwd_range,
                                     height_range=height_range)
        top_image = draw_top_image(top_view) 
    
        if 1:           # if visualize
            scale = top_image.shape[1]/camera_image.shape[1]
            camera_img = cv2.resize(camera_image,(top_image.shape[1], int(camera_image.shape[0]*scale)))
            image_out = np.concatenate((top_image, camera_img), axis=0)
            cv2.imshow("top", image_out)
            cv2.waitKey(1)

if __name__ == '__main__':
    print(sys.version)

    rospy.init_node('test_node')
    f = Frame()

    rospy.Subscriber('/image_raw', Image, f.image_callback)
    rospy.Subscriber('/velodyne_points', PointCloud2, f.velodyne_callback)
    #rospy.Subscriber('/objects/capture_vehicle/front/gps/fix', NavSatFix, gpsfix_front_callback)
    #rospy.Subscriber('/objects/capture_vehicle/rear/gps/fix', NavSatFix, gpsfix_rear_callback)
    #rospy.Subscriber('/radar/points', PointCloud2, radar_points_callback)

    # Spin until ctrl + c
    rospy.spin()
