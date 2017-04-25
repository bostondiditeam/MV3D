#!/usr/bin/env python

import rospy
import numpy as np
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField, Image
from velodyne_msgs.msg import VelodyneScan
from cv_bridge import CvBridge, CvBridgeError

class OneFrameData():
    def __init__(self):
        self.clearData()

    def clearData(self):
        self.cap_r = None
        self.cap_f = None
        self.obs_r = None
        self.velodyne_points = None
        self.image_raw = None
        self.complete = False

    def addData(self, data, type):
        if type == 'cap_r':
            self.cap_r = data
        elif type == 'cap_f':
            self.cap_f = data
        elif type == 'obs_r':
            self.obs_r = data
        elif type == 'velodyne_points':
            self.velodyne_points = data
        elif type == 'image_raw':
            self.image_raw = data
        self.checkComplete()

    def checkComplete(self):
        if ((self.cap_r is not None) and
            (self.cap_f is not None) and
            (self.obs_r is not None) and
            (self.velodyne_points is not None) and
            (self.image_raw is not None)):
            self.complete = True
        else:
            self.complete = False
        return self.complete

dataframe = OneFrameData()


##################################################################################################################################
# Ref: https://github.com/hengck23/didi-udacity-2017/blob/master/baseline-04/didi_data/ros_scripts/run_dump_lidar.py
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

##################################################################################################################################

def rtk_position_to_numpy(msg):
    assert isinstance(msg, Odometry)
    timestamp = msg.header.stamp.to_nsec()
    print('Odometry : seq=%d, timestamp=%19d'%(
        msg.header.seq, timestamp
    ))
    p = msg.pose.pose.position
    return np.array([p.x, p.y, p.z])

def velodyne_to_numpy(msg):
    assert isinstance(msg, PointCloud2)
    timestamp = msg.header.stamp.to_nsec()
    print('PointCloud : seq=%d, timestamp=%19d'%(
        msg.header.seq, timestamp
    ))
    arr = msg_to_arr(msg)
    return arr

#### from bg_to_kittti.py ###
def image_to_numpy(msg):
    assert isinstance(msg, Image)
    timestamp = msg.header.stamp.to_nsec()
    print('Image : seq=%d, timestamp=%19d'%(
        msg.header.seq, timestamp
    ))

    bridge = CvBridge()
    results = {}
    if hasattr(msg, 'format') and 'compressed' in msg.format:
        buf = np.ndarray(shape=(1, len(msg.data)), dtype=np.uint8, buffer=msg.data)
        cv_image = cv2.imdecode(buf, cv2.IMREAD_ANYCOLOR)
        if cv_image.shape[2] != 3:
            print("Invalid image %s" % image_filename)
            return results
        results['height'] = cv_image.shape[0]
        results['width'] = cv_image.shape[1]
    else:
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        #cv2.imwrite(image_filename, cv_image)

    return cv_image

def process_frame():
    print('...call MV3D...')
    dataframe.clearData()

def handle_msg(msg, who):
    # rospy.loginfo(who + ': I heard %s', msg)
    rospy.loginfo(who)
    if who == 'cap_r':
        last_cap_r = rtk_position_to_numpy(msg)
        dataframe.addData(last_cap_r, who)
        # print(last_cap_r)
    elif who == 'cap_f':
        last_cap_f = rtk_position_to_numpy(msg)
        dataframe.addData(last_cap_f, who)
        # print(last_cap_f)
    elif who == 'obs_r':
        last_obs_r = rtk_position_to_numpy(msg)
        dataframe.addData(last_obs_r, who)
        # print(last_obs_r)
    elif who == 'velodyne_points':
        last_cloudpoint_array = velodyne_to_numpy(msg)
        print(last_cloudpoint_array.shape)
        dataframe.addData(last_cloudpoint_array, who)
    elif who == 'image_raw':
        image_array = image_to_numpy(msg)
        print(image_array.shape)
        dataframe.addData(image_array, who)

    dataframe.checkComplete();
    if dataframe.complete:
        process_frame()

def listener():
    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)

    # Odometry topics
    rospy.Subscriber('/objects/capture_vehicle/rear/gps/rtkfix', Odometry, handle_msg, 'cap_r')
    rospy.Subscriber('/objects/capture_vehicle/front/gps/rtkfix', Odometry, handle_msg, 'cap_f')
    rospy.Subscriber('/objects/obs1/rear/gps/rtkfix', Odometry, handle_msg, 'obs_r')
    # Lidar topics
    rospy.Subscriber('/velodyne_packets', VelodyneScan, handle_msg, 'velodyne_packets')
    rospy.Subscriber('/velodyne_points', PointCloud2, handle_msg, 'velodyne_points')
    # Camera topic
    rospy.Subscriber('/image_raw', Image, handle_msg, 'image_raw')

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    print('listener started...')
    listener()
