import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField, Image
from nav_msgs.msg import Odometry
from velodyne_msgs.msg import VelodyneScan
from cv_bridge import CvBridge, CvBridgeError

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
