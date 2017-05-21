#!/usr/bin/env python

import numpy as np
import csv
import sys
import os

import rospy
import tf
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker


last_cap_r = None
last_cap_f = None
last_cap_yaw = None
current_time = rospy.Time()


def reset():
    global last_cap_r, last_cap_f, last_cap_yaw
    last_cap_r = None
    last_cap_f = None
    last_cap_yaw = None


def load_metadata(md_path):
    data = []
    with open(md_path, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # convert str to float
            row['l'] = float(row['l'])
            row['w'] = float(row['w'])
            row['h'] = float(row['h'])
            row['gps_l'] = float(row['gps_l'])
            row['gps_w'] = float(row['gps_w'])
            row['gps_h'] = float(row['gps_h'])
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


def handle_msg(msg, who):
    assert isinstance(msg, Odometry)

    global last_cap_r, last_cap_f, last_cap_yaw, current_time

    now = rospy.get_rostime()
    if now<current_time :
        reset()
        try:
            rospy.delete_param('obs_centroid')
        except KeyError:
            pass
    current_time = now

    if who == 'cap_r':
        last_cap_r = rtk_position_to_numpy(msg)
    elif who == 'cap_f' and last_cap_r is not None:
        cap_f = rtk_position_to_numpy(msg)
        cap_r = last_cap_r

        last_cap_f = cap_f
        last_cap_yaw = get_yaw(cap_f, cap_r)
    elif who == 'obs_r' and last_cap_f is not None and last_cap_yaw is not None:
        md = None
        for obs in metadata:
            if obs['obstacle_name'] == 'obs1':
                md = obs
        assert md, 'obs1 metadata not found'

        # find obstacle rear RTK to centroid vector
        lrg_to_gps = [md['gps_l'], -md['gps_w'], md['gps_h']]
        lrg_to_centroid = [md['l'] / 2., -md['w'] / 2., md['h'] / 2.]
        obs_r_to_centroid = np.subtract(lrg_to_centroid, lrg_to_gps)

        # in the fixed GPS frame 
        cap_f = last_cap_f
        obs_r = rtk_position_to_numpy(msg)
        
        # in the capture vehicle front RTK frame
        cap_to_obs = np.dot(rotMatZ(-last_cap_yaw), obs_r - cap_f)
        cap_to_obs_centroid = cap_to_obs + obs_r_to_centroid
        velo_to_front = np.array([-1.0922, 0, -0.0508])
        cap_to_obs_centroid += velo_to_front

        rospy.set_param('obs_centroid', cap_to_obs_centroid.tolist())

        #br = tf.TransformBroadcaster()
        #br.sendTransform(tuple(cap_to_obs_centroid), (0,0,0,1), 
        #    rospy.Time.now(), 'obs_centroid', 'velodyne')

        ## publish obstacle bounding box
        #marker = Marker()
        #marker.header.frame_id = "obs_centroid"
        #marker.header.stamp = rospy.Time.now()

        #marker.type = Marker.CUBE
        #marker.action = Marker.ADD

        #marker.scale.x = md['l']
        #marker.scale.y = md['w']
        #marker.scale.z = md['h']

        #marker.color.r = 0.2
        #marker.color.g = 0.5
        #marker.color.b = 0.2
        #marker.color.a = 0.5

        #marker.lifetime = rospy.Duration()

        #pub = rospy.Publisher("obs_bbox", Marker, queue_size=10)
        #pub.publish(marker)


if __name__ == '__main__':
    argv = rospy.myargv()
    rospy.init_node('base_link_to_obs1_tf_broadcaster')
    
    # [filepath, argument1, argument2, ..., argumentN, nodename, logpath]
    assert len(argv) == 2
    
    # compose path to metadata file
    bag_path = argv[1]
    bag_dir = os.path.dirname(bag_path)
    md_path = os.path.join(bag_dir, 'metadata.csv')
    assert os.path.isfile(md_path), 'Metadata file %s does not exists' % md_path
    
    metadata = load_metadata(md_path)

    obj_topics = {
        'cap_r': '/objects/capture_vehicle/rear/gps/rtkfix',
        'cap_f': '/objects/capture_vehicle/front/gps/rtkfix',
        'obs_r': '/objects/obs1/rear/gps/rtkfix'
    }
    
    for obj in obj_topics:
        rospy.Subscriber(obj_topics[obj],
                         Odometry,
                         handle_msg,
                         obj)

    rospy.spin()
