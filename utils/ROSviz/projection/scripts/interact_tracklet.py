#!/usr/bin/env python

from __future__ import print_function
import rospy, rosbag, tf
from geometry_msgs.msg import Point, Quaternion
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, InteractiveMarker,  InteractiveMarkerControl, InteractiveMarkerFeedback
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from interactive_markers.menu_handler import MenuHandler
import cv2, cv_bridge
from image_geometry import PinholeCameraModel
import numpy as np
import csv, sys, os, copy
from collections import defaultdict
import PyKDL as kd
from camera_info import *
from utils import *
from parse_tracklet import *


CAMERA_TOPICS = ["/image_raw"]

class Frame():
    def __init__(self, trans, rotq, object_type, size):
        self.trans = trans
        self.rotq = rotq
        self.object_type = object_type
        self.size = size

def extract_bag_timestamps(bag_file):
    timestamp_map = {}
    index = 0
    with rosbag.Bag(bag_file, "r") as bag:
        for topic, msg, ts in bag.read_messages(topics=CAMERA_TOPICS):
            timestamp_map[msg.header.stamp.to_nsec()] = index
            index += 1
    return timestamp_map


def generate_frame_map(tracklets):
        # map all tracklets to one timeline
    frame_map = defaultdict(list)
    for t in tracklets:
        for i in range(t.num_frames):
            frame_index = i + t.first_frame
            rot = t.rots[i]
            rotq = kd.Rotation.RPY(rot[0], rot[1], rot[2]).GetQuaternion()
            frame_map[frame_index].append(
                Frame(
                    t.trans[i],
                    rotq,
                    t.object_type,
                    t.size))
    return frame_map


class Projection:

    def __init__(self, bag_file, md_path, calib_file, tracklets):
        self.timestamp_map = extract_bag_timestamps(bag_file)
        self.calib_file = calib_file
        self.frame_map = generate_frame_map(tracklets)
        
        md = None
        metadata = load_metadata(md_path)
        for obs in metadata:
            if obs['obstacle_name'] == 'obs1':
                md = obs
        assert md, 'obs1 metadata not found'
        self.metadata = md
        self.server = InteractiveMarkerServer("obstacle_marker")
        self.br = tf.TransformBroadcaster()

        self.offset = [0,0,0]
        self.rotation_offset = [0,0,0,1]
        self.orient = (0,0,0,1)
    
        self.marker = Marker()
        self.marker.type = Marker.CUBE
        self.marker.header.frame_id = "velodyne"
    
        md = self.metadata
        self.marker.scale.x = md['l']
        self.marker.scale.y = md['w']
        self.marker.scale.z = md['h']
    
    
        self.marker.color.r = 0.2
        self.marker.color.g = 0.5
        self.marker.color.b = 0.2
        self.marker.color.a = 0.7
        
        outputName = '/image_bbox'
        self.imgOutput = rospy.Publisher(outputName, Image, queue_size=1)
        #self.markOutput = rospy.Publisher("bbox", Marker, queue_size=1)


        self.velodyne_marker = self.setup_marker(frame = "velodyne",
                            name = "capture vehicle", translation=True)
        self.obs_marker = self.setup_marker(frame = "velodyne",
                            name = "obstacle vehicle", translation=False)

    def setup_marker(self, frame="velodyne", name = "capture vehicle", translation=True):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = frame
        int_marker.name = name
        int_marker.description = name
        int_marker.scale = 3

        marker_control = InteractiveMarkerControl()
        marker_control.always_visible = True
        marker_control.markers.append(self.marker)
        int_marker.controls.append(marker_control)
    
        control = InteractiveMarkerControl()
        control.name = "rotate_x"
        control.orientation.w = 1
        control.orientation.x = 1
        control.orientation.y = 0
        control.orientation.z = 0
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.name = "rotate_z"
        control.orientation.w = 1
        control.orientation.x = 0
        control.orientation.y = 1
        control.orientation.z = 0
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(control)

        control = InteractiveMarkerControl()
        control.name = "rotate_y"
        control.orientation.w = 1
        control.orientation.x = 0
        control.orientation.y = 0
        control.orientation.z = 1
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(control)

        if not translation :
            #int_marker.pose.position = Point(0,0,0)
            return int_marker

        control = InteractiveMarkerControl()
        control.name = "move_x"
        control.orientation.w = 1
        control.orientation.x = 1
        control.orientation.y = 0
        control.orientation.z = 0
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control)
    

        control = InteractiveMarkerControl()
        control.name = "move_z"
        control.orientation.w = 1
        control.orientation.x = 0
        control.orientation.y = 1
        control.orientation.z = 0
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control)


        control = InteractiveMarkerControl()
        control.name = "move_y"
        control.orientation.w = 1
        control.orientation.x = 0
        control.orientation.y = 0
        control.orientation.z = 1
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control)
        return int_marker

    def add_bbox(self):
        inputName = '/image_raw'
        rospy.Subscriber(inputName, Image, self.handle_img_msg, queue_size=1)

    def handle_img_msg(self, img_msg):
        now = rospy.get_rostime()
        img = None
        bridge = cv_bridge.CvBridge()
        try:
            img = bridge.imgmsg_to_cv2(img_msg, 'bgr8')
        except cv_bridge.CvBridgeError as e:
            rospy.logerr( 'image message to cv conversion failed :' )
            rospy.logerr( e )
            print( e )
            return
   
        self.frame_index = self.timestamp_map[img_msg.header.stamp.to_nsec()]
        f = self.frame_map[self.frame_index][0]
        obs_centroid = np.array(f.trans) + np.array(self.offset)
        R = tf.transformations.quaternion_matrix(self.rotation_offset)
        rotated_centroid = R.dot(list(obs_centroid)+[1])
        obs_centroid = rotated_centroid[:3]
        #self.orient = list(f.rotq)
 
        self.marker.header.stamp = now
        self.marker.pose.position = Point(*list(obs_centroid))
        self.marker.pose.orientation = Quaternion(*self.orient)
        self.obs_marker.pose.position = Point(*list(obs_centroid))
        self.obs_marker.pose.orientation = Quaternion(*self.orient)
        self.add_bbox_lidar()

        md = self.metadata
        dims = np.array([md['l'], md['w'], md['h']])
        outputName = '/image_bbox'
   
        if obs_centroid is None:
            rospy.loginfo("Couldn't find obstacle centroid")
            imgOutput.publish(bridge.cv2_to_imgmsg(img, 'bgr8'))
            return
        
        # print centroid info
        rospy.loginfo(str(obs_centroid))

        # case when obstacle is not in camera frame
        if obs_centroid[0]<2.5 :
            self.imgOutput.publish(bridge.cv2_to_imgmsg(img, 'bgr8'))
            return
    
        # get bbox 
        R = tf.transformations.quaternion_matrix(self.orient)
        corners = [0.5*np.array([i,j,k])*dims for i in [-1,1] 
                    for j in [-1,1] for k in [-1,1]]
        corners = [obs_centroid + R.dot(list(c)+[1])[:3] for c in corners]
        projected_pts = []
        cameraModel = PinholeCameraModel()
        cam_info = load_cam_info(self.calib_file)
        cameraModel.fromCameraInfo(cam_info)
        projected_pts = [cameraModel.project3dToPixel(list(pt)+[1]) for pt in corners]
        projected_pts = np.array(projected_pts)
        center = np.mean(projected_pts, axis=0)
        out_img = drawBbox(img, projected_pts)
        self.imgOutput.publish(bridge.cv2_to_imgmsg(out_img, 'bgr8'))

    def processFeedback(self, feedback ):
        p = feedback.pose.orientation
        self.rotation_offset = [p.x, p.y, p.z, p.w]
        p = feedback.pose.position
        self.offset = [p.x, p.y, p.z]
        self.server.applyChanges()
    
    def obs_processFeedback(self, feedback ):
        p = feedback.pose.orientation
        self.orient = (p.x, p.y, p.z, p.w)
        self.marker.pose.orientation = Quaternion(*self.orient)
        self.server.applyChanges()
    
    def add_bbox_lidar(self):
        #if obs_centroid is None :
        #    return
    
        now = rospy.get_rostime() 
        self.velodyne_marker.header.stamp = now 
        self.obs_marker.header.stamp = now 
        
        # tell the server to call processFeedback() when feedback arrives for it
        self.server.insert(self.velodyne_marker, self.processFeedback)
        self.server.applyChanges()
        self.server.insert(self.obs_marker, self.obs_processFeedback)
        self.server.applyChanges()
    



if __name__ == "__main__" :
    argv = rospy.myargv()
    rospy.init_node('projection')
    assert len(argv) == 4, 'usage: \n{} <bag_file> <tracklet_file> <calib_file>'.format(argv[0])
    
    bag_file = argv[1]
    bag_dir = os.path.dirname(bag_file)
    md_path = os.path.join(bag_dir, 'metadata.csv')
    tracklet_file = argv[2] 
    calib_file = argv[3]
    assert os.path.isfile(md_path), 'Metadata file %s does not exist' % md_path
    assert os.path.isfile(tracklet_file), 'Tracklet file %s does not exist' % tracklet_file
    assert os.path.isfile(calib_file), 'Calibration file %s does not exist' % calib_file

    tracklets = parse_xml(tracklet_file)

    try :
        p = Projection(bag_file, md_path, calib_file, tracklets)    
        p.add_bbox()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

