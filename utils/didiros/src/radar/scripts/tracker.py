#!/usr/bin/env python

from __future__ import print_function
from __future__ import division
import rospy, rosbag, tf
from geometry_msgs.msg import Point, Quaternion, PoseArray, Pose
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from radar_driver.msg import RadarTracks 
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray, InteractiveMarker,  InteractiveMarkerControl, InteractiveMarkerFeedback
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from interactive_markers.menu_handler import MenuHandler
import cv2, cv_bridge
from image_geometry import PinholeCameraModel
import numpy as np
import csv, sys, os, copy
from collections import defaultdict
import PyKDL as kd
from parse_tracklet import *
from utils import *
from camera_info import *
import argparse
import pandas as pd
import math

pi = math.pi

# https://stackoverflow.com/questions/470690/how-to-automatically-generate-n-distinct-colors
kelly_colors_dict = dict(
    vivid_yellow=(255, 179, 0),
    strong_purple=(128, 62, 117),
    vivid_orange=(255, 104, 0),
    #very_light_blue=(166, 189, 215),
    vivid_red=(193, 0, 32),
    #grayish_yellow=(206, 162, 98),
    #medium_gray=(129, 112, 102),
    vivid_green=(0, 125, 52),
    strong_purplish_pink=(246, 118, 142),
    strong_blue=(0, 83, 138),
    strong_yellowish_pink=(255, 122, 92),
    #strong_violet=(83, 55, 122),
    vivid_orange_yellow=(255, 142, 0),
    #strong_purplish_red=(179, 40, 81),
    vivid_greenish_yellow=(244, 200, 0),
    #strong_reddish_brown=(127, 24, 13),
    #vivid_yellowish_green=(147, 170, 0),
    #deep_yellowish_brown=(89, 51, 21),
    #vivid_reddish_orange=(241, 58, 19),
    #dark_olive_green=(35, 44, 22),
    blue=(0,255,255),
    green=(0,255,150))
kelly_colors_list = kelly_colors_dict.values()


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
        self.offset = None
        
        md = None
        metadata = load_metadata(md_path)
        for obs in metadata:
            if obs['obstacle_name'] == 'obs1':
                md = obs
        assert md, 'obs1 metadata not found'
        self.metadata = md
        self.br = tf.TransformBroadcaster()

        self.markerArray = MarkerArray()
        outputName = '/image_bbox'
        self.imgOutput = rospy.Publisher(outputName, Image, queue_size=1)
        self.markOutput = rospy.Publisher("bbox", MarkerArray, queue_size=1)

    def add_offset(self, offset):
        self.offset = offset

    def add_bbox(self):
        inputName = '/image_raw'
        rospy.Subscriber(inputName, Image, self.handle_img_msg, queue_size=1)

    def create_marker(self):
        marker = Marker()
        marker.type = Marker.CUBE
        marker.header.frame_id = "velodyne"
        md = self.metadata
        marker.scale.x = md['l']
        marker.scale.y = md['w']
        marker.scale.z = md['h']
        return marker

    def handle_img_msg(self, img_msg):

        now = rospy.get_rostime()
        for m in self.markerArray.markers:
            m.action = Marker.DELETE

        img = None
        bridge = cv_bridge.CvBridge()
        try:
            img = bridge.imgmsg_to_cv2(img_msg, 'bgr8')
        except cv_bridge.CvBridgeError as e:
            rospy.logerr( 'image message to cv conversion failed :' )
            rospy.logerr( e )
            print( e )
            return

        timestamp = img_msg.header.stamp.to_nsec()
        self.frame_index = self.timestamp_map[timestamp]
        if self.offset :
            self.frame_index -= self.offset
        out_img = np.copy(img)
        for i, f in enumerate(self.frame_map[self.frame_index]):
            #f = self.frame_map[self.frame_index][0]
            #color = (255,255,0)
            color = kelly_colors_list[i % len(kelly_colors_list)]
            marker = self.create_marker() 
            marker.color.r = color[0]/255
            marker.color.g = color[1]/255
            marker.color.b = color[2]/255
            marker.color.a = 0.5
        
            marker.header.stamp = now
            marker.pose.position = Point(*f.trans)
            marker.pose.orientation = Quaternion(*f.rotq)

            marker.id = i
   
            self.markerArray.markers.append(marker) 
            #self.markOutput.publish(marker)
            

            md = self.metadata
            dims = np.array([md['l'], md['w'], md['h']])
            obs_centroid = np.array(f.trans)
            orient = list(f.rotq)
            #orient[2] -= 0.035
            #R = tf.transformations.quaternion_matrix((0,0,-0.0065,1))
            #obs_centroid = R.dot(list(obs_centroid)+[1])[:3]
    
   
            if obs_centroid is None:
                rospy.loginfo("Couldn't find obstacle centroid")
                continue
                #self.imgOutput.publish(bridge.cv2_to_imgmsg(img, 'bgr8'))
                #return
            
            # print centroid info
            rospy.loginfo(str(obs_centroid))

            # case when obstacle is not in camera frame
            if obs_centroid[0]<3 :
                continue
                #self.imgOutput.publish(bridge.cv2_to_imgmsg(img, 'bgr8'))
                #return
    
            # get bbox 
            R = tf.transformations.quaternion_matrix(orient)
            #R = tf.transformations.quaternion_matrix([0,0,0,1])
            corners = [0.5*np.array([i,j,k])*dims for i in [-1,1] 
                        for j in [-1,1] for k in [-1,1]]
            corners = np.array([obs_centroid + R.dot(list(c)+[1])[:3] for c in corners])
            #if self.ground_corr is not None:
            #    z_min, x_min, y_min = self.ground_corr.loc[timestamp]
            #    z_offset = z_min - min(corners[:,2])
            #    x_offset = x_min - min(corners[:,0])
            #    y_offset = y_min - min(corners[:,1])
            #    corr = np.array([0, 0, z_offset])
            #    #corr = np.array([0, 0,0])
            #    #corr = np.array([x_offset, y_offset, z_offset])
            #    corr[np.isnan(corr)]=0
            #    corners+=corr
            #print(corners)
            cameraModel = PinholeCameraModel()
            cam_info = load_cam_info(self.calib_file)
            cameraModel.fromCameraInfo(cam_info)
            projected_pts = [cameraModel.project3dToPixel(list(pt)+[1]) for pt in corners]
            projected_pts = np.array(projected_pts)
            center = np.mean(projected_pts, axis=0)
            out_img = drawBbox(out_img, projected_pts, color=color[::-1])
       

        self.markOutput.publish(self.markerArray)
        self.imgOutput.publish(bridge.cv2_to_imgmsg(out_img, 'bgr8'))


class Tracker:

    def __init__(self, bag_file, md_path, tracklets):
        self.calib_file = 'ost_new.yaml'
        self.timestamp_map = extract_bag_timestamps(bag_file)
        self.frame_map = generate_frame_map(tracklets)
        self.offset = None
        
        md = None
        metadata = load_metadata(md_path)
        for obs in metadata:
            if obs['obstacle_name'] == 'obs1':
                md = obs
        assert md, 'obs1 metadata not found'
        self.metadata = md
        self.posePub = rospy.Publisher("bbox_pose", PoseArray, queue_size=1) #TODO : integrate with detector
        self.radarPub = rospy.Publisher("radar/proposal", PoseArray, queue_size=1)
        self.radarPub = rospy.Publisher("radar/proposal", PoseArray, queue_size=1)
        self.detected_poses = None
        self.detection_time = None
        self.markerArray = MarkerArray()

        outputName = '/image_bbox'
        self.imgOutput = rospy.Publisher(outputName, Image, queue_size=1)
        self.markOutput = rospy.Publisher("bbox", MarkerArray, queue_size=1)
        self.raw_image = None
    
    def startlistening(self):
        rospy.init_node('tracker', anonymous=True)
        rospy.Subscriber('/image_raw', Image, self.handle_image_msg) #TODO : integrate with detector 
        rospy.Subscriber("/detected_pose", PoseArray, self.handle_pose_msg)
        rospy.Subscriber("/radar/tracks", RadarTracks, self.handle_radar_msg)
        print('tracker node initialzed')
        rospy.spin()

    def create_marker(self):
        marker = Marker()
        marker.type = Marker.CUBE
        marker.header.frame_id = "velodyne"
        md = self.metadata
        marker.scale.x = md['l']
        marker.scale.y = md['l']
        marker.scale.z = md['h']+1.
        return marker

    def handle_image_msg(self, img_msg):
        now = rospy.get_rostime()
        poseArray = PoseArray()
        poseArray.header.stamp = now
        timestamp = img_msg.header.stamp.to_nsec()
        self.frame_index = self.timestamp_map[timestamp]
        for i, f in enumerate(self.frame_map[self.frame_index]):
            pose = Pose()
            pose.position = Point(*f.trans)
            pose.orientation = Quaternion(*f.rotq)
            poseArray.poses.append(pose) 
        self.posePub.publish(poseArray)
        self.raw_image = img_msg

    def handle_pose_msg(self, pose_msg):
        self.detection_time = rospy.get_rostime()
        self.detected_poses = pose_msg 

    def handle_radar_msg(self, radar_msg):
        md = self.metadata
        now = radar_msg.header.stamp
        poseArray = PoseArray()
        for track in radar_msg.tracks :
            r = track.range
            # ignore obstacles farther than 55 m
            if (track.status!=3) or (r>45) or (r<6): 
                continue
            pose = Pose()
            theta = track.angle/180.*pi
            pose.position.x = (r+md['l']/2.) * math.cos(theta) + 1.5
            pose.position.y = -(r+md['l']/2.) * math.sin(theta) 
            pose.position.z = -0.8
            poseArray.poses.append(pose)
        self.detected_poses = poseArray
        self.radarPub.publish(poseArray)
                 

        img = None
        bridge = cv_bridge.CvBridge()
        if self.raw_image is None :
            return

        try:
            img = bridge.imgmsg_to_cv2(self.raw_image, 'bgr8')
        except cv_bridge.CvBridgeError as e:
            rospy.logerr( 'image message to cv conversion failed :' )
            rospy.logerr( e )
            print( e )
            return
        out_img = np.copy(img)

        if self.detected_poses is None:
            return

        for i, pose in enumerate(self.detected_poses.poses):
            color = kelly_colors_list[i % len(kelly_colors_list)]
            marker = self.create_marker() 
            marker.color.r = color[0]/255
            marker.color.g = color[1]/255
            marker.color.b = color[2]/255
            marker.color.a = 0.5
        
            marker.header.stamp = now
            marker.pose = pose

            marker.id = i
            self.markerArray.markers.append(marker) 

            dims = np.array([md['l'], md['l'], md['h']])
            obs_centroid = np.array([pose.position.x,
                                     pose.position.y,
                                     pose.position.z])
            orient = list([pose.orientation.x,
                           pose.orientation.y,
                           pose.orientation.z,
                           pose.orientation.w])

            # case when obstacle is not in camera frame
            if obs_centroid[0]<4 :
                continue
    
            # get bbox 
            R = tf.transformations.quaternion_matrix(orient)
            corners = [0.5*np.array([i,j,k])*dims for i in [-1,1] 
                        for j in [-1,1] for k in [-1,1]]
            corners = np.array([obs_centroid + R.dot(list(c)+[1])[:3] for c in corners])
            cameraModel = PinholeCameraModel()
            cam_info = load_cam_info(self.calib_file)
            cameraModel.fromCameraInfo(cam_info)
            projected_pts = [cameraModel.project3dToPixel(list(pt)+[1]) for pt in corners]
            projected_pts = np.array(projected_pts)
            center = np.mean(projected_pts, axis=0)
            out_img = drawBbox(out_img, projected_pts, color=color[::-1])
        
        self.markOutput.publish(self.markerArray)
        self.imgOutput.publish(bridge.cv2_to_imgmsg(out_img, 'bgr8'))


if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description="tracker")
    parser.add_argument('bag', type=str, nargs='?', default='', help='bag filename')
    parser.add_argument('tracklet', type=str, nargs='?', default='', help='tracklet filename')
    parser.add_argument('--offset', type=int, help='Number of frames to offset')
    args = parser.parse_args(rospy.myargv()[1:])
    
    bag_file = args.bag
    bag_dir = os.path.dirname(bag_file)
    md_path = os.path.join(bag_dir, 'metadata.csv')
    tracklet_file = args.tracklet
    assert os.path.isfile(md_path), 'Metadata file %s does not exist' % md_path
    assert os.path.isfile(tracklet_file), 'Tracklet file %s does not exist' % tracklet_file
    tracklets = parse_xml(tracklet_file)

    try :
        tracker = Tracker(bag_file, md_path, tracklets)
        tracker.startlistening()
    except rospy.ROSInterruptException:
        pass

