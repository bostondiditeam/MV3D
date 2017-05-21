#!/usr/bin/env python

from __future__ import print_function
import rospy, tf
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, InteractiveMarker,  InteractiveMarkerControl, InteractiveMarkerFeedback
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from interactive_markers.menu_handler import MenuHandler
import cv2, cv_bridge
from image_geometry import PinholeCameraModel
import numpy as np
import csv, sys, os
from camera_info import *
from utils import *



class Projection:

    def __init__(self, md_path, calib_file):
        self.reset()
        self.current_time = rospy.Time()
        self.calib_file = calib_file
        
        md = None
        metadata = load_metadata(md_path)
        for obs in metadata:
            if obs['obstacle_name'] == 'obs1':
                md = obs
        assert md, 'obs1 metadata not found'
        self.metadata = md
        self.server = InteractiveMarkerServer("simple_marker")
        self.menu_handler = MenuHandler()
        self.menu_handler.insert( "First Entry", callback=self.processFeedback )

    def reset(self):
        self.last_cap_r = None
        self.last_cap_f = None
        self.last_cap_yaw = None
        self.obs_centroid = None

    def track_obstacle(self) :
        obj_topics = {
            'cap_r': '/objects/capture_vehicle/rear/gps/rtkfix',
            'cap_f': '/objects/capture_vehicle/front/gps/rtkfix',
            'obs_r': '/objects/obs1/rear/gps/rtkfix'
        }

        for obj in obj_topics:
            rospy.Subscriber(obj_topics[obj],
                         Odometry,
                         self.handle_msg,
                         obj)
        
    def handle_msg(self, msg, who):
        assert isinstance(msg, Odometry)
    
        now = rospy.get_rostime()
        if now < self.current_time :
            self.reset()
        self.current_time = now
    
        if who == 'cap_r':
            self.last_cap_r = rtk_position_to_numpy(msg)
        elif who == 'cap_f' and self.last_cap_r is not None:
            cap_f = rtk_position_to_numpy(msg)
            cap_r = self.last_cap_r
    
            self.last_cap_f = cap_f
            self.last_cap_yaw = get_yaw(cap_f, cap_r)
        elif who == 'obs_r' and self.last_cap_f is not None and self.last_cap_yaw is not None:
            md = self.metadata
    
            # find obstacle rear RTK to centroid vector
            lrg_to_gps = [md['gps_l'], -md['gps_w'], md['gps_h']]
            lrg_to_centroid = [md['l'] / 2., -md['w'] / 2., md['h'] / 2.]
            obs_r_to_centroid = np.subtract(lrg_to_centroid, lrg_to_gps)
    
            # in the fixed GPS frame 
            cap_f = self.last_cap_f
            obs_r = rtk_position_to_numpy(msg)
            
            # in the capture vehicle velodyne frame
            cap_to_obs = np.dot(rotMatZ(-self.last_cap_yaw), obs_r - cap_f)
            cap_to_obs_centroid = cap_to_obs + obs_r_to_centroid
            velo_to_front = np.array([-1.0922, 0, -0.0508])
            cap_to_obs_centroid += velo_to_front
            self.obs_centroid = cap_to_obs_centroid
            
                        
            br = tf.TransformBroadcaster()
            br.sendTransform(tuple(self.obs_centroid), (0,0,0,1), now, 
                            'obs_centroid', 'velodyne')
            self.add_bbox_lidar('obs_centroid', 'obstacle vehicle')
            self.add_bbox_lidar('velodyne', 'capture vehicle')


    def add_bbox(self):
        inputName = '/image_raw'
        rospy.Subscriber(inputName, Image, self.handle_img_msg, queue_size=1)
         

    def handle_img_msg(self, img_msg):
        img = None
        bridge = cv_bridge.CvBridge()
        try:
            img = bridge.imgmsg_to_cv2(img_msg, 'bgr8')
        except cv_bridge.CvBridgeError as e:
            rospy.logerr( 'image message to cv conversion failed :' )
            rospy.logerr( e )
            print( e )
            return
    
        tx, ty, tz, yaw, pitch, roll = [0.00749025, -0.40459941, -0.51372948, 
                                        -1.66780896, -1.59875352, -3.05415572]
        translation = [tx, ty, tz, 1]
        rotationMatrix = tf.transformations.euler_matrix(roll, pitch, yaw)
        rotationMatrix[:, 3] = translation
        md = self.metadata
        dims = np.array([md['l'], md['w'], md['h']])
        outputName = '/image_bbox'
        imgOutput = rospy.Publisher(outputName, Image, queue_size=1)
        obs_centroid = self.obs_centroid
   
        if self.obs_centroid is None:
            rospy.loginfo("Couldn't find obstacle centroid")
            imgOutput.publish(bridge.cv2_to_imgmsg(img, 'bgr8'))
            return
        
        # print centroid info
        rospy.loginfo(str(obs_centroid))

        # case when obstacle is not in camera frame
        if obs_centroid[0]<2.5 :
            imgOutput.publish(bridge.cv2_to_imgmsg(img, 'bgr8'))
            return
    
        # get bbox 
        corners = [obs_centroid  +0.5*np.array([i,j,k])*dims for i in [-1,1] 
                    for j in [-1,1] for k in [-1,1]]
        projected_pts = []
        cameraModel = PinholeCameraModel()
        cam_info = load_cam_info(self.calib_file)
        cameraModel.fromCameraInfo(cam_info)
        for pt in corners:
            rotated_pt = rotationMatrix.dot(list(pt)+[1])
            projected_pts.append(cameraModel.project3dToPixel(rotated_pt))
        projected_pts = np.array(projected_pts)
        center = np.mean(projected_pts, axis=0)
        out_img = drawBbox(img, projected_pts)
        imgOutput.publish(bridge.cv2_to_imgmsg(out_img, 'bgr8'))


    def processFeedback(self, feedback ):
        p = feedback.pose.position 
        print(p.x, p.y, p.z)
        s = "Feedback from marker '" + feedback.marker_name
        s += "' / control '" + feedback.control_name + "'"
    
        mp = ""
        if feedback.mouse_point_valid:
            mp = " at " + str(feedback.mouse_point.x)
            mp += ", " + str(feedback.mouse_point.y)
            mp += ", " + str(feedback.mouse_point.z)
            mp += " in frame " + feedback.header.frame_id
    
        if feedback.event_type == InteractiveMarkerFeedback.BUTTON_CLICK:
            rospy.loginfo( s + ": button click" + mp + "." )
        elif feedback.event_type == InteractiveMarkerFeedback.MENU_SELECT:
            rospy.loginfo( s + ": menu item " + str(feedback.menu_entry_id) + " clicked" + mp + "." )
        elif feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
            rospy.loginfo( s + ": pose changed")
        elif feedback.event_type == InteractiveMarkerFeedback.MOUSE_DOWN:
            rospy.loginfo( s + ": mouse down" + mp + "." )
        elif feedback.event_type == InteractiveMarkerFeedback.MOUSE_UP:
            rospy.loginfo( s + ": mouse up" + mp + "." )
        self.server.applyChanges()
    
    
    def add_bbox_lidar(self, frame_id, frame_name):
        if self.obs_centroid is None :
            return
    
        now = rospy.get_rostime() 
        #now = rospy.Time.now()
    
        #br = tf.TransformBroadcaster()
        #br.sendTransform(tuple(self.obs_centroid), (0,0,0,1), now, i
        #                    'obs_centroid', 'velodyne')
    
        # create an interactive marker server on the topic namespace simple_marker
        #server = InteractiveMarkerServer("simple_marker")
    
        int_marker = InteractiveMarker()
        #int_marker.header.frame_id = "obs_centroid"
        int_marker.header.frame_id = frame_id
        #int_marker.pose.position = Point(0,0,0)
        int_marker.header.stamp = rospy.get_rostime()
        int_marker.name = frame_name
        int_marker.description = frame_name
        int_marker.scale = 5
    
        marker = Marker()
        marker.type = Marker.CUBE
        marker.header.frame_id = "obs_centroid"
        #marker.header.frame_id = "velodyne"
        #marker.header.stamp = now
        #marker.action = Marker.ADD
    
        md = self.metadata
        marker.scale.x = md['l']
        marker.scale.y = md['w']
        marker.scale.z = md['h']
    
        #int_marker.scale.x = md['l']
        #int_marker.scale.y = md['w']
        #int_marker.scale.z = md['h']
    
        marker.color.r = 0.2
        marker.color.g = 0.5
        marker.color.b = 0.2
        marker.color.a = 0.7
    
        #marker.lifetime = rospy.Duration()
        #int_marker.lifetime = rospy.Duration()
    
        #pub = rospy.Publisher("obs_bbox", Marker, queue_size=10)
        #pub.publish(marker)
        marker_control = InteractiveMarkerControl()
        marker_control.always_visible = True
        #marker_control.interaction_mode = InteractiveMarkerControl.BUTTON
        marker_control.markers.append(marker)
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
        control.name = "move_x"
        control.orientation.w = 1
        control.orientation.x = 1
        control.orientation.y = 0
        control.orientation.z = 0
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
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
        control.name = "move_z"
        control.orientation.w = 1
        control.orientation.x = 0
        control.orientation.y = 1
        control.orientation.z = 0
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control)


        control = InteractiveMarkerControl()
        control.name = "rotate_y"
        control.orientation.w = 1
        control.orientation.x = 0
        control.orientation.y = 0
        control.orientation.z = 1
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(control)


        control = InteractiveMarkerControl()
        control.name = "move_y"
        control.orientation.w = 1
        control.orientation.x = 0
        control.orientation.y = 0
        control.orientation.z = 1
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control)
       # # add the control to the interactive marker
       # int_marker.controls.append( box_control )
    
       # # create a control which will move the box
       # # this control does not contain any markers,
       # # which will cause RViz to insert two arrows
       # rotate_control = InteractiveMarkerControl()
       # rotate_control.name = "move_x"
       # rotate_control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
    
       # # add the control to the interactive marker
       # int_marker.controls.append(rotate_control)
    
       # # add the interactive marker to our collection &
       # # tell the server to call processFeedback() when feedback arrives for it
        self.server.insert(int_marker, self.processFeedback)
        self.menu_handler.apply(self.server, int_marker.name)
    
       # # 'commit' changes and send to all clients
        self.server.applyChanges()
    



if __name__ == "__main__" :
    argv = rospy.myargv()
    rospy.init_node('projection')
    assert len(argv) == 3, 'usage: \n{} <bag_file> <calib_file>'.format(argv[0])
    
    bag_dir = os.path.dirname(argv[1])
    md_path = os.path.join(bag_dir, 'metadata.csv')
    calib_file = argv[2]
    assert os.path.isfile(md_path), 'Metadata file %s does not exist' % md_path
    assert os.path.isfile(calib_file), 'Calibration file %s does not exist' % calib_file

    try :
        p = Projection(md_path, calib_file)    
        p.track_obstacle()
        p.add_bbox()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

