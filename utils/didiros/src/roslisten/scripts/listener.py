#!/usr/bin/env python

import rospy
import numpy as np
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField, Image
from velodyne_msgs.msg import VelodyneScan
#from cv_bridge import CvBridge, CvBridgeError
import Converter
from dataframe import OneFrameData

# def process_frame():
#     print('....................call MV3D....................')
#     Predictor.pred_and_pub(dataframe)
#     dataframe.clearData()

class Listener():

    def __init__(self):
        self.dataframe = OneFrameData()

    def publish_dataframe(self):
        print('............... Publish dataframe ...............')
        pub = rospy.Publisher('chatter', String, queue_size=10)
        hello_str = "hello world %s" % rospy.get_time()
        rospy.loginfo(hello_str)
        pub.publish(hello_str) # python testlisten.py to listen to test messages
        self.dataframe.clearData()

    def handle_msg(self, msg, who):
        # rospy.loginfo(who + ': I heard %s', msg)
        rospy.loginfo(who)
        if who == 'cap_r':
            last_cap_r = Converter.rtk_position_to_numpy(msg)
            self.dataframe.addData(last_cap_r, who)
            # print(last_cap_r)
        elif who == 'cap_f':
            last_cap_f = Converter.rtk_position_to_numpy(msg)
            self.dataframe.addData(last_cap_f, who)
            # print(last_cap_f)
        elif who == 'obs_r':
            last_obs_r = Converter.rtk_position_to_numpy(msg)
            self.dataframe.addData(last_obs_r, who)
            # print(last_obs_r)
        elif who == 'velodyne_points':
            last_cloudpoint_array = Converter.velodyne_to_numpy(msg)
            print(last_cloudpoint_array.shape)
            self.dataframe.addData(last_cloudpoint_array, who)
        elif who == 'image_raw':
            image_array = Converter.image_to_numpy(msg)
            print(image_array.shape)
            self.dataframe.addData(image_array, who)

        # dataframe.checkComplete();
        if (self.dataframe.checkComplete()==True):
            #process_frame()
            self.publish_dataframe()

    def startlisten(self):
        # In ROS, nodes are uniquely named. If two nodes with the same
        # name are launched, the previous one is kicked off. The
        # anonymous=True flag means that rospy will choose a unique
        # name for our 'listener' node so that multiple listeners can
        # run simultaneously.
        rospy.init_node('listener', anonymous=True)
        #rospy.init_node('talker', anonymous=True)

        # Odometry topics
        rospy.Subscriber('/objects/capture_vehicle/rear/gps/rtkfix', Odometry, self.handle_msg, 'cap_r')
        rospy.Subscriber('/objects/capture_vehicle/front/gps/rtkfix', Odometry, self.handle_msg, 'cap_f')
        rospy.Subscriber('/objects/obs1/rear/gps/rtkfix', Odometry, self.handle_msg, 'obs_r')
        # Lidar topics
        rospy.Subscriber('/velodyne_packets', VelodyneScan, self.handle_msg, 'velodyne_packets')
        rospy.Subscriber('/velodyne_points', PointCloud2, self.handle_msg, 'velodyne_points')
        # Camera topic
        rospy.Subscriber('/image_raw', Image, self.handle_msg, 'image_raw')

        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()

if __name__ == '__main__':
    #dataframe = OneFrameData()
    listener = Listener()
    print('listener started...')
    listener.startlisten()
