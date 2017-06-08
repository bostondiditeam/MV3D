import numpy as np
import rospy, tf
from dataframe import OneFrameData
from std_msgs.msg import String

def pred_and_pub(dataframe):
    print('predicting one frame')
    #.... do some magic ...
    publish_bbox(0.00749025, -0.40459941, -0.51372948, -1.66780896, -1.59875352, -3.05415572)

def publish_bbox(tx, ty, tz, yaw, pitch, roll):
    # tx, ty, tz, yaw, pitch, roll = [0.00749025, -0.40459941, -0.51372948,
    #                                 -1.66780896, -1.59875352, -3.05415572]
    translation = [tx, ty, tz, 1]
    rotationMatrix = tf.transformations.euler_matrix(roll, pitch, yaw)
    rotationMatrix[:, 3] = translation
    outputName = '/image_bbox'
    #imgOutput = rospy.Publisher(outputName, Image, queue_size=1)
    # out_img = drawBbox(img, projected_pts)
    # imgOutput.publish(bridge.cv2_to_imgmsg(out_img, 'bgr8'))
    talker()

def talker():
    #pub = rospy.Publisher('chatter', String, queue_size=10)

    hello_str = "hello world %s" % rospy.get_time()
    rospy.loginfo(hello_str)
    #pub.publish(hello_str)
