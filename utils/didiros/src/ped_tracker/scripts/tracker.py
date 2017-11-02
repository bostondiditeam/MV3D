#!/usr/bin/env python

from __future__ import print_function, division
import rospy, rosbag
from geometry_msgs.msg import Point, Quaternion, PoseArray, Pose
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from bbox_driver.msg import BboxArray, Bbox 
import numpy as np
import csv, sys, os, copy
from collections import defaultdict
import PyKDL as kd
from parse_tracklet import parse_xml
from sort import Sort
import argparse
import math
import time

pi = math.pi



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


class Tracker:

    def __init__(self, bag_file, tracklets):
        self.timestamp_map = extract_bag_timestamps(bag_file)
        self.frame_map = generate_frame_map(tracklets)
        
        #TODO : detector should publish BBoxArray:
        # below I just read bboxes from tracklet (see handle_img_msg function)
        self.detect_pub = rospy.Publisher("bbox/detections", BboxArray, queue_size=1)
        self.n_skip = 25 # to simulate delay by MV3D, skip detections (about 1 sec)
        self.skip_count = 0 # keep count of skips
 
        self.predict_pub = rospy.Publisher("bbox/predictions", BboxArray, queue_size=1)
        self.detected_bboxes = None
        self.latest_detection_time = None

        self.mot_tracker = Sort(max_age=3, 
                                min_hits=6, 
                                iou_threshold=0.1, 
                                max_time_elapsed=2)
        self.min_detections = 7
        self.track_count = {}
        self.is_busy = False
   
 
    def startlistening(self):
        rospy.init_node('tracker', anonymous=True)
        rospy.Subscriber('/image_raw', Image, self.handle_image_msg) #TODO : just for time keeping (to be removed) 
        rospy.Subscriber("bbox/detections", BboxArray, self.handle_bbox_msg)
        print('tracker node initialzed')
        rospy.Timer(rospy.Duration(0.1), self.publish_predictions)
        rospy.spin()


    def handle_bbox_msg(self, bbox_msg):
        """ saves the latest bbox detections and latest detection time
        """
        self.latest_detection_time = rospy.get_rostime()
        print("Bboxes detected at", self.latest_detection_time.to_sec())
        self.detected_bboxes = bbox_msg 


    def publish_predictions(self, event):
        # wait until first detection
        # print('enter here? ', time.time())

        if (not self.latest_detection_time) or (not self.detected_bboxes):
            return

        # if no new detections since last call :
        if self.latest_detection_time < event.last_real:
            # predict tracks without update 
            for track in self.mot_tracker.trackers:
                t = rospy.get_rostime().to_nsec()
                track.predict(t, is_update=False)
            tracks = self.mot_tracker.good_tracks()
        # if new detections since last call :
        else : 
            detections = self.detected_bboxes.bboxes
            t = self.latest_detection_time.to_nsec()
            d = []
            for bbox in detections:
                d.append([bbox.x, bbox.y, bbox.z, bbox.h, bbox.w, bbox.l])
            tracks = self.mot_tracker.update(np.array(d),t)

        # publish all tracks
        print(tracks)

        bboxArray = BboxArray()
        bboxArray.header.stamp = rospy.get_rostime()
        for track in tracks:
            trk_id = int(track[5])
            if trk_id not in self.track_count.keys():
                self.track_count[trk_id] = 0
            else :
                self.track_count[trk_id] += 1
            if self.track_count[trk_id] < self.min_detections:
                continue
            bbox = Bbox()
            bbox.x, bbox.y, bbox.z = track[0:3]
            bbox.h = track[4]
            bbox.w =  bbox.l = 2*track[3]
            bbox.yaw = 0 # TODO : needs to be changed for cars
            bboxArray.bboxes.append(bbox)
        # rospy.logerr('here: ', bboxArray)
        rospy.logerr('bboxArray={} '.format(bboxArray))
        self.predict_pub.publish(bboxArray)


    #------------------------------------------------------------
    #TODO : used here for publishing bbox (to be removed)
    #------------------------------------------------------------
    def handle_image_msg(self, img_msg):
        if self.is_busy:
            return
        self.is_busy = True
        now = rospy.get_rostime()
        bboxArray = BboxArray()
        bboxArray.header.stamp = now
        timestamp = img_msg.header.stamp.to_nsec()
        self.frame_index = self.timestamp_map[timestamp]
        for i, f in enumerate(self.frame_map[self.frame_index]):
            bbox = Bbox()
            bbox.x, bbox.y, bbox.z = f.trans
            bbox.h, bbox.w, bbox.l = f.size
            bbox.score=1.
            bboxArray.bboxes.append(bbox)
        time.sleep(0.3) #simulate MV3D  delay
        self.detect_pub.publish(bboxArray)
        rospy.logerr('detect_pub bboxArray={} '.format(bboxArray))
        self.is_busy=False
    #------------------------------------------------------------

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description="tracker")
    parser.add_argument('bag', type=str, nargs='?', default='', help='bag filename')
    parser.add_argument('tracklet', type=str, nargs='?', default='', help='tracklet filename')
    args = parser.parse_args(rospy.myargv()[1:])
    
    bag_file = args.bag
    bag_dir = os.path.dirname(bag_file)
    tracklet_file = args.tracklet
    assert os.path.isfile(bag_file), 'Bag file %s does not exist' % bag_file
    assert os.path.isfile(tracklet_file), 'Tracklet file %s does not exist' % tracklet_file
    tracklets = parse_xml(tracklet_file)

    try :
        tracker = Tracker(bag_file, tracklets)  # TODO : remove tracklets, use detector node instead  
        tracker.startlistening()
    except rospy.ROSInterruptException:
        pass

