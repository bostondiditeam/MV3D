#!/usr/bin/env python
import rospy, rosbag, tf
import numpy as np
import PyKDL as kd
from collections import defaultdict

CAMERA_TOPICS = ["/image_raw"]

class Frame():
    def __init__(self, trans, rotq, object_type, size):
        self.trans = trans
        self.rotq = rotq
        self.object_type = object_type
        self.size = size

def extract_bag_timestamps(bag_file, topics=CAMERA_TOPICS):
    timestamp_map, index = {},{}
    for topic in topics:
        timestamp_map[topic] = {}
        index[topic] = 0
    with rosbag.Bag(bag_file, "r") as bag:
        for topic, msg, ts in bag.read_messages(topics):
            timestamp_map[topic][msg.header.stamp.to_nsec()] = index[topic]
            index[topic] += 1
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
