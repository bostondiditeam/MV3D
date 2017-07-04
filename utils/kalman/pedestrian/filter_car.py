
from __future__ import print_function, division
import rosbag, rospy

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from filterpy.kalman import KalmanFilter, MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.common import Q_discrete_white_noise
from parse_tracklet import parse_xml
# from sync import generate_frame_map
from generate_tracklet import Tracklet, TrackletCollection
# from shapely.geometry import Polygon
import pandas as pd

import random, math
from bisect import bisect_left
import numpy as np
from numba import jit
from itertools import combinations
from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.cluster.hierarchy import linkage, fcluster
import os
import pickle
from sort_car import Sort

bag_name = 'ford01'
bag_file = os.path.join('/home/stu/competition_data/didi_dataset/round2/test_car', bag_name + '.bag')
tracklet_dir = './'
tracklet_file = os.path.join(tracklet_dir, 'ori', bag_name + '.xml')
new_tracklet_file = os.path.join(tracklet_dir, 'corrected', bag_name + '.xml')


tracklets = parse_xml(tracklet_file)


bag = rosbag.Bag(bag_file)
print('Reading timestamps from bag ', bag_file)
n_stamps = bag.get_message_count(topic_filters=['/image_raw'])
timestamps, radar_timestamps = [],[]
radar_tracks = {}
for topic,msg,t in bag.read_messages(topics=['/image_raw','/radar/tracks']):
    if topic=='/image_raw':
        timestamps.append(t.to_nsec())
    else :
        radar_time = t.to_nsec()
        radar_timestamps.append(radar_time)
        radar_tracks[radar_time] = []
        for track in msg.tracks :
            if (track.status!=3) or (track.range>60) or (track.range<4):
                continue
            radar_tracks[radar_time].append([track.number, track.range, track.angle, track.rate, track.late_rate])
detections = {t:[] for t in timestamps}

if 1:
    dumps = {'radar_tracks':radar_tracks,
             'detections':detections,
             'n_stamps':n_stamps
             }
    pickle.dump(dumps,open('./dumps','wb'))
else:
    dumps= pickle.load(open('./dumps', 'rb'))
    radar_tracks,detections,n_stamps = dumps['radar_tracks'],dumps['detections'],dumps['n_stamps']


for track in tracklets:
    stamp = timestamps[track.first_frame]
    detections[stamp].append(np.concatenate((track.trans[0],[track.rots[0][2]], track.size))) # (x,y,z,yaw, h,w,l)    detections[stamp].append(np.concatenate((track.trans[0],[track.rots[0][2]], track.size))) # (x,y,z,yaw, h,w,l)



cam_df = pd.DataFrame(index=timestamps, columns=['sensor'])
cam_df['sensor'] = 'C'
radar_df = pd.DataFrame(index=radar_timestamps, columns=['sensor'])
radar_df['sensor'] = 'R'
sensor_df = pd.merge(cam_df, radar_df, left_index=True, right_index=True, on='sensor', how='outer')
sensor_df

# KalmanBoxTracker.count = 0
mot_tracker = Sort(max_age=3, min_hits=4, iou_threshold=0.3, max_time_elapsed=2)
collection = TrackletCollection()
min_detections = 7

tracklets = {}
frame_count=0
track_count = {}
for timestamp,sensor in sensor_df.iterrows():
    if sensor['sensor'] == 'R':
        print('radar detections : ', radar_tracks[timestamp])
        mot_tracker.radar_update(timestamp, radar_tracks[timestamp])
    if sensor['sensor'] == 'C':
        d = detections[timestamp]
        print('MV3D detections : ', d)
        tracks =  mot_tracker.update(np.array(d),timestamp)
        print("~~~~ tracks ~~~~~ ", tracks)
        for track in tracks :
            trk_id = int(track[5])
            if trk_id not in tracklets.keys() :
                tracklets[trk_id] = Tracklet(
                        object_type = 'Pedestrian',
                        l = track[4],
                        w = track[5],
                        h = track[6],
                        first_frame=frame_count)
                track_count[trk_id] = 0
            else :
                track_count[trk_id] += 1
            if track_count[trk_id] < min_detections :
                continue
            frame = dict(tx=track[0],
                     ty=track[1],
                     tz=track[2],
                     rx=0.,
                     ry=0.,
                     rz=track[3])
            tracklets[trk_id].poses.append(frame)
        frame_count += 1
for trk_id in tracklets.keys() :
    collection.tracklets.append(tracklets[trk_id])


collection.write_xml(new_tracklet_file)