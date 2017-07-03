#!/usr/bin/env python

import numpy as np
import rospy, rosbag
from parse_tracklet import parse_xml
from generate_tracklet import Tracklet, TrackletCollection
from sort import Sort


tag="004"
bag_file = '/media/prerit/Data/didi_data/ped/ped_test_'+tag+'/'+tag+'.bag'
tracklet_file = '/media/prerit/Data/didi_data/ped/ped_test_'+tag+'/'+tag+'.xml'
new_tracklet_file = '/media/prerit/Data/didi_data/ped/ped_test_'+tag+'/'+tag+'_corrected.xml' 


tracklets = parse_xml(tracklet_file)
bag = rosbag.Bag(bag_file)
print('Reading timestamps from bag ', bag_file)
n_stamps = bag.get_message_count(topic_filters=['/image_raw'])
timestamps= [t.to_sec() for _,_,t in bag.read_messages(topics=['/image_raw'])]
detections = [[] for i in range(n_stamps)]
for track in tracklets:
    detections[track.first_frame].append(
    np.concatenate((track.trans[0],track.size))) # (x,y,z,h,w,l)


mot_tracker = Sort(max_age=3, min_hits=6, iou_threshold=0.1, max_time_elapsed=2)
min_detections = 7

collection = TrackletCollection()


tracklets = {}
frame_count=0
track_count = {}
for t,d in zip(timestamps,detections):
    #print("detections : ", d)
    tracks =  mot_tracker.update(np.array(d),t)
    print(tracks)
    for track in tracks :    
        trk_id = int(track[5]) 
        if trk_id not in tracklets.keys() :
            tracklets[trk_id] = Tracklet(
                    object_type = 'Pedestrian', 
                    l = 2*track[3], 
                    w = 2*track[3], 
                    h = track[4], 
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
                 rz=0.)
        tracklets[trk_id].poses.append(frame)
    frame_count += 1
for trk_id in tracklets.keys() :
    collection.tracklets.append(tracklets[trk_id])
collection.write_xml(new_tracklet_file)


