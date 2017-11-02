#!/usr/bin/env python

import numpy as np
import rospy, rosbag
from parse_tracklet import parse_xml
from generate_tracklet import Tracklet, TrackletCollection
from sort import Sort
import os
import argparse
import pickle
import config

TYPE = 'car'

if __name__ == "__main__" :
    # parser = argparse.ArgumentParser(description="filter poses")
    # parser.add_argument('bag', type=str, nargs='?', default='', help='bag name in abs path')
    # parser.add_argument('tracklet_dir', type=str, nargs='?', default='', help='tracklet directory')
    # parser.add_argument('tracklet_name', type=str, nargs='?', default='', help='tracklet directory')
    # args = parser.parse_args()
    #
    # bag_tag = args.bag
    # bag_file = os.path.join('/hdd/data/didi_competition/didi_dataset/round2/Data/', bag_tag + '.bag')
    # tracklet_dir = args.tracklet_dir
    # tracklet_tag = args.tracklet_name
    # tracklet_file = os.path.join(tracklet_dir, tracklet_tag+'.xml')
    # new_tracklet_file = tracklet_tag+'_corrected.xml'

    bag_name = 'ford01'
    bag_file = os.path.join('/home/stu/competition_data/didi_dataset/round2/test_car',bag_name+'.bag')
    tracklet_dir = './'
    tracklet_file = os.path.join(tracklet_dir,'ori',bag_name+'.xml')
    new_tracklet_file = os.path.join(tracklet_dir,'corrected', bag_name+'.xml')


    tracklets = parse_xml(tracklet_file)
    if 0:
        bag = rosbag.Bag(bag_file)
        print('Reading timestamps from bag ', bag_file)
        n_stamps = bag.get_message_count(topic_filters=['/image_raw'])
        timestamps= [t.to_sec() for _,_,t in bag.read_messages(topics=['/image_raw'])]
    else:
        timestamps =  pickle.load(open('./timestamps'+'_'+bag_name,'rb'))
    pickle.dump(timestamps,open('./timestamps'+'_'+bag_name,'wb'))
    detections = [[] for i in range(len(timestamps))]
    for track in tracklets:
        detections[track.first_frame].append(
        np.concatenate((track.trans[0],track.size))) # (x,y,z,h,w,l)

    if config.cfg.OBJ_TYPE == 'ped':
        mot_tracker = Sort(max_age=3, min_hits=5, iou_threshold=0.1, max_time_elapsed=2)
    else:
        mot_tracker = Sort(max_age=3, min_hits=5, iou_threshold=0.01, max_time_elapsed=2)
    collection = TrackletCollection()


    tracklets = {}
    frame_count=0
    fix_size = False
    for t,d in zip(timestamps,detections):
        #print("detections : ", d)
        tracks = mot_tracker.update(np.array(d), t)
        print(tracks)
        for track in tracks :
            trk_id = int(track[5])
            if trk_id not in tracklets.keys() :
                if fix_size:
                    tracklets[trk_id] = Tracklet(
                        object_type='Pedestrian',
                        l=0.8*1.2,
                        w=0.8*1.2,
                        h=1.7*1.2,
                        first_frame=frame_count)
                else:
                    tracklets[trk_id] = Tracklet(
                            object_type = 'Pedestrian',
                            l = 2*track[3],
                            w = 2*track[3],
                            h = track[4],
                            first_frame=frame_count)
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


