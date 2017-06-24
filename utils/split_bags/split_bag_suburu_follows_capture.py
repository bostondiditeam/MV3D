#!/usr/bin/env python

import rosbag
import os
from shutil import copyfile


bag_file = '/media/prerit/Data/didi_data/suburu/suburu_follows_capture/suburu09.bag'
bag = rosbag.Bag(bag_file)
bag_prefix = '/media/prerit/Data/didi_data/suburu/suburu_follows_capture_'
time_step = 35

print "Reading bag. Please wait..."
it = bag.read_messages()
last_ts = bag.get_start_time()
index = 1 
print "Reading done!"

tag = str(index).zfill(3)
bag_dir = bag_prefix+tag
if not os.path.exists(bag_dir):
    os.makedirs(bag_dir)
src = os.path.join(os.path.dirname(bag_file), 'metadata.csv')
dst = os.path.join(bag_prefix+tag, 'metadata.csv')
copyfile(src, dst)
wbag = rosbag.Bag(os.path.join(bag_prefix+tag,  tag+'.bag'), 'w')
for msg in it:
    ts = msg.timestamp
    ts = ts.to_sec()
    if (ts-last_ts) > time_step:
        wbag.close()
        print 'Bag # '+str(index)+' written.'
        index += 1
        tag = str(index).zfill(3)
        bag_dir = bag_prefix+tag
        if not os.path.exists(bag_dir):
            os.makedirs(bag_dir)
        src = os.path.join(os.path.dirname(bag_file), 'metadata.csv')
        dst = os.path.join(bag_prefix+tag, 'metadata.csv')
        copyfile(src, dst)
        wbag = rosbag.Bag(os.path.join(bag_prefix+tag,  tag+'.bag'), 'w')
        last_ts = ts

    wbag.write(msg.topic, msg.message, msg.timestamp)

wbag.close()
