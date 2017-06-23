#!/usr/bin/env python

import rosbag
import os

bag = rosbag.Bag('/media/prerit/Data/didi_data/ped/ped_train/ped_train.bag')
bag_prefix = '/media/prerit/Data/didi_data/ped'
time_step = 30.

count = 0
it = bag.read_messages()
last_ts = bag.get_start_time()
index = 1 

ped_tag = str(index).zfill(3)
bag_dir = os.path.join(bag_prefix, 'ped_'+ped_tag)
if not os.path.exists(bag_dir):
    os.makedirs(bag_dir)
wbag = rosbag.Bag(os.path.join(bag_prefix, 'ped_'+ped_tag,  ped_tag+'.bag'), 'w')
for msg in it:
    ts = msg.message.header.stamp if hasattr(msg.message, 'header') else msg.timestamp
    ts = ts.to_sec()
    if ts - last_ts > time_step:
        wbag.close()
        print 'Bag # '+str(index)+' written.'
        index += 1
        ped_tag = str(index).zfill(3)
        bag_dir = os.path.join(bag_prefix, 'ped_'+ped_tag)
        if not os.path.exists(bag_dir):
            os.makedirs(bag_dir)
        wbag = rosbag.Bag(os.path.join(bag_prefix, 'ped_'+ped_tag,  ped_tag+'.bag'), 'w')

        last_ts = ts
    wbag.write(msg.topic, msg.message,
        msg.message.header.stamp if hasattr(msg.message, 'header') else msg.timestamp)

wbag.close()
