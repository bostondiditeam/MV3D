#! /usr/bin/python
""" Udacity Self-Driving Car Challenge Bag Processing
"""

from __future__ import print_function
# import os
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
from tracklets.generate_tracklet import *
import os
import sys
from config import cfg
# from random import random
import random
random.seed()
import numpy as np
from utils.tracklet_tools import read_objects

class Tracklet_saver():
    def __init__(self, dir_path, tracklet_name, exist_ok=False):
        # check if the tracklet file already exist, if yes, exit and print error message.
        file_path = os.path.join(dir_path, tracklet_name + '.xml')
        if exist_ok==False and os.path.isfile(file_path):
            sys.stderr.write("Error: The tracklet file %s already exists, change file name before prediction.\n" % file_path)
            exit(-1)
        else:
            self.path = file_path
        self.collection = TrackletCollection()

    def add_tracklet_pose(self, obs_tracklet, translation, rotation):
        keys = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']
        values = [translation[0], translation[1], translation[2], rotation[0], rotation[1], rotation[2]]
        pose = {k: v for k, v in zip(keys, values)}
        obs_tracklet.poses = [pose]
        pass

    # for add new tracklets
    # size is [h, w, l]
    def add_tracklet(self, first_frame_nb, size, transition, rotation):
        if cfg.OBJ_TYPE == 'car':
            obs_tracklet = Tracklet(object_type='Car', l=size[2], w=size[1], h=size[0], first_frame=first_frame_nb)
        elif cfg.OBJ_TYPE == 'ped':
            obs_tracklet = Tracklet(object_type='Pedestrian', l=size[2], w=size[1], h=size[0], first_frame=first_frame_nb)
        self.add_tracklet_pose(obs_tracklet, transition, rotation)
        self.collection.tracklets.append(obs_tracklet)
        # if 0<transition[1]<8:
        #     self.add_tracklet_pose(obs_tracklet, transition, rotation)
        #     self.collection.tracklets.append(obs_tracklet)

    def write_tracklet(self):
        self.collection.write_xml(self.path)


if __name__ == '__main__':
    #a test case
    os.makedirs('./test_output/', exist_ok=True)
    a = Tracklet_saver('./test_output/', 'ford05.xml')
    # The size is for obstacle car, the order for size is [height, width, length]
    rate = 1.2
    size = [1.466, 1.824, 4.359]
    for i in range(3):
        size[i] = size[i] * rate


    # for tx, ty, tz for different poses.
    transition = [0,3,0]
    # for rx, ry, rz for different poses.
    rotation = [0, 0, 0]
    # which frames you want the above posed in. Like the belowing example, I want to write size, transition and
    # rotation defined above to be in frame 324 to frame 647, then I define it in the following way.
    # for i in range(324,647):
    #     a.add_tracklet(i, size, transition, rotation)

    objects = read_objects('./ford05.xml')


    # list here:

    for frame in objects: # (i, size, transition, rotation)
        for item in frame:
            after_size = [0,0,0]
            random_list = [random.random()*0.12, random.random()*0.16, random.random()*0.17]
            after_size = [0, 0, 0]
            for i, s in enumerate(item[3]):
                after_size[i] = size[i] + random_list[i]
            a.add_tracklet(item[0], after_size, item[1], item[2])


    a.write_tracklet()
