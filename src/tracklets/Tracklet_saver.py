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

class Tracklet_saver():
    def __init__(self, dir_path):
        # check if the tracklet file already exist, if yes, exit and print error message.
        file_path = os.path.join(dir_path, 'tracklet_labels_pred.xml')
        if os.path.isfile(file_path):
            sys.stderr.write("Error: The tracklet file %s already exists, change file name before prediction.\n" % file_path)
            exit(-1)
        else:
            self.path = file_path
        self.collection = TrackletCollection()

    def add_tracklet_pose(self, obs_tracklet, translation, rotation):
        keys = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']
        values = [translation[0], translation[1], translation[2], rotation[0], rotation[1], rotation[2]]
        pose = {k: v for k, v in zip(keys, values)}
        # print("what is its type: ", type(first_pose['tx']))
        obs_tracklet.poses = [pose]
        pass

    # for add new tracklets
    # size is [h, w, l]
    def add_tracklet(self, first_frame_nb, size, transition, rotation):
        obs_tracklet = Tracklet(object_type='Car', l=size[2], w=size[1], h=size[0], first_frame=first_frame_nb)
        self.add_tracklet_pose(obs_tracklet, transition, rotation)
        self.collection.tracklets.append(obs_tracklet)

    def write_tracklet(self):
        self.collection.write_xml(self.path)


# a test case
# a = Tracklet_saver('./test/')
# size = [1,2,3]
# transition = [10,20,30]
# rotation = [0.1, 0.2, 0.3]
# a.add_tracklet(100, size, transition, rotation)
# a.add_tracklet(100, size, transition, rotation)
# a.write_tracklet()
