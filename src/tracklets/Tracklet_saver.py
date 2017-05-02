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


if __name__ == '__main__':
    #a test case
    os.makedirs('./test_output/',exist_ok=True)
    a = Tracklet_saver('./test_output/')
    size = [4.2418,1.4478*10,1.5748*4]
    transition = [0,3,0]
    rotation = [0, 0, 0]
    # for i in range(243,260):
    #     a.add_tracklet(i, size, transition, rotation)
    for i in range(324,647):
        a.add_tracklet(i, size, transition, rotation)
    a.write_tracklet()
