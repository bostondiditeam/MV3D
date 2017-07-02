#!/usr/bin/env python

from __future__ import print_function
from generate_tracklet import *
import parse_tracklet
import sys, os
import PyKDL as kd
import numpy as np
import tf
import yaml
import argparse



class Tracklet_offset :

    def __init__(self, tracklet_file):
        self.offset = [0,0,0]
        self.rotation_offset = [0,0,0,1]
        self.orient = (0,0,0,1)
        self.tracklets = parse_tracklet.parse_xml(tracklet_file)
        self.collection = TrackletCollection()

    def set_offset(self, offset_file) :
        with open(offset_file,'r') as f :
            meta = yaml.load(f)
            self.offset = meta['translation']  
            self.rotation_offset = meta['rotation']    
            self.orient = tuple(meta['orientation'])        

    def apply_offset(self) :
        for t in self.tracklets:
            obs_tracklet = Tracklet(
                            object_type = t.object_type, 
                            l = t.size[2], 
                            w = t.size[1], 
                            h = t.size[0], 
                            first_frame=0)
            for i in range(t.num_frames):
                obs_centroid = np.array(t.trans[i]) + np.array(self.offset)
                R = tf.transformations.quaternion_matrix(self.rotation_offset)
                rotated_centroid = R.dot(list(obs_centroid)+[1])
                obs_centroid = rotated_centroid[:3]
                rot = tf.transformations.euler_from_quaternion(self.orient) #t.rots[i]
                frame = dict(tx=obs_centroid[0],
                             ty=obs_centroid[1],
                             tz=obs_centroid[2],
                             rx=rot[0],
                             ry=rot[1],
                             rz=rot[2])
                obs_tracklet.poses.append(frame)
            self.collection.tracklets.append(obs_tracklet)

    def write_tracklet(self, out_file):
        self.collection.write_xml(out_file)


if __name__ == "__main__" :
    offset_file = None 
    assert len(sys.argv)==4, 'usage: \n{} <input_tracklet> <output_tracklet> <offset_file>'.format(sys.argv[0])
    input_tracklet = sys.argv[1] 
    output_tracklet = sys.argv[2]
    offset_file = sys.argv[3]  

    assert os.path.isfile(input_tracklet), 'Tracklet file %s does not exist' % input_tracklet
    assert os.path.isfile(offset_file), 'Meta file %s does not exist' % offset_file
    new_tracklet = Tracklet_offset(input_tracklet)
    new_tracklet.set_offset(offset_file)
    new_tracklet.apply_offset()
    new_tracklet.write_tracklet(output_tracklet)


