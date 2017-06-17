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
import pandas as pd



class Tracklet_offset :

    def __init__(self, tracklet_file):
        self.offset = [0,0,0]
        self.rotation_offset = [0,0,0,1]
        self.orient_offset = (0,0,0,1)
        self.tracklets = parse_tracklet.parse_xml(tracklet_file)
        self.collection = TrackletCollection()
        self.ground_corr = None

    def set_offset(self, offset_file) :
        with open(offset_file,'r') as f :
            meta = yaml.load(f)
            self.offset = meta['translation']  
            self.rotation_offset = meta['rotation']    
            self.orient_offset = tuple(meta['orientation'])        

    def import_ground_corr(self, ground_corr_file) :
        self.ground_corr = pd.read_csv(ground_corr_file)

    def apply_corr(self) :
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
                q0 = kd.Rotation.RPY(*t.rots[i]).GetQuaternion()
                q = tf.transformations.quaternion_multiply(self.orient_offset, q0)
                rot = tf.transformations.euler_from_quaternion(q)
                #rot = t.rots[i]
                if self.ground_corr is not None:
                    obs_centroid[2] = self.ground_corr.iloc[i]['z_min'] + 0.5*t.size[0]
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
    parser = argparse.ArgumentParser(description="visulaize tracklet")
    parser.add_argument('input', type=str, nargs='?', default='', help='input tracklet')
    parser.add_argument('output', type=str, nargs='?', default='', help='output tracklet')
    parser.add_argument('--offset', type=str, help='yaml file with offset corrections')
    parser.add_argument('--ground', type=str, help='Ground height corrections')
    args = parser.parse_args()
    
    input_tracklet = args.input 
    output_tracklet = args.output
    offset_file = None
    ground_corr = None  

    assert os.path.isfile(input_tracklet), 'Tracklet file %s does not exist' % input_tracklet
    new_tracklet = Tracklet_offset(input_tracklet)
    
    if args.offset :
        offset_file = args.offset
        assert os.path.isfile(offset_file), 'Offset file %s does not exist' % offset_file
        new_tracklet.set_offset(offset_file)
    if args.ground :
        ground_corr = args.ground
        assert os.path.isfile(ground_corr), 'Offset file %s does not exist' % ground_corr
        new_tracklet.import_ground_corr(ground_corr)
        
    new_tracklet.apply_corr()
    new_tracklet.write_tracklet(output_tracklet)


