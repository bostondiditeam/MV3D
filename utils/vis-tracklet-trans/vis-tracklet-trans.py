import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from parse_tracklet import *

def generate_obstacles(tracklets, override_size=None):
    for tracklet_idx, tracklet in enumerate(tracklets):
        frame_idx = tracklet.first_frame
        for trans, rot in zip(tracklet.trans, tracklet.rots):
            # obstacle = Obs(
            #     tracklet_idx,
            #     tracklet.object_type,
            #     override_size if override_size is not None else tracklet.size,
            #     trans,
            #     rot)
            # yield frame_idx, obstacle
            frame_idx += 1

def main():
    print '----------------------------------------------------------\n'
    print 'Plot translation coordinates in tracklet file.\n'
    if(len(sys.argv)<=1):
        print 'Usage: python vis-tracklet-trans.py input-tracklet-file [output file]'
        print 'Number of arguments:', len(sys.argv), 'arguments.'
        print 'Argument List:', str(sys.argv)
        exit(-1)

    infile = sys.argv[1]
    if not os.path.exists(infile):
        sys.stderr.write('Error: Input file %s not found.\n' % infile)
        exit(-1)

    dir_path = os.path.dirname(os.path.realpath(__file__))

    ## TODO: Extract vectors to output file for Kalman Filter
    outfile = ''
    if(len(sys.argv)>2):
        outfile = sys.argv[2]
    if not os.path.exists(outfile):
        outfile = os.path.join(dir_path,"output.txt")
        print 'Warning: Output file not found, using', outfile

    print '\nInput : ', infile, '\nOutput: ', outfile , '\n'

    pred_tracklets = parse_xml(infile)
    if not pred_tracklets:
        sys.stderr.write('Error: No Tracklets parsed for predictions.\n')
        exit(-1)

    num_pred_frames = 0
    for p_idx, pred_tracklet in enumerate(pred_tracklets):
        num_pred_frames = max(num_pred_frames, pred_tracklet.first_frame + pred_tracklet.num_frames)

    print 'number of frames = ', num_pred_frames

    xs = []
    ys = []
    zs = []
    for tracklet_idx, tracklet in enumerate(pred_tracklets):
        frame_idx = tracklet.first_frame
        print 'tracklet_idx = ', tracklet_idx
        print 'Frame#', frame_idx, ': '
        print "size:  ", tracklet.size
        for trans, rot in zip(tracklet.trans, tracklet.rots):
            print "trans: ", trans
            print "rot:   ", rot
            xs.append(trans[0])
            ys.append(trans[1])
            zs.append(trans[2])

    fig = plt.figure()

    # 3D version
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs)

    # 2D version
    # plt.plot(xs, ys, 'ro')

    plt.show()

if __name__ == "__main__":
    main()
