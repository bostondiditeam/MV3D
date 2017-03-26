from kitti_data import pykitti
from kitti_data.pykitti.tracklet import parseXML, TRUNC_IN_IMAGE, TRUNC_TRUNCATED

import numpy as np
import math

def read_objects(tracklet_file, num_frames):

    objects = []  #grouped by frames
    for n in range(num_frames): objects.append([])

    # read tracklets from file
    tracklets = parseXML(tracklet_file)
    num = len(tracklets)

    for n in range(num):
        tracklet = tracklets[n]

        # this part is inspired by kitti object development kit matlab code: computeBox3D
        h,w,l = tracklet.size
        trackletBox = np.array([ # in velodyne coordinates around zero point and without orientation yet\
            [-l/2, -l/2,  l/2, l/2, -l/2, -l/2,  l/2, l/2], \
            [ w/2, -w/2, -w/2, w/2,  w/2, -w/2, -w/2, w/2], \
            [ 0.0,  0.0,  0.0, 0.0,    h,     h,   h,   h]])

        # loop over all data in tracklet
        t  = tracklet.firstFrame
        for translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absoluteFrameNumber in tracklet:

            # determine if object is in the image; otherwise continue
            if truncation not in (TRUNC_IN_IMAGE, TRUNC_TRUNCATED):
               continue

            # re-create 3D bounding box in velodyne coordinate system
            yaw = rotation[2]   # other rotations are 0 in all xml files I checked
            assert np.abs(rotation[:2]).sum() == 0, 'object rotations other than yaw given!'
            rotMat = np.array([\
              [np.cos(yaw), -np.sin(yaw), 0.0], \
              [np.sin(yaw),  np.cos(yaw), 0.0], \
              [        0.0,          0.0, 1.0]])
            cornerPosInVelo = np.dot(rotMat, trackletBox) + np.tile(translation, (8,1)).T

            # calc yaw as seen from the camera (i.e. 0 degree = facing away from cam), as opposed to
            #   car-centered yaw (i.e. 0 degree = same orientation as car).
            #   makes quite a difference for objects in periphery!
            # Result is in [0, 2pi]
            x, y, z = translation
            yawVisual = ( yaw - np.arctan2(y, x) ) % (2*math.pi)

            o = type('', (), {})()
            o.box = cornerPosInVelo.transpose()
            o.type = tracklet.objectType
            o.tracklet_id = n
            objects[t].append(o)
            t = t+1

    return objects