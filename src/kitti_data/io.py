# from kitti_data import pykitti
from kitti_data.pykitti.tracklet import parseXML, TRUNC_IN_IMAGE, TRUNC_TRUNCATED

import numpy as np
import math
from config import cfg

def read_objects(tracklet_file, frames_index):
    objects = []  #grouped by frames
    # frames_index = range(15)
    for n in frames_index: objects.append([])

    # read tracklets from file
    tracklets = parseXML(tracklet_file)
    num = len(tracklets)    #number of obs

    for n in range(num):
        tracklet = tracklets[n]

        # this part is inspired by kitti object development kit matlab code: computeBox3D
        h,w,l = tracklet.size
        if cfg.DATA_SETS_TYPE == 'didi2' or cfg.DATA_SETS_TYPE == 'didi' or cfg.DATA_SETS_TYPE == 'test':
            h, w = h*1.1, l
            trackletBox = np.array([
                [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2], \
                [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], \
                [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]])
        elif cfg.DATA_SETS_TYPE == 'kitti':
            trackletBox = np.array([  # in velodyne coordinates around zero point and without orientation yet\
                [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2], \
                [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], \
                [0.0, 0.0, 0.0, 0.0, h, h, h, h]])
        else:
            raise ValueError('unexpected type in cfg.DATA_SETS_TYPE :{}!'.format(cfg.DATA_SETS_TYPE))

        # loop over all data in tracklet
        start_frame  = tracklet.firstFrame
        end_frame=tracklet.firstFrame+tracklet.nFrames

        object_in_frames_index = [i for i in frames_index if i in range(start_frame, end_frame)]
        object_in_tracklet_index=[i-start_frame for i in object_in_frames_index]

        for i in object_in_tracklet_index:
            translation = tracklet.trans[i]
            rotation = tracklet.rots[i]
            state = tracklet.states[i]
            occlusion = tracklet.occs[i]
            truncation = tracklet.truncs[i]


            if cfg.DATA_SETS_TYPE == 'kitti':
                # print('truncation filter disable')
                # determine if object is in the image; otherwise continue
                if truncation not in (TRUNC_IN_IMAGE, TRUNC_TRUNCATED):
                   continue
                # pass
            elif cfg.DATA_SETS_TYPE == 'didi2':
                # todo : 'truncation filter disable'
                pass
            elif cfg.DATA_SETS_TYPE == 'didi':
                # todo : 'truncation filter disable'
                pass
            elif cfg.DATA_SETS_TYPE == 'test':
                pass
            else:
                raise ValueError('unexpected type in cfg.DATA_SETS_TYPE :{}!'.format(cfg.DATA_SETS_TYPE))

            # re-create 3D bounding box in velodyne coordinate system
            yaw = rotation[2]   # other rotations are 0 in all xml files I checked
            #assert np.abs(rotation[:2]).sum() == 0, 'object rotations other than yaw given!'
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

            if o.type == 'Van' or o.type == 'Truck' or o.type == 'Car' or o.type == 'Tram':  # todo : only  support 'Van'
                o.translation=translation
                o.rotation=rotation
                o.size=tracklet.size
            else:
                continue

            objects[frames_index.index(i+start_frame)].append(o)

    return objects