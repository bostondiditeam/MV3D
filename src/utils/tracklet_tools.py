from raw_data import read_objects
from kitti_data.pykitti.tracklet import parseXML, TRUNC_IN_IMAGE, TRUNC_TRUNCATED
from config import cfg

def read_objects(tracklet_file):
    objects = []  #grouped by frames

    # read tracklets from file
    tracklets = parseXML(tracklet_file)
    num = len(tracklets)    #number of obs
    objects = [[] for n in range(num)]

    for n in range(num):
        tracklet = tracklets[n]

        # this part is inspired by kitti object development kit matlab code: computeBox3D
        h,w,l = tracklet.size

        # loop over all data in tracklet
        start_frame  = tracklet.firstFrame
        end_frame=tracklet.firstFrame+tracklet.nFrames

        object_in_frames_index = range(start_frame, end_frame)
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


            o = type('', (), {})()
            o.type = tracklet.objectType
            o.tracklet_id = n

            o.translation=translation
            o.rotation=rotation
            o.size=tracklet.size
            o.start_frame = start_frame
            objects[n].append((start_frame, translation, rotation, tracklet.size ))


    return objects


if __name__ == '__main__':
    objects = read_objects('../../tracklets/20170721_2039/ford01.xml')
    for one_frame_objs in objects:
        for obj in one_frame_objs:
            print(obj)
