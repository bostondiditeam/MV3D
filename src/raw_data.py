from typing import List,Tuple
import config
from config import cfg
import os
import numpy as np
import glob
import cv2
# from kitti_data import pykitti
from kitti_data.pykitti.tracklet import parseXML, TRUNC_IN_IMAGE, TRUNC_TRUNCATED

import math
from config import cfg
from data import is_evaluation_dataset


class RawData(object):
    def __init__(self):
        pass

    def get_synced_nframe(self, dir_tag:str) -> int:
        name = dir_tag.split('/')
        path = os.path.join(cfg.RAW_DATA_SETS_DIR, name[0], name[1], 'image_02', 'data', '*')
        return len(glob.glob(path))


class Image(RawData):

    def __init__(self):
        RawData.__init__(self)
        self.files_path_mapping= self.get_paths_mapping()


    def load(self, frame_tag:str)-> np.ndarray:
        return cv2.imread(self.files_path_mapping[frame_tag])

    def get_tags(self)-> [str]:
        tags = [tag for tag in self.files_path_mapping]
        tags.sort()
        return tags



    def get_paths_mapping(self):
        raw_dir = cfg.RAW_DATA_SETS_DIR
        mapping={}

        # foreach dir1
        for dir1 in  glob.glob(os.path.join(raw_dir, '*')):
            name1 = os.path.basename(dir1)

            # foreach dir2
            for dir2 in glob.glob(os.path.join(dir1, '*')):

                name2 = os.path.basename(dir2)

                # foreach files in dir2
                files_path = glob.glob(os.path.join(dir2,'image_02','data', '*'))
                files_path.sort()
                for i,file_path in enumerate(files_path):
                    key = '%s/%s/%05d' % (name1,name2,i)
                    mapping[key] =file_path

        return mapping







class Tracklet(RawData):

    def __init__(self):
        RawData.__init__(self)
        self.frames_object=self.get_frames_objects()


    def load(self, frame_tag:str):
        objs = self.frames_object[frame_tag]

        return objs

    def get_tags(self)-> [str]:
        tags = [tag for tag in self.frames_object]
        tags.sort()
        return tags

    def frame_tag_to_path(self):
        pass

    def get_frames_objects(self):
        raw_dir = cfg.RAW_DATA_SETS_DIR
        frames_objects={}

        # foreach dir1
        for dir1 in  glob.glob(os.path.join(raw_dir, '*')):
            name1 = os.path.basename(dir1)

            # foreach dir2
            for dir2 in glob.glob(os.path.join(dir1, '*')):

                name2 = os.path.basename(dir2)

                dir_tag = '%s/%s' % (name1, name2)
                nframe = self.get_synced_nframe(dir_tag)

                # read one tracklet
                tracklet_file = os.path.join(dir2, 'tracklet_labels.xml')
                if os.path.isfile(tracklet_file)==False: continue
                one_frame_objects = read_objects(tracklet_file, range(nframe))

                for i,objects  in enumerate(one_frame_objects):
                    frame_tag = '%s/%05d' % (dir_tag, i)
                    frames_objects[frame_tag] =objects

        return frames_objects


class Lidar(RawData):
    def __init__(self):
        RawData.__init__(self)
        self.files_path_mapping = self.get_paths_mapping()

    def load(self, frame_tag: str) -> np.dtype:
        lidar =np.fromfile(self.files_path_mapping[frame_tag], np.float32)
        return lidar.reshape((-1, 4))

    def get_tags(self) -> [str]:
        tags = [tag for tag in self.files_path_mapping]
        tags.sort()
        return tags

    def get_paths_mapping(self):
        raw_dir = cfg.RAW_DATA_SETS_DIR
        mapping = {}

        # foreach dir1
        for dir1 in glob.glob(os.path.join(raw_dir, '*')):
            name1 = os.path.basename(dir1)

            # foreach dir2
            for dir2 in glob.glob(os.path.join(dir1, '*')):
                name2 = os.path.basename(dir2)

                # foreach files in dir2
                files_path = glob.glob(os.path.join(dir2, 'velodyne_points', 'data', '*'))
                files_path.sort()
                for i, file_path in enumerate(files_path):
                    key = '%s/%s/%05d' % (name1, name2, i)
                    mapping[key] = file_path

        return mapping



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


            o = type('', (), {})()
            o.type = tracklet.objectType
            o.tracklet_id = n

            o.translation=translation
            o.rotation=rotation
            o.size=tracklet.size

            objects[frames_index.index(i+start_frame)].append(o)

    return objects


if __name__ == '__main__':
    import data
    import net.utility.draw as draw
    from sklearn.utils import shuffle

    preprocess = data.Preprocess()

    raw_img = Image()
    raw_tracklet = Tracklet()
    raw_lidar = Lidar()

    # tags = shuffle(raw_tracklet.get_tags())
    tags = raw_tracklet.get_tags()


    os.makedirs(os.path.join(config.cfg.LOG_DIR,'test','rgb') ,exist_ok=True)
    os.makedirs(os.path.join(config.cfg.LOG_DIR, 'test','top'), exist_ok=True)
    for one_frame_tag in tags:

        # load
        objs = raw_tracklet.load(one_frame_tag)
        rgb = raw_img.load(one_frame_tag)
        lidar = raw_lidar.load(one_frame_tag)

        # preprocess
        rgb = preprocess.rgb(rgb)
        top = preprocess.lidar_to_top(lidar)
        boxes3d = [preprocess.bbox3d(obj) for obj in objs]
        labels = [preprocess.label(obj) for obj in objs]


        # dump rgb
        img = draw.draw_box3d_on_camera(rgb, boxes3d)
        new_size = (img.shape[1] // 3, img.shape[0] // 3)
        img = cv2.resize(img, new_size)
        path = os.path.join(config.cfg.LOG_DIR,'test','rgb', '%s.png' % one_frame_tag.replace('/','_'))
        cv2.imwrite(path, img)
        print('write %s finished' % path)

        # draw bbox on top image
        path = os.path.join(config.cfg.LOG_DIR, 'test', 'top', '%s.png' % one_frame_tag.replace('/', '_'))
        top_image = data.draw_top_image(top)
        top_image = data.draw_box3d_on_top(top_image, boxes3d, color=(0, 0, 80))
        cv2.imwrite(path, top_image)
        print('write %s finished' % path)



