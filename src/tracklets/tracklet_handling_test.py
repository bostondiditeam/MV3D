# from net.processing.boxes3d import boxes3d_for_evaluation
# from tracklets import Tracklet_saver
# import data
import numpy
from config import *
import os
from kitti_data.io import read_objects
from net.processing.boxes3d import boxes3d_for_evaluation
from tracklets import Tracklet_saver


# test if 3D bbox output from MV3D net is correctly saved in tracklet_label.xml, both in format and numerical value.
# load from tracklet.xml -> convert it into 3D bbox from data.py, then

if __name__ == '__main__':
    tracklet_file = os.path.join(cfg.DATA_SETS_DIR, '2011_09_26/tracklet_labels.xml')

    num_frames = 154
    objects = read_objects(tracklet_file, num_frames)
    # object is counted as frame.
    frame1 = objects[0]
    obj1 = frame1[0]
    coordinate_3d_1 = obj1.box
    # rotation_1 = obj1.rotation

    obj2 = frame1[1]
    coordinate_3d_2 = obj2.box
    # rotation_2 = obj2.rotation

    coordinate_3d_1 = np.array(coordinate_3d_1)
    coordinate_3d_2 = np.array(coordinate_3d_2)
    translation1, size1, rotation1 = boxes3d_for_evaluation(coordinate_3d_1)
    translation2, size2, rotation2 = boxes3d_for_evaluation(coordinate_3d_1)

    a = Tracklet_saver('./test/')
    size = size1
    transition = translation1
    rotation = rotation1
    a.add_tracklet(0, size, transition, rotation)

    size = size2
    transition = translation2
    rotation = rotation2
    a.add_tracklet(0, size, transition, rotation)
    a.write_tracklet()


    # load tracklets from

    # # test boxes3d_for_evaluation
    # gt_boxes3d=np.load('gt_boxes3d_135.npy')
    # translation, size, rotation =boxes3d_for_evaluation(gt_boxes3d[0])
    # print(translation,size,rotation)