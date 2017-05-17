# from net.processing.boxes3d import boxes3d_for_evaluation
# from tracklets import Tracklet_saver
# import data
import numpy
from config import *
import os
from kitti_data.io import read_objects
from net.processing.boxes3d import boxes3d_decompose
from tracklets.Tracklet_saver import Tracklet_saver
from data import obj_to_gt_boxes3d


# test if 3D bbox output from MV3D net is correctly saved in tracklet_label.xml, both in format and numerical value.
# load from tracklet.xml -> convert it into 3D bbox from data.py, then

def test_case_first_frame():
    tracklet_file = os.path.join(cfg.RAW_DATA_SETS_DIR, '2011_09_26', '2011_09_26_drive_0005_sync',
                                 'tracklet_labels.xml')
    num_frames = 809
    objects = read_objects(tracklet_file, num_frames)

    a = Tracklet_saver('./test/')

    # object is counted as frame.
    for i in range(num_frames):
        frame_no = i
        frame1 = objects[frame_no]
        coordinate_3d_1, _ = obj_to_gt_boxes3d(frame1)

        translation1, size1, rotation1 = boxes3d_decompose(coordinate_3d_1)


        size = size1
        transition = translation1
        rotation = rotation1

        # frame 1 has multiple cars
        for i in range(len(translation1)):
            a.add_tracklet(frame_no, size[i], transition[i], rotation[i])

    a.write_tracklet()

# def load_generated_xml(path):
#     objects = read_objects(path, 1)
#     # object is counted as frame.
#     frame1 = objects[0]
#     coordinate_3d_1, _ = obj_to_gt_boxes3d(frame1)
#     pass

if __name__ == '__main__':
    test_case_first_frame()
