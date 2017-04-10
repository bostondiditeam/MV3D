import model as mod
import data
from net.utility.draw import imsave ,draw_boxed3d_to_rgb
from net.processing.boxes3d import boxes3d_for_evaluation
from tracklets.Tracklet_saver import Tracklet_saver
import argparse
import os
from config import *

# Set true if you want score after export predicted tracklet xml
# set false if you just want to export tracklet xml

def pred_and_save(tracklet_pred_dir):
    # Tracklet_saver will check whether the file already exists.
    tracklet = Tracklet_saver(tracklet_pred_dir)

    rgbs, tops, fronts, gt_labels, gt_boxes3d = data.load([0])
    m3=mod.MV3D()
    m3.tracking_init(tops[0].shape,fronts[0].shape,rgbs[0].shape)

    load_indexs=[ 135, 0,  99, 23]
    for i in load_indexs:
        rgbs, tops, fronts, gt_labels, gt_boxes3d = data.load([i])
        boxes3d,probs=m3.tacking(tops[0],fronts[0],rgbs[0])

        # for debugging: save image and show image.
        # file_name='tacking_test_img_{}'.format(i)
        # img_tracking=draw_boxed3d_to_rgb(rgbs[0],boxes3d)
        # imsave(file_name,img_tracking)
        # print(file_name+' save ok'

        # save boxes3d as tracklet files.
        translation, size, rotation = boxes3d_for_evaluation(boxes3d)
        for j in range(len(translation)):
            tracklet.add_tracklet(i, size[j], translation[j], rotation[j])

    tracklet.write_tracklet()
    print("tracklet file named tracklet_labels.xml is written successfully.")
    return tracklet.path


from tracklets.evaluate_tracklets import tracklet_score
if_score = True

if __name__ == '__main__':
    tracklet_pred_dir = os.path.join(cfg.DATA_SETS_DIR, 'predicted')

    # generate tracklet file
    pred_file = pred_and_save(tracklet_pred_dir)

    if(if_score):
        # compare newly generated tracklet_label_pred.xml with tracklet_labels_gt.xml. Change the path accordingly to
        #  fits you needs.
        gt_tracklet_file = os.path.join(cfg.DATA_SETS_DIR, '2011_09_26', 'tracklet_labels.xml')
        tracklet_score(pred_file=pred_file, gt_file=gt_tracklet_file, output_dir=tracklet_pred_dir)
        print("scores are save under {%s} directory.".format(tracklet_pred_dir))

    print("Completed")
