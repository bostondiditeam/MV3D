import os
os.environ["DISPLAY"] = ":0"

# std libs
import glob


# num libs
import math
import random
import numpy as np

import cv2
import config
import data
import net.utility.draw as draw



if __name__ == '__main__':

    # preprocessed_dir=config.cfg.PREPROCESSED_DATA_SETS_DIR
    preprocessed_dir=config.cfg.PREPROCESSING_DATA_SETS_DIR
    dataset='/1/15/00070.npy'
    top_view_dir      =preprocessed_dir+ '/top'+dataset
    gt_boxes3d_dir =preprocessed_dir+'/gt_boxes3d'+dataset
    top=np.load(top_view_dir)
    gt_boxes3d=np.load(gt_boxes3d_dir)

    top_img=data.draw_top_image(top)
    top_img_marked=data.draw_box3d_on_top(top_img,gt_boxes3d)
    draw.imsave('top_img_marked',top_img_marked,'debug')
    print('top_img_marked dump finished!!')



