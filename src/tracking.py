import model as mod
import data
from net.utility.draw import imsave ,draw_boxed3d_to_rgb
from net.processing.boxes3d import boxes3d_decompose
from tracklets.Tracklet_saver import Tracklet_saver
import argparse
import os
from config import cfg
import time
import utils.batch_loading as ub
import cv2
import numpy as np
import net.utility.draw as draw
import skvideo.io
from utils.timer import timer
from time import localtime, strftime
from task import copy_weigths

log_dir = os.path.join(cfg.LOG_DIR, 'tracking',strftime("%Y_%m_%d_%H_%M_%S", localtime()))

# Set true if you want score after export predicted tracklet xml
# set false if you just want to export tracklet xml

def pred_and_save(tracklet_pred_dir, dataset, generate_video=False, frame_offset=17):
    # Tracklet_saver will check whether the file already exists.
    tracklet = Tracklet_saver(tracklet_pred_dir)
    os.makedirs (os.path.join(log_dir,'image'),exist_ok=True)


    top_shape, front_shape, rgb_shape=dataset.get_shape()
    m3=mod.MV3D()
    m3.tracking_init(top_shape,front_shape,rgb_shape)

    if generate_video:
        vid_in = skvideo.io.FFmpegWriter(os.path.join(log_dir,'output.mp4'))

    # timer
    timer_step = 100
    if cfg.TRACKING_TIMER:
        time_it = timer()

    frame_num = 0
    for i in range(dataset.size):

        rgb, top, front, _, _,_= dataset.load(1)

        frame_num = i - frame_offset
        if frame_num < 0:
            continue

        boxes3d,probs=m3.tracking(top[0],front[0],rgb[0])

        # time timer_step iterations. Turn it on/off in config.py
        if cfg.TRACKING_TIMER and i%timer_step ==0 and i!=0:
            m3.track_log.write('It takes %0.2f secs for inferring %d frames. \n' % \
                                   (time_it.time_diff_per_n_loops(), timer_step))

        # for debugging: save image and show image.
        top_image = data.draw_top_image(top[0])
        rgb_image = rgb[0]


        if len(boxes3d)!=0:
            top_image = data.draw_box3d_on_top(top_image, boxes3d[:,:,:], color=(80, 80, 0), thickness=3)
            rgb_image = draw.draw_boxed3d_to_rgb(rgb_image, boxes3d[:,:,:], color=(0, 0, 80), thickness=3)
            translation, size, rotation = boxes3d_decompose(boxes3d[:, :, :])
            for j in range(len(translation)):
                tracklet.add_tracklet(frame_num, size[j], translation[j], rotation[j])
        rgb_image = cv2.resize(rgb_image, (500, 400))
        resize_scale=top_image.shape[0]/rgb_image.shape[0]
        rgb_image = cv2.resize(rgb_image,(int(rgb_image.shape[1]*resize_scale), top_image.shape[0]))
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        new_image = np.concatenate((top_image, rgb_image), axis = 1)
        cv2.imwrite(os.path.join(log_dir,'image','%5d_image.jpg'%frame_num), new_image)

        if generate_video:
            vid_in.writeFrame(new_image)
            vid_in.close()

    tracklet.write_tracklet()

    if cfg.TRACKING_TIMER:
        m3.log.write('It takes %0.2f secs for inferring the whole test dataset. \n' % \
                       (time_it.total_time()))

    print("tracklet file named tracklet_labels.xml is written successfully.")
    return tracklet.path


from tracklets.evaluate_tracklets import tracklet_score

if __name__ == '__main__':


    tracklet_pred_dir = os.path.join(log_dir, 'tracklet')
    os.makedirs(tracklet_pred_dir,exist_ok=True)

    dataset = {
        'Round1Test': ['19_f2']
               }

    dataset_loader = ub.batch_loading(cfg.PREPROCESSED_DATA_SETS_DIR,dataset,is_testset=True)

    # generate tracklet file
    print("tracklet_pred_dir: " + tracklet_pred_dir)
    pred_file = pred_and_save(tracklet_pred_dir, dataset_loader)
    if_score = False

    if(if_score):
        # compare newly generated tracklet_label_pred.xml with tracklet_labels_gt.xml. Change the path accordingly to
        #  fits you needs.
        gt_tracklet_file = os.path.join(cfg.RAW_DATA_SETS_DIR, '2011_09_26', '2011_09_26_drive_0005_sync', 'tracklet_labels.xml')
        tracklet_score(pred_file=pred_file, gt_file=gt_tracklet_file, output_dir=tracklet_pred_dir)
        print("scores are save under {} directory.".format(tracklet_pred_dir))

    copy_weigths(os.path.join(log_dir, 'pretrained_model'))
    print("Completed")