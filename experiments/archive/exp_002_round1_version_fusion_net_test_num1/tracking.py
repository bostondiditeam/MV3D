import mv3d
from data import draw_top_image,draw_box3d_on_top
from net.utility.draw import imsave ,draw_box3d_on_camera, draw_box3d_on_camera
from net.processing.boxes3d import boxes3d_decompose
from tracklets.Tracklet_saver import Tracklet_saver
import argparse
import os
import config
from config import cfg
import time
import utils.batch_loading as ub
import cv2
import numpy as np
import net.utility.draw as draw
import skvideo.io
from utils.timer import timer
from time import localtime, strftime

log_subdir=os.path.join('tracking',strftime("%Y_%m_%d_%H_%M_%S", localtime()))
log_dir = os.path.join(cfg.LOG_DIR, log_subdir)

fast_test = False
def pred_and_save(tracklet_pred_dir, dataset, generate_video=False,
                  frame_offset=16, log_tag=None, weights_tag=None):
    # Tracklet_saver will check whether the file already exists.
    tracklet = Tracklet_saver(tracklet_pred_dir)
    os.makedirs (os.path.join(log_dir,'image'),exist_ok=True)


    top_shape, front_shape, rgb_shape=dataset.get_shape()
    predict=mv3d.Predictor(top_shape, front_shape, rgb_shape, log_tag=log_tag, weights_tag=weights_tag)

    if generate_video:
        vid_in = skvideo.io.FFmpegWriter(os.path.join(log_dir,'output.mp4'))

    # timer
    timer_step = 100
    if cfg.TRACKING_TIMER:
        time_it = timer()

    frame_num = 0
    for i in range(dataset.size if fast_test==False else frame_offset + 1):

        rgb, top, front, _, _,_= dataset.load(1)

        frame_num = i - frame_offset
        if frame_num < 0:
            continue

        boxes3d,probs=predict(top, front, rgb)
        predict.dump_log(log_subdir=log_subdir,n_frame=i)

        # time timer_step iterations. Turn it on/off in config.py
        if cfg.TRACKING_TIMER and i%timer_step ==0 and i!=0:
            predict.track_log.write('It takes %0.2f secs for inferring %d frames. \n' % \
                                   (time_it.time_diff_per_n_loops(), timer_step))

        # for debugging: save image and show image.
        top_image = draw_top_image(top[0])
        rgb_image = rgb[0]


        if len(boxes3d)!=0:
            top_image = draw_box3d_on_top(top_image, boxes3d[:,:,:], color=(80, 80, 0), thickness=3)
            rgb_image = draw_box3d_on_camera(rgb_image, boxes3d[:, :, :], color=(0, 0, 80), thickness=3)
            translation, size, rotation = boxes3d_decompose(boxes3d[:, :, :])
            #todo: remove it after gtbox is ok
            size[:,1:3] = size[:,1:3]/cfg.TRACKLET_GTBOX_LENGTH_SCALE

            for j in range(len(translation)):
                if 0<translation[j,1]<8:
                    tracklet.add_tracklet(frame_num, size[j], translation[j], rotation[j])
        resize_scale=top_image.shape[0]/rgb_image.shape[0]
        rgb_image = cv2.resize(rgb_image,(int(rgb_image.shape[1]*resize_scale), top_image.shape[0]))
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        new_image = np.concatenate((top_image, rgb_image), axis = 1)
        cv2.imwrite(os.path.join(log_dir,'image','%5d_image.jpg'%i), new_image)

        if generate_video:
            vid_in.writeFrame(new_image)
            vid_in.close()

    tracklet.write_tracklet()
    predict.dump_weigths(os.path.join(log_dir, 'pretrained_model'))

    if cfg.TRACKING_TIMER:
        predict.log_msg.write('It takes %0.2f secs for inferring the whole test dataset. \n' % \
                              (time_it.total_time()))

    print("tracklet file named tracklet_labels.xml is written successfully.")
    return tracklet.path


from tracklets.evaluate_tracklets import tracklet_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tracking')
    parser.add_argument('-n', '--tag', type=str, nargs='?', default='unknown_tag',
                        help='set log tag')
    parser.add_argument('-w', '--weights', type=str, nargs='?', default='',
                        help='set weigths tag name')
    parser.add_argument('-t', '--fast_test', type=bool, nargs='?', default=False,
                        help='set fast_test model')
    args = parser.parse_args()

    print('\n\n{}\n\n'.format(args))
    tag = args.tag
    if tag == 'unknow_tag':
        tag = input('Enter log tag : ')
        print('\nSet log tag :"%s" ok !!\n' %tag)
    weights_tag = args.weights if args.weights!= '' else None

    fast_test = args.fast_test

    tracklet_pred_dir = os.path.join(log_dir, 'tracklet')
    os.makedirs(tracklet_pred_dir, exist_ok=True)

    # Set true if you want score after export predicted tracklet xml
    # set false if you just want to export tracklet xml
    frame_offset=0
    dataset_loader=None

    if config.cfg.DATA_SETS_TYPE == 'didi':
        if_score = True
        if 1:
            dataset = {'Round1Test': ['19_f2']}

        else:
            car='3'
            data='7'
            dataset = {
                car: [data]
            }

            # compare newly generated tracklet_label_pred.xml with tracklet_labels_gt.xml. Change the path accordingly to
            #  fits you needs.
            gt_tracklet_file = os.path.join(cfg.RAW_DATA_SETS_DIR, car, data, 'tracklet_labels.xml')

    elif config.cfg.DATA_SETS_TYPE == 'kitti':
        if_score = False
        car = '2011_09_26'
        data = '0013'
        dataset = {
            car: [data]
        }

        # compare newly generated tracklet_label_pred.xml with tracklet_labels_gt.xml. Change the path accordingly to
        #  fits you needs.
        gt_tracklet_file = os.path.join(cfg.RAW_DATA_SETS_DIR, car, car + '_drive_' + data + '_sync',
                                        'tracklet_labels.xml')


    dataset_loader = ub.batch_loading(cfg.PREPROCESSED_DATA_SETS_DIR, dataset, is_testset=True)

    print("tracklet_pred_dir: " + tracklet_pred_dir)
    pred_file = pred_and_save(tracklet_pred_dir, dataset_loader,
                              frame_offset=0, log_tag=tag, weights_tag=weights_tag)

    if if_score:
        tracklet_score(pred_file=pred_file, gt_file=gt_tracklet_file, output_dir=tracklet_pred_dir)
        print("scores are save under {} directory.".format(tracklet_pred_dir))

    print("Completed")