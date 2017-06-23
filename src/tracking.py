import mv3d
from data import draw_top_image, draw_box3d_on_top
from net.utility.draw import imsave, draw_box3d_on_camera, draw_box3d_on_camera
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
from utils.batch_loading import BatchLoading2 as BatchLoading
from utils.training_validation_data_splitter import get_test_tags

log_dir = None

fast_test = False


def pred_and_save(tracklet_pred_dir, dataset,frame_offset=0, log_tag=None, weights_tag=None):
    # Tracklet_saver will check whether the file already exists.
    tracklet = Tracklet_saver(tracklet_pred_dir, exist_ok=True)

    top_shape, front_shape, rgb_shape = dataset.get_shape()
    predict = mv3d.Predictor(top_shape, front_shape, rgb_shape, log_tag=log_tag, weights_tag=weights_tag)

    # timer
    timer_step = 100
    if cfg.TRACKING_TIMER:
        time_it = timer()

    for i in range(dataset.size if fast_test == False else frame_offset + 1):

        rgb, top, front, _, _, frame_id = dataset.load()

        frame_num = i - frame_offset
        if frame_num < 0:
            continue

        # detection
        boxes3d, probs = predict(top, front, rgb)
        predict.dump_log(log_subdir=os.path.join('tracking',log_tag), n_frame=i, frame_tag=frame_id)

        # time timer_step iterations. Turn it on/off in config.py
        if cfg.TRACKING_TIMER and i % timer_step == 0 and i != 0:
            predict.track_log.write('It takes %0.2f secs for inferring %d frames. \n' % \
                                    (time_it.time_diff_per_n_loops(), timer_step))

        if len(boxes3d) != 0:
            translation, size, rotation = boxes3d_decompose(boxes3d[:, :, :])

            # add to tracklets
            for j in range(len(translation)):
                if 0 < translation[j, 1] < 8:
                    tracklet.add_tracklet(frame_num, size[j], translation[j], rotation[j])


    tracklet.write_tracklet()
    predict.save_weights(dir=os.path.join(log_dir, 'pretrained_model'))

    if cfg.TRACKING_TIMER:
        predict.log_msg.write('It takes %0.2f secs for inferring the whole test dataset. \n' % \
                              (time_it.total_time()))

    print("tracklet file named tracklet_labels.xml is written successfully.")
    return tracklet.path


def str2bool(v):
    if v.lower() in ('true'):
        return True
    elif v.lower() in ('false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


from tracklets.evaluate_tracklets import tracklet_score

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='tracking')
    parser.add_argument('-n', '--tag', type=str, nargs='?', default='unknown_tag',
                        help='set log tag')
    parser.add_argument('-w', '--weights', type=str, nargs='?', default='',
                        help='set weigths tag name')
    parser.add_argument('-t', '--fast_test', type=str2bool, nargs='?', default=False,
                        help='set fast_test model')
    parser.add_argument('-s', '--n_skip_frames', type=int, nargs='?', default=0,
                        help='set number of skip frames')
    args = parser.parse_args()

    print('\n\n{}\n\n'.format(args))
    tag = args.tag
    if tag == 'unknow_tag':
        tag = input('Enter log tag : ')
        print('\nSet log tag :"%s" ok !!\n' % tag)
    weights_tag = args.weights if args.weights != '' else None

    fast_test = args.fast_test
    n_skip_frames = args.n_skip_frames

    #log dir
    log_dir = os.path.join(config.cfg.LOG_DIR,'tracking', tag)


    tracklet_pred_dir = os.path.join(log_dir, 'tracklet')
    os.makedirs(tracklet_pred_dir, exist_ok=True)



    frame_offset = 0
    dataset_loader = None
    gt_tracklet_file = None

    # Set true if you want score after export predicted tracklet xml
    # set false if you just want to export tracklet xml
    if_score =False

    if config.cfg.DATA_SETS_TYPE == 'didi2':
        test_bags = [
            'test_car/ford01',
            'test_car/ford02',
            'test_car/ford03',
            'test_car/ford04',
            'test_car/ford05',
            'test_car/ford06',
            'test_car/ford07',
            'test_car/mustang01'
        ]

    elif config.cfg.DATA_SETS_TYPE == 'didi':
        pass #todo
        # if_score = True
        # if 1:
        #     dataset = {'Round1Test': ['19_f2']}
        #
        # else:
        #     car = '3'
        #     data = '7'
        #     dataset = {
        #         car: [data]
        #     }
        #
        #     # compare newly generated tracklet_label_pred.xml with tracklet_labels_gt.xml. Change the path accordingly to
        #     #  fits you needs.
        #     gt_tracklet_file = os.path.join(cfg.RAW_DATA_SETS_DIR, car, data, 'tracklet_labels.xml')

    elif config.cfg.DATA_SETS_TYPE == 'kitti':
        pass #todo
        # if_score = False
        # car = '2011_09_26'
        # data = '0013'
        # dataset = {
        #     car: [data]
        # }
        #
        # # compare newly generated tracklet_label_pred.xml with tracklet_labels_gt.xml. Change the path accordingly to
        # #  fits you needs.
        # gt_tracklet_file = os.path.join(cfg.RAW_DATA_SETS_DIR, car, car + '_drive_' + data + '_sync',
        #                                 'tracklet_labels.xml')


    ## detecting
    test_tags = get_test_tags(test_bags)
    with BatchLoading(test_tags, require_shuffle=False, is_testset=True,
                      n_skip_frames=0 if fast_test else n_skip_frames) as dataset_loader:

        # dataset_loader = ub.batch_loading(cfg.PREPROCESSED_DATA_SETS_DIR, dataset, is_testset=True)

        print("tracklet_pred_dir: " + tracklet_pred_dir)
        pred_file = pred_and_save(tracklet_pred_dir, dataset_loader,
                                  frame_offset=0, log_tag=tag, weights_tag=weights_tag)

        if if_score:
            tracklet_score(pred_file=pred_file, gt_file=gt_tracklet_file, output_dir=tracklet_pred_dir)
            print("scores are save under {} directory.".format(tracklet_pred_dir))

        print("Completed")
