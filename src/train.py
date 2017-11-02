import numpy as np
import mv3d
import mv3d_net
import glob
from sklearn.utils import shuffle
from config import *
# import utils.batch_loading as ub
import argparse
import os
from utils.training_validation_data_splitter import TrainingValDataSplitter
from utils.batch_loading import BatchLoading2 as BatchLoading


def str2bool(v):
    if v.lower() in ('true'):
        return True
    elif v.lower() in ('false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training')

    all = '%s,%s,%s' % (mv3d_net.top_view_rpn_name, mv3d_net.imfeature_net_name, mv3d_net.fusion_net_name)

    parser.add_argument('-w', '--weights', type=str, nargs='?', default='',
                        help='use pre trained weights example: -w "%s" ' % (all))

    parser.add_argument('-t', '--targets', type=str, nargs='?', default=all,
                        help='train targets example: -w "%s" ' % (all))

    parser.add_argument('-i', '--max_iter', type=int, nargs='?', default=1000,
                        help='max count of train iter')

    parser.add_argument('-n', '--tag', type=str, nargs='?', default='unknown_tag',
                        help='set log tag')

    parser.add_argument('-c', '--continue_train', type=str2bool, nargs='?', default=False,
                        help='set continue train flag')
    args = parser.parse_args()

    print('\n\n{}\n\n'.format(args))
    tag = args.tag
    if tag == 'unknown_tag':
        tag = input('Enter log tag : ')
        print('\nSet log tag :"%s" ok !!\n' % tag)

    max_iter = args.max_iter
    weights = []
    if args.weights != '':
        weights = args.weights.split(',')

    targets = []
    if args.targets != '':
        targets = args.targets.split(',')

    dataset_dir = cfg.PREPROCESSED_DATA_SETS_DIR

    if cfg.DATA_SETS_TYPE == 'didi2':
        assert cfg.OBJ_TYPE == 'car' or cfg.OBJ_TYPE == 'ped'
        if cfg.OBJ_TYPE == 'car':
            train_n_val_dataset=[
            'suburu_pulling_up_to_it/suburu02',
            'nissan_brief/nissan06',
            # 'cmax_sitting_still/cmax01',
            # 'suburu_pulling_to_right/suburu03_bak',
            'nissan_pulling_up_to_it/nissan02',
            # 'ped_train/ped_train',
            # 'suburu_sitting_still/suburu01', # tracklet is very inaccurate.
            'nissan_pulling_to_left/nissan03',
            # 'bmw_sitting_still/bmw01',
            'nissan_pulling_away/nissan05',
            'suburu_pulling_to_left/suburu04',
            'nissan_pulling_to_right/nissan04',
            'suburu_driving_towards_it/suburu06',
            'suburu_not_visible/suburu12',
            'suburu_leading_front_left/suburu11',
            # 'nissan_sitting_still/nissan01',
            'suburu_driving_away/suburu05',
            'nissan_driving_past_it/nissan07',
            'suburu_driving_past_it/suburu07',
            'suburu_driving_parallel/suburu10',

            'bmw_following_long_split/001',
            'bmw_following_long_split/002',
            'bmw_following_long_split/003',
            'bmw_following_long_split/004',
            'bmw_following_long_split/005',
            'bmw_following_long_split/006',
            'bmw_following_long_split/007',
            'bmw_following_long_split/008',
            'bmw_following_long_split/009',
            'bmw_following_long_split/010',
            'bmw_following_long_split/011',

            'cmax_following_long_split/001',
            'cmax_following_long_split/002',
            'cmax_following_long_split/003',
            'cmax_following_long_split/004',
            'cmax_following_long_split/005',
            'cmax_following_long_split/006',
            'cmax_following_long_split/007',
            'cmax_following_long_split/008',
            'cmax_following_long_split/009',
            'cmax_following_long_split/010',
            'cmax_following_long_split/011',

            'nissan_following_long_split/001',
            'nissan_following_long_split/002',
            'nissan_following_long_split/003',
            'nissan_following_long_split/004',
            'nissan_following_long_split/005',
            'nissan_following_long_split/006',
            'nissan_following_long_split/007',
            'nissan_following_long_split/008',
            'nissan_following_long_split/009',
            'nissan_following_long_split/010',
            'nissan_following_long_split/011',

            'suburu_following_long_split/001',
            'suburu_following_long_split/002',
            'suburu_following_long_split/003',
            'suburu_following_long_split/004',
            'suburu_following_long_split/005',
            'suburu_following_long_split/006',
            'suburu_following_long_split/007',

            'suburu_follows_capture_split/001',
            'suburu_follows_capture_split/002',
            'suburu_follows_capture_split/003',
            'suburu_follows_capture_split/004',
            'suburu_follows_capture_split/005',

            'suburu_leading_at_distance_split/001',
            'suburu_leading_at_distance_split/002',
            'suburu_leading_at_distance_split/003',
            'suburu_leading_at_distance_split/004',

        ]
        else:
            train_n_val_dataset=[
                'ped_train/ped_train',
            ]


    elif cfg.DATA_SETS_TYPE == 'didi' or cfg.DATA_SETS_TYPE == 'test':
        training_dataset = {
            '1': ['6_f', '9_f', '10', '13', '20', '21_f', '15', '19'],
            '2': ['3_f', '6_f', '8_f'],
            '3': ['2_f', '4', '6', '8', '7', '11_f']}

        validation_dataset = {
            '1': ['15']}

    elif cfg.DATA_SETS_TYPE == 'kitti':
        train_n_val_dataset = [
            # '2011_09_26/2011_09_26_drive_0001_sync', # for tracking
            '2011_09_26/2011_09_26_drive_0002_sync',
            '2011_09_26/2011_09_26_drive_0005_sync',
            # '2011_09_26/2011_09_26_drive_0009_sync',
            '2011_09_26/2011_09_26_drive_0011_sync',
            '2011_09_26/2011_09_26_drive_0013_sync',
            '2011_09_26/2011_09_26_drive_0014_sync',
            '2011_09_26/2011_09_26_drive_0015_sync',
            '2011_09_26/2011_09_26_drive_0017_sync',
            '2011_09_26/2011_09_26_drive_0018_sync',
            '2011_09_26/2011_09_26_drive_0019_sync',
            '2011_09_26/2011_09_26_drive_0020_sync',
            '2011_09_26/2011_09_26_drive_0022_sync',
            '2011_09_26/2011_09_26_drive_0023_sync',
            '2011_09_26/2011_09_26_drive_0027_sync',
            '2011_09_26/2011_09_26_drive_0028_sync',
            '2011_09_26/2011_09_26_drive_0029_sync',
            '2011_09_26/2011_09_26_drive_0032_sync',
            '2011_09_26/2011_09_26_drive_0035_sync',
            '2011_09_26/2011_09_26_drive_0036_sync',
            '2011_09_26/2011_09_26_drive_0039_sync',
            '2011_09_26/2011_09_26_drive_0046_sync',
            '2011_09_26/2011_09_26_drive_0048_sync',
            '2011_09_26/2011_09_26_drive_0051_sync',
            '2011_09_26/2011_09_26_drive_0052_sync',
            '2011_09_26/2011_09_26_drive_0056_sync',
            '2011_09_26/2011_09_26_drive_0057_sync',
            '2011_09_26/2011_09_26_drive_0059_sync',
            '2011_09_26/2011_09_26_drive_0060_sync',
            '2011_09_26/2011_09_26_drive_0061_sync',
            '2011_09_26/2011_09_26_drive_0064_sync',
            '2011_09_26/2011_09_26_drive_0070_sync',
            '2011_09_26/2011_09_26_drive_0079_sync',
            '2011_09_26/2011_09_26_drive_0084_sync',
            '2011_09_26/2011_09_26_drive_0086_sync',
            '2011_09_26/2011_09_26_drive_0087_sync',
            '2011_09_26/2011_09_26_drive_0091_sync',
            # '2011_09_26/2011_09_26_drive_0093_sync',  #data size not same
            # '2011_09_26/2011_09_26_drive_0095_sync',
            # '2011_09_26/2011_09_26_drive_0096_sync',
            # '2011_09_26/2011_09_26_drive_0104_sync',
            # '2011_09_26/2011_09_26_drive_0106_sync',
            # '2011_09_26/2011_09_26_drive_0113_sync',
            # '2011_09_26/2011_09_26_drive_0117_sync',
            '2011_09_26/2011_09_26_drive_0119_sync',
        ]

    # shuffle bag list or same kind of bags will only be in training or validation set.
    train_n_val_dataset = shuffle(train_n_val_dataset, random_state=666)
    data_splitter = TrainingValDataSplitter(train_n_val_dataset)

    with BatchLoading(tags=data_splitter.training_tags, require_shuffle=True, random_num=np.random.randint(100),
                      is_flip=False) as training:
        with BatchLoading(tags=data_splitter.val_tags, queue_size=1, require_shuffle=True,random_num=666) as validation:
            train = mv3d.Trainer(train_set=training, validation_set=validation,
                                 pre_trained_weights=weights, train_targets=targets, log_tag=tag,
                                 continue_train=args.continue_train,
                                 fast_test_mode=True if max_iter == 1 else False)
            train(max_iter=max_iter)
