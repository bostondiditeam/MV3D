from sync_img_lidar_tracklet_tool.bag_to_kitti import *
import subprocess as subproc
import os
import glob

def check_lidar_data(point_cloud_dir):
    print(point_cloud_dir)
    next_level_dir = os.listdir(point_cloud_dir)
    next_level_dir = os.path.join(point_cloud_dir, next_level_dir[0])
    print('next_level here: ', next_level_dir)
    # get all txt files under
    all_txt_files = glob.glob(next_level_dir+'/*.txt')
    all_bin_files = glob.glob(next_level_dir + '/*.bin')
    # print('txt_files: ', all_txt_files)
    # print('bin files: ', all_bin_files)
    if len(all_txt_files) == len(all_bin_files):
        return True
    else:
        return False


if __name__ == '__main__':

    input_common = '/hdd/data/didi_competition/didi_dataset/round2/Data'
    output_common = '/home/stu/hdd_tmp/round2'
    pc_common = '/home/stu/round12_data/unsynced/data3'

    bag_dir_list = ['suburu_pulling_to_left',
                     'nissan_following_long',
                     'test_ped',
                     'suburu_following_long',
                     'nissan_pulling_to_right',
                     'suburu_pulling_to_right',
                     'suburu_not_visible',
                     'cmax_following_long',
                     'nissan_driving_past_it',
                     'cmax_sitting_still',
                     'suburu_pulling_up_to_it',
                     'suburu_driving_towards_it',
                     'suburu_sitting_still',
                     'suburu_driving_away',
                     'suburu_follows_capture',
                     'bmw_sitting_still',
                     'suburu_leading_front_left',
                     'nissan_sitting_still',
                     'nissan_brief',
                     'suburu_leading_at_distance',
                     'bmw_following_long',
                     'suburu_driving_past_it',
                     'test_car',
                     'ped_train',
                     'nissan_pulling_up_to_it',
                     'suburu_driving_parallel',
                     'nissan_pulling_to_left',
                     'nissan_pulling_away']
    bag_dir_list = ['suburu_not_visible',
                    'cmax_following_long',
                    'nissan_driving_past_it',
                    'cmax_sitting_still',
                    'suburu_pulling_up_to_it',
                    'suburu_driving_towards_it',
                    'suburu_sitting_still',
                    'suburu_driving_away',
                    'suburu_follows_capture',
                    'bmw_sitting_still',
                    'suburu_leading_front_left',
                    'nissan_sitting_still',
                    'nissan_brief',
                    'suburu_leading_at_distance',
                    'bmw_following_long',
                    'suburu_driving_past_it',
                    'test_car',
                    'ped_train',
                    'nissan_pulling_up_to_it',
                    'suburu_driving_parallel',
                    'nissan_pulling_to_left',
                    'nissan_pulling_away']

    bag_dir_list = ['nissan_pulling_up_to_it',
                    'suburu_driving_parallel',
                    'nissan_pulling_to_left',
                    'nissan_pulling_away']


    input_common_dir = [os.path.join(input_common, bag_dir) for bag_dir in bag_dir_list]
    output_common_dir = [os.path.join(output_common, bag_dir) for bag_dir in bag_dir_list]
    point_cloud_dir = [os.path.join(pc_common, bag_dir) for bag_dir in bag_dir_list]

    for i, o, pc in zip(input_common_dir, output_common_dir, point_cloud_dir):
        assert check_lidar_data(pc)

        subproc.call(('python', '-m', 'sync_img_lidar_tracklet_tool.bag_to_kitti', '-i', i, '-o', o, '-pc', pc))









