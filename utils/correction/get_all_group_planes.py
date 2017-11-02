import subprocess32 as sub
import os

bag_dir = '/hdd/data/didi_competition/didi_dataset/round2/Data'

tracklet_dir = '/hdd/round12_data/raw/didi2/'

train_key_list = ['suburu_pulling_up_to_it',
                  'nissan_brief',
                  'cmax_sitting_still',
                  'nissan_pulling_up_to_it',
                  'suburu_sitting_still',
                  'nissan_pulling_to_left',
                  'bmw_sitting_still',
                  'suburu_follows_capture',
                  'nissan_pulling_away',
                  'suburu_pulling_to_left',
                  'bmw_following_long',
                  'nissan_pulling_to_right',
                  'suburu_driving_towards_it',
                  'suburu_following_long',
                  'suburu_not_visible',
                  'suburu_leading_front_left',
                  'nissan_sitting_still',
                  'cmax_following_long',
                  'nissan_following_long',
                  'suburu_driving_away',
                  'suburu_leading_at_distance',
                  'nissan_driving_past_it',
                  'suburu_driving_past_it',
                  'suburu_driving_parallel',
                  ]

raw_data_dir = ''

train_key_full_path_list = [os.path.join(tracklet_dir, key) for key in train_key_list]
train_value_list = [os.listdir(value)[0] for value in train_key_full_path_list]

train_n_val_dataset = [k + '/' + v for k, v in zip(train_key_list, train_value_list)]

bag_path_list = [os.path.join(bag_dir, path+'.bag') for path in train_n_val_dataset]

tracklet_list = [os.path.join(tracklet_dir, path, 'tracklet_labels.xml') for path in train_n_val_dataset]

for i, (bag, tracklet) in enumerate(zip(bag_path_list, tracklet_list)):
    command = 'python ground.py ' + bag + ' ' + tracklet + ' --data ./output/' + train_key_list[i]
    sub.call(command, shell=True)