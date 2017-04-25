from data import load
from config import cfg
import os
import glob
from sklearn.utils import shuffle


class batch_loading:
    def __init__(self, dir_path, dates_to_drivers=None, indice=None, cache_num=10):
        self.dates_to_drivers = dates_to_drivers
        self.indice = indice
        self.cache_num = cache_num
        self.preprocess_path = dir_path
        # load_file_names is like 1_15_1490991691546439436 for didi or 2012_09_26_0005_00001 for kitti.
        if indice is None:
            self.load_file_names = self.get_all_load_index(self.preprocess_path, self.dates_to_drivers)
        else:
            self.load_file_names = indice
            self.load_once = True
        self.size = len(self.load_file_names)
        self.shuffled_file_names = shuffle(self.load_file_names, random_state=1)
        # for getting current index in shuffled_file_names
        self.batch_start_index = 0

        # num_frame_used means how many frames are used in current batch, if all frame are used, load another batch
        self.num_frame_used = cache_num

        # current batch contents
        self.train_rgbs = []
        self.train_tops = []
        self.train_fronts = []
        self.train_gt_labels = []
        self.train_gt_boxes3d = []

    def get_all_load_index(self, data_seg, dates_to_drivers):
        # todo: check if all files from lidar, rgb, gt_boxes3d is the same
        lidar_dir = os.path.join(data_seg, "top")
        print('lidar data here: ', lidar_dir)
        load_indexs = []
        for date, drivers in dates_to_drivers.items():
            for driver in drivers:
                # file_prefix is something like /home/stu/data/preprocessed/didi/lidar/2011_09_26_0001_*
                file_prefix = lidar_dir + '/' + date + '_' + driver + '_*'
                driver_files = glob.glob(file_prefix)
                name_list = [file.split('/')[-1].split('.')[0] for file in driver_files]
                load_indexs += name_list
        return load_indexs

    def load_test_frames(self, size):
        # just load it once
        if self.load_once:
            self.train_rgbs, self.train_tops, self.train_fronts, self.train_gt_labels, self.train_gt_boxes3d = \
                load(self.shuffled_file_names)
            self.num_frame_used = 0
            self.load_once = False
        # if there are still frames left
        frame_end = min(self.num_frame_used + size, self.cache_num)
        train_rgbs = self.train_rgbs[self.num_frame_used:frame_end]
        train_tops = self.train_tops[self.num_frame_used:frame_end]
        train_fronts = self.train_fronts[self.num_frame_used:frame_end]
        train_gt_labels = self.train_gt_labels[self.num_frame_used:frame_end]
        train_gt_boxes3d = self.train_gt_boxes3d[self.num_frame_used:frame_end]
        print("start index is here: ", self.num_frame_used)
        self.num_frame_used = frame_end
        if self.num_frame_used >= self.size:
            self.num_frame_used = 0
        # return number of batches according to current size.
        return train_rgbs, train_tops, train_fronts, train_gt_labels, train_gt_boxes3d


    # size is for loading how many frames per time.
    def load_batch(self, size):
        # if all frames are used up, reload another batch according to cache_num
        if self.num_frame_used >= self.cache_num:
            batch_end_index = self.batch_start_index + self.cache_num

            if batch_end_index < self.size:
                loaded_file_names = self.shuffled_file_names[self.batch_start_index:batch_end_index]
                self.batch_start_index = batch_end_index

            else:
                # print("end of the data is here: ", self.batch_start_index)
                diff_to_end = self.size - self.batch_start_index
                start_offset = self.cache_num - diff_to_end

                file_names_to_end = self.shuffled_file_names[self.batch_start_index:self.size]
                self.shuffled_file_names = shuffle(self.shuffled_file_names)
                file_names_from_start = self.shuffled_file_names[0:start_offset]

                loaded_file_names = file_names_to_end + file_names_from_start
                self.batch_start_index = start_offset
                # print("after reloop: ", self.batch_start_index)

            print('The loaded file name here: ', loaded_file_names)
            self.train_rgbs, self.train_tops, self.train_fronts, self.train_gt_labels, self.train_gt_boxes3d = \
                load(loaded_file_names)
            self.num_frame_used = 0

        # if there are still frames left
        frame_end = min(self.num_frame_used + size, self.cache_num)
        train_rgbs = self.train_rgbs[self.num_frame_used:frame_end]
        train_tops = self.train_tops[self.num_frame_used:frame_end]
        train_fronts = self.train_fronts[self.num_frame_used:frame_end]
        train_gt_labels = self.train_gt_labels[self.num_frame_used:frame_end]
        train_gt_boxes3d = self.train_gt_boxes3d[self.num_frame_used:frame_end]
        # print("start index is here: ", self.num_frame_used)
        self.num_frame_used = frame_end
        # return number of batches according to current size.
        return train_rgbs, train_tops, train_fronts, train_gt_labels, train_gt_boxes3d


    def load(self, size, batch=True):
        if batch:
            train_rgbs, train_tops, train_fronts, train_gt_labels, train_gt_boxes3d = self.load_batch(size)
        else:
            train_rgbs, train_tops, train_fronts, train_gt_labels, train_gt_boxes3d = self.load_test_frames(size)

        return train_rgbs, train_tops, train_fronts, train_gt_labels, train_gt_boxes3d


if __name__ == '__main__':
    # testing image testing, single frames

    # batch frame testing.
    dataset_dir = cfg.PREPROCESSED_DATA_SETS_DIR

    dates_to_drivers = {'1':['15', '6_f']}
    load_indexs = None
    batches = batch_loading(dataset_dir, dates_to_drivers, load_indexs)

    for i in range(1000):
        train_rgbs, train_tops, train_fronts, train_gt_labels, train_gt_boxes3d = batches.load(2, batch=True)

    # this code is for single testing.
    # load_indexs = ['1_15_00000', '1_15_00001', '1_15_00002']
    # batches = batch_loading(dataset_dir, dates_to_drivers, load_indexs)
    #
    # for i in range(1000):
    #     train_rgbs, train_tops, train_fronts, train_gt_labels, train_gt_boxes3d = batches.load(1, False)
    #
