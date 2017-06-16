import cv2
import numpy as np
from config import cfg
import os
import glob
from sklearn.utils import shuffle
from utils.check_data import check_preprocessed_data, get_file_names
import net.processing.boxes3d  as box

# disable print
# import sys
# f = open(os.devnull, 'w')
# sys.stdout = f

def load(file_names,is_testset=False):

    # here the file names is like /home/stu/round12_data_out_range/preprocessed/didi/top/2/14_f/00013, the top inside
    first_item = file_names[0].split('/')
    prefix = '/'.join(first_item[:-4])
    #  need to be replaced.
    frame_num_list = ['/'.join(name.split('/')[-3:]) for name in file_names]

    # print('rgb path here: ', os.path.join(prefix,'rgb', date, driver, file + '.png'))
    train_rgbs=[cv2.imread(os.path.join(prefix,'rgb', file + '.png'),1) for file in frame_num_list]
    train_tops = [np.load(os.path.join(prefix, 'top', file + '.npy.npz'))['top_view'] for file in frame_num_list]
    train_fronts=[np.zeros((1, 1), dtype=np.float32) for file in frame_num_list]

    if is_testset==True:
        train_gt_boxes3d=None
        train_gt_labels=None
    else:
        train_gt_boxes3d = [np.load(os.path.join(prefix, 'gt_boxes3d', file + '.npy')) for file in frame_num_list]

        train_gt_labels = [np.load(os.path.join(prefix, 'gt_labels', file + '.npy')) for file in
                           frame_num_list]

    return train_rgbs,train_tops,train_fronts,train_gt_labels,train_gt_boxes3d


class batch_loading:
    def __init__(self, dir_path, dates_to_drivers=None, indice=None, cache_num=10, is_testset=False):
        self.dates_to_drivers = dates_to_drivers
        self.indice = indice
        self.cache_num = cache_num
        self.preprocess_path = dir_path
        self.is_testset = is_testset
        # load_file_names is like 1_15_1490991691546439436 for didi or 2012_09_26_0005_00001 for kitti.
        if indice is None:
            self.load_file_names = self.get_all_load_index(self.preprocess_path, self.dates_to_drivers, is_testset)
        else:
            # self.load_file_names = indice
            self.load_file_names = self.get_specific_load_index(indice, self.preprocess_path, self.dates_to_drivers,
                                                           is_testset)
            self.load_once = True
        self.size = len(self.load_file_names)

        # self.shuffled_file_names = shuffle(self.load_file_names, random_state=1)
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
        self.current_batch_file_names = []


    def get_shape(self):

        #print("file name is here: ", self.load_file_names[0])
        train_rgbs, train_tops, train_fronts, train_gt_labels, train_gt_boxes3d = load([self.load_file_names[0]],
                                                                                       is_testset=self.is_testset)

        top_shape = train_tops[0].shape
        front_shape = train_fronts[0].shape
        rgb_shape = train_rgbs[0].shape

        return top_shape, front_shape, rgb_shape

    def get_all_load_index(self, data_seg, dates_to_drivers, gt_included):
        # check if input data (rgb, top, gt_labels, gt_boxes) have the same amount.
        check_preprocessed_data(data_seg, dates_to_drivers, gt_included)
        top_dir = os.path.join(data_seg, "top")
        # print('lidar data here: ', lidar_dir)
        load_indexs = []
        for date, drivers in dates_to_drivers.items():
            for driver in drivers:
                # file_prefix is something like /home/stu/data/preprocessed/didi/lidar/2011_09_26_0001_*
                file_prefix = os.path.join(data_seg, "top", date, driver)
                driver_files = get_file_names(data_seg, "top", driver, date)
                if len(driver_files) == 0:
                    raise ValueError('Directory has no data starts from {}, please revise.'.format(file_prefix))

                name_list = [file.split('/')[-1].split('.')[0] for file in driver_files]
                name_list = [file.split('.')[0] for file in driver_files]
                load_indexs += name_list
        load_indexs = sorted(load_indexs)
        return load_indexs

    def get_specific_load_index(self, index, data_seg, dates_to_drivers, gt_included):
        # check if input data (rgb, top, gt_labels, gt_boxes) have the same amount.
        check_preprocessed_data(data_seg, dates_to_drivers, gt_included)
        top_dir = os.path.join(data_seg, "top")
        # print('lidar data here: ', lidar_dir)
        load_indexs = []
        for date, drivers in dates_to_drivers.items():
            for driver in drivers:
                # file_prefix is something like /home/stu/data/preprocessed/didi/lidar/2011_09_26_0001_*
                file_prefix = os.path.join(data_seg, "top", driver, date)
                driver_files = get_file_names(data_seg, "top", driver, date, index)
                if len(driver_files) == 0:
                    raise ValueError('Directory has no data starts from {}, please revise.'.format(file_prefix))

                name_list = [file.split('/')[-1].split('.')[0] for file in driver_files]
                name_list = [file.split('.')[0] for file in driver_files]
                load_indexs += name_list
        load_indexs = sorted(load_indexs)
        return load_indexs

    def load_test_frames(self, size, shuffled):
        # just load it once
        if self.load_once:
            if shuffled:
                self.load_file_names = shuffle(self.load_file_names)
            self.train_rgbs, self.train_tops, self.train_fronts, self.train_gt_labels, self.train_gt_boxes3d = \
                load(self.load_file_names)
            self.num_frame_used = 0
            self.load_once = False
        # if there are still frames left
        self.current_batch_file_names = self.load_file_names
        frame_end = min(self.num_frame_used + size, self.cache_num)
        train_rgbs = self.train_rgbs[self.num_frame_used:frame_end]
        train_tops = self.train_tops[self.num_frame_used:frame_end]
        train_fronts = self.train_fronts[self.num_frame_used:frame_end]
        train_gt_labels = self.train_gt_labels[self.num_frame_used:frame_end]
        train_gt_boxes3d = self.train_gt_boxes3d[self.num_frame_used:frame_end]
        handle_id = self.current_batch_file_names[self.num_frame_used:frame_end]
        handle_id = ['/'.join(name.split('/')[-3:]) for name in handle_id]
        # print("start index is here: ", self.num_frame_used)
        self.num_frame_used = frame_end
        if self.num_frame_used >= self.size:
            self.num_frame_used = 0
        # return number of batches according to current size.
        return train_rgbs, train_tops, train_fronts, train_gt_labels, train_gt_boxes3d, handle_id


    # size is for loading how many frames per time.
    def load_batch(self, size, shuffled):
        if shuffled:
            self.load_file_names = shuffle(self.load_file_names)

        # if all frames are used up, reload another batch according to cache_num
        if self.num_frame_used >= self.cache_num:
            batch_end_index = self.batch_start_index + self.cache_num

            if batch_end_index < self.size:
                loaded_file_names = self.load_file_names[self.batch_start_index:batch_end_index]
                self.batch_start_index = batch_end_index

            else:
                # print("end of the data is here: ", self.batch_start_index)
                diff_to_end = self.size - self.batch_start_index
                start_offset = self.cache_num - diff_to_end

                file_names_to_end = self.load_file_names[self.batch_start_index:self.size]
                if shuffled:
                    self.load_file_names = shuffle(self.load_file_names)

                file_names_from_start = self.load_file_names[0:start_offset]

                loaded_file_names = file_names_to_end + file_names_from_start
                self.batch_start_index = start_offset
                # print("after reloop: ", self.batch_start_index)

            # print('The loaded file name here: ', loaded_file_names)
            self.current_batch_file_names = loaded_file_names
            self.train_rgbs, self.train_tops, self.train_fronts, self.train_gt_labels, self.train_gt_boxes3d = \
                load(loaded_file_names, is_testset=self.is_testset)
            self.num_frame_used = 0

        # if there are still frames left
        frame_end = min(self.num_frame_used + size, self.cache_num)
        train_rgbs = self.train_rgbs[self.num_frame_used:frame_end]
        train_tops = self.train_tops[self.num_frame_used:frame_end]
        train_fronts = self.train_fronts[self.num_frame_used:frame_end]
        if self.is_testset:
            train_gt_labels = None
            train_gt_boxes3d = None
        else:
            train_gt_labels = self.train_gt_labels[self.num_frame_used:frame_end]
            train_gt_boxes3d = self.train_gt_boxes3d[self.num_frame_used:frame_end]
        # print("start index is here: ", self.num_frame_used)
        handle_id = self.current_batch_file_names[self.num_frame_used:frame_end]
        handle_id = ['/'.join(name.split('/')[-3:]) for name in handle_id]
        # print('handle id here: ', handle_id)
        self.num_frame_used = frame_end
        # return number of batches according to current size.
        return train_rgbs, train_tops, train_fronts, train_gt_labels, train_gt_boxes3d, handle_id

    def get_date_and_driver(self, handle_id):
        date_n_driver = ['/'.join(item.split('/')[0:2]) for item in handle_id]
        return date_n_driver

    def get_frame_info(self, handle_id):
        return handle_id

    def keep_gt_inside_range(self, train_gt_labels, train_gt_boxes3d):
        # todo : support batch size >1
        if train_gt_labels.shape[0] == 0:
            return False, None, None
        assert train_gt_labels.shape[0] == train_gt_boxes3d.shape[0]

        # get limited train_gt_boxes3d and train_gt_labels.
        keep = np.zeros((len(train_gt_labels)), dtype=bool)

        for i in range(len(train_gt_labels)):
            if box.box3d_in_top_view(train_gt_boxes3d[i]):
                keep[i] = 1

        # if all targets are out of range in selected top view, return True.
        if np.sum(keep) == 0:
            return False, None, None

        train_gt_labels = train_gt_labels[keep]
        train_gt_boxes3d = train_gt_boxes3d[keep]
        return True, train_gt_labels, train_gt_boxes3d

    def load(self, size, batch=True, shuffled=False):
        load_frames = True
        while load_frames:
            if batch:
                train_rgbs, train_tops, train_fronts, train_gt_labels, train_gt_boxes3d, frame_id = self.load_batch(size,
                                                                                                              shuffled)
            else:
                train_rgbs, train_tops, train_fronts, train_gt_labels, train_gt_boxes3d, frame_id =  \
                    self.load_test_frames(size, shuffled)
            load_frames = False
            
            if not self.is_testset:
                # for keeping all gt labels and gt boxes inside range, and discard gt out of selected range.
                is_gt_inside_range, batch_gt_labels_in_range, batch_gt_boxes3d_in_range = \
                    self.keep_gt_inside_range(train_gt_labels[0], train_gt_boxes3d[0])

                if not is_gt_inside_range:
                    load_frames = True
                    continue

                # modify gt_labels and gt_boxes3d values to be inside range.
                # todo current support only batch_size == 1
                train_gt_labels = np.zeros((1, batch_gt_labels_in_range.shape[0]), dtype=np.int32)
                train_gt_boxes3d = np.zeros((1, batch_gt_labels_in_range.shape[0], 8, 3), dtype=np.float32)
                train_gt_labels[0] = batch_gt_labels_in_range
                train_gt_boxes3d[0] = batch_gt_boxes3d_in_range


        return np.array(train_rgbs), np.array(train_tops), np.array(train_fronts), np.array(train_gt_labels), \
               np.array(train_gt_boxes3d), frame_id


if __name__ == '__main__':
    # testing image testing, single frames
    # batch frame testing.
    dataset_dir = cfg.PREPROCESSED_DATA_SETS_DIR

    dates_to_drivers = {'1':['11']}
    # dates_to_drivers = {'Round1Test': ['19_f2']}
    # load_indexs = None
    # batches = batch_loading(dataset_dir, dates_to_drivers, load_indexs, is_testset=True)
    # # get_shape is used for getting shape.
    # top_shape, front_shape, rgb_shape = batches.get_shape()
    # for i in range(1000):
    #     train_rgbs, train_tops, train_fronts, train_gt_labels, train_gt_boxes3d = batches.load(2, batch=True,
    #                                                                                            shuffled=False)

    # this code is for single testing.
    load_indexs = ['00000', '00001', '00002', '00003']
    batches = batch_loading(dataset_dir, dates_to_drivers, load_indexs, is_testset=True)

    for i in range(1000):
        train_rgbs, train_tops, train_fronts, train_gt_labels, train_gt_boxes3d, handle_id = batches.load(1, False)

