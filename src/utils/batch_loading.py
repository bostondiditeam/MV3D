import cv2
import numpy as np
from config import cfg
import os
import glob
from sklearn.utils import shuffle
from utils.check_data import check_preprocessed_data, get_file_names
import net.processing.boxes3d  as box
from multiprocessing import Process,Queue as Queue, Value,Array
# import queue
import time

import config
import os
import numpy as np
import glob
import cv2
from kitti_data.pykitti.tracklet import parseXML, TRUNC_IN_IMAGE, TRUNC_TRUNCATED
from config import cfg
import data
import net.utility.draw as draw
from raw_data import *
from utils.training_validation_data_splitter import TrainingValDataSplitter
import pickle
import array
import data
from sklearn.utils import shuffle
import threading


# disable print
# import sys
# f = open(os.devnull, 'w')
# sys.stdout = f

def load(file_names, is_testset=False):
    # here the file names is like /home/stu/round12_data_out_range/preprocessed/didi/top/2/14_f/00013, the top inside
    first_item = file_names[0].split('/')
    prefix = '/'.join(first_item[:-4])
    #  need to be replaced.
    frame_num_list = ['/'.join(name.split('/')[-3:]) for name in file_names]

    # print('rgb path here: ', os.path.join(prefix,'rgb', date, driver, file + '.png'))
    train_rgbs = [cv2.imread(os.path.join(prefix, 'rgb', file + '.png'), 1) for file in frame_num_list]
    train_tops = [np.load(os.path.join(prefix, 'top', file + '.npy.npz'))['top_view'] for file in frame_num_list]
    train_fronts = [np.zeros((1, 1), dtype=np.float32) for file in frame_num_list]

    if is_testset == True:
        train_gt_boxes3d = None
        train_gt_labels = None
    else:
        train_gt_boxes3d = [np.load(os.path.join(prefix, 'gt_boxes3d', file + '.npy')) for file in frame_num_list]

        train_gt_labels = [np.load(os.path.join(prefix, 'gt_labels', file + '.npy')) for file in
                           frame_num_list]

    return train_rgbs, train_tops, train_fronts, train_gt_labels, train_gt_boxes3d


class batch_loading:
    def __init__(self, dir_path, dates_to_drivers=None, indice=None, cache_num=10, is_testset=False):
        self.dates_to_drivers = dates_to_drivers
        self.indice = indice
        self.cache_num = cache_num
        self.preprocess_path = dir_path
        self.is_testset = is_testset

        self.preprocess = data.Preprocess()
        self.raw_img = Image()
        self.raw_tracklet = Tracklet()
        self.raw_lidar = Lidar()

        # load_file_names is like 1_15_1490991691546439436 for didi or 2012_09_26_0005_00001 for kitti.
        if indice is None:
            self.load_file_names = self.get_all_load_index(self.preprocess_path, self.dates_to_drivers, is_testset)
            self.tags = self.raw_img.get_tags()
        else:
            # self.load_file_names = indice
            self.load_file_names = self.get_specific_load_index(indice, self.preprocess_path, self.dates_to_drivers,
                                                                is_testset)
            self.load_once = True
        self.size = len(self.tags)

        # self.shuffled_file_names = shuffle(self.load_tags, random_state=1)
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

    def load_from_one_tag(self, one_frame_tag):
        obstacles = self.raw_tracklet.load(one_frame_tag)
        rgb = self.raw_img.load(one_frame_tag)
        lidar = self.raw_lidar.load(one_frame_tag)
        return obstacles, rgb, lidar

    def preprocess(self, rgb, lidar, obstacles):
        rgb = preprocess.rgb(rgb)
        top = preprocess.lidar_to_top(lidar)
        boxes3d = [preprocess.bbox3d(obs) for obs in obstacles]
        labels = [preprocess.label(obs) for obs in obstacles]
        return rgb, top, boxes3d, labels

    def draw_bbox_on_rgb(self, rgb, boxes3d):
        img = draw.draw_box3d_on_camera(rgb, boxes3d)
        new_size = (img.shape[1] // 3, img.shape[0] // 3)
        img = cv2.resize(img, new_size)
        path = os.path.join(config.cfg.LOG_DIR, 'test', 'rgb', '%s.png' % one_frame_tag.replace('/', '_'))
        cv2.imwrite(path, img)
        print('write %s finished' % path)

    def draw_bbox_on_lidar_top(self, top, boxes3d):
        path = os.path.join(config.cfg.LOG_DIR, 'test', 'top', '%s.png' % one_frame_tag.replace('/', '_'))
        top_image = data.draw_top_image(top)
        top_image = data.draw_box3d_on_top(top_image, boxes3d, color=(0, 0, 80))
        cv2.imwrite(path, top_image)
        print('write %s finished' % path)

    def get_shape(self):

        # print("file name is here: ", self.load_file_names[0])
        train_rgbs, train_tops, train_fronts, train_gt_labels, train_gt_boxes3d = load([self.load_file_names[0]],
                                                                                       is_testset=self.is_testset)

        obstacles, rgb, lidar = self.load_from_one_tag([self.tags[0]],
                                                       is_testset=self.is_testset)
        train_rgbs, train_tops, train_fronts, train_gt_labels, train_gt_boxes3d = self.preprocess()

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
                train_rgbs, train_tops, train_fronts, train_gt_labels, train_gt_boxes3d, frame_id = \
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


def draw_bbox_on_rgb(rgb, boxes3d, one_frame_tag):
    img = draw.draw_box3d_on_camera(rgb, boxes3d)
    new_size = (img.shape[1] // 3, img.shape[0] // 3)
    img = cv2.resize(img, new_size)
    path = os.path.join(config.cfg.LOG_DIR, 'test', 'rgb', '%s.png' % one_frame_tag.replace('/', '_'))
    cv2.imwrite(path, img)
    print('write %s finished' % path)


def draw_bbox_on_lidar_top(top, boxes3d, one_frame_tag):
    path = os.path.join(config.cfg.LOG_DIR, 'test', 'top', '%s.png' % one_frame_tag.replace('/', '_'))
    top_image = data.draw_top_image(top)
    top_image = data.draw_box3d_on_top(top_image, boxes3d, color=(0, 0, 80))
    cv2.imwrite(path, top_image)
    print('write %s finished' % path)


use_thread = True

class BatchLoading2:

    def __init__(self, bags, tags, queue_size=20, require_shuffle=False,
                 require_log=False, is_testset=False):
        self.is_testset = is_testset
        self.shuffled = require_shuffle
        self.preprocess = data.Preprocess()
        self.raw_img = Image()
        self.raw_tracklet = Tracklet()
        self.raw_lidar = Lidar()

        self.bags = bags
        # get all tags
        self.tags = tags

        if self.shuffled:
            self.tags = shuffle(self.tags)

        self.tag_index = 0
        self.size = len(self.tags)

        self.require_log = require_log

        self.cache_size = queue_size
        self.loader_need_exit = Value('i', 0)

        if use_thread:
            self.prepr_data=[]
            self.lodaer_processing = threading.Thread(target=self.loader)
        else:
            self.preproc_data_queue = Queue()
            self.buffer_blocks = [Array('h', 41246691) for i in range(queue_size)]
            self.blocks_usage = Array('i', range(queue_size))
            self.lodaer_processing = Process(target=self.loader)
        self.lodaer_processing.start()


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.loader_need_exit.value=True
        if self.require_log: print('set loader_need_exit True')
        self.lodaer_processing.join()
        if self.require_log: print('exit lodaer_processing')

    def load_from_one_tag(self, one_frame_tag):
        if self.is_testset:
            obstacles = None
        else:
            obstacles = self.raw_tracklet.load(one_frame_tag)
        rgb = self.raw_img.load(one_frame_tag)
        lidar = self.raw_lidar.load(one_frame_tag)
        return obstacles, rgb, lidar


    def preprocess_one_frame(self, rgb, lidar, obstacles):
        rgb = self.preprocess.rgb(rgb)
        top = self.preprocess.lidar_to_top(lidar)
        if self.is_testset:
            return rgb, top, None, None
        boxes3d = [self.preprocess.bbox3d(obs) for obs in obstacles]
        labels = [self.preprocess.label(obs) for obs in obstacles]
        return rgb, top, boxes3d, labels

    def get_shape(self):
        train_rgbs, train_tops, train_fronts, train_gt_labels, train_gt_boxes3d, _ = self.load()
        top_shape = train_tops[0].shape
        front_shape = train_fronts[0].shape
        rgb_shape = train_rgbs[0].shape

        return top_shape, front_shape, rgb_shape

    def data_preprocessed(self):
        fronts = []
        frame_tag = self.tags[self.tag_index]
        obstacles, rgb, lidar = self.load_from_one_tag(frame_tag)
        rgb, top, boxes3d, labels = self.preprocess_one_frame(rgb, lidar, obstacles)
        if self.require_log and not self.is_testset:
            draw_bbox_on_rgb(rgb, boxes3d, frame_tag)
            draw_bbox_on_lidar_top(top, boxes3d, frame_tag)

        self.tag_index += 1

        # reset self tag_index to 0 and shuffle tag list
        if self.tag_index >= self.size:
            self.tag_index = 0
            if self.shuffled:
                self.tags = shuffle(self.tags)



        return np.array([rgb]), np.array([top]), np.array([fronts]), np.array([labels]), \
               np.array([boxes3d]), frame_tag

    def find_empty_block(self):
        idx = -1
        for i in range(self.cache_size):
            if self.blocks_usage[i] == 1:
                continue
            else:
                idx = i
                break
        return idx


    def loader(self):
        if use_thread:
            while self.loader_need_exit.value == 0:

                if len(self.prepr_data) >=self.cache_size:
                    time.sleep(1)
                    # print('sleep ')
                else:
                    self.prepr_data = [(self.data_preprocessed())]+self.prepr_data
                    # print('data_preprocessed')
        else:
            while self.loader_need_exit.value == 0:
                empty_idx = self.find_empty_block()
                if empty_idx == -1:
                    time.sleep(1)
                    # print('sleep ')
                else:
                    prepr_data = (self.data_preprocessed())
                    # print('data_preprocessed')
                    dumps = pickle.dumps(prepr_data)
                    length = len(dumps)
                    self.buffer_blocks[empty_idx][0:length] = dumps[0:length]

                    self.preproc_data_queue.put({
                        'index': empty_idx,
                        'length': length
                    })


        if self.require_log:print('loader exit')



    def load(self):
        if use_thread:
            while len(self.prepr_data)==0:
                time.sleep(1)
            data_ori = self.prepr_data.pop()


        else:

            # print('self.preproc_data_queue.qsize() = ', self.preproc_data_queue.qsize())
            info = self.preproc_data_queue.get(block=True)
            length = info['length']
            block_index = info['index']
            dumps = self.buffer_blocks[block_index][0:length]

            #set flag
            self.blocks_usage[block_index] = 0

            # convert to bytes string
            dumps = array.array('B',dumps).tostring()
            data_ori = pickle.loads(dumps)

        return data_ori



    def get_frame_info(self):
        return self.tags[self.tag_index]



if __name__ == '__main__':
    # testing image testing, single frames
    # batch frame testing.
    dataset_dir = cfg.PREPROCESSED_DATA_SETS_DIR

    dates_to_drivers = {'1': ['11']}
    # dates_to_drivers = {'Round1Test': ['19_f2']}
    # load_indexs = None
    # batches = batch_loading(dataset_dir, dates_to_drivers, load_indexs, is_testset=True)
    # # get_shape is used for getting shape.
    # top_shape, front_shape, rgb_shape = batches.get_shape()
    # for i in range(1000):
    #     train_rgbs, train_tops, train_fronts, train_gt_labels, train_gt_boxes3d = batches.load(2, batch=True,
    #                                                                                            shuffled=False)

    # this code is for single testing.
    # load_indexs = ['00000', '00001', '00002', '00003']
    # batches = batch_loading(dataset_dir, dates_to_drivers, load_indexs, is_testset=True)
    #
    # for i in range(1000):
    #     train_rgbs, train_tops, train_fronts, train_gt_labels, train_gt_boxes3d, handle_id = batches.load(1, False)
    train_key_list = ['nissan_pulling_away',
                      'nissan_pulling_up_to_it',
                      'suburu_follows_capture',
                      'nissan_pulling_to_left',
                      'nissan_driving_past_it',
                      'nissan_pulling_to_right',
                      'suburu_driving_away',
                      'nissan_following_long',
                      'suburu_driving_parallel',
                      'suburu_driving_towards_it',
                      'suburu_pulling_to_left',
                      'suburu_not_visible',

                      'suburu_leading_front_left',
                      'ped_train',
                      'bmw_following_long',
                      'cmax_following_long',
                      'suburu_following_long',
                      'suburu_driving_past_it',
                      'nissan_brief',
                      'suburu_leading_at_distance']

    train_key_full_path_list = [os.path.join(cfg.RAW_DATA_SETS_DIR, key) for key in train_key_list]
    train_value_list = [os.listdir(value)[0] for value in train_key_full_path_list]

    train_n_val_dataset = [k + '/' + v for k, v in zip(train_key_list, train_value_list)]

    splitter = TrainingValDataSplitter(train_n_val_dataset)

    # bl = BatchLoading2(splitter.training_bags, splitter.training_tags)

    with BatchLoading2(splitter.training_bags, splitter.training_tags) as bl:
        time.sleep(5)
        for i in range(5):
            t0 = time.time()
            data = bl.load()
            print('use time =', time.time()-t0)
            print(data)
            time.sleep(3)

        print('Done')
