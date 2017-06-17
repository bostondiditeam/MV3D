from sklearn.utils import shuffle
from raw_data import Image
import re
from config import cfg
import os

def get_test_tags(bags):
    raw_img = Image()
    tags_all = raw_img.get_tags()
    # get only tes
    all_tags = []
    for bag in bags:
        # get all tags start from bag string.
        r = re.compile(bag + "*")
        tag_list = filter(r.match, tags_all)
        bag_tag_list = list(tag_list)
        all_tags += bag_tag_list
    return all_tags


class TrainingValDataSplitter:
    def __init__(self, bags, split_rate=0.7):
        self.bags = bags
        self.raw_img = Image()

        # get all tags
        self.tags_all = self.raw_img.get_tags()
        self.size = len(self.tags_all)

        # for holding bags, like '1/15'
        self.training_bags = []
        # for holding tags, like '1/15/00000'
        self.training_tags = []

        self.val_bags = []
        self.val_tags = []
        self.split_rate = split_rate
        self.real_split_rate = -1

        # get training_bags, training_tags, val_bags, val_tags.
        self.split_bags_by_tag_name()

    def split_bags_by_tag_name(self):
        # input tags, split rate,
        # record: training_bags(a name list), training_tags(list), val_bags(a name list), val_tags(list)
        self.bags = shuffle(self.bags, random_state=0)
        # shuffle bags:
        all_tags = []
        for bag in self.bags:
            # get all tags start from bag string.
            r = re.compile(bag + "*")
            tag_list = filter(r.match, self.tags_all)
            bag_tag_list = list(tag_list)
            all_tags += bag_tag_list

        tag_size = len(all_tags)
        split_point = round(tag_size * self.split_rate)

        for i in range(split_point, tag_size):
            first_frame = all_tags[i]
            sec_frame = all_tags[i + 1]
            if ('/').join(first_frame.split('/')[:2]) != ('/').join(sec_frame.split('/')[:2]):
                split_point = i
                break

        self.training_tags = all_tags[:split_point + 1]
        self.val_tags = all_tags[split_point + 1:]

        self.real_split_rate = 1. * split_point / tag_size
        print('real split rate is here: ', self.real_split_rate)
        print('first frame is here: ', all_tags[i], ' and sec is: ', all_tags[i + 1])

        split_bag = ('/').join(all_tags[i + 1].split('/')[:2])

        in_training_bag = True
        for i in self.bags:
            if i == split_bag:
                in_training_bag = False

            if in_training_bag:
                self.training_bags += [i]
            else:
                self.val_bags += [i]


if __name__ == '__main__':
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
    # splitter.split_bags_by_tag_name()
    print('hello')

    # with BatchLoading2(train_n_val_dataset) as bl:
    #     time.sleep(1)
    #     for i in range(40):
    #         data = bl.load()
    #         print(data)
    #     print('Done')
