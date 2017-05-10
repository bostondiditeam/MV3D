import os
import glob
from collections import defaultdict
from warnings import warn


# get all file names starts from a prefix string.
def get_file_names(data_dir, data_type, driver, date, index=None):
    dir_path = os.path.join(data_dir, data_type)
    driver_path = os.path.join(dir_path, date, driver)
    if index is None:
        prefix = driver_path + '/*'
        driver_files = glob.glob(prefix)
    else:
        prefix = [os.path.join(driver_path, file_name) for file_name in index]
        driver_files = [glob.glob(i + '*')[0] for i in prefix]
    return driver_files

def check_preprocessed_data(data_seg, dates_to_drivers, is_testset=False):
    problem_driver =  defaultdict(list)
    right_driver = defaultdict(list)

    for date, drivers in dates_to_drivers.items():
        for driver in drivers:
            rgb_files = get_file_names(data_seg, "rgb", driver, date)
            top_files = get_file_names(data_seg, "top", driver, date)
            # front_files = get_file_names(data_seg, "front", driver, date)
            gt_labels_files = get_file_names(data_seg, "gt_labels", driver, date)
            gt_boxes3d_files = get_file_names(data_seg, "gt_boxes3d", driver, date)

            if is_testset:
                value_set = set([len(rgb_files), len(top_files)])
            else:
                value_set = set([len(rgb_files), len(top_files), len(gt_labels_files), len(gt_boxes3d_files)])
            if len(value_set) != 1:
                # print('date is here {} and driver here {}'.format(date, driver))
                problem_driver[date].append(driver)
            else:
                right_driver[date].append(driver)

    for key, value in right_driver.items():
        print("CORRECT!, date {{'{}':{}}} has same number of rgbs, tops, gt_labels or gt_boxes".format(key, value))

    if len(problem_driver.keys()) != 0:
        for key, value in problem_driver.items():
            warn("INCORRECT! date {{'{}':{}}} has different number of rgbs, tops, gt_labels or gt_boxes".format(key,
                                                                                                             value))
        raise ValueError('Check above warning info to find which date and driver data is incomplete. ')
    return True
